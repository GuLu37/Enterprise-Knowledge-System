//! 文档入库：上传 → 文本提取 → Parent-Child 切块 → 向量化 → milvus_lite 追加 + SQLite 元数据。
//!
//! 支持格式: txt, md (直接读取); docx (段落+表格顺序); xlsx (calamine); pdf (pdf-extract + lopdf)。
//! milvus_lite 写入: 新 Parquet data 文件 + manifest 版本更新（对齐 milvus_lite 存储格式）。

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema, Fields};
use parquet::arrow::ArrowWriter;
use serde_json::json;

/// 对齐 Python `_extract_text_from_txt`：utf-8 → utf-8-sig → gbk → gb18030。
fn decode_text_bytes(data: &[u8]) -> Result<String, String> {
    if let Ok(text) = std::str::from_utf8(data) {
        return Ok(text.to_string());
    }
    if let Some(without_bom) = data.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        if let Ok(text) = std::str::from_utf8(without_bom) {
            return Ok(text.to_string());
        }
    }
    for encoding in [encoding_rs::GBK, encoding_rs::GB18030] {
        let (text, _, had_errors) = encoding.decode(data);
        if !had_errors {
            return Ok(text.into_owned());
        }
    }
    Err("TXT 文件编码无法识别".into())
}

const PARENT_CHUNK_SIZE: usize = 500;
const PARENT_CHUNK_OVERLAP: usize = 60;
const CHILD_CHUNK_SIZE: usize = 150;
const CHILD_CHUNK_OVERLAP: usize = 30;
const DOCUMENT_CHUNK_SIZE: usize = 600;
const DOCUMENT_CHUNK_OVERLAP: usize = 100;
const DOCUMENT_ARTICLE_CHUNK_SIZE: usize = 280;
const DOCUMENT_ARTICLE_CHUNK_OVERLAP: usize = 50;
const STRUCTURED_FILE_TYPES: [&str; 4] = ["csv", "tsv", "xls", "xlsx"];
const ARTICLE_FILE_TYPES: [&str; 9] = [
    "pdf", "doc", "docx", "md", "txt", "html", "htm", "rtf", "markdown",
];
const EMBEDDING_DIM: usize = 768;

// ==================== 文本提取 ====================

pub fn extract_text(filename: &str, data: &[u8]) -> Result<String, String> {
    let ext = filename
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "txt" | "md" => decode_text_bytes(data),
        "csv" | "tsv" => decode_text_bytes(data),
        "html" | "htm" => extract_html_text(data),
        "docx" => extract_docx_text(data),
        "xlsx" | "xls" => extract_xlsx_text(data),
        "pdf" => extract_pdf_text(data),
        "pptx" => extract_pptx_text(data),
        _ => Err(format!("不支持的文件类型: .{ext}")),
    }
}

/// 当前内核可抽取的后缀（不含需 unstructured/OCR 的 .doc/.ppt/.rtf）。
pub fn is_extractable_type(file_type: &str) -> bool {
    matches!(
        file_type.trim().trim_start_matches('.').to_lowercase().as_str(),
        "txt" | "md" | "csv" | "tsv" | "html" | "htm" | "docx" | "xlsx" | "xls" | "pdf" | "pptx"
    )
}

fn xml_unescape(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn collapse_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_xml_name_boundary(ch: u8) -> bool {
    matches!(ch, b' ' | b'>' | b'/' | b'\n' | b'\r' | b'\t')
}

fn parse_open_tag(xml: &str) -> Option<(String, usize, bool)> {
    if !xml.starts_with('<') || xml.starts_with("</") {
        return None;
    }
    let end = xml.find('>')?;
    let raw = &xml[1..end];
    let self_closing = raw.trim_end().ends_with('/');
    let name = raw
        .trim_end_matches('/')
        .split_whitespace()
        .next()?
        .to_string();
    Some((name, end + 1, self_closing))
}

fn find_named_tag(haystack: &str, prefix: &str) -> Option<usize> {
    let mut search = 0;
    while let Some(rel) = haystack[search..].find(prefix) {
        let abs = search + rel;
        let next = haystack
            .as_bytes()
            .get(abs + prefix.len())
            .copied()
            .unwrap_or(b'>');
        if is_xml_name_boundary(next) {
            return Some(abs);
        }
        search = abs + prefix.len();
    }
    None
}

fn find_matching_close(xml: &str, name: &str) -> Option<usize> {
    let open = format!("<{name}");
    let close = format!("</{name}>");
    let mut depth = 1usize;
    let mut i = 0usize;
    while i < xml.len() {
        let rest = &xml[i..];
        let open_at = find_named_tag(rest, &open);
        let close_at = rest.find(&close);
        match (open_at, close_at) {
            (Some(open_idx), Some(close_idx)) if open_idx < close_idx => {
                depth += 1;
                i += open_idx + open.len();
            }
            (_, Some(close_idx)) => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + close_idx);
                }
                i += close_idx + close.len();
            }
            _ => return None,
        }
    }
    None
}

fn iter_xml_children(xml: &str) -> Vec<(String, String)> {
    let mut items = Vec::new();
    let mut pos = 0;
    while pos < xml.len() {
        let rest = &xml[pos..];
        let Some(rel) = rest.find('<') else {
            break;
        };
        pos += rel;
        let rest = &xml[pos..];
        if rest.starts_with("</") || rest.starts_with("<?") || rest.starts_with("<!") {
            pos += rest.find('>').map(|i| i + 1).unwrap_or(rest.len());
            continue;
        }
        let Some((name, after_open, self_closing)) = parse_open_tag(rest) else {
            break;
        };
        if self_closing {
            items.push((name, String::new()));
            pos += after_open;
            continue;
        }
        let inner_start = pos + after_open;
        let Some(close_at) = find_matching_close(&xml[inner_start..], &name) else {
            break;
        };
        let inner = xml[inner_start..inner_start + close_at].to_string();
        let close_len = name.len() + 3;
        items.push((name, inner));
        pos = inner_start + close_at + close_len;
    }
    items
}

fn extract_tagged_text(xml: &str, tag: &str) -> String {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut text = String::new();
    let mut rest = xml;
    while let Some(pos) = find_named_tag(rest, &open) {
        let after = &rest[pos..];
        let Some((_, after_open, self_closing)) = parse_open_tag(after) else {
            break;
        };
        if self_closing {
            rest = &after[after_open..];
            continue;
        }
        if let Some(close_at) = after[after_open..].find(&close) {
            text.push_str(&after[after_open..after_open + close_at]);
            rest = &after[after_open + close_at + close.len()..];
        } else {
            break;
        }
    }
    xml_unescape(&text)
}

fn extract_wt_text(xml: &str) -> String {
    extract_tagged_text(xml, "w:t")
}

fn extract_at_text(xml: &str) -> String {
    extract_tagged_text(xml, "a:t")
}

fn xml_body(xml: &str) -> &str {
    let Some(start_tag) = find_named_tag(xml, "<w:body") else {
        return xml;
    };
    let after = &xml[start_tag..];
    let Some((_, after_open, self_closing)) = parse_open_tag(after) else {
        return xml;
    };
    if self_closing {
        return "";
    }
    let inner_start = start_tag + after_open;
    find_matching_close(&xml[inner_start..], "w:body")
        .map(|close_at| &xml[inner_start..inner_start + close_at])
        .unwrap_or(&xml[inner_start..])
}

fn extract_table_block(tbl_xml: &str) -> Option<String> {
    let mut rows = Vec::new();
    for (name, tr_xml) in iter_xml_children(tbl_xml) {
        if name != "w:tr" {
            continue;
        }
        let mut cells = Vec::new();
        for (cell_name, tc_xml) in iter_xml_children(&tr_xml) {
            if cell_name != "w:tc" {
                continue;
            }
            let mut parts = Vec::new();
            for (para_name, p_xml) in iter_xml_children(&tc_xml) {
                if para_name != "w:p" {
                    continue;
                }
                let text = collapse_ws(&extract_wt_text(&p_xml));
                if !text.is_empty() {
                    parts.push(text);
                }
            }
            cells.push(parts.join(" "));
        }
        if cells.iter().any(|cell| !cell.is_empty()) {
            rows.push(cells.join(" | "));
        }
    }
    if rows.is_empty() {
        None
    } else {
        Some(format!("[表格]\n{}", rows.join("\n")))
    }
}

/// 对齐 Python `_extract_text_from_docx`：按 body 顺序抽取段落与表格。
fn extract_docx_text(data: &[u8]) -> Result<String, String> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| format!("docx zip 打开失败: {e}"))?;
    let mut file = archive
        .by_name("word/document.xml")
        .map_err(|e| format!("docx 缺少 document.xml: {e}"))?;
    let mut xml = String::new();
    std::io::Read::read_to_string(&mut file, &mut xml)
        .map_err(|e| format!("docx xml 读取失败: {e}"))?;

    let mut blocks = Vec::new();
    for (name, inner) in iter_xml_children(xml_body(&xml)) {
        if name == "w:p" {
            let text = collapse_ws(&extract_wt_text(&inner));
            if !text.is_empty() {
                blocks.push(text);
            }
        } else if name == "w:tbl" {
            if let Some(table) = extract_table_block(&inner) {
                blocks.push(table);
            }
        }
    }
    Ok(blocks.join("\n\n"))
}

const EXCEL_FILL_COLUMN_HINTS: [&str; 22] = [
    "部门", "团队", "小组", "班组", "科室", "事业部", "中心", "区域", "地区", "办公地点",
    "岗位", "职级", "级别", "序列", "分类", "类别", "状态", "负责人", "主管", "上级", "项目",
    "产品",
];
const EXCEL_NO_FILL_HINTS: [&str; 17] = [
    "金额", "工资", "薪资", "绩效", "应发", "实发", "数量", "单价", "税", "社保", "公积金",
    "合计", "总计", "余额", "成本", "收入", "支出",
];

fn collapse_excel_text(value: &str) -> String {
    value
        .replace('\r', " ")
        .replace('\n', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_excel_cell(cell: &calamine::Data) -> String {
    match cell {
        calamine::Data::Empty => String::new(),
        calamine::Data::Float(value) if value.fract() == 0.0 => format!("{}", *value as i64),
        _ => collapse_excel_text(&cell.to_string()),
    }
}

fn serialize_excel_row(values: &[String]) -> String {
    values
        .iter()
        .map(|value| {
            if value.contains([',', '"', '\n', '\r']) {
                format!("\"{}\"", value.replace('"', "\"\""))
            } else {
                value.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn looks_numeric_excel_value(value: &str) -> bool {
    let normalized = value.trim().replace(',', "");
    if normalized.is_empty() {
        return false;
    }
    let mut rest = normalized.as_str();
    if rest.starts_with('+') || rest.starts_with('-') {
        rest = &rest[1..];
    }
    if let Some(stripped) = rest.strip_suffix("万元") {
        rest = stripped;
    } else if let Some(stripped) = rest.strip_suffix('万') {
        rest = stripped;
    } else if let Some(stripped) = rest.strip_suffix('元') {
        rest = stripped;
    } else if let Some(stripped) = rest.strip_suffix('%') {
        rest = stripped;
    }
    if rest.is_empty() {
        return false;
    }
    let mut seen_digit = false;
    let mut seen_dot = false;
    for (index, ch) in rest.chars().enumerate() {
        if ch.is_ascii_digit() {
            seen_digit = true;
            continue;
        }
        if ch == '.' && !seen_dot && (index > 0 || rest.len() > 1) {
            seen_dot = true;
            continue;
        }
        return false;
    }
    seen_digit
}

fn looks_like_excel_fill_column(header: &str, column_values: &[String]) -> bool {
    let normalized_header = header.split_whitespace().collect::<String>().to_lowercase();
    if normalized_header.is_empty() {
        return false;
    }
    if EXCEL_NO_FILL_HINTS
        .iter()
        .any(|hint| normalized_header.contains(hint))
    {
        return false;
    }
    if EXCEL_FILL_COLUMN_HINTS
        .iter()
        .any(|hint| normalized_header.contains(hint))
    {
        return true;
    }
    let values: Vec<String> = column_values
        .iter()
        .map(|value| value.trim().to_string())
        .collect();
    let nonempty: Vec<&str> = values
        .iter()
        .map(String::as_str)
        .filter(|value| !value.is_empty())
        .collect();
    if nonempty.len() < 2 {
        return false;
    }
    let text_ratio = nonempty
        .iter()
        .filter(|value| !looks_numeric_excel_value(value))
        .count() as f64
        / nonempty.len() as f64;
    let blank_ratio = (values.len() - nonempty.len()) as f64 / values.len() as f64;
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for value in &nonempty {
        *counts.entry(*value).or_insert(0) += 1;
    }
    let repeated_ratio = counts.values().copied().max().unwrap_or(0) as f64 / nonempty.len() as f64;
    text_ratio >= 0.7 && blank_ratio >= 0.25 && repeated_ratio >= 0.25
}

fn forward_fill_excel_rows(headers: &[String], rows: &[Vec<String>]) -> Vec<Vec<String>> {
    if rows.is_empty() {
        return Vec::new();
    }
    let width = headers.len();
    let column_candidates: Vec<bool> = (0..width)
        .map(|column_index| {
            let column_values: Vec<String> = rows
                .iter()
                .map(|row| {
                    row.get(column_index)
                        .cloned()
                        .unwrap_or_default()
                })
                .collect();
            looks_like_excel_fill_column(&headers[column_index], &column_values)
        })
        .collect();
    let mut previous_values = vec![String::new(); width];
    let mut filled_rows = Vec::new();
    for row in rows {
        let mut normalized_row = row.clone();
        normalized_row.resize(width, String::new());
        for column_index in 0..width {
            let cell_value = normalized_row[column_index].trim().to_string();
            if !cell_value.is_empty() {
                previous_values[column_index] = cell_value;
                continue;
            }
            if column_candidates[column_index] && !previous_values[column_index].is_empty() {
                normalized_row[column_index] = previous_values[column_index].clone();
            }
        }
        filled_rows.push(normalized_row);
    }
    filled_rows
}

fn infer_excel_header_index(rows: &[Vec<String>]) -> usize {
    let nonempty_rows: Vec<(usize, &Vec<String>)> = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.iter().any(|cell| !cell.trim().is_empty()))
        .collect();
    if nonempty_rows.is_empty() {
        return 0;
    }
    if nonempty_rows.len() == 1 {
        return nonempty_rows[0].0;
    }
    let max_width = nonempty_rows
        .iter()
        .map(|(_, row)| row.iter().filter(|cell| !cell.trim().is_empty()).count())
        .max()
        .unwrap_or(0);
    let mut best: Option<(f64, usize, i64, usize)> = None;
    for &(index, row) in &nonempty_rows {
        let values: Vec<String> = row
            .iter()
            .map(|cell| cell.trim().to_string())
            .filter(|cell| !cell.is_empty())
            .collect();
        let width = values.len();
        if width < 2 {
            continue;
        }
        let unique_ratio = values
            .iter()
            .map(|value| value.to_lowercase())
            .collect::<std::collections::HashSet<_>>()
            .len() as f64
            / width as f64;
        let text_ratio = values
            .iter()
            .filter(|value| !looks_numeric_excel_value(value))
            .count() as f64
            / width as f64;
        let similar_rows = rows
            .iter()
            .skip(index + 1)
            .take(3)
            .filter(|candidate| candidate.iter().any(|cell| !cell.trim().is_empty()))
            .filter(|candidate| {
                candidate.iter().filter(|cell| !cell.trim().is_empty()).count()
                    >= 2.max(width / 2)
            })
            .count();
        let mut score = width as f64 * 2.0
            + unique_ratio * 3.0
            + text_ratio * 3.0
            + similar_rows as f64 * 2.0;
        if width >= 2.max(max_width / 2) {
            score += 4.0;
        }
        let candidate = (score, width, -(index as i64), index);
        if best.map(|current| candidate > current).unwrap_or(true) {
            best = Some(candidate);
        }
    }
    best.map(|item| item.3).unwrap_or(nonempty_rows[0].0)
}

fn normalize_excel_headers(headers: &[String]) -> Vec<String> {
    let mut normalized_headers = Vec::new();
    let mut used_counts: HashMap<String, usize> = HashMap::new();
    for (index, header) in headers.iter().enumerate() {
        let mut value = header.split_whitespace().collect::<Vec<_>>().join(" ");
        if value.is_empty() || value.to_lowercase().starts_with("unnamed") {
            value = format!("列{}", index + 1);
        }
        let count = used_counts.entry(value.clone()).or_insert(0);
        *count += 1;
        if *count == 1 {
            normalized_headers.push(value);
        } else {
            normalized_headers.push(format!("{value}_{count}"));
        }
    }
    normalized_headers
}

fn format_excel_sheet(sheet_name: &str, raw_rows: Vec<Vec<String>>) -> Option<String> {
    let mut raw_rows: Vec<Vec<String>> = raw_rows
        .into_iter()
        .filter(|row| row.iter().any(|cell| !cell.trim().is_empty()))
        .collect();
    if raw_rows.is_empty() {
        return None;
    }
    let max_width = raw_rows.iter().map(Vec::len).max().unwrap_or(0);
    for row in &mut raw_rows {
        row.resize(max_width, String::new());
    }
    while !raw_rows.is_empty() && raw_rows.iter().all(|row| row.last().map(|cell| cell.trim().is_empty()).unwrap_or(true))
    {
        for row in &mut raw_rows {
            row.pop();
        }
    }
    let header_index = infer_excel_header_index(&raw_rows);
    let headers = normalize_excel_headers(&raw_rows[header_index]);
    let width = headers.len();
    let data_rows = forward_fill_excel_rows(&headers, &raw_rows[header_index + 1..]);
    let mut blocks = vec![
        format!("[Sheet] {sheet_name}"),
        serialize_excel_row(&headers),
    ];
    for row in &raw_rows[..header_index] {
        let values: Vec<&str> = row
            .iter()
            .take(width)
            .map(String::as_str)
            .filter(|cell| !cell.trim().is_empty())
            .collect();
        if !values.is_empty() {
            blocks.push(format!("[说明] {}", values.join(" | ")));
        }
    }
    for row in data_rows {
        if row.iter().any(|cell| !cell.trim().is_empty()) {
            blocks.push(serialize_excel_row(&row));
        }
    }
    Some(blocks.join("\n"))
}

/// xlsx/xls: calamine 读取后做表头推断、说明行保留与合并单元格向下填充。
fn extract_xlsx_text(data: &[u8]) -> Result<String, String> {
    use calamine::{open_workbook_auto_from_rs, Reader};
    let cursor = std::io::Cursor::new(data);
    let mut workbook = open_workbook_auto_from_rs(cursor)
        .map_err(|err| format!("Excel 打开失败: {err}"))?;
    let mut blocks = Vec::new();
    for (title, range) in workbook.worksheets() {
        let raw_rows: Vec<Vec<String>> = range
            .rows()
            .map(|row| row.iter().map(normalize_excel_cell).collect())
            .collect();
        if let Some(block) = format_excel_sheet(&title, raw_rows) {
            blocks.push(block);
        }
    }
    Ok(blocks.join("\n"))
}

fn clean_pdf_text(text: String) -> Result<String, String> {
    let cleaned = text.replace('\0', "");
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        return Err("PDF 未提取到文本，可能是扫描件".into());
    }
    Ok(trimmed.to_string())
}

fn extract_pdf_with_pdf_extract(data: &[u8]) -> Result<String, String> {
    let text = pdf_extract::extract_text_from_mem(data)
        .map_err(|err| format!("PDF 解析失败: {err}"))?;
    clean_pdf_text(text)
}

fn extract_pdf_with_lopdf(data: &[u8]) -> Result<String, String> {
    let document = lopdf::Document::load_mem(data)
        .map_err(|err| format!("PDF 解析失败: {err}"))?;
    let page_numbers: Vec<u32> = document.get_pages().keys().copied().collect();
    let text = document
        .extract_text(&page_numbers)
        .map_err(|err| format!("PDF 解析失败: {err}"))?;
    clean_pdf_text(text)
}

/// 对齐 Python `_extract_text_from_pdf`：主解析失败后走第二解析器（跳过 unstructured）。
fn extract_pdf_text(data: &[u8]) -> Result<String, String> {
    match extract_pdf_with_pdf_extract(data) {
        Ok(text) => Ok(text),
        Err(err) => {
            tracing::warn!("pdf-extract 解析失败，尝试 lopdf 兜底: {err}");
            extract_pdf_with_lopdf(data)
        }
    }
}

fn pptx_slide_number(name: &str) -> Option<u32> {
    let name = name.replace('\\', "/");
    if !name.contains("ppt/slides/slide") || name.contains("slideLayout") {
        return None;
    }
    let file = name.rsplit('/').next()?;
    let rest = file.strip_prefix("slide")?.strip_suffix(".xml")?;
    if rest.is_empty() || !rest.bytes().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

fn collect_paragraph_texts(xml: &str, tag: &str, out: &mut Vec<String>) {
    for (name, inner) in iter_xml_children(xml) {
        if name == tag {
            let text = extract_at_text(&inner);
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
        } else {
            collect_paragraph_texts(&inner, tag, out);
        }
    }
}

fn extract_slide_text(xml: &str) -> String {
    let mut paragraphs = Vec::new();
    collect_paragraph_texts(xml, "a:p", &mut paragraphs);
    paragraphs.join("\n")
}

/// 对齐 Python `_extract_text_from_pptx`：按幻灯片序号抽取，页与页之间空一行。
fn extract_pptx_text(data: &[u8]) -> Result<String, String> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| format!("pptx zip 打开失败: {e}"))?;
    let mut slides = Vec::new();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
        let name = file.name().to_string();
        let Some(number) = pptx_slide_number(&name) else {
            continue;
        };
        let mut xml = String::new();
        std::io::Read::read_to_string(&mut file, &mut xml).map_err(|e| e.to_string())?;
        slides.push((number, extract_slide_text(&xml)));
    }
    slides.sort_by_key(|(number, _)| *number);
    Ok(slides
        .into_iter()
        .map(|(_, text)| text)
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n"))
}

fn strip_html(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let mut text = String::new();
    let mut i = 0usize;
    while i < html.len() {
        let rest = &html[i..];
        let Some(rel) = rest.find('<') else {
            text.push_str(rest);
            break;
        };
        text.push_str(&rest[..rel]);
        let tag_start = i + rel;
        let lowered = &lower[tag_start..];
        let skip_until = if lowered.starts_with("<script") {
            lower[tag_start..].find("</script>").map(|end| tag_start + end + 9)
        } else if lowered.starts_with("<style") {
            lower[tag_start..].find("</style>").map(|end| tag_start + end + 8)
        } else {
            None
        };
        if let Some(end) = skip_until {
            i = end;
            continue;
        }
        let tag_end = rest[rel..].find('>').map(|end| tag_start + end + 1).unwrap_or(html.len());
        let tag = &lower[tag_start..tag_end];
        if tag.starts_with("<p")
            || tag.starts_with("<br")
            || tag.starts_with("<div")
            || tag.starts_with("<h")
            || tag.starts_with("<li")
            || tag.starts_with("<tr")
            || tag.starts_with("</p")
            || tag.starts_with("</div")
            || tag.starts_with("</h")
            || tag.starts_with("</li")
            || tag.starts_with("</tr")
        {
            text.push('\n');
        }
        i = tag_end;
    }
    xml_unescape(&text)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_html_text(data: &[u8]) -> Result<String, String> {
    let raw = decode_text_bytes(data)?;
    Ok(strip_html(&raw))
}

// ==================== 切块 ====================

const DEFAULT_TEXT_SEPARATORS: &[&str] = &["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""];
const ARTICLE_TEXT_SEPARATORS: &[&str] = &[
    "\n\n", "\n", "。", "！", "？", "；", "：", "，", ".", "!", "?", ";", ":", " ", "",
];

fn char_len(text: &str) -> usize {
    text.chars().count()
}

fn split_keep_nonempty(text: &str, separator: &str) -> Vec<String> {
    if separator.is_empty() {
        return text.chars().map(|ch| ch.to_string()).filter(|item| !item.is_empty()).collect();
    }
    text.split(separator)
        .filter(|item| !item.is_empty())
        .map(str::to_string)
        .collect()
}

fn join_docs(parts: &[String], separator: &str) -> Option<String> {
    let joined = parts.join(separator);
    let trimmed = joined.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn merge_splits(splits: &[String], separator: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let separator_len = char_len(separator);
    let mut docs = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut total = 0usize;
    for piece in splits {
        let piece_len = char_len(piece);
        let extra = if current.is_empty() { 0 } else { separator_len };
        if total + piece_len + extra > chunk_size && !current.is_empty() {
            if let Some(doc) = join_docs(&current, separator) {
                docs.push(doc);
            }
            while !current.is_empty()
                && (total > overlap
                    || total + piece_len + (if current.is_empty() { 0 } else { separator_len })
                        > chunk_size)
            {
                let first_len = char_len(&current[0]);
                let sep = if current.len() > 1 { separator_len } else { 0 };
                total = total.saturating_sub(first_len + sep);
                current.remove(0);
            }
        }
        current.push(piece.clone());
        total += piece_len + if current.len() > 1 { separator_len } else { 0 };
    }
    if let Some(doc) = join_docs(&current, separator) {
        docs.push(doc);
    }
    docs
}

fn recursive_split(text: &str, separators: &[&str], chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    let mut separator = separators.last().copied().unwrap_or("");
    let mut rest: &[&str] = &[];
    for (index, candidate) in separators.iter().enumerate() {
        if candidate.is_empty() {
            separator = "";
            rest = &separators[index + 1..];
            break;
        }
        if text.contains(candidate) {
            separator = candidate;
            rest = &separators[index + 1..];
            break;
        }
    }
    let pieces = split_keep_nonempty(text, separator);
    let mut final_chunks = Vec::new();
    let mut good = Vec::new();
    for piece in pieces {
        if char_len(&piece) <= chunk_size {
            good.push(piece);
            continue;
        }
        if !good.is_empty() {
            final_chunks.extend(merge_splits(&good, separator, chunk_size, overlap));
            good.clear();
        }
        if rest.is_empty() {
            final_chunks.push(piece);
        } else {
            final_chunks.extend(recursive_split(&piece, rest, chunk_size, overlap));
        }
    }
    if !good.is_empty() {
        final_chunks.extend(merge_splits(&good, separator, chunk_size, overlap));
    }
    final_chunks
}

/// 对齐 Python `split_text_chunks` / LangChain RecursiveCharacterTextSplitter。
fn split_text_with_separators(text: &str, chunk_size: usize, overlap: usize, separators: &[&str]) -> Vec<String> {
    let normalized = text.trim();
    if normalized.is_empty() {
        return Vec::new();
    }
    if char_len(normalized) <= chunk_size {
        return vec![normalized.to_string()];
    }
    recursive_split(normalized, separators, chunk_size.max(1), overlap)
        .into_iter()
        .map(|chunk| chunk.trim().to_string())
        .filter(|chunk| !chunk.is_empty())
        .collect()
}

fn split_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    split_text_with_separators(text, chunk_size, overlap, DEFAULT_TEXT_SEPARATORS)
}

/// 对齐 Python `split_text_chunks`，供对话记忆切块复用。
pub(crate) fn split_text_chunks(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    split_text(text, chunk_size, overlap)
}

struct HeadingRegexes {
    markdown: Regex,
    numbered: Regex,
    chapter: Regex,
    chinese: Regex,
    paren: Regex,
    hashes: Regex,
}

fn heading_regexes() -> &'static HeadingRegexes {
    static RE: OnceLock<HeadingRegexes> = OnceLock::new();
    RE.get_or_init(|| HeadingRegexes {
        markdown: Regex::new(r"^(#{1,6})\s+(\S.{0,80})$").expect("markdown heading"),
        numbered: Regex::new(r"^(?P<number>\d+(?:\.\d+)*)(?:、|[.．])\s*(?P<title>\S.{0,70})$")
            .expect("numbered heading"),
        chapter: Regex::new(r"^第[一二三四五六七八九十百千万0-9]+[章节条部分][、.．\s:：-]*\S.{0,70}$")
            .expect("chapter heading"),
        chinese: Regex::new(r"^[一二三四五六七八九十]+[、.．]\s*\S.{0,70}$").expect("chinese heading"),
        paren: Regex::new(r"^[（(][一二三四五六七八九十0-9]+[）)]\s*\S.{0,70}$").expect("paren heading"),
        hashes: Regex::new(r"^#{1,6}\s*").expect("heading hashes"),
    })
}

fn normalize_heading_line(line: &str) -> String {
    collapse_ws(line)
}

fn parse_section_heading(line: &str) -> Option<(usize, String)> {
    let normalized = normalize_heading_line(line);
    if normalized.is_empty() || normalized.chars().count() > 140 {
        return None;
    }
    let sentence_marks = normalized
        .chars()
        .filter(|c| matches!(*c, '。' | '；' | ';'))
        .count();
    if sentence_marks > 0 {
        return None;
    }
    let regexes = heading_regexes();
    if let Some(captures) = regexes.markdown.captures(&normalized) {
        return Some((captures[1].chars().count(), captures[2].trim().to_string()));
    }
    if let Some(captures) = regexes.numbered.captures(&normalized) {
        let level = captures.name("number").map(|m| m.as_str().matches('.').count() + 1).unwrap_or(1);
        return Some((level, normalized));
    }
    if regexes.chapter.is_match(&normalized) {
        return Some((1, normalized));
    }
    if regexes.chinese.is_match(&normalized) {
        return Some((1, normalized));
    }
    if regexes.paren.is_match(&normalized) {
        return Some((2, normalized));
    }
    None
}

fn clean_section_heading(line: &str) -> String {
    let heading = normalize_heading_line(line);
    heading_regexes().hashes.replace(&heading, "").trim().to_string()
}

fn split_text_into_sections(text: &str) -> Vec<(String, String)> {
    let mut sections = Vec::new();
    let mut heading_stack: Vec<(usize, String)> = Vec::new();
    let mut current_lines: Vec<String> = Vec::new();
    let mut heading_count = 0usize;

    let mut flush = |force: bool, heading_stack: &[(usize, String)], current_lines: &mut Vec<String>| {
        let body = current_lines.join("\n").trim().to_string();
        if !body.is_empty() || (force && !heading_stack.is_empty()) {
            let heading = heading_stack
                .iter()
                .map(|item| item.1.as_str())
                .collect::<Vec<_>>()
                .join(" / ");
            sections.push((heading, body));
        }
        current_lines.clear();
    };

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            if current_lines.last().is_some_and(|item| !item.is_empty()) {
                current_lines.push(String::new());
            }
            continue;
        }
        if let Some((level, heading)) = parse_section_heading(line) {
            if !current_lines.is_empty() {
                flush(false, &heading_stack, &mut current_lines);
            }
            heading_stack.retain(|item| item.0 < level);
            heading_stack.push((level, clean_section_heading(&heading)));
            heading_count += 1;
            continue;
        }
        current_lines.push(line.to_string());
    }
    flush(true, &heading_stack, &mut current_lines);
    if heading_count == 0 {
        Vec::new()
    } else {
        sections
    }
}

fn split_section_chunks(heading: &str, body: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    let prefix = if heading.trim().is_empty() {
        String::new()
    } else {
        format!("[章节] {}", heading.trim())
    };
    let normalized_body = body.trim();
    if !prefix.is_empty() && normalized_body.is_empty() {
        return Vec::new();
    }
    if prefix.is_empty() {
        return split_text_with_separators(
            normalized_body,
            chunk_size,
            chunk_overlap,
            ARTICLE_TEXT_SEPARATORS,
        );
    }
    let available_size = 120.max(chunk_size.saturating_sub(prefix.chars().count().saturating_add(1)));
    let overlap = chunk_overlap.min(available_size / 3);
    split_text_with_separators(normalized_body, available_size, overlap, ARTICLE_TEXT_SEPARATORS)
        .into_iter()
        .map(|chunk| format!("{prefix}\n{chunk}").trim().to_string())
        .collect()
}

fn split_document_text_by_sections(text: &str, chunk_size: usize, chunk_overlap: usize) -> Option<Vec<String>> {
    let sections = split_text_into_sections(text);
    if sections.is_empty() {
        return None;
    }
    let mut chunks = Vec::new();
    for (heading, body) in sections {
        chunks.extend(split_section_chunks(&heading, &body, chunk_size, chunk_overlap));
    }
    chunks.retain(|chunk| !chunk.trim().is_empty());
    Some(chunks)
}

fn is_article_file_type(file_type: &str) -> bool {
    ARTICLE_FILE_TYPES.contains(&file_type.trim().trim_start_matches('.').to_lowercase().as_str())
}

fn split_document_text(text: &str, chunk_size: usize, chunk_overlap: usize, file_type: &str) -> Vec<String> {
    if is_article_file_type(file_type) {
        let resolved_size = if chunk_size == DOCUMENT_CHUNK_SIZE {
            DOCUMENT_ARTICLE_CHUNK_SIZE
        } else {
            chunk_size
        };
        let resolved_overlap = if chunk_overlap == DOCUMENT_CHUNK_OVERLAP {
            DOCUMENT_ARTICLE_CHUNK_OVERLAP
        } else {
            chunk_overlap
        };
        if let Some(chunks) = split_document_text_by_sections(text, resolved_size, resolved_overlap) {
            return chunks;
        }
        return split_text_with_separators(text, resolved_size, resolved_overlap, ARTICLE_TEXT_SEPARATORS);
    }
    split_text(text, chunk_size, chunk_overlap)
}

fn parse_structured_sections(text: &str) -> Vec<(String, Vec<String>)> {
    let lines: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let mut sections = Vec::new();
    let mut current_name = String::new();
    let mut current_lines = Vec::new();
    let flush = |name: &mut String, lines: &mut Vec<String>, sections: &mut Vec<(String, Vec<String>)>| {
        if !name.is_empty() || !lines.is_empty() {
            sections.push((name.clone(), std::mem::take(lines)));
        }
    };
    for line in lines {
        if let Some(name) = eks_kernels::structured_sheet_name(&line) {
            flush(&mut current_name, &mut current_lines, &mut sections);
            current_name = name.to_string();
            continue;
        }
        current_lines.push(line);
    }
    flush(&mut current_name, &mut current_lines, &mut sections);
    sections
}

fn split_structured_text_chunks(text: &str, chunk_size: usize) -> Vec<String> {
    let sections = parse_structured_sections(text);
    if !sections.iter().any(|(name, _)| !name.is_empty()) {
        return split_text(text, chunk_size, PARENT_CHUNK_OVERLAP);
    }
    let mut chunks = Vec::new();
    for (sheet_name, sheet_lines) in sections {
        if sheet_lines.is_empty() {
            continue;
        }
        let sheet_prefix = if sheet_name.is_empty() {
            String::new()
        } else {
            format!("[Sheet] {sheet_name}")
        };
        let header = sheet_lines[0].clone();
        let rows = &sheet_lines[1..];
        let mut prefix_parts = Vec::new();
        if !sheet_prefix.is_empty() {
            prefix_parts.push(sheet_prefix);
        }
        if !header.is_empty() {
            prefix_parts.push(header);
        }
        let prefix = prefix_parts.join("\n");
        let available_size = 120.max(chunk_size.saturating_sub(prefix.chars().count().saturating_add(1)));
        if rows.is_empty() {
            if !prefix.is_empty() {
                chunks.push(prefix);
            }
            continue;
        }
        let mut current_rows: Vec<String> = Vec::new();
        let mut current_length = 0usize;
        for row in rows {
            let row_length = row.chars().count() + usize::from(!current_rows.is_empty());
            if !current_rows.is_empty() && current_length + row_length > available_size {
                chunks.push(prefix_parts.iter().cloned().chain(current_rows.iter().cloned()).collect::<Vec<_>>().join("\n").trim().to_string());
                current_rows.clear();
                current_length = 0;
            }
            current_rows.push(row.clone());
            current_length += row_length;
        }
        if !current_rows.is_empty() {
            chunks.push(
                prefix_parts
                    .iter()
                    .cloned()
                    .chain(current_rows.into_iter())
                    .collect::<Vec<_>>()
                    .join("\n")
                    .trim()
                    .to_string(),
            );
        }
    }
    chunks.into_iter().filter(|chunk| !chunk.trim().is_empty()).collect()
}

fn split_structured_row_chunks(text: &str) -> Vec<String> {
    let sections = parse_structured_sections(text);
    if !sections.iter().any(|(name, _)| !name.is_empty()) {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    for (sheet_name, sheet_lines) in sections {
        if sheet_lines.len() < 2 {
            continue;
        }
        let sheet_prefix = if sheet_name.is_empty() {
            String::new()
        } else {
            format!("[Sheet] {sheet_name}")
        };
        let header = &sheet_lines[0];
        let mut prefix_parts = Vec::new();
        if !sheet_prefix.is_empty() {
            prefix_parts.push(sheet_prefix.as_str());
        }
        if !header.is_empty() {
            prefix_parts.push(header.as_str());
        }
        for row in &sheet_lines[1..] {
            let row_text = row.trim();
            if row_text.is_empty() || row_text.starts_with('[') {
                continue;
            }
            let mut parts = prefix_parts.clone();
            parts.push(row_text);
            chunks.push(parts.join("\n"));
        }
    }
    chunks.into_iter().filter(|chunk| !chunk.trim().is_empty()).collect()
}

#[derive(Debug)]
pub struct ChunkResult {
    pub parents: Vec<ParentRow>,
    pub children: Vec<ChildRow>,
}

#[derive(Debug, Clone)]
pub struct ParentRow {
    pub parent_id: String,
    pub document_id: String,
    pub parent_index: i64,
    pub parent_text: String,
}

#[derive(Debug, Clone)]
pub struct ChildRow {
    pub child_id: String,
    pub parent_id: String,
    pub document_id: String,
    pub parent_index: i64,
    pub child_index: i64,
    pub child_text: String,
}

pub fn build_parent_child_chunks(
    document_id: &str,
    text: &str,
    file_type: &str,
) -> ChunkResult {
    let normalized_type = file_type.trim().trim_start_matches('.').to_lowercase();
    let is_structured = STRUCTURED_FILE_TYPES.contains(&normalized_type.as_str());
    let parent_texts = if is_structured {
        let structured = split_structured_text_chunks(text, PARENT_CHUNK_SIZE);
        if structured.is_empty() {
            split_text(text, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)
        } else {
            structured
        }
    } else {
        split_document_text(text, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, &normalized_type)
    };

    let mut parents = Vec::new();
    let mut children = Vec::new();

    for (parent_index, parent_text) in parent_texts.iter().enumerate() {
        let parent_id = format!("{}:p{:06}", document_id, parent_index);
        parents.push(ParentRow {
            parent_id: parent_id.clone(),
            document_id: document_id.to_string(),
            parent_index: parent_index as i64,
            parent_text: parent_text.clone(),
        });

        let child_texts = if is_structured {
            let structured = split_structured_row_chunks(parent_text);
            if structured.is_empty() {
                vec![parent_text.clone()]
            } else {
                structured
            }
        } else {
            let overlap = CHILD_CHUNK_OVERLAP.min(CHILD_CHUNK_SIZE / 3);
            let children = split_document_text(parent_text, CHILD_CHUNK_SIZE, overlap, &normalized_type);
            if children.is_empty() {
                vec![parent_text.clone()]
            } else {
                children
            }
        };

        for child_text in child_texts {
            let child_index = children.len();
            children.push(ChildRow {
                child_id: format!("{}:c{:06}", document_id, child_index),
                parent_id: parent_id.clone(),
                document_id: document_id.to_string(),
                parent_index: parent_index as i64,
                child_index: child_index as i64,
                child_text: child_text.trim().to_string(),
            });
        }
    }

    ChunkResult { parents, children }
}

/// 抽取文本并切块；空正文或无 child 时返回与 Python 相同的错误文案。
pub fn prepare_upload_chunks(
    document_id: &str,
    filename: &str,
    data: &[u8],
    file_type: &str,
) -> Result<ChunkResult, String> {
    let text = extract_text(filename, data)?;
    if text.trim().is_empty() {
        return Err("文档内容为空，无法生成向量".into());
    }
    let chunks = build_parent_child_chunks(document_id, &text, file_type);
    if chunks.children.is_empty() {
        return Err("文档切块失败，未生成有效内容".into());
    }
    Ok(chunks)
}

// ==================== milvus_lite 写入 ====================

pub async fn append_to_milvus_lite(
    store: &crate::milvus_lite::MilvusLiteStore,
    collection: &str,
    child_rows: &[ChildRow],
    source_name: &str,
    file_type: &str,
    content_type: &str,
    vectors: &[Vec<f32>],
) -> Result<(), String> {
    if child_rows.is_empty() || vectors.is_empty() {
        return Ok(());
    }

    let collection_dir = store
        .root
        .join("collections")
        .join(collection);
    let manifest_path = collection_dir.join("manifest.json");
    let manifest_str = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("manifest 读取失败: {e}"))?;
    let mut manifest: serde_json::Value =
        serde_json::from_str(&manifest_str).map_err(|e| format!("manifest 解析失败: {e}"))?;

    let current_seq = manifest["current_seq"].as_i64().unwrap_or(0);
    let data_dir = collection_dir.join("partitions").join("_default").join("data");
    std::fs::create_dir_all(&data_dir).map_err(|e| e.to_string())?;

    // 生成新 data 文件名
    let start_seq = current_seq + 1;
    let end_seq = current_seq + child_rows.len() as i64;
    let new_file = format!("data_{:06}_{:06}.parquet", start_seq, end_seq);
    let new_path = data_dir.join(&new_file);

    // 构建 Arrow RecordBatch（对齐 milvus_lite Parquet 列序）
    let n = child_rows.len();
    let pk_start = current_seq + 100; // 避免与现有 pk 冲突
    let pk_values: Vec<i64> = (0..n).map(|i| pk_start + i as i64).collect();
    let seq_values: Vec<i64> = (0..n).map(|i| start_seq + i as i64).collect();
    let chunk_indexes: Vec<i64> = child_rows.iter().map(|c| c.child_index).collect();
    let parent_indexes: Vec<i64> = child_rows.iter().map(|c| c.parent_index).collect();

    let fields = Fields::from(vec![
        Field::new("_seq", DataType::Int64, false),
        Field::new("pk", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
        Field::new("vector", DataType::FixedSizeList(
            std::sync::Arc::new(Field::new("item", DataType::Float32, true)), EMBEDDING_DIM as i32
        ), false),
        Field::new("document_id", DataType::Utf8, false),
        Field::new("chunk_index", DataType::Int64, false),
        Field::new("source_name", DataType::Utf8, false),
        Field::new("chunk_text", DataType::Utf8, false),
        Field::new("child_id", DataType::Utf8, false),
        Field::new("parent_id", DataType::Utf8, false),
        Field::new("parent_index", DataType::Int64, false),
        Field::new("child_text", DataType::Utf8, false),
        Field::new("file_type", DataType::Utf8, false),
        Field::new("content_type", DataType::Utf8, false),
    ]);

    // 展平向量
    let flat_vectors: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
    let vector_array = FixedSizeListArray::new(
        std::sync::Arc::new(Field::new("item", DataType::Float32, true)),
        EMBEDDING_DIM as i32,
        std::sync::Arc::new(Float32Array::from(flat_vectors)) as ArrayRef,
        None,
    );

    let columns: Vec<ArrayRef> = vec![
        std::sync::Arc::new(Int64Array::from(seq_values)),
        std::sync::Arc::new(Int64Array::from(pk_values)),
        std::sync::Arc::new(StringArray::from(vec![""; n])),
        std::sync::Arc::new(vector_array) as ArrayRef,
        std::sync::Arc::new(StringArray::from(child_rows.iter().map(|c| c.document_id.clone()).collect::<Vec<_>>())),
        std::sync::Arc::new(Int64Array::from(chunk_indexes)),
        std::sync::Arc::new(StringArray::from(vec![source_name; n])),
        std::sync::Arc::new(StringArray::from(child_rows.iter().map(|c| c.child_text.clone()).collect::<Vec<_>>())),
        std::sync::Arc::new(StringArray::from(child_rows.iter().map(|c| c.child_id.clone()).collect::<Vec<_>>())),
        std::sync::Arc::new(StringArray::from(child_rows.iter().map(|c| c.parent_id.clone()).collect::<Vec<_>>())),
        std::sync::Arc::new(Int64Array::from(parent_indexes)),
        std::sync::Arc::new(StringArray::from(child_rows.iter().map(|c| c.child_text.clone()).collect::<Vec<_>>())),
        std::sync::Arc::new(StringArray::from(vec![file_type; n])),
        std::sync::Arc::new(StringArray::from(vec![content_type; n])),
    ];

    let schema = Schema::new(fields);
    let batch = RecordBatch::try_new(std::sync::Arc::new(schema), columns)
        .map_err(|e| format!("RecordBatch 构建失败: {e}"))?;

    let file = std::fs::File::create(&new_path).map_err(|e| e.to_string())?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)
        .map_err(|e| format!("Parquet writer 创建失败: {e}"))?;
    writer.write(&batch).map_err(|e| format!("Parquet 写入失败: {e}"))?;
    writer.close().map_err(|e| format!("Parquet 关闭失败: {e}"))?;

    // 更新 manifest
    manifest["current_seq"] = json!(end_seq);
    let version = manifest["version"].as_i64().unwrap_or(0) + 1;
    manifest["version"] = json!(version);
    if let Some(partition) = manifest["partitions"]["_default"]["data_files"].as_array_mut() {
        partition.push(json!(format!("data/{new_file}")));
    }
    let manifest_out = serde_json::to_string_pretty(&manifest)
        .map_err(|e| e.to_string())?;
    std::fs::write(&manifest_path, manifest_out).map_err(|e| e.to_string())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn decodes_utf8_and_gbk_text_files() {
        assert_eq!(extract_text("手册.txt", "你好".as_bytes()).unwrap(), "你好");
        let (gbk, _, _) = encoding_rs::GBK.encode("差旅报销");
        assert_eq!(extract_text("手册.md", &gbk).unwrap(), "差旅报销");
        assert_eq!(extract_text("花名册.csv", &gbk).unwrap(), "差旅报销");
        let mut bom = vec![0xEF, 0xBB, 0xBF];
        bom.extend_from_slice("带BOM".as_bytes());
        assert_eq!(extract_text("a.txt", &bom).unwrap(), "\u{feff}带BOM");
    }

    #[test]
    fn rejects_unrecognized_text_encoding() {
        let err = extract_text("bad.txt", &[0xFF, 0xFE, 0x00]).unwrap_err();
        assert_eq!(err, "TXT 文件编码无法识别");
    }

    #[test]
    fn prepare_upload_chunks_rejects_empty_text() {
        let err = prepare_upload_chunks("doc-1", "empty.txt", b"   \n", "txt").unwrap_err();
        assert_eq!(err, "文档内容为空，无法生成向量");
        let chunks =
            prepare_upload_chunks("doc-1", "手册.md", "差旅报销需要提前申请。".as_bytes(), "md")
                .unwrap();
        assert!(!chunks.parents.is_empty());
        assert!(!chunks.children.is_empty());
    }

    #[test]
    fn excel_headers_fill_and_csv_serialize() {
        assert!(looks_numeric_excel_value("12.5%"));
        assert!(looks_numeric_excel_value("3万元"));
        assert!(!looks_numeric_excel_value("部门"));
        assert_eq!(
            normalize_excel_headers(&["部门".into(), "".into(), "unnamed".into(), "部门".into()]),
            vec!["部门", "列2", "列3", "部门_2"]
        );
        assert_eq!(serialize_excel_row(&["上海,浦东".into(), "ok".into()]), "\"上海,浦东\",ok");
        let rows = vec![
            vec!["标题".into(), "".into()],
            vec!["部门".into(), "姓名".into()],
            vec!["研发".into(), "张三".into()],
            vec!["".into(), "李四".into()],
        ];
        assert_eq!(infer_excel_header_index(&rows), 1);
        let headers = normalize_excel_headers(&rows[1]);
        let filled = forward_fill_excel_rows(&headers, &rows[2..]);
        assert_eq!(filled[1][0], "研发");
        let text = format_excel_sheet("花名册", rows).unwrap();
        assert!(text.starts_with("[Sheet] 花名册"));
        assert!(text.contains("[说明] 标题"));
        assert!(text.contains("研发,李四"));
    }

    #[test]
    fn structured_chunks_copy_sheet_header_and_skip_notes() {
        let text = "[Sheet] 花名册\n部门,姓名\n[说明] 内部资料\n研发,张三\n市场,李四";
        let parents = split_structured_text_chunks(text, 500);
        assert_eq!(parents.len(), 1);
        assert!(parents[0].contains("[Sheet] 花名册"));
        assert!(parents[0].contains("部门,姓名"));
        let children = split_structured_row_chunks(&parents[0]);
        assert_eq!(children.len(), 2);
        assert!(children[0].contains("部门,姓名"));
        assert!(children[0].contains("研发,张三"));
        assert!(!children.iter().any(|item| item.contains("[说明]")));
        let chunks = build_parent_child_chunks("doc-x", text, "xlsx");
        assert_eq!(chunks.children.len(), 2);
        assert!(chunks.children[1].child_text.contains("市场,李四"));
    }

    #[test]
    fn structured_chunks_accept_uppercase_sheet_marker() {
        let text = "[SHEET] 花名册\n部门,姓名\n研发,张三";
        let parents = split_structured_text_chunks(text, 500);
        assert_eq!(parents.len(), 1);
        assert!(parents[0].starts_with("[Sheet] 花名册"));
        assert!(parents[0].contains("研发,张三"));
        assert!(eks_kernels::structured_sheet_name("[sHeEt] Roster") == Some("Roster"));
    }

    fn make_docx(document_xml: &str) -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("word/document.xml", options).unwrap();
        zip.write_all(document_xml.as_bytes()).unwrap();
        zip.finish().unwrap().into_inner()
    }

    fn make_pptx(slides: &[(&str, &str)]) -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default();
        for (name, xml) in slides {
            zip.start_file(*name, options).unwrap();
            zip.write_all(xml.as_bytes()).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    fn slide_xml(text: &str) -> String {
        format!(
            r#"<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"#
        )
    }

    #[test]
    fn docx_extracts_paragraphs_and_tables_in_order() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>前言  段落</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>部门</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>负责人</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>研发</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>张三</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
    <w:p><w:r><w:t>结尾</w:t></w:r></w:p>
  </w:body>
</w:document>"#;
        let text = extract_text("手册.docx", &make_docx(xml)).unwrap();
        assert_eq!(
            text,
            "前言 段落\n\n[表格]\n部门 | 负责人\n研发 | 张三\n\n结尾"
        );
    }

    fn sample_pdf_bytes(text: &str) -> Vec<u8> {
        use lopdf::{Dictionary, Document, Object, Stream};
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();

        let mut font = Dictionary::new();
        font.set("Type", "Font");
        font.set("Subtype", "Type1");
        font.set("BaseFont", "Helvetica");
        let font_id = doc.add_object(font);

        let mut fonts = Dictionary::new();
        fonts.set("F1", font_id);
        let mut resources = Dictionary::new();
        resources.set("Font", fonts);
        let resources_id = doc.add_object(resources);

        let content = format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET");
        let content_id = doc.add_object(Stream::new(Dictionary::new(), content.into_bytes()));

        let mut page = Dictionary::new();
        page.set("Type", "Page");
        page.set("Parent", pages_id);
        page.set("Contents", content_id);
        page.set("Resources", resources_id);
        page.set(
            "MediaBox",
            vec![
                Object::Integer(0),
                Object::Integer(0),
                Object::Integer(612),
                Object::Integer(792),
            ],
        );
        let page_id = doc.add_object(page);

        let mut pages = Dictionary::new();
        pages.set("Type", "Pages");
        pages.set("Kids", vec![Object::from(page_id)]);
        pages.set("Count", 1);
        doc.objects.insert(pages_id, Object::Dictionary(pages));

        let mut catalog = Dictionary::new();
        catalog.set("Type", "Catalog");
        catalog.set("Pages", pages_id);
        let catalog_id = doc.add_object(catalog);
        doc.trailer.set("Root", catalog_id);

        let mut out = Vec::new();
        doc.save_to(&mut out).unwrap();
        out
    }

    #[test]
    fn pdf_extract_falls_back_to_lopdf() {
        let bytes = sample_pdf_bytes("HelloPDF");
        let from_lopdf = extract_pdf_with_lopdf(&bytes).unwrap();
        assert!(from_lopdf.contains("HelloPDF"), "{from_lopdf}");
        let extracted = extract_pdf_text(&bytes).unwrap();
        assert!(extracted.contains("HelloPDF"), "{extracted}");
        let err = extract_pdf_text(b"not-a-pdf").unwrap_err();
        assert!(err.contains("PDF 解析失败"), "{err}");
    }

    #[test]
    fn article_sections_prefix_parent_chunks() {
        let text = "# 差旅制度\n出差需提前申请。\n## 报销\n票据必须原件。\n第一章 总则\n本规定适用于全公司。\n一、适用范围\n全体员工。\n（一）例外\n实习生除外。";
        assert_eq!(parse_section_heading("# 差旅制度"), Some((1, "差旅制度".into())));
        assert_eq!(parse_section_heading("第一章 总则").unwrap().0, 1);
        assert_eq!(parse_section_heading("一、适用范围").unwrap().0, 1);
        assert_eq!(parse_section_heading("（一）例外").unwrap().0, 2);
        let chunks = build_parent_child_chunks("doc-a", text, "md");
        assert!(chunks.parents.iter().any(|item| item.parent_text.contains("[章节] 差旅制度")));
        assert!(chunks.parents.iter().any(|item| item.parent_text.contains("[章节] 差旅制度 / 报销")));
        assert!(chunks.parents.iter().any(|item| item.parent_text.contains("[章节] 总则")
            || item.parent_text.contains("[章节] 第一章 总则")));
        let default_chunks = split_document_text(text, DOCUMENT_CHUNK_SIZE, DOCUMENT_CHUNK_OVERLAP, "md");
        assert!(default_chunks.iter().all(|item| item.chars().count() <= DOCUMENT_ARTICLE_CHUNK_SIZE + 80));
        let plain = build_parent_child_chunks("doc-b", "没有标题的普通说明文字。", "md");
        assert!(!plain.parents[0].parent_text.contains("[章节]"));
    }

    #[test]
    fn pptx_extracts_slides_in_number_order() {
        assert_eq!(pptx_slide_number("ppt/slides/slide2.xml"), Some(2));
        assert_eq!(pptx_slide_number("ppt/slideLayouts/slideLayout1.xml"), None);
        let bytes = make_pptx(&[
            ("ppt/slides/slide2.xml", &slide_xml("第二页")),
            ("ppt/slides/slide1.xml", &slide_xml("第一页")),
        ]);
        assert_eq!(extract_text("介绍.pptx", &bytes).unwrap(), "第一页\n\n第二页");
    }

    #[test]
    fn html_strips_tags_and_skips_script() {
        let html = "<html><head><style>p{color:red}</style></head><body><h1>制度</h1><p>出差需<b>提前</b>申请。</p><script>alert(1)</script></body></html>";
        let text = extract_text("制度.html", html.as_bytes()).unwrap();
        assert!(text.contains("制度"));
        assert!(text.contains("出差需提前申请。"));
        assert!(!text.contains("alert"));
        assert!(!text.contains("color:red"));
    }

    #[test]
    fn recursive_split_breaks_on_paragraph_separators() {
        let text = "第一段说明。\n\n第二段内容需要单独成块。\n\n第三段收尾。";
        let chunks = split_text(text, 18, 4);
        assert!(chunks.len() >= 2, "{chunks:?}");
        assert!(chunks.iter().any(|chunk| chunk.contains("第一段")));
        assert!(chunks.iter().any(|chunk| chunk.contains("第二段")));
        let joined = chunks.join("");
        assert!(joined.contains("第三段"));
        assert!(!is_extractable_type("doc"));
        assert!(!is_extractable_type("ppt"));
        assert!(is_extractable_type("pptx"));
        assert!(is_extractable_type("html"));
    }
}
