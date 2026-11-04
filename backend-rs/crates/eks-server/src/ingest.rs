//! 文档入库：上传 → 文本提取 → Parent-Child 切块 → 向量化 → milvus_lite 追加 + SQLite 元数据。
//!
//! 支持格式: txt, md (直接读取); docx (zip+xml); xlsx (calamine); pdf (pdf-extract)。
//! milvus_lite 写入: 新 Parquet data 文件 + manifest 版本更新（对齐 milvus_lite 存储格式）。

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema, Fields};
use parquet::arrow::ArrowWriter;
use serde_json::json;

use crate::auth::AuthError;
use crate::state::AppState;

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
const STRUCTURED_FILE_TYPES: [&str; 4] = ["csv", "tsv", "xls", "xlsx"];
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
        "csv" | "tsv" => {
            String::from_utf8(data.to_vec()).map_err(|e| format!("UTF-8 解码失败: {e}"))
        }
        "docx" => extract_docx_text(data),
        "xlsx" | "xls" => extract_xlsx_text(data),
        "pdf" => extract_pdf_text(data),
        "pptx" => extract_pptx_text(data),
        _ => Err(format!("不支持的文件类型: .{ext}")),
    }
}

/// docx: zip 中的 word/document.xml，提取 <w:t> 文本。
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

    // 简易 XML 文本提取: <w:t> 标签内容 + <w:p> 换行
    let mut text = String::new();
    let mut in_paragraph = false;
    let mut rest = xml.as_str();
    while let Some(pos) = rest.find('<') {
        let tag_end = rest[pos..].find('>').map(|i| pos + i + 1).unwrap_or(rest.len());
        let tag = &rest[pos..tag_end];
        if tag.starts_with("<w:p ") || tag == "<w:p>" || tag.starts_with("<w:p/") {
            in_paragraph = true;
        } else if tag.starts_with("</w:p>") {
            if in_paragraph {
                text.push('\n');
            }
            in_paragraph = false;
        } else if tag.starts_with("<w:t") {
            // 提取 <w:t>...</w:t> 之间的文本
            if let Some(close) = rest[tag_end..].find("</w:t>") {
                text.push_str(&rest[tag_end..tag_end + close]);
            }
        }
        rest = &rest[tag_end..];
    }
    Ok(text)
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

/// pdf: 用 pdf-extract 从内容流提取文字。扫描件没有文本层时会失败。
fn extract_pdf_text(data: &[u8]) -> Result<String, String> {
    let text = pdf_extract::extract_text_from_mem(data)
        .map_err(|err| format!("PDF 解析失败: {err}"))?;
    let cleaned = text.replace('\0', "");
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        return Err("PDF 未提取到文本，可能是扫描件".into());
    }
    Ok(trimmed.to_string())
}

/// pptx: zip 中的 ppt/slides/slideN.xml。
fn extract_pptx_text(data: &[u8]) -> Result<String, String> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| format!("pptx zip 打开失败: {e}"))?;
    let mut text = String::new();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
        let name = file.name().to_string();
        if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
            let mut xml = String::new();
            std::io::Read::read_to_string(&mut file, &mut xml)
                .map_err(|e| e.to_string())?;
            // 提取 <a:t> 标签
            let mut rest = xml.as_str();
            while let Some(pos) = rest.find("<a:t>") {
                let after = &rest[pos + 5..];
                if let Some(close) = after.find("</a:t>") {
                    text.push_str(&after[..close]);
                    text.push('\n');
                }
                rest = &after[after.find("</a:t>").map(|i| i + 6).unwrap_or(after.len())..];
            }
        }
    }
    Ok(text)
}

// ==================== 切块 ====================

/// 简化版递归文本切分（对齐 split_text_chunks 的核心逻辑）。
fn split_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        // 尝试在段落/句子/空格处断开
        let break_at = if end < chars.len() {
            let window = &chars[(end - 50).max(start)..end];
            let mut best = end;
            for (i, c) in window.iter().enumerate().rev() {
                if *c == '\n' || *c == '。' || *c == '.' || *c == '；' || *c == ';' {
                    best = (end - 50).max(start) + i + 1;
                    break;
                }
            }
            best
        } else {
            end
        };

        let chunk: String = chars[start..break_at].iter().collect();
        let trimmed = chunk.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }

        if break_at >= chars.len() {
            break;
        }
        start = if break_at > overlap {
            break_at - overlap
        } else {
            break_at
        };
    }
    chunks
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
        if let Some(name) = line
            .strip_prefix("[Sheet]")
            .or_else(|| line.strip_prefix("[sheet]"))
        {
            flush(&mut current_name, &mut current_lines, &mut sections);
            current_name = name.trim().to_string();
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
    let is_structured = STRUCTURED_FILE_TYPES.contains(&file_type.to_lowercase().as_str());
    let parent_texts = if is_structured {
        let structured = split_structured_text_chunks(text, PARENT_CHUNK_SIZE);
        if structured.is_empty() {
            split_text(text, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)
        } else {
            structured
        }
    } else {
        split_text(text, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)
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
            split_text(parent_text, CHILD_CHUNK_SIZE, overlap)
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

    #[test]
    fn decodes_utf8_and_gbk_text_files() {
        assert_eq!(extract_text("手册.txt", "你好".as_bytes()).unwrap(), "你好");
        let (gbk, _, _) = encoding_rs::GBK.encode("差旅报销");
        assert_eq!(extract_text("手册.md", &gbk).unwrap(), "差旅报销");
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
}
