//! 文档入库：上传 → 文本提取 → Parent-Child 切块 → 向量化 → milvus_lite 追加 + SQLite 元数据。
//!
//! 支持格式: txt, md (直接读取); docx (zip+xml); xlsx (calamine); pdf (pdf-extract)。
//! milvus_lite 写入: 新 Parquet data 文件 + manifest 版本更新（对齐 milvus_lite 存储格式）。

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
        "txt" | "md" | "csv" | "tsv" => {
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

/// xlsx: 用 calamine 读所有 sheet。
fn extract_xlsx_text(data: &[u8]) -> Result<String, String> {
    use calamine::{Reader, Xlsx, Data};
    let cursor = std::io::Cursor::new(data);

    let mut workbook = Xlsx::new(cursor)
        .map_err(|e| format!("xlsx 打开失败: {e}"))?;

    let mut text = String::new();
    for range in workbook.worksheets() {
        let (title, range) = range;
        text.push_str(&format!("[Sheet] {}
", title));
        for row in range.rows() {
            let cells: Vec<String> = row
                .iter()
                .map(|cell| match cell {
                    Data::Empty => String::new(),
                    _ => cell.to_string(),
                })
                .collect();
            text.push_str(&cells.join(","));
            text.push('\n');
        }
        text.push('\n');
    }
    Ok(text)
}

/// pdf: 用 pdf-extract crate（简化实现）。
fn extract_pdf_text(data: &[u8]) -> Result<String, String> {
    // pdf-extract 在纯 Rust 环境下对中文 PDF 支持有限
    // 生产环境建议用外部工具或 pypdf 兜底
    let _ = data;
    Err("PDF 解析暂未在 Rust 后端实现，请使用 Python 后端上传 PDF".into())
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

/// xlsx 行级切块: 每个 [Sheet] 段落作为 parent，每行作为 child。
fn split_structured_rows(text: &str, _chunk_size: usize) -> Vec<String> {
    let mut children = Vec::new();
    let mut current_sheet = String::new();
    let mut header = String::new();
    let mut first_data = true;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("[Sheet]") {
            current_sheet = line.to_string();
            header = String::new();
            first_data = true;
            continue;
        }
        if first_data {
            header = line.to_string();
            first_data = false;
            continue;
        }
        // 数据行 → child = sheet名 + 表头 + 行
        children.push(format!("{}\n{}\n{}", current_sheet, header, line));
    }
    if children.is_empty() {
        children.push(text.to_string());
    }
    children
}

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
    let parent_texts = split_text(text, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP);
    let is_structured = STRUCTURED_FILE_TYPES.contains(&file_type.to_lowercase().as_str());

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
            split_structured_rows(parent_text, CHILD_CHUNK_SIZE)
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
