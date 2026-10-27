//! 文档与技能路由（列表/删除/技能发现）。

use axum::{
    Json,
    extract::{Path, State},
    http::HeaderMap,
};
use serde_json::{Value, json};

use crate::auth::AuthError;
use crate::db;
use crate::routes::require_current_user;
use crate::state::AppState;

#[derive(Debug, sqlx::FromRow)]
pub struct DocumentRow {
    pub document_id: String,
    pub original_filename: String,
    pub file_type: String,
    pub file_size: i64,
    pub status: String,
    pub chunk_count: i64,
    pub error_message: Option<String>,
    pub created_at: String,
}

fn doc_to_json(row: &DocumentRow) -> Value {
    json!({
        "document_id": row.document_id,
        "original_filename": row.original_filename,
        "file_type": row.file_type,
        "file_size": row.file_size,
        "status": row.status,
        "chunk_count": row.chunk_count,
        "error_message": row.error_message,
        "created_at": db::dt_to_iso(&row.created_at),
    })
}

// GET /api/v1/documents
pub async fn list_documents(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    let rows: Vec<DocumentRow> = sqlx::query_as(
        "SELECT document_id, original_filename, file_type, file_size, status, \
         chunk_count, error_message, created_at FROM documents ORDER BY created_at DESC",
    )
    .fetch_all(&state.pool)
    .await
    .map_err(AuthError::from)?;
    let documents: Vec<Value> = rows.iter().map(doc_to_json).collect();
    Ok(Json(json!({ "documents": documents, "total": documents.len() })))
}

// DELETE /api/v1/documents/{document_id}
pub async fn delete_document(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(document_id): Path<String>,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;

    // 查是否存在
    let row: Option<DocumentRow> = sqlx::query_as(
        "SELECT document_id, original_filename, file_type, file_size, status, \
         chunk_count, error_message, created_at FROM documents WHERE document_id = ?",
    )
    .bind(&document_id)
    .fetch_optional(&state.pool)
    .await
    .map_err(AuthError::from)?;

    let Some(row) = row else {
        return Err(AuthError::not_found(&document_id));
    };

    // 删除 SQLite 中的 chunk 记录
    sqlx::query("DELETE FROM document_chunks WHERE document_id = ?")
        .bind(&document_id)
        .execute(&state.pool)
        .await
        .map_err(AuthError::from)?;

    // 删除文档元数据
    sqlx::query("DELETE FROM documents WHERE document_id = ?")
        .bind(&document_id)
        .execute(&state.pool)
        .await
        .map_err(AuthError::from)?;

    // 删除本地文件（尽力而为）
    if !row.original_filename.is_empty() {
        let upload_dir = std::path::Path::new(&state.config.sqlite_path)
            .parent()
            .map(|p| p.join("uploads"))
            .unwrap_or_default();
        let _ = std::fs::remove_dir_all(&upload_dir.join(&document_id));
    }

    // 失效检索缓存
    state.engine.invalidate().await;

    Ok(Json(json!({
        "document_id": document_id,
        "original_filename": row.original_filename,
        "deleted": true,
    })))
}

// GET /api/v1/skills
pub async fn list_skills(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    // 工具 schema 对齐 Python skills/registry.py 的 function calling 目录
    Ok(Json(json!({
        "skills": [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "在企业知识库中检索相关文档片段。支持按制度名、流程、编号、人名等关键词检索，也可按文件名查询某份文档是否存在。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "检索关键词或文件名"},
                            "top_k": {"type": "integer", "description": "返回条数(1-50)", "default": 5},
                            "retrieval_method": {"type": "string", "enum": ["dense", "sparse", "hybrid"], "description": "检索方式", "default": "hybrid"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_knowledge_base_files",
                    "description": "列出当前知识库中已上传的所有文档文件名和状态。",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_available_tools",
                    "description": "列出当前系统可用的全部工具及其参数说明。",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
    })))
}

impl AuthError {
    pub fn not_found(detail: &str) -> Self {
        AuthError {
            status: axum::http::StatusCode::NOT_FOUND,
            detail: format!("资源不存在: {detail}"),
        }
    }
}

// POST /api/v1/documents/upload (multipart)
pub async fn upload_document(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: axum::extract::Multipart,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;

    // 读取第一个文件字段
    let mut filename = String::new();
    let mut file_data: Vec<u8> = Vec::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AuthError::bad_request(&format!("multipart 读取失败: {e}")))?
    {
        if field.name() == Some("file") {
            filename = field.file_name().unwrap_or("unknown").to_string();
            file_data = field
                .bytes()
                .await
                .map_err(|e| AuthError::bad_request(&format!("文件读取失败: {e}")))?
                .to_vec();
            break;
        }
    }

    if filename.is_empty() || file_data.is_empty() {
        return Err(AuthError::bad_request("文件不能为空"));
    }

    let document_id = uuid::Uuid::new_v4().simple().to_string();
    let file_type = filename.rsplit('.').next().unwrap_or("").to_lowercase();

    // 1) 文本提取
    let text = crate::ingest::extract_text(&filename, &file_data)
        .map_err(|e| AuthError::bad_request(&format!("文档解析失败: {e}")))?;

    if text.trim().is_empty() {
        return Err(AuthError::bad_request("文档内容为空，无法生成向量"));
    }

    // 2) Parent-Child 切块
    let chunks = crate::ingest::build_parent_child_chunks(&document_id, &text, &file_type);
    if chunks.children.is_empty() {
        return Err(AuthError::bad_request("文档切块失败，未生成有效内容"));
    }

    // 3) 向量化
    let child_texts: Vec<String> = chunks
        .children
        .iter()
        .map(|c| c.child_text.clone())
        .collect();
    let vectors = state
        .engine
        .embedder
        .embed_documents(&child_texts)
        .map_err(|e| AuthError {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            detail: format!("向量化失败: {e}"),
        })?;

    // 4) milvus_lite 追加写入
    crate::ingest::append_to_milvus_lite(
        &state.engine.store,
        "doc_chunk",
        &chunks.children,
        &filename,
        &file_type,
        "application/octet-stream",
        &vectors,
    )
    .await
    .map_err(|e| AuthError {
        status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        detail: format!("向量库写入失败: {e}"),
    })?;

    // 5) SQLite 元数据
    let now = crate::db::fmt_dt(&crate::db::now_naive());
    sqlx::query(
        "INSERT INTO documents (document_id, original_filename, stored_filename, file_path,          content_type, file_type, file_size, status, chunk_count, error_message, created_at, updated_at)          VALUES (?, ?, ?, ?, ?, ?, ?, 'ready', ?, NULL, ?, ?)",
    )
    .bind(&document_id)
    .bind(&filename)
    .bind(&format!("{document_id}.{file_type}"))
    .bind("") // file_path 由 Python 端管理
    .bind("application/octet-stream")
    .bind(&file_type)
    .bind(file_data.len() as i64)
    .bind(chunks.children.len() as i64)
    .bind(&now)
    .bind(&now)
    .execute(&state.pool)
    .await
    .map_err(AuthError::from)?;

    // parent 行写入 document_chunks
    for parent in &chunks.parents {
        sqlx::query(
            "INSERT INTO document_chunks (document_id, chunk_index, parent_id, chunk_text, created_at)              VALUES (?, ?, ?, ?, ?)",
        )
        .bind(&parent.document_id)
        .bind(parent.parent_index)
        .bind(&parent.parent_id)
        .bind(&parent.parent_text)
        .bind(&now)
        .execute(&state.pool)
        .await
        .map_err(AuthError::from)?;
    }

    // 6) 失效缓存
    state.engine.invalidate().await;

    // 保存上传文件
    let upload_dir = std::path::Path::new(&state.config.sqlite_path)
        .parent()
        .map(|p| p.join("uploads"))
        .unwrap_or_default();
    let _ = std::fs::create_dir_all(&upload_dir);
    let _ = std::fs::write(upload_dir.join(&format!("{document_id}.{file_type}")), &file_data);

    Ok(Json(json!({
        "document_id": document_id,
        "original_filename": filename,
        "file_type": file_type,
        "file_size": file_data.len(),
        "status": "ready",
        "chunk_count": chunks.children.len(),
    })))
}
