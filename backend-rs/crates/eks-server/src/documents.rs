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
