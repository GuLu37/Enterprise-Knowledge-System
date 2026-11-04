//! 文档路由（列表/正文/上传/删除）。

use axum::{
    Json,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::auth::AuthError;
use crate::db;
use crate::routes::require_current_user;
use crate::state::AppState;

#[derive(Debug, sqlx::FromRow)]
pub struct DocumentRow {
    pub document_id: String,
    pub original_filename: String,
    pub stored_filename: String,
    pub file_path: String,
    pub content_type: Option<String>,
    pub file_type: String,
    pub file_size: i64,
    pub status: String,
    pub chunk_count: i64,
    pub error_message: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

const DOCUMENT_COLUMNS: &str = "document_id, original_filename, stored_filename, file_path, \
     content_type, file_type, file_size, status, chunk_count, error_message, created_at, updated_at";

#[derive(Debug, Deserialize)]
pub struct DocumentListQuery {
    #[serde(default)]
    pub skip: i64,
    pub limit: Option<i64>,
}

fn doc_to_json(row: &DocumentRow) -> Value {
    json!({
        "document_id": row.document_id,
        "original_filename": row.original_filename,
        "stored_filename": row.stored_filename,
        "file_path": row.file_path,
        "content_type": row.content_type,
        "file_type": row.file_type,
        "file_size": row.file_size,
        "status": row.status,
        "chunk_count": row.chunk_count,
        "error_message": row.error_message,
        "created_at": db::dt_to_iso(&row.created_at),
        "updated_at": db::dt_to_iso(&row.updated_at),
    })
}

pub fn guess_content_type(file_type: &str) -> &'static str {
    match file_type.trim().trim_start_matches('.').to_lowercase().as_str() {
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "md" => "text/markdown",
        "csv" | "tsv" => "text/csv",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        _ => "application/octet-stream",
    }
}

async fn list_documents_page(
    state: &AppState,
    headers: &HeaderMap,
    skip: i64,
    limit: i64,
) -> Result<Json<Value>, AuthError> {
    require_current_user(state, headers).await?;
    let skip = skip.max(0);
    let limit = limit.clamp(1, 1000);
    let total: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM documents")
        .fetch_one(&state.pool)
        .await
        .map_err(AuthError::from)?;
    let sql = format!(
        "SELECT {DOCUMENT_COLUMNS} FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?"
    );
    let rows: Vec<DocumentRow> = sqlx::query_as(&sql)
        .bind(limit)
        .bind(skip)
        .fetch_all(&state.pool)
        .await
        .map_err(AuthError::from)?;
    let documents: Vec<Value> = rows.iter().map(doc_to_json).collect();
    Ok(Json(json!({
        "documents": documents,
        "total": total,
        "skip": skip,
        "limit": limit,
    })))
}

// GET /api/v1/documents — 保留旧路径，默认返回较多条，避免既有调用只看到第一页。
pub async fn list_documents(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<DocumentListQuery>,
) -> Result<Json<Value>, AuthError> {
    list_documents_page(&state, &headers, query.skip, query.limit.unwrap_or(1000)).await
}

// GET /api/v1/documents/list — 与前端、Python 契约一致。
pub async fn list_documents_compat(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<DocumentListQuery>,
) -> Result<Json<Value>, AuthError> {
    list_documents_page(&state, &headers, query.skip, query.limit.unwrap_or(10).clamp(1, 100)).await
}

fn deletion_rejected(status: &str) -> Option<String> {
    match status.trim() {
        "deleting" | "deleted" => Some(format!("文档当前状态为 {status}，无法重复删除")),
        _ => None,
    }
}

async fn load_document(
    pool: &sqlx::SqlitePool,
    document_id: &str,
) -> Result<Option<DocumentRow>, AuthError> {
    sqlx::query_as(&format!(
        "SELECT {DOCUMENT_COLUMNS} FROM documents WHERE document_id = ?"
    ))
    .bind(document_id)
    .fetch_optional(pool)
    .await
    .map_err(AuthError::from)
}

async fn mark_document_deleting(
    pool: &sqlx::SqlitePool,
    document_id: &str,
) -> Result<DocumentRow, AuthError> {
    let Some(row) = load_document(pool, document_id).await? else {
        return Err(AuthError::bad_request("文档不存在"));
    };
    if let Some(detail) = deletion_rejected(&row.status) {
        return Err(AuthError::bad_request(&detail));
    }
    let now = db::fmt_dt(&db::now_naive());
    sqlx::query(
        "UPDATE documents SET status = 'deleting', error_message = NULL, updated_at = ? \
         WHERE document_id = ?",
    )
    .bind(&now)
    .bind(document_id)
    .execute(pool)
    .await
    .map_err(AuthError::from)?;
    let mut updated = row;
    updated.status = "deleting".into();
    updated.error_message = None;
    updated.updated_at = now;
    Ok(updated)
}

fn remove_document_files(sqlite_path: &str, row: &DocumentRow) -> bool {
    let upload_dir = std::path::Path::new(sqlite_path)
        .parent()
        .map(|parent| parent.join("uploads"))
        .unwrap_or_default();
    let removed_stored = std::fs::remove_file(upload_dir.join(&row.stored_filename)).is_ok();
    let removed_dir = std::fs::remove_dir_all(upload_dir.join(&row.document_id)).is_ok();
    let removed_path = !row.file_path.is_empty() && std::fs::remove_file(&row.file_path).is_ok();
    removed_stored || removed_dir || removed_path
}

async fn restore_document_after_delete_failure(
    pool: &sqlx::SqlitePool,
    document_id: &str,
    err: &str,
) {
    let now = db::fmt_dt(&db::now_naive());
    if let Err(restore_err) = sqlx::query(
        "UPDATE documents SET status = 'ready', error_message = ?, updated_at = ? \
         WHERE document_id = ?",
    )
    .bind(format!("删除失败: {err}"))
    .bind(&now)
    .bind(document_id)
    .execute(pool)
    .await
    {
        tracing::error!(document_id, "文档删除失败后回写状态失败: {restore_err}");
    }
}

async fn finalize_document_deletion(state: &AppState, document_id: &str) -> Result<(), String> {
    let row = load_document(&state.pool, document_id)
        .await
        .map_err(|err| err.detail)?;
    let Some(row) = row else {
        return Ok(());
    };
    if row.status == "deleted" {
        return Ok(());
    }

    tracing::info!(document_id, "开始删除 Milvus 文档切块");
    let milvus_deleted = state.engine.delete_document_chunks(document_id).await?;
    tracing::info!(document_id, milvus_deleted, "Milvus 文档切块删除完成");

    let file_deleted = remove_document_files(&state.config.sqlite_path, &row);
    tracing::info!(document_id, file_deleted, "本地文件删除完成");

    sqlx::query("DELETE FROM document_chunks WHERE document_id = ?")
        .bind(document_id)
        .execute(&state.pool)
        .await
        .map_err(|err| err.to_string())?;
    sqlx::query("DELETE FROM documents WHERE document_id = ?")
        .bind(document_id)
        .execute(&state.pool)
        .await
        .map_err(|err| err.to_string())?;
    state.engine.invalidate().await;
    tracing::info!(document_id, "文档删除完成");
    Ok(())
}

fn deletion_accepted_payload(row: &DocumentRow) -> Value {
    let mut payload = doc_to_json(row);
    if let Some(object) = payload.as_object_mut() {
        object.insert("status".into(), json!("deleting"));
        object.insert("milvus_deleted".into(), json!(false));
        object.insert("file_deleted".into(), json!(false));
    }
    payload
}

// DELETE /api/v1/documents/{document_id}
pub async fn delete_document(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(document_id): Path<String>,
) -> Result<(StatusCode, Json<Value>), AuthError> {
    require_current_user(&state, &headers).await?;
    let row = mark_document_deleting(&state.pool, &document_id).await?;
    let payload = deletion_accepted_payload(&row);
    let state = state.clone();
    let document_id = document_id.clone();
    tokio::spawn(async move {
        if let Err(err) = finalize_document_deletion(&state, &document_id).await {
            tracing::error!(document_id = %document_id, "文档后台删除失败: {err}");
            restore_document_after_delete_failure(&state.pool, &document_id, &err).await;
        }
    });
    Ok((StatusCode::ACCEPTED, Json(payload)))
}

// GET /api/v1/documents/content/{document_id}
pub async fn get_document_content(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(document_id): Path<String>,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    let row: Option<DocumentRow> = sqlx::query_as(&format!(
        "SELECT {DOCUMENT_COLUMNS} FROM documents WHERE document_id = ?"
    ))
    .bind(&document_id)
    .fetch_optional(&state.pool)
    .await
    .map_err(AuthError::from)?;
    let Some(row) = row else {
        return Err(AuthError {
            status: axum::http::StatusCode::NOT_FOUND,
            detail: "文档不存在".into(),
        });
    };

    let chunks: Vec<String> = sqlx::query_scalar(
        "SELECT chunk_text FROM document_chunks WHERE document_id = ? ORDER BY chunk_index ASC",
    )
    .bind(&document_id)
    .fetch_all(&state.pool)
    .await
    .map_err(AuthError::from)?;
    let mut content = chunks
        .into_iter()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if content.is_empty() && !row.file_path.is_empty() {
        if let Ok(bytes) = std::fs::read(&row.file_path) {
            if let Ok(text) = crate::ingest::extract_text(&row.original_filename, &bytes) {
                content = text;
            }
        }
    }
    if content.trim().is_empty() {
        return Err(AuthError::bad_request("文档正文不存在或尚未处理完成"));
    }
    Ok(Json(json!({
        "document_id": row.document_id,
        "original_filename": row.original_filename,
        "file_type": row.file_type,
        "status": row.status,
        "content": content,
        "content_type": row.content_type,
        "updated_at": db::dt_to_iso(&row.updated_at),
    })))
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
    let content_type = guess_content_type(&file_type);
    let stored_filename = format!("{document_id}.{file_type}");
    let upload_dir = std::path::Path::new(&state.config.sqlite_path)
        .parent()
        .map(|parent| parent.join("uploads"))
        .unwrap_or_default();
    let _ = std::fs::create_dir_all(&upload_dir);
    let file_path = upload_dir.join(&stored_filename);
    std::fs::write(&file_path, &file_data)
        .map_err(|err| AuthError::bad_request(&format!("保存上传文件失败: {err}")))?;
    let file_path_text = file_path.display().to_string();

    // 1) 文本提取
    let text = match crate::ingest::extract_text(&filename, &file_data) {
        Ok(text) => text,
        Err(err) => {
            let _ = std::fs::remove_file(&file_path);
            return Err(AuthError::bad_request(&format!("文档解析失败: {err}")));
        }
    };

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
        content_type,
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
    .bind(&stored_filename)
    .bind(&file_path_text)
    .bind(content_type)
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

    state.engine.invalidate().await;

    Ok(Json(json!({
        "document_id": document_id,
        "original_filename": filename,
        "stored_filename": stored_filename,
        "file_path": file_path_text,
        "content_type": content_type,
        "file_type": file_type,
        "file_size": file_data.len(),
        "status": "ready",
        "chunk_count": chunks.children.len(),
        "error_message": null,
        "created_at": db::dt_to_iso(&now),
        "updated_at": db::dt_to_iso(&now),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    fn sample_row(status: &str) -> DocumentRow {
        DocumentRow {
            document_id: "doc-1".into(),
            original_filename: "手册.md".into(),
            stored_filename: "doc-1.md".into(),
            file_path: "/tmp/doc-1.md".into(),
            content_type: Some("text/markdown".into()),
            file_type: "md".into(),
            file_size: 12,
            status: status.into(),
            chunk_count: 3,
            error_message: None,
            created_at: "2026-01-01 00:00:00.000000".into(),
            updated_at: "2026-01-01 00:00:00.000000".into(),
        }
    }

    async fn memory_pool() -> sqlx::SqlitePool {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(
            "CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                content_type TEXT,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                status TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        pool
    }

    async fn insert_row(pool: &sqlx::SqlitePool, row: &DocumentRow) {
        sqlx::query(
            "INSERT INTO documents (
                document_id, original_filename, stored_filename, file_path, content_type,
                file_type, file_size, status, chunk_count, error_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&row.document_id)
        .bind(&row.original_filename)
        .bind(&row.stored_filename)
        .bind(&row.file_path)
        .bind(&row.content_type)
        .bind(&row.file_type)
        .bind(row.file_size)
        .bind(&row.status)
        .bind(row.chunk_count)
        .bind(&row.error_message)
        .bind(&row.created_at)
        .bind(&row.updated_at)
        .execute(pool)
        .await
        .unwrap();
    }

    #[test]
    fn rejects_deleting_or_deleted_status() {
        assert_eq!(
            deletion_rejected("deleting").as_deref(),
            Some("文档当前状态为 deleting，无法重复删除")
        );
        assert_eq!(
            deletion_rejected("deleted").as_deref(),
            Some("文档当前状态为 deleted，无法重复删除")
        );
        assert!(deletion_rejected("ready").is_none());
        assert!(deletion_rejected("processing").is_none());
        assert!(deletion_rejected("failed").is_none());
    }

    #[test]
    fn accepted_payload_keeps_python_request_shape() {
        let payload = deletion_accepted_payload(&sample_row("ready"));
        assert_eq!(payload["status"], json!("deleting"));
        assert_eq!(payload["milvus_deleted"], json!(false));
        assert_eq!(payload["file_deleted"], json!(false));
        assert_eq!(payload["document_id"], json!("doc-1"));
        assert!(payload.get("deleted").is_none());
    }

    #[tokio::test]
    async fn mark_missing_document_is_bad_request() {
        let pool = memory_pool().await;
        let err = mark_document_deleting(&pool, "missing").await.unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.detail, "文档不存在");
    }

    #[tokio::test]
    async fn mark_repeat_delete_is_rejected() {
        let pool = memory_pool().await;
        insert_row(&pool, &sample_row("deleting")).await;
        let err = mark_document_deleting(&pool, "doc-1").await.unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.detail, "文档当前状态为 deleting，无法重复删除");
    }

    #[tokio::test]
    async fn marks_ready_document_deleting() {
        let pool = memory_pool().await;
        insert_row(&pool, &sample_row("ready")).await;
        let updated = mark_document_deleting(&pool, "doc-1").await.unwrap();
        assert_eq!(updated.status, "deleting");
        assert!(updated.error_message.is_none());
        let status: String = sqlx::query_scalar("SELECT status FROM documents WHERE document_id = ?")
            .bind("doc-1")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(status, "deleting");
    }

    #[tokio::test]
    async fn restore_after_delete_failure_rewrites_ready() {
        let pool = memory_pool().await;
        insert_row(&pool, &sample_row("deleting")).await;
        restore_document_after_delete_failure(&pool, "doc-1", "向量库不可用").await;
        let row = load_document(&pool, "doc-1").await.unwrap().unwrap();
        assert_eq!(row.status, "ready");
        assert_eq!(row.error_message.as_deref(), Some("删除失败: 向量库不可用"));
    }

    #[test]
    fn remove_document_files_deletes_stored_and_path() {
        let root = std::env::temp_dir().join(format!(
            "eks-doc-delete-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        let upload_dir = root.join("uploads");
        std::fs::create_dir_all(&upload_dir).unwrap();
        let stored = upload_dir.join("doc-1.md");
        let extra = root.join("extra.md");
        std::fs::write(&stored, b"hello").unwrap();
        std::fs::write(&extra, b"hello").unwrap();
        let mut row = sample_row("deleting");
        row.stored_filename = "doc-1.md".into();
        row.file_path = extra.display().to_string();
        let sqlite_path = root.join("rag_metadata.db");
        assert!(remove_document_files(sqlite_path.to_str().unwrap(), &row));
        assert!(!stored.exists());
        assert!(!extra.exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
