//! 文档路由（列表/正文/上传/删除）。

use std::collections::HashSet;
use std::path::Path as FsPath;

use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::auth::AuthError;
use crate::db;
use crate::extract::ApiQuery;
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

fn sanitize_original_filename(raw: &str) -> String {
    raw.trim()
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

fn upload_file_type(filename: &str) -> String {
    FsPath::new(filename)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .trim()
        .trim_start_matches('.')
        .to_lowercase()
}

fn normalized_allowed_types(types: &[String]) -> HashSet<String> {
    types
        .iter()
        .map(|item| {
            item.trim()
                .trim_matches('"')
                .trim_matches('\'')
                .trim_start_matches('.')
                .to_lowercase()
        })
        .filter(|item| !item.is_empty())
        .collect()
}

fn reject_unsupported_file_type(file_type: &str, allowed: &[String]) -> Option<String> {
    let file_type = file_type.trim().trim_start_matches('.').to_lowercase();
    if !normalized_allowed_types(allowed).contains(&file_type)
        || !crate::ingest::is_extractable_type(&file_type)
    {
        Some(format!("不支持的文件类型: {file_type}"))
    } else {
        None
    }
}

fn reject_oversized_upload(size: usize, max_upload_size: i64) -> Option<&'static str> {
    if (size as i64) > max_upload_size {
        Some("文件超过最大上传限制")
    } else {
        None
    }
}

pub fn guess_content_type(file_type: &str) -> &'static str {
    match file_type.trim().trim_start_matches('.').to_lowercase().as_str() {
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "md" => "text/markdown",
        "html" | "htm" => "text/html",
        "csv" | "tsv" => "text/csv",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        _ => "application/octet-stream",
    }
}

/// 对齐 Python `UploadFile.content_type`：原样保留 multipart 声明，缺省为 None，不按扩展名猜测。
pub fn upload_content_type(declared: Option<&str>) -> Option<String> {
    declared.map(str::to_string)
}

/// 对齐 Python milvus 写入 `content_type or ""`。
pub fn milvus_content_type(declared: Option<&str>) -> String {
    declared.unwrap_or("").to_string()
}

async fn list_documents_page(
    state: &AppState,
    headers: &HeaderMap,
    skip: i64,
    limit: i64,
) -> Result<Json<Value>, AuthError> {
    require_current_user(state, headers).await?;
    let skip = crate::validate::document_list_skip(skip)?;
    let limit = limit.clamp(1, 1000);
    let total: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM documents")
        .fetch_one(&state.pool)
        .await
        .map_err(|err| document_sql_error("获取文档列表失败", err))?;
    let sql = format!(
        "SELECT {DOCUMENT_COLUMNS} FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?"
    );
    let rows: Vec<DocumentRow> = sqlx::query_as(&sql)
        .bind(limit)
        .bind(skip)
        .fetch_all(&state.pool)
        .await
        .map_err(|err| document_sql_error("获取文档列表失败", err))?;
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
    ApiQuery(query): ApiQuery<DocumentListQuery>,
) -> Result<Json<Value>, AuthError> {
    list_documents_page(&state, &headers, query.skip, query.limit.unwrap_or(1000)).await
}

// GET /api/v1/documents/list — 与前端、Python 契约一致。
pub async fn list_documents_compat(
    State(state): State<AppState>,
    headers: HeaderMap,
    ApiQuery(query): ApiQuery<DocumentListQuery>,
) -> Result<Json<Value>, AuthError> {
    let skip = crate::validate::document_list_skip(query.skip)?;
    let limit = crate::validate::document_list_limit(query.limit)?;
    list_documents_page(&state, &headers, skip, limit).await
}

/// 对齐 Python `_log_delete_step`：`[文档删除][n/4] {message} (document_id=…)`。
fn format_delete_step(document_id: &str, step: &str, message: &str) -> String {
    format!("[文档删除][{step}] {message} (document_id={document_id})")
}

/// 对齐 Python `logger.exception("[文档删除][n/4] …失败 (document_id=…): {exc}")`。
fn format_delete_fail(document_id: &str, step: &str, message: &str, err: &str) -> String {
    format!("[文档删除][{step}] {message} (document_id={document_id}): {err}")
}

fn log_delete_step(document_id: &str, step: &str, message: &str) {
    tracing::info!("{}", format_delete_step(document_id, step, message));
}

/// 对齐 Python `_log_upload_step`：`[文档上传][n/3] {message} (document_id=…)`。
fn format_upload_step(document_id: &str, step: &str, message: &str) -> String {
    format!("[文档上传][{step}] {message} (document_id={document_id})")
}

fn log_upload_step(document_id: &str, step: &str, message: &str) {
    tracing::info!("{}", format_upload_step(document_id, step, message));
}

fn format_upload_task_submitted(document_id: &str) -> String {
    format!("[文档上传] 后台处理任务已提交 (document_id={document_id})")
}

fn format_milvus_content_skip(error: &str, document_id: &str) -> String {
    format!("从 Milvus 读取文档正文切片失败，已跳过: {error} (document_id={document_id})")
}

fn deletion_rejected(status: &str) -> Option<String> {
    match status.trim() {
        "deleting" | "deleted" => Some(format!("文档当前状态为 {status}，无法重复删除")),
        _ => None,
    }
}

fn format_document_route_error(action: &str, err: &str) -> String {
    format!("{action}: {err}")
}

/// 对齐 Python `ingest_document` 的入库异常日志。
fn format_document_ingest_fail(document_id: &str, err: &str) -> String {
    format!("文档入库失败 (document_id={document_id}): {err}")
}

fn missing_upload_file() -> AuthError {
    AuthError::unprocessable_at(&["body", "file"], "Field required", "missing")
}

fn document_sql_error(action: &str, err: sqlx::Error) -> AuthError {
    // 对齐 Python 路由：500 detail 回传真实错误信息（仅日志再叠加动作前缀）。
    let detail = err.to_string();
    tracing::error!("{}", format_document_route_error(action, &detail));
    AuthError::http(StatusCode::INTERNAL_SERVER_ERROR, detail)
}

async fn load_document(
    pool: &sqlx::SqlitePool,
    document_id: &str,
) -> Result<Option<DocumentRow>, sqlx::Error> {
    sqlx::query_as(&format!(
        "SELECT {DOCUMENT_COLUMNS} FROM documents WHERE document_id = ?"
    ))
    .bind(document_id)
    .fetch_optional(pool)
    .await
}

async fn mark_document_deleting(
    pool: &sqlx::SqlitePool,
    document_id: &str,
) -> Result<DocumentRow, AuthError> {
    log_delete_step(document_id, "1/4", "收到删除请求");
    let Some(row) = load_document(pool, document_id)
        .await
        .map_err(|err| document_sql_error("文档删除失败", err))?
    else {
        return Err(AuthError::bad_request("文档不存在"));
    };
    if let Some(detail) = deletion_rejected(&row.status) {
        return Err(AuthError::bad_request(&detail));
    }
    log_delete_step(
        document_id,
        "1/4",
        &format!(
            "SQLite 元数据校验通过: status={}, file_path={}",
            row.status, row.file_path
        ),
    );
    let now = db::fmt_dt(&db::now_naive());
    sqlx::query(
        "UPDATE documents SET status = 'deleting', error_message = NULL, updated_at = ? \
         WHERE document_id = ?",
    )
    .bind(&now)
    .bind(document_id)
    .execute(pool)
    .await
    .map_err(|err| document_sql_error("文档删除失败", err))?;
    let mut updated = row;
    updated.status = "deleting".into();
    updated.error_message = None;
    updated.updated_at = now;
    log_delete_step(document_id, "1/4", "SQLite 状态已更新为 deleting");
    Ok(updated)
}

fn remove_one_file(path: &FsPath) -> Result<bool, String> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err.to_string()),
    }
}

fn remove_one_dir(path: &FsPath) -> Result<bool, String> {
    match std::fs::remove_dir_all(path) {
        Ok(()) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err.to_string()),
    }
}

/// 对齐 Python `file_path.unlink(missing_ok=True)`：找不到文件视为成功，其它 IO 错误中止删除。
fn remove_document_files(upload_dir: &FsPath, row: &DocumentRow) -> Result<bool, String> {
    let mut deleted = false;
    deleted |= remove_one_file(&upload_dir.join(&row.stored_filename))?;
    deleted |= remove_one_dir(&upload_dir.join(&row.document_id))?;
    if !row.file_path.is_empty() {
        deleted |= remove_one_file(FsPath::new(&row.file_path))?;
    }
    Ok(deleted)
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
        return;
    }
    log_delete_step(document_id, "1/4", "后台删除失败，文档状态已恢复为 ready");
}

async fn finalize_document_deletion(state: &AppState, document_id: &str) -> Result<(), String> {
    let row = load_document(&state.pool, document_id)
        .await
        .map_err(|err| err.to_string())?;
    // 对齐 Python：记录缺失或已删除时按删除失败处理（走失败日志与状态恢复路径）。
    let row = match row {
        Some(row) if row.status != "deleted" => row,
        Some(_) => return Err("文档当前状态为 deleted，无法重复删除".to_string()),
        None => return Err("文档不存在".to_string()),
    };

    log_delete_step(document_id, "1/4", "开始执行后台删除");
    log_delete_step(document_id, "2/4", "开始删除 Milvus 文档切块");
    match state.engine.delete_document_chunks(document_id).await {
        Ok(deleted) => log_delete_step(
            document_id,
            "2/4",
            &format!("Milvus 文档切块删除完成: deleted={deleted}"),
        ),
        Err(err) => {
            tracing::error!(
                "{}",
                format_delete_fail(document_id, "2/4", "Milvus 文档切块删除失败", &err)
            );
            return Err(err);
        }
    }

    let upload_dir = crate::config::resolve_upload_dir(
        &state.config.upload_dir,
        &state.config.sqlite_path,
    );
    log_delete_step(
        document_id,
        "3/4",
        &format!("开始删除本地文件: {}", row.file_path),
    );
    let file_deleted = match remove_document_files(&upload_dir, &row) {
        Ok(deleted) => deleted,
        Err(err) => {
            tracing::error!(
                "{}",
                format_delete_fail(document_id, "3/4", "本地文件删除失败", &err)
            );
            return Err(err);
        }
    };
    log_delete_step(
        document_id,
        "3/4",
        &format!("本地文件删除完成: deleted={file_deleted}"),
    );

    log_delete_step(document_id, "4/4", "开始删除 SQLite 文档记录");
    sqlx::query("DELETE FROM document_chunks WHERE document_id = ?")
        .bind(document_id)
        .execute(&state.pool)
        .await
        .map_err(|err| {
            tracing::error!(
                "{}",
                format_delete_fail(
                    document_id,
                    "4/4",
                    "SQLite 文档记录删除失败",
                    &err.to_string()
                )
            );
            err.to_string()
        })?;
    sqlx::query("DELETE FROM documents WHERE document_id = ?")
        .bind(document_id)
        .execute(&state.pool)
        .await
        .map_err(|err| {
            tracing::error!(
                "{}",
                format_delete_fail(
                    document_id,
                    "4/4",
                    "SQLite 文档记录删除失败",
                    &err.to_string()
                )
            );
            err.to_string()
        })?;
    log_delete_step(document_id, "4/4", "SQLite 文档记录删除完成");
    state.engine.invalidate().await;
    tracing::info!("✓ 文档删除完成 (document_id={document_id})");
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
            tracing::error!("文档删除失败 (document_id={document_id}): {err}");
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
    .map_err(|err| document_sql_error("获取文档内容失败", err))?;
    let Some(row) = row else {
        return Err(document_content_exception("文档不存在"));
    };

    let chunks: Vec<String> = sqlx::query_scalar(
        "SELECT chunk_text FROM document_chunks WHERE document_id = ? ORDER BY chunk_index ASC",
    )
    .bind(&document_id)
    .fetch_all(&state.pool)
    .await
    .map_err(|err| document_sql_error("获取文档内容失败", err))?;
    let mut content = chunks
        .into_iter()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if content.is_empty() {
        match state
            .engine
            .store
            .collection(&state.engine.child_collection)
            .await
        {
            Ok(snapshot) => {
                content = crate::milvus_lite::document_content_from_rows(&snapshot.rows, &document_id);
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => tracing::warn!("{}", format_milvus_content_skip(&err.to_string(), &document_id)),
        }
    }
    if content.is_empty() && !row.file_path.is_empty() {
        if let Ok(bytes) = std::fs::read(&row.file_path) {
            if let Ok(text) = crate::ingest::extract_text(&row.original_filename, &bytes) {
                content = text;
            }
        }
    }
    if content.trim().is_empty() {
        return Err(document_content_exception("文档正文不存在或尚未处理完成"));
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

/// 对齐 Python `/documents/content`：detail 含「不存在」时 404，其余 400。
fn document_content_exception(detail: &str) -> AuthError {
    AuthError::http(
        if detail.contains("不存在") {
            StatusCode::NOT_FOUND
        } else {
            StatusCode::BAD_REQUEST
        },
        detail,
    )
}

fn document_delete_requested(status: Option<&str>) -> bool {
    match status.map(str::trim) {
        None => true,
        Some("deleting" | "deleted") => true,
        Some(_) => false,
    }
}

fn delete_local_uploaded_file(file_path: &str, document_id: &str, reason: &str) -> bool {
    let path = std::path::Path::new(file_path);
    let existed = path.exists();
    match std::fs::remove_file(path) {
        Ok(()) => {
            tracing::info!(
                document_id,
                existed,
                reason,
                path = file_path,
                "本地上传原始文件清理完成"
            );
            existed
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            tracing::info!(
                document_id,
                existed = false,
                reason,
                path = file_path,
                "本地上传原始文件清理完成"
            );
            false
        }
        Err(err) => {
            tracing::warn!(
                document_id,
                reason,
                path = file_path,
                "本地上传原始文件清理失败: {err}"
            );
            false
        }
    }
}

async fn insert_processing_document(
    pool: &sqlx::SqlitePool,
    document_id: &str,
    original_filename: &str,
    stored_filename: &str,
    file_path: &str,
    content_type: Option<&str>,
    file_type: &str,
    file_size: i64,
) -> Result<DocumentRow, AuthError> {
    let now = db::fmt_dt(&db::now_naive());
    sqlx::query(
        "INSERT INTO documents (
            document_id, original_filename, stored_filename, file_path, content_type,
            file_type, file_size, status, chunk_count, error_message, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'processing', 0, NULL, ?, ?)",
    )
    .bind(document_id)
    .bind(original_filename)
    .bind(stored_filename)
    .bind(file_path)
    .bind(content_type)
    .bind(file_type)
    .bind(file_size)
    .bind(&now)
    .bind(&now)
    .execute(pool)
    .await
    .map_err(|err| document_sql_error("文档上传失败", err))?;
    load_document(pool, document_id)
        .await
        .map_err(|err| document_sql_error("文档上传失败", err))?
        .ok_or_else(|| {
            tracing::error!("{}", format_document_route_error("文档上传失败", "文档元数据写入失败"));
            AuthError::from("文档元数据写入失败".to_string())
        })
}

async fn mark_document_failed(pool: &sqlx::SqlitePool, document_id: &str, err: &str) {
    let now = db::fmt_dt(&db::now_naive());
    if let Err(restore_err) = sqlx::query(
        "UPDATE documents SET status = 'failed', error_message = ?, updated_at = ? \
         WHERE document_id = ?",
    )
    .bind(err)
    .bind(&now)
    .bind(document_id)
    .execute(pool)
    .await
    {
        tracing::error!(document_id, "文档失败状态回写失败: {restore_err}");
    }
}

async fn replace_document_chunk_records(
    pool: &sqlx::SqlitePool,
    document_id: &str,
    parents: &[crate::ingest::ParentRow],
) -> Result<(), String> {
    sqlx::query("DELETE FROM document_chunks WHERE document_id = ?")
        .bind(document_id)
        .execute(pool)
        .await
        .map_err(|err| err.to_string())?;
    let now = db::fmt_dt(&db::now_naive());
    for parent in parents {
        if parent.parent_text.trim().is_empty() {
            continue;
        }
        sqlx::query(
            "INSERT INTO document_chunks (document_id, chunk_index, parent_id, chunk_text, created_at) \
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(&parent.document_id)
        .bind(parent.parent_index)
        .bind(&parent.parent_id)
        .bind(&parent.parent_text)
        .bind(&now)
        .execute(pool)
        .await
        .map_err(|err| err.to_string())?;
    }
    Ok(())
}

async fn mark_document_ready(
    pool: &sqlx::SqlitePool,
    document_id: &str,
    chunk_count: i64,
) -> Result<(), String> {
    let now = db::fmt_dt(&db::now_naive());
    sqlx::query(
        "UPDATE documents SET status = 'ready', chunk_count = ?, error_message = NULL, updated_at = ? \
         WHERE document_id = ?",
    )
    .bind(chunk_count)
    .bind(&now)
    .bind(document_id)
    .execute(pool)
    .await
    .map_err(|err| err.to_string())?;
    Ok(())
}

async fn current_document_status(
    pool: &sqlx::SqlitePool,
    document_id: &str,
) -> Result<Option<String>, String> {
    Ok(load_document(pool, document_id)
        .await
        .map_err(|err| err.to_string())?
        .map(|row| row.status))
}

async fn process_document_upload(
    state: AppState,
    document_id: String,
    stored_path: String,
    original_name: String,
    file_type: String,
    content_type: String,
) {
    tracing::info!(
        "后台处理开始 (document_id={document_id}, file={original_name}, path={stored_path})"
    );
    let result = process_document_upload_inner(
        &state,
        &document_id,
        &stored_path,
        &original_name,
        &file_type,
        &content_type,
    )
    .await;
    if let Err(err) = result {
        tracing::error!("文档后台处理失败 (document_id={document_id}): {err}");
        if !document_delete_requested(
            current_document_status(&state.pool, &document_id)
                .await
                .ok()
                .flatten()
                .as_deref(),
        ) {
            tracing::info!("开始更新文档状态为 failed (document_id={document_id})");
            mark_document_failed(&state.pool, &document_id, &err).await;
            tracing::info!("文档状态更新完成 (document_id={document_id}, status=failed)");
        }
    }
    delete_local_uploaded_file(&stored_path, &document_id, "background_processed");
}

async fn process_document_upload_inner(
    state: &AppState,
    document_id: &str,
    stored_path: &str,
    original_name: &str,
    file_type: &str,
    content_type: &str,
) -> Result<(), String> {
    let stored_path_buf = std::path::PathBuf::from(stored_path);
    let extract_name = original_name.to_string();
    let extract_type = file_type.to_string();
    let extract_id = document_id.to_string();
    tracing::info!("开始抽取文本 (document_id={document_id})");
    let chunks = tokio::task::spawn_blocking(move || {
        let data = std::fs::read(&stored_path_buf).map_err(|err| {
            let wrapped = format!("提取文本失败: {err}");
            tracing::error!("{wrapped}");
            wrapped
        })?;
        crate::ingest::prepare_upload_chunks(&extract_id, &extract_name, &data, &extract_type)
    })
    .await
    .map_err(|err| err.to_string())??;
    tracing::info!(
        "文本抽取完成 (document_id={document_id}, text_len={})",
        chunks.text_len
    );
    tracing::info!("开始 Parent-Child 切块 (document_id={document_id})");
    tracing::info!(
        "切块完成 (document_id={document_id}, parents={}, children={})",
        chunks.parents.len(),
        chunks.children.len()
    );

    if document_delete_requested(
        current_document_status(&state.pool, document_id)
            .await?
            .as_deref(),
    ) {
        tracing::info!(document_id, "文档已请求删除，跳过后台向量写入");
        return Ok(());
    }

    log_upload_step(
        document_id,
        "3/3",
        &format!(
            "开始写入 Milvus (parents={}, children={})",
            chunks.parents.len(),
            chunks.children.len()
        ),
    );
    let embedder = state.engine.embedder.clone();
    let child_texts: Vec<String> = chunks
        .children
        .iter()
        .map(|child| child.child_text.clone())
        .collect();
    tracing::info!(
        "开始初始化 BGE Embedding (document_id={document_id}, children={})",
        child_texts.len()
    );
    tracing::info!(
        "BGE Embedding 就绪，开始向量化 (document_id={document_id}, children={})",
        child_texts.len()
    );
    let vectors = tokio::task::spawn_blocking(move || embedder.embed_documents(&child_texts))
        .await
        .map_err(|err| err.to_string())?
        .map_err(|err| format!("向量化失败: {err}"))?;
    tracing::info!(
        "BGE 向量化完成 (document_id={document_id}, vectors={})",
        vectors.len()
    );

    crate::ingest::append_to_milvus_lite(
        &state.engine.store,
        &state.engine.child_collection,
        &chunks.children,
        original_name,
        file_type,
        content_type,
        &vectors,
        state.config.embedding_dimension,
    )
    .await
    .map_err(|err| format!("向量库写入失败: {err}"))?;
    tracing::info!(
        "Milvus Child 写入成功 (document_id={document_id}, parents={}, children={})",
        chunks.parents.len(),
        chunks.children.len()
    );
    log_upload_step(
        document_id,
        "3/3",
        &format!("Milvus 写入成功: children={}", chunks.children.len()),
    );

    if document_delete_requested(
        current_document_status(&state.pool, document_id)
            .await?
            .as_deref(),
    ) {
        let _ = state.engine.delete_document_chunks(document_id).await;
        tracing::info!(document_id, "文档已在向量写入期间删除，已清理向量块");
        return Ok(());
    }

    tracing::info!("开始更新文档状态为 ready (document_id={document_id})");
    replace_document_chunk_records(&state.pool, document_id, &chunks.parents).await?;
    mark_document_ready(&state.pool, document_id, chunks.children.len() as i64).await?;
    tracing::info!(
        "文档状态更新完成 (document_id={document_id}, status=ready, children={})",
        chunks.children.len()
    );
    state.engine.invalidate().await;
    tracing::info!(
        "✓ 文档后台处理完成 (document_id={document_id}, children={})",
        chunks.children.len()
    );
    Ok(())
}

// POST /api/v1/documents/upload (multipart)
pub async fn upload_document(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: axum::extract::Multipart,
) -> Result<Json<Value>, AuthError> {
    require_current_user(&state, &headers).await?;
    db::ensure_document_schema(&state.pool)
        .await
        .map_err(|err| document_sql_error("文档上传失败", err))?;

    let mut filename = String::new();
    let mut file_data: Vec<u8> = Vec::new();
    let mut declared_content_type: Option<String> = None;
    let mut saw_file_field = false;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AuthError::bad_request(&format!("multipart 读取失败: {e}")))?
    {
        if field.name() == Some("file") {
            saw_file_field = true;
            filename = field.file_name().unwrap_or("unknown").to_string();
            declared_content_type = field.content_type().map(str::to_string);
            file_data = field
                .bytes()
                .await
                .map_err(|e| AuthError::bad_request(&format!("文件读取失败: {e}")))?
                .to_vec();
            break;
        }
    }
    // 对齐 FastAPI `file: UploadFile = File(...)`：缺字段是 422 loc=["body","file"]。
    if !saw_file_field {
        return Err(missing_upload_file());
    }

    let filename = sanitize_original_filename(&filename);
    if filename.is_empty() {
        return Err(AuthError::bad_request("文件名不能为空"));
    }
    let file_type = upload_file_type(&filename);
    if let Some(detail) = reject_unsupported_file_type(&file_type, &state.config.allowed_file_types)
    {
        return Err(AuthError::bad_request(&detail));
    }

    // 对齐 Python 时序：两条 [1/3] 日志先于大小校验，超限请求留有日志痕迹。
    let document_id = uuid::Uuid::new_v4().simple().to_string();
    log_upload_step(
        &document_id,
        "1/3",
        &format!("开始接收上传文件: {filename}"),
    );
    let content_type = upload_content_type(declared_content_type.as_deref());
    let stored_filename = format!("{document_id}.{file_type}");
    let upload_dir = crate::config::resolve_upload_dir(
        &state.config.upload_dir,
        &state.config.sqlite_path,
    );
    let _ = std::fs::create_dir_all(&upload_dir);
    let file_path = upload_dir.join(&stored_filename);
    std::fs::write(&file_path, &file_data)
        .map_err(|err| AuthError::bad_request(&format!("保存上传文件失败: {err}")))?;
    let file_path_text = file_path.display().to_string();
    log_upload_step(
        &document_id,
        "1/3",
        &format!("本地文件保存成功: {filename} -> {file_path_text}"),
    );
    if let Some(detail) = reject_oversized_upload(file_data.len(), state.config.max_upload_size) {
        let _ = std::fs::remove_file(&file_path);
        return Err(AuthError::bad_request(detail));
    }
    let file_size = file_data.len() as i64;
    drop(file_data);

    let row = match insert_processing_document(
        &state.pool,
        &document_id,
        &filename,
        &stored_filename,
        &file_path_text,
        content_type.as_deref(),
        &file_type,
        file_size,
    )
    .await
    {
        Ok(row) => row,
        Err(err) => {
            tracing::error!("{}", format_document_ingest_fail(&document_id, &err.detail));
            let _ = std::fs::remove_file(&file_path);
            return Err(err);
        }
    };
    log_upload_step(
        &document_id,
        "2/3",
        &format!("SQLite 元数据写入成功: status=processing, file_size={file_size}"),
    );

    let state = state.clone();
    tokio::spawn(process_document_upload(
        state,
        document_id.clone(),
        file_path_text,
        filename,
        file_type,
        milvus_content_type(content_type.as_deref()),
    ));
    tracing::info!("{}", format_upload_task_submitted(&document_id));
    Ok(Json(doc_to_json(&row)))
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
        db::ensure_document_schema(&pool).await.unwrap();
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
    fn empty_document_body_is_not_found_like_python() {
        let empty = document_content_exception("文档正文不存在或尚未处理完成");
        assert_eq!(empty.status, StatusCode::NOT_FOUND);
        assert_eq!(empty.detail, "文档正文不存在或尚未处理完成");
        let missing = document_content_exception("文档不存在");
        assert_eq!(missing.status, StatusCode::NOT_FOUND);
        let other = document_content_exception("文档切块失败，未生成有效内容");
        assert_eq!(other.status, StatusCode::BAD_REQUEST);
        let missing_file = missing_upload_file();
        assert_eq!(missing_file.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(missing_file.detail, "Field required");
        assert_eq!(missing_file.loc, vec!["body".to_string(), "file".to_string()]);
        assert_eq!(missing_file.error_type, "missing");
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
    fn delete_step_log_matches_python() {
        assert_eq!(
            format_delete_step("doc-1", "1/4", "收到删除请求"),
            "[文档删除][1/4] 收到删除请求 (document_id=doc-1)"
        );
        assert_eq!(
            format_delete_step("doc-1", "2/4", "Milvus 文档切块删除完成: deleted=true"),
            "[文档删除][2/4] Milvus 文档切块删除完成: deleted=true (document_id=doc-1)"
        );
        assert_eq!(
            format_delete_step("doc-1", "4/4", "SQLite 文档记录删除完成"),
            "[文档删除][4/4] SQLite 文档记录删除完成 (document_id=doc-1)"
        );
        assert_eq!(
            format_delete_fail("doc-1", "2/4", "Milvus 文档切块删除失败", "timeout"),
            "[文档删除][2/4] Milvus 文档切块删除失败 (document_id=doc-1): timeout"
        );
        assert_eq!(
            format_delete_fail("doc-1", "4/4", "SQLite 文档记录删除失败", "locked"),
            "[文档删除][4/4] SQLite 文档记录删除失败 (document_id=doc-1): locked"
        );
        assert_eq!(
            format_delete_fail("doc-1", "3/4", "本地文件删除失败", "permission denied"),
            "[文档删除][3/4] 本地文件删除失败 (document_id=doc-1): permission denied"
        );
    }

    #[test]
    fn document_route_error_logs_match_python() {
        assert_eq!(
            format_document_route_error("获取文档内容失败", "boom"),
            "获取文档内容失败: boom"
        );
        assert_eq!(
            format_document_route_error("文档上传失败", "disk full"),
            "文档上传失败: disk full"
        );
        assert_eq!(
            format_document_route_error("获取文档列表失败", "locked"),
            "获取文档列表失败: locked"
        );
        assert_eq!(
            format_document_route_error("文档删除失败", "busy"),
            "文档删除失败: busy"
        );
        assert_eq!(
            format_document_ingest_fail("doc-1", "database is locked"),
            "文档入库失败 (document_id=doc-1): database is locked"
        );
    }

    #[test]
    fn upload_step_log_matches_python() {
        assert_eq!(
            format_upload_step("doc-1", "1/3", "开始接收上传文件: 手册.md"),
            "[文档上传][1/3] 开始接收上传文件: 手册.md (document_id=doc-1)"
        );
        assert_eq!(
            format_upload_step(
                "doc-1",
                "1/3",
                "本地文件保存成功: 手册.md -> /tmp/doc-1.md"
            ),
            "[文档上传][1/3] 本地文件保存成功: 手册.md -> /tmp/doc-1.md (document_id=doc-1)"
        );
        assert_eq!(
            format_upload_step(
                "doc-1",
                "2/3",
                "SQLite 元数据写入成功: status=processing, file_size=12"
            ),
            "[文档上传][2/3] SQLite 元数据写入成功: status=processing, file_size=12 (document_id=doc-1)"
        );
        assert_eq!(
            format_upload_step("doc-1", "3/3", "Milvus 写入成功: children=4"),
            "[文档上传][3/3] Milvus 写入成功: children=4 (document_id=doc-1)"
        );
        assert_eq!(
            format_upload_task_submitted("doc-1"),
            "[文档上传] 后台处理任务已提交 (document_id=doc-1)"
        );
        assert_eq!(
            format_milvus_content_skip("connection reset", "doc-1"),
            "从 Milvus 读取文档正文切片失败，已跳过: connection reset (document_id=doc-1)"
        );
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
        assert!(remove_document_files(&upload_dir, &row).unwrap());
        assert!(!stored.exists());
        assert!(!extra.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn remove_document_files_fails_when_path_is_directory() {
        let root = std::env::temp_dir().join(format!(
            "eks-doc-delete-dir-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        let upload_dir = root.join("uploads");
        std::fs::create_dir_all(&upload_dir).unwrap();
        let dir_path = root.join("not-a-file");
        std::fs::create_dir_all(&dir_path).unwrap();
        let mut row = sample_row("deleting");
        row.stored_filename = "missing.md".into();
        row.file_path = dir_path.display().to_string();
        let err = remove_document_files(&upload_dir, &row).unwrap_err();
        assert!(!err.is_empty(), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sanitizes_upload_filename_and_type() {
        assert_eq!(
            sanitize_original_filename(r"C:\uploads\..\手册.PDF"),
            "手册.PDF"
        );
        assert_eq!(sanitize_original_filename("/tmp/a/b.md"), "b.md");
        assert_eq!(sanitize_original_filename("  "), "");
        assert_eq!(upload_file_type("手册.PDF"), "pdf");
        assert_eq!(upload_file_type("noext"), "");
    }

    #[test]
    fn upload_keeps_multipart_content_type_like_python() {
        assert_eq!(
            upload_content_type(Some("text/x-markdown")).as_deref(),
            Some("text/x-markdown")
        );
        assert_eq!(upload_content_type(None), None);
        assert_eq!(upload_content_type(Some("")).as_deref(), Some(""));
        assert_ne!(
            upload_content_type(Some("text/x-markdown")).as_deref(),
            Some(guess_content_type("md"))
        );
        assert_eq!(milvus_content_type(Some("text/x-markdown")), "text/x-markdown");
        assert_eq!(milvus_content_type(None), "");
    }

    #[test]
    fn rejects_unsupported_type_and_oversized_upload() {
        let allowed = vec!["pdf".into(), "md".into(), "txt".into()];
        assert_eq!(
            reject_unsupported_file_type("exe", &allowed).as_deref(),
            Some("不支持的文件类型: exe")
        );
        assert_eq!(
            reject_unsupported_file_type("", &allowed).as_deref(),
            Some("不支持的文件类型: ")
        );
        assert!(reject_unsupported_file_type("PDF", &allowed).is_none());
        assert!(reject_unsupported_file_type("md", &allowed).is_none());
        let defaults = crate::config::parse_allowed_file_types("");
        assert_eq!(
            reject_unsupported_file_type("doc", &defaults).as_deref(),
            Some("不支持的文件类型: doc")
        );
        assert_eq!(
            reject_unsupported_file_type("ppt", &defaults).as_deref(),
            Some("不支持的文件类型: ppt")
        );
        assert!(reject_unsupported_file_type("pptx", &defaults).is_none());
        assert!(reject_unsupported_file_type("html", &defaults).is_none());
        assert!(reject_unsupported_file_type("csv", &defaults).is_none());
        assert_eq!(
            reject_oversized_upload(104_857_601, 104_857_600),
            Some("文件超过最大上传限制")
        );
        assert!(reject_oversized_upload(104_857_600, 104_857_600).is_none());
        assert!(reject_oversized_upload(0, 104_857_600).is_none());
    }

    #[test]
    fn delete_requested_matches_python() {
        assert!(document_delete_requested(None));
        assert!(document_delete_requested(Some("deleting")));
        assert!(document_delete_requested(Some("deleted")));
        assert!(!document_delete_requested(Some("processing")));
        assert!(!document_delete_requested(Some("ready")));
        assert!(!document_delete_requested(Some("failed")));
    }

    #[tokio::test]
    async fn insert_processing_then_ready_or_failed() {
        let pool = memory_pool().await;
        let row = insert_processing_document(
            &pool,
            "doc-up",
            "手册.md",
            "doc-up.md",
            "/tmp/doc-up.md",
            Some("text/markdown"),
            "md",
            12,
        )
        .await
        .unwrap();
        assert_eq!(row.status, "processing");
        assert_eq!(row.chunk_count, 0);
        assert_eq!(row.content_type.as_deref(), Some("text/markdown"));
        assert!(row.error_message.is_none());

        let missing_type = insert_processing_document(
            &pool,
            "doc-up-none",
            "手册.md",
            "doc-up-none.md",
            "/tmp/doc-up-none.md",
            None,
            "md",
            12,
        )
        .await
        .unwrap();
        assert!(missing_type.content_type.is_none());
        let custom_type = insert_processing_document(
            &pool,
            "doc-up-custom",
            "手册.md",
            "doc-up-custom.md",
            "/tmp/doc-up-custom.md",
            Some("text/x-markdown"),
            "md",
            12,
        )
        .await
        .unwrap();
        assert_eq!(custom_type.content_type.as_deref(), Some("text/x-markdown"));
        let payload = doc_to_json(&row);
        assert_eq!(payload["status"], json!("processing"));
        assert_eq!(payload["chunk_count"], json!(0));

        mark_document_ready(&pool, "doc-up", 4).await.unwrap();
        let ready = load_document(&pool, "doc-up").await.unwrap().unwrap();
        assert_eq!(ready.status, "ready");
        assert_eq!(ready.chunk_count, 4);
        assert!(ready.error_message.is_none());

        mark_document_failed(&pool, "doc-up", "文档内容为空，无法生成向量").await;
        let failed = load_document(&pool, "doc-up").await.unwrap().unwrap();
        assert_eq!(failed.status, "failed");
        assert_eq!(
            failed.error_message.as_deref(),
            Some("文档内容为空，无法生成向量")
        );
    }

    #[tokio::test]
    async fn skips_ready_persist_when_delete_requested() {
        let pool = memory_pool().await;
        insert_row(&pool, &sample_row("deleting")).await;
        assert!(document_delete_requested(
            current_document_status(&pool, "doc-1")
                .await
                .unwrap()
                .as_deref()
        ));
        let chunks = crate::ingest::prepare_upload_chunks(
            "doc-1",
            "手册.md",
            "差旅报销流程如下。".as_bytes(),
            "md",
        )
        .unwrap();
        assert!(!chunks.children.is_empty());
        if !document_delete_requested(
            current_document_status(&pool, "doc-1")
                .await
                .unwrap()
                .as_deref(),
        ) {
            panic!("deleting 文档不应继续写入 ready");
        }
        let row = load_document(&pool, "doc-1").await.unwrap().unwrap();
        assert_eq!(row.status, "deleting");
        assert_eq!(row.chunk_count, 3);
    }
}
