//! 对话管理 CRUD（对齐 Python conversation_service）。
//! 同表 user_conversations，同字段语义，与 Python 后端读写兼容。

use axum::{
    Json,
    extract::{Path, State},
    http::HeaderMap,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::auth::AuthError;
use crate::db;
use crate::routes::require_current_user;
use crate::state::AppState;

#[derive(Serialize)]
pub struct ConversationRecord {
    pub conversation_id: String,
    pub session_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub messages: Value,
}

#[derive(Deserialize)]
pub struct ConversationUpsertRequest {
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub messages: Value,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
}

/// 过滤临时占位消息，只保留 user/assistant 角色（对齐 _normalize_messages）。
fn normalize_messages(messages: &Value) -> String {
    let list = match messages.as_array() {
        Some(list) => list,
        None => return "[]".to_string(),
    };
    let filtered: Vec<&Value> = list
        .iter()
        .take(500)
        .filter(|item| {
            item.get("ephemeral")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
                == false
        })
        .filter(|item| {
            item.get("role")
                .and_then(|v| v.as_str())
                .map(|r| r == "user" || r == "assistant")
                .unwrap_or(false)
        })
        .collect();
    serde_json::to_string(&filtered).unwrap_or_else(|_| "[]".to_string())
}

fn record_to_json(row: &ConversationRow) -> Value {
    let messages: Value =
        serde_json::from_str(&row.payload).unwrap_or_else(|_| json!([]));
    json!({
        "conversation_id": row.conversation_id,
        "session_id": row.session_id,
        "created_at": db::dt_to_iso(&row.created_at),
        "updated_at": db::dt_to_iso(&row.updated_at),
        "messages": messages,
    })
}

#[derive(Debug, sqlx::FromRow)]
pub struct ConversationRow {
    pub user_id: String,
    pub conversation_id: String,
    pub session_id: Option<String>,
    pub payload: String,
    pub created_at: String,
    pub updated_at: String,
}

// GET /api/v1/chat/conversations
pub async fn list_conversations(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, AuthError> {
    let user = require_current_user(&state, &headers).await?;
    let rows: Vec<ConversationRow> = sqlx::query_as(
        "SELECT user_id, conversation_id, session_id, payload, created_at, updated_at \
         FROM user_conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT 100",
    )
    .bind(&user.user_id)
    .fetch_all(&state.pool)
    .await
    .map_err(AuthError::from)?;
    let conversations: Vec<Value> = rows.iter().map(record_to_json).collect();
    Ok(Json(json!({ "conversations": conversations })))
}

// PUT /api/v1/chat/conversations/{conversation_id}
pub async fn upsert_conversation(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(conversation_id): Path<String>,
    Json(request): Json<ConversationUpsertRequest>,
) -> Result<Json<Value>, AuthError> {
    let user = require_current_user(&state, &headers).await?;
    let cid = conversation_id.trim().to_string();
    if cid.is_empty() {
        return Err(AuthError::bad_request("conversation_id 不能为空"));
    }

    let payload = normalize_messages(&request.messages);
    let now = db::fmt_dt(&db::now_naive());
    let session_id = request
        .session_id
        .as_deref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let created_at = request.created_at.as_deref().and_then(db::parse_dt)
        .map(|dt| db::fmt_dt(&dt))
        .unwrap_or_else(|| now.clone());

    // upsert: INSERT OR REPLACE 保持与 Python 的 first-then-update 等价
    sqlx::query(
        "INSERT INTO user_conversations (user_id, conversation_id, session_id, payload, created_at, updated_at) \
         VALUES (?, ?, ?, ?, ?, ?) \
         ON CONFLICT(user_id, conversation_id) DO UPDATE SET \
         session_id = excluded.session_id, payload = excluded.payload, updated_at = excluded.updated_at",
    )
    .bind(&user.user_id)
    .bind(&cid)
    .bind(&session_id)
    .bind(&payload)
    .bind(&created_at)
    .bind(&now)
    .execute(&state.pool)
    .await
    .map_err(AuthError::from)?;

    let row: Option<ConversationRow> = sqlx::query_as(
        "SELECT user_id, conversation_id, session_id, payload, created_at, updated_at \
         FROM user_conversations WHERE user_id = ? AND conversation_id = ?",
    )
    .bind(&user.user_id)
    .bind(&cid)
    .fetch_optional(&state.pool)
    .await
    .map_err(AuthError::from)?;

    match row {
        Some(row) => Ok(Json(record_to_json(&row))),
        None => Err(AuthError::bad_request("保存失败")),
    }
}

// DELETE /api/v1/chat/conversations/{conversation_id}
pub async fn delete_conversation(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(conversation_id): Path<String>,
) -> Result<Json<Value>, AuthError> {
    let user = require_current_user(&state, &headers).await?;
    let cid = conversation_id.trim().to_string();
    let result = sqlx::query(
        "DELETE FROM user_conversations WHERE user_id = ? AND conversation_id = ?",
    )
    .bind(&user.user_id)
    .bind(&cid)
    .execute(&state.pool)
    .await
    .map_err(AuthError::from)?;
    let memory_deleted = crate::memory::delete_conversation_memory(&state.pool, &user.user_id, &cid)
        .await
        .map_err(AuthError::from)?
        > 0;
    Ok(Json(json!({
        "conversation_id": cid,
        "deleted": result.rows_affected() > 0,
        "memory_deleted": memory_deleted,
    })))
}

async fn list_user_conversation_ids(
    pool: &sqlx::SqlitePool,
    user_id: &str,
) -> Result<Vec<String>, sqlx::Error> {
    sqlx::query_scalar(
        "SELECT conversation_id FROM user_conversations WHERE user_id = ? ORDER BY updated_at DESC",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await
}

/// 按会话统计长期记忆删除次数，对齐 Python `memory_deleted_count`（会话数而非事件行数）。
async fn memory_deleted_conversation_count(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    conversation_ids: &[String],
) -> Result<i64, sqlx::Error> {
    let mut count = 0i64;
    for conversation_id in conversation_ids {
        if crate::memory::delete_conversation_memory(pool, user_id, conversation_id).await? > 0 {
            count += 1;
        }
    }
    Ok(count)
}

// DELETE /api/v1/chat/conversations (clear all)
pub async fn clear_conversations(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, AuthError> {
    let user = require_current_user(&state, &headers).await?;
    let conversation_ids = list_user_conversation_ids(&state.pool, &user.user_id)
        .await
        .map_err(AuthError::from)?;
    let result = sqlx::query("DELETE FROM user_conversations WHERE user_id = ?")
        .bind(&user.user_id)
        .execute(&state.pool)
        .await
        .map_err(AuthError::from)?;
    let memory_deleted_count =
        memory_deleted_conversation_count(&state.pool, &user.user_id, &conversation_ids)
            .await
            .map_err(AuthError::from)?;
    Ok(Json(json!({
        "deleted_count": result.rows_affected(),
        "memory_deleted_count": memory_deleted_count,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn memory_pool() -> sqlx::SqlitePool {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(
            "CREATE TABLE user_conversations (
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                session_id TEXT,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, conversation_id)
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        crate::memory::ensure_schema(&pool).await.unwrap();
        pool
    }

    async fn insert_conversation(pool: &sqlx::SqlitePool, user_id: &str, conversation_id: &str) {
        sqlx::query(
            "INSERT INTO user_conversations (
                user_id, conversation_id, session_id, payload, created_at, updated_at
            ) VALUES (?, ?, 's1', '[]', '2026-01-01 00:00:00.000000', '2026-01-01 00:00:00.000000')",
        )
        .bind(user_id)
        .bind(conversation_id)
        .execute(pool)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn memory_deleted_count_is_conversation_not_event_rows() {
        let pool = memory_pool().await;
        insert_conversation(&pool, "u1", "c-with-memory").await;
        insert_conversation(&pool, "u1", "c-empty").await;
        insert_conversation(&pool, "u1", "c-multi").await;
        crate::memory::persist_turn(
            &pool,
            None,
            "u1",
            "c-with-memory",
            "s1",
            "你好",
            "你好，我在。",
            90,
        )
        .await;
        crate::memory::persist_turn(&pool, None, "u1", "c-multi", "s1", "第一问", "第一答", 90).await;
        crate::memory::persist_turn(&pool, None, "u1", "c-multi", "s1", "第二问", "第二答", 90).await;
        crate::memory::persist_turn(&pool, None, "u2", "c-other", "s1", "别人的问题", "别人的回答", 90)
            .await;

        let ids = list_user_conversation_ids(&pool, "u1").await.unwrap();
        assert_eq!(ids.len(), 3);
        let count = memory_deleted_conversation_count(&pool, "u1", &ids)
            .await
            .unwrap();
        assert_eq!(count, 2);

        let remaining_other: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_records WHERE user_id = 'u2' AND status = 'active'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(remaining_other, 1);
    }
}
