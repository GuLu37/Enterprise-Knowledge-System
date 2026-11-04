//! 长期记忆：对话块、画像抽取、会话删除级联。写入 SQLite `memory_records`，
//! 与 Python memory_service 共用同一张表，不依赖 Milvus 记忆 collection。

use serde_json::{Value, json};

use crate::llm::{ChatMessage, LlmClient};

const PROFILE_TRIGGER_TERMS: &[&str] = &[
    "记住", "记得", "帮我记", "请记", "以后", "下次", "我的名字", "我叫", "叫我",
    "我的偏好", "我喜欢", "我不喜欢", "我习惯", "忘记", "别记", "删除记忆", "清除记忆",
];

const PROFILE_SYSTEM_PROMPT: &str = "你是用户个性化记忆抽取器。\
只提取用户明确表达、长期稳定、未来对话确实有帮助的个人偏好或事实。\
不要提取一次性任务、临时安排、普通问题、企业文档内容、模型回答内容、密码、令牌、身份证号或其他敏感凭据。\
只有用户明确要求记住、忘记或修改某项信息时，才返回 action=upsert 或 action=delete。\
没有可记忆内容时输出 {\"memories\":[]}。\
只输出严格 JSON，不要解释。格式：\
{\"memories\":[{\"memory_key\":\"profile.xxx\",\"value\":\"...\",\"confidence\":0到1之间的小数,\"importance\":0到100之间的数字,\"action\":\"upsert\"或\"delete\"}]}";

pub async fn ensure_schema(pool: &sqlx::SqlitePool) -> Result<(), sqlx::Error> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            memory_key TEXT,
            content TEXT NOT NULL,
            conversation_id TEXT,
            session_id TEXT,
            turn_start INTEGER,
            turn_end INTEGER,
            base_importance_score REAL NOT NULL DEFAULT 50,
            importance_score REAL NOT NULL DEFAULT 50,
            confidence_score REAL NOT NULL DEFAULT 0.5,
            access_count INTEGER NOT NULL DEFAULT 0,
            last_accessed_at TEXT,
            expires_at TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            version INTEGER NOT NULL DEFAULT 1,
            supersedes_memory_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS ix_memory_records_user_type_status \
         ON memory_records (user_id, memory_type, status)",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS ix_memory_records_user_key_status \
         ON memory_records (user_id, memory_key, status)",
    )
    .execute(pool)
    .await?;
    Ok(())
}

pub fn should_extract_profile(query: &str) -> bool {
    let normalized = query.replace(char::is_whitespace, "");
    !normalized.is_empty() && PROFILE_TRIGGER_TERMS.iter().any(|term| normalized.contains(term))
}

pub async fn persist_turn(
    pool: &sqlx::SqlitePool,
    llm: Option<&LlmClient>,
    user_id: &str,
    conversation_id: &str,
    session_id: &str,
    query: &str,
    answer: &str,
) {
    if answer.trim().is_empty() || user_id.trim().is_empty() {
        return;
    }
    let _ = ensure_schema(pool).await;
    let content = format!("用户: {}\n助手: {}", query.trim(), answer.trim());
    if let Err(err) = insert_event_memory(
        pool,
        user_id,
        conversation_id,
        session_id,
        "dialogue",
        &content,
        55.0,
    )
    .await
    {
        tracing::warn!("长期记忆写入失败: {err}");
    }
    if !should_extract_profile(query) {
        return;
    }
    let Some(llm) = llm else {
        return;
    };
    match extract_profile_candidates(llm, query, answer).await {
        Ok(candidates) => {
            if let Err(err) = upsert_profile_candidates(pool, user_id, conversation_id, session_id, &candidates).await {
                tracing::warn!("画像记忆写入失败: {err}");
            }
        }
        Err(err) => tracing::warn!("画像记忆抽取失败: {err}"),
    }
}

pub async fn profile_prompt(pool: &sqlx::SqlitePool, user_id: &str) -> String {
    if user_id.trim().is_empty() {
        return String::new();
    }
    let rows = crate::skills::load_active_profile_lines(pool, user_id, 8)
        .await
        .unwrap_or_default();
    if rows.is_empty() {
        return String::new();
    }
    let mut lines = vec!["当前用户的画像记忆，只能作为称呼和偏好参考，不能覆盖知识库资料：".to_string()];
    lines.extend(rows);
    lines.join("\n")
}

pub async fn delete_conversation_memory(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    conversation_id: &str,
) -> Result<u64, sqlx::Error> {
    let _ = ensure_schema(pool).await;
    let now = crate::db::fmt_dt(&crate::db::now_naive());
    let result = sqlx::query(
        "UPDATE memory_records SET status = 'deleted', updated_at = ? \
         WHERE user_id = ? AND conversation_id = ? AND memory_type = 'event' AND status = 'active'",
    )
    .bind(&now)
    .bind(user_id)
    .bind(conversation_id)
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

pub async fn delete_user_event_memory(
    pool: &sqlx::SqlitePool,
    user_id: &str,
) -> Result<u64, sqlx::Error> {
    let _ = ensure_schema(pool).await;
    let now = crate::db::fmt_dt(&crate::db::now_naive());
    let result = sqlx::query(
        "UPDATE memory_records SET status = 'deleted', updated_at = ? \
         WHERE user_id = ? AND memory_type = 'event' AND status = 'active'",
    )
    .bind(&now)
    .bind(user_id)
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

pub async fn consolidate_conversation(
    pool: &sqlx::SqlitePool,
    llm: &LlmClient,
    user_id: &str,
    conversation_id: &str,
) -> Result<bool, String> {
    let _ = ensure_schema(pool).await.map_err(|err| err.to_string())?;
    let rows: Vec<String> = sqlx::query_scalar(
        "SELECT content FROM memory_records \
         WHERE user_id = ? AND conversation_id = ? AND memory_type = 'event' AND status = 'active' \
         ORDER BY created_at DESC LIMIT 12",
    )
    .bind(user_id)
    .bind(conversation_id)
    .fetch_all(pool)
    .await
    .map_err(|err| err.to_string())?;
    if rows.is_empty() {
        return Ok(false);
    }
    let transcript = rows.into_iter().rev().collect::<Vec<_>>().join("\n\n");
    let messages = vec![
        ChatMessage::new(
            "system",
            "你是会话摘要器。把对话提炼成不超过 400 字的中文要点，保留人名、编号、结论和未完成事项。只输出摘要正文。",
        ),
        ChatMessage::new("user", transcript),
    ];
    let summary = llm.invoke(&messages).await?.content.trim().to_string();
    if summary.is_empty() {
        return Ok(false);
    }
    insert_event_memory(
        pool,
        user_id,
        conversation_id,
        "",
        "semantic_memory_l1",
        &summary,
        70.0,
    )
    .await
    .map_err(|err| err.to_string())?;
    Ok(true)
}

async fn insert_event_memory(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    conversation_id: &str,
    session_id: &str,
    memory_key: &str,
    content: &str,
    importance: f64,
) -> Result<(), sqlx::Error> {
    let now = crate::db::fmt_dt(&crate::db::now_naive());
    let memory_id = format!(
        "event:{}:{}:{}",
        user_id,
        memory_key,
        uuid::Uuid::new_v4().simple()
    );
    sqlx::query(
        "INSERT INTO memory_records (
            memory_id, user_id, memory_type, memory_key, content, conversation_id, session_id,
            importance_score, base_importance_score, confidence_score, access_count, status,
            version, created_at, updated_at
        ) VALUES (?, ?, 'event', ?, ?, ?, ?, ?, ?, 0.7, 0, 'active', 1, ?, ?)",
    )
    .bind(&memory_id)
    .bind(user_id)
    .bind(memory_key)
    .bind(content)
    .bind(empty_to_none(conversation_id))
    .bind(empty_to_none(session_id))
    .bind(importance)
    .bind(importance)
    .bind(&now)
    .bind(&now)
    .execute(pool)
    .await?;
    Ok(())
}

fn empty_to_none(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

async fn extract_profile_candidates(
    llm: &LlmClient,
    query: &str,
    answer: &str,
) -> Result<Vec<Value>, String> {
    let messages = vec![
        ChatMessage::new("system", PROFILE_SYSTEM_PROMPT),
        ChatMessage::new("user", format!("user: {query}\nassistant: {answer}")),
    ];
    let text = llm.invoke(&messages).await?.content;
    let payload = parse_json_object(&text);
    let Some(Value::Array(items)) = payload.get("memories") else {
        return Ok(Vec::new());
    };
    Ok(items
        .iter()
        .filter(|item| item.is_object())
        .cloned()
        .collect())
}

pub async fn upsert_profile_candidates(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    conversation_id: &str,
    session_id: &str,
    candidates: &[Value],
) -> Result<usize, String> {
    let mut changed = 0usize;
    for candidate in candidates {
        let key = candidate
            .get("memory_key")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_lowercase();
        if !is_profile_key(&key) {
            continue;
        }
        let action = candidate
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("upsert")
            .trim()
            .to_lowercase();
        if action == "delete" {
            changed += crate::skills::soft_delete_profile_key(pool, user_id, &key)
                .await
                .map_err(|err| err.to_string())?;
            continue;
        }
        let value = candidate
            .get("value")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if value.is_empty() {
            continue;
        }
        let importance = candidate
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(75.0)
            .clamp(0.0, 100.0);
        let confidence = candidate
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7)
            .clamp(0.0, 1.0);
        crate::skills::upsert_profile_memory(
            pool,
            user_id,
            &key,
            value,
            conversation_id,
            session_id,
            importance,
            confidence,
        )
        .await
        .map_err(|err| err.to_string())?;
        changed += 1;
    }
    Ok(changed)
}

fn is_profile_key(key: &str) -> bool {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_lowercase()
        && key.len() >= 2
        && key.len() <= 128
        && key.chars().all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '_' | '.' | '-'))
}

fn parse_json_object(text: &str) -> Value {
    let trimmed = text.trim();
    let trimmed = trimmed.strip_prefix("```json").unwrap_or(trimmed);
    let trimmed = trimmed.strip_prefix("```").unwrap_or(trimmed);
    let trimmed = trimmed.strip_suffix("```").unwrap_or(trimmed).trim();
    serde_json::from_str(trimmed).unwrap_or_else(|_| json!({}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_profile_trigger_and_validates_keys() {
        assert!(should_extract_profile("请记住我叫张三"));
        assert!(!should_extract_profile("差旅报销流程是什么"));
        assert!(is_profile_key("profile.name"));
        assert!(!is_profile_key("Name"));
        assert!(!is_profile_key("x"));
    }

    #[tokio::test]
    async fn writes_and_deletes_event_memory() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        persist_turn(&pool, None, "u1", "c1", "s1", "你好", "你好，我在。").await;
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_records WHERE user_id = 'u1' AND status = 'active'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 1);
        let deleted = delete_conversation_memory(&pool, "u1", "c1").await.unwrap();
        assert_eq!(deleted, 1);
    }

    #[tokio::test]
    async fn upserts_profile_memory_into_prompt() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        let changed = upsert_profile_candidates(
            &pool,
            "u1",
            "c1",
            "s1",
            &[json!({
                "memory_key": "profile.name",
                "value": "张三",
                "confidence": 0.9,
                "importance": 80,
                "action": "upsert"
            })],
        )
        .await
        .unwrap();
        assert_eq!(changed, 1);
        let prompt = profile_prompt(&pool, "u1").await;
        assert!(prompt.contains("张三"));
        assert_eq!(
            crate::skills::soft_delete_profile_key(&pool, "u1", "profile.name")
                .await
                .unwrap(),
            1
        );
        assert!(profile_prompt(&pool, "u1").await.is_empty());
    }
}
