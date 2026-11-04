//! 长期记忆：对话块、画像抽取、会话删除级联、分层召回。
//! 写入 SQLite `memory_records`，与 Python memory_service 共用同一张表；
//! 召回用词法 + 可选 BGE 余弦，不依赖 Milvus 记忆 collection。

use serde_json::{Value, json};

use crate::embedder::Embedder;
use crate::llm::{ChatMessage, LlmClient};

const LONG_TERM_MEMORY_TOP_K: usize = 3;
const LONG_TERM_MEMORY_SNIPPET_LIMIT: usize = 240;
const CANDIDATE_LIMIT: i64 = 200;
const COSINE_RERANK_LIMIT: usize = 24;
const MIN_LEXICAL_SCORE: f64 = 0.0;

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

const LONG_TERM_MEMORY_HINTS: &[&str] = &[
    "之前", "刚才", "上次", "前面", "记得", "记忆", "偏好", "我说过", "你还记得", "根据我们",
    "继续", "沿用", "完整对话", "原始对话", "展开", "我的名字", "我叫什么",
];

const LONG_TERM_MEMORY_REFERENCE_HINTS: &[&str] = &["这个", "那个", "它", "这件事", "该问题"];

const MEMORY_TYPE_EVENT: &str = "event";
const MEMORY_STATUS_ACTIVE: &str = "active";
const DEFAULT_EVENT_EXPIRY_DAYS: i64 = 90;

#[derive(Debug, Clone, Copy)]
pub struct MemoryCleanupSettings {
    pub event_expiry_days: i64,
    pub profile_stale_days: i64,
    pub score_threshold: f64,
    pub purge_retention_days: i64,
}

impl MemoryCleanupSettings {
    pub fn from_config(config: &crate::config::Config) -> Self {
        MemoryCleanupSettings {
            event_expiry_days: config.memory_event_expiry_days.max(1),
            profile_stale_days: config.memory_profile_stale_days.max(1),
            score_threshold: config.memory_score_threshold.clamp(0.0, 100.0),
            purge_retention_days: config.memory_purge_retention_days.max(1),
        }
    }
}

impl Default for MemoryCleanupSettings {
    fn default() -> Self {
        MemoryCleanupSettings {
            event_expiry_days: DEFAULT_EVENT_EXPIRY_DAYS,
            profile_stale_days: 180,
            score_threshold: 20.0,
            purge_retention_days: 30,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryCleanupStats {
    pub rescored: i64,
    pub expired: i64,
    pub purged: i64,
}

impl MemoryCleanupStats {
    pub fn any(&self) -> bool {
        self.rescored > 0 || self.expired > 0 || self.purged > 0
    }
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct EventMemoryRow {
    memory_id: String,
    memory_key: Option<String>,
    content: String,
    conversation_id: Option<String>,
    session_id: Option<String>,
    turn_start: Option<i64>,
    turn_end: Option<i64>,
    importance_score: f64,
}

/// 分层召回命中（chunk_type 存在 SQLite `memory_key`：L3/L2/L1/dialogue）。
#[derive(Debug, Clone)]
pub struct EventMemoryHit {
    pub memory_id: String,
    pub content: String,
    pub memory_key: String,
    pub conversation_id: Option<String>,
    pub session_id: Option<String>,
    pub turn_start: Option<i64>,
    pub turn_end: Option<i64>,
    pub score: f64,
    pub lexical_score: f64,
    pub importance_score: f64,
}

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

pub fn clamp_score(value: f64) -> f64 {
    value.clamp(0.0, 100.0)
}

pub fn decay_importance_score(
    memory_type: &str,
    base_importance: f64,
    confidence: f64,
    access_count: i64,
    age_days: f64,
) -> f64 {
    let recency_days = if memory_type == MEMORY_TYPE_EVENT { 90.0 } else { 365.0 };
    let decayed = base_importance * confidence * (-age_days.max(0.0) / recency_days).exp();
    let access_bonus = 15.0_f64.min(((access_count.max(0) as f64) + 1.0).ln() * 2.5);
    (clamp_score(decayed + access_bonus) * 100.0).round() / 100.0
}

pub fn should_expire_memory(
    memory_type: &str,
    importance_score: f64,
    age_days: f64,
    expires_at: Option<&str>,
    now: chrono::NaiveDateTime,
    settings: &MemoryCleanupSettings,
) -> bool {
    if expires_at
        .and_then(crate::db::parse_dt)
        .is_some_and(|expiry| expiry <= now)
    {
        return true;
    }
    importance_score < settings.score_threshold
        && (memory_type == MEMORY_TYPE_EVENT || age_days >= settings.profile_stale_days as f64)
}

fn age_days_since(reference: Option<&str>, created_at: &str, now: chrono::NaiveDateTime) -> f64 {
    let reference_time = reference
        .and_then(crate::db::parse_dt)
        .or_else(|| crate::db::parse_dt(created_at))
        .unwrap_or(now);
    (now - reference_time).num_seconds().max(0) as f64 / 86400.0
}

#[derive(sqlx::FromRow)]
struct ActiveMemoryRow {
    memory_id: String,
    memory_type: String,
    base_importance_score: f64,
    confidence_score: f64,
    access_count: i64,
    last_accessed_at: Option<String>,
    created_at: String,
    expires_at: Option<String>,
}

pub async fn cleanup_memory_records(
    pool: &sqlx::SqlitePool,
    settings: &MemoryCleanupSettings,
) -> Result<MemoryCleanupStats, sqlx::Error> {
    let _ = ensure_schema(pool).await?;
    let now = crate::db::now_naive();
    let now_text = crate::db::fmt_dt(&now);
    let rows: Vec<ActiveMemoryRow> = sqlx::query_as(
        "SELECT memory_id, memory_type, base_importance_score, confidence_score, access_count, \
         last_accessed_at, created_at, expires_at \
         FROM memory_records WHERE status = 'active'",
    )
    .fetch_all(pool)
    .await?;

    let mut stats = MemoryCleanupStats::default();
    for row in rows {
        let age_days = age_days_since(row.last_accessed_at.as_deref(), &row.created_at, now);
        let score = decay_importance_score(
            &row.memory_type,
            row.base_importance_score,
            row.confidence_score,
            row.access_count,
            age_days,
        );
        let expire = should_expire_memory(
            &row.memory_type,
            score,
            age_days,
            row.expires_at.as_deref(),
            now,
            settings,
        );
        let status = if expire {
            stats.expired += 1;
            "expired"
        } else {
            MEMORY_STATUS_ACTIVE
        };
        sqlx::query(
            "UPDATE memory_records SET importance_score = ?, status = ?, updated_at = ? \
             WHERE memory_id = ?",
        )
        .bind(score)
        .bind(status)
        .bind(&now_text)
        .bind(&row.memory_id)
        .execute(pool)
        .await?;
        stats.rescored += 1;
    }

    let purge_before = crate::db::fmt_dt(&(now - chrono::Duration::days(settings.purge_retention_days.max(1))));
    let purged = sqlx::query(
        "DELETE FROM memory_records \
         WHERE status IN ('expired', 'superseded', 'deleted') AND updated_at < ?",
    )
    .bind(&purge_before)
    .execute(pool)
    .await?;
    stats.purged = purged.rows_affected() as i64;
    Ok(stats)
}

pub fn spawn_memory_maintenance(
    pool: sqlx::SqlitePool,
    settings: MemoryCleanupSettings,
    interval_secs: u64,
) {
    tokio::spawn(async move {
        let interval = std::time::Duration::from_secs(interval_secs.max(60));
        loop {
            match cleanup_memory_records(&pool, &settings).await {
                Ok(stats) if stats.any() => tracing::info!(
                    rescored = stats.rescored,
                    expired = stats.expired,
                    purged = stats.purged,
                    "长期记忆维护完成"
                ),
                Ok(_) => {}
                Err(err) => tracing::warn!("长期记忆维护失败: {err}"),
            }
            tokio::time::sleep(interval).await;
        }
    });
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
    expiry_days: i64,
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
        expiry_days,
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

/// 对齐 Python `_should_retrieve_long_term_memory`：direct/clarify 跳过，其余看记忆线索。
pub fn should_retrieve_event_memory(query: &str, route: Option<&str>) -> bool {
    match route.map(str::trim) {
        Some("direct") | Some("clarify") => return false,
        Some(route) if !matches!(route, "rag" | "skill") => return false,
        _ => {}
    }
    let normalized: String = query
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>()
        .to_lowercase();
    if normalized.is_empty() {
        return false;
    }
    LONG_TERM_MEMORY_HINTS
        .iter()
        .chain(LONG_TERM_MEMORY_REFERENCE_HINTS.iter())
        .any(|hint| normalized.contains(hint))
}

/// 按 L3 > L2 > L1 > dialogue 分层召回事件记忆。
pub async fn search_event_memories(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    query: &str,
    conversation_id: Option<&str>,
    session_id: Option<&str>,
    top_k: usize,
    embedder: Option<&Embedder>,
) -> Vec<EventMemoryHit> {
    if user_id.trim().is_empty() {
        return Vec::new();
    }
    let _ = ensure_schema(pool).await;
    let rows: Vec<EventMemoryRow> = match sqlx::query_as(
        "SELECT memory_id, memory_key, content, conversation_id, session_id, \
         turn_start, turn_end, importance_score \
         FROM memory_records \
         WHERE user_id = ? AND memory_type = 'event' AND status = 'active' \
         ORDER BY created_at DESC LIMIT ?",
    )
    .bind(user_id)
    .bind(CANDIDATE_LIMIT)
    .fetch_all(pool)
    .await
    {
        Ok(rows) => rows,
        Err(err) => {
            tracing::warn!("长期记忆读取失败: {err}");
            return Vec::new();
        }
    };
    if rows.is_empty() {
        return Vec::new();
    }

    let query = query.trim();
    let query_tokens = if query.is_empty() {
        Vec::new()
    } else {
        eks_kernels::tokenize_cjk(query)
    };
    let conversation_id = conversation_id.map(str::trim).filter(|item| !item.is_empty());
    let session_id = session_id.map(str::trim).filter(|item| !item.is_empty());
    let mut scored: Vec<EventMemoryHit> = rows
        .into_iter()
        .map(|row| {
            let memory_key = row
                .memory_key
                .as_deref()
                .unwrap_or("dialogue")
                .trim()
                .to_string();
            let lexical_score = if query.is_empty() {
                0.0
            } else {
                let row_tokens = eks_kernels::tokenize_cjk(&row.content);
                eks_kernels::lexical::lexical_match_score(&query_tokens, &row_tokens, query, &row.content)
            };
            let mut score = lexical_score * 0.55 + (row.importance_score / 100.0).clamp(0.0, 1.0) * 0.20;
            if conversation_id.is_some() && row.conversation_id.as_deref() == conversation_id {
                score += 0.12;
            }
            if session_id.is_some() && row.session_id.as_deref() == session_id {
                score += 0.04;
            }
            EventMemoryHit {
                memory_id: row.memory_id,
                content: row.content,
                memory_key,
                conversation_id: row.conversation_id,
                session_id: row.session_id,
                turn_start: row.turn_start,
                turn_end: row.turn_end,
                score,
                lexical_score,
                importance_score: row.importance_score,
            }
        })
        .collect();

    if let Some(embedder) = embedder.filter(|_| !query.is_empty()) {
        blend_cosine_scores(&mut scored, query, embedder);
    }

    if !query.is_empty() {
        let compact_query = eks_kernels::normalize_text(query);
        scored.retain(|hit| {
            hit.lexical_score > MIN_LEXICAL_SCORE
                || hit.score >= 0.35
                || (!compact_query.is_empty()
                    && eks_kernels::normalize_text(&hit.content).contains(&compact_query))
        });
    }
    scored.sort_by(|left, right| {
        level_rank(&left.memory_key)
            .cmp(&level_rank(&right.memory_key))
            .then_with(|| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    let limit = top_k.max(1);
    scored.truncate(limit);
    mark_accessed(pool, user_id, &scored).await;
    scored
}

/// 对齐 Python `_build_hierarchical_memory_lines`：有命中才生成 system 提示。
pub async fn hierarchical_prompt(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    query: &str,
    conversation_id: Option<&str>,
    session_id: Option<&str>,
    top_k: usize,
    embedder: Option<&Embedder>,
    route: Option<&str>,
) -> String {
    if !should_retrieve_event_memory(query, route) {
        return String::new();
    }
    let hits = search_event_memories(
        pool,
        user_id,
        query,
        conversation_id,
        session_id,
        if top_k == 0 { LONG_TERM_MEMORY_TOP_K } else { top_k },
        embedder,
    )
    .await;
    format_hierarchical_prompt(&hits)
}

/// 按摘要引用、会话或检索提示展开当前用户的原始对话。
#[derive(Debug, Clone, Default)]
pub struct OriginalConversationQuery {
    pub memory_ids: Vec<String>,
    pub query: String,
    pub conversation_id: String,
    pub turn_start: Option<i64>,
    pub turn_end: Option<i64>,
    pub include_neighbors: i64,
    pub include_all_conversation: bool,
    pub top_k: usize,
    pub max_chars: usize,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct OriginalMemoryRow {
    memory_id: String,
    memory_key: Option<String>,
    content: String,
    conversation_id: Option<String>,
    session_id: Option<String>,
    turn_start: Option<i64>,
    turn_end: Option<i64>,
    created_at: String,
}

fn empty_original_conversation(summary: &str) -> Value {
    json!({
        "mode": "original_conversation",
        "summary": summary,
        "summary_references": [],
        "original_chunks": [],
    })
}

fn serialize_original_row(row: &OriginalMemoryRow) -> Value {
    json!({
        "memory_id": row.memory_id,
        "content": row.content,
        "conversation_id": row.conversation_id,
        "session_id": row.session_id,
        "chunk_type": row.memory_key.as_deref().unwrap_or("dialogue"),
        "topic": "",
        "turn_start": row.turn_start,
        "turn_end": row.turn_end,
        "status": "active",
    })
}

fn clamp_original_query(query: &OriginalConversationQuery) -> OriginalConversationQuery {
    let mut turn_start = query.turn_start.map(|value| value.max(0));
    let mut turn_end = query.turn_end.map(|value| value.max(0));
    if let (Some(start), Some(end)) = (turn_start, turn_end) {
        if end < start {
            turn_start = Some(end);
            turn_end = Some(start);
        }
    }
    OriginalConversationQuery {
        memory_ids: query
            .memory_ids
            .iter()
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect(),
        query: query.query.trim().to_string(),
        conversation_id: query.conversation_id.trim().to_string(),
        turn_start,
        turn_end,
        include_neighbors: query.include_neighbors.clamp(0, 5),
        include_all_conversation: query.include_all_conversation,
        top_k: query.top_k.clamp(1, 50),
        max_chars: query.max_chars.clamp(500, 50_000),
    }
}

fn bound_original_chunks(chunks: Vec<Value>, max_chars: usize) -> (Vec<Value>, bool) {
    let mut total_chars = 0usize;
    let mut bounded = Vec::new();
    for mut chunk in chunks {
        let remaining = max_chars.saturating_sub(total_chars);
        if remaining == 0 {
            break;
        }
        let content = chunk
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if content.chars().count() > remaining {
            let truncated: String = content.chars().take(remaining).collect();
            chunk["content"] = json!(truncated);
        }
        total_chars += chunk
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .chars()
            .count();
        bounded.push(chunk);
    }
    (bounded, total_chars >= max_chars)
}

fn chunk_sort_key(chunk: &Value) -> (String, i64, String) {
    (
        chunk
            .get("conversation_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        chunk.get("turn_start").and_then(Value::as_i64).unwrap_or(0),
        chunk
            .get("memory_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    )
}

fn turns_overlap(row: &OriginalMemoryRow, start: Option<i64>, end: Option<i64>) -> bool {
    if row.turn_start.is_none() && row.turn_end.is_none() {
        return true;
    }
    if let Some(start) = start {
        if row.turn_end.is_some_and(|turn_end| turn_end < start) {
            return false;
        }
    }
    if let Some(end) = end {
        if row.turn_start.is_some_and(|turn_start| turn_start > end) {
            return false;
        }
    }
    true
}

fn recovered_from_summary(reference: &Value, conversation_id: &str, start: Option<i64>, end: Option<i64>) -> Option<Value> {
    let content = reference.get("content").and_then(Value::as_str).unwrap_or("");
    let transcript = content.split_once("【原文】")?.1.trim();
    if transcript.is_empty() {
        return None;
    }
    Some(json!({
        "memory_id": reference.get("memory_id"),
        "conversation_id": conversation_id,
        "session_id": reference.get("session_id"),
        "chunk_type": "dialogue_recovered",
        "turn_start": start,
        "turn_end": end,
        "content": transcript,
        "source_summary_memory_id": reference.get("memory_id"),
        "score": reference.get("score").and_then(Value::as_f64).unwrap_or(0.0),
    }))
}

async fn load_active_event_rows(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    memory_ids: &[String],
    conversation_id: Option<&str>,
    dialogue_only: bool,
    limit: i64,
) -> Vec<OriginalMemoryRow> {
    let mut builder = sqlx::QueryBuilder::new(
        "SELECT memory_id, memory_key, content, conversation_id, session_id, \
         turn_start, turn_end, created_at \
         FROM memory_records WHERE user_id = ",
    );
    builder.push_bind(user_id);
    builder.push(" AND memory_type = 'event' AND status = 'active'");
    if !memory_ids.is_empty() {
        builder.push(" AND memory_id IN (");
        {
            let mut separated = builder.separated(", ");
            for memory_id in memory_ids {
                separated.push_bind(memory_id);
            }
        }
        builder.push(")");
    }
    if let Some(conversation_id) = conversation_id.filter(|item| !item.is_empty()) {
        builder.push(" AND conversation_id = ");
        builder.push_bind(conversation_id);
    }
    if dialogue_only {
        builder.push(" AND memory_key = 'dialogue'");
    }
    builder.push(" ORDER BY created_at ASC LIMIT ");
    builder.push_bind(limit);
    builder
        .build_query_as::<OriginalMemoryRow>()
        .fetch_all(pool)
        .await
        .unwrap_or_else(|err| {
            tracing::warn!("原文记忆读取失败: {err}");
            Vec::new()
        })
}

async fn mark_ids_accessed(pool: &sqlx::SqlitePool, user_id: &str, memory_ids: &[String]) {
    let hits: Vec<EventMemoryHit> = memory_ids
        .iter()
        .filter(|item| !item.is_empty())
        .map(|memory_id| EventMemoryHit {
            memory_id: memory_id.clone(),
            content: String::new(),
            memory_key: String::new(),
            conversation_id: None,
            session_id: None,
            turn_start: None,
            turn_end: None,
            score: 0.0,
            lexical_score: 0.0,
            importance_score: 0.0,
        })
        .collect();
    mark_accessed(pool, user_id, &hits).await;
}

pub async fn retrieve_original_conversation(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    query: OriginalConversationQuery,
) -> Value {
    if user_id.trim().is_empty() {
        return empty_original_conversation("请提供摘要 memory_id、检索提示或 conversation_id。");
    }
    let _ = ensure_schema(pool).await;
    let query = clamp_original_query(&query);
    if query.memory_ids.is_empty() && query.query.is_empty() && query.conversation_id.is_empty() {
        return empty_original_conversation("请提供摘要 memory_id、检索提示或 conversation_id。");
    }

    let mut summary_references: Vec<Value> = Vec::new();
    if !query.memory_ids.is_empty() {
        let rows = load_active_event_rows(
            pool,
            user_id,
            &query.memory_ids,
            None,
            false,
            query.memory_ids.len() as i64,
        )
        .await;
        summary_references.extend(rows.iter().map(serialize_original_row));
    } else if query.include_all_conversation && !query.conversation_id.is_empty() {
        let mut rows = load_active_event_rows(
            pool,
            user_id,
            &[],
            Some(query.conversation_id.as_str()),
            true,
            1000,
        )
        .await;
        rows.sort_by_key(|row| (row.turn_start.unwrap_or(0), row.created_at.clone()));
        let chunks = rows.iter().map(serialize_original_row).collect();
        let (bounded, truncated) = bound_original_chunks(chunks, query.max_chars);
        mark_ids_accessed(
            pool,
            user_id,
            &bounded
                .iter()
                .filter_map(|item| item.get("memory_id").and_then(Value::as_str).map(str::to_string))
                .collect::<Vec<_>>(),
        )
        .await;
        return json!({
            "mode": "original_conversation",
            "summary": format!("已展开当前会话的 {} 个原始对话块。", bounded.len()),
            "summary_references": [],
            "original_chunks": bounded,
            "truncated": truncated,
        });
    } else if !query.query.is_empty() {
        let hits = search_event_memories(
            pool,
            user_id,
            &query.query,
            if query.conversation_id.is_empty() {
                None
            } else {
                Some(query.conversation_id.as_str())
            },
            None,
            query.top_k,
            None,
        )
        .await;
        summary_references.extend(hits.iter().filter(|hit| hit.memory_key.starts_with("semantic_memory")).map(|hit| {
            json!({
                "memory_id": hit.memory_id,
                "content": hit.content,
                "conversation_id": hit.conversation_id,
                "session_id": hit.session_id,
                "chunk_type": hit.memory_key,
                "topic": "",
                "turn_start": hit.turn_start,
                "turn_end": hit.turn_end,
                "score": hit.score,
            })
        }));
        if summary_references.is_empty() {
            let raw_chunks: Vec<Value> = hits
                .iter()
                .filter(|hit| hit.memory_key == "dialogue")
                .map(|hit| {
                    json!({
                        "memory_id": hit.memory_id,
                        "conversation_id": hit.conversation_id,
                        "turn_start": hit.turn_start,
                        "turn_end": hit.turn_end,
                        "content": hit.content,
                        "score": hit.score,
                    })
                })
                .collect();
            return json!({
                "mode": "original_conversation",
                "summary": format!("已按语义找到 {} 个原始对话块。", raw_chunks.len()),
                "summary_references": [],
                "original_chunks": raw_chunks,
            });
        }
    }

    let mut original_chunks: Vec<Value> = Vec::new();
    let mut seen_raw_ids = std::collections::HashSet::new();
    for reference in summary_references.iter().take(query.top_k) {
        let reference_conversation = reference
            .get("conversation_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if !query.conversation_id.is_empty()
            && !reference_conversation.is_empty()
            && reference_conversation != query.conversation_id
        {
            continue;
        }
        let ref_conversation = if reference_conversation.is_empty() {
            query.conversation_id.clone()
        } else {
            reference_conversation
        };
        if ref_conversation.is_empty() {
            continue;
        }
        let mut ref_start = if query.turn_start.is_some() {
            query.turn_start
        } else {
            reference.get("turn_start").and_then(Value::as_i64)
        };
        let mut ref_end = if query.turn_end.is_some() {
            query.turn_end
        } else {
            reference.get("turn_end").and_then(Value::as_i64)
        };
        if let Some(start) = ref_start {
            ref_start = Some((start - query.include_neighbors).max(0));
        }
        if let Some(end) = ref_end {
            ref_end = Some(end + query.include_neighbors);
        }
        let rows = load_active_event_rows(
            pool,
            user_id,
            &[],
            Some(ref_conversation.as_str()),
            true,
            (query.top_k * 4) as i64,
        )
        .await;
        let mut matched_any = false;
        for row in rows {
            if !turns_overlap(&row, ref_start, ref_end) || !seen_raw_ids.insert(row.memory_id.clone()) {
                continue;
            }
            matched_any = true;
            let mut serialized = serialize_original_row(&row);
            serialized["source_summary_memory_id"] = reference.get("memory_id").cloned().unwrap_or(Value::Null);
            serialized["score"] = json!(reference.get("score").and_then(Value::as_f64).unwrap_or(0.0));
            original_chunks.push(serialized);
        }
        if !matched_any {
            if let Some(recovered) = recovered_from_summary(reference, &ref_conversation, ref_start, ref_end) {
                original_chunks.push(recovered);
            }
        }
    }

    original_chunks.sort_by_key(chunk_sort_key);
    let (bounded, truncated) = bound_original_chunks(original_chunks, query.max_chars);
    let mut accessed: Vec<String> = summary_references
        .iter()
        .filter_map(|item| item.get("memory_id").and_then(Value::as_str).map(str::to_string))
        .collect();
    accessed.extend(
        bounded
            .iter()
            .filter_map(|item| item.get("memory_id").and_then(Value::as_str).map(str::to_string)),
    );
    mark_ids_accessed(pool, user_id, &accessed).await;
    let summary_preview: Vec<Value> = summary_references
        .iter()
        .take(query.top_k)
        .map(|item| {
            json!({
                "memory_id": item.get("memory_id"),
                "conversation_id": item.get("conversation_id"),
                "chunk_type": item.get("chunk_type"),
                "topic": item.get("topic"),
                "turn_start": item.get("turn_start"),
                "turn_end": item.get("turn_end"),
            })
        })
        .collect();
    json!({
        "mode": "original_conversation",
        "summary": format!("已展开 {} 个原始对话块。", bounded.len()),
        "summary_references": summary_preview,
        "original_chunks": bounded,
        "truncated": truncated,
    })
}

pub fn format_hierarchical_prompt(hits: &[EventMemoryHit]) -> String {
    if hits.is_empty() {
        return String::new();
    }
    let mut lines = vec!["以下是与当前问题相关的长期记忆，仅在相关时参考，不能把未确认信息当作事实：".to_string()];
    for (index, hit) in hits.iter().enumerate() {
        let reference = json!({
            "memory_id": hit.memory_id,
            "conversation_id": hit.conversation_id,
            "chunk_type": hit.memory_key,
            "topic": "",
            "turn_start": hit.turn_start,
            "turn_end": hit.turn_end,
            "disclosure_tool": "memory_retrieve_original",
        });
        lines.push(format!(
            "{}. {}\n   摘要引用：{}",
            index + 1,
            compact_content(&hit.content),
            reference
        ));
    }
    lines.join("\n")
}

fn level_rank(memory_key: &str) -> i32 {
    match memory_key.trim() {
        "semantic_memory_l3" => 0,
        "semantic_memory_l2" => 1,
        "semantic_memory_l1" => 2,
        "dialogue" => 3,
        _ => 9,
    }
}

fn compact_content(content: &str) -> String {
    let normalized = content.trim();
    if normalized.is_empty() {
        return String::new();
    }
    let mut topic = String::new();
    let mut summary = String::new();
    for line in normalized.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("【主题】") {
            topic = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("【摘要】") {
            summary = rest.trim().to_string();
        }
    }
    if !topic.is_empty() && !summary.is_empty() {
        return format!("{topic}：{summary}");
    }
    if !summary.is_empty() {
        return summary;
    }
    if !topic.is_empty() {
        return topic;
    }
    normalized.replace('\n', " ").chars().take(LONG_TERM_MEMORY_SNIPPET_LIMIT).collect()
}

fn blend_cosine_scores(hits: &mut [EventMemoryHit], query: &str, embedder: &Embedder) {
    if hits.is_empty() {
        return;
    }
    let mut order: Vec<usize> = (0..hits.len()).collect();
    order.sort_by(|left, right| {
        hits[*right]
            .score
            .partial_cmp(&hits[*left].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(COSINE_RERANK_LIMIT);
    let query_vec = match embedder.embed_queries(&[query.to_string()]) {
        Ok(vectors) if !vectors.is_empty() => vectors.into_iter().next().unwrap_or_default(),
        Ok(_) => return,
        Err(err) => {
            tracing::warn!("长期记忆 query 向量化失败: {err}");
            return;
        }
    };
    let texts: Vec<String> = order.iter().map(|&index| hits[index].content.clone()).collect();
    let doc_vecs = match embedder.embed_documents(&texts) {
        Ok(vectors) => vectors,
        Err(err) => {
            tracing::warn!("长期记忆文档向量化失败: {err}");
            return;
        }
    };
    for (offset, index) in order.into_iter().enumerate() {
        if let Some(vector) = doc_vecs.get(offset) {
            hits[index].score += 0.25 * cosine(&query_vec, vector).max(0.0);
        }
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let n = left.len().min(right.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let a = left[i] as f64;
        let b = right[i] as f64;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

async fn mark_accessed(pool: &sqlx::SqlitePool, user_id: &str, hits: &[EventMemoryHit]) {
    if hits.is_empty() {
        return;
    }
    let now = crate::db::fmt_dt(&crate::db::now_naive());
    for hit in hits {
        if let Err(err) = sqlx::query(
            "UPDATE memory_records SET access_count = access_count + 1, last_accessed_at = ?, updated_at = ? \
             WHERE user_id = ? AND memory_id = ? AND status = 'active'",
        )
        .bind(&now)
        .bind(&now)
        .bind(user_id)
        .bind(&hit.memory_id)
        .execute(pool)
        .await
        {
            tracing::warn!("更新长期记忆访问统计失败: {err}");
        }
    }
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
    expiry_days: i64,
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
        expiry_days,
    )
    .await
    .map_err(|err| err.to_string())?;
    Ok(true)
}

pub(crate) async fn insert_event_memory(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    conversation_id: &str,
    session_id: &str,
    memory_key: &str,
    content: &str,
    importance: f64,
    expiry_days: i64,
) -> Result<String, sqlx::Error> {
    insert_event_memory_at(
        pool,
        user_id,
        conversation_id,
        session_id,
        memory_key,
        content,
        importance,
        expiry_days,
        None,
        None,
    )
    .await
}

pub(crate) async fn insert_event_memory_at(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    conversation_id: &str,
    session_id: &str,
    memory_key: &str,
    content: &str,
    importance: f64,
    expiry_days: i64,
    turn_start: Option<i64>,
    turn_end: Option<i64>,
) -> Result<String, sqlx::Error> {
    let now = crate::db::now_naive();
    let now_text = crate::db::fmt_dt(&now);
    let expires_at = crate::db::fmt_dt(&(now + chrono::Duration::days(expiry_days.max(1))));
    let memory_id = format!(
        "event:{}:{}:{}",
        user_id,
        memory_key,
        uuid::Uuid::new_v4().simple()
    );
    sqlx::query(
        "INSERT INTO memory_records (
            memory_id, user_id, memory_type, memory_key, content, conversation_id, session_id,
            turn_start, turn_end, importance_score, base_importance_score, confidence_score,
            access_count, status, version, created_at, updated_at, expires_at
        ) VALUES (?, ?, 'event', ?, ?, ?, ?, ?, ?, ?, ?, 0.7, 0, 'active', 1, ?, ?, ?)",
    )
    .bind(&memory_id)
    .bind(user_id)
    .bind(memory_key)
    .bind(content)
    .bind(empty_to_none(conversation_id))
    .bind(empty_to_none(session_id))
    .bind(turn_start)
    .bind(turn_end)
    .bind(importance)
    .bind(importance)
    .bind(&now_text)
    .bind(&now_text)
    .bind(&expires_at)
    .execute(pool)
    .await?;
    Ok(memory_id)
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
        persist_turn(&pool, None, "u1", "c1", "s1", "你好", "你好，我在。", DEFAULT_EVENT_EXPIRY_DAYS).await;
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

    #[test]
    fn gates_event_memory_retrieval_like_python() {
        assert!(should_retrieve_event_memory("你还记得上次说的报销时限吗", Some("rag")));
        assert!(should_retrieve_event_memory("继续刚才的话题", Some("rag")));
        assert!(should_retrieve_event_memory("这个怎么处理", Some("skill")));
        assert!(!should_retrieve_event_memory("差旅报销流程是什么", Some("rag")));
        assert!(!should_retrieve_event_memory("你还记得上次说的吗", Some("direct")));
        assert!(!should_retrieve_event_memory("你还记得吗", Some("clarify")));
        assert!(!should_retrieve_event_memory("", Some("rag")));
    }

    #[test]
    fn compacts_topic_summary_blocks() {
        assert_eq!(
            compact_content("【主题】差旅\n【摘要】报销时限三天"),
            "差旅：报销时限三天"
        );
        assert!(compact_content("用户: 你好\n助手: 你好").contains("用户: 你好"));
    }

    #[tokio::test]
    async fn ranks_l1_over_dialogue_and_drops_unrelated() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        insert_event_memory(
            &pool,
            "u1",
            "c1",
            "s1",
            "dialogue",
            "用户: 报销时限是几天？\n助手: 差旅报销需要在发生后 3 天内提交。",
            55.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
        )
        .await
        .unwrap();
        insert_event_memory(
            &pool,
            "u1",
            "c1",
            "s1",
            "semantic_memory_l1",
            "【主题】差旅报销\n【摘要】差旅费用须在发生后 3 天内提交报销申请。",
            70.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
        )
        .await
        .unwrap();
        insert_event_memory(
            &pool,
            "u1",
            "c2",
            "s2",
            "semantic_memory_l1",
            "【主题】天气闲聊\n【摘要】今天下午可能下雨，出门记得带伞。",
            90.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
        )
        .await
        .unwrap();
        insert_event_memory(
            &pool,
            "u2",
            "c9",
            "s9",
            "semantic_memory_l1",
            "【主题】差旅报销\n【摘要】另一名用户的报销时限是 7 天。",
            99.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
        )
        .await
        .unwrap();

        let hits = search_event_memories(&pool, "u1", "报销时限是多久", Some("c1"), Some("s1"), 3, None).await;
        assert!(!hits.is_empty());
        assert_eq!(hits[0].memory_key, "semantic_memory_l1");
        assert!(hits[0].content.contains("3 天"));
        assert!(hits.iter().all(|hit| !hit.content.contains("下雨") && !hit.content.contains("7 天")));

        let prompt = hierarchical_prompt(
            &pool,
            "u1",
            "你还记得上次报销时限吗",
            Some("c1"),
            Some("s1"),
            3,
            None,
            Some("rag"),
        )
        .await;
        assert!(prompt.contains("长期记忆"));
        assert!(prompt.contains("差旅报销"));
        assert!(prompt.contains("memory_retrieve_original"));
        assert!(prompt.contains("semantic_memory_l1"));

        let access: i64 = sqlx::query_scalar(
            "SELECT access_count FROM memory_records \
             WHERE user_id = 'u1' AND memory_key = 'semantic_memory_l1' AND content LIKE '%3 天%'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(access >= 1);

        assert!(
            hierarchical_prompt(&pool, "u1", "差旅报销流程是什么", Some("c1"), None, 3, None, Some("rag"))
                .await
                .is_empty()
        );
    }

    #[tokio::test]
    async fn prefers_higher_semantic_level_when_both_match() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        insert_event_memory(&pool, "u1", "c1", "s1", "dialogue", "用户问了差旅报销时限。", 80.0, DEFAULT_EVENT_EXPIRY_DAYS)
            .await
            .unwrap();
        insert_event_memory(
            &pool,
            "u1",
            "c1",
            "s1",
            "semantic_memory_l3",
            "差旅报销时限：发生后 3 天内提交。",
            40.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
        )
        .await
        .unwrap();
        let hits = search_event_memories(&pool, "u1", "差旅报销时限", None, None, 2, None).await;
        assert_eq!(hits[0].memory_key, "semantic_memory_l3");
        assert_eq!(hits[1].memory_key, "dialogue");
    }

    #[test]
    fn decay_matches_python_formula() {
        assert_eq!(decay_importance_score("event", 50.0, 0.5, 0, 0.0), 25.0);
        let aged = decay_importance_score("event", 50.0, 1.0, 0, 90.0);
        let expected = (50.0 * (-1.0_f64).exp() * 100.0).round() / 100.0;
        assert_eq!(aged, expected);
        assert_eq!(decay_importance_score("profile", 80.0, 1.0, 0, 0.0), 80.0);
        let with_access = decay_importance_score("event", 50.0, 1.0, 10, 0.0);
        assert!(with_access > 50.0);
        assert!(with_access <= 65.0);
    }

    #[test]
    fn expire_rules_cover_deadline_score_and_profile_stale() {
        let settings = MemoryCleanupSettings::default();
        let now = crate::db::now_naive();
        let past = crate::db::fmt_dt(&(now - chrono::Duration::days(1)));
        assert!(should_expire_memory("event", 90.0, 0.0, Some(&past), now, &settings));
        assert!(should_expire_memory("event", 10.0, 0.0, None, now, &settings));
        assert!(!should_expire_memory("profile", 10.0, 10.0, None, now, &settings));
        assert!(should_expire_memory("profile", 10.0, 180.0, None, now, &settings));
        assert!(!should_expire_memory("event", 50.0, 0.0, None, now, &settings));
    }

    #[tokio::test]
    async fn cleanup_expires_low_score_events_and_purges_old_rows() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        insert_event_memory(
            &pool,
            "u1",
            "c1",
            "s1",
            "dialogue",
            "低分事件",
            5.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
        )
        .await
        .unwrap();
        upsert_profile_candidates(
            &pool,
            "u1",
            "c1",
            "s1",
            &[json!({
                "memory_key": "profile.name",
                "value": "张三",
                "confidence": 0.1,
                "importance": 5,
                "action": "upsert"
            })],
        )
        .await
        .unwrap();

        let stats = cleanup_memory_records(&pool, &MemoryCleanupSettings::default())
            .await
            .unwrap();
        assert_eq!(stats.expired, 1);
        assert_eq!(stats.rescored, 2);
        let event_status: String = sqlx::query_scalar(
            "SELECT status FROM memory_records WHERE memory_type = 'event' AND content = '低分事件'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(event_status, "expired");
        let profile_status: String = sqlx::query_scalar(
            "SELECT status FROM memory_records WHERE memory_type = 'profile' AND memory_key = 'profile.name'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(profile_status, "active");

        let old = crate::db::fmt_dt(&(crate::db::now_naive() - chrono::Duration::days(40)));
        sqlx::query("UPDATE memory_records SET updated_at = ? WHERE status = 'expired'")
            .bind(&old)
            .execute(&pool)
            .await
            .unwrap();
        let purged = cleanup_memory_records(&pool, &MemoryCleanupSettings::default())
            .await
            .unwrap();
        assert_eq!(purged.purged, 1);
        let remaining: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_records WHERE content = '低分事件'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(remaining, 0);
    }

    #[tokio::test]
    async fn original_conversation_expands_summary_range_for_current_user() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        let summary_id = insert_event_memory_at(
            &pool,
            "u1",
            "c1",
            "s1",
            "semantic_memory_l1",
            "【主题】项目 Alpha\n【摘要】用户确认了上线计划。",
            70.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
            Some(2),
            Some(3),
        )
        .await
        .unwrap();
        let raw_id = insert_event_memory_at(
            &pool,
            "u1",
            "c1",
            "s1",
            "dialogue",
            "用户: 请确认 Alpha 上线计划\n助手: 已确认本周五上线。",
            55.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
            Some(2),
            Some(3),
        )
        .await
        .unwrap();
        insert_event_memory_at(
            &pool,
            "u1",
            "c1",
            "s1",
            "dialogue",
            "用户: 昨天的天气如何\n助手: 晴。",
            40.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
            Some(0),
            Some(0),
        )
        .await
        .unwrap();
        insert_event_memory_at(
            &pool,
            "u2",
            "c9",
            "s9",
            "semantic_memory_l1",
            "其他用户的摘要",
            90.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
            Some(0),
            Some(0),
        )
        .await
        .unwrap();

        let result = retrieve_original_conversation(
            &pool,
            "u1",
            OriginalConversationQuery {
                memory_ids: vec![summary_id.clone()],
                include_neighbors: 0,
                top_k: 8,
                max_chars: 12000,
                ..OriginalConversationQuery::default()
            },
        )
        .await;
        assert_eq!(
            result.pointer("/summary_references/0/memory_id").and_then(Value::as_str),
            Some(summary_id.as_str())
        );
        assert_eq!(
            result.pointer("/original_chunks/0/memory_id").and_then(Value::as_str),
            Some(raw_id.as_str())
        );
        assert!(result.pointer("/original_chunks/0/content").and_then(Value::as_str).unwrap_or("").contains("Alpha 上线计划"));
        assert_eq!(result.get("original_chunks").and_then(Value::as_array).map(Vec::len), Some(1));

        let hidden = retrieve_original_conversation(
            &pool,
            "u1",
            OriginalConversationQuery {
                memory_ids: vec!["missing-other".into()],
                top_k: 8,
                max_chars: 12000,
                ..OriginalConversationQuery::default()
            },
        )
        .await;
        assert_eq!(hidden.get("summary_references").and_then(Value::as_array).map(Vec::len), Some(0));
        assert_eq!(hidden.get("original_chunks").and_then(Value::as_array).map(Vec::len), Some(0));
    }

    #[tokio::test]
    async fn original_conversation_recovers_embedded_transcript_and_whole_session() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        let summary_id = insert_event_memory_at(
            &pool,
            "u1",
            "c1",
            "s1",
            "semantic_memory_l1",
            "【主题】Alpha\n【摘要】确认上线。\n【原文】用户: 周五上线吗\n助手: 是的",
            70.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
            Some(2),
            Some(3),
        )
        .await
        .unwrap();
        let recovered = retrieve_original_conversation(
            &pool,
            "u1",
            OriginalConversationQuery {
                memory_ids: vec![summary_id],
                include_neighbors: 0,
                top_k: 8,
                max_chars: 12000,
                ..OriginalConversationQuery::default()
            },
        )
        .await;
        assert_eq!(
            recovered.pointer("/original_chunks/0/chunk_type").and_then(Value::as_str),
            Some("dialogue_recovered")
        );
        assert!(recovered.pointer("/original_chunks/0/content").and_then(Value::as_str).unwrap_or("").contains("周五上线吗"));

        insert_event_memory_at(
            &pool,
            "u1",
            "c2",
            "s2",
            "dialogue",
            "用户: 第一句\n助手: 收到",
            55.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
            Some(0),
            Some(0),
        )
        .await
        .unwrap();
        insert_event_memory_at(
            &pool,
            "u1",
            "c2",
            "s2",
            "dialogue",
            "用户: 第二句\n助手: 记下了",
            55.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
            Some(1),
            Some(1),
        )
        .await
        .unwrap();
        let whole = retrieve_original_conversation(
            &pool,
            "u1",
            OriginalConversationQuery {
                conversation_id: "c2".into(),
                include_all_conversation: true,
                top_k: 8,
                max_chars: 12000,
                ..OriginalConversationQuery::default()
            },
        )
        .await;
        assert_eq!(whole.get("original_chunks").and_then(Value::as_array).map(Vec::len), Some(2));
        assert!(whole.get("summary").and_then(Value::as_str).unwrap_or("").contains("当前会话"));
    }
}
