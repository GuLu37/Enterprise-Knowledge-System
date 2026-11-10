//! 长期记忆：对话块、画像抽取、会话删除级联、分层召回。
//! SQLite `memory_records` 管生命周期；有 milvus_lite `memory_chunks` 时写入并 ANN 召回，
//! 否则回退词法 + 可选 BGE 余弦。

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use serde_json::{Value, json};

use crate::embedder::Embedder;
use crate::llm::{ChatMessage, LlmClient};
use crate::memory_chunk::{
    build_conversation_chunks, build_semantic_memory_chunks, high_confidence_topic_continuation,
    normalize_conversation_message, user_assistant_messages, CHAT_CHUNK_OVERLAP, CHAT_CHUNK_SIZE,
    LONG_TERM_MEMORY_TOKEN_LIMIT, ConversationMessage,
};
use crate::milvus_lite::MilvusLiteStore;

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
const EMPTY_USER_ID_MESSAGE: &str = "user_id 不能为空，长期记忆必须绑定到用户";

/// 对齐 Python `_require_user_id`。
pub fn require_user_id(user_id: &str) -> Result<&str, String> {
    let normalized = user_id.trim();
    if normalized.is_empty() {
        Err(EMPTY_USER_ID_MESSAGE.into())
    } else {
        Ok(normalized)
    }
}
pub const MEMORY_PERSIST_QUEUE_SIZE: usize = 100;
pub const CONSOLIDATION_LEVELS: [(i64, i32); 3] = [(10, 1), (20, 2), (40, 3)];
const CONSOLIDATION_RETRY_DELAYS: [u64; 3] = [60, 300, 1800];
const L1_SUMMARY_SYSTEM_PROMPT: &str = "你是长期记忆摘要器。请将给定对话总结成适合长期记忆的一句话摘要，并给出主题。\
只输出 JSON，格式为 {\"topic\": \"主题\", \"summary\": \"一句摘要\"}，不要输出额外解释。\
主题要短，摘要要准确，不要编造。";
const TOPIC_CHANGE_SYSTEM_PROMPT: &str = "你是对话话题判断器。请判断下面两段对话是否已经发生明显话题切换。\
只输出 JSON，格式为 {\"changed\": true/false}，不要输出额外解释。";
const MERGE_SUMMARY_SYSTEM_PROMPT: &str = "你是长期记忆递进摘要器。把以下多个摘要块压缩成一段更精炼的高层摘要。\
保留跨块共有的主题、结论、偏好和待办；丢弃单块琐碎细节。\
只输出严格 JSON，格式：{\"topic\":\"主题\",\"summary\":\"一段摘要\"}";
const LOW_INFORMATION_DIALOGUE: &[&str] = &[
    "好", "好的", "收到", "明白", "了解", "知道了", "可以", "行", "嗯", "嗯嗯", "谢谢", "感谢",
    "ok", "okay", "thanks", "thankyou",
];

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
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS memory_consolidation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            level INTEGER NOT NULL,
            last_merged_turn INTEGER NOT NULL DEFAULT 0,
            last_merged_at TEXT,
            UNIQUE(user_id, conversation_id, level)
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS ix_memory_consolidation_user_conv \
         ON memory_consolidation (user_id, conversation_id)",
    )
    .execute(pool)
    .await?;
    migrate_memory_turn_columns(pool).await?;
    Ok(())
}

async fn migrate_memory_turn_columns(pool: &sqlx::SqlitePool) -> Result<(), sqlx::Error> {
    let columns = crate::db::table_column_names(pool, "memory_records").await?;
    if !columns.contains("turn_start") {
        sqlx::query("ALTER TABLE memory_records ADD COLUMN turn_start INTEGER")
            .execute(pool)
            .await?;
    }
    if !columns.contains("turn_end") {
        sqlx::query("ALTER TABLE memory_records ADD COLUMN turn_end INTEGER")
            .execute(pool)
            .await?;
    }
    Ok(())
}

/// 对齐 Python milvus `memory_chunks` 写入/召回。测试不配置时走 SQLite 词法回退。
#[derive(Clone)]
pub struct MemoryVectorSink {
    pub store: MilvusLiteStore,
    pub collection: String,
    pub embedder: Arc<Embedder>,
    pub embedding_dimension: usize,
}

static MEMORY_VECTOR_SINK: OnceLock<MemoryVectorSink> = OnceLock::new();

pub fn configure_memory_vectors(sink: MemoryVectorSink) {
    let _ = MEMORY_VECTOR_SINK.set(sink);
}

fn memory_vector_sink() -> Option<&'static MemoryVectorSink> {
    MEMORY_VECTOR_SINK.get()
}

/// SQLite `memory_key` 在 Rust 历史行里是 chunk_type，在 Python 行里是 logical_key。
pub fn event_chunk_type(memory_key: Option<&str>) -> String {
    let key = memory_key.unwrap_or("dialogue").trim();
    if key.is_empty() {
        return "dialogue".into();
    }
    if key == "dialogue" || key.starts_with("semantic_memory") {
        return key.to_string();
    }
    let parts: Vec<&str> = key.split(':').collect();
    if parts.len() >= 4 && parts[0] == "event" {
        return parts[3].to_string();
    }
    key.to_string()
}

fn event_logical_key(
    user_id: &str,
    conversation_id: &str,
    chunk_type: &str,
    memory_id: &str,
    chunk_index: i64,
) -> String {
    let conv = if conversation_id.trim().is_empty() {
        "unknown"
    } else {
        conversation_id.trim()
    };
    if chunk_type == "dialogue" {
        format!("event:{user_id}:{conv}:dialogue:{memory_id}")
    } else {
        format!("event:{user_id}:{conv}:{chunk_type}:{chunk_index}")
    }
}

fn chunk_type_like(chunk_type: &str) -> String {
    format!("event:%:%:{chunk_type}:%")
}

pub fn clamp_score(value: f64) -> f64 {
    value.clamp(0.0, 100.0)
}

pub fn clamp_confidence(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

/// 对齐 Python `_clamp_score(value, default)`：`float(value)` 失败则用 default，再夹 0–100。
pub fn clamp_score_any(value: Option<&Value>, default: f64) -> f64 {
    clamp_score(coerce_json_float(value, default))
}

/// 对齐 Python `_clamp_confidence(value, default)`：`float(value)` 失败则用 default，再夹 0–1。
pub fn clamp_confidence_any(value: Option<&Value>, default: f64) -> f64 {
    clamp_confidence(coerce_json_float(value, default))
}

fn coerce_json_float(value: Option<&Value>, default: f64) -> f64 {
    let Some(value) = value else {
        return default;
    };
    match value {
        Value::Null => default,
        Value::Number(n) => n.as_f64().unwrap_or(default),
        Value::String(text) => text.trim().parse::<f64>().unwrap_or(default),
        Value::Bool(true) => 1.0,
        Value::Bool(false) => 0.0,
        _ => default,
    }
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
    let mut stats = MemoryCleanupStats::default();
    let mut expired_event_ids: Vec<String> = Vec::new();
    if let Err(err) =
        cleanup_memory_lifecycle(pool, settings, &mut stats, &mut expired_event_ids).await
    {
        tracing::warn!(error = %err, "长期记忆生命周期清理失败");
    }
    delete_expired_event_vectors(&expired_event_ids).await;
    Ok(stats)
}

async fn cleanup_memory_lifecycle(
    pool: &sqlx::SqlitePool,
    settings: &MemoryCleanupSettings,
    stats: &mut MemoryCleanupStats,
    expired_event_ids: &mut Vec<String>,
) -> Result<(), sqlx::Error> {
    let now = crate::db::now_naive();
    let now_text = crate::db::fmt_dt(&now);
    let rows: Vec<ActiveMemoryRow> = sqlx::query_as(
        "SELECT memory_id, memory_type, base_importance_score, confidence_score, access_count, \
         last_accessed_at, created_at, expires_at \
         FROM memory_records WHERE status = 'active'",
    )
    .fetch_all(pool)
    .await?;

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
            if row.memory_type == MEMORY_TYPE_EVENT {
                expired_event_ids.push(row.memory_id.clone());
            }
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

    let purge_before =
        crate::db::fmt_dt(&(now - chrono::Duration::days(settings.purge_retention_days.max(1))));
    let purged = sqlx::query(
        "DELETE FROM memory_records \
         WHERE status IN ('expired', 'superseded', 'deleted') AND updated_at < ?",
    )
    .bind(&purge_before)
    .execute(pool)
    .await?;
    stats.purged = purged.rows_affected() as i64;
    Ok(())
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
                    "{}",
                    format_memory_maintenance_complete(stats.rescored, stats.expired, stats.purged)
                ),
                Ok(_) => {}
                Err(err) => tracing::warn!("长期记忆维护失败: {err}"),
            }
            tokio::time::sleep(interval).await;
        }
    });
}

struct PersistJob {
    pool: sqlx::SqlitePool,
    llm: Option<Arc<LlmClient>>,
    user_id: String,
    conversation_id: String,
    session_id: String,
    query: String,
    answer: String,
    expiry_days: i64,
}

pub struct MemoryPersistQueue {
    tx: mpsc::Sender<PersistJob>,
}

impl MemoryPersistQueue {
    pub fn start(capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(capacity.max(1));
        tokio::spawn(run_persist_worker(rx));
        MemoryPersistQueue { tx }
    }

    #[cfg(test)]
    fn bounded(capacity: usize) -> (Self, mpsc::Receiver<PersistJob>) {
        let (tx, rx) = mpsc::channel(capacity.max(1));
        (MemoryPersistQueue { tx }, rx)
    }

    fn try_enqueue(&self, job: PersistJob) -> bool {
        match self.tx.try_send(job) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(job)) => {
                tracing::warn!(
                    "长期记忆写入队列已满({})，丢弃本轮记忆: query={}",
                    MEMORY_PERSIST_QUEUE_SIZE,
                    job.query.chars().take(32).collect::<String>()
                );
                false
            }
            Err(mpsc::error::TrySendError::Closed(job)) => {
                tracing::warn!(
                    "长期记忆写入队列已关闭，丢弃本轮记忆: query={}",
                    job.query.chars().take(32).collect::<String>()
                );
                false
            }
        }
    }
}

async fn run_persist_worker(mut rx: mpsc::Receiver<PersistJob>) {
    while let Some(job) = rx.recv().await {
        if let Err(err) = persist_turn_result(
            &job.pool,
            job.llm.as_deref(),
            &job.user_id,
            &job.conversation_id,
            &job.session_id,
            &job.query,
            &job.answer,
            job.expiry_days,
        )
        .await
        {
            tracing::warn!("{}", format_persist_worker_fail(&err));
        }
    }
}

fn global_persist_queue() -> &'static MemoryPersistQueue {
    static QUEUE: OnceLock<MemoryPersistQueue> = OnceLock::new();
    QUEUE.get_or_init(|| MemoryPersistQueue::start(MEMORY_PERSIST_QUEUE_SIZE))
}

/// 对齐 Python `_persist_long_term_memory_async`：单 worker + 有界队列串行写入。
pub fn enqueue_persist_turn(
    pool: sqlx::SqlitePool,
    llm: Option<Arc<LlmClient>>,
    user_id: String,
    conversation_id: String,
    session_id: String,
    query: String,
    answer: String,
    expiry_days: i64,
) -> bool {
    if answer.trim().is_empty() || user_id.trim().is_empty() {
        return false;
    }
    global_persist_queue().try_enqueue(PersistJob {
        pool,
        llm,
        user_id,
        conversation_id,
        session_id,
        query,
        answer,
        expiry_days,
    })
}

pub fn should_extract_profile(query: &str) -> bool {
    let normalized = query.replace(char::is_whitespace, "");
    !normalized.is_empty() && PROFILE_TRIGGER_TERMS.iter().any(|term| normalized.contains(term))
}

const EVENT_STRONG_MARKERS: &[&str] = &["必须", "明确", "决定", "结论", "长期", "偏好", "目标", "完成"];
const EVENT_TEMPORARY_MARKERS: &[&str] = &["今天", "明天", "刚才", "临时", "这次", "当前"];

fn marker_weighted_count(text: &str, markers: &[&str], weight: f64) -> f64 {
    markers
        .iter()
        .map(|marker| text.matches(marker).count() as f64 * weight)
        .sum()
}

/// 对齐 Python `_calculate_event_importance`：按稳定/临时标记生成初始分。
pub(crate) fn calculate_event_importance(content: &str) -> (f64, f64) {
    let normalized = crate::retrieval::collapse_spaces(content);
    let mut importance = 45.0;
    let mut confidence = 0.65;
    importance += marker_weighted_count(&normalized, EVENT_STRONG_MARKERS, 5.0).min(25.0);
    importance -= marker_weighted_count(&normalized, EVENT_TEMPORARY_MARKERS, 4.0).min(20.0);
    if normalized.chars().count() < 20 {
        confidence -= 0.1;
    }
    (clamp_score(importance), clamp_confidence(confidence))
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
    let _ = persist_turn_result(
        pool,
        llm,
        user_id,
        conversation_id,
        session_id,
        query,
        answer,
        expiry_days,
    )
    .await;
}

async fn persist_turn_result(
    pool: &sqlx::SqlitePool,
    llm: Option<&LlmClient>,
    user_id: &str,
    conversation_id: &str,
    session_id: &str,
    query: &str,
    answer: &str,
    expiry_days: i64,
) -> Result<(), String> {
    if answer.trim().is_empty() || user_id.trim().is_empty() {
        return Ok(());
    }
    let _ = ensure_schema(pool).await;
    let messages = user_assistant_messages(query, answer);
    let chunks = if messages.is_empty() {
        Vec::new()
    } else {
        build_conversation_chunks(&messages, CHAT_CHUNK_SIZE, CHAT_CHUNK_OVERLAP, false)
    };
    if chunks.is_empty() {
        tracing::info!("{}", format_empty_memory_chunks());
    } else {
        let turn_offset = count_dialogue_turns(pool, user_id, conversation_id).await;
        let mut written = 0usize;
        for chunk in &chunks {
            let (importance, confidence) = calculate_event_importance(&chunk.text);
            let local_index = chunk.chunk_index as i64;
            match insert_event_memory_with_confidence(
                pool,
                user_id,
                conversation_id,
                session_id,
                "dialogue",
                &chunk.text,
                importance,
                confidence,
                expiry_days,
                Some(chunk.turn_start + turn_offset),
                Some(chunk.turn_end + turn_offset),
                Some(local_index + turn_offset),
                Some(versioned_chunk_memory_id(user_id, conversation_id, local_index)),
            )
            .await
            {
                Ok(_) => written += 1,
                Err(err) => tracing::warn!("长期记忆写入失败，已跳过: {err}"),
            }
        }
        if written > 0 {
            tracing::info!(
                "{}",
                format_memory_write_success(user_id, conversation_id, written)
            );
        }
    }
    if !conversation_id.trim().is_empty() {
        if let Err(err) = persist_consolidate_check(
            pool,
            llm,
            user_id,
            conversation_id,
            expiry_days,
        )
        .await
        {
            tracing::warn!("{}", format_consolidate_check_fail(&err));
        }
    }
    if !should_extract_profile(query) {
        return Ok(());
    }
    let Some(llm) = llm else {
        return Ok(());
    };
    match extract_profile_candidates(llm, query, answer).await {
        Ok(candidates) => match upsert_profile_candidates(
            pool,
            user_id,
            conversation_id,
            session_id,
            &candidates,
        )
        .await
        {
            Ok(updated) if updated > 0 => {
                tracing::info!("{}", format_profile_update_success(user_id, updated));
            }
            Ok(_) => {}
            Err(err) => tracing::warn!("{}", format_profile_write_fail(&err)),
        },
        Err(err) => tracing::warn!("{}", format_profile_extract_fail(&err)),
    }
    Ok(())
}

async fn persist_consolidate_check(
    pool: &sqlx::SqlitePool,
    llm: Option<&LlmClient>,
    user_id: &str,
    conversation_id: &str,
    expiry_days: i64,
) -> Result<i64, String> {
    Ok(consolidate_check(pool, llm, user_id, conversation_id, expiry_days).await)
}

const PROFILE_PROMPT_LIMIT: usize = 20;
const PROFILE_PROMPT_INTRO: &str = "以下是当前用户已确认的个性化记忆，仅在相关时参考：";

pub async fn profile_prompt(pool: &sqlx::SqlitePool, user_id: &str) -> String {
    if user_id.trim().is_empty() {
        return String::new();
    }
    let rows = match crate::skills::load_active_profile_lines(pool, user_id, PROFILE_PROMPT_LIMIT)
        .await
    {
        Ok(rows) => rows,
        Err(err) => {
            tracing::warn!("个性化记忆读取失败，已跳过: {err}");
            return String::new();
        }
    };
    if rows.is_empty() {
        return String::new();
    }
    let mut lines = vec![PROFILE_PROMPT_INTRO.to_string()];
    lines.extend(rows);
    lines.join("\n")
}

/// 对齐 Python `_recent_conversation_memory_lines`：取当前请求历史最后 top_k 轮原文。
pub fn recent_unsaved_conversation_prompt(
    history: &[crate::short_memory::MemoryMessage],
    top_k: usize,
) -> String {
    let lines = recent_unsaved_conversation_lines(history, top_k);
    if lines.is_empty() {
        return String::new();
    }
    format!(
        "以下是当前会话最近尚未进入长期记忆的对话原文，仅用于补足刚刚发生的上下文：\n{}",
        lines.join("\n\n")
    )
}

fn recent_unsaved_conversation_lines(
    history: &[crate::short_memory::MemoryMessage],
    top_k: usize,
) -> Vec<String> {
    let (_, turns) = crate::short_memory::split_history(history);
    if turns.is_empty() {
        return Vec::new();
    }
    let keep = top_k.max(1);
    let start = turns.len().saturating_sub(keep);
    let mut lines = Vec::new();
    for (index, turn) in turns[start..].iter().enumerate() {
        let mut rendered = Vec::new();
        for message in turn {
            let role = message.role.trim();
            let content = message.content.trim();
            if content.is_empty() {
                continue;
            }
            if role == "user" {
                rendered.push(format!("用户: {content}"));
            } else if role == "assistant" {
                rendered.push(format!("助手: {content}"));
            }
        }
        if !rendered.is_empty() {
            lines.push(format!("{}. {}", index + 1, rendered.join("\n")));
        }
    }
    lines
}

/// 对齐 Python `_build_long_term_memory_messages_uncached`：
/// 无记忆线索时跳过画像、未入库原文和事件召回。
pub async fn build_long_term_memory_messages(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    query: &str,
    conversation_id: Option<&str>,
    session_id: Option<&str>,
    top_k: usize,
    embedder: Option<&Embedder>,
    route: Option<&str>,
    recent_history: &[crate::short_memory::MemoryMessage],
) -> Vec<ChatMessage> {
    build_long_term_memory_messages_inner(
        pool,
        user_id,
        query,
        conversation_id,
        session_id,
        top_k,
        embedder,
        route,
        recent_history,
        None,
    )
    .await
}

/// 对齐 Python `MemoryContext`：一次请求内复用画像读取与按 route 的最终消息，
/// 消除 skill 多轮 + 最终组消息路径上的重复画像 SQL 与事件记忆检索。
pub struct MemoryContext<'a> {
    pool: &'a sqlx::SqlitePool,
    user_id: &'a str,
    query: &'a str,
    conversation_id: Option<&'a str>,
    session_id: Option<&'a str>,
    embedder: Option<&'a Embedder>,
    profile: std::sync::Mutex<Option<String>>,
    routes:
        std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<Vec<ChatMessage>>>>,
}

impl<'a> MemoryContext<'a> {
    pub fn new(
        pool: &'a sqlx::SqlitePool,
        user_id: &'a str,
        query: &'a str,
        conversation_id: Option<&'a str>,
        session_id: Option<&'a str>,
        embedder: Option<&'a Embedder>,
    ) -> Self {
        MemoryContext {
            pool,
            user_id,
            query,
            conversation_id,
            session_id,
            embedder,
            profile: std::sync::Mutex::new(None),
            routes: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// 等价 `build_long_term_memory_messages`，同 route 二次调用直接命中缓存。
    pub async fn messages(
        &self,
        route: Option<&str>,
        top_k: usize,
        recent_history: &[crate::short_memory::MemoryMessage],
    ) -> std::sync::Arc<Vec<ChatMessage>> {
        let key = route.map(str::trim).unwrap_or("").to_string();
        if let Some(cached) = self.routes.lock().unwrap().get(&key) {
            return cached.clone();
        }
        let messages = std::sync::Arc::new(
            build_long_term_memory_messages_inner(
                self.pool,
                self.user_id,
                self.query,
                self.conversation_id,
                self.session_id,
                top_k,
                self.embedder,
                route,
                recent_history,
                Some(&self.profile),
            )
            .await,
        );
        self.routes
            .lock()
            .unwrap()
            .insert(key, messages.clone());
        messages
    }
}

async fn build_long_term_memory_messages_inner(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    query: &str,
    conversation_id: Option<&str>,
    session_id: Option<&str>,
    top_k: usize,
    embedder: Option<&Embedder>,
    route: Option<&str>,
    recent_history: &[crate::short_memory::MemoryMessage],
    profile_cache: Option<&std::sync::Mutex<Option<String>>>,
) -> Vec<ChatMessage> {
    if user_id.trim().is_empty() {
        return Vec::new();
    }
    if !should_retrieve_event_memory(query, route) {
        tracing::debug!("{}", format_memory_skip_recall());
        return Vec::new();
    }
    let top_k = if top_k == 0 {
        LONG_TERM_MEMORY_TOP_K
    } else {
        top_k
    };
    let mut messages = Vec::new();
    let profile = match profile_cache {
        Some(cache) => {
            // 先把缓存值 clone 出来再 match，避免 MutexGuard 跨 await 持有（Send 约束）。
            let cached = cache.lock().unwrap().clone();
            match cached {
                Some(value) => value,
                None => {
                    let fresh = profile_prompt(pool, user_id).await;
                    *cache.lock().unwrap() = Some(fresh.clone());
                    fresh
                }
            }
        }
        None => profile_prompt(pool, user_id).await,
    };
    if !profile.is_empty() {
        messages.push(ChatMessage::new("system", profile));
    }
    let recent = recent_unsaved_conversation_prompt(recent_history, top_k);
    if !recent.is_empty() {
        messages.push(ChatMessage::new("system", recent));
    }
    let event = hierarchical_prompt(
        pool,
        user_id,
        query,
        conversation_id,
        session_id,
        top_k,
        embedder,
        route,
    )
    .await;
    if !event.is_empty() {
        messages.push(ChatMessage::new("system", event));
    }
    messages
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
) -> Result<Vec<EventMemoryHit>, String> {
    let user_id = require_user_id(user_id)?;
    let _ = ensure_schema(pool).await;
    let conversation_id = conversation_id.map(str::trim).filter(|item| !item.is_empty());
    let session_id = session_id.map(str::trim).filter(|item| !item.is_empty());
    let query = query.trim();
    match search_event_memories_ann(
        pool,
        user_id,
        query,
        conversation_id,
        session_id,
        top_k,
    )
    .await
    {
        AnnSearch::Hits(hits) => return Ok(hits),
        AnnSearch::Failed(err) => {
            let wrapped = format_memory_search_fail(&err);
            tracing::error!("{wrapped}");
            return Err(wrapped);
        }
        AnnSearch::Unavailable => {}
    }

    let now_text = crate::db::fmt_dt(&crate::db::now_naive());
    let mut builder = sqlx::QueryBuilder::new(
        "SELECT memory_id, memory_key, content, conversation_id, session_id, \
         turn_start, turn_end, importance_score \
         FROM memory_records \
         WHERE user_id = ",
    );
    builder.push_bind(user_id);
    builder.push(" AND memory_type = 'event' AND status = 'active'");
    builder.push(" AND (expires_at IS NULL OR expires_at > ");
    builder.push_bind(&now_text);
    builder.push(")");
    if let Some(conversation_id) = conversation_id {
        builder.push(" AND conversation_id = ");
        builder.push_bind(conversation_id);
    }
    if let Some(session_id) = session_id {
        builder.push(" AND session_id = ");
        builder.push_bind(session_id);
    }
    builder.push(" ORDER BY created_at DESC LIMIT ");
    builder.push_bind(CANDIDATE_LIMIT);
    let rows: Vec<EventMemoryRow> = match builder.build_query_as().fetch_all(pool).await {
        Ok(rows) => rows,
        Err(err) => {
            tracing::warn!("长期记忆检索失败，已跳过: {err}");
            return Ok(Vec::new());
        }
    };
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let query_tokens = if query.is_empty() {
        Vec::new()
    } else {
        eks_kernels::tokenize_cjk(query)
    };
    let mut scored: Vec<EventMemoryHit> = rows
        .into_iter()
        .map(|row| {
            let memory_key = event_chunk_type(row.memory_key.as_deref());
            let lexical_score = if query.is_empty() {
                0.0
            } else {
                let row_tokens = eks_kernels::tokenize_cjk(&row.content);
                eks_kernels::lexical::lexical_match_score(&query_tokens, &row_tokens, query, &row.content)
            };
            let score = lexical_score * 0.55 + (row.importance_score / 100.0).clamp(0.0, 1.0) * 0.20;
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
    mark_accessed_async(pool, user_id, &scored).await;
    Ok(scored)
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
    let hits = match search_event_memories(
        pool,
        user_id,
        query,
        conversation_id,
        session_id,
        if top_k == 0 { LONG_TERM_MEMORY_TOP_K } else { top_k },
        embedder,
    )
    .await
    {
        Ok(hits) => hits,
        Err(err) => {
            tracing::warn!("长期记忆层级召回失败: {err}");
            Vec::new()
        }
    };
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
        "chunk_type": event_chunk_type(row.memory_key.as_deref()),
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
        builder.push(" AND (memory_key = 'dialogue' OR memory_key LIKE 'event:%:%:dialogue:%')");
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
    mark_accessed_async(pool, user_id, &hits).await;
}

pub async fn retrieve_original_conversation(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    query: OriginalConversationQuery,
) -> Result<Value, String> {
    let user_id = require_user_id(user_id)?;
    let _ = ensure_schema(pool).await;
    let query = clamp_original_query(&query);
    if query.memory_ids.is_empty() && query.query.is_empty() && query.conversation_id.is_empty() {
        return Ok(empty_original_conversation(
            "请提供摘要 memory_id、检索提示或 conversation_id。",
        ));
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
        return Ok(json!({
            "mode": "original_conversation",
            "summary": format!("已展开当前会话的 {} 个原始对话块。", bounded.len()),
            "summary_references": [],
            "original_chunks": bounded,
            "truncated": truncated,
        }));
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
        .await?;
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
            return Ok(json!({
                "mode": "original_conversation",
                "summary": format!("已按语义找到 {} 个原始对话块。", raw_chunks.len()),
                "summary_references": [],
                "original_chunks": raw_chunks,
            }));
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
    Ok(json!({
        "mode": "original_conversation",
        "summary": format!("已展开 {} 个原始对话块。", bounded.len()),
        "summary_references": summary_preview,
        "original_chunks": bounded,
        "truncated": truncated,
    }))
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

/// 记忆文本 → 向量的进程级缓存。
///
/// `blend_cosine_scores` 与 ANN 召回此前对同一批记忆文本/查询反复整段重嵌入，
/// 是记忆类请求延迟与并发排队的主要来源。记忆内容写入后不可变（更新只动
/// access/importance 字段），同一文本的向量恒定，可安全复用；键为内容哈希，
/// 命中时再做全文比对，哈希碰撞只会退化为未命中，不影响正确性。
/// 方向前缀区分查询指令（embed_queries 会前置 query_instruction）。
static MEMORY_VECTOR_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<u64, (String, std::sync::Arc<Vec<f32>>)>>,
> = std::sync::OnceLock::new();
const MEMORY_VECTOR_CACHE_CAP: usize = 8192;

fn memory_vector_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<u64, (String, std::sync::Arc<Vec<f32>>)>> {
    MEMORY_VECTOR_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn content_hash(kind: &str, text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (kind, text).hash(&mut hasher);
    hasher.finish()
}

fn cached_memory_vector(kind: &str, text: &str) -> Option<std::sync::Arc<Vec<f32>>> {
    let cache = memory_vector_cache().lock().unwrap();
    cache
        .get(&content_hash(kind, text))
        .filter(|(stored, _)| stored == text)
        .map(|(_, vector)| vector.clone())
}

fn store_memory_vector(kind: &str, text: &str, vector: Vec<f32>) {
    let mut cache = memory_vector_cache().lock().unwrap();
    // 简单容量上限：超限整体清空（一次性冷启动代价，避免 LRU 复杂度）
    if cache.len() >= MEMORY_VECTOR_CACHE_CAP {
        cache.clear();
    }
    cache.insert(
        content_hash(kind, text),
        (text.to_string(), std::sync::Arc::new(vector)),
    );
}

/// 查询向量（带查询指令）缓存版：未命中时嵌入一次并写入缓存。
fn embed_query_cached(embedder: &Embedder, query: &str) -> Option<Vec<f32>> {
    if let Some(vector) = cached_memory_vector("q", query) {
        return Some((*vector).clone());
    }
    let vectors = embedder.embed_queries(&[query.to_string()]).ok()?;
    let vector = vectors.into_iter().next()?;
    store_memory_vector("q", query, vector.clone());
    Some(vector)
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
    let query_vec = match embed_query_cached(embedder, query) {
        Some(vector) if !vector.is_empty() => vector,
        _ => {
            tracing::warn!("长期记忆 query 向量化失败");
            return;
        }
    };
    // 先取缓存命中；未命中的文本一次性批量补嵌入（每条文本进程生命周期仅此一次）
    let mut misses: Vec<(usize, usize)> = Vec::new(); // (order 内位置, 行号)
    let mut doc_vecs: Vec<Option<std::sync::Arc<Vec<f32>>>> = Vec::with_capacity(order.len());
    for (position, &index) in order.iter().enumerate() {
        if let Some(vector) = cached_memory_vector("d", &hits[index].content) {
            doc_vecs.push(Some(vector));
        } else {
            doc_vecs.push(None);
            misses.push((position, index));
        }
    }
    if !misses.is_empty() {
        let texts: Vec<String> = misses
            .iter()
            .map(|(_, index)| hits[*index].content.clone())
            .collect();
        match embedder.embed_documents(&texts) {
            Ok(vectors) => {
                for (offset, &(position, index)) in misses.iter().enumerate() {
                    if let Some(vector) = vectors.get(offset) {
                        store_memory_vector("d", &hits[index].content, vector.clone());
                        doc_vecs[position] = Some(std::sync::Arc::new(vector.clone()));
                    }
                }
            }
            Err(err) => {
                tracing::warn!("长期记忆文档向量化失败: {err}");
            }
        }
    }
    for (offset, index) in order.into_iter().enumerate() {
        if let Some(Some(vector)) = doc_vecs.get(offset) {
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

fn format_memory_search_fail(error: &str) -> String {
    format!("长期记忆检索失败: {error}")
}

#[cfg_attr(not(test), allow(dead_code))]
fn format_access_stat_async_fail(error: &str) -> String {
    format!("异步更新长期记忆访问统计失败: {error}")
}

fn format_access_stat_submit_fail(error: &str) -> String {
    format!("提交长期记忆访问统计任务失败: {error}")
}

async fn mark_accessed_async(pool: &sqlx::SqlitePool, user_id: &str, hits: &[EventMemoryHit]) {
    if hits.is_empty() {
        return;
    }
    let ids: Vec<String> = hits.iter().map(|hit| hit.memory_id.clone()).collect();
    #[cfg(test)]
    {
        mark_accessed_ids(pool, user_id, &ids).await;
    }
    #[cfg(not(test))]
    {
        let pool = pool.clone();
        let user_id = user_id.to_string();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    mark_accessed_ids(&pool, &user_id, &ids).await;
                });
            }
            Err(err) => tracing::warn!("{}", format_access_stat_submit_fail(&err.to_string())),
        }
    }
}

async fn mark_accessed_ids(pool: &sqlx::SqlitePool, user_id: &str, ids: &[String]) {
    let now = crate::db::fmt_dt(&crate::db::now_naive());
    for memory_id in ids {
        if let Err(err) = sqlx::query(
            "UPDATE memory_records SET access_count = access_count + 1, last_accessed_at = ?, updated_at = ? \
             WHERE user_id = ? AND memory_id = ? AND status = 'active'",
        )
        .bind(&now)
        .bind(&now)
        .bind(user_id)
        .bind(memory_id)
        .execute(pool)
        .await
        {
            tracing::warn!("更新长期记忆访问统计失败: {err}");
        }
    }
}

fn format_memory_delete_success(user_id: &str, conversation_id: &str) -> String {
    format!("✓ 长期记忆删除成功 (user_id={user_id}, conversation_id={conversation_id})")
}

fn format_memory_write_success(user_id: &str, conversation_id: &str, chunks: usize) -> String {
    format!(
        "✓ 长期记忆写入成功 (user_id={user_id}, conversation_id={conversation_id}, chunks={chunks})"
    )
}

fn format_summary_write_success(
    level: i32,
    user_id: &str,
    conversation_id: &str,
    turn_start: i64,
    turn_end: i64,
) -> String {
    format!(
        "✓ 递进摘要 L{level} 写入成功 (user_id={user_id}, conversation_id={conversation_id}, turns={turn_start}-{turn_end})"
    )
}

fn format_profile_update_success(user_id: &str, updated: usize) -> String {
    format!("个性化记忆更新完成 (user_id={user_id}, updated={updated})")
}

fn format_memory_maintenance_complete(rescored: i64, expired: i64, purged: i64) -> String {
    format!("长期记忆维护完成: rescored={rescored} expired={expired} purged={purged}")
}

fn format_empty_memory_chunks() -> &'static str {
    "没有可写入的长期记忆块。"
}

fn format_profile_extract_fail(err: &str) -> String {
    format!("个性化记忆抽取失败，已跳过: {err}")
}

fn format_profile_write_fail(err: &str) -> String {
    format!("个性化记忆写入失败，已跳过: {err}")
}

fn format_memory_skip_recall() -> &'static str {
    "当前问题未命中长期记忆关联信号，跳过记忆召回"
}

fn format_consolidate_check_fail(err: &str) -> String {
    format!("递进摘要合并检查失败，下次触发点重试: {err}")
}

fn format_persist_worker_fail(err: &str) -> String {
    format!("后台长期记忆写入失败: {err}")
}

/// 对齐 Python `delete_long_term_memory`：collection 可写时删除成功即 True，不看行数。
pub async fn delete_conversation_memory(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    conversation_id: &str,
) -> Result<bool, sqlx::Error> {
    let cid = conversation_id.trim();
    if cid.is_empty() || user_id.trim().is_empty() {
        return Ok(false);
    }
    let _ = ensure_schema(pool).await;
    if !delete_conversation_memory_vectors(user_id, cid).await {
        return Ok(false);
    }
    let now = crate::db::fmt_dt(&crate::db::now_naive());
    sqlx::query(
        "UPDATE memory_records SET status = 'deleted', updated_at = ? \
         WHERE user_id = ? AND conversation_id = ? AND memory_type = 'event' AND status = 'active'",
    )
    .bind(&now)
    .bind(user_id)
    .bind(cid)
    .execute(pool)
    .await?;
    tracing::info!("{}", format_memory_delete_success(user_id, conversation_id));
    Ok(true)
}

pub async fn delete_user_event_memory(
    pool: &sqlx::SqlitePool,
    user_id: &str,
) -> Result<u64, sqlx::Error> {
    let _ = ensure_schema(pool).await;
    let ids: Vec<String> = sqlx::query_scalar(
        "SELECT memory_id FROM memory_records \
         WHERE user_id = ? AND memory_type = 'event' AND status = 'active'",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    delete_memory_vectors(&ids).await;
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

async fn delete_memory_vectors(memory_ids: &[String]) {
    delete_memory_vectors_logged(memory_ids, "长期记忆向量删除失败").await;
}

async fn delete_expired_event_vectors(memory_ids: &[String]) {
    delete_memory_vectors_logged(memory_ids, "清理过期长期记忆向量失败").await;
}

async fn delete_memory_vectors_logged(memory_ids: &[String], fail_log: &str) {
    let Some(sink) = memory_vector_sink() else {
        return;
    };
    if memory_ids.is_empty() {
        return;
    }
    if let Err(err) = sink
        .store
        .delete_by_field_values(&sink.collection, "memory_id", memory_ids)
        .await
    {
        tracing::warn!("{fail_log}: {err}");
    }
}

/// 对齐 Python `client.delete(filter=user_id == ".." and conversation_id == "..")`。
/// 未配置向量库或 collection 尚未物化时仍允许 SQL 软删；向量删除失败则中止。
async fn delete_conversation_memory_vectors(user_id: &str, conversation_id: &str) -> bool {
    let Some(sink) = memory_vector_sink() else {
        return true;
    };
    if !sink.store.has_collection(&sink.collection).await {
        return true;
    }
    match sink
        .store
        .delete_by_matching_fields(
            &sink.collection,
            &[("user_id", user_id), ("conversation_id", conversation_id)],
        )
        .await
    {
        Ok(_) => true,
        Err(err) => {
            tracing::error!("长期记忆删除失败: {err}");
            false
        }
    }
}

enum MemorySummarizer<'a> {
    Llm(&'a LlmClient),
    #[cfg(test)]
    Fixed { topic: String, summary: String },
}

impl MemorySummarizer<'_> {
    async fn summarize(&self, system: &str, user: &str) -> Option<(String, String)> {
        match self {
            MemorySummarizer::Llm(llm) => {
                let text = llm
                    .invoke(&[
                        ChatMessage::new("system", system),
                        ChatMessage::new("user", user),
                    ])
                    .await
                    .ok()?
                    .content;
                let payload = parse_json_object(&text);
                let topic = payload
                    .get("topic")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let mut summary = payload
                    .get("summary")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if summary.is_empty() {
                    summary = text.trim().to_string();
                }
                if summary.is_empty() {
                    None
                } else {
                    Some((topic, summary))
                }
            }
            #[cfg(test)]
            MemorySummarizer::Fixed { topic, summary } => {
                let _ = (system, user);
                Some((topic.clone(), summary.clone()))
            }
        }
    }

    #[allow(dead_code)]
    async fn topic_changed(&self, current_text: &str, next_text: &str) -> bool {
        if high_confidence_topic_continuation(current_text, next_text) {
            return false;
        }
        match self {
            MemorySummarizer::Llm(llm) => {
                let text = llm
                    .invoke(&[
                        ChatMessage::new("system", TOPIC_CHANGE_SYSTEM_PROMPT),
                        ChatMessage::new(
                            "user",
                            &format!(
                                "当前缓冲区对话：\n{}\n\n候选新增对话：\n{}\n\n如果已经明显换题，请返回 changed=true；否则返回 changed=false。",
                                current_text.trim(),
                                next_text.trim()
                            ),
                        ),
                    ])
                    .await
                    .ok()
                    .map(|message| message.content)
                    .unwrap_or_default();
                let payload = parse_json_object(&text);
                if payload.get("changed").is_some() {
                    return payload
                        .get("changed")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                }
                let lowered = text.trim().to_lowercase();
                ["true", "yes", "changed", "是", "切换", "换题"]
                    .iter()
                    .any(|token| lowered.contains(token))
            }
            #[cfg(test)]
            MemorySummarizer::Fixed { .. } => true,
        }
    }
}

fn consolidation_in_flight() -> &'static Mutex<HashSet<(String, String)>> {
    static LOCK: std::sync::OnceLock<Mutex<HashSet<(String, String)>>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(HashSet::new()))
}

fn consolidation_retry_state() -> &'static Mutex<HashMap<(String, String, i32), (Instant, u32)>> {
    static LOCK: std::sync::OnceLock<Mutex<HashMap<(String, String, i32), (Instant, u32)>>> =
        std::sync::OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(HashMap::new()))
}

fn normalize_dialogue_token(text: &str) -> String {
    text.chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>()
        .to_lowercase()
        .chars()
        .filter(|ch| {
            !matches!(
                *ch,
                '，' | '。' | '！' | '？' | '、' | ',' | '.' | '!' | '?' | '：' | ':' | '；' | ';'
                    | '‘' | '’' | '“' | '”' | '"' | '\'' | '（' | '）' | '(' | ')' | '[' | ']'
                    | '{' | '}'
            )
        })
        .collect()
}

pub fn is_substantive_dialogue(messages: &[(String, String)]) -> bool {
    messages.iter().any(|(role, content)| {
        if role != "user" {
            return false;
        }
        let normalized = normalize_dialogue_token(content);
        !normalized.is_empty() && !LOW_INFORMATION_DIALOGUE.contains(&normalized.as_str())
    })
}

fn format_memory_chunk(topic: &str, summary: &str, transcript: &str) -> String {
    let mut parts = Vec::new();
    if !topic.trim().is_empty() {
        parts.push(format!("【主题】{}", topic.trim()));
    }
    parts.push(format!("【摘要】{}", summary.trim()));
    parts.push("【原文】".into());
    parts.push(transcript.trim().to_string());
    parts.join("\n")
}

fn parse_dialogue_messages(content: &str) -> Vec<(String, String)> {
    let mut messages = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("用户:") {
            messages.push(("user".into(), rest.trim().to_string()));
        } else if let Some(rest) = line.strip_prefix("助手:") {
            messages.push(("assistant".into(), rest.trim().to_string()));
        }
    }
    messages
}

pub async fn count_dialogue_turns(pool: &sqlx::SqlitePool, user_id: &str, conversation_id: &str) -> i64 {
    if user_id.trim().is_empty() || conversation_id.trim().is_empty() {
        return 0;
    }
    let rows: Vec<(Option<i64>, Option<i64>)> = sqlx::query_as(
        "SELECT turn_start, turn_end FROM memory_records \
         WHERE user_id = ? AND conversation_id = ? AND memory_type = 'event' \
         AND (memory_key = 'dialogue' OR memory_key LIKE 'event:%:%:dialogue:%') \
         AND status = 'active'",
    )
    .bind(user_id)
    .bind(conversation_id)
    .fetch_all(pool)
    .await
    .unwrap_or_default();
    if rows.is_empty() {
        return 0;
    }
    let known_end: Vec<i64> = rows.iter().filter_map(|(_, turn_end)| *turn_end).collect();
    let unknown_count = (rows.len() as i64) - (known_end.len() as i64);
    let known = known_end.into_iter().max().map(|value| value + 1).unwrap_or(0);
    known + unknown_count
}

async fn get_consolidation_cursor(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    conversation_id: &str,
    level: i32,
) -> i64 {
    sqlx::query_scalar(
        "SELECT last_merged_turn FROM memory_consolidation \
         WHERE user_id = ? AND conversation_id = ? AND level = ?",
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(level)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()
    .unwrap_or(0)
}

async fn update_consolidation_cursor(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    conversation_id: &str,
    level: i32,
    new_turn: i64,
) -> Result<(), sqlx::Error> {
    let now = crate::db::fmt_dt(&crate::db::now_naive());
    sqlx::query(
        "INSERT INTO memory_consolidation (user_id, conversation_id, level, last_merged_turn, last_merged_at) \
         VALUES (?, ?, ?, ?, ?) \
         ON CONFLICT(user_id, conversation_id, level) DO UPDATE SET \
         last_merged_turn = MAX(last_merged_turn, excluded.last_merged_turn), \
         last_merged_at = excluded.last_merged_at",
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(level)
    .bind(new_turn.max(0))
    .bind(&now)
    .execute(pool)
    .await?;
    Ok(())
}

async fn load_dialogue_range(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    conversation_id: &str,
    start_turn: i64,
    end_turn: i64,
) -> Vec<(String, String)> {
    let rows: Vec<(String, Option<i64>, Option<i64>)> = sqlx::query_as(
        "SELECT content, turn_start, turn_end FROM memory_records \
         WHERE user_id = ? AND conversation_id = ? AND memory_type = 'event' \
         AND (memory_key = 'dialogue' OR memory_key LIKE 'event:%:%:dialogue:%') \
         AND status = 'active' \
         AND (turn_end IS NULL OR turn_end >= ?) \
         AND (turn_start IS NULL OR turn_start < ?) \
         ORDER BY turn_start ASC, created_at ASC",
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(start_turn)
    .bind(end_turn)
    .fetch_all(pool)
    .await
    .unwrap_or_default();
    rows.into_iter()
        .flat_map(|(content, _, _)| parse_dialogue_messages(&content))
        .collect()
}

async fn load_summary_chunks(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    conversation_id: &str,
    level: i32,
    start_turn: i64,
    end_turn: i64,
) -> Vec<(String, String)> {
    let memory_key = format!("semantic_memory_l{level}");
    let like_pattern = chunk_type_like(&memory_key);
    let rows: Vec<String> = sqlx::query_scalar(
        "SELECT content FROM memory_records \
         WHERE user_id = ? AND conversation_id = ? AND memory_type = 'event' \
         AND (memory_key = ? OR memory_key LIKE ?) AND status = 'active' \
         AND (turn_end IS NULL OR turn_end >= ?) \
         AND (turn_start IS NULL OR turn_start < ?) \
         ORDER BY turn_start ASC, created_at ASC",
    )
    .bind(user_id)
    .bind(conversation_id)
    .bind(&memory_key)
    .bind(&like_pattern)
    .bind(start_turn)
    .bind(end_turn)
    .fetch_all(pool)
    .await
    .unwrap_or_default();
    rows.into_iter()
        .filter_map(|content| {
            let mut topic = String::new();
            let mut summary = String::new();
            for line in content.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("【主题】") {
                    topic = rest.trim().to_string();
                } else if let Some(rest) = line.strip_prefix("【摘要】") {
                    summary = rest.trim().to_string();
                }
            }
            if summary.is_empty() {
                None
            } else {
                Some((topic, summary))
            }
        })
        .collect()
}

async fn write_semantic_summary(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    conversation_id: &str,
    level: i32,
    topic: &str,
    summary: &str,
    transcript: &str,
    turn_start: i64,
    turn_end: i64,
    expiry_days: i64,
    chunk_index: i64,
    memory_id: String,
) -> Result<String, sqlx::Error> {
    let content = format_memory_chunk(topic, summary, transcript);
    let (importance, confidence) = calculate_event_importance(&content);
    let inserted = insert_event_memory_with_confidence(
        pool,
        user_id,
        conversation_id,
        conversation_id,
        &format!("semantic_memory_l{level}"),
        &content,
        importance,
        confidence,
        expiry_days,
        Some(turn_start),
        Some(turn_end),
        Some(chunk_index),
        Some(memory_id),
    )
    .await?;
    tracing::info!(
        "{}",
        format_summary_write_success(level, user_id, conversation_id, turn_start, turn_end)
    );
    Ok(inserted)
}

fn conversation_key(conversation_id: &str) -> &str {
    if conversation_id.trim().is_empty() {
        "unknown"
    } else {
        conversation_id.trim()
    }
}

/// 对齐 Python `_build_insert_rows`：`{user}:{conv}:{chunk.chunk_index}:{uuid}`。
fn versioned_chunk_memory_id(user_id: &str, conversation_id: &str, local_index: i64) -> String {
    format!(
        "{}:{}:{}:{}",
        user_id,
        conversation_key(conversation_id),
        local_index,
        uuid::Uuid::new_v4().simple()
    )
}

/// 对齐 Python `_write_semantic_summary_block`：`{user}:{conv}:L{level}:{start}:{end}:{uuid}`。
fn semantic_level_memory_id(
    user_id: &str,
    conversation_id: &str,
    level: i32,
    turn_start: i64,
    turn_end: i64,
) -> String {
    format!(
        "{}:{}:L{level}:{turn_start}:{turn_end}:{}",
        user_id,
        conversation_key(conversation_id),
        uuid::Uuid::new_v4().simple()
    )
}

async fn consolidate_level(
    pool: &sqlx::SqlitePool,
    summarizer: Option<&MemorySummarizer<'_>>,
    user_id: &str,
    conversation_id: &str,
    level: i32,
    threshold: i64,
    expiry_days: i64,
) -> i64 {
    let cursor = get_consolidation_cursor(pool, user_id, conversation_id, level).await;
    let segment_start = (cursor / threshold) * threshold;
    let inserted = if level == 1 {
        let messages = load_dialogue_range(
            pool,
            user_id,
            conversation_id,
            segment_start,
            segment_start + threshold,
        )
        .await;
        if messages.is_empty() {
            return 0;
        }
        if !is_substantive_dialogue(&messages) {
            tracing::info!(
                user_id,
                conversation_id,
                start = segment_start,
                end = segment_start + threshold - 1,
                "L1 摘要跳过低信息对话"
            );
            let _ = update_consolidation_cursor(
                pool,
                user_id,
                conversation_id,
                level,
                segment_start + threshold,
            )
            .await;
            return 0;
        }
        let Some(summarizer) = summarizer else {
            return 0;
        };
        let conv_messages: Vec<ConversationMessage> = messages
            .iter()
            .filter_map(|(role, content)| {
                normalize_conversation_message(role, content, false)
            })
            .collect();
        // L1 按固定轮次窗口切块，不再对每个相邻轮次调用话题判断 LLM。
        let skeletons = build_semantic_memory_chunks(
            &conv_messages,
            |_current, _next| false,
            |_| (String::new(), String::new()),
            LONG_TERM_MEMORY_TOKEN_LIMIT,
            false,
        );
        if skeletons.is_empty() {
            return 0;
        }
        let mut inserted = 0i64;
        for chunk in skeletons {
            let (topic, summary) = match summarizer
                .summarize(
                    L1_SUMMARY_SYSTEM_PROMPT,
                    &format!("请基于下面的对话生成长期记忆摘要：\n\n{}", chunk.transcript),
                )
                .await
            {
                Some(pair) => pair,
                None => {
                    let fallback: String = chunk.transcript.chars().take(120).collect();
                    (fallback.clone(), fallback)
                }
            };
            match write_semantic_summary(
                pool,
                user_id,
                conversation_id,
                1,
                &topic,
                &summary,
                &chunk.transcript,
                segment_start + chunk.turn_start,
                segment_start + chunk.turn_end,
                expiry_days,
                chunk.chunk_index as i64 + segment_start,
                versioned_chunk_memory_id(user_id, conversation_id, chunk.chunk_index as i64),
            )
            .await
            {
                Ok(_) => inserted += 1,
                Err(err) => {
                    tracing::warn!("L1 合并失败，下次触发点重试: {err}");
                    return 0;
                }
            }
        }
        inserted
    } else {
        let lower_chunks = load_summary_chunks(
            pool,
            user_id,
            conversation_id,
            level - 1,
            segment_start,
            segment_start + threshold,
        )
        .await;
        if lower_chunks.is_empty() {
            return 0;
        }
        let Some(summarizer) = summarizer else {
            return 0;
        };
        let texts: Vec<String> = lower_chunks
            .iter()
            .map(|(topic, summary)| {
                if topic.is_empty() {
                    summary.clone()
                } else {
                    format!("[{topic}] {summary}")
                }
            })
            .collect();
        let joined = texts.join("\n\n");
        let (topic, summary) = match summarizer.summarize(MERGE_SUMMARY_SYSTEM_PROMPT, &joined).await {
            Some(pair) => pair,
            None => (String::from("对话摘要"), texts.first().cloned().unwrap_or_default().chars().take(120).collect()),
        };
        if summary.trim().is_empty() {
            return 0;
        }
        let dialogue_turns = count_dialogue_turns(pool, user_id, conversation_id).await;
        match write_semantic_summary(
            pool,
            user_id,
            conversation_id,
            level,
            if topic.is_empty() { "对话摘要" } else { &topic },
            &summary,
            &joined,
            segment_start,
            segment_start + threshold,
            expiry_days,
            dialogue_turns + segment_start,
            semantic_level_memory_id(
                user_id,
                conversation_id,
                level,
                segment_start,
                segment_start + threshold,
            ),
        )
        .await
        {
            Ok(_) => 1,
            Err(err) => {
                tracing::error!("递进摘要 L{level} 写入失败: {err}");
                0
            }
        }
    };
    if inserted <= 0 {
        return 0;
    }
    let _ = update_consolidation_cursor(
        pool,
        user_id,
        conversation_id,
        level,
        segment_start + threshold,
    )
    .await;
    inserted
}

async fn consolidate_with(
    pool: &sqlx::SqlitePool,
    summarizer: Option<&MemorySummarizer<'_>>,
    user_id: &str,
    conversation_id: &str,
    expiry_days: i64,
    force: bool,
) -> i64 {
    let user_id = user_id.trim();
    let conversation_id = conversation_id.trim();
    if user_id.is_empty() || conversation_id.is_empty() {
        return 0;
    }
    let _ = ensure_schema(pool).await;
    let lock_key = (user_id.to_string(), conversation_id.to_string());
    {
        let mut in_flight = consolidation_in_flight().lock().unwrap_or_else(|err| err.into_inner());
        if !in_flight.insert(lock_key.clone()) {
            return 0;
        }
    }
    let result = async {
        let current_turn = count_dialogue_turns(pool, user_id, conversation_id).await;
        if current_turn <= 0 {
            return 0;
        }
        let mut merged = 0i64;
        let now = Instant::now();
        for (threshold, level) in CONSOLIDATION_LEVELS {
            if !force && current_turn < threshold {
                continue;
            }
            let cursor = get_consolidation_cursor(pool, user_id, conversation_id, level).await;
            if !force && current_turn / threshold <= cursor / threshold {
                continue;
            }
            let retry_key = (user_id.to_string(), conversation_id.to_string(), level);
            if !force {
                let retry = consolidation_retry_state()
                    .lock()
                    .unwrap_or_else(|err| err.into_inner())
                    .get(&retry_key)
                    .cloned();
                if let Some((until, _)) = retry {
                    if now < until {
                        continue;
                    }
                }
            }
            let inserted = consolidate_level(
                pool,
                summarizer,
                user_id,
                conversation_id,
                level,
                threshold,
                expiry_days,
            )
            .await;
            let updated_cursor = get_consolidation_cursor(pool, user_id, conversation_id, level).await;
            if inserted > 0 || updated_cursor > cursor {
                merged += inserted;
                consolidation_retry_state()
                    .lock()
                    .unwrap_or_else(|err| err.into_inner())
                    .remove(&retry_key);
            } else if !force {
                let mut retry = consolidation_retry_state()
                    .lock()
                    .unwrap_or_else(|err| err.into_inner());
                let previous = retry.get(&retry_key).map(|item| item.1).unwrap_or(0);
                let delay_index = previous.min((CONSOLIDATION_RETRY_DELAYS.len() - 1) as u32);
                retry.insert(
                    retry_key,
                    (
                        Instant::now() + Duration::from_secs(CONSOLIDATION_RETRY_DELAYS[delay_index as usize]),
                        previous + 1,
                    ),
                );
            }
        }
        merged
    }
    .await;
    consolidation_in_flight()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .remove(&lock_key);
    result
}

pub async fn consolidate_check(
    pool: &sqlx::SqlitePool,
    llm: Option<&LlmClient>,
    user_id: &str,
    conversation_id: &str,
    expiry_days: i64,
) -> i64 {
    let summarizer = llm.map(MemorySummarizer::Llm);
    consolidate_with(pool, summarizer.as_ref(), user_id, conversation_id, expiry_days, false).await
}

pub async fn consolidate_conversation_memory(
    pool: &sqlx::SqlitePool,
    llm: Option<&LlmClient>,
    user_id: &str,
    conversation_id: &str,
    expiry_days: i64,
) -> Result<i64, String> {
    let summarizer = llm.map(MemorySummarizer::Llm);
    Ok(consolidate_with(pool, summarizer.as_ref(), user_id, conversation_id, expiry_days, true).await)
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
    insert_event_memory_with_confidence(
        pool,
        user_id,
        conversation_id,
        session_id,
        memory_key,
        content,
        importance,
        0.7,
        expiry_days,
        turn_start,
        turn_end,
        None,
        None,
    )
    .await
}

async fn insert_event_memory_with_confidence(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    conversation_id: &str,
    session_id: &str,
    memory_key: &str,
    content: &str,
    importance: f64,
    confidence: f64,
    expiry_days: i64,
    turn_start: Option<i64>,
    turn_end: Option<i64>,
    chunk_index: Option<i64>,
    memory_id: Option<String>,
) -> Result<String, sqlx::Error> {
    let now = crate::db::now_naive();
    let now_text = crate::db::fmt_dt(&now);
    let expires_at = crate::db::fmt_dt(&(now + chrono::Duration::days(expiry_days.max(1))));
    let chunk_type = if memory_key.trim().is_empty() {
        "dialogue"
    } else {
        memory_key.trim()
    };
    let chunk_index = chunk_index.unwrap_or_else(|| turn_start.unwrap_or(0));
    // 对齐 Python：conversation_key = conversation_id or session_id or uuid，
    // SQL/向量侧 conversation_id 均写该兜底键，保证按 session 仍可召回/删除。
    let conv_key = if !conversation_id.trim().is_empty() {
        conversation_id.trim().to_string()
    } else if !session_id.trim().is_empty() {
        session_id.trim().to_string()
    } else {
        uuid::Uuid::new_v4().simple().to_string()
    };
    let memory_id = memory_id.unwrap_or_else(|| {
        format!(
            "{}:{}:{}:{}",
            user_id,
            conv_key,
            chunk_index,
            uuid::Uuid::new_v4().simple()
        )
    });
    let stored_key = event_logical_key(user_id, &conv_key, chunk_type, &memory_id, chunk_index);
    let current: Option<(String, i64)> = sqlx::query_as(
        "SELECT memory_id, version FROM memory_records \
         WHERE user_id = ? AND memory_type = 'event' AND memory_key = ? AND status = 'active' \
         ORDER BY version DESC, updated_at DESC LIMIT 1",
    )
    .bind(user_id)
    .bind(&stored_key)
    .fetch_optional(pool)
    .await?;
    let mut version = 1i64;
    let mut supersedes_memory_id: Option<String> = None;
    if let Some((old_id, old_version)) = current {
        sqlx::query(
            "UPDATE memory_records SET status = 'superseded', updated_at = ? WHERE memory_id = ?",
        )
        .bind(&now_text)
        .bind(&old_id)
        .execute(pool)
        .await?;
        version = old_version + 1;
        supersedes_memory_id = Some(old_id);
    }
    write_memory_vector(
        &memory_id,
        chunk_index,
        content,
        user_id,
        &conv_key,
        session_id,
        chunk_type,
        turn_start.unwrap_or(0),
        turn_end.unwrap_or(turn_start.unwrap_or(0)),
        &crate::db::utc_isoformat(now.and_utc()),
    )
    .await;
    sqlx::query(
        "INSERT INTO memory_records (
            memory_id, user_id, memory_type, memory_key, content, conversation_id, session_id,
            turn_start, turn_end, importance_score, base_importance_score, confidence_score,
            access_count, status, version, supersedes_memory_id, created_at, updated_at, expires_at
        ) VALUES (?, ?, 'event', ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'active', ?, ?, ?, ?, ?)",
    )
    .bind(&memory_id)
    .bind(user_id)
    .bind(&stored_key)
    .bind(content)
    .bind(&conv_key)
    .bind(empty_to_none(session_id))
    .bind(turn_start)
    .bind(turn_end)
    .bind(clamp_score(importance))
    .bind(clamp_score(importance))
    .bind(clamp_confidence(confidence))
    .bind(version)
    .bind(supersedes_memory_id.as_deref())
    .bind(&now_text)
    .bind(&now_text)
    .bind(&expires_at)
    .execute(pool)
    .await?;
    if let Some(retired) = supersedes_memory_id {
        delete_memory_vectors(&[retired]).await;
    }
    Ok(memory_id)
}

async fn write_memory_vector(
    memory_id: &str,
    chunk_index: i64,
    content: &str,
    user_id: &str,
    conversation_id: &str,
    session_id: &str,
    chunk_type: &str,
    turn_start: i64,
    turn_end: i64,
    created_at: &str,
) {
    let Some(sink) = memory_vector_sink() else {
        return;
    };
    let vectors = match sink.embedder.embed_documents(&[content.to_string()]) {
        Ok(vectors) => vectors,
        Err(err) => {
            tracing::warn!("长期记忆向量化失败，仅写入 SQLite: {err}");
            return;
        }
    };
    let Some(vector) = vectors.into_iter().next() else {
        return;
    };
    let row = crate::milvus_lite::MemoryChunkRow {
        memory_id: memory_id.to_string(),
        chunk_index,
        source_name: "conversation".into(),
        chunk_text: content.to_string(),
        user_id: user_id.to_string(),
        conversation_id: conversation_id.trim().to_string(),
        session_id: session_id.trim().to_string(),
        chunk_type: chunk_type.to_string(),
        topic: String::new(),
        turn_start,
        turn_end,
        created_at: created_at.to_string(),
        vector,
    };
    if let Err(err) = crate::milvus_lite::append_memory_chunks(
        &sink.store.root,
        &sink.collection,
        &[row],
        sink.embedding_dimension,
    ) {
        tracing::warn!("长期记忆向量写入失败: {err}");
        return;
    }
    // 不硬失效快照：manifest mtime 变化会让下一次 collection() 走增量重载，
    // 保留旧快照可避免每轮持久化后全量重读 memory_chunks 的所有 parquet。
}

enum AnnSearch {
    Hits(Vec<EventMemoryHit>),
    Unavailable,
    Failed(String),
}

async fn search_event_memories_ann(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    query: &str,
    conversation_id: Option<&str>,
    session_id: Option<&str>,
    top_k: usize,
) -> AnnSearch {
    let Some(sink) = memory_vector_sink() else {
        return AnnSearch::Unavailable;
    };
    if query.is_empty() || !sink.store.has_collection(&sink.collection).await {
        return AnnSearch::Unavailable;
    }
    if let Err(err) = crate::milvus_lite::ensure_collection_dimension(
        &sink.store.root,
        &sink.collection,
        sink.embedding_dimension,
    ) {
        return AnnSearch::Failed(err);
    }
    let Some(vector) = embed_query_cached(&sink.embedder, query) else {
        return AnnSearch::Unavailable;
    };
    if let Err(err) = crate::milvus_lite::validate_vector_dimension(
        &sink.collection,
        vector.len(),
        sink.embedding_dimension,
    ) {
        return AnnSearch::Failed(err);
    }
    let conv = conversation_id;
    let sess = session_id;
    let hits = match sink
        .store
        .search_cosine_filtered(
            &sink.collection,
            &vector,
            (top_k.max(1) * 3).max(6),
            |row| {
                let row_user = row
                    .fields
                    .get("user_id")
                    .and_then(crate::milvus_lite::FieldValue::as_str)
                    .unwrap_or("");
                if row_user != user_id {
                    return false;
                }
                if let Some(conversation_id) = conv {
                    let row_conv = row
                        .fields
                        .get("conversation_id")
                        .and_then(crate::milvus_lite::FieldValue::as_str)
                        .unwrap_or("");
                    if row_conv != conversation_id {
                        return false;
                    }
                }
                if let Some(session_id) = sess {
                    let row_sess = row
                        .fields
                        .get("session_id")
                        .and_then(crate::milvus_lite::FieldValue::as_str)
                        .unwrap_or("");
                    if row_sess != session_id {
                        return false;
                    }
                }
                true
            },
        )
        .await
    {
        Ok(hits) => hits,
        Err(_) => return AnnSearch::Unavailable,
    };
    let snapshot = match sink.store.collection(&sink.collection).await {
        Ok(snapshot) => snapshot,
        Err(_) => return AnnSearch::Unavailable,
    };
    let now_text = crate::db::fmt_dt(&crate::db::now_naive());
    let mut scored = Vec::new();
    for (index, score) in hits {
        let Some(row) = snapshot.rows.get(index) else {
            continue;
        };
        let memory_id = row
            .fields
            .get("memory_id")
            .and_then(crate::milvus_lite::FieldValue::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if memory_id.is_empty() {
            continue;
        }
        let record: Option<EventMemoryRow> = match sqlx::query_as(
            "SELECT memory_id, memory_key, content, conversation_id, session_id, \
             turn_start, turn_end, importance_score \
             FROM memory_records \
             WHERE user_id = ? AND memory_id = ? AND memory_type = 'event' AND status = 'active' \
             AND (expires_at IS NULL OR expires_at > ?)",
        )
        .bind(user_id)
        .bind(&memory_id)
        .bind(&now_text)
        .fetch_optional(pool)
        .await
        {
            Ok(record) => record,
            Err(err) => {
                tracing::warn!("长期记忆生命周期读取失败: {err}");
                continue;
            }
        };
        let Some(record) = record else {
            continue;
        };
        let content = row
            .fields
            .get("chunk_text")
            .or_else(|| row.fields.get("text"))
            .and_then(crate::milvus_lite::FieldValue::as_str)
            .unwrap_or(&record.content)
            .to_string();
        scored.push(EventMemoryHit {
            memory_id: record.memory_id,
            content,
            memory_key: event_chunk_type(record.memory_key.as_deref()),
            conversation_id: record.conversation_id,
            session_id: record.session_id,
            turn_start: record.turn_start,
            turn_end: record.turn_end,
            score,
            lexical_score: 0.0,
            importance_score: record.importance_score,
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
    scored.truncate(top_k.max(1));
    mark_accessed_async(pool, user_id, &scored).await;
    AnnSearch::Hits(scored)
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
        let importance = clamp_score_any(candidate.get("importance"), 75.0);
        let confidence = clamp_confidence_any(candidate.get("confidence"), 0.7);
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

    #[tokio::test]
    async fn memory_context_caches_messages_by_route() {
        // 空 user_id 走短路路径不触库；同 route 二次调用必须返回同一份 Arc（缓存命中）。
        let pool = sqlx::SqlitePool::connect_lazy("sqlite::memory:").unwrap();
        let ctx = MemoryContext::new(&pool, "", "查一下差旅标准", None, None, None);
        let history = vec![crate::short_memory::MemoryMessage {
            role: "user".into(),
            content: "差旅标准是什么".into(),
        }];
        let first = ctx.messages(Some("skill"), 0, &history).await;
        let second = ctx.messages(Some("skill"), 0, &history).await;
        assert!(std::sync::Arc::ptr_eq(&first, &second));
        assert!(first.is_empty());
    }

    #[test]
    fn clamp_score_any_parses_strings_like_python() {
        assert_eq!(clamp_score_any(Some(&json!("80")), 75.0), 80.0);
        assert_eq!(clamp_score_any(Some(&json!(" 90.5 ")), 75.0), 90.5);
        assert_eq!(clamp_score_any(Some(&json!(80)), 75.0), 80.0);
        assert_eq!(clamp_score_any(Some(&json!(null)), 75.0), 75.0);
        assert_eq!(clamp_score_any(None, 75.0), 75.0);
        assert_eq!(clamp_score_any(Some(&json!("bad")), 75.0), 75.0);
        assert_eq!(clamp_score_any(Some(&json!(-10)), 75.0), 0.0);
        assert_eq!(clamp_score_any(Some(&json!(150)), 75.0), 100.0);
        assert_eq!(clamp_confidence_any(Some(&json!("0.9")), 0.7), 0.9);
        assert_eq!(clamp_confidence_any(Some(&json!("nope")), 0.7), 0.7);
        assert_eq!(clamp_confidence_any(Some(&json!(true)), 0.7), 1.0);
        assert_eq!(clamp_confidence_any(Some(&json!(false)), 0.7), 0.0);
    }

    #[test]
    fn require_user_id_matches_python() {
        assert_eq!(require_user_id("u1").unwrap(), "u1");
        assert_eq!(require_user_id("  u1  ").unwrap(), "u1");
        assert_eq!(
            require_user_id("").unwrap_err(),
            "user_id 不能为空，长期记忆必须绑定到用户"
        );
        assert_eq!(
            require_user_id("   ").unwrap_err(),
            "user_id 不能为空，长期记忆必须绑定到用户"
        );
    }

    #[tokio::test]
    async fn empty_user_id_hard_fails_search_and_original() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        let err = search_event_memories(&pool, "", "报销时限", None, None, 3, None)
            .await
            .unwrap_err();
        assert_eq!(err, "user_id 不能为空，长期记忆必须绑定到用户");
        let original_err = retrieve_original_conversation(
            &pool,
            "  ",
            OriginalConversationQuery {
                memory_ids: vec!["m1".into()],
                top_k: 8,
                max_chars: 12000,
                ..OriginalConversationQuery::default()
            },
        )
        .await
        .unwrap_err();
        assert_eq!(
            original_err,
            "user_id 不能为空，长期记忆必须绑定到用户"
        );
    }

    #[test]
    fn detects_profile_trigger_and_validates_keys() {
        assert!(should_extract_profile("请记住我叫张三"));
        assert!(!should_extract_profile("差旅报销流程是什么"));
        assert!(is_profile_key("profile.name"));
        assert!(!is_profile_key("Name"));
        assert!(!is_profile_key("x"));
        assert_eq!(event_chunk_type(Some("dialogue")), "dialogue");
        assert_eq!(
            event_chunk_type(Some("event:u1:c1:dialogue:u1:c1:0:abc")),
            "dialogue"
        );
        assert_eq!(
            event_chunk_type(Some("event:u1:c1:semantic_memory_l1:0")),
            "semantic_memory_l1"
        );
    }

    #[tokio::test]
    async fn persist_uses_python_memory_id_and_logical_key() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        persist_turn(
            &pool,
            None,
            "u1",
            "c1",
            "s1",
            "你好",
            "你好，我在。",
            DEFAULT_EVENT_EXPIRY_DAYS,
        )
        .await;
        let row: (String, String) = sqlx::query_as(
            "SELECT memory_id, memory_key FROM memory_records WHERE user_id = 'u1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(row.0.starts_with("u1:c1:0:"), "{}", row.0);
        assert!(
            row.1.starts_with("event:u1:c1:dialogue:"),
            "{}",
            row.1
        );
    }

    #[tokio::test]
    async fn semantic_l1_uses_chunk_index_memory_id_like_python() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        for index in 0..10 {
            insert_event_memory_at(
                &pool,
                "u1",
                "c-sem",
                "s1",
                "dialogue",
                &format!("用户: 确认事项 {index}\n助手: 已经完成后续相关安排。"),
                55.0,
                DEFAULT_EVENT_EXPIRY_DAYS,
                Some(index),
                Some(index),
            )
            .await
            .unwrap();
        }
        let summarizer = MemorySummarizer::Fixed {
            topic: "确认".into(),
            summary: "已确认十轮事项。".into(),
        };
        let merged = consolidate_with(
            &pool,
            Some(&summarizer),
            "u1",
            "c-sem",
            DEFAULT_EVENT_EXPIRY_DAYS,
            false,
        )
        .await;
        assert!(merged >= 1, "merged={merged}");
        let row: (String, String) = sqlx::query_as(
            "SELECT memory_id, memory_key FROM memory_records WHERE conversation_id = 'c-sem' \
             AND (memory_key = 'semantic_memory_l1' OR memory_key LIKE 'event:%:%:semantic_memory_l1:%')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(
            row.0.starts_with("u1:c-sem:0:"),
            "L1 memory_id should use local chunk_index, got {}",
            row.0
        );
        assert!(
            !row.0.contains(":L1:"),
            "L1 should not use L-level memory_id, got {}",
            row.0
        );
        assert_eq!(row.1, "event:u1:c-sem:semantic_memory_l1:0");
    }

    #[tokio::test]
    async fn rewriting_same_logical_key_supersedes_previous_event() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        let first = insert_event_memory_at(
            &pool,
            "u1",
            "c1",
            "s1",
            "semantic_memory_l1",
            "【主题】报销\n【摘要】第一版摘要",
            70.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
            Some(0),
            Some(9),
        )
        .await
        .unwrap();
        let second = insert_event_memory_at(
            &pool,
            "u1",
            "c1",
            "s1",
            "semantic_memory_l1",
            "【主题】报销\n【摘要】第二版摘要",
            80.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
            Some(0),
            Some(9),
        )
        .await
        .unwrap();
        assert_ne!(first, second);
        let rows: Vec<(String, String, i64, Option<String>)> = sqlx::query_as(
            "SELECT memory_id, status, version, supersedes_memory_id \
             FROM memory_records WHERE user_id = 'u1' ORDER BY version ASC",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, first);
        assert_eq!(rows[0].1, "superseded");
        assert_eq!(rows[0].2, 1);
        assert!(rows[0].3.is_none());
        assert_eq!(rows[1].0, second);
        assert_eq!(rows[1].1, "active");
        assert_eq!(rows[1].2, 2);
        assert_eq!(rows[1].3.as_deref(), Some(first.as_str()));
        let active: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_records WHERE user_id = 'u1' AND status = 'active'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(active, 1);
    }

    #[tokio::test]
    async fn dialogue_events_keep_independent_logical_keys() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        insert_event_memory(
            &pool,
            "u1",
            "c1",
            "s1",
            "dialogue",
            "用户: 第一轮\n助手: 回答一",
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
            "dialogue",
            "用户: 第二轮\n助手: 回答二",
            55.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
        )
        .await
        .unwrap();
        let rows: Vec<(String, i64)> = sqlx::query_as(
            "SELECT status, version FROM memory_records WHERE user_id = 'u1' ORDER BY created_at",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(rows, vec![("active".into(), 1), ("active".into(), 1)]);
    }

    #[tokio::test]
    async fn search_hard_filters_conversation_and_expired_rows() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        insert_event_memory(
            &pool,
            "u1",
            "c1",
            "s1",
            "dialogue",
            "用户: 报销时限是几天？\n助手: 差旅报销需要在发生后 3 天内提交。",
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
            "dialogue",
            "用户: 报销时限是几天？\n助手: 另一会话说差旅报销时限是 7 天。",
            90.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
        )
        .await
        .unwrap();
        let expired_id = insert_event_memory(
            &pool,
            "u1",
            "c1",
            "s1",
            "dialogue",
            "用户: 报销时限是几天？\n助手: 已过期的 1 天说法。",
            99.0,
            DEFAULT_EVENT_EXPIRY_DAYS,
        )
        .await
        .unwrap();
        let past = crate::db::fmt_dt(&(crate::db::now_naive() - chrono::Duration::days(1)));
        sqlx::query("UPDATE memory_records SET expires_at = ? WHERE memory_id = ?")
            .bind(&past)
            .bind(&expired_id)
            .execute(&pool)
            .await
            .unwrap();

        let hits =
            search_event_memories(&pool, "u1", "报销时限是多久", Some("c1"), Some("s1"), 8, None)
                .await
                .unwrap();
        assert!(!hits.is_empty());
        assert!(hits.iter().all(|hit| hit.conversation_id.as_deref() == Some("c1")));
        assert!(hits.iter().all(|hit| hit.session_id.as_deref() == Some("s1")));
        assert!(hits.iter().all(|hit| !hit.content.contains("7 天")));
        assert!(hits.iter().all(|hit| !hit.content.contains("已过期")));
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
        assert!(deleted);
        let remaining: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_records WHERE user_id = 'u1' AND status = 'active'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(remaining, 0);
        let empty = delete_conversation_memory(&pool, "u1", "c-empty").await.unwrap();
        assert!(empty);
        assert!(!delete_conversation_memory(&pool, "u1", "  ").await.unwrap());
        assert!(!delete_conversation_memory(&pool, "", "c1").await.unwrap());
    }

    #[test]
    fn memory_success_logs_match_python() {
        assert_eq!(
            format_memory_delete_success("u1", "c1"),
            "✓ 长期记忆删除成功 (user_id=u1, conversation_id=c1)"
        );
        assert_eq!(
            format_memory_write_success("u1", "c1", 2),
            "✓ 长期记忆写入成功 (user_id=u1, conversation_id=c1, chunks=2)"
        );
        assert_eq!(
            format_summary_write_success(1, "u1", "c1", 0, 10),
            "✓ 递进摘要 L1 写入成功 (user_id=u1, conversation_id=c1, turns=0-10)"
        );
        assert_eq!(
            format_profile_update_success("u1", 2),
            "个性化记忆更新完成 (user_id=u1, updated=2)"
        );
        assert_eq!(format_empty_memory_chunks(), "没有可写入的长期记忆块。");
        assert_eq!(
            format_profile_extract_fail("llm timeout"),
            "个性化记忆抽取失败，已跳过: llm timeout"
        );
        assert_eq!(
            format_profile_write_fail("db locked"),
            "个性化记忆写入失败，已跳过: db locked"
        );
        assert_eq!(
            format_memory_skip_recall(),
            "当前问题未命中长期记忆关联信号，跳过记忆召回"
        );
        assert_eq!(
            format_consolidate_check_fail("busy"),
            "递进摘要合并检查失败，下次触发点重试: busy"
        );
        assert_eq!(
            format_persist_worker_fail("panic"),
            "后台长期记忆写入失败: panic"
        );
        assert_eq!(
            format_memory_maintenance_complete(2, 1, 3),
            "长期记忆维护完成: rescored=2 expired=1 purged=3"
        );
        assert_eq!(
            format_memory_search_fail("timeout"),
            "长期记忆检索失败: timeout"
        );
        assert_eq!(
            format_access_stat_async_fail("db locked"),
            "异步更新长期记忆访问统计失败: db locked"
        );
        assert_eq!(
            format_access_stat_submit_fail("runtime shutdown"),
            "提交长期记忆访问统计任务失败: runtime shutdown"
        );
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
        assert!(prompt.starts_with("以下是当前用户已确认的个性化记忆，仅在相关时参考："));
        assert!(prompt.contains("- profile.name: 张三"));
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

    fn mem(role: &str, content: &str) -> crate::short_memory::MemoryMessage {
        crate::short_memory::MemoryMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    #[test]
    fn recent_unsaved_conversation_keeps_last_top_k_turns() {
        let history = vec![
            mem("user", "第一轮问题"),
            mem("assistant", "第一轮回答"),
            mem("user", "第二轮问题"),
            mem("assistant", "第二轮回答"),
            mem("user", "第三轮问题"),
            mem("assistant", "第三轮回答"),
            mem("user", "第四轮问题"),
            mem("assistant", "第四轮回答"),
        ];
        let prompt = recent_unsaved_conversation_prompt(&history, 3);
        assert!(prompt.starts_with(
            "以下是当前会话最近尚未进入长期记忆的对话原文，仅用于补足刚刚发生的上下文："
        ));
        assert!(!prompt.contains("第一轮问题"));
        assert!(prompt.contains("1. 用户: 第二轮问题\n助手: 第二轮回答"));
        assert!(prompt.contains("2. 用户: 第三轮问题\n助手: 第三轮回答"));
        assert!(prompt.contains("3. 用户: 第四轮问题\n助手: 第四轮回答"));
        assert!(recent_unsaved_conversation_prompt(&[], 3).is_empty());
    }

    #[tokio::test]
    async fn long_term_memory_skips_profile_and_recent_without_hints() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        upsert_profile_candidates(
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
        let history = vec![mem("user", "我叫张三"), mem("assistant", "好的")];
        let messages = build_long_term_memory_messages(
            &pool,
            "u1",
            "差旅报销流程是什么",
            Some("c1"),
            Some("s1"),
            3,
            None,
            Some("rag"),
            &history,
        )
        .await;
        assert!(messages.is_empty());
        assert!(
            build_long_term_memory_messages(
                &pool,
                "u1",
                "你还记得我叫什么吗",
                Some("c1"),
                Some("s1"),
                3,
                None,
                Some("direct"),
                &history,
            )
            .await
            .is_empty()
        );
    }

    #[tokio::test]
    async fn long_term_memory_includes_profile_recent_and_event() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        upsert_profile_candidates(
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
        let history = vec![
            mem("user", "报销时限是几天"),
            mem("assistant", "发生后三天内提交"),
        ];
        let messages = build_long_term_memory_messages(
            &pool,
            "u1",
            "你还记得上次说的报销时限吗",
            Some("c1"),
            Some("s1"),
            3,
            None,
            Some("rag"),
            &history,
        )
        .await;
        assert_eq!(messages.len(), 3);
        assert!(messages[0].content.contains("个性化记忆"));
        assert!(messages[0].content.contains("张三"));
        assert!(messages[1].content.contains("尚未进入长期记忆"));
        assert!(messages[1].content.contains("用户: 报销时限是几天"));
        assert!(messages[2].content.contains("长期记忆"));
        assert!(messages[2].content.contains("差旅报销"));
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

        let hits = search_event_memories(&pool, "u1", "报销时限是多久", Some("c1"), Some("s1"), 3, None)
            .await
            .unwrap();
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
             WHERE user_id = 'u1' AND content LIKE '%3 天%' \
             AND (memory_key = 'semantic_memory_l1' OR memory_key LIKE 'event:%:%:semantic_memory_l1:%')",
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
        let hits = search_event_memories(&pool, "u1", "差旅报销时限", None, None, 2, None)
            .await
            .unwrap();
        assert_eq!(hits[0].memory_key, "semantic_memory_l3");
        assert_eq!(hits[1].memory_key, "dialogue");
    }

    #[test]
    fn event_importance_matches_python_rules() {
        assert_eq!(calculate_event_importance(""), (45.0, 0.55));
        assert_eq!(calculate_event_importance("你好"), (45.0, 0.55));

        let (strong, strong_conf) =
            calculate_event_importance("必须明确这是长期目标，一定要完成后续相关安排和检查事项。");
        assert_eq!(strong, 70.0);
        assert_eq!(strong_conf, 0.65);

        let (temporary, _) = calculate_event_importance("今天明天刚才临时这次当前都要看一下情况");
        assert_eq!(temporary, 25.0);

        let (mixed, mixed_conf) = calculate_event_importance("必须今天完成");
        assert_eq!(mixed, 51.0);
        assert_eq!(mixed_conf, 0.55);

        let (capped, _) = calculate_event_importance("必须必须必须必须必须必须");
        assert_eq!(capped, 70.0);
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
        .await
        .unwrap();
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
        .await
        .unwrap();
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
        .await
        .unwrap();
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
        .await
        .unwrap();
        assert_eq!(whole.get("original_chunks").and_then(Value::as_array).map(Vec::len), Some(2));
        assert!(whole.get("summary").and_then(Value::as_str).unwrap_or("").contains("当前会话"));
    }

    #[test]
    fn consolidation_thresholds_are_10_20_40() {
        assert_eq!(CONSOLIDATION_LEVELS, [(10, 1), (20, 2), (40, 3)]);
        assert!(is_substantive_dialogue(&[("user".into(), "确认周五上线".into())]));
        assert!(!is_substantive_dialogue(&[("user".into(), "好的".into()), ("assistant".into(), "收到".into())]));
        assert!(!is_substantive_dialogue(&[("user".into(), "谢谢！".into())]));
    }

    #[tokio::test]
    async fn persist_turn_numbers_dialogue_turns() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        persist_turn(&pool, None, "u1", "c-turns", "s1", "第一问", "第一答", DEFAULT_EVENT_EXPIRY_DAYS).await;
        persist_turn(&pool, None, "u1", "c-turns", "s1", "第二问", "第二答", DEFAULT_EVENT_EXPIRY_DAYS).await;
        persist_turn(&pool, None, "u1", "c-turns", "s1", "第三问", "第三答", DEFAULT_EVENT_EXPIRY_DAYS).await;
        assert_eq!(count_dialogue_turns(&pool, "u1", "c-turns").await, 3);
        let turns: Vec<(i64, i64)> = sqlx::query_as(
            "SELECT turn_start, turn_end FROM memory_records \
             WHERE conversation_id = 'c-turns' \
             AND (memory_key = 'dialogue' OR memory_key LIKE 'event:%:%:dialogue:%') \
             ORDER BY turn_start",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(turns, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[tokio::test]
    async fn persist_turn_splits_long_answer_into_same_turn_chunks() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        let answer = "甲".repeat(600);
        persist_turn(
            &pool,
            None,
            "u1",
            "c-long",
            "s1",
            "请详细说明",
            &answer,
            DEFAULT_EVENT_EXPIRY_DAYS,
        )
        .await;
        let turns: Vec<(String, i64, i64)> = sqlx::query_as(
            "SELECT content, turn_start, turn_end FROM memory_records \
             WHERE conversation_id = 'c-long' \
             AND (memory_key = 'dialogue' OR memory_key LIKE 'event:%:%:dialogue:%') \
             ORDER BY created_at",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert!(turns.len() >= 2);
        assert!(turns.iter().all(|(_, start, end)| *start == 0 && *end == 0));
        assert_eq!(count_dialogue_turns(&pool, "u1", "c-long").await, 1);
        assert!(turns.iter().any(|(content, _, _)| content.contains("用户: 请详细说明")));
        let ids: Vec<String> = sqlx::query_scalar(
            "SELECT memory_id FROM memory_records WHERE conversation_id = 'c-long' \
             AND (memory_key = 'dialogue' OR memory_key LIKE 'event:%:%:dialogue:%') \
             ORDER BY created_at",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert!(ids[0].starts_with("u1:c-long:0:"), "{}", ids[0]);
        assert!(
            ids.iter().any(|id| id.starts_with("u1:c-long:1:")),
            "{ids:?}"
        );
    }

    #[tokio::test]
    async fn topic_changed_skips_high_confidence_continuation() {
        let summarizer = MemorySummarizer::Fixed {
            topic: "t".into(),
            summary: "s".into(),
        };
        assert!(!summarizer.topic_changed("差旅报销标准", "那住宿费怎么报").await);
        assert!(summarizer.topic_changed("差旅报销标准", "服务器发布窗口").await);
    }

    #[tokio::test]
    async fn persist_turn_uses_content_importance() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        persist_turn(
            &pool,
            None,
            "u1",
            "c-imp",
            "s1",
            "必须明确这是长期目标",
            "已经完成后续相关安排和检查事项。",
            DEFAULT_EVENT_EXPIRY_DAYS,
        )
        .await;
        let row: (f64, f64, f64) = sqlx::query_as(
            "SELECT importance_score, base_importance_score, confidence_score \
             FROM memory_records WHERE conversation_id = 'c-imp' \
             AND (memory_key = 'dialogue' OR memory_key LIKE 'event:%:%:dialogue:%')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let content = "用户: 必须明确这是长期目标\n\n助手: 已经完成后续相关安排和检查事项。";
        let expected = calculate_event_importance(content);
        assert_eq!(row, (expected.0, expected.0, expected.1));
        assert_eq!(expected.0, 70.0);
        assert_eq!(expected.1, 0.65);
    }

    #[tokio::test]
    async fn l1_skips_low_information_and_advances_cursor() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        for _ in 0..10 {
            persist_turn(&pool, None, "u1", "c-low", "s1", "好的", "收到", DEFAULT_EVENT_EXPIRY_DAYS).await;
        }
        let l1: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_records WHERE conversation_id = 'c-low' \
             AND (memory_key = 'semantic_memory_l1' OR memory_key LIKE 'event:%:%:semantic_memory_l1:%')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(l1, 0);
        assert_eq!(get_consolidation_cursor(&pool, "u1", "c-low", 1).await, 10);
    }

    #[tokio::test]
    async fn cursor_waits_for_successful_write() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        for index in 0..10 {
            insert_event_memory_at(
                &pool,
                "u1",
                "c-wait",
                "s1",
                "dialogue",
                &format!("用户: 确认 Alpha 上线计划 {index}\n助手: 已记录"),
                55.0,
                DEFAULT_EVENT_EXPIRY_DAYS,
                Some(index),
                Some(index),
            )
            .await
            .unwrap();
        }
        let merged = consolidate_level(
            &pool,
            None,
            "u1",
            "c-wait",
            1,
            10,
            DEFAULT_EVENT_EXPIRY_DAYS,
        )
        .await;
        assert_eq!(merged, 0);
        assert_eq!(get_consolidation_cursor(&pool, "u1", "c-wait", 1).await, 0);

        let summarizer = MemorySummarizer::Fixed {
            topic: "Alpha".into(),
            summary: "已确认 Alpha 上线计划。".into(),
        };
        let written = consolidate_with(
            &pool,
            Some(&summarizer),
            "u1",
            "c-wait",
            DEFAULT_EVENT_EXPIRY_DAYS,
            false,
        )
        .await;
        assert_eq!(written, 1);
        assert_eq!(get_consolidation_cursor(&pool, "u1", "c-wait", 1).await, 10);
        let content: String = sqlx::query_scalar(
            "SELECT content FROM memory_records WHERE conversation_id = 'c-wait' \
             AND (memory_key = 'semantic_memory_l1' OR memory_key LIKE 'event:%:%:semantic_memory_l1:%')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(content.contains("【主题】Alpha"));
        assert!(content.contains("【摘要】已确认 Alpha 上线计划。"));
        assert!(content.contains("【原文】"));
        assert!(content.contains("确认 Alpha 上线计划"));
    }

    #[tokio::test]
    async fn higher_levels_merge_lower_summaries() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        for index in 0..20 {
            insert_event_memory_at(
                &pool,
                "u1",
                "c-l2",
                "s1",
                "dialogue",
                &format!("用户: 讨论项目进度 {index}\n助手: 记下了"),
                55.0,
                DEFAULT_EVENT_EXPIRY_DAYS,
                Some(index),
                Some(index),
            )
            .await
            .unwrap();
        }
        let summarizer = MemorySummarizer::Fixed {
            topic: "项目进度".into(),
            summary: "持续跟踪项目进度。".into(),
        };
        let first = consolidate_with(
            &pool,
            Some(&summarizer),
            "u1",
            "c-l2",
            DEFAULT_EVENT_EXPIRY_DAYS,
            false,
        )
        .await;
        let second = consolidate_with(
            &pool,
            Some(&summarizer),
            "u1",
            "c-l2",
            DEFAULT_EVENT_EXPIRY_DAYS,
            false,
        )
        .await;
        assert_eq!(first, 2);
        assert_eq!(second, 1);
        let l1: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_records WHERE conversation_id = 'c-l2' \
             AND (memory_key = 'semantic_memory_l1' OR memory_key LIKE 'event:%:%:semantic_memory_l1:%')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let l2: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_records WHERE conversation_id = 'c-l2' \
             AND (memory_key = 'semantic_memory_l2' OR memory_key LIKE 'event:%:%:semantic_memory_l2:%')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(l1, 2);
        assert_eq!(l2, 1);
        assert_eq!(get_consolidation_cursor(&pool, "u1", "c-l2", 1).await, 20);
        assert_eq!(get_consolidation_cursor(&pool, "u1", "c-l2", 2).await, 20);
        let l2_row: (String, String) = sqlx::query_as(
            "SELECT memory_id, memory_key FROM memory_records WHERE conversation_id = 'c-l2' \
             AND (memory_key = 'semantic_memory_l2' OR memory_key LIKE 'event:%:%:semantic_memory_l2:%')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(
            l2_row.0.starts_with("u1:c-l2:L2:0:20:"),
            "L2 memory_id should follow Python L-level shape, got {}",
            l2_row.0
        );
        assert_eq!(l2_row.1, "event:u1:c-l2:semantic_memory_l2:20");
    }

    fn persist_job(
        pool: &sqlx::SqlitePool,
        conversation_id: &str,
        query: &str,
        answer: &str,
    ) -> PersistJob {
        PersistJob {
            pool: pool.clone(),
            llm: None,
            user_id: "u1".into(),
            conversation_id: conversation_id.into(),
            session_id: "s1".into(),
            query: query.into(),
            answer: answer.into(),
            expiry_days: DEFAULT_EVENT_EXPIRY_DAYS,
        }
    }

    #[tokio::test]
    async fn persist_queue_drops_when_full_and_skips_empty_answer() {
        assert!(!enqueue_persist_turn(
            sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap(),
            None,
            "u1".into(),
            "c-empty".into(),
            "s1".into(),
            "问题".into(),
            "  ".into(),
            DEFAULT_EVENT_EXPIRY_DAYS,
        ));
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        let (queue, _rx) = MemoryPersistQueue::bounded(1);
        assert!(queue.try_enqueue(persist_job(&pool, "c-full", "第一问", "第一答")));
        assert!(!queue.try_enqueue(persist_job(&pool, "c-full", "第二问", "第二答")));
        assert_eq!(MEMORY_PERSIST_QUEUE_SIZE, 100);
    }

    #[tokio::test]
    async fn persist_queue_serializes_turn_numbers() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_schema(&pool).await.unwrap();
        let queue = MemoryPersistQueue::start(8);
        assert!(queue.try_enqueue(persist_job(&pool, "c-serial", "第一问", "第一答")));
        assert!(queue.try_enqueue(persist_job(&pool, "c-serial", "第二问", "第二答")));
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if count_dialogue_turns(&pool, "u1", "c-serial").await >= 2 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("串行写入超时");
        let turns: Vec<(i64, i64)> = sqlx::query_as(
            "SELECT turn_start, turn_end FROM memory_records \
             WHERE conversation_id = 'c-serial' \
             AND (memory_key = 'dialogue' OR memory_key LIKE 'event:%:%:dialogue:%') \
             ORDER BY turn_start",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(turns, vec![(0, 0), (1, 1)]);
    }
}
