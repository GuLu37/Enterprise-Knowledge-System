//! SQLite 访问层：与 Python 版共用同一个 rag_metadata.db（WAL 已开启）。
//! 时间列沿用 SQLAlchemy 的 naive-UTC 文本格式 "YYYY-MM-DD HH:MM:SS.ffffff"。
use chrono::{DateTime, FixedOffset, NaiveDateTime, Utc};

pub fn now_naive() -> NaiveDateTime {
    Utc::now().naive_utc()
}

/// 与 Python `datetime.utcnow()` 写入 SQLite 的字符串格式一致（6 位微秒）。
pub fn fmt_dt(dt: &NaiveDateTime) -> String {
    dt.format("%Y-%m-%d %H:%M:%S%.6f").to_string()
}

/// 解析 SQLite 文本时间（兼容不同微秒位数）。
pub fn parse_dt(text: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f").ok()
        .or_else(|| NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").ok())
}

/// 对齐 Python `conversation_service._parse_client_datetime`：
/// `fromisoformat(str(value).replace("Z", "+00:00"))`，带时区则转到 naive UTC。
pub fn parse_client_datetime(value: &str) -> Option<NaiveDateTime> {
    if value.is_empty() {
        return None;
    }
    let normalized = value.replace('Z', "+00:00");
    parse_client_aware(&normalized)
        .map(|dt| dt.with_timezone(&Utc).naive_utc())
        .or_else(|| parse_client_naive(&normalized))
}

fn parse_client_aware(text: &str) -> Option<DateTime<FixedOffset>> {
    DateTime::parse_from_rfc3339(text)
        .ok()
        .or_else(|| DateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f%:z").ok())
        .or_else(|| DateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%:z").ok())
        .or_else(|| DateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f%:z").ok())
        .or_else(|| DateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%:z").ok())
}

fn parse_client_naive(text: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f")
        .ok()
        .or_else(|| NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S").ok())
        .or_else(|| parse_dt(text))
}

/// JSON 序列化对齐 Python naive `datetime.isoformat()` / pydantic v2：
/// 空格→T，零微秒省略小数，None → null。
pub fn dt_to_iso(text: &str) -> Option<String> {
    parse_dt(text).map(|dt| naive_isoformat(&dt))
}

fn naive_isoformat(dt: &NaiveDateTime) -> String {
    if dt.and_utc().timestamp_subsec_micros() == 0 {
        dt.format("%Y-%m-%dT%H:%M:%S").to_string()
    } else {
        dt.format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
    }
}

/// 对齐 Python `datetime.now(timezone.utc).isoformat()`：`+00:00`，零微秒省略小数。
pub fn utc_isoformat(dt: DateTime<Utc>) -> String {
    let naive = dt.naive_utc();
    if dt.timestamp_subsec_micros() == 0 {
        format!("{}+00:00", naive.format("%Y-%m-%dT%H:%M:%S"))
    } else {
        format!("{}+00:00", naive.format("%Y-%m-%dT%H:%M:%S%.6f"))
    }
}

pub fn utc_isoformat_now() -> String {
    utc_isoformat(Utc::now())
}

/// 对齐 Python `init_metadata_db` 中的 users / auth_sessions 表。
pub async fn ensure_auth_schema(pool: &sqlx::SqlitePool) -> Result<(), sqlx::Error> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'admin',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_login_at TEXT
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_username ON users (username)")
        .execute(pool)
        .await?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS auth_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            refresh_token_hash TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            revoked_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_used_at TEXT
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_auth_sessions_refresh_token_hash \
         ON auth_sessions (refresh_token_hash)",
    )
    .execute(pool)
    .await?;
    sqlx::query("CREATE INDEX IF NOT EXISTS ix_auth_sessions_user_id ON auth_sessions (user_id)")
        .execute(pool)
        .await?;
    Ok(())
}

/// 对齐 Python `init_metadata_db` 收尾 debug。
pub fn format_sqlite_schema_ready() -> &'static str {
    "✓ SQLite 元数据与认证表初始化完成"
}

/// 对齐 Python `init_metadata_db` 中的 documents / document_chunks 表。
pub async fn ensure_document_schema(pool: &sqlx::SqlitePool) -> Result<(), sqlx::Error> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            content_type TEXT,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'processing',
            chunk_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            parent_id TEXT,
            chunk_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_document_order \
         ON document_chunks (document_id, chunk_index)",
    )
    .execute(pool)
    .await?;
    migrate_document_parent_id(pool).await?;
    Ok(())
}

/// 对齐 Python `init_metadata_db` 中的 user_conversations 表。
pub async fn ensure_conversation_schema(pool: &sqlx::SqlitePool) -> Result<(), sqlx::Error> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS user_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            session_id TEXT,
            payload TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, conversation_id)
        )",
    )
    .execute(pool)
    .await?;
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS ix_user_conversations_user_updated \
         ON user_conversations (user_id, updated_at)",
    )
    .execute(pool)
    .await?;
    Ok(())
}

pub(crate) async fn table_column_names(
    pool: &sqlx::SqlitePool,
    table: &str,
) -> Result<std::collections::HashSet<String>, sqlx::Error> {
    let rows = sqlx::query(&format!("PRAGMA table_info({table})"))
        .fetch_all(pool)
        .await?;
    let mut names = std::collections::HashSet::new();
    for row in rows {
        if let Ok(name) = sqlx::Row::try_get::<String, _>(&row, "name") {
            names.insert(name);
        }
    }
    Ok(names)
}

/// 对齐 Python：旧库补 `parent_id` 列，并按 `document_id:p{chunk_index:06}` 回填。
async fn migrate_document_parent_id(pool: &sqlx::SqlitePool) -> Result<(), sqlx::Error> {
    let columns = table_column_names(pool, "document_chunks").await?;
    if !columns.contains("parent_id") {
        sqlx::query("ALTER TABLE document_chunks ADD COLUMN parent_id TEXT")
            .execute(pool)
            .await?;
        tracing::info!("SQLite document_chunks 已补充 parent_id 字段");
    }
    let result = sqlx::query(
        "UPDATE document_chunks \
         SET parent_id = document_id || char(58) || 'p' || printf('%06d', chunk_index) \
         WHERE parent_id IS NULL OR parent_id = ''",
    )
    .execute(pool)
    .await?;
    if result.rows_affected() > 0 {
        tracing::info!(
            migrated = result.rows_affected(),
            "SQLite document_chunks 已补齐历史 Parent ID"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct UserRow {
    pub user_id: String,
    pub username: String,
    pub password_hash: String,
    pub password_salt: String,
    pub role: String,
    pub is_active: bool,
    pub created_at: String,
    pub updated_at: String,
    pub last_login_at: Option<String>,
}

impl UserRow {
    /// 与 Python `_serialize_user` 字段一致。
    pub fn to_public(&self) -> serde_json::Value {
        serde_json::json!({
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": dt_to_iso(&self.created_at),
            "updated_at": dt_to_iso(&self.updated_at),
            "last_login_at": self.last_login_at.as_deref().and_then(dt_to_iso),
        })
    }
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct SessionRow {
    pub session_id: String,
    pub user_id: String,
    pub refresh_token_hash: String,
    pub expires_at: String,
    pub revoked_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub last_used_at: Option<String>,
}

pub async fn get_user_by_username(
    pool: &sqlx::SqlitePool,
    username: &str,
) -> Result<Option<UserRow>, sqlx::Error> {
    sqlx::query_as::<_, UserRow>(
        "SELECT user_id, username, password_hash, password_salt, role, is_active, \
         created_at, updated_at, last_login_at FROM users WHERE username = ? LIMIT 1",
    )
    .bind(username.trim().to_lowercase())
    .fetch_optional(pool)
    .await
}

pub async fn get_user_by_id(
    pool: &sqlx::SqlitePool,
    user_id: &str,
) -> Result<Option<UserRow>, sqlx::Error> {
    sqlx::query_as::<_, UserRow>(
        "SELECT user_id, username, password_hash, password_salt, role, is_active, \
         created_at, updated_at, last_login_at FROM users WHERE user_id = ? LIMIT 1",
    )
    .bind(user_id)
    .fetch_optional(pool)
    .await
}

pub async fn get_session_by_id(
    pool: &sqlx::SqlitePool,
    session_id: &str,
) -> Result<Option<SessionRow>, sqlx::Error> {
    sqlx::query_as::<_, SessionRow>(
        "SELECT session_id, user_id, refresh_token_hash, expires_at, revoked_at, \
         created_at, updated_at, last_used_at FROM auth_sessions WHERE session_id = ? LIMIT 1",
    )
    .bind(session_id)
    .fetch_optional(pool)
    .await
}

/// 与 Python `_revoke_user_sessions` 一致：吊销该用户所有未吊销会话（可排除一个）。
pub async fn revoke_user_sessions(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    exclude_session_id: Option<&str>,
) -> Result<(), sqlx::Error> {
    let now = fmt_dt(&now_naive());
    match exclude_session_id {
        Some(exclude) => {
            sqlx::query(
                "UPDATE auth_sessions SET revoked_at = ?, updated_at = ? \
                 WHERE user_id = ? AND revoked_at IS NULL AND session_id != ?",
            )
            .bind(&now)
            .bind(&now)
            .bind(user_id)
            .bind(exclude)
            .execute(pool)
            .await?;
        }
        None => {
            sqlx::query(
                "UPDATE auth_sessions SET revoked_at = ?, updated_at = ? \
                 WHERE user_id = ? AND revoked_at IS NULL",
            )
            .bind(&now)
            .bind(&now)
            .bind(user_id)
            .execute(pool)
            .await?;
        }
    }
    Ok(())
}

pub async fn insert_user(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    username: &str,
    password_hash: &str,
    password_salt: &str,
    role: &str,
) -> Result<(), sqlx::Error> {
    let now = fmt_dt(&now_naive());
    sqlx::query(
        "INSERT INTO users (user_id, username, password_hash, password_salt, role, \
         is_active, created_at, updated_at, last_login_at) \
         VALUES (?, ?, ?, ?, ?, 1, ?, ?, NULL)",
    )
    .bind(user_id)
    .bind(username)
    .bind(password_hash)
    .bind(password_salt)
    .bind(role)
    .bind(&now)
    .bind(&now)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn touch_last_login(
    pool: &sqlx::SqlitePool,
    user_id: &str,
) -> Result<(), sqlx::Error> {
    let now = fmt_dt(&now_naive());
    sqlx::query("UPDATE users SET last_login_at = ? WHERE user_id = ?")
        .bind(&now)
        .bind(user_id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn update_password(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    password_hash: &str,
    password_salt: &str,
) -> Result<(), sqlx::Error> {
    let now = fmt_dt(&now_naive());
    sqlx::query(
        "UPDATE users SET password_hash = ?, password_salt = ?, updated_at = ? \
         WHERE user_id = ?",
    )
    .bind(password_hash)
    .bind(password_salt)
    .bind(&now)
    .bind(user_id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn insert_session(
    pool: &sqlx::SqlitePool,
    session_id: &str,
    user_id: &str,
    refresh_token_hash: &str,
    expires_at: &str,
) -> Result<(), sqlx::Error> {
    let now = fmt_dt(&now_naive());
    sqlx::query(
        "INSERT INTO auth_sessions (session_id, user_id, refresh_token_hash, expires_at, \
         revoked_at, created_at, updated_at, last_used_at) \
         VALUES (?, ?, ?, ?, NULL, ?, ?, ?)",
    )
    .bind(session_id)
    .bind(user_id)
    .bind(refresh_token_hash)
    .bind(expires_at)
    .bind(&now)
    .bind(&now)
    .bind(&now)
    .execute(pool)
    .await?;
    Ok(())
}

/// 与 Python `_renew_session_access_token` 一致：滑动续期会话。
pub async fn slide_session(
    pool: &sqlx::SqlitePool,
    session_id: &str,
    idle_timeout_minutes: i64,
) -> Result<(), sqlx::Error> {
    let now = now_naive();
    let expires = fmt_dt(&(now + chrono::Duration::minutes(idle_timeout_minutes)));
    sqlx::query(
        "UPDATE auth_sessions SET last_used_at = ?, updated_at = ?, expires_at = ? \
         WHERE session_id = ?",
    )
    .bind(fmt_dt(&now))
    .bind(fmt_dt(&now))
    .bind(expires)
    .bind(session_id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn revoke_session(pool: &sqlx::SqlitePool, session_id: &str) -> Result<(), sqlx::Error> {
    let now = fmt_dt(&now_naive());
    sqlx::query(
        "UPDATE auth_sessions SET revoked_at = ?, updated_at = ? WHERE session_id = ?",
    )
    .bind(&now)
    .bind(&now)
    .bind(session_id)
    .execute(pool)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqlite_schema_ready_log_matches_python() {
        assert_eq!(
            format_sqlite_schema_ready(),
            "✓ SQLite 元数据与认证表初始化完成"
        );
    }

    #[test]
    fn parse_client_datetime_accepts_frontend_iso_z_like_python() {
        let parsed = parse_client_datetime("2026-03-15T08:30:00.123Z").unwrap();
        assert_eq!(fmt_dt(&parsed), "2026-03-15 08:30:00.123000");
        let offset = parse_client_datetime("2026-03-15T16:30:00+08:00").unwrap();
        assert_eq!(fmt_dt(&offset), "2026-03-15 08:30:00.000000");
        let plus = parse_client_datetime("2026-03-15T08:30:00+00:00").unwrap();
        assert_eq!(fmt_dt(&plus), "2026-03-15 08:30:00.000000");
        let naive = parse_client_datetime("2026-03-15T08:30:00").unwrap();
        assert_eq!(fmt_dt(&naive), "2026-03-15 08:30:00.000000");
        let sqlite = parse_client_datetime("2026-03-15 08:30:00.123000").unwrap();
        assert_eq!(fmt_dt(&sqlite), "2026-03-15 08:30:00.123000");
        assert!(parse_client_datetime("").is_none());
        assert!(parse_client_datetime("not-a-date").is_none());
        assert!(parse_dt("2026-03-15T08:30:00.123Z").is_none());
    }

    #[test]
    fn dt_to_iso_matches_python_naive_isoformat() {
        assert_eq!(
            dt_to_iso("2026-01-01 00:00:00.000000").as_deref(),
            Some("2026-01-01T00:00:00")
        );
        assert_eq!(
            dt_to_iso("2026-01-01 00:00:00").as_deref(),
            Some("2026-01-01T00:00:00")
        );
        assert_eq!(
            dt_to_iso("2026-03-15 08:30:00.123000").as_deref(),
            Some("2026-03-15T08:30:00.123000")
        );
        assert_eq!(
            dt_to_iso("2026-03-15 08:30:00.000123").as_deref(),
            Some("2026-03-15T08:30:00.000123")
        );
        assert!(dt_to_iso("").is_none());
        assert!(dt_to_iso("not-a-date").is_none());
    }

    #[test]
    fn utc_isoformat_matches_python_timezone_utc() {
        let zero = DateTime::parse_from_rfc3339("2026-03-15T08:30:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(utc_isoformat(zero), "2026-03-15T08:30:00+00:00");
        let frac = DateTime::parse_from_rfc3339("2026-03-15T08:30:00.123456Z")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(utc_isoformat(frac), "2026-03-15T08:30:00.123456+00:00");
        assert!(!utc_isoformat_now().ends_with('Z'));
        assert!(utc_isoformat_now().ends_with("+00:00"));
    }

    #[tokio::test]
    async fn conversation_schema_supports_upsert_conflict() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        ensure_conversation_schema(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO user_conversations (user_id, conversation_id, session_id, payload, created_at, updated_at) \
             VALUES ('u1', 'c1', 's1', '[]', '2026-01-01 00:00:00.000000', '2026-01-01 00:00:00.000000')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO user_conversations (user_id, conversation_id, session_id, payload, created_at, updated_at) \
             VALUES ('u1', 'c1', 's2', '[1]', '2026-01-01 00:00:00.000000', '2026-01-02 00:00:00.000000') \
             ON CONFLICT(user_id, conversation_id) DO UPDATE SET \
             session_id = excluded.session_id, payload = excluded.payload, updated_at = excluded.updated_at",
        )
        .execute(&pool)
        .await
        .unwrap();
        let row: (String, String) = sqlx::query_as(
            "SELECT session_id, payload FROM user_conversations WHERE user_id = 'u1' AND conversation_id = 'c1'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(row, ("s2".into(), "[1]".into()));
    }

    #[tokio::test]
    async fn document_parent_id_is_backfilled_for_legacy_rows() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(
            "CREATE TABLE document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO document_chunks (document_id, chunk_index, chunk_text, created_at) \
             VALUES ('doc-a', 2, '正文', '2026-01-01 00:00:00.000000')",
        )
        .execute(&pool)
        .await
        .unwrap();
        ensure_document_schema(&pool).await.unwrap();
        let parent_id: String =
            sqlx::query_scalar("SELECT parent_id FROM document_chunks WHERE document_id = 'doc-a'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(parent_id, "doc-a:p000002");
    }
}
