//! SQLite 访问层：与 Python 版共用同一个 rag_metadata.db（WAL 已开启）。
//! 时间列沿用 SQLAlchemy 的 naive-UTC 文本格式 "YYYY-MM-DD HH:MM:SS.ffffff"。
use chrono::{NaiveDateTime, Utc};

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

/// JSON 序列化格式与 pydantic v2 一致：naive ISO（空格→T），None → null。
pub fn dt_to_iso(text: &str) -> Option<String> {
    parse_dt(text).map(|dt| dt.format("%Y-%m-%dT%H:%M:%S%.6f").to_string())
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
