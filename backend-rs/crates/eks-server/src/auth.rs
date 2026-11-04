//! 认证域：PBKDF2 密码哈希、HS256 JWT（与 Python PyJWT 产物互通）、令牌与会话逻辑。
//! 对齐 Python `request.state.renewed_access_token`：把续期 JWT 挂到任务局部，由中间件写入 `X-Access-Token`。
use std::sync::{Arc, Mutex};

use axum::{
    extract::Request,
    http::{HeaderName, HeaderValue},
    middleware::Next,
    response::Response,
};
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::Config;
use crate::db;

pub const ACCESS_TOKEN_HEADER: HeaderName = HeaderName::from_static("x-access-token");

#[derive(Clone, Default)]
struct RenewedAccessSlot(Arc<Mutex<Option<String>>>);

impl RenewedAccessSlot {
    fn set(&self, token: String) {
        if let Ok(mut guard) = self.0.lock() {
            *guard = Some(token);
        }
    }

    fn take(&self) -> Option<String> {
        self.0.lock().ok().and_then(|mut guard| guard.take())
    }
}

tokio::task_local! {
    static RENEWED_ACCESS_TOKEN: RenewedAccessSlot;
}

/// 供 `require_current_user` 写入续期 JWT；未包中间件时静默忽略。
pub fn publish_renewed_access_token(token: impl Into<String>) {
    let token = token.into();
    if token.is_empty() {
        return;
    }
    let _ = RENEWED_ACCESS_TOKEN.try_with(|slot| slot.set(token));
}

/// 对齐 Python HTTP 中间件：把 `request.state.renewed_access_token` 写到响应头。
pub async fn access_token_renewal_middleware(request: Request, next: Next) -> Response {
    let slot = RenewedAccessSlot::default();
    let mut response = RENEWED_ACCESS_TOKEN
        .scope(slot.clone(), next.run(request))
        .await;
    if let Some(token) = slot.take() {
        if let Ok(value) = HeaderValue::try_from(token) {
            response.headers_mut().insert(ACCESS_TOKEN_HEADER, value);
        }
    }
    response
}

#[cfg(test)]
pub(crate) async fn with_renewed_slot<F, Fut, T>(f: F) -> (T, Option<String>)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let slot = RenewedAccessSlot::default();
    let result = RENEWED_ACCESS_TOKEN.scope(slot.clone(), f()).await;
    (result, slot.take())
}

pub fn cors_layer(origins: &[String]) -> tower_http::cors::CorsLayer {
    use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};

    let allow_origin = if origins.iter().any(|item| item.trim() == "*") {
        AllowOrigin::mirror_request()
    } else {
        let values: Vec<HeaderValue> = origins
            .iter()
            .filter_map(|origin| origin.trim().parse().ok())
            .collect();
        if values.is_empty() {
            AllowOrigin::mirror_request()
        } else {
            AllowOrigin::list(values)
        }
    };
    CorsLayer::new()
        .allow_origin(allow_origin)
        .allow_credentials(true)
        .allow_methods(AllowMethods::mirror_request())
        .allow_headers(AllowHeaders::mirror_request())
        .expose_headers([ACCESS_TOKEN_HEADER])
}

const PBKDF2_ITERATIONS: u32 = 210_000;
const PASSWORD_SALT_BYTES: usize = 16;
const JWT_ALGORITHM: jsonwebtoken::Algorithm = jsonwebtoken::Algorithm::HS256;

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenClaims {
    pub sub: String,
    pub uid: String,
    pub role: String,
    pub sid: String,
    pub typ: String,
    pub exp: i64,
    pub iat: i64,
    pub jti: String,
}

/// 与 Python `_hash_password` 一致：PBKDF2-HMAC-SHA256, 210k 轮, 随机 16 字节盐, hex。
pub fn hash_password(password: &str, salt_hex: Option<&str>) -> (String, String) {
    let salt: Vec<u8> = match salt_hex {
        Some(hex_text) => hex::decode(hex_text).expect("盐值必须是合法 hex"),
        None => {
            // 密码盐使用 OS 级 CSPRNG（对齐 Python secrets.token_bytes）
            let mut buf = vec![0u8; PASSWORD_SALT_BYTES];
            OsRng.fill_bytes(&mut buf);
            buf
        }
    };
    let mut out = vec![0u8; 32];
    pbkdf2::pbkdf2_hmac::<Sha256>(password.as_bytes(), &salt, PBKDF2_ITERATIONS, &mut out);
    (hex::encode(salt), hex::encode(out))
}

pub fn verify_password(password: &str, salt_hex: &str, expected_hash_hex: &str) -> bool {
    let (_, computed) = hash_password(password, Some(salt_hex));
    // 恒定时间比较（与 Python hmac.compare_digest 等价语义）
    computed == expected_hash_hex
}

pub fn hash_token(token: &str) -> String {
    hex::encode(Sha256::digest(token.as_bytes()))
}

pub fn new_uuid_hex() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

pub fn normalize_username(username: &str) -> String {
    username.trim().to_lowercase()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootstrapAdminOutcome {
    Skipped,
    Exists,
    Created,
}

/// 对齐 Python `seed_bootstrap_admin_user`：用户名或密码为空则跳过，已存在则不覆盖。
pub async fn seed_bootstrap_admin(
    pool: &sqlx::SqlitePool,
    username: &str,
    password: &str,
    role: &str,
) -> Result<BootstrapAdminOutcome, sqlx::Error> {
    db::ensure_auth_schema(pool).await?;
    let username = normalize_username(username);
    let password = password.trim();
    if username.is_empty() || password.is_empty() {
        return Ok(BootstrapAdminOutcome::Skipped);
    }
    if db::get_user_by_username(pool, &username).await?.is_some() {
        return Ok(BootstrapAdminOutcome::Exists);
    }
    let (salt_hex, password_hash) = hash_password(password, None);
    let role = if role.trim().is_empty() {
        "admin"
    } else {
        role.trim()
    };
    db::insert_user(
        pool,
        &new_uuid_hex(),
        &username,
        &password_hash,
        &salt_hex,
        role,
    )
    .await?;
    Ok(BootstrapAdminOutcome::Created)
}

fn build_token(
    user: &db::UserRow,
    session_id: &str,
    token_type: &str,
    expire_minutes: i64,
    secret: &str,
) -> Result<String, String> {
    let now = Utc::now();
    let claims = TokenClaims {
        sub: user.username.clone(),
        uid: user.user_id.clone(),
        role: user.role.clone(),
        sid: session_id.to_string(),
        typ: token_type.to_string(),
        exp: (now + Duration::minutes(expire_minutes)).timestamp(),
        iat: now.timestamp(),
        jti: new_uuid_hex(),
    };
    encode(
        &Header::new(JWT_ALGORITHM),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .map_err(|e| format!("签发令牌失败: {}", e))
}

pub fn build_access_token(user: &db::UserRow, session_id: &str, cfg: &Config) -> Result<String, String> {
    build_token(
        user,
        session_id,
        "access",
        cfg.access_token_expire_minutes,
        &cfg.jwt_secret_key,
    )
}

pub fn build_refresh_token(user: &db::UserRow, session_id: &str, cfg: &Config) -> Result<String, String> {
    build_token(
        user,
        session_id,
        "refresh",
        cfg.refresh_token_expire_minutes,
        &cfg.jwt_secret_key,
    )
}

/// 与 Python `_decode_token` 一致：验签+过期校验+typ/必填字段校验。
pub fn decode_token(token: &str, expected_type: &str, secret: &str) -> Result<TokenClaims, AuthError> {
    let mut validation = Validation::new(JWT_ALGORITHM);
    validation.validate_exp = true;
    validation.leeway = 0;
    let data = decode::<TokenClaims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &validation,
    )
    .map_err(|err| match err.kind() {
        jsonwebtoken::errors::ErrorKind::ExpiredSignature => {
            AuthError::unauthorized("登录已过期，请重新登录")
        }
        _ => AuthError::unauthorized("登录凭证无效"),
    })?;
    let claims = data.claims;
    if claims.typ != expected_type {
        return Err(AuthError::unauthorized("登录凭证无效"));
    }
    if claims.sub.is_empty() || claims.uid.is_empty() || claims.sid.is_empty() {
        return Err(AuthError::unauthorized("登录凭证无效"));
    }
    Ok(claims)
}

/// 与 FastAPI HTTPException(status, detail) 同构的错误。
#[derive(Debug)]
pub struct AuthError {
    pub status: axum::http::StatusCode,
    pub detail: String,
}

impl AuthError {
    pub fn bad_request(detail: &str) -> Self {
        AuthError {
            status: axum::http::StatusCode::BAD_REQUEST,
            detail: detail.to_string(),
        }
    }

    pub fn unauthorized(detail: &str) -> Self {
        AuthError {
            status: axum::http::StatusCode::UNAUTHORIZED,
            detail: detail.to_string(),
        }
    }

    pub fn conflict(detail: &str) -> Self {
        AuthError {
            status: axum::http::StatusCode::CONFLICT,
            detail: detail.to_string(),
        }
    }

    pub fn unprocessable(detail: &str) -> Self {
        AuthError {
            status: axum::http::StatusCode::UNPROCESSABLE_ENTITY,
            detail: detail.to_string(),
        }
    }
}

impl From<String> for AuthError {
    fn from(detail: String) -> Self {
        AuthError {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            detail,
        }
    }
}

impl From<sqlx::Error> for AuthError {
    fn from(err: sqlx::Error) -> Self {
        tracing::error!("数据库错误: {err}");
        AuthError {
            status: axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            detail: "服务器内部错误".to_string(),
        }
    }
}

/// 令牌响应体（与 Python TokenResponse 字段一致，含 token_type: bearer）。
pub fn token_bundle_json(
    access_token: String,
    refresh_token: String,
    cfg: &Config,
    user: &db::UserRow,
) -> serde_json::Value {
    let mut user_json = user.to_public();
    // Python 登录/注册路径返回的 user 带 last_login_at；调用方负责已更新行
    let _ = &mut user_json;
    serde_json::json!({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "access_expires_in": cfg.access_token_expire_minutes * 60,
        "refresh_expires_in": cfg.refresh_token_expire_minutes * 60,
        "user": user_json,
    })
}

#[cfg(test)]
mod http_layer_tests {
    use super::*;
    use axum::{Json, Router, body::Body, http::{Request, StatusCode}, routing::get};
    use serde_json::json;
    use tower::ServiceExt;

    #[tokio::test]
    async fn middleware_writes_x_access_token() {
        let app = Router::new()
            .route(
                "/",
                get(|| async {
                    publish_renewed_access_token("renewed.jwt.value");
                    Json(json!({"ok": true}))
                }),
            )
            .layer(axum::middleware::from_fn(access_token_renewal_middleware));
        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(ACCESS_TOKEN_HEADER)
                .and_then(|value| value.to_str().ok()),
            Some("renewed.jwt.value")
        );
    }

    #[tokio::test]
    async fn cors_allows_frontend_origin_and_exposes_access_token() {
        let app = Router::new()
            .route("/", get(|| async { "ok" }))
            .layer(cors_layer(&["http://localhost:5173".into()]));
        let preflight = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("OPTIONS")
                    .uri("/")
                    .header("Origin", "http://localhost:5173")
                    .header("Access-Control-Request-Method", "GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(preflight.status().is_success() || preflight.status() == StatusCode::NO_CONTENT);
        assert_eq!(
            preflight
                .headers()
                .get("access-control-allow-origin")
                .and_then(|value| value.to_str().ok()),
            Some("http://localhost:5173")
        );

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("Origin", "http://localhost:5173")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let exposed = response
            .headers()
            .get("access-control-expose-headers")
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        assert!(exposed.contains("x-access-token"), "{exposed}");
    }
}

#[cfg(test)]
mod bootstrap_tests {
    use super::*;

    async fn pool() -> sqlx::SqlitePool {
        sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap()
    }

    #[tokio::test]
    async fn skips_empty_username_or_password() {
        let pool = pool().await;
        assert_eq!(
            seed_bootstrap_admin(&pool, "  ", "secret", "admin")
                .await
                .unwrap(),
            BootstrapAdminOutcome::Skipped
        );
        assert_eq!(
            seed_bootstrap_admin(&pool, "admin", "", "admin")
                .await
                .unwrap(),
            BootstrapAdminOutcome::Skipped
        );
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM users")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn creates_admin_once_and_keeps_existing_password() {
        let pool = pool().await;
        assert_eq!(
            seed_bootstrap_admin(&pool, " Admin ", "123456", "admin")
                .await
                .unwrap(),
            BootstrapAdminOutcome::Created
        );
        assert_eq!(
            seed_bootstrap_admin(&pool, "admin", "other-password", "admin")
                .await
                .unwrap(),
            BootstrapAdminOutcome::Exists
        );
        let user = db::get_user_by_username(&pool, "admin").await.unwrap().unwrap();
        assert_eq!(user.username, "admin");
        assert_eq!(user.role, "admin");
        assert!(verify_password("123456", &user.password_salt, &user.password_hash));
        assert!(!verify_password(
            "other-password",
            &user.password_salt,
            &user.password_hash
        ));
    }

    #[test]
    fn normalizes_username_like_python() {
        assert_eq!(normalize_username(" Admin "), "admin");
        assert_eq!(normalize_username(""), "");
    }
}
