//! 认证域：PBKDF2 密码哈希、HS256 JWT（与 Python PyJWT 产物互通）、令牌与会话逻辑。
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::Config;
use crate::db;

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
