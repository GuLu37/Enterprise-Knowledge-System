//! 配置：复用 backend/.env 的同名键，附加 EKS_* 前缀的 Rust 侧专属配置。
use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub api_prefix: String,
    pub host: String,
    pub port: u16,
    pub sqlite_path: String,
    pub jwt_secret_key: String,
    pub access_token_expire_minutes: i64,
    pub refresh_token_expire_minutes: i64,
    pub session_idle_timeout_minutes: i64,
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_i64(key: &str, default: i64) -> i64 {
    env::var(key)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

impl Config {
    /// 从 .env 文件（默认 ../backend/.env，可用 EKS_ENV_FILE 覆盖）+ 环境变量加载。
    pub fn load() -> Self {
        let env_file = env_or("EKS_ENV_FILE", "../backend/.env");
        let _ = dotenvy::from_filename(&env_file);
        Config {
            api_prefix: env_or("API_PREFIX", "/api/v1"),
            host: env_or("EKS_HOST", "127.0.0.1"),
            port: env_i64("EKS_PORT", 8001) as u16,
            sqlite_path: env_or(
                "EKS_SQLITE_PATH",
                "../backend/app/data/rag_metadata.db",
            ),
            jwt_secret_key: env_or("JWT_SECRET_KEY", "change-me-in-production"),
            access_token_expire_minutes: env_i64("ACCESS_TOKEN_EXPIRE_MINUTES", 60),
            refresh_token_expire_minutes: env_i64("REFRESH_TOKEN_EXPIRE_MINUTES", 60),
            session_idle_timeout_minutes: env_i64("SESSION_IDLE_TIMEOUT_MINUTES", 60),
        }
    }
}
