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
    pub bge_model_dir: String,
    pub bge_query_instruction: String,
    pub bge_max_length: i64,
    pub milvus_lite_path: String,
    pub cross_encoder_enabled: bool,
    pub cross_encoder_onnx: String,
    pub cross_encoder_tokenizer: String,
    pub cross_encoder_candidate_k: usize,
    pub cross_encoder_weight: f64,
    pub rule_rerank_weight: f64,
    pub cross_encoder_confidence_threshold: f64,
    pub cross_encoder_confidence_margin: f64,
    pub cross_encoder_trigger_mode: String,
    pub llm_temperature: f64,
    pub llm_timeout: u64,
    pub llm_max_fallback_attempts: usize,
    pub deepseek_api_key: String,
    pub deepseek_base_url: String,
    pub deepseek_model: String,
    pub openrouter_api_key: String,
    pub openrouter_api_base: String,
    pub openrouter_model: String,
    pub openai_api_key: String,
    pub openai_model: String,
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

fn env_f64(key: &str, default: f64) -> f64 {
    env::var(key)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .map(|value| {
            let normalized = value.trim().to_lowercase();
            normalized == "true" || normalized == "1" || normalized == "yes" || normalized == "on"
        })
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
            bge_model_dir: env_or(
                "EKS_BGE_MODEL_DIR",
                "../backend/app/models/bge-base-zh-v1.5",
            ),
            bge_query_instruction: env_or(
                "BGE_QUERY_INSTRUCTION",
                "为这个句子生成表示以用于检索相关文章：",
            ),
            bge_max_length: env_i64("BGE_MAX_LENGTH", 512),
            milvus_lite_path: env_or(
                "EKS_MILVUS_LITE_PATH",
                "../backend/app/data/milvus_lite.db",
            ),
            cross_encoder_enabled: env_bool("CROSS_ENCODER_ENABLED", true),
            cross_encoder_onnx: env_or("EKS_CROSS_ENCODER_ONNX", ""),
            cross_encoder_tokenizer: env_or("EKS_CROSS_ENCODER_TOKENIZER", ""),
            cross_encoder_candidate_k: env_i64("CROSS_ENCODER_CANDIDATE_K", 30).max(1) as usize,
            cross_encoder_weight: env_f64("CROSS_ENCODER_WEIGHT", 0.65).clamp(0.0, 1.0),
            rule_rerank_weight: env_f64("RULE_RERANK_WEIGHT", 0.35).clamp(0.0, 1.0),
            cross_encoder_confidence_threshold: env_f64("CROSS_ENCODER_CONFIDENCE_THRESHOLD", 0.60).clamp(0.0, 1.0),
            cross_encoder_confidence_margin: env_f64("CROSS_ENCODER_CONFIDENCE_MARGIN", 0.08).clamp(0.0, 1.0),
            cross_encoder_trigger_mode: env_or("CROSS_ENCODER_TRIGGER_MODE", "adaptive"),
            llm_temperature: env_f64("LLM_TEMPERATURE", 0.7),
            llm_timeout: env_i64("LLM_TIMEOUT", 60).max(1) as u64,
            llm_max_fallback_attempts: env_i64("LLM_MAX_FALLBACK_ATTEMPTS", 2).max(1) as usize,
            deepseek_api_key: env_or("DEEPSEEK_API_KEY", ""),
            deepseek_base_url: env_or("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model: env_or("DEEPSEEK_MODEL", "deepseek-chat"),
            openrouter_api_key: env_or("OPENROUTER_API_KEY", ""),
            openrouter_api_base: env_or("OPENROUTER_API_BASE", ""),
            openrouter_model: env_or("OPENROUTER_MODEL", ""),
            openai_api_key: env_or("OPENAI_API_KEY", ""),
            openai_model: env_or("OPENAI_MODEL", "gpt-4o-mini"),
        }
    }
}
