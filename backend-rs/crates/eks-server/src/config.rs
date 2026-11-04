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
    pub llm_provider: String,
    pub llm_temperature: f64,
    pub llm_timeout: u64,
    pub llm_max_tokens: u32,
    pub llm_max_fallback_attempts: usize,
    pub deepseek_api_key: String,
    pub deepseek_base_url: String,
    pub deepseek_model: String,
    pub openrouter_api_key: String,
    pub openrouter_api_base: String,
    pub openrouter_model: String,
    pub openai_api_key: String,
    pub openai_model: String,
    pub openai_base_url: String,
    pub anthropic_api_key: String,
    pub anthropic_model: String,
    pub ollama_base_url: String,
    pub ollama_model: String,
    pub chat_max_conversations: i64,
    pub cors_origins: Vec<String>,
    pub mcp_remote_tool_manifests: Vec<serde_json::Value>,
    pub unified_react_enabled: bool,
    pub unified_react_max_iterations: usize,
    pub unified_react_parallel_workers: usize,
    pub react_max_iterations: usize,
    pub retrieval_planner_llm_decision_enabled: bool,
    pub enable_cache: bool,
    pub cache_ttl: u64,
    pub memory_cleanup_interval_seconds: u64,
    pub memory_event_expiry_days: i64,
    pub memory_profile_stale_days: i64,
    pub memory_score_threshold: f64,
    pub memory_purge_retention_days: i64,
}

const DEFAULT_CORS_ORIGINS: &str =
    r#"["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080"]"#;

/// 对齐 Python `_parse_list`：JSON 数组，失败则按逗号拆。
pub fn parse_cors_origins(raw: &str) -> Vec<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return parse_cors_origins(DEFAULT_CORS_ORIGINS);
    }
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) {
        let items = match parsed {
            serde_json::Value::Array(items) => items,
            other => vec![other],
        };
        let origins: Vec<String> = items
            .into_iter()
            .filter_map(|item| match item {
                serde_json::Value::String(value) => Some(value),
                other => Some(other.to_string()),
            })
            .map(|value| value.trim().trim_matches('"').trim_matches('\'').to_string())
            .filter(|value| !value.is_empty())
            .collect();
        if !origins.is_empty() {
            return origins;
        }
    }
    trimmed
        .trim_matches(|ch| ch == '[' || ch == ']')
        .split(',')
        .map(|item| item.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|item| !item.is_empty())
        .collect()
}

/// 对齐 Python `_parse_dict_list`：JSON 对象数组，失败或非对象则跳过。
pub fn parse_dict_list(raw: &str) -> Vec<serde_json::Value> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let parsed = match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(value) => value,
        Err(_) => return Vec::new(),
    };
    match parsed {
        serde_json::Value::Object(_) => vec![parsed],
        serde_json::Value::Array(items) => items.into_iter().filter(serde_json::Value::is_object).collect(),
        _ => Vec::new(),
    }
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
            port: env_i64("EKS_PORT", 8000) as u16,
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
            llm_provider: env_or("LLM_PROVIDER", "ollama"),
            llm_temperature: env_f64("LLM_TEMPERATURE", 0.7),
            llm_timeout: env_i64("LLM_TIMEOUT", 60).max(1) as u64,
            llm_max_tokens: env_i64("LLM_MAX_TOKENS", 4096).max(1) as u32,
            llm_max_fallback_attempts: env_i64("LLM_MAX_FALLBACK_ATTEMPTS", 2).max(1) as usize,
            deepseek_api_key: env_or("DEEPSEEK_API_KEY", ""),
            deepseek_base_url: env_or("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model: env_or("DEEPSEEK_MODEL", "deepseek-chat"),
            openrouter_api_key: env_or("OPENROUTER_API_KEY", ""),
            openrouter_api_base: env_or("OPENROUTER_API_BASE", ""),
            openrouter_model: env_or("OPENROUTER_MODEL", ""),
            openai_api_key: env_or("OPENAI_API_KEY", ""),
            openai_model: env_or("OPENAI_MODEL", "gpt-4o-mini"),
            openai_base_url: env_or("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            anthropic_api_key: env_or("ANTHROPIC_API_KEY", ""),
            anthropic_model: env_or("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            ollama_base_url: env_or("OLLAMA_BASE_URL", ""),
            ollama_model: env_or("OLLAMA_MODEL", ""),
            chat_max_conversations: env_i64("CHAT_MAX_CONVERSATIONS", 20).clamp(1, 200),
            cors_origins: parse_cors_origins(&env::var("CORS_ORIGINS").unwrap_or_default()),
            mcp_remote_tool_manifests: parse_dict_list(
                &env::var("MCP_REMOTE_TOOL_MANIFESTS").unwrap_or_default(),
            ),
            unified_react_enabled: env_bool("UNIFIED_REACT_ENABLED", true),
            unified_react_max_iterations: env_i64("UNIFIED_REACT_MAX_ITERATIONS", 3).max(1) as usize,
            unified_react_parallel_workers: env_i64("UNIFIED_REACT_PARALLEL_WORKERS", 4).max(1)
                as usize,
            react_max_iterations: env_i64("REACT_MAX_ITERATIONS", 3).max(1) as usize,
            retrieval_planner_llm_decision_enabled: env_bool(
                "RETRIEVAL_PLANNER_LLM_DECISION_ENABLED",
                true,
            ),
            enable_cache: env_bool("ENABLE_CACHE", true),
            cache_ttl: env_i64("CACHE_TTL", 3600).max(0) as u64,
            memory_cleanup_interval_seconds: env_i64("MEMORY_CLEANUP_INTERVAL_SECONDS", 3600)
                .max(60) as u64,
            memory_event_expiry_days: env_i64("MEMORY_EVENT_EXPIRY_DAYS", 90).max(1),
            memory_profile_stale_days: env_i64("MEMORY_PROFILE_STALE_DAYS", 180).max(1),
            memory_score_threshold: env_f64("MEMORY_SCORE_THRESHOLD", 20.0).clamp(0.0, 100.0),
            memory_purge_retention_days: env_i64("MEMORY_PURGE_RETENTION_DAYS", 30).max(1),
        }
    }

    #[cfg(test)]
    pub fn jwt_test() -> Self {
        Config {
            api_prefix: "/api/v1".into(),
            host: "127.0.0.1".into(),
            port: 8000,
            sqlite_path: ":memory:".into(),
            jwt_secret_key: "unit-test-secret".into(),
            access_token_expire_minutes: 60,
            refresh_token_expire_minutes: 120,
            session_idle_timeout_minutes: 60,
            bge_model_dir: String::new(),
            bge_query_instruction: String::new(),
            bge_max_length: 512,
            milvus_lite_path: String::new(),
            cross_encoder_enabled: false,
            cross_encoder_onnx: String::new(),
            cross_encoder_tokenizer: String::new(),
            cross_encoder_candidate_k: 30,
            cross_encoder_weight: 0.65,
            rule_rerank_weight: 0.35,
            cross_encoder_confidence_threshold: 0.6,
            cross_encoder_confidence_margin: 0.08,
            cross_encoder_trigger_mode: "adaptive".into(),
            llm_provider: "ollama".into(),
            llm_temperature: 0.7,
            llm_timeout: 60,
            llm_max_tokens: 4096,
            llm_max_fallback_attempts: 2,
            deepseek_api_key: String::new(),
            deepseek_base_url: String::new(),
            deepseek_model: String::new(),
            openrouter_api_key: String::new(),
            openrouter_api_base: String::new(),
            openrouter_model: String::new(),
            openai_api_key: String::new(),
            openai_model: String::new(),
            openai_base_url: "https://api.openai.com/v1".into(),
            anthropic_api_key: String::new(),
            anthropic_model: "claude-sonnet-4-6".into(),
            ollama_base_url: String::new(),
            ollama_model: String::new(),
            chat_max_conversations: 20,
            cors_origins: parse_cors_origins(""),
            mcp_remote_tool_manifests: Vec::new(),
            unified_react_enabled: true,
            unified_react_max_iterations: 3,
            unified_react_parallel_workers: 4,
            react_max_iterations: 3,
            retrieval_planner_llm_decision_enabled: true,
            enable_cache: true,
            cache_ttl: 3600,
            memory_cleanup_interval_seconds: 3600,
            memory_event_expiry_days: 90,
            memory_profile_stale_days: 180,
            memory_score_threshold: 20.0,
            memory_purge_retention_days: 30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_json_and_csv_cors_origins() {
        assert_eq!(
            parse_cors_origins(r#"["http://localhost:5173", "http://127.0.0.1:5173"]"#),
            vec!["http://localhost:5173", "http://127.0.0.1:5173"]
        );
        assert_eq!(
            parse_cors_origins("http://localhost:3000, http://localhost:8080"),
            vec!["http://localhost:3000", "http://localhost:8080"]
        );
        assert!(parse_cors_origins("").contains(&"http://localhost:5173".to_string()));
    }

    #[test]
    fn parses_mcp_manifest_object_or_array() {
        let single = parse_dict_list(r#"{"name":"crm_lookup"}"#);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0]["name"], "crm_lookup");
        let many = parse_dict_list(r#"[{"name":"a"},"skip",{"name":"b"}]"#);
        assert_eq!(many.len(), 2);
        assert!(parse_dict_list("not-json").is_empty());
        assert!(parse_dict_list("").is_empty());
    }

    #[test]
    fn jwt_test_keeps_react_parallel_workers() {
        assert_eq!(Config::jwt_test().unified_react_parallel_workers, 4);
        assert!(Config::jwt_test().enable_cache);
        assert_eq!(Config::jwt_test().cache_ttl, 3600);
        assert_eq!(Config::jwt_test().memory_cleanup_interval_seconds, 3600);
        assert_eq!(Config::jwt_test().memory_event_expiry_days, 90);
        assert_eq!(Config::jwt_test().memory_score_threshold, 20.0);
    }
}
