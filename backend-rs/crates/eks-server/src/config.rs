//! 配置：复用 backend/.env 的同名键，附加 EKS_* 前缀的 Rust 侧专属配置。
use std::env;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Config {
    pub app_name: String,
    pub app_version: String,
    pub environment: String,
    pub debug: bool,
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
    pub bootstrap_admin_username: String,
    pub bootstrap_admin_password: String,
    pub bootstrap_admin_role: String,
    pub startup_init_llm: bool,
    pub max_upload_size: i64,
    pub allowed_file_types: Vec<String>,
    pub log_dir: String,
    pub log_level: String,
    pub upload_dir: String,
    pub search_top_k: usize,
    pub similarity_threshold: f64,
    pub milvus_child_collection: String,
    pub dense_weight: f64,
    pub sparse_weight: f64,
    pub use_dense_retriever: bool,
    pub use_sparse_retriever: bool,
    pub use_hybrid_retriever: bool,
    pub sparse_index_max_rows: usize,
}

const DEFAULT_CORS_ORIGINS: &str =
    r#"["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080"]"#;
const DEFAULT_ALLOWED_FILE_TYPES: &str =
    r#"["pdf", "txt", "md", "doc", "docx", "ppt", "pptx", "xlsx", "xls"]"#;
const DEFAULT_MAX_UPLOAD_SIZE: i64 = 104_857_600;

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

/// 对齐 Python `_parse_list` + `_normalize_allowed_types`。
pub fn parse_allowed_file_types(raw: &str) -> Vec<String> {
    let source = if raw.trim().is_empty() {
        DEFAULT_ALLOWED_FILE_TYPES
    } else {
        raw
    };
    parse_cors_origins(source)
        .into_iter()
        .map(|item| {
            item.trim()
                .trim_matches('"')
                .trim_matches('\'')
                .trim_start_matches('.')
                .to_lowercase()
        })
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

/// 对齐 Python `_validate_runtime_security`：非 DEBUG 拒绝弱 JWT / 默认管理员口令。
pub fn validate_runtime_security(
    debug: bool,
    jwt_secret_key: &str,
    bootstrap_admin_password: &str,
) -> Result<(), String> {
    if debug {
        return Ok(());
    }
    if jwt_secret_key.is_empty()
        || jwt_secret_key == "change-me-in-production"
        || jwt_secret_key.len() < 32
    {
        return Err("生产环境必须配置长度至少 32 个字符的 JWT_SECRET_KEY".into());
    }
    if matches!(
        bootstrap_admin_password,
        "123456" | "password" | "change-me"
    ) {
        return Err("生产环境禁止使用默认 BOOTSTRAP_ADMIN_PASSWORD".into());
    }
    Ok(())
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

fn env_first(keys: &[&str], default: &str) -> String {
    for key in keys {
        if let Ok(value) = env::var(key) {
            if !value.trim().is_empty() {
                return value;
            }
        }
    }
    default.to_string()
}

fn env_i64_first(keys: &[&str], default: i64) -> i64 {
    for key in keys {
        if let Some(value) = env::var(key)
            .ok()
            .and_then(|value| value.trim().parse().ok())
        {
            return value;
        }
    }
    default
}

/// 对齐 Python `_resolve_sqlite_url`：从 `DATABASE_URL` 抽出 sqlite 文件路径。
pub fn sqlite_path_from_database_url(raw: &str) -> Option<String> {
    let path = raw.trim().strip_prefix("sqlite:///")?;
    if path.is_empty() || path == ":memory:" || path.starts_with("file:") {
        return Some(if path.is_empty() {
            ":memory:".into()
        } else {
            path.to_string()
        });
    }
    let parsed = Path::new(path);
    if parsed.is_absolute() {
        return Some(path.to_string());
    }
    parsed
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("../backend/app/data/{name}"))
}

/// 对齐 Python `_resolve_backend_model_path`：相对路径相对 backend 目录解析。
pub fn resolve_backend_model_path(raw: &str, default: &str) -> String {
    let value = raw.trim();
    if value.is_empty() {
        return default.to_string();
    }
    let path = PathBuf::from(value);
    if path.is_absolute() {
        return value.to_string();
    }
    let looks_like_path = value.starts_with('.')
        || value.starts_with("app/")
        || value.contains("/models/")
        || value.ends_with(".onnx");
    let relative = value.strip_prefix("./").unwrap_or(value);
    let candidate = PathBuf::from("../backend").join(relative);
    if looks_like_path || candidate.exists() {
        return candidate.to_string_lossy().into_owned();
    }
    value.to_string()
}

/// 对齐 Python `_resolve_app_path`：相对路径相对 `backend/app` 解析。
pub fn resolve_app_path(raw: &str, default: &str) -> String {
    let value = if raw.trim().is_empty() {
        default
    } else {
        raw.trim()
    };
    let path = Path::new(value);
    if path.is_absolute() {
        return value.to_string();
    }
    let relative = value.strip_prefix("./").unwrap_or(value);
    PathBuf::from("../backend/app")
        .join(relative)
        .to_string_lossy()
        .into_owned()
}

fn load_sqlite_path() -> String {
    if let Ok(value) = env::var("EKS_SQLITE_PATH") {
        if !value.trim().is_empty() {
            return value;
        }
    }
    if let Ok(value) = env::var("DATABASE_URL") {
        if let Some(path) = sqlite_path_from_database_url(&value) {
            return path;
        }
    }
    "../backend/app/data/rag_metadata.db".into()
}

fn load_bge_model_dir() -> String {
    if let Ok(value) = env::var("EKS_BGE_MODEL_DIR") {
        if !value.trim().is_empty() {
            return value;
        }
    }
    resolve_backend_model_path(
        &env::var("BGE_MODEL_NAME").unwrap_or_default(),
        "../backend/app/models/bge-base-zh-v1.5",
    )
}

fn load_milvus_lite_path() -> String {
    if let Ok(value) = env::var("EKS_MILVUS_LITE_PATH") {
        if !value.trim().is_empty() {
            return value;
        }
    }
    resolve_app_path(
        &env::var("MILVUS_LITE_PATH").unwrap_or_default(),
        "data/milvus_lite.db",
    )
}

/// 对齐 Python `UPLOAD_DIR`：相对路径相对 sqlite 所在 app 目录解析。
pub fn resolve_upload_dir(upload_dir: &str, sqlite_path: &str) -> PathBuf {
    let data_dir = Path::new(sqlite_path).parent().unwrap_or(Path::new("."));
    let raw = upload_dir.trim();
    if raw.is_empty() {
        return data_dir.join("uploads");
    }
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        data_dir.parent().unwrap_or(data_dir).join(path)
    }
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
        let config = Config {
            app_name: env_or("APP_NAME", "Enterprise Knowledge System"),
            app_version: env_or("APP_VERSION", "1.0.0"),
            environment: env_or("ENVIRONMENT", "开发版"),
            debug: env_bool("DEBUG", true),
            api_prefix: env_or("API_PREFIX", "/api/v1"),
            host: env_first(&["EKS_HOST", "SERVER_HOST"], "127.0.0.1"),
            port: env_i64_first(&["EKS_PORT", "SERVER_PORT"], 8000) as u16,
            sqlite_path: load_sqlite_path(),
            jwt_secret_key: env_or("JWT_SECRET_KEY", "change-me-in-production"),
            access_token_expire_minutes: env_i64("ACCESS_TOKEN_EXPIRE_MINUTES", 60),
            refresh_token_expire_minutes: env_i64("REFRESH_TOKEN_EXPIRE_MINUTES", 60),
            session_idle_timeout_minutes: env_i64("SESSION_IDLE_TIMEOUT_MINUTES", 60),
            bge_model_dir: load_bge_model_dir(),
            bge_query_instruction: env_or(
                "BGE_QUERY_INSTRUCTION",
                "为这个句子生成表示以用于检索相关文章：",
            ),
            bge_max_length: env_i64("BGE_MAX_LENGTH", 512),
            milvus_lite_path: load_milvus_lite_path(),
            cross_encoder_enabled: env_bool("CROSS_ENCODER_ENABLED", false),
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
            bootstrap_admin_username: env_or("BOOTSTRAP_ADMIN_USERNAME", "admin"),
            bootstrap_admin_password: env_or("BOOTSTRAP_ADMIN_PASSWORD", "123456"),
            bootstrap_admin_role: env_or("BOOTSTRAP_ADMIN_ROLE", "admin"),
            startup_init_llm: env_bool("STARTUP_INIT_LLM", true),
            max_upload_size: env_i64("MAX_UPLOAD_SIZE", DEFAULT_MAX_UPLOAD_SIZE).max(0),
            allowed_file_types: parse_allowed_file_types(
                &env::var("ALLOWED_FILE_TYPES").unwrap_or_default(),
            ),
            log_dir: env_or("LOG_DIR", "logs"),
            log_level: env_or("LOG_LEVEL", "INFO"),
            upload_dir: env_or("UPLOAD_DIR", ""),
            search_top_k: env_i64("SEARCH_TOP_K", 5).clamp(1, 50) as usize,
            similarity_threshold: env_f64("SIMILARITY_THRESHOLD", 0.5).clamp(0.0, 1.0),
            milvus_child_collection: env_first(
                &["MILVUS_CHILD_COLLECTION_NAME", "MILVUS_DOC_COLLECTION_NAME"],
                "doc_chunk",
            ),
            dense_weight: env_f64("DENSE_WEIGHT", 0.5).clamp(0.0, 1.0),
            sparse_weight: env_f64("SPARSE_WEIGHT", 0.5).clamp(0.0, 1.0),
            use_dense_retriever: env_bool("USE_DENSE_RETRIEVER", true),
            use_sparse_retriever: env_bool("USE_SPARSE_RETRIEVER", true),
            use_hybrid_retriever: env_bool("USE_HYBRID_RETRIEVER", true),
            sparse_index_max_rows: env_i64("SPARSE_INDEX_MAX_ROWS", 50_000).max(1000) as usize,
        };
        if let Err(err) = validate_runtime_security(
            config.debug,
            &config.jwt_secret_key,
            &config.bootstrap_admin_password,
        ) {
            panic!("{err}");
        }
        config
    }

    #[cfg(test)]
    pub fn jwt_test() -> Self {
        Config {
            app_name: "Enterprise Knowledge System".into(),
            app_version: "1.0.0".into(),
            environment: "开发版".into(),
            debug: true,
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
            bootstrap_admin_username: "admin".into(),
            bootstrap_admin_password: "123456".into(),
            bootstrap_admin_role: "admin".into(),
            startup_init_llm: true,
            max_upload_size: DEFAULT_MAX_UPLOAD_SIZE,
            allowed_file_types: parse_allowed_file_types(""),
            log_dir: "logs".into(),
            log_level: "INFO".into(),
            upload_dir: String::new(),
            search_top_k: 5,
            similarity_threshold: 0.5,
            milvus_child_collection: "doc_chunk".into(),
            dense_weight: 0.5,
            sparse_weight: 0.5,
            use_dense_retriever: true,
            use_sparse_retriever: true,
            use_hybrid_retriever: true,
            sparse_index_max_rows: 50_000,
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
        assert_eq!(Config::jwt_test().bootstrap_admin_username, "admin");
        assert_eq!(Config::jwt_test().bootstrap_admin_role, "admin");
        assert!(Config::jwt_test().startup_init_llm);
        assert_eq!(Config::jwt_test().app_name, "Enterprise Knowledge System");
        assert_eq!(Config::jwt_test().app_version, "1.0.0");
        assert_eq!(Config::jwt_test().environment, "开发版");
        assert!(Config::jwt_test().debug);
        assert_eq!(Config::jwt_test().max_upload_size, 104_857_600);
        assert!(Config::jwt_test().allowed_file_types.contains(&"pdf".into()));
        assert!(Config::jwt_test().allowed_file_types.contains(&"xlsx".into()));
        assert_eq!(Config::jwt_test().log_dir, "logs");
        assert_eq!(Config::jwt_test().log_level, "INFO");
        assert_eq!(Config::jwt_test().search_top_k, 5);
        assert_eq!(Config::jwt_test().similarity_threshold, 0.5);
        assert!(Config::jwt_test().upload_dir.is_empty());
        assert!(!Config::jwt_test().cross_encoder_enabled);
        assert_eq!(Config::jwt_test().milvus_child_collection, "doc_chunk");
        assert_eq!(Config::jwt_test().dense_weight, 0.5);
        assert_eq!(Config::jwt_test().sparse_weight, 0.5);
        assert!(Config::jwt_test().use_dense_retriever);
        assert!(Config::jwt_test().use_sparse_retriever);
        assert!(Config::jwt_test().use_hybrid_retriever);
        assert_eq!(Config::jwt_test().sparse_index_max_rows, 50_000);
    }

    #[test]
    fn parses_python_env_path_aliases() {
        assert_eq!(
            sqlite_path_from_database_url("sqlite:///./app/data/rag_metadata.db").as_deref(),
            Some("../backend/app/data/rag_metadata.db")
        );
        assert_eq!(
            sqlite_path_from_database_url("sqlite:////var/eks/rag_metadata.db").as_deref(),
            Some("/var/eks/rag_metadata.db")
        );
        assert!(sqlite_path_from_database_url("postgresql://localhost/rag").is_none());
        assert_eq!(
            resolve_backend_model_path("./app/models/bge-base-zh-v1.5", "def"),
            "../backend/app/models/bge-base-zh-v1.5"
        );
        assert_eq!(
            resolve_backend_model_path("BAAI/bge-base-zh-v1.5", "def"),
            "BAAI/bge-base-zh-v1.5"
        );
        assert_eq!(
            resolve_app_path("./data/milvus_lite.db", "data/milvus_lite.db"),
            "../backend/app/data/milvus_lite.db"
        );
        assert_eq!(
            resolve_app_path("", "data/milvus_lite.db"),
            "../backend/app/data/milvus_lite.db"
        );
    }

    #[test]
    fn resolves_upload_dir_next_to_sqlite_data() {
        let sqlite = "/workspace/backend/app/data/rag_metadata.db";
        assert_eq!(
            resolve_upload_dir("", sqlite),
            PathBuf::from("/workspace/backend/app/data/uploads")
        );
        assert_eq!(
            resolve_upload_dir("./data/uploads", sqlite),
            PathBuf::from("/workspace/backend/app/data/uploads")
        );
        assert_eq!(
            resolve_upload_dir("/var/eks/uploads", sqlite),
            PathBuf::from("/var/eks/uploads")
        );
    }

    #[test]
    fn debug_skips_runtime_security_checks() {
        assert!(validate_runtime_security(true, "change-me-in-production", "123456").is_ok());
        assert!(validate_runtime_security(true, "", "password").is_ok());
    }

    #[test]
    fn production_rejects_weak_jwt_and_bootstrap_password() {
        assert_eq!(
            validate_runtime_security(false, "change-me-in-production", "strong-pass").unwrap_err(),
            "生产环境必须配置长度至少 32 个字符的 JWT_SECRET_KEY"
        );
        assert_eq!(
            validate_runtime_security(false, "", "strong-pass").unwrap_err(),
            "生产环境必须配置长度至少 32 个字符的 JWT_SECRET_KEY"
        );
        assert_eq!(
            validate_runtime_security(false, "short-secret-key", "strong-pass").unwrap_err(),
            "生产环境必须配置长度至少 32 个字符的 JWT_SECRET_KEY"
        );
        let ok_secret = "a".repeat(32);
        assert!(validate_runtime_security(false, &ok_secret, "strong-pass").is_ok());
        assert_eq!(
            validate_runtime_security(false, &ok_secret, "123456").unwrap_err(),
            "生产环境禁止使用默认 BOOTSTRAP_ADMIN_PASSWORD"
        );
        assert_eq!(
            validate_runtime_security(false, &ok_secret, "password").unwrap_err(),
            "生产环境禁止使用默认 BOOTSTRAP_ADMIN_PASSWORD"
        );
        assert_eq!(
            validate_runtime_security(false, &ok_secret, "change-me").unwrap_err(),
            "生产环境禁止使用默认 BOOTSTRAP_ADMIN_PASSWORD"
        );
    }

    #[test]
    fn parses_allowed_file_types_like_python() {
        assert_eq!(
            parse_allowed_file_types(r#"["PDF", ".Txt", "md"]"#),
            vec!["pdf", "txt", "md"]
        );
        assert_eq!(
            parse_allowed_file_types("docx, xlsx, pptx"),
            vec!["docx", "xlsx", "pptx"]
        );
        let defaults = parse_allowed_file_types("");
        assert!(defaults.contains(&"pdf".to_string()));
        assert!(defaults.contains(&"xls".to_string()));
        assert!(defaults.contains(&"doc".to_string()));
    }
}
