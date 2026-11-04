//! eks-server 二进制入口：Enterprise Knowledge System 的 Rust 后端
//! （分阶段绞杀 Python 版）。
use eks_server::config::Config;
use eks_server::routes;
use eks_server::state::{AppState, RuntimeWarmup};
use sqlx::sqlite::SqliteConnectOptions;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = Config::load();
    let addr = format!("{}:{}", config.host, config.port);

    // 与 Python 共用同一 SQLite 文件（WAL 模式，读写共存）
    let db_options = SqliteConnectOptions::new()
        .filename(&config.sqlite_path)
        .create_if_missing(true)
        .foreign_keys(true)
        .busy_timeout(std::time::Duration::from_secs(15))
        .pragma("journal_mode", "WAL");
    let pool = sqlx::SqlitePool::connect_with(db_options).await?;
    eks_server::db::ensure_auth_schema(&pool).await?;
    match eks_server::auth::seed_bootstrap_admin(
        &pool,
        &config.bootstrap_admin_username,
        &config.bootstrap_admin_password,
        &config.bootstrap_admin_role,
    )
    .await
    {
        Ok(eks_server::auth::BootstrapAdminOutcome::Created) => {
            tracing::info!(
                username = eks_server::auth::normalize_username(&config.bootstrap_admin_username),
                "已创建 bootstrap 管理员账号"
            );
        }
        Ok(eks_server::auth::BootstrapAdminOutcome::Exists) => {}
        Ok(eks_server::auth::BootstrapAdminOutcome::Skipped) => {
            tracing::info!("未配置 bootstrap 管理员，跳过账号初始化");
        }
        Err(err) => tracing::error!("初始化 bootstrap 管理员失败: {err}"),
    }
    eks_server::memory::ensure_schema(&pool).await?;

    // 检索引擎（Milvus Lite 直读 + ONNX embedder + SQLite 父块）
    let embedder = eks_server::embedder::shared_embedder(&config)
        .map_err(|e| format!("Embedding 内核初始化失败: {e}"))?;
    let store = eks_server::milvus_lite::MilvusLiteStore::open(&config.milvus_lite_path);
    let mut engine = eks_server::retrieval_engine::RetrievalEngine::new(
        store, embedder, pool.clone(),
    )
    .with_result_cache(config.enable_cache, config.cache_ttl);
    if config.cross_encoder_enabled {
        match eks_server::cross_encoder::CrossEncoder::load(
            &eks_server::cross_encoder::CrossEncoderConfig {
                onnx_path: std::path::PathBuf::from(&config.cross_encoder_onnx),
                tokenizer_path: std::path::PathBuf::from(&config.cross_encoder_tokenizer),
                max_length: 512,
                batch_size: 8,
            },
        ) {
            Ok(ce) => {
                tracing::info!("✓ Cross-Encoder ONNX 内核就绪 (adaptive)");
                engine = engine.with_cross_encoder(
                    std::sync::Arc::new(ce),
                    config.cross_encoder_candidate_k,
                    config.cross_encoder_weight,
                    config.rule_rerank_weight,
                    config.cross_encoder_confidence_threshold,
                    config.cross_encoder_confidence_margin,
                    config.cross_encoder_trigger_mode.clone(),
                );
            }
            Err(exc) => {
                tracing::warn!(
                    "Cross-Encoder 内核不可用，检索将走纯规则路径: {exc}"
                );
            }
        }
    }
    let engine = std::sync::Arc::new(engine);

    eks_server::memory::spawn_memory_maintenance(
        pool.clone(),
        eks_server::memory::MemoryCleanupSettings::from_config(&config),
        config.memory_cleanup_interval_seconds,
    );

    let mcp_skills = eks_server::remote_mcp::skills_from_manifests(&config.mcp_remote_tool_manifests);
    if mcp_skills.is_empty() {
        tracing::info!("远程 MCP 未配置 manifest");
    } else {
        let names: Vec<&str> = mcp_skills.iter().map(|item| item.name.as_str()).collect();
        tracing::info!("远程 MCP 已注册: {}", names.join(", "));
    }

    tracing::info!(
        "eks-server 启动: {addr} | sqlite={} | prefix={}",
        config.sqlite_path,
        config.api_prefix
    );

    let mut warmup = RuntimeWarmup::initializing(&config);
    if config.startup_init_llm {
        let llm_started_at = std::time::Instant::now();
        match warmup.apply_startup_llm(&config) {
            Ok(()) => tracing::info!(
                provider = ?warmup.provider,
                duration_ms = llm_started_at.elapsed().as_secs_f64() * 1000.0,
                "启动阶段 LLM 初始化完成"
            ),
            Err(exc) => tracing::warn!("启动阶段 LLM 初始化失败，将回退到后台预热: {exc}"),
        }
    }

    let state = AppState {
        pool,
        config: Arc::new(config),
        engine,
        runtime_warmup: Arc::new(Mutex::new(warmup)),
    };
    eks_server::chat_stream::spawn_background_warmup(state.clone());

    let app = routes::router(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}
