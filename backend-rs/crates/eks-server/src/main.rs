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
    let config = Config::load();
    eks_server::logging::init_tracing(&config.log_dir, &config.log_level);
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
    eks_server::db::ensure_document_schema(&pool).await?;
    eks_server::db::ensure_conversation_schema(&pool).await?;
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
                "✓ 已创建 bootstrap 管理员账号: {}",
                eks_server::auth::normalize_username(&config.bootstrap_admin_username)
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
        store.clone(), embedder, pool.clone(),
    )
    .with_result_cache(config.enable_cache, config.cache_ttl)
    .with_retrieval_options(&config);
    if config.cross_encoder_enabled {
        match eks_server::cross_encoder::CrossEncoder::load(
            &eks_server::cross_encoder::CrossEncoderConfig {
                onnx_path: std::path::PathBuf::from(&config.cross_encoder_onnx),
                tokenizer_path: std::path::PathBuf::from(&config.cross_encoder_tokenizer),
                max_length: config.cross_encoder_max_length,
                batch_size: config.cross_encoder_batch_size,
                device: config.cross_encoder_device.clone(),
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
                // 保留根因（工件路径等），经 /chat/runtime-status 透出供诊断。
                engine.cross_encoder_load_error = Some(exc);
            }
        }
    }
    let engine = std::sync::Arc::new(engine);
    if config.use_sparse_retriever
        && config.sparse_index_mode.trim().eq_ignore_ascii_case("cached")
    {
        let warm_engine = engine.clone();
        tokio::spawn(async move {
            let started = std::time::Instant::now();
            match warm_engine.warm_sparse_index().await {
                Ok(rows) => tracing::info!(
                    "稀疏持久索引预热完成: rows={rows} elapsed={:.2}s",
                    started.elapsed().as_secs_f64()
                ),
                Err(err) => tracing::warn!("稀疏持久索引预热失败，检索将回退 LIKE 路径: {err}"),
            }
        });
    }
    eks_server::memory::configure_memory_vectors(eks_server::memory::MemoryVectorSink {
        store: store.clone(),
        collection: config.milvus_memory_collection.clone(),
        embedder: engine.embedder.clone(),
        embedding_dimension: config.embedding_dimension,
    });

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

    tracing::info!("==================================================");
    tracing::info!("🚀 启动 {} (v{})", config.app_name, config.app_version);
    tracing::info!("环境: {}", config.environment);
    tracing::info!("调试模式: {}", config.debug);
    tracing::info!("系统正在初始化各类设置...请稍等...");
    tracing::info!(
        "eks-server 监听: {addr} | sqlite={} | prefix={}",
        config.sqlite_path,
        config.api_prefix
    );
    tracing::info!("核心服务已就绪，模型与 Milvus 正在后台初始化。");
    tracing::info!("==================================================");

    let default_llm = eks_server::llm::LlmClient::from_env(&config);
    let mut warmup = RuntimeWarmup::initializing(&config);
    if config.startup_init_llm {
        let llm_started_at = std::time::Instant::now();
        match warmup.apply_startup_llm(&default_llm).await {
            Ok(()) => tracing::info!(
                "{}",
                format!(
                    "启动阶段 LLM 初始化完成: provider={} duration={:.2}ms",
                    warmup.provider.as_deref().unwrap_or("None"),
                    llm_started_at.elapsed().as_secs_f64() * 1000.0
                )
            ),
            Err(exc) => tracing::warn!("启动阶段 LLM 初始化失败，将回退到后台预热: {exc}"),
        }
    }
    let state = AppState {
        pool,
        config: Arc::new(config),
        engine,
        default_llm,
        runtime_warmup: Arc::new(Mutex::new(warmup)),
    };
    eks_server::chat_stream::spawn_background_warmup(state.clone());

    let app = routes::router(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;
    Ok(())
}

/// Ctrl+C / SIGTERM / Ctrl+Break 时先打关闭横幅再退出，
/// 对齐 Python lifespan 的「🛑 关闭应用...」日志可达性。
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("🛑 关闭应用...");
}
