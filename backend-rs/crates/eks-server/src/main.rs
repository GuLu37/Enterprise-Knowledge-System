//! eks-server 二进制入口：Enterprise Knowledge System 的 Rust 后端
//! （分阶段绞杀 Python 版）。
use eks_server::config::Config;
use eks_server::routes;
use eks_server::state::AppState;
use sqlx::sqlite::SqliteConnectOptions;
use std::sync::Arc;

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

    // 检索引擎（Milvus Lite 直读 + ONNX embedder + SQLite 父块）
    let embedder = eks_server::embedder::shared_embedder(&config)
        .map_err(|e| format!("Embedding 内核初始化失败: {e}"))?;
    let store = eks_server::milvus_lite::MilvusLiteStore::open(&config.milvus_lite_path);
    let engine = std::sync::Arc::new(eks_server::retrieval_engine::RetrievalEngine::new(
        store, embedder, pool.clone(),
    ));

    tracing::info!(
        "eks-server 启动: {addr} | sqlite={} | prefix={}",
        config.sqlite_path,
        config.api_prefix
    );

    let app = routes::router(AppState {
        pool,
        config: Arc::new(config),
        engine,
    });

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
