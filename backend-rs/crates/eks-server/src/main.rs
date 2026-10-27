//! eks-server: Enterprise Knowledge System 的 Rust 后端（分阶段绞杀 Python 版）。
//! Phase A: 服务骨架 + 与 Python 完全互通的认证域。
mod auth;
mod config;
mod db;
mod milvus_lite;
mod routes;
mod state;
#[cfg(test)]
mod tests;

use sqlx::sqlite::SqliteConnectOptions;
use state::AppState;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = config::Config::load();
    let addr = format!("{}:{}", config.host, config.port);

    // 与 Python 共用同一 SQLite 文件（WAL 模式，读写共存）
    let db_options = SqliteConnectOptions::new()
        .filename(&config.sqlite_path)
        .create_if_missing(true)
        .foreign_keys(true)
        .busy_timeout(std::time::Duration::from_secs(15));
    // journal_mode=WAL
    let db_options = db_options.pragma("journal_mode", "WAL");
    let pool = sqlx::SqlitePool::connect_with(db_options).await?;

    tracing::info!(
        "eks-server 启动: {addr} | sqlite={} | prefix={}",
        config.sqlite_path,
        config.api_prefix
    );

    let app = routes::router(AppState {
        pool,
        config: std::sync::Arc::new(config),
    });

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
