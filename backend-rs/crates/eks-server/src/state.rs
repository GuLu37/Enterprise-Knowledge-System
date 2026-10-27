//! 共享应用状态。
use std::sync::Arc;

use crate::config::Config;
use crate::retrieval_engine::RetrievalEngine;

#[derive(Clone)]
pub struct AppState {
    pub pool: sqlx::SqlitePool,
    pub config: Arc<Config>,
    pub engine: Arc<RetrievalEngine>,
}
