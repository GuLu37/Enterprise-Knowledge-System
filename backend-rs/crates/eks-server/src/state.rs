//! 共享应用状态。
use std::sync::{Arc, Mutex};

use crate::config::Config;
use crate::retrieval_engine::RetrievalEngine;

/// 对齐 Python `app.state.runtime_warmup`：后台预热快照。
#[derive(Clone, Debug, Default)]
pub struct RuntimeWarmup {
    pub llm_warmed: bool,
    pub embedding_warmed: bool,
    pub cross_encoder_enabled: bool,
    pub cross_encoder_warmed: bool,
    pub cross_encoder_model: Option<String>,
    pub cross_encoder_device: Option<String>,
    pub cross_encoder_error: Option<String>,
    pub provider: Option<String>,
    pub status: String,
    pub error: Option<String>,
}

impl RuntimeWarmup {
    pub fn initializing(config: &Config) -> Self {
        RuntimeWarmup {
            llm_warmed: false,
            embedding_warmed: false,
            cross_encoder_enabled: config.cross_encoder_enabled,
            cross_encoder_warmed: false,
            cross_encoder_model: if config.cross_encoder_onnx.trim().is_empty() {
                None
            } else {
                Some(config.cross_encoder_onnx.clone())
            },
            cross_encoder_device: Some("cpu".into()),
            cross_encoder_error: None,
            provider: None,
            status: "initializing".into(),
            error: None,
        }
    }

    /// Python GET `/chat/runtime-status`：LLM + embedding，且 CE 未启用或已预热。
    pub fn is_ready(&self) -> bool {
        self.llm_warmed
            && self.embedding_warmed
            && (!self.cross_encoder_enabled || self.cross_encoder_warmed)
    }

    pub fn finalize_status(&mut self) {
        self.status = if self.is_ready() {
            "ready".into()
        } else {
            "failed".into()
        };
    }

    /// 对齐 `STARTUP_INIT_LLM`：构造默认 LLM，不发起实际对话。
    pub fn apply_startup_llm(&mut self, config: &Config) -> Result<(), String> {
        let llm = crate::llm::LlmConfig::from_env(config);
        if llm.providers.is_empty() {
            return Err("没有配置任何 LLM 提供商".into());
        }
        let _client = crate::llm::LlmClient::new(llm.clone());
        self.llm_warmed = true;
        self.provider = llm.primary_provider_name();
        Ok(())
    }
}

#[derive(Clone)]
pub struct AppState {
    pub pool: sqlx::SqlitePool,
    pub config: Arc<Config>,
    pub engine: Arc<RetrievalEngine>,
    pub runtime_warmup: Arc<Mutex<RuntimeWarmup>>,
}

impl AppState {
    pub fn store_warmup(&self, snapshot: RuntimeWarmup) {
        match self.runtime_warmup.lock() {
            Ok(mut guard) => *guard = snapshot,
            Err(poisoned) => *poisoned.into_inner() = snapshot,
        }
    }

    pub fn warmup_snapshot(&self) -> RuntimeWarmup {
        match self.runtime_warmup.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }
}
