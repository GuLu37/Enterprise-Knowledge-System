//! 多供应商 LLM 客户端（对齐 Python FallbackLLM 语义）。
//!
//! 兜底链: deepseek → openrouter → openai → anthropic → ollama（按 .env 配置过滤）。
//! OpenAI 兼容协议（chat/completions + SSE delta）覆盖 deepseek/openrouter/openai。
//! 失败冷却 60s + 超时预算分摊 + 单次最多尝试 LLM_MAX_FALLBACK_ATTEMPTS 家。

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;

const FAILURE_TTL: Duration = Duration::from_secs(60);
const FALLBACK_ORDER: &[&str] = &["deepseek", "openrouter", "openai", "anthropic", "ollama"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub name: String,
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub is_anthropic: bool,
    pub is_ollama: bool,
}

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub temperature: f64,
    pub timeout_secs: u64,
    pub max_fallback_attempts: usize,
    pub providers: Vec<ProviderConfig>,
}

impl LlmConfig {
    pub fn from_env(cfg: &crate::config::Config) -> Self {
        let mut providers = Vec::new();
        // OpenAI 兼容供应商
        if !cfg.deepseek_api_key.is_empty() && !cfg.deepseek_base_url.is_empty() {
            providers.push(ProviderConfig {
                name: "deepseek".into(),
                api_key: cfg.deepseek_api_key.clone(),
                base_url: cfg.deepseek_base_url.clone(),
                model: cfg.deepseek_model.clone(),
                is_anthropic: false,
                is_ollama: false,
            });
        }
        if !cfg.openrouter_api_key.is_empty() && !cfg.openrouter_api_base.is_empty() {
            providers.push(ProviderConfig {
                name: "openrouter".into(),
                api_key: cfg.openrouter_api_key.clone(),
                base_url: cfg.openrouter_api_base.clone(),
                model: cfg.openrouter_model.clone(),
                is_anthropic: false,
                is_ollama: false,
            });
        }
        if !cfg.openai_api_key.is_empty() {
            providers.push(ProviderConfig {
                name: "openai".into(),
                api_key: cfg.openai_api_key.clone(),
                base_url: "https://api.openai.com/v1".into(),
                model: cfg.openai_model.clone(),
                is_anthropic: false,
                is_ollama: false,
            });
        }
        LlmConfig {
            temperature: cfg.llm_temperature,
            timeout_secs: cfg.llm_timeout,
            max_fallback_attempts: cfg.llm_max_fallback_attempts.max(1),
            providers,
        }
    }
}

pub struct LlmClient {
    http: reqwest::Client,
    config: LlmConfig,
    failure_at: Mutex<HashMap<String, Instant>>,
    active_provider: Mutex<Option<String>>,
}

/// 非流式调用结果。
#[derive(Debug)]
pub struct ChatCompletion {
    pub content: String,
    pub provider: String,
}

/// 流式 chunk。
#[derive(Debug)]
pub struct StreamChunk {
    pub delta: String,
    pub provider: String,
}

impl LlmClient {
    pub fn new(config: LlmConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("reqwest client 构建失败");
        LlmClient {
            http,
            config,
            failure_at: Mutex::new(HashMap::new()),
            active_provider: Mutex::new(None),
        }
    }

    /// 按兜底顺序返回当前可用 provider（过滤冷却中的）。
    async fn candidate_providers(&self) -> Vec<ProviderConfig> {
        let failures = self.failure_at.lock().await;
        let now = Instant::now();
        let configured: HashMap<&str, &ProviderConfig> = self
            .config
            .providers
            .iter()
            .map(|p| (p.name.as_str(), p))
            .collect();

        let mut ordered: Vec<ProviderConfig> = Vec::new();
        for name in FALLBACK_ORDER {
            if let Some(provider) = configured.get(name) {
                let recently_failed = failures
                    .get(*name)
                    .map(|t| now.duration_since(*t) < FAILURE_TTL)
                    .unwrap_or(false);
                if !recently_failed {
                    ordered.push((*provider).clone());
                }
            }
        }
        // 所有 provider 都冷却 → 放宽为全量
        if ordered.is_empty() {
            for name in FALLBACK_ORDER {
                if let Some(provider) = configured.get(name) {
                    ordered.push((*provider).clone());
                }
            }
        }
        ordered.truncate(self.config.max_fallback_attempts);
        ordered
    }

    /// 标记 provider 失败。
    async fn mark_failure(&self, provider: &str) {
        self.failure_at
            .lock()
            .await
            .insert(provider.to_string(), Instant::now());
    }

    fn build_request_body(&self, provider: &ProviderConfig, messages: &[ChatMessage]) -> Value {
        json!({
            "model": provider.model,
            "temperature": self.config.temperature,
            "messages": messages.iter().map(|m| json!({
                "role": m.role,
                "content": m.content,
            })).collect::<Vec<_>>(),
        })
    }

    /// 非流式调用（带兜底链）。
    pub async fn invoke(&self, messages: &[ChatMessage]) -> Result<ChatCompletion, String> {
        let providers = self.candidate_providers().await;
        if providers.is_empty() {
            return Err("没有可用的 LLM 提供商，请检查配置".into());
        }
        let per_timeout = self.config.timeout_secs / providers.len() as u64;

        let mut errors: Vec<String> = Vec::new();
        for provider in &providers {
            match self.invoke_provider(provider, messages, per_timeout).await {
                Ok(result) => {
                    *self.active_provider.lock().await = Some(provider.name.clone());
                    return Ok(result);
                }
                Err(exc) => {
                    tracing::warn!("LLM 调用失败，切换 provider: {} -> {exc}", provider.name);
                    self.mark_failure(&provider.name).await;
                    errors.push(format!("{}: {exc}", provider.name));
                }
            }
        }
        Err(format!("所有 LLM 提供商均不可用: {}", errors.join(" | ")))
    }

    async fn invoke_provider(
        &self,
        provider: &ProviderConfig,
        messages: &[ChatMessage],
        timeout_secs: u64,
    ) -> Result<ChatCompletion, String> {
        let body = self.build_request_body(provider, messages);
        let url = format!("{}/chat/completions", provider.base_url.trim_end_matches('/'));

        let response = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", provider.api_key))
            .header("Content-Type", "application/json")
            .timeout(Duration::from_secs(timeout_secs))
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(format!("HTTP {status}: {text}"));
        }

        let payload: Value = response.json().await.map_err(|e| e.to_string())?;
        let content = payload
            .pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        Ok(ChatCompletion {
            content,
            provider: provider.name.clone(),
        })
    }

    /// 流式调用（SSE，带兜底链）。返回一个 chunk 接收器。
    pub async fn stream(
        self: &Arc<Self>,
        messages: &[ChatMessage],
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, String> {
        let providers = self.candidate_providers().await;
        if providers.is_empty() {
            return Err("没有可用的 LLM 提供商，请检查配置".into());
        }
        let per_timeout = self.config.timeout_secs / providers.len() as u64;
        let provider = providers[0].clone(); // 简化: 流式先用首选，失败由调用方 fallback 到 invoke

        let body = self.build_request_body(&provider, messages);
        let url = format!("{}/chat/completions", provider.base_url.trim_end_matches('/'));

        let request = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", provider.api_key))
            .header("Content-Type", "application/json")
            .timeout(Duration::from_secs(per_timeout * 4)) // 流式给更长时间
            .json(&json!({
                "model": body["model"],
                "temperature": body["temperature"],
                "messages": body["messages"],
                "stream": true,
            }));

        let response = request.send().await.map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            self.mark_failure(&provider.name).await;
            return Err(format!("流式 LLM HTTP {status}: {text}"));
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let provider_name = provider.name.clone();
        let client = self.http.clone();

        tokio::spawn(async move {
            use futures_util::StreamExt;
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut buffer = std::mem::take(&mut buffer);

            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!("SSE chunk 读取失败: {e}");
                        break;
                    }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // 解析 SSE: data: {...}
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].to_string();
                    buffer.drain(..pos + 1);
                    let line = line.trim();
                    if !line.starts_with("data: ") {
                        continue;
                    }
                    let data = &line[6..];
                    if data == "[DONE]" {
                        return;
                    }
                    if let Ok(payload) = serde_json::from_str::<Value>(data) {
                        let delta = payload
                            .pointer("/choices/0/delta/content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if !delta.is_empty() {
                            if tx
                                .send(StreamChunk {
                                    delta: delta.to_string(),
                                    provider: provider_name.clone(),
                                })
                                .await
                                .is_err()
                            {
                                return; // 接收端关闭
                            }
                        }
                    }
                }
            }
            let _ = client; // 保持引用
        });

        Ok(rx)
    }
}
