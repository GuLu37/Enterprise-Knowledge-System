//! 多供应商 LLM 客户端（对齐 Python FallbackLLM 语义）。
//!
//! 兜底链: preferred → deepseek → openrouter → openai → anthropic → ollama（按 .env 配置过滤）。
//! OpenAI 兼容协议覆盖 deepseek/openrouter/openai；Anthropic 走 Messages API；Ollama 走 /api/chat。
//! 失败冷却 60s + 超时预算分摊 + 单次最多尝试 LLM_MAX_FALLBACK_ATTEMPTS 家。

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;

const FAILURE_TTL: Duration = Duration::from_secs(60);
const FALLBACK_ORDER: &[&str] = &["deepseek", "openrouter", "openai", "anthropic", "ollama"];

/// 对齐 Python `FallbackLLM.prime` / `LLMException` 无可用 provider 文案。
pub const LLM_NO_PROVIDER_MESSAGE: &str = "没有可用的 LLM 提供商，请检查配置: 无可用 provider";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn to_api_json(&self) -> Value {
        let mut object = serde_json::Map::new();
        object.insert("role".into(), json!(self.role));
        object.insert("content".into(), json!(self.content));
        if let Some(calls) = &self.tool_calls {
            object.insert("tool_calls".into(), calls.clone());
        }
        if let Some(id) = &self.tool_call_id {
            object.insert("tool_call_id".into(), json!(id));
        }
        Value::Object(object)
    }
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    pub raw: Value,
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
    pub preferred_provider: String,
    pub temperature: f64,
    pub timeout_secs: u64,
    pub max_tokens: u32,
    pub max_fallback_attempts: usize,
    pub providers: Vec<ProviderConfig>,
}

impl LlmConfig {
    pub fn from_env(cfg: &crate::config::Config) -> Self {
        Self::from_env_with_overrides(cfg, None, None, None)
    }

    pub fn from_env_with_overrides(
        cfg: &crate::config::Config,
        provider: Option<&str>,
        model: Option<&str>,
        temperature: Option<f64>,
    ) -> Self {
        let preferred = normalize_provider(provider).unwrap_or_else(|| normalize_provider(Some(&cfg.llm_provider)).unwrap_or_default());
        let model_override = model.map(str::trim).filter(|item| !item.is_empty());
        let mut providers = Vec::new();
        // 对齐 Python `_provider_is_configured`：缺 model/base_url 时不把 provider 当可用。
        if !cfg.deepseek_api_key.is_empty() && !cfg.deepseek_base_url.trim().is_empty() {
            if let Some(model) = resolved_provider_model(model_override, &cfg.deepseek_model) {
                providers.push(openai_compat_provider(
                    "deepseek",
                    &cfg.deepseek_api_key,
                    cfg.deepseek_base_url.trim(),
                    model,
                ));
            }
        }
        if !cfg.openrouter_api_key.is_empty() && !cfg.openrouter_api_base.trim().is_empty() {
            if let Some(model) = resolved_provider_model(model_override, &cfg.openrouter_model) {
                providers.push(openai_compat_provider(
                    "openrouter",
                    &cfg.openrouter_api_key,
                    cfg.openrouter_api_base.trim(),
                    model,
                ));
            }
        }
        if !cfg.openai_api_key.is_empty() {
            if let Some(model) = resolved_provider_model(model_override, &cfg.openai_model) {
                providers.push(openai_compat_provider(
                    "openai",
                    &cfg.openai_api_key,
                    openai_base_url(&cfg.openai_base_url),
                    model,
                ));
            }
        }
        if !cfg.anthropic_api_key.is_empty() {
            providers.push(ProviderConfig {
                name: "anthropic".into(),
                api_key: cfg.anthropic_api_key.clone(),
                base_url: "https://api.anthropic.com".into(),
                model: model_override.unwrap_or(&cfg.anthropic_model).to_string(),
                is_anthropic: true,
                is_ollama: false,
            });
        }
        if !cfg.ollama_base_url.is_empty() && !cfg.ollama_model.is_empty() {
            providers.push(ProviderConfig {
                name: "ollama".into(),
                api_key: String::new(),
                base_url: cfg.ollama_base_url.clone(),
                model: model_override.unwrap_or(&cfg.ollama_model).to_string(),
                is_anthropic: false,
                is_ollama: true,
            });
        }
        LlmConfig {
            preferred_provider: preferred,
            temperature: temperature.unwrap_or(cfg.llm_temperature),
            timeout_secs: cfg.llm_timeout,
            max_tokens: cfg.llm_max_tokens.max(1),
            max_fallback_attempts: cfg.llm_max_fallback_attempts.max(1),
            providers,
        }
    }

    /// 按 preferred → 兜底链选出第一个已配置的供应商名称。
    pub fn primary_provider_name(&self) -> Option<String> {
        for name in build_provider_order(&self.preferred_provider) {
            if self.providers.iter().any(|item| item.name == name) {
                return Some(name);
            }
        }
        self.providers.first().map(|item| item.name.clone())
    }
}

fn resolved_provider_model<'a>(model_override: Option<&'a str>, configured: &'a str) -> Option<&'a str> {
    model_override.or_else(|| {
        let trimmed = configured.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

fn openai_base_url(configured: &str) -> &str {
    let trimmed = configured.trim();
    if trimmed.is_empty() {
        "https://api.openai.com/v1"
    } else {
        trimmed
    }
}

fn openai_compat_provider(name: &str, api_key: &str, base_url: &str, model: &str) -> ProviderConfig {
    ProviderConfig {
        name: name.into(),
        api_key: api_key.to_string(),
        base_url: base_url.to_string(),
        model: model.to_string(),
        is_anthropic: false,
        is_ollama: false,
    }
}

fn normalize_provider(provider: Option<&str>) -> Option<String> {
    let value = provider.map(str::trim).filter(|item| !item.is_empty())?.to_lowercase();
    Some(value)
}

fn build_provider_order(preferred: &str) -> Vec<String> {
    let mut ordered = Vec::new();
    let preferred = preferred.trim().to_lowercase();
    if FALLBACK_ORDER.iter().any(|name| *name == preferred) {
        ordered.push(preferred);
    }
    for name in FALLBACK_ORDER {
        if !ordered.iter().any(|item| item == name) {
            ordered.push((*name).to_string());
        }
    }
    ordered
}

/// 对齐 Python `_fallback_call`：若粘滞供应商仍在候选里，提到队首继续用。
fn promote_active_provider(providers: &mut Vec<ProviderConfig>, active: Option<&str>) {
    let Some(active) = active.map(str::trim).filter(|name| !name.is_empty()) else {
        return;
    };
    if let Some(index) = providers.iter().position(|item| item.name == active) {
        if index > 0 {
            let provider = providers.remove(index);
            providers.insert(0, provider);
        }
    }
}

/// 对齐 Python `FallbackLLM._attempt_timeout`：把总超时整除分摊到本次候选数。
fn attempt_timeout(total_timeout_secs: u64, attempt_count: usize) -> u64 {
    let total = total_timeout_secs.max(1);
    let count = attempt_count.max(1) as u64;
    (total / count).max(1)
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
    pub model: String,
    pub tool_calls: Vec<ToolCall>,
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

    /// 对齐 Python `get_default_llm()`：进程级默认实例。
    pub fn from_env(cfg: &crate::config::Config) -> Arc<Self> {
        Arc::new(Self::new(LlmConfig::from_env(cfg)))
    }

    pub fn from_env_with_overrides(
        cfg: &crate::config::Config,
        provider: Option<&str>,
        model: Option<&str>,
        temperature: Option<f64>,
    ) -> Arc<Self> {
        Arc::new(Self::new(LlmConfig::from_env_with_overrides(
            cfg,
            provider,
            model,
            temperature,
        )))
    }

    pub fn has_configured_providers(&self) -> bool {
        !self.config.providers.is_empty()
    }

    /// 对齐 Python `FallbackLLM.active_provider`。
    pub async fn active_provider_name(&self) -> Option<String> {
        self.active_provider.lock().await.clone()
    }

    /// 对齐 Python `FallbackLLM.prime`：预先挑选当前可用 provider 并粘滞到进程级实例。
    pub async fn prime(&self) -> Result<String, String> {
        let providers = self.candidate_providers().await;
        if providers.is_empty() {
            return Err(LLM_NO_PROVIDER_MESSAGE.into());
        }
        let name = providers[0].name.clone();
        self.remember_success(&name).await;
        Ok(name)
    }

    /// 按兜底顺序返回当前可用 provider（过滤冷却中的）。
    /// 对齐 Python `_fallback_call` / `stream`：粘滞 `active_provider` 提到队首。
    async fn candidate_providers(&self) -> Vec<ProviderConfig> {
        let mut ordered = {
            let failures = self.failure_at.lock().await;
            let now = Instant::now();
            let configured: HashMap<&str, &ProviderConfig> = self
                .config
                .providers
                .iter()
                .map(|p| (p.name.as_str(), p))
                .collect();

            let order = build_provider_order(&self.config.preferred_provider);
            let mut ordered: Vec<ProviderConfig> = Vec::new();
            for name in &order {
                if let Some(provider) = configured.get(name.as_str()) {
                    let recently_failed = failures
                        .get(name.as_str())
                        .map(|t| now.duration_since(*t) < FAILURE_TTL)
                        .unwrap_or(false);
                    let skip_cooldown = name == &self.config.preferred_provider;
                    if !recently_failed || skip_cooldown {
                        ordered.push((*provider).clone());
                    }
                }
            }
            if ordered.is_empty() {
                for name in &order {
                    if let Some(provider) = configured.get(name.as_str()) {
                        ordered.push((*provider).clone());
                    }
                }
            }
            ordered
        };
        ordered.truncate(self.config.max_fallback_attempts);
        let active = self.active_provider.lock().await.clone();
        promote_active_provider(&mut ordered, active.as_deref());
        ordered
    }

    /// 标记 provider 失败。
    async fn mark_failure(&self, provider: &str) {
        self.failure_at
            .lock()
            .await
            .insert(provider.to_string(), Instant::now());
    }

    async fn remember_success(&self, provider: &str) {
        *self.active_provider.lock().await = Some(provider.to_string());
    }

    /// 对齐 Python：失败时冷却，并在该供应商正是粘滞项时清空。
    async fn record_failure(&self, provider: &str) {
        self.mark_failure(provider).await;
        let mut active = self.active_provider.lock().await;
        if active.as_deref() == Some(provider) {
            *active = None;
        }
    }

    fn build_request_body(
        &self,
        provider: &ProviderConfig,
        messages: &[ChatMessage],
        tools: Option<&[Value]>,
    ) -> Value {
        let mut body = json!({
            "model": provider.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": messages.iter().map(ChatMessage::to_api_json).collect::<Vec<_>>(),
        });
        if let Some(tools) = tools {
            if !tools.is_empty() {
                body["tools"] = json!(tools);
                body["tool_choice"] = json!("auto");
            }
        }
        body
    }

    /// 非流式调用（带兜底链）。
    pub async fn invoke(&self, messages: &[ChatMessage]) -> Result<ChatCompletion, String> {
        let providers = self.candidate_providers().await;
        if providers.is_empty() {
            return Err("没有可用的 LLM 提供商，请检查配置".into());
        }
        let per_timeout = attempt_timeout(self.config.timeout_secs, providers.len());

        let mut errors: Vec<String> = Vec::new();
        for provider in &providers {
            match self.invoke_provider(provider, messages, None, per_timeout).await {
                Ok(result) => {
                    self.remember_success(&provider.name).await;
                    return Ok(result);
                }
                Err(exc) => {
                    tracing::warn!("LLM 调用失败，切换 provider: {} -> {exc}", provider.name);
                    self.record_failure(&provider.name).await;
                    errors.push(format!("{}: {exc}", provider.name));
                }
            }
        }
        Err(format!("所有 LLM 提供商均不可用: {}", errors.join(" | ")))
    }

    /// 绑定工具后的非流式调用。供应商不支持 tools 时返回错误，由调用方降级。
    pub async fn invoke_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[Value],
    ) -> Result<ChatCompletion, String> {
        let providers = self.candidate_providers().await;
        if providers.is_empty() {
            return Err("没有可用的 LLM 提供商，请检查配置".into());
        }
        let per_timeout = attempt_timeout(self.config.timeout_secs, providers.len());
        let mut errors: Vec<String> = Vec::new();
        for provider in &providers {
            match self
                .invoke_provider(provider, messages, Some(tools), per_timeout)
                .await
            {
                Ok(result) => {
                    self.remember_success(&provider.name).await;
                    return Ok(result);
                }
                Err(exc) => {
                    tracing::warn!("带工具 LLM 调用失败: {} -> {exc}", provider.name);
                    self.record_failure(&provider.name).await;
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
        tools: Option<&[Value]>,
        timeout_secs: u64,
    ) -> Result<ChatCompletion, String> {
        if provider.is_anthropic {
            return self.invoke_anthropic(provider, messages, tools, timeout_secs).await;
        }
        if provider.is_ollama {
            return self.invoke_ollama(provider, messages, tools, timeout_secs).await;
        }
        let body = self.build_request_body(provider, messages, tools);
        let url = format!("{}/chat/completions", provider.base_url.trim_end_matches('/'));
        let payload = self
            .post_json(&url, Some(&provider.api_key), None, &body, timeout_secs)
            .await?;
        completion_from_openai(provider, payload)
    }

    async fn invoke_anthropic(
        &self,
        provider: &ProviderConfig,
        messages: &[ChatMessage],
        tools: Option<&[Value]>,
        timeout_secs: u64,
    ) -> Result<ChatCompletion, String> {
        let body = build_anthropic_body(
            provider,
            messages,
            tools,
            self.config.temperature,
            self.config.max_tokens,
            false,
        );
        let url = format!("{}/v1/messages", provider.base_url.trim_end_matches('/'));
        let payload = self
            .post_json(
                &url,
                None,
                Some(&provider.api_key),
                &body,
                timeout_secs,
            )
            .await?;
        completion_from_anthropic(provider, payload)
    }

    async fn invoke_ollama(
        &self,
        provider: &ProviderConfig,
        messages: &[ChatMessage],
        tools: Option<&[Value]>,
        timeout_secs: u64,
    ) -> Result<ChatCompletion, String> {
        let body = build_ollama_body(
            provider,
            messages,
            tools,
            self.config.temperature,
            self.config.max_tokens,
            false,
        );
        let url = format!("{}/api/chat", provider.base_url.trim_end_matches('/'));
        let payload = self.post_json(&url, None, None, &body, timeout_secs).await?;
        completion_from_ollama(provider, payload)
    }

    async fn post_json(
        &self,
        url: &str,
        bearer: Option<&str>,
        anthropic_key: Option<&str>,
        body: &Value,
        timeout_secs: u64,
    ) -> Result<Value, String> {
        let mut request = self
            .http
            .post(url)
            .header("Content-Type", "application/json")
            .timeout(Duration::from_secs(timeout_secs.max(1)))
            .json(body);
        if let Some(key) = bearer.filter(|item| !item.is_empty()) {
            request = request.header("Authorization", format!("Bearer {key}"));
        }
        if let Some(key) = anthropic_key.filter(|item| !item.is_empty()) {
            request = request
                .header("x-api-key", key)
                .header("anthropic-version", "2023-06-01");
        }
        let response = request.send().await.map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(format!("HTTP {status}: {text}"));
        }
        response.json().await.map_err(|e| e.to_string())
    }

    /// 流式调用（SSE，带兜底链）。HTTP/连接失败时切换下一供应商，对齐 Python `FallbackLLM.stream`。
    pub async fn stream(
        self: &Arc<Self>,
        messages: &[ChatMessage],
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, String> {
        let providers = self.candidate_providers().await;
        if providers.is_empty() {
            return Err("没有可用的 LLM 提供商，请检查配置".into());
        }
        let per_timeout = attempt_timeout(self.config.timeout_secs, providers.len());
        let mut errors: Vec<String> = Vec::new();
        for provider in &providers {
            match self.open_stream_response(provider, messages, per_timeout).await {
                Ok(response) => {
                    self.remember_success(&provider.name).await;
                    return Ok(spawn_stream_reader(response, provider));
                }
                Err(exc) => {
                    tracing::warn!(
                        "LLM 流式调用失败，切换到下一个 provider: {} -> {exc}",
                        provider.name
                    );
                    self.record_failure(&provider.name).await;
                    errors.push(format!("{}: {exc}", provider.name));
                }
            }
        }
        Err(format!(
            "所有 LLM 提供商均不可用: {}",
            if errors.is_empty() {
                "无可用 provider".to_string()
            } else {
                errors.join(" | ")
            }
        ))
    }

    async fn open_stream_response(
        &self,
        provider: &ProviderConfig,
        messages: &[ChatMessage],
        timeout_secs: u64,
    ) -> Result<reqwest::Response, String> {
        let (url, body, bearer, anthropic_key) = if provider.is_anthropic {
            (
                format!("{}/v1/messages", provider.base_url.trim_end_matches('/')),
                build_anthropic_body(
                    provider,
                    messages,
                    None,
                    self.config.temperature,
                    self.config.max_tokens,
                    true,
                ),
                None,
                Some(provider.api_key.clone()),
            )
        } else if provider.is_ollama {
            (
                format!("{}/api/chat", provider.base_url.trim_end_matches('/')),
                build_ollama_body(
                    provider,
                    messages,
                    None,
                    self.config.temperature,
                    self.config.max_tokens,
                    true,
                ),
                None,
                None,
            )
        } else {
            let mut body = self.build_request_body(provider, messages, None);
            body["stream"] = json!(true);
            (
                format!("{}/chat/completions", provider.base_url.trim_end_matches('/')),
                body,
                Some(provider.api_key.clone()),
                None,
            )
        };

        let mut request = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .timeout(Duration::from_secs(timeout_secs.max(1)))
            .json(&body);
        if let Some(key) = bearer.filter(|item| !item.is_empty()) {
            request = request.header("Authorization", format!("Bearer {key}"));
        }
        if let Some(key) = anthropic_key.filter(|item| !item.is_empty()) {
            request = request
                .header("x-api-key", key)
                .header("anthropic-version", "2023-06-01");
        }

        let response = request.send().await.map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(format!("流式 LLM HTTP {status}: {text}"));
        }
        Ok(response)
    }
}

/// 对齐 Python `get_default_llm` / `get_llm`：无 provider/model/temperature 覆盖时复用进程级实例。
pub fn resolve_llm_client(
    default_llm: &Arc<LlmClient>,
    cfg: &crate::config::Config,
    provider: Option<&str>,
    model: Option<&str>,
    temperature: Option<f64>,
) -> Arc<LlmClient> {
    if provider.is_none() && model.is_none() && temperature.is_none() {
        default_llm.clone()
    } else {
        LlmClient::from_env_with_overrides(cfg, provider, model, temperature)
    }
}

fn spawn_stream_reader(
    response: reqwest::Response,
    provider: &ProviderConfig,
) -> tokio::sync::mpsc::Receiver<StreamChunk> {
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    let provider_name = provider.name.clone();
    let is_ollama = provider.is_ollama;
    let is_anthropic = provider.is_anthropic;

    tokio::spawn(async move {
        use futures_util::StreamExt;
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("SSE chunk 读取失败: {e}");
                    break;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].to_string();
                buffer.drain(..pos + 1);
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let delta = if is_ollama {
                    stream_delta_from_ollama(line)
                } else if is_anthropic {
                    stream_delta_from_anthropic(line)
                } else {
                    stream_delta_from_openai(line)
                };
                if let Some(delta) = delta {
                    if tx
                        .send(StreamChunk {
                            delta,
                            provider: provider_name.clone(),
                        })
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            }
        }
    });

    rx
}

fn json_line(line: &str) -> Option<&str> {
    let line = line.trim();
    if line.is_empty() || line.starts_with(':') || line.starts_with("event:") {
        return None;
    }
    let payload = line.strip_prefix("data:").map(str::trim).unwrap_or(line);
    if payload.is_empty() || payload == "[DONE]" {
        return None;
    }
    Some(payload)
}

fn openai_tool_to_anthropic(tool: &Value) -> Option<Value> {
    if tool.get("name").and_then(Value::as_str).is_some() && tool.get("input_schema").is_some() {
        return Some(tool.clone());
    }
    let function = tool.get("function").unwrap_or(tool);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if name.is_empty() {
        return None;
    }
    let description = function.get("description").cloned().unwrap_or(json!(""));
    let input_schema = function
        .get("parameters")
        .cloned()
        .or_else(|| function.get("input_schema").cloned())
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
    Some(json!({
        "name": name,
        "description": description,
        "input_schema": input_schema,
    }))
}

fn anthropic_tools(tools: Option<&[Value]>) -> Vec<Value> {
    tools
        .unwrap_or(&[])
        .iter()
        .filter_map(openai_tool_to_anthropic)
        .collect()
}

fn parse_json_object(value: &Value) -> Value {
    if let Some(text) = value.as_str() {
        serde_json::from_str::<Value>(text).unwrap_or_else(|_| json!({}))
    } else {
        value.clone()
    }
}

fn anthropic_tool_use_from_openai(item: &Value) -> Option<Value> {
    let name = item
        .pointer("/function/name")
        .and_then(Value::as_str)
        .or_else(|| item.get("name").and_then(Value::as_str))
        .unwrap_or("")
        .trim();
    if name.is_empty() {
        return None;
    }
    let id = item
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("call_{name}"));
    let input = item
        .pointer("/function/arguments")
        .or_else(|| item.get("input"))
        .map(parse_json_object)
        .unwrap_or_else(|| json!({}));
    Some(json!({
        "type": "tool_use",
        "id": id,
        "name": name,
        "input": input,
    }))
}

fn anthropic_assistant_message(message: &ChatMessage) -> Value {
    let mut blocks = Vec::new();
    if !message.content.is_empty() {
        blocks.push(json!({"type": "text", "text": message.content}));
    }
    if let Some(Value::Array(items)) = &message.tool_calls {
        for item in items {
            if let Some(block) = anthropic_tool_use_from_openai(item) {
                blocks.push(block);
            }
        }
    }
    if blocks.is_empty() {
        json!({"role": "assistant", "content": message.content})
    } else {
        json!({"role": "assistant", "content": blocks})
    }
}

fn anthropic_tool_result(message: &ChatMessage) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": message.tool_call_id.clone().unwrap_or_default(),
        "content": message.content,
    })
}

fn last_user_is_tool_results(messages: &[Value]) -> bool {
    let Some(Value::Object(object)) = messages.last() else {
        return false;
    };
    if object.get("role").and_then(Value::as_str) != Some("user") {
        return false;
    }
    match object.get("content") {
        Some(Value::Array(items)) => items
            .iter()
            .all(|item| item.get("type").and_then(Value::as_str) == Some("tool_result")),
        _ => false,
    }
}

fn split_anthropic_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
    let mut system_parts = Vec::new();
    let mut converted = Vec::new();
    for message in messages {
        match message.role.as_str() {
            "system" => {
                let text = message.content.trim();
                if !text.is_empty() {
                    system_parts.push(text.to_string());
                }
            }
            "tool" => {
                let result = anthropic_tool_result(message);
                if last_user_is_tool_results(&converted) {
                    if let Some(Value::Object(object)) = converted.last_mut() {
                        if let Some(Value::Array(content)) = object.get_mut("content") {
                            content.push(result);
                            continue;
                        }
                    }
                }
                converted.push(json!({
                    "role": "user",
                    "content": [result],
                }));
            }
            "assistant" => converted.push(anthropic_assistant_message(message)),
            _ => converted.push(json!({
                "role": "user",
                "content": message.content,
            })),
        }
    }
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    (system, converted)
}

fn build_anthropic_body(
    provider: &ProviderConfig,
    messages: &[ChatMessage],
    tools: Option<&[Value]>,
    temperature: f64,
    max_tokens: u32,
    stream: bool,
) -> Value {
    let (system, converted) = split_anthropic_messages(messages);
    let mut body = json!({
        "model": provider.model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": converted,
    });
    if let Some(system) = system {
        body["system"] = json!(system);
    }
    let converted_tools = anthropic_tools(tools);
    if !converted_tools.is_empty() {
        body["tools"] = json!(converted_tools);
    }
    if stream {
        body["stream"] = json!(true);
    }
    body
}

fn normalize_ollama_tool_call(item: &Value) -> Value {
    let mut mapped = item.clone();
    let Some(text) = mapped
        .pointer("/function/arguments")
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return mapped;
    };
    if let Some(args) = mapped.pointer_mut("/function/arguments") {
        *args = serde_json::from_str::<Value>(&text).unwrap_or_else(|_| json!({}));
    }
    mapped
}

fn ollama_message(message: &ChatMessage) -> Value {
    let mut object = serde_json::Map::new();
    object.insert("role".into(), json!(message.role));
    object.insert("content".into(), json!(message.content));
    if let Some(calls) = &message.tool_calls {
        let normalized = match calls {
            Value::Array(items) => Value::Array(items.iter().map(normalize_ollama_tool_call).collect()),
            other => other.clone(),
        };
        object.insert("tool_calls".into(), normalized);
    }
    if let Some(id) = &message.tool_call_id {
        object.insert("tool_call_id".into(), json!(id));
    }
    Value::Object(object)
}

fn build_ollama_body(
    provider: &ProviderConfig,
    messages: &[ChatMessage],
    tools: Option<&[Value]>,
    temperature: f64,
    max_tokens: u32,
    stream: bool,
) -> Value {
    let mut body = json!({
        "model": provider.model,
        "messages": messages.iter().map(ollama_message).collect::<Vec<_>>(),
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    });
    if let Some(tools) = tools {
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
    }
    body
}

fn message_content(value: &Value) -> String {
    match value.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.as_str())
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn completion_from_openai(provider: &ProviderConfig, payload: Value) -> Result<ChatCompletion, String> {
    let message = payload.pointer("/choices/0/message").cloned().unwrap_or(json!({}));
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&provider.model)
        .to_string();
    Ok(ChatCompletion {
        content: message_content(&message),
        provider: provider.name.clone(),
        model,
        tool_calls: parse_tool_calls(message.get("tool_calls")),
    })
}

fn tool_call_from_anthropic(block: &Value) -> Option<ToolCall> {
    let name = block
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if name.is_empty() {
        return None;
    }
    let id = block
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("call_{name}"));
    let arguments = block.get("input").cloned().unwrap_or_else(|| json!({}));
    let raw = json!({
        "id": id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".into()),
        }
    });
    Some(ToolCall {
        id,
        name,
        arguments,
        raw,
    })
}

fn completion_from_anthropic(provider: &ProviderConfig, payload: Value) -> Result<ChatCompletion, String> {
    let mut texts = Vec::new();
    let mut tool_calls = Vec::new();
    match payload.get("content") {
        Some(Value::String(text)) => texts.push(text.clone()),
        Some(Value::Array(blocks)) => {
            for block in blocks {
                match block.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text" => {
                        if let Some(text) = block.get("text").and_then(Value::as_str) {
                            if !text.is_empty() {
                                texts.push(text.to_string());
                            }
                        }
                    }
                    "tool_use" => {
                        if let Some(call) = tool_call_from_anthropic(block) {
                            tool_calls.push(call);
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&provider.model)
        .to_string();
    Ok(ChatCompletion {
        content: texts.join(""),
        provider: provider.name.clone(),
        model,
        tool_calls,
    })
}

fn completion_from_ollama(provider: &ProviderConfig, payload: Value) -> Result<ChatCompletion, String> {
    let message = payload.get("message").cloned().unwrap_or(json!({}));
    let model = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&provider.model)
        .to_string();
    Ok(ChatCompletion {
        content: message_content(&message),
        provider: provider.name.clone(),
        model,
        tool_calls: parse_tool_calls(message.get("tool_calls")),
    })
}

fn stream_delta_from_openai(line: &str) -> Option<String> {
    let payload: Value = serde_json::from_str(json_line(line)?).ok()?;
    payload
        .pointer("/choices/0/delta/content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

fn stream_delta_from_anthropic(line: &str) -> Option<String> {
    let payload: Value = serde_json::from_str(json_line(line)?).ok()?;
    if payload.get("type").and_then(Value::as_str) == Some("content_block_delta") {
        return payload
            .pointer("/delta/text")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
            .map(str::to_string);
    }
    payload
        .pointer("/delta/text")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

fn stream_delta_from_ollama(line: &str) -> Option<String> {
    let payload: Value = serde_json::from_str(json_line(line)?).ok()?;
    payload
        .pointer("/message/content")
        .and_then(Value::as_str)
        .or_else(|| payload.get("response").and_then(Value::as_str))
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

fn parse_tool_calls(value: Option<&Value>) -> Vec<ToolCall> {
    let Some(Value::Array(items)) = value else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let id = item
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name = item
                .pointer("/function/name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if name.is_empty() {
                return None;
            }
            let arguments = item
                .pointer("/function/arguments")
                .and_then(|v| {
                    if let Some(text) = v.as_str() {
                        serde_json::from_str::<Value>(text).ok()
                    } else {
                        Some(v.clone())
                    }
                })
                .unwrap_or_else(|| json!({}));
            Some(ToolCall {
                id: if id.is_empty() {
                    format!("call_{name}")
                } else {
                    id
                },
                name,
                arguments,
                raw: item.clone(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_tool_calls() {
        let payload = json!([{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "arguments": "{\"query\":\"报销\"}"
            }
        }]);
        let calls = parse_tool_calls(Some(&payload));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search_knowledge_base");
        assert_eq!(calls[0].arguments["query"], "报销");
        assert_eq!(calls[0].id, "call_1");
    }

    #[test]
    fn chat_message_omits_empty_tool_fields() {
        let message = ChatMessage::new("user", "你好");
        let encoded = message.to_api_json();
        assert_eq!(encoded["role"], "user");
        assert_eq!(encoded["content"], "你好");
        assert!(encoded.get("tool_calls").is_none());
        assert!(encoded.get("tool_call_id").is_none());
    }

    fn demo_provider(name: &str, anthropic: bool, ollama: bool) -> ProviderConfig {
        ProviderConfig {
            name: name.into(),
            api_key: "k".into(),
            base_url: if ollama {
                "http://127.0.0.1:11434".into()
            } else {
                "https://api.example.com".into()
            },
            model: "demo".into(),
            is_anthropic: anthropic,
            is_ollama: ollama,
        }
    }

    #[test]
    fn provider_requires_python_configured_fields() {
        let mut cfg = crate::config::Config::jwt_test();
        cfg.openai_api_key = "sk-test".into();
        assert!(LlmConfig::from_env(&cfg).providers.is_empty());
        cfg.openai_model = "gpt-4o-mini".into();
        let openai = LlmConfig::from_env(&cfg);
        assert_eq!(openai.providers.len(), 1);
        assert_eq!(openai.providers[0].name, "openai");
        assert_eq!(openai.providers[0].model, "gpt-4o-mini");
        assert_eq!(openai.providers[0].base_url, "https://api.openai.com/v1");

        cfg.openai_api_key.clear();
        cfg.openai_model.clear();
        cfg.deepseek_api_key = "ds-test".into();
        assert!(LlmConfig::from_env(&cfg).providers.is_empty());
        cfg.deepseek_base_url = "https://api.deepseek.com".into();
        assert!(LlmConfig::from_env(&cfg).providers.is_empty());
        cfg.deepseek_model = "deepseek-chat".into();
        assert_eq!(LlmConfig::from_env(&cfg).providers[0].name, "deepseek");

        let override_cfg = crate::config::Config::jwt_test();
        let mut override_cfg = override_cfg;
        override_cfg.openai_api_key = "sk-test".into();
        let with_request_model =
            LlmConfig::from_env_with_overrides(&override_cfg, None, Some("gpt-4o"), None);
        assert_eq!(with_request_model.providers[0].model, "gpt-4o");
    }

    #[test]
    fn preferred_provider_leads_fallback_order() {
        assert_eq!(
            build_provider_order("anthropic"),
            vec!["anthropic", "deepseek", "openrouter", "openai", "ollama"]
        );
        assert_eq!(
            build_provider_order("unknown"),
            vec!["deepseek", "openrouter", "openai", "anthropic", "ollama"]
        );
    }

    #[test]
    fn request_overrides_preferred_model_and_temperature() {
        let mut cfg = crate::config::Config::jwt_test();
        cfg.openai_api_key = "sk-test".into();
        cfg.openai_model = "gpt-4o-mini".into();
        cfg.anthropic_api_key = "ak-test".into();
        cfg.ollama_base_url = "http://127.0.0.1:11434".into();
        cfg.ollama_model = "qwen".into();
        cfg.llm_provider = "openai".into();
        cfg.llm_temperature = 0.7;

        let llm = LlmConfig::from_env_with_overrides(
            &cfg,
            Some("Anthropic"),
            Some("claude-sonnet-4-6"),
            Some(0.2),
        );
        assert_eq!(llm.preferred_provider, "anthropic");
        assert_eq!(llm.temperature, 0.2);
        assert_eq!(llm.primary_provider_name().as_deref(), Some("anthropic"));
        assert!(llm.providers.iter().all(|item| item.model == "claude-sonnet-4-6"));
        assert!(llm.providers.iter().any(|item| item.is_anthropic));
        assert!(llm.providers.iter().any(|item| item.is_ollama));
    }

    #[test]
    fn anthropic_body_extracts_system_and_converts_tools() {
        let provider = demo_provider("anthropic", true, false);
        let messages = vec![
            ChatMessage::new("system", "你是助手"),
            ChatMessage::new("user", "查报销"),
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: Some(json!([{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_knowledge_base",
                        "arguments": "{\"query\":\"报销\"}"
                    }
                }])),
                tool_call_id: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: "{\"ok\":true}".into(),
                tool_calls: None,
                tool_call_id: Some("call_1".into()),
            },
        ];
        let tools = [json!({
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "检索",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        })];
        let body = build_anthropic_body(&provider, &messages, Some(&tools), 0.3, 1024, true);
        assert_eq!(body["model"], "demo");
        assert_eq!(body["system"], "你是助手");
        assert_eq!(body["max_tokens"], 1024);
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(body["messages"][1]["content"][0]["input"]["query"], "报销");
        assert_eq!(body["messages"][2]["role"], "user");
        assert_eq!(body["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(body["tools"][0]["name"], "search_knowledge_base");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn ollama_body_uses_native_chat_options() {
        let provider = demo_provider("ollama", false, true);
        let messages = vec![
            ChatMessage::new("system", "你是助手"),
            ChatMessage {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: Some(json!([{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_knowledge_base",
                        "arguments": "{\"query\":\"报销\"}"
                    }
                }])),
                tool_call_id: None,
            },
        ];
        let body = build_ollama_body(&provider, &messages, None, 0.4, 2048, false);
        assert_eq!(body["model"], "demo");
        assert_eq!(body["stream"], false);
        assert_eq!(body["options"]["temperature"], 0.4);
        assert_eq!(body["options"]["num_predict"], 2048);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["arguments"]["query"],
            "报销"
        );
    }

    #[test]
    fn parses_anthropic_and_ollama_completions() {
        let anthropic = demo_provider("anthropic", true, false);
        let parsed = completion_from_anthropic(
            &anthropic,
            json!({
                "model": "claude-sonnet-4-6",
                "content": [
                    {"type": "text", "text": "先查知识库"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "search_knowledge_base",
                        "input": {"query": "报销"}
                    }
                ]
            }),
        )
        .unwrap();
        assert_eq!(parsed.content, "先查知识库");
        assert_eq!(parsed.model, "claude-sonnet-4-6");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].id, "toolu_1");
        assert_eq!(parsed.tool_calls[0].arguments["query"], "报销");
        assert_eq!(parsed.tool_calls[0].raw["function"]["name"], "search_knowledge_base");

        let ollama = demo_provider("ollama", false, true);
        let parsed = completion_from_ollama(
            &ollama,
            json!({
                "model": "qwen",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "search_knowledge_base",
                            "arguments": {"query": "差旅"}
                        }
                    }]
                }
            }),
        )
        .unwrap();
        assert_eq!(parsed.model, "qwen");
        assert_eq!(parsed.tool_calls[0].name, "search_knowledge_base");
        assert_eq!(parsed.tool_calls[0].arguments["query"], "差旅");
    }

    #[test]
    fn parses_provider_stream_deltas() {
        assert_eq!(
            stream_delta_from_openai(r#"data: {"choices":[{"delta":{"content":"你好"}}]}"#).as_deref(),
            Some("你好")
        );
        assert_eq!(stream_delta_from_openai("data: [DONE]"), None);
        assert_eq!(
            stream_delta_from_anthropic(
                r#"data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}"#
            )
            .as_deref(),
            Some("Hello")
        );
        assert_eq!(
            stream_delta_from_ollama(r#"{"message":{"role":"assistant","content":"世界"},"done":false}"#)
                .as_deref(),
            Some("世界")
        );
    }

    fn test_client(providers: Vec<ProviderConfig>, preferred: &str) -> LlmClient {
        LlmClient::new(LlmConfig {
            preferred_provider: preferred.into(),
            temperature: 0.2,
            timeout_secs: 8,
            max_tokens: 64,
            max_fallback_attempts: 4,
            providers,
        })
    }

    fn named_provider(name: &str, base_url: &str) -> ProviderConfig {
        ProviderConfig {
            name: name.into(),
            api_key: "k".into(),
            base_url: base_url.into(),
            model: "demo".into(),
            is_anthropic: false,
            is_ollama: false,
        }
    }

    #[test]
    fn attempt_timeout_matches_python_integer_division() {
        assert_eq!(attempt_timeout(60, 1), 60);
        assert_eq!(attempt_timeout(60, 2), 30);
        assert_eq!(attempt_timeout(60, 4), 15);
        assert_eq!(attempt_timeout(8, 3), 2);
        assert_eq!(attempt_timeout(1, 4), 1);
        assert_eq!(attempt_timeout(0, 2), 1);
        assert_eq!(attempt_timeout(8, 0), 8);
        let stream_old_quadruple = attempt_timeout(8, 2).max(1) * 4;
        assert_eq!(attempt_timeout(8, 2), 4);
        assert_eq!(stream_old_quadruple, 16);
    }

    #[tokio::test]
    async fn stream_times_out_per_attempt_then_falls_back() {
        const SSE: &str =
            "data: {\"choices\":[{\"delta\":{\"content\":\"fallback\"}}]}\n\ndata: [DONE]\n\n";
        let hang = axum::Router::new().route(
            "/chat/completions",
            axum::routing::post(|| async {
                tokio::time::sleep(Duration::from_secs(30)).await;
                "never"
            }),
        );
        let hang_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let hang_addr = hang_listener.local_addr().unwrap();
        let hang_handle = tokio::spawn(async move {
            axum::serve(hang_listener, hang).await.unwrap();
        });
        let (ok_url, ok_handle) =
            serve_chat_completions(axum::http::StatusCode::OK, SSE, "text/event-stream").await;

        let client = std::sync::Arc::new(LlmClient::new(LlmConfig {
            preferred_provider: "openai".into(),
            temperature: 0.2,
            timeout_secs: 2,
            max_tokens: 64,
            max_fallback_attempts: 4,
            providers: vec![
                named_provider("openai", &format!("http://{hang_addr}")),
                named_provider("deepseek", &ok_url),
            ],
        }));
        let started = Instant::now();
        let mut rx = client
            .stream(&[ChatMessage::new("user", "hi")])
            .await
            .expect("hanging first provider should time out and fall back");
        let mut deltas = Vec::new();
        while let Some(chunk) = rx.recv().await {
            deltas.push((chunk.provider, chunk.delta));
        }
        let elapsed = started.elapsed();
        assert_eq!(deltas, vec![("deepseek".into(), "fallback".into())]);
        assert!(
            elapsed < Duration::from_secs(4),
            "per-attempt timeout is 1s (2/2), old *4 budget would wait 4s, elapsed={elapsed:?}"
        );
        hang_handle.abort();
        ok_handle.abort();
    }

    #[test]
    fn sticky_active_provider_moves_to_front() {
        let mut providers = vec![
            demo_provider("deepseek", false, false),
            demo_provider("openai", false, false),
            demo_provider("ollama", false, true),
        ];
        promote_active_provider(&mut providers, Some("openai"));
        assert_eq!(
            providers.iter().map(|item| item.name.as_str()).collect::<Vec<_>>(),
            vec!["openai", "deepseek", "ollama"]
        );
        promote_active_provider(&mut providers, Some("openai"));
        assert_eq!(providers[0].name, "openai");
        promote_active_provider(&mut providers, Some("missing"));
        assert_eq!(providers[0].name, "openai");
    }

    #[tokio::test]
    async fn candidate_providers_promote_sticky_and_skip_cooled_non_preferred() {
        let client = test_client(
            vec![
                demo_provider("deepseek", false, false),
                demo_provider("openai", false, false),
            ],
            "deepseek",
        );
        *client.active_provider.lock().await = Some("openai".into());
        let names: Vec<_> = client
            .candidate_providers()
            .await
            .into_iter()
            .map(|item| item.name)
            .collect();
        assert_eq!(names, vec!["openai", "deepseek"]);

        client.mark_failure("openai").await;
        let names: Vec<_> = client
            .candidate_providers()
            .await
            .into_iter()
            .map(|item| item.name)
            .collect();
        assert_eq!(names, vec!["deepseek"]);
    }

    async fn serve_chat_completions(
        status: axum::http::StatusCode,
        body: &'static str,
        content_type: &'static str,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let app = axum::Router::new().route(
            "/chat/completions",
            axum::routing::post(move || async move {
                (
                    status,
                    [(axum::http::header::CONTENT_TYPE, content_type)],
                    body.to_string(),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), handle)
    }

    #[tokio::test]
    async fn stream_falls_back_to_next_provider_and_sticks() {
        const SSE: &str =
            "data: {\"choices\":[{\"delta\":{\"content\":\"你好\"}}]}\n\ndata: [DONE]\n\n";
        let (fail_url, fail_handle) = serve_chat_completions(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "boom",
            "text/plain",
        )
        .await;
        let (ok_url, ok_handle) =
            serve_chat_completions(axum::http::StatusCode::OK, SSE, "text/event-stream").await;

        let client = std::sync::Arc::new(test_client(
            vec![
                named_provider("openai", &fail_url),
                named_provider("deepseek", &ok_url),
            ],
            "openai",
        ));
        let mut rx = client
            .stream(&[ChatMessage::new("user", "hi")])
            .await
            .expect("stream should fall back");
        let mut deltas = Vec::new();
        while let Some(chunk) = rx.recv().await {
            deltas.push((chunk.provider, chunk.delta));
        }
        assert_eq!(deltas, vec![("deepseek".into(), "你好".into())]);
        assert_eq!(
            client.active_provider.lock().await.as_deref(),
            Some("deepseek")
        );
        fail_handle.abort();
        ok_handle.abort();
    }

    #[tokio::test]
    async fn prime_sticks_first_available_provider() {
        let client = test_client(
            vec![
                demo_provider("deepseek", false, false),
                demo_provider("openai", false, false),
            ],
            "openai",
        );
        assert_eq!(client.prime().await.unwrap(), "openai");
        assert_eq!(client.active_provider.lock().await.as_deref(), Some("openai"));
        let empty = test_client(Vec::new(), "openai");
        let err = empty.prime().await.expect_err("no providers");
        assert_eq!(err, LLM_NO_PROVIDER_MESSAGE);
        assert!(empty.active_provider.lock().await.is_none());
    }

    #[test]
    fn default_request_reuses_process_client() {
        let mut cfg = crate::config::Config::jwt_test();
        cfg.openai_api_key = "sk-test".into();
        cfg.openai_model = "gpt-4o-mini".into();
        cfg.anthropic_api_key = "ak-test".into();
        let default = LlmClient::from_env(&cfg);
        let reused = resolve_llm_client(&default, &cfg, None, None, None);
        assert!(std::sync::Arc::ptr_eq(&default, &reused));
        let isolated = resolve_llm_client(&default, &cfg, Some("anthropic"), None, None);
        assert!(!std::sync::Arc::ptr_eq(&default, &isolated));
        let by_temperature = resolve_llm_client(&default, &cfg, None, None, Some(0.2));
        assert!(!std::sync::Arc::ptr_eq(&default, &by_temperature));
    }

    #[tokio::test]
    async fn process_client_keeps_sticky_across_resolves() {
        let mut cfg = crate::config::Config::jwt_test();
        cfg.openai_api_key = "sk-test".into();
        cfg.openai_model = "gpt-4o-mini".into();
        cfg.anthropic_api_key = "ak-test".into();
        cfg.llm_provider = "openai".into();
        let default = LlmClient::from_env(&cfg);
        *default.active_provider.lock().await = Some("anthropic".into());
        let reused = resolve_llm_client(&default, &cfg, None, None, None);
        let names: Vec<_> = reused
            .candidate_providers()
            .await
            .into_iter()
            .map(|item| item.name)
            .collect();
        assert_eq!(names[0], "anthropic");
        let isolated = resolve_llm_client(&default, &cfg, Some("openai"), None, None);
        let names: Vec<_> = isolated
            .candidate_providers()
            .await
            .into_iter()
            .map(|item| item.name)
            .collect();
        assert_eq!(names[0], "openai");
        assert!(isolated.active_provider.lock().await.is_none());
    }

    #[tokio::test]
    async fn stream_exhausts_fallback_chain_and_clears_sticky() {
        let (fail_a, handle_a) = serve_chat_completions(
            axum::http::StatusCode::BAD_GATEWAY,
            "down",
            "text/plain",
        )
        .await;
        let (fail_b, handle_b) = serve_chat_completions(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "busy",
            "text/plain",
        )
        .await;
        let client = std::sync::Arc::new(test_client(
            vec![
                named_provider("openai", &fail_a),
                named_provider("deepseek", &fail_b),
            ],
            "openai",
        ));
        *client.active_provider.lock().await = Some("openai".into());
        let err = client
            .stream(&[ChatMessage::new("user", "hi")])
            .await
            .expect_err("all providers fail");
        assert!(err.contains("所有 LLM 提供商均不可用"), "{err}");
        assert!(err.contains("openai"), "{err}");
        assert!(err.contains("deepseek"), "{err}");
        assert!(client.active_provider.lock().await.is_none());
        handle_a.abort();
        handle_b.abort();
    }
}
