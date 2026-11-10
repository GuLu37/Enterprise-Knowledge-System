//! 短期记忆：滑动窗口，或把超出窗口的更早轮次递进摘要后注入 prompt。
//! 对齐 Python `_build_sliding_window_history` / `_build_cached_summary_history`。

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use serde_json::json;
use sha2::{Digest, Sha256};

use crate::llm::{ChatMessage, LlmClient};

const SUMMARY_CACHE_MAX_ENTRIES: usize = 256;
const SUMMARY_SYSTEM_PROMPT: &str = "你是对话摘要器。把历史对话压缩成一段可继续对话使用的中文摘要。\
只保留用户目标、已确认事实、约束条件、偏好、待办事项和关键结论。\
不要补充未出现的信息，不要输出标题、编号或解释。";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ShortMemoryOptions {
    pub strategy: String,
    pub window_n: usize,
    pub summary_m: usize,
    pub conversation_id: String,
}

impl ShortMemoryOptions {
    pub fn validate(&self) -> Result<(), String> {
        if self.window_n < 1 {
            return Err("short_memory_n 必须大于 0".into());
        }
        if self.strategy != "window" && self.strategy != "summary" {
            return Err(format!("不支持的短期记忆策略: {}", self.strategy));
        }
        if self.strategy == "summary" && self.summary_m <= self.window_n {
            return Err("short_memory_m 必须大于 short_memory_n".into());
        }
        if self.strategy == "summary" && self.conversation_id.trim().is_empty() {
            return Err("short_memory_strategy=summary 时必须提供 conversation_id".into());
        }
        Ok(())
    }
}

type CacheKey = (String, String, usize, usize);

fn summary_cache() -> &'static Mutex<HashMap<CacheKey, String>> {
    static CACHE: OnceLock<Mutex<HashMap<CacheKey, String>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn inflight() -> &'static Mutex<HashSet<CacheKey>> {
    static SET: OnceLock<Mutex<HashSet<CacheKey>>> = OnceLock::new();
    SET.get_or_init(|| Mutex::new(HashSet::new()))
}

pub fn split_history(history: &[MemoryMessage]) -> (Vec<MemoryMessage>, Vec<Vec<MemoryMessage>>) {
    let mut system_messages = Vec::new();
    let mut turns: Vec<Vec<MemoryMessage>> = Vec::new();
    let mut current: Vec<MemoryMessage> = Vec::new();
    for message in history {
        let role = message.role.trim();
        let content = message.content.trim();
        if content.is_empty() {
            continue;
        }
        if role == "system" {
            system_messages.push(MemoryMessage {
                role: "system".into(),
                content: content.to_string(),
            });
            continue;
        }
        if role != "user" && role != "assistant" {
            continue;
        }
        let normalized = MemoryMessage {
            role: role.to_string(),
            content: content.to_string(),
        };
        if role == "user" {
            if !current.is_empty() {
                turns.push(std::mem::take(&mut current));
            }
            current = vec![normalized];
            continue;
        }
        if current.is_empty() {
            current = vec![normalized];
        } else {
            current.push(normalized);
        }
    }
    if !current.is_empty() {
        turns.push(current);
    }
    (system_messages, turns)
}

pub fn recent_pairs(history: &[MemoryMessage], window_n: usize) -> Vec<(String, String)> {
    let (_, turns) = split_history(history);
    let keep = if window_n == 0 { 0 } else { window_n };
    turns
        .iter()
        .rev()
        .take(keep)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .flat_map(|turn| {
            turn.iter()
                .map(|item| (item.role.clone(), item.content.clone()))
        })
        .collect()
}

pub fn assemble(
    history: &[MemoryMessage],
    options: &ShortMemoryOptions,
    llm: Option<std::sync::Arc<LlmClient>>,
) -> Result<Vec<ChatMessage>, String> {
    options.validate()?;
    if options.strategy == "summary" {
        Ok(assemble_summary(history, options, llm))
    } else {
        Ok(assemble_window(history, options.window_n))
    }
}

fn assemble_window(history: &[MemoryMessage], window_n: usize) -> Vec<ChatMessage> {
    let (system_messages, turns) = split_history(history);
    let recent = if window_n == 0 || turns.len() <= window_n {
        turns.as_slice()
    } else {
        &turns[turns.len() - window_n..]
    };
    compose_messages(&system_messages, recent, "", None, 0)
}

fn assemble_summary(
    history: &[MemoryMessage],
    options: &ShortMemoryOptions,
    llm: Option<std::sync::Arc<LlmClient>>,
) -> Vec<ChatMessage> {
    let (system_messages, turns) = split_history(history);
    if turns.len() <= options.window_n {
        return compose_messages(&system_messages, &turns, "", None, 0);
    }
    let recent = &turns[turns.len() - options.window_n..];
    let older_turn_count = turns.len() - options.window_n;
    if let Some(key) = cache_key(&options.conversation_id, &turns, options.window_n, options.summary_m) {
        if let Some(summary) = get_cached(&key) {
            return compose_messages(
                &system_messages,
                recent,
                &summary,
                Some(&options.conversation_id),
                older_turn_count,
            );
        }
        if let Some(llm) = llm {
            schedule_summary(history.to_vec(), options.clone(), llm, key);
        }
    }
    compose_messages(&system_messages, recent, "", None, 0)
}

fn compose_messages(
    system_messages: &[MemoryMessage],
    recent_turns: &[Vec<MemoryMessage>],
    summary_text: &str,
    conversation_id: Option<&str>,
    older_turn_count: usize,
) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    for item in system_messages {
        messages.push(ChatMessage::new("system", &item.content));
    }
    let summary = summary_text.trim();
    if !summary.is_empty() {
        let reference = json!({
            "source": "short_term_summary",
            "conversation_id": conversation_id.unwrap_or(""),
            "turn_start": 0,
            "turn_end": older_turn_count.saturating_sub(1),
            "disclosure_tool": "memory_retrieve_original",
        });
        messages.push(ChatMessage::new(
            "system",
            format!(
                "以下是更早历史的递进摘要，仅供继续对话参考：\n{summary}\n\
如需核对这段摘要对应的完整原始对话，应调用 memory_retrieve_original，\
并优先使用下面的摘要引用：\n{reference}"
            ),
        ));
    }
    for turn in recent_turns {
        for item in turn {
            messages.push(ChatMessage::new(&item.role, &item.content));
        }
    }
    messages
}

fn cache_key(
    conversation_id: &str,
    turns: &[Vec<MemoryMessage>],
    window_n: usize,
    summary_m: usize,
) -> Option<CacheKey> {
    let conversation_id = conversation_id.trim();
    if conversation_id.is_empty() || turns.len() <= window_n {
        return None;
    }
    let payload: Vec<Vec<serde_json::Value>> = turns
        .iter()
        .map(|turn| {
            turn.iter()
                .map(|item| json!({"role": item.role, "content": item.content}))
                .collect()
        })
        .collect();
    let encoded = serde_json::to_string(&payload).unwrap_or_else(|_| "[]".into());
    let digest = hex::encode(Sha256::digest(encoded.as_bytes()));
    Some((conversation_id.to_string(), digest, window_n, summary_m))
}

fn get_cached(key: &CacheKey) -> Option<String> {
    summary_cache().lock().ok()?.get(key).cloned()
}

fn set_cached(key: CacheKey, summary: String) {
    let summary = summary.trim().to_string();
    if summary.is_empty() {
        return;
    }
    if let Ok(mut cache) = summary_cache().lock() {
        cache.insert(key.clone(), summary);
        while cache.len() > SUMMARY_CACHE_MAX_ENTRIES {
            let first = cache.keys().next().cloned();
            match first {
                Some(first) => {
                    cache.remove(&first);
                }
                None => break,
            }
        }
    }
}

fn format_summary_submit_fail(error: &str) -> String {
    format!("后台短期摘要任务提交失败: {error}")
}

fn schedule_summary(
    history: Vec<MemoryMessage>,
    options: ShortMemoryOptions,
    llm: std::sync::Arc<LlmClient>,
    key: CacheKey,
) {
    {
        let Ok(mut pending) = inflight().lock() else {
            tracing::warn!("{}", format_summary_submit_fail("inflight lock poisoned"));
            return;
        };
        if pending.contains(&key) || get_cached(&key).is_some() {
            return;
        }
        pending.insert(key.clone());
    }
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            handle.spawn(async move {
                let result = build_summary_text(&history, options.window_n, options.summary_m, &llm).await;
                match result {
                    Ok(summary) => set_cached(key.clone(), summary),
                    Err(err) => tracing::warn!("后台短期摘要生成失败: {err}"),
                }
                if let Ok(mut pending) = inflight().lock() {
                    pending.remove(&key);
                }
            });
        }
        Err(err) => {
            if let Ok(mut pending) = inflight().lock() {
                pending.remove(&key);
            }
            tracing::warn!("{}", format_summary_submit_fail(&err.to_string()));
        }
    }
}

async fn build_summary_text(
    history: &[MemoryMessage],
    window_n: usize,
    summary_m: usize,
    llm: &LlmClient,
) -> Result<String, String> {
    let (_, turns) = split_history(history);
    if turns.len() <= window_n {
        return Ok(String::new());
    }
    let older = &turns[..turns.len() - window_n];
    let batch_size = summary_m.saturating_sub(window_n).max(1);
    let mut summary = String::new();
    for batch in older.chunks(batch_size) {
        summary = summarize_batch(llm, batch, &summary).await?;
    }
    Ok(summary)
}

async fn summarize_batch(
    llm: &LlmClient,
    batch: &[Vec<MemoryMessage>],
    previous_summary: &str,
) -> Result<String, String> {
    let mut messages = vec![ChatMessage::new("system", SUMMARY_SYSTEM_PROMPT)];
    if !previous_summary.trim().is_empty() {
        messages.push(ChatMessage::new(
            "system",
            format!(
                "已有摘要如下，请在保留其核心信息的基础上，结合新增历史继续压缩：\n{}",
                previous_summary.trim()
            ),
        ));
    }
    messages.push(ChatMessage::new(
        "user",
        format!("请概括下面的历史对话，输出一段中文摘要：\n\n{}", render_turns(batch)),
    ));
    messages.push(ChatMessage::new(
        "user",
        "请只输出摘要正文，不要输出前缀、编号或解释。",
    ));
    Ok(llm.invoke(&messages).await?.content.trim().to_string())
}

fn render_turns(turns: &[Vec<MemoryMessage>]) -> String {
    let mut blocks = Vec::new();
    for turn in turns {
        for item in turn {
            let label = match item.role.as_str() {
                "user" => "用户",
                "assistant" => "助手",
                _ => "系统",
            };
            blocks.push(format!("{label}: {}", item.content));
        }
    }
    blocks.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> MemoryMessage {
        MemoryMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    #[test]
    fn summary_submit_fail_log_matches_python() {
        assert_eq!(
            format_summary_submit_fail("executor shutdown"),
            "后台短期摘要任务提交失败: executor shutdown"
        );
    }

    #[test]
    fn splits_system_and_turns() {
        let history = vec![
            msg("system", "规则"),
            msg("user", "q1"),
            msg("assistant", "a1"),
            msg("user", "q2"),
            msg("assistant", "a2"),
            msg("user", "q3"),
        ];
        let (system, turns) = split_history(&history);
        assert_eq!(system.len(), 1);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[2].len(), 1);
    }

    #[test]
    fn window_keeps_latest_turns() {
        let history = vec![
            msg("system", "规则"),
            msg("user", "q1"),
            msg("assistant", "a1"),
            msg("user", "q2"),
            msg("assistant", "a2"),
            msg("user", "q3"),
            msg("assistant", "a3"),
        ];
        let options = ShortMemoryOptions {
            strategy: "window".into(),
            window_n: 2,
            summary_m: 10,
            conversation_id: String::new(),
        };
        let assembled = assemble(&history, &options, None).unwrap();
        let contents: Vec<&str> = assembled.iter().map(|item| item.content.as_str()).collect();
        assert_eq!(contents, vec!["规则", "q2", "a2", "q3", "a3"]);
        let pairs = recent_pairs(&history, 1);
        assert_eq!(pairs, vec![("user".into(), "q3".into()), ("assistant".into(), "a3".into())]);
    }

    #[test]
    fn summary_without_cache_falls_back_to_window() {
        let mut history = Vec::new();
        for index in 1..=4 {
            history.push(msg("user", &format!("q{index}")));
            history.push(msg("assistant", &format!("a{index}")));
        }
        let options = ShortMemoryOptions {
            strategy: "summary".into(),
            window_n: 2,
            summary_m: 5,
            conversation_id: "c1".into(),
        };
        let assembled = assemble(&history, &options, None).unwrap();
        let contents: Vec<&str> = assembled.iter().map(|item| item.content.as_str()).collect();
        assert_eq!(contents, vec!["q3", "a3", "q4", "a4"]);
    }

    #[test]
    fn rejects_invalid_summary_options() {
        let options = ShortMemoryOptions {
            strategy: "summary".into(),
            window_n: 5,
            summary_m: 5,
            conversation_id: "c1".into(),
        };
        assert!(options.validate().is_err());
        let missing_id = ShortMemoryOptions {
            strategy: "summary".into(),
            window_n: 2,
            summary_m: 5,
            conversation_id: String::new(),
        };
        assert!(missing_id.validate().is_err());
    }

    #[test]
    fn compose_injects_summary_reference() {
        let recent = vec![vec![msg("user", "q"), msg("assistant", "a")]];
        let messages = compose_messages(&[], &recent, "更早在讨论报销", Some("c1"), 3);
        assert_eq!(messages[0].role, "system");
        assert!(messages[0].content.contains("更早在讨论报销"));
        assert!(messages[0].content.contains("memory_retrieve_original"));
        assert_eq!(messages[1].content, "q");
        assert_eq!(messages[2].content, "a");
    }
}
