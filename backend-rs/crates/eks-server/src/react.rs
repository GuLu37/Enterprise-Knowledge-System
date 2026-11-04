//! ReAct 检索升级：统一编排（RAG + Skill + MCP）失败或关闭时回退 rag 专用循环。

use futures_util::stream::{self, StreamExt};
use serde_json::{Value, json};

use crate::llm::{ChatMessage, LlmClient, ToolCall};
use crate::planner;
use crate::retrieval::RetrievalItem;
use crate::retrieval_engine::RetrievalEngine;
use crate::skills;

const UNIFIED_REACT_SYSTEM_PROMPT: &str = "你是企业知识库助手的任务执行者。你的目标是为后续回答收集到足够的证据，不要直接回答用户问题本身。\n\
可用工具不止一个，按用途分为三类，你可以按需组合、跨工具多次调用：\n\
- 检索类：search_knowledge_base，查文档、制度、流程、内部事实，可按 retrieval_method 与 top_k 自主调策略。\n\
- 技能类：记忆查询与原文展开、知识库文件清单、热词统计等本地能力。\n\
- 远程 MCP：由外部服务提供的能力，若列表中没有则不可用，不要臆造工具名。\n\
规则信号：{signal}\n\
流程：每一轮先判断还缺什么，选择最合适的工具调用；观察结果后再决定是换工具、换参数继续，还是收手。\n\
约束：\n\
1. 同一轮内多个互不依赖的调用，请一次性并行发出。\n\
2. 不要重复调用相同参数的同一个工具。\n\
3. 工具返回 error 时，改参数或换工具重试一次即可，不要反复重试同一个失败调用。\n\
4. 优先在 {max_iterations} 轮内完成，避免低价值查询。\n\
当你已获得足以回答问题的资料，直接输出简短说明（如「资料已足够」）并停止调用工具。";

const RAG_REACT_SYSTEM_PROMPT: &str = "你是企业知识库检索决策者。你的任务是检索到足够回答用户问题的资料，不要回答用户问题本身。\n\
可用工具：\n\
- search_knowledge_base：执行检索，可通过参数 retrieval_method(dense/sparse/hybrid) 与 top_k 自主选择检索策略，也支持按文件名/文档名查询某份文档是否存在。\n\
规则信号：{signal}\n\
流程：调用 search_knowledge_base 检索；如果结果不足，可换 retrieval_method 或改写 query 再次检索。\n\
优先在 1-2 轮内完成检索，避免重复或低价值查询；尽量一次性覆盖问题的多个方面。\n\
示例：当用户问『04_产品与账号常见问题呢』这类文件名查询时，直接用文件名作为 query 调用 search_knowledge_base；若未检索到，说明该文档未在知识库中。\n\
当你认为已经检索到足够资料时，直接输出简短说明（如「检索完成」）并停止调用工具。";

pub struct ReactOutcome {
    pub sources: Vec<RetrievalItem>,
    pub context: String,
    pub notes: Vec<String>,
    pub used_react: bool,
    pub kind: String,
}

pub enum ReactAttempt {
    Unavailable,
    Ran(ReactOutcome),
}

pub async fn run_unified_react(
    engine: &RetrievalEngine,
    pool: &sqlx::SqlitePool,
    llm: &LlmClient,
    user_id: &str,
    query: &str,
    history: &[(String, String)],
    top_k: usize,
    first_results: Vec<RetrievalItem>,
    mcp_manifests: &[Value],
    enabled: bool,
    max_iterations: usize,
    max_workers: usize,
) -> ReactAttempt {
    if !enabled {
        tracing::info!("统一 ReAct 编排已关闭，回退 rag 专用循环");
        return ReactAttempt::Unavailable;
    }
    run_react_loop(
        engine,
        pool,
        llm,
        user_id,
        query,
        history,
        top_k,
        first_results,
        skills::function_calling_tools_from(mcp_manifests),
        UNIFIED_REACT_SYSTEM_PROMPT,
        mcp_manifests,
        max_iterations,
        max_workers,
        "unified_react",
    )
    .await
}

pub async fn run_rag_react(
    engine: &RetrievalEngine,
    pool: &sqlx::SqlitePool,
    llm: &LlmClient,
    user_id: &str,
    query: &str,
    history: &[(String, String)],
    top_k: usize,
    first_results: Vec<RetrievalItem>,
    max_iterations: usize,
    max_workers: usize,
) -> ReactAttempt {
    run_react_loop(
        engine,
        pool,
        llm,
        user_id,
        query,
        history,
        top_k,
        first_results,
        skills::rag_function_calling_tools(),
        RAG_REACT_SYSTEM_PROMPT,
        &[],
        max_iterations,
        max_workers,
        "rag_react",
    )
    .await
}

async fn run_react_loop(
    engine: &RetrievalEngine,
    pool: &sqlx::SqlitePool,
    llm: &LlmClient,
    user_id: &str,
    query: &str,
    history: &[(String, String)],
    top_k: usize,
    first_results: Vec<RetrievalItem>,
    tools: Vec<Value>,
    system_prompt: &str,
    mcp_manifests: &[Value],
    max_iterations: usize,
    max_workers: usize,
    kind: &str,
) -> ReactAttempt {
    let mut sources = first_results;
    let mut notes = Vec::new();
    let signal = rule_retrieval_signal(query);
    let mut messages = vec![ChatMessage::new(
        "system",
        system_prompt
            .replace("{signal}", &signal)
            .replace("{max_iterations}", &max_iterations.max(1).to_string()),
    )];
    for (role, content) in history.iter().rev().take(6).collect::<Vec<_>>().into_iter().rev() {
        if role == "user" || role == "assistant" {
            messages.push(ChatMessage::new(role, content));
        }
    }
    messages.push(ChatMessage::new("user", query));

    let mut seen = std::collections::HashSet::new();
    let mut used_react = false;
    for iteration in 0..max_iterations.max(1) {
        let completion = match llm.invoke_with_tools(&messages, &tools).await {
            Ok(result) => result,
            Err(err) => {
                if !used_react {
                    tracing::warn!("{kind} bind_tools 失败: {err}");
                    return ReactAttempt::Unavailable;
                }
                tracing::warn!("{kind} 后续轮次失败，使用已有观察: {err}");
                break;
            }
        };
        if completion.tool_calls.is_empty() {
            if !completion.content.trim().is_empty() {
                notes.push(completion.content.trim().to_string());
            }
            break;
        }
        used_react = true;
        let calls = completion.tool_calls;
        let raw_calls: Vec<Value> = calls.iter().map(|call| call.raw.clone()).collect();
        messages.push(ChatMessage {
            role: "assistant".into(),
            content: completion.content,
            tool_calls: Some(json!(raw_calls)),
            tool_call_id: None,
        });
        let tool_names = calls.iter().map(|call| call.name.as_str()).collect::<Vec<_>>().join("、");
        let mut observations = vec![json!({"error": "missing_execution"}); calls.len()];
        let mut extras: Vec<Vec<RetrievalItem>> = vec![Vec::new(); calls.len()];
        let mut pending = Vec::new();
        for (index, call) in calls.iter().enumerate() {
            if !seen.insert(call_dedupe_key(call)) {
                observations[index] = json!({"skipped": "duplicate"});
                continue;
            }
            pending.push(index);
        }
        let workers = parallel_worker_limit(pending.len(), max_workers);
        tracing::info!(
            kind,
            iteration = iteration + 1,
            max_iterations = max_iterations.max(1),
            tools = tool_names.as_str(),
            pending = pending.len(),
            workers,
            "ReAct 本轮工具调用"
        );
        if !pending.is_empty() {
            let mut stream = stream::iter(pending).map(|index| {
                let call = calls[index].clone();
                async move {
                    tracing::info!(
                        kind,
                        iteration = iteration + 1,
                        tool = call.name.as_str(),
                        method = call
                            .arguments
                            .get("retrieval_method")
                            .and_then(|value| value.as_str())
                            .unwrap_or("默认"),
                        "ReAct 工具执行"
                    );
                    let (observation, extra) = execute_tool(
                        engine,
                        pool,
                        user_id,
                        query,
                        top_k,
                        &call,
                        mcp_manifests,
                        llm,
                        history,
                    )
                    .await;
                    (index, observation, extra)
                }
            }).buffer_unordered(workers);
            while let Some((index, observation, extra)) = stream.next().await {
                observations[index] = observation;
                extras[index] = extra;
            }
        }
        let mut searched = false;
        for (index, call) in calls.iter().enumerate() {
            let observation = std::mem::take(&mut observations[index]);
            notes.push(format!("{}: {}", call.name, truncate(&observation.to_string(), 240)));
            messages.push(tool_message(call, observation));
            merge_sources(&mut sources, std::mem::take(&mut extras[index]));
            searched |= call.name == "search_knowledge_base";
        }
        if searched && planner::assess_evidence_rule(query, &sources).sufficient {
            tracing::info!(kind, "证据评估充分，提前结束 ReAct");
            return ReactAttempt::Ran(finish(sources, notes, used_react, kind));
        }
    }
    ReactAttempt::Ran(finish(sources, notes, used_react, kind))
}

pub fn rule_retrieval_signal(query: &str) -> String {
    planner::select_initial_rule(query, 5, planner::RETRIEVAL_METHOD_HYBRID).signal_text()
}

fn finish(sources: Vec<RetrievalItem>, notes: Vec<String>, used_react: bool, kind: &str) -> ReactOutcome {
    let max_chars = if sources.iter().any(|item| item.structured_aggregation) {
        30000
    } else {
        6500
    };
    let mut context = crate::retrieval::assemble_rag_context(&sources, max_chars);
    if !notes.is_empty() && context.chars().count() < max_chars {
        context.push_str("\n\n工具观察：\n");
        context.push_str(&notes.join("\n"));
    }
    ReactOutcome {
        sources,
        context,
        notes,
        used_react,
        kind: kind.to_string(),
    }
}

fn tool_message(call: &ToolCall, observation: Value) -> ChatMessage {
    ChatMessage {
        role: "tool".into(),
        content: observation.to_string(),
        tool_calls: None,
        tool_call_id: Some(call.id.clone()),
    }
}

fn parallel_worker_limit(pending: usize, configured: usize) -> usize {
    pending.min(configured.max(1)).max(1)
}

fn call_dedupe_key(call: &ToolCall) -> String {
    if call.name == "search_knowledge_base" {
        let query_key = eks_kernels::normalize_text(
            call.arguments
                .get("query")
                .and_then(|value| value.as_str())
                .unwrap_or(""),
        );
        if !query_key.is_empty() {
            return format!("search_knowledge_base:{query_key}");
        }
    }
    format!("{}:{}", call.name, compact_args(&call.arguments))
}

async fn execute_tool(
    engine: &RetrievalEngine,
    pool: &sqlx::SqlitePool,
    user_id: &str,
    original_query: &str,
    default_top_k: usize,
    call: &ToolCall,
    mcp_manifests: &[Value],
    llm: &LlmClient,
    history: &[(String, String)],
) -> (Value, Vec<RetrievalItem>) {
    if call.name == "search_knowledge_base" {
        let query = call
            .arguments
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or(original_query)
            .trim();
        let top_k = call
            .arguments
            .get("top_k")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default_top_k)
            .clamp(1, 50);
        let method = call
            .arguments
            .get("retrieval_method")
            .and_then(|v| v.as_str())
            .unwrap_or("hybrid");
        return match engine.retrieve_planned(query, method, top_k, Some(llm)).await {
            Ok((items, _)) => {
                let payload: Vec<Value> = items.iter().take(8).map(|item| item.to_json()).collect();
                (
                    json!({"ok": true, "count": payload.len(), "results": payload}),
                    items,
                )
            }
            Err(err) => (json!({"error": err}), Vec::new()),
        };
    }
    let recent_history: Vec<crate::short_memory::MemoryMessage> = history
        .iter()
        .map(|(role, content)| crate::short_memory::MemoryMessage {
            role: role.clone(),
            content: content.clone(),
        })
        .collect();
    match skills::execute_named_from(
        pool,
        user_id,
        &call.name,
        &call.arguments,
        mcp_manifests,
        None,
        &recent_history,
    )
    .await {
        Ok(result) => {
            let result = if call.name == "memory_query" {
                match skills::maybe_disclose_original_memory(
                    pool,
                    user_id,
                    original_query,
                    None,
                    result,
                    mcp_manifests,
                    false,
                )
                .await
                {
                    Ok(merged) => merged,
                    Err(err) => return (json!({"error": err}), Vec::new()),
                }
            } else {
                result
            };
            (json!({"ok": true, "result": result}), Vec::new())
        }
        Err(err) => (json!({"error": err}), Vec::new()),
    }
}

fn merge_sources(existing: &mut Vec<RetrievalItem>, extra: Vec<RetrievalItem>) {
    let mut seen: std::collections::HashSet<(String, String, String)> =
        existing.iter().map(crate::retrieval::citation_chunk_key).collect();
    for item in extra {
        if seen.insert(crate::retrieval::citation_chunk_key(&item)) {
            existing.push(item);
        }
    }
}

fn compact_args(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

fn truncate(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (index, ch) in text.chars().enumerate() {
        if index >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_and_dedupes_args() {
        assert_eq!(truncate("abcdef", 3), "abc…");
        assert_eq!(compact_args(&json!({"query":"a"})), "{\"query\":\"a\"}");
    }

    #[test]
    fn parallel_workers_cap_to_pending_and_config() {
        assert_eq!(parallel_worker_limit(1, 4), 1);
        assert_eq!(parallel_worker_limit(3, 4), 3);
        assert_eq!(parallel_worker_limit(8, 4), 4);
        assert_eq!(parallel_worker_limit(8, 0), 1);
    }

    #[test]
    fn search_dedupe_key_ignores_method_and_keeps_skill_args() {
        let search = ToolCall {
            id: "1".into(),
            name: "search_knowledge_base".into(),
            arguments: json!({"query": " 差旅报销 ", "retrieval_method": "dense"}),
            raw: json!({}),
        };
        let search_again = ToolCall {
            id: "2".into(),
            name: "search_knowledge_base".into(),
            arguments: json!({"query": "差旅报销", "retrieval_method": "sparse"}),
            raw: json!({}),
        };
        assert_eq!(call_dedupe_key(&search), call_dedupe_key(&search_again));
        let skill = ToolCall {
            id: "3".into(),
            name: "memory_query".into(),
            arguments: json!({"query": "偏好", "top_k": 5}),
            raw: json!({}),
        };
        assert!(call_dedupe_key(&skill).starts_with("memory_query:"));
    }

    #[test]
    fn restores_original_order_from_unordered_completions() {
        let mut slots = vec![None, None, None];
        for (index, value) in [(2, "c"), (0, "a"), (1, "b")] {
            slots[index] = Some(value);
        }
        let ordered: Vec<_> = slots.into_iter().map(Option::unwrap).collect();
        assert_eq!(ordered, ["a", "b", "c"]);
    }

    #[test]
    fn rag_react_binds_only_search_tool() {
        let tools = skills::rag_function_calling_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(
            tools[0].pointer("/function/name").and_then(Value::as_str),
            Some("search_knowledge_base")
        );
    }

    #[test]
    fn retrieval_signal_matches_planner_rules() {
        let identifier = rule_retrieval_signal("员工号 E0011 的姓名和部门");
        assert!(identifier.contains("identifier_exact"));
        assert!(identifier.contains("sparse"));
        let structured = rule_retrieval_signal("列出各产品名单");
        assert!(structured.contains("structured_multi_query"));
        let procedural = rule_retrieval_signal("差旅报销流程怎么走");
        assert!(procedural.contains("procedural_hybrid"));
        assert_eq!(
            rule_retrieval_signal("随便聊聊天"),
            "查询特征策略=default_hybrid（通用知识库问题使用混合检索作为稳定起点），规则建议检索方式=hybrid"
        );
    }
}
