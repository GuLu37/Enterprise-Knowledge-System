//! ReAct 检索升级：统一编排（RAG + Skill + MCP）失败或关闭时回退 rag 专用循环。

use serde_json::{Value, json};

use crate::llm::{ChatMessage, LlmClient, ToolCall};
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

const STRUCTURED_QUERY_TERMS: &[&str] = &[
    "列出", "各", "分别", "有哪些", "名单", "列表", "清单", "明细", "统计", "数量", "多少",
    "一共", "总数", "比较", "对比", "同时", "并且",
];
const PROCEDURAL_QUERY_TERMS: &[&str] = &[
    "如何", "怎么", "流程", "步骤", "要求", "条件", "标准", "规定", "政策", "制度",
];

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
        let raw_calls: Vec<Value> = completion.tool_calls.iter().map(|call| call.raw.clone()).collect();
        messages.push(ChatMessage {
            role: "assistant".into(),
            content: completion.content,
            tool_calls: Some(json!(raw_calls)),
            tool_call_id: None,
        });
        for call in completion.tool_calls {
            let query_key = if call.name == "search_knowledge_base" {
                eks_kernels::normalize_text(
                    call.arguments
                        .get("query")
                        .and_then(|value| value.as_str())
                        .unwrap_or(""),
                )
            } else {
                String::new()
            };
            let dedupe = if call.name == "search_knowledge_base" && !query_key.is_empty() {
                format!("search_knowledge_base:{query_key}")
            } else {
                format!("{}:{}", call.name, compact_args(&call.arguments))
            };
            if !seen.insert(dedupe) {
                messages.push(tool_message(&call, json!({"skipped": "duplicate"})));
                continue;
            }
            tracing::info!(
                kind,
                iteration = iteration + 1,
                max_iterations = max_iterations.max(1),
                tool = call.name.as_str(),
                method = call
                    .arguments
                    .get("retrieval_method")
                    .and_then(|value| value.as_str())
                    .unwrap_or("默认"),
                "ReAct 工具调用"
            );
            let observation =
                execute_tool(engine, pool, user_id, query, top_k, &call, &mut sources, mcp_manifests).await;
            notes.push(format!("{}: {}", call.name, truncate(&observation.to_string(), 240)));
            messages.push(tool_message(&call, observation));
            if retrieval_quality_ok(&sources) && call.name == "search_knowledge_base" {
                return ReactAttempt::Ran(finish(sources, notes, used_react, kind));
            }
        }
    }
    ReactAttempt::Ran(finish(sources, notes, used_react, kind))
}

pub fn rule_retrieval_signal(query: &str) -> String {
    let normalized = query.trim();
    if eks_kernels::rerank::is_identifier_query(normalized) {
        return "查询特征策略=identifier_exact（检测到编号、工号、产品号或编码，优先使用稀疏检索保留精确匹配），规则建议检索方式=sparse".into();
    }
    if STRUCTURED_QUERY_TERMS.iter().any(|term| normalized.contains(term)) {
        return "查询特征策略=structured_multi_query（检测到多条件或结构化查询，保留混合检索并扩大候选范围），规则建议检索方式=hybrid".into();
    }
    if PROCEDURAL_QUERY_TERMS.iter().any(|term| normalized.contains(term)) {
        return "查询特征策略=procedural_hybrid（检测到流程、条件或制度类问题，优先使用混合检索），规则建议检索方式=hybrid".into();
    }
    "通用知识库问题，建议 hybrid 检索".into()
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

fn retrieval_quality_ok(results: &[RetrievalItem]) -> bool {
    results.iter().any(|item| item.score >= 0.30)
}

fn tool_message(call: &ToolCall, observation: Value) -> ChatMessage {
    ChatMessage {
        role: "tool".into(),
        content: observation.to_string(),
        tool_calls: None,
        tool_call_id: Some(call.id.clone()),
    }
}

async fn execute_tool(
    engine: &RetrievalEngine,
    pool: &sqlx::SqlitePool,
    user_id: &str,
    original_query: &str,
    default_top_k: usize,
    call: &ToolCall,
    sources: &mut Vec<RetrievalItem>,
    mcp_manifests: &[Value],
) -> Value {
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
        let result = match method {
            "dense" => engine.search_dense(query, top_k).await.map(|items| (items, String::new())),
            "sparse" => engine.search_sparse(query, top_k).await.map(|items| (items, String::new())),
            _ => engine.run_rag_workflow(query, top_k).await,
        };
        return match result {
            Ok((items, _)) => {
                let payload: Vec<Value> = items.iter().take(8).map(|item| item.to_json()).collect();
                merge_sources(sources, items);
                json!({"ok": true, "count": payload.len(), "results": payload})
            }
            Err(err) => json!({"error": err}),
        };
    }
    match skills::execute_named_from(pool, user_id, &call.name, &call.arguments, mcp_manifests).await {
        Ok(result) => json!({"ok": true, "result": result}),
        Err(err) => json!({"error": err}),
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
            "通用知识库问题，建议 hybrid 检索"
        );
    }
}
