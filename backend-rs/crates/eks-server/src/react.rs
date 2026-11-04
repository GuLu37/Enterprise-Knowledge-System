//! 统一 ReAct：首轮检索证据不足时，绑定全量本地工具让模型改写、换工具或补检索。

use serde_json::{Value, json};

use crate::llm::{ChatMessage, LlmClient, ToolCall};
use crate::retrieval::RetrievalItem;
use crate::retrieval_engine::RetrievalEngine;
use crate::skills;

const MAX_ITERATIONS: usize = 3;

const SYSTEM_PROMPT: &str = "你是企业知识库助手的任务执行者。你的目标是为后续回答收集到足够的证据，不要直接回答用户问题本身。\n\
可用工具不止一个，按用途分为两类：\n\
- 检索类：search_knowledge_base，查文档、制度、流程、内部事实，可按 retrieval_method 与 top_k 自主调策略。\n\
- 技能类：记忆查询与原文展开、知识库文件清单、热词统计等本地能力。\n\
规则信号：{signal}\n\
流程：每一轮先判断还缺什么，选择最合适的工具调用；观察结果后再决定是换工具、换参数继续，还是收手。\n\
约束：\n\
1. 不要重复调用相同参数的同一个工具。\n\
2. 工具返回 error 时，改参数或换工具重试一次即可。\n\
3. 优先在 {max_iterations} 轮内完成。\n\
当你已获得足以回答问题的资料，直接输出简短说明（如「资料已足够」）并停止调用工具。";

pub struct ReactOutcome {
    pub sources: Vec<RetrievalItem>,
    pub context: String,
    pub notes: Vec<String>,
    pub used_react: bool,
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
) -> Result<ReactOutcome, String> {
    let mut sources = first_results;
    let mut notes = Vec::new();
    let tools = skills::function_calling_tools();
    let mut messages = vec![ChatMessage::new(
        "system",
        SYSTEM_PROMPT
            .replace("{signal}", "规则建议检索方式=hybrid")
            .replace("{max_iterations}", &MAX_ITERATIONS.to_string()),
    )];
    for (role, content) in history.iter().rev().take(6).collect::<Vec<_>>().into_iter().rev() {
        if role == "user" || role == "assistant" {
            messages.push(ChatMessage::new(role, content));
        }
    }
    messages.push(ChatMessage::new("user", query));

    let mut seen = std::collections::HashSet::new();
    let mut used_react = false;
    for _ in 0..MAX_ITERATIONS {
        let completion = match llm.invoke_with_tools(&messages, &tools).await {
            Ok(result) => result,
            Err(err) => {
                tracing::warn!("统一 ReAct bind_tools 失败，回退改写检索: {err}");
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
            let dedupe = format!("{}:{}", call.name, compact_args(&call.arguments));
            if !seen.insert(dedupe) {
                messages.push(tool_message(&call, json!({"skipped": "duplicate"})));
                continue;
            }
            let observation = execute_tool(engine, pool, user_id, query, top_k, &call, &mut sources).await;
            notes.push(format!("{}: {}", call.name, truncate(&observation.to_string(), 240)));
            messages.push(tool_message(&call, observation));
            if retrieval_quality_ok(&sources) && call.name == "search_knowledge_base" {
                return Ok(finish(sources, notes, used_react));
            }
        }
    }
    Ok(finish(sources, notes, used_react))
}

fn finish(sources: Vec<RetrievalItem>, notes: Vec<String>, used_react: bool) -> ReactOutcome {
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
    match skills::execute_named(pool, user_id, &call.name, &call.arguments).await {
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
}
