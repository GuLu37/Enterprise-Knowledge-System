//! 本地 Skill：规则路由 + SQLite 执行。
//!
//! 覆盖前端会直接碰到的工具目录、知识库文件、热词和画像记忆。
//! 同名 markdown（`backend/app/skills/{name}/{name}.md`）作为 `skill_instructions`
//! 写入执行封装，供最终回答把该 Skill 的详细规则当作补充规范。
//! 远程 MCP 通过 `MCP_REMOTE_TOOL_MANIFESTS` 注册进同一套目录；未就绪工具
//! 只出现在 capabilities，不进入 Function Calling。

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

use serde_json::{Value, json};

use crate::llm::ChatMessage;

const TOOL_CATALOG_HINTS: &[&str] = &[
    "列出你的工具",
    "列出工具",
    "有什么工具",
    "你有什么技能",
    "有哪些技能",
    "可用技能",
    "可用工具",
    "可用能力",
    "有哪些能力",
    "工具列表",
    "技能列表",
    "工具注册表",
    "function calling",
    "函数调用",
];

const PROFILE_NAME_HINTS: &[&str] = &[
    "我叫什么",
    "我叫什么名字",
    "我的名字",
    "你记得我叫什么",
    "还记得我叫什么",
    "记得我叫",
    "我叫啥",
];

const ORIGINAL_MEMORY_HINTS: &[&str] = &[
    "完整对话",
    "原始对话",
    "原文",
    "完整上下文",
    "具体细节",
    "逐字",
    "前后文",
    "展开",
    "细节",
];

const MEMORY_QUERY_HINTS: &[&str] = &[
    "记得什么",
    "我之前说过",
    "我的偏好",
    "记忆",
    "画像",
];

const HOT_WORD_HINTS: &[&str] = &[
    "热词",
    "高频词",
    "关键词",
    "文件里",
    "文档里",
    "统计词频",
    "词频统计",
];

const NOISY_TERMS: &[&str] = &[
    "文档", "文件", "章节", "内容", "信息", "资料", "数据", "记录", "系统", "知识库",
    "查询", "统计", "热点", "热点词", "全部", "所有", "其他", "相关", "说明", "问题",
    "回答", "表格", "附件", "任务", "document", "file", "content", "data",
];

const FILE_EXTENSIONS: &[&str] = &[
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".md", ".txt", ".csv",
];

#[derive(Debug, Clone)]
pub struct RoutedSkill {
    pub name: &'static str,
    /// 对齐 Python `_rule_route_skill_intent` 的 reason 语义值。
    pub reason: &'static str,
    pub arguments: Value,
}

/// 规则优先的本地技能路由。命中后聊天不再进入知识库检索。
pub fn route_local_skill(query: &str) -> Option<RoutedSkill> {
    let normalized = query.trim();
    if normalized.is_empty() {
        return None;
    }
    let lowered = normalized.to_lowercase();

    if is_profile_name_query(normalized) {
        return Some(RoutedSkill {
            name: "memory_query",
            reason: "profile_name_query",
            arguments: json!({
                "query": normalized,
                "top_k": 5,
                "memory_scope": "all",
            }),
        });
    }
    if TOOL_CATALOG_HINTS.iter().any(|hint| lowered.contains(hint)) {
        return Some(RoutedSkill {
            name: "list_available_tools",
            reason: "tool_catalog_query",
            arguments: json!({}),
        });
    }
    let wants_original = ORIGINAL_MEMORY_HINTS.iter().any(|hint| lowered.contains(hint));
    let memory_context = ["摘要", "记忆", "之前", "刚才", "对话", "上下文", "说过"]
        .iter()
        .any(|hint| lowered.contains(hint));
    if wants_original && memory_context {
        return Some(RoutedSkill {
            name: "memory_retrieve_original",
            reason: "original_conversation_disclosure",
            arguments: json!({
                "query": normalized,
                "top_k": 8,
                "max_chars": 12000,
            }),
        });
    }
    if MEMORY_QUERY_HINTS.iter().any(|hint| lowered.contains(hint)) {
        return Some(RoutedSkill {
            name: "memory_query",
            reason: "memory_keyword",
            arguments: json!({
                "query": normalized,
                "top_k": 5,
                "memory_scope": "all",
            }),
        });
    }
    if HOT_WORD_HINTS.iter().any(|hint| lowered.contains(hint)) {
        let mut arguments = json!({ "top_k": 10, "include_samples": true });
        let file_name = extract_file_name_hint(normalized);
        if !file_name.is_empty() {
            arguments["file_name"] = json!(file_name);
        }
        return Some(RoutedSkill {
            name: "document_hot_word_statistics",
            reason: "document_hot_word_keyword",
            arguments,
        });
    }
    None
}

/// 对齐 Python `_is_profile_name_query`：去掉空白后再匹配。
fn is_profile_name_query(query: &str) -> bool {
    let normalized: String = query
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>()
        .to_lowercase();
    !normalized.is_empty() && PROFILE_NAME_HINTS.iter().any(|hint| normalized.contains(hint))
}

pub fn extract_file_name_hint(query: &str) -> String {
    let lower = query.to_lowercase();
    for ext in FILE_EXTENSIONS {
        if let Some(end_rel) = lower.find(ext) {
            let end = end_rel + ext.len();
            let start = lower[..end_rel]
                .char_indices()
                .rev()
                .take_while(|(_, ch)| !is_filename_break(*ch))
                .last()
                .map(|(index, _)| index)
                .unwrap_or(end_rel);
            let name = query[start..end].trim().trim_matches(|ch: char| ch == '"' || ch == '\'' || ch == '「' || ch == '」');
            if name.chars().count() >= ext.len() + 1 {
                return name.to_string();
            }
        }
    }
    String::new()
}

fn is_filename_break(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '，' | '。' | '；' | '：' | ',' | ';' | ':' | '、' | '《' | '》' | '（' | '）' | '(' | ')')
}

/// 对齐 Python `_default_action_arguments`。
pub fn default_action_arguments(tool_name: &str, query: &str) -> Value {
    match tool_name {
        "search_knowledge_base" => json!({
            "query": query,
            "top_k": 5,
            "retrieval_method": "hybrid",
        }),
        "memory_query" => json!({
            "query": query,
            "top_k": 5,
            "memory_scope": "all",
        }),
        "memory_retrieve_original" => json!({
            "query": query,
            "top_k": 8,
            "max_chars": 12000,
        }),
        "document_hot_word_statistics" => {
            let mut defaults = json!({
                "top_k": 10,
                "include_samples": true,
            });
            let file_name = extract_file_name_hint(query);
            if !file_name.is_empty() {
                defaults["file_name"] = json!(file_name);
            }
            defaults
        }
        _ => json!({}),
    }
}

/// 对齐 Python `_normalize_action_arguments`：只补缺省，不覆盖调用方已给的键。
pub fn normalize_action_arguments(tool_name: &str, arguments: &Value, query: &str) -> Value {
    let mut normalized = if arguments.is_object() {
        arguments.clone()
    } else {
        json!({})
    };
    let defaults = default_action_arguments(tool_name, query);
    if let (Some(target), Some(defaults)) = (normalized.as_object_mut(), defaults.as_object()) {
        for (key, value) in defaults {
            target.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
    normalized
}

pub fn should_disclose_original_memory(query: &str) -> bool {
    let normalized: String = query
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>()
        .to_lowercase();
    !normalized.is_empty() && ORIGINAL_MEMORY_HINTS.iter().any(|hint| normalized.contains(hint))
}

fn matched_memory_ids(result: &Value) -> Vec<String> {
    result
        .get("matched_memories")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    item.get("memory_id")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// 对齐 Python `_run_agent_tool_turn` 的第二次工具：独立 `memory_retrieve_original`。
pub struct ProgressiveDisclosure {
    pub arguments: Value,
    pub result: Value,
}

/// 对齐 Python `SkillExecutionContext`：聊天回填请求 provider/model，HTTP execute 保持空。
#[derive(Clone, Debug)]
pub struct SkillExecutionContext<'a> {
    pub user_id: &'a str,
    pub conversation_id: Option<&'a str>,
    pub session_id: Option<&'a str>,
    pub query: Option<&'a str>,
    pub provider: Option<&'a str>,
    pub model: Option<&'a str>,
    pub trace_id: Option<&'a str>,
    pub metadata: Value,
}

impl<'a> SkillExecutionContext<'a> {
    pub fn new(user_id: &'a str) -> Self {
        Self {
            user_id,
            conversation_id: None,
            session_id: None,
            query: None,
            provider: None,
            model: None,
            trace_id: None,
            metadata: json!({}),
        }
    }

    /// 对齐 Python `execute_skill` 的 `context_payload`：含 provider/model，metadata 去掉 memory_context。
    pub fn public_payload(&self) -> Value {
        json!({
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "query": self.query,
            "provider": self.provider,
            "model": self.model,
            "trace_id": self.trace_id,
            "metadata": public_context_metadata(self.metadata.clone()),
        })
    }
}

pub async fn maybe_run_progressive_disclosure(
    pool: &sqlx::SqlitePool,
    context: &SkillExecutionContext<'_>,
    query: &str,
    memory_query_result: &Value,
    manifests: &[Value],
    allow_conversation_fallback: bool,
) -> Result<Option<ProgressiveDisclosure>, String> {
    if !should_disclose_original_memory(query) {
        return Ok(None);
    }
    let memory_ids = matched_memory_ids(memory_query_result);
    let disclosure_arguments = if !memory_ids.is_empty() {
        json!({
            "memory_ids": memory_ids,
            "include_neighbors": 1,
            "top_k": 8,
            "max_chars": 12000,
        })
    } else if allow_conversation_fallback {
        let conversation_id = context.conversation_id.unwrap_or("").trim();
        if conversation_id.is_empty() {
            return Ok(None);
        }
        json!({
            "conversation_id": conversation_id,
            "include_all_conversation": true,
            "top_k": 8,
            "max_chars": 12000,
        })
    } else {
        return Ok(None);
    };
    let result = execute_named_from(
        pool,
        "memory_retrieve_original",
        &disclosure_arguments,
        context,
        manifests,
        &[],
        None,
    )
    .await?;
    Ok(Some(ProgressiveDisclosure {
        arguments: disclosure_arguments,
        result,
    }))
}

pub async fn maybe_disclose_original_memory(
    pool: &sqlx::SqlitePool,
    context: &SkillExecutionContext<'_>,
    query: &str,
    memory_query_result: Value,
    manifests: &[Value],
    allow_conversation_fallback: bool,
) -> Result<Value, String> {
    match maybe_run_progressive_disclosure(
        pool,
        context,
        query,
        &memory_query_result,
        manifests,
        allow_conversation_fallback,
    )
    .await?
    {
        Some(disclosure) => Ok(merge_progressive_disclosure_payload(
            memory_query_result,
            disclosure.result,
        )),
        None => Ok(memory_query_result),
    }
}

/// 对齐 Python `_run_skill_action`：generate 把 memory_query 与原文合并进同一 result。
pub fn merge_progressive_disclosure_payload(memory_query: Value, original: Value) -> Value {
    json!({
        "memory_query": memory_query,
        "original_conversation": original,
        "progressive_disclosure": true,
    })
}

pub async fn execute_named(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    name: &str,
    arguments: &Value,
) -> Result<Value, String> {
    let context = SkillExecutionContext::new(user_id);
    execute_named_from(pool, name, arguments, &context, &[], &[], None).await
}

/// 对齐 Python `logger.exception(f"Skill 执行失败: {name}")`。
fn format_skill_exec_fail(name: &str) -> String {
    format!("Skill 执行失败: {name}")
}

pub async fn execute_named_from(
    pool: &sqlx::SqlitePool,
    name: &str,
    arguments: &Value,
    context: &SkillExecutionContext<'_>,
    manifests: &[Value],
    recent_history: &[crate::short_memory::MemoryMessage],
    embedder: Option<&crate::embedder::Embedder>,
) -> Result<Value, String> {
    let definition = get_skill_definition_from(name, manifests);
    let kind = crate::trace::trace_kind(definition.as_ref().map(|item| item.kind.as_str()), name);
    let display_name = definition
        .as_ref()
        .map(|item| item.name.as_str())
        .unwrap_or(name);
    let mut arguments = arguments.clone();
    if name == "memory_retrieve_original" {
        let existing = arg_str(&arguments, "conversation_id");
        if existing.is_empty() {
            if let Some(conversation_id) = context
                .conversation_id
                .map(str::trim)
                .filter(|item| !item.is_empty())
            {
                arguments["conversation_id"] = json!(conversation_id);
            }
        }
    }
    let arguments = &arguments;
    let context_payload = context.public_payload();
    crate::trace::log_tool_call(kind, display_name, arguments, &context_payload);
    if let Some(definition) = definition.as_ref() {
        if definition.kind == "remote_mcp" {
            let err = crate::remote_mcp::execution_error_message(definition);
            crate::trace::log_tool_error(kind, display_name, &err, &context_payload);
            return Err(err);
        }
    }
    let result = match name {
        "list_available_tools" => Ok(list_available_tools(manifests)),
        "list_knowledge_base_files" => list_knowledge_base_files(pool, arguments).await,
        "document_hot_word_statistics" => hot_words(pool, arguments).await,
        "memory_query" => {
            memory_query(
                pool,
                context.user_id,
                arguments,
                context.conversation_id,
                context.session_id,
                recent_history,
                embedder,
            )
            .await
        }
        "memory_retrieve_original" => memory_original(pool, context.user_id, arguments).await,
        "delete_profile_memory" => delete_profile_memory(pool, context.user_id, arguments).await,
        "search_knowledge_base" => Err("检索请走 /retrieval 或聊天接口".into()),
        other => Err(format!("未知技能: {other}")),
    };
    match result {
        Ok(payload) => {
            crate::trace::log_tool_result(
                kind,
                display_name,
                &crate::trace::skill_result_summary(&payload),
                &context_payload,
            );
            Ok(payload)
        }
        Err(err) => {
            crate::trace::log_tool_error(kind, display_name, &err, &context_payload);
            if definition.is_some() {
                tracing::error!("{}", format_skill_exec_fail(display_name));
            }
            Err(err)
        }
    }
}

pub async fn answer_routed_skill(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    route: &RoutedSkill,
) -> Result<String, String> {
    let result = execute_named(pool, user_id, route.name, &route.arguments).await?;
    Ok(format_skill_answer(route.name, &result))
}

pub fn format_skill_answer(name: &str, result: &Value) -> String {
    match name {
        "list_available_tools" => format_tools(result),
        "list_knowledge_base_files" => format_files(result),
        "document_hot_word_statistics" => format_hot_words(result),
        "memory_query" => format_memory_query(result),
        "memory_retrieve_original" => format_original(result),
        "delete_profile_memory" => result
            .get("summary")
            .and_then(|value| value.as_str())
            .unwrap_or("已处理画像记忆删除请求。")
            .to_string(),
        _ => result.to_string(),
    }
}

fn format_tools(result: &Value) -> String {
    let tools = result.get("tools").and_then(|value| value.as_array());
    let Some(tools) = tools else {
        return "当前没有可展示的工具。".into();
    };
    let mut lines = vec![format!("当前可用工具共 {} 个：", tools.len())];
    for tool in tools {
        let name = tool.get("name").and_then(|value| value.as_str()).unwrap_or("");
        let description = tool.get("description").and_then(|value| value.as_str()).unwrap_or("");
        lines.push(format!("- {name}：{description}"));
    }
    lines.join("\n")
}

fn format_files(result: &Value) -> String {
    let count = result
        .pointer("/summary/document_count")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    if count == 0 {
        return result
            .pointer("/summary/message")
            .and_then(|value| value.as_str())
            .unwrap_or("未找到匹配的已上传文档。")
            .to_string();
    }
    let mut lines = vec![format!("知识库中匹配到 {count} 份文档：")];
    let documents = result.get("documents").and_then(|value| value.as_array()).cloned().unwrap_or_default();
    for (index, doc) in documents.iter().enumerate() {
        let name = doc.get("original_filename").and_then(|value| value.as_str()).unwrap_or("未命名");
        let file_type = doc.get("file_type").and_then(|value| value.as_str()).unwrap_or("");
        let chunks = doc.get("chunk_count").and_then(|value| value.as_i64()).unwrap_or(0);
        lines.push(format!("{}. {name}（{file_type}，{chunks} 个父块）", index + 1));
        if let Some(keywords) = doc.get("topic_keywords").and_then(|value| value.as_array()) {
            let words: Vec<&str> = keywords.iter().filter_map(|item| item.as_str()).take(6).collect();
            if !words.is_empty() {
                lines.push(format!("   关键词：{}", words.join("、")));
            }
        }
        if let Some(preview) = doc.get("content_preview").and_then(|value| value.as_str()) {
            let preview = preview.trim();
            if !preview.is_empty() {
                let short: String = preview.chars().take(180).collect();
                lines.push(format!("   预览：{short}"));
            }
        }
    }
    lines.join("\n")
}

fn format_hot_words(result: &Value) -> String {
    let count = result
        .pointer("/summary/document_count")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    if count == 0 {
        return "没有找到可统计的文档。".into();
    }
    let words = result.get("top_words").and_then(|value| value.as_array()).cloned().unwrap_or_default();
    if words.is_empty() {
        return format!("已检查 {count} 份文档，没有统计到有效高频词。");
    }
    let mut lines = vec![format!("在 {count} 份文档中，高频词如下：")];
    for (index, word) in words.iter().enumerate() {
        let term = word.get("term").and_then(|value| value.as_str()).unwrap_or("");
        let freq = word.get("document_count").and_then(|value| value.as_u64()).unwrap_or(0);
        lines.push(format!("{}. {term}（{freq}）", index + 1));
    }
    lines.join("\n")
}

fn format_memory_query(result: &Value) -> String {
    let disclosed = result
        .get("progressive_disclosure")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let payload = if disclosed {
        result.get("memory_query").unwrap_or(result)
    } else {
        result
    };
    let profiles = payload.get("profile_memories").and_then(|value| value.as_array()).cloned().unwrap_or_default();
    let matched = payload.get("matched_memories").and_then(|value| value.as_array()).cloned().unwrap_or_default();
    if profiles.is_empty() && matched.is_empty() && !disclosed {
        return "当前账号还没有可用的画像或长期记忆。".into();
    }
    let mut lines = Vec::new();
    if !profiles.is_empty() {
        lines.push("画像记忆：".into());
        for item in &profiles {
            let key = item.get("memory_key").and_then(|value| value.as_str()).unwrap_or("profile");
            let content = item.get("content").and_then(|value| value.as_str()).unwrap_or("");
            lines.push(format!("- {key}：{content}"));
        }
    }
    if !matched.is_empty() {
        lines.push("相关记忆：".into());
        for item in matched.iter().take(5) {
            let content = item.get("content").and_then(|value| value.as_str()).unwrap_or("");
            let short: String = content.chars().take(240).collect();
            lines.push(format!("- {short}"));
        }
    }
    if disclosed {
        if let Some(original) = result.get("original_conversation") {
            let expanded = format_original(original);
            if !expanded.is_empty() {
                if !lines.is_empty() {
                    lines.push(String::new());
                }
                lines.push(expanded);
            }
        }
    }
    if lines.is_empty() {
        "当前账号还没有可用的画像或长期记忆。".into()
    } else {
        lines.join("\n")
    }
}

fn format_original(result: &Value) -> String {
    let chunks = result
        .get("original_chunks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let joined = if chunks.is_empty() {
        result.get("content").and_then(Value::as_str).unwrap_or("").trim().to_string()
    } else {
        chunks
            .iter()
            .filter_map(|item| item.get("content").and_then(Value::as_str))
            .filter(|item| !item.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    if joined.trim().is_empty() {
        return result
            .get("summary")
            .and_then(Value::as_str)
            .filter(|item| !item.trim().is_empty())
            .unwrap_or("没有找到可展开的原始对话。")
            .to_string();
    }
    format!("按你的要求展开原始对话：\n\n{joined}")
}

#[derive(Clone)]
pub struct SkillDefinition {
    pub name: String,
    pub description: String,
    pub category: String,
    pub kind: String,
    pub tags: Vec<String>,
    pub provider: Option<String>,
    pub parameters: Value,
    pub metadata: Value,
}

impl SkillDefinition {
    pub fn executor_ready(&self) -> bool {
        self.metadata
            .get("executor_ready")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }

    fn to_function_calling(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        })
    }

    fn to_public_dict(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "kind": self.kind,
            "tags": self.tags,
            "provider": self.provider,
            "metadata": self.metadata,
            "function_calling": self.to_function_calling(),
        })
    }
}

fn skill_def(
    name: &str,
    description: &str,
    category: &str,
    tags: &[&str],
    parameters: Value,
    returns: &str,
) -> SkillDefinition {
    SkillDefinition {
        name: name.to_string(),
        description: description.to_string(),
        category: category.to_string(),
        kind: "local".into(),
        tags: tags.iter().map(|item| item.to_string()).collect(),
        provider: None,
        parameters,
        // 对齐 Python 本地技能 metadata：只有 returns；executor_ready 缺省视为 true。
        metadata: json!({"returns": returns}),
    }
}

pub fn rag_function_calling_tools() -> Vec<Value> {
    vec![rag_tool_definition()]
}

fn rag_tool_definition() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "检索企业知识库，适用于文档内容、制度流程、上传资料、内部事实核验，以及按文件名/文档名查询某份文档是否存在。",
            "parameters": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于知识库检索的改写问题。当用户提到文件名/文档名时，可直接用文件名作为检索词。"
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 5,
                        "description": "返回多少条检索结果。"
                    },
                    "retrieval_method": {
                        "type": "string",
                        "enum": ["hybrid", "dense", "sparse"],
                        "default": "hybrid",
                        "description": "检索方式。"
                    }
                },
                "required": ["query"]
            }
        }
    })
}

fn local_skill_catalog() -> Vec<SkillDefinition> {
    let mut skills = vec![
        skill_def(
            "list_available_tools",
            "展示当前 Agent Function Calling 注册表中的全部可用工具和 Skill，包括知识库检索、本地 Skill 以及已注册的远程 MCP Skill。适用于用户询问『列出你的工具』『你有什么技能』『有哪些可用能力』等问题。",
            "system",
            &["tools", "skills", "function_calling", "registry"],
            json!({"type": "object", "additionalProperties": false, "properties": {}}),
            "registered_agent_tools",
        ),
        skill_def(
            "document_hot_word_statistics",
            "按文件名或文档 ID 定位文档并统计高频热词。适用于用户询问具体文件中的热词、关键词或出现频率；不用于回答文档正文、制度流程或内部事实。",
            "document",
            &["statistics", "documents", "keywords"],
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10, "description": "返回前多少个热点词。"},
                    "file_name": {"type": "string", "description": "可选，按原始文件名或存储文件名模糊定位目标文档；用户提到文件名时优先填写。"},
                    "document_ids": {"type": "array", "items": {"type": "string"}, "description": "可选，只统计指定文档。"},
                    "file_types": {"type": "array", "items": {"type": "string"}, "description": "可选，只统计指定文件类型。"},
                    "statuses": {"type": "array", "items": {"type": "string", "enum": ["processing", "ready", "failed", "deleting", "deleted"]}, "description": "可选，只统计指定状态。"},
                    "include_samples": {"type": "boolean", "default": true, "description": "是否返回示例文档。"}
                }
            }),
            "hot_word_statistics",
        ),
        skill_def(
            "memory_query",
            "查询当前用户自己的长期记忆或画像记忆。适用于‘你记得什么’、‘我之前说过什么’、‘我的偏好是什么’等问题；不用于查询企业文档，也不能跨用户读取记忆。",
            "memory",
            &["memory", "profile", "search"],
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "query": {"type": "string", "description": "可选，按语义召回长期记忆时使用。"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5, "description": "最多返回多少条记忆。"},
                    "memory_scope": {"type": "string", "enum": ["all", "profile", "event"], "default": "all", "description": "记忆范围：画像记忆、事件记忆或两者都查。"}
                }
            }),
            "memory_query_result",
        ),
        {
            let mut skill = skill_def(
                "memory_retrieve_original",
                "按长期记忆摘要的引用渐进式披露当前用户对应的原始对话。当摘要不足以回答问题、用户要求查看原文/完整上下文/具体细节，或需要核对原始措辞时使用；只能读取当前用户自己的对话。",
                "memory",
                &["memory", "conversation", "original", "progressive_disclosure"],
                json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "memory_ids": {"type": "array", "items": {"type": "string"}, "description": "可选，摘要注入中提供的 memory_id 列表；优先使用。"},
                        "query": {"type": "string", "description": "可选，摘要 memory_id 不明确时用于语义定位原始对话。"},
                        "conversation_id": {"type": "string", "description": "可选，限定某个会话；默认使用当前会话。"},
                        "turn_start": {"type": "integer", "minimum": 0, "description": "可选，原始对话起始轮次。"},
                        "turn_end": {"type": "integer", "minimum": 0, "description": "可选，原始对话结束轮次。"},
                        "include_neighbors": {"type": "integer", "minimum": 0, "maximum": 5, "default": 0, "description": "可选，在定位范围前后各多取几块上下文。"},
                        "include_all_conversation": {"type": "boolean", "default": false, "description": "可选，无摘要引用时是否展开当前会话的全部原始对话。"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 8, "description": "最多展开多少个摘要对应的原始块。"},
                        "max_chars": {"type": "integer", "minimum": 500, "maximum": 50000, "default": 12000, "description": "本次最多返回的原文字符数。"}
                    }
                }),
                "original_conversation_result",
            );
            skill.metadata["progressive_disclosure"] = json!(true);
            skill
        },
        skill_def(
            "delete_profile_memory",
            "删除用户的个性化记忆（画像记忆）。当用户明确要求删除、遗忘、清除某条个人信息（如名字、偏好、地点、习惯等）时使用。可指定 memory_key 精确删除，或提供 keywords 按内容模糊匹配删除；用户表达模糊时先查询并确认，不能删除事件记忆或其他用户数据。",
            "memory",
            &["memory", "profile", "delete", "privacy"],
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "memory_keys": {"type": "array", "items": {"type": "string"}, "description": "可选，精确删除的 memory_key 列表，如 profile.location、profile.name。"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "可选，按内容关键词模糊匹配要删除的记忆，如 ['北京']。"},
                    "max_deletes": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50, "description": "单次最多删除条数，超过会拒绝并要求更精确条件。"}
                }
            }),
            "delete_profile_memory_result",
        ),
        skill_def(
            "list_knowledge_base_files",
            "列出知识库中已上传的文档，可按文件名、文档 ID、文件类型或状态筛选；并可概括指定文档的内容（章节大纲、内容预览、主题关键词）。适用于『知识库里有哪些文件』『帮我看看 04_产品与账号常见问题 这个文档讲了什么』『列出全部已上传文档』『这份文档包含哪些章节』等需求；不用于按内容检索具体答案，那应使用 search_knowledge_base。",
            "document",
            &["documents", "list", "summary", "overview"],
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "file_name": {"type": "string", "description": "可选，按文件名模糊筛选；用户提到具体文件名时填写，如『产品与账号常见问题』。"},
                    "document_ids": {"type": "array", "items": {"type": "string"}, "description": "可选，只列出指定 document_id 的文档。"},
                    "file_types": {"type": "array", "items": {"type": "string"}, "description": "可选，只列出指定文件类型，如 ['pdf','docx','md']。"},
                    "statuses": {"type": "array", "items": {"type": "string", "enum": ["processing", "ready", "failed", "deleting", "deleted"]}, "description": "可选，按状态筛选，默认只列已入库(ready)的文档。"},
                    "summarize": {"type": "boolean", "default": true, "description": "是否概括内容（章节大纲、内容预览、主题关键词）。"},
                    "preview_chars": {"type": "integer", "minimum": 200, "maximum": 6000, "default": 1500, "description": "每个文档内容预览的最大字符数。"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20, "description": "最多返回多少条文档。"}
                }
            }),
            "knowledge_base_files",
        ),
    ];
    skills.sort_by(|left, right| left.name.cmp(&right.name));
    skills
}

fn merged_skill_catalog(manifests: &[Value]) -> Vec<SkillDefinition> {
    let mut by_name = BTreeMap::new();
    for skill in local_skill_catalog() {
        by_name.insert(skill.name.clone(), skill);
    }
    for skill in crate::remote_mcp::skills_from_manifests(manifests) {
        by_name.insert(skill.name.clone(), skill);
    }
    by_name.into_values().collect()
}

pub fn get_skill_definition(name: &str) -> Option<SkillDefinition> {
    get_skill_definition_from(name, &[])
}

pub fn get_skill_definition_from(name: &str, manifests: &[Value]) -> Option<SkillDefinition> {
    let normalized = name.trim();
    merged_skill_catalog(manifests)
        .into_iter()
        .find(|item| item.name == normalized)
}

/// GET /skills：技能定义数组（不含 RAG 工具）。
pub fn list_skill_definitions() -> Vec<Value> {
    list_skill_definitions_from(&[])
}

pub fn list_skill_definitions_from(manifests: &[Value]) -> Vec<Value> {
    merged_skill_catalog(manifests)
        .iter()
        .map(SkillDefinition::to_public_dict)
        .collect()
}

/// GET /skills/function-calling：RAG + 已就绪 Skill 的 Function Calling schema。
pub fn function_calling_tools() -> Vec<Value> {
    function_calling_tools_from(&[])
}

pub fn function_calling_tools_from(manifests: &[Value]) -> Vec<Value> {
    let mut tools = vec![rag_tool_definition()];
    tools.extend(
        merged_skill_catalog(manifests)
            .iter()
            .filter(|skill| skill.executor_ready())
            .map(SkillDefinition::to_function_calling),
    );
    tools
}

/// GET /skills/capabilities：统一能力目录（含 RAG，未就绪 MCP 标 available=false）。
pub fn list_agent_capabilities() -> Vec<Value> {
    list_agent_capabilities_from(&[])
}

pub fn list_agent_capabilities_from(manifests: &[Value]) -> Vec<Value> {
    let rag = rag_tool_definition();
    let mut capabilities = vec![json!({
        "name": "search_knowledge_base",
        "description": rag.pointer("/function/description").cloned().unwrap_or(json!("")),
        "category": "retrieval",
        "kind": "tool",
        "source": "rag",
        "available": true,
        "location": "app.tools.rag.run_rag_tool",
        "function_calling": rag,
    })];
    for skill in merged_skill_catalog(manifests) {
        let mut item = skill.to_public_dict();
        if let Some(object) = item.as_object_mut() {
            object.insert("source".into(), json!("skill"));
            object.insert("available".into(), json!(skill.executor_ready()));
            object.insert(
                "location".into(),
                json!(format!("backend/app/skills/{0}/{0}.md", skill.name)),
            );
        }
        capabilities.push(item);
    }
    capabilities
}

pub fn skill_capabilities_payload() -> Value {
    skill_capabilities_payload_from(&[])
}

pub fn skill_capabilities_payload_from(manifests: &[Value]) -> Value {
    let tools = function_calling_tools_from(manifests);
    json!({
        "capabilities": list_agent_capabilities_from(manifests),
        "tools": tools,
        "function_calling": tools,
    })
}

pub fn default_skills_dir() -> PathBuf {
    if let Ok(value) = std::env::var("EKS_SKILLS_DIR") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    candidate_skills_dirs()
        .into_iter()
        .find(|path| path.is_dir())
        .unwrap_or_else(|| PathBuf::from("app/skills"))
}

fn candidate_skills_dirs() -> Vec<PathBuf> {
    vec![
        // CWD=backend（标准运行方式）与 CWD=仓库根 两种情况
        PathBuf::from("app/skills"),
        PathBuf::from("backend/app/skills"),
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../app/skills"),
    ]
}

/// 对齐 Python `get_skill_document_path`：拒绝空名、`.` / `..` 与路径分隔符。
pub fn get_skill_document_path(skill_name: &str) -> Result<PathBuf, String> {
    get_skill_document_path_in(skill_name, &default_skills_dir())
}

fn get_skill_document_path_in(skill_name: &str, skills_root: &Path) -> Result<PathBuf, String> {
    let normalized = skill_name.trim();
    if normalized.is_empty()
        || matches!(normalized, "." | "..")
        || Path::new(normalized)
            .file_name()
            .and_then(|name| name.to_str())
            != Some(normalized)
    {
        return Err(format!("非法 Skill 名称: {skill_name}"));
    }
    Ok(skills_root.join(normalized).join(format!("{normalized}.md")))
}

pub fn read_skill_document(skill_name: &str) -> Result<String, String> {
    let definition =
        get_skill_definition(skill_name).ok_or_else(|| format!("未找到技能: {skill_name}"))?;
    read_skill_document_for_definition(&definition)
}

fn read_skill_document_for_definition(definition: &SkillDefinition) -> Result<String, String> {
    read_skill_document_for_definition_from(definition, &default_skills_dir())
}

fn read_skill_document_for_definition_from(
    definition: &SkillDefinition,
    skills_root: &Path,
) -> Result<String, String> {
    if definition.kind != "local" {
        return Ok(String::new());
    }
    let document_path = get_skill_document_path_in(&definition.name, skills_root)?;
    ensure_document_within_skills_root(skills_root, &document_path, &definition.name)?;
    if !document_path.is_file() {
        tracing::warn!(path = %document_path.display(), "Skill 详细文档不存在");
        return Ok(String::new());
    }
    std::fs::read_to_string(&document_path)
        .map_err(|err| format!("读取 Skill 文档失败 {}: {err}", document_path.display()))
}

fn ensure_document_within_skills_root(
    skills_root: &Path,
    document_path: &Path,
    skill_name: &str,
) -> Result<(), String> {
    let root = normalize_path(skills_root);
    let document = normalize_path(document_path);
    if document.starts_with(&root) && document != root {
        Ok(())
    } else {
        Err(format!("Skill 文档路径超出技能目录: {skill_name}"))
    }
}

fn normalize_path(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        // Windows canonicalize 会加 \\?\ 前缀，与未 canonicalize 的绝对路径比较会失配；
        // 去掉前缀保证两侧口径一致（Linux 无前缀，行为不变）。
        let text = canonical.to_string_lossy().into_owned();
        return PathBuf::from(text.strip_prefix(r"\\?\").unwrap_or(&text).to_string());
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// 对齐 Python `SkillNotFoundError(KeyError)`：HTTP 404 文案带引号。
pub fn skill_not_found_http_detail(name: &str) -> String {
    format!("未找到技能: '{name}'")
}

/// 对齐 Python `SkillExecuteResponse`：HTTP 不回传 `skill_instructions`，context 含 provider/model。
pub fn skill_execute_http_payload(wrapped: Value) -> Value {
    let mut payload = wrapped;
    if let Some(obj) = payload.as_object_mut() {
        obj.remove("skill_instructions");
        if let Some(context) = obj.get_mut("context").and_then(Value::as_object_mut) {
            context.entry("provider".to_string()).or_insert(Value::Null);
            context.entry("model".to_string()).or_insert(Value::Null);
        }
    }
    payload
}

fn public_context_metadata(metadata: Value) -> Value {
    match metadata {
        Value::Null => json!({}),
        Value::Object(mut map) => {
            map.remove("memory_context");
            Value::Object(map)
        }
        other => other,
    }
}

pub fn wrap_skill_execution(
    name: &str,
    arguments: &Value,
    context: &SkillExecutionContext<'_>,
    result: Value,
) -> Result<Value, String> {
    let definition = get_skill_definition(name).ok_or_else(|| format!("未找到技能: {name}"))?;
    let skill_instructions = read_skill_document_for_definition(&definition)?;
    let executed_at = crate::db::utc_isoformat_now();
    Ok(json!({
        "skill": definition.name,
        "category": definition.category,
        "kind": definition.kind,
        "provider": definition.provider,
        "arguments": arguments,
        "context": context.public_payload(),
        "skill_instructions": skill_instructions,
        "result": result,
        "executed_at": executed_at,
    }))
}

pub const SKILL_SYSTEM_PROMPT: &str = "\
你是企业知识库问答助手。当前问题已经调用了一个技能工具。\
如果消息中提供了当前 Skill 的详细规则，必须把它当作当前 Skill 的补充执行规范。\
只能依据技能返回结果回答，不要编造未返回的信息。\
如果技能结果为空，直接说明当前没有查到可用结果。\
默认用简洁中文回答，可以用短标题或列表，但不要重复原始 JSON。";

pub fn skill_result_message(skill_result: &Value) -> ChatMessage {
    let payload = json!({
        "skill": skill_result.get("skill"),
        "category": skill_result.get("category"),
        "kind": skill_result.get("kind"),
        "provider": skill_result.get("provider"),
        "result": skill_result.get("result").cloned().unwrap_or(json!({})),
        "executed_at": skill_result.get("executed_at"),
    });
    let serialized = serde_json::to_string(&payload).unwrap_or_else(|_| payload.to_string());
    ChatMessage::new("system", format!("技能执行结果：\n{serialized}"))
}

/// 对齐 Python `build_skill_messages`：系统规范 + Skill 文档 + 执行结果 + 记忆 + 历史 + 用户问题。
pub fn build_skill_messages(
    query: &str,
    skill_result: &Value,
    long_term_memory: &[ChatMessage],
    history: Vec<ChatMessage>,
) -> Vec<ChatMessage> {
    let mut messages = vec![ChatMessage::new("system", SKILL_SYSTEM_PROMPT)];
    if let Some(instructions) = skill_result
        .get("skill_instructions")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        messages.push(ChatMessage::new(
            "system",
            format!("当前 Skill 的详细规则（仅适用于本次已选中的 Skill）：\n{instructions}"),
        ));
    }
    messages.push(skill_result_message(skill_result));
    messages.extend(long_term_memory.iter().cloned());
    messages.extend(history);
    messages.push(ChatMessage::new("user", query.trim()));
    messages
}

fn list_available_tools(manifests: &[Value]) -> Value {
    let schemas = function_calling_tools_from(manifests);
    let capabilities: HashMap<String, Value> = list_agent_capabilities_from(manifests)
        .into_iter()
        .filter_map(|item| {
            let name = item.get("name").and_then(|value| value.as_str())?.to_string();
            Some((name, item))
        })
        .collect();
    let mut tools = Vec::new();
    let mut categories: HashMap<String, u64> = HashMap::new();
    for schema in schemas {
        let function = schema.get("function").cloned().unwrap_or(json!({}));
        let name = function
            .get("name")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if name.is_empty() {
            continue;
        }
        let capability = capabilities.get(&name).cloned().unwrap_or(json!({}));
        let category = capability
            .get("category")
            .and_then(|value| value.as_str())
            .unwrap_or("general")
            .to_string();
        *categories.entry(category.clone()).or_insert(0) += 1;
        tools.push(json!({
            "name": name,
            "description": capability.get("description").cloned().unwrap_or_else(|| function.get("description").cloned().unwrap_or(json!(""))),
            "category": category,
            "kind": capability.get("kind").cloned().unwrap_or(json!("skill")),
            "source": capability.get("source").cloned().unwrap_or(json!("skill")),
            "provider": capability.get("provider").cloned().unwrap_or(Value::Null),
            "location": capability.get("location").cloned().unwrap_or(Value::Null),
            "parameters": function.get("parameters").cloned().unwrap_or(json!({})),
        }));
    }
    json!({
        "summary": {
            "tool_count": tools.len(),
            "categories": categories,
            "message": "以下是当前统一 Agent Function Calling 注册表中的全部工具。",
        },
        "tools": tools,
    })
}

pub async fn load_active_profile_lines(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    limit: usize,
) -> Result<Vec<String>, String> {
    Ok(load_memories(pool, user_id, Some("profile"), 100)
        .await?
        .into_iter()
        .filter(memory_active)
        .take(limit)
        .map(|row| {
            let key = row.memory_key.unwrap_or_else(|| "profile".into());
            format!("- {key}: {}", row.content)
        })
        .collect())
}

pub async fn soft_delete_profile_key(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    memory_key: &str,
) -> Result<usize, sqlx::Error> {
    let now = crate::db::fmt_dt(&crate::db::now_naive());
    let result = sqlx::query(
        "UPDATE memory_records SET status = 'deleted', updated_at = ? \
         WHERE user_id = ? AND memory_type = 'profile' AND memory_key = ? AND status = 'active'",
    )
    .bind(&now)
    .bind(user_id)
    .bind(memory_key)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() as usize)
}

pub async fn upsert_profile_memory(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    memory_key: &str,
    content: &str,
    conversation_id: &str,
    session_id: &str,
    importance: f64,
    confidence: f64,
) -> Result<(), sqlx::Error> {
    let now = crate::db::fmt_dt(&crate::db::now_naive());
    let current: Option<(String, i64, String)> = sqlx::query_as(
        "SELECT memory_id, version, content FROM memory_records \
         WHERE user_id = ? AND memory_type = 'profile' AND memory_key = ? AND status = 'active' \
         ORDER BY version DESC LIMIT 1",
    )
    .bind(user_id)
    .bind(memory_key)
    .fetch_optional(pool)
    .await?;
    if let Some((memory_id, version, old_content)) = current {
        if old_content.trim() == content.trim() {
            sqlx::query(
                "UPDATE memory_records SET importance_score = ?, base_importance_score = ?, \
                 confidence_score = ?, updated_at = ? WHERE memory_id = ?",
            )
            .bind(importance)
            .bind(importance)
            .bind(confidence)
            .bind(&now)
            .bind(&memory_id)
            .execute(pool)
            .await?;
            return Ok(());
        }
        sqlx::query("UPDATE memory_records SET status = 'superseded', updated_at = ? WHERE memory_id = ?")
            .bind(&now)
            .bind(&memory_id)
            .execute(pool)
            .await?;
        let memory_id_new = format!("profile:{user_id}:{memory_key}:{}", uuid::Uuid::new_v4().simple());
        sqlx::query(
            "INSERT INTO memory_records (
                memory_id, user_id, memory_type, memory_key, content, conversation_id, session_id,
                importance_score, base_importance_score, confidence_score, access_count, status,
                version, supersedes_memory_id, created_at, updated_at
            ) VALUES (?, ?, 'profile', ?, ?, ?, ?, ?, ?, ?, 0, 'active', ?, ?, ?, ?)",
        )
        .bind(&memory_id_new)
        .bind(user_id)
        .bind(memory_key)
        .bind(content)
        .bind(if conversation_id.is_empty() { None } else { Some(conversation_id) })
        .bind(if session_id.is_empty() { None } else { Some(session_id) })
        .bind(importance)
        .bind(importance)
        .bind(confidence)
        .bind(version + 1)
        .bind(&memory_id)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await?;
        return Ok(());
    }
    let memory_id = format!("profile:{user_id}:{memory_key}:{}", uuid::Uuid::new_v4().simple());
    sqlx::query(
        "INSERT INTO memory_records (
            memory_id, user_id, memory_type, memory_key, content, conversation_id, session_id,
            importance_score, base_importance_score, confidence_score, access_count, status,
            version, created_at, updated_at
        ) VALUES (?, ?, 'profile', ?, ?, ?, ?, ?, ?, ?, 0, 'active', 1, ?, ?)",
    )
    .bind(&memory_id)
    .bind(user_id)
    .bind(memory_key)
    .bind(content)
    .bind(if conversation_id.is_empty() { None } else { Some(conversation_id) })
    .bind(if session_id.is_empty() { None } else { Some(session_id) })
    .bind(importance)
    .bind(importance)
    .bind(confidence)
    .bind(&now)
    .bind(&now)
    .execute(pool)
    .await?;
    Ok(())
}

#[derive(Debug, sqlx::FromRow)]
struct SkillDocument {
    document_id: String,
    original_filename: String,
    stored_filename: String,
    file_type: String,
    status: String,
    file_size: i64,
    chunk_count: i64,
    created_at: String,
    updated_at: String,
}

/// 对齐 Python `_skill_list_knowledge_base_files`：默认 ready、limit 1–100，
/// 支持 filename/status 别名，filters 含 document_ids/file_types。
async fn list_knowledge_base_files(pool: &sqlx::SqlitePool, arguments: &Value) -> Result<Value, String> {
    let file_name = skill_file_name(arguments);
    let limit = arg_usize(arguments, "limit", 20, 1, 100);
    let summarize = arg_bool(arguments, "summarize", true);
    let preview_chars = arg_usize(arguments, "preview_chars", 1500, 200, 6000);
    let mut statuses = skill_status_filters(arguments);
    if statuses.is_empty() {
        statuses = vec!["ready".to_string()];
    }
    let file_types = arg_strings(arguments, "file_types");
    let document_ids = arg_strings(arguments, "document_ids");
    let filters = json!({
        "document_ids": document_ids,
        "file_types": file_types,
        "statuses": statuses,
        "file_name": file_name,
        "limit": limit,
    });

    let mut sql = String::from(
        "SELECT document_id, original_filename, stored_filename, file_type, status, \
         file_size, chunk_count, created_at, updated_at FROM documents WHERE 1=1",
    );
    let mut binds: Vec<String> = Vec::new();
    push_in_clause(&mut sql, &mut binds, "status", &statuses);
    push_in_clause(&mut sql, &mut binds, "file_type", &file_types);
    push_in_clause(&mut sql, &mut binds, "document_id", &document_ids);
    if !file_name.is_empty() {
        sql.push_str(" AND (original_filename LIKE ? OR stored_filename LIKE ?)");
        let pattern = format!("%{file_name}%");
        binds.push(pattern.clone());
        binds.push(pattern);
    }
    sql.push_str(" ORDER BY original_filename ASC LIMIT ?");

    let mut query = sqlx::query_as::<_, SkillDocument>(&sql);
    for bind in &binds {
        query = query.bind(bind);
    }
    query = query.bind(limit as i64);
    let documents = query.fetch_all(pool).await.map_err(|err| err.to_string())?;
    if documents.is_empty() {
        return Ok(json!({
            "summary": {"document_count": 0, "message": "未找到匹配的已上传文档，请确认文件名或状态。"},
            "documents": [],
            "filters": filters,
        }));
    }

    let ids: Vec<String> = documents.iter().map(|record| record.document_id.clone()).collect();
    let chunks_by_document = if summarize {
        load_chunks_for_documents(pool, &ids).await?
    } else {
        HashMap::new()
    };

    let mut documents_out = Vec::new();
    for record in &documents {
        let mut doc = skill_document_json(record);
        if summarize {
            let chunks = chunks_by_document
                .get(&record.document_id)
                .cloned()
                .unwrap_or_default();
            let combined = chunks.join("\n");
            let outline: Vec<String> = combined
                .lines()
                .filter_map(|line| line.trim().strip_prefix("[章节]"))
                .map(|line| line.trim().trim_matches(|ch: char| ch == ':' || ch == '：').to_string())
                .filter(|line| !line.is_empty())
                .collect();
            let preview: String = combined.chars().take(preview_chars).collect();
            let keywords = top_terms(&combined, 10);
            doc["outline"] = json!(outline);
            doc["content_preview"] = json!(preview);
            doc["topic_keywords"] = json!(keywords);
        }
        documents_out.push(doc);
    }
    let chunk_count: i64 = documents.iter().map(|record| record.chunk_count).sum();
    Ok(json!({
        "summary": {"document_count": documents_out.len(), "chunk_count": chunk_count},
        "documents": documents_out,
        "filters": filters,
    }))
}

fn skill_file_name(arguments: &Value) -> String {
    let name = arg_str(arguments, "file_name");
    if name.is_empty() {
        arg_str(arguments, "filename")
    } else {
        name
    }
}

fn skill_status_filters(arguments: &Value) -> Vec<String> {
    let statuses = arg_strings(arguments, "statuses");
    if statuses.is_empty() {
        arg_strings(arguments, "status")
    } else {
        statuses
    }
}

fn skill_document_json(record: &SkillDocument) -> Value {
    json!({
        "document_id": record.document_id,
        "original_filename": record.original_filename,
        "stored_filename": record.stored_filename,
        "file_type": record.file_type,
        "status": record.status,
        "file_size": record.file_size,
        "chunk_count": record.chunk_count,
        "created_at": crate::db::dt_to_iso(&record.created_at),
        "updated_at": crate::db::dt_to_iso(&record.updated_at),
    })
}

/// 对齐 Python `_skill_document_hot_words`：不过滤默认 ready、不加 100 条上限，
/// 词频文本包含 file_type/status，响应带 filters。
async fn hot_words(pool: &sqlx::SqlitePool, arguments: &Value) -> Result<Value, String> {
    let top_k = arg_usize(arguments, "top_k", 10, 1, 50);
    let file_name = skill_file_name(arguments);
    let document_ids = arg_strings(arguments, "document_ids");
    let file_types = arg_strings(arguments, "file_types");
    let statuses = skill_status_filters(arguments);
    let include_samples = arg_bool(arguments, "include_samples", true);
    let filters = json!({
        "document_ids": document_ids,
        "file_types": file_types,
        "statuses": statuses,
        "file_name": file_name,
    });

    let mut sql = String::from(
        "SELECT document_id, original_filename, stored_filename, file_type, status, \
         file_size, chunk_count, created_at, updated_at FROM documents WHERE 1=1",
    );
    let mut binds: Vec<String> = Vec::new();
    push_in_clause(&mut sql, &mut binds, "status", &statuses);
    push_in_clause(&mut sql, &mut binds, "file_type", &file_types);
    push_in_clause(&mut sql, &mut binds, "document_id", &document_ids);
    if !file_name.is_empty() {
        sql.push_str(" AND (original_filename LIKE ? OR stored_filename LIKE ?)");
        let pattern = format!("%{file_name}%");
        binds.push(pattern.clone());
        binds.push(pattern);
    }
    sql.push_str(" ORDER BY updated_at DESC");

    let mut query = sqlx::query_as::<_, SkillDocument>(&sql);
    for bind in &binds {
        query = query.bind(bind);
    }
    let documents = query.fetch_all(pool).await.map_err(|err| err.to_string())?;
    if documents.is_empty() {
        return Ok(json!({
            "scope": "documents",
            "filters": filters,
            "summary": {"document_count": 0, "chunk_count": 0},
            "top_words": [],
        }));
    }

    let ids: Vec<String> = documents.iter().map(|record| record.document_id.clone()).collect();
    let chunks_by_document = load_chunks_for_documents(pool, &ids).await?;

    let mut term_frequency: HashMap<String, usize> = HashMap::new();
    let mut term_documents: HashMap<String, Vec<Value>> = HashMap::new();
    for record in &documents {
        let mut parts = vec![record.file_type.clone(), record.status.clone()];
        if let Some(chunks) = chunks_by_document.get(&record.document_id) {
            parts.extend(chunks.iter().cloned());
        }
        let combined = parts
            .into_iter()
            .filter(|part| !part.trim().is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        let limited: String = combined.chars().take(12000).collect();
        let document_term_frequency = extract_term_frequencies(&limited);
        for (term, count) in document_term_frequency {
            *term_frequency.entry(term.clone()).or_default() += count;
            if include_samples {
                let samples = term_documents.entry(term).or_default();
                if samples.len() < 3
                    && !samples.iter().any(|item| {
                        item.get("document_id").and_then(Value::as_str) == Some(record.document_id.as_str())
                    })
                {
                    samples.push(skill_document_json(record));
                }
            }
        }
    }

    let mut ranked: Vec<(String, usize)> = term_frequency.into_iter().collect();
    // 对齐 Python 排序键 (-count, -len(term), term)：len 按字符数而非字节数。
    ranked.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| right.0.chars().count().cmp(&left.0.chars().count()))
            .then_with(|| left.0.cmp(&right.0))
    });
    let top_words: Vec<Value> = ranked
        .into_iter()
        .take(top_k)
        .map(|(term, count)| {
            json!({
                "term": term,
                "document_count": count,
                "sample_documents": term_documents.get(&term).cloned().unwrap_or_default(),
            })
        })
        .collect();
    let chunk_count: i64 = documents.iter().map(|record| record.chunk_count).sum();
    Ok(json!({
        "scope": "documents",
        "filters": filters,
        "summary": {"document_count": documents.len(), "chunk_count": chunk_count},
        "top_words": top_words,
    }))
}

#[derive(Debug, sqlx::FromRow)]
struct MemoryRow {
    memory_id: String,
    memory_type: String,
    memory_key: Option<String>,
    content: String,
    conversation_id: Option<String>,
    session_id: Option<String>,
    importance_score: f64,
    confidence_score: f64,
    access_count: i64,
    status: String,
    created_at: String,
    updated_at: String,
    expires_at: Option<String>,
}

fn memory_active(row: &MemoryRow) -> bool {
    if row.status != "active" {
        return false;
    }
    match row.expires_at.as_deref().and_then(crate::db::parse_dt) {
        Some(expires) => expires > crate::db::now_naive(),
        None => true,
    }
}

fn json_none_or_display(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".into(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    }
}

fn memory_record_dedupe_key(record: &Value) -> String {
    let metadata = record.get("metadata");
    let memory_id = metadata
        .and_then(|value| value.get("memory_id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            record
                .get("memory_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
        })
        .unwrap_or("");
    if !memory_id.is_empty() {
        return memory_id.to_string();
    }
    let conversation_id = record
        .get("conversation_id")
        .and_then(Value::as_str)
        .or_else(|| metadata.and_then(|value| value.get("conversation_id")).and_then(Value::as_str))
        .unwrap_or("");
    let content = record.get("content").and_then(Value::as_str).unwrap_or("");
    let prefix: String = content.chars().take(120).collect();
    format!(
        "{}:{}:{}:{}",
        conversation_id,
        json_none_or_display(metadata.and_then(|value| value.get("turn_start"))),
        json_none_or_display(metadata.and_then(|value| value.get("turn_end"))),
        eks_kernels::normalize_text(&prefix)
    )
}

fn dedupe_matched_memories(records: Vec<Value>) -> Vec<Value> {
    let mut seen = HashSet::new();
    let mut deduplicated = Vec::new();
    for record in records {
        if !seen.insert(memory_record_dedupe_key(&record)) {
            continue;
        }
        deduplicated.push(record);
    }
    deduplicated
}

fn merge_current_history(
    original: Vec<Value>,
    recent_history: &[crate::short_memory::MemoryMessage],
    conversation_id: Option<&str>,
    top_k: usize,
) -> Vec<Value> {
    if recent_history.is_empty() {
        return original;
    }
    let (_, turns) = crate::short_memory::split_history(recent_history);
    if turns.is_empty() {
        return original;
    }
    let keep = top_k.max(1);
    let start = turns.len().saturating_sub(keep);
    let mut existing_texts: HashSet<String> = original
        .iter()
        .filter_map(|record| record.get("content").and_then(Value::as_str))
        .map(eks_kernels::normalize_text)
        .collect();
    let mut merged = original;
    let conversation_id = conversation_id.map(str::trim).filter(|value| !value.is_empty());
    for turn in &turns[start..] {
        let mut block_lines = Vec::new();
        for message in turn {
            let role = message.role.trim().to_lowercase();
            let content = message.content.trim();
            if content.is_empty() {
                continue;
            }
            if role == "user" {
                block_lines.push(format!("用户: {content}"));
            } else if role == "assistant" {
                block_lines.push(format!("助手: {content}"));
            }
        }
        let block_text = block_lines.join("\n");
        let block_text = block_text.trim();
        if block_text.is_empty() {
            continue;
        }
        let normalized_block = eks_kernels::normalize_text(block_text);
        if !existing_texts.insert(normalized_block) {
            continue;
        }
        merged.push(json!({
            "memory_id": Value::Null,
            "content": block_text,
            "score": 1.0,
            "source": "conversation",
            "metadata": {
                "memory_id": Value::Null,
                "conversation_id": conversation_id,
                "chunk_type": "dialogue_current_turn",
                "turn_start": Value::Null,
                "turn_end": Value::Null,
            },
        }));
    }
    merged
}

async fn memory_query(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    arguments: &Value,
    conversation_id: Option<&str>,
    session_id: Option<&str>,
    recent_history: &[crate::short_memory::MemoryMessage],
    embedder: Option<&crate::embedder::Embedder>,
) -> Result<Value, String> {
    let user_id = crate::memory::require_user_id(user_id)?;
    let query = arg_str(arguments, "query");
    let top_k = arg_usize(arguments, "top_k", 5, 1, 50);
    let mut scope = arg_str(arguments, "memory_scope").to_lowercase();
    if scope.is_empty() {
        scope = "all".to_string();
    }
    let mut include_profile = scope == "all" || scope == "profile";
    let mut include_event = scope == "all" || scope == "event";
    if !include_profile && !include_event {
        scope = "all".to_string();
        include_profile = true;
        include_event = true;
    }

    let profile_memories = if include_profile {
        load_memories(pool, user_id, Some("profile"), 100).await?
            .into_iter()
            .filter(memory_active)
            .take(top_k)
            .map(|row| json!({
                "memory_id": row.memory_id,
                "memory_type": row.memory_type,
                "memory_key": row.memory_key,
                "content": row.content,
                "status": row.status,
                "importance_score": row.importance_score,
                "confidence_score": row.confidence_score,
                "access_count": row.access_count,
                "conversation_id": row.conversation_id,
                "session_id": row.session_id,
                "created_at": crate::db::dt_to_iso(&row.created_at),
                "updated_at": crate::db::dt_to_iso(&row.updated_at),
                "expires_at": row
                    .expires_at
                    .as_deref()
                    .and_then(crate::db::dt_to_iso),
            }))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let matched_memories = if include_event && !query.trim().is_empty() {
        let stored = crate::memory::search_event_memories(
            pool,
            user_id,
            &query,
            conversation_id,
            session_id,
            top_k,
            embedder,
        )
            .await?
            .into_iter()
            .map(|hit| json!({
                "memory_id": hit.memory_id,
                "content": hit.content,
                "score": hit.score,
                "source": hit.memory_key,
                "metadata": {
                    "conversation_id": hit.conversation_id,
                    "memory_id": hit.memory_id,
                    "memory_key": hit.memory_key,
                    "chunk_type": hit.memory_key,
                    "turn_start": hit.turn_start,
                    "turn_end": hit.turn_end,
                },
            }))
            .collect::<Vec<_>>();
        merge_current_history(
            dedupe_matched_memories(stored),
            recent_history,
            conversation_id,
            top_k,
        )
    } else {
        Vec::new()
    };
    let profile_count = profile_memories.len();
    let matched_count = matched_memories.len();

    Ok(json!({
        "query": query,
        "memory_scope": scope,
        "filters": {
            "conversation_id": conversation_id,
            "session_id": session_id,
            "top_k": top_k,
        },
        "profile_memories": profile_memories,
        "matched_memories": matched_memories,
        "summary": {
            "profile_count": profile_count,
            "matched_count": matched_count,
        },
    }))
}

async fn memory_original(pool: &sqlx::SqlitePool, user_id: &str, arguments: &Value) -> Result<Value, String> {
    let mut memory_ids = arg_strings(arguments, "memory_ids");
    if memory_ids.is_empty() {
        memory_ids = arg_strings(arguments, "summary_memory_ids");
    }
    crate::memory::retrieve_original_conversation(
        pool,
        user_id,
        crate::memory::OriginalConversationQuery {
            memory_ids,
            query: arg_str(arguments, "query"),
            conversation_id: arg_str(arguments, "conversation_id"),
            turn_start: arg_i64(arguments, "turn_start"),
            turn_end: arg_i64(arguments, "turn_end"),
            include_neighbors: arg_usize(arguments, "include_neighbors", 0, 0, 5) as i64,
            include_all_conversation: arg_bool(arguments, "include_all_conversation", false),
            top_k: arg_usize(arguments, "top_k", 8, 1, 50),
            max_chars: arg_usize(arguments, "max_chars", 12000, 500, 50000),
        },
    )
    .await
}

async fn delete_profile_memory(pool: &sqlx::SqlitePool, user_id: &str, arguments: &Value) -> Result<Value, String> {
    let keys = arg_strings(arguments, "memory_keys");
    let keywords = arg_strings(arguments, "keywords");
    let max_deletes = arg_usize(arguments, "max_deletes", 50, 1, 200);
    if keys.is_empty() && keywords.is_empty() {
        return Ok(json!({
            "deleted_count": 0,
            "matched": [],
            "summary": "未提供删除条件，请指定 memory_keys 或 keywords。",
        }));
    }
    let normalized_keys: Vec<String> = keys.iter().map(|key| key.trim().to_lowercase()).filter(|key| !key.is_empty()).collect();
    let normalized_keywords: Vec<String> = keywords.iter().map(|word| eks_kernels::normalize_text(word)).filter(|word| !word.is_empty()).collect();
    let rows = load_memories(pool, user_id, Some("profile"), 200).await?;
    let active: Vec<MemoryRow> = rows.into_iter().filter(memory_active).collect();
    if active.is_empty() {
        return Ok(json!({
            "deleted_count": 0,
            "matched": [],
            "summary": "当前没有可删除的个性化记忆。",
        }));
    }
    let matched: Vec<MemoryRow> = active
        .into_iter()
        .filter(|row| {
            let key = row.memory_key.as_deref().unwrap_or("").trim().to_lowercase();
            let content = eks_kernels::normalize_text(&row.content);
            normalized_keys.iter().any(|item| item == &key)
                || normalized_keywords.iter().any(|word| content.contains(word) || key.contains(word))
        })
        .collect();
    if matched.is_empty() {
        return Ok(json!({
            "deleted_count": 0,
            "matched": [],
            "summary": "没有匹配到符合条件的个性化记忆。",
        }));
    }
    if matched.len() > max_deletes {
        // 对齐 Python：超限时一条都不删，只返回预览，避免误删。
        let previews: Vec<Value> = matched
            .iter()
            .take(max_deletes)
            .map(|row| {
                json!({
                    "memory_key": row.memory_key.clone().unwrap_or_default(),
                    "content": row.content.chars().take(200).collect::<String>(),
                })
            })
            .collect();
        return Ok(json!({
            "deleted_count": 0,
            "matched": previews,
            "summary": format!(
                "匹配到 {} 条记忆，超过单次删除上限 {}，为避免误删请用更精确的 memory_key 或关键词重试。",
                matched.len(),
                max_deletes
            ),
        }));
    }
    let now = crate::db::fmt_dt(&crate::db::now_naive());
    for row in &matched {
        sqlx::query("UPDATE memory_records SET status = 'deleted', updated_at = ? WHERE memory_id = ? AND user_id = ?")
            .bind(&now)
            .bind(&row.memory_id)
            .bind(user_id)
            .execute(pool)
            .await
            .map_err(|err| err.to_string())?;
    }
    let matched_json: Vec<Value> = matched
        .iter()
        .map(|row| {
            json!({
                "memory_id": row.memory_id,
                "memory_key": row.memory_key,
                "content": row.content.chars().take(400).collect::<String>(),
                "confidence_score": row.confidence_score,
            })
        })
        .collect();
    Ok(json!({
        "deleted_count": matched_json.len(),
        "matched": matched_json,
        "summary": format!("已删除 {} 条个性化记忆。", matched.len()),
    }))
}

async fn load_memories(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    memory_type: Option<&str>,
    limit: i64,
) -> Result<Vec<MemoryRow>, String> {
    let sql = if memory_type.is_some() {
        "SELECT memory_id, memory_type, memory_key, content, conversation_id, session_id, \
         importance_score, confidence_score, access_count, status, created_at, updated_at, expires_at \
         FROM memory_records WHERE user_id = ? AND memory_type = ? \
         ORDER BY importance_score DESC, updated_at DESC LIMIT ?"
    } else {
        "SELECT memory_id, memory_type, memory_key, content, conversation_id, session_id, \
         importance_score, confidence_score, access_count, status, created_at, updated_at, expires_at \
         FROM memory_records WHERE user_id = ? \
         ORDER BY importance_score DESC, updated_at DESC LIMIT ?"
    };
    let query = sqlx::query_as::<_, MemoryRow>(sql).bind(user_id);
    let query = if let Some(memory_type) = memory_type {
        query.bind(memory_type).bind(limit)
    } else {
        query.bind(limit)
    };
    query.fetch_all(pool).await.map_err(|err| err.to_string())
}

#[derive(Debug, sqlx::FromRow)]
struct ChunkTextRow {
    document_id: String,
    chunk_text: String,
}

async fn load_chunks_for_documents(
    pool: &sqlx::SqlitePool,
    document_ids: &[String],
) -> Result<HashMap<String, Vec<String>>, String> {
    let mut chunks_by_document: HashMap<String, Vec<String>> = HashMap::new();
    if document_ids.is_empty() {
        return Ok(chunks_by_document);
    }
    let placeholders = vec!["?"; document_ids.len()].join(",");
    let sql = format!(
        "SELECT document_id, chunk_text FROM document_chunks \
         WHERE document_id IN ({placeholders}) ORDER BY document_id, chunk_index"
    );
    let mut query = sqlx::query_as::<_, ChunkTextRow>(&sql);
    for document_id in document_ids {
        query = query.bind(document_id);
    }
    let rows = query.fetch_all(pool).await.map_err(|err| err.to_string())?;
    for row in rows {
        let text = row.chunk_text.trim();
        if text.is_empty() {
            continue;
        }
        chunks_by_document
            .entry(row.document_id)
            .or_default()
            .push(text.to_string());
    }
    Ok(chunks_by_document)
}

fn push_in_clause(sql: &mut String, binds: &mut Vec<String>, column: &str, values: &[String]) {
    if values.is_empty() {
        return;
    }
    let placeholders = vec!["?"; values.len()].join(",");
    sql.push_str(&format!(" AND {column} IN ({placeholders})"));
    binds.extend(values.iter().cloned());
}

fn arg_str(arguments: &Value, key: &str) -> String {
    arguments.get(key).and_then(|value| value.as_str()).unwrap_or("").trim().to_string()
}

/// 对齐 Python `_normalize_string_list`：字符串是单元素（不按逗号拆分），
/// 大小写不敏感去重、跳过空项。
fn arg_strings(arguments: &Value, key: &str) -> Vec<String> {
    let items: Vec<Value> = match arguments.get(key) {
        Some(Value::Array(list)) => list.clone(),
        Some(Value::Null) | None => return Vec::new(),
        Some(other) => vec![other.clone()],
    };
    let mut seen = std::collections::HashSet::new();
    let mut normalized = Vec::new();
    for item in items {
        // 对齐 Python `str(item or "")`：0/False/空串归一为空后跳过。
        let text = match &item {
            Value::String(value) => value.trim().to_string(),
            Value::Number(value) if value.as_f64() == Some(0.0) => continue,
            Value::Bool(false) => continue,
            Value::Null => continue,
            other => other.to_string().trim().to_string(),
        };
        if text.is_empty() {
            continue;
        }
        if seen.insert(text.to_lowercase()) {
            normalized.push(text);
        }
    }
    normalized
}

/// 对齐 Python `_coerce_int`：接受数字/数字字符串/布尔，失败取默认，最后夹取。
fn arg_usize(arguments: &Value, key: &str, default: usize, min: usize, max: usize) -> usize {
    let resolved: i64 = match arguments.get(key) {
        Some(Value::Number(number)) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|value| value.trunc() as i64))
            .unwrap_or(default as i64),
        Some(Value::String(text)) => text.trim().parse::<i64>().unwrap_or(default as i64),
        Some(Value::Bool(value)) => i64::from(*value),
        _ => default as i64,
    };
    resolved.clamp(min as i64, max as i64) as usize
}

fn arg_i64(arguments: &Value, key: &str) -> Option<i64> {
    arguments.get(key).and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_u64().map(|item| item as i64))
            .or_else(|| value.as_str().and_then(|item| item.trim().parse().ok()))
    })
}

fn arg_bool(arguments: &Value, key: &str, default: bool) -> bool {
    // 对齐 Python `bool(value)` 真值语义：非空字符串/非零数字为真。
    match arguments.get(key) {
        None => default,
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(value)) => value.as_f64().unwrap_or(0.0) != 0.0,
        Some(Value::String(value)) => !value.trim().is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(map)) => !map.is_empty(),
        Some(Value::Null) => false,
    }
}

fn top_terms(text: &str, limit: usize) -> Vec<String> {
    let mut ranked: Vec<(String, usize)> = extract_term_frequencies(text).into_iter().collect();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    ranked.into_iter().take(limit).map(|(term, _)| term).collect()
}

fn extract_term_frequencies(text: &str) -> HashMap<String, usize> {
    let mut frequencies = HashMap::new();
    if text.is_empty() {
        return frequencies;
    }
    let chars: Vec<char> = text.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        if chars[index].is_ascii_alphanumeric() {
            let start = index;
            while index < chars.len() && chars[index].is_ascii_alphanumeric() {
                index += 1;
            }
            if index - start >= 2 {
                let token: String = chars[start..index].iter().collect();
                push_term(&mut frequencies, &token);
            }
            continue;
        }
        if is_cjk(chars[index]) {
            let start = index;
            while index < chars.len() && is_cjk(chars[index]) {
                index += 1;
            }
            for size in [2usize, 3] {
                if index - start < size {
                    continue;
                }
                for offset in start..=index - size {
                    let gram: String = chars[offset..offset + size].iter().collect();
                    push_term(&mut frequencies, &gram);
                }
            }
            continue;
        }
        index += 1;
    }
    frequencies
}

fn push_term(frequencies: &mut HashMap<String, usize>, token: &str) {
    let cleaned = eks_kernels::normalize_text(token);
    if is_noisy_term(&cleaned) {
        return;
    }
    *frequencies.entry(cleaned).or_default() += 1;
}

fn is_noisy_term(term: &str) -> bool {
    term.chars().count() < 2 || NOISY_TERMS.contains(&term) || term.chars().all(|ch| ch.is_ascii_digit())
}

fn is_cjk(ch: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&ch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn routes_rule_skills_like_python() {
        let tools = route_local_skill("列出你的工具").expect("tool");
        assert_eq!(tools.name, "list_available_tools");
        assert!(route_local_skill("知识库里有哪些文档").is_none());
        assert!(route_local_skill("忘掉张三").is_none());
        assert!(route_local_skill("差旅报销流程是什么").is_none());
        let profile = route_local_skill("我 叫 什么 名字").expect("profile");
        assert_eq!(profile.name, "memory_query");
        assert_eq!(profile.arguments["memory_scope"], "all");
        let hot = route_local_skill("这份文档里的关键词").expect("hot");
        assert_eq!(hot.name, "document_hot_word_statistics");
        let catalog = function_calling_tools();
        let names: Vec<&str> = catalog
            .iter()
            .filter_map(|tool| tool.pointer("/function/name").and_then(|value| value.as_str()))
            .collect();
        assert!(names.contains(&"search_knowledge_base"));
        assert!(names.contains(&"memory_query"));
        assert_eq!(names.len(), 7);
        assert_eq!(names[0], "search_knowledge_base");
        assert_eq!(
            format_skill_exec_fail("memory_query"),
            "Skill 执行失败: memory_query"
        );
    }

    #[test]
    fn skill_catalog_shapes_match_python() {
        let definitions = list_skill_definitions();
        assert_eq!(definitions.len(), 6);
        let names: Vec<&str> = definitions
            .iter()
            .filter_map(|item| item.get("name").and_then(|value| value.as_str()))
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
        assert!(!names.contains(&"search_knowledge_base"));
        assert!(definitions[0].get("function_calling").is_some());
        assert_eq!(definitions[0].get("kind").and_then(|value| value.as_str()), Some("local"));

        let tools = function_calling_tools();
        assert_eq!(
            tools[0].pointer("/function/name").and_then(|value| value.as_str()),
            Some("search_knowledge_base")
        );
        assert_eq!(tools.len(), 7);

        let capabilities = list_agent_capabilities();
        assert_eq!(capabilities.len(), 7);
        assert_eq!(capabilities[0].get("source").and_then(|value| value.as_str()), Some("rag"));
        assert_eq!(capabilities[1].get("source").and_then(|value| value.as_str()), Some("skill"));
        let payload = skill_capabilities_payload();
        assert_eq!(payload.get("tools"), payload.get("function_calling"));
        assert!(payload.get("capabilities").and_then(|value| value.as_array()).is_some());

        let wrapped = wrap_skill_execution(
            "memory_query",
            &json!({"query": "我的名字"}),
            &SkillExecutionContext {
                user_id: "u1",
                conversation_id: Some("c1"),
                query: Some("我叫什么"),
                ..SkillExecutionContext::new("u1")
            },
            json!({"ok": true}),
        )
        .unwrap();
        assert_eq!(wrapped.get("skill").and_then(|value| value.as_str()), Some("memory_query"));
        assert_eq!(wrapped.get("category").and_then(|value| value.as_str()), Some("memory"));
        assert_eq!(wrapped.get("kind").and_then(|value| value.as_str()), Some("local"));
        let executed_at = wrapped.get("executed_at").and_then(|value| value.as_str()).unwrap();
        assert!(executed_at.ends_with("+00:00"), "{executed_at}");
        assert!(!executed_at.ends_with('Z'));
        assert_eq!(
            wrapped.pointer("/context/conversation_id").and_then(|value| value.as_str()),
            Some("c1")
        );
        let instructions = wrapped
            .get("skill_instructions")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        assert!(instructions.contains("记忆查询"));
        assert!(instructions.contains("不能跨用户"));
        assert!(
            wrap_skill_execution(
                "missing",
                &json!({}),
                &SkillExecutionContext::new("u1"),
                json!({})
            )
            .is_err()
        );
        assert_eq!(
            skill_not_found_http_detail("memory_query"),
            "未找到技能: 'memory_query'"
        );
        let http_payload = skill_execute_http_payload(wrapped.clone());
        assert!(http_payload.get("skill_instructions").is_none());
        assert_eq!(http_payload["skill"], "memory_query");
        assert_eq!(http_payload["context"]["provider"], json!(null));
        assert_eq!(http_payload["context"]["model"], json!(null));
        assert_eq!(http_payload["context"]["conversation_id"], "c1");
        assert!(wrapped.get("skill_instructions").is_some());
        let filtered = wrap_skill_execution(
            "memory_query",
            &json!({}),
            &SkillExecutionContext {
                metadata: json!({"keep": true, "memory_context": {"secret": 1}}),
                ..SkillExecutionContext::new("u1")
            },
            json!({"ok": true}),
        )
        .unwrap();
        assert_eq!(filtered["context"]["metadata"], json!({"keep": true}));
        assert!(filtered["context"]["metadata"].get("memory_context").is_none());
        let chat_wrapped = wrap_skill_execution(
            "memory_query",
            &json!({"query": "我的名字"}),
            &SkillExecutionContext {
                conversation_id: Some("c1"),
                session_id: Some("s1"),
                query: Some("我叫什么"),
                provider: Some("openai"),
                model: Some("gpt-4o-mini"),
                ..SkillExecutionContext::new("u1")
            },
            json!({"ok": true}),
        )
        .unwrap();
        assert_eq!(chat_wrapped["context"]["provider"], "openai");
        assert_eq!(chat_wrapped["context"]["model"], "gpt-4o-mini");
        assert_eq!(chat_wrapped["context"]["session_id"], "s1");
        assert_eq!(
            SkillExecutionContext {
                provider: Some("deepseek"),
                model: Some("deepseek-chat"),
                query: Some("差旅"),
                metadata: json!({"react_mode": "unified"}),
                ..SkillExecutionContext::new("u1")
            }
            .public_payload(),
            json!({
                "user_id": "u1",
                "conversation_id": null,
                "session_id": null,
                "query": "差旅",
                "provider": "deepseek",
                "model": "deepseek-chat",
                "trace_id": null,
                "metadata": {"react_mode": "unified"},
            })
        );

        let messages = build_skill_messages(
            " 我叫什么 ",
            &wrapped,
            &[ChatMessage::new("system", "画像：张三")],
            vec![
                ChatMessage::new("user", "上次问题"),
                ChatMessage::new("assistant", "上次回答"),
            ],
        );
        assert_eq!(messages[0].content, SKILL_SYSTEM_PROMPT);
        assert!(messages[1].content.contains("当前 Skill 的详细规则"));
        assert!(messages[1].content.contains("# 记忆查询 Skill"));
        assert!(messages[2].content.starts_with("技能执行结果："));
        assert!(messages[2].content.contains("\"skill\":\"memory_query\""));
        assert!(!messages[2].content.contains("skill_instructions"));
        assert_eq!(messages[3].content, "画像：张三");
        assert_eq!(messages[4].content, "上次问题");
        assert_eq!(messages[5].content, "上次回答");
        assert_eq!(messages[6].role, "user");
        assert_eq!(messages[6].content, "我叫什么");
        assert_eq!(messages.len(), 7);

        let empty_instructions = json!({
            "skill": "list_available_tools",
            "result": {"ok": true},
            "skill_instructions": "  \n",
        });
        let compact = build_skill_messages("列出工具", &empty_instructions, &[], vec![]);
        assert_eq!(compact.len(), 3);
        assert_eq!(compact[0].content, SKILL_SYSTEM_PROMPT);
        assert!(!compact[1].content.contains("详细规则"));
        assert_eq!(compact[2].content, "列出工具");
    }

    #[test]
    fn rejects_illegal_skill_document_names() {
        let root = Path::new("/tmp/skills");
        for name in ["", ".", "..", "../etc", "foo/bar", "a/../b"] {
            let err = get_skill_document_path_in(name, root).unwrap_err();
            assert!(err.contains("非法 Skill 名称"), "{name}: {err}");
        }
        assert_eq!(
            get_skill_document_path_in("memory_query", root).unwrap(),
            root.join("memory_query").join("memory_query.md")
        );
    }

    #[test]
    fn reads_local_markdown_skips_remote_and_missing_files() {
        let text = read_skill_document("memory_query").unwrap();
        assert!(text.contains("# 记忆查询 Skill"));
        assert!(text.contains("## 使用规则"));

        let remote = SkillDefinition {
            name: "crm_lookup".into(),
            description: "查询客户".into(),
            category: "mcp".into(),
            kind: "remote_mcp".into(),
            tags: vec![],
            provider: Some("crm".into()),
            parameters: json!({"type": "object"}),
            metadata: json!({"executor_ready": false}),
        };
        assert_eq!(read_skill_document_for_definition(&remote).unwrap(), "");

        let missing_root = std::env::temp_dir().join(format!("eks-skill-missing-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&missing_root).unwrap();
        let missing = skill_def(
            "memory_query",
            "d",
            "memory",
            &[],
            json!({"type": "object"}),
            "r",
        );
        assert_eq!(
            read_skill_document_for_definition_from(&missing, &missing_root).unwrap(),
            ""
        );
        std::fs::remove_dir_all(&missing_root).ok();
    }

    #[test]
    fn reads_markdown_from_explicit_root_and_rejects_escaped_paths() {
        let root = std::env::temp_dir().join(format!("eks-skill-docs-{}", uuid::Uuid::new_v4()));
        let doc_dir = root.join("fixture_skill");
        std::fs::create_dir_all(&doc_dir).unwrap();
        std::fs::write(doc_dir.join("fixture_skill.md"), "# Fixture\n规则甲").unwrap();
        let definition = skill_def(
            "fixture_skill",
            "d",
            "system",
            &[],
            json!({"type": "object"}),
            "r",
        );
        let text = read_skill_document_for_definition_from(&definition, &root).unwrap();
        assert_eq!(text, "# Fixture\n规则甲");

        let err = ensure_document_within_skills_root(
            Path::new("/tmp/eks-skills-root"),
            Path::new("/etc/passwd"),
            "fixture_skill",
        )
        .unwrap_err();
        assert!(err.contains("超出技能目录"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn remote_mcp_catalog_filters_unready_from_function_calling() {
        let manifests = vec![
            json!({
                "skill_name": "crm_lookup",
                "server_name": "crm",
                "description": "查询客户"
            }),
            json!({
                "name": "ticket_create",
                "endpoint": "https://mcp.example.internal/ticket",
                "provider": "itsm"
            }),
        ];
        let definitions = list_skill_definitions_from(&manifests);
        let names: Vec<&str> = definitions
            .iter()
            .filter_map(|item| item.get("name").and_then(Value::as_str))
            .collect();
        assert!(names.contains(&"crm_lookup"));
        assert!(names.contains(&"ticket_create"));
        assert!(names.contains(&"memory_query"));
        let crm = definitions.iter().find(|item| item["name"] == "crm_lookup").unwrap();
        assert_eq!(crm["kind"], "remote_mcp");
        assert_eq!(crm["metadata"]["executor_ready"], false);

        let tools = function_calling_tools_from(&manifests);
        let tool_names: Vec<&str> = tools
            .iter()
            .filter_map(|tool| tool.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert!(tool_names.contains(&"ticket_create"));
        assert!(!tool_names.contains(&"crm_lookup"));
        assert_eq!(tool_names[0], "search_knowledge_base");

        let capabilities = list_agent_capabilities_from(&manifests);
        let crm_cap = capabilities.iter().find(|item| item["name"] == "crm_lookup").unwrap();
        assert_eq!(crm_cap["available"], false);
        let ticket_cap = capabilities.iter().find(|item| item["name"] == "ticket_create").unwrap();
        assert_eq!(ticket_cap["available"], true);

        let listed = list_available_tools(&manifests);
        let listed_names: Vec<&str> = listed["tools"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|item| item.get("name").and_then(Value::as_str))
            .collect();
        assert!(listed_names.contains(&"ticket_create"));
        assert!(!listed_names.contains(&"crm_lookup"));

        let err = execute_named_from(
            &sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap(),
            "crm_lookup",
            &json!({}),
            &SkillExecutionContext::new("u1"),
            &manifests,
            &[],
            None,
        )
        .await
        .unwrap_err();
        assert!(err.contains("尚未接入实际执行器"));
        let ready_err = execute_named_from(
            &sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap(),
            "ticket_create",
            &json!({}),
            &SkillExecutionContext::new("u1"),
            &manifests,
            &[],
            None,
        )
        .await
        .unwrap_err();
        assert!(ready_err.contains("itsm/ticket_create"));
    }

    #[test]
    fn extracts_filename_and_chinese_terms() {
        assert_eq!(extract_file_name_hint("统计 05_费用标准表.xlsx 的热词"), "05_费用标准表.xlsx");
        let freq = extract_term_frequencies("账号密码");
        assert!(freq.get("账号").copied().unwrap_or(0) >= 1);
        assert!(freq.get("密码").copied().unwrap_or(0) >= 1);
        assert!(!freq.contains_key("账号密码"));
    }

    #[test]
    fn tool_defaults_are_shared_by_direct_and_agent_paths() {
        let query = "请展开之前摘要对应的完整对话";
        let direct = normalize_action_arguments("memory_retrieve_original", &json!({}), query);
        let agent = default_action_arguments("memory_retrieve_original", query);
        assert_eq!(direct, agent);
        assert_eq!(direct["query"], query);
        assert_eq!(direct["top_k"], 8);
        assert_eq!(direct["max_chars"], 12000);

        let provided = normalize_action_arguments(
            "memory_query",
            &json!({"query": "只要这个", "top_k": 2}),
            "原始问题",
        );
        assert_eq!(provided["query"], "只要这个");
        assert_eq!(provided["top_k"], 2);
        assert_eq!(provided["memory_scope"], "all");

        let hot = default_action_arguments(
            "document_hot_word_statistics",
            "统计 05_费用标准表.xlsx 的热词",
        );
        assert_eq!(hot["top_k"], 10);
        assert_eq!(hot["include_samples"], true);
        assert_eq!(hot["file_name"], "05_费用标准表.xlsx");
    }

    #[tokio::test]
    async fn hot_words_counts_all_statuses_and_returns_filters() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::ensure_document_schema(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO documents (document_id, original_filename, stored_filename, file_path, \
             file_type, file_size, status, chunk_count, created_at, updated_at) VALUES \
             ('d-ready','报销.md','d-ready.md','/tmp/a','md',10,'ready',1, \
              '2026-01-01 00:00:00.000000','2026-01-02 00:00:00.000000')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO documents (document_id, original_filename, stored_filename, file_path, \
             file_type, file_size, status, chunk_count, created_at, updated_at) VALUES \
             ('d-proc','发布.md','d-proc.md','/tmp/b','md',10,'processing',1, \
              '2026-01-01 00:00:00.000000','2026-01-03 00:00:00.000000')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO document_chunks (document_id, chunk_index, parent_id, chunk_text, created_at) \
             VALUES ('d-ready',0,'d-ready:p000000','差旅报销需要发票','2026-01-01 00:00:00.000000')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO document_chunks (document_id, chunk_index, parent_id, chunk_text, created_at) \
             VALUES ('d-proc',0,'d-proc:p000000','服务器发布窗口确认','2026-01-01 00:00:00.000000')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let all = hot_words(&pool, &json!({"top_k": 50})).await.unwrap();
        assert_eq!(all["summary"]["document_count"], 2);
        assert_eq!(all["summary"]["chunk_count"], 2);
        assert_eq!(all["scope"], "documents");
        assert!(all["filters"]["statuses"].as_array().unwrap().is_empty());
        assert!(all["filters"]["document_ids"].as_array().unwrap().is_empty());
        assert!(all["filters"]["file_types"].as_array().unwrap().is_empty());
        assert_eq!(all["filters"]["file_name"], "");
        let terms: Vec<&str> = all["top_words"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|word| word["term"].as_str())
            .collect();
        assert!(terms.iter().any(|term| *term == "报销"));
        assert!(terms.iter().any(|term| *term == "发布"));
        assert!(terms.iter().any(|term| *term == "ready"));
        assert!(terms.iter().any(|term| *term == "processing"));
        assert!(!all["top_words"][0]["sample_documents"].as_array().unwrap().is_empty());

        let ready_only = hot_words(&pool, &json!({"top_k": 50, "status": "ready"})).await.unwrap();
        assert_eq!(ready_only["summary"]["document_count"], 1);
        assert_eq!(ready_only["filters"]["statuses"][0], "ready");
        let ready_terms: Vec<&str> = ready_only["top_words"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|word| word["term"].as_str())
            .collect();
        assert!(ready_terms.iter().any(|term| *term == "报销"));
        assert!(!ready_terms.iter().any(|term| *term == "发布"));

        let no_samples = hot_words(&pool, &json!({"top_k": 5, "include_samples": false})).await.unwrap();
        assert!(no_samples["top_words"][0]["sample_documents"]
            .as_array()
            .unwrap()
            .is_empty());

        let by_filename = hot_words(&pool, &json!({"filename": "报销.md"})).await.unwrap();
        assert_eq!(by_filename["summary"]["document_count"], 1);
        assert_eq!(by_filename["filters"]["file_name"], "报销.md");
    }

    #[tokio::test]
    async fn lists_ready_files_and_returns_python_filters() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::db::ensure_document_schema(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO documents (document_id, original_filename, stored_filename, file_path, \
             file_type, file_size, status, chunk_count, created_at, updated_at) VALUES \
             ('d-ready','报销.md','d-ready.md','/tmp/a','md',10,'ready',1, \
              '2026-01-01 00:00:00.000000','2026-01-02 00:00:00.000000')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO documents (document_id, original_filename, stored_filename, file_path, \
             file_type, file_size, status, chunk_count, created_at, updated_at) VALUES \
             ('d-proc','发布.md','d-proc.md','/tmp/b','md',10,'processing',1, \
              '2026-01-01 00:00:00.000000','2026-01-03 00:00:00.000000')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO document_chunks (document_id, chunk_index, parent_id, chunk_text, created_at) \
             VALUES ('d-ready',0,'d-ready:p000000','[章节] 差旅\n差旅报销需要发票','2026-01-01 00:00:00.000000')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let listed = list_knowledge_base_files(&pool, &json!({})).await.unwrap();
        assert_eq!(listed["summary"]["document_count"], 1);
        assert_eq!(listed["filters"]["statuses"][0], "ready");
        assert!(listed["filters"]["document_ids"].as_array().unwrap().is_empty());
        assert!(listed["filters"]["file_types"].as_array().unwrap().is_empty());
        assert_eq!(listed["filters"]["file_name"], "");
        assert_eq!(listed["filters"]["limit"], 20);
        assert_eq!(listed["documents"][0]["document_id"], "d-ready");
        assert_eq!(listed["documents"][0]["outline"][0], "差旅");
        assert!(listed["documents"][0]["content_preview"]
            .as_str()
            .unwrap()
            .contains("差旅报销需要发票"));

        let processing = list_knowledge_base_files(&pool, &json!({"status": "processing"}))
            .await
            .unwrap();
        assert_eq!(processing["summary"]["document_count"], 1);
        assert_eq!(processing["filters"]["statuses"][0], "processing");
        assert_eq!(processing["documents"][0]["document_id"], "d-proc");

        let by_filename = list_knowledge_base_files(&pool, &json!({"filename": "报销.md"}))
            .await
            .unwrap();
        assert_eq!(by_filename["summary"]["document_count"], 1);
        assert_eq!(by_filename["filters"]["file_name"], "报销.md");

        let empty = list_knowledge_base_files(&pool, &json!({"filename": "不存在.xlsx"}))
            .await
            .unwrap();
        assert_eq!(empty["summary"]["document_count"], 0);
        assert!(empty["filters"].get("document_ids").is_some());
        assert!(empty["filters"].get("file_types").is_some());
    }

    #[tokio::test]
    async fn lists_and_deletes_profile_memory() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(
            "CREATE TABLE memory_records (
                memory_id TEXT PRIMARY KEY, user_id TEXT, memory_type TEXT, memory_key TEXT,
                content TEXT, conversation_id TEXT, session_id TEXT, importance_score REAL,
                confidence_score REAL, access_count INTEGER, status TEXT, created_at TEXT,
                updated_at TEXT, expires_at TEXT
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO memory_records VALUES ('m1','u1','profile','profile.name','张三',NULL,NULL,80,0.9,0,'active','2026-01-01 00:00:00.000000','2026-01-01 00:00:00.000000',NULL)",
        )
        .execute(&pool)
        .await
        .unwrap();
        let queried = memory_query(
            &pool,
            "u1",
            &json!({"query":"我的名字","memory_scope":"profile","top_k":5}),
            None,
            None,
            &[],
            None,
        )
            .await
            .unwrap();
        assert_eq!(queried.pointer("/summary/profile_count").and_then(|value| value.as_u64()), Some(1));
        assert_eq!(queried.pointer("/filters/top_k").and_then(|value| value.as_u64()), Some(5));
        assert!(queried.pointer("/profile_memories/0/expires_at").unwrap().is_null());
        let mixed_scope = memory_query(
            &pool,
            "u1",
            &json!({"query":"我的名字","memory_scope":"PROFILE","top_k":5}),
            None,
            None,
            &[],
            None,
        )
        .await
        .unwrap();
        assert_eq!(mixed_scope["memory_scope"], "profile");
        assert_eq!(mixed_scope.pointer("/summary/profile_count").and_then(|value| value.as_u64()), Some(1));
        let invalid_scope = memory_query(
            &pool,
            "u1",
            &json!({"query":"我的名字","memory_scope":"unknown","top_k":5}),
            None,
            None,
            &[],
            None,
        )
        .await
        .unwrap();
        assert_eq!(invalid_scope["memory_scope"], "all");
        assert_eq!(invalid_scope.pointer("/summary/profile_count").and_then(|value| value.as_u64()), Some(1));
        let empty = delete_profile_memory(&pool, "u1", &json!({})).await.unwrap();
        assert_eq!(
            empty["summary"],
            "未提供删除条件，请指定 memory_keys 或 keywords。"
        );
        assert_eq!(empty["deleted_count"], 0);
        let deleted = delete_profile_memory(&pool, "u1", &json!({"keywords":["张三"]})).await.unwrap();
        assert_eq!(deleted.get("deleted_count").and_then(|value| value.as_u64()), Some(1));
    }

    #[tokio::test]
    async fn queries_event_memory_by_hierarchical_rank() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::memory::ensure_schema(&pool).await.unwrap();
        crate::memory::insert_event_memory(
            &pool,
            "u1",
            "c1",
            "s1",
            "dialogue",
            "用户: 报销时限几天？\n助手: 3 天内提交。",
            55.0,
            90,
        )
        .await
        .unwrap();
        crate::memory::insert_event_memory(
            &pool,
            "u1",
            "c1",
            "s1",
            "semantic_memory_l1",
            "差旅报销时限是发生后 3 天。",
            70.0,
            90,
        )
        .await
        .unwrap();
        let queried = memory_query(
            &pool,
            "u1",
            &json!({"query":"报销时限","memory_scope":"event","top_k":2}),
            None,
            None,
            &[],
            None,
        )
        .await
        .unwrap();
        assert_eq!(queried.pointer("/summary/matched_count").and_then(|value| value.as_u64()), Some(2));
        assert_eq!(queried.pointer("/filters/top_k").and_then(|value| value.as_u64()), Some(2));
        assert_eq!(
            queried.pointer("/matched_memories/0/metadata/memory_key").and_then(|value| value.as_str()),
            Some("semantic_memory_l1")
        );
    }

    #[test]
    fn original_memory_hints_include_details_and_strip_whitespace() {
        assert!(should_disclose_original_memory("请展开 完整 对话"));
        assert!(should_disclose_original_memory("把刚才的细节说清楚"));
        assert!(!should_disclose_original_memory("我之前说过什么"));
        assert!(!should_disclose_original_memory("   "));
        let intent = route_local_skill("请展开之前摘要对应的完整对话").expect("original");
        assert_eq!(intent.name, "memory_retrieve_original");
        assert_eq!(intent.arguments.get("top_k").and_then(Value::as_u64), Some(8));
        assert_eq!(intent.arguments.get("max_chars").and_then(Value::as_u64), Some(12000));
    }

    #[tokio::test]
    async fn memory_query_merges_progressive_disclosure_from_matched_ids() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::memory::ensure_schema(&pool).await.unwrap();
        let summary_id = crate::memory::insert_event_memory_at(
            &pool,
            "u1",
            "c1",
            "s1",
            "semantic_memory_l1",
            "【主题】项目 Alpha\n【摘要】用户确认了上线计划。",
            70.0,
            90,
            Some(2),
            Some(3),
        )
        .await
        .unwrap();
        crate::memory::insert_event_memory_at(
            &pool,
            "u1",
            "c1",
            "s1",
            "dialogue",
            "用户: 请确认 Alpha 上线计划\n助手: 已确认本周五上线。",
            55.0,
            90,
            Some(2),
            Some(3),
        )
        .await
        .unwrap();
        let queried = memory_query(
            &pool,
            "u1",
            &json!({"query":"Alpha 上线","memory_scope":"event","top_k":5}),
            None,
            None,
            &[],
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            queried.pointer("/matched_memories/0/memory_id").and_then(Value::as_str),
            Some(summary_id.as_str())
        );
        let disclosure = maybe_run_progressive_disclosure(
            &pool,
            &SkillExecutionContext {
                conversation_id: Some("c1"),
                ..SkillExecutionContext::new("u1")
            },
            "请展开刚才摘要对应的完整对话",
            &queried,
            &[],
            true,
        )
        .await
        .unwrap()
        .expect("independent disclosure tool");
        let disclosed_ids = disclosure
            .arguments
            .get("memory_ids")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        assert!(
            disclosed_ids
                .iter()
                .any(|id| id.as_str() == Some(summary_id.as_str()))
        );
        assert_eq!(disclosure.arguments.get("include_neighbors").and_then(Value::as_u64), Some(1));
        assert!(
            disclosure
                .result
                .pointer("/original_chunks/0/content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .contains("Alpha 上线计划")
        );
        let merged = maybe_disclose_original_memory(
            &pool,
            &SkillExecutionContext {
                conversation_id: Some("c1"),
                ..SkillExecutionContext::new("u1")
            },
            "请展开刚才摘要对应的完整对话",
            queried,
            &[],
            true,
        )
        .await
        .unwrap();
        assert_eq!(merged.get("progressive_disclosure").and_then(Value::as_bool), Some(true));
        assert!(
            merged
                .pointer("/original_conversation/original_chunks/0/content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .contains("Alpha 上线计划")
        );
        let formatted = format_skill_answer("memory_query", &merged);
        assert!(formatted.contains("按你的要求展开原始对话"));
        assert!(formatted.contains("Alpha 上线计划"));
    }

    #[tokio::test]
    async fn memory_query_disclosure_falls_back_to_conversation_id() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::memory::ensure_schema(&pool).await.unwrap();
        crate::memory::insert_event_memory_at(
            &pool,
            "u1",
            "c1",
            "s1",
            "dialogue",
            "用户: 完整原文在这里\n助手: 已记下",
            55.0,
            90,
            Some(0),
            Some(0),
        )
        .await
        .unwrap();
        let queried = json!({
            "matched_memories": [],
            "profile_memories": [],
            "summary": {"profile_count": 0, "matched_count": 0},
        });
        let skipped = maybe_disclose_original_memory(
            &pool,
            &SkillExecutionContext {
                conversation_id: Some("c1"),
                ..SkillExecutionContext::new("u1")
            },
            "请展开原始对话",
            queried.clone(),
            &[],
            false,
        )
        .await
        .unwrap();
        assert!(skipped.get("progressive_disclosure").is_none());
        let merged = maybe_disclose_original_memory(
            &pool,
            &SkillExecutionContext {
                conversation_id: Some("c1"),
                ..SkillExecutionContext::new("u1")
            },
            "请展开原始对话",
            queried,
            &[],
            true,
        )
        .await
        .unwrap();
        assert_eq!(merged.get("progressive_disclosure").and_then(Value::as_bool), Some(true));
        assert_eq!(
            merged
                .pointer("/original_conversation/original_chunks")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(1)
        );
    }

    fn history(items: &[(&str, &str)]) -> Vec<crate::short_memory::MemoryMessage> {
        items
            .iter()
            .map(|(role, content)| crate::short_memory::MemoryMessage {
                role: (*role).into(),
                content: (*content).into(),
            })
            .collect()
    }

    #[tokio::test]
    async fn memory_query_merges_recent_history() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::memory::ensure_schema(&pool).await.unwrap();
        let queried = memory_query(
            &pool,
            "u1",
            &json!({"query":"我刚才说过什么","memory_scope":"event","top_k":3}),
            Some("c1"),
            Some("s1"),
            &history(&[
                ("user", "我刚才说过项目 Alpha 的上线计划"),
                ("assistant", "已记录在案"),
            ]),
            None,
        )
        .await
        .unwrap();
        let texts: Vec<&str> = queried
            .get("matched_memories")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|item| item.get("content").and_then(Value::as_str))
            .collect();
        assert!(texts.iter().any(|text| text.contains("项目 Alpha")));
        assert_eq!(
            queried
                .pointer("/matched_memories/0/metadata/chunk_type")
                .and_then(Value::as_str),
            Some("dialogue_current_turn")
        );
        assert_eq!(
            queried.pointer("/summary/matched_count").and_then(Value::as_u64),
            Some(1)
        );
    }

    #[tokio::test]
    async fn memory_query_skips_duplicate_recent_history_and_older_turns() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::memory::ensure_schema(&pool).await.unwrap();
        let same_turn = "用户: 我刚才说过项目 Alpha 的上线计划\n助手: 已记录在案";
        crate::memory::insert_event_memory(
            &pool,
            "u1",
            "c1",
            "s1",
            "dialogue",
            same_turn,
            55.0,
            90,
        )
        .await
        .unwrap();
        let queried = memory_query(
            &pool,
            "u1",
            &json!({"query":"项目 Alpha","memory_scope":"event","top_k":1}),
            Some("c1"),
            Some("s1"),
            &history(&[
                ("user", "更早的旧问题"),
                ("assistant", "更早的旧回答"),
                ("user", "我刚才说过项目 Alpha 的上线计划"),
                ("assistant", "已记录在案"),
            ]),
            None,
        )
        .await
        .unwrap();
        let matched = queried
            .get("matched_memories")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let current_turns: Vec<&Value> = matched
            .iter()
            .filter(|item| {
                item.pointer("/metadata/chunk_type").and_then(Value::as_str)
                    == Some("dialogue_current_turn")
            })
            .collect();
        assert!(current_turns.is_empty(), "已入库的同一轮不应再合并");
        assert!(
            matched
                .iter()
                .all(|item| !item.get("content").and_then(Value::as_str).unwrap_or("").contains("更早的旧问题")),
            "超出 top_k 的更早轮次不应进入结果"
        );
    }

    #[tokio::test]
    async fn memory_query_profile_scope_skips_recent_history() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        crate::memory::ensure_schema(&pool).await.unwrap();
        let queried = memory_query(
            &pool,
            "u1",
            &json!({"query":"我刚才说过什么","memory_scope":"profile","top_k":3}),
            Some("c1"),
            Some("s1"),
            &history(&[("user", "我刚才说过项目 Alpha 的上线计划")]),
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            queried.pointer("/summary/matched_count").and_then(Value::as_u64),
            Some(0)
        );
    }
}
