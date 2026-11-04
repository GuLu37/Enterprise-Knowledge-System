//! 本地 Skill：规则路由 + SQLite 执行。
//!
//! 覆盖前端会直接碰到的工具目录、知识库文件、热词和画像记忆。
//! 远程 MCP 通过 `MCP_REMOTE_TOOL_MANIFESTS` 注册进同一套目录；未就绪工具
//! 只出现在 capabilities，不进入 Function Calling。

use std::collections::{BTreeMap, HashMap};

use serde_json::{Value, json};

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
];

const MEMORY_QUERY_HINTS: &[&str] = &[
    "记得什么",
    "我之前说过",
    "我的偏好",
    "记忆",
    "画像",
];

const HOT_WORD_HINTS: &[&str] = &["热词", "高频词", "统计词频", "词频统计"];

const FILE_LIST_HINTS: &[&str] = &[
    "有哪些文件",
    "有哪些文档",
    "有什么文件",
    "有什么文档",
    "列出文件",
    "列出文档",
    "文件列表",
    "文档列表",
    "已上传的文档",
    "已上传的文件",
    "上传了哪些",
    "知识库文件",
    "知识库文档",
    "这份文档",
    "这个文档",
    "讲了什么",
];

const FILE_LIST_CONTENT_HINTS: &[&str] = &[
    "怎么办", "如何", "是什么", "流程", "规定", "标准", "多少钱",
];

const DELETE_PROFILE_HINTS: &[&str] = &[
    "删除记忆",
    "清除记忆",
    "忘掉",
    "别再记",
    "忘记我的",
    "删除我的",
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
    pub arguments: Value,
}

/// 规则优先的本地技能路由。命中后聊天不再进入知识库检索。
pub fn route_local_skill(query: &str) -> Option<RoutedSkill> {
    let normalized = query.trim();
    if normalized.is_empty() {
        return None;
    }
    let lowered = normalized.to_lowercase();

    if PROFILE_NAME_HINTS.iter().any(|hint| lowered.contains(hint)) {
        return Some(RoutedSkill {
            name: "memory_query",
            arguments: json!({
                "query": normalized,
                "top_k": 5,
                "memory_scope": "profile",
            }),
        });
    }
    if TOOL_CATALOG_HINTS.iter().any(|hint| lowered.contains(hint)) {
        return Some(RoutedSkill {
            name: "list_available_tools",
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
            arguments,
        });
    }
    if DELETE_PROFILE_HINTS.iter().any(|hint| lowered.contains(hint)) {
        let keywords = extract_delete_keywords(normalized);
        return Some(RoutedSkill {
            name: "delete_profile_memory",
            arguments: json!({ "keywords": keywords, "max_deletes": 50 }),
        });
    }
    if is_file_list_query(&lowered) {
        let mut arguments = json!({ "summarize": true, "limit": 20 });
        let file_name = extract_file_name_hint(normalized);
        if !file_name.is_empty() {
            arguments["file_name"] = json!(file_name);
        }
        return Some(RoutedSkill {
            name: "list_knowledge_base_files",
            arguments,
        });
    }
    None
}

fn is_file_list_query(lowered: &str) -> bool {
    if !FILE_LIST_HINTS.iter().any(|hint| lowered.contains(hint)) {
        return false;
    }
    !FILE_LIST_CONTENT_HINTS.iter().any(|hint| lowered.contains(hint))
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

fn extract_delete_keywords(query: &str) -> Vec<String> {
    let mut text = query.to_string();
    for hint in DELETE_PROFILE_HINTS {
        text = text.replace(hint, " ");
    }
    text.split(|ch: char| ch.is_whitespace() || matches!(ch, '，' | '。' | '、' | ',' | '的' | '了'))
        .map(str::trim)
        .filter(|part| part.chars().count() >= 1 && !matches!(*part, "我" | "你" | "把" | "请"))
        .map(str::to_string)
        .take(8)
        .collect()
}

pub async fn execute_named(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    name: &str,
    arguments: &Value,
) -> Result<Value, String> {
    execute_named_from(pool, user_id, name, arguments, &[]).await
}

pub async fn execute_named_from(
    pool: &sqlx::SqlitePool,
    user_id: &str,
    name: &str,
    arguments: &Value,
    manifests: &[Value],
) -> Result<Value, String> {
    if let Some(definition) = get_skill_definition_from(name, manifests) {
        if definition.kind == "remote_mcp" {
            return Err(crate::remote_mcp::execution_error_message(&definition));
        }
    }
    match name {
        "list_available_tools" => Ok(list_available_tools(manifests)),
        "list_knowledge_base_files" => list_knowledge_base_files(pool, arguments).await,
        "document_hot_word_statistics" => hot_words(pool, arguments).await,
        "memory_query" => memory_query(pool, user_id, arguments).await,
        "memory_retrieve_original" => memory_original(pool, user_id, arguments).await,
        "delete_profile_memory" => delete_profile_memory(pool, user_id, arguments).await,
        "search_knowledge_base" => Err("检索请走 /retrieval 或聊天接口".into()),
        other => Err(format!("未知技能: {other}")),
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
    let profiles = result.get("profile_memories").and_then(|value| value.as_array()).cloned().unwrap_or_default();
    let matched = result.get("matched_memories").and_then(|value| value.as_array()).cloned().unwrap_or_default();
    if profiles.is_empty() && matched.is_empty() {
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
    lines.join("\n")
}

fn format_original(result: &Value) -> String {
    let content = result.get("content").and_then(|value| value.as_str()).unwrap_or("").trim();
    if content.is_empty() {
        return "没有找到可展开的原始对话。".into();
    }
    format!("按你的要求展开原始对话：\n\n{content}")
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
        metadata: json!({"returns": returns, "executor_ready": true}),
    }
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
                    "file_name": {"type": "string", "description": "可选，按原始文件名或存储文件名模糊定位目标文档。"},
                    "document_ids": {"type": "array", "items": {"type": "string"}},
                    "file_types": {"type": "array", "items": {"type": "string"}},
                    "statuses": {"type": "array", "items": {"type": "string", "enum": ["processing", "ready", "failed", "deleting", "deleted"]}},
                    "include_samples": {"type": "boolean", "default": true}
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
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                    "memory_scope": {"type": "string", "enum": ["all", "profile", "event"], "default": "all"}
                }
            }),
            "memory_query_result",
        ),
        skill_def(
            "memory_retrieve_original",
            "按长期记忆摘要的引用渐进式披露当前用户对应的原始对话。当摘要不足以回答问题、用户要求查看原文/完整上下文/具体细节，或需要核对原始措辞时使用。",
            "memory",
            &["memory", "conversation", "original", "progressive_disclosure"],
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "memory_ids": {"type": "array", "items": {"type": "string"}},
                    "query": {"type": "string"},
                    "conversation_id": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 500, "maximum": 50000, "default": 12000}
                }
            }),
            "original_conversation_result",
        ),
        skill_def(
            "delete_profile_memory",
            "删除用户的个性化记忆（画像记忆）。当用户明确要求删除、遗忘、清除某条个人信息时使用。",
            "memory",
            &["memory", "profile", "delete", "privacy"],
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "memory_keys": {"type": "array", "items": {"type": "string"}},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "max_deletes": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50}
                }
            }),
            "delete_profile_memory_result",
        ),
        skill_def(
            "list_knowledge_base_files",
            "列出知识库中已上传的文档，可按文件名、文档 ID、文件类型或状态筛选，并可概括指定文档的内容。不用于按内容检索具体答案，那应使用 search_knowledge_base。",
            "document",
            &["documents", "list", "summary", "overview"],
            json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "file_name": {"type": "string"},
                    "document_ids": {"type": "array", "items": {"type": "string"}},
                    "file_types": {"type": "array", "items": {"type": "string"}},
                    "statuses": {"type": "array", "items": {"type": "string"}},
                    "summarize": {"type": "boolean", "default": true},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}
                }
            }),
            "knowledge_base_file_list",
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

pub fn wrap_skill_execution(
    name: &str,
    arguments: &Value,
    user_id: &str,
    conversation_id: Option<&str>,
    session_id: Option<&str>,
    query: Option<&str>,
    trace_id: Option<&str>,
    metadata: Value,
    result: Value,
) -> Result<Value, String> {
    let definition = get_skill_definition(name).ok_or_else(|| format!("未找到技能: {name}"))?;
    let executed_at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true);
    Ok(json!({
        "skill": definition.name,
        "category": definition.category,
        "kind": definition.kind,
        "provider": definition.provider,
        "arguments": arguments,
        "context": {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "query": query,
            "trace_id": trace_id,
            "metadata": if metadata.is_null() { json!({}) } else { metadata },
        },
        "result": result,
        "executed_at": executed_at,
    }))
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
            format!("- {key}：{}", row.content)
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

async fn list_knowledge_base_files(pool: &sqlx::SqlitePool, arguments: &Value) -> Result<Value, String> {
    let file_name = arg_str(arguments, "file_name");
    let limit = arg_usize(arguments, "limit", 20, 1, 100);
    let summarize = arguments.get("summarize").and_then(|value| value.as_bool()).unwrap_or(true);
    let preview_chars = arg_usize(arguments, "preview_chars", 1500, 200, 6000);
    let statuses = arg_strings(arguments, "statuses");
    let statuses = if statuses.is_empty() { vec!["ready".to_string()] } else { statuses };
    let file_types = arg_strings(arguments, "file_types");
    let document_ids = arg_strings(arguments, "document_ids");

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
            "filters": {"file_name": file_name, "statuses": statuses, "limit": limit},
        }));
    }

    let mut documents_out = Vec::new();
    for record in &documents {
        let mut doc = json!({
            "document_id": record.document_id,
            "original_filename": record.original_filename,
            "stored_filename": record.stored_filename,
            "file_type": record.file_type,
            "status": record.status,
            "file_size": record.file_size,
            "chunk_count": record.chunk_count,
            "created_at": crate::db::dt_to_iso(&record.created_at),
            "updated_at": crate::db::dt_to_iso(&record.updated_at),
        });
        if summarize {
            let chunks = load_chunk_texts(pool, &record.document_id).await?;
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
        "filters": {"file_name": file_name, "statuses": statuses, "limit": limit},
    }))
}

async fn hot_words(pool: &sqlx::SqlitePool, arguments: &Value) -> Result<Value, String> {
    let top_k = arg_usize(arguments, "top_k", 10, 1, 50);
    let file_name = arg_str(arguments, "file_name");
    let mut listed = list_knowledge_base_files(
        pool,
        &json!({
            "file_name": file_name,
            "statuses": arguments.get("statuses").cloned().unwrap_or(json!([])),
            "file_types": arguments.get("file_types").cloned().unwrap_or(json!([])),
            "document_ids": arguments.get("document_ids").cloned().unwrap_or(json!([])),
            "summarize": false,
            "limit": 100,
        }),
    )
    .await?;
    let documents = listed
        .get_mut("documents")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();
    if documents.is_empty() {
        return Ok(json!({
            "scope": "documents",
            "summary": {"document_count": 0, "chunk_count": 0},
            "top_words": [],
        }));
    }

    let mut term_frequency: HashMap<String, usize> = HashMap::new();
    let mut term_documents: HashMap<String, Vec<Value>> = HashMap::new();
    for doc in &documents {
        let document_id = doc.get("document_id").and_then(|value| value.as_str()).unwrap_or("");
        let chunks = load_chunk_texts(pool, document_id).await?;
        let combined = chunks.join(" ");
        let limited: String = combined.chars().take(12000).collect();
        for (term, count) in extract_term_frequencies(&limited) {
            *term_frequency.entry(term.clone()).or_default() += count;
            let samples = term_documents.entry(term).or_default();
            if samples.len() < 3 && !samples.iter().any(|item| item.get("document_id") == doc.get("document_id")) {
                samples.push(doc.clone());
            }
        }
    }
    let mut ranked: Vec<(String, usize)> = term_frequency.into_iter().collect();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| right.0.len().cmp(&left.0.len())).then_with(|| left.0.cmp(&right.0)));
    let top_words: Vec<Value> = ranked
        .into_iter()
        .take(top_k)
        .map(|(term, count)| {
            let samples = term_documents.get(&term).cloned().unwrap_or_default();
            json!({
                "term": term,
                "document_count": count,
                "sample_documents": samples,
            })
        })
        .collect();
    let chunk_count = documents.iter().map(|doc| doc.get("chunk_count").and_then(|value| value.as_i64()).unwrap_or(0)).sum::<i64>();
    Ok(json!({
        "scope": "documents",
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

async fn memory_query(pool: &sqlx::SqlitePool, user_id: &str, arguments: &Value) -> Result<Value, String> {
    let query = arg_str(arguments, "query");
    let top_k = arg_usize(arguments, "top_k", 5, 1, 50);
    let scope = arg_str(arguments, "memory_scope");
    let scope = if scope.is_empty() { "all".to_string() } else { scope };
    let include_profile = scope == "all" || scope == "profile";
    let include_event = scope == "all" || scope == "event";

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
            }))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let matched_memories = if include_event {
        crate::memory::search_event_memories(pool, user_id, &query, None, None, top_k, None)
            .await
            .into_iter()
            .map(|hit| json!({
                "memory_id": hit.memory_id,
                "content": hit.content,
                "score": hit.score,
                "source": hit.memory_key,
                "metadata": {
                    "conversation_id": hit.conversation_id,
                    "memory_key": hit.memory_key,
                    "chunk_type": hit.memory_key,
                    "turn_start": hit.turn_start,
                    "turn_end": hit.turn_end,
                },
            }))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let profile_count = profile_memories.len();
    let matched_count = matched_memories.len();

    Ok(json!({
        "query": query,
        "memory_scope": scope,
        "profile_memories": profile_memories,
        "matched_memories": matched_memories,
        "summary": {
            "profile_count": profile_count,
            "matched_count": matched_count,
        },
    }))
}

async fn memory_original(pool: &sqlx::SqlitePool, user_id: &str, arguments: &Value) -> Result<Value, String> {
    let max_chars = arg_usize(arguments, "max_chars", 12000, 500, 50000);
    let query = arg_str(arguments, "query");
    let terms = query_terms(&query);
    let rows = load_memories(pool, user_id, None, 400).await?;
    let mut pieces = Vec::new();
    let mut used = 0usize;
    for row in rows.into_iter().filter(memory_active).filter(|row| row.memory_type != "profile") {
        let normalized = eks_kernels::normalize_text(&row.content);
        if !terms.is_empty() && !terms.iter().any(|term| normalized.contains(term)) {
            continue;
        }
        if used >= max_chars {
            break;
        }
        let remain = max_chars - used;
        let piece: String = row.content.chars().take(remain).collect();
        used += piece.chars().count();
        pieces.push(piece);
        if pieces.len() >= 8 {
            break;
        }
    }
    Ok(json!({
        "query": query,
        "content": pieces.join("\n\n"),
        "piece_count": pieces.len(),
    }))
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
    let matched: Vec<MemoryRow> = rows
        .into_iter()
        .filter(memory_active)
        .filter(|row| {
            let key = row.memory_key.as_deref().unwrap_or("").trim().to_lowercase();
            let content = eks_kernels::normalize_text(&row.content);
            normalized_keys.iter().any(|item| item == &key)
                || normalized_keywords.iter().any(|word| content.contains(word) || key.contains(word))
        })
        .take(max_deletes)
        .collect();
    if matched.is_empty() {
        return Ok(json!({
            "deleted_count": 0,
            "matched": [],
            "summary": "当前没有匹配的画像记忆。",
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
        .map(|row| json!({"memory_id": row.memory_id, "memory_key": row.memory_key, "content": row.content}))
        .collect();
    Ok(json!({
        "deleted_count": matched_json.len(),
        "matched": matched_json,
        "summary": format!("已删除 {} 条画像记忆。", matched.len()),
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

async fn load_chunk_texts(pool: &sqlx::SqlitePool, document_id: &str) -> Result<Vec<String>, String> {
    let rows: Vec<String> = sqlx::query_scalar(
        "SELECT chunk_text FROM document_chunks WHERE document_id = ? ORDER BY chunk_index ASC",
    )
    .bind(document_id)
    .fetch_all(pool)
    .await
    .map_err(|err| err.to_string())?;
    Ok(rows.into_iter().map(|text| text.trim().to_string()).filter(|text| !text.is_empty()).collect())
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

fn arg_strings(arguments: &Value, key: &str) -> Vec<String> {
    match arguments.get(key) {
        Some(Value::Array(items)) => items.iter().filter_map(|item| item.as_str()).map(|item| item.trim().to_string()).filter(|item| !item.is_empty()).collect(),
        Some(Value::String(item)) => item.split(',').map(str::trim).filter(|item| !item.is_empty()).map(str::to_string).collect(),
        _ => Vec::new(),
    }
}

fn arg_usize(arguments: &Value, key: &str, default: usize, min: usize, max: usize) -> usize {
    arguments
        .get(key)
        .and_then(|value| value.as_u64().or_else(|| value.as_i64().map(|item| item as u64)))
        .map(|value| value as usize)
        .unwrap_or(default)
        .clamp(min, max)
}

fn query_terms(query: &str) -> Vec<String> {
    extract_term_frequencies(query)
        .into_iter()
        .filter(|(_, count)| *count > 0)
        .map(|(term, _)| term)
        .filter(|term| term.chars().count() >= 2)
        .take(12)
        .collect()
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

    #[test]
    fn routes_tool_catalog_and_file_list() {
        let tools = route_local_skill("列出你的工具").expect("tool");
        assert_eq!(tools.name, "list_available_tools");
        let files = route_local_skill("知识库里有哪些文档").expect("files");
        assert_eq!(files.name, "list_knowledge_base_files");
        assert!(route_local_skill("差旅报销流程是什么").is_none());
        let catalog = function_calling_tools();
        let names: Vec<&str> = catalog
            .iter()
            .filter_map(|tool| tool.pointer("/function/name").and_then(|value| value.as_str()))
            .collect();
        assert!(names.contains(&"search_knowledge_base"));
        assert!(names.contains(&"memory_query"));
        assert_eq!(names.len(), 7);
        assert_eq!(names[0], "search_knowledge_base");
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
            "u1",
            Some("c1"),
            None,
            Some("我叫什么"),
            None,
            json!({}),
            json!({"ok": true}),
        )
        .unwrap();
        assert_eq!(wrapped.get("skill").and_then(|value| value.as_str()), Some("memory_query"));
        assert_eq!(wrapped.get("category").and_then(|value| value.as_str()), Some("memory"));
        assert_eq!(wrapped.get("kind").and_then(|value| value.as_str()), Some("local"));
        assert!(wrapped.get("executed_at").and_then(|value| value.as_str()).is_some());
        assert_eq!(
            wrapped.pointer("/context/conversation_id").and_then(|value| value.as_str()),
            Some("c1")
        );
        assert!(wrap_skill_execution("missing", &json!({}), "u1", None, None, None, None, json!({}), json!({})).is_err());
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
            "u1",
            "crm_lookup",
            &json!({}),
            &manifests,
        )
        .await
        .unwrap_err();
        assert!(err.contains("尚未接入实际执行器"));
        let ready_err = execute_named_from(
            &sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap(),
            "u1",
            "ticket_create",
            &json!({}),
            &manifests,
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
        let queried = memory_query(&pool, "u1", &json!({"query":"我的名字","memory_scope":"profile","top_k":5}))
            .await
            .unwrap();
        assert_eq!(queried.pointer("/summary/profile_count").and_then(|value| value.as_u64()), Some(1));
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
        )
        .await
        .unwrap();
        let queried = memory_query(
            &pool,
            "u1",
            &json!({"query":"报销时限","memory_scope":"event","top_k":2}),
        )
        .await
        .unwrap();
        assert_eq!(queried.pointer("/summary/matched_count").and_then(|value| value.as_u64()), Some(2));
        assert_eq!(
            queried.pointer("/matched_memories/0/metadata/memory_key").and_then(|value| value.as_str()),
            Some("semantic_memory_l1")
        );
    }
}
