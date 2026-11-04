//! 意图路由：规则快判 skill/direct/rag，低置信时再交给 LLM 消歧。
//!
//! 对齐 Python `route_query_action` / `route_query_intent`：
//! 明确技能词走本地 Skill；编号查询和知识词走 RAG；闲聊直答；
//! 直答词与知识词冲突或默认模糊时，才调用 LLM 在统一工具表里选择动作。

use serde_json::{json, Value};

use crate::llm::{ChatMessage, LlmClient};
use crate::skills;

const KNOWLEDGE_INTENT_KEYWORDS: &[&str] = &[
    "知识库", "文档", "资料", "上传", "附件", "引用", "来源", "检索", "搜索", "查一下", "查询",
    "根据", "基于", "结合", "公司", "企业", "内部", "制度", "流程", "规范", "政策", "手册",
    "项目", "合同", "编号", "编码", "工号", "员工编号", "员工号", "产品", "产品编号", "产品号",
    "型号", "物料", "订单", "申请", "权限", "配置", "环境", "接口", "操作", "使用说明", "账号",
    "密码", "登录", "注册", "常见问题", "修改密码", "重置密码", "找回密码", "账号异常", "登录失败",
    "报销", "审批", "退款", "退货", "退订", "售后", "员工", "客户",
];

const DIRECT_INTENT_KEYWORDS: &[&str] = &[
    "你好", "在吗", "谢谢", "翻译", "润色", "改写", "写一段", "生成", "创作", "代码", "解释一下这段",
];

const AGENT_ROUTER_SYSTEM_PROMPT: &str = "你是企业知识库问答的统一动作路由器。\
请查看工具列表，然后决定最合适的动作。工具列表同时包含 RAG 工具和可调用 Skill；\
工具描述与参数说明是判断 Skill 适用场景的唯一发现入口。\
工具列表里的 function calling schema 才是可执行入口，选中的工具必须通过 function calling 执行。\
RAG 也是工具，名字是 search_knowledge_base，不是特例。\
只输出严格 JSON，不要解释、前后缀或代码块。\
格式：\
{\"route\":\"direct\"|\"skill\"|\"rag\"|\"clarify\",\
\"reason\":\"简短原因\",\
\"confidence\":0到1之间的小数,\
\"query\":\"需要时的检索改写\",\
\"skill_name\":\"需要调用的工具或技能名\",\
\"skill_arguments\":{},\
\"clarify_question\":\"需要追问时的一句问题\"}\
规则：\
如果能直接回答就 route=direct。\
如果应该调用工具，route=skill，并在 skill_name 里填写工具名；RAG 也写成 skill，但 tool name 必须是 search_knowledge_base。\
如果信息不够，route=clarify。";

#[derive(Debug, Clone)]
pub struct QueryIntent {
    pub route: String,
    pub needs_retrieval: bool,
    pub reason: String,
    pub source: String,
    pub confidence: Option<f64>,
    pub rewritten_query: String,
    pub skill_name: Option<String>,
    pub skill_arguments: Value,
    pub clarify_question: String,
}

impl QueryIntent {
    fn direct(reason: &str, source: &str) -> Self {
        QueryIntent {
            route: "direct".into(),
            needs_retrieval: false,
            reason: reason.into(),
            source: source.into(),
            confidence: None,
            rewritten_query: String::new(),
            skill_name: None,
            skill_arguments: json!({}),
            clarify_question: String::new(),
        }
    }

    fn rag(reason: &str, source: &str, query: &str) -> Self {
        QueryIntent {
            route: "rag".into(),
            needs_retrieval: true,
            reason: reason.into(),
            source: source.into(),
            confidence: None,
            rewritten_query: query.trim().to_string(),
            skill_name: None,
            skill_arguments: json!({}),
            clarify_question: String::new(),
        }
    }

    pub fn retrieval_query<'a>(&'a self, original: &'a str) -> &'a str {
        let rewritten = self.rewritten_query.trim();
        if rewritten.is_empty() {
            original
        } else {
            rewritten
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "route": self.route,
            "needs_retrieval": self.needs_retrieval,
            "reason": self.reason,
            "source": self.source,
            "confidence": self.confidence,
            "query": self.rewritten_query,
            "skill_name": self.skill_name,
            "skill_arguments": self.skill_arguments,
            "clarify_question": self.clarify_question,
        })
    }
}

pub fn route_signals(query: &str) -> (bool, bool) {
    let lowered = query.trim().to_lowercase();
    let has_direct = DIRECT_INTENT_KEYWORDS.iter().any(|keyword| lowered.contains(keyword));
    let has_knowledge = KNOWLEDGE_INTENT_KEYWORDS.iter().any(|keyword| lowered.contains(keyword));
    (has_direct, has_knowledge)
}

pub fn has_signal_conflict(query: &str) -> bool {
    let (has_direct, has_knowledge) = route_signals(query);
    has_direct && has_knowledge
}

pub fn rule_route_query_intent(query: &str, use_retrieval: bool) -> QueryIntent {
    let normalized = query.trim();
    if !use_retrieval {
        return QueryIntent::direct("retrieval_disabled", "rule");
    }
    if normalized.is_empty() {
        return QueryIntent::direct("empty_query", "rule");
    }
    let (has_direct, has_knowledge) = route_signals(normalized);
    if has_direct && has_knowledge {
        let mut intent = QueryIntent::rag("signal_conflict", "rule", normalized);
        intent.confidence = Some(0.2);
        return intent;
    }
    if eks_kernels::rerank::is_identifier_query(normalized) {
        return QueryIntent::rag("identifier_query", "rule", normalized);
    }
    if has_direct {
        return QueryIntent::direct("direct_keyword", "rule");
    }
    if has_knowledge {
        return QueryIntent::rag("knowledge_keyword", "rule", normalized);
    }
    QueryIntent::direct("direct_default", "rule")
}

pub fn fallback_route_query_intent(query: &str, use_retrieval: bool) -> QueryIntent {
    let normalized = query.trim();
    if !use_retrieval {
        return QueryIntent::direct("retrieval_disabled", "fallback");
    }
    if normalized.is_empty() {
        return QueryIntent::direct("empty_query", "fallback");
    }
    let (has_direct, has_knowledge) = route_signals(normalized);
    if eks_kernels::rerank::is_identifier_query(normalized) || has_knowledge {
        let mut reason = if eks_kernels::rerank::is_identifier_query(normalized) {
            "identifier_query"
        } else {
            "knowledge_keyword"
        };
        if has_direct {
            reason = "signal_conflict";
        }
        let mut intent = QueryIntent::rag(&format!("fallback:{reason}"), "fallback", normalized);
        intent.confidence = Some(0.0);
        return intent;
    }
    if has_direct {
        let mut intent = QueryIntent::direct("fallback:direct_keyword", "fallback");
        intent.confidence = Some(0.0);
        return intent;
    }
    let mut intent = QueryIntent::direct("fallback:ambiguous_llm_default", "fallback");
    intent.confidence = Some(0.0);
    intent
}

fn intent_from_routed_skill(skill: &skills::RoutedSkill) -> QueryIntent {
    QueryIntent {
        route: "skill".into(),
        needs_retrieval: false,
        reason: format!("rule_skill:{}", skill.name),
        source: "rule".into(),
        confidence: None,
        rewritten_query: String::new(),
        skill_name: Some(skill.name.to_string()),
        skill_arguments: skill.arguments.clone(),
        clarify_question: String::new(),
    }
}

pub fn rule_route_query_action(query: &str, use_retrieval: bool) -> QueryIntent {
    let normalized = query.trim();
    if normalized.is_empty() {
        return QueryIntent::direct("empty_query", "rule");
    }
    if let Some(skill) = skills::route_local_skill(normalized) {
        return intent_from_routed_skill(&skill);
    }
    let mut intent = rule_route_query_intent(normalized, use_retrieval);
    if intent.reason == "direct_keyword" || !intent.needs_retrieval {
        intent.route = "direct".into();
        intent.needs_retrieval = false;
    } else {
        intent.route = "rag".into();
        intent.needs_retrieval = true;
    }
    intent
}

pub async fn route_query_action(
    query: &str,
    use_retrieval: bool,
    llm: Option<&LlmClient>,
    history: &[(String, String)],
    mcp_manifests: &[Value],
) -> QueryIntent {
    route_query_action_with(query, use_retrieval, llm, history, mcp_manifests, true).await
}

fn py_optional_float(value: Option<f64>) -> String {
    match value {
        None => "None".into(),
        Some(value) if value.fract() == 0.0 && value.is_finite() => format!("{value:.1}"),
        Some(value) => value.to_string(),
    }
}

fn skill_log_value(skill_name: &Option<String>) -> &str {
    skill_name.as_deref().unwrap_or("None")
}

fn format_intent_route_result(intent: &QueryIntent, with_confidence: bool) -> String {
    let route = if intent.needs_retrieval { "rag" } else { "direct" };
    if with_confidence {
        format!(
            "意图路由结果: source={} route={} confidence={} reason={}",
            intent.source,
            route,
            py_optional_float(intent.confidence),
            intent.reason
        )
    } else {
        format!(
            "意图路由结果: source={} route={} reason={}",
            intent.source, route, intent.reason
        )
    }
}

fn format_action_route_result(intent: &QueryIntent, with_confidence: bool) -> String {
    if with_confidence {
        format!(
            "动作路由结果: source={} route={} confidence={} reason={} skill={}",
            intent.source,
            intent.route,
            py_optional_float(intent.confidence),
            intent.reason,
            skill_log_value(&intent.skill_name)
        )
    } else {
        format!(
            "动作路由结果: source={} route={} reason={} skill={}",
            intent.source,
            intent.route,
            intent.reason,
            skill_log_value(&intent.skill_name)
        )
    }
}

fn log_action_route(intent: QueryIntent, with_confidence: bool) -> QueryIntent {
    tracing::info!("{}", format_action_route_result(&intent, with_confidence));
    intent
}

/// 对齐 Python `route_query_intent(..., llm=None)`：规则早退打无 confidence 日志，
/// 冲突/默认直答走 fallback 并带 confidence。
fn route_query_intent_without_llm(query: &str, use_retrieval: bool) -> QueryIntent {
    let normalized = query.trim();
    if !use_retrieval {
        return QueryIntent::direct("retrieval_disabled", "rule");
    }
    if normalized.is_empty() {
        return QueryIntent::direct("empty_query", "rule");
    }
    if eks_kernels::rerank::is_identifier_query(normalized) && !has_signal_conflict(normalized) {
        let intent = QueryIntent::rag("identifier_query", "rule", normalized);
        tracing::info!("{}", format_intent_route_result(&intent, false));
        return intent;
    }
    let rule_intent = rule_route_query_intent(normalized, use_retrieval);
    if rule_intent.reason == "direct_keyword" {
        tracing::info!("{}", format_intent_route_result(&rule_intent, false));
        return rule_intent;
    }
    if rule_intent.needs_retrieval && rule_intent.reason != "signal_conflict" {
        tracing::info!("{}", format_intent_route_result(&rule_intent, false));
        return rule_intent;
    }
    let fallback = fallback_route_query_intent(normalized, use_retrieval);
    tracing::info!("{}", format_intent_route_result(&fallback, true));
    fallback
}

pub async fn route_query_action_with(
    query: &str,
    use_retrieval: bool,
    llm: Option<&LlmClient>,
    history: &[(String, String)],
    mcp_manifests: &[Value],
    prefer_rule_skill: bool,
) -> QueryIntent {
    let normalized = query.trim();
    if normalized.is_empty() {
        return QueryIntent::direct("empty_query", "rule");
    }
    let rule_skill = skills::route_local_skill(normalized);
    if prefer_rule_skill {
        if let Some(skill) = &rule_skill {
            return log_action_route(intent_from_routed_skill(skill), false);
        }
    }

    let fallback = route_query_intent_without_llm(normalized, use_retrieval);
    if let Some(llm) = llm {
        match llm_select_action(llm, normalized, history, mcp_manifests).await {
            Ok(Some(mut action)) => {
                if action.route == "skill" && action.skill_name.as_deref() == Some("search_knowledge_base") {
                    let retrieval_query = action
                        .skill_arguments
                        .get("query")
                        .and_then(|value| value.as_str())
                        .unwrap_or(normalized)
                        .trim()
                        .to_string();
                    action.route = "rag".into();
                    action.needs_retrieval = true;
                    action.rewritten_query = if retrieval_query.is_empty() {
                        normalized.to_string()
                    } else {
                        retrieval_query
                    };
                } else if action.route == "rag" && !use_retrieval {
                    action = QueryIntent::direct("retrieval_disabled", &action.source);
                }
                let skill_missing =
                    action.route == "skill" && action.skill_name.as_deref().unwrap_or("").is_empty();
                if !skill_missing {
                    if action.route == "direct" {
                        if prefer_rule_skill {
                            if let Some(skill) = &rule_skill {
                                action = intent_from_routed_skill(skill);
                            }
                        }
                        if action.route == "direct" && fallback.needs_retrieval {
                            action = fallback.clone();
                            action = finalize_fallback(action);
                        }
                    }
                    return log_action_route(action, true);
                }
            }
            Ok(None) => {}
            Err(err) => tracing::warn!("LLM 动作路由失败，进入最终兜底: {err}"),
        }
    }

    if let Some(skill) = &rule_skill {
        return log_action_route(intent_from_routed_skill(skill), false);
    }
    log_action_route(finalize_fallback(fallback), true)
}

fn finalize_fallback(mut intent: QueryIntent) -> QueryIntent {
    if intent.reason == "direct_keyword" || !intent.needs_retrieval {
        intent.route = "direct".into();
        intent.needs_retrieval = false;
    } else {
        intent.route = "rag".into();
        intent.needs_retrieval = true;
    }
    intent
}

async fn llm_select_action(
    llm: &LlmClient,
    query: &str,
    history: &[(String, String)],
    mcp_manifests: &[Value],
) -> Result<Option<QueryIntent>, String> {
    let tools = skills::function_calling_tools_from(mcp_manifests);
    let catalog: Vec<Value> = tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.pointer("/function/name"),
                "description": tool.pointer("/function/description"),
            })
        })
        .collect();
    let mut messages = vec![
        ChatMessage::new("system", AGENT_ROUTER_SYSTEM_PROMPT),
        ChatMessage::new(
            "user",
            format!(
                "用户消息：\n{}\n\n可用工具列表（包含 Skill）：\n{}",
                query,
                serde_json::to_string(&catalog).unwrap_or_else(|_| "[]".into())
            ),
        ),
    ];
    let history_hint = render_history(history, 5);
    if !history_hint.is_empty() {
        messages.push(ChatMessage::new("user", format!("最近对话上下文：\n{history_hint}")));
    }

    let completion = match llm.invoke_with_tools(&messages, &tools).await {
        Ok(result) => result,
        Err(_) => llm.invoke(&messages).await?,
    };
    if let Some(call) = completion.tool_calls.first() {
        if let Some(intent) = intent_from_tool_call(&call.name, &call.arguments, query) {
            return Ok(Some(intent));
        }
    }
    Ok(intent_from_action_payload(&parse_json_object(&completion.content), query))
}

fn intent_from_tool_call(name: &str, arguments: &Value, query: &str) -> Option<QueryIntent> {
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    if name == "search_knowledge_base" {
        let retrieval_query = arguments
            .get("query")
            .and_then(|value| value.as_str())
            .unwrap_or(query)
            .trim();
        if retrieval_query.is_empty() {
            return None;
        }
        return Some(QueryIntent {
            route: "rag".into(),
            needs_retrieval: true,
            reason: "llm:tool_call".into(),
            source: "llm".into(),
            confidence: Some(1.0),
            rewritten_query: retrieval_query.to_string(),
            skill_name: None,
            skill_arguments: arguments.clone(),
            clarify_question: String::new(),
        });
    }
    Some(QueryIntent {
        route: "skill".into(),
        needs_retrieval: false,
        reason: "llm:tool_call".into(),
        source: "llm".into(),
        confidence: Some(1.0),
        rewritten_query: String::new(),
        skill_name: Some(name.to_string()),
        skill_arguments: crate::skills::normalize_action_arguments(name, arguments, query),
        clarify_question: String::new(),
    })
}

fn intent_from_action_payload(payload: &Value, query: &str) -> Option<QueryIntent> {
    let route = payload
        .get("route")
        .or_else(|| payload.get("intent"))
        .or_else(|| payload.get("decision"))
        .or_else(|| payload.get("action"))
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim()
        .to_lowercase();
    if !matches!(
        route.as_str(),
        "direct" | "skill" | "rag" | "clarify" | "chat" | "answer" | "none" | "no_rag"
    ) {
        return None;
    }
    let confidence = payload
        .get("confidence")
        .and_then(|value| value.as_f64().or_else(|| value.as_i64().map(|item| item as f64)))
        .map(|value| value.clamp(0.0, 1.0));
    let reason = payload
        .get("reason")
        .and_then(|value| value.as_str())
        .unwrap_or(&route)
        .trim()
        .to_string();
    if matches!(route.as_str(), "direct" | "chat" | "answer" | "none" | "no_rag") {
        let mut intent = QueryIntent::direct(&format!("llm:{reason}"), "llm");
        intent.confidence = confidence;
        return Some(intent);
    }
    if route == "clarify" {
        return Some(QueryIntent {
            route: "clarify".into(),
            needs_retrieval: false,
            reason: format!("llm:{reason}"),
            source: "llm".into(),
            confidence,
            rewritten_query: String::new(),
            skill_name: None,
            skill_arguments: json!({}),
            clarify_question: payload
                .get("clarify_question")
                .or_else(|| payload.get("question"))
                .or_else(|| payload.get("clarification"))
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .trim()
                .to_string(),
        });
    }
    if route == "skill" {
        let skill_name = payload
            .get("skill_name")
            .or_else(|| payload.get("tool_name"))
            .or_else(|| payload.get("name"))
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if skill_name.is_empty() {
            return None;
        }
        return Some(QueryIntent {
            route: "skill".into(),
            needs_retrieval: false,
            reason: format!("llm:{reason}"),
            source: "llm".into(),
            confidence,
            rewritten_query: String::new(),
            skill_name: Some(skill_name),
            skill_arguments: payload
                .get("skill_arguments")
                .or_else(|| payload.get("arguments"))
                .or_else(|| payload.get("tool_arguments"))
                .cloned()
                .unwrap_or_else(|| json!({})),
            clarify_question: String::new(),
        });
    }
    let rewritten = payload
        .get("query")
        .or_else(|| payload.get("rewritten_query"))
        .or_else(|| payload.get("search_query"))
        .or_else(|| payload.get("retrieval_query"))
        .and_then(|value| value.as_str())
        .unwrap_or(query)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if rewritten.is_empty() {
        return None;
    }
    let mut intent = QueryIntent::rag(&format!("llm:{reason}"), "llm", &rewritten);
    intent.confidence = confidence;
    Some(intent)
}

fn parse_json_object(text: &str) -> Value {
    let trimmed = text.trim();
    let trimmed = trimmed.strip_prefix("```json").unwrap_or(trimmed);
    let trimmed = trimmed.strip_prefix("```").unwrap_or(trimmed);
    let trimmed = trimmed.strip_suffix("```").unwrap_or(trimmed).trim();
    let start = trimmed.find('{').unwrap_or(0);
    let end = trimmed.rfind('}').map(|index| index + 1).unwrap_or(trimmed.len());
    serde_json::from_str(&trimmed[start..end]).unwrap_or_else(|_| json!({}))
}

fn render_history(history: &[(String, String)], turns: usize) -> String {
    history
        .iter()
        .rev()
        .filter(|(role, _)| role == "user" || role == "assistant")
        .take(turns)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|(role, content)| format!("{role}: {content}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_direct_knowledge_identifier_and_skill() {
        let greeting = rule_route_query_action("你好", true);
        assert_eq!(greeting.route, "direct");
        assert_eq!(greeting.reason, "direct_keyword");

        let knowledge = rule_route_query_action("差旅报销流程是什么", true);
        assert_eq!(knowledge.route, "rag");
        assert_eq!(knowledge.reason, "knowledge_keyword");

        let identifier = rule_route_query_action("员工号 E0011 的姓名和部门", true);
        assert_eq!(identifier.route, "rag");
        assert_eq!(identifier.reason, "identifier_query");

        let tools = rule_route_query_action("列出你的工具", true);
        assert_eq!(tools.route, "skill");
        assert_eq!(tools.skill_name.as_deref(), Some("list_available_tools"));
    }

    #[test]
    fn conflict_falls_back_to_rag_without_llm() {
        assert!(has_signal_conflict("翻译这段报销制度"));
        let intent = rule_route_query_action("翻译这段报销制度", true);
        assert_eq!(intent.route, "rag");
        assert_eq!(intent.reason, "signal_conflict");
        let fallback = fallback_route_query_intent("翻译这段报销制度", true);
        assert_eq!(fallback.route, "rag");
        assert_eq!(fallback.reason, "fallback:signal_conflict");
    }

    #[test]
    fn default_chat_skips_retrieval() {
        let intent = rule_route_query_action("写一首诗", true);
        assert_eq!(intent.route, "direct");
        assert_eq!(intent.reason, "direct_default");
        let disabled = rule_route_query_action("差旅报销流程是什么", false);
        assert_eq!(disabled.route, "direct");
        assert_eq!(disabled.reason, "retrieval_disabled");
    }

    #[tokio::test]
    async fn prefer_rule_skill_false_still_falls_back_without_llm() {
        let preferred = route_query_action(
            "列出你的工具",
            true,
            None,
            &[],
            &[],
        )
        .await;
        assert_eq!(preferred.route, "skill");
        assert_eq!(preferred.skill_name.as_deref(), Some("list_available_tools"));

        let retry = route_query_action_with(
            "列出你的工具",
            true,
            None,
            &[],
            &[],
            false,
        )
        .await;
        assert_eq!(retry.route, "skill");
        assert_eq!(retry.skill_name.as_deref(), Some("list_available_tools"));
    }

    #[test]
    fn parses_llm_action_payload_and_tool_call() {
        let payload = json!({
            "route": "clarify",
            "reason": "缺编号",
            "clarify_question": "请提供员工号"
        });
        let intent = intent_from_action_payload(&payload, "查一下").unwrap();
        assert_eq!(intent.route, "clarify");
        assert_eq!(intent.clarify_question, "请提供员工号");

        let tool = intent_from_tool_call(
            "search_knowledge_base",
            &json!({"query": "差旅报销"}),
            "这个怎么报",
        )
        .unwrap();
        assert_eq!(tool.route, "rag");
        assert_eq!(tool.rewritten_query, "差旅报销");
    }

    #[test]
    fn action_and_intent_logs_match_python() {
        let skill = intent_from_routed_skill(&skills::route_local_skill("列出你的工具").unwrap());
        assert_eq!(
            format_action_route_result(&skill, false),
            "动作路由结果: source=rule route=skill reason=rule_skill:list_available_tools skill=list_available_tools"
        );
        let mut rag = QueryIntent::rag("knowledge_keyword", "rule", "差旅报销");
        assert_eq!(
            format_intent_route_result(&rag, false),
            "意图路由结果: source=rule route=rag reason=knowledge_keyword"
        );
        rag.confidence = Some(0.0);
        assert_eq!(
            format_action_route_result(&rag, true),
            "动作路由结果: source=rule route=rag confidence=0.0 reason=knowledge_keyword skill=None"
        );
        let fallback = fallback_route_query_intent("翻译这段报销制度", true);
        assert_eq!(
            format_intent_route_result(&fallback, true),
            "意图路由结果: source=fallback route=rag confidence=0.0 reason=fallback:signal_conflict"
        );
    }

    #[tokio::test]
    async fn without_llm_conflict_uses_python_fallback_intent() {
        let intent = route_query_action("翻译这段报销制度", true, None, &[], &[]).await;
        assert_eq!(intent.route, "rag");
        assert_eq!(intent.source, "fallback");
        assert_eq!(intent.reason, "fallback:signal_conflict");
        assert_eq!(intent.confidence, Some(0.0));

        let poem = route_query_action("写一首诗", true, None, &[], &[]).await;
        assert_eq!(poem.route, "direct");
        assert_eq!(poem.source, "fallback");
        assert_eq!(poem.reason, "fallback:ambiguous_llm_default");
    }
}
