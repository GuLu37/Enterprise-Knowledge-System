//! 常规检索规划器（对齐 Python `app.rag.planner.RetrievalPlanner`）。
//!
//! 第一轮怎么查、证据够不够、不足时怎么补。真正的文档检索仍由
//! `search_knowledge_base` / 检索引擎执行。无强信号或证据落在模糊区间时，
//! 可交给 LLM 消歧，失败回退规则。

use serde_json::{Value, json};

use crate::llm::{ChatMessage, LlmClient};
use crate::retrieval::RetrievalItem;

pub const RETRIEVAL_METHOD_HYBRID: &str = "hybrid";
pub const RETRIEVAL_METHOD_DENSE: &str = "dense";
pub const RETRIEVAL_METHOD_SPARSE: &str = "sparse";

const MAX_PLAN_ATTEMPTS: u32 = 2;
const DEFAULT_TOP_K: usize = 5;
const MAX_TOP_K: usize = 50;
const MIN_EVIDENCE_SCORE: f64 = 0.28;
const MIN_KEYWORD_COVERAGE: f64 = 0.22;
const HIGH_EVIDENCE_SCORE: f64 = 0.45;
const LOW_EVIDENCE_SCORE: f64 = 0.15;
const HIGH_KEYWORD_COVERAGE: f64 = 0.50;
const LOW_KEYWORD_COVERAGE: f64 = 0.10;

const SELECT_SYSTEM_PROMPT: &str = "你是企业知识库检索策略决策器。规则未给出强信号时，由你做最终决策。\
可选检索方式：hybrid（混合，默认稳妥）、dense（纯语义召回）、sparse（纯关键词/精确匹配）。\
只输出严格 JSON，格式：{\"retrieval_method\":\"hybrid|dense|sparse\",\"top_k\":整数,\"strategy\":\"简短策略名\",\"reason\":\"简短中文理由\"}";

const ASSESS_SYSTEM_PROMPT: &str = "你是检索证据评估器。根据用户问题与给定检索信号，判断现有检索结果是否足以回答问题。\
只依据提供的信号做判断，不要臆测知识库内容。\
只输出严格 JSON，格式：{\"sufficient\":true|false,\"reason\":\"简短中文理由\"}";

const REWRITE_SYSTEM_PROMPT: &str = "你是企业知识库检索查询改写器。证据不足时，改写查询用于二次检索。\
必须原样保留编号、工号、产品号、人名、文档名等专名，不得丢失问题焦点。\
只输出严格 JSON，格式：{\"query\":\"改写后的检索查询\"}";

const STRUCTURED_QUERY_TERMS: &[&str] = &[
    "列出", "各", "分别", "有哪些", "名单", "列表", "清单", "明细", "统计", "数量", "多少",
    "一共", "总数", "比较", "对比", "同时", "并且",
];
const PROCEDURAL_QUERY_TERMS: &[&str] = &[
    "如何", "怎么", "流程", "步骤", "要求", "条件", "标准", "规定", "政策", "制度",
];

#[derive(Debug, Clone)]
pub struct RetrievalPlan {
    pub query: String,
    pub retrieval_method: String,
    pub top_k: usize,
    pub strategy: String,
    pub attempt: u32,
    pub reason: String,
}

impl RetrievalPlan {
    pub fn signal_text(&self) -> String {
        format!(
            "查询特征策略={}（{}），规则建议检索方式={}",
            self.strategy, self.reason, self.retrieval_method
        )
    }
}

#[derive(Debug, Clone)]
pub struct EvidenceAssessment {
    pub sufficient: bool,
    pub result_count: usize,
    pub best_score: f64,
    pub keyword_coverage: f64,
    pub identifier_hit: bool,
    pub reason: String,
    pub missing_identifiers: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct QuerySignals {
    pub is_identifier: bool,
    pub structured_hit: bool,
    pub procedural_hit: bool,
}

pub fn normalize_method(method: &str) -> String {
    let normalized = method.trim().to_lowercase();
    if matches!(normalized.as_str(), RETRIEVAL_METHOD_HYBRID | RETRIEVAL_METHOD_DENSE | RETRIEVAL_METHOD_SPARSE)
    {
        normalized
    } else {
        RETRIEVAL_METHOD_HYBRID.to_string()
    }
}

pub fn resolve_method_for_query(query: &str, method: &str) -> String {
    let normalized = normalize_method(method);
    if normalized == RETRIEVAL_METHOD_DENSE && eks_kernels::rerank::is_identifier_query(query) {
        RETRIEVAL_METHOD_HYBRID.to_string()
    } else {
        normalized
    }
}

pub fn normalize_top_k(top_k: usize) -> usize {
    top_k.clamp(1, MAX_TOP_K)
}

fn collapse_spaces(query: &str) -> String {
    query.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn contains_any(query: &str, terms: &[&str]) -> bool {
    let normalized = query.trim().to_lowercase();
    terms.iter().any(|term| normalized.contains(term))
}

pub fn extract_query_signals(query: &str) -> QuerySignals {
    let normalized = collapse_spaces(query);
    QuerySignals {
        is_identifier: eks_kernels::rerank::is_identifier_query(&normalized),
        structured_hit: contains_any(&normalized, STRUCTURED_QUERY_TERMS),
        procedural_hit: contains_any(&normalized, PROCEDURAL_QUERY_TERMS),
    }
}

pub fn select_initial_rule(query: &str, top_k: usize, retrieval_method: &str) -> RetrievalPlan {
    let normalized_query = collapse_spaces(query);
    let method = normalize_method(retrieval_method);
    let top_k = normalize_top_k(if top_k == 0 { DEFAULT_TOP_K } else { top_k });
    rule_select(&normalized_query, method, top_k, extract_query_signals(&normalized_query))
}

fn rule_select(query: &str, mut method: String, mut top_k: usize, signals: QuerySignals) -> RetrievalPlan {
    let (strategy, reason) = if signals.is_identifier {
        if method == RETRIEVAL_METHOD_HYBRID {
            method = RETRIEVAL_METHOD_SPARSE.to_string();
        }
        (
            "identifier_exact",
            "检测到编号、工号、产品号或编码，优先使用稀疏检索保留精确匹配",
        )
    } else if signals.structured_hit {
        top_k = top_k.max(8);
        (
            "structured_multi_query",
            "检测到多条件或结构化查询，保留混合检索并扩大候选范围",
        )
    } else if signals.procedural_hit {
        if method == RETRIEVAL_METHOD_DENSE {
            method = RETRIEVAL_METHOD_HYBRID.to_string();
        }
        (
            "procedural_hybrid",
            "检测到流程、条件或制度类问题，优先使用混合检索",
        )
    } else {
        method = normalize_method(&method);
        (
            "default_hybrid",
            "通用知识库问题使用混合检索作为稳定起点",
        )
    };
    RetrievalPlan {
        query: query.to_string(),
        retrieval_method: method,
        top_k,
        strategy: strategy.into(),
        attempt: 1,
        reason: reason.into(),
    }
}

pub async fn select_initial(
    query: &str,
    top_k: usize,
    retrieval_method: &str,
    llm: Option<&LlmClient>,
) -> RetrievalPlan {
    let normalized_query = collapse_spaces(query);
    let method = normalize_method(retrieval_method);
    let top_k = normalize_top_k(if top_k == 0 { DEFAULT_TOP_K } else { top_k });
    let signals = extract_query_signals(&normalized_query);
    if signals.is_identifier || signals.structured_hit || signals.procedural_hit {
        return rule_select(&normalized_query, method, top_k, signals);
    }
    if let Some(llm) = llm {
        match llm_select(&normalized_query, &signals, top_k, llm).await {
            Ok(plan) => return plan,
            Err(err) => tracing::warn!("常规检索策略选择 LLM 决策失败，回退规则: {err}"),
        }
    }
    rule_select(&normalized_query, method, top_k, signals)
}

async fn llm_select(
    query: &str,
    signals: &QuerySignals,
    top_k: usize,
    llm: &LlmClient,
) -> Result<RetrievalPlan, String> {
    let human = format!(
        "用户问题：{query}\n规则信号：编号查询={}，结构化查询={}，流程类查询={}\n默认 top_k={top_k}\n请输出最终检索策略 JSON。",
        signals.is_identifier, signals.structured_hit, signals.procedural_hit
    );
    let payload = invoke_llm_decision(llm, SELECT_SYSTEM_PROMPT, &human).await?;
    let method = normalize_method(payload.get("retrieval_method").and_then(Value::as_str).unwrap_or(RETRIEVAL_METHOD_HYBRID));
    let resolved_top_k = payload
        .get("top_k")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(top_k);
    let strategy = payload
        .get("strategy")
        .and_then(Value::as_str)
        .unwrap_or("llm_hybrid")
        .trim();
    let reason = payload
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or("LLM 决策选择检索策略")
        .trim();
    Ok(RetrievalPlan {
        query: query.to_string(),
        retrieval_method: method,
        top_k: normalize_top_k(resolved_top_k),
        strategy: if strategy.is_empty() { "llm_hybrid".into() } else { strategy.into() },
        attempt: 1,
        reason: if reason.is_empty() {
            "LLM 决策选择检索策略".into()
        } else {
            reason.into()
        },
    })
}

fn result_text(item: &RetrievalItem) -> String {
    format!("{}\n{}\n{}", item.source_name, item.document_id, item.content)
}

fn keyword_coverage(item: &RetrievalItem) -> f64 {
    let Some(signals) = &item.signals else {
        return 0.0;
    };
    let coverage = if signals.keyword_coverage > 0.0 {
        signals.keyword_coverage
    } else {
        signals.focus_coverage
    };
    coverage.clamp(0.0, 1.0)
}

fn exact_match(item: &RetrievalItem) -> bool {
    item.signals.as_ref().map(|signals| signals.exact_match).unwrap_or(false)
}

pub fn assess_evidence_rule(query: &str, results: &[RetrievalItem]) -> EvidenceAssessment {
    if results.is_empty() {
        return EvidenceAssessment {
            sufficient: false,
            result_count: 0,
            best_score: 0.0,
            keyword_coverage: 0.0,
            identifier_hit: false,
            reason: "no_results".into(),
            missing_identifiers: Vec::new(),
        };
    }
    let best_score = results.iter().map(|item| item.score).fold(0.0_f64, f64::max);
    let coverage = results.iter().map(keyword_coverage).fold(0.0_f64, f64::max);
    let identifiers = eks_kernels::rerank::extract_identifier_terms(query);
    let missing: Vec<String> = identifiers
        .iter()
        .filter(|identifier| {
            let needle = eks_kernels::normalize_text(identifier);
            !results.iter().any(|item| eks_kernels::normalize_text(&result_text(item)).contains(&needle))
        })
        .cloned()
        .collect();
    let identifier_hit = !identifiers.is_empty() && missing.is_empty();
    rule_assess(results, &identifiers, best_score, coverage, identifier_hit, missing)
}

fn evidence_high_confidence(identifiers: &[String], best_score: f64, coverage: f64) -> bool {
    if !identifiers.is_empty() {
        return true;
    }
    if best_score >= HIGH_EVIDENCE_SCORE || coverage >= HIGH_KEYWORD_COVERAGE {
        return true;
    }
    best_score <= LOW_EVIDENCE_SCORE && coverage <= LOW_KEYWORD_COVERAGE
}

fn rule_assess(
    results: &[RetrievalItem],
    identifiers: &[String],
    best_score: f64,
    coverage: f64,
    identifier_hit: bool,
    missing: Vec<String>,
) -> EvidenceAssessment {
    let (sufficient, reason) = if !identifiers.is_empty() {
        (
            identifier_hit,
            if identifier_hit { "identifier_hit" } else { "identifier_incomplete" },
        )
    } else {
        let sufficient = best_score >= MIN_EVIDENCE_SCORE
            || coverage >= MIN_KEYWORD_COVERAGE
            || results.iter().any(exact_match);
        let reason = if sufficient {
            "evidence_sufficient"
        } else if results.len() < 2 {
            "weak_single_result"
        } else {
            "low_score_or_coverage"
        };
        (sufficient, reason)
    };
    EvidenceAssessment {
        sufficient,
        result_count: results.len(),
        best_score: round6(best_score.max(0.0)),
        keyword_coverage: round6(coverage.max(0.0)),
        identifier_hit,
        reason: reason.into(),
        missing_identifiers: missing,
    }
}

pub async fn assess_evidence(
    query: &str,
    results: &[RetrievalItem],
    llm: Option<&LlmClient>,
) -> EvidenceAssessment {
    if results.is_empty() {
        return assess_evidence_rule(query, results);
    }
    let best_score = results.iter().map(|item| item.score).fold(0.0_f64, f64::max);
    let coverage = results.iter().map(keyword_coverage).fold(0.0_f64, f64::max);
    let identifiers = eks_kernels::rerank::extract_identifier_terms(query);
    let missing: Vec<String> = identifiers
        .iter()
        .filter(|identifier| {
            let needle = eks_kernels::normalize_text(identifier);
            !results.iter().any(|item| eks_kernels::normalize_text(&result_text(item)).contains(&needle))
        })
        .cloned()
        .collect();
    let identifier_hit = !identifiers.is_empty() && missing.is_empty();
    if evidence_high_confidence(&identifiers, best_score, coverage) {
        return rule_assess(results, &identifiers, best_score, coverage, identifier_hit, missing);
    }
    if let Some(llm) = llm {
        match llm_assess(query, results, best_score, coverage, identifier_hit, &missing, llm).await {
            Ok(Some(assessment)) => return assessment,
            Ok(None) => {}
            Err(err) => tracing::warn!("常规检索证据评估 LLM 决策失败，回退规则: {err}"),
        }
    }
    rule_assess(results, &identifiers, best_score, coverage, identifier_hit, missing)
}

async fn llm_assess(
    query: &str,
    results: &[RetrievalItem],
    best_score: f64,
    coverage: f64,
    identifier_hit: bool,
    missing: &[String],
    llm: &LlmClient,
) -> Result<Option<EvidenceAssessment>, String> {
    let snippet = results
        .iter()
        .take(3)
        .map(|item| item.content.chars().take(60).collect::<String>())
        .collect::<Vec<_>>()
        .join(" | ");
    let human = format!(
        "用户问题：{query}\n检索信号：结果数={}，最高分={best_score:.3}，关键词覆盖={coverage:.3}\n命中片段示例：{snippet}\n请判断现有检索结果是否足以回答用户问题。",
        results.len()
    );
    let payload = invoke_llm_decision(llm, ASSESS_SYSTEM_PROMPT, &human).await?;
    let sufficient = match parse_sufficient(payload.get("sufficient")) {
        Some(value) => value,
        None => return Ok(None),
    };
    let reason = payload
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or("llm_assessed")
        .trim();
    Ok(Some(EvidenceAssessment {
        sufficient,
        result_count: results.len(),
        best_score: round6(best_score.max(0.0)),
        keyword_coverage: round6(coverage.max(0.0)),
        identifier_hit,
        reason: if reason.is_empty() { "llm_assessed".into() } else { reason.into() },
        missing_identifiers: missing.to_vec(),
    }))
}

pub fn plan_follow_up_rule(
    original_query: &str,
    decision: &RetrievalPlan,
    assessment: &EvidenceAssessment,
) -> Option<RetrievalPlan> {
    if assessment.sufficient || decision.attempt >= MAX_PLAN_ATTEMPTS {
        return None;
    }
    let (next_method, strategy, reason) = if eks_kernels::rerank::is_identifier_query(original_query)
        || !assessment.missing_identifiers.is_empty()
    {
        (
            RETRIEVAL_METHOD_SPARSE,
            "identifier_exact_retry",
            "上一轮未覆盖全部精确标识，改用稀疏检索并强化编号匹配",
        )
    } else if decision.retrieval_method == RETRIEVAL_METHOD_SPARSE {
        (
            RETRIEVAL_METHOD_HYBRID,
            "hybrid_recall_retry",
            "稀疏检索证据不足，切换混合召回补充语义结果",
        )
    } else if decision.retrieval_method == RETRIEVAL_METHOD_DENSE {
        (
            RETRIEVAL_METHOD_SPARSE,
            "sparse_exact_retry",
            "密集检索证据不足，切换稀疏检索补充关键词命中",
        )
    } else {
        (
            RETRIEVAL_METHOD_SPARSE,
            "sparse_focus_retry",
            "混合检索证据不足，切换稀疏检索强化核心词和文件名命中",
        )
    };
    let next_query = build_follow_up_query(original_query, &decision.query, assessment);
    Some(RetrievalPlan {
        query: next_query,
        retrieval_method: next_method.to_string(),
        top_k: if assessment.result_count == 0 {
            decision.top_k.max(8)
        } else {
            decision.top_k
        },
        strategy: strategy.into(),
        attempt: decision.attempt + 1,
        reason: reason.into(),
    })
}

pub async fn plan_follow_up(
    original_query: &str,
    decision: &RetrievalPlan,
    assessment: &EvidenceAssessment,
    llm: Option<&LlmClient>,
) -> Option<RetrievalPlan> {
    let mut plan = plan_follow_up_rule(original_query, decision, assessment)?;
    if let Some(llm) = llm {
        match llm_rewrite_query(original_query, &decision.query, assessment, llm).await {
            Ok(Some(rewritten)) => plan.query = rewritten,
            Ok(None) => {}
            Err(err) => tracing::warn!("常规检索补检索改写 LLM 失败，回退模板: {err}"),
        }
    }
    Some(plan)
}

async fn llm_rewrite_query(
    original_query: &str,
    first_query: &str,
    assessment: &EvidenceAssessment,
    llm: &LlmClient,
) -> Result<Option<String>, String> {
    let missing = if assessment.missing_identifiers.is_empty() {
        "无".to_string()
    } else {
        assessment.missing_identifiers.join("、")
    };
    let human = format!(
        "用户原问题：{original_query}\n上次检索查询：{first_query}\n缺失编号：{missing}\n请输出改写后的二次检索查询。"
    );
    let payload = invoke_llm_decision(llm, REWRITE_SYSTEM_PROMPT, &human).await?;
    let rewritten = collapse_spaces(payload.get("query").and_then(Value::as_str).unwrap_or(""));
    Ok(if rewritten.is_empty() { None } else { Some(rewritten) })
}

fn query_signature(query: &str) -> String {
    eks_kernels::normalize_text(&collapse_spaces(query))
}

fn build_follow_up_query(original_query: &str, first_query: &str, assessment: &EvidenceAssessment) -> String {
    let original = collapse_spaces(original_query);
    let first = collapse_spaces(first_query);
    if query_signature(&first) != query_signature(&original) && !original.is_empty() {
        return original;
    }
    if !assessment.missing_identifiers.is_empty() {
        return format!("{} 精确匹配 {}", original.or_fallback(&first), assessment.missing_identifiers.join(" "));
    }
    let focus_terms = eks_kernels::rerank::build_query_focus_terms(original.or_fallback(&first), 3);
    if !focus_terms.is_empty() {
        return format!("{} 具体规定 {}", original.or_fallback(&first), focus_terms.join(" "));
    }
    format!("{} 具体内容", original.or_fallback(&first))
}

trait OrFallback {
    fn or_fallback<'a>(&'a self, fallback: &'a str) -> &'a str;
}

impl OrFallback for String {
    fn or_fallback<'a>(&'a self, fallback: &'a str) -> &'a str {
        if self.is_empty() { fallback } else { self }
    }
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn parse_sufficient(value: Option<&Value>) -> Option<bool> {
    match value? {
        Value::Bool(flag) => Some(*flag),
        Value::Number(number) => Some(number.as_f64().unwrap_or(0.0) != 0.0),
        Value::String(text) => {
            let normalized = text.trim().to_lowercase();
            Some(matches!(
                normalized.as_str(),
                "true" | "1" | "yes" | "on" | "是" | "充分" | "足够" | "能" | "可以"
            ))
        }
        _ => None,
    }
}

pub fn parse_json_payload(text: &str) -> Value {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return json!({});
    }
    let trimmed = trimmed.strip_prefix("```json").unwrap_or(trimmed);
    let trimmed = trimmed.strip_prefix("```").unwrap_or(trimmed);
    let trimmed = trimmed.strip_suffix("```").unwrap_or(trimmed).trim();
    let start = trimmed.find('{').unwrap_or(0);
    let end = trimmed.rfind('}').map(|index| index + 1).unwrap_or(trimmed.len());
    serde_json::from_str(&trimmed[start..end]).unwrap_or_else(|_| json!({}))
}

async fn invoke_llm_decision(llm: &LlmClient, system: &str, human: &str) -> Result<Value, String> {
    let completion = llm
        .invoke(&[ChatMessage::new("system", system), ChatMessage::new("user", human)])
        .await?;
    Ok(parse_json_payload(&completion.content))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(content: &str, score: f64, coverage: f64, exact: bool) -> RetrievalItem {
        RetrievalItem {
            content: content.into(),
            score,
            source: "hybrid".into(),
            document_id: "d1".into(),
            source_name: "制度.pdf".into(),
            chunk_index: 0,
            child_id: "c1".into(),
            parent_id: "p1".into(),
            parent_index: 0,
            file_type: "pdf".into(),
            content_type: "text".into(),
            signals: Some(crate::retrieval::ItemSignals {
                keyword_coverage: coverage,
                focus_coverage: coverage,
                heading_coverage: 0.0,
                heading_focus_coverage: 0.0,
                exact_match: exact,
                is_reference_index: false,
                keyword_specificity: 0.0,
                keyword_hits: Vec::new(),
                focus_hits: Vec::new(),
                long_keyword_hit_count: 0,
            }),
            parent_context_text: None,
            rerank_stage: None,
            cross_encoder_score: None,
            rerank_trace: crate::retrieval::RerankTrace::default(),
            structured_aggregation: false,
            section_name: None,
            chunk_indices: Vec::new(),
            matched_sources: Vec::new(),
            fusion_score: None,
        }
    }

    #[test]
    fn rule_select_identifier_structured_procedural_and_default() {
        let identifier = select_initial_rule("员工号 E0011 的姓名和部门", 5, "hybrid");
        assert_eq!(identifier.strategy, "identifier_exact");
        assert_eq!(identifier.retrieval_method, "sparse");
        let structured = select_initial_rule("列出各产品名单", 5, "hybrid");
        assert_eq!(structured.strategy, "structured_multi_query");
        assert_eq!(structured.top_k, 8);
        let procedural = select_initial_rule("差旅报销流程怎么走", 5, "dense");
        assert_eq!(procedural.strategy, "procedural_hybrid");
        assert_eq!(procedural.retrieval_method, "hybrid");
        let default = select_initial_rule("随便聊聊天", 5, "hybrid");
        assert_eq!(default.strategy, "default_hybrid");
        assert!(default.signal_text().contains("hybrid"));
    }

    #[test]
    fn dense_identifier_query_is_forced_to_hybrid() {
        assert_eq!(
            resolve_method_for_query("员工号 E0011", "dense"),
            "hybrid"
        );
    }

    #[test]
    fn evidence_rules_cover_empty_identifier_and_score() {
        let empty = assess_evidence_rule("报销", &[]);
        assert!(!empty.sufficient);
        assert_eq!(empty.reason, "no_results");

        let strong = assess_evidence_rule("报销标准", &[item("差旅报销标准", 0.51, 0.6, false)]);
        assert!(strong.sufficient);
        assert_eq!(strong.reason, "evidence_sufficient");

        let exact = assess_evidence_rule("报销", &[item("内容", 0.05, 0.0, true)]);
        assert!(exact.sufficient);

        let identifier = assess_evidence_rule(
            "员工号 E0011",
            &[item("员工号 E0011 张三 研发部", 0.2, 0.1, false)],
        );
        assert!(identifier.sufficient);
        assert_eq!(identifier.reason, "identifier_hit");

        let missing = assess_evidence_rule("员工号 E0011", &[item("无关内容", 0.9, 0.9, false)]);
        assert!(!missing.sufficient);
        assert_eq!(missing.reason, "identifier_incomplete");
        assert!(!missing.missing_identifiers.is_empty());
    }

    #[test]
    fn follow_up_switches_method_and_keeps_focus() {
        let first = select_initial_rule("差旅报销标准", 5, "hybrid");
        let weak = assess_evidence_rule("差旅报销标准", &[item("不相关", 0.05, 0.05, false)]);
        let follow = plan_follow_up_rule("差旅报销标准", &first, &weak).unwrap();
        assert_eq!(follow.retrieval_method, "sparse");
        assert_eq!(follow.strategy, "sparse_focus_retry");
        assert_eq!(follow.attempt, 2);
        assert!(plan_follow_up_rule("差旅报销标准", &follow, &weak).is_none());
    }

    #[test]
    fn parses_llm_json_and_sufficient_aliases() {
        let payload = parse_json_payload("```json\n{\"sufficient\": \"是\", \"reason\": \"够了\"}\n```");
        assert_eq!(parse_sufficient(payload.get("sufficient")), Some(true));
        assert_eq!(payload["reason"], "够了");
        assert_eq!(parse_sufficient(Some(&json!("false"))), Some(false));
        assert!(parse_sufficient(None).is_none());
    }
}
