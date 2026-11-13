//! 检索编排：retrieval_service.py 的 Rust 移植（B4a 子集）。
//!
//! 覆盖: 多查询启发式扩展、启发式不足时 LLM 补写、dense(milvus_lite 直读 COSINE)+sparse(内核 BM25 两段式)、
//! 变体间 RRF 融合与原始查询保护、规则重排(含 adaptive 精排跳过语义)、
//! 关键词门控过滤、编号保护、结构化 schema 打分、父块上下文批量回填、
//! 上下文组装与序列化。
//! 暂缺(B4b): _expand_structured_section_evidence 同 Sheet 聚合展开——由 bench 门禁判定是否必要。

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use eks_kernels::bm25::BM25Okapi;
use eks_kernels::fuse::{FuseItem, fuse_ranked_results};
use eks_kernels::lexical::{build_filter_terms, lexical_match_score};
use eks_kernels::rerank::{
    Candidate, build_query_keywords, build_relevance_signals, extract_identifier_terms,
    is_identifier_query, is_reference_index_result, normalize_source_score,
    passes_keyword_relevance_gate, score_result,
};
use eks_kernels::text::{normalize_text, tokenize_cjk};
use serde_json::{Value, json};

use crate::db;
use crate::llm::{ChatMessage, LlmClient};
use crate::milvus_lite::{FieldValue, MilvusLiteStore, Row};

const MIN_MULTI_QUERY_COUNT: usize = 3;
const MAX_MULTI_QUERY_COUNT: usize = 3;
const MULTI_QUERY_SYSTEM_PROMPT: &str = "\
你是企业知识库检索查询扩展器。\
把用户问题改写为 3 条适合检索的查询，不回答问题本身。\
必须覆盖：原始问法、关键词短语、可能的文件名或标题表达、同义表达。\
如果问题包含编号、工号、产品号、人名、文档名或带书名号的名称，必须原样保留。\
只输出严格 JSON，格式：{\"queries\":[\"查询1\",\"查询2\",\"查询3\"]}";
pub const MAX_CITATION_RESULTS: usize = 3;
pub const STRUCTURED_QUERY_CITATION_RESULTS: usize = 12;
pub const RAG_CONTEXT_MAX_CHARS: usize = 6500;
pub const STRUCTURED_CONTEXT_MAX_CHARS: usize = 30000;
pub const MIN_FINAL_RELEVANCE_SCORE: f64 = 0.28;
const MIN_RAG_CANDIDATE_POOL: usize = 60;

const STRUCTURED_QUERY_HINT_TERMS: [&str; 19] = [
    "员工", "部门", "岗位", "职位", "工号", "姓名", "婚姻", "状态", "入职", "离职", "手机号",
    "电话", "邮箱", "地址", "身份证", "证件", "账号", "编码", "编号",
];

const STRUCTURED_META_SCHEMA_TERMS: [&str; 12] = [
    "说明", "填表说明", "字段说明", "版本", "变更", "速查", "索引", "目录", "FAQ", "常见问题",
    "检索提示", "回答参考",
];

const STRUCTURED_META_HEADER_TERMS: [&str; 8] = [
    "字段", "用途", "填写规则", "问题", "答案", "关键词", "优先命中表", "检索提示",
];

const STRUCTURED_DATA_SHEET_TERMS: [&str; 8] = [
    "明细", "名单", "名册", "花名册", "通讯录", "清单", "台账", "总表",
];

// ---------------- 结果结构 ----------------

#[derive(Debug, Clone)]
pub struct RetrievalItem {
    pub content: String,
    pub score: f64,
    pub source: String,
    pub document_id: String,
    pub source_name: String,
    pub chunk_index: i64,
    pub child_id: String,
    pub parent_id: String,
    pub parent_index: i64,
    pub file_type: String,
    pub content_type: String,
    /// 重排信号（规则重排后填充）。
    pub signals: Option<ItemSignals>,
    pub parent_context_text: Option<String>,
    pub rerank_stage: Option<String>,
    /// Cross-Encoder 精排分数（触发精排时填充）。
    pub cross_encoder_score: Option<f64>,
    /// 对齐 Python cascade metadata：跳过/精排可观测字段。
    pub rerank_trace: RerankTrace,
    /// 结构化同 Sheet 聚合标记（_expand_structured_section_evidence 产物）。
    pub structured_aggregation: bool,
    pub section_name: Option<String>,
    pub chunk_indices: Vec<i64>,
    /// 融合来源标记与融合分（fuse_ranked_results 产物，进 metadata）。
    pub matched_sources: Vec<String>,
    pub fusion_score: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct RerankTrace {
    pub cross_encoder_used: Option<bool>,
    pub rule_confidence: Option<f64>,
    pub rule_score: Option<f64>,
    pub rule_score_normalized: Option<f64>,
    pub final_score: Option<f64>,
    pub cross_encoder_candidate_rank: Option<i64>,
    pub cross_encoder_error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ItemSignals {
    pub keyword_coverage: f64,
    pub focus_coverage: f64,
    pub heading_coverage: f64,
    pub heading_focus_coverage: f64,
    pub exact_match: bool,
    pub is_reference_index: bool,
    pub keyword_specificity: f64,
    /// 对齐 Python signals：命中词列表与计数（序列化进 metadata）。
    pub keyword_hits: Vec<String>,
    pub focus_hits: Vec<String>,
    pub long_keyword_hit_count: usize,
}

impl RetrievalItem {
    pub fn from_row(row: &Row, score: f64, source: &str) -> Self {
        let get_str = |name: &str| {
            row.fields
                .get(name)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let get_int = |name: &str| {
            row.fields
                .get(name)
                .and_then(|v| v.as_int())
                .unwrap_or(-1)
        };
        RetrievalItem {
            // 对齐 Python：内容取 child_text/chunk_text/text 逐级兜底。
            content: ["child_text", "chunk_text", "text"]
                .into_iter()
                .map(|key| {
                    row.fields
                        .get(key)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string()
                })
                .find(|value| !value.is_empty())
                .unwrap_or_default(),
            score,
            source: source.to_string(),
            document_id: get_str("document_id"),
            source_name: get_str("source_name"),
            chunk_index: get_int("chunk_index"),
            child_id: get_str("child_id"),
            parent_id: get_str("parent_id"),
            parent_index: get_int("parent_index"),
            file_type: get_str("file_type"),
            content_type: get_str("content_type"),
            signals: None,
            parent_context_text: None,
            rerank_stage: None,
            cross_encoder_score: None,
            rerank_trace: RerankTrace::default(),
            structured_aggregation: false,
            section_name: None,
            chunk_indices: Vec::new(),
            matched_sources: Vec::new(),
            fusion_score: None,
        }
    }

    /// 与 Python serialize_retrieval_results 相同的 JSON 形态（过滤 _ 前缀）。
    pub fn to_json(&self) -> Value {
        let mut metadata = json!({
            "child_id": self.child_id,
            "document_id": self.document_id,
            "parent_id": self.parent_id,
            "parent_index": self.parent_index,
            "chunk_index": self.chunk_index,
            "source_name": self.source_name,
            "file_type": self.file_type,
            "content_type": self.content_type,
        });
        let map = metadata.as_object_mut().unwrap();
        if let Some(fusion_score) = self.fusion_score {
            map.insert("matched_sources".into(), json!(self.matched_sources));
            map.insert("fusion_score".into(), json!(fusion_score));
        }
        if let Some(signals) = &self.signals {
            map.insert(
                "keyword_hits".into(),
                json!(signals.keyword_hits),
            );
            map.insert("focus_hits".into(), json!(signals.focus_hits));
            map.insert(
                "keyword_hit_count".into(),
                json!(signals.keyword_hits.len()),
            );
            map.insert("focus_hit_count".into(), json!(signals.focus_hits.len()));
            map.insert(
                "long_keyword_hit_count".into(),
                json!(signals.long_keyword_hit_count),
            );
            map.insert("keyword_coverage".into(), json!(signals.keyword_coverage));
            map.insert("focus_coverage".into(), json!(signals.focus_coverage));
            map.insert("heading_coverage".into(), json!(signals.heading_coverage));
            map.insert(
                "heading_focus_coverage".into(),
                json!(signals.heading_focus_coverage),
            );
            map.insert("exact_match".into(), json!(signals.exact_match));
            map.insert(
                "is_reference_index".into(),
                json!(signals.is_reference_index),
            );
            map.insert(
                "keyword_specificity".into(),
                json!(signals.keyword_specificity),
            );
        }
        if let Some(stage) = &self.rerank_stage {
            map.insert("rerank_stage".into(), json!(stage));
        }
        if self.parent_context_text.is_some() {
            map.insert("context_expansion".into(), json!(true));
        }
        if let Some(used) = self.rerank_trace.cross_encoder_used {
            map.insert("cross_encoder_used".into(), json!(used));
        } else if self.cross_encoder_score.is_some() {
            map.insert("cross_encoder_used".into(), json!(true));
        }
        match self.cross_encoder_score {
            Some(ce_score) => {
                map.insert("cross_encoder_score".into(), json!(ce_score));
            }
            None if self.rerank_stage.as_deref() == Some("rule_then_cross_encoder") => {
                map.insert("cross_encoder_score".into(), Value::Null);
            }
            None => {}
        }
        if let Some(value) = self.rerank_trace.rule_confidence {
            map.insert("rule_confidence".into(), json!(value));
        }
        if let Some(value) = self.rerank_trace.rule_score {
            map.insert("rule_score".into(), json!(value));
        }
        if let Some(value) = self.rerank_trace.rule_score_normalized {
            map.insert("rule_score_normalized".into(), json!(value));
        }
        if let Some(value) = self.rerank_trace.final_score {
            map.insert("final_score".into(), json!(value));
        }
        match self.rerank_trace.cross_encoder_candidate_rank {
            Some(rank) => {
                map.insert("cross_encoder_candidate_rank".into(), json!(rank));
            }
            None if self.rerank_stage.as_deref() == Some("rule_then_cross_encoder") => {
                map.insert("cross_encoder_candidate_rank".into(), Value::Null);
            }
            None => {}
        }
        if let Some(error) = &self.rerank_trace.cross_encoder_error {
            map.insert("cross_encoder_error".into(), json!(error));
        }
        if self.structured_aggregation {
            map.insert("structured_aggregation".into(), json!(true));
            if let Some(section) = &self.section_name {
                map.insert("section_name".into(), json!(section));
            }
            if !self.chunk_indices.is_empty() {
                map.insert("chunk_indices".into(), json!(self.chunk_indices));
            }
        }
        json!({
            "content": self.content,
            "metadata": metadata,
            "score": self.score,
            "source": self.source,
        })
    }
}

// ---------------- 多查询扩展 ----------------

pub fn collapse_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn compact_query_key(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_whitespace() && !"：:，,。？?、.-".contains(*c))
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// 丢掉几乎重复的改写，避免 hybrid 为同一句话嵌三次向量。
pub fn collapse_near_duplicate_queries(queries: Vec<String>) -> Vec<String> {
    let mut kept: Vec<String> = Vec::new();
    let mut keys: Vec<String> = Vec::new();
    for query in queries {
        let key = compact_query_key(&query);
        if key.is_empty() {
            continue;
        }
        let duplicate = keys.iter().any(|existing| {
            if existing == &key {
                return true;
            }
            let shorter = existing.len().min(key.len());
            let longer = existing.len().max(key.len());
            shorter * 10 >= longer * 8
                && (existing.contains(&key) || key.contains(existing))
        });
        if duplicate {
            continue;
        }
        keys.push(key);
        kept.push(query);
    }
    kept
}

fn normalize_query_variants(candidates: &[String], base_query: &str, max_queries: usize) -> Vec<String> {
    let normalized_base = collapse_spaces(base_query.trim());
    let mut expanded: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for candidate in candidates {
        let mut item = collapse_spaces(&candidate.trim());
        // 去掉行首编号前缀 "1. "、"一、" 等
        item = item
            .trim_start_matches(|c: char| c.is_whitespace() || c.is_ascii_digit() || c == '.' || c == '-' || c == '、' || c == ':' || c == '：')
            .trim()
            .to_string();
        if item.is_empty() {
            continue;
        }
        let key = item.to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        expanded.push(item);
        if expanded.len() >= max_queries.max(1) {
            break;
        }
    }
    if !normalized_base.is_empty()
        && !normalized_base.to_lowercase().is_empty()
        && !seen.contains(&normalized_base.to_lowercase())
    {
        expanded.insert(0, normalized_base);
    }
    expanded.into_iter().take(max_queries.max(1)).collect()
}

fn strip_prefix_phrase(cleaned: &str) -> String {
    // ^(请问|麻烦|帮我|帮忙|查询|检索|搜索|查一下|查找|根据|基于|结合)(一下|一下子|知识库|文档|资料|上传文档|企业知识库|公司资料)?[：:\s，,]*
    let prefixes = [
        "请问", "麻烦", "帮我", "帮忙", "查询", "检索", "搜索", "查一下", "查找", "根据", "基于", "结合",
    ];
    let suffixes = [
        "一下", "一下子", "知识库", "文档", "资料", "上传文档", "企业知识库", "公司资料",
    ];
    let mut current = cleaned.to_string();
    for _ in 0..2 {
        let mut next = current.clone();
        for prefix in prefixes {
            if let Some(rest) = next.strip_prefix(prefix) {
                let mut rest = rest.to_string();
                for suffix in suffixes {
                    if let Some(after) = rest.strip_prefix(suffix) {
                        rest = after.to_string();
                        break;
                    }
                }
                next = rest
                    .trim_start_matches(|c: char| "：:，, \t".contains(c))
                    .trim()
                    .to_string();
                break;
            }
        }
        current = next;
    }
    current
}

fn strip_middle_phrase(cleaned: &str) -> String {
    // (根据|基于|结合)?(知识库|企业知识库|文档|资料|上传文档|公司资料)(里面|中|里)?
    let optional_head = ["根据", "基于", "结合"];
    let body = [
        "知识库", "企业知识库", "文档", "资料", "上传文档", "公司资料",
    ];
    let tails = ["里面", "中", "里"];
    let mut result = cleaned.to_string();
    // 对齐 Python re.sub：删除全部匹配（循环删除直到无匹配），不只删第一处。
    for _ in 0..16 {
        let chars: Vec<char> = result.chars().collect();
        let mut replaced = false;
        'scan: for start in 0..chars.len() {
            let mut rest: String = chars[start..].iter().collect();
            for head in optional_head {
                if let Some(after) = rest.strip_prefix(head) {
                    rest = after.to_string();
                    break;
                }
            }
            for mid in body {
                if let Some(after_mid) = rest.strip_prefix(mid) {
                    let mut after_all = after_mid;
                    for tail in tails {
                        if let Some(after_tail) = after_all.strip_prefix(tail) {
                            after_all = after_tail;
                            break;
                        }
                    }
                    let prefix: String = chars[..start].iter().collect();
                    result = (prefix + after_all)
                        .trim_matches(|c: char| " ：:，,。？?".contains(c))
                        .to_string();
                    replaced = true;
                    break 'scan;
                }
            }
        }
        if !replaced {
            break;
        }
    }
    result
}

fn heuristic_multi_query_variants(query: &str, max_queries: usize) -> Vec<String> {
    let normalized_query = collapse_spaces(query.trim());
    if normalized_query.is_empty() {
        return Vec::new();
    }
    let mut candidates = vec![normalized_query.clone()];
    let mut cleaned = strip_prefix_phrase(&normalized_query);
    cleaned = strip_middle_phrase(&cleaned);

    let remove_question_words = |text: &str| -> String {
        let words = [
            "是什么", "有哪些", "如何", "怎么", "为什么", "是否", "能否", "可以吗", "吗", "呢",
            "请说明", "请解释", "介绍一下", "说明一下",
        ];
        let mut result = text.to_string();
        for word in words {
            result = result.replace(word, "");
        }
        result
            .trim_matches(|c: char| " ：:，,。？?".contains(c))
            .to_string()
    };

    if !cleaned.is_empty() {
        candidates.push(cleaned.clone());
    }
    let focus = remove_question_words(if cleaned.is_empty() {
        &normalized_query
    } else {
        &cleaned
    });
    if !focus.is_empty() {
        candidates.push(focus.clone());
    }

    // 结构化别名
    for alias in structured_query_aliases(&normalized_query) {
        candidates.push(alias);
    }

    // 语义片段扩展（去停词后的连续片段，空格拼接）
    let semantic_source = if !focus.is_empty() {
        focus.clone()
    } else if !cleaned.is_empty() {
        cleaned.clone()
    } else {
        normalized_query.clone()
    };
    let dropped = drop_stop_terms(&semantic_source);
    let terms = extract_semantic_terms(&dropped);
    if !terms.is_empty() {
        // 对齐 Python：语义片段只保留前 8 个词。
        let head: Vec<String> = terms.iter().take(8).cloned().collect();
        candidates.push(head.join(" "));
    }

    normalize_query_variants(&candidates, &normalized_query, max_queries)
}

/// 去 "(根据|基于|...|啊)" → " " 与虚词单字 → " "。
fn drop_stop_terms(text: &str) -> String {
    let words = [
        "根据", "基于", "结合", "请问", "麻烦", "帮我", "帮忙", "查询", "检索", "搜索", "查一下",
        "查找", "公司", "企业", "每个", "各个", "所有", "全部", "分别", "哪些", "哪几", "是谁",
        "是什么", "在哪里", "在哪", "多少", "如何", "怎么", "为什么", "是否", "有没有", "可以吗",
        "能否", "吗", "呢", "吧", "啊",
    ];
    let mut result = text.to_string();
    for word in words {
        result = result.replace(word, " ");
    }
    result
        .chars()
        .map(|c| {
            if "的之与和及在是有为从到对把将".contains(c) {
                ' '
            } else {
                c
            }
        })
        .collect()
}

/// `[A-Za-z0-9]+|[一-鿿]{2,}` 片段提取并保序去重。
fn extract_semantic_terms(text: &str) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();
    let mut run = String::new();
    let mut run_kind = 0; // 0 无 1 ascii 2 cjk
    for c in text.chars() {
        let kind = if c.is_ascii_alphanumeric() {
            1
        } else if ('\u{4e00}'..='\u{9fff}').contains(&c) {
            2
        } else {
            0
        };
        if kind != 0 && run_kind != 0 && kind != run_kind {
            push_term(&mut terms, &run);
            run.clear();
        }
        if kind != 0 {
            run.push(c);
        } else if !run.is_empty() {
            push_term(&mut terms, &run);
            run.clear();
        }
        run_kind = kind;
    }
    if !run.is_empty() {
        push_term(&mut terms, &run);
    }
    // CJK 连续段按 {2,} 语义: 单字 CJK 段丢弃
    terms
}

fn push_term(terms: &mut Vec<String>, term: &str) {
    let trimmed = term.trim();
    if trimmed.is_empty() {
        return;
    }
    if trimmed.chars().all(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)) && trimmed.chars().count() < 2
    {
        return;
    }
    if !terms.iter().any(|t| t == trimmed) {
        terms.push(trimmed.to_string());
    }
}

fn structured_query_aliases(query: &str) -> Vec<String> {
    // 对齐 Python re.sub(r"\s+","",...)：删除全部空白（不是合并成单空格）。
    let normalized: String = query.chars().filter(|c| !c.is_whitespace()).collect();
    if normalized.is_empty() {
        return Vec::new();
    }
    let mut aliases: Vec<String> = Vec::new();
    let has = |terms: &[&str]| terms.iter().any(|t| normalized.contains(t));
    let mut add = |value: &str, aliases: &mut Vec<String>| {
        let cleaned: String = value.chars().filter(|c| !c.is_whitespace()).collect();
        if cleaned.is_empty() || cleaned == normalized {
            return;
        }
        if !aliases.contains(&cleaned) {
            aliases.push(cleaned);
        }
    };
    let has_employee = has(&["员工", "人员", "人事", "花名册", "名册"]);
    let has_department = has(&["部门", "组织", "通讯录", "架构"]);
    let has_salary = has(&["工资", "薪资", "薪酬", "月薪", "应发", "实发", "奖金", "扣款"]);
    let has_count = has(&["统计", "总数", "总人数", "数量", "多少", "共有", "一共"]);
    let has_list = has(&["名单", "列表", "清单", "明细", "分别", "有哪些", "有谁", "是谁"]);
    let has_role = has(&["负责人", "主管", "上级"]);

    if has_salary {
        for value in ["工资明细", "薪资明细", "实发工资", "应发工资", "员工工资"] {
            add(value, &mut aliases);
        }
    }
    if has_employee {
        for value in ["员工名单", "员工花名册", "员工明细"] {
            add(value, &mut aliases);
        }
        if has_count {
            for value in ["员工人数", "员工总数", "员工统计"] {
                add(value, &mut aliases);
            }
        }
    }
    if has_department {
        for value in ["部门通讯录", "部门负责人", "部门名单", "部门数量", "部门统计"] {
            add(value, &mut aliases);
        }
    }
    if has_role {
        for value in ["负责人名单", "各部门负责人"] {
            add(value, &mut aliases);
        }
    }
    if has_count {
        if has_employee {
            add("在职员工人数", &mut aliases);
        }
        if has_department {
            add("部门数量统计", &mut aliases);
        }
    }
    if has_list {
        if has_employee {
            add("员工明细", &mut aliases);
        }
        if has_department {
            add("部门清单", &mut aliases);
        }
    }
    aliases
}

/// 复刻 generate_multi_query_variants（llm=None 路径）。
pub fn expand_multi_query(query: &str) -> Vec<String> {
    expand_multi_query_with_limit(query, MAX_MULTI_QUERY_COUNT)
}

fn expand_multi_query_with_limit(query: &str, max_queries: usize) -> Vec<String> {
    let normalized_query = collapse_spaces(query.trim());
    if normalized_query.is_empty() {
        return Vec::new();
    }
    let variants = if is_identifier_query(&normalized_query) {
        identifier_query_variants(&normalized_query)
    } else {
        heuristic_multi_query_variants(&normalized_query, max_queries.max(1))
    };
    collapse_near_duplicate_queries(variants)
}

fn variant_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

pub fn parse_multi_query_payload(text: &str) -> Value {
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

/// 对齐 Python：LLM 产出的 queries / query_variants / variants 规范化后至少 3 条才采用。
pub fn apply_llm_multi_query_payload(
    query: &str,
    payload: &Value,
    max_queries: usize,
) -> Option<Vec<String>> {
    let variants = payload
        .get("queries")
        .or_else(|| payload.get("query_variants"))
        .or_else(|| payload.get("variants"))
        .and_then(Value::as_array)?;
    let candidates: Vec<String> = variants.iter().map(variant_text).collect();
    let normalized = normalize_query_variants(&candidates, query, max_queries.max(1));
    (normalized.len() >= MIN_MULTI_QUERY_COUNT).then_some(normalized)
}

/// 启发式变体不足 2 条且配置了 LLM 时补一次改写；编号查询与 LLM 失败都回退启发式。
pub async fn expand_multi_query_with_llm(
    query: &str,
    llm: Option<&LlmClient>,
    max_queries: usize,
) -> Vec<String> {
    let max_queries = max_queries.max(1);
    let heuristic = expand_multi_query_with_limit(query, max_queries);
    let normalized_query = collapse_spaces(query.trim());
    if normalized_query.is_empty() || is_identifier_query(&normalized_query) {
        return heuristic;
    }
    let Some(llm) = llm else {
        return heuristic;
    };
    if heuristic.len() >= 2 {
        return heuristic;
    }
    let human = format!(
        "用户问题：{normalized_query}\n\n请输出 3 条查询变体，避免长句，优先提升 source_name、标题、编号和正文关键词召回。"
    );
    match llm
        .invoke(&[
            ChatMessage::new("system", MULTI_QUERY_SYSTEM_PROMPT),
            ChatMessage::new("user", human),
        ])
        .await
    {
        Ok(result) => {
            let payload = parse_multi_query_payload(&result.content);
            if let Some(variants) = apply_llm_multi_query_payload(&normalized_query, &payload, max_queries)
            {
                let variants = collapse_near_duplicate_queries(variants);
                tracing::info!(
                    query = normalized_query.as_str(),
                    count = variants.len(),
                    "LLM multi-query 补写成功"
                );
                return variants;
            }
            tracing::info!(query = normalized_query.as_str(), "LLM multi-query 变体不足，回退启发式");
        }
        Err(err) => tracing::warn!("LLM multi-query 失败，回退启发式: {err}"),
    }
    heuristic
}

fn identifier_query_variants(query: &str) -> Vec<String> {
    let normalized_query = collapse_spaces(query.trim());
    if normalized_query.is_empty() {
        return Vec::new();
    }
    // 工号题只保留原句。把裸编号再当一条查询会把工资表和花名册一起抬上来。
    vec![normalized_query]
}

/// 复刻 _canonical_identifier_terms：编号按压缩键去重保留代表值。
pub fn canonical_identifier_terms(query: &str) -> Vec<String> {
    let mut identifiers: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for item in extract_identifier_terms(query) {
        let key: String = item
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
            .collect::<String>()
            .to_lowercase();
        if key.is_empty() || seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        identifiers.push(item);
    }
    identifiers
}

/// 对齐 SparseRetriever：LIKE/持久索引候选上限 `max(top_k * 30, 80)`。
pub fn sparse_candidate_limit(top_k: usize) -> usize {
    top_k.saturating_mul(30).max(80)
}

/// 对齐 Python：仅 `cached` 且未超 `SPARSE_INDEX_MAX_ROWS` 时走进程内持久 BM25。
pub fn should_use_cached_sparse(mode: &str, row_count: usize, max_rows: usize) -> bool {
    mode.trim().eq_ignore_ascii_case("cached") && row_count <= max_rows
}

fn row_field<'a>(row: &'a Row, key: &str) -> &'a str {
    row.fields
        .get(key)
        .and_then(|value| value.as_str())
        .unwrap_or("")
}

fn searchable_row_text(row: &Row) -> String {
    let source_name = row_field(row, "source_name");
    let content = ["child_text", "chunk_text", "text"]
        .into_iter()
        .map(|key| row_field(row, key))
        .find(|value| !value.is_empty())
        .unwrap_or("");
    [source_name, content]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn row_matches_filter_terms(row: &Row, terms: &[String]) -> bool {
    let source_name = row_field(row, "source_name");
    let child_text = row_field(row, "child_text");
    terms.iter().any(|term| {
        !term.is_empty() && (child_text.contains(term) || source_name.contains(term))
    })
}

fn collect_like_candidates(rows: &[Row], terms: &[String], candidate_limit: usize) -> Vec<usize> {
    rows.iter()
        .enumerate()
        .filter(|(_, row)| row_matches_filter_terms(row, terms))
        .map(|(index, _)| index)
        .take(candidate_limit.max(1))
        .collect()
}

fn score_sparse_subset(rows: &[Row], indexes: &[usize], query: &str) -> Vec<(usize, f64, f64)> {
    if indexes.is_empty() {
        return Vec::new();
    }
    let texts: Vec<String> = indexes
        .iter()
        .map(|&index| searchable_row_text(&rows[index]))
        .collect();
    let tokens: Vec<Vec<String>> = texts.iter().map(|text| tokenize_cjk(text)).collect();
    let bm25 = BM25Okapi::build(&tokens);
    let query_tokens = tokenize_cjk(query);
    let scores = bm25.get_scores(&query_tokens);
    indexes
        .iter()
        .enumerate()
        .map(|(position, &row_index)| {
            let lexical = lexical_match_score(
                &query_tokens,
                &tokens[position],
                query,
                &texts[position],
            );
            (row_index, scores[position], lexical)
        })
        .collect()
}

/// 复刻 SparseRetriever._legacy_like_search：词法预筛候选后再在子集上重建 BM25。
pub fn legacy_like_search(rows: &[Row], query: &str, candidate_limit: usize) -> Vec<(usize, f64, f64)> {
    let filter_terms = build_filter_terms(query, 12);
    if filter_terms.is_empty() {
        return Vec::new();
    }
    let cap = candidate_limit.max(1);
    let mut candidates = collect_like_candidates(rows, &filter_terms, cap);
    if candidates.is_empty() && filter_terms.len() > 1 {
        let fallback = &filter_terms[..filter_terms.len().min(4)];
        candidates = collect_like_candidates(rows, fallback, cap);
    }
    score_sparse_subset(rows, &candidates, query)
}

/// 对齐 Python：`final_score = max(0, bm25) + lexical`，按分数降序、chunk_index 升序截断。
pub fn rank_sparse_hits(
    rows: &[Row],
    hits: Vec<(usize, f64, f64)>,
    top_k: usize,
) -> Vec<(usize, f64)> {
    let mut ranked: Vec<(f64, i64, usize)> = hits
        .into_iter()
        .map(|(row_index, bm25, lexical)| {
            let score = bm25.max(0.0) + lexical;
            let chunk_index = rows
                .get(row_index)
                .and_then(|row| row.fields.get("chunk_index"))
                .and_then(|value| value.as_int())
                .unwrap_or(0);
            (score, chunk_index, row_index)
        })
        .filter(|(score, _, _)| *score > 0.0)
        .collect();
    ranked.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    ranked
        .into_iter()
        .take(top_k.max(1))
        .map(|(score, _, row_index)| (row_index, score))
        .collect()
}

// ---------------- 稀疏索引（两段式 BM25，语义与 Python sparse_index 一致） ----------------

pub struct SparseIndexSnapshot {
    row_texts: Vec<String>,
    row_tokens: Vec<Vec<String>>,
    row_index: Vec<usize>, // 文本下标 -> 快照行下标
    bm25: BM25Okapi,
}

impl SparseIndexSnapshot {
    pub fn build(rows: &[Row]) -> Self {
        let mut row_texts = Vec::with_capacity(rows.len());
        let mut row_tokens = Vec::with_capacity(rows.len());
        let mut row_index = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let source_name = row
                .fields
                .get("source_name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let content = row
                .fields
                .get("child_text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let text = [source_name, content]
                .into_iter()
                .filter(|p| !p.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            row_tokens.push(tokenize_cjk(&text));
            row_texts.push(text);
            row_index.push(index);
        }
        let bm25 = BM25Okapi::build(&row_tokens);
        SparseIndexSnapshot {
            row_texts,
            row_tokens,
            row_index,
            bm25,
        }
    }

    /// 返回 (行下标, 切片 BM25 分, 词法分)。
    pub fn search(&self, query: &str, limit: usize) -> Vec<(usize, f64, f64)> {
        let query_tokens = tokenize_cjk(query);
        let global_scores = self.bm25.get_scores(&query_tokens);
        let mut order: Vec<usize> = (0..global_scores.len())
            .filter(|&index| global_scores[index] > 0.0)
            .collect();
        order.sort_by(|&a, &b| {
            global_scores[b]
                .partial_cmp(&global_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let cap = limit.max(1);
        if order.len() > cap {
            order.truncate(cap);
        }
        if order.is_empty() {
            return Vec::new();
        }
        let slice_corpus: Vec<&[String]> = order
            .iter()
            .map(|&index| self.row_tokens[index].as_slice())
            .collect();
        let slice_bm25 = BM25Okapi::build_from_slices(&slice_corpus);
        let slice_scores = slice_bm25.get_scores(&query_tokens);
        order
            .iter()
            .enumerate()
            .map(|(position, &text_index)| {
                let lexical = lexical_match_score(
                    &query_tokens,
                    &self.row_tokens[text_index],
                    query,
                    &self.row_texts[text_index],
                );
                (self.row_index[text_index], slice_scores[position], lexical)
            })
            .collect()
    }
}


/// 复刻 _is_structured_lookup_query。
pub fn is_structured_lookup_query(query: &str) -> bool {
    let normalized: String = query.split_whitespace().collect();
    if normalized.is_empty() {
        return false;
    }
    if eks_kernels::rerank::is_identifier_query(&normalized) {
        return true;
    }
    if normalized.chars().count() > 32 {
        return false;
    }
    STRUCTURED_QUERY_HINT_TERMS.iter().any(|t| normalized.contains(t))
}

/// 复刻 _citation_chunk_key。
pub fn citation_chunk_key(item: &RetrievalItem) -> (String, String, String) {
    let chunk = if !item.child_id.is_empty() {
        item.child_id.clone()
    } else {
        item.chunk_index.to_string()
    };
    (item.document_id.clone(), item.source_name.clone(), chunk)
}

/// 对齐 assemble_rag_context。
pub fn assemble_rag_context(results: &[RetrievalItem], max_chars: usize) -> String {
    let mut blocks: Vec<String> = Vec::new();
    let mut used_chars = 0usize;
    for (index, item) in results.iter().enumerate() {
        let number = index + 1;
        let source = if !item.source_name.is_empty() {
            item.source_name.clone()
        } else if !item.document_id.is_empty() {
            item.document_id.clone()
        } else {
            "未知来源".to_string()
        };
        let mut header = format!("[资料{number}] 来源: {source}");
        // 对齐 Python：chunk_index 缺失（-1）时不追加 chunk 后缀。
        if item.chunk_index >= 0 {
            header.push_str(&format!(" / chunk: {}", item.chunk_index));
        }
        let mut content = collapse_spaces(item.content.trim());
        if let Some(parent) = &item.parent_context_text {
            let parent = collapse_spaces(parent.trim());
            if !parent.is_empty() && parent != content {
                content = format!("[命中行] {content}
[父块上下文] {parent}");
            }
        }
        let mut block = format!("{header}
{content}");
        if used_chars + block.chars().count() > max_chars {
            let remaining = max_chars.saturating_sub(used_chars);
            if remaining <= 120 {
                break;
            }
            block = block.chars().take(remaining).collect::<String>().trim_end().to_string();
        }
        used_chars += block.chars().count();
        blocks.push(block);
        if used_chars >= max_chars {
            break;
        }
    }
    blocks.join("

")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_query_has_fewer_than_two_heuristic_variants() {
        assert!(expand_multi_query("报销").len() < 2);
        assert!(expand_multi_query("差旅报销流程是什么").len() >= 2);
        assert!(expand_multi_query("").is_empty());
    }

    #[test]
    fn identifier_query_keeps_number_terms() {
        let queries = expand_multi_query("查询工号 E0011");
        assert!(queries.iter().any(|item| item.contains("E0011")));
        assert_eq!(
            expand_multi_query("员工号 E0011 的姓名和所属部门是什么？").len(),
            1,
            "工号题不应再扩出裸编号变体"
        );
    }

    #[test]
    fn near_duplicate_query_variants_collapse() {
        let collapsed = collapse_near_duplicate_queries(vec![
            "差旅报销流程是什么".into(),
            "差旅报销流程是什么".into(),
            "差旅报销流程是什么吗".into(),
            "费用标准".into(),
        ]);
        assert_eq!(collapsed, vec!["差旅报销流程是什么", "费用标准"]);
    }

    #[test]
    fn llm_payload_requires_three_normalized_variants() {
        let accepted = apply_llm_multi_query_payload(
            "报销",
            &json!({"queries": ["差旅报销", "费用标准", "报销流程"]}),
            3,
        )
        .unwrap();
        assert_eq!(accepted.len(), 3);
        assert_eq!(accepted[0], "报销");

        let by_alias = apply_llm_multi_query_payload(
            "报销",
            &json!({"query_variants": ["报销", "差旅报销", "费用标准"]}),
            3,
        )
        .unwrap();
        assert_eq!(by_alias[0], "报销");
        assert_eq!(by_alias.len(), 3);

        assert!(
            apply_llm_multi_query_payload("报销", &json!({"queries": ["报销", "差旅报销"]}), 3)
                .is_none()
        );
        assert!(apply_llm_multi_query_payload("报销", &json!({"queries": "not-a-list"}), 3).is_none());
    }

    #[test]
    fn parses_fenced_multi_query_json() {
        let payload = parse_multi_query_payload(
            "```json\n{\"variants\":[\"报销标准\",\"差旅报销\",\"费用报销\"]}\n```",
        );
        let variants = apply_llm_multi_query_payload("报销", &payload, 3).unwrap();
        assert_eq!(variants.len(), 3);
        assert!(variants.contains(&"报销标准".to_string()));
    }

    #[tokio::test]
    async fn without_llm_keeps_heuristic_path() {
        assert_eq!(
            expand_multi_query_with_llm("报销", None, 3).await,
            expand_multi_query("报销")
        );
        assert_eq!(
            expand_multi_query_with_llm("差旅报销流程是什么", None, 3).await,
            expand_multi_query("差旅报销流程是什么")
        );
    }

    fn test_row(pk: i64, source: &str, text: &str, chunk_index: i64) -> Row {
        let mut fields = HashMap::new();
        fields.insert("source_name".into(), FieldValue::Str(source.into()));
        fields.insert("child_text".into(), FieldValue::Str(text.into()));
        fields.insert("chunk_index".into(), FieldValue::Int(chunk_index));
        fields.insert("document_id".into(), FieldValue::Str("doc".into()));
        Row {
            pk,
            vector: Vec::new(),
            fields,
        }
    }

    #[test]
    fn cached_sparse_skips_when_over_max_or_legacy() {
        assert!(should_use_cached_sparse("cached", 10, 50_000));
        assert!(should_use_cached_sparse("CACHED", 50_000, 50_000));
        assert!(!should_use_cached_sparse("cached", 50_001, 50_000));
        assert!(!should_use_cached_sparse("legacy", 10, 50_000));
        assert_eq!(sparse_candidate_limit(1), 80);
        assert_eq!(sparse_candidate_limit(2), 80);
        assert_eq!(sparse_candidate_limit(3), 90);
        assert_eq!(sparse_candidate_limit(5), 150);
    }

    #[test]
    fn legacy_like_keeps_shared_bigram_rows() {
        let rows = vec![
            test_row(1, "手册.pdf", "如何修改密码", 2),
            test_row(2, "差旅.xlsx", "差旅报销流程说明", 1),
        ];
        let hits = legacy_like_search(&rows, "账号密码", 80);
        let ranked = rank_sparse_hits(&rows, hits, 5);
        assert!(
            ranked.iter().any(|(index, _)| *index == 0),
            "「修改密码」应被「账号密码」的二元词预筛命中"
        );
        assert!(!ranked.iter().any(|(index, _)| *index == 1));
    }
}
