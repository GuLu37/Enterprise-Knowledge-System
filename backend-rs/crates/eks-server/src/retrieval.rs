//! 检索编排：retrieval_service.py 的 Rust 移植（B4a 子集）。
//!
//! 覆盖: 多查询启发式扩展、dense(milvus_lite 直读 COSINE)+sparse(内核 BM25 两段式)、
//! 变体间 RRF 融合与原始查询保护、规则重排(含 adaptive 精排跳过语义)、
//! 关键词门控过滤、编号保护、结构化 schema 打分、父块上下文批量回填、
//! 上下文组装与序列化。
//! 暂缺(B4b): _expand_structured_section_evidence 同 Sheet 聚合展开——由 bench 门禁判定是否必要。

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use eks_kernels::bm25::BM25Okapi;
use eks_kernels::fuse::{FuseItem, fuse_ranked_results};
use eks_kernels::rerank::{
    Candidate, build_query_keywords, build_relevance_signals, extract_identifier_terms,
    is_identifier_query, is_reference_index_result, normalize_source_score,
    passes_keyword_relevance_gate, score_result,
};
use eks_kernels::text::{normalize_text, tokenize_cjk};
use serde_json::{Value, json};

use crate::db;
use crate::milvus_lite::{FieldValue, MilvusLiteStore, Row};

const MAX_MULTI_QUERY_COUNT: usize = 3;
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
    /// 结构化同 Sheet 聚合标记（_expand_structured_section_evidence 产物）。
    pub structured_aggregation: bool,
    pub section_name: Option<String>,
    pub chunk_indices: Vec<i64>,
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
            content: row
                .fields
                .get("child_text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
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
            structured_aggregation: false,
            section_name: None,
            chunk_indices: Vec::new(),
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
        if let Some(signals) = &self.signals {
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
    // 逐位置扫描第一个可匹配组合
    'outer: for start in 0..result.chars().count() {
        let mut rest: String = result.chars().skip(start).collect();
        let mut head_offset = 0usize;
        for head in optional_head {
            if let Some(after) = rest.strip_prefix(head) {
                head_offset = head.chars().count();
                rest = after.to_string();
                break;
            }
        }
        let _ = head_offset;
        for mid in body {
            if let Some(after_mid) = rest.strip_prefix(mid) {
                let mut tail_len = 0;
                for tail in tails {
                    if let Some(after_tail) = after_mid.strip_prefix(tail) {
                        tail_len = tail.chars().count();
                        let after_all = after_tail;
                        let prefix: String = result.chars().take(start).collect();
                        let replacement: String =
                            prefix + after_all;
                        result = replacement
                            .trim_matches(|c: char| " ：:，,。？?".contains(c))
                            .to_string();
                        break 'outer;
                    }
                }
                if tail_len == 0 {
                    let prefix: String = result.chars().take(start).collect();
                    let replacement = prefix + after_mid;
                    result = replacement
                        .trim_matches(|c: char| " ：:，,。？?".contains(c))
                        .to_string();
                    break 'outer;
                }
            }
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
        candidates.push(terms.join(" "));
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
    let normalized: String = query.split_whitespace().collect();
    if normalized.is_empty() {
        return Vec::new();
    }
    let mut aliases: Vec<String> = Vec::new();
    let has = |terms: &[&str]| terms.iter().any(|t| normalized.contains(t));
    let mut add = |value: &str, aliases: &mut Vec<String>| {
        let cleaned: String = value.split_whitespace().collect();
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
    let normalized_query = collapse_spaces(query.trim());
    if normalized_query.is_empty() {
        return Vec::new();
    }
    if is_identifier_query(&normalized_query) {
        return identifier_query_variants(&normalized_query);
    }
    heuristic_multi_query_variants(&normalized_query, MAX_MULTI_QUERY_COUNT)
}

fn identifier_query_variants(query: &str) -> Vec<String> {
    let normalized_query = collapse_spaces(query.trim());
    let mut candidates = vec![normalized_query.clone()];
    for term in canonical_identifier_terms(&normalized_query) {
        candidates.push(term);
    }
    let max = MAX_MULTI_QUERY_COUNT.max(candidates.len());
    normalize_query_variants(&candidates, &normalized_query, max)
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
        let mut order: Vec<usize> = (0..global_scores.len()).collect();
        order.sort_by(|&a, &b| {
            global_scores[b]
                .partial_cmp(&global_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut slice: Vec<usize> = Vec::new();
        for index in order.into_iter().take(limit.max(1)) {
            if global_scores[index] <= 0.0 {
                break;
            }
            slice.push(index);
        }
        if slice.is_empty() {
            return Vec::new();
        }
        let slice_corpus: Vec<Vec<String>> = slice.iter().map(|&i| self.row_tokens[i].clone()).collect();
        let slice_bm25 = BM25Okapi::build(&slice_corpus);
        let slice_scores = slice_bm25.get_scores(&query_tokens);
        slice
            .iter()
            .enumerate()
            .map(|(position, &text_index)| {
                let lexical = eks_kernels::lexical::lexical_match_score(
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
        header.push_str(&format!(" / chunk: {}", item.chunk_index));
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
