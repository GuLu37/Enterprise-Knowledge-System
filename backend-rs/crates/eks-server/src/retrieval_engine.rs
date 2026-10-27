//! 检索引擎主体：多查询混合召回、变体融合与保护、规则重排过滤、父块回填。
//! （retrieval.rs 提供查询扩展与稀疏索引；本模块对齐 retrieval_service.py 编排层。）

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use eks_kernels::fuse::{FuseItem, fuse_ranked_results};
use eks_kernels::rerank::{
    Candidate, build_query_focus_terms, build_query_keywords, build_relevance_signals,
    build_keyword_specificity_scores, normalize_source_score, passes_keyword_relevance_gate,
    score_result, is_reference_index_result,
};

use crate::embedder::Embedder;
use crate::milvus_lite::MilvusLiteStore;
use crate::retrieval::{
    MAX_CITATION_RESULTS, MIN_FINAL_RELEVANCE_SCORE, STRUCTURED_QUERY_CITATION_RESULTS,
    RetrievalItem, SparseIndexSnapshot, canonical_identifier_terms, citation_chunk_key,
    collapse_spaces, expand_multi_query, is_structured_lookup_query,
};

const MAX_MULTI_QUERY_COUNT: usize = 3;
const MIN_RAG_CANDIDATE_POOL: usize = 60;
const STRUCTURED_SECTION_PARENT_LIMIT: usize = 200;
const STRUCTURED_SECTION_ROW_CHAR_LIMIT: usize = 24000;

fn candidate_pool_limit(top_k: usize) -> usize {
    (top_k * 8).max(top_k + 20).max(MIN_RAG_CANDIDATE_POOL)
}

fn is_structured_result(item: &RetrievalItem) -> bool {
    let file_type = item.file_type.trim().trim_start_matches('.').to_lowercase();
    file_type == "csv" || file_type == "tsv" || file_type == "xls" || file_type == "xlsx"
        || item.content.trim_start().starts_with("[Sheet]")
}

fn structured_schema_parts(content: &str) -> (String, String) {
    let lines: Vec<&str> = content.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if lines.is_empty() {
        return (String::new(), String::new());
    }
    if lines[0].starts_with("[Sheet]") {
        let sheet = lines[0].strip_prefix("[Sheet]").map(|s| s.trim().to_string()).unwrap_or_default();
        let header = lines.get(1).copied().unwrap_or("").to_string();
        (sheet, header)
    } else {
        (String::new(), lines[0].to_string())
    }
}

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

fn structured_schema_score(item: &RetrievalItem, query: &str) -> f64 {
    let (sheet_name, header) = structured_schema_parts(&item.content);
    let schema_text = eks_kernels::text::normalize_text(
        &[sheet_name.as_str(), header.as_str()]
            .into_iter()
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
    );
    if schema_text.is_empty() {
        return 0.0;
    }
    let sheet_text = eks_kernels::text::normalize_text(&sheet_name);
    let header_text = eks_kernels::text::normalize_text(&header);

    let normalized_query: String = query.split_whitespace().collect();
    let mut terms: Vec<String> = Vec::new();
    let has = |list: &[&str]| list.iter().any(|t| normalized_query.contains(t));
    if has(&["工资", "薪资", "薪酬", "月薪", "应发", "实发", "奖金", "扣款"]) {
        for v in ["工资", "薪资", "薪酬", "基本工资", "应发工资", "实发工资", "奖金", "扣款", "个税", "社保", "公积金"] {
            if !terms.iter().any(|t| t == v) { terms.push(v.into()); }
        }
    }
    if has(&["员工", "人员", "人事", "花名册", "名册", "姓名", "工号"]) {
        for v in ["员工", "人员", "员工号", "工号", "姓名", "员工姓名", "部门", "岗位", "职位", "状态", "入职", "邮箱", "电话"] {
            if !terms.iter().any(|t| t == v) { terms.push(v.into()); }
        }
    }
    if has(&["部门", "组织", "通讯录", "架构", "负责人", "主管"]) {
        for v in ["部门", "负责人", "主管", "直属主管", "办公地点", "办公", "地点", "联系人", "电话", "邮箱"] {
            if !terms.iter().any(|t| t == v) { terms.push(v.into()); }
        }
    }
    for term in build_query_keywords(query, 32) {
        if !terms.iter().any(|t| *t == term) { terms.push(term); }
    }
    let stop_terms = ["名单", "列表", "清单", "统计", "总数", "总人数", "数量", "多少", "共有", "一共"];
    let mut useful: Vec<String> = Vec::new();
    for term in terms {
        let normalized = eks_kernels::text::normalize_text(&term);
        if normalized.chars().count() < 2 || stop_terms.contains(&normalized.as_str()) {
            continue;
        }
        if !useful.contains(&normalized) {
            useful.push(normalized);
        }
    }
    if useful.is_empty() {
        return 0.0;
    }

    let mut hit_weight = 0.0f64;
    let mut max_weight = 0.0f64;
    for term in &useful {
        let weight = term.chars().count().min(8) as f64 / 8.0;
        max_weight += weight;
        if header_text.contains(term.as_str()) {
            hit_weight += weight;
        } else if sheet_text.contains(term.as_str()) {
            hit_weight += weight * 0.65;
        }
    }
    let mut score = if max_weight > 0.0 { hit_weight / max_weight } else { 0.0 };

    let meta_hits = STRUCTURED_META_SCHEMA_TERMS
        .iter()
        .filter(|t| schema_text.contains(&eks_kernels::text::normalize_text(t)))
        .count();
    let meta_header_hits = STRUCTURED_META_HEADER_TERMS
        .iter()
        .filter(|t| header_text.contains(&eks_kernels::text::normalize_text(t)))
        .count();
    let data_sheet_hit = STRUCTURED_DATA_SHEET_TERMS
        .iter()
        .any(|t| sheet_text.contains(&eks_kernels::text::normalize_text(t)));
    let data_field_hits = useful.iter().filter(|t| header_text.contains(t.as_str())).count();
    if data_sheet_hit && data_field_hits >= 2 {
        score += 0.15;
    }
    if meta_hits + meta_header_hits >= 2 && data_field_hits < 2 {
        score *= 0.25;
    }
    score.min(1.0)
}

pub struct RetrievalEngine {
    pub store: MilvusLiteStore,
    pub embedder: Arc<Embedder>,
    pub pool: sqlx::SqlitePool,
    sparse: tokio::sync::RwLock<Option<Arc<SparseIndexSnapshot>>>,
}

impl RetrievalEngine {
    pub fn new(store: MilvusLiteStore, embedder: Arc<Embedder>, pool: sqlx::SqlitePool) -> Self {
        RetrievalEngine {
            store,
            embedder,
            pool,
            sparse: tokio::sync::RwLock::new(None),
        }
    }

    async fn sparse_snapshot(&self) -> Result<Arc<SparseIndexSnapshot>, String> {
        {
            let cached = self.sparse.read().await;
            if let Some(snapshot) = cached.as_ref() {
                return Ok(snapshot.clone());
            }
        }
        let snapshot = self
            .store
            .collection("doc_chunk")
            .await
            .map_err(|e| e.to_string())?;
        let built = Arc::new(SparseIndexSnapshot::build(&snapshot.rows));
        *self.sparse.write().await = Some(built.clone());
        Ok(built)
    }

    pub async fn invalidate(&self) {
        self.store.invalidate("doc_chunk").await;
        *self.sparse.write().await = None;
    }

    /// 完整 RAG 工作流（对齐 run_rag_workflow, llm=None）。
    pub async fn run_rag_workflow(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<(Vec<RetrievalItem>, String), String> {
        let normalized_query = query.trim();
        if normalized_query.is_empty() {
            return Ok((Vec::new(), String::new()));
        }
        let recalled = self.retrieve_documents_multi_query(normalized_query, top_k).await?;
        let processed = self.post_process(normalized_query, recalled, top_k).await?;
        let max_chars = if processed
            .iter()
            .any(|item| item.structured_aggregation)
        {
            crate::retrieval::STRUCTURED_CONTEXT_MAX_CHARS
        } else {
            crate::retrieval::RAG_CONTEXT_MAX_CHARS
        };
        let context = crate::retrieval::assemble_rag_context(&processed, max_chars);
        Ok((processed, context))
    }

    pub async fn retrieve_documents_multi_query(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievalItem>, String> {
        let mut queries = expand_multi_query(query);
        if !eks_kernels::rerank::is_identifier_query(query) {
            queries.truncate(MAX_MULTI_QUERY_COUNT);
        }
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let resolved_top_k = top_k.clamp(1, 50);
        let candidate_k = candidate_pool_limit(resolved_top_k);

        let snapshot = self.store.collection("doc_chunk").await.map_err(|e| e.to_string())?;
        let sparse_index = self.sparse_snapshot().await?;

        // dense: 一次 embedding 批量 + 每变体一次暴力 COSINE
        let query_vectors = self.embedder.embed_queries(&queries)?;
        let mut variant_sets: Vec<Vec<RetrievalItem>> = Vec::with_capacity(queries.len());
        for (position, q) in queries.iter().enumerate() {
            let dense_hits = self
                .store
                .search_cosine("doc_chunk", &query_vectors[position], candidate_k)
                .await
                .map_err(|e| e.to_string())?;
            let dense: Vec<FuseItem<RetrievalItem>> = dense_hits
                .iter()
                .map(|(row_index, score)| {
                    RetrievalItem::from_row(&snapshot.rows[*row_index], *score, "dense").to_fuse_item()
                })
                .collect();
            let sparse_hits = sparse_index.search(q, candidate_k.max(80));
            let sparse: Vec<FuseItem<RetrievalItem>> = sparse_hits
                .iter()
                .map(|(row_index, bm25, lexical)| {
                    let final_score = bm25.max(0.0) + lexical;
                    RetrievalItem::from_row(&snapshot.rows[*row_index], final_score, "sparse").to_fuse_item()
                })
                .collect();
            if dense.is_empty() && sparse.is_empty() {
                variant_sets.push(Vec::new());
                continue;
            }
            let fused = fuse_ranked_results(vec![dense, sparse], &[0.5, 0.5], &["dense", "sparse"], 60.0);
            variant_sets.push(fused.into_iter().take(resolved_top_k).map(|out| out.value).collect());
        }

        // 变体间融合（原始查询 1.4 / 变体 0.75）
        let nonempty_indexes: Vec<usize> = (0..variant_sets.len()).filter(|&i| !variant_sets[i].is_empty()).collect();
        if nonempty_indexes.is_empty() {
            return Ok(Vec::new());
        }
        let sets: Vec<Vec<FuseItem<RetrievalItem>>> = nonempty_indexes
            .iter()
            .map(|&i| variant_sets[i].iter().map(|item| item.to_fuse_item()).collect())
            .collect();
        let weights: Vec<f64> = nonempty_indexes.iter().map(|&i| if i == 0 { 1.4 } else { 0.75 }).collect();
        let labels: Vec<String> = nonempty_indexes.iter().map(|&i| format!("query_{}", i + 1)).collect();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        let fused = fuse_ranked_results(sets, &weights, &label_refs, 60.0);

        let mut selected: Vec<RetrievalItem> = fused.into_iter().take(candidate_k).map(|out| out.value).collect();
        let mut seen: HashSet<(String, String, String)> = selected.iter().map(citation_chunk_key).collect();

        let keep_candidate = |item: &RetrievalItem,
                              selected: &mut Vec<RetrievalItem>,
                              seen: &mut HashSet<(String, String, String)>| {
            let key = citation_chunk_key(item);
            if seen.contains(&key) {
                return;
            }
            if selected.len() >= candidate_k {
                if let Some(removed) = selected.pop() {
                    seen.remove(&citation_chunk_key(&removed));
                }
            }
            seen.insert(key);
            selected.push(item.clone());
        };

        // 原始查询保护
        let keywords = build_query_keywords(query, 24);
        let focus_terms = build_query_focus_terms(query, 8);
        if let Some(direct) = variant_sets.first() {
            for item in direct.iter().take((resolved_top_k * 2).max(12)) {
                if item.content.trim().is_empty() {
                    continue;
                }
                let candidate = item.as_candidate();
                let signals = build_relevance_signals(query, &candidate, &keywords, &focus_terms);
                if passes_keyword_relevance_gate(query, &candidate, &keywords, &focus_terms, &signals) {
                    keep_candidate(item, &mut selected, &mut seen);
                }
            }
        }

        // 结构化路由保护（每变体最多 3 条）
        if is_structured_lookup_query(query) {
            for set in &variant_sets {
                let mut kept = 0;
                for item in set {
                    if !is_structured_result(item)
                        || is_reference_index_result(&item.content)
                        || structured_schema_score(item, query) <= 0.12
                    {
                        continue;
                    }
                    keep_candidate(item, &mut selected, &mut seen);
                    kept += 1;
                    if kept >= 3 {
                        break;
                    }
                }
            }
        }
        Ok(selected)
    }

    /// 规则重排（Rust 侧无 cross-encoder，等价 adaptive 跳过路径的打分）。
    fn rule_rerank(&self, query: &str, items: Vec<RetrievalItem>) -> Vec<RetrievalItem> {
        if items.is_empty() {
            return items;
        }
        let keywords = build_query_keywords(query, 24);
        let focus_terms = build_query_focus_terms(query, 8);
        let candidates: Vec<Candidate> = items.iter().map(|item| item.as_candidate()).collect();
        let signals: Vec<_> = candidates
            .iter()
            .map(|candidate| build_relevance_signals(query, candidate, &keywords, &focus_terms))
            .collect();
        let specificity = build_keyword_specificity_scores(
            &keywords,
            &candidates.iter().map(|c| (c.content, c.source_name)).collect::<Vec<_>>(),
        );
        let raw_scores: Vec<f64> = items
            .iter()
            .map(|item| normalize_source_score(item.score, &item.source))
            .collect();
        let min_score = raw_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = raw_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let span = max_score - min_score;

        let mut scored: Vec<(usize, f64, f64)> = Vec::with_capacity(items.len());
        for (index, item) in items.iter().enumerate() {
            let raw_norm = if span > 1e-9 {
                (raw_scores[index] - min_score) / span
            } else if raw_scores[index] > 0.0 {
                1.0
            } else {
                0.0
            };
            let gate = passes_keyword_relevance_gate(
                query,
                &candidates[index],
                &keywords,
                &focus_terms,
                &signals[index],
            );
            let score = score_result(&item.source, raw_norm, specificity[index], &signals[index], gate);
            scored.push((index, score, item.score));
        }
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.0.cmp(&b.0))
        });

        scored
            .into_iter()
            .map(|(index, score, _)| {
                let mut item = items[index].clone();
                item.score = score;
                item.signals = Some(crate::retrieval::ItemSignals {
                    keyword_coverage: signals[index].keyword_coverage,
                    focus_coverage: signals[index].focus_coverage,
                    heading_coverage: signals[index].heading_coverage,
                    heading_focus_coverage: signals[index].heading_focus_coverage,
                    exact_match: signals[index].exact_match,
                    is_reference_index: signals[index].is_reference_index,
                    keyword_specificity: specificity[index],
                });
                item.rerank_stage = Some("rule_only_rust".into());
                item
            })
            .collect()
    }

    pub async fn post_process(
        &self,
        query: &str,
        results: Vec<RetrievalItem>,
        top_k: usize,
    ) -> Result<Vec<RetrievalItem>, String> {
        let filtered = self.filter_retrieval_results(query, results, top_k);
        let expanded = self.expand_structured_section_evidence(query, filtered).await;
        self.expand_parent_evidence(expanded).await
    }

    /// 复刻 _expand_structured_section_evidence：同 Sheet parent 聚合为可读证据。
    async fn expand_structured_section_evidence(
        &self,
        query: &str,
        results: Vec<RetrievalItem>,
    ) -> Vec<RetrievalItem> {
        if !results.iter().any(is_structured_result) {
            return results;
        }
        let mut parent_cache: HashMap<String, Vec<ParentRecord>> = HashMap::new();
        let mut seen_sections: HashSet<(String, String)> = HashSet::new();
        let mut expanded: Vec<RetrievalItem> = Vec::with_capacity(results.len());
        for item in results {
            if !is_structured_result(&item) || is_reference_index_result(&item.content) {
                expanded.push(item);
                continue;
            }
            let section_name = structured_section_marker(&item.content);
            let section_key = (item.document_id.clone(), section_name.clone());
            if item.document_id.is_empty() || section_name.is_empty()
                || seen_sections.contains(&section_key)
            {
                expanded.push(item);
                continue;
            }
            seen_sections.insert(section_key);
            if !parent_cache.contains_key(&item.document_id) {
                let rows = self.query_parent_rows_by_document(&item.document_id).await;
                parent_cache.insert(item.document_id.clone(), rows);
            }
            let parent_rows = parent_cache
                .get(&item.document_id)
                .cloned()
                .unwrap_or_default();
            match build_structured_section_summary(&item, &parent_rows, query) {
                Some(summary) => expanded.push(summary),
                None => expanded.push(item),
            }
        }
        expanded
    }

    /// 按 document_id 读取全部 parent 行（chunk_index 升序）。
    async fn query_parent_rows_by_document(
        &self,
        document_id: &str,
    ) -> Vec<ParentRecord> {
        #[derive(sqlx::FromRow)]
        struct Row {
            parent_id: Option<String>,
            document_id: String,
            chunk_index: i64,
            chunk_text: String,
        }
        let rows: Vec<Row> = sqlx::query_as(
            "SELECT parent_id, document_id, chunk_index, chunk_text FROM document_chunks \
             WHERE document_id = ? ORDER BY chunk_index ASC",
        )
        .bind(document_id)
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();
        rows.into_iter()
            .filter(|row| !row.chunk_text.trim().is_empty())
            .map(|row| ParentRecord {
                parent_id: row
                    .parent_id
                    .filter(|v| !v.is_empty())
                    .unwrap_or_else(|| format!("{}:p{:06}", row.document_id, row.chunk_index)),
                parent_index: row.chunk_index,
                parent_text: row.chunk_text,
            })
            .collect()
    }

    fn filter_retrieval_results(
        &self,
        query: &str,
        results: Vec<RetrievalItem>,
        top_k: usize,
    ) -> Vec<RetrievalItem> {
        let identifiers = canonical_identifier_terms(query);
        let structured_lookup = is_structured_lookup_query(query);
        let mut max_citation = if structured_lookup { STRUCTURED_QUERY_CITATION_RESULTS } else { MAX_CITATION_RESULTS };
        let mut requested_top_k = top_k.max(1);
        if !identifiers.is_empty() {
            max_citation = max_citation.max(identifiers.len());
            requested_top_k = requested_top_k.max(identifiers.len());
        }
        let resolved_top_k = requested_top_k.min(max_citation).max(1);

        let candidates: Vec<RetrievalItem> = results
            .into_iter()
            .filter(|item| !item.content.trim().is_empty())
            .collect();
        if candidates.is_empty() {
            return Vec::new();
        }
        let reranked = self.rule_rerank(query, candidates);
        if reranked.is_empty() {
            return Vec::new();
        }

        let keywords = build_query_keywords(query, 24);
        let focus_terms = build_query_focus_terms(query, 8);
        let keyword_matched: Vec<usize> = reranked
            .iter()
            .enumerate()
            .filter(|(_, item)| {
                let candidate = item.as_candidate();
                let signals = build_relevance_signals(query, &candidate, &keywords, &focus_terms);
                passes_keyword_relevance_gate(query, &candidate, &keywords, &focus_terms, &signals)
            })
            .map(|(index, _)| index)
            .collect();

        let ranking_pool: Vec<usize> = if structured_lookup || keyword_matched.is_empty() {
            (0..reranked.len()).collect()
        } else {
            keyword_matched.clone()
        };
        let best_score = ranking_pool
            .iter()
            .map(|&i| reranked[i].score)
            .fold(f64::NEG_INFINITY, f64::max);

        let filtered_indexes: Vec<usize> = if structured_lookup {
            let structured_matched: Vec<usize> = ranking_pool
                .iter()
                .copied()
                .filter(|&i| is_structured_result(&reranked[i]))
                .collect();
            if !structured_matched.is_empty() {
                let non_reference: Vec<usize> = structured_matched
                    .iter()
                    .copied()
                    .filter(|&i| !is_reference_index_result(&reranked[i].content))
                    .collect();
                let pool = if non_reference.is_empty() { structured_matched } else { non_reference };
                let scores: Vec<f64> = pool.iter().map(|&i| structured_schema_score(&reranked[i], query)).collect();
                let best = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut chosen: Vec<usize> = if best > 0.0 {
                    let minimum = 0.25f64.max(best * 0.5);
                    pool.iter().zip(scores.iter()).filter(|(_, s)| **s >= minimum).map(|(i, _)| *i).collect()
                } else {
                    pool
                };
                chosen.sort_by(|&a, &b| {
                    structured_schema_score(&reranked[b], query)
                        .partial_cmp(&structured_schema_score(&reranked[a], query))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            reranked[b].score.partial_cmp(&reranked[a].score).unwrap_or(std::cmp::Ordering::Equal)
                        })
                });
                chosen
            } else {
                ranking_pool[..resolved_top_k.min(ranking_pool.len())].to_vec()
            }
        } else if keyword_matched.is_empty() {
            ranking_pool[..resolved_top_k.min(ranking_pool.len())].to_vec()
        } else {
            let score_floor = MIN_FINAL_RELEVANCE_SCORE.max(best_score * 0.62);
            let mut chosen: Vec<usize> = ranking_pool
                .iter()
                .copied()
                .filter(|&i| reranked[i].score >= score_floor)
                .collect();
            if chosen.is_empty() && best_score >= MIN_FINAL_RELEVANCE_SCORE {
                chosen = vec![ranking_pool[0]];
            }
            chosen
        };

        // 多编号保护
        let mut protected: Vec<usize> = Vec::new();
        for identifier in &identifiers {
            let key: String = identifier
                .chars()
                .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
                .collect::<String>()
                .to_lowercase();
            let matched = ranking_pool.iter().find(|&&i| {
                let item = &reranked[i];
                let text = format!("{}\n{}\n{}", item.content, item.source_name, item.document_id);
                let text_key: String = text
                    .chars()
                    .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
                    .collect::<String>()
                    .to_lowercase();
                text_key.contains(&key)
            });
            if let Some(&index) = matched {
                protected.push(index);
            }
        }

        let mut deduplicated: Vec<RetrievalItem> = Vec::new();
        let mut seen: HashSet<(String, String, String)> = HashSet::new();
        let mut append = |item: &RetrievalItem,
                          out: &mut Vec<RetrievalItem>,
                          seen: &mut HashSet<(String, String, String)>| {
            if out.len() >= resolved_top_k {
                return;
            }
            let key = if structured_lookup {
                (
                    item.document_id.clone(),
                    item.parent_id.clone(),
                    if item.child_id.is_empty() { item.chunk_index.to_string() } else { item.child_id.clone() },
                )
            } else {
                citation_chunk_key(item)
            };
            if seen.contains(&key) {
                return;
            }
            seen.insert(key);
            out.push(item.clone());
        };
        for &index in &protected {
            append(&reranked[index], &mut deduplicated, &mut seen);
        }
        for &index in &filtered_indexes {
            if deduplicated.len() >= resolved_top_k {
                break;
            }
            append(&reranked[index], &mut deduplicated, &mut seen);
        }
        deduplicated
    }

    async fn expand_parent_evidence(
        &self,
        results: Vec<RetrievalItem>,
    ) -> Result<Vec<RetrievalItem>, String> {
        if results.is_empty() {
            return Ok(results);
        }
        let parents = self.load_parent_texts(&results).await;
        let mut output: Vec<RetrievalItem> = Vec::with_capacity(results.len());
        let mut seen: HashSet<(String, String, String)> = HashSet::new();
        for item in results {
            if item.structured_aggregation {
                // 聚合证据已是完整上下文，跳过父块回填（对齐 Python 语义）。
                output.push(item);
                continue;
            }
            let key = citation_chunk_key(&item);
            if seen.contains(&key) {
                continue;
            }
            seen.insert(key);
            let mut enriched = item.clone();
            let parent_text = parents
                .get(&(item.document_id.clone(), item.parent_id.clone()))
                .or_else(|| {
                    parents
                        .get(&(item.document_id.clone(), format!("idx:{}", item.parent_index)))
                });
            if let Some(text) = parent_text {
                enriched.parent_context_text = Some(text.clone());
            }
            output.push(enriched);
        }
        Ok(output)
    }

    /// 一次会话批量读 parent（parent_id 优先，chunk_index 兜底）。
    async fn load_parent_texts(
        &self,
        items: &[RetrievalItem],
    ) -> HashMap<(String, String), String> {
        #[derive(sqlx::FromRow)]
        struct ParentRow {
            document_id: String,
            parent_id: String,
            chunk_index: i64,
            chunk_text: String,
        }
        let mut result: HashMap<(String, String), String> = HashMap::new();
        let mut need_by_parent: Vec<(String, String)> = Vec::new();
        let mut need_by_index: Vec<(String, i64)> = Vec::new();
        for item in items {
            if item.document_id.is_empty() {
                continue;
            }
            if !item.parent_id.is_empty() {
                need_by_parent.push((item.document_id.clone(), item.parent_id.clone()));
            }
            if item.parent_index >= 0 {
                need_by_index.push((item.document_id.clone(), item.parent_index));
            }
        }

        if !need_by_parent.is_empty() {
            let placeholders = vec!["(?,?)"; need_by_parent.len()].join(",");
            let sql = format!(
                "SELECT document_id, parent_id, chunk_index, chunk_text FROM document_chunks \
                 WHERE (document_id, parent_id) IN ({placeholders})"
            );
            let mut query = sqlx::query_as::<_, ParentRow>(&sql);
            for (doc, parent) in &need_by_parent {
                query = query.bind(doc).bind(parent);
            }
            if let Ok(rows) = query.fetch_all(&self.pool).await {
                for row in rows {
                    let text = row.chunk_text.trim().to_string();
                    if !text.is_empty() {
                        result
                            .entry((row.document_id, row.parent_id))
                            .or_insert(text);
                    }
                }
            }
        }
        if !need_by_index.is_empty() {
            let placeholders = vec!["(?,?)"; need_by_index.len()].join(",");
            let sql = format!(
                "SELECT document_id, parent_id, chunk_index, chunk_text FROM document_chunks \
                 WHERE (document_id, chunk_index) IN ({placeholders})"
            );
            let mut query = sqlx::query_as::<_, ParentRow>(&sql);
            for (doc, index) in &need_by_index {
                query = query.bind(doc).bind(index);
            }
            if let Ok(rows) = query.fetch_all(&self.pool).await {
                for row in rows {
                    let text = row.chunk_text.trim().to_string();
                    if !text.is_empty() {
                        result
                            .entry((row.document_id.clone(), row.parent_id.clone()))
                            .or_insert_with(|| text.clone());
                        // chunk_index 兜底命中也要能被查到: 以 index 键补充
                        result
                            .entry((row.document_id, format!("idx:{}", row.chunk_index)))
                            .or_insert(text);
                    }
                }
            }
        }
        result
    }
}

fn key_for_parent(item: &RetrievalItem) -> (String, String) {
    (item.document_id.clone(), item.parent_id.clone())
}

impl RetrievalItem {
    fn to_fuse_item(&self) -> FuseItem<RetrievalItem> {
        FuseItem {
            document_id: self.document_id.clone(),
            chunk_key: self.chunk_index,
            chunk_sort: self.chunk_index,
            content: self.content.clone(),
            value: self.clone(),
        }
    }

    fn as_candidate(&self) -> Candidate<'_> {
        Candidate {
            content: &self.content,
            source_name: &self.source_name,
            score: self.score,
            source: &self.source,
        }
    }
}

// ---------------- 结构化聚合辅助（复刻 _parse_structured_table 等） ----------------

#[derive(Debug, Clone)]
pub struct ParentRecord {
    pub parent_id: String,
    pub parent_index: i64,
    pub parent_text: String,
}

/// 复刻 _structured_section_marker_from_text：首个非空行匹配 [Sheet] 名。
fn structured_section_marker(text: &str) -> String {
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let lower = line.to_lowercase();
        if let Some(rest) = lower.strip_prefix("[sheet]") {
            return rest.trim().to_string();
        }
        return String::new();
    }
    String::new()
}

fn parse_structured_table(
    parent_texts: &[String],
) -> (String, Vec<String>, Vec<String>) {
    let mut sheet_name = String::new();
    let mut header = String::new();
    let mut rows: Vec<String> = Vec::new();
    let mut seen_rows: HashSet<String> = HashSet::new();

    for parent_text in parent_texts {
        let lines: Vec<&str> = parent_text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        for line in lines {
            let lower = line.to_lowercase();
            if let Some(rest) = lower.strip_prefix("[sheet]") {
                let current = rest.trim().to_string();
                if sheet_name.is_empty() {
                    sheet_name = current;
                }
                continue;
            }
            if header.is_empty() {
                header = line.to_string();
                continue;
            }
            if eks_kernels::text::normalize_text(line) == eks_kernels::text::normalize_text(&header) {
                continue;
            }
            let row_key = eks_kernels::text::normalize_text(line);
            if row_key.is_empty() || seen_rows.contains(&row_key) {
                continue;
            }
            seen_rows.insert(row_key);
            rows.push(line.to_string());
        }
    }
    let headers: Vec<String> = header
        .split(',')
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect();
    (sheet_name, headers, rows)
}

fn row_matches_query(row: &str, query: &str) -> bool {
    let keywords = build_query_keywords(query, 32);
    let normalized_row = eks_kernels::text::normalize_text(row);
    let stop = ["员工", "人员", "部门", "名单", "列表", "清单", "统计", "总数", "数量"];
    let useful: Vec<&String> = keywords
        .iter()
        .filter(|k| !k.is_empty() && !stop.contains(&k.as_str()) && k.chars().count() >= 2)
        .collect();
    if useful.is_empty() {
        return false;
    }
    useful
        .iter()
        .any(|keyword| normalized_row.contains(keyword.as_str()))
}

/// 复刻 _build_structured_section_summary：同 sheet parent 压缩为可读证据。
fn build_structured_section_summary(
    anchor: &RetrievalItem,
    parent_rows: &[ParentRecord],
    query: &str,
) -> Option<RetrievalItem> {
    let section_name = structured_section_marker(&anchor.content);
    if section_name.is_empty() {
        return None;
    }
    let mut same_section: Vec<&ParentRecord> = parent_rows
        .iter()
        .filter(|row| structured_section_marker(&row.parent_text) == section_name)
        .collect();
    same_section.sort_by_key(|row| row.parent_index);
    same_section.truncate(crate::retrieval_engine::STRUCTURED_SECTION_PARENT_LIMIT);
    if same_section.is_empty() {
        return None;
    }
    let texts: Vec<String> = same_section
        .iter()
        .map(|row| row.parent_text.clone())
        .collect();
    let (sheet_name, headers, rows) = parse_structured_table(&texts);
    if rows.is_empty() {
        return None;
    }

    let matched_rows: Vec<&String> = rows.iter().filter(|row| row_matches_query(row, query)).collect();
    let matched_set: HashSet<&String> = matched_rows.iter().copied().collect();
    let mut ordered: Vec<&String> = matched_rows.clone();
    ordered.extend(rows.iter().filter(|row| !matched_set.contains(row)));

    let mut rendered: Vec<&String> = Vec::new();
    let mut used_chars = 0usize;
    for row in ordered {
        let next_len = row.chars().count() + 1;
        if used_chars + next_len > STRUCTURED_SECTION_ROW_CHAR_LIMIT {
            break;
        }
        rendered.push(row);
        used_chars += next_len;
    }
    if rendered.is_empty() {
        return None;
    }

    let content = format!(
        "[Sheet] {}\n字段: {}\n去重记录数: {}\n优先匹配行数: {}\n数据行:\n{}",
        if sheet_name.is_empty() { section_name.clone() } else { sheet_name.clone() },
        if headers.is_empty() { "未知".to_string() } else { headers.join("、") },
        rows.len(),
        matched_rows.len(),
        rendered.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n"),
    );
    let mut summary = anchor.clone();
    summary.content = content;
    summary.structured_aggregation = true;
    summary.section_name = Some(if sheet_name.is_empty() { section_name } else { sheet_name });
    summary.chunk_indices = same_section.iter().map(|row| row.parent_index).collect();
    summary.score = anchor.score.max(0.0) + 0.08;
    summary.source = "structured_evidence".to_string();
    Some(summary)
}
