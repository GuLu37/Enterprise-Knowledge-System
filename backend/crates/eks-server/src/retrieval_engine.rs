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

use serde_json::{json, Value};

use crate::embedder::Embedder;
use crate::milvus_lite::MilvusLiteStore;
use crate::llm::LlmClient;
use crate::retrieval::{
    MAX_CITATION_RESULTS, MIN_FINAL_RELEVANCE_SCORE, STRUCTURED_QUERY_CITATION_RESULTS,
    RetrievalItem, SparseIndexSnapshot, canonical_identifier_terms, citation_chunk_key,
    collapse_spaces, expand_multi_query_with_llm, is_structured_lookup_query,
    legacy_like_search, rank_sparse_hits, should_use_cached_sparse, sparse_candidate_limit,
};

const MAX_MULTI_QUERY_COUNT: usize = 3;
const MIN_RAG_CANDIDATE_POOL: usize = 60;
const STRUCTURED_SECTION_PARENT_LIMIT: usize = 200;
const STRUCTURED_SECTION_ROW_CHAR_LIMIT: usize = 24000;

fn candidate_pool_limit(top_k: usize) -> usize {
    (top_k * 8).max(top_k + 20).max(MIN_RAG_CANDIDATE_POOL)
}

/// 对齐 Python `_create_retriever`：hybrid 开启且双路可用时按方式选通道；
/// 否则退化为可用单路（dense 优先）。
fn multi_query_channels(
    method: &str,
    use_dense: bool,
    use_sparse: bool,
    use_hybrid: bool,
) -> (bool, bool) {
    if use_hybrid && use_dense && use_sparse {
        return match method {
            crate::planner::RETRIEVAL_METHOD_DENSE => (true, false),
            crate::planner::RETRIEVAL_METHOD_SPARSE => (false, true),
            _ => (true, true),
        };
    }
    // 对齐 Python `dense_retriever or sparse_retriever` / `sparse_retriever or dense_retriever`。
    match method {
        crate::planner::RETRIEVAL_METHOD_DENSE => {
            if use_dense {
                (true, false)
            } else {
                (false, use_sparse)
            }
        }
        crate::planner::RETRIEVAL_METHOD_SPARSE => {
            if use_sparse {
                (false, true)
            } else {
                (use_dense, false)
            }
        }
        _ => {
            if use_dense {
                (true, false)
            } else {
                (false, use_sparse)
            }
        }
    }
}

/// 规划检索一轮的产物：召回结果、组装上下文、实际使用的 Multi-Query 变体。
pub struct PlannedRetrieval {
    pub results: Vec<RetrievalItem>,
    pub context: String,
    pub expanded_queries: Vec<String>,
}

fn cross_encoder_used(item: &RetrievalItem) -> bool {
    item.cross_encoder_score.is_some()
        || item.rerank_stage.as_deref() == Some("rule_then_cross_encoder")
}

/// 对齐 Python `filter_retrieval_results` 的 ranking_pool：
/// 结构化查询或 Cross-Encoder 已使用时取全量 reranked，否则优先关键词命中，空则回退全量。
fn ranking_pool_indexes(
    structured_lookup: bool,
    has_cross_encoder: bool,
    keyword_matched: &[usize],
    reranked_len: usize,
) -> Vec<usize> {
    if structured_lookup || has_cross_encoder || keyword_matched.is_empty() {
        (0..reranked_len).collect()
    } else {
        keyword_matched.to_vec()
    }
}

fn assemble_processed_context(processed: &[RetrievalItem]) -> String {
    let max_chars = if processed.iter().any(|item| item.structured_aggregation) {
        crate::retrieval::STRUCTURED_CONTEXT_MAX_CHARS
    } else {
        crate::retrieval::RAG_CONTEXT_MAX_CHARS
    };
    crate::retrieval::assemble_rag_context(processed, max_chars)
}

fn json_optional_str(value: &str) -> Value {
    if value.is_empty() {
        Value::Null
    } else {
        json!(value)
    }
}

/// 对齐 Python `run_rag_workflow` 的 `RAG 检索诊断` JSON 日志。
fn format_rag_diagnostic(
    query: &str,
    expanded_queries: &[String],
    recalled_count: usize,
    processed: &[RetrievalItem],
    context: &str,
) -> String {
    let best_score = processed
        .iter()
        .map(|item| item.score)
        .fold(0.0_f64, f64::max);
    let result_chunks: Vec<Value> = processed
        .iter()
        .map(|item| {
            json!({
                "source_name": json_optional_str(&item.source_name),
                "document_id": json_optional_str(&item.document_id),
                "chunk_index": item.chunk_index,
                "chunk_count": if item.chunk_indices.is_empty() {
                    1
                } else {
                    item.chunk_indices.len()
                },
                "parent_id": json_optional_str(&item.parent_id),
            })
        })
        .collect();
    let payload = json!({
        "query": query,
        "identifier_query": eks_kernels::rerank::is_identifier_query(query),
        "expanded_queries": expanded_queries,
        "recalled_count": recalled_count,
        "filtered_count": processed.len(),
        "result_chunks": result_chunks,
        "context_chars": context.chars().count(),
        "best_score": best_score,
    });
    format!("RAG 检索诊断: {payload}")
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
    pub child_collection: String,
    pub dense_weight: f64,
    pub sparse_weight: f64,
    pub use_dense: bool,
    pub use_sparse: bool,
    pub use_hybrid: bool,
    pub sparse_index_mode: String,
    pub sparse_index_max_rows: usize,
    pub embedding_dimension: usize,
    pub similarity_threshold: f64,
    pub search_top_k: i64,
    pub cross_encoder: Option<crate::cross_encoder::SharedCrossEncoder>,
    pub ce_candidate_k: usize,
    pub ce_weight: f64,
    pub rule_weight: f64,
    pub ce_threshold: f64,
    pub ce_margin: f64,
    pub ce_trigger_mode: String,
    /// 启动期 Cross-Encoder 工件加载失败的原因（透出到 runtime-status 供诊断）。
    pub cross_encoder_load_error: Option<String>,
    sparse: tokio::sync::RwLock<Option<Arc<SparseIndexSnapshot>>>,
    /// 稀疏索引 single-flight 构建锁（并发首查只建一次）。
    sparse_build: tokio::sync::RwLock<()>,
    result_cache: tokio::sync::Mutex<crate::rag_cache::RagResultCache>,
}

fn passes_similarity_threshold(score: f64, threshold: f64) -> bool {
    threshold <= 0.0 || score >= threshold
}

/// HTTP dense 默认强制阈值；聊天/规划器对齐 Python `enforce_similarity_threshold=False`。
fn filter_dense_hits(
    hits: Vec<(usize, f64)>,
    threshold: f64,
    enforce_similarity_threshold: bool,
) -> Vec<(usize, f64)> {
    if !enforce_similarity_threshold {
        return hits;
    }
    hits.into_iter()
        .filter(|(_, score)| passes_similarity_threshold(*score, threshold))
        .collect()
}

impl RetrievalEngine {
    pub fn new(store: MilvusLiteStore, embedder: Arc<Embedder>, pool: sqlx::SqlitePool) -> Self {
        RetrievalEngine {
            store,
            embedder,
            pool,
            child_collection: "doc_chunk".into(),
            dense_weight: 0.5,
            sparse_weight: 0.5,
            use_dense: true,
            use_sparse: true,
            use_hybrid: true,
            sparse_index_mode: "cached".into(),
            sparse_index_max_rows: 50_000,
            embedding_dimension: crate::milvus_lite::DEFAULT_EMBEDDING_DIMENSION,
            similarity_threshold: 0.5,
            search_top_k: 5,
            cross_encoder: None,
            ce_candidate_k: 30,
            ce_weight: 0.65,
            rule_weight: 0.35,
            ce_threshold: 0.60,
            ce_margin: 0.08,
            ce_trigger_mode: "adaptive".to_string(),
            cross_encoder_load_error: None,
            sparse: tokio::sync::RwLock::new(None),
            sparse_build: tokio::sync::RwLock::new(()),
            result_cache: tokio::sync::Mutex::new(crate::rag_cache::RagResultCache::new(true, 3600)),
        }
    }

    pub fn with_retrieval_options(mut self, config: &crate::config::Config) -> Self {
        self.child_collection = config.milvus_child_collection.clone();
        self.dense_weight = config.dense_weight;
        self.sparse_weight = config.sparse_weight;
        self.use_dense = config.use_dense_retriever;
        self.use_sparse = config.use_sparse_retriever;
        self.use_hybrid = config.use_hybrid_retriever;
        self.sparse_index_mode = config.sparse_index_mode.clone();
        self.sparse_index_max_rows = config.sparse_index_max_rows;
        self.embedding_dimension = config.embedding_dimension;
        self.similarity_threshold = config.similarity_threshold;
        self.search_top_k = config.search_top_k;
        self
    }

    fn child_collection(&self) -> &str {
        if self.child_collection.trim().is_empty() {
            "doc_chunk"
        } else {
            self.child_collection.as_str()
        }
    }

    pub fn with_result_cache(self, enabled: bool, ttl_secs: u64) -> Self {
        RetrievalEngine {
            result_cache: tokio::sync::Mutex::new(crate::rag_cache::RagResultCache::new(
                enabled, ttl_secs,
            )),
            ..self
        }
    }

    pub fn with_cross_encoder(
        mut self,
        ce: crate::cross_encoder::SharedCrossEncoder,
        candidate_k: usize,
        ce_weight: f64,
        rule_weight: f64,
        threshold: f64,
        margin: f64,
        trigger_mode: String,
    ) -> Self {
        self.cross_encoder = Some(ce);
        self.ce_candidate_k = candidate_k;
        self.ce_weight = ce_weight;
        self.rule_weight = rule_weight;
        self.ce_threshold = threshold;
        self.ce_margin = margin;
        self.ce_trigger_mode = trigger_mode;
        self
    }

    async fn ensure_cached_sparse(
        &self,
        rows: &[crate::milvus_lite::Row],
    ) -> Arc<SparseIndexSnapshot> {
        {
            let cached = self.sparse.read().await;
            if let Some(snapshot) = cached.as_ref() {
                return snapshot.clone();
            }
        }
        // single-flight：并发首查只允许一个任务构建，其余等待复用，避免重复秒级重建。
        {
            let mut builder = self.sparse_build.write().await;
            {
                let cached = self.sparse.read().await;
                if let Some(snapshot) = cached.as_ref() {
                    return snapshot.clone();
                }
            }
            let started = std::time::Instant::now();
            let built = Arc::new(SparseIndexSnapshot::build(rows));
            tracing::info!(
                "稀疏持久索引构建完成: rows={} kernel=rust elapsed={:.2}s",
                rows.len(),
                started.elapsed().as_secs_f64()
            );
            *self.sparse.write().await = Some(built.clone());
            built
        }
    }

    fn uses_cached_sparse(&self, row_count: usize) -> bool {
        should_use_cached_sparse(&self.sparse_index_mode, row_count, self.sparse_index_max_rows)
    }

    async fn sparse_hits(
        &self,
        rows: &[crate::milvus_lite::Row],
        query: &str,
        limit: usize,
    ) -> Vec<(usize, f64, f64)> {
        if self.uses_cached_sparse(rows.len()) {
            let index = self.ensure_cached_sparse(rows).await;
            return index.search(query, limit.max(1));
        }
        if self.sparse_index_mode.trim().eq_ignore_ascii_case("cached")
            && rows.len() > self.sparse_index_max_rows
        {
            tracing::warn!(
                "child 行数 {} 超过稀疏持久索引上限 {}，继续使用 LIKE 路径",
                rows.len(),
                self.sparse_index_max_rows
            );
        }
        legacy_like_search(rows, query, limit.max(1))
    }

    /// 对齐 Python `warm_build_sparse_index`：cached 模式下预构建进程内 BM25。
    pub async fn warm_sparse_index(&self) -> Result<usize, String> {
        if !self.use_sparse || !self.sparse_index_mode.trim().eq_ignore_ascii_case("cached") {
            return Ok(0);
        }
        let snapshot = self
            .store
            .collection(self.child_collection())
            .await
            .map_err(|err| err.to_string())?;
        if snapshot.rows.len() > self.sparse_index_max_rows {
            tracing::warn!(
                "child 行数 {} 超过稀疏持久索引上限 {}，继续使用 LIKE 路径",
                snapshot.rows.len(),
                self.sparse_index_max_rows
            );
            return Ok(0);
        }
        let count = snapshot.rows.len();
        let _ = self.ensure_cached_sparse(&snapshot.rows).await;
        Ok(count)
    }

    pub async fn invalidate(&self) {
        // 不硬失效集合快照：mtime 变化会触发 collection() 增量重载（只读新增
        // parquet 并与缓存行合并），保留旧快照作增量基座；删除路径已由
        // delete_* 内部自行硬失效。稀疏索引与检索结果缓存仍需重建/清空。
        self.sparse.write().await.take();
        self.result_cache.lock().await.clear();
        tracing::info!("{}", format_sparse_index_invalidated());
    }

    pub async fn delete_document_chunks(&self, document_id: &str) -> Result<bool, String> {
        self.store
            .delete_by_document_id(self.child_collection(), document_id)
            .await
    }

    /// 对齐 Python `_is_collection_ready`：集合缺失/未加载时告警并软跳过。
    async fn collection_snapshot(&self, channel: &str) -> Result<Option<Arc<crate::milvus_lite::CollectionSnapshot>>, String> {
        match self.store.collection(self.child_collection()).await {
            Ok(snapshot) => Ok(Some(snapshot)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                tracing::warn!(
                    "collection {} 尚未加载完成，跳过本次{}检索",
                    self.child_collection(),
                    channel
                );
                Ok(None)
            }
            Err(err) => Err(err.to_string()),
        }
    }

    async fn search_dense_inner(
        &self,
        query: &str,
        top_k: usize,
        enforce_similarity_threshold: bool,
    ) -> Result<Vec<RetrievalItem>, String> {
        let snapshot = match self.collection_snapshot("密集").await? {
            Some(snapshot) => snapshot,
            None => return Ok(Vec::new()),
        };
        // 嵌入是 CPU/GPU 密集同步操作，放阻塞线程池避免占住 tokio worker。
        let embedder = self.embedder.clone();
        let query_text = query.to_string();
        let vectors = tokio::task::spawn_blocking(move || {
            embedder.embed_queries(&[query_text])
        })
        .await
        .map_err(|err| err.to_string())??;
        let vector = vectors.first().ok_or_else(|| "查询向量为空".to_string())?;
        let hits = self
            .store
            .search_cosine(self.child_collection(), vector, top_k)
            .await
            .map_err(|err| err.to_string())?;
        Ok(filter_dense_hits(hits, self.similarity_threshold, enforce_similarity_threshold)
            .into_iter()
            .map(|(index, score)| RetrievalItem::from_row(&snapshot.rows[index], score, "dense"))
            .collect())
    }

    async fn search_sparse_inner(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievalItem>, String> {
        let snapshot = match self.collection_snapshot("稀疏").await? {
            Some(snapshot) => snapshot,
            None => return Ok(Vec::new()),
        };
        let hits = self
            .sparse_hits(&snapshot.rows, query, sparse_candidate_limit(top_k))
            .await;
        Ok(rank_sparse_hits(&snapshot.rows, hits, top_k)
            .into_iter()
            .map(|(row_index, score)| RetrievalItem::from_row(&snapshot.rows[row_index], score, "sparse"))
            .collect())
    }

    /// 单次 Dense 检索（HTTP /search/dense）。
    /// 对齐 Python 路由：直接实例化 DenseRetriever，不看 USE_* 开关、不回退另一路。
    pub async fn search_dense(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievalItem>, String> {
        let top_k = top_k.clamp(1, 50);
        let query = query.trim();
        if query.is_empty() {
            return Ok(Vec::new());
        }
        self.search_dense_inner(query, top_k, true).await
    }

    /// 聊天/规划器 Dense：高召回，不在召回阶段用 SIMILARITY_THRESHOLD 硬过滤。
    pub async fn search_dense_high_recall(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievalItem>, String> {
        let top_k = top_k.clamp(1, 50);
        let query = query.trim();
        if query.is_empty() {
            return Ok(Vec::new());
        }
        if self.use_dense {
            self.search_dense_inner(query, top_k, false).await
        } else if self.use_sparse {
            self.search_sparse_inner(query, top_k).await
        } else {
            Ok(Vec::new())
        }
    }

    /// 单次 Sparse（BM25）检索（HTTP /search/sparse）。
    /// 对齐 Python 路由：直接实例化 SparseRetriever，不看 USE_* 开关、不回退另一路。
    pub async fn search_sparse(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievalItem>, String> {
        let top_k = top_k.clamp(1, 50);
        let query = query.trim();
        if query.is_empty() {
            return Ok(Vec::new());
        }
        self.search_sparse_inner(query, top_k).await
    }

    /// 完整 RAG 工作流（对齐 run_rag_workflow）。HTTP 检索入口不传 LLM，聊天/ReAct 传入。
    pub async fn run_rag_workflow(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<PlannedRetrieval, String> {
        self.retrieve_planned(query, crate::planner::RETRIEVAL_METHOD_HYBRID, top_k, None)
            .await
    }

    /// 按规划器选定的检索方式执行一轮检索。对齐 Python `run_rag_workflow`：
    /// dense/sparse/hybrid 都走 Multi-Query，并回传实际扩展查询。
    pub async fn retrieve_planned(
        &self,
        query: &str,
        method: &str,
        top_k: usize,
        llm: Option<&LlmClient>,
    ) -> Result<PlannedRetrieval, String> {
        let query = query.trim();
        if query.is_empty() {
            return Ok(PlannedRetrieval {
                results: Vec::new(),
                context: String::new(),
                expanded_queries: Vec::new(),
            });
        }
        let method = crate::planner::resolve_method_for_query(query, method);
        let arguments = json!({
            "query": query,
            "top_k": top_k,
            "retrieval_method": method,
        });
        crate::trace::log_tool_call("tool", "search_knowledge_base", &arguments, &json!({}));
        let cache_key = crate::rag_cache::rag_cache_key(query, top_k, &method);
        let outcome: Result<PlannedRetrieval, String> = async {
            {
                let cache = self.result_cache.lock().await;
                if let Some((results, context, expanded_queries)) = cache.get(&cache_key) {
                    tracing::info!(query, method = method.as_str(), "检索结果缓存命中");
                    return Ok(PlannedRetrieval {
                        results,
                        context,
                        expanded_queries,
                    });
                }
            }
            let (recalled, expanded_queries) = self
                .retrieve_documents_multi_query(query, top_k, &method, llm)
                .await?;
            let recalled_count = recalled.len();
            let processed = self.post_process(query, recalled, top_k).await?;
            let context = assemble_processed_context(&processed);
            tracing::info!(
                "{}",
                format_rag_diagnostic(
                    query,
                    &expanded_queries,
                    recalled_count,
                    &processed,
                    &context
                )
            );
            self.result_cache.lock().await.put(
                cache_key,
                processed.clone(),
                context.clone(),
                expanded_queries.clone(),
            );
            Ok(PlannedRetrieval {
                results: processed,
                context,
                expanded_queries,
            })
        }
        .await;
        match &outcome {
            Ok(planned) => crate::trace::log_tool_result(
                "tool",
                "search_knowledge_base",
                &crate::trace::rag_result_summary(
                    &method,
                    planned.results.len(),
                    planned.results.is_empty(),
                ),
                &json!({}),
            ),
            Err(err) => {
                crate::trace::log_tool_error("tool", "search_knowledge_base", err, &json!({}));
            }
        }
        outcome
    }

    pub async fn retrieve_documents_multi_query(
        &self,
        query: &str,
        top_k: usize,
        method: &str,
        llm: Option<&LlmClient>,
    ) -> Result<(Vec<RetrievalItem>, Vec<String>), String> {
        let mut queries = expand_multi_query_with_llm(query, llm, MAX_MULTI_QUERY_COUNT).await;
        if !eks_kernels::rerank::is_identifier_query(query) {
            queries.truncate(MAX_MULTI_QUERY_COUNT);
        }
        if queries.is_empty() {
            return Ok((Vec::new(), queries));
        }
        let method = crate::planner::resolve_method_for_query(query, method);
        let (want_dense, want_sparse) =
            multi_query_channels(&method, self.use_dense, self.use_sparse, self.use_hybrid);
        let resolved_top_k =
            crate::config::resolve_search_top_k(top_k as i64, self.search_top_k);
        let candidate_k = candidate_pool_limit(resolved_top_k);

        let snapshot = match self.collection_snapshot("密集").await? {
            Some(snapshot) => snapshot,
            None => {
                tracing::warn!(
                    "collection {} 尚未加载完成，跳过本次稀疏检索",
                    self.child_collection()
                );
                return Ok((Vec::new(), queries));
            }
        };

        let query_vectors = if want_dense {
            // 批量嵌入放阻塞线程池，多用户并发时不占 tokio worker。
            let embedder = self.embedder.clone();
            let batch = queries.clone();
            match tokio::task::spawn_blocking(move || embedder.embed_queries(&batch)).await {
                Ok(Ok(vectors)) => vectors,
                Ok(Err(err)) => {
                    tracing::warn!("批量密集检索失败，跳过该路结果: {err}");
                    Vec::new()
                }
                Err(err) => {
                    tracing::warn!("批量密集检索任务失败，跳过该路结果: {err}");
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };
        let dense_ready = want_dense && query_vectors.len() == queries.len();
        let mut variant_sets: Vec<Vec<RetrievalItem>> = Vec::with_capacity(queries.len());
        for (position, q) in queries.iter().enumerate() {
            let mut channel_sets: Vec<Vec<FuseItem<RetrievalItem>>> = Vec::new();
            let mut channel_weights: Vec<f64> = Vec::new();
            let mut channel_labels: Vec<&str> = Vec::new();
            if dense_ready {
                match self
                    .store
                    .search_cosine(self.child_collection(), &query_vectors[position], candidate_k)
                    .await
                {
                    Ok(dense_hits) => {
                        let dense: Vec<FuseItem<RetrievalItem>> = dense_hits
                            .iter()
                            .map(|(row_index, score)| {
                                RetrievalItem::from_row(&snapshot.rows[*row_index], *score, "dense")
                                    .to_fuse_item()
                            })
                            .collect();
                        if !dense.is_empty() {
                            channel_sets.push(dense);
                            channel_weights.push(self.dense_weight);
                            channel_labels.push("dense");
                        }
                    }
                    Err(err) => {
                        tracing::warn!("密集 检索失败，跳过该路结果: {err}");
                    }
                }
            }
            if want_sparse {
                let hits = self
                    .sparse_hits(&snapshot.rows, q, sparse_candidate_limit(candidate_k))
                    .await;
                let sparse: Vec<FuseItem<RetrievalItem>> =
                    rank_sparse_hits(&snapshot.rows, hits, candidate_k)
                        .into_iter()
                        .map(|(row_index, score)| {
                            RetrievalItem::from_row(&snapshot.rows[row_index], score, "sparse")
                                .to_fuse_item()
                        })
                        .collect();
                if !sparse.is_empty() {
                    channel_sets.push(sparse);
                    channel_weights.push(self.sparse_weight);
                    channel_labels.push("sparse");
                }
            }
            if channel_sets.is_empty() {
                variant_sets.push(Vec::new());
                continue;
            }
            let fused = fuse_ranked_results(channel_sets, &channel_weights, &channel_labels, 60.0);
            variant_sets.push(
                fused
                    .into_iter()
                    .take(candidate_k)
                    .map(|out| {
                        // 对齐 Python fuse_ranked_results：融合产物 source=hybrid
                        // 并携带 matched_sources/fusion_score（进 metadata）。
                        let mut item = out.value;
                        item.source = "hybrid".into();
                        item.matched_sources = out.sources;
                        item.fusion_score = Some(out.fused_score);
                        item
                    })
                    .collect(),
            );
        }

        // 变体间融合（原始查询 1.4 / 变体 0.75）
        let nonempty_indexes: Vec<usize> = (0..variant_sets.len()).filter(|&i| !variant_sets[i].is_empty()).collect();
        if nonempty_indexes.is_empty() {
            return Ok((Vec::new(), queries));
        }
        let sets: Vec<Vec<FuseItem<RetrievalItem>>> = nonempty_indexes
            .iter()
            .map(|&i| variant_sets[i].iter().map(|item| item.to_fuse_item()).collect())
            .collect();
        let weights: Vec<f64> = nonempty_indexes.iter().map(|&i| if i == 0 { 1.4 } else { 0.75 }).collect();
        let labels: Vec<String> = nonempty_indexes.iter().map(|&i| format!("query_{}", i + 1)).collect();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        let fused = fuse_ranked_results(sets, &weights, &label_refs, 60.0);

        let mut selected: Vec<RetrievalItem> = fused
            .into_iter()
            .take(candidate_k)
            .map(|out| {
                let mut item = out.value;
                item.source = "hybrid".into();
                item.matched_sources = out.sources;
                item.fusion_score = Some(out.fused_score);
                item
            })
            .collect();
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
        Ok((selected, queries))
    }

    /// 规则重排（Rust 侧无 cross-encoder，等价 adaptive 跳过路径的打分）。
    pub fn rule_rerank_pub(&self, query: &str, items: Vec<RetrievalItem>) -> Vec<RetrievalItem> {
        self.rule_rerank(query, items)
    }

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
                    keyword_hits: signals[index].keyword_hits.clone(),
                    focus_hits: signals[index].focus_hits.clone(),
                    long_keyword_hit_count: signals[index].long_keyword_hit_count,
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
        let rows: Vec<Row> = match sqlx::query_as(
            "SELECT parent_id, document_id, chunk_index, chunk_text FROM document_chunks \
             WHERE document_id = ? ORDER BY chunk_index ASC",
        )
        .bind(document_id)
        .fetch_all(&self.pool)
        .await
        {
            Ok(rows) => rows,
            Err(err) => {
                tracing::debug!("{}", format_structured_parent_load_fail(&err.to_string()));
                return Vec::new();
            }
        };
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
        let mut requested_top_k =
            crate::config::coalesce_search_top_k(top_k as i64, self.search_top_k);
        if !identifiers.is_empty() {
            max_citation = max_citation.max(identifiers.len());
            requested_top_k = requested_top_k.max(identifiers.len() as i64);
        }
        let resolved_top_k = requested_top_k.min(max_citation as i64).max(1) as usize;

        let candidates: Vec<RetrievalItem> = results
            .into_iter()
            .filter(|item| !item.content.trim().is_empty())
            .collect();
        if candidates.is_empty() {
            return Vec::new();
        }
        let candidates_count = candidates.len();
        // Python 语义: filter 以 top_k=len(candidates) 调用 cascade —— small_pool
        // 门控因此几乎总跳过精排，只有显式低置信路径才会触发 CE。
        let cascade_limit = candidates_count;
        let reranked = self.cascade_rerank(query, candidates, cascade_limit);
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

        // 对齐 Python：结构化查询或 Cross-Encoder 已打分时，用全量 reranked
        // 做 ranking_pool，避免关键词门控把语义精排结果清掉。
        let ranking_pool: Vec<usize> = ranking_pool_indexes(
            structured_lookup,
            reranked.iter().any(cross_encoder_used),
            &keyword_matched,
            reranked.len(),
        );
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
            // 对齐 Python `_expand_parent_evidence` 去重键：
            // (document_id, child_id/chunk_index, content.strip())。
            let child = if !item.child_id.is_empty() {
                item.child_id.clone()
            } else {
                item.chunk_index.to_string()
            };
            let key = (
                item.document_id.clone(),
                child,
                item.content.trim().to_string(),
            );
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
    /// 批量 SQL 失败时逐条回退，对齐 Python `_load_parent_contexts_batch`。
    async fn load_parent_texts(
        &self,
        items: &[RetrievalItem],
    ) -> HashMap<(String, String), String> {
        match self.load_parent_texts_batch(items).await {
            Ok(result) => result,
            Err(err) => {
                tracing::debug!("{}", format_parent_batch_fail(&err.to_string()));
                let mut result = HashMap::new();
                for item in items {
                    match self.load_parent_text_single(item).await {
                        Ok(Some(text)) => {
                            insert_parent_text(&mut result, item, text);
                        }
                        Ok(None) => {}
                        Err(single_err) => {
                            tracing::debug!(
                                "{}",
                                format_parent_single_fail(&single_err.to_string())
                            );
                        }
                    }
                }
                result
            }
        }
    }

    async fn load_parent_texts_batch(
        &self,
        items: &[RetrievalItem],
    ) -> Result<HashMap<(String, String), String>, sqlx::Error> {
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
            for row in query.fetch_all(&self.pool).await? {
                record_parent_row(&mut result, row);
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
            for row in query.fetch_all(&self.pool).await? {
                record_parent_row(&mut result, row);
            }
        }
        Ok(result)
    }

    /// 对齐 Python `_load_parent_context`：parent_id 优先，chunk_index 兜底。
    async fn load_parent_text_single(
        &self,
        item: &RetrievalItem,
    ) -> Result<Option<String>, sqlx::Error> {
        if item.document_id.is_empty() {
            return Ok(None);
        }
        if !item.parent_id.is_empty() {
            let text: Option<String> = sqlx::query_scalar(
                "SELECT chunk_text FROM document_chunks WHERE document_id = ? AND parent_id = ?",
            )
            .bind(&item.document_id)
            .bind(&item.parent_id)
            .fetch_optional(&self.pool)
            .await?;
            if let Some(text) = text {
                let text = text.trim().to_string();
                if !text.is_empty() {
                    return Ok(Some(text));
                }
            }
        }
        if item.parent_index >= 0 {
            let text: Option<String> = sqlx::query_scalar(
                "SELECT chunk_text FROM document_chunks WHERE document_id = ? AND chunk_index = ?",
            )
            .bind(&item.document_id)
            .bind(item.parent_index)
            .fetch_optional(&self.pool)
            .await?;
            if let Some(text) = text {
                let text = text.trim().to_string();
                if !text.is_empty() {
                    return Ok(Some(text));
                }
            }
        }
        Ok(None)
    }
}

#[derive(sqlx::FromRow)]
struct ParentRow {
    document_id: String,
    parent_id: String,
    chunk_index: i64,
    chunk_text: String,
}

fn record_parent_row(result: &mut HashMap<(String, String), String>, row: ParentRow) {
    let text = row.chunk_text.trim().to_string();
    if text.is_empty() {
        return;
    }
    result
        .entry((row.document_id.clone(), row.parent_id.clone()))
        .or_insert_with(|| text.clone());
    result
        .entry((row.document_id, format!("idx:{}", row.chunk_index)))
        .or_insert(text);
}

fn insert_parent_text(
    result: &mut HashMap<(String, String), String>,
    item: &RetrievalItem,
    text: String,
) {
    if !item.parent_id.is_empty() {
        result
            .entry((item.document_id.clone(), item.parent_id.clone()))
            .or_insert_with(|| text.clone());
    }
    if item.parent_index >= 0 {
        result
            .entry((item.document_id.clone(), format!("idx:{}", item.parent_index)))
            .or_insert(text);
    } else if item.parent_id.is_empty() {
        result.insert((item.document_id.clone(), String::new()), text);
    }
}

fn format_sparse_index_invalidated() -> &'static str {
    "稀疏持久索引已失效，将在下次检索时重建"
}

fn format_parent_batch_fail(err: &str) -> String {
    format!("SQLite Parent 上下文批量读取失败，逐条回退: {err}")
}

fn format_parent_single_fail(err: &str) -> String {
    format!("SQLite Parent 上下文读取失败，继续使用原始命中: {err}")
}

fn format_structured_parent_load_fail(err: &str) -> String {
    format!("结构化 SQLite Parent 扩展读取失败，继续使用原始命中: {err}")
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
        return eks_kernels::structured_sheet_name(line)
            .unwrap_or("")
            .to_string();
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
            if let Some(current) = eks_kernels::structured_sheet_name(line) {
                if sheet_name.is_empty() {
                    sheet_name = current.to_string();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_threshold_keeps_scores_at_or_above_cutoff() {
        assert!(passes_similarity_threshold(0.5, 0.5));
        assert!(passes_similarity_threshold(0.9, 0.5));
        assert!(!passes_similarity_threshold(0.49, 0.5));
        assert!(passes_similarity_threshold(0.01, 0.0));
        assert!(passes_similarity_threshold(-1.0, 0.0));
    }

    #[test]
    fn chat_dense_skips_similarity_threshold_like_python() {
        let hits = vec![(0, 0.49), (1, 0.9)];
        let http = filter_dense_hits(hits.clone(), 0.5, true);
        assert_eq!(http, vec![(1, 0.9)]);
        let chat = filter_dense_hits(hits, 0.5, false);
        assert_eq!(chat, vec![(0, 0.49), (1, 0.9)]);
    }

    #[test]
    fn multi_query_channels_follow_planned_method() {
        assert_eq!(multi_query_channels("dense", true, true, true), (true, false));
        assert_eq!(multi_query_channels("sparse", true, true, true), (false, true));
        assert_eq!(multi_query_channels("hybrid", true, true, true), (true, true));
        assert_eq!(
            multi_query_channels("dense", false, true, true),
            (false, true),
            "无 dense 时回退 sparse，对齐 _create_retriever"
        );
        assert_eq!(
            multi_query_channels("sparse", true, false, true),
            (true, false),
            "无 sparse 时回退 dense，对齐 _create_retriever"
        );
        assert_eq!(multi_query_channels("hybrid", true, false, true), (true, false));
        // USE_HYBRID_RETRIEVER=false：对齐 Python 退化为可用单路（dense 优先）。
        assert_eq!(multi_query_channels("hybrid", true, true, false), (true, false));
        assert_eq!(multi_query_channels("hybrid", false, true, false), (false, true));
        assert_eq!(multi_query_channels("hybrid", false, false, false), (false, false));
    }

    #[test]
    fn ranking_pool_keeps_full_reranked_after_cross_encoder() {
        let keyword_matched = vec![0usize];
        assert_eq!(
            ranking_pool_indexes(false, true, &keyword_matched, 3),
            vec![0, 1, 2],
            "CE 已打分时不能只保留关键词命中"
        );
        assert_eq!(ranking_pool_indexes(false, false, &keyword_matched, 3), vec![0]);
        assert_eq!(
            ranking_pool_indexes(true, false, &keyword_matched, 3),
            vec![0, 1, 2]
        );
        assert_eq!(ranking_pool_indexes(false, false, &[], 2), vec![0, 1]);
    }

    #[test]
    fn cross_encoder_used_detects_score_or_stage() {
        let mut item = RetrievalItem {
            content: "报销标准".into(),
            score: 0.8,
            source: "dense".into(),
            document_id: "d1".into(),
            source_name: "手册.md".into(),
            chunk_index: 0,
            child_id: "d1:c000000".into(),
            parent_id: "d1:p000000".into(),
            parent_index: 0,
            file_type: "md".into(),
            content_type: "text/markdown".into(),
            signals: None,
            parent_context_text: None,
            rerank_stage: None,
            cross_encoder_score: None,
            rerank_trace: crate::retrieval::RerankTrace::default(),
            structured_aggregation: false,
            section_name: None,
            chunk_indices: Vec::new(),
            matched_sources: Vec::new(),
            fusion_score: None,
        };
        assert!(!cross_encoder_used(&item));
        item.rerank_stage = Some("rule_only_high_confidence".into());
        assert!(!cross_encoder_used(&item));
        item.rerank_stage = Some("rule_then_cross_encoder".into());
        assert!(cross_encoder_used(&item));
        item.rerank_stage = None;
        item.cross_encoder_score = Some(0.91);
        assert!(cross_encoder_used(&item));
    }

    #[test]
    fn rag_diagnostic_log_matches_python_shape() {
        let item = RetrievalItem {
            content: "报销标准".into(),
            score: 0.8,
            source: "dense".into(),
            document_id: "d1".into(),
            source_name: "手册.md".into(),
            chunk_index: 0,
            child_id: "d1:c000000".into(),
            parent_id: "d1:p000000".into(),
            parent_index: 0,
            file_type: "md".into(),
            content_type: "text/markdown".into(),
            signals: None,
            parent_context_text: None,
            rerank_stage: None,
            cross_encoder_score: None,
                rerank_trace: crate::retrieval::RerankTrace::default(),
                structured_aggregation: false,
                section_name: None,
                chunk_indices: Vec::new(),
                matched_sources: Vec::new(),
                fusion_score: None,
            };
            let line = format_rag_diagnostic("差旅报销", &["差旅报销".into()], 4, &[item], "上下文");
        assert!(line.starts_with("RAG 检索诊断: "));
        let payload: Value =
            serde_json::from_str(line.strip_prefix("RAG 检索诊断: ").unwrap()).unwrap();
        assert_eq!(payload["query"], "差旅报销");
        assert_eq!(payload["identifier_query"], false);
        assert_eq!(payload["recalled_count"], 4);
        assert_eq!(payload["filtered_count"], 1);
        assert_eq!(payload["result_chunks"][0]["chunk_count"], 1);
        assert_eq!(payload["result_chunks"][0]["source_name"], "手册.md");
        assert_eq!(payload["context_chars"], 3);
        assert_eq!(payload["best_score"], 0.8);
    }

    #[test]
    fn sparse_invalidate_log_matches_python() {
        assert_eq!(
            format_sparse_index_invalidated(),
            "稀疏持久索引已失效，将在下次检索时重建"
        );
    }

    #[test]
    fn parent_context_fallback_logs_match_python() {
        assert_eq!(
            format_parent_batch_fail("busy"),
            "SQLite Parent 上下文批量读取失败，逐条回退: busy"
        );
        assert_eq!(
            format_parent_single_fail("locked"),
            "SQLite Parent 上下文读取失败，继续使用原始命中: locked"
        );
        assert_eq!(
            format_structured_parent_load_fail("no such table"),
            "结构化 SQLite Parent 扩展读取失败，继续使用原始命中: no such table"
        );
    }
}
