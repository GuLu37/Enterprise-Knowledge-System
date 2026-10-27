//! pyo3 FFI 壳：纯逻辑在 eks-kernels crate（与 eks-server 共用同一源码）。
use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use eks_kernels::rerank::{
    Candidate, RelevanceSignals, build_keyword_specificity_scores, build_query_focus_terms,
    build_query_keywords, build_relevance_signals, extract_identifier_terms,
    is_identifier_query, normalize_source_score, passes_keyword_relevance_gate, score_result,
};

// ---------------- pyo3 暴露 ----------------

/// Python 侧输入: (content, source_name, score, source)。
#[derive(FromPyObject)]
pub struct PyCandidate {
    pub content: String,
    pub source_name: String,
    pub score: f64,
    pub source: String,
}

/// 复刻 rerank_results：返回 [(原始下标, 综合分, 信号字典)]，
/// pydantic 重建与 metadata 合并由 Python 侧完成。
#[pyfunction(signature = (query, items, top_k=None))]
pub fn rerank(
    py: Python<'_>,
    query: String,
    items: Vec<PyCandidate>,
    top_k: Option<usize>,
) -> PyResult<Vec<(usize, f64, PyObject)>> {
    // 过滤空内容候选但保留原始下标映射。
    let kept: Vec<(usize, PyCandidate)> = items
        .into_iter()
        .enumerate()
        .filter(|(_, item)| !item.content.trim().is_empty())
        .collect();
    if kept.is_empty() {
        return Ok(Vec::new());
    }
    let candidates: Vec<Candidate> = kept
        .iter()
        .map(|(_, item)| Candidate {
            content: item.content.as_str(),
            source_name: item.source_name.as_str(),
            score: item.score,
            source: item.source.as_str(),
        })
        .collect();

    let keywords = build_query_keywords(&query, 24);
    let focus_terms = build_query_focus_terms(&query, 8);

    let signals: Vec<RelevanceSignals> = candidates
        .iter()
        .map(|candidate| build_relevance_signals(&query, candidate, &keywords, &focus_terms))
        .collect();
    let specificity = build_keyword_specificity_scores(
        &keywords,
        &candidates
            .iter()
            .map(|c| (c.content, c.source_name))
            .collect::<Vec<_>>(),
    );

    let raw_scores: Vec<f64> = candidates
        .iter()
        .map(|candidate| normalize_source_score(candidate.score, candidate.source))
        .collect();
    let min_score = raw_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_score = raw_scores
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let span = max_score - min_score;

    // (原始下标, 综合分, 候选原始分, 信号下标)
    let mut scored: Vec<(usize, f64, f64, usize)> = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        let raw_norm = if span > 1e-9 {
            (raw_scores[index] - min_score) / span
        } else if raw_scores[index] > 0.0 {
            1.0
        } else {
            0.0
        };
        let gate = passes_keyword_relevance_gate(
            &query,
            candidate,
            &keywords,
            &focus_terms,
            &signals[index],
        );
        let score = score_result(
            candidate.source,
            raw_norm,
            specificity[index],
            &signals[index],
            gate,
        );
        scored.push((kept[index].0, score, candidate.score, index));
    }

    // Python 排序键: (-综合分, -候选原始分, 下标)
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.2.partial_cmp(&a.2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.0.cmp(&b.0))
    });

    let limit = top_k.unwrap_or(scored.len()).max(1);
    let mut output = Vec::with_capacity(limit.min(scored.len()));
    for (original_index, score, _raw, signal_index) in scored.into_iter().take(limit) {
        let signals = &signals[signal_index];
        let dict = PyDict::new(py);
        dict.set_item("_query_key", &signals.query_key)?;
        dict.set_item("query_keywords", keywords.clone())?;
        dict.set_item("query_focus_terms", focus_terms.clone())?;
        dict.set_item("keyword_hits", signals.keyword_hits.clone())?;
        dict.set_item("focus_hits", signals.focus_hits.clone())?;
        dict.set_item("keyword_hit_count", signals.keyword_hits.len() as u64)?;
        dict.set_item("focus_hit_count", signals.focus_hits.len() as u64)?;
        dict.set_item(
            "long_keyword_hit_count",
            signals.long_keyword_hit_count as u64,
        )?;
        dict.set_item("keyword_coverage", signals.keyword_coverage)?;
        dict.set_item("focus_coverage", signals.focus_coverage)?;
        dict.set_item("heading_coverage", signals.heading_coverage)?;
        dict.set_item("heading_focus_coverage", signals.heading_focus_coverage)?;
        dict.set_item("exact_match", signals.exact_match)?;
        dict.set_item("is_reference_index", signals.is_reference_index)?;
        dict.set_item("keyword_specificity", specificity[signal_index])?;
        output.push((original_index, score, dict.into_any().unbind()));
    }
    Ok(output)
}

/// Python 侧输入 item: (document_id, chunk_index|None, content, score)。
#[derive(FromPyObject)]
pub struct PyFuseItem {
    pub document_id: String,
    pub chunk_index: Option<i64>,
    pub content: String,
    pub score: f64,
}

/// 复刻 fuse_ranked_results（RRF 融合）。
/// 输出: (document_id, chunk_index(-1 缺省), content, fused_score, sources)。
#[pyfunction(signature = (result_sets, weights=None, source_labels=None, rrf_k=60))]
pub fn fuse_ranked(
    result_sets: Vec<Vec<PyFuseItem>>,
    weights: Option<Vec<f64>>,
    source_labels: Option<Vec<String>>,
    rrf_k: u64,
) -> PyResult<Vec<(String, i64, String, f64, Vec<String>)>> {
    if rrf_k == 0 {
        return Err(PyValueError::new_err("rrf_k must be positive"));
    }
    let empty_weights: Vec<f64> = Vec::new();
    let empty_labels: Vec<String> = Vec::new();
    let weights = weights.unwrap_or(empty_weights);
    let labels = source_labels.unwrap_or(empty_labels);
    let rrf_k = rrf_k as f64;

    // (doc, chunk_key(-1 缺省), content, fused_score, chunk_sort(0 缺省), sources)
    let mut fused: Vec<(String, i64, String, f64, i64, Vec<String>)> = Vec::new();
    let mut index_map: HashMap<(String, i64, String), usize> = HashMap::new();

    for (set_index, results) in result_sets.into_iter().enumerate() {
        let weight = weights.get(set_index).copied().unwrap_or(1.0);
        let source_name = labels
            .get(set_index)
            .cloned()
            .unwrap_or_else(|| format!("route_{}", set_index + 1));
        for (rank0, item) in results.into_iter().enumerate() {
            let chunk_key = item.chunk_index.unwrap_or(-1);
            let chunk_sort = item.chunk_index.unwrap_or(0);
            let content = item.content.trim().to_string();
            let key = (item.document_id.clone(), chunk_key, content.clone());
            let contribution = weight / (rrf_k + (rank0 as f64 + 1.0));
            match index_map.get(&key) {
                Some(&position) => {
                    fused[position].3 += contribution;
                    if !fused[position].5.contains(&source_name) {
                        fused[position].5.push(source_name.clone());
                    }
                }
                None => {
                    index_map.insert(key, fused.len());
                    fused.push((
                        item.document_id.clone(),
                        chunk_key,
                        content,
                        contribution,
                        chunk_sort,
                        vec![source_name.clone()],
                    ));
                }
            }
        }
    }

    // Python 排序键: (-fused_score, 首见结果的 chunk_index(缺省 0))
    let mut order: Vec<usize> = (0..fused.len()).collect();
    order.sort_by(|&a, &b| {
        fused[b]
            .3
            .partial_cmp(&fused[a].3)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| fused[a].4.cmp(&fused[b].4))
    });

    Ok(order
        .into_iter()
        .map(|position| {
            let (doc, chunk, content, score, _, sources) = &fused[position];
            (doc.clone(), *chunk, content.clone(), *score, sources.clone())
        })
        .collect())
}

/// parity/诊断入口。
#[pyfunction]
pub fn extract_identifiers(query: String) -> Vec<String> {
    extract_identifier_terms(&query)
}

#[pyfunction]
pub fn identifier_query(query: String) -> bool {
    is_identifier_query(&query)
}

#[pyfunction(signature = (query, max_terms=24))]
pub fn query_keywords(query: String, max_terms: usize) -> Vec<String> {
    build_query_keywords(&query, max_terms)
}

#[pyfunction(signature = (query, max_terms=8))]
pub fn query_focus_terms(query: String, max_terms: usize) -> Vec<String> {
    build_query_focus_terms(&query, max_terms)
}
