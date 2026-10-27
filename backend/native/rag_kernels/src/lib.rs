//! pyo3 暴露层：SparseIndex（常驻 BM25 索引 + 两段式检索）与调试/parity 函数。

use pyo3::prelude::*;

mod bm25;
mod lexical;
mod rerank;
mod text;

use crate::bm25::BM25Okapi;
use crate::lexical::lexical_match_score;
use crate::text::{normalize_text, tokenize_cjk};

/// 进程内常驻稀疏索引，替代 Python 侧 rank_bm25 的每次查询重建与全语料打分循环。
///
/// 构建语义与 app/rag/retrieval/sparse_index.py 一致：
/// texts 为 `source_name + "\n" + child_text` 的行文本。
/// search 返回 (行下标, 切片内 BM25 分, 词法覆盖分)，
/// 两段式：全语料分选候选（等价旧 LIKE 预筛）→ 切片内重建 BM25 重打分
///（保持 Python 修复后的子集统计语义）。
#[pyclass(module = "rag_kernels")]
pub struct SparseIndex {
    row_texts: Vec<String>,
    row_tokens: Vec<Vec<String>>,
    bm25: BM25Okapi,
}

#[pymethods]
impl SparseIndex {
    #[new]
    fn new(texts: Vec<String>) -> PyResult<Self> {
        if texts.len() > 500_000 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "corpus too large: {} rows",
                texts.len()
            )));
        }
        let row_texts = texts;
        let row_tokens: Vec<Vec<String>> =
            row_texts.iter().map(|t| tokenize_cjk(t)).collect();
        let bm25 = BM25Okapi::build(&row_tokens);
        Ok(SparseIndex {
            row_texts,
            row_tokens,
            bm25,
        })
    }

    /// 返回 [(row_index, slice_bm25_score, lexical_score)]，按全语料 BM25 降序取前 limit 个正分候选。
    /// Python 侧负责 final = max(0.0, bm25) + lexical 的合成与最终排序。
    #[pyo3(signature = (query, limit))]
    fn search(&self, query: String, limit: usize) -> Vec<(usize, f64, f64)> {
        let query_tokens = tokenize_cjk(&query);
        let global_scores = self.bm25.get_scores(&query_tokens);

        // 稳定排序保持并列时原始行序，与 Python sorted(range, key=-score) 一致。
        let mut order: Vec<usize> = (0..global_scores.len()).collect();
        order.sort_by(|&a, &b| {
            global_scores[b]
                .partial_cmp(&global_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut slice_indexes: Vec<usize> = Vec::with_capacity(limit.min(order.len()));
        for row_index in order.into_iter().take(limit.max(1)) {
            if global_scores[row_index] <= 0.0 {
                break;
            }
            slice_indexes.push(row_index);
        }
        if slice_indexes.is_empty() {
            return Vec::new();
        }

        let slice_corpus: Vec<Vec<String>> = slice_indexes
            .iter()
            .map(|&i| self.row_tokens[i].clone())
            .collect();
        let slice_bm25 = BM25Okapi::build(&slice_corpus);
        let slice_scores = slice_bm25.get_scores(&query_tokens);

        slice_indexes
            .iter()
            .enumerate()
            .map(|(position, &row_index)| {
                let lexical = lexical_match_score(
                    &query_tokens,
                    &self.row_tokens[row_index],
                    &query,
                    &self.row_texts[row_index],
                );
                (row_index, slice_scores[position], lexical)
            })
            .collect()
    }

    /// 行数（诊断用）。
    #[getter]
    fn row_count(&self) -> usize {
        self.row_texts.len()
    }
}

/// parity/诊断：CJK n-gram 分词。
#[pyfunction]
fn tokenize(text: String) -> Vec<String> {
    tokenize_cjk(&text)
}

/// parity/诊断：文本归一化。
#[pyfunction]
fn normalize(text: String) -> String {
    normalize_text(&text)
}

/// parity/诊断：BM25Okapi 打分（构建一次、打一次，慢速路径仅供测试对比）。
#[pyfunction]
fn bm25_scores(corpus: Vec<Vec<String>>, query: Vec<String>) -> Vec<f64> {
    BM25Okapi::build(&corpus).get_scores(&query)
}

/// parity/诊断：词法覆盖分（内部自行分词，与线上调用方式一致）。
#[pyfunction]
fn lexical_score(query: String, row_text: String) -> f64 {
    let q = tokenize_cjk(&query);
    let r = tokenize_cjk(&row_text);
    lexical_match_score(&q, &r, &query, &row_text)
}

#[pymodule]
fn rag_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SparseIndex>()?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(bm25_scores, m)?)?;
    m.add_function(wrap_pyfunction!(lexical_score, m)?)?;
    m.add_function(wrap_pyfunction!(rerank::rerank, m)?)?;
    m.add_function(wrap_pyfunction!(rerank::fuse_ranked, m)?)?;
    m.add_function(wrap_pyfunction!(rerank::extract_identifiers, m)?)?;
    m.add_function(wrap_pyfunction!(rerank::identifier_query, m)?)?;
    m.add_function(wrap_pyfunction!(rerank::query_keywords, m)?)?;
    m.add_function(wrap_pyfunction!(rerank::query_focus_terms, m)?)?;
    Ok(())
}
