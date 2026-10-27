//! BM25Okapi 的 Rust 移植，数学与 `rank_bm25==0.2.2` 逐位一致：
//! - idf(word) = ln((N - df + 0.5) / (df + 0.5))，负值地板为 epsilon * average_idf
//! - 单词贡献 = idf * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))
//! - get_scores 按查询 token 出现顺序迭代（重复 token 重复计贡献）
//! 由 parity 测试（对比 rank_bm25 输出，误差 <1e-9）钉死。

use std::collections::HashMap;

const K1: f64 = 1.5;
const B: f64 = 0.75;
const EPSILON: f64 = 0.25;

pub struct BM25Okapi {
    doc_count: usize,
    doc_len: Vec<usize>,
    avgdl: f64,
    /// 每篇文档 token -> 词频。
    doc_term_freqs: Vec<HashMap<String, u32>>,
    /// 词 -> idf（已应用负值地板）。
    idf: HashMap<String, f64>,
}

impl BM25Okapi {
    pub fn build(corpus: &[Vec<String>]) -> Self {
        let doc_count = corpus.len();
        let mut doc_len = Vec::with_capacity(doc_count);
        let mut doc_term_freqs = Vec::with_capacity(doc_count);
        let mut doc_frequency: HashMap<String, u32> = HashMap::new();
        let mut total_len = 0usize;

        for tokens in corpus {
            let mut freqs: HashMap<String, u32> = HashMap::with_capacity(tokens.len());
            for token in tokens {
                *freqs.entry(token.clone()).or_insert(0u32) += 1;
            }
            for token in freqs.keys() {
                *doc_frequency.entry(token.clone()).or_insert(0u32) += 1;
            }
            total_len += tokens.len();
            doc_len.push(tokens.len());
            doc_term_freqs.push(freqs);
        }

        let avgdl = if doc_count > 0 {
            total_len as f64 / doc_count as f64
        } else {
            0.0
        };

        // rank_bm25: average_idf = 原始 idf 均值，负 idf 替换为 epsilon * average_idf。
        let mut raw_idf: HashMap<String, f64> = HashMap::with_capacity(doc_frequency.len());
        let mut idf_sum = 0.0f64;
        for (word, df) in &doc_frequency {
            let value = ((doc_count as f64 - *df as f64 + 0.5) / (*df as f64 + 0.5)).ln();
            raw_idf.insert(word.clone(), value);
            idf_sum += value;
        }
        let average_idf = if raw_idf.is_empty() {
            0.0
        } else {
            idf_sum / raw_idf.len() as f64
        };
        let floor = EPSILON * average_idf;

        let mut idf: HashMap<String, f64> = HashMap::with_capacity(raw_idf.len());
        for (word, value) in raw_idf {
            idf.insert(word, if value < 0.0 { floor } else { value });
        }

        BM25Okapi {
            doc_count,
            doc_len,
            avgdl,
            doc_term_freqs,
            idf,
        }
    }

    /// 与 rank_bm25.BM25Okapi.get_scores 一致：返回每篇文档的 BM25 总分。
    /// 语料中未出现的查询词贡献为 0（等价 `idf.get(q) or 0`）。
    pub fn get_scores(&self, query: &[String]) -> Vec<f64> {
        let mut scores = vec![0.0f64; self.doc_count];
        if self.doc_count == 0 {
            return scores;
        }
        for doc_index in 0..self.doc_count {
            let freqs = &self.doc_term_freqs[doc_index];
            let dl = self.doc_len[doc_index] as f64;
            let norm = if self.avgdl > 0.0 {
                K1 * (1.0 - B + B * dl / self.avgdl)
            } else {
                K1 * (1.0 - B)
            };
            let mut score = 0.0f64;
            for token in query {
                let Some(idf) = self.idf.get(token) else {
                    continue;
                };
                let tf = *freqs.get(token.as_str()).unwrap_or(&0) as f64;
                if tf == 0.0 {
                    continue;
                }
                score += idf * (tf * (K1 + 1.0)) / (tf + norm);
            }
            scores[doc_index] = score;
        }
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens_of(texts: &[&str]) -> Vec<Vec<String>> {
        texts
            .iter()
            .map(|t| t.split_whitespace().map(|s| s.to_string()).collect())
            .collect()
    }

    #[test]
    fn scores_positive_and_zero() {
        let corpus = tokens_of(&["a b a", "b c", "d"]);
        let bm25 = BM25Okapi::build(&corpus);
        let query: Vec<String> = vec!["a".into(), "b".into()];
        let scores = bm25.get_scores(&query);
        assert!(scores[0] > scores[2]);
        assert!(scores[1] > 0.0);
    }

    #[test]
    fn unknown_token_contributes_zero() {
        let corpus = tokens_of(&["a b"]);
        let bm25 = BM25Okapi::build(&corpus);
        let query: Vec<String> = vec!["zzz".into()];
        assert_eq!(bm25.get_scores(&query), vec![0.0]);
    }
}
