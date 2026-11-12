//! BM25Okapi 的 Rust 移植，数学与 `rank_bm25==0.2.2` 逐位一致：
//! - idf(word) = ln((N - df + 0.5) / (df + 0.5))，负值地板为 epsilon * average_idf
//! - 单词贡献 = idf * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))
//! - get_scores 按查询 token 出现顺序迭代（重复 token 重复计贡献）
//! 由 parity 测试（对比 rank_bm25 输出，误差 <1e-9）钉死。
//!
//! 内部用倒排表打分：只扫查询词命中的文档，分数与逐文档 HashMap 扫描逐位相同。

use std::collections::HashMap;

const K1: f64 = 1.5;
const B: f64 = 0.75;
const EPSILON: f64 = 0.25;

pub struct BM25Okapi {
    doc_count: usize,
    /// 每篇文档的 BM25 长度归一化分母：k1 * (1 - b + b * dl / avgdl)。
    norms: Vec<f64>,
    /// 词 -> 词元 id。
    term_index: HashMap<String, u32>,
    /// 词元 id -> 已应用负值地板的 idf。
    idf: Vec<f64>,
    /// 词元 id -> [(doc_id, tf)]。
    postings: Vec<Vec<(u32, u32)>>,
}

impl BM25Okapi {
    pub fn build(corpus: &[Vec<String>]) -> Self {
        let slices: Vec<&[String]> = corpus.iter().map(|doc| doc.as_slice()).collect();
        Self::build_from_slices(&slices)
    }

    /// 与 [`Self::build`] 同语义，避免切片重建时 clone 词元向量。
    pub fn build_from_slices(corpus: &[&[String]]) -> Self {
        let doc_count = corpus.len();
        let mut doc_len = Vec::with_capacity(doc_count);
        let mut total_len = 0usize;
        let mut term_index: HashMap<String, u32> = HashMap::new();
        let mut df: Vec<u32> = Vec::new();
        let mut postings: Vec<Vec<(u32, u32)>> = Vec::new();

        for (doc_id, tokens) in corpus.iter().enumerate() {
            let mut freqs: HashMap<&str, u32> = HashMap::with_capacity(tokens.len());
            for token in *tokens {
                *freqs.entry(token.as_str()).or_insert(0u32) += 1;
            }
            for (token, tf) in freqs {
                let id = if let Some(&id) = term_index.get(token) {
                    id
                } else {
                    let id = term_index.len() as u32;
                    term_index.insert(token.to_string(), id);
                    df.push(0);
                    postings.push(Vec::new());
                    id
                };
                df[id as usize] += 1;
                postings[id as usize].push((doc_id as u32, tf));
            }
            total_len += tokens.len();
            doc_len.push(tokens.len());
        }

        let avgdl = if doc_count > 0 {
            total_len as f64 / doc_count as f64
        } else {
            0.0
        };

        // rank_bm25: average_idf = 原始 idf 均值，负 idf 替换为 epsilon * average_idf。
        let mut raw_idf = Vec::with_capacity(df.len());
        let mut idf_sum = 0.0f64;
        for &doc_freq in &df {
            let value = ((doc_count as f64 - doc_freq as f64 + 0.5) / (doc_freq as f64 + 0.5)).ln();
            raw_idf.push(value);
            idf_sum += value;
        }
        let average_idf = if raw_idf.is_empty() {
            0.0
        } else {
            idf_sum / raw_idf.len() as f64
        };
        let floor = EPSILON * average_idf;
        let idf = raw_idf
            .into_iter()
            .map(|value| if value < 0.0 { floor } else { value })
            .collect();

        let mut norms = Vec::with_capacity(doc_count);
        for &length in &doc_len {
            let dl = length as f64;
            norms.push(if avgdl > 0.0 {
                K1 * (1.0 - B + B * dl / avgdl)
            } else {
                K1 * (1.0 - B)
            });
        }

        BM25Okapi {
            doc_count,
            norms,
            term_index,
            idf,
            postings,
        }
    }

    /// 与 rank_bm25.BM25Okapi.get_scores 一致：返回每篇文档的 BM25 总分。
    /// 语料中未出现的查询词贡献为 0（等价 `idf.get(q) or 0`）。
    pub fn get_scores(&self, query: &[String]) -> Vec<f64> {
        let mut scores = vec![0.0f64; self.doc_count];
        if self.doc_count == 0 {
            return scores;
        }
        for token in query {
            let Some(&term_id) = self.term_index.get(token.as_str()) else {
                continue;
            };
            let idf = self.idf[term_id as usize];
            for &(doc_index, tf) in &self.postings[term_id as usize] {
                let tf = tf as f64;
                let doc_index = doc_index as usize;
                scores[doc_index] += idf * (tf * (K1 + 1.0)) / (tf + self.norms[doc_index]);
            }
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

    fn naive_scores(corpus: &[Vec<String>], query: &[String]) -> Vec<f64> {
        let doc_count = corpus.len();
        let mut doc_len = Vec::with_capacity(doc_count);
        let mut doc_term_freqs = Vec::with_capacity(doc_count);
        let mut doc_frequency: HashMap<String, u32> = HashMap::new();
        let mut total_len = 0usize;
        for tokens in corpus {
            let mut freqs: HashMap<String, u32> = HashMap::new();
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
        let mut raw_idf: HashMap<String, f64> = HashMap::new();
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
        let mut idf: HashMap<String, f64> = HashMap::new();
        for (word, value) in raw_idf {
            idf.insert(word, if value < 0.0 { floor } else { value });
        }
        let mut scores = vec![0.0f64; doc_count];
        for doc_index in 0..doc_count {
            let freqs = &doc_term_freqs[doc_index];
            let dl = doc_len[doc_index] as f64;
            let norm = if avgdl > 0.0 {
                K1 * (1.0 - B + B * dl / avgdl)
            } else {
                K1 * (1.0 - B)
            };
            let mut score = 0.0f64;
            for token in query {
                let Some(idf) = idf.get(token) else {
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

    #[test]
    fn inverted_matches_per_document_hashmap_including_repeated_query_tokens() {
        let corpus = tokens_of(&["a b a c", "b c d", "d e", "a a a", ""]);
        let query: Vec<String> = vec!["a".into(), "b".into(), "a".into(), "zzz".into()];
        let bm25 = BM25Okapi::build(&corpus);
        let slices: Vec<&[String]> = corpus.iter().map(|doc| doc.as_slice()).collect();
        let from_slices = BM25Okapi::build_from_slices(&slices);
        let expected = naive_scores(&corpus, &query);
        let actual = bm25.get_scores(&query);
        let sliced = from_slices.get_scores(&query);
        assert_eq!(actual.len(), expected.len());
        for i in 0..expected.len() {
            assert!(
                (actual[i] - expected[i]).abs() < 1e-12,
                "doc {i}: inverted {} vs hashmap {}",
                actual[i],
                expected[i]
            );
            assert!((sliced[i] - expected[i]).abs() < 1e-12);
        }
    }
}
