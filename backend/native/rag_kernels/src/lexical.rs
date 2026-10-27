//! 词法覆盖打分，复刻 sparse_retriever._lexical_match_score：
//! 去重查询 token 的加权命中率 + 整句紧凑子串加分(0.35)。

use std::collections::HashSet;

use crate::text::normalize_text;

/// 与 Python 签名 `_lexical_match_score(query_tokens, row_tokens, query, row_text)` 对齐。
pub fn lexical_match_score(
    query_tokens: &[String],
    row_tokens: &[String],
    query: &str,
    row_text: &str,
) -> f64 {
    // 去重保持顺序
    let mut seen: HashSet<&str> = HashSet::with_capacity(query_tokens.len());
    let unique: Vec<&str> = query_tokens
        .iter()
        .map(|s| s.as_str())
        .filter(|t| seen.insert(t))
        .collect();
    if unique.is_empty() {
        return 0.0;
    }

    let row_token_set: HashSet<&str> = row_tokens.iter().map(|s| s.as_str()).collect();
    let mut max_weight = 0.0f64;
    let mut hit_weight = 0.0f64;
    for token in &unique {
        let len = token.chars().count();
        let weight = len.clamp(1, 8) as f64 / 8.0;
        max_weight += weight;
        if row_token_set.contains(*token) {
            hit_weight += weight;
        }
    }
    let mut score = if max_weight > 0.0 {
        hit_weight / max_weight
    } else {
        0.0
    };
    let compact_query = normalize_text(query);
    let compact_row = normalize_text(row_text);
    if !compact_query.is_empty() && compact_row.contains(&compact_query) {
        score += 0.35;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::tokenize_cjk;

    #[test]
    fn hit_and_bonus() {
        let query = "报销时限";
        let q = tokenize_cjk(query);
        let row_text = "报销时限说明";
        let r = tokenize_cjk(row_text);
        let score = lexical_match_score(&q, &r, query, row_text);
        assert!(score > 0.35); // 命中 + 整句子串加分
    }
}
