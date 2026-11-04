//! 词法覆盖打分与稀疏 LIKE 预筛词，复刻 sparse_retriever。

use std::collections::HashSet;

use crate::rerank::extract_identifier_terms;
use crate::text::normalize_text;

/// 对齐 `_build_filter_terms` 的口语/问句停用短语（按 Python 正则从左到右优先）。
const FILTER_STOP_PHRASES: &[&str] = &[
    "根据", "基于", "结合", "请问", "麻烦", "帮我", "帮忙", "查询", "检索", "搜索",
    "查一下", "查找", "公司", "企业", "每个", "各个", "所有", "全部", "分别", "哪些",
    "哪几", "是谁", "是什么", "在哪里", "在哪", "多少", "如何", "怎么", "为什么", "是否",
    "有没有", "可以吗", "能否", "吗", "呢", "吧", "啊",
];
const FILTER_STOP_CHARS: &[char] = &[
    '的', '之', '与', '和', '及', '在', '是', '有', '为', '从', '到', '对', '把', '将',
];

/// 与 Python 签名 `_lexical_match_score(query_tokens, row_tokens, query, row_text)` 对齐。
pub fn lexical_match_score(
    query_tokens: &[String],
    row_tokens: &[String],
    query: &str,
    row_text: &str,
) -> f64 {
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

fn is_filter_stop_char(c: char) -> bool {
    FILTER_STOP_CHARS.contains(&c)
}

fn strip_sparse_filter_stopwords(query: &str) -> String {
    let lowered = query.trim().to_lowercase();
    let chars: Vec<char> = lowered.chars().collect();
    let mut stripped = String::new();
    let mut index = 0usize;
    while index < chars.len() {
        let rest: String = chars[index..].iter().collect();
        if let Some(phrase) = FILTER_STOP_PHRASES
            .iter()
            .find(|phrase| rest.starts_with(*phrase))
        {
            stripped.push(' ');
            index += phrase.chars().count();
            continue;
        }
        stripped.push(chars[index]);
        index += 1;
    }
    stripped
        .chars()
        .map(|ch| if is_filter_stop_char(ch) { ' ' } else { ch })
        .collect()
}

fn push_semantic_parts(text: &str, out: &mut Vec<String>) {
    let mut ascii = String::new();
    let mut cjk: Vec<char> = Vec::new();
    let flush_ascii = |ascii: &mut String, out: &mut Vec<String>| {
        if !ascii.is_empty() {
            out.push(std::mem::take(ascii));
        }
    };
    let flush_cjk = |cjk: &mut Vec<char>, out: &mut Vec<String>| {
        if cjk.len() >= 2 {
            out.push(cjk.iter().collect());
        }
        cjk.clear();
    };
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            flush_cjk(&mut cjk, out);
            ascii.push(ch);
        } else if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
            flush_ascii(&mut ascii, out);
            cjk.push(ch);
        } else {
            flush_ascii(&mut ascii, out);
            flush_cjk(&mut cjk, out);
        }
    }
    flush_ascii(&mut ascii, out);
    flush_cjk(&mut cjk, out);
}

fn ngrams_skipping_stop_chars(part: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = part.chars().collect();
    if chars.len() < n {
        return Vec::new();
    }
    (0..=chars.len() - n)
        .filter_map(|start| {
            let gram = &chars[start..start + n];
            if gram.iter().any(|ch| is_filter_stop_char(*ch)) {
                None
            } else {
                Some(gram.iter().collect())
            }
        })
        .collect()
}

/// 复刻 sparse_retriever._build_filter_terms：编号 + 去停用词语义片段 + CJK 2/3-gram。
pub fn build_filter_terms(query: &str, max_terms: usize) -> Vec<String> {
    let max_terms = max_terms.max(1);
    let mut query_terms = extract_identifier_terms(query);
    let stripped = strip_sparse_filter_stopwords(query);
    let mut semantic_parts = Vec::new();
    push_semantic_parts(&stripped, &mut semantic_parts);
    for part in &semantic_parts {
        query_terms.push(part.clone());
        let char_len = part.chars().count();
        let is_cjk = part.chars().all(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch));
        if is_cjk && char_len >= 3 {
            query_terms.extend(ngrams_skipping_stop_chars(part, 2));
        }
        if is_cjk && char_len >= 4 {
            query_terms.extend(ngrams_skipping_stop_chars(part, 3));
        }
    }
    if query_terms.is_empty() {
        return Vec::new();
    }

    let mut filter_terms: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for term in &query_terms {
        let normalized = term.trim();
        if normalized.chars().count() < 2 || !seen.insert(normalized.to_string()) {
            continue;
        }
        filter_terms.push(normalized.to_string());
        if filter_terms.len() >= max_terms {
            break;
        }
    }
    if !filter_terms.is_empty() {
        return filter_terms;
    }

    let mut fallback = Vec::new();
    for term in &query_terms {
        let normalized = term.trim();
        if normalized.is_empty() || !seen.insert(normalized.to_string()) {
            continue;
        }
        fallback.push(normalized.to_string());
        if fallback.len() >= max_terms {
            break;
        }
    }
    fallback
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
        assert!(score > 0.35);
    }

    #[test]
    fn filter_terms_keep_password_bigrams() {
        let terms = build_filter_terms("账号密码", 12);
        assert!(terms.iter().any(|term| term == "账号"));
        assert!(terms.iter().any(|term| term == "密码"));
        assert!(terms.iter().any(|term| term == "账号密码"));
    }

    #[test]
    fn filter_terms_strip_question_fillers() {
        let terms = build_filter_terms("请问公司的报销时限是什么", 12);
        assert!(terms.iter().any(|term| term.contains("报销") || term == "报销时限"));
        assert!(!terms.iter().any(|term| term == "请问" || term == "公司"));
    }
}
