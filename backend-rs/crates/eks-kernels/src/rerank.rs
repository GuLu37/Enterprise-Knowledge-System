//! 规则重排内核：`app/rag/retrieval/reranker.py` 纯计算部分的逐位移植。
//!
//! 覆盖: 标识符提取、查询关键词/焦点词构建、相关性信号、关键词特异度、
//! 门控判定与综合打分（RRF 融合的纯逻辑在 FFI/服务层）。
//! 所有常量词表与 Python 侧一致，由 Python 侧 parity 测试对拍钉死。

use std::collections::HashMap;

use crate::text::normalize_text;

const QUESTION_FILLER_CHARS: &str = "的吗呢吧么嘛是否有没有请帮想要了解查询检索一下";

const QUERY_STOP_TERMS: [&str; 10] = [
    "公司", "企业", "知识库", "文档", "资料", "里面", "相关", "标准吗", "有没有", "是否有",
];

const QUERY_INTENT_TERMS: [&str; 26] = [
    "完整名单", "全体名单", "名单列表", "清单列表", "分别是什么", "有哪些", "有谁", "是谁",
    "多少个", "多少名", "一共有多少", "共有多少", "总人数", "总数", "数量", "名单", "列表",
    "清单", "明细", "构成", "分别", "全部", "所有", "全体", "各个", "各",
];

const DOMAIN_TERMS: [&str; 46] = [
    "办公地点", "直属主管", "负责人", "员工", "部门", "岗位", "职位", "工号", "姓名", "状态",
    "学历", "邮箱", "城市交通", "市内出行", "出租车", "网约车", "火车站", "行程单", "高铁",
    "飞机", "住宿", "餐补", "报销", "审批", "流程", "标准", "费用", "凭证", "发票", "材料",
    "差旅", "采购", "合同", "付款", "验收", "上线", "发布", "灰度", "回滚", "账号", "登录",
    "密码", "权限", "项目", "文档", "资料",
];

const GENERIC_FOCUS_TERMS: [&str; 16] = [
    "公司", "企业", "城市", "知识库", "文档", "资料", "报销", "审批", "流程", "标准", "费用",
    "制度", "规范", "政策", "手册", "相关",
];

const REFERENCE_INDEX_MARKERS: [&str; 5] =
    ["查询速查", "优先命中表", "检索提示", "回答参考", "常见问题"];

const IDENTIFIER_HINT_TERMS: [&str; 16] = [
    "员工编号", "员工号", "工号", "人员编号", "用户编号", "产品编号", "产品号", "商品编号",
    "物料编号", "订单编号", "合同编号", "项目编号", "编号", "编码", "型号", "id",
];

fn is_filler(c: char) -> bool {
    QUESTION_FILLER_CHARS.contains(c)
}

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

// ---------------- 标识符提取（等价 _IDENTIFIER_PATTERN 的手写匹配器） ----------------

/// 交替式 `(?<![A-Za-z0-9])(?:[A-Za-z]{1,12}[-_ ]?\d{2,}[A-Za-z0-9_-]*|\d{4,}[A-Za-z0-9_-]*)(?![A-Za-z0-9])`
/// 的非重叠匹配。尾类贪婪保证前瞻恒成立；唯一回溯点是可选分隔符。
fn identifier_match_spans(text: &[char]) -> Vec<(usize, usize)> {
    let n = text.len();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < n {
        let boundary_ok = i == 0 || !text[i - 1].is_ascii_alphanumeric();
        if boundary_ok {
            if let Some(end) = try_identifier_alt1(text, i) {
                spans.push((i, end));
                i = end;
                continue;
            }
            if let Some(end) = try_identifier_alt2(text, i) {
                spans.push((i, end));
                i = end;
                continue;
            }
        }
        i += 1;
    }
    spans
}

fn is_tail_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '-' || c == '_'
}

fn try_identifier_alt1(text: &[char], start: usize) -> Option<usize> {
    let n = text.len();
    let mut j = start;
    let mut letters = 0;
    while letters < 12 && j < n && text[j].is_ascii_alphabetic() {
        j += 1;
        letters += 1;
    }
    if letters == 0 {
        return None;
    }
    // 正则回溯顺序: 先尝试消费一个 [-_ ] 分隔符，失败再试空分隔符。
    for consume_sep in [true, false] {
        let mut k = j;
        if consume_sep {
            if k < n && matches!(text[k], '-' | '_' | ' ') {
                k += 1;
            } else {
                continue;
            }
        }
        let digit_start = k;
        while k < n && text[k].is_ascii_digit() {
            k += 1;
        }
        if k - digit_start < 2 {
            continue;
        }
        while k < n && is_tail_char(text[k]) {
            k += 1;
        }
        return Some(k);
    }
    None
}

fn try_identifier_alt2(text: &[char], start: usize) -> Option<usize> {
    let n = text.len();
    if !text[start].is_ascii_digit() {
        return None;
    }
    let mut k = start;
    while k < n && text[k].is_ascii_digit() {
        k += 1;
    }
    if k - start < 4 {
        return None;
    }
    while k < n && is_tail_char(text[k]) {
        k += 1;
    }
    Some(k)
}

/// 复刻 extract_identifier_terms: compact/plain 变体 × 大小写变体去重追加。
pub fn extract_identifier_terms(query: &str) -> Vec<String> {
    let raw = query.trim();
    if raw.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = raw.chars().collect();
    let mut identifiers: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (start, end) in identifier_match_spans(&chars) {
        let matched: String = chars[start..end].iter().collect();
        let compact: String = matched.chars().filter(|c| !c.is_whitespace()).collect();
        let plain: String = matched
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
            .collect();
        for base in [compact, plain] {
            if base.is_empty() {
                continue;
            }
            for value in [base.clone(), base.to_uppercase(), base.to_lowercase()] {
                if !value.is_empty() && seen.insert(value.clone()) {
                    identifiers.push(value);
                }
            }
        }
    }
    identifiers
}

/// 复刻 is_identifier_query。
pub fn is_identifier_query(query: &str) -> bool {
    let normalized = query.trim();
    if normalized.is_empty() {
        return false;
    }
    let lowered = normalized.to_lowercase();
    if !extract_identifier_terms(normalized).is_empty() {
        return true;
    }
    if IDENTIFIER_HINT_TERMS
        .iter()
        .any(|term| *term != "id" && lowered.contains(*term))
    {
        return true;
    }
    // `(?<![A-Za-z0-9])id(?![A-Za-z0-9])`
    let chars: Vec<char> = lowered.chars().collect();
    for i in 0..chars.len().saturating_sub(1) {
        if chars[i] == 'i' && chars[i + 1] == 'd' {
            let prev_ok = i == 0 || !chars[i - 1].is_ascii_alphanumeric();
            let next = i + 2;
            let next_ok = next >= chars.len() || !chars[next].is_ascii_alphanumeric();
            if prev_ok && next_ok {
                return true;
            }
        }
    }
    false
}

// ---------------- 查询词构建 ----------------

/// 复刻 _normalize_query_for_keywords（替换顺序与 Python 完全一致）。
fn normalize_query_for_keywords(query: &str) -> String {
    let mut normalized = normalize_text(query);
    if normalized.is_empty() {
        return String::new();
    }
    let replacements: [(&str, &str); 17] = [
        ("根据知识库", ""),
        ("基于知识库", ""),
        ("结合知识库", ""),
        ("企业知识库", ""),
        ("公司资料", ""),
        ("上传文档", ""),
        ("查一下", ""),
        ("查询", ""),
        ("检索", ""),
        ("搜索", ""),
        ("请问", ""),
        ("帮我", ""),
        ("有没有", ""),
        ("是否有", ""),
        ("是否", ""),
        ("有什么", "什么"),
        ("有标准", "标准"),
    ];
    for (old, new) in replacements {
        normalized = normalized.replace(old, new);
    }
    let suffixes: [&str; 9] = [
        "是什么", "有哪些", "如何", "怎么", "为什么", "可以吗", "能否", "请说明", "请解释",
    ];
    for suffix in suffixes {
        if let Some(stripped) = normalized.strip_suffix(suffix) {
            normalized = stripped.to_string();
            break;
        }
    }
    let mut stop_sorted: Vec<&str> = QUERY_STOP_TERMS.to_vec();
    stop_sorted.sort_by_key(|t| std::cmp::Reverse(t.chars().count()));
    for term in stop_sorted {
        normalized = normalized.replace(term, "");
    }
    let mut intent_sorted: Vec<&str> = QUERY_INTENT_TERMS.to_vec();
    intent_sorted.sort_by_key(|t| std::cmp::Reverse(t.chars().count()));
    for term in intent_sorted {
        normalized = normalized.replace(term, "");
    }
    normalized = normalized.replace("有标准", "标准");
    normalized
        .trim_matches(|c: char| "的吗呢吧么嘛请帮想要了解一下".contains(c))
        .to_string()
}

/// 复刻 build_query_terms: `[A-Za-z0-9]+|[一-鿿]{2,}` 切分 + 长词 n-gram 展开 + 稳定去重。
fn build_query_terms(query: &str, max_terms: usize) -> Vec<String> {
    let lowered = query.trim().to_lowercase();
    if lowered.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = lowered.chars().collect();
    let mut parts: Vec<(Vec<char>, bool)> = Vec::new(); // (run, is_cjk)
    let mut run: Vec<char> = Vec::new();
    let mut run_cjk = false;
    for &c in &chars {
        if c.is_ascii_alphanumeric() {
            if run_cjk && !run.is_empty() {
                parts.push((std::mem::take(&mut run), run_cjk));
            }
            run_cjk = false;
            run.push(c);
        } else if is_cjk(c) {
            if !run_cjk && !run.is_empty() {
                parts.push((std::mem::take(&mut run), run_cjk));
            }
            run_cjk = true;
            run.push(c);
        } else if !run.is_empty() {
            parts.push((std::mem::take(&mut run), run_cjk));
        }
    }
    if !run.is_empty() {
        parts.push((run, run_cjk));
    }

    let mut terms: Vec<String> = extract_identifier_terms(query);
    for (part, part_cjk) in parts {
        let len = part.len();
        let text: String = part.iter().collect();
        if part_cjk && len >= 2 {
            if len >= 4 {
                terms.push(text.clone());
                for start in 0..len - 1 {
                    terms.push(part[start..start + 2].iter().collect());
                }
                if len >= 6 {
                    for start in 0..len - 2 {
                        terms.push(part[start..start + 3].iter().collect());
                    }
                }
            } else {
                terms.push(text);
            }
        } else if !part_cjk {
            terms.push(text);
        }
        // 单个 CJK 字符被 `{2,}` 量词丢弃，与 Python 一致。
    }

    // Python `sorted(terms, key=len, reverse=True)` 是稳定排序：长度并列保持原序。
    let mut indexed: Vec<usize> = (0..terms.len()).collect();
    indexed.sort_by(|&a, &b| {
        terms[b]
            .chars()
            .count()
            .cmp(&terms[a].chars().count())
            .then_with(|| a.cmp(&b))
    });
    let mut deduped: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for index in indexed {
        let term = terms[index].trim().to_string();
        if term.is_empty() || seen.contains(&term) {
            continue;
        }
        seen.insert(term.clone());
        deduped.push(term);
        if deduped.len() >= max_terms {
            break;
        }
    }
    deduped
}

fn append_unique(
    items: &mut Vec<String>,
    seen: &mut std::collections::HashSet<String>,
    value: &str,
    max_terms: usize,
) -> bool {
    let normalized = normalize_text(value);
    if normalized.chars().count() < 2 || seen.contains(&normalized) {
        return items.len() >= max_terms;
    }
    seen.insert(normalized.clone());
    items.push(normalized);
    items.len() >= max_terms
}

/// 复刻 build_query_keywords。
pub fn build_query_keywords(query: &str, max_terms: usize) -> Vec<String> {
    let normalized_query = normalize_query_for_keywords(query);
    let normalized_chars: Vec<char> = normalized_query.chars().collect();
    let mut keywords: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for term in DOMAIN_TERMS {
        if normalized_query.contains(term)
            && append_unique(&mut keywords, &mut seen, term, max_terms)
        {
            return keywords;
        }
    }

    let n = normalized_chars.len();
    if (2..=16).contains(&n) {
        append_unique(&mut keywords, &mut seen, &normalized_query, max_terms);
    }

    for size in [4usize, 3, 2] {
        if n < size {
            continue;
        }
        for start in 0..=n - size {
            let term: String = normalized_chars[start..start + size].iter().collect();
            if term.chars().any(is_filler) {
                continue;
            }
            if append_unique(&mut keywords, &mut seen, &term, max_terms) {
                // Python 在 _append_unique 返回 True 时直接 return 整个函数。
                return keywords;
            }
        }
    }

    let raw_terms = build_query_terms(&normalized_query, max_terms * 4);
    for term in raw_terms {
        let nt = normalize_text(&term);
        if nt.chars().count() < 2 {
            continue;
        }
        if QUERY_STOP_TERMS.contains(&nt.as_str()) {
            continue;
        }
        if !nt.is_empty() && nt.chars().all(is_cjk) {
            if nt.chars().all(is_filler) {
                continue;
            }
            if nt.chars().count() >= 3 && nt.chars().any(is_filler) {
                continue;
            }
        }
        if seen.contains(&nt) {
            continue;
        }
        seen.insert(nt.clone());
        keywords.push(nt);
        if keywords.len() >= max_terms {
            break;
        }
    }
    keywords
}

/// 复刻 build_query_focus_terms。
pub fn build_query_focus_terms(query: &str, max_terms: usize) -> Vec<String> {
    let normalized_query = normalize_query_for_keywords(query);
    let query_chars: Vec<char> = normalized_query.chars().collect();
    let mut focus_terms: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // (出现位置, 词) 按 (位置升序, 词长降序) 排序。Python 的 find 是码点下标，
    // 因此这里全程在码点域运算。
    let mut matched: Vec<(usize, &str)> = DOMAIN_TERMS
        .iter()
        .filter_map(|term| find_char_index(&query_chars, term).map(|pos| (pos, *term)))
        .collect();
    matched.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(b.1.chars().count().cmp(&a.1.chars().count()))
    });

    for start_index in 0..matched.len() {
        let start_position = matched[start_index].0;
        let mut end_position = start_position;
        for (next_position, next_term) in matched.iter().skip(start_index) {
            if *next_position > end_position + 1 {
                break;
            }
            end_position = end_position.max(next_position + next_term.chars().count());
        }
        let phrase: String = query_chars[start_position..end_position].iter().collect();
        if phrase.chars().count() >= 4 {
            if append_unique(&mut focus_terms, &mut seen, &phrase, max_terms) {
                return focus_terms;
            }
            if focus_terms.len() >= max_terms {
                return focus_terms;
            }
        }
    }

    for term in DOMAIN_TERMS {
        if GENERIC_FOCUS_TERMS.contains(&term) || !normalized_query.contains(term) {
            continue;
        }
        if append_unique(&mut focus_terms, &mut seen, term, max_terms) {
            return focus_terms;
        }
        if term == "城市交通"
            && append_unique(&mut focus_terms, &mut seen, "交通", max_terms)
        {
            return focus_terms;
        }
        if term == "市内出行"
            && append_unique(&mut focus_terms, &mut seen, "出行", max_terms)
        {
            return focus_terms;
        }
    }
    if !focus_terms.is_empty() {
        return focus_terms;
    }

    // Python 兜底: keywords = build_query_keywords(query, max_terms=32)
    let keywords = build_query_keywords(query, 32);
    for term in keywords {
        if GENERIC_FOCUS_TERMS.contains(&term.as_str()) {
            continue;
        }
        if term.chars().count() > 2
            && !DOMAIN_TERMS.contains(&term.as_str())
            && term != normalized_query
        {
            continue;
        }
        if term.chars().count() < 2 {
            continue;
        }
        if seen.contains(&term) {
            continue;
        }
        seen.insert(term.clone());
        focus_terms.push(term);
        if focus_terms.len() >= max_terms {
            break;
        }
    }
    focus_terms
}

/// 码点域子串查找（等价 Python str.find 返回码点下标）。
fn find_char_index(haystack: &[char], needle: &str) -> Option<usize> {
    let needle_chars: Vec<char> = needle.chars().collect();
    if needle_chars.is_empty() || haystack.len() < needle_chars.len() {
        return None;
    }
    'outer: for start in 0..=haystack.len() - needle_chars.len() {
        for (offset, nc) in needle_chars.iter().enumerate() {
            if haystack[start + offset] != *nc {
                continue 'outer;
            }
        }
        return Some(start);
    }
    None
}

// ---------------- 覆盖度 / 信号 / 特异度 / 门控 ----------------

/// 复刻 _match_coverage_score。
fn match_coverage_score(query_terms: &[String], content: &str) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let normalized_content = normalize_text(content);
    if normalized_content.is_empty() {
        return 0.0;
    }
    let mut hit_score = 0.0f64;
    let mut max_score = 0.0f64;
    let joined: String = query_terms.iter().take(3).cloned().collect::<Vec<_>>().join("");
    let normalized_query = normalize_text(&joined);
    if !normalized_query.is_empty() && normalized_content.contains(&normalized_query) {
        hit_score += 1.0;
    }
    for term in query_terms {
        let normalized_term = normalize_text(term);
        if normalized_term.is_empty() {
            continue;
        }
        let weight = (normalized_term.chars().count().clamp(1, 12)) as f64 / 12.0;
        max_score += weight;
        if normalized_content.contains(&normalized_term) {
            hit_score += weight;
        }
    }
    if max_score <= 0.0 {
        0.0
    } else {
        (hit_score / max_score).min(1.0)
    }
}

/// 复刻 _build_result_heading_text。
fn build_result_heading_text(content: &str) -> String {
    let lines: Vec<&str> = content
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        return String::new();
    }
    if lines[0].starts_with("[Sheet]") && lines.len() > 1 {
        format!("{}\n{}", lines[0], lines[1])
    } else {
        lines[0].to_string()
    }
}

/// 复刻 is_reference_index_result。
pub fn is_reference_index_result(content: &str) -> bool {
    REFERENCE_INDEX_MARKERS
        .iter()
        .filter(|marker| content.contains(*marker))
        .count()
        >= 2
}

/// 单条候选的借用视图。
pub struct Candidate<'a> {
    pub content: &'a str,
    pub source_name: &'a str,
    pub score: f64,
    pub source: &'a str,
}

fn search_text_of(content: &str, source_name: &str) -> String {
    let parts: Vec<&str> = [source_name, content].into_iter().filter(|p| !p.is_empty()).collect();
    parts.join("\n")
}

pub struct RelevanceSignals {
    pub query_key: String,
    pub keyword_hits: Vec<String>,
    pub focus_hits: Vec<String>,
    pub long_keyword_hit_count: usize,
    pub keyword_coverage: f64,
    pub focus_coverage: f64,
    pub heading_coverage: f64,
    pub heading_focus_coverage: f64,
    pub exact_match: bool,
    pub is_reference_index: bool,
}

/// 复刻 build_relevance_signals（keywords/focus 由调用方预构建一次复用）。
pub fn build_relevance_signals(
    query: &str,
    candidate: &Candidate,
    keywords: &[String],
    focus_terms: &[String],
) -> RelevanceSignals {
    let text = search_text_of(candidate.content, candidate.source_name);
    let normalized_content = normalize_text(&text);
    let normalized_query = normalize_text(query);

    let keyword_hits: Vec<String> = keywords
        .iter()
        .filter(|t| !t.is_empty() && normalized_content.contains(t.as_str()))
        .cloned()
        .collect();
    let focus_hits: Vec<String> = focus_terms
        .iter()
        .filter(|t| !t.is_empty() && normalized_content.contains(t.as_str()))
        .cloned()
        .collect();

    let mut unique_keyword_hits: Vec<String> = Vec::new();
    for hit in &keyword_hits {
        if !unique_keyword_hits.contains(hit) {
            unique_keyword_hits.push(hit.clone());
        }
    }
    let mut unique_focus_hits: Vec<String> = Vec::new();
    for hit in &focus_hits {
        if !unique_focus_hits.contains(hit) {
            unique_focus_hits.push(hit.clone());
        }
    }
    let long_hit_count = unique_keyword_hits
        .iter()
        .filter(|t| t.chars().count() >= 3)
        .count();
    let exact_match =
        !normalized_query.is_empty() && normalized_content.contains(&normalized_query);

    let heading = build_result_heading_text(candidate.content);
    RelevanceSignals {
        query_key: normalized_query,
        keyword_hits: unique_keyword_hits,
        focus_hits: unique_focus_hits,
        long_keyword_hit_count: long_hit_count,
        keyword_coverage: match_coverage_score(keywords, &text),
        focus_coverage: match_coverage_score(focus_terms, &text),
        heading_coverage: match_coverage_score(keywords, &heading),
        heading_focus_coverage: match_coverage_score(focus_terms, &heading),
        exact_match,
        is_reference_index: is_reference_index_result(candidate.content),
    }
}

/// 复刻 _build_keyword_specificity_scores。
pub fn build_keyword_specificity_scores(
    keywords: &[String],
    contents: &[(&str, &str)], // (content, source_name)
) -> Vec<f64> {
    if keywords.is_empty() || contents.is_empty() {
        return vec![0.0; contents.len()];
    }
    let normalized_texts: Vec<String> = contents
        .iter()
        .map(|(content, source_name)| normalize_text(&search_text_of(content, source_name)))
        .collect();
    let total = contents.len() as f64;
    let mut max_weight = 0.0f64;
    let mut weights: Vec<(String, f64)> = Vec::with_capacity(keywords.len());
    for keyword in keywords {
        if keyword.is_empty() {
            continue;
        }
        let df = normalized_texts
            .iter()
            .filter(|text| text.contains(keyword.as_str()))
            .count() as f64;
        let idf = ((total + 1.0) / (df + 1.0)).ln() + 1.0;
        let weight = idf * ((keyword.chars().count().clamp(1, 8)) as f64 / 8.0);
        weights.push((keyword.clone(), weight));
        max_weight += weight;
    }
    if max_weight <= 0.0 {
        return vec![0.0; contents.len()];
    }
    normalized_texts
        .iter()
        .map(|text| {
            let hit_weight: f64 = weights
                .iter()
                .filter(|(kw, _)| text.contains(kw.as_str()))
                .map(|(_, w)| *w)
                .sum();
            (hit_weight / max_weight).min(1.0)
        })
        .collect()
}

/// 复刻 _passes_keyword_relevance_gate（完整版，含标识符分支与焦点词约束）。
pub fn passes_keyword_relevance_gate(
    query: &str,
    candidate: &Candidate,
    keywords: &[String],
    focus_terms: &[String],
    signals: &RelevanceSignals,
) -> bool {
    let identifier_terms = extract_identifier_terms(query);
    if !identifier_terms.is_empty() {
        let normalized_content =
            normalize_text(&search_text_of(candidate.content, candidate.source_name));
        return identifier_terms
            .iter()
            .any(|term| normalized_content.contains(&normalize_text(term)));
    }
    if keywords.is_empty() {
        return true;
    }
    if signals.exact_match {
        return true;
    }
    let hit_count = signals.keyword_hits.len();
    let focus_hit_count = signals.focus_hits.len();
    let long_hit_count = signals.long_keyword_hit_count;

    if !focus_terms.is_empty()
        && focus_terms.len() >= 2
        && focus_hit_count < 2
        && signals.focus_coverage < 0.18
        && signals.heading_coverage < 0.30
    {
        return false;
    }
    if keywords.len() <= 2 {
        return hit_count >= 1 || signals.keyword_coverage >= 0.28;
    }
    long_hit_count >= 1 || hit_count >= 2 || signals.keyword_coverage >= 0.22
}

/// 复刻 _normalize_source_score。
pub fn normalize_source_score(score: f64, source: &str) -> f64 {
    if source == "sparse" {
        let v = score.max(0.0);
        v / (v + 1.0)
    } else if source == "dense" {
        let v = score.clamp(-1.0, 1.0);
        (v + 1.0) / 2.0
    } else {
        score.max(0.0)
    }
}

/// 复刻 _score_result 综合分。
pub fn score_result(
    source: &str,
    raw_norm: f64,
    specificity: f64,
    s: &RelevanceSignals,
    gate_passed: bool,
) -> f64 {
    let exact_bonus = if s.exact_match { 1.0 } else { 0.0 };
    let gate_bonus = if gate_passed { 0.22 } else { -0.22 };
    let source_bonus = match source {
        "hybrid" => 0.02,
        "dense" => 0.03,
        "sparse" => 0.03,
        _ => 0.0,
    };
    let reference_penalty = if s.is_reference_index { -0.10 } else { 0.0 };
    (0.20 * raw_norm)
        + (0.20 * s.keyword_coverage)
        + (0.24 * specificity)
        + (0.16 * s.focus_coverage)
        + (0.08 * s.heading_coverage)
        + (0.12 * s.heading_focus_coverage)
        + (0.25 * exact_bonus)
        + gate_bonus
        + source_bonus
        + reference_penalty
}
