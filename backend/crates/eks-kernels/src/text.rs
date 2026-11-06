//! 文本归一化与 CJK n-gram 分词。
//!
//! 语义必须与 Python 侧逐位一致（`app/rag/retrieval/reranker.py normalize_text`
//! 与 `sparse_retriever.py _tokenize`），由 tests/parity 向量钉死。

/// 复刻 reranker._CLEANUP_PATTERN 的字符集：这些字符在归一化时整体移除。
fn is_cleanup_char(c: char) -> bool {
    c.is_whitespace() // 对齐 Python re 的 \s（含全角空格 U+3000 等 Unicode 空白）
        || matches!(
            c,
            '-' | '_'
            | '.' | ',' | ';' | ':' | '!' | '?'
            | '，' | '。' | '！' | '？' | '；' | '：' | '、'
            | '\u{201c}' | '\u{201d}' // “ ”
            | '\u{2018}' | '\u{2019}' // ‘ ’
            | '"' | '\'' | '`' | '~' | '…' | '·'
            | '/' | '\\' | '|'
            | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>'
            | '《' | '》'
        )
}

/// 复刻 reranker.normalize_text：strip + lower + 移除标点/空白字符。
pub fn normalize_text(text: &str) -> String {
    text.trim()
        .to_lowercase()
        .chars()
        .filter(|c| !is_cleanup_char(*c))
        .collect()
}

#[inline]
fn is_ascii_alnum(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

#[inline]
fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c) // 一 .. 鿿
}

/// 复刻 sparse_retriever._tokenize：
/// 对 lower 后的文本按 `[A-Za-z0-9]+|[一-鿿]+` 切段；
/// CJK 段额外展开 2-gram（长度>2 时）与 3-gram（长度>=4 时），保留整词。
pub fn tokenize_cjk(text: &str) -> Vec<String> {
    let lowered = text.to_lowercase();
    let mut tokens: Vec<String> = Vec::new();
    let mut run: Vec<char> = Vec::new();
    let mut run_cjk = false;

    let mut flush = |run: &mut Vec<char>, run_cjk: bool, tokens: &mut Vec<String>| {
        if run.is_empty() {
            return;
        }
        let part: String = run.iter().collect();
        if run_cjk {
            let len = run.len();
            tokens.push(part.clone());
            if len > 2 {
                for start in 0..len - 1 {
                    tokens.push(run[start..start + 2].iter().collect());
                }
                if len >= 4 {
                    for start in 0..len - 2 {
                        tokens.push(run[start..start + 3].iter().collect());
                    }
                }
            }
        } else {
            tokens.push(part);
        }
        run.clear();
    };

    for c in lowered.chars() {
        if is_ascii_alnum(c) {
            if run_cjk && !run.is_empty() {
                flush(&mut run, run_cjk, &mut tokens);
            }
            run_cjk = false;
            run.push(c);
        } else if is_cjk(c) {
            if !run_cjk && !run.is_empty() {
                flush(&mut run, run_cjk, &mut tokens);
            }
            run_cjk = true;
            run.push(c);
        } else if !run.is_empty() {
            flush(&mut run, run_cjk, &mut tokens);
            run_cjk = false;
        }
    }
    flush(&mut run, run_cjk, &mut tokens);
    tokens
}

/// 对齐 Python `STRUCTURED_SECTION_PATTERN`（`re.IGNORECASE`）：
/// `^[Sheet]\s*(?P<name>.+?)\s*$`。名称保留原文大小写；无名称则不算表头。
pub fn structured_sheet_name(line: &str) -> Option<&str> {
    const PREFIX: &str = "[sheet]";
    let trimmed = line.trim();
    let prefix = trimmed.get(..PREFIX.len())?;
    if !prefix.eq_ignore_ascii_case(PREFIX) {
        return None;
    }
    let name = trimmed[PREFIX.len()..].trim();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_basic_cjk() {
        assert_eq!(
            tokenize_cjk("报销时限"),
            vec!["报销时限", "报销", "销时", "时限", "报销时", "销时限"]
        );
    }

    #[test]
    fn tokenize_mixed() {
        let tokens = tokenize_cjk("E0011 的工资");
        assert!(tokens.contains(&"e0011".to_string()));
        assert!(tokens.contains(&"工资".to_string()));
    }

    #[test]
    fn normalize_removes_punct() {
        assert_eq!(normalize_text("《差旅-报销》制度!"), "差旅报销制度");
        assert_eq!(normalize_text("  A B_c "), "abc");
    }

    #[test]
    fn structured_sheet_name_is_case_insensitive_and_keeps_original() {
        assert_eq!(structured_sheet_name("[Sheet] 花名册"), Some("花名册"));
        assert_eq!(structured_sheet_name("[SHEET] 花名册"), Some("花名册"));
        assert_eq!(structured_sheet_name("[sHeEt] Roster"), Some("Roster"));
        assert_eq!(structured_sheet_name("  [sheet]  明细  "), Some("明细"));
        assert_eq!(structured_sheet_name("[Sheet]"), None);
        assert_eq!(structured_sheet_name("[Sheet]   "), None);
        assert_eq!(structured_sheet_name("部门,姓名"), None);
        assert_eq!(structured_sheet_name("花名册"), None);
    }
}
