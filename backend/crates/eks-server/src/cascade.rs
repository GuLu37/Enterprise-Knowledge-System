//! cascade_rerank（adaptive 精排门控）与置信度/混合打分辅助。
//! 语义复刻 reranker.cascade_rerank_results：小候选池/编号/高置信跳过，
//! 低置信走 Cross-Encoder 前候选 K 精排并以 0.65/0.35 权重混合。

/// 复刻 _rule_top1_confidence（基于规则重排后的信号）。
pub fn rule_top1_confidence(rule_ranked: &[crate::retrieval::RetrievalItem]) -> f64 {
    let top = &rule_ranked[0];
    let signals = match top.signals.as_ref() {
        Some(signals) => signals,
        None => return 0.0,
    };
    let coverage = signals.keyword_coverage.clamp(0.0, 1.0);
    let focus = signals.focus_coverage.clamp(0.0, 1.0);
    let specificity = signals.keyword_specificity.clamp(0.0, 1.0);
    let mut confidence = 0.45 * coverage + 0.30 * focus + 0.25 * specificity;
    if signals.exact_match {
        confidence += 0.20;
    }
    confidence.min(1.0)
}

/// 复刻 _normalize_rank_scores（min-max 到 [0,1]）。
pub fn normalize_rank_scores(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let span = max - min;
    if span <= 1e-9 {
        return scores.iter().map(|s| if *s > 0.0 { 1.0 } else { 0.0 }).collect();
    }
    scores.iter().map(|s| (s - min) / span).collect()
}

/// 复刻 _build_cross_encoder_text：来源 + 标题 + 内容。
pub fn cross_encoder_text(item: &crate::retrieval::RetrievalItem) -> String {
    let lines: Vec<&str> = item
        .content
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let heading = if !lines.is_empty() && lines[0].starts_with("[Sheet]") && lines.len() > 1 {
        format!("{}\n{}", lines[0], lines[1])
    } else {
        lines.first().copied().unwrap_or("").to_string()
    };
    let mut parts: Vec<String> = Vec::new();
    if !item.source_name.is_empty() {
        parts.push(format!("来源: {}", item.source_name));
    }
    if !heading.is_empty() && heading != item.content {
        parts.push(format!("标题: {}", heading));
    }
    if !item.content.is_empty() {
        parts.push(format!("内容: {}", item.content));
    }
    parts.join("\n")
}

fn take_return_limit<T>(items: Vec<T>, return_limit: usize) -> Vec<T> {
    items.into_iter().take(return_limit.max(1)).collect()
}

/// 前两名来自不同文档，或规则分差过小，视为排序不确定，值得走精排。
fn rank_is_ambiguous(items: &[crate::retrieval::RetrievalItem], margin: f64) -> bool {
    let Some(second) = items.get(1) else {
        return false;
    };
    let top = &items[0];
    if (top.score - second.score) < margin {
        return true;
    }
    let left = top.source_name.trim();
    let right = second.source_name.trim();
    !left.is_empty() && !right.is_empty() && left != right
}

/// 复刻 `_attach_rule_only_metadata`：跳过精排时写入可观测标记并截断。
fn attach_rule_only(
    mut items: Vec<crate::retrieval::RetrievalItem>,
    stage: &str,
    confidence: Option<f64>,
    return_limit: usize,
) -> Vec<crate::retrieval::RetrievalItem> {
    for item in items.iter_mut() {
        item.rerank_stage = Some(stage.to_string());
        item.rerank_trace.cross_encoder_used = Some(false);
        item.rerank_trace.rule_confidence = confidence;
        item.rerank_trace.cross_encoder_error = None;
    }
    take_return_limit(items, return_limit)
}

fn format_ce_fallback(err: &str) -> String {
    format!("Cross-Encoder 精排失败，回退规则重排: {err}")
}

fn attach_rule_fallback(
    mut items: Vec<crate::retrieval::RetrievalItem>,
    error: impl Into<String>,
    return_limit: usize,
) -> Vec<crate::retrieval::RetrievalItem> {
    let error = error.into();
    tracing::warn!("{}", format_ce_fallback(&error));
    for item in items.iter_mut() {
        item.rerank_stage = Some("rule_fallback".into());
        item.rerank_trace.cross_encoder_used = Some(false);
        item.rerank_trace.cross_encoder_error = Some(error.clone());
        item.rerank_trace.rule_confidence = None;
    }
    take_return_limit(items, return_limit)
}

fn attach_cross_encoder_item(
    item: &mut crate::retrieval::RetrievalItem,
    original_rule_score: f64,
    rule_score_normalized: f64,
    final_score: f64,
    ce_score: Option<f64>,
    original_index: usize,
) {
    item.score = final_score;
    item.rerank_stage = Some("rule_then_cross_encoder".into());
    item.cross_encoder_score = ce_score;
    item.rerank_trace.cross_encoder_used = Some(ce_score.is_some());
    item.rerank_trace.rule_score = Some(original_rule_score);
    item.rerank_trace.rule_score_normalized = Some(rule_score_normalized);
    item.rerank_trace.final_score = Some(final_score);
    item.rerank_trace.cross_encoder_candidate_rank =
        ce_score.map(|_| (original_index + 1) as i64);
    item.rerank_trace.rule_confidence = None;
    item.rerank_trace.cross_encoder_error = None;
}

impl crate::retrieval_engine::RetrievalEngine {
    /// 复刻 cascade_rerank_results（adaptive）。
    pub fn cascade_rerank(
        &self,
        query: &str,
        items: Vec<crate::retrieval::RetrievalItem>,
        return_limit: usize,
    ) -> Vec<crate::retrieval::RetrievalItem> {
        use crate::retrieval::RetrievalItem;

        let rule_ranked = self.rule_rerank_pub(query, items);
        if rule_ranked.is_empty() {
            return rule_ranked;
        }
        let trigger_mode = self.ce_trigger_mode.trim().to_lowercase();
        let ce = match self.cross_encoder.as_ref() {
            Some(ce) if trigger_mode != "off" => ce,
            _ => {
                return attach_rule_only(rule_ranked, "rule_only_disabled", None, return_limit);
            }
        };
        if eks_kernels::rerank::is_identifier_query(query) {
            return attach_rule_only(rule_ranked, "rule_only_identifier", None, return_limit);
        }
        if trigger_mode != "always" {
            let confidence = rule_top1_confidence(&rule_ranked);
            let ambiguous = rank_is_ambiguous(&rule_ranked, self.ce_margin);
            if rule_ranked.len() <= return_limit && !ambiguous {
                return attach_rule_only(
                    rule_ranked,
                    "rule_only_small_pool",
                    Some(confidence),
                    return_limit,
                );
            }
            if confidence >= self.ce_threshold && !ambiguous {
                return attach_rule_only(
                    rule_ranked,
                    "rule_only_high_confidence",
                    Some(confidence),
                    return_limit,
                );
            }
        }

        // 低置信: 前候选 K 名交给 cross-encoder 精排，权重混合后重排
        let candidate_limit = rule_ranked.len().min(self.ce_candidate_k).max(1);
        let ce_texts: Vec<String> = rule_ranked[..candidate_limit]
            .iter()
            .map(cross_encoder_text)
            .collect();
        // 注：CE 触发频率低（adaptive 门控大多跳过），此处保持同步调用；
        // 会话池已解除并发互斥，偶发阻塞可接受。
        let ce_scores = match ce.score(query, &ce_texts) {
            Ok(scores) if scores.len() == candidate_limit => scores,
            Ok(scores) => {
                return attach_rule_fallback(
                    rule_ranked,
                    format!(
                        "Cross-Encoder 返回分数数量不匹配: expected={candidate_limit}, actual={}",
                        scores.len()
                    ),
                    return_limit,
                );
            }
            Err(err) => return attach_rule_fallback(rule_ranked, err, return_limit),
        };

        let rule_scores: Vec<f64> = rule_ranked.iter().map(|item| item.score).collect();
        let rule_norm = normalize_rank_scores(&rule_scores);
        let weight_total = self.ce_weight + self.rule_weight;
        let (ce_w, rule_w) = if weight_total <= 1e-9 {
            (0.65, 0.35)
        } else {
            (self.ce_weight / weight_total, self.rule_weight / weight_total)
        };

        // (下标, final, ce_score)
        let mut scored: Vec<(usize, f64, Option<f64>)> = rule_ranked
            .iter()
            .enumerate()
            .map(|(index, _)| (index, rule_norm[index], None))
            .collect();
        for (position, raw) in ce_scores.iter().enumerate() {
            let clamped = raw.clamp(0.0, 1.0);
            scored[position].1 = ce_w * clamped + rule_w * rule_norm[position];
            scored[position].2 = Some(clamped);
        }
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let ca = a.2.unwrap_or(-1.0);
                    let cb = b.2.unwrap_or(-1.0);
                    cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    rule_scores[b.0]
                        .partial_cmp(&rule_scores[a.0])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.0.cmp(&b.0))
        });

        let original = rule_ranked;
        take_return_limit(
            scored
                .into_iter()
                .map(|(index, final_score, ce_score)| {
                    let mut item: RetrievalItem = original[index].clone();
                    attach_cross_encoder_item(
                        &mut item,
                        rule_scores[index],
                        rule_norm[index],
                        final_score,
                        ce_score,
                        index,
                    );
                    item
                })
                .collect(),
            return_limit,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::{ItemSignals, RetrievalItem, RerankTrace};

    fn item(content: &str, score: f64) -> RetrievalItem {
        item_from("手册.md", content, score)
    }

    fn item_from(source_name: &str, content: &str, score: f64) -> RetrievalItem {
        RetrievalItem {
            content: content.into(),
            score,
            source: "hybrid".into(),
            document_id: source_name.into(),
            source_name: source_name.into(),
            chunk_index: 0,
            child_id: "c1".into(),
            parent_id: "p1".into(),
            parent_index: 0,
            file_type: "md".into(),
            content_type: "text".into(),
            signals: Some(ItemSignals {
                keyword_coverage: 0.8,
                focus_coverage: 0.7,
                heading_coverage: 0.0,
                heading_focus_coverage: 0.0,
                exact_match: true,
                is_reference_index: false,
                keyword_specificity: 0.6,
                keyword_hits: Vec::new(),
                focus_hits: Vec::new(),
                long_keyword_hit_count: 0,
            }),
            parent_context_text: None,
            rerank_stage: None,
            cross_encoder_score: None,
            rerank_trace: RerankTrace::default(),
            structured_aggregation: false,
            section_name: None,
            chunk_indices: Vec::new(),
            matched_sources: Vec::new(),
            fusion_score: None,
        }
    }

    #[test]
    fn skip_metadata_marks_unused_encoder_and_optional_confidence() {
        let skipped = attach_rule_only(
            vec![item("报销标准", 0.9), item("差旅流程", 0.4)],
            "rule_only_small_pool",
            Some(0.72),
            1,
        );
        assert_eq!(skipped.len(), 1);
        assert_eq!(skipped[0].rerank_stage.as_deref(), Some("rule_only_small_pool"));
        assert_eq!(skipped[0].rerank_trace.cross_encoder_used, Some(false));
        assert_eq!(skipped[0].rerank_trace.rule_confidence, Some(0.72));
        let payload = skipped[0].to_json();
        assert_eq!(payload["metadata"]["cross_encoder_used"], false);
        assert_eq!(payload["metadata"]["rule_confidence"], 0.72);
        assert_eq!(payload["metadata"]["rerank_stage"], "rule_only_small_pool");

        let disabled = attach_rule_only(vec![item("编号 E001", 1.0)], "rule_only_disabled", None, 8);
        assert_eq!(disabled[0].rerank_trace.cross_encoder_used, Some(false));
        assert!(disabled[0].rerank_trace.rule_confidence.is_none());
        let disabled_json = disabled[0].to_json();
        assert_eq!(disabled_json["metadata"]["cross_encoder_used"], false);
        assert!(disabled_json["metadata"].get("rule_confidence").is_none());
    }

    #[test]
    fn fallback_metadata_includes_error_and_slices() {
        let fallback = attach_rule_fallback(
            vec![item("a", 0.9), item("b", 0.8), item("c", 0.1)],
            "Cross-Encoder 返回分数数量不匹配: expected=2, actual=1",
            2,
        );
        assert_eq!(fallback.len(), 2);
        assert_eq!(fallback[0].rerank_stage.as_deref(), Some("rule_fallback"));
        assert_eq!(fallback[0].rerank_trace.cross_encoder_used, Some(false));
        assert!(fallback[0]
            .rerank_trace
            .cross_encoder_error
            .as_deref()
            .unwrap()
            .contains("分数数量不匹配"));
        let payload = fallback[0].to_json();
        assert_eq!(payload["metadata"]["cross_encoder_used"], false);
        assert!(payload["metadata"]["cross_encoder_error"]
            .as_str()
            .unwrap()
            .contains("expected=2"));
        assert_eq!(
            format_ce_fallback("onnx timeout"),
            "Cross-Encoder 精排失败，回退规则重排: onnx timeout"
        );
    }

    #[test]
    fn cross_encoder_trace_writes_scores_and_null_rank_outside_pool() {
        let mut scored = item("报销标准", 0.4);
        attach_cross_encoder_item(&mut scored, 0.4, 1.0, 0.82, Some(0.91), 0);
        let payload = scored.to_json();
        assert_eq!(payload["metadata"]["rerank_stage"], "rule_then_cross_encoder");
        assert_eq!(payload["metadata"]["cross_encoder_used"], true);
        assert_eq!(payload["metadata"]["cross_encoder_score"], 0.91);
        assert_eq!(payload["metadata"]["rule_score"], 0.4);
        assert_eq!(payload["metadata"]["rule_score_normalized"], 1.0);
        assert_eq!(payload["metadata"]["final_score"], 0.82);
        assert_eq!(payload["metadata"]["cross_encoder_candidate_rank"], 1);
        assert_eq!(payload["score"], 0.82);

        let mut outside = item("目录", 0.1);
        attach_cross_encoder_item(&mut outside, 0.1, 0.0, 0.0, None, 3);
        let outside_json = outside.to_json();
        assert_eq!(outside_json["metadata"]["cross_encoder_used"], false);
        assert!(outside_json["metadata"]["cross_encoder_score"].is_null());
        assert!(outside_json["metadata"]["cross_encoder_candidate_rank"].is_null());
        assert_eq!(outside_json["metadata"]["rule_score"], 0.1);
    }

    #[test]
    fn ambiguous_when_margin_is_small_or_sources_differ() {
        let close = vec![item("报销标准", 0.90), item("差旅流程", 0.89)];
        assert!(rank_is_ambiguous(&close, 0.08));
        let same_source = vec![item("报销标准", 0.90), item("差旅流程", 0.40)];
        assert!(!rank_is_ambiguous(&same_source, 0.08));
        let different = vec![
            item_from("02_采购审批流程.docx", "采购审批", 0.90),
            item_from("04_产品与账号常见问题.md", "引用资料", 0.70),
        ];
        assert!(rank_is_ambiguous(&different, 0.08));
        assert!(!rank_is_ambiguous(&close[..1], 0.08));
    }
}
