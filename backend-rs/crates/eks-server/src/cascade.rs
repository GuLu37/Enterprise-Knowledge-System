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

impl crate::retrieval_engine::RetrievalEngine {
    /// 复刻 cascade_rerank_results（adaptive）。
    pub fn cascade_rerank(
        &self,
        query: &str,
        items: Vec<crate::retrieval::RetrievalItem>,
        return_limit: usize,
    ) -> Vec<crate::retrieval::RetrievalItem> {
        use crate::retrieval::RetrievalItem;

        let mut rule_ranked = self.rule_rerank_pub(query, items);
        if rule_ranked.is_empty() {
            return rule_ranked;
        }
        let trigger_mode = self.ce_trigger_mode.trim().to_lowercase();
        let ce = match self.cross_encoder.as_ref() {
            Some(ce) if trigger_mode != "off" => ce,
            _ => {
                for item in rule_ranked.iter_mut() {
                    item.rerank_stage = Some("rule_only_disabled".into());
                }
                return rule_ranked;
            }
        };
        if eks_kernels::rerank::is_identifier_query(query) {
            for item in rule_ranked.iter_mut() {
                item.rerank_stage = Some("rule_only_identifier".into());
            }
            return rule_ranked;
        }
        if trigger_mode != "always" {
            if rule_ranked.len() <= return_limit {
                for item in rule_ranked.iter_mut() {
                    item.rerank_stage = Some("rule_only_small_pool".into());
                }
                return rule_ranked;
            }
            let confidence = rule_top1_confidence(&rule_ranked);
            let top1 = rule_ranked[0].score;
            let top2 = rule_ranked.get(1).map(|item| item.score).unwrap_or(0.0);
            if confidence >= self.ce_threshold && (top1 - top2) >= self.ce_margin {
                for item in rule_ranked.iter_mut() {
                    item.rerank_stage = Some("rule_only_high_confidence".into());
                }
                return rule_ranked;
            }
        }

        // 低置信: 前候选 K 名交给 cross-encoder 精排，权重混合后重排
        let candidate_limit = rule_ranked.len().min(self.ce_candidate_k);
        let ce_texts: Vec<String> = rule_ranked[..candidate_limit]
            .iter()
            .map(cross_encoder_text)
            .collect();
        let ce_scores = match ce.score(query, &ce_texts) {
            Ok(scores) if scores.len() == candidate_limit => scores,
            _ => {
                for item in rule_ranked.iter_mut() {
                    item.rerank_stage = Some("rule_fallback".into());
                }
                return rule_ranked;
            }
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
        scored
            .into_iter()
            .map(|(index, final_score, ce_score)| {
                let mut item: RetrievalItem = original[index].clone();
                item.score = final_score;
                item.rerank_stage = Some("rule_then_cross_encoder".into());
                item.cross_encoder_score = ce_score;
                item
            })
            .collect()
    }
}
