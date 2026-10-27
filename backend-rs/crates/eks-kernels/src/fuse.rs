//! RRF 融合（fuse_ranked_results 的纯 Rust 版，语义与 Python 侧一致）。

/// 融合输入项：带去重键与排序键的有序候选。
pub struct FuseItem<T> {
    /// 去重键组成部分: document_id。
    pub document_id: String,
    /// chunk_index（缺省 -1，用于去重键）。
    pub chunk_key: i64,
    /// 排序用的 chunk_index（缺省 0，取首见结果）。
    pub chunk_sort: i64,
    pub content: String,
    pub value: T,
}

pub struct FuseOutput<T> {
    pub document_id: String,
    pub chunk_key: i64,
    pub content: String,
    pub fused_score: f64,
    pub sources: Vec<String>,
    pub value: T,
}

/// 对多路有序结果执行 RRF 融合（rrf_k 缺省 60）。
/// 排序键: (-fused_score, 首见结果的 chunk_index)；并列保持首见插入序（稳定排序）。
pub fn fuse_ranked_results<T>(
    result_sets: Vec<Vec<FuseItem<T>>>,
    weights: &[f64],
    source_labels: &[&str],
    rrf_k: f64,
) -> Vec<FuseOutput<T>> {
    use std::collections::HashMap;

    struct Slot<T> {
        document_id: String,
        chunk_key: i64,
        content: String,
        chunk_sort: i64,
        score: f64,
        sources: Vec<String>,
        value: T,
    }

    let mut slots: Vec<Slot<T>> = Vec::new();
    let mut index_map: HashMap<(String, i64, String), usize> = HashMap::new();

    for (set_index, results) in result_sets.into_iter().enumerate() {
        let weight = weights.get(set_index).copied().unwrap_or(1.0);
        let source_name = source_labels
            .get(set_index)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("route_{}", set_index + 1));
        for (rank0, item) in results.into_iter().enumerate() {
            let content = item.content.trim().to_string();
            let key = (item.document_id.clone(), item.chunk_key, content.clone());
            let contribution = weight / (rrf_k + (rank0 as f64 + 1.0));
            match index_map.get(&key) {
                Some(&position) => {
                    slots[position].score += contribution;
                    if !slots[position].sources.contains(&source_name) {
                        slots[position].sources.push(source_name.clone());
                    }
                }
                None => {
                    index_map.insert(key, slots.len());
                    slots.push(Slot {
                        document_id: item.document_id,
                        chunk_key: item.chunk_key,
                        content,
                        chunk_sort: item.chunk_sort,
                        score: contribution,
                        sources: vec![source_name.clone()],
                        value: item.value,
                    });
                }
            }
        }
    }

    // 稳定排序保证 (score, chunk_sort) 全并列时保持首见插入序。
    slots.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.chunk_sort.cmp(&b.chunk_sort))
    });

    slots
        .into_iter()
        .map(|slot| FuseOutput {
            document_id: slot.document_id,
            chunk_key: slot.chunk_key,
            content: slot.content,
            fused_score: slot.score,
            sources: slot.sources,
            value: slot.value,
        })
        .collect()
}
