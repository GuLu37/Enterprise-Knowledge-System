//! 检索结果内存缓存（对齐 Python `_RAG_RESULT_CACHE`）。
//!
//! 相同规范化 query + top_k + 检索方式在 TTL 内直接返回，降低重复问题和
//! ReAct 循环内重复检索的延迟。文档入库/删除时清空，避免返回过期结果。

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::retrieval::RetrievalItem;

pub const RAG_CACHE_MAX_ENTRIES: usize = 1000;

#[derive(Clone)]
struct CachedRag {
    results: Vec<RetrievalItem>,
    context: String,
}

pub struct RagResultCache {
    enabled: bool,
    ttl: Duration,
    max_entries: usize,
    entries: HashMap<String, (Instant, CachedRag)>,
}

impl RagResultCache {
    pub fn new(enabled: bool, ttl_secs: u64) -> Self {
        Self::with_limit(enabled, ttl_secs, RAG_CACHE_MAX_ENTRIES)
    }

    pub fn with_limit(enabled: bool, ttl_secs: u64, max_entries: usize) -> Self {
        RagResultCache {
            enabled,
            ttl: Duration::from_secs(ttl_secs),
            max_entries: max_entries.max(1),
            entries: HashMap::new(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn get(&self, key: &str) -> Option<(Vec<RetrievalItem>, String)> {
        self.get_at(key, Instant::now())
    }

    pub fn get_at(&self, key: &str, now: Instant) -> Option<(Vec<RetrievalItem>, String)> {
        if !self.enabled {
            return None;
        }
        let (cached_at, value) = self.entries.get(key)?;
        if now.saturating_duration_since(*cached_at) < self.ttl {
            Some((value.results.clone(), value.context.clone()))
        } else {
            None
        }
    }

    pub fn put(&mut self, key: String, results: Vec<RetrievalItem>, context: String) {
        self.put_at(key, Instant::now(), results, context);
    }

    pub fn put_at(
        &mut self,
        key: String,
        now: Instant,
        results: Vec<RetrievalItem>,
        context: String,
    ) {
        if !self.enabled {
            return;
        }
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&key) {
            let expired: Vec<String> = self
                .entries
                .iter()
                .filter(|(_, (cached_at, _))| now.saturating_duration_since(*cached_at) >= self.ttl)
                .map(|(item, _)| item.clone())
                .collect();
            for item in expired {
                self.entries.remove(&item);
            }
            if self.entries.len() >= self.max_entries {
                let mut oldest_keys: Vec<(Instant, String)> = self
                    .entries
                    .iter()
                    .map(|(item, (cached_at, _))| (*cached_at, item.clone()))
                    .collect();
                oldest_keys.sort_by_key(|(cached_at, _)| *cached_at);
                let overflow = self.entries.len() - self.max_entries + 1;
                for (_, item) in oldest_keys.into_iter().take(overflow) {
                    self.entries.remove(&item);
                }
            }
        }
        self.entries.insert(
            key,
            (
                now,
                CachedRag {
                    results,
                    context,
                },
            ),
        );
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

pub fn rag_cache_key(query: &str, top_k: usize, retrieval_method: &str) -> String {
    format!(
        "{}::t{top_k}::{retrieval_method}",
        eks_kernels::normalize_text(query)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::ItemSignals;

    fn item(content: &str) -> RetrievalItem {
        RetrievalItem {
            content: content.into(),
            score: 0.5,
            source: "hybrid".into(),
            document_id: "d1".into(),
            source_name: "制度.pdf".into(),
            chunk_index: 0,
            child_id: "c1".into(),
            parent_id: "p1".into(),
            parent_index: 0,
            file_type: "pdf".into(),
            content_type: "text".into(),
            signals: Some(ItemSignals::default()),
            parent_context_text: None,
            rerank_stage: None,
            cross_encoder_score: None,
            structured_aggregation: false,
            section_name: None,
            chunk_indices: Vec::new(),
        }
    }

    #[test]
    fn cache_key_normalizes_query_and_keeps_method() {
        assert_eq!(
            rag_cache_key(" 差旅报销标准。 ", 5, "sparse"),
            rag_cache_key("差旅报销标准", 5, "sparse")
        );
        assert_ne!(
            rag_cache_key("差旅报销标准", 5, "hybrid"),
            rag_cache_key("差旅报销标准", 8, "hybrid")
        );
        assert!(rag_cache_key("差旅报销", 5, "dense").ends_with("::t5::dense"));
    }

    #[test]
    fn disabled_cache_neither_stores_nor_hits() {
        let mut cache = RagResultCache::new(false, 3600);
        cache.put("k".into(), vec![item("内容")], "ctx".into());
        assert_eq!(cache.len(), 0);
        assert!(cache.get("k").is_none());
    }

    #[test]
    fn hit_returns_clone_and_expired_misses() {
        let mut cache = RagResultCache::new(true, 10);
        let now = Instant::now();
        cache.put_at("k".into(), now, vec![item("原文")], "ctx".into());
        let (mut results, context) = cache.get_at("k", now).expect("hit");
        assert_eq!(context, "ctx");
        results[0].content = "被污染".into();
        let (again, _) = cache.get_at("k", now).expect("still original");
        assert_eq!(again[0].content, "原文");
        assert!(cache.get_at("k", now + Duration::from_secs(10)).is_none());
    }

    #[test]
    fn evicts_expired_then_oldest_when_full() {
        let mut cache = RagResultCache::with_limit(true, 10, 2);
        let t0 = Instant::now();
        let t1 = t0 + Duration::from_secs(1);
        let t2 = t0 + Duration::from_secs(2);
        cache.put_at("a".into(), t0, vec![item("a")], "a".into());
        cache.put_at("b".into(), t1, vec![item("b")], "b".into());
        cache.put_at("c".into(), t2, vec![item("c")], "c".into());
        assert_eq!(cache.len(), 2);
        assert!(cache.get_at("a", t2).is_none());
        assert!(cache.get_at("b", t2).is_some());
        assert!(cache.get_at("c", t2).is_some());

        let mut cache = RagResultCache::with_limit(true, 5, 2);
        cache.put_at("old".into(), t0, vec![item("old")], "old".into());
        cache.put_at("keep".into(), t0 + Duration::from_secs(4), vec![item("keep")], "keep".into());
        let later = t0 + Duration::from_secs(6);
        cache.put_at("new".into(), later, vec![item("new")], "new".into());
        assert!(cache.get_at("old", later).is_none());
        assert!(cache.get_at("keep", later).is_some());
        assert!(cache.get_at("new", later).is_some());
    }

    #[test]
    fn clear_drops_all_entries() {
        let mut cache = RagResultCache::new(true, 3600);
        cache.put("k".into(), vec![item("内容")], "ctx".into());
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.get("k").is_none());
    }
}
