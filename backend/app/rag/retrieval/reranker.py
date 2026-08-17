"""检索结果打分、重排与融合。"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .base import RetrievalResult

_QUERY_TERM_PATTERN = re.compile(r"[A-Za-z0-9]+|[一-鿿]{2,}")
_CLEANUP_PATTERN = re.compile(r"[\s\-_.,;:!?，。！？；：、“”‘’\"'`~…·/\\|()[\]{}<>《》]+")


def normalize_text(text: str) -> str:
    """把文本压成便于匹配的形式。"""
    normalized = (text or "").strip().lower()
    if not normalized:
        return ""
    return _CLEANUP_PATTERN.sub("", normalized)


def build_query_terms(query: str, max_terms: int = 16) -> List[str]:
    """提取适合做命中判断的查询词。"""
    normalized_query = (query or "").strip().lower()
    if not normalized_query:
        return []

    terms: List[str] = []
    for part in _QUERY_TERM_PATTERN.findall(normalized_query):
        if re.fullmatch(r"[一-鿿]{2,}", part):
            if len(part) >= 4:
                terms.append(part)
                terms.extend(part[index:index + 2] for index in range(len(part) - 1))
                if len(part) >= 6:
                    terms.extend(part[index:index + 3] for index in range(len(part) - 2))
            else:
                terms.append(part)
        else:
            terms.append(part)

    deduplicated: List[str] = []
    seen = set()
    for term in sorted(terms, key=len, reverse=True):
        normalized_term = term.strip()
        if not normalized_term or normalized_term in seen:
            continue
        seen.add(normalized_term)
        deduplicated.append(normalized_term)
        if len(deduplicated) >= max_terms:
            break
    return deduplicated


def _normalize_source_score(score: float, source: str = "") -> float:
    """把不同来源的原始分数压到可比较区间。"""
    value = float(score or 0.0)
    if source == "sparse":
        value = max(0.0, value)
        return value / (value + 1.0)

    if source == "dense":
        if value < -1.0:
            value = -1.0
        if value > 1.0:
            value = 1.0
        return (value + 1.0) / 2.0

    return max(0.0, value)


def _match_coverage_score(query_terms: Sequence[str], content: str) -> float:
    """用查询词覆盖率衡量内容相关性。"""
    if not query_terms:
        return 0.0

    normalized_content = normalize_text(content)
    if not normalized_content:
        return 0.0

    hit_score = 0.0
    max_score = 0.0
    normalized_query = normalize_text("".join(query_terms[:3]))
    if normalized_query and normalized_query in normalized_content:
        hit_score += 1.0

    for term in query_terms:
        normalized_term = normalize_text(term)
        if not normalized_term:
            continue
        weight = min(max(len(normalized_term), 1), 12) / 12.0
        max_score += weight
        if normalized_term in normalized_content:
            hit_score += weight

    if max_score <= 0.0:
        return 0.0
    return min(1.0, hit_score / max_score)


def _build_result_search_text(result: RetrievalResult) -> str:
    """组合切片正文与来源文件名，供关键词覆盖度和精确匹配使用。"""
    metadata = result.metadata or {}
    source_name = str(metadata.get("source_name") or "")
    content = result.content or ""
    return "\n".join(part for part in (source_name, content) if part)


def _score_result(
    query_terms: Sequence[str],
    normalized_query: str,
    result: RetrievalResult,
    raw_norm: float,
) -> float:
    """对单条检索结果计算综合分。"""
    searchable_text = _build_result_search_text(result)
    coverage = _match_coverage_score(query_terms, searchable_text)
    normalized_content = normalize_text(searchable_text)
    exact_bonus = 1.0 if normalized_query and normalized_query in normalized_content else 0.0
    source_bonus = 0.0
    if result.source == "hybrid":
        source_bonus = 0.02
    elif result.source == "dense":
        source_bonus = 0.03
    elif result.source == "sparse":
        source_bonus = 0.03

    return (0.35 * raw_norm) + (0.35 * coverage) + (0.30 * exact_bonus) + source_bonus


def rerank_results(
    query: str,
    results: Sequence[RetrievalResult],
    top_k: int | None = None,
) -> List[RetrievalResult]:
    """根据查询词覆盖和原始分数重排结果。"""
    query_terms = build_query_terms(query)
    normalized_query = normalize_text(query)
    candidates = [item for item in results if item and (item.content or "").strip()]
    if not candidates:
        return []

    raw_scores = [_normalize_source_score(item.score, item.source) for item in candidates]
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    score_span = max_score - min_score

    scored_items: List[Tuple[int, RetrievalResult, float]] = []
    for index, item in enumerate(candidates):
        if score_span > 1e-9:
            raw_norm = (raw_scores[index] - min_score) / score_span
        else:
            raw_norm = 1.0 if raw_scores[index] > 0 else 0.0
        scored_items.append((index, item, _score_result(query_terms, normalized_query, item, raw_norm)))

    scored_items.sort(
        key=lambda item: (
            -item[2],
            -float(item[1].score or 0.0),
            item[0],
        )
    )

    limited = scored_items[: max(1, top_k or len(scored_items))]
    return [
        RetrievalResult(
            content=item.content,
            metadata=item.metadata,
            score=float(score),
            source=item.source,
        )
        for _, item, score in limited
    ]


def _dedup_key(result: RetrievalResult) -> Tuple[str, int, str]:
    """生成跨检索器稳定去重键。"""
    metadata = result.metadata or {}
    document_id = str(metadata.get("document_id") or "")
    chunk_index = metadata.get("chunk_index")
    try:
        chunk_index_value = int(chunk_index)
    except (TypeError, ValueError):
        chunk_index_value = -1
    return (document_id, chunk_index_value, (result.content or "").strip())


def fuse_ranked_results(
    result_sets: Sequence[Sequence[RetrievalResult]],
    weights: Sequence[float] | None = None,
    source_labels: Sequence[str] | None = None,
    rrf_k: int = 60,
) -> List[RetrievalResult]:
    """对多路有序结果执行 RRF 融合，并做初步去重。"""
    fused: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    resolved_weights = list(weights or [])
    resolved_labels = list(source_labels or [])

    for set_index, results in enumerate(result_sets):
        weight = resolved_weights[set_index] if set_index < len(resolved_weights) else 1.0
        source_name = resolved_labels[set_index] if set_index < len(resolved_labels) else f"route_{set_index + 1}"
        for rank, result in enumerate(results, start=1):
            key = _dedup_key(result)

            current = fused.get(key)
            contribution = weight / (rrf_k + rank)
            if current is None:
                fused[key] = {
                    "result": result,
                    "score": contribution,
                    "sources": [source_name],
                }
            else:
                current["score"] += contribution
                if source_name not in current["sources"]:
                    current["sources"].append(source_name)

    ordered = sorted(
        fused.values(),
        key=lambda item: (
            -float(item["score"]),
            item["result"].metadata.get("chunk_index", 0),
        ),
    )

    merged: List[RetrievalResult] = []
    for item in ordered:
        base_result: RetrievalResult = item["result"]
        metadata = dict(base_result.metadata or {})
        metadata["matched_sources"] = list(item["sources"])
        metadata["fusion_score"] = float(item["score"])
        merged.append(
            RetrievalResult(
                content=base_result.content,
                metadata=metadata,
                score=float(item["score"]),
                source="hybrid",
            )
        )
    return merged


def fuse_retrieval_results(
    dense_results: Sequence[RetrievalResult],
    sparse_results: Sequence[RetrievalResult],
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[RetrievalResult]:
    """按 reciprocal rank fusion 方式融合 dense/sparse 两路结果。"""
    return fuse_ranked_results(
        result_sets=[dense_results, sparse_results],
        weights=[dense_weight, sparse_weight],
        source_labels=["dense", "sparse"],
        rrf_k=60,
    )
