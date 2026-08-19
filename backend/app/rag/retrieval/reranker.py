"""检索结果打分、重排与融合。"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .base import RetrievalResult

_QUERY_TERM_PATTERN = re.compile(r"[A-Za-z0-9]+|[一-鿿]{2,}")
_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]{1,12}[-_ ]?\d{2,}[A-Za-z0-9_-]*|\d{4,}[A-Za-z0-9_-]*)(?![A-Za-z0-9])"
)
_CLEANUP_PATTERN = re.compile(r"[\s\-_.,;:!?，。！？；：、“”‘’\"'`~…·/\\|()[\]{}<>《》]+")
_QUESTION_FILLER_CHARS = set("的吗呢吧么嘛是否有没有请帮想要了解查询检索一下")
_QUERY_STOP_TERMS = {
    "公司",
    "企业",
    "知识库",
    "文档",
    "资料",
    "里面",
    "相关",
    "标准吗",
    "有没有",
    "是否有",
}
_QUERY_INTENT_TERMS = (
    "完整名单",
    "全体名单",
    "名单列表",
    "清单列表",
    "分别是什么",
    "有哪些",
    "有谁",
    "是谁",
    "多少个",
    "多少名",
    "一共有多少",
    "共有多少",
    "总人数",
    "总数",
    "数量",
    "名单",
    "列表",
    "清单",
    "明细",
    "构成",
    "分别",
    "全部",
    "所有",
    "全体",
    "各个",
    "各",
)
_DOMAIN_TERMS = (
    "办公地点",
    "直属主管",
    "负责人",
    "员工",
    "部门",
    "岗位",
    "职位",
    "工号",
    "姓名",
    "状态",
    "学历",
    "邮箱",
    "城市交通",
    "市内出行",
    "出租车",
    "网约车",
    "火车站",
    "行程单",
    "高铁",
    "飞机",
    "住宿",
    "餐补",
    "报销",
    "审批",
    "流程",
    "标准",
    "费用",
    "凭证",
    "发票",
    "材料",
    "差旅",
    "采购",
    "合同",
    "付款",
    "验收",
    "上线",
    "发布",
    "灰度",
    "回滚",
    "账号",
    "登录",
    "密码",
    "权限",
    "项目",
    "文档",
)
_GENERIC_FOCUS_TERMS = {
    "公司",
    "企业",
    "城市",
    "知识库",
    "文档",
    "资料",
    "报销",
    "审批",
    "流程",
    "标准",
    "费用",
    "制度",
    "规范",
    "政策",
    "手册",
    "相关",
}
_IDENTIFIER_HINT_TERMS = (
    "员工编号",
    "员工号",
    "工号",
    "人员编号",
    "用户编号",
    "产品编号",
    "产品号",
    "商品编号",
    "物料编号",
    "订单编号",
    "合同编号",
    "项目编号",
    "编号",
    "编码",
    "型号",
    "id",
)
_REFERENCE_INDEX_MARKERS = (
    "查询速查",
    "优先命中表",
    "检索提示",
    "回答参考",
    "常见问题",
)


def extract_identifier_terms(query: str) -> List[str]:
    """提取用户问题中的编号、工号、产品号等精确检索词。"""
    raw_query = (query or "").strip()
    if not raw_query:
        return []

    identifiers: List[str] = []
    seen = set()
    for match in _IDENTIFIER_PATTERN.findall(raw_query):
        compact = re.sub(r"\s+", "", match).strip()
        plain = re.sub(r"[\s_-]+", "", match).strip()
        for base_value in (compact, plain):
            for value in (base_value, base_value.upper(), base_value.lower()):
                if value and value not in seen:
                    seen.add(value)
                    identifiers.append(value)
    return identifiers


def is_identifier_query(query: str) -> bool:
    """判断是否像员工号、产品号、订单号这类应优先精确检索的查询。"""
    normalized = (query or "").strip()
    if not normalized:
        return False
    lowered = normalized.lower()
    if extract_identifier_terms(normalized):
        return True
    if any(term in lowered for term in _IDENTIFIER_HINT_TERMS if term != "id"):
        return True
    return bool(re.search(r"(?<![A-Za-z0-9])id(?![A-Za-z0-9])", lowered))


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

    terms: List[str] = extract_identifier_terms(query)
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


def _normalize_query_for_keywords(query: str) -> str:
    """清理问句噪声，避免生成跨越虚词的中文 ngram。"""
    normalized = normalize_text(query)
    if not normalized:
        return ""

    replacements = (
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
    )
    for old, new in replacements:
        normalized = normalized.replace(old, new)

    normalized = re.sub(r"(是什么|有哪些|如何|怎么|为什么|可以吗|能否|请说明|请解释)$", "", normalized)
    for stop_term in sorted(_QUERY_STOP_TERMS, key=len, reverse=True):
        normalized = normalized.replace(stop_term, "")
    for intent_term in sorted(_QUERY_INTENT_TERMS, key=len, reverse=True):
        normalized = normalized.replace(intent_term, "")
    normalized = normalized.replace("有标准", "标准")
    normalized = normalized.strip("的吗呢吧么嘛请帮想要了解一下")
    return normalized


def _append_unique(items: List[str], seen: set[str], value: str, max_terms: int) -> bool:
    normalized_value = normalize_text(value)
    if len(normalized_value) < 2 or normalized_value in seen:
        return len(items) >= max_terms
    seen.add(normalized_value)
    items.append(normalized_value)
    return len(items) >= max_terms


def build_query_keywords(query: str, max_terms: int = 24) -> List[str]:
    """提取用于最终相关性过滤的有效关键词。"""
    normalized_query = _normalize_query_for_keywords(query)
    keywords: List[str] = []
    seen = set()

    for term in _DOMAIN_TERMS:
        if term in normalized_query and _append_unique(keywords, seen, term, max_terms):
            return keywords

    if 2 <= len(normalized_query) <= 16:
        _append_unique(keywords, seen, normalized_query, max_terms)

    for size in (4, 3, 2):
        if len(normalized_query) < size:
            continue
        for index in range(len(normalized_query) - size + 1):
            term = normalized_query[index:index + size]
            if any(char in _QUESTION_FILLER_CHARS for char in term):
                continue
            if _append_unique(keywords, seen, term, max_terms):
                return keywords

    raw_terms = build_query_terms(normalized_query, max_terms=max_terms * 4)
    for term in raw_terms:
        normalized_term = normalize_text(term)
        if len(normalized_term) < 2:
            continue
        if normalized_term in _QUERY_STOP_TERMS:
            continue
        if re.fullmatch(r"[一-鿿]+", normalized_term):
            if all(char in _QUESTION_FILLER_CHARS for char in normalized_term):
                continue
            if len(normalized_term) >= 3 and any(char in _QUESTION_FILLER_CHARS for char in normalized_term):
                continue
        if normalized_term in seen:
            continue
        seen.add(normalized_term)
        keywords.append(normalized_term)
        if len(keywords) >= max_terms:
            break
    return keywords


def build_query_focus_terms(query: str, max_terms: int = 8) -> List[str]:
    """提取能代表用户真正主题的核心焦点词。"""
    normalized_query = _normalize_query_for_keywords(query)
    focus_terms: List[str] = []
    seen = set()

    matched_domain_terms = sorted(
        (
            (normalized_query.find(term), term)
            for term in _DOMAIN_TERMS
            if term in normalized_query
        ),
        key=lambda item: (item[0], -len(item[1])),
    )
    # 连续业务词组成的短语比单个宽泛词更能区分主题，例如：
    # “差旅 + 报销 + 标准”应形成“差旅报销标准”。
    for start_index, (start_position, _) in enumerate(matched_domain_terms):
        end_position = start_position
        for next_position, next_term in matched_domain_terms[start_index:]:
            if next_position > end_position + 1:
                break
            end_position = max(
                end_position,
                next_position + len(next_term),
            )
        phrase = normalized_query[start_position:end_position]
        if len(phrase) >= 4:
            _append_unique(focus_terms, seen, phrase, max_terms)
            if len(focus_terms) >= max_terms:
                return focus_terms

    for term in _DOMAIN_TERMS:
        if term in _GENERIC_FOCUS_TERMS or term not in normalized_query:
            continue
        if _append_unique(focus_terms, seen, term, max_terms):
            return focus_terms
        if term == "城市交通" and _append_unique(focus_terms, seen, "交通", max_terms):
            return focus_terms
        if term == "市内出行" and _append_unique(focus_terms, seen, "出行", max_terms):
            return focus_terms

    if focus_terms:
        return focus_terms

    keywords = build_query_keywords(query, max_terms=32)
    for term in keywords:
        if term in _GENERIC_FOCUS_TERMS:
            continue
        if len(term) > 2 and term not in _DOMAIN_TERMS and term != normalized_query:
            continue
        if len(term) < 2:
            continue
        if term in seen:
            continue
        seen.add(term)
        focus_terms.append(term)
        if len(focus_terms) >= max_terms:
            break
    return focus_terms


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


def _build_result_heading_text(result: RetrievalResult) -> str:
    """提取 chunk 的结构标题，供章节/表格标题命中加权。"""
    lines = [
        line.strip()
        for line in (result.content or "").splitlines()
        if line.strip()
    ]
    if not lines:
        return ""

    heading_lines = [lines[0]]
    if lines[0].startswith("[Sheet]") and len(lines) > 1:
        heading_lines.append(lines[1])
    return "\n".join(heading_lines)


def is_reference_index_result(result: RetrievalResult) -> bool:
    """识别用于导航的索引/速查片段，避免它压过事实数据片段。

    这是按内容角色判断，而不是按文件名或具体业务字段判断。索引片段
    仍然可以作为兜底证据，但在存在原始记录时不应主导最终答案。
    """
    content = str(result.content or "")
    marker_count = sum(1 for marker in _REFERENCE_INDEX_MARKERS if marker in content)
    return marker_count >= 2


def _build_keyword_specificity_scores(
    keywords: Sequence[str],
    candidates: Sequence[RetrievalResult],
) -> List[float]:
    """按候选集合内的关键词稀有度计算区分度分数。"""
    if not keywords or not candidates:
        return [0.0 for _ in candidates]

    normalized_texts = [
        normalize_text(_build_result_search_text(item))
        for item in candidates
    ]
    document_frequency: Dict[str, int] = {}
    for keyword in keywords:
        document_frequency[keyword] = sum(
            1
            for text in normalized_texts
            if keyword and keyword in text
        )

    total = len(candidates)
    max_weight = 0.0
    keyword_weights: Dict[str, float] = {}
    for keyword in keywords:
        if not keyword:
            continue
        idf = math.log((total + 1) / (document_frequency.get(keyword, 0) + 1)) + 1.0
        weight = idf * (min(max(len(keyword), 1), 8) / 8.0)
        keyword_weights[keyword] = weight
        max_weight += weight

    if max_weight <= 0.0:
        return [0.0 for _ in candidates]

    scores: List[float] = []
    for text in normalized_texts:
        hit_weight = sum(
            weight
            for keyword, weight in keyword_weights.items()
            if keyword in text
        )
        scores.append(min(1.0, hit_weight / max_weight))
    return scores


def build_relevance_signals(
    query: str,
    result: RetrievalResult,
    query_keywords: Sequence[str] | None = None,
    query_focus_terms: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """生成最终过滤可复用的相关性信号。"""
    keywords = list(query_keywords) if query_keywords is not None else build_query_keywords(query)
    focus_terms = (
        list(query_focus_terms)
        if query_focus_terms is not None
        else build_query_focus_terms(query)
    )
    searchable_text = _build_result_search_text(result)
    normalized_content = normalize_text(searchable_text)
    normalized_query = normalize_text(query)
    keyword_hits = [
        term
        for term in keywords
        if term and term in normalized_content
    ]
    focus_hits = [
        term
        for term in focus_terms
        if term and term in normalized_content
    ]
    unique_hits = list(dict.fromkeys(keyword_hits))
    unique_focus_hits = list(dict.fromkeys(focus_hits))
    long_hit_count = sum(1 for term in unique_hits if len(term) >= 3)
    exact_match = bool(normalized_query and normalized_query in normalized_content)
    return {
        "_query_key": normalize_text(query),
        "query_keywords": keywords,
        "query_focus_terms": focus_terms,
        "keyword_hits": unique_hits,
        "focus_hits": unique_focus_hits,
        "keyword_hit_count": len(unique_hits),
        "focus_hit_count": len(unique_focus_hits),
        "long_keyword_hit_count": long_hit_count,
        "keyword_coverage": _match_coverage_score(keywords, searchable_text),
        "focus_coverage": _match_coverage_score(focus_terms, searchable_text),
        "heading_coverage": _match_coverage_score(keywords, _build_result_heading_text(result)),
        "heading_focus_coverage": _match_coverage_score(focus_terms, _build_result_heading_text(result)),
        "exact_match": exact_match,
        "is_reference_index": is_reference_index_result(result),
    }


def _passes_keyword_relevance_gate(
    query: str,
    result: RetrievalResult,
    signals: Dict[str, Any],
) -> bool:
    """判断结果是否满足最终引用所需的关键词相关性。"""
    identifier_terms = extract_identifier_terms(query)
    if identifier_terms:
        normalized_content = normalize_text(_build_result_search_text(result))
        # 编号查询的完整性优先于普通主题阈值，命中的每个编号都必须
        # 有机会进入后续的逐编号保护逻辑。
        return any(normalize_text(term) in normalized_content for term in identifier_terms)

    keywords = signals["query_keywords"]
    focus_terms = signals["query_focus_terms"]
    if not keywords:
        return True
    if signals["exact_match"]:
        return True

    hit_count = int(signals["keyword_hit_count"])
    focus_hit_count = int(signals["focus_hit_count"])
    long_hit_count = int(signals["long_keyword_hit_count"])
    coverage = float(signals["keyword_coverage"] or 0.0)
    focus_coverage = float(signals["focus_coverage"] or 0.0)
    heading_coverage = float(signals["heading_coverage"] or 0.0)

    if focus_terms and len(focus_terms) >= 2 and focus_hit_count < 2 and focus_coverage < 0.18:
        # 复合主题不能只凭一个宽泛词通过；如果标题本身明确命中主题，
        # 则允许正文 chunk 只覆盖其中一部分（例如制度首页）。
        if heading_coverage < 0.30:
            return False

    if len(keywords) <= 2:
        return hit_count >= 1 or coverage >= 0.28
    return long_hit_count >= 1 or hit_count >= 2 or coverage >= 0.22


def passes_keyword_relevance_gate(query: str, result: RetrievalResult) -> bool:
    """判断结果是否满足最终引用所需的关键词相关性。"""
    cached_signals = (result.metadata or {}).get("_relevance_signals")
    if (
        isinstance(cached_signals, dict)
        and cached_signals.get("_query_key") == normalize_text(query)
    ):
        signals = cached_signals
    else:
        signals = build_relevance_signals(query, result)
    return _passes_keyword_relevance_gate(query, result, signals)


def _score_result(
    result: RetrievalResult,
    raw_norm: float,
    specificity: float,
    signals: Dict[str, Any],
    keyword_gate_passed: bool,
) -> float:
    """对单条检索结果计算综合分。"""
    coverage = float(signals["keyword_coverage"] or 0.0)
    focus_coverage = float(signals["focus_coverage"] or 0.0)
    heading_coverage = float(signals["heading_coverage"] or 0.0)
    heading_focus_coverage = float(signals["heading_focus_coverage"] or 0.0)
    exact_bonus = 1.0 if signals["exact_match"] else 0.0
    keyword_gate_bonus = 0.22 if keyword_gate_passed else -0.22
    source_bonus = 0.0
    if result.source == "hybrid":
        source_bonus = 0.02
    elif result.source == "dense":
        source_bonus = 0.03
    elif result.source == "sparse":
        source_bonus = 0.03
    reference_index_penalty = -0.10 if signals["is_reference_index"] else 0.0

    return (
        (0.20 * raw_norm)
        + (0.20 * coverage)
        + (0.24 * specificity)
        + (0.16 * focus_coverage)
        + (0.08 * heading_coverage)
        + (0.12 * heading_focus_coverage)
        + (0.25 * exact_bonus)
        + keyword_gate_bonus
        + source_bonus
        + reference_index_penalty
    )


def rerank_results(
    query: str,
    results: Sequence[RetrievalResult],
    top_k: int | None = None,
) -> List[RetrievalResult]:
    """根据查询词覆盖和原始分数重排结果。"""
    candidates = [item for item in results if item and (item.content or "").strip()]
    if not candidates:
        return []

    query_keywords = build_query_keywords(query)
    query_focus_terms = build_query_focus_terms(query)
    relevance_signals = [
        build_relevance_signals(
            query,
            item,
            query_keywords=query_keywords,
            query_focus_terms=query_focus_terms,
        )
        for item in candidates
    ]
    specificity_scores = _build_keyword_specificity_scores(query_keywords, candidates)
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
        scored_items.append((
            index,
            item,
            _score_result(
                item,
                raw_norm,
                specificity_scores[index],
                relevance_signals[index],
                _passes_keyword_relevance_gate(
                    query,
                    item,
                    relevance_signals[index],
                ),
            ),
        ))

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
            metadata={
                **(item.metadata or {}),
                **{
                    key: value
                    for key, value in relevance_signals[index].items()
                    if key not in {"query_keywords", "query_focus_terms"}
                },
                "_relevance_signals": relevance_signals[index],
                "keyword_specificity": float(specificity_scores[index]),
            },
            score=float(score),
            source=item.source,
        )
        for index, item, score in limited
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
