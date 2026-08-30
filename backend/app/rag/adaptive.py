"""Adaptive-RAG 的轻量决策层。

该模块只做可解释的策略决策，不负责真正的文档检索。检索仍由
``search_knowledge_base`` Function Calling 工具完成。
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.rag.retrieval.reranker import (
    build_query_focus_terms,
    build_query_keywords,
    extract_identifier_terms,
    is_identifier_query,
    normalize_text,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

RETRIEVAL_METHOD_HYBRID = "hybrid"
RETRIEVAL_METHOD_DENSE = "dense"
RETRIEVAL_METHOD_SPARSE = "sparse"
SUPPORTED_RETRIEVAL_METHODS = {
    RETRIEVAL_METHOD_HYBRID,
    RETRIEVAL_METHOD_DENSE,
    RETRIEVAL_METHOD_SPARSE,
}

MAX_ADAPTIVE_ATTEMPTS = 2
DEFAULT_TOP_K = 5
MAX_TOP_K = 50
MIN_EVIDENCE_SCORE = 0.28
MIN_KEYWORD_COVERAGE = 0.22

# 置信度分界：高于上界直接规则判「充分」，低于下界直接规则判「不足」，
# 落在中间模糊区间才交给 LLM 消歧，避免每次检索都多调一次模型。
HIGH_EVIDENCE_SCORE = 0.45
LOW_EVIDENCE_SCORE = 0.15
HIGH_KEYWORD_COVERAGE = 0.50
LOW_KEYWORD_COVERAGE = 0.10

_ADAPTIVE_SELECT_SYSTEM_PROMPT = (
    "你是企业知识库检索策略决策器。规则未给出强信号时，由你做最终决策。"
    "可选检索方式：hybrid（混合，默认稳妥）、dense（纯语义召回）、sparse（纯关键词/精确匹配）。"
    "只输出严格 JSON，格式：{\"retrieval_method\":\"hybrid|dense|sparse\",\"top_k\":整数,\"strategy\":\"简短策略名\",\"reason\":\"简短中文理由\"}"
)

_ADAPTIVE_ASSESS_SYSTEM_PROMPT = (
    "你是检索证据评估器。根据用户问题与给定检索信号，判断现有检索结果是否足以回答问题。"
    "只依据提供的信号做判断，不要臆测知识库内容。"
    "只输出严格 JSON，格式：{\"sufficient\":true|false,\"reason\":\"简短中文理由\"}"
)

_ADAPTIVE_REWRITE_SYSTEM_PROMPT = (
    "你是企业知识库检索查询改写器。证据不足时，改写查询用于二次检索。"
    "必须原样保留编号、工号、产品号、人名、文档名等专名，不得丢失问题焦点。"
    "只输出严格 JSON，格式：{\"query\":\"改写后的检索查询\"}"
)

_STRUCTURED_QUERY_TERMS = (
    "列出",
    "各",
    "分别",
    "有哪些",
    "名单",
    "列表",
    "清单",
    "明细",
    "统计",
    "数量",
    "多少",
    "一共",
    "总数",
    "比较",
    "对比",
    "同时",
    "并且",
)
_PROCEDURAL_QUERY_TERMS = (
    "如何",
    "怎么",
    "流程",
    "步骤",
    "要求",
    "条件",
    "标准",
    "规定",
    "政策",
    "制度",
)


@dataclass(frozen=True)
class AdaptiveRAGDecision:
    """一次 RAG 尝试的结构化计划。"""

    query: str
    retrieval_method: str
    top_k: int
    strategy: str
    attempt: int = 1
    reason: str = ""
    should_retrieve: bool = True
    stop_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceAssessment:
    """对一次 RAG 工具返回结果的证据质量判断。"""

    sufficient: bool
    result_count: int
    best_score: float
    keyword_coverage: float
    identifier_hit: bool
    reason: str
    missing_identifiers: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["missing_identifiers"] = list(self.missing_identifiers)
        return payload


@dataclass(frozen=True)
class AdaptiveSignals:
    """规则层产出的软信号，只描述事实，不下结论。"""

    is_identifier: bool = False
    structured_hit: bool = False
    procedural_hit: bool = False
    best_score: float = 0.0
    keyword_coverage: float = 0.0
    result_count: int = 0
    missing_identifiers: Tuple[str, ...] = ()


def _normalize_method(method: str) -> str:
    normalized = (method or RETRIEVAL_METHOD_HYBRID).strip().lower()
    return (
        normalized
        if normalized in SUPPORTED_RETRIEVAL_METHODS
        else RETRIEVAL_METHOD_HYBRID
    )


def _normalize_top_k(top_k: Optional[int]) -> int:
    try:
        resolved = int(top_k or DEFAULT_TOP_K)
    except (TypeError, ValueError):
        resolved = DEFAULT_TOP_K
    return max(1, min(MAX_TOP_K, resolved))


def _contains_any(query: str, terms: Sequence[str]) -> bool:
    normalized = (query or "").strip().lower()
    return any(term in normalized for term in terms)


def _result_text(result: Dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    return "\n".join(
        str(value or "")
        for value in (
            metadata.get("source_name"),
            metadata.get("document_id"),
            result.get("content"),
        )
    )


def _numeric_score(result: Dict[str, Any]) -> float:
    try:
        return float(result.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _result_keyword_coverage(result: Dict[str, Any]) -> float:
    metadata = result.get("metadata") or {}
    for key in ("keyword_coverage", "focus_coverage"):
        try:
            value = float(metadata.get(key) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0:
            return min(1.0, max(0.0, value))
    return 0.0


def _query_signature(query: str) -> str:
    return normalize_text(re.sub(r"\s+", " ", (query or "").strip()))


def _build_follow_up_query(
    original_query: str,
    first_query: str,
    assessment: EvidenceAssessment,
) -> str:
    """生成与上一轮不同、但仍保留原问题焦点的补检索 query。"""
    original = re.sub(r"\s+", " ", (original_query or "").strip())
    first = re.sub(r"\s+", " ", (first_query or "").strip())
    if _query_signature(first) != _query_signature(original) and original:
        return original

    if assessment.missing_identifiers:
        identifiers = " ".join(assessment.missing_identifiers)
        return f"{original or first} 精确匹配 {identifiers}".strip()

    focus_terms = build_query_focus_terms(original or first, max_terms=3)
    if focus_terms:
        focus = " ".join(focus_terms)
        return f"{original or first} 具体规定 {focus}".strip()

    return f"{original or first} 具体内容".strip()


def _extract_text(message: Any) -> str:
    """提取 LangChain 消息或字典中的纯文本。"""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def _parse_json_payload(text: str) -> Dict[str, Any]:
    """从 LLM 输出中提取 JSON 对象，失败返回空字典。"""
    normalized = (text or "").strip()
    if not normalized:
        return {}
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?", "", normalized, flags=re.IGNORECASE).strip()
        normalized = re.sub(r"```$", "", normalized).strip()
    match = re.search(r"\{[\s\S]*\}", normalized)
    candidate = match.group(0) if match else normalized
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _invoke_llm_decision(llm: Any, system_prompt: str, human_prompt: str) -> Dict[str, Any]:
    """用字符串 prompt 调 LLM 并解析 JSON，失败由调用方兜底。"""
    prompt = f"{system_prompt}\n\n{human_prompt}"
    response = llm.invoke(prompt)
    return _parse_json_payload(_extract_text(response))


def _extract_query_signals(query: str) -> AdaptiveSignals:
    """从查询中提取规则软信号。"""
    normalized = re.sub(r"\s+", " ", (query or "").strip())
    return AdaptiveSignals(
        is_identifier=is_identifier_query(normalized),
        structured_hit=_contains_any(normalized, _STRUCTURED_QUERY_TERMS),
        procedural_hit=_contains_any(normalized, _PROCEDURAL_QUERY_TERMS),
    )


class AdaptiveRAGController:
    """基于规则的 Adaptive-RAG 控制器。

    第一版优先保证可解释性和确定性。后续如果引入模型评分，只需要替换
    ``assess_evidence`` 或 ``plan_follow_up``，不会改变工具接口。
    """

    def select_initial(
        self,
        query: str,
        top_k: Optional[int] = None,
        retrieval_method: str = RETRIEVAL_METHOD_HYBRID,
        llm: Optional[Any] = None,
    ) -> AdaptiveRAGDecision:
        normalized_query = re.sub(r"\s+", " ", (query or "").strip())
        resolved_method = _normalize_method(retrieval_method)
        resolved_top_k = _normalize_top_k(top_k)
        signals = _extract_query_signals(normalized_query)

        # 第一层：规则评估置信度。有强特征时直接规则判，零额外 LLM 延迟。
        if signals.is_identifier or signals.structured_hit or signals.procedural_hit:
            return self._rule_select(normalized_query, resolved_method, resolved_top_k, signals)

        # 第二层：无强信号（default），交给 LLM 做最终决策。
        if llm is not None:
            try:
                return self._llm_select(normalized_query, signals, resolved_top_k, llm)
            except Exception as exc:
                logger.warning("Adaptive-RAG 策略选择 LLM 决策失败，回退规则: %s", exc)

        # 第三层：LLM 失败或未提供，回退规则兜底。
        return self._rule_select(normalized_query, resolved_method, resolved_top_k, signals)

    def _rule_select(
        self,
        query: str,
        resolved_method: str,
        resolved_top_k: int,
        signals: AdaptiveSignals,
    ) -> AdaptiveRAGDecision:
        """规则直判：按强信号选择检索策略。"""
        if signals.is_identifier:
            strategy = "identifier_exact"
            reason = "检测到编号、工号、产品号或编码，优先使用稀疏检索保留精确匹配"
            if resolved_method == RETRIEVAL_METHOD_HYBRID:
                resolved_method = RETRIEVAL_METHOD_SPARSE
        elif signals.structured_hit:
            strategy = "structured_multi_query"
            reason = "检测到多条件或结构化查询，保留混合检索并扩大候选范围"
            resolved_top_k = max(resolved_top_k, 8)
        elif signals.procedural_hit:
            strategy = "procedural_hybrid"
            reason = "检测到流程、条件或制度类问题，优先使用混合检索"
            if resolved_method == RETRIEVAL_METHOD_DENSE:
                resolved_method = RETRIEVAL_METHOD_HYBRID
        else:
            strategy = "default_hybrid"
            reason = "通用知识库问题使用混合检索作为稳定起点"
            if resolved_method not in SUPPORTED_RETRIEVAL_METHODS:
                resolved_method = RETRIEVAL_METHOD_HYBRID

        return AdaptiveRAGDecision(
            query=query,
            retrieval_method=resolved_method,
            top_k=resolved_top_k,
            strategy=strategy,
            attempt=1,
            reason=reason,
        )

    def _llm_select(
        self,
        query: str,
        signals: AdaptiveSignals,
        resolved_top_k: int,
        llm: Any,
    ) -> AdaptiveRAGDecision:
        """LLM 决策：无强信号时，让模型选择检索方式与 top_k。"""
        human_prompt = (
            f"用户问题：{query}\n"
            f"规则信号：编号查询={signals.is_identifier}，结构化查询={signals.structured_hit}，流程类查询={signals.procedural_hit}\n"
            f"默认 top_k={resolved_top_k}\n"
            "请输出最终检索策略 JSON。"
        )
        payload = _invoke_llm_decision(llm, _ADAPTIVE_SELECT_SYSTEM_PROMPT, human_prompt)
        method = _normalize_method(str(payload.get("retrieval_method") or RETRIEVAL_METHOD_HYBRID))
        top_k = _normalize_top_k(payload.get("top_k") or resolved_top_k)
        strategy = str(payload.get("strategy") or "llm_hybrid").strip() or "llm_hybrid"
        reason = str(payload.get("reason") or "LLM 决策选择检索策略").strip()
        return AdaptiveRAGDecision(
            query=query,
            retrieval_method=method,
            top_k=top_k,
            strategy=strategy,
            attempt=1,
            reason=reason,
        )

    def assess_evidence(
        self,
        query: str,
        payload: Optional[Dict[str, Any]],
        llm: Optional[Any] = None,
    ) -> EvidenceAssessment:
        result_payload = payload if isinstance(payload, dict) else {}
        results = [
            item
            for item in (result_payload.get("results") or [])
            if isinstance(item, dict)
        ]
        if not results:
            return EvidenceAssessment(
                sufficient=False,
                result_count=0,
                best_score=0.0,
                keyword_coverage=0.0,
                identifier_hit=False,
                reason="no_results",
            )

        best_score = max((_numeric_score(item) for item in results), default=0.0)
        keyword_coverage = max(
            (_result_keyword_coverage(item) for item in results),
            default=0.0,
        )
        identifiers = extract_identifier_terms(query)
        missing_identifiers = tuple(
            identifier
            for identifier in identifiers
            if not any(
                normalize_text(identifier) in normalize_text(_result_text(item))
                for item in results
            )
        )
        identifier_hit = bool(identifiers) and not missing_identifiers

        # 第一层：规则评估置信度。明确充分/明确不足/编号查询直接规则判。
        if self._evidence_high_confidence(identifiers, best_score, keyword_coverage):
            return self._rule_assess(
                results, identifiers, best_score, keyword_coverage, identifier_hit, missing_identifiers
            )

        # 第二层：模糊区间，交 LLM 决策。
        if llm is not None:
            try:
                assessment = self._llm_assess(
                    query, results, best_score, keyword_coverage,
                    identifier_hit, missing_identifiers, llm,
                )
                if assessment is not None:
                    return assessment
            except Exception as exc:
                logger.warning("Adaptive-RAG 证据评估 LLM 决策失败，回退规则: %s", exc)

        # 第三层：LLM 失败或未提供，回退规则兜底。
        return self._rule_assess(
            results, identifiers, best_score, keyword_coverage, identifier_hit, missing_identifiers
        )

    def _evidence_high_confidence(
        self,
        identifiers: Sequence[str],
        best_score: float,
        keyword_coverage: float,
    ) -> bool:
        """判断证据是否处于可直接规则判定的高置信区间。"""
        if identifiers:
            return True
        if best_score >= HIGH_EVIDENCE_SCORE or keyword_coverage >= HIGH_KEYWORD_COVERAGE:
            return True
        if best_score <= LOW_EVIDENCE_SCORE and keyword_coverage <= LOW_KEYWORD_COVERAGE:
            return True
        return False

    def _rule_assess(
        self,
        results: Sequence[Dict[str, Any]],
        identifiers: Sequence[str],
        best_score: float,
        keyword_coverage: float,
        identifier_hit: bool,
        missing_identifiers: Tuple[str, ...],
    ) -> EvidenceAssessment:
        """规则兜底：按阈值硬判定证据充分性。"""
        if identifiers:
            sufficient = identifier_hit
            reason = "identifier_hit" if sufficient else "identifier_incomplete"
        else:
            sufficient = (
                best_score >= MIN_EVIDENCE_SCORE
                or keyword_coverage >= MIN_KEYWORD_COVERAGE
                or any(
                    bool((item.get("metadata") or {}).get("exact_match"))
                    for item in results
                )
            )
            if sufficient:
                reason = "evidence_sufficient"
            elif len(results) < 2:
                reason = "weak_single_result"
            else:
                reason = "low_score_or_coverage"

        return EvidenceAssessment(
            sufficient=sufficient,
            result_count=len(results),
            best_score=round(max(0.0, best_score), 6),
            keyword_coverage=round(max(0.0, keyword_coverage), 6),
            identifier_hit=identifier_hit,
            reason=reason,
            missing_identifiers=missing_identifiers,
        )

    def _llm_assess(
        self,
        query: str,
        results: Sequence[Dict[str, Any]],
        best_score: float,
        keyword_coverage: float,
        identifier_hit: bool,
        missing_identifiers: Tuple[str, ...],
        llm: Any,
    ) -> Optional[EvidenceAssessment]:
        """LLM 决策：模糊区间让模型判断证据是否充分。"""
        snippet = " | ".join(
            str(item.get("content") or "")[:60] for item in results[:3]
        )
        human_prompt = (
            f"用户问题：{query}\n"
            f"检索信号：结果数={len(results)}，最高分={best_score:.3f}，关键词覆盖={keyword_coverage:.3f}\n"
            f"命中片段示例：{snippet}\n"
            "请判断现有检索结果是否足以回答用户问题。"
        )
        payload = _invoke_llm_decision(llm, _ADAPTIVE_ASSESS_SYSTEM_PROMPT, human_prompt)
        raw_sufficient = payload.get("sufficient")
        if raw_sufficient is None:
            return None
        if isinstance(raw_sufficient, bool):
            sufficient = raw_sufficient
        else:
            sufficient = str(raw_sufficient).strip().lower() in {
                "true", "1", "yes", "on", "是", "充分", "足够", "能", "可以",
            }
        reason = str(payload.get("reason") or "llm_assessed").strip()
        return EvidenceAssessment(
            sufficient=sufficient,
            result_count=len(results),
            best_score=round(max(0.0, best_score), 6),
            keyword_coverage=round(max(0.0, keyword_coverage), 6),
            identifier_hit=identifier_hit,
            reason=reason,
            missing_identifiers=missing_identifiers,
        )

    def plan_follow_up(
        self,
        original_query: str,
        decision: AdaptiveRAGDecision,
        assessment: EvidenceAssessment,
        llm: Optional[Any] = None,
    ) -> Optional[AdaptiveRAGDecision]:
        """证据不足时生成最多一次不同策略的补检索计划。"""
        if assessment.sufficient or decision.attempt >= MAX_ADAPTIVE_ATTEMPTS:
            return None

        # 第一层：策略切换用规则（确定性高，无需 LLM）。
        if is_identifier_query(original_query) or assessment.missing_identifiers:
            next_method = RETRIEVAL_METHOD_SPARSE
            strategy = "identifier_exact_retry"
            reason = "上一轮未覆盖全部精确标识，改用稀疏检索并强化编号匹配"
        elif decision.retrieval_method == RETRIEVAL_METHOD_SPARSE:
            next_method = RETRIEVAL_METHOD_HYBRID
            strategy = "hybrid_recall_retry"
            reason = "稀疏检索证据不足，切换混合召回补充语义结果"
        elif decision.retrieval_method == RETRIEVAL_METHOD_DENSE:
            next_method = RETRIEVAL_METHOD_SPARSE
            strategy = "sparse_exact_retry"
            reason = "密集检索证据不足，切换稀疏检索补充关键词命中"
        else:
            next_method = RETRIEVAL_METHOD_SPARSE
            strategy = "sparse_focus_retry"
            reason = "混合检索证据不足，切换稀疏检索强化核心词和文件名命中"

        next_query = _build_follow_up_query(
            original_query=original_query,
            first_query=decision.query,
            assessment=assessment,
        )

        # 第二层：查询改写交 LLM（可选），失败回退模板改写。
        if llm is not None:
            try:
                rewritten = self._llm_rewrite_query(original_query, decision.query, assessment, llm)
                if rewritten:
                    next_query = rewritten
            except Exception as exc:
                logger.warning("Adaptive-RAG 补检索改写 LLM 失败，回退模板: %s", exc)

        return AdaptiveRAGDecision(
            query=next_query,
            retrieval_method=next_method,
            top_k=max(decision.top_k, 8 if assessment.result_count == 0 else decision.top_k),
            strategy=strategy,
            attempt=decision.attempt + 1,
            reason=reason,
        )

    def _llm_rewrite_query(
        self,
        original_query: str,
        first_query: str,
        assessment: EvidenceAssessment,
        llm: Any,
    ) -> Optional[str]:
        """LLM 改写二次检索查询。"""
        human_prompt = (
            f"用户原问题：{original_query}\n"
            f"上次检索查询：{first_query}\n"
            f"缺失编号：{'、'.join(assessment.missing_identifiers) if assessment.missing_identifiers else '无'}\n"
            "请输出改写后的二次检索查询。"
        )
        payload = _invoke_llm_decision(llm, _ADAPTIVE_REWRITE_SYSTEM_PROMPT, human_prompt)
        rewritten = re.sub(r"\s+", " ", str(payload.get("query") or "")).strip()
        return rewritten or None


__all__ = [
    "AdaptiveRAGController",
    "AdaptiveRAGDecision",
    "AdaptiveSignals",
    "EvidenceAssessment",
    "MAX_ADAPTIVE_ATTEMPTS",
    "MIN_EVIDENCE_SCORE",
    "MIN_KEYWORD_COVERAGE",
]
