"""统一的文档检索业务服务。"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.config import settings
from app.rag.retrieval.base import RetrievalResult
from app.rag.retrieval.reranker import (
    build_query_terms,
    extract_identifier_terms,
    fuse_ranked_results,
    is_identifier_query,
    passes_keyword_relevance_gate,
    rerank_results,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

MIN_MULTI_QUERY_COUNT = 3
MAX_MULTI_QUERY_COUNT = 5
MAX_CITATION_RESULTS = 3
STRUCTURED_QUERY_CITATION_RESULTS = 5
RAG_CONTEXT_MAX_CHARS = 6500
MIN_FINAL_RELEVANCE_SCORE = 0.28
MIN_RAG_CANDIDATE_POOL = 60
NO_RESULTS_MESSAGE = "当前知识库未检索到足够相关资料"
MULTI_QUERY_SYSTEM_PROMPT = (
    "你是企业知识库检索的多查询扩展器。"
    "你的任务是根据用户问题，生成 3 到 5 条同义、不同角度的检索查询变体。"
    "这些变体必须适合直接用于知识库检索，不要回答问题本身，不要解释，不要输出编号。"
    "请覆盖同义词、不同表述、关键词式、问题式等角度，尽量提升召回。"
    "只输出严格 JSON，格式必须是：{\"queries\":[\"查询1\",\"查询2\",\"查询3\"]}"
)

STRUCTURED_QUERY_HINT_TERMS = (
    "员工",
    "部门",
    "岗位",
    "职位",
    "工号",
    "姓名",
    "婚姻",
    "状态",
    "入职",
    "离职",
    "手机号",
    "电话",
    "邮箱",
    "地址",
    "身份证",
    "证件",
    "账号",
    "编码",
    "编号",
)


@dataclass
class RagWorkflowResult:
    """一次 RAG 工具执行后的结构化结果。"""

    query: str
    retrieval_method: str = "hybrid"
    expanded_queries: List[str] = field(default_factory=list)
    results: List[RetrievalResult] = field(default_factory=list)
    context: str = ""
    message: str = "未检索到相关内容"


def _extract_text_content(message: Any) -> str:
    """提取模型返回中的纯文本。"""
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


def _build_retriever(top_k: int, retrieval_method: str = "hybrid"):
    """根据配置和请求的检索方式构造检索器。"""
    normalized_method = (retrieval_method or "hybrid").strip().lower()
    dense_retriever = None
    sparse_retriever = None

    if settings.USE_DENSE_RETRIEVER:
        from app.rag.retrieval.dense_retriever import DenseRetriever
        from app.core.embeddings import get_default_embeddings

        dense_retriever = DenseRetriever(
            embeddings=get_default_embeddings(),
            top_k=top_k,
        )

    if settings.USE_SPARSE_RETRIEVER:
        from app.rag.retrieval.sparse_retriever import SparseRetriever

        sparse_retriever = SparseRetriever(top_k=top_k)

    if settings.USE_HYBRID_RETRIEVER and dense_retriever and sparse_retriever:
        from app.rag.retrieval.hybrid_retriever import HybridRetriever

        if normalized_method == "dense":
            return dense_retriever
        if normalized_method == "sparse":
            return sparse_retriever
        return HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            top_k=top_k,
            dense_weight=settings.DENSE_WEIGHT,
            sparse_weight=settings.SPARSE_WEIGHT,
        )

    if normalized_method == "dense":
        return dense_retriever or sparse_retriever
    if normalized_method == "sparse":
        return sparse_retriever or dense_retriever

    return dense_retriever or sparse_retriever


def _resolve_retrieval_method_for_query(query: str, retrieval_method: str) -> str:
    """编号型查询自动避免纯向量检索，优先保留关键词/精确匹配机会。"""
    normalized_method = (retrieval_method or "hybrid").strip().lower()
    if normalized_method == "dense" and is_identifier_query(query):
        return "hybrid"
    return normalized_method


def retrieve_documents(
    query: str,
    top_k: Optional[int] = None,
    retrieval_method: str = "hybrid",
) -> List[RetrievalResult]:
    """执行一次文档检索，返回统一的 RetrievalResult 列表。"""
    normalized_query = (query or "").strip()
    if not normalized_query:
        return []

    resolved_top_k = max(1, min(top_k or settings.SEARCH_TOP_K, 50))
    candidate_k = max(resolved_top_k * 8, resolved_top_k + 20, MIN_RAG_CANDIDATE_POOL)
    resolved_method = _resolve_retrieval_method_for_query(normalized_query, retrieval_method)
    retriever = _build_retriever(candidate_k, retrieval_method=resolved_method)
    if retriever is None:
        return []

    candidates = retriever.retrieve(normalized_query, top_k=candidate_k)
    return rerank_results(query=normalized_query, results=candidates, top_k=resolved_top_k)


def _normalize_query_variants(
    candidates: Sequence[Any],
    base_query: str,
    max_queries: int = MAX_MULTI_QUERY_COUNT,
) -> List[str]:
    """清洗并去重 query 变体。"""
    normalized_base = re.sub(r"\s+", " ", (base_query or "").strip())
    expanded_queries: List[str] = []
    seen = set()

    for candidate in candidates:
        item = re.sub(r"\s+", " ", str(candidate or "")).strip()
        item = re.sub(r"^[\s\d\.\-、:：]+", "", item).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        expanded_queries.append(item)
        if len(expanded_queries) >= max(1, max_queries):
            break

    if normalized_base and normalized_base.lower() not in seen:
        expanded_queries.insert(0, normalized_base)

    return expanded_queries[: max(1, max_queries)]


def _heuristic_multi_query_variants(query: str, max_queries: int = MAX_MULTI_QUERY_COUNT) -> List[str]:
    """LLM 失败时的兜底 query 变体生成。"""
    normalized_query = re.sub(r"\s+", " ", (query or "").strip())
    if not normalized_query:
        return []

    candidates = [normalized_query]
    cleaned = normalized_query
    for _ in range(2):
        cleaned = re.sub(
            r"^(请问|麻烦|帮我|帮忙|查询|检索|搜索|查一下|查找|根据|基于|结合)"
            r"(一下|一下子|知识库|文档|资料|上传文档|企业知识库|公司资料)?[：:\s，,]*",
            "",
            cleaned,
        ).strip()
    cleaned = re.sub(
        r"(根据|基于|结合)?(知识库|企业知识库|文档|资料|上传文档|公司资料)(里面|中|里)?",
        "",
        cleaned,
    ).strip(" ：:，,。？?")
    if cleaned:
        candidates.append(cleaned)

    focus = re.sub(
        r"(是什么|有哪些|如何|怎么|为什么|是否|能否|可以吗|吗|呢|请说明|请解释|介绍一下|说明一下)",
        "",
        cleaned or normalized_query,
    ).strip(" ：:，,。？?")
    if focus:
        candidates.append(focus)

    query_terms = build_query_terms(focus or cleaned or normalized_query, max_terms=8)
    if query_terms:
        candidates.append(" ".join(query_terms[:6]))

    return _normalize_query_variants(candidates, normalized_query, max_queries=max_queries)


def _identifier_query_variants(query: str) -> List[str]:
    """编号型查询不要交给 LLM 改写，优先保留原始编号和大小写变体。"""
    normalized_query = re.sub(r"\s+", " ", (query or "").strip())
    candidates = [normalized_query]
    candidates.extend(extract_identifier_terms(normalized_query))
    return _normalize_query_variants(candidates, normalized_query, max_queries=MAX_MULTI_QUERY_COUNT)


def _parse_multi_query_payload(text: str) -> Dict[str, Any]:
    """解析 LLM 输出的多查询 JSON。"""
    normalized = (text or "").strip()
    if not normalized:
        return {}
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?", "", normalized, flags=re.IGNORECASE).strip()
        normalized = re.sub(r"```$", "", normalized).strip()
    match = re.search(r"\{[\s\S]*\}", normalized)
    candidate = match.group(0) if match else normalized
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def generate_multi_query_variants(
    query: str,
    llm: Optional[Any] = None,
    max_queries: int = MAX_MULTI_QUERY_COUNT,
) -> List[str]:
    """优先用 LLM 生成多查询变体，失败时回退到启发式变体。"""
    normalized_query = re.sub(r"\s+", " ", (query or "").strip())
    if not normalized_query:
        return []
    if is_identifier_query(normalized_query):
        return _identifier_query_variants(normalized_query)

    if llm is None:
        return _heuristic_multi_query_variants(normalized_query, max_queries=max_queries)

    prompt = (
        f"{MULTI_QUERY_SYSTEM_PROMPT}\n\n"
        f"用户问题：{normalized_query}\n\n"
        f"请输出 3 到 5 条查询变体，优先保证同义改写、不同角度改写和关键词改写都被覆盖。"
    )

    try:
        response = llm.invoke(prompt)
        payload = _parse_multi_query_payload(_extract_text_content(response))
        variants = payload.get("queries") or payload.get("query_variants") or payload.get("variants")
        if isinstance(variants, list):
            normalized_variants = _normalize_query_variants(variants, normalized_query, max_queries=max_queries)
            if len(normalized_variants) >= MIN_MULTI_QUERY_COUNT:
                return normalized_variants
    except Exception:
        pass

    heuristic_variants = _heuristic_multi_query_variants(normalized_query, max_queries=max_queries)
    return heuristic_variants


def expand_multi_query(
    query: str,
    llm: Optional[Any] = None,
    max_queries: int = MAX_MULTI_QUERY_COUNT,
) -> List[str]:
    """生成多查询变体。优先由 LLM 产出，再用启发式兜底。"""
    return generate_multi_query_variants(query, llm=llm, max_queries=max_queries)


def retrieve_documents_multi_query(
    query: str,
    top_k: Optional[int] = None,
    retrieval_method: str = "hybrid",
    expanded_queries: Optional[Sequence[str]] = None,
) -> List[RetrievalResult]:
    """执行 Multi-Query 检索，并用 RRF 融合多路召回结果。"""
    queries = list(expanded_queries or expand_multi_query(query))
    if not queries:
        return []

    resolved_top_k = max(1, min(top_k or settings.SEARCH_TOP_K, 50))
    candidate_k = max(resolved_top_k * 8, resolved_top_k + 20, MIN_RAG_CANDIDATE_POOL)
    resolved_method = _resolve_retrieval_method_for_query(query, retrieval_method)
    retriever = _build_retriever(candidate_k, retrieval_method=resolved_method)
    if retriever is None:
        return []

    result_sets: List[List[RetrievalResult]] = []
    for item in queries:
        results = retriever.retrieve(item, top_k=candidate_k)
        if results:
            result_sets.append(results)

    if not result_sets:
        return []

    fused_results = fuse_ranked_results(
        result_sets=result_sets,
        weights=[1.0] + [0.85 for _ in result_sets[1:]],
        source_labels=[f"query_{index + 1}" for index in range(len(result_sets))],
        rrf_k=60,
    )
    return fused_results[: max(resolved_top_k * 8, resolved_top_k + 20, MIN_RAG_CANDIDATE_POOL)]


def _citation_document_key(result: RetrievalResult) -> str:
    """引用资料优先按文档去重，避免同一文件多个片段挤占展示位。"""
    metadata = result.metadata or {}
    return str(metadata.get("document_id") or metadata.get("source_name") or result.content or "")


def _citation_chunk_key(result: RetrievalResult) -> tuple[str, str, Any]:
    metadata = result.metadata or {}
    return (
        str(metadata.get("document_id") or ""),
        str(metadata.get("source_name") or ""),
        metadata.get("chunk_index"),
    )


def _is_structured_lookup_query(query: str) -> bool:
    """判断是否是员工、部门、状态这类结构化字段查询。"""
    normalized = re.sub(r"\s+", "", (query or "").strip())
    if not normalized:
        return False
    if is_identifier_query(normalized):
        return True
    if len(normalized) > 32:
        return False
    return any(term in normalized for term in STRUCTURED_QUERY_HINT_TERMS)


def filter_retrieval_results(
    query: str,
    results: Sequence[RetrievalResult],
    top_k: Optional[int] = None,
) -> List[RetrievalResult]:
    """检索后过滤：保留高相关、去重后的候选引用资料。"""
    structured_lookup = _is_structured_lookup_query(query)
    max_citation_results = STRUCTURED_QUERY_CITATION_RESULTS if structured_lookup else MAX_CITATION_RESULTS
    resolved_top_k = max(1, min(top_k or settings.SEARCH_TOP_K, max_citation_results))
    candidates = [item for item in results if item and (item.content or "").strip()]
    if not candidates:
        return []

    reranked = rerank_results(query=query, results=candidates, top_k=len(candidates))
    if not reranked:
        return []

    keyword_matched = [item for item in reranked if passes_keyword_relevance_gate(query, item)]
    if not keyword_matched:
        return []

    best_score = max(float(item.score or 0.0) for item in keyword_matched)
    score_floor = max(MIN_FINAL_RELEVANCE_SCORE, best_score * 0.62)
    filtered = [item for item in keyword_matched if float(item.score or 0.0) >= score_floor]

    if not filtered and best_score >= MIN_FINAL_RELEVANCE_SCORE:
        filtered = keyword_matched[:1]

    deduplicated: List[RetrievalResult] = []
    seen_documents = set()
    seen_chunks = set()

    def append_candidate(item: RetrievalResult, enforce_document_diversity: bool) -> None:
        if len(deduplicated) >= resolved_top_k:
            return
        chunk_key = _citation_chunk_key(item)
        document_key = _citation_document_key(item)
        if chunk_key in seen_chunks:
            return
        if enforce_document_diversity and document_key in seen_documents:
            return
        seen_chunks.add(chunk_key)
        seen_documents.add(document_key)
        deduplicated.append(item)

    for item in filtered:
        append_candidate(item, enforce_document_diversity=not structured_lookup)
        if len(deduplicated) >= resolved_top_k:
            break

    return deduplicated


def assemble_rag_context(results: Sequence[RetrievalResult], max_chars: int = RAG_CONTEXT_MAX_CHARS) -> str:
    """把过滤后的检索片段组装成可注入 Prompt 的上下文。"""
    blocks: List[str] = []
    used_chars = 0

    for index, result in enumerate(results, start=1):
        metadata = result.metadata or {}
        source_name = metadata.get("source_name") or metadata.get("document_id") or "未知来源"
        chunk_index = metadata.get("chunk_index")
        header = f"[资料{index}] 来源: {source_name}"
        if chunk_index is not None:
            header += f" / chunk: {chunk_index}"
        content = re.sub(r"\s+", " ", (result.content or "").strip())
        block = f"{header}\n{content}"
        if used_chars + len(block) > max_chars:
            remaining = max_chars - used_chars
            if remaining <= 120:
                break
            block = block[:remaining].rstrip()
        blocks.append(block)
        used_chars += len(block)
        if used_chars >= max_chars:
            break

    return "\n\n".join(blocks)


def run_rag_workflow(
    query: str,
    top_k: Optional[int] = None,
    retrieval_method: str = "hybrid",
    llm: Optional[Any] = None,
) -> RagWorkflowResult:
    """执行完整 RAG 工具流程：Multi-Query、混合召回、RRF、重排、过滤、上下文组装。"""
    normalized_query = (query or "").strip()
    if not normalized_query:
        return RagWorkflowResult(query="")

    resolved_method = _resolve_retrieval_method_for_query(normalized_query, retrieval_method)
    expanded_queries = expand_multi_query(normalized_query, llm=llm)
    recalled_results = retrieve_documents_multi_query(
        normalized_query,
        top_k=top_k,
        retrieval_method=resolved_method,
        expanded_queries=expanded_queries,
    )
    filtered_results = filter_retrieval_results(
        normalized_query,
        recalled_results,
        top_k=top_k,
    )
    context = assemble_rag_context(filtered_results)
    logger_data = {
        "query": normalized_query,
        "identifier_query": is_identifier_query(normalized_query),
        "expanded_queries": expanded_queries,
        "recalled_count": len(recalled_results),
        "filtered_count": len(filtered_results),
        "best_score": max((float(item.score or 0.0) for item in filtered_results), default=0.0),
    }
    logger.info(f"RAG 检索诊断: {json.dumps(logger_data, ensure_ascii=False)}")
    return RagWorkflowResult(
        query=normalized_query,
        retrieval_method=resolved_method,
        expanded_queries=expanded_queries,
        results=filtered_results,
        context=context,
        message="检索完成" if filtered_results else NO_RESULTS_MESSAGE,
    )


def serialize_retrieval_results(results: List[RetrievalResult]) -> List[Dict[str, Any]]:
    """把检索结果转换成 API、工具和日志都能使用的普通字典。"""
    return [
        {
            "content": result.content,
            "metadata": result.metadata,
            "score": result.score,
            "source": result.source,
        }
        for result in results
    ]
