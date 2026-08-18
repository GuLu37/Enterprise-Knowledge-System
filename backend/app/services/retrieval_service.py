"""统一的文档检索业务服务。"""
import csv
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.config import settings
from app.rag.retrieval.base import RetrievalResult
from app.rag.retrieval.reranker import (
    extract_identifier_terms,
    fuse_ranked_results,
    build_query_keywords,
    is_reference_index_result,
    is_identifier_query,
    normalize_text,
    passes_keyword_relevance_gate,
    rerank_results,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

MIN_MULTI_QUERY_COUNT = 3
MAX_MULTI_QUERY_COUNT = 4
MAX_CITATION_RESULTS = 3
STRUCTURED_QUERY_CITATION_RESULTS = 5
RAG_CONTEXT_MAX_CHARS = 6500
MAX_STRUCTURED_CONTEXT_CHARS = 24000
MIN_FINAL_RELEVANCE_SCORE = 0.28
MIN_RAG_CANDIDATE_POOL = 60
NO_RESULTS_MESSAGE = "当前知识库未检索到足够相关资料"
MULTI_QUERY_SYSTEM_PROMPT = (
    "你是企业知识库检索查询扩展器。"
    "把用户问题改写为 3 到 5 条适合检索的查询，不回答问题本身。"
    "必须覆盖：原始问法、关键词短语、可能的文件名或标题表达、同义表达。"
    "如果问题包含编号、工号、产品号、人名、文档名或带书名号的名称，必须原样保留。"
    "只输出严格 JSON，格式：{\"queries\":[\"查询1\",\"查询2\",\"查询3\"]}"
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


def _identifier_key(value: str) -> str:
    """把编号压成大小写和分隔符无关的覆盖判断 key。"""
    return re.sub(r"[\s_-]+", "", (value or "")).lower()


def _canonical_identifier_terms(query: str) -> List[str]:
    """每个编号只保留一个代表值，避免大小写变体挤掉后面的编号。"""
    identifiers: List[str] = []
    seen = set()
    for item in extract_identifier_terms(query):
        key = _identifier_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        identifiers.append(item)
    return identifiers


def _result_contains_identifier(result: RetrievalResult, identifier: str) -> bool:
    metadata = result.metadata or {}
    search_text = "\n".join(
        str(part or "")
        for part in (
            result.content,
            metadata.get("source_name"),
            metadata.get("document_id"),
        )
    )
    return _identifier_key(identifier) in _identifier_key(search_text)


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
            # 内部候选召回追求高召回，相关性阈值交给后续重排/过滤阶段。
            enforce_similarity_threshold=False,
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

    # 中文没有空格分词时，直接拼接 build_query_terms 会产生“公司办、
    # 司办公、办公地”这类重叠 ngram，容易让扩展查询把无关分片带进 RRF。
    # 兜底扩展只保留去掉疑问词、语气词后的连续语义片段。
    semantic_query = re.sub(
        r"(根据|基于|结合|请问|麻烦|帮我|帮忙|查询|检索|搜索|查一下|查找|"
        r"公司|企业|每个|各个|所有|全部|分别|哪些|哪几|是谁|是什么|在哪里|在哪|"
        r"多少|如何|怎么|为什么|是否|有没有|可以吗|能否|吗|呢|吧|啊)",
        " ",
        focus or cleaned or normalized_query,
    )
    semantic_query = re.sub(r"[的之与和及在是有为从到对把将]", " ", semantic_query)
    semantic_terms = re.findall(r"[A-Za-z0-9]+|[一-鿿]{2,}", semantic_query)
    semantic_terms = list(dict.fromkeys(term.strip() for term in semantic_terms if term.strip()))
    if semantic_terms:
        candidates.append(" ".join(semantic_terms[:8]))

    return _normalize_query_variants(candidates, normalized_query, max_queries=max_queries)


def _identifier_query_variants(query: str) -> List[str]:
    """编号型查询保留原句，并为每个编号生成独立查询。"""
    normalized_query = re.sub(r"\s+", " ", (query or "").strip())
    candidates = [normalized_query]
    candidates.extend(_canonical_identifier_terms(normalized_query))
    return _normalize_query_variants(
        candidates,
        normalized_query,
        max_queries=max(MAX_MULTI_QUERY_COUNT, len(candidates)),
    )


def _load_structured_section_chunks(
    anchor: RetrievalResult,
) -> List[RetrievalResult]:
    """从向量库读取同一结构化区域的全部分片。

    旧索引可能没有在每个表格分片重复表头，单靠关键词过滤无法判断
    后续数据行属于哪个字段。这里仅在已经命中结构化锚点时读取同一文档
    的完整工作表区域，并按工作表标记确认边界，避免跨表拼接无关内容。
    """
    if not _is_structured_result(anchor):
        return []

    metadata = anchor.metadata or {}
    document_id = str(metadata.get("document_id") or "").strip()
    anchor_index = metadata.get("chunk_index")
    anchor_section = _structured_section_marker(anchor)
    if not document_id or anchor_index is None or not anchor_section:
        return []

    try:
        anchor_index = int(anchor_index)
    except (TypeError, ValueError):
        return []

    try:
        from app.storage.milvus_store import get_milvus_client

        client = get_milvus_client()
        escaped_document_id = document_id.replace("\\", "\\\\").replace('"', '\\"')
        query_kwargs = {
            "collection_name": settings.MILVUS_DOC_COLLECTION_NAME,
            "filter": f'document_id == "{escaped_document_id}"',
            "output_fields": [
                "chunk_text",
                "text",
                "document_id",
                "chunk_index",
                "source_name",
                "file_type",
                "content_type",
            ],
        }
        if hasattr(client, "query_iterator"):
            iterator = client.query_iterator(
                **query_kwargs,
                batch_size=1000,
                limit=-1,
            )
            rows = []
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    rows.extend(batch)
            finally:
                close = getattr(iterator, "close", None)
                if close:
                    close()
        else:
            # 兼容旧版客户端和测试替身；生产客户端使用上面的迭代器，
            # 因此不会受单次 query 的返回数量上限影响。
            rows = client.query(**query_kwargs, limit=10000)
    except Exception as exc:
        logger.debug("结构化分片邻域补取失败，继续使用已有召回结果: %s", exc)
        return []

    ordered_rows = sorted(
        rows or [],
        key=lambda row: int(row.get("chunk_index") or 0),
    )
    current_section = ""
    section_by_index: Dict[int, str] = {}
    for row in ordered_rows:
        try:
            chunk_index = int(row.get("chunk_index"))
        except (TypeError, ValueError):
            continue
        content = str(row.get("chunk_text") or row.get("text") or "")
        marker = _structured_section_marker(
            RetrievalResult(
                content=content,
                metadata={},
                score=0.0,
                source="structured",
            )
        )
        if marker:
            current_section = marker
        section_by_index[chunk_index] = current_section

    section_chunks: List[RetrievalResult] = []
    for row in ordered_rows:
        try:
            chunk_index = int(row.get("chunk_index"))
        except (TypeError, ValueError):
            continue
        if chunk_index == anchor_index:
            continue
        same_section = section_by_index.get(chunk_index) == anchor_section
        if not same_section:
            continue

        content = str(row.get("chunk_text") or row.get("text") or "").strip()
        if not content:
            continue
        section_chunks.append(
            RetrievalResult(
                content=content,
                metadata={
                    "document_id": row.get("document_id") or document_id,
                    "chunk_index": chunk_index,
                    "source_name": row.get("source_name") or metadata.get("source_name"),
                    "file_type": row.get("file_type") or metadata.get("file_type"),
                    "content_type": row.get("content_type") or metadata.get("content_type"),
                    "context_expansion": True,
                    "section_name": section_by_index.get(chunk_index),
                },
                score=max(float(anchor.score or 0.0) * 0.92, 0.0),
                source=anchor.source,
            )
        )

    return sorted(
        section_chunks,
        key=lambda item: int(item.metadata.get("chunk_index") or 0),
    )


def _compact_structured_section(
    anchor: RetrievalResult,
    section_chunks: Sequence[RetrievalResult],
    query: str,
) -> Optional[RetrievalResult]:
    """把较大的结构化区域压缩为相关字段和完整记录集合。

    结构化覆盖问题不能只取锚点附近的 chunk，也不能把整张大表原样
    塞入 Prompt。这里根据表头与查询词选择字段，跨全部分片收集记录，
    并去除切块重叠产生的重复行。
    """
    ordered_chunks = sorted(
        [anchor, *section_chunks],
        key=lambda item: int((item.metadata or {}).get("chunk_index") or 0),
    )
    header: Optional[List[str]] = None
    rows: List[List[str]] = []

    for result in ordered_chunks:
        for raw_line in str(result.content or "").splitlines():
            line = raw_line.strip()
            if not line or line.lower().startswith("[sheet]"):
                continue
            try:
                parsed = [str(cell).strip() for cell in next(csv.reader([line]))]
            except (csv.Error, StopIteration):
                continue
            if len(parsed) < 2:
                continue
            if header is None:
                header = parsed
                continue
            if [normalize_text(cell) for cell in parsed] == [
                normalize_text(cell) for cell in header
            ]:
                continue
            rows.append(parsed)

    if not header or not rows:
        return None

    query_keywords = build_query_keywords(query, max_terms=32)
    normalized_headers = [normalize_text(cell) for cell in header]
    selected_indexes = [0]
    for index, normalized_header in enumerate(normalized_headers):
        if index == 0:
            continue
        if any(
            keyword
            and (
                keyword in normalized_header
                or normalized_header in keyword
            )
            for keyword in query_keywords
        ):
            selected_indexes.append(index)

    if len(selected_indexes) == 1:
        selected_indexes.extend(range(1, min(len(header), 4)))
    selected_indexes = list(dict.fromkeys(selected_indexes))

    rendered_rows: List[str] = []
    seen_rows = set()
    for row in rows:
        row_key = tuple(normalize_text(cell) for cell in row)
        if not any(row_key) or row_key in seen_rows:
            continue
        seen_rows.add(row_key)
        values = [
            f"{header[index]}={row[index] if index < len(row) else ''}"
            for index in selected_indexes
            if index < len(header) and index < len(row) and row[index]
        ]
        if values:
            rendered_rows.append("；".join(values))

    if not rendered_rows:
        return None

    metadata = dict(anchor.metadata or {})
    metadata.update(
        {
            "structured_aggregation": True,
            "chunk_indices": [
                (item.metadata or {}).get("chunk_index")
                for item in ordered_chunks
            ],
            "section_name": _structured_section_marker(anchor),
        }
    )
    section_name = _structured_section_marker(anchor)
    content = (
        f"[Sheet] {section_name}\n"
        f"字段: {'、'.join(header[index] for index in selected_indexes)}\n"
        + "\n".join(rendered_rows)
    )
    return RetrievalResult(
        content=content,
        metadata=metadata,
        score=max(float(item.score or 0.0) for item in ordered_chunks),
        source="structured_evidence",
    )


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

    heuristic_variants = _heuristic_multi_query_variants(
        normalized_query,
        max_queries=max_queries,
    )
    # 大多数问题用规则扩展已经足够。只有问题过短或语义不足时才
    # 追加一次 LLM 改写，避免每次 RAG 请求都多消耗一次模型调用。
    if llm is None or len(heuristic_variants) >= 2:
        return heuristic_variants

    prompt = (
        f"{MULTI_QUERY_SYSTEM_PROMPT}\n\n"
        f"用户问题：{normalized_query}\n\n"
        f"请输出 3 到 5 条查询变体，避免长句，优先提升 source_name、标题、编号和正文关键词召回。"
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
    if not is_identifier_query(query):
        queries = queries[:MAX_MULTI_QUERY_COUNT]
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
        # 原始问题是最可靠的召回基线，扩展问题只负责补充表达，不能
        # 因为多个较弱变体的 RRF 累积而把原始命中挤出候选池。
        weights=[1.4] + [0.75 for _ in result_sets[1:]],
        source_labels=[f"query_{index + 1}" for index in range(len(result_sets))],
        rrf_k=60,
    )
    candidate_limit = max(
        resolved_top_k * 8,
        resolved_top_k + 20,
        MIN_RAG_CANDIDATE_POOL,
    )
    selected_results = list(fused_results[:candidate_limit])
    seen_keys = {_citation_chunk_key(item) for item in selected_results}

    # Multi-Query 的职责是扩召回，不应破坏原始问题已经召回的有效证据。
    # 如果原始查询结果因融合排序落到候选池之外，替换候选池尾部保留它。
    direct_results = result_sets[0]
    protected_direct_results = [
        item
        for item in direct_results[: max(resolved_top_k * 2, 12)]
        if item
        and (item.content or "").strip()
        and passes_keyword_relevance_gate(query, item)
    ]
    for item in protected_direct_results:
        key = _citation_chunk_key(item)
        if key in seen_keys:
            continue
        if len(selected_results) >= candidate_limit:
            removed = selected_results.pop()
            seen_keys.discard(_citation_chunk_key(removed))
        selected_results.append(item)
        seen_keys.add(key)

    return selected_results


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


def _is_structured_result(result: RetrievalResult) -> bool:
    metadata = result.metadata or {}
    file_type = str(metadata.get("file_type") or "").strip().lower().lstrip(".")
    return file_type in {"csv", "tsv", "xls", "xlsx"} or str(result.content or "").lstrip().startswith("[Sheet]")


def _structured_section_marker(result: RetrievalResult) -> str:
    first_line = next(
        (
            line.strip()
            for line in str(result.content or "").splitlines()
            if line.strip()
        ),
        "",
    )
    match = re.match(r"^\[Sheet\]\s*(.+?)\s*$", first_line, flags=re.IGNORECASE)
    return str(match.group(1) or "").strip().lower() if match else ""


def _structured_header_score(result: RetrievalResult) -> float:
    """按表头与查询的匹配度衡量结构化分片是否属于主证据区域。"""
    metadata = result.metadata or {}
    try:
        heading_coverage = float(metadata.get("heading_coverage") or 0.0)
    except (TypeError, ValueError):
        heading_coverage = 0.0
    try:
        heading_focus_coverage = float(metadata.get("heading_focus_coverage") or 0.0)
    except (TypeError, ValueError):
        heading_focus_coverage = 0.0
    return heading_coverage + heading_focus_coverage


def _select_primary_structured_anchors(
    anchors: Sequence[RetrievalResult],
) -> List[RetrievalResult]:
    """保留与查询表头最匹配的一个或多个结构化证据区域。"""
    if not anchors:
        return []

    scored = [(item, _structured_header_score(item)) for item in anchors]
    best_score = max(score for _, score in scored)
    if best_score <= 0.0:
        # 旧索引可能没有保存重排信号，至少保留最高原始相关结果所在区域。
        return [scored[0][0]]

    minimum_score = max(best_score * 0.5, 0.05)
    return [item for item, score in scored if score >= minimum_score]


def _same_structured_section(anchor: RetrievalResult, candidate: RetrievalResult) -> bool:
    """判断两个结构化分片是否仍属于同一连续表格区域。"""
    if not _is_structured_result(anchor) or not _is_structured_result(candidate):
        return False

    anchor_metadata = anchor.metadata or {}
    candidate_metadata = candidate.metadata or {}
    if str(anchor_metadata.get("document_id") or "") != str(candidate_metadata.get("document_id") or ""):
        return False

    anchor_marker = _structured_section_marker(anchor)
    candidate_marker = _structured_section_marker(candidate)
    if anchor_marker and candidate_marker:
        return anchor_marker == candidate_marker
    if candidate_marker and not anchor_marker:
        return False
    return True


def _expand_structured_evidence(
    results: Sequence[RetrievalResult],
    query: str,
    top_k: Optional[int] = None,
) -> List[RetrievalResult]:
    """把已命中的结构化锚点扩展成同一工作表的完整证据。"""
    structured_anchors = _select_primary_structured_anchors(
        [
            item
            for item in results
            if _is_structured_result(item) and _structured_section_marker(item)
        ]
    )
    if not structured_anchors:
        return list(results)

    citation_limit = max(1, min(top_k or settings.SEARCH_TOP_K, STRUCTURED_QUERY_CITATION_RESULTS))
    expanded_results: List[RetrievalResult] = []
    expanded_keys = set()
    expanded_sections = set()

    for result in structured_anchors:
        metadata = result.metadata or {}
        section_key = (
            str(metadata.get("document_id") or ""),
            _structured_section_marker(result),
        )
        if section_key in expanded_sections:
            continue
        expanded_sections.add(section_key)

        section_chunks = [result, *_load_structured_section_chunks(result)]
        if len(section_chunks) > max(citation_limit, 8):
            compacted = _compact_structured_section(
                anchor=result,
                section_chunks=section_chunks[1:],
                query=query,
            )
            if compacted is not None:
                expanded_results.append(compacted)
                continue

        for item in section_chunks:
            item_key = _citation_chunk_key(item)
            if item_key in expanded_keys:
                continue
            expanded_results.append(item)
            expanded_keys.add(item_key)

    return expanded_results


def filter_retrieval_results(
    query: str,
    results: Sequence[RetrievalResult],
    top_k: Optional[int] = None,
) -> List[RetrievalResult]:
    """检索后过滤：保留高相关、去重后的候选引用资料。"""
    identifiers = _canonical_identifier_terms(query)
    structured_lookup = _is_structured_lookup_query(query)
    max_citation_results = (
        STRUCTURED_QUERY_CITATION_RESULTS if structured_lookup else MAX_CITATION_RESULTS
    )
    requested_top_k = top_k or settings.SEARCH_TOP_K
    if identifiers:
        max_citation_results = max(max_citation_results, len(identifiers))
        requested_top_k = max(requested_top_k, len(identifiers))
    resolved_top_k = max(1, min(requested_top_k, max_citation_results))
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
    structured_matched = [item for item in keyword_matched if _is_structured_result(item)]
    if structured_lookup and structured_matched:
        filtered = structured_matched
    else:
        score_floor = max(MIN_FINAL_RELEVANCE_SCORE, best_score * 0.62)
        filtered = [item for item in keyword_matched if float(item.score or 0.0) >= score_floor]

        if not filtered and best_score >= MIN_FINAL_RELEVANCE_SCORE:
            filtered = keyword_matched[:1]

    # 多编号查询必须保留每个已命中的编号，避免一个高分片段挤掉其他编号。
    protected_identifier_results: List[RetrievalResult] = []
    for identifier in identifiers:
        matching = [
            item
            for item in keyword_matched
            if _result_contains_identifier(item, identifier)
        ]
        if matching:
            protected_identifier_results.append(matching[0])

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

    for item in protected_identifier_results:
        append_candidate(item, enforce_document_diversity=False)

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
    """执行完整 RAG 工具流程：Multi-Query、混合召回、RRF、重排、过滤、结构化补全、上下文组装。"""
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
    filtered_results = _expand_structured_evidence(
        filtered_results,
        query=normalized_query,
        top_k=top_k,
    )

    context_max_chars = RAG_CONTEXT_MAX_CHARS
    if any(
        bool((item.metadata or {}).get("structured_aggregation"))
        for item in filtered_results
    ):
        context_max_chars = MAX_STRUCTURED_CONTEXT_CHARS
    context = assemble_rag_context(
        filtered_results,
        max_chars=context_max_chars,
    )
    logger_data = {
        "query": normalized_query,
        "identifier_query": is_identifier_query(normalized_query),
        "expanded_queries": expanded_queries,
        "recalled_count": len(recalled_results),
        "filtered_count": len(filtered_results),
        "result_chunks": [
            {
                "source_name": (item.metadata or {}).get("source_name"),
                "document_id": (item.metadata or {}).get("document_id"),
                "chunk_index": (item.metadata or {}).get("chunk_index"),
                "chunk_count": len((item.metadata or {}).get("chunk_indices") or [])
                or 1,
                "structured_aggregation": bool(
                    (item.metadata or {}).get("structured_aggregation")
                ),
            }
            for item in filtered_results
        ],
        "context_chars": len(context),
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
            "metadata": {
                key: value
                for key, value in (result.metadata or {}).items()
                if not str(key).startswith("_")
            },
            "score": result.score,
            "source": result.source,
        }
        for result in results
    ]
