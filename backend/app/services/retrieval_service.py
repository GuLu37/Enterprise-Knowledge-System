"""统一的文档检索业务服务。"""
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.config import settings
from app.rag.retrieval.base import RetrievalResult
from app.rag.retrieval.reranker import (
    build_query_keywords,
    extract_identifier_terms,
    fuse_ranked_results,
    is_reference_index_result,
    is_identifier_query,
    normalize_text,
    passes_keyword_relevance_gate,
    rerank_results,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
_RETRIEVER_CACHE: Dict[str, Any] = {}
_RETRIEVER_CACHE_LOCK = threading.Lock()

MIN_MULTI_QUERY_COUNT = 3
MAX_MULTI_QUERY_COUNT = 4
MAX_CITATION_RESULTS = 3
STRUCTURED_QUERY_CITATION_RESULTS = 12
RAG_CONTEXT_MAX_CHARS = 6500
STRUCTURED_CONTEXT_MAX_CHARS = 30000
MIN_FINAL_RELEVANCE_SCORE = 0.28
MIN_RAG_CANDIDATE_POOL = 60
STRUCTURED_SECTION_PARENT_LIMIT = 200
STRUCTURED_SECTION_ROW_CHAR_LIMIT = 24000
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
STRUCTURED_SUMMARY_TERMS = (
    "统计",
    "总数",
    "总人数",
    "数量",
    "多少",
    "共有",
    "一共",
    "名单",
    "列表",
    "清单",
    "明细",
    "构成",
    "分别",
    "哪些",
    "有谁",
)
STRUCTURED_META_SCHEMA_TERMS = (
    "说明",
    "填表说明",
    "字段说明",
    "版本",
    "变更",
    "速查",
    "索引",
    "目录",
    "FAQ",
    "常见问题",
    "检索提示",
    "回答参考",
)
STRUCTURED_META_HEADER_TERMS = (
    "字段",
    "用途",
    "填写规则",
    "问题",
    "答案",
    "关键词",
    "优先命中表",
    "检索提示",
)
STRUCTURED_DATA_SHEET_TERMS = (
    "明细",
    "名单",
    "名册",
    "花名册",
    "通讯录",
    "清单",
    "台账",
    "总表",
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


def _candidate_pool_limit(top_k: int) -> int:
    """统一控制候选池大小，避免不同流程各自写一套上限公式。"""
    return max(top_k * 8, top_k + 20, MIN_RAG_CANDIDATE_POOL)


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
    cache_key = normalized_method
    with _RETRIEVER_CACHE_LOCK:
        cached_retriever = _RETRIEVER_CACHE.get(cache_key)
        if cached_retriever is not None:
            return cached_retriever

        retriever = _create_retriever(top_k, normalized_method)
        if retriever is not None:
            _RETRIEVER_CACHE[cache_key] = retriever
        return retriever


def _create_retriever(top_k: int, normalized_method: str):
    """创建可复用的检索器实例，运行时 top_k 仍由 retrieve 参数控制。"""
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


def _post_process_retrieval_results(
    query: str,
    results: Sequence[RetrievalResult],
    top_k: Optional[int] = None,
) -> List[RetrievalResult]:
    """统一收口检索结果：先过滤，再补结构化证据和父块上下文。"""
    filtered_results = filter_retrieval_results(
        query,
        results,
        top_k=top_k,
    )
    filtered_results = _expand_structured_section_evidence(
        filtered_results,
        query=query,
    )
    return _expand_parent_evidence(
        filtered_results,
        query=query,
    )


def retrieve_documents(
    query: str,
    top_k: Optional[int] = None,
    retrieval_method: str = "hybrid",
) -> List[RetrievalResult]:
    """执行一次文档检索，返回统一的 RetrievalResult 列表。"""
    normalized_query = (query or "").strip()
    if not normalized_query:
        return []

    resolved_method = _resolve_retrieval_method_for_query(normalized_query, retrieval_method)
    resolved_top_k = max(1, min(top_k or settings.SEARCH_TOP_K, 50))
    reranked = retrieve_documents_multi_query(
        normalized_query,
        top_k=resolved_top_k,
        retrieval_method=resolved_method,
    )
    return _post_process_retrieval_results(
        normalized_query,
        reranked,
        top_k=resolved_top_k,
    )


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

    structured_aliases = _structured_query_aliases(normalized_query)
    if structured_aliases:
        candidates.extend(structured_aliases)

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


def _structured_query_aliases(query: str) -> List[str]:
    """把员工、部门这类结构化问法展开成通用口径别名。"""
    normalized = re.sub(r"\s+", "", (query or "").strip())
    if not normalized:
        return []

    aliases: List[str] = []
    seen = set()

    def add(value: str) -> None:
        value = re.sub(r"\s+", "", (value or "").strip())
        if not value or value == normalized or value in seen:
            return
        seen.add(value)
        aliases.append(value)

    has_employee = any(term in normalized for term in ("员工", "人员", "人事", "花名册", "名册"))
    has_department = any(term in normalized for term in ("部门", "组织", "通讯录", "架构"))
    has_salary = any(term in normalized for term in ("工资", "薪资", "薪酬", "月薪", "应发", "实发", "奖金", "扣款"))
    has_count = any(term in normalized for term in ("统计", "总数", "总人数", "数量", "多少", "共有", "一共"))
    has_list = any(term in normalized for term in ("名单", "列表", "清单", "明细", "分别", "有哪些", "有谁", "是谁"))
    has_role = any(term in normalized for term in ("负责人", "主管", "上级"))

    if has_salary:
        add("工资明细")
        add("薪资明细")
        add("实发工资")
        add("应发工资")
        add("员工工资")
    if has_employee:
        add("员工名单")
        add("员工花名册")
        add("员工明细")
        if has_count:
            add("员工人数")
            add("员工总数")
            add("员工统计")
    if has_department:
        add("部门通讯录")
        add("部门负责人")
        add("部门名单")
        add("部门数量")
        add("部门统计")
    if has_role:
        add("负责人名单")
        add("各部门负责人")
    if has_count:
        if has_employee:
            add("在职员工人数")
        if has_department:
            add("部门数量统计")
    if has_list:
        if has_employee:
            add("员工明细")
        if has_department:
            add("部门清单")

    return aliases


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


def _load_parent_context(
    anchor: RetrievalResult,
) -> Optional[RetrievalResult]:
    """通过 parent_id 从 Milvus 读取对应 Parent 文本。"""
    metadata = anchor.metadata or {}
    parent_id = str(metadata.get("parent_id") or "").strip()
    if not parent_id:
        return None

    try:
        from app.storage.milvus_store import get_milvus_client

        client = get_milvus_client()
        escaped_parent_id = parent_id.replace("\\", "\\\\").replace('"', '\\"')
        parent_rows = client.query(
            collection_name=settings.MILVUS_PARENT_COLLECTION_NAME,
            filter=f'parent_id == "{escaped_parent_id}"',
            output_fields=[
                "parent_id",
                "document_id",
                "parent_index",
                "parent_text",
                "source_name",
                "file_type",
                "content_type",
            ],
            limit=1,
        )
        if not parent_rows:
            return None

        parent = parent_rows[0]
        parent_text = str(parent.get("parent_text") or "").strip()
        if not parent_text:
            return None

        return RetrievalResult(
            content=parent_text,
            metadata={
                **metadata,
                "document_id": parent.get("document_id") or metadata.get("document_id"),
                "parent_id": parent.get("parent_id") or parent_id,
                "parent_index": parent.get("parent_index") or metadata.get("parent_index"),
                "source_name": parent.get("source_name") or metadata.get("source_name"),
                "file_type": parent.get("file_type") or metadata.get("file_type"),
                "content_type": parent.get("content_type") or metadata.get("content_type"),
                "context_expansion": True,
            },
            score=float(anchor.score or 0.0),
            source=anchor.source,
        )
    except Exception as exc:
        logger.debug("Parent 上下文读取失败，继续使用原始命中: %s", exc)
        return None


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
    candidate_k = _candidate_pool_limit(resolved_top_k)
    resolved_method = _resolve_retrieval_method_for_query(query, retrieval_method)
    retriever = _build_retriever(candidate_k, retrieval_method=resolved_method)
    if retriever is None:
        return []

    retrieve_many = getattr(retriever, "retrieve_many", None)
    if callable(retrieve_many):
        result_sets = retrieve_many(queries, top_k=candidate_k)
    else:
        result_sets = [
            retriever.retrieve(item, top_k=candidate_k)
            for item in queries
        ]

    if not any(result_sets):
        return []

    nonempty_result_sets = [
        results
        for results in result_sets
        if results
    ]
    nonempty_indexes = [
        index
        for index, results in enumerate(result_sets)
        if results
    ]
    fused_results = fuse_ranked_results(
        result_sets=nonempty_result_sets,
        # 原始问题是最可靠的召回基线，扩展问题只负责补充表达，不能
        # 因为多个较弱变体的 RRF 累积而把原始命中挤出候选池。
        weights=[
            1.4 if index == 0 else 0.75
            for index in nonempty_indexes
        ],
        source_labels=[
            f"query_{index + 1}"
            for index in nonempty_indexes
        ],
        rrf_k=60,
    )
    candidate_limit = _candidate_pool_limit(resolved_top_k)
    selected_results = list(fused_results[:candidate_limit])
    seen_keys = {_citation_chunk_key(item) for item in selected_results}

    def keep_candidate(item: RetrievalResult) -> None:
        """把必须保留的候选塞回融合池尾部。"""
        key = _citation_chunk_key(item)
        if key in seen_keys:
            return
        if len(selected_results) >= candidate_limit:
            removed = selected_results.pop()
            seen_keys.discard(_citation_chunk_key(removed))
        selected_results.append(item)
        seen_keys.add(key)

    # Multi-Query 的职责是扩召回，不应破坏原始问题已经召回的有效证据。
    # 如果原始查询结果因融合排序落到候选池之外，替换候选池尾部保留它。
    direct_results = result_sets[0] if result_sets else []
    protected_direct_results = [
        item
        for item in direct_results[: max(resolved_top_k * 2, 12)]
        if item
        and (item.content or "").strip()
        and passes_keyword_relevance_gate(query, item)
    ]
    for item in protected_direct_results:
        keep_candidate(item)

    if _is_structured_lookup_query(query):
        # 统计/名单类表格问题里，真正的数据表可能只被“员工花名册/
        # 部门通讯录/工资明细”某个变体强命中；RRF 会偏向多路擦边命中
        # 的说明页。每个变体保留少量 schema 明确匹配的数据行，保证后续
        # 同 Sheet 聚合有机会展开完整证据。
        for results in result_sets:
            kept_for_route = 0
            for item in results:
                if not _is_structured_result(item):
                    continue
                if is_reference_index_result(item):
                    continue
                if _structured_schema_score(item, query) <= 0.12:
                    continue
                keep_candidate(item)
                kept_for_route += 1
                if kept_for_route >= 3:
                    break

    return selected_results


def _citation_chunk_key(result: RetrievalResult) -> tuple[str, str, Any]:
    metadata = result.metadata or {}
    return (
        str(metadata.get("document_id") or ""),
        str(metadata.get("source_name") or ""),
        metadata.get("child_id") or metadata.get("chunk_index"),
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


def _is_structured_summary_query(query: str) -> bool:
    """判断是否需要跨多行表格证据来回答。"""
    normalized = re.sub(r"\s+", "", (query or "").strip())
    return _is_structured_lookup_query(normalized) and any(
        term in normalized
        for term in STRUCTURED_SUMMARY_TERMS
    )


def _structured_section_marker_from_text(text: str) -> str:
    """从结构化 chunk/parent 中提取 sheet 名。"""
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^\[Sheet\]\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""
    return ""


def _structured_section_marker(result: RetrievalResult) -> str:
    return _structured_section_marker_from_text(result.content)


def _structured_schema_text(result: RetrievalResult) -> str:
    """取结构化结果的 sheet 名和表头，用于判断表格是否对题。"""
    lines = [
        line.strip()
        for line in str(result.content or "").splitlines()
        if line.strip()
    ]
    return "\n".join(lines[:2])


def _structured_schema_parts(result: RetrievalResult) -> tuple[str, str]:
    """拆出 sheet 名和表头，后续只用结构信息判断表格是否对题。"""
    lines = [
        line.strip()
        for line in str(result.content or "").splitlines()
        if line.strip()
    ]
    if not lines:
        return "", ""

    sheet_name = ""
    header = lines[0]
    if lines[0].startswith("[Sheet]"):
        sheet_name = re.sub(r"^\[Sheet\]\s*", "", lines[0], flags=re.IGNORECASE).strip()
        header = lines[1] if len(lines) > 1 else ""
    return sheet_name, header


def _structured_query_schema_terms(query: str) -> List[str]:
    """按查询主题生成可迁移的表头/Sheet 匹配词。"""
    normalized = re.sub(r"\s+", "", (query or "").strip())
    terms: List[str] = []

    def extend(values: Sequence[str]) -> None:
        for value in values:
            if value and value not in terms:
                terms.append(value)

    has_employee = any(term in normalized for term in ("员工", "人员", "人事", "花名册", "名册", "姓名", "工号"))
    has_department = any(term in normalized for term in ("部门", "组织", "通讯录", "架构", "负责人", "主管"))
    has_salary = any(term in normalized for term in ("工资", "薪资", "薪酬", "月薪", "应发", "实发", "奖金", "扣款"))

    if has_salary:
        extend(("工资", "薪资", "薪酬", "基本工资", "应发工资", "实发工资", "奖金", "扣款", "个税", "社保", "公积金"))
    if has_employee:
        extend(("员工", "人员", "员工号", "工号", "姓名", "员工姓名", "部门", "岗位", "职位", "状态", "入职", "邮箱", "电话"))
    if has_department:
        extend(("部门", "负责人", "主管", "直属主管", "办公地点", "办公", "地点", "联系人", "电话", "邮箱"))

    extend(build_query_keywords(query, max_terms=32))
    return [term for term in terms if len(normalize_text(term)) >= 2]


def _structured_schema_score(result: RetrievalResult, query: str) -> float:
    """按 sheet/表头字段匹配度给结构化候选打分。"""
    sheet_name, header = _structured_schema_parts(result)
    schema_text = normalize_text("\n".join(part for part in (sheet_name, header) if part))
    if not schema_text:
        return 0.0

    sheet_text = normalize_text(sheet_name)
    header_text = normalize_text(header)
    useful_terms = [
        normalize_text(term)
        for term in _structured_query_schema_terms(query)
        if normalize_text(term)
        and normalize_text(term) not in {"名单", "列表", "清单", "统计", "总数", "总人数", "数量", "多少", "共有", "一共"}
    ]
    if not useful_terms:
        return 0.0

    hit_weight = 0.0
    max_weight = 0.0
    for term in list(dict.fromkeys(useful_terms)):
        weight = min(len(term), 8) / 8.0
        max_weight += weight
        if term in header_text:
            hit_weight += weight
        elif term in sheet_text:
            hit_weight += weight * 0.65

    score = hit_weight / max_weight if max_weight > 0.0 else 0.0
    meta_hits = sum(
        1
        for term in STRUCTURED_META_SCHEMA_TERMS
        if normalize_text(term) and normalize_text(term) in schema_text
    )
    meta_header_hits = sum(
        1
        for term in STRUCTURED_META_HEADER_TERMS
        if normalize_text(term) and normalize_text(term) in header_text
    )
    data_sheet_hit = any(
        normalize_text(term) in sheet_text
        for term in STRUCTURED_DATA_SHEET_TERMS
        if normalize_text(term)
    )
    data_field_hits = sum(1 for term in useful_terms if term in header_text)

    if data_sheet_hit and data_field_hits >= 2:
        score += 0.15
    if meta_hits + meta_header_hits >= 2 and data_field_hits < 2:
        score *= 0.25
    return min(1.0, score)


def _escape_milvus_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _query_parent_rows_by_document(document_id: str) -> List[Dict[str, Any]]:
    """按 document_id 读取 parent，避免运行时扫全库。"""
    if not document_id:
        return []
    try:
        from app.storage.milvus_store import get_milvus_client

        client = get_milvus_client()
        if not client.has_collection(settings.MILVUS_PARENT_COLLECTION_NAME):
            return []
        rows = client.query(
            collection_name=settings.MILVUS_PARENT_COLLECTION_NAME,
            filter=f'document_id == "{_escape_milvus_value(document_id)}"',
            output_fields=[
                "parent_id",
                "document_id",
                "parent_index",
                "parent_text",
                "source_name",
                "file_type",
                "content_type",
            ],
            limit=10000,
        )
        return list(rows or [])
    except Exception as exc:
        logger.debug("结构化 parent 扩展读取失败，继续使用原始命中: %s", exc)
        return []


def _parse_structured_table(parent_texts: Sequence[str]) -> tuple[str, List[str], List[str], List[int]]:
    """从多个 parent 中还原同一张 sheet 的表头与去重数据行。"""
    sheet_name = ""
    header = ""
    rows: List[str] = []
    seen_rows = set()
    parent_indexes: List[int] = []

    for parent_text in parent_texts:
        current_sheet = ""
        lines = [
            line.strip()
            for line in str(parent_text or "").splitlines()
            if line.strip()
        ]
        for line in lines:
            match = re.match(r"^\[Sheet\]\s*(.+?)\s*$", line, flags=re.IGNORECASE)
            if match:
                current_sheet = match.group(1).strip()
                if not sheet_name:
                    sheet_name = current_sheet
                continue
            if not header:
                header = line
                continue
            if normalize_text(line) == normalize_text(header):
                continue
            row_key = normalize_text(line)
            if not row_key or row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            rows.append(line)
        if current_sheet and current_sheet == sheet_name:
            parent_indexes.append(len(parent_indexes))

    return sheet_name, [item.strip() for item in header.split(",") if item.strip()], rows, parent_indexes


def _row_matches_query(row: str, query: str) -> bool:
    """粗粒度判断某行是否包含用户限定条件，命中行优先展示。"""
    keywords = build_query_keywords(query, max_terms=32)
    normalized_row = normalize_text(row)
    useful_keywords = [
        keyword
        for keyword in keywords
        if keyword
        and keyword not in {"员工", "人员", "部门", "名单", "列表", "清单", "统计", "总数", "数量"}
        and len(keyword) >= 2
    ]
    if not useful_keywords:
        return False
    return any(keyword in normalized_row for keyword in useful_keywords)


def _build_structured_section_summary(
    anchor: RetrievalResult,
    parent_rows: Sequence[Dict[str, Any]],
    query: str,
) -> Optional[RetrievalResult]:
    """把同一 sheet 的 parent 压缩成统计/名单类问题可读的证据。"""
    section_name = _structured_section_marker(anchor)
    metadata = anchor.metadata or {}
    if not section_name:
        return None

    same_section_rows = sorted(
        [
            row
            for row in parent_rows
            if _structured_section_marker_from_text(str(row.get("parent_text") or "")) == section_name
        ],
        key=lambda row: int(row.get("parent_index") or 0),
    )[:STRUCTURED_SECTION_PARENT_LIMIT]
    if not same_section_rows:
        return None

    sheet_name, headers, rows, _ = _parse_structured_table(
        [str(row.get("parent_text") or "") for row in same_section_rows]
    )
    if not rows:
        return None

    matched_rows = [row for row in rows if _row_matches_query(row, query)]
    ordered_rows = matched_rows + [row for row in rows if row not in set(matched_rows)]
    rendered_rows: List[str] = []
    used_chars = 0
    for row in ordered_rows:
        next_len = len(row) + 1
        if used_chars + next_len > STRUCTURED_SECTION_ROW_CHAR_LIMIT:
            break
        rendered_rows.append(row)
        used_chars += next_len

    if not rendered_rows:
        return None

    content = (
        f"[Sheet] {sheet_name or section_name}\n"
        f"字段: {'、'.join(headers) if headers else '未知'}\n"
        f"去重记录数: {len(rows)}\n"
        f"优先匹配行数: {len(matched_rows)}\n"
        "数据行:\n"
        + "\n".join(rendered_rows)
    )
    return RetrievalResult(
        content=content,
        metadata={
            **metadata,
            "structured_aggregation": True,
            "section_name": sheet_name or section_name,
            "chunk_indices": [
                row.get("parent_index")
                for row in same_section_rows
            ],
        },
        score=max(float(anchor.score or 0.0), 0.0) + 0.08,
        source="structured_evidence",
    )


def _expand_parent_evidence(
    results: Sequence[RetrievalResult],
    query: str,
) -> List[RetrievalResult]:
    """保留 Child 命中，并把 Parent 上下文挂到元数据里供 Prompt 使用。"""
    if not results:
        return list(results)

    expanded_results: List[RetrievalResult] = []
    seen_children = set()

    for result in results:
        if (result.metadata or {}).get("structured_aggregation"):
            expanded_results.append(result)
            continue

        metadata = result.metadata or {}
        child_key = (
            str(metadata.get("document_id") or ""),
            str(metadata.get("child_id") or metadata.get("chunk_index") or ""),
            str((result.content or "").strip()),
        )
        if child_key in seen_children:
            continue
        seen_children.add(child_key)

        parent_result = _load_parent_context(result)
        metadata_with_parent = dict(metadata)
        if parent_result and (parent_result.content or "").strip():
            metadata_with_parent["parent_context_text"] = parent_result.content
            metadata_with_parent["context_expansion"] = True

        expanded_results.append(
            RetrievalResult(
                content=result.content,
                metadata=metadata_with_parent,
                score=result.score,
                source=result.source,
            )
        )

    return expanded_results


def _expand_structured_section_evidence(
    results: Sequence[RetrievalResult],
    query: str,
) -> List[RetrievalResult]:
    """结构化文档命中 sheet 后，补齐同一 sheet 的多行 parent 证据。

    xlsx/csv 的一行通常只是最小事实单元。用户问统计、筛选、名单、
    范围比较时，单个 child 命中不足以让 LLM 得到完整答案，所以只要
    已经召回到结构化 sheet，就把同 sheet 的 parent 行聚合成可读证据。
    """
    if not any(_is_structured_result(result) for result in results):
        return list(results)

    expanded: List[RetrievalResult] = []
    seen_sections = set()
    parent_cache: Dict[str, List[Dict[str, Any]]] = {}

    for result in results:
        if not _is_structured_result(result) or is_reference_index_result(result):
            expanded.append(result)
            continue
        metadata = result.metadata or {}
        document_id = str(metadata.get("document_id") or "").strip()
        section_name = _structured_section_marker(result)
        section_key = (document_id, section_name)
        if not document_id or not section_name or section_key in seen_sections:
            expanded.append(result)
            continue

        seen_sections.add(section_key)
        if document_id not in parent_cache:
            parent_cache[document_id] = _query_parent_rows_by_document(document_id)
        summary = _build_structured_section_summary(
            anchor=result,
            parent_rows=parent_cache[document_id],
            query=query,
        )
        expanded.append(summary or result)

    return expanded


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
    # 关键词相关性只做优先级信号，不做硬拒绝。否则同义词、数值范围、
    # 表格字段表达不一致时，会把 Dense/RRF 已经召回的候选直接清零。
    ranking_pool = list(reranked) if structured_lookup else (keyword_matched or list(reranked))

    best_score = max(float(item.score or 0.0) for item in ranking_pool)
    structured_matched = [item for item in ranking_pool if _is_structured_result(item)]
    if structured_lookup and structured_matched:
        non_reference_structured = [
            item
            for item in structured_matched
            if not is_reference_index_result(item)
        ]
        structured_pool = non_reference_structured or structured_matched
        schema_scores = [
            _structured_schema_score(item, query)
            for item in structured_pool
        ]
        best_schema_score = max(schema_scores, default=0.0)
        if best_schema_score > 0.0:
            minimum_schema_score = max(0.25, best_schema_score * 0.5)
            filtered = [
                item
                for item, schema_score in zip(structured_pool, schema_scores)
                if schema_score >= minimum_schema_score
            ]
            filtered.sort(
                key=lambda item: (
                    -_structured_schema_score(item, query),
                    -float(item.score or 0.0),
                )
            )
        else:
            filtered = structured_pool
    elif not keyword_matched:
        filtered = ranking_pool[:resolved_top_k]
    else:
        score_floor = max(MIN_FINAL_RELEVANCE_SCORE, best_score * 0.62)
        filtered = [item for item in ranking_pool if float(item.score or 0.0) >= score_floor]

        if not filtered and best_score >= MIN_FINAL_RELEVANCE_SCORE:
            filtered = ranking_pool[:1]

    # 多编号查询必须保留每个已命中的编号，避免一个高分片段挤掉其他编号。
    protected_identifier_results: List[RetrievalResult] = []
    for identifier in identifiers:
        matching = [
            item
            for item in ranking_pool
            if _result_contains_identifier(item, identifier)
        ]
        if matching:
            protected_identifier_results.append(matching[0])

    deduplicated: List[RetrievalResult] = []
    seen_chunks = set()

    def append_candidate(item: RetrievalResult) -> None:
        if len(deduplicated) >= resolved_top_k:
            return
        chunk_key = _citation_chunk_key(item)
        if structured_lookup:
            metadata = item.metadata or {}
            chunk_key = (
                str(metadata.get("document_id") or ""),
                str(metadata.get("parent_id") or ""),
                str(metadata.get("child_id") or metadata.get("chunk_index") or ""),
            )
        if chunk_key in seen_chunks:
            return
        seen_chunks.add(chunk_key)
        deduplicated.append(item)

    for item in protected_identifier_results:
        append_candidate(item)

    for item in filtered:
        append_candidate(item)
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
        parent_context = re.sub(
            r"\s+",
            " ",
            str(metadata.get("parent_context_text") or "").strip(),
        )
        if parent_context and parent_context != content:
            content = f"[命中行] {content}\n[父块上下文] {parent_context}"
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
    """执行完整 RAG 工具流程：Multi-Query、混合召回、RRF、重排、Parent 回收、上下文组装。"""
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
    filtered_results = _post_process_retrieval_results(
        normalized_query,
        recalled_results,
        top_k=top_k,
    )

    context_max_chars = (
        STRUCTURED_CONTEXT_MAX_CHARS
        if any((item.metadata or {}).get("structured_aggregation") for item in filtered_results)
        else RAG_CONTEXT_MAX_CHARS
    )
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
                "parent_id": (item.metadata or {}).get("parent_id"),
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
