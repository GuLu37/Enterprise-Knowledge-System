"""统一的文档检索业务服务。"""
from typing import Any, Dict, List, Optional

from app.config import settings
from app.rag.retrieval.base import RetrievalResult
from app.rag.retrieval.dense_retriever import DenseRetriever
from app.rag.retrieval.hybrid_retriever import HybridRetriever
from app.rag.retrieval.sparse_retriever import SparseRetriever


def _build_retriever(top_k: int):
    """根据配置构造当前启用的检索器。"""
    dense_retriever = None
    sparse_retriever = None

    if settings.USE_DENSE_RETRIEVER:
        from app.core.embeddings import get_default_embeddings

        dense_retriever = DenseRetriever(
            embeddings=get_default_embeddings(),
            top_k=top_k,
        )

    if settings.USE_SPARSE_RETRIEVER:
        sparse_retriever = SparseRetriever(top_k=top_k)

    if settings.USE_HYBRID_RETRIEVER and dense_retriever and sparse_retriever:
        return HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            top_k=top_k,
            dense_weight=settings.DENSE_WEIGHT,
            sparse_weight=settings.SPARSE_WEIGHT,
        )

    return dense_retriever or sparse_retriever


def retrieve_documents(query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
    """执行一次文档检索，返回统一的 RetrievalResult 列表。"""
    normalized_query = (query or "").strip()
    if not normalized_query:
        return []

    resolved_top_k = max(1, min(top_k or settings.SEARCH_TOP_K, 50))
    retriever = _build_retriever(resolved_top_k)
    if retriever is None:
        return []

    return retriever.retrieve(normalized_query, top_k=resolved_top_k)


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

