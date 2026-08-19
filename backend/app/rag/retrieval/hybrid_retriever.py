"""混合检索器 (密集 + 稀疏)"""
from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence

from .base import BaseRetriever, RetrievalResult
from .reranker import fuse_retrieval_results
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridRetriever(BaseRetriever):
    """直接融合 dense 和 sparse 的最简混合检索。"""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        top_k: int = 5,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        super().__init__(top_k)
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def _safe_retrieve(self, retriever: BaseRetriever, query: str, top_k: int, label: str) -> List[RetrievalResult]:
        try:
            return retriever.retrieve(query, top_k=top_k) or []
        except Exception as e:
            logger.warning(f"{label} 检索失败，跳过该路结果: {str(e)}")
            return []

    def _safe_retrieve_many(
        self,
        queries: Sequence[str],
        top_k: int,
    ) -> List[List[RetrievalResult]]:
        retrieve_many = getattr(self.dense_retriever, "retrieve_many", None)
        if not callable(retrieve_many):
            return [
                self._safe_retrieve(self.dense_retriever, query, top_k, "密集")
                for query in queries
            ]
        try:
            return retrieve_many(list(queries), top_k=top_k) or [
                [] for _ in queries
            ]
        except Exception as e:
            logger.warning(f"批量密集检索失败，跳过该路结果: {str(e)}")
            return [[] for _ in queries]

    def retrieve_many(
        self,
        queries: Sequence[str],
        top_k: int = None,
        sparse_max_workers: int = 3,
    ) -> List[List[RetrievalResult]]:
        """统一调度多查询：Dense 批量检索，Sparse 有界并行检索。"""
        resolved_top_k = top_k or self.top_k
        normalized_queries = [(query or "").strip() for query in queries]
        if not normalized_queries:
            return []

        worker_count = max(1, min(sparse_max_workers, len(normalized_queries)))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="rag-sparse",
        ) as executor:
            sparse_futures = [
                executor.submit(
                    self._safe_retrieve,
                    self.sparse_retriever,
                    query,
                    resolved_top_k,
                    "稀疏",
                )
                if query
                else None
                for query in normalized_queries
            ]
            dense_result_sets = self._safe_retrieve_many(
                normalized_queries,
                resolved_top_k,
            )
            sparse_result_sets = [
                future.result() if future is not None else []
                for future in sparse_futures
            ]

        merged_result_sets: List[List[RetrievalResult]] = []
        for index in range(len(normalized_queries)):
            dense_results = (
                dense_result_sets[index]
                if index < len(dense_result_sets)
                else []
            )
            sparse_results = (
                sparse_result_sets[index]
                if index < len(sparse_result_sets)
                else []
            )
            if not dense_results and not sparse_results:
                merged_result_sets.append([])
                continue
            merged_result_sets.append(
                fuse_retrieval_results(
                    dense_results=dense_results,
                    sparse_results=sparse_results,
                    dense_weight=self.dense_weight,
                    sparse_weight=self.sparse_weight,
                )[:resolved_top_k]
            )
        return merged_result_sets

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        normalized_query = (query or "").strip()
        if not normalized_query:
            return []
        return self.retrieve_many([normalized_query], top_k=top_k)[0]
