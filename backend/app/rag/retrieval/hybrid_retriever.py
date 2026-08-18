"""混合检索器 (密集 + 稀疏)"""
from concurrent.futures import ThreadPoolExecutor
from typing import List

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

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        try:
            if top_k is None:
                top_k = self.top_k

            query = (query or "").strip()
            if not query:
                return []

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-retrieval") as executor:
                dense_future = executor.submit(
                    self._safe_retrieve,
                    self.dense_retriever,
                    query,
                    top_k,
                    "密集",
                )
                sparse_future = executor.submit(
                    self._safe_retrieve,
                    self.sparse_retriever,
                    query,
                    top_k,
                    "稀疏",
                )
                dense_results = dense_future.result()
                sparse_results = sparse_future.result()

            if not dense_results and not sparse_results:
                return []

            fused_results = fuse_retrieval_results(
                dense_results=dense_results,
                sparse_results=sparse_results,
                dense_weight=self.dense_weight,
                sparse_weight=self.sparse_weight,
            )
            return fused_results[:top_k]
        except Exception as e:
            logger.error(f"混合检索失败: {str(e)}")
            raise
