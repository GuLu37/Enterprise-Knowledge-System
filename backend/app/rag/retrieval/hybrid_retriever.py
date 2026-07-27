"""混合检索器 (密集 + 稀疏)"""
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseRetriever, RetrievalResult
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def _make_result_key(result: RetrievalResult) -> Tuple[str, int, str]:
    metadata = result.metadata or {}
    document_id = str(metadata.get("document_id") or "")
    chunk_index = metadata.get("chunk_index")
    try:
        chunk_index_value = int(chunk_index)
    except (TypeError, ValueError):
        chunk_index_value = -1
    content = (result.content or "").strip()
    return document_id, chunk_index_value, content


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

            candidate_k = max(top_k * 2, top_k)
            dense_results = self._safe_retrieve(self.dense_retriever, query, candidate_k, "密集")
            sparse_results = self._safe_retrieve(self.sparse_retriever, query, candidate_k, "稀疏")

            if not dense_results and not sparse_results:
                return []

            fused: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

            for source_name, results, weight in (
                ("dense", dense_results, self.dense_weight),
                ("sparse", sparse_results, self.sparse_weight),
            ):
                for rank, result in enumerate(results, start=1):
                    key = _make_result_key(result)
                    contribution = weight / rank
                    current = fused.get(key)
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

            ranked = sorted(
                fused.values(),
                key=lambda item: (
                    -float(item["score"]),
                    item["result"].metadata.get("chunk_index", 0),
                ),
            )

            results: List[RetrievalResult] = []
            for item in ranked[:top_k]:
                base_result: RetrievalResult = item["result"]
                results.append(
                    RetrievalResult(
                        content=base_result.content,
                        metadata=base_result.metadata,
                        score=float(item["score"]),
                        source="hybrid",
                    )
                )

            return results
        except Exception as e:
            logger.error(f"混合检索失败: {str(e)}")
            raise
