"""密集向量检索器 (语义相似度)"""
import time
from typing import Any, Dict, List, Optional

from .base import BaseRetriever, RetrievalResult
from app.config import settings
from app.storage.milvus_store import (
    is_collection_loaded,
    get_milvus_client,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
_COLLECTION_RECHECK_SECONDS = 5.0


def _get_field(source: Any, name: str, default: Any = None) -> Any:
    """兼容 Milvus 返回的 dict 和对象实体。"""
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _get_hit_entity(hit: Any) -> Any:
    """提取命中项中的实体字段。"""
    return _get_field(hit, "entity", hit)


class DenseRetriever(BaseRetriever):
    """
    密集向量检索器

    基于向量相似度的语义检索，使用 Embedding 模型
    """

    def __init__(
        self,
        embeddings,
        top_k: int = 5,
        vector_store: Optional[Any] = None,
        enforce_similarity_threshold: bool = True,
    ):
        """
        初始化密集检索器

        Args:
            embeddings: Embedding 模型实例
            top_k: 默认返回结果数
            vector_store: 可选的 MilvusClient 实例，便于测试或外部注入
            enforce_similarity_threshold: 是否在本次召回阶段执行相似度阈值
        """
        super().__init__(top_k)
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.enforce_similarity_threshold = enforce_similarity_threshold
        self._collection_ready: Optional[bool] = None
        self._last_collection_check_at = 0.0

    def _is_collection_ready(self, client: Any) -> bool:
        """缓存已加载状态；未就绪时定期复检以兼容后台初始化完成。"""
        now = time.monotonic()
        if self._collection_ready is True:
            return True
        if (
            self._collection_ready is False
            and now - self._last_collection_check_at < _COLLECTION_RECHECK_SECONDS
        ):
            return False

        collection_name = settings.MILVUS_CHILD_COLLECTION_NAME
        self._last_collection_check_at = now
        self._collection_ready = (
            client.has_collection(collection_name)
            and is_collection_loaded(client, collection_name)
        )
        if not self._collection_ready:
            logger.warning(f"collection {collection_name} 尚未加载完成，跳过本次密集检索")
        return self._collection_ready

    def _build_results(self, hits: Any, top_k: int) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        for hit in hits or []:
            entity = _get_hit_entity(hit)
            content = (
                _get_field(entity, "child_text")
                or _get_field(entity, "chunk_text")
                or _get_field(entity, "text")
                or _get_field(hit, "chunk_text")
                or _get_field(hit, "text")
                or ""
            )
            metadata = {
                "child_id": _get_field(entity, "child_id", _get_field(hit, "child_id")),
                "document_id": _get_field(entity, "document_id", _get_field(hit, "document_id")),
                "parent_id": _get_field(entity, "parent_id", _get_field(hit, "parent_id")),
                "parent_index": _get_field(entity, "parent_index", _get_field(hit, "parent_index")),
                "chunk_index": _get_field(entity, "chunk_index", _get_field(hit, "chunk_index")),
                "source_name": _get_field(entity, "source_name", _get_field(hit, "source_name")),
                "file_type": _get_field(entity, "file_type", _get_field(hit, "file_type")),
                "content_type": _get_field(entity, "content_type", _get_field(hit, "content_type")),
            }
            score = float(_get_field(hit, "distance", _get_field(hit, "score", 0.0)) or 0.0)
            if (
                self.enforce_similarity_threshold
                and settings.SIMILARITY_THRESHOLD
                and score < settings.SIMILARITY_THRESHOLD
            ):
                continue
            results.append(
                RetrievalResult(
                    content=content,
                    metadata=metadata,
                    score=score,
                    source="dense",
                )
            )
        return results[:top_k]

    def retrieve_many(self, queries: List[str], top_k: int = None) -> List[List[RetrievalResult]]:
        """批量生成查询向量并通过一次 Milvus Search 召回多路结果。"""
        resolved_top_k = top_k or self.top_k
        normalized_queries = [(query or "").strip() for query in queries]
        if not normalized_queries:
            return []

        valid_indexes = [
            index
            for index, query in enumerate(normalized_queries)
            if query
        ]
        result_sets: List[List[RetrievalResult]] = [[] for _ in normalized_queries]
        if not valid_indexes:
            return result_sets

        try:
            client = self.vector_store or get_milvus_client()
            if not self._is_collection_ready(client):
                return result_sets

            valid_queries = [normalized_queries[index] for index in valid_indexes]
            embed_queries = getattr(self.embeddings, "embed_queries", None)
            if callable(embed_queries):
                query_vectors = embed_queries(valid_queries)
            else:
                query_vectors = [
                    self.embeddings.embed_query(query)
                    for query in valid_queries
                ]

            search_result = client.search(
                collection_name=settings.MILVUS_CHILD_COLLECTION_NAME,
                data=query_vectors,
                anns_field="vector",
                limit=resolved_top_k,
                output_fields=[
                    "text",
                    "child_id",
                    "document_id",
                    "parent_id",
                    "parent_index",
                    "chunk_index",
                    "source_name",
                    "chunk_text",
                    "child_text",
                    "file_type",
                    "content_type",
                ],
                search_params={
                    "metric_type": "COSINE",
                    "params": {},
                },
            )

            if len(valid_queries) == 1 and (
                not search_result
                or not isinstance(search_result[0], list)
            ):
                hit_sets = [search_result or []]
            else:
                hit_sets = list(search_result or [])

            for result_index, query_index in enumerate(valid_indexes):
                hits = hit_sets[result_index] if result_index < len(hit_sets) else []
                result_sets[query_index] = self._build_results(hits, resolved_top_k)
            return result_sets
        except Exception as e:
            logger.error(f"批量密集检索失败: {str(e)}")
            raise

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        使用向量相似度检索

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            检索结果列表
        """

        normalized_query = (query or "").strip()
        if not normalized_query:
            return []
        logger.debug(f"密集检索: {normalized_query} (top_k: {top_k or self.top_k})")
        return self.retrieve_many([normalized_query], top_k=top_k)[0]
