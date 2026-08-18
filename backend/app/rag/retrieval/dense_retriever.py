"""密集向量检索器 (语义相似度)"""
from typing import Any, Dict, List, Optional

from .base import BaseRetriever, RetrievalResult
from app.config import settings
from app.storage.milvus_store import (
    is_collection_loaded,
    get_milvus_client,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


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

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        使用向量相似度检索

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            检索结果列表
        """

        try:
            if top_k is None:
                top_k = self.top_k

            query = (query or "").strip()
            if not query:
                return []

            logger.debug(f"密集检索: {query} (top_k: {top_k})")

            client = self.vector_store or get_milvus_client()
            collection_name = settings.MILVUS_DOC_COLLECTION_NAME
            if self._collection_ready is None:
                self._collection_ready = (
                    client.has_collection(collection_name)
                    and is_collection_loaded(client, collection_name)
                )
                if not self._collection_ready:
                    logger.warning(f"collection {collection_name} 尚未加载完成，跳过本次密集检索")
            if not self._collection_ready:
                return []

            query_vector = self.embeddings.embed_query(query)
            search_result = client.search(
                collection_name=collection_name,
                data=[query_vector],
                anns_field="vector",
                limit=top_k,
                output_fields=[
                    "text",
                    "document_id",
                    "chunk_index",
                    "source_name",
                    "chunk_text",
                    "file_type",
                    "content_type",
                ],
                search_params={
                    "metric_type": "COSINE",
                    "params": {},
                },
            )

            hits = search_result[0] if search_result and isinstance(search_result[0], list) else search_result or []

            results: List[RetrievalResult] = []
            for hit in hits:
                entity = _get_hit_entity(hit)
                content = (
                    _get_field(entity, "chunk_text")
                    or _get_field(entity, "text")
                    or _get_field(hit, "chunk_text")
                    or _get_field(hit, "text")
                    or ""
                )
                metadata = {
                    "document_id": _get_field(entity, "document_id", _get_field(hit, "document_id")),
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

            if not results:
                return []

            return results[:top_k]

        except Exception as e:
            logger.error(f"密集检索失败: {str(e)}")
            raise
