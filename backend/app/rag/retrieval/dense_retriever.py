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


class DenseRetriever(BaseRetriever):
    """
    密集向量检索器

    基于向量相似度的语义检索，使用 Embedding 模型
    """

    def __init__(self, embeddings, top_k: int = 5, vector_store: Optional[Any] = None):
        """
        初始化密集检索器

        Args:
            embeddings: Embedding 模型实例
            top_k: 默认返回结果数
            vector_store: 可选的 MilvusClient 实例，便于测试或外部注入
        """
        super().__init__(top_k)
        self.embeddings = embeddings
        self.vector_store = vector_store

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
            if not client.has_collection(collection_name):
                return []

            if not is_collection_loaded(client, collection_name):
                logger.warning(f"collection {collection_name} 尚未加载完成，跳过本次密集检索")
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
                entity = getattr(hit, "entity", hit)
                content = (
                    getattr(entity, "chunk_text", None)
                    or getattr(entity, "text", None)
                    or getattr(hit, "chunk_text", None)
                    or getattr(hit, "text", None)
                    or ""
                )
                metadata = {
                    "document_id": getattr(entity, "document_id", getattr(hit, "document_id", None)),
                    "chunk_index": getattr(entity, "chunk_index", getattr(hit, "chunk_index", None)),
                    "source_name": getattr(entity, "source_name", getattr(hit, "source_name", None)),
                    "file_type": getattr(entity, "file_type", getattr(hit, "file_type", None)),
                    "content_type": getattr(entity, "content_type", getattr(hit, "content_type", None)),
                }
                score = float(getattr(hit, "distance", getattr(hit, "score", 0.0)) or 0.0)
                results.append(
                    RetrievalResult(
                        content=content,
                        metadata=metadata,
                        score=score,
                        source="dense",
                    )
                )

            return results

        except Exception as e:
            logger.error(f"密集检索失败: {str(e)}")
            raise
