"""混合检索器 (密集 + 稀疏)"""
from typing import List, Dict, Any
from .base import BaseRetriever, RetrievalResult
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridRetriever(BaseRetriever):
    """
    混合检索器

    融合密集检索和稀疏检索的结果
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        top_k: int = 5,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        """
        初始化混合检索器

        Args:
            dense_retriever: 密集检索器实例
            sparse_retriever: 稀疏检索器实例
            top_k: 默认返回结果数
            dense_weight: 密集检索权重
            sparse_weight: 稀疏检索权重
        """
        super().__init__(top_k)
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        混合检索

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            融合后的检索结果列表
        """
        try:
            if top_k is None:
                top_k = self.top_k

            logger.debug(
                f"混合检索: {query} (top_k: {top_k}, "
                f"dense_w: {self.dense_weight}, sparse_w: {self.sparse_weight})"
            )

            # TODO: 实现混合检索逻辑
            # 1. 分别执行密集和稀疏检索
            # 2. 对结果进行权重融合
            # 3. 返回排序后的结果

            return []
        except Exception as e:
            logger.error(f"混合检索失败: {str(e)}")
            raise

    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        同时添加文档到两个检索器

        Args:
            content: 文档内容
            metadata: 文档元数据

        Returns:
            文档ID
        """
        try:
            logger.debug(f"添加文档到混合索引 (长度: {len(content)})")

            # 同时添加到密集和稀疏检索器
            doc_id = self.dense_retriever.add_document(content, metadata)
            self.sparse_retriever.add_document(content, metadata)

            return doc_id
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            raise

    def delete_document(self, document_id: str) -> bool:
        """
        同时从两个检索器删除文档

        Args:
            document_id: 文档ID

        Returns:
            删除是否成功
        """
        try:
            logger.debug(f"从混合索引删除文档: {document_id}")

            # 同时从两个检索器删除
            self.dense_retriever.delete_document(document_id)
            self.sparse_retriever.delete_document(document_id)

            return True
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            raise
