"""检索器基类"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel


class RetrievalResult(BaseModel):
    """检索结果模型"""

    content: str
    metadata: Dict[str, Any]
    score: float = 0.0
    source: str = ""


class BaseRetriever(ABC):
    """检索器基类"""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    @abstractmethod
    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            检索结果列表
        """
        pass

    @abstractmethod
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """
        添加文档

        Args:
            content: 文档内容
            metadata: 文档元数据

        Returns:
            文档ID
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """
        删除文档

        Args:
            document_id: 文档ID

        Returns:
            删除是否成功
        """
        pass
