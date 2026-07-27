"""检索器基类"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
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
