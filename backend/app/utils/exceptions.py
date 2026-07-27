"""自定义异常"""


class RAGException(Exception):
    """RAG 系统基类异常"""
    def __init__(self, message: str, code: int = 500):
        self.message = message
        self.code = code
        super().__init__(self.message)


class DocumentException(RAGException):
    """文档处理异常"""
    def __init__(self, message: str):
        super().__init__(message, 400)


class EmbeddingException(RAGException):
    """向量化异常"""
    def __init__(self, message: str):
        super().__init__(message, 500)


class RetrievalException(RAGException):
    """检索异常"""
    def __init__(self, message: str):
        super().__init__(message, 500)


class LLMException(RAGException):
    """LLM 调用异常"""
    def __init__(self, message: str):
        super().__init__(message, 500)


class VectorStoreException(RAGException):
    """向量库异常"""
    def __init__(self, message: str):
        super().__init__(message, 500)


class ConfigException(RAGException):
    """配置异常"""
    def __init__(self, message: str):
        super().__init__(message, 500)
