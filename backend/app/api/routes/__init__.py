"""API 路由模块"""
from .documents import router as documents_router
from .retrieval import router as retrieval_router
from .chat import router as chat_router
from .health import router as health_router

__all__ = [
    "documents_router",
    "retrieval_router",
    "chat_router",
    "health_router",
]
