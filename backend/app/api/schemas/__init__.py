"""API 请求/响应模型。"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    document_id: str
    original_filename: str
    stored_filename: str
    file_path: str
    content_type: Optional[str] = None
    file_type: str
    file_size: int
    status: Literal["processing", "ready", "failed", "deleting", "deleted"]
    chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    skip: int
    limit: int


class DocumentContentResponse(BaseModel):
    document_id: str
    original_filename: str
    file_type: str
    status: str
    content: str
    content_type: Optional[str] = None
    updated_at: Optional[datetime] = None


class DocumentDeleteResponse(DocumentResponse):
    milvus_deleted: bool = False
    file_deleted: bool = False


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class RetrievalResultResponse(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    source: str = ""


class RetrievalResponse(BaseModel):
    query: str
    retrieval_method: Literal["hybrid", "dense", "sparse"]
    results: List[RetrievalResultResponse]
    top_k: int


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    history: Optional[List[ChatMessage]] = None
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=50)
    use_retrieval: bool = True
    retrieval_method: Literal["hybrid", "dense", "sparse"] = "hybrid"
    short_memory_strategy: Literal["window", "summary"] = "window"
    short_memory_n: int = Field(default=5, ge=1)
    short_memory_m: int = Field(default=10, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    provider: Optional[str] = None
    model: Optional[str] = None


class ChatResponse(BaseModel):
    query: str
    response: str
    sources: List[RetrievalResultResponse] = Field(default_factory=list)
    model: str


class ChatSettingsResponse(BaseModel):
    max_conversations: int


class ConversationDeleteResponse(BaseModel):
    conversation_id: str
    memory_deleted: bool = False


__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatSettingsResponse",
    "ConversationDeleteResponse",
    "DocumentDeleteResponse",
    "DocumentContentResponse",
    "DocumentListResponse",
    "DocumentResponse",
    "RetrievalRequest",
    "RetrievalResponse",
    "RetrievalResultResponse",
]
