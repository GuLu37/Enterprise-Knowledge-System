"""项目配置管理"""
from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    """应用配置"""

    # ==================== 应用基础配置 ====================
    APP_NAME: str = "LangChain RAG Tutorial"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"  # development, staging, production
    LOG_LEVEL: str = "INFO"

    # ==================== 服务器配置 ====================
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    # ==================== LLM 配置 ====================
    # Ollama 配置（本地模型）
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"  # qwen, llama2, mistral 等

    # OpenAI 配置（可选）
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_BASE_URL: Optional[str] = None

    # DeepSeek 配置（可选）
    DEEPSEEK_API_KEY: Optional[str] = None

    # Anthropic 配置（可选）
    ANTHROPIC_API_KEY: Optional[str] = None

    # LLM 通用参数
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2000
    LLM_TIMEOUT: int = 60

    # ==================== 向量化配置 ====================
    # Ollama Embedding 配置
    EMBEDDING_MODEL: str = "nomic-embed-text"  # Ollama 向量模型
    EMBEDDING_BASE_URL: str = "http://localhost:11434"

    # 向量维度
    EMBEDDING_DIMENSION: int = 768

    # ==================== 向量数据库配置 ====================
    # Milvus 配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_DB_NAME: str = "rag_db"
    MILVUS_COLLECTION_NAME: str = "documents"

    # 向量库参数
    VECTOR_STORE_TYPE: str = "milvus"  # milvus, chroma, pinecone
    SEARCH_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.5

    # ==================== 元数据数据库配置 ====================
    # SQLite 配置（默认）
    DATABASE_URL: str = "sqlite:///./rag_metadata.db"

    # PostgreSQL 配置（可选，用于生产环境）
    # DATABASE_URL: str = "postgresql://user:password@localhost/rag_db"

    # ==================== 文件存储配置 ====================
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_TYPES: List[str] = [
        "pdf", "txt", "md", "doc", "docx", "ppt", "pptx", "xlsx", "xls"
    ]

    # ==================== 文本处理配置 ====================
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    TEXT_SPLITTER_TYPE: str = "recursive"  # recursive, token, semantic

    # ==================== 检索配置 ====================
    # 三重检索参数
    USE_DENSE_RETRIEVER: bool = True
    USE_SPARSE_RETRIEVER: bool = True  # BM25
    USE_HYBRID_RETRIEVER: bool = True

    # 检索权重
    DENSE_WEIGHT: float = 0.6
    SPARSE_WEIGHT: float = 0.4

    # ==================== 缓存配置 ====================
    ENABLE_CACHE: bool = True
    CACHE_DIR: str = "./data/cache"
    CACHE_TTL: int = 3600  # 1小时

    # ==================== 日志配置 ====================
    LOG_DIR: str = "./logs"
    LOG_FILE_NAME: str = "app.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings
