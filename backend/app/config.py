"""项目配置管理"""
import os
from typing import Optional, List
from dotenv import load_dotenv

# 加载 .env 文件（绝对路径，兼容任意工作目录）
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(_env_path, override=True)


class Settings:
    """应用配置 - 所有值从 .env 文件通过 os.getenv() 加载"""

    # ==================== 应用基础配置 ====================
    APP_NAME: str = os.getenv("APP_NAME")
    APP_VERSION: str = os.getenv("APP_VERSION")
    DEBUG: bool = os.getenv("DEBUG").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL")

    # ==================== 服务器配置 ====================
    SERVER_HOST: str = os.getenv("SERVER_HOST")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT"))
    API_PREFIX: str = os.getenv("API_PREFIX")
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS").strip("[]").replace('"', '').split(",")

    # ==================== LLM 配置 ====================
    # Ollama 配置（本地模型）
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL")

    # OpenAI 配置（可选）
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")

    # DeepSeek 配置（可选）
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL: Optional[str] = os.getenv("DEEPSEEK_BASE_URL")
    DEEPSEEK_MODEL: Optional[str] = os.getenv("DEEPSEEK_MODEL")

    # OpenRouter 配置（可选）
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_API_BASE: Optional[str] = os.getenv("OPENROUTER_API_BASE")

    # Anthropic 配置（可选）
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # LLM 通用参数
    # 温度: 0=确定性最强(代码/数据抽取), 1=默认平衡, 2=最有创意
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE"))
    # 最大输出TOKENS数量
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS"))
    # LLM请求超时时间（秒）
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT"))

    # ==================== 向量化配置 ====================
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION"))

    # ==================== 向量数据库配置 ====================
    MILVUS_HOST: str = os.getenv("MILVUS_HOST")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT"))
    MILVUS_DB_NAME: str = os.getenv("MILVUS_DB_NAME")
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME")

    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE")
    SEARCH_TOP_K: int = int(os.getenv("SEARCH_TOP_K"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD"))

    # ==================== 元数据数据库配置 ====================
    # SQLite（开发）/ PostgreSQL（生产）
    DATABASE_URL: str = os.getenv("DATABASE_URL")

    # ==================== 文件存储配置 ====================
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR")
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE"))
    ALLOWED_FILE_TYPES: List[str] = os.getenv("ALLOWED_FILE_TYPES").strip("[]").replace('"', '').split(",")

    # ==================== 文本处理配置 ====================
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP"))
    TEXT_SPLITTER_TYPE: str = os.getenv("TEXT_SPLITTER_TYPE")

    # ==================== 检索配置 ====================
    USE_DENSE_RETRIEVER: bool = os.getenv("USE_DENSE_RETRIEVER").lower() == "true"
    USE_SPARSE_RETRIEVER: bool = os.getenv("USE_SPARSE_RETRIEVER").lower() == "true"
    USE_HYBRID_RETRIEVER: bool = os.getenv("USE_HYBRID_RETRIEVER").lower() == "true"

    DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT"))
    SPARSE_WEIGHT: float = float(os.getenv("SPARSE_WEIGHT"))

    # ==================== 缓存配置 ====================
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE").lower() == "true"
    CACHE_DIR: str = os.getenv("CACHE_DIR")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL"))

    # ==================== 日志配置 ====================
    LOG_DIR: str = os.getenv("LOG_DIR")
    LOG_FILE_NAME: str = os.getenv("LOG_FILE_NAME")

    # ==================== Langsmith 配置（可选）====================
    LANGSMITH_TRACING: Optional[bool] = (
        os.getenv("LANGSMITH_TRACING").lower() == "true"
        if os.getenv("LANGSMITH_TRACING") else None
    )
    LANGSMITH_ENDPOINT: Optional[str] = os.getenv("LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: Optional[str] = os.getenv("LANGSMITH_PROJECT")


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings
