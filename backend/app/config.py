"""项目配置管理"""
import json
import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

# 加载 .env 文件（绝对路径，兼容任意工作目录）
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
_env_path = BACKEND_DIR / ".env"
load_dotenv(_env_path, override=True)


def _resolve_app_path(path_value: Optional[str], default: str) -> str:
    """把应用内部路径固定解析到 backend/app 下，避免受启动目录影响。"""
    path = Path(path_value or default)
    if not path.is_absolute():
        path = APP_DIR / path
    return str(path.resolve())


def _parse_list(value: Optional[str], default: str = "") -> List[str]:
    """解析 JSON 数组配置，并统一清理空白和引号。"""
    raw_value = (value or default).strip()
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = raw_value.strip("[]").split(",") if raw_value else []

    if not isinstance(parsed, list):
        parsed = [parsed]
    return [str(item).strip().strip('"').strip("'") for item in parsed if str(item).strip()]


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
    CORS_ORIGINS: List[str] = _parse_list(
        os.getenv("CORS_ORIGINS"),
        '["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080"]',
    )

    # ==================== LLM 配置 ====================
    # 默认 LLM 提供商
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")

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
    OPENROUTER_MODEL: Optional[str] = os.getenv("OPENROUTER_MODEL")

    # Anthropic 配置（可选）
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: Optional[str] = os.getenv("ANTHROPIC_MODEL")

    # LLM 通用参数
    # 温度: 0=确定性最强(代码/数据抽取), 1=默认平衡, 2=最有创意
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE"))
    # 最大输出TOKENS数量
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS"))
    # LLM请求超时时间（秒）
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT"))

    # ==================== 聊天配置 ====================
    # 前端可保留的最大对话线程数，防止本地历史过多导致页面和请求阻塞。
    CHAT_MAX_CONVERSATIONS: int = int(os.getenv("CHAT_MAX_CONVERSATIONS"))

    # ==================== 向量化配置 ====================
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION"))
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "bge")

    # BGE 本地向量化配置
    BGE_MODEL_NAME: str = os.getenv("BGE_MODEL_NAME", "BAAI/bge-base-zh-v1.5")
    BGE_DEVICE: str = os.getenv("BGE_DEVICE", "auto")
    BGE_MAX_LENGTH: int = int(os.getenv("BGE_MAX_LENGTH", "512"))
    BGE_BATCH_SIZE: int = int(os.getenv("BGE_BATCH_SIZE", "16"))
    BGE_NORMALIZE_EMBEDDINGS: bool = os.getenv("BGE_NORMALIZE_EMBEDDINGS", "true").lower() == "true"
    BGE_QUERY_INSTRUCTION: str = os.getenv("BGE_QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关文章：")
    # BGE 模型缓存目录；为空时使用 HuggingFace/Transformers 默认缓存目录
    BGE_CACHE_DIR: Optional[str] = os.getenv("BGE_CACHE_DIR") or None
    # 是否只从本地读取模型；true 时不会尝试联网下载 HuggingFace 模型
    BGE_LOCAL_FILES_ONLY: bool = os.getenv("BGE_LOCAL_FILES_ONLY", "false").lower() == "true"

    # ==================== 向量数据库配置 ====================
    MILVUS_HOST: str = os.getenv("MILVUS_HOST")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT"))
    MILVUS_DB_NAME: str = os.getenv("MILVUS_DB_NAME")
    # 文档向量和长期记忆分开存储，分别使用不同的 collection
    MILVUS_DOC_COLLECTION_NAME: str = os.getenv(
        "MILVUS_DOC_COLLECTION_NAME",
        "doc_chunks",
    )
    MILVUS_MEMORY_COLLECTION_NAME: str = os.getenv(
        "MILVUS_MEMORY_COLLECTION_NAME",
        "memory_chunks",
    )

    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE")
    SEARCH_TOP_K: int = int(os.getenv("SEARCH_TOP_K"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD"))

    # ==================== 元数据数据库配置 ====================
    # SQLite（开发）/ PostgreSQL（生产）
    DATABASE_URL: str = os.getenv("DATABASE_URL")

    # ==================== 文件存储配置 ====================
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR")
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE"))
    ALLOWED_FILE_TYPES: List[str] = _parse_list(os.getenv("ALLOWED_FILE_TYPES"))

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
    LOG_DIR: str = _resolve_app_path(os.getenv("LOG_DIR"), "logs")
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
