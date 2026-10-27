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
load_dotenv(_env_path, override=False)


def _configure_langsmith_tracing() -> None:
    """默认关闭 LangSmith/LangChain 追踪，避免本地开发反复上报。"""
    tracing = (os.getenv("LANGSMITH_TRACING") or "false").strip().lower()
    if tracing not in {"true", "1", "yes", "on"}:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


_configure_langsmith_tracing()


def _configure_huggingface_offline() -> None:
    """默认离线加载 HuggingFace 模型，避免本地模型已缓存时仍联网检查导致超时。

    transformers 5.x 的 local_files_only 在部分路径下仍会访问 hub；
    强制 HF_HUB_OFFLINE 可彻底切断联网。需要联网下载模型时设 HF_HUB_OFFLINE=false。
    """
    offline = (os.getenv("HF_HUB_OFFLINE") or "true").strip().lower()
    if offline in {"true", "1", "yes", "on"}:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


_configure_huggingface_offline()


def _resolve_app_path(path_value: Optional[str], default: str) -> str:
    """把应用内部路径固定解析到 backend/app 下，避免受启动目录影响。"""
    path = Path(path_value or default)
    if not path.is_absolute():
        path = APP_DIR / path
    return str(path.resolve())


def _resolve_backend_path(path_value: Optional[str], default: str) -> str:
    """把路径固定解析到 backend 目录下，适合 data/uploads、data/cache 这类目录。"""
    path = Path(path_value or default)
    if not path.is_absolute():
        path = BACKEND_DIR / path
    return str(path.resolve())


def _resolve_backend_model_path(path_value: Optional[str], default: str) -> str:
    """把本地模型路径优先解析到 backend 目录下；远程模型名则原样保留。"""
    raw_value = (path_value or default).strip()
    if not raw_value:
        return raw_value

    path = Path(raw_value)
    if path.is_absolute():
        return str(path.resolve())

    candidate = (BACKEND_DIR / path).resolve()
    if candidate.exists():
        return str(candidate)

    return raw_value


def _resolve_sqlite_url(url_value: Optional[str], default: str) -> str:
    """把 SQLite 路径固定到 app/data 下，避免散落到其他目录。"""
    raw_url = (url_value or default).strip()
    if not raw_url.startswith("sqlite:///"):
        return raw_url

    database_path = raw_url.removeprefix("sqlite:///")
    if not database_path or database_path == ":memory:" or database_path.startswith("file:"):
        return raw_url

    data_dir = APP_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = Path(database_path)
    if path.is_absolute():
        try:
            path.relative_to(data_dir)
            target_path = path
        except ValueError:
            target_path = data_dir / path.name
    else:
        target_path = data_dir / path.name
    return f"sqlite:///{target_path.resolve().as_posix()}"


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


def _parse_dict_list(value: Optional[str], default: str = "[]") -> List[dict]:
    """解析 JSON 对象数组配置，失败时返回空列表。"""
    raw_value = (value or default).strip()
    if not raw_value:
        return []
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


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
    # 服务启动时是否同步初始化默认 LLM，避免第一次对话请求等待 provider 初始化。
    STARTUP_INIT_LLM: bool = (
        os.getenv("STARTUP_INIT_LLM", "true").strip().lower() == "true"
    )
    # 温度: 0=确定性最强(代码/数据抽取), 1=默认平衡, 2=最有创意
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE"))
    # 最大输出TOKENS数量
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS"))
    # LLM请求超时时间（秒）
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT"))
    # 单次请求最多尝试的 LLM 提供商数量，包含首次调用。
    LLM_MAX_FALLBACK_ATTEMPTS: int = max(
        1,
        int(os.getenv("LLM_MAX_FALLBACK_ATTEMPTS", "2")),
    )

    # ==================== 聊天配置 ====================
    # 前端可保留的最大对话线程数，防止本地历史过多导致页面和请求阻塞。
    CHAT_MAX_CONVERSATIONS: int = int(os.getenv("CHAT_MAX_CONVERSATIONS"))
    MEMORY_CLEANUP_INTERVAL_SECONDS: int = max(
        60,
        int(os.getenv("MEMORY_CLEANUP_INTERVAL_SECONDS", "3600")),
    )
    MEMORY_EVENT_EXPIRY_DAYS: int = max(
        1,
        int(os.getenv("MEMORY_EVENT_EXPIRY_DAYS", "90")),
    )
    MEMORY_PROFILE_STALE_DAYS: int = max(
        1,
        int(os.getenv("MEMORY_PROFILE_STALE_DAYS", "180")),
    )
    MEMORY_SCORE_THRESHOLD: float = max(
        0.0,
        min(100.0, float(os.getenv("MEMORY_SCORE_THRESHOLD", "20"))),
    )
    MEMORY_PURGE_RETENTION_DAYS: int = max(
        1,
        int(os.getenv("MEMORY_PURGE_RETENTION_DAYS", "30")),
    )

    # ==================== 认证配置 ====================
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
    # 登录态默认 1 小时；每次有效交互都会把数据库会话的空闲过期时间重新延长。
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    REFRESH_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", "60"))
    SESSION_IDLE_TIMEOUT_MINUTES: int = int(
        os.getenv("SESSION_IDLE_TIMEOUT_MINUTES", "60")
    )
    BOOTSTRAP_ADMIN_USERNAME: Optional[str] = os.getenv("BOOTSTRAP_ADMIN_USERNAME", "admin")
    BOOTSTRAP_ADMIN_PASSWORD: Optional[str] = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "123456")
    BOOTSTRAP_ADMIN_ROLE: str = os.getenv("BOOTSTRAP_ADMIN_ROLE", "admin")

    # ==================== 向量化配置 ====================
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION"))
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "bge")

    # BGE 本地向量化配置
    BGE_MODEL_NAME: str = _resolve_backend_model_path(
        os.getenv("BGE_MODEL_NAME"),
        "./app/models/bge-base-zh-v1.5",
    )
    BGE_DEVICE: str = os.getenv("BGE_DEVICE", "auto")
    BGE_MAX_LENGTH: int = int(os.getenv("BGE_MAX_LENGTH", "512"))
    BGE_BATCH_SIZE: int = int(os.getenv("BGE_BATCH_SIZE", "16"))
    BGE_NORMALIZE_EMBEDDINGS: bool = os.getenv("BGE_NORMALIZE_EMBEDDINGS", "true").lower() == "true"
    BGE_QUERY_INSTRUCTION: str = os.getenv("BGE_QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关文章：")
    # BGE 模型缓存目录；为空时使用 HuggingFace/Transformers 默认缓存目录
    BGE_CACHE_DIR: Optional[str] = _resolve_backend_path(
        os.getenv("BGE_CACHE_DIR"),
        "app/models",
    )
    # 是否只从本地读取模型；true 时不会尝试联网下载 HuggingFace 模型
    BGE_LOCAL_FILES_ONLY: bool = os.getenv("BGE_LOCAL_FILES_ONLY", "false").lower() == "true"

    # ==================== 向量数据库配置 ====================
    # 向量库模式：milvus / milvus_lite
    VECTOR_STORE_TYPE: str = (os.getenv("VECTOR_STORE_TYPE") or "milvus").strip().lower()
    # 仅服务端 Milvus 使用；Lite 模式会忽略这两个值。
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_DB_NAME: str = os.getenv("MILVUS_DB_NAME", "default")
    # Milvus Lite 本地文件路径，最终会解析到 backend/app/data 下。
    MILVUS_LITE_PATH: str = _resolve_app_path(
        os.getenv("MILVUS_LITE_PATH"),
        "data/milvus_lite.db",
    )
    # 文档 Child 向量和长期记忆分开存储；Parent 正文统一由 SQLite 保存。
    MILVUS_DOC_COLLECTION_NAME: str = os.getenv(
        "MILVUS_DOC_COLLECTION_NAME",
        "doc_chunk",
    )
    MILVUS_CHILD_COLLECTION_NAME: str = os.getenv(
        "MILVUS_CHILD_COLLECTION_NAME",
        os.getenv("MILVUS_DOC_COLLECTION_NAME", "doc_chunk"),
    )
    MILVUS_MEMORY_COLLECTION_NAME: str = os.getenv(
        "MILVUS_MEMORY_COLLECTION_NAME",
        "memory_chunks",
    )

    SEARCH_TOP_K: int = int(os.getenv("SEARCH_TOP_K"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD"))

    # ==================== 元数据数据库配置 ====================
    # SQLite（开发）/ PostgreSQL（生产）
    DATABASE_URL: str = _resolve_sqlite_url(
        os.getenv("DATABASE_URL"),
        "sqlite:///./app/data/rag_metadata.db",
    )

    # ==================== 文件存储配置 ====================
    UPLOAD_DIR: str = _resolve_app_path(
        os.getenv("UPLOAD_DIR"),
        "data/uploads",
    )
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE"))
    ALLOWED_FILE_TYPES: List[str] = _parse_list(os.getenv("ALLOWED_FILE_TYPES"))

    # ==================== 检索配置 ====================
    USE_DENSE_RETRIEVER: bool = os.getenv("USE_DENSE_RETRIEVER").lower() == "true"
    USE_SPARSE_RETRIEVER: bool = os.getenv("USE_SPARSE_RETRIEVER").lower() == "true"
    USE_HYBRID_RETRIEVER: bool = os.getenv("USE_HYBRID_RETRIEVER").lower() == "true"

    DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT"))
    SPARSE_WEIGHT: float = float(os.getenv("SPARSE_WEIGHT"))

    # ==================== 重排配置 ====================
    # Cross-Encoder 默认关闭；配置模型路径并显式打开后，会在启动预热阶段加载并参与规则后精排。
    CROSS_ENCODER_ENABLED: bool = (
        os.getenv("CROSS_ENCODER_ENABLED", "false").strip().lower()
        in {"true", "1", "yes", "on"}
    )
    CROSS_ENCODER_MODEL_NAME: str = _resolve_backend_model_path(
        os.getenv("CROSS_ENCODER_MODEL_NAME"),
        "BAAI/bge-reranker-base",
    )
    CROSS_ENCODER_DEVICE: str = os.getenv("CROSS_ENCODER_DEVICE", "auto")
    CROSS_ENCODER_MAX_LENGTH: int = int(
        os.getenv("CROSS_ENCODER_MAX_LENGTH", "512")
    )
    CROSS_ENCODER_BATCH_SIZE: int = int(
        os.getenv("CROSS_ENCODER_BATCH_SIZE", "8")
    )
    CROSS_ENCODER_CACHE_DIR: Optional[str] = (
        _resolve_backend_path(
            os.getenv("CROSS_ENCODER_CACHE_DIR"),
            "app/models",
        )
    )
    CROSS_ENCODER_LOCAL_FILES_ONLY: bool = (
        os.getenv("CROSS_ENCODER_LOCAL_FILES_ONLY", "false").strip().lower()
        in {"true", "1", "yes", "on"}
    )
    # 规则只负责宽松预排序和精确约束，Cross-Encoder 只精排前 N 个候选。
    CROSS_ENCODER_CANDIDATE_K: int = max(
        1,
        int(os.getenv("CROSS_ENCODER_CANDIDATE_K", "30")),
    )
    CROSS_ENCODER_WEIGHT: float = min(
        1.0,
        max(0.0, float(os.getenv("CROSS_ENCODER_WEIGHT", "0.65"))),
    )
    RULE_RERANK_WEIGHT: float = min(
        1.0,
        max(0.0, float(os.getenv("RULE_RERANK_WEIGHT", "0.35"))),
    )
    # 精排触发模式: adaptive=规则高置信跳过精排（默认，基准显示规则重排
    # recall@5 已达 1.0，精排仅 MRR +0.02 却多 ~4s）；always=始终精排；
    # off=等同于关闭精排（保留 CROSS_ENCODER_ENABLED 原有总开关语义）。
    CROSS_ENCODER_TRIGGER_MODE: str = (
        os.getenv("CROSS_ENCODER_TRIGGER_MODE", "adaptive").strip().lower()
    )
    # adaptive 模式的置信度阈值: 规则 top-1 综合置信分超过该值即跳过精排。
    CROSS_ENCODER_CONFIDENCE_THRESHOLD: float = min(
        1.0,
        max(0.0, float(os.getenv("CROSS_ENCODER_CONFIDENCE_THRESHOLD", "0.60"))),
    )
    # adaptive 模式的领先幅度阈值: top-1 需领先 top-2 的规则分差。
    CROSS_ENCODER_CONFIDENCE_MARGIN: float = min(
        1.0,
        max(0.0, float(os.getenv("CROSS_ENCODER_CONFIDENCE_MARGIN", "0.08"))),
    )

    # ==================== 缓存配置 ====================
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE").lower() == "true"
    CACHE_DIR: str = _resolve_backend_path(
        os.getenv("CACHE_DIR"),
        "data/cache",
    )
    CACHE_TTL: int = int(os.getenv("CACHE_TTL"))

    # ==================== MCP / Skill 适配配置 ====================
    MCP_REMOTE_TOOL_MANIFESTS: List[dict] = _parse_dict_list(
        os.getenv("MCP_REMOTE_TOOL_MANIFESTS"),
        "[]",
    )

    # ==================== 常规检索规划器（RetrievalPlanner）配置 ====================
    # 是否启用 LLM 消歧决策（规则评估置信度 → LLM 决策 → 规则兜底）。
    # 关闭后规划器退回纯规则判定，不产生额外 LLM 调用。
    RETRIEVAL_PLANNER_LLM_DECISION_ENABLED: bool = (
        os.getenv("RETRIEVAL_PLANNER_LLM_DECISION_ENABLED", "true").strip().lower()
        in {"true", "1", "yes", "on"}
    )

    # ==================== ReAct 检索决策配置 ====================
    # ReAct 检索决策循环最大迭代轮数（rag 分支模型自主迭代检索的上限），
    # 防止模型无限调用工具；ReAct 失败或无结果时回退常规检索规划（规则决策）。
    REACT_MAX_ITERATIONS: int = max(1, int(os.getenv("REACT_MAX_ITERATIONS", "3")))

    # ==================== 统一 ReAct 编排配置 ====================
    # 是否启用统一 ReAct 编排循环。开启后，模型在同一条 messages 轨迹里跨工具自主
    # 编排（工具空间 = list_agent_function_calling_tools()，含 RAG 检索 + 本地 Skills
    # + 远程 MCP），而不是每轮重建路由后再执行单步工具的「伪 ReAct」。
    # 关闭、或 bind_tools 不被供应商支持时，回退 rag 专用循环 _run_rag_react_loop
    # （只暴露 search_knowledge_base），保证链路不中断。
    UNIFIED_REACT_ENABLED: bool = (
        os.getenv("UNIFIED_REACT_ENABLED", "true").strip().lower()
        in {"true", "1", "yes", "on"}
    )

    # 统一 ReAct 编排循环的迭代上限（模型自主编排工具的最大轮数），防止无限调用工具。
    UNIFIED_REACT_MAX_ITERATIONS: int = max(
        1, int(os.getenv("UNIFIED_REACT_MAX_ITERATIONS", "3"))
    )

    # 一轮内并行执行多个工具调用的线程上限（检索与远程 MCP 均为 IO 密集）。
    UNIFIED_REACT_PARALLEL_WORKERS: int = max(
        1, int(os.getenv("UNIFIED_REACT_PARALLEL_WORKERS", "4"))
    )

    # ==================== 日志配置 ====================
    LOG_DIR: str = _resolve_app_path(os.getenv("LOG_DIR"), "logs")

    # ==================== Langsmith 配置（可选）====================
    LANGSMITH_TRACING: Optional[bool] = (
        os.getenv("LANGSMITH_TRACING").lower() == "true"
        if os.getenv("LANGSMITH_TRACING") else None
    )
    LANGSMITH_ENDPOINT: Optional[str] = os.getenv("LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: Optional[str] = os.getenv("LANGSMITH_PROJECT")


def _validate_runtime_security(config: Settings) -> None:
    """非调试环境禁止使用示例认证配置。"""
    if config.DEBUG:
        return

    if (
        not config.JWT_SECRET_KEY
        or config.JWT_SECRET_KEY == "change-me-in-production"
        or len(config.JWT_SECRET_KEY) < 32
    ):
        raise RuntimeError(
            "生产环境必须配置长度至少 32 个字符的 JWT_SECRET_KEY"
        )

    if config.BOOTSTRAP_ADMIN_PASSWORD in {"123456", "password", "change-me"}:
        raise RuntimeError(
            "生产环境禁止使用默认 BOOTSTRAP_ADMIN_PASSWORD"
        )


# 全局配置实例
settings = Settings()
_validate_runtime_security(settings)


def get_settings() -> Settings:
    """获取配置实例"""
    return settings
