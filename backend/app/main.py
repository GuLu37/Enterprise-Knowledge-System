"""FastAPI 主程序入口"""
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.storage.sqlite_metadata import init_metadata_db
from app.storage.milvus_store import initialize_collections
from app.services.auth_service import require_current_user
from app.utils.logger import setup_logger, _init_loguru

# 全局日志初始化（在任何模块 import 之前完成，确保 uvicorn 日志也被接管）
_init_loguru()

# 设置日志
logger = setup_logger(__name__)


def _include_routes(app: FastAPI) -> None:
    """延迟导入并注册路由，减少启动阶段的重型依赖加载。"""
    from app.api.routes.auth import router as auth_router
    from app.api.routes.documents import router as documents_router
    from app.api.routes.retrieval import router as retrieval_router
    from app.api.routes.chat import router as chat_router

    app.include_router(
        auth_router,
        prefix=f"{settings.API_PREFIX}/auth",
        tags=["Auth"],
    )
    app.include_router(
        documents_router,
        prefix=f"{settings.API_PREFIX}/documents",
        tags=["Documents"],
        dependencies=[Depends(require_current_user)],
    )
    app.include_router(
        retrieval_router,
        prefix=f"{settings.API_PREFIX}/retrieval",
        tags=["Retrieval"],
        dependencies=[Depends(require_current_user)],
    )
    app.include_router(
        chat_router,
        prefix=f"{settings.API_PREFIX}/chat",
        tags=["Chat"],
        dependencies=[Depends(require_current_user)],
    )


def _warmup_chat_runtime_in_background(app: FastAPI) -> None:
    """后台预热模型，避免本地模型加载阻塞服务启动。"""
    try:
        from app.services.chat_service import warmup_chat_runtime

        warmup_started_at = time.perf_counter()
        result = warmup_chat_runtime()
        result["status"] = (
            "ready"
            if result.get("llm_warmed") and result.get("embedding_warmed")
            else "failed"
        )
        app.state.runtime_warmup = result
        logger.info(
            "聊天运行时后台预热完成: llm=%s embedding=%s provider=%s duration=%.2fms",
            result.get("llm_warmed"),
            result.get("embedding_warmed"),
            result.get("provider"),
            (time.perf_counter() - warmup_started_at) * 1000,
        )
    except Exception as exc:
        app.state.runtime_warmup = {
            "llm_warmed": False,
            "embedding_warmed": False,
            "provider": None,
            "error": str(exc),
        }
        logger.warning("聊天运行时后台预热出错，首次请求将按需初始化: %s", exc)


def _initialize_milvus_in_background(app: FastAPI) -> None:
    """后台初始化 Milvus，避免连接或 collection 预热拖慢服务可用时间。"""
    try:
        app.state.milvus_ready = initialize_collections()
        if app.state.milvus_ready:
            logger.info("✓ Milvus 后台初始化完成")
        else:
            logger.warning("Milvus 后台初始化未完成，检索与入库将在服务恢复后可用")
    except Exception as exc:
        app.state.milvus_ready = False
        logger.warning("Milvus 后台初始化出错: %s", exc)


def _memory_maintenance_loop(app: FastAPI) -> None:
    """定期刷新记忆重要性并清理过期数据。"""
    from app.services.memory_service import cleanup_memory_records

    stop_event = app.state.memory_maintenance_stop_event
    while not stop_event.is_set():
        try:
            result = cleanup_memory_records()
            if any(result.values()):
                logger.info(
                    "长期记忆维护完成: rescored=%s expired=%s purged=%s",
                    result["rescored"],
                    result["expired"],
                    result["purged"],
                )
        except Exception as exc:
            logger.warning("长期记忆维护失败: %s", exc)
        stop_event.wait(settings.MEMORY_CLEANUP_INTERVAL_SECONDS)


def _start_background_initialization(app: FastAPI) -> None:
    """启动不影响 HTTP 就绪状态的后台初始化任务。"""
    app.state.runtime_warmup = {
        "llm_warmed": False,
        "embedding_warmed": False,
        "provider": None,
        "status": "initializing",
    }
    app.state.milvus_ready = False
    app.state.memory_maintenance_stop_event = threading.Event()

    threading.Thread(
        target=_warmup_chat_runtime_in_background,
        args=(app,),
        name="chat-runtime-warmup",
        daemon=True,
    ).start()
    threading.Thread(
        target=_initialize_milvus_in_background,
        args=(app,),
        name="milvus-initialization",
        daemon=True,
    ).start()
    threading.Thread(
        target=_memory_maintenance_loop,
        args=(app,),
        name="memory-maintenance",
        daemon=True,
    ).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # Startup
    logger.info("==================================================")
    logger.info(f"🚀 启动 {settings.APP_NAME} (v{settings.APP_VERSION})")
    logger.info(f"环境: {settings.ENVIRONMENT}")
    logger.info(f"调试模式: {settings.DEBUG}")
    logger.info("系统正在初始化各类设置...请稍等...")
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    init_metadata_db()
    _start_background_initialization(app)

    logger.info("核心服务已就绪，模型与 Milvus 正在后台初始化。")
    logger.info("==================================================")

    yield

    # Shutdown
    stop_event = getattr(app.state, "memory_maintenance_stop_event", None)
    if stop_event is not None:
        stop_event.set()
    logger.info("🛑 关闭应用...")


# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="基于 LangChain 的 RAG 三重检索系统",
    lifespan=lifespan,
)
_include_routes(app)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Access-Token"],
)

# 记录每次 HTTP 调用，未捕获异常会带 traceback 写入日志文件。
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有 HTTP 请求、响应状态和未捕获异常。"""
    start_time = time.perf_counter()
    client_host = request.client.host if request.client else "-"
    request_line = f"{request.method} {request.url.path}"
    if request.url.query:
        request_line = f"{request_line}?{request.url.query}"

    logger.info(f"请求开始: {request_line} client={client_host}")
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(f"请求异常: {request_line} duration={duration_ms:.2f}ms")
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000
    renewed_access_token = getattr(
        getattr(request, "state", None),
        "renewed_access_token",
        None,
    )
    if renewed_access_token:
        response.headers["X-Access-Token"] = renewed_access_token
    logger.info(
        f"请求完成: {request_line} status={response.status_code} duration={duration_ms:.2f}ms"
    )
    return response


@app.get("/")
async def root():
    """根路由"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """健康检查接口，供反向代理、监控和外部探活使用。"""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.DEBUG,
        log_config=None,  # 禁用 uvicorn 默认日志配置，由 loguru 统一接管
    )
