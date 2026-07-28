"""FastAPI 主程序入口"""
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.storage.sqlite_metadata import init_metadata_db
from app.storage.milvus_store import warmup_collections
from app.utils.logger import setup_logger, _init_loguru

# 全局日志初始化（在任何模块 import 之前完成，确保 uvicorn 日志也被接管）
_init_loguru()

# 设置日志
logger = setup_logger(__name__)


def _include_routes(app: FastAPI) -> None:
    """延迟导入并注册路由，减少启动阶段的重型依赖加载。"""
    from app.api.routes.documents import router as documents_router
    from app.api.routes.retrieval import router as retrieval_router
    from app.api.routes.chat import router as chat_router
    from app.api.routes.health import router as health_router

    app.include_router(health_router, tags=["Health"])
    app.include_router(
        documents_router,
        prefix=f"{settings.API_PREFIX}/documents",
        tags=["Documents"],
    )
    app.include_router(
        retrieval_router,
        prefix=f"{settings.API_PREFIX}/retrieval",
        tags=["Retrieval"],
    )
    app.include_router(
        chat_router,
        prefix=f"{settings.API_PREFIX}/chat",
        tags=["Chat"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # Startup
    logger.info(f"🚀 启动 {settings.APP_NAME} (v{settings.APP_VERSION})")
    logger.info(f"环境: {settings.ENVIRONMENT}")
    logger.info(f"调试模式: {settings.DEBUG}")
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    init_metadata_db()
    warmup_collections()
    _include_routes(app)

    yield

    # Shutdown
    logger.info("🛑 关闭应用...")


# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="基于 LangChain 的 RAG 三重检索系统",
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        "health": "/health",
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
