"""FastAPI 主程序入口"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path

from app.config import settings
from app.storage.sqlite_metadata import init_metadata_db
from app.utils.logger import setup_logger, _init_loguru
from app.api.routes import (
    documents_router,
    retrieval_router,
    chat_router,
    health_router,
)

# 全局日志初始化（在任何模块 import 之前完成，确保 uvicorn 日志也被接管）
_init_loguru()

# 设置日志
logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # Startup
    logger.info(f"🚀 启动 {settings.APP_NAME} (v{settings.APP_VERSION})")
    logger.info(f"环境: {settings.ENVIRONMENT}")
    logger.info(f"调试模式: {settings.DEBUG}")
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    init_metadata_db()

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

# 注册路由
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
