"""健康检查路由"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/ready")
async def readiness_check():
    """就绪检查端点"""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
    }
