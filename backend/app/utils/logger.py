"""日志配置"""
import logging
from pathlib import Path
from loguru import logger as loguru_logger
from app.config import settings

# 创建日志目录
log_dir = Path(settings.LOG_DIR)
log_dir.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str) -> logging.Logger:
    """配置日志"""

    # 移除默认处理器
    loguru_logger.remove()

    # 添加控制台处理器
    loguru_logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
    )

    # 添加文件处理器
    log_file = log_dir / settings.LOG_FILE_NAME
    loguru_logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.LOG_LEVEL,
        rotation="500 MB",
        retention="10 days",
    )

    return loguru_logger.bind(name=name)
