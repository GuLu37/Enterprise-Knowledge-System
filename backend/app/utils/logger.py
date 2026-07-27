"""日志配置"""
import logging
import sys
from pathlib import Path
from loguru import logger as loguru_logger
from app.config import settings

# 创建日志目录
log_dir = Path(settings.LOG_DIR)
log_dir.mkdir(parents=True, exist_ok=True)

_initialized = False


class _InterceptHandler(logging.Handler):
    """把标准 logging（uvicorn/fastapi/第三方库）的日志桥接到 loguru。"""

    def emit(self, record: logging.LogRecord) -> None:
        # 找到对应的 loguru 级别
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到真实调用栈深度（跳过 logging 内部帧）
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _init_loguru() -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True

    # 移除 loguru 默认处理器
    loguru_logger.remove()

    log_fmt = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    color_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # 控制台（带色彩）
    loguru_logger.add(
        sys.stderr,
        format=color_fmt,
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    # 文件（纯文本，自动轮转）
    log_file = log_dir / settings.LOG_FILE_NAME
    loguru_logger.add(
        str(log_file),
        format=log_fmt,
        level=settings.LOG_LEVEL,
        rotation="500 MB",
        retention="10 days",
        encoding="utf-8",
    )

    # 桥接标准 logging → loguru（覆盖 uvicorn、fastapi、sqlalchemy 等）
    intercept = _InterceptHandler()
    intercept.setLevel(0)

    # 接管所有已有 logger
    for name in logging.root.manager.loggerDict:
        log = logging.getLogger(name)
        log.handlers = [intercept]
        log.propagate = False

    # 接管 root logger，捕获未来动态创建的 logger
    logging.root.handlers = [intercept]
    logging.root.setLevel(0)

    # 单独确保 uvicorn 系列被接管
    for uvicorn_logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        uv_log = logging.getLogger(uvicorn_logger_name)
        uv_log.handlers = [intercept]
        uv_log.propagate = False


def setup_logger(name: str):
    """获取绑定了模块名的 loguru logger，首次调用时完成全局初始化。"""
    _init_loguru()
    return loguru_logger.bind(name=name)
