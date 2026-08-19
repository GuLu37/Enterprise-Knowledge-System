"""日志配置"""
import asyncio
import logging
import sys
import threading
import warnings
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger as loguru_logger

from app.config import settings

# 创建日志目录
log_dir = Path(settings.LOG_DIR)
log_dir.mkdir(parents=True, exist_ok=True)

_initialized = False
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_LOG_ROTATION_MAX_BYTES = 100 * 1024 * 1024


class _StreamToLogger:
    """把 stdout/stderr 的原始输出按行写入 loguru，覆盖 print 和未接管的报错输出。"""

    def __init__(self, level: str):
        self.level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        if not message:
            return 0

        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                loguru_logger.opt(depth=1).log(self.level, line)
        return len(message)

    def flush(self) -> None:
        if self._buffer.strip():
            loguru_logger.opt(depth=1).log(self.level, self._buffer.strip())
        self._buffer = ""

    def isatty(self) -> bool:
        return False


class _InterceptHandler(logging.Handler):
    """把标准 logging（uvicorn/fastapi/第三方库）的日志桥接到 loguru。"""

    def emit(self, record: logging.LogRecord) -> None:
        # 找到对应的 loguru 级别
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到真实调用栈深度（跳过 logging 内部帧）
        try:
            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
        except ValueError:
            depth = 2

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _find_reusable_log_file(now: datetime) -> Path | None:
    """优先复用当天且未超过阈值的最新日志文件。"""
    candidates: list[tuple[float, Path]] = []

    for path in log_dir.glob("*.log"):
        try:
            stat = path.stat()
        except OSError:
            continue

        file_day = datetime.fromtimestamp(stat.st_mtime).date()
        if file_day != now.date():
            continue
        if stat.st_size >= _LOG_ROTATION_MAX_BYTES:
            continue

        candidates.append((stat.st_mtime, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _build_log_file_pattern(now: datetime | None = None) -> str:
    """优先复用当天日志，否则新建纯时间戳文件名。"""
    now = now or datetime.now()
    reusable = _find_reusable_log_file(now)
    if reusable is not None:
        return str(reusable)

    candidate = log_dir / f"{now:%Y%m%d%H%M%S}.log"
    offset_seconds = 1
    while candidate.exists():
        try:
            size = candidate.stat().st_size
        except OSError:
            break

        if size < _LOG_ROTATION_MAX_BYTES:
            return str(candidate)

        candidate = log_dir / f"{(now + timedelta(seconds=offset_seconds)):%Y%m%d%H%M%S}.log"
        offset_seconds += 1

    return str(candidate)


def _build_rotation_rule():
    """按自然日和文件大小触发轮转。"""
    current_day = datetime.now().date()

    def should_rotate(message, file) -> bool:
        nonlocal current_day

        record_time = message.record["time"]
        if record_time.date() != current_day:
            current_day = record_time.date()
            return True

        try:
            file.seek(0, 2)
            current_size = file.tell()
        except Exception:
            current_size = 0

        return current_size >= _LOG_ROTATION_MAX_BYTES

    return should_rotate


def _install_exception_hooks() -> None:
    """记录主线程、子线程和 asyncio 中未捕获的异常。"""

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        loguru_logger.opt(exception=(exc_type, exc_value, exc_traceback)).critical(
            "未捕获异常"
        )

    def handle_thread_exception(args: threading.ExceptHookArgs):
        loguru_logger.opt(
            exception=(args.exc_type, args.exc_value, args.exc_traceback)
        ).critical(f"线程未捕获异常: {args.thread.name if args.thread else 'unknown'}")

    def handle_asyncio_exception(loop, context):
        exception = context.get("exception")
        message = context.get("message", "asyncio 未捕获异常")
        if exception:
            loguru_logger.opt(exception=exception).critical(message)
        else:
            loguru_logger.critical(message)

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception

    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(handle_asyncio_exception)
    except RuntimeError:
        pass


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
        _ORIGINAL_STDERR,
        format=color_fmt,
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    # 文件（纯文本，按天/按大小轮转，文件名包含精确时间戳）。
    loguru_logger.add(
        _build_log_file_pattern(),
        format=log_fmt,
        level=0,
        rotation=_build_rotation_rule(),
        retention="10 days",
        encoding="utf-8",
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )

    # 桥接标准 logging → loguru（覆盖 uvicorn、fastapi、sqlalchemy 等）
    intercept = _InterceptHandler()
    intercept.setLevel(0)

    # 接管所有已有 logger
    for name in logging.root.manager.loggerDict:
        log = logging.getLogger(name)
        log.handlers = [intercept]
        log.propagate = False

    # httpx/httpcore 在 INFO 级别会把每一次模型 HTTP 请求都打出来；
    # 业务日志保留即可，第三方成功请求降到 WARNING，避免控制台刷屏。
    for noisy_logger_name in ("httpx", "httpcore"):
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

    # 接管 root logger，捕获未来动态创建的 logger
    logging.root.handlers = [intercept]
    logging.root.setLevel(0)
    logging.captureWarnings(True)
    warnings.simplefilter("default")

    # 单独确保 uvicorn 系列被接管
    for uvicorn_logger_name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "sqlalchemy",
    ):
        uv_log = logging.getLogger(uvicorn_logger_name)
        uv_log.handlers = [intercept]
        uv_log.propagate = False

    for noisy_logger_name in ("httpx", "httpcore"):
        noisy_log = logging.getLogger(noisy_logger_name)
        noisy_log.handlers = [intercept]
        noisy_log.propagate = False
        noisy_log.setLevel(logging.WARNING)

    sys.stdout = _StreamToLogger("INFO")
    sys.stderr = _StreamToLogger("ERROR")
    _install_exception_hooks()


def setup_logger(name: str):
    """获取绑定了模块名的 loguru logger，首次调用时完成全局初始化。"""
    _init_loguru()
    return loguru_logger.bind(name=name)
