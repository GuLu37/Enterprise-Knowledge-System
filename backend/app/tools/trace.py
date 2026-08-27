"""工具与技能调用轨迹日志。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

MAX_LOG_STRING_LENGTH = 300
MAX_LOG_LIST_ITEMS = 6
MAX_LOG_DICT_ITEMS = 12


def _truncate_text(value: str) -> str:
    text = str(value)
    if len(text) <= MAX_LOG_STRING_LENGTH:
        return text
    return f"{text[:MAX_LOG_STRING_LENGTH]}...<len={len(text)}>"


def _compact_for_log(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat() if value.tzinfo else value.isoformat()
    if isinstance(value, dict):
        items = list(value.items())[:MAX_LOG_DICT_ITEMS]
        return {
            str(key): _compact_for_log(item, depth + 1)
            for key, item in items
        }
    if isinstance(value, (list, tuple, set)):
        items = list(value)[:MAX_LOG_LIST_ITEMS]
        return [_compact_for_log(item, depth + 1) for item in items]
    return _truncate_text(value)


def _format_payload(kind: str, stage: str, name: str, payload: Dict[str, Any]) -> str:
    event = {
        "kind": kind,
        "stage": stage,
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    return json.dumps(_compact_for_log(event), ensure_ascii=False)


def log_tool_call(
    kind: str,
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """记录一次工具/技能/MCP 调用开始。"""
    logger.info(
        "调用开始: {}",
        _format_payload(
            kind=kind,
            stage="call",
            name=name,
            payload={
                "arguments": arguments or {},
                "context": context or {},
            },
        ),
    )


def log_tool_result(
    kind: str,
    name: str,
    result: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """记录一次工具/技能/MCP 调用结果。"""
    logger.info(
        "调用结果: {}",
        _format_payload(
            kind=kind,
            stage="result",
            name=name,
            payload={
                "result": result or {},
                "context": context or {},
            },
        ),
    )


def log_tool_error(
    kind: str,
    name: str,
    error: Exception | str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """记录一次工具/技能/MCP 调用异常。"""
    logger.error(
        "调用异常: {}",
        _format_payload(
            kind=kind,
            stage="error",
            name=name,
            payload={
                "error": str(error),
                "context": context or {},
            },
        ),
    )
