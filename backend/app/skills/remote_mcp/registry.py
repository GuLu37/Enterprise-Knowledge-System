"""远程 MCP -> Skill 的注册适配器。"""
from __future__ import annotations

from collections.abc import Iterable as IterableABC
from typing import Any, Dict, List, Optional, Sequence

from app.config import get_settings
from app.skills.registry import (
    SkillDefinition,
    SkillExecutionContext,
    SkillExecutionError,
    register_skill,
)
from app.utils.logger import setup_logger

_DEFAULT_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
}

logger = setup_logger(__name__)


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, IterableABC):
        items = list(value)
    else:
        items = [value]
    normalized: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
    return normalized


def _first_non_empty(mapping: Dict[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _coerce_parameters(manifest: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("parameters", "schema", "input_schema", "inputSchema", "arguments_schema"):
        candidate = manifest.get(key)
        if isinstance(candidate, dict):
            return dict(candidate)
    return dict(_DEFAULT_PARAMETERS)


def _build_execution_error_message(manifest: Dict[str, Any]) -> str:
    tool_name = _first_non_empty(manifest, ("tool_name", "remote_tool_name", "name"), "unknown")
    server_name = _first_non_empty(manifest, ("server_name", "server", "provider"), "remote_mcp")
    return (
        f"远程 MCP 工具 {server_name}/{tool_name} 尚未接入实际执行器，当前不可用。"
        f"不要重试该工具，请改用其他可用工具或直接回答。"
    )


def is_remote_mcp_ready(manifest: Dict[str, Any]) -> bool:
    """判断远程 MCP 是否已接入真实执行器（配置了 endpoint 才算就绪）。"""
    return bool(_first_non_empty(manifest, ("endpoint", "url", "server_url", "serverUrl"), ""))


def build_remote_mcp_skill_definition(manifest: Dict[str, Any]) -> SkillDefinition:
    """把远程 MCP manifest 转成标准 SkillDefinition。"""
    name = _first_non_empty(manifest, ("skill_name", "name", "tool_name"))
    if not name:
        raise ValueError("远程 MCP manifest 缺少 name/skill_name/tool_name")

    raw_description = _first_non_empty(
        manifest,
        ("description", "summary", "title"),
        f"远程 MCP 工具 {name}",
    )
    # 未接入执行器的工具在描述里显式标记：即便将来某个入口把它暴露给模型，
    # 模型也能一眼看出不可用，不会白调一轮。
    ready = is_remote_mcp_ready(manifest)
    description = raw_description if ready else f"[未接入执行器·当前不可用] {raw_description}"
    category = _first_non_empty(manifest, ("category",), "mcp")
    provider = _first_non_empty(manifest, ("provider", "server_name", "server"), None)
    tags = _normalize_string_list(manifest.get("tags"))
    metadata = dict(manifest.get("metadata") or {})
    metadata.update(
        {
            "source": "remote_mcp",
            "executor_ready": ready,
            "server_name": _first_non_empty(manifest, ("server_name", "server", "provider"), ""),
            "tool_name": _first_non_empty(manifest, ("tool_name", "remote_tool_name", "name"), name),
            "manifest": {
                key: value
                for key, value in manifest.items()
                if key not in {"executor", "parameters", "schema", "input_schema", "inputSchema", "arguments_schema"}
            },
        }
    )

    def _executor(_context: SkillExecutionContext, _arguments: Dict[str, Any]) -> Dict[str, Any]:
        raise SkillExecutionError(_build_execution_error_message(manifest))

    return SkillDefinition(
        name=name,
        description=description,
        category=category,
        kind="remote_mcp",
        tags=tuple(tags),
        provider=provider or None,
        parameters=_coerce_parameters(manifest),
        executor=_executor,
        metadata=metadata,
    )


def register_remote_mcp_skills(manifests: Optional[Sequence[Dict[str, Any]]] = None) -> List[str]:
    """注册远程 MCP manifest 对应的 Skill。"""
    resolved_manifests = list(manifests or [])
    registered_names: List[str] = []
    for manifest in resolved_manifests:
        if not isinstance(manifest, dict):
            continue
        enabled = manifest.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in {"false", "0", "no", "off"}
        if not enabled:
            continue
        try:
            definition = build_remote_mcp_skill_definition(manifest)
        except Exception:
            logger.warning("跳过无效的远程 MCP manifest: {}", manifest)
            continue
        try:
            register_skill(definition)
        except Exception as exc:
            logger.warning("注册远程 MCP skill 失败: {} reason={}", definition.name, exc)
            continue
        registered_names.append(definition.name)
    return registered_names


def register_remote_mcp_skills_from_settings() -> List[str]:
    """从配置中读取远程 MCP manifests 并注册。"""
    settings = get_settings()
    manifests = getattr(settings, "MCP_REMOTE_TOOL_MANIFESTS", []) or []
    return register_remote_mcp_skills(manifests)
