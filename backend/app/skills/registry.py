"""统一 Skill 注册与 Function Calling 目录。"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.config import APP_DIR
from app.tools.trace import log_tool_call, log_tool_error, log_tool_result
from app.utils.logger import setup_logger
logger = setup_logger(__name__)

SkillExecutor = Callable[["SkillExecutionContext", Dict[str, Any]], Dict[str, Any]]

_SKILL_REGISTRY: Dict[str, "SkillDefinition"] = {}
_BUILTIN_SKILLS_LOADED = False
SKILLS_DIR = APP_DIR / "skills"


def get_skill_document_path(skill_name: str) -> Path:
    """返回 Skill 对应的同名目录和文档路径。"""
    normalized_name = (skill_name or "").strip()
    if (
        not normalized_name
        or normalized_name in {".", ".."}
        or Path(normalized_name).name != normalized_name
    ):
        raise ValueError(f"非法 Skill 名称: {skill_name}")
    return SKILLS_DIR / normalized_name / f"{normalized_name}.md"


def read_skill_document(skill_name: str) -> str:
    """按已注册 Skill 名称读取详细规则，未配置文档时返回空字符串。"""
    definition = get_skill_definition(skill_name)
    return _read_skill_document_for_definition(definition)


def _read_skill_document_for_definition(definition: "SkillDefinition") -> str:
    """读取已解析 Skill 定义对应的详细规则。"""
    if definition.kind != "local":
        return ""

    skills_root = SKILLS_DIR.resolve()
    document_path = get_skill_document_path(definition.name).resolve()
    if skills_root not in document_path.parents:
        raise ValueError(f"Skill 文档路径超出技能目录: {definition.name}")
    if not document_path.is_file():
        logger.warning("Skill 详细文档不存在: %s", document_path)
        return ""
    return document_path.read_text(encoding="utf-8")

RAG_TOOL_DEFINITION: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "检索企业知识库，适用于文档内容、制度流程、上传资料、内部事实核验，以及按文件名/文档名查询某份文档是否存在。",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用于知识库检索的改写问题。当用户提到文件名/文档名时，可直接用文件名作为检索词。",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5,
                    "description": "返回多少条检索结果。",
                },
                "retrieval_method": {
                    "type": "string",
                    "enum": ["hybrid", "dense", "sparse"],
                    "default": "hybrid",
                    "description": "检索方式。",
                },
            },
            "required": ["query"],
        },
    },
}


@dataclass(frozen=True)
class SkillExecutionContext:
    """一次技能调用的上下文信息。"""

    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    query: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillDefinition:
    """标准化 Skill 定义。"""

    name: str
    description: str
    parameters: Dict[str, Any]
    category: str = "general"
    kind: str = "local"
    tags: Sequence[str] = field(default_factory=tuple)
    provider: Optional[str] = None
    executor: Optional[SkillExecutor] = field(default=None, compare=False, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_function_calling_tool(self) -> Dict[str, Any]:
        """转换为 OpenAI/LangChain 风格的 Function Calling 工具描述。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_public_dict(self) -> Dict[str, Any]:
        """转换为 API 可直接返回的公开信息。"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "kind": self.kind,
            "tags": list(self.tags),
            "provider": self.provider,
            "metadata": dict(self.metadata),
            "function_calling": self.to_function_calling_tool(),
        }


class SkillNotFoundError(KeyError):
    """技能不存在。"""


class SkillExecutionError(RuntimeError):
    """技能执行失败。"""


def register_skill(definition: SkillDefinition) -> SkillDefinition:
    """注册一个技能定义。"""
    if not definition.name or not definition.name.strip():
        raise ValueError("skill name 不能为空")
    if definition.executor is None and definition.kind == "local":
        raise ValueError(f"本地 skill 必须提供 executor: {definition.name}")
    _SKILL_REGISTRY[definition.name] = definition
    return definition


def get_skill_definition(name: str) -> SkillDefinition:
    """读取单个技能定义。"""
    ensure_builtin_skills_loaded()
    normalized = (name or "").strip()
    if not normalized or normalized not in _SKILL_REGISTRY:
        raise SkillNotFoundError(normalized)
    return _SKILL_REGISTRY[normalized]


def list_skill_definitions() -> List[Dict[str, Any]]:
    """列出所有已注册技能。"""
    ensure_builtin_skills_loaded()
    return [definition.to_public_dict() for definition in sorted(_SKILL_REGISTRY.values(), key=lambda item: item.name)]


def list_function_calling_tools() -> List[Dict[str, Any]]:
    """列出所有技能对应的 Function Calling 规格。"""
    ensure_builtin_skills_loaded()
    return [definition.to_function_calling_tool() for definition in sorted(_SKILL_REGISTRY.values(), key=lambda item: item.name)]


def list_agent_capabilities() -> List[Dict[str, Any]]:
    """列出给 Agent 的统一能力目录。"""
    capabilities: List[Dict[str, Any]] = [
        {
            "name": RAG_TOOL_DEFINITION["function"]["name"],
            "description": RAG_TOOL_DEFINITION["function"]["description"],
            "category": "retrieval",
            "kind": "tool",
            "source": "rag",
            "location": "app.tools.rag.run_rag_tool",
            "function_calling": dict(RAG_TOOL_DEFINITION),
        }
    ]

    for skill in list_skill_definitions():
        capabilities.append(
            {
                **skill,
                "source": "skill",
                "location": get_skill_document_path(skill["name"])
                .relative_to(APP_DIR.parent.parent)
                .as_posix(),
            }
        )

    return capabilities


def list_agent_function_calling_tools() -> List[Dict[str, Any]]:
    """列出给模型做 Function Calling 的工具 schema。"""
    return [
        dict(RAG_TOOL_DEFINITION),
        *[
            dict(skill["function_calling"])
            for skill in list_skill_definitions()
        ],
    ]


def execute_agent_tool(
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    context: Optional[SkillExecutionContext] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """按统一入口执行 Agent 选中的工具。"""
    from app.tools.rag import run_rag_tool

    normalized_name = (name or "").strip()
    resolved_arguments = dict(arguments or {})

    if normalized_name == RAG_TOOL_DEFINITION["function"]["name"]:
        context_payload = {
            "user_id": getattr(context, "user_id", None),
            "conversation_id": getattr(context, "conversation_id", None),
            "session_id": getattr(context, "session_id", None),
            "query": getattr(context, "query", None),
            "provider": getattr(context, "provider", None),
            "model": getattr(context, "model", None),
            "trace_id": getattr(context, "trace_id", None),
            "metadata": dict(getattr(context, "metadata", {}) or {}),
        }
        context_payload.update(
            {
                "query": str(resolved_arguments.get("query") or ""),
                "top_k": int(resolved_arguments.get("top_k") or 5),
                "retrieval_method": resolved_arguments.get("retrieval_method") or "hybrid",
            }
        )
        return {
            "tool": normalized_name,
            "kind": "tool",
            "category": "retrieval",
            "provider": None,
            "arguments": resolved_arguments,
            "context": context_payload,
            "result": run_rag_tool(
                query=str(resolved_arguments.get("query") or ""),
                top_k=resolved_arguments.get("top_k"),
                retrieval_method=str(resolved_arguments.get("retrieval_method") or "hybrid"),
                context=context_payload,
                llm=llm,
            ),
        }

    skill_result = execute_skill(normalized_name, resolved_arguments, context)
    skill_result["tool"] = skill_result.get("skill", normalized_name)
    return skill_result


def execute_skill(
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    context: Optional[SkillExecutionContext] = None,
) -> Dict[str, Any]:
    """执行单个技能，并返回统一封装结果。"""
    ensure_builtin_skills_loaded()
    definition = get_skill_definition(name)
    if definition.executor is None:
        raise SkillExecutionError(f"skill 暂未配置执行器: {name}")

    resolved_context = context or SkillExecutionContext()
    skill_instructions = _read_skill_document_for_definition(definition)
    resolved_arguments = dict(arguments or {})
    trace_kind = "mcp" if definition.kind == "remote_mcp" else "skill"
    context_payload = {
        "user_id": resolved_context.user_id,
        "conversation_id": resolved_context.conversation_id,
        "session_id": resolved_context.session_id,
        "query": resolved_context.query,
        "provider": resolved_context.provider,
        "model": resolved_context.model,
        "trace_id": resolved_context.trace_id,
        "metadata": dict(resolved_context.metadata),
    }
    log_tool_call(
        kind=trace_kind,
        name=definition.name,
        arguments=resolved_arguments,
        context=context_payload,
    )
    execution_context = SkillExecutionContext(
        user_id=resolved_context.user_id,
        conversation_id=resolved_context.conversation_id,
        session_id=resolved_context.session_id,
        query=resolved_context.query,
        provider=resolved_context.provider,
        model=resolved_context.model,
        trace_id=resolved_context.trace_id,
        metadata={
            **dict(resolved_context.metadata),
            "skill_instructions": skill_instructions,
        },
    )
    try:
        payload = definition.executor(execution_context, resolved_arguments)
    except Exception as exc:
        log_tool_error(
            kind=trace_kind,
            name=definition.name,
            error=exc,
            context=context_payload,
        )
        logger.exception(f"Skill 执行失败: {name}")
        raise SkillExecutionError(str(exc)) from exc

    log_tool_result(
        kind=trace_kind,
        name=definition.name,
        result={
            "result_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
            "summary": payload.get("summary", {}) if isinstance(payload, dict) else {},
        },
        context=context_payload,
    )

    return {
        "skill": definition.name,
        "category": definition.category,
        "kind": definition.kind,
        "provider": definition.provider,
        "arguments": resolved_arguments,
        "context": context_payload,
        "skill_instructions": skill_instructions,
        "result": payload,
        "executed_at": datetime.now(timezone.utc).isoformat(),
    }


def ensure_builtin_skills_loaded() -> None:
    """延迟加载内置技能，避免启动阶段拉起重型依赖。"""
    global _BUILTIN_SKILLS_LOADED
    if _BUILTIN_SKILLS_LOADED:
        from app.skills.remote_mcp.registry import register_remote_mcp_skills_from_settings

        register_remote_mcp_skills_from_settings()
        return
    from app.skills.builtin import register_builtin_skills

    register_builtin_skills()
    from app.skills.remote_mcp.registry import register_remote_mcp_skills_from_settings

    register_remote_mcp_skills_from_settings()
    _BUILTIN_SKILLS_LOADED = True


def reset_skill_registry() -> None:
    """清空技能注册表，主要供测试使用。"""
    global _BUILTIN_SKILLS_LOADED
    _SKILL_REGISTRY.clear()
    _BUILTIN_SKILLS_LOADED = False


__all__ = [
    "RAG_TOOL_DEFINITION",
    "SKILLS_DIR",
    "SkillExecutionContext",
    "SkillDefinition",
    "SkillExecutionError",
    "SkillNotFoundError",
    "execute_agent_tool",
    "execute_skill",
    "get_skill_definition",
    "get_skill_document_path",
    "list_agent_capabilities",
    "list_agent_function_calling_tools",
    "list_function_calling_tools",
    "list_skill_definitions",
    "read_skill_document",
    "register_skill",
    "reset_skill_registry",
]
