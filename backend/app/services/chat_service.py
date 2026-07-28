"""聊天业务编排：让 LLM 自主决定是否调用 RAG 工具。"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.tools.rag import create_rag_tools

MAX_TOOL_ROUNDS = 3

RAG_SYSTEM_PROMPT = (
    "你是企业知识库助手。对于闲聊、写作、翻译和不需要企业内部资料的问题，直接回答。"
    "对于公司制度、业务流程、项目资料、上传文档或其他需要事实核验的问题，"
    "请先调用 search_knowledge_base。调用工具后只能依据检索结果回答，"
    "不能把没有检索到的内容当成事实；如果资料不足，请明确说明。"
)

DIRECT_CHAT_SYSTEM_PROMPT = (
    "你是企业知识库助手。当前对话未启用知识库检索，请直接回答用户问题。"
    "不要假装查阅过企业内部文档；涉及内部事实且无法确认时，请明确说明。"
)


@dataclass
class ChatRunResult:
    """一次聊天运行的结果。"""

    text: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    model: str = "default"


def build_chat_messages(
    history: Optional[Iterable[Any]],
    query: str,
    allow_retrieval: bool = True,
) -> List[Any]:
    """把 API 请求转换成 LangChain 消息，并注入工具使用策略。"""
    system_prompt = RAG_SYSTEM_PROMPT if allow_retrieval else DIRECT_CHAT_SYSTEM_PROMPT
    messages: List[Any] = [SystemMessage(content=system_prompt)]

    for message in history or []:
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            raise ValueError(f"不支持的消息角色: {role}")

    messages.append(HumanMessage(content=(query or "").strip()))
    return messages


def _extract_text(message: Any) -> str:
    """提取 LangChain 消息中的纯文本。"""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def _normalize_tool_calls(message: Any) -> List[Dict[str, Any]]:
    """兼容不同 LangChain provider 返回的 tool call 结构。"""
    tool_calls = getattr(message, "tool_calls", None) or []
    normalized: List[Dict[str, Any]] = []

    for index, call in enumerate(tool_calls):
        if isinstance(call, dict):
            normalized.append(
                {
                    "id": call.get("id") or f"tool-call-{index}",
                    "name": call.get("name") or call.get("function", {}).get("name"),
                    "args": call.get("args") or call.get("function", {}).get("arguments", {}),
                }
            )

    if normalized:
        return normalized

    raw_calls = getattr(message, "additional_kwargs", {}).get("tool_calls", [])
    for index, call in enumerate(raw_calls):
        function = call.get("function", {}) if isinstance(call, dict) else {}
        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        normalized.append(
            {
                "id": call.get("id") or f"tool-call-{index}",
                "name": function.get("name"),
                "args": args,
            }
        )

    return normalized


def _deduplicate_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按文档和 chunk 去重，避免模型重复调用工具时来源重复。"""
    deduplicated: List[Dict[str, Any]] = []
    seen = set()
    for source in sources:
        metadata = source.get("metadata") or {}
        key = (
            metadata.get("document_id"),
            metadata.get("chunk_index"),
            source.get("content"),
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(source)
    return deduplicated


def _invoke_tools(
    tool_calls: List[Dict[str, Any]],
    tool_map: Dict[str, Any],
    messages: List[Any],
) -> None:
    """执行当前轮次的工具调用，并把 ToolMessage 写回上下文。"""
    for call in tool_calls:
        tool_name = call.get("name")
        tool = tool_map.get(tool_name)
        if tool is None:
            result = f"工具不存在: {tool_name}"
        else:
            try:
                result = tool.invoke(call.get("args") or {})
            except Exception as exc:
                result = f"工具调用失败: {exc}"

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call["id"],
            )
        )


def run_chat(
    query: str,
    history: Optional[Iterable[Any]] = None,
    use_retrieval: bool = True,
    top_k: int = 5,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    llm: Optional[Any] = None,
) -> ChatRunResult:
    """运行一次支持自主 RAG 工具调用的聊天。"""
    if llm is None:
        from app.core.llm import get_llm

        llm = get_llm(
            provider=provider,
            model=model,
            temperature=temperature,
        )
    messages = build_chat_messages(history, query, allow_retrieval=use_retrieval)
    if not use_retrieval:
        return ChatRunResult(
            text=_extract_text(llm.invoke(messages)),
            model=model or "default",
        )

    sources: List[Dict[str, Any]] = []
    tools = create_rag_tools(
        sources,
        default_top_k=top_k,
        max_top_k=top_k,
    )
    tool_map = {tool.name: tool for tool in tools}
    model_with_tools = llm.bind_tools(tools)

    for _ in range(MAX_TOOL_ROUNDS):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        tool_calls = _normalize_tool_calls(response)
        if not tool_calls:
            return ChatRunResult(
                text=_extract_text(response),
                sources=_deduplicate_sources(sources),
                model=model or "default",
            )
        _invoke_tools(tool_calls, tool_map, messages)

    raise RuntimeError("LLM 工具调用超过最大轮数")


def stream_chat(
    query: str,
    history: Optional[Iterable[Any]] = None,
    use_retrieval: bool = True,
    top_k: int = 5,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    llm: Optional[Any] = None,
):
    """运行流式聊天，工具决策和工具结果通过事件交给 API 层。"""
    if llm is None:
        from app.core.llm import get_llm

        llm = get_llm(
            provider=provider,
            model=model,
            temperature=temperature,
        )
    messages = build_chat_messages(history, query, allow_retrieval=use_retrieval)
    sources: List[Dict[str, Any]] = []

    if not use_retrieval:
        for chunk in llm.stream(messages):
            text = _extract_text(chunk)
            if text:
                yield {"event": "message", "data": {"content": text}}
        yield {"event": "done", "data": {"sources": []}}
        return

    tools = create_rag_tools(
        sources,
        default_top_k=top_k,
        max_top_k=top_k,
    )
    tool_map = {tool.name: tool for tool in tools}
    model_with_tools = llm.bind_tools(tools)

    for _ in range(MAX_TOOL_ROUNDS):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        tool_calls = _normalize_tool_calls(response)

        if not tool_calls:
            text = _extract_text(response)
            if text:
                yield {"event": "message", "data": {"content": text}}
            yield {
                "event": "done",
                "data": {"sources": _deduplicate_sources(sources)},
            }
            return

        for call in tool_calls:
            yield {
                "event": "tool_call",
                "data": {
                    "name": call.get("name"),
                    "args": call.get("args") or {},
                },
            }

        _invoke_tools(tool_calls, tool_map, messages)
        yield {
            "event": "tool_result",
            "data": {"sources": _deduplicate_sources(sources)},
        }

    raise RuntimeError("LLM 工具调用超过最大轮数")


def _validate_chat_options(query: str, top_k: int) -> None:
    """校验聊天业务参数。"""
    if not query or not query.strip():
        raise ValueError("查询文本不能为空")
    if top_k < 1 or top_k > 50:
        raise ValueError("top_k 必须在 1 到 50 之间")


def generate_chat(
    query: str,
    history: Optional[Iterable[Any]] = None,
    top_k: int = 5,
    use_retrieval: bool = True,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ChatRunResult:
    """Chat API 使用的同步业务入口。"""
    _validate_chat_options(query, top_k)
    return run_chat(
        query=query,
        history=history,
        use_retrieval=use_retrieval,
        top_k=top_k,
        provider=provider,
        model=model,
        temperature=temperature,
    )


def stream_chat_events(
    query: str,
    history: Optional[Iterable[Any]] = None,
    top_k: int = 5,
    use_retrieval: bool = True,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """Chat API 使用的流式业务入口。"""
    _validate_chat_options(query, top_k)
    yield from stream_chat(
        query=query,
        history=history,
        use_retrieval=use_retrieval,
        top_k=top_k,
        provider=provider,
        model=model,
        temperature=temperature,
    )
