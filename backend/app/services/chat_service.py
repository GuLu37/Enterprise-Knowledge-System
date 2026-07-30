"""聊天业务编排：让 LLM 自主决定是否调用 RAG 工具。"""
import logging
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.services.memory_service import (
    search_long_term_memory,
    store_semantic_long_term_memory,
)
from app.tools.rag import create_rag_tools

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3
SHORT_MEMORY_STRATEGY_WINDOW = "window"
SHORT_MEMORY_STRATEGY_SUMMARY = "summary"
RETRIEVAL_METHOD_HYBRID = "hybrid"
RETRIEVAL_METHOD_DENSE = "dense"
RETRIEVAL_METHOD_SPARSE = "sparse"
DEFAULT_SHORT_MEMORY_N = 5
DEFAULT_SHORT_MEMORY_M = 10
LONG_TERM_MEMORY_TOP_K = 3
LONG_TERM_MEMORY_SNIPPET_LIMIT = 240

SUMMARY_SYSTEM_PROMPT = (
    "你是对话摘要器。请将给定的历史对话压缩成一段简洁、准确、可继续对话使用的中文摘要。"
    "保留用户目标、已确认事实、约束条件、偏好、待办事项和关键结论。"
    "不要编造，不要加入未出现的信息。"
)

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


def _normalize_history_message(message: Any) -> Dict[str, str]:
    """把 history 消息统一成可处理的 role/content 结构。"""
    if isinstance(message, dict):
        role = str(message.get("role", "")).strip().lower()
        content = _extract_text(message.get("content")).strip()
    else:
        role = str(getattr(message, "role", "")).strip().lower()
        content = _extract_text(getattr(message, "content", "")).strip()

    if not role or not content:
        return {}
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"不支持的消息角色: {role}")
    return {"role": role, "content": content}


def _split_history_messages(history: Optional[Iterable[Any]]) -> Tuple[List[Dict[str, str]], List[List[Dict[str, str]]]]:
    """把 history 拆成系统消息和按轮次组织的对话消息。"""
    system_messages: List[Dict[str, str]] = []
    turns: List[List[Dict[str, str]]] = []
    current_turn: List[Dict[str, str]] = []

    for message in history or []:
        normalized = _normalize_history_message(message)
        if not normalized:
            continue

        role = normalized["role"]
        if role == "system":
            system_messages.append(normalized)
            continue

        if role == "user":
            if current_turn:
                turns.append(current_turn)
            current_turn = [normalized]
            continue

        if not current_turn:
            current_turn = [normalized]
        else:
            current_turn.append(normalized)

    if current_turn:
        turns.append(current_turn)

    return system_messages, turns


def _render_conversation_blocks(turns: Sequence[Sequence[Dict[str, str]]]) -> str:
    """把若干轮对话渲染成适合摘要的纯文本。"""
    role_labels = {
        "user": "用户",
        "assistant": "助手",
        "system": "系统",
    }
    blocks: List[str] = []
    for turn in turns:
        for message in turn:
            blocks.append(f"{role_labels[message['role']]}: {message['content']}")
    return "\n\n".join(blocks)


def _build_summary_messages(
    llm: Any,
    history_batch: Sequence[Sequence[Dict[str, str]]],
    previous_summary: str = "",
) -> str:
    """用大模型对一批较早历史进行递进式摘要。"""
    if llm is None:
        raise ValueError("历史摘要模式需要提供 llm")
    batch_text = _render_conversation_blocks(history_batch)
    summary_messages: List[Any] = [SystemMessage(content=SUMMARY_SYSTEM_PROMPT)]
    if previous_summary.strip():
        summary_messages.append(
            SystemMessage(
                content=(
                    "已有摘要如下，请在保留其核心信息的基础上，结合新增历史继续压缩：\n"
                    f"{previous_summary.strip()}"
                )
            )
        )
    summary_messages.append(
        HumanMessage(
            content=(
                "请概括下面的历史对话，输出一段中文摘要：\n\n"
                f"{batch_text}"
            )
        )
    )
    summary_messages.append(
        HumanMessage(
            content="请只输出摘要正文，不要输出前缀、编号或解释。"
        )
    )
    return _extract_text(llm.invoke(summary_messages)).strip()


def _build_sliding_window_history(
    history: Optional[Iterable[Any]],
    window_n: int,
) -> Tuple[List[Any], List[Dict[str, str]]]:
    """滑动窗口： 只保留最近 n 轮对话作为短期记忆。"""
    system_messages, turns = _split_history_messages(history)
    recent_turns = turns[-window_n:] if window_n > 0 else []
    messages: List[Any] = [SystemMessage(content=item["content"]) for item in system_messages]
    for turn in recent_turns:
        for message in turn:
            if message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                messages.append(AIMessage(content=message["content"]))
            else:
                messages.append(SystemMessage(content=message["content"]))
    return messages, system_messages


def _build_summary_history(
    history: Optional[Iterable[Any]],
    window_n: int,
    summary_m: int,
    llm: Any,
) -> List[Any]:
    """历史摘要： 把超出窗口的历史递进摘要后注入 prompt。"""
    if summary_m <= window_n:
        raise ValueError("short_memory_m 必须大于 short_memory_n")

    system_messages, turns = _split_history_messages(history)
    if len(turns) <= window_n:
        messages: List[Any] = [SystemMessage(content=item["content"]) for item in system_messages]
        for turn in turns:
            for message in turn:
                if message["role"] == "user":
                    messages.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    messages.append(AIMessage(content=message["content"]))
        return messages

    recent_turns = turns[-window_n:]
    older_turns = turns[:-window_n]
    batch_size = max(summary_m - window_n, 1)
    summary_text = ""

    for start in range(0, len(older_turns), batch_size):
        batch = older_turns[start:start + batch_size]
        summary_text = _build_summary_messages(
            llm=llm,
            history_batch=batch,
            previous_summary=summary_text,
        )

    messages = [SystemMessage(content=item["content"]) for item in system_messages]
    if summary_text:
        messages.append(
            SystemMessage(
                content=(
                    "以下是更早历史的递进摘要，仅供继续对话参考：\n"
                    f"{summary_text}"
                )
            )
        )

    for turn in recent_turns:
        for message in turn:
            if message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                messages.append(AIMessage(content=message["content"]))
    return messages


def _compact_long_term_memory_content(content: str) -> str:
    """把长期记忆块压缩成适合 prompt 的简短文本。"""
    normalized = (content or "").strip()
    if not normalized:
        return ""

    topic = ""
    summary = ""
    for line in normalized.splitlines():
        line = line.strip()
        if line.startswith("【主题】"):
            topic = line.removeprefix("【主题】").strip()
        elif line.startswith("【摘要】"):
            summary = line.removeprefix("【摘要】").strip()

    if topic and summary:
        return f"{topic}：{summary}"
    if summary:
        return summary
    if topic:
        return topic
    return normalized.replace("\n", " ")[:LONG_TERM_MEMORY_SNIPPET_LIMIT]


def _build_long_term_memory_messages(
    query: str,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    top_k: int = LONG_TERM_MEMORY_TOP_K,
) -> List[Any]:
    """检索长期记忆并转成系统提示消息。"""
    try:
        memories = search_long_term_memory(
            query=query,
            top_k=top_k,
            conversation_id=conversation_id,
            session_id=session_id,
        )
    except Exception as exc:
        logger.warning("长期记忆检索失败，已跳过: %s", exc)
        return []

    if not memories:
        return []

    lines = [
        "以下是与当前问题相关的长期记忆，仅在相关时参考，不能把未确认信息当作事实："
    ]
    for index, memory in enumerate(memories, start=1):
        lines.append(
            f"{index}. {_compact_long_term_memory_content(getattr(memory, 'content', ''))}"
        )

    return [SystemMessage(content="\n".join(lines))]


def _persist_long_term_memory(
    history: Optional[Iterable[Any]],
    query: str,
    answer: str,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    short_window_n: int = DEFAULT_SHORT_MEMORY_N,
    llm: Optional[Any] = None,
) -> None:
    """把本次对话写入长期记忆。"""
    if not answer or not answer.strip():
        return

    conversation_messages: List[Any] = list(history or [])
    conversation_messages.append({"role": "user", "content": query})
    conversation_messages.append({"role": "assistant", "content": answer})

    try:
        store_semantic_long_term_memory(
            messages=conversation_messages,
            conversation_id=conversation_id,
            session_id=session_id,
            short_window_n=short_window_n,
            llm=llm,
        )
    except Exception as exc:
        logger.warning("长期记忆写入失败，已跳过: %s", exc)


def _persist_long_term_memory_async(
    history: Optional[Iterable[Any]],
    query: str,
    answer: str,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    short_window_n: int = DEFAULT_SHORT_MEMORY_N,
    llm: Optional[Any] = None,
) -> None:
    """后台写入长期记忆，不阻塞本次回答结束。"""
    if not answer or not answer.strip():
        return

    worker = threading.Thread(
        target=_persist_long_term_memory,
        kwargs={
            "history": history,
            "query": query,
            "answer": answer,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "short_window_n": short_window_n,
            "llm": llm,
        },
        daemon=True,
    )
    worker.start()


def build_chat_messages(
    history: Optional[Iterable[Any]],
    query: str,
    allow_retrieval: bool = True,
    long_term_memory_messages: Optional[Iterable[Any]] = None,
    short_memory_strategy: str = SHORT_MEMORY_STRATEGY_WINDOW,
    short_memory_n: int = DEFAULT_SHORT_MEMORY_N,
    short_memory_m: int = DEFAULT_SHORT_MEMORY_M,
    llm: Optional[Any] = None,
) -> List[Any]:
    """把 API 请求转换成 LangChain 消息，并注入工具使用和短期记忆策略。"""
    system_prompt = RAG_SYSTEM_PROMPT if allow_retrieval else DIRECT_CHAT_SYSTEM_PROMPT
    messages: List[Any] = [SystemMessage(content=system_prompt)]
    for message in long_term_memory_messages or []:
        messages.append(message)

    if short_memory_strategy == SHORT_MEMORY_STRATEGY_WINDOW:
        history_messages, _ = _build_sliding_window_history(history, short_memory_n)
        messages.extend(history_messages)
    elif short_memory_strategy == SHORT_MEMORY_STRATEGY_SUMMARY:
        history_messages = _build_summary_history(history, short_memory_n, short_memory_m, llm)
        messages.extend(history_messages)
    else:
        raise ValueError(f"不支持的短期记忆策略: {short_memory_strategy}")

    messages.append(HumanMessage(content=(query or "").strip()))
    return messages


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

    raw_tool_call_chunks = getattr(message, "tool_call_chunks", None) or []
    for index, call in enumerate(raw_tool_call_chunks):
        if not isinstance(call, dict):
            continue
        args = call.get("args", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        normalized.append(
            {
                "id": call.get("id") or f"tool-call-{index}",
                "name": call.get("name"),
                "args": args,
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


def _stream_model_response(model: Any, messages: List[Any]):
    """逐块生成模型响应，同时聚合最终消息以解析工具调用。"""
    response = None
    for chunk in model.stream(messages):
        response = chunk if response is None else response + chunk
        text = _extract_text(chunk)
        if text:
            yield text, None

    if response is None:
        response = model.invoke(messages)
    yield "", response


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
    retrieval_method: str = RETRIEVAL_METHOD_HYBRID,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    short_memory_strategy: str = SHORT_MEMORY_STRATEGY_WINDOW,
    short_memory_n: int = DEFAULT_SHORT_MEMORY_N,
    short_memory_m: int = DEFAULT_SHORT_MEMORY_M,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    llm: Optional[Any] = None,
) -> ChatRunResult:
    """运行一次支持自主 RAG 工具调用的聊天。"""
    normalized_retrieval_method = _validate_retrieval_method(retrieval_method)
    if llm is None:
        from app.core.llm import get_llm

        llm = get_llm(
            provider=provider,
            model=model,
            temperature=temperature,
        )
    long_term_memory_messages = _build_long_term_memory_messages(
        query=query,
        conversation_id=conversation_id,
        session_id=session_id,
    )
    messages = build_chat_messages(
        history,
        query,
        allow_retrieval=use_retrieval,
        long_term_memory_messages=long_term_memory_messages,
        short_memory_strategy=short_memory_strategy,
        short_memory_n=short_memory_n,
        short_memory_m=short_memory_m,
        llm=llm,
    )
    if not use_retrieval:
        text = _extract_text(llm.invoke(messages))
        _persist_long_term_memory_async(
            history=history,
            query=query,
            answer=text,
            conversation_id=conversation_id,
            session_id=session_id,
            short_window_n=short_memory_n,
            llm=llm,
        )
        return ChatRunResult(
            text=text,
            model=model or "default",
        )

    sources: List[Dict[str, Any]] = []
    tools = create_rag_tools(
        sources,
        default_top_k=top_k,
        max_top_k=top_k,
        retrieval_method=normalized_retrieval_method,
    )
    tool_map = {tool.name: tool for tool in tools}
    model_with_tools = llm.bind_tools(tools)

    for _ in range(MAX_TOOL_ROUNDS):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        tool_calls = _normalize_tool_calls(response)
        if not tool_calls:
            text = _extract_text(response)
            _persist_long_term_memory_async(
                history=history,
                query=query,
                answer=text,
                conversation_id=conversation_id,
                session_id=session_id,
                short_window_n=short_memory_n,
                llm=llm,
            )
            return ChatRunResult(
                text=text,
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
    retrieval_method: str = RETRIEVAL_METHOD_HYBRID,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    short_memory_strategy: str = SHORT_MEMORY_STRATEGY_WINDOW,
    short_memory_n: int = DEFAULT_SHORT_MEMORY_N,
    short_memory_m: int = DEFAULT_SHORT_MEMORY_M,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    llm: Optional[Any] = None,
):
    """运行流式聊天，工具决策和工具结果通过事件交给 API 层。"""
    normalized_retrieval_method = _validate_retrieval_method(retrieval_method)
    if llm is None:
        from app.core.llm import get_llm

        llm = get_llm(
            provider=provider,
            model=model,
            temperature=temperature,
        )
    long_term_memory_messages = _build_long_term_memory_messages(
        query=query,
        conversation_id=conversation_id,
        session_id=session_id,
    )
    messages = build_chat_messages(
        history,
        query,
        allow_retrieval=use_retrieval,
        long_term_memory_messages=long_term_memory_messages,
        short_memory_strategy=short_memory_strategy,
        short_memory_n=short_memory_n,
        short_memory_m=short_memory_m,
        llm=llm,
    )
    sources: List[Dict[str, Any]] = []
    final_text = ""

    if not use_retrieval:
        for chunk in llm.stream(messages):
            text = _extract_text(chunk)
            if text:
                final_text += text
                yield {"event": "message", "data": {"content": text}}
        yield {"event": "done", "data": {"sources": []}}
        _persist_long_term_memory_async(
            history=history,
            query=query,
            answer=final_text,
            conversation_id=conversation_id,
            session_id=session_id,
            short_window_n=short_memory_n,
            llm=llm,
        )
        return

    tools = create_rag_tools(
        sources,
        default_top_k=top_k,
        max_top_k=top_k,
        retrieval_method=normalized_retrieval_method,
    )
    tool_map = {tool.name: tool for tool in tools}
    model_with_tools = llm.bind_tools(tools)

    for _ in range(MAX_TOOL_ROUNDS):
        response = None
        streamed_text = ""
        for text, final_response in _stream_model_response(model_with_tools, messages):
            if text:
                streamed_text += text
                yield {"event": "message", "data": {"content": text}}
            if final_response is not None:
                response = final_response

        if response is None:
            response = model_with_tools.invoke(messages)
        messages.append(response)
        tool_calls = _normalize_tool_calls(response)

        if not tool_calls:
            final_text = streamed_text or _extract_text(response)
            if not streamed_text and final_text:
                yield {"event": "message", "data": {"content": final_text}}
            yield {
                "event": "done",
                "data": {"sources": _deduplicate_sources(sources)},
            }
            _persist_long_term_memory_async(
                history=history,
                query=query,
                answer=final_text,
                conversation_id=conversation_id,
                session_id=session_id,
                short_window_n=short_memory_n,
                llm=llm,
            )
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


def _validate_short_memory_options(
    short_memory_strategy: str,
    short_memory_n: int,
    short_memory_m: int,
) -> None:
    """校验短期记忆参数。"""
    if short_memory_n < 1:
        raise ValueError("short_memory_n 必须大于 0")
    if short_memory_strategy not in {SHORT_MEMORY_STRATEGY_WINDOW, SHORT_MEMORY_STRATEGY_SUMMARY}:
        raise ValueError(f"不支持的短期记忆策略: {short_memory_strategy}")
    if short_memory_strategy == SHORT_MEMORY_STRATEGY_SUMMARY and short_memory_m <= short_memory_n:
        raise ValueError("short_memory_m 必须大于 short_memory_n")


def _validate_retrieval_method(retrieval_method: str) -> str:
    """校验检索方式。"""
    normalized = (retrieval_method or RETRIEVAL_METHOD_HYBRID).strip().lower()
    if normalized not in {
        RETRIEVAL_METHOD_HYBRID,
        RETRIEVAL_METHOD_DENSE,
        RETRIEVAL_METHOD_SPARSE,
    }:
        raise ValueError(f"不支持的检索方式: {retrieval_method}")
    return normalized


def generate_chat(
    query: str,
    history: Optional[Iterable[Any]] = None,
    top_k: int = 5,
    use_retrieval: bool = True,
    retrieval_method: str = RETRIEVAL_METHOD_HYBRID,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    short_memory_strategy: str = SHORT_MEMORY_STRATEGY_WINDOW,
    short_memory_n: int = DEFAULT_SHORT_MEMORY_N,
    short_memory_m: int = DEFAULT_SHORT_MEMORY_M,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ChatRunResult:
    """Chat API 使用的同步业务入口。"""
    normalized_retrieval_method = _validate_retrieval_method(retrieval_method)
    _validate_chat_options(query, top_k)
    _validate_short_memory_options(short_memory_strategy, short_memory_n, short_memory_m)
    return run_chat(
        query=query,
        history=history,
        use_retrieval=use_retrieval,
        top_k=top_k,
        retrieval_method=normalized_retrieval_method,
        conversation_id=conversation_id,
        session_id=session_id,
        short_memory_strategy=short_memory_strategy,
        short_memory_n=short_memory_n,
        short_memory_m=short_memory_m,
        provider=provider,
        model=model,
        temperature=temperature,
    )


def stream_chat_events(
    query: str,
    history: Optional[Iterable[Any]] = None,
    top_k: int = 5,
    use_retrieval: bool = True,
    retrieval_method: str = RETRIEVAL_METHOD_HYBRID,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    short_memory_strategy: str = SHORT_MEMORY_STRATEGY_WINDOW,
    short_memory_n: int = DEFAULT_SHORT_MEMORY_N,
    short_memory_m: int = DEFAULT_SHORT_MEMORY_M,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """Chat API 使用的流式业务入口。"""
    normalized_retrieval_method = _validate_retrieval_method(retrieval_method)
    _validate_chat_options(query, top_k)
    _validate_short_memory_options(short_memory_strategy, short_memory_n, short_memory_m)
    yield from stream_chat(
        query=query,
        history=history,
        use_retrieval=use_retrieval,
        top_k=top_k,
        retrieval_method=normalized_retrieval_method,
        conversation_id=conversation_id,
        session_id=session_id,
        short_memory_strategy=short_memory_strategy,
        short_memory_n=short_memory_n,
        short_memory_m=short_memory_m,
        provider=provider,
        model=model,
        temperature=temperature,
    )


def warmup_chat_runtime() -> Dict[str, Any]:
    """预热聊天链路里最重的初始化步骤。"""
    result: Dict[str, Any] = {
        "llm_warmed": False,
        "embedding_warmed": False,
        "provider": None,
    }

    try:
        from app.core.llm import get_default_llm

        llm = get_default_llm()
        result["llm_warmed"] = True
        result["provider"] = getattr(llm, "active_provider", None)
    except Exception as exc:
        logger.warning("LLM 预热失败，已跳过: %s", exc)

    try:
        from app.core.embeddings import get_default_embeddings

        embeddings = get_default_embeddings()
        embeddings.embed_query("warmup")
        result["embedding_warmed"] = True
    except Exception as exc:
        logger.warning("Embedding 预热失败，已跳过: %s", exc)

    return result
