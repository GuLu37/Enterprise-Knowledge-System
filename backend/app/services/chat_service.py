"""聊天业务编排：先做意图路由，命中知识库意图后调用 RAG 工具。"""
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from app.services.memory_service import (
    search_long_term_memory,
    store_semantic_long_term_memory,
)
from app.tools.rag import run_rag_tool

logger = logging.getLogger(__name__)

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
    "你是企业知识库助手。当前问题已识别为需要知识库检索。"
    "请优先依据已检索到的资料回答，不能把没有检索到的内容当成企业内部事实。"
    "如果资料不足，请明确说明缺少哪些依据，并给出可继续补充资料的方向。"
    "请使用简洁的纯文本段落回答，不要使用 Markdown 标题、加粗、代码围栏、引用或列表符号。"
)

DIRECT_CHAT_SYSTEM_PROMPT = (
    "你是企业知识库助手。当前问题不需要知识库检索，请直接回答用户问题。"
    "不要假装查阅过企业内部文档；涉及内部事实且无法确认时，请明确说明。"
    "请使用简洁的纯文本段落回答，不要使用 Markdown 标题、加粗、代码围栏、引用或列表符号。"
)

INTENT_ROUTER_SYSTEM_PROMPT = (
    "你是一个企业对话意图路由器。"
    "你的唯一任务是判断当前用户问题是否需要进入企业知识库检索流程。"
    "请只输出严格 JSON，不要输出解释、前后缀或代码块。"
    "JSON 结构必须是："
    "{\"route\":\"rag\"|\"direct\",\"reason\":\"简短原因\",\"confidence\":0到1之间的小数}"
    "判定规则："
    "1. 需要企业制度、流程、项目资料、上传文档、公司内部事实核验、引用来源时，route=rag。"
    "2. 只是闲聊、写作、翻译、润色、代码解释、总结、通用知识问答时，route=direct。"
    "3. 如果当前问题依赖近期对话上下文，请结合上下文一起判断。"
)

INTENT_ROUTER_HUMAN_PROMPT = (
    "请判断下面这条用户消息是否应该调用企业知识库检索。"
    "如果需要调用知识库，请输出 route=rag；否则输出 route=direct。"
    "用户消息：\n{query}"
)

INTENT_JSON_PATTERN = re.compile(r"\{[\s\S]*\}")
MARKDOWN_EMPHASIS_PATTERN = re.compile(r"(\*\*|__)(.*?)\1", re.DOTALL)
MARKDOWN_CODE_PATTERN = re.compile(r"(`{1,3})(.*?)\1", re.DOTALL)
MARKDOWN_HEADING_PATTERN = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
MARKDOWN_QUOTE_PATTERN = re.compile(r"(?m)^\s*>\s?")
MARKDOWN_LIST_PATTERN = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)")


@dataclass
class ChatRunResult:
    """一次聊天运行的结果。"""

    text: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    model: str = "default"


@dataclass
class QueryIntent:
    """用户 query 的轻量意图路由结果。"""

    needs_retrieval: bool
    reason: str = "direct"
    source: str = "rule"
    confidence: Optional[float] = None


KNOWLEDGE_INTENT_KEYWORDS = (
    "知识库",
    "文档",
    "资料",
    "上传",
    "附件",
    "引用",
    "来源",
    "检索",
    "搜索",
    "查一下",
    "查询",
    "根据",
    "基于",
    "结合",
    "公司",
    "企业",
    "内部",
    "制度",
    "流程",
    "规范",
    "政策",
    "手册",
    "项目",
    "合同",
    "申请",
    "权限",
    "配置",
    "环境",
    "接口",
    "操作",
    "使用说明",
    "报销",
    "审批",
    "员工",
    "客户",
)

DIRECT_INTENT_KEYWORDS = (
    "你好",
    "在吗",
    "谢谢",
    "翻译",
    "润色",
    "改写",
    "写一段",
    "生成",
    "创作",
    "代码",
    "解释一下这段",
)


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


def clean_rag_response_text(text: str) -> str:
    """清理 RAG 回答中的 Markdown 展示符号，保留段落和语义文本。"""
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"```[^\n`]*\n?", "", normalized)
    normalized = normalized.replace("```", "")
    normalized = MARKDOWN_EMPHASIS_PATTERN.sub(r"\2", normalized)
    normalized = MARKDOWN_CODE_PATTERN.sub(r"\2", normalized)
    normalized = MARKDOWN_HEADING_PATTERN.sub("", normalized)
    normalized = MARKDOWN_QUOTE_PATTERN.sub("", normalized)
    normalized = MARKDOWN_LIST_PATTERN.sub("", normalized)
    normalized = "\n".join(line.strip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _build_intent_router_messages(query: str, history: Optional[Iterable[Any]] = None) -> List[Any]:
    """构造用于意图路由的轻量提示消息。"""
    history_hint = ""
    if history:
        try:
            _, turns = _split_history_messages(history)
            recent_turns = turns[-2:]
            if recent_turns:
                history_hint = _render_conversation_blocks(recent_turns)
        except Exception:
            history_hint = ""

    human_prompt = INTENT_ROUTER_HUMAN_PROMPT.format(query=(query or "").strip())
    if history_hint.strip():
        human_prompt += f"\n\n最近对话上下文：\n{history_hint.strip()}"

    return [
        SystemMessage(content=INTENT_ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]


def _parse_intent_router_payload(text: str) -> Dict[str, Any]:
    """从 LLM 输出中提取意图路由 JSON。"""
    normalized = (text or "").strip()
    if not normalized:
        return {}

    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?", "", normalized, flags=re.IGNORECASE).strip()
        normalized = re.sub(r"```$", "", normalized).strip()

    match = INTENT_JSON_PATTERN.search(normalized)
    candidate = match.group(0) if match else normalized
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _intent_from_payload(payload: Dict[str, Any], source: str = "llm") -> Optional[QueryIntent]:
    """把解析后的路由结果标准化成 QueryIntent。"""
    route = str(
        payload.get("route")
        or payload.get("intent")
        or payload.get("decision")
        or payload.get("action")
        or ""
    ).strip().lower()
    if route in {"rag", "retrieve", "retrieval", "knowledge", "kb"}:
        needs_retrieval = True
    elif route in {"direct", "chat", "answer", "none", "no_rag"}:
        needs_retrieval = False
    else:
        return None

    reason = str(payload.get("reason") or payload.get("explanation") or route).strip()
    confidence_value = payload.get("confidence")
    confidence: Optional[float] = None
    try:
        if confidence_value is not None and confidence_value != "":
            confidence = max(0.0, min(1.0, float(confidence_value)))
    except (TypeError, ValueError):
        confidence = None

    return QueryIntent(
        needs_retrieval=needs_retrieval,
        reason=f"{source}:{reason}" if reason else source,
        source=source,
        confidence=confidence,
    )


def _rule_route_query_intent(query: str, use_retrieval: bool = True) -> QueryIntent:
    """用低延迟规则判断本轮是否需要进入知识库 RAG 工具。"""
    normalized_query = (query or "").strip()
    if not use_retrieval:
        return QueryIntent(needs_retrieval=False, reason="retrieval_disabled", source="rule")
    if not normalized_query:
        return QueryIntent(needs_retrieval=False, reason="empty_query", source="rule")

    lowered = normalized_query.lower()
    has_knowledge_signal = any(keyword in lowered for keyword in KNOWLEDGE_INTENT_KEYWORDS)
    has_direct_signal = any(keyword in lowered for keyword in DIRECT_INTENT_KEYWORDS)

    if has_knowledge_signal:
        return QueryIntent(needs_retrieval=True, reason="knowledge_keyword", source="rule")

    if has_direct_signal:
        return QueryIntent(needs_retrieval=False, reason="direct_keyword", source="rule")

    return QueryIntent(needs_retrieval=False, reason="direct_default", source="rule")


def route_query_intent(
    query: str,
    use_retrieval: bool = True,
    llm: Optional[Any] = None,
    history: Optional[Iterable[Any]] = None,
) -> QueryIntent:
    """规则优先判断是否走 RAG，仅对知识库相关问题使用 LLM 消歧。"""
    normalized_query = (query or "").strip()
    if not use_retrieval:
        return QueryIntent(needs_retrieval=False, reason="retrieval_disabled", source="rule")
    if not normalized_query:
        return QueryIntent(needs_retrieval=False, reason="empty_query", source="rule")

    rule_intent = _rule_route_query_intent(normalized_query, use_retrieval=use_retrieval)
    if not rule_intent.needs_retrieval:
        logger.info(
            "意图路由结果: source=%s route=direct reason=%s",
            rule_intent.source,
            rule_intent.reason,
        )
        return rule_intent

    if llm is not None:
        try:
            response = llm.invoke(_build_intent_router_messages(normalized_query, history=history))
            payload = _parse_intent_router_payload(_extract_text(response))
            intent = _intent_from_payload(payload, source="llm")
            if intent is not None:
                logger.info(
                    "意图路由结果: source=%s route=%s confidence=%s reason=%s",
                    intent.source,
                    "rag" if intent.needs_retrieval else "direct",
                    intent.confidence,
                    intent.reason,
                )
                return intent
        except Exception as exc:
            logger.warning("LLM 意图路由失败，已回退规则判断: %s", exc)

    logger.info(
        "意图路由结果: source=%s route=%s reason=%s",
        rule_intent.source,
        "rag" if rule_intent.needs_retrieval else "direct",
        rule_intent.reason,
    )
    return rule_intent


def _build_rag_context_message(rag_context: str, rag_message: str = "") -> Optional[SystemMessage]:
    """把 RAG 工具输出转换成回答阶段的系统上下文。"""
    context = (rag_context or "").strip()
    if context:
        return SystemMessage(
            content=(
                "以下是本轮 RAG 工具从企业知识库检索、融合、重排和过滤后的资料：\n\n"
                f"{context}\n\n"
                "请基于这些资料作答。资料没有覆盖的信息不要编造。"
            )
        )

    return SystemMessage(
        content=(
            "本轮已进入 RAG 工具，但知识库未检索到足够相关资料。"
            f"{rag_message or '请直接说明资料不足，不要编造企业内部事实。'}"
        )
    )


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
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    top_k: int = LONG_TERM_MEMORY_TOP_K,
) -> List[Any]:
    """检索长期记忆并转成系统提示消息。"""
    try:
        memories = search_long_term_memory(
            query=query,
            top_k=top_k,
            user_id=user_id,
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
    user_id: Optional[str] = None,
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
            user_id=user_id,
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
    user_id: Optional[str] = None,
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
            "user_id": user_id,
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
    rag_context: str = "",
    rag_message: str = "",
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
    if allow_retrieval:
        context_message = _build_rag_context_message(rag_context, rag_message=rag_message)
        if context_message is not None:
            messages.append(context_message)

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


def run_chat(
    query: str,
    history: Optional[Iterable[Any]] = None,
    use_retrieval: bool = True,
    top_k: int = 5,
    retrieval_method: str = RETRIEVAL_METHOD_HYBRID,
    user_id: Optional[str] = None,
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
    """运行一次先路由、再按需调用 RAG 工具的聊天。"""
    normalized_retrieval_method = _validate_retrieval_method(retrieval_method)
    if llm is None:
        if provider is None and model is None and temperature is None:
            from app.core.llm import get_default_llm

            llm = get_default_llm()
        else:
            from app.core.llm import get_llm

            llm = get_llm(
                provider=provider,
                model=model,
                temperature=temperature,
            )
    intent = route_query_intent(query, use_retrieval=use_retrieval, llm=llm, history=history)
    sources: List[Dict[str, Any]] = []
    rag_payload: Dict[str, Any] = {}
    if intent.needs_retrieval:
        rag_payload = run_rag_tool(
            query=query,
            default_top_k=top_k,
            max_top_k=top_k,
            retrieval_method=normalized_retrieval_method,
            sources_sink=sources,
            llm=llm,
        )

    long_term_memory_messages = _build_long_term_memory_messages(
        query=query,
        user_id=user_id,
        conversation_id=conversation_id,
        session_id=session_id,
    )
    messages = build_chat_messages(
        history,
        query,
        allow_retrieval=intent.needs_retrieval,
        rag_context=str(rag_payload.get("context") or ""),
        rag_message=str(rag_payload.get("message") or ""),
        long_term_memory_messages=long_term_memory_messages,
        short_memory_strategy=short_memory_strategy,
        short_memory_n=short_memory_n,
        short_memory_m=short_memory_m,
        llm=llm,
    )
    text = clean_rag_response_text(_extract_text(llm.invoke(messages)))
    _persist_long_term_memory_async(
        history=history,
        query=query,
        answer=text,
        user_id=user_id,
        conversation_id=conversation_id,
        session_id=session_id,
        short_window_n=short_memory_n,
        llm=llm,
    )
    return ChatRunResult(
        text=text,
        sources=_deduplicate_sources(sources) if intent.needs_retrieval else [],
        model=model or "default",
    )


def stream_chat(
    query: str,
    history: Optional[Iterable[Any]] = None,
    use_retrieval: bool = True,
    top_k: int = 5,
    retrieval_method: str = RETRIEVAL_METHOD_HYBRID,
    user_id: Optional[str] = None,
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
    """运行流式聊天，先做意图路由，命中后再调用 RAG 工具。"""
    normalized_retrieval_method = _validate_retrieval_method(retrieval_method)
    if llm is None:
        if provider is None and model is None and temperature is None:
            from app.core.llm import get_default_llm

            llm = get_default_llm()
        else:
            from app.core.llm import get_llm

            llm = get_llm(
                provider=provider,
                model=model,
                temperature=temperature,
            )
    intent = route_query_intent(query, use_retrieval=use_retrieval, llm=llm, history=history)
    sources: List[Dict[str, Any]] = []
    rag_payload: Dict[str, Any] = {}
    if intent.needs_retrieval:
        yield {
            "event": "tool_call",
            "data": {
                "name": "search_knowledge_base",
                "args": {
                    "query": query,
                    "top_k": top_k,
                    "retrieval_method": normalized_retrieval_method,
                    "reason": intent.reason,
                },
            },
        }
        rag_payload = run_rag_tool(
            query=query,
            default_top_k=top_k,
            max_top_k=top_k,
            retrieval_method=normalized_retrieval_method,
            sources_sink=sources,
            llm=llm,
        )

    long_term_memory_messages = _build_long_term_memory_messages(
        query=query,
        user_id=user_id,
        conversation_id=conversation_id,
        session_id=session_id,
    )
    messages = build_chat_messages(
        history,
        query,
        allow_retrieval=intent.needs_retrieval,
        rag_context=str(rag_payload.get("context") or ""),
        rag_message=str(rag_payload.get("message") or ""),
        long_term_memory_messages=long_term_memory_messages,
        short_memory_strategy=short_memory_strategy,
        short_memory_n=short_memory_n,
        short_memory_m=short_memory_m,
        llm=llm,
    )
    final_text = ""

    for chunk in llm.stream(messages):
        text = _extract_text(chunk)
        if text:
            final_text += text
            yield {"event": "message", "data": {"content": text}}

    deduplicated_sources = _deduplicate_sources(sources) if intent.needs_retrieval else []
    if intent.needs_retrieval:
        yield {
            "event": "tool_result",
            "data": {
                "sources": deduplicated_sources,
                "expanded_queries": rag_payload.get("expanded_queries") or [],
                "message": rag_payload.get("message") or "",
            },
        }

    cleaned_final_text = clean_rag_response_text(final_text)
    yield {
        "event": "done",
        "data": {
            "sources": deduplicated_sources,
            "content": cleaned_final_text,
        },
    }
    _persist_long_term_memory_async(
        history=history,
        query=query,
        answer=cleaned_final_text,
        user_id=user_id,
        conversation_id=conversation_id,
        session_id=session_id,
        short_window_n=short_memory_n,
        llm=llm,
    )
    return


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
    user_id: Optional[str] = None,
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
        user_id=user_id,
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
    user_id: Optional[str] = None,
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
        user_id=user_id,
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
