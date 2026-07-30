"""通用切块工具。"""
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Callable, Iterable, List, Mapping, Sequence

from app.core.constants import DOCUMENT_ARTICLE_CHUNK_OVERLAP, DOCUMENT_ARTICLE_CHUNK_SIZE
from app.core.constants import DOCUMENT_CHUNK_OVERLAP, DOCUMENT_CHUNK_SIZE
from app.core.constants import CHAT_CHUNK_OVERLAP, CHAT_CHUNK_SIZE

DEFAULT_TEXT_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", "")
ARTICLE_TEXT_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "：", "，", ".", "!", "?", ";", ":", " ", "")
LONG_TERM_MEMORY_TOKEN_LIMIT = 700

ARTICLE_FILE_TYPES = {"pdf", "doc", "docx", "md", "txt", "html", "htm", "rtf", "markdown"}

_ROLE_LABELS = {
    "system": "系统",
    "user": "用户",
    "assistant": "助手",
}


@dataclass(frozen=True)
class ConversationMessage:
    """标准化后的对话消息。"""

    role: str
    content: str


@dataclass(frozen=True)
class ConversationChunk:
    """对话记忆块。"""

    chunk_index: int
    text: str
    turn_start: int
    turn_end: int
    message_count: int
    topic: str = ""
    summary: str = ""
    trigger_reason: str = ""


@dataclass(frozen=True)
class LongTermMemoryChunk:
    """语义长期记忆块。"""

    chunk_index: int
    topic: str
    summary: str
    transcript: str
    text: str
    turn_start: int
    turn_end: int
    message_count: int
    trigger_reason: str


def normalize_text_content(content: Any) -> str:
    """把任意内容统一转换为可切块的纯文本。

    这里主要处理消息、列表片段和普通对象三类输入，避免不同模块重复写
    “把内容转成字符串”的逻辑。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


@lru_cache(maxsize=32)
def _get_recursive_text_splitter(chunk_size: int, chunk_overlap: int, separators: tuple[str, ...]):
    """创建可复用的递归文本切分器。

    splitters 本身创建成本不高，但在高频入库/查询场景里复用实例更省心。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators),
    )


def split_text_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str] = DEFAULT_TEXT_SEPARATORS,
) -> List[str]:
    """按递归分隔符把纯文本切成 chunk 列表。

    这个函数是文档、对话记忆等场景的公共底座；上层模块只需要决定
    chunk 大小和分隔符顺序即可。
    """
    normalized_text = (text or "").strip()
    if not normalized_text:
        return []

    separator_tuple = tuple(separators)
    splitter = _get_recursive_text_splitter(chunk_size, chunk_overlap, separator_tuple)
    return [chunk.strip() for chunk in splitter.split_text(normalized_text) if chunk.strip()]


def split_document_text(
    text: str,
    chunk_size: int = DOCUMENT_CHUNK_SIZE,
    chunk_overlap: int = DOCUMENT_CHUNK_OVERLAP,
    file_type: str | None = None,
) -> List[str]:
    """按文档默认规则切分文本。

    文章类文档优先走更细的句子级切分，表格/结构化文档保留较大的块。
    """
    normalized_file_type = (file_type or "").strip().lower().lstrip(".")
    if normalized_file_type in ARTICLE_FILE_TYPES:
        return split_text_chunks(
            text=text,
            chunk_size=DOCUMENT_ARTICLE_CHUNK_SIZE,
            chunk_overlap=DOCUMENT_ARTICLE_CHUNK_OVERLAP,
            separators=ARTICLE_TEXT_SEPARATORS,
        )

    return split_text_chunks(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_TEXT_SEPARATORS,
    )


def _normalize_message(message: Any, include_system: bool = False) -> ConversationMessage | None:
    """把任意消息对象转换成统一结构。"""
    if isinstance(message, ConversationMessage):
        role = message.role.strip().lower()
        content = message.content.strip()
    elif isinstance(message, Mapping):
        role = str(message.get("role", "")).strip().lower()
        content = normalize_text_content(message.get("content", "")).strip()
    else:
        role = str(getattr(message, "role", "")).strip().lower()
        content = normalize_text_content(getattr(message, "content", "")).strip()

    if not role or not content:
        return None
    if role == "system" and not include_system:
        return None
    if role not in _ROLE_LABELS:
        return None
    return ConversationMessage(role=role, content=content)


def _normalize_messages(messages: Iterable[Any], include_system: bool = False) -> List[ConversationMessage]:
    """过滤无效消息，并统一成标准结构。"""
    normalized: List[ConversationMessage] = []
    for message in messages:
        normalized_message = _normalize_message(message, include_system=include_system)
        if normalized_message is not None:
            normalized.append(normalized_message)
    return normalized


def _message_to_block(message: ConversationMessage) -> str:
    """把单条消息渲染成适合入库的文本块。"""
    return f"{_ROLE_LABELS[message.role]}: {message.content.strip()}"


def format_conversation_messages(messages: Sequence[Any], include_system: bool = False) -> str:
    """把消息序列渲染成连续对话文本。"""
    normalized_messages = _normalize_messages(messages, include_system=include_system)
    return "\n\n".join(_message_to_block(message) for message in normalized_messages)


def split_conversation_turns(
    messages: Sequence[Any],
    include_system: bool = False,
) -> List[List[ConversationMessage]]:
    """把消息序列拆成按轮次组织的对话。"""
    normalized_messages = _normalize_messages(messages, include_system=include_system)
    if not normalized_messages:
        return []

    turns: List[List[ConversationMessage]] = []
    current_turn: List[ConversationMessage] = []

    for message in normalized_messages:
        if message.role == "user":
            if current_turn:
                turns.append(current_turn)
            current_turn = [message]
            continue

        if not current_turn:
            current_turn = [message]
        else:
            current_turn.append(message)

    if current_turn:
        turns.append(current_turn)

    return turns


def split_conversation_chunks(
    conversation_text: str,
    chunk_size: int = CHAT_CHUNK_SIZE,
    chunk_overlap: int = CHAT_CHUNK_OVERLAP,
) -> List[str]:
    """把原始对话文本切成适合长期记忆入库的 chunk 列表。"""
    return split_text_chunks(
        text=conversation_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def estimate_text_tokens(text: str) -> int:
    """用轻量启发式估算 token 数。"""
    normalized_text = (text or "").strip()
    if not normalized_text:
        return 0
    tokens = re.findall(r"[A-Za-z0-9_]+|[一-鿿]", normalized_text)
    return max(1, len(tokens))


def render_long_term_memory_transcript(turns: Sequence[Sequence[ConversationMessage]]) -> str:
    """把轮次对话渲染成长文本原文。"""
    if not turns:
        return ""
    return "\n\n".join(
        "\n".join(_message_to_block(message) for message in turn)
        for turn in turns
    )


def format_long_term_memory_chunk(
    summary: str,
    transcript: str,
    topic: str = "",
) -> str:
    """把摘要和原文拼成标准化长期记忆文本。"""
    parts: List[str] = []
    if topic.strip():
        parts.append(f"【主题】{topic.strip()}")
    parts.append(f"【摘要】{(summary or '').strip()}")
    parts.append("【原文】")
    parts.append((transcript or "").strip())
    return "\n".join(parts).strip()


def _coerce_summary_result(result: Any) -> dict[str, str]:
    """兼容摘要函数返回 dict、tuple 或字符串。"""
    if isinstance(result, dict):
        return {
            "topic": str(result.get("topic", "") or "").strip(),
            "summary": str(result.get("summary", "") or "").strip(),
        }
    if isinstance(result, tuple) and result:
        if len(result) == 1:
            return {"topic": "", "summary": str(result[0] or "").strip()}
        return {"topic": str(result[0] or "").strip(), "summary": str(result[1] or "").strip()}
    return {"topic": "", "summary": str(result or "").strip()}


def _build_single_turn_memory_chunk(
    chunk_index: int,
    turn: Sequence[ConversationMessage],
    summarize_chunk: Callable[[str], Any],
    turn_index: int,
    trigger_reason: str,
) -> LongTermMemoryChunk:
    """把超长单轮直接封成长期记忆块。"""
    transcript = render_long_term_memory_transcript([turn])
    summary_result = _coerce_summary_result(summarize_chunk(transcript))
    summary = summary_result["summary"] or transcript[:120].strip()
    topic = summary_result["topic"] or summary
    return LongTermMemoryChunk(
        chunk_index=chunk_index,
        topic=topic,
        summary=summary,
        transcript=transcript,
        text=format_long_term_memory_chunk(summary=summary, transcript=transcript, topic=topic),
        turn_start=turn_index,
        turn_end=turn_index,
        message_count=1,
        trigger_reason=trigger_reason,
    )


def build_semantic_memory_chunks(
    messages: Sequence[Any],
    *,
    should_split_topic: Callable[[str, str], bool],
    summarize_chunk: Callable[[str], Any],
    token_limit: int = LONG_TERM_MEMORY_TOKEN_LIMIT,
    include_system: bool = False,
) -> List[LongTermMemoryChunk]:
    """按语义单元构建长期记忆块。"""
    turns = split_conversation_turns(messages, include_system=include_system)
    if not turns:
        return []

    chunks: List[LongTermMemoryChunk] = []
    buffer_turns: List[List[ConversationMessage]] = []
    buffer_start = 0

    def flush(reason: str) -> None:
        nonlocal buffer_turns, buffer_start
        if not buffer_turns:
            return

        transcript = render_long_term_memory_transcript(buffer_turns)
        summary_result = _coerce_summary_result(summarize_chunk(transcript))
        summary = summary_result["summary"] or transcript[:120].strip()
        topic = summary_result["topic"] or summary
        chunks.append(
            LongTermMemoryChunk(
                chunk_index=len(chunks),
                topic=topic,
                summary=summary,
                transcript=transcript,
                text=format_long_term_memory_chunk(summary=summary, transcript=transcript, topic=topic),
                turn_start=buffer_start,
                turn_end=buffer_start + len(buffer_turns) - 1,
                message_count=len(buffer_turns),
                trigger_reason=reason,
            )
        )
        buffer_turns = []

    for turn_index, turn in enumerate(turns):
        turn_text = render_long_term_memory_transcript([turn])
        if not turn_text.strip():
            continue

        if not buffer_turns:
            if estimate_text_tokens(turn_text) > token_limit:
                chunks.append(
                    _build_single_turn_memory_chunk(
                        chunk_index=len(chunks),
                        turn=turn,
                        summarize_chunk=summarize_chunk,
                        turn_index=turn_index,
                        trigger_reason="token_limit",
                    )
                )
                continue
            buffer_turns = [turn]
            buffer_start = turn_index
            continue

        current_text = render_long_term_memory_transcript(buffer_turns)
        candidate_text = f"{current_text}\n\n{turn_text}".strip()

        if should_split_topic(current_text, turn_text):
            flush("topic_change")
            if estimate_text_tokens(turn_text) > token_limit:
                chunks.append(
                    _build_single_turn_memory_chunk(
                        chunk_index=len(chunks),
                        turn=turn,
                        summarize_chunk=summarize_chunk,
                        turn_index=turn_index,
                        trigger_reason="topic_change",
                    )
                )
                continue
            buffer_turns = [turn]
            buffer_start = turn_index
            continue

        if estimate_text_tokens(candidate_text) > token_limit:
            flush("token_limit")
            if estimate_text_tokens(turn_text) > token_limit:
                chunks.append(
                    _build_single_turn_memory_chunk(
                        chunk_index=len(chunks),
                        turn=turn,
                        summarize_chunk=summarize_chunk,
                        turn_index=turn_index,
                        trigger_reason="token_limit",
                    )
                )
                continue
            buffer_turns = [turn]
            buffer_start = turn_index
            continue

        buffer_turns.append(turn)

    flush("flush")
    return chunks


def _split_message_into_blocks(
    message: ConversationMessage,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """把单条过长消息拆成多个块，同时保留角色前缀。"""
    block = _message_to_block(message)
    if len(block) <= chunk_size:
        return [block]

    label = _ROLE_LABELS[message.role]
    content_chunk_size = max(chunk_size - len(label) - 2, 1)
    content_chunk_overlap = max(chunk_overlap - len(label) - 2, 0)
    pieces = split_text_chunks(
        text=message.content.strip(),
        chunk_size=content_chunk_size,
        chunk_overlap=content_chunk_overlap,
    )
    return [f"{label}: {piece.strip()}" for piece in pieces if piece.strip()]


def _joined_length(blocks: Sequence[tuple[int, str]]) -> int:
    """估算块拼接后的字符长度。"""
    if not blocks:
        return 0
    return sum(len(text) for _, text in blocks) + max(len(blocks) - 1, 0) * 2


def _build_overlap_tail(
    blocks: Sequence[tuple[int, str]],
    chunk_overlap: int,
) -> List[tuple[int, str]]:
    """从上一块尾部保留少量上下文，减少相邻 chunk 的断裂感。"""
    if not blocks or chunk_overlap <= 0:
        return []

    tail: List[tuple[int, str]] = []
    current_length = 0
    for turn_index, text in reversed(blocks):
        item_length = len(text)
        projected = item_length if not tail else current_length + 2 + item_length
        if tail and projected > chunk_overlap:
            break
        tail.append((turn_index, text))
        current_length = projected
        if current_length >= chunk_overlap:
            break

    if not tail:
        tail.append(blocks[-1])

    return list(reversed(tail))


def _build_chunk(
    blocks: Sequence[tuple[int, str]],
    chunk_index: int,
) -> ConversationChunk:
    """把若干块合并成一个结构化记忆 chunk。"""
    turn_indexes = [turn_index for turn_index, _ in blocks]
    return ConversationChunk(
        chunk_index=chunk_index,
        text="\n\n".join(text for _, text in blocks),
        turn_start=min(turn_indexes),
        turn_end=max(turn_indexes),
        message_count=len(set(turn_indexes)),
    )


def build_conversation_chunks(
    messages: Sequence[Any],
    chunk_size: int = CHAT_CHUNK_SIZE,
    chunk_overlap: int = CHAT_CHUNK_OVERLAP,
    include_system: bool = False,
) -> List[ConversationChunk]:
    """把消息序列切成适合长期记忆入库的结构化 chunk。"""
    normalized_messages = _normalize_messages(messages, include_system=include_system)
    if not normalized_messages:
        return []

    chunks: List[ConversationChunk] = []
    current_blocks: List[tuple[int, str]] = []

    for turn_index, message in enumerate(normalized_messages):
        message_blocks = _split_message_into_blocks(message, chunk_size, chunk_overlap)
        for block in message_blocks:
            block_length = len(block)
            if current_blocks and _joined_length(current_blocks + [(turn_index, block)]) > chunk_size:
                chunks.append(_build_chunk(current_blocks, len(chunks)))
                current_blocks = _build_overlap_tail(current_blocks, chunk_overlap)
                if current_blocks and _joined_length(current_blocks + [(turn_index, block)]) > chunk_size:
                    current_blocks = []

            if not current_blocks and block_length > chunk_size:
                chunks.append(
                    ConversationChunk(
                        chunk_index=len(chunks),
                        text=block,
                        turn_start=turn_index,
                        turn_end=turn_index,
                        message_count=1,
                    )
                )
                continue

            current_blocks.append((turn_index, block))

    if current_blocks:
        chunks.append(_build_chunk(current_blocks, len(chunks)))

    return chunks
