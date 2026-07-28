"""通用切块工具。"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, List, Mapping, Sequence

from app.core.constants import DOCUMENT_CHUNK_OVERLAP, DOCUMENT_CHUNK_SIZE
from app.core.constants import CHAT_CHUNK_OVERLAP, CHAT_CHUNK_SIZE

DEFAULT_TEXT_SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", "")

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
) -> List[str]:
    """按文档默认规则切分文本。

    文档场景优先保留段落和标点边界，必要时再退化为更细粒度切分。
    """
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
