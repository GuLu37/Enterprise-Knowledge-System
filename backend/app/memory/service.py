"""长期记忆业务层。"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
import uuid

from app.config import settings
from app.core.constants import CHAT_CHUNK_OVERLAP, CHAT_CHUNK_SIZE
from app.rag.retrieval.base import RetrievalResult
from app.utils.chunking import (
    LongTermMemoryChunk,
    build_semantic_memory_chunks,
    split_conversation_turns,
    split_conversation_chunks,
)
from app.utils.exceptions import RetrievalException, VectorStoreException
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


def _escape_filter_value(value: str) -> str:
    """转义 Milvus 过滤表达式中的字符串值。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_memory_filter(
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
    topic: Optional[str] = None,
) -> str:
    """构建记忆检索过滤条件。"""
    clauses: List[str] = []
    if conversation_id:
        clauses.append(f'conversation_id == "{_escape_filter_value(conversation_id)}"')
    if session_id:
        clauses.append(f'session_id == "{_escape_filter_value(session_id)}"')
    if chunk_type:
        clauses.append(f'chunk_type == "{_escape_filter_value(chunk_type)}"')
    if topic:
        clauses.append(f'topic == "{_escape_filter_value(topic)}"')
    return " and ".join(clauses)


def _coerce_float(value: Any) -> float:
    """把返回分数统一转成浮点数。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _get_field(source: Any, name: str, default: Any = None) -> Any:
    """从 dict 或对象中提取字段。"""
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _get_hit_entity(hit: Any) -> Any:
    """提取 Milvus 命中项里的实体内容。"""
    entity = _get_field(hit, "entity")
    return entity if entity is not None else hit


def _flatten_search_hits(search_result: Any) -> List[Any]:
    """兼容不同 Milvus 返回结构。"""
    if not search_result:
        return []
    if isinstance(search_result, list) and search_result and isinstance(search_result[0], list):
        return list(search_result[0])
    if isinstance(search_result, tuple) and search_result and isinstance(search_result[0], list):
        return list(search_result[0])
    return list(search_result)


def _extract_text(message: Any) -> str:
    """提取 LLM 输出中的纯文本。"""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """尽量把模型输出解析成 JSON。"""
    normalized = (text or "").strip()
    if not normalized:
        return {}
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        start = normalized.find("{")
        end = normalized.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(normalized[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _flatten_turn_messages(turns: Sequence[Sequence[Any]]) -> List[Any]:
    """把轮次结构拍平，方便交给 chunk builder。"""
    flattened: List[Any] = []
    for turn in turns:
        flattened.extend(turn)
    return flattened


def _build_topic_change_detector(llm: Any):
    """构建话题切换判断函数。"""
    if llm is None:
        raise ValueError("长期记忆语义切块需要提供 llm")

    system_prompt = (
        "你是对话话题判断器。请判断下面两段对话是否已经发生明显话题切换。"
        "只输出 JSON，格式为 {\"changed\": true/false}，不要输出额外解释。"
    )

    def detect(current_text: str, next_text: str) -> bool:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        "当前缓冲区对话：\n"
                        f"{current_text.strip()}\n\n"
                        "候选新增对话：\n"
                        f"{next_text.strip()}\n\n"
                        "如果已经明显换题，请返回 changed=true；否则返回 changed=false。"
                    )
                ),
            ]
        )
        payload = _safe_json_loads(_extract_text(response))
        if "changed" in payload:
            return bool(payload.get("changed"))

        result_text = _extract_text(response).strip().lower()
        return any(token in result_text for token in ("true", "yes", "changed", "是", "切换", "换题"))

    return detect


def _build_chunk_summarizer(llm: Any):
    """构建摘要函数。"""
    if llm is None:
        raise ValueError("长期记忆语义切块需要提供 llm")

    system_prompt = (
        "你是长期记忆摘要器。请将给定对话总结成适合长期记忆的一句话摘要，并给出主题。"
        "只输出 JSON，格式为 {\"topic\": \"主题\", \"summary\": \"一句摘要\"}，不要输出额外解释。"
        "主题要短，摘要要准确，不要编造。"
    )

    def summarize(transcript: str) -> Dict[str, str]:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        "请基于下面的对话生成长期记忆摘要：\n\n"
                        f"{transcript.strip()}"
                    )
                ),
            ]
        )
        payload = _safe_json_loads(_extract_text(response))
        topic = str(payload.get("topic", "") or "").strip()
        summary = str(payload.get("summary", "") or "").strip()
        if not summary:
            summary = _extract_text(response).strip()
        return {"topic": topic, "summary": summary}

    return summarize


def _build_insert_rows(
    chunks: Sequence[Any],
    vectors: Sequence[Sequence[float]],
    conversation_id: str,
    session_id: str,
    source_name: str,
    chunk_type: str,
    topic: str,
) -> List[Dict[str, Any]]:
    """组装 Milvus 插入行。"""
    created_at = datetime.now(timezone.utc).isoformat()
    rows: List[Dict[str, Any]] = []
    for chunk in chunks:
        chunk_topic = str(getattr(chunk, "topic", "") or topic).strip()
        rows.append(
            {
                "text": chunk.text,
                "vector": list(vectors[chunk.chunk_index]),
                "memory_id": f"{conversation_id}:{chunk.chunk_index}",
                "chunk_index": chunk.chunk_index,
                "source_name": source_name,
                "chunk_text": chunk.text,
                "conversation_id": conversation_id,
                "session_id": session_id,
                "chunk_type": chunk_type,
                "topic": chunk_topic,
                "turn_start": chunk.turn_start,
                "turn_end": chunk.turn_end,
                "created_at": created_at,
            }
        )
    return rows


def store_semantic_long_term_memory(
    messages: Sequence[Any] | str,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    source_name: str = "conversation",
    topic: str = "",
    chunk_type: str = "semantic_memory",
    include_system: bool = False,
    short_window_n: int = 5,
    token_limit: int = 700,
    llm: Optional[Any] = None,
    chunk_size: int = CHAT_CHUNK_SIZE,
    chunk_overlap: int = CHAT_CHUNK_OVERLAP,
) -> List[str]:
    """把长期记忆写入 Milvus 的 `memory_chunks` collection。"""
    try:
        if isinstance(messages, str):
            chunk_texts = split_conversation_chunks(
                messages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = [
                LongTermMemoryChunk(
                    chunk_index=index,
                    topic=topic.strip() or "conversation",
                    summary=text[:120].strip(),
                    transcript=text,
                    text=text,
                    turn_start=0,
                    turn_end=0,
                    message_count=1,
                    trigger_reason="legacy_text",
                )
                for index, text in enumerate(chunk_texts)
            ]
        else:
            turns = split_conversation_turns(messages, include_system=include_system)
            older_turns = turns[:-short_window_n] if short_window_n > 0 else turns
            if not older_turns:
                logger.info("短期窗口内没有可入库的长期记忆。")
                return []

            semantic_messages = _flatten_turn_messages(older_turns)
            chunks = build_semantic_memory_chunks(
                semantic_messages,
                should_split_topic=_build_topic_change_detector(llm),
                summarize_chunk=_build_chunk_summarizer(llm),
                token_limit=token_limit,
                include_system=include_system,
            )

        if not chunks:
            logger.info("没有可写入的长期记忆块。")
            return []

        conversation_key = conversation_id or session_id or uuid.uuid4().hex
        session_key = session_id or conversation_key
        topic_value = topic.strip()

        from app.core.embeddings import get_default_embeddings
        from app.storage.milvus_store import (
            _ensure_memory_collection,
            _validate_vector_dimension,
            get_milvus_client,
            insert_rows_with_retry,
        )

        client = get_milvus_client()
        _ensure_memory_collection(client)
        _validate_vector_dimension(client, settings.MILVUS_MEMORY_COLLECTION_NAME)

        embeddings = get_default_embeddings()
        vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
        rows = _build_insert_rows(
            chunks=chunks,
            vectors=vectors,
            conversation_id=conversation_key,
            session_id=session_key,
            source_name=source_name,
            chunk_type=chunk_type,
            topic=topic_value,
        )

        inserted_ids = insert_rows_with_retry(
            client=client,
            collection_name=settings.MILVUS_MEMORY_COLLECTION_NAME,
            rows=rows,
        )
        logger.info(
            "✓ 长期记忆写入成功 (conversation_id=%s, chunks=%s)",
            conversation_key,
            len(inserted_ids),
        )
        return inserted_ids
    except Exception as error:
        logger.error(f"长期记忆写入失败: {str(error)}")
        raise VectorStoreException(f"长期记忆写入失败: {str(error)}")


def store_long_term_memory(*args, **kwargs) -> List[str]:
    """兼容旧路径的长期记忆写入入口。"""
    return store_semantic_long_term_memory(*args, **kwargs)


def search_long_term_memory(
    query: str,
    top_k: int = 5,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
    topic: Optional[str] = None,
) -> List[RetrievalResult]:
    """按向量相似度检索长期记忆。"""
    try:
        if not query or not query.strip():
            raise RetrievalException("查询文本不能为空")

        from app.core.embeddings import get_default_embeddings
        from app.storage.milvus_store import (
            _ensure_memory_collection,
            _validate_vector_dimension,
            get_milvus_client,
            is_collection_loaded,
        )

        client = get_milvus_client()
        if not client.has_collection(settings.MILVUS_MEMORY_COLLECTION_NAME):
            return []

        _ensure_memory_collection(client)
        _validate_vector_dimension(client, settings.MILVUS_MEMORY_COLLECTION_NAME)
        if not is_collection_loaded(client, settings.MILVUS_MEMORY_COLLECTION_NAME):
            logger.warning("长期记忆 collection 尚未加载完成，跳过本次检索")
            return []

        embeddings = get_default_embeddings()
        query_vector = embeddings.embed_query(query.strip())

        filter_expr = _build_memory_filter(
            conversation_id=conversation_id,
            session_id=session_id,
            chunk_type=chunk_type,
            topic=topic,
        )

        search_kwargs: Dict[str, Any] = {
            "collection_name": settings.MILVUS_MEMORY_COLLECTION_NAME,
            "data": [query_vector],
            "limit": top_k,
            "output_fields": [
                "text",
                "memory_id",
                "chunk_index",
                "source_name",
                "chunk_text",
                "conversation_id",
                "session_id",
                "chunk_type",
                "topic",
                "turn_start",
                "turn_end",
                "created_at",
            ],
        }
        if filter_expr:
            search_kwargs["filter"] = filter_expr

        search_result = client.search(**search_kwargs)
        hits = _flatten_search_hits(search_result)
        results: List[RetrievalResult] = []
        for hit in hits:
            entity = _get_hit_entity(hit)
            content = str(
                _get_field(entity, "chunk_text")
                or _get_field(entity, "text")
                or _get_field(hit, "chunk_text")
                or _get_field(hit, "text")
                or ""
            )
            metadata = {
                "memory_id": _get_field(entity, "memory_id", _get_field(hit, "memory_id")),
                "chunk_index": _get_field(entity, "chunk_index", _get_field(hit, "chunk_index")),
                "source_name": _get_field(entity, "source_name", _get_field(hit, "source_name")),
                "chunk_type": _get_field(entity, "chunk_type", _get_field(hit, "chunk_type")),
                "topic": _get_field(entity, "topic", _get_field(hit, "topic")),
                "conversation_id": _get_field(entity, "conversation_id", _get_field(hit, "conversation_id")),
                "session_id": _get_field(entity, "session_id", _get_field(hit, "session_id")),
                "turn_start": _get_field(entity, "turn_start", _get_field(hit, "turn_start")),
                "turn_end": _get_field(entity, "turn_end", _get_field(hit, "turn_end")),
                "created_at": _get_field(entity, "created_at", _get_field(hit, "created_at")),
            }
            score = _coerce_float(_get_field(hit, "distance", _get_field(hit, "score", 0.0)))
            results.append(
                RetrievalResult(
                    content=content,
                    metadata=metadata,
                    score=score,
                    source=str(metadata.get("source_name") or "conversation"),
                )
            )

        return results
    except RetrievalException:
        raise
    except Exception as error:
        logger.error(f"长期记忆检索失败: {str(error)}")
        raise RetrievalException(f"长期记忆检索失败: {str(error)}")


def delete_long_term_memory(conversation_id: str) -> bool:
    """按 conversation_id 删除长期记忆。"""
    try:
        if not conversation_id or not conversation_id.strip():
            raise VectorStoreException("conversation_id 不能为空")

        from app.storage.milvus_store import _ensure_memory_collection, get_milvus_client

        client = get_milvus_client()
        if not client.has_collection(settings.MILVUS_MEMORY_COLLECTION_NAME):
            return False

        _ensure_memory_collection(client)
        filter_expr = f'conversation_id == "{_escape_filter_value(conversation_id.strip())}"'
        client.delete(
            collection_name=settings.MILVUS_MEMORY_COLLECTION_NAME,
            filter=filter_expr,
            timeout=30,
        )
        logger.info("✓ 长期记忆删除成功 (conversation_id=%s)", conversation_id)
        return True
    except Exception as error:
        logger.error(f"长期记忆删除失败: {str(error)}")
        raise VectorStoreException(f"长期记忆删除失败: {str(error)}")


store_conversation_memory = store_long_term_memory
search_conversation_memory = search_long_term_memory
delete_conversation_memory = delete_long_term_memory


__all__ = [
    "store_semantic_long_term_memory",
    "store_long_term_memory",
    "search_long_term_memory",
    "delete_long_term_memory",
    "store_conversation_memory",
    "search_conversation_memory",
    "delete_conversation_memory",
]
