"""对话长期记忆服务。"""
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import uuid
import logging

from app.utils.chunking import ConversationChunk, build_conversation_chunks, split_conversation_chunks
from app.utils.exceptions import RetrievalException, VectorStoreException
from app.rag.retrieval.base import RetrievalResult
from app.config import settings
from app.core.constants import CHAT_CHUNK_OVERLAP, CHAT_CHUNK_SIZE

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


def _build_insert_rows(
    chunks: Sequence[ConversationChunk],
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
                "topic": topic,
                "turn_start": chunk.turn_start,
                "turn_end": chunk.turn_end,
                "created_at": created_at,
            }
        )
    return rows


def store_conversation_memory(
    messages: Sequence[Any] | str,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    source_name: str = "conversation",
    topic: str = "",
    chunk_type: str = "dialogue",
    include_system: bool = False,
    chunk_size: int = CHAT_CHUNK_SIZE,
    chunk_overlap: int = CHAT_CHUNK_OVERLAP,
) -> List[str]:
    """把对话写入长期记忆向量库。"""
    try:
        if isinstance(messages, str):
            chunk_texts = split_conversation_chunks(messages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = [
                ConversationChunk(
                    chunk_index=index,
                    text=text,
                    turn_start=0,
                    turn_end=0,
                    message_count=1,
                )
                for index, text in enumerate(chunk_texts)
            ]
        else:
            chunks = build_conversation_chunks(
                messages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                include_system=include_system,
            )

        if not chunks:
            logger.info("没有可写入的长期记忆块。")
            return []

        conversation_key = conversation_id or session_id or uuid.uuid4().hex
        session_key = session_id or conversation_key
        topic_value = topic.strip()

        from app.storage.milvus_store import (
            _ensure_memory_collection,
            _validate_vector_dimension,
            insert_rows_with_retry,
            get_milvus_client,
        )

        client = get_milvus_client()
        _ensure_memory_collection(client)
        _validate_vector_dimension(client, settings.MILVUS_MEMORY_COLLECTION_NAME)

        from app.core.embeddings import get_default_embeddings

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
    except Exception as e:
        logger.error(f"长期记忆写入失败: {str(e)}")
        raise VectorStoreException(f"长期记忆写入失败: {str(e)}")


def search_conversation_memory(
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

        from app.storage.milvus_store import (
            _ensure_memory_collection,
            _validate_vector_dimension,
            get_milvus_client,
        )

        client = get_milvus_client()
        if not client.has_collection(settings.MILVUS_MEMORY_COLLECTION_NAME):
            return []

        _ensure_memory_collection(client)
        _validate_vector_dimension(client, settings.MILVUS_MEMORY_COLLECTION_NAME)
        from app.storage.milvus_store import is_collection_loaded

        if not is_collection_loaded(client, settings.MILVUS_MEMORY_COLLECTION_NAME):
            logger.warning("长期记忆 collection 尚未加载完成，跳过本次检索")
            return []

        from app.core.embeddings import get_default_embeddings

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
    except Exception as e:
        logger.error(f"长期记忆检索失败: {str(e)}")
        raise RetrievalException(f"长期记忆检索失败: {str(e)}")


def delete_conversation_memory(
    conversation_id: str,
) -> bool:
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
    except Exception as e:
        logger.error(f"长期记忆删除失败: {str(e)}")
        raise VectorStoreException(f"长期记忆删除失败: {str(e)}")
