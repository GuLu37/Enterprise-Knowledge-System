"""对话长期记忆服务。"""
import json
import logging
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import or_

from app.config import settings
from app.core.constants import CHAT_CHUNK_OVERLAP, CHAT_CHUNK_SIZE
from app.rag.retrieval.base import RetrievalResult
from app.storage.sqlite_metadata import MemoryRecord, SessionLocal, init_metadata_db
from app.utils.chunking import (
    ConversationChunk,
    LongTermMemoryChunk,
    build_conversation_chunks,
    build_semantic_memory_chunks,
    split_conversation_chunks,
    split_conversation_turns,
)
from app.utils.exceptions import RetrievalException, VectorStoreException

logger = logging.getLogger(__name__)

MEMORY_TYPE_PROFILE = "profile"
MEMORY_TYPE_EVENT = "event"
MEMORY_STATUS_ACTIVE = "active"
MEMORY_STATUS_SUPERSEDED = "superseded"
MEMORY_STATUS_EXPIRED = "expired"
MEMORY_STATUS_DELETED = "deleted"
PROFILE_MEMORY_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{1,127}$")

PROFILE_MEMORY_SYSTEM_PROMPT = (
    "你是用户个性化记忆抽取器。"
    "只提取用户明确表达、长期稳定、未来对话确实有帮助的个人偏好或事实。"
    "不要提取一次性任务、临时安排、普通问题、企业文档内容、模型回答内容、密码、令牌、身份证号或其他敏感凭据。"
    "只有用户明确要求记住、忘记或修改某项信息时，才返回 action=upsert 或 action=delete。"
    "没有可记忆内容时输出 {\"memories\":[]}。"
    "只输出严格 JSON，不要解释。格式："
    "{\"memories\":[{\"memory_key\":\"profile.xxx\",\"value\":\"...\","
    "\"confidence\":0到1之间的小数,\"importance\":0到100之间的数字,"
    "\"action\":\"upsert\"或\"delete\"}]}"
)


def _escape_filter_value(value: str) -> str:
    """转义 Milvus 过滤表达式中的字符串值。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_memory_filter(
    user_id: str,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
    topic: Optional[str] = None,
) -> str:
    """构建记忆检索过滤条件。"""
    clauses: List[str] = [f'user_id == "{_escape_filter_value(user_id)}"']
    if conversation_id:
        clauses.append(f'conversation_id == "{_escape_filter_value(conversation_id)}"')
    if session_id:
        clauses.append(f'session_id == "{_escape_filter_value(session_id)}"')
    if chunk_type:
        clauses.append(f'chunk_type == "{_escape_filter_value(chunk_type)}"')
    if topic:
        clauses.append(f'topic == "{_escape_filter_value(topic)}"')
    return " and ".join(clauses)


def _require_user_id(user_id: Optional[str]) -> str:
    """确保长期记忆操作始终绑定到已认证用户。"""
    normalized = (user_id or "").strip()
    if not normalized:
        raise VectorStoreException("user_id 不能为空，长期记忆必须绑定到用户")
    return normalized


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


def _utcnow_naive() -> datetime:
    """返回与 SQLAlchemy DateTime 字段兼容的 UTC 时间。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _clamp_score(value: Any, default: float = 50.0) -> float:
    """把重要性或置信度限制在安全范围。"""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(100.0, numeric))


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _normalize_memory_content(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _calculate_event_importance(content: str) -> tuple[float, float]:
    """为事件记忆生成可解释的初始重要性和置信度。"""
    normalized = _normalize_memory_content(content)
    importance = 45.0
    confidence = 0.65
    strong_markers = ("必须", "明确", "决定", "结论", "长期", "偏好", "目标", "完成")
    temporary_markers = ("今天", "明天", "刚才", "临时", "这次", "当前")
    importance += min(25.0, sum(normalized.count(marker) * 5 for marker in strong_markers))
    importance -= min(20.0, sum(normalized.count(marker) * 4 for marker in temporary_markers))
    if len(normalized) < 20:
        confidence -= 0.1
    return _clamp_score(importance), _clamp_confidence(confidence)


def _extract_profile_memory_candidates(
    messages: Sequence[Any],
    llm: Optional[Any],
) -> List[Dict[str, Any]]:
    """从最近对话中抽取稳定的用户个性化记忆候选。"""
    if llm is None:
        return []

    transcript_lines = []
    for message in list(messages)[-20:]:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        content = _normalize_memory_content(content)
        if role in {"user", "assistant"} and content:
            transcript_lines.append(f"{role}: {content}")
    if not transcript_lines:
        return []

    try:
        response = llm.invoke(
            [
                SystemMessage(content=PROFILE_MEMORY_SYSTEM_PROMPT),
                HumanMessage(content="\n".join(transcript_lines)),
            ]
        )
        payload = _safe_json_loads(_extract_text(response))
    except Exception as exc:
        logger.warning("个性化记忆抽取失败，已跳过: %s", exc)
        return []

    candidates = payload.get("memories")
    if not isinstance(candidates, list):
        return []

    normalized_candidates: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        key = str(candidate.get("memory_key") or "").strip().lower()
        value = _normalize_memory_content(candidate.get("value"))
        action = str(candidate.get("action") or "upsert").strip().lower()
        if not PROFILE_MEMORY_KEY_PATTERN.match(key):
            continue
        if action == "delete":
            normalized_candidates.append({"memory_key": key, "action": "delete"})
            continue
        if not value:
            continue
        normalized_candidates.append(
            {
                "memory_key": key,
                "value": value[:4000],
                "confidence": _clamp_confidence(candidate.get("confidence"), 0.7),
                "importance": _clamp_score(candidate.get("importance"), 75.0),
                "action": "upsert",
            }
        )
    return normalized_candidates


def get_active_profile_memories(user_id: Optional[str], limit: int = 20) -> List[MemoryRecord]:
    """读取当前用户跨对话共享的有效个性化记忆。"""
    user_key = _require_user_id(user_id)
    init_metadata_db()
    db = SessionLocal()
    try:
        now = _utcnow_naive()
        records = (
            db.query(MemoryRecord)
            .filter(
                MemoryRecord.user_id == user_key,
                MemoryRecord.memory_type == MEMORY_TYPE_PROFILE,
                MemoryRecord.status == MEMORY_STATUS_ACTIVE,
                or_(MemoryRecord.expires_at.is_(None), MemoryRecord.expires_at > now),
            )
            .order_by(MemoryRecord.importance_score.desc(), MemoryRecord.updated_at.desc())
            .limit(max(1, min(limit, 100)))
            .all()
        )
        for record in records:
            record.access_count += 1
            record.last_accessed_at = now
        db.commit()
        return records
    finally:
        db.close()


def upsert_profile_memory_candidates(
    user_id: Optional[str],
    candidates: Sequence[Dict[str, Any]],
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> int:
    """按 memory_key 更新个性化记忆，保留被替代版本。"""
    user_key = _require_user_id(user_id)
    if not candidates:
        return 0
    init_metadata_db()
    db = SessionLocal()
    changed = 0
    now = _utcnow_naive()
    try:
        for candidate in candidates:
            key = str(candidate.get("memory_key") or "").strip().lower()
            action = str(candidate.get("action") or "upsert").strip().lower()
            if not PROFILE_MEMORY_KEY_PATTERN.match(key):
                continue
            current = (
                db.query(MemoryRecord)
                .filter(
                    MemoryRecord.user_id == user_key,
                    MemoryRecord.memory_type == MEMORY_TYPE_PROFILE,
                    MemoryRecord.memory_key == key,
                    MemoryRecord.status == MEMORY_STATUS_ACTIVE,
                )
                .order_by(MemoryRecord.version.desc(), MemoryRecord.updated_at.desc())
                .first()
            )
            if action == "delete":
                if current:
                    current.status = MEMORY_STATUS_DELETED
                    current.updated_at = now
                    changed += 1
                continue

            value = _normalize_memory_content(candidate.get("value"))
            if not value:
                continue
            confidence = _clamp_confidence(candidate.get("confidence"), 0.7)
            importance = _clamp_score(candidate.get("importance"), 75.0)
            if current and _normalize_memory_content(current.content) == value:
                current.base_importance_score = importance
                current.importance_score = importance
                current.confidence_score = confidence
                current.last_accessed_at = now
                current.updated_at = now
                changed += 1
                continue

            if current:
                current.status = MEMORY_STATUS_SUPERSEDED
                current.updated_at = now
                version = current.version + 1
                supersedes_memory_id = current.memory_id
            else:
                version = 1
                supersedes_memory_id = None

            record = MemoryRecord(
                memory_id=f"profile:{user_key}:{key}:{uuid.uuid4().hex}",
                user_id=user_key,
                memory_type=MEMORY_TYPE_PROFILE,
                memory_key=key,
                content=value,
                conversation_id=conversation_id,
                session_id=session_id,
                base_importance_score=importance,
                importance_score=importance,
                confidence_score=confidence,
                access_count=0,
                last_accessed_at=None,
                expires_at=None,
                status=MEMORY_STATUS_ACTIVE,
                version=version,
                supersedes_memory_id=supersedes_memory_id,
                created_at=now,
                updated_at=now,
            )
            db.add(record)
            changed += 1
        db.commit()
        return changed
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def register_event_memory_records(
    user_id: Optional[str],
    rows: Sequence[Dict[str, Any]],
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> int:
    """为已写入 Milvus 的事件记忆建立 SQL 生命周期记录。"""
    user_key = _require_user_id(user_id)
    if not rows:
        return 0
    init_metadata_db()
    db = SessionLocal()
    now = _utcnow_naive()
    expiry = now + timedelta(days=settings.MEMORY_EVENT_EXPIRY_DAYS)
    changed = 0
    retired_memory_ids: List[str] = []
    try:
        for row in rows:
            memory_id = str(row.get("memory_id") or "").strip()
            content = _normalize_memory_content(row.get("chunk_text") or row.get("text"))
            if not memory_id or not content:
                continue
            base_score, confidence = _calculate_event_importance(content)
            row_conversation_id = str(
                conversation_id or row.get("conversation_id") or ""
            ).strip() or None
            row_session_id = str(session_id or row.get("session_id") or "").strip() or None
            chunk_index = row.get("chunk_index")
            logical_key = (
                f"event:{user_key}:{row_conversation_id or 'unknown'}:"
                f"{chunk_index if chunk_index is not None else memory_id}"
            )
            current = (
                db.query(MemoryRecord)
                .filter(
                    MemoryRecord.user_id == user_key,
                    MemoryRecord.memory_type == MEMORY_TYPE_EVENT,
                    MemoryRecord.memory_key == logical_key,
                    MemoryRecord.status == MEMORY_STATUS_ACTIVE,
                )
                .order_by(MemoryRecord.version.desc(), MemoryRecord.updated_at.desc())
                .first()
            )
            version = 1
            supersedes_memory_id = None
            if current:
                current.status = MEMORY_STATUS_SUPERSEDED
                current.updated_at = now
                version = current.version + 1
                supersedes_memory_id = current.memory_id
                retired_memory_ids.append(current.memory_id)

            db.add(
                MemoryRecord(
                    memory_id=memory_id,
                    user_id=user_key,
                    memory_type=MEMORY_TYPE_EVENT,
                    memory_key=logical_key,
                    content=content,
                    conversation_id=row_conversation_id,
                    session_id=row_session_id,
                    base_importance_score=base_score,
                    importance_score=base_score,
                    confidence_score=confidence,
                    access_count=0,
                    last_accessed_at=None,
                    expires_at=expiry,
                    status=MEMORY_STATUS_ACTIVE,
                    version=version,
                    supersedes_memory_id=supersedes_memory_id,
                    created_at=now,
                    updated_at=now,
                )
            )
            changed += 1
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    if retired_memory_ids:
        _delete_event_vectors(retired_memory_ids)
    return changed


def _get_active_event_records(
    user_id: str,
    memory_ids: Sequence[str],
) -> Dict[str, MemoryRecord]:
    """只返回 SQL 中仍然有效的事件记忆。"""
    normalized_ids = [str(item).strip() for item in memory_ids if str(item).strip()]
    if not normalized_ids:
        return {}
    db = SessionLocal()
    try:
        now = _utcnow_naive()
        records = (
            db.query(MemoryRecord)
            .filter(
                MemoryRecord.user_id == user_id,
                MemoryRecord.memory_type == MEMORY_TYPE_EVENT,
                MemoryRecord.memory_id.in_(normalized_ids),
                MemoryRecord.status == MEMORY_STATUS_ACTIVE,
                or_(MemoryRecord.expires_at.is_(None), MemoryRecord.expires_at > now),
            )
            .all()
        )
        return {record.memory_id: record for record in records}
    finally:
        db.close()


def _mark_memory_accessed(user_id: str, memory_ids: Sequence[str]) -> None:
    """更新被召回记忆的访问统计。"""
    normalized_ids = [str(item).strip() for item in memory_ids if str(item).strip()]
    if not normalized_ids:
        return
    db = SessionLocal()
    try:
        now = _utcnow_naive()
        records = (
            db.query(MemoryRecord)
            .filter(
                MemoryRecord.user_id == user_id,
                MemoryRecord.memory_id.in_(normalized_ids),
                MemoryRecord.status == MEMORY_STATUS_ACTIVE,
            )
            .all()
        )
        for record in records:
            record.access_count += 1
            record.last_accessed_at = now
            record.updated_at = now
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("更新长期记忆访问统计失败", exc_info=True)
    finally:
        db.close()


def _delete_event_vectors(memory_ids: Sequence[str]) -> None:
    """删除已过期事件记忆对应的 Milvus 向量。"""
    normalized_ids = [str(item).strip() for item in memory_ids if str(item).strip()]
    if not normalized_ids:
        return
    try:
        from app.storage.milvus_store import get_milvus_client

        client = get_milvus_client()
        if not client.has_collection(settings.MILVUS_MEMORY_COLLECTION_NAME):
            return
        escaped = ", ".join(f'"{_escape_filter_value(item)}"' for item in normalized_ids)
        client.delete(
            collection_name=settings.MILVUS_MEMORY_COLLECTION_NAME,
            filter=f"memory_id in [{escaped}]",
            timeout=30,
        )
    except Exception:
        logger.warning("清理过期长期记忆向量失败", exc_info=True)


def cleanup_memory_records() -> Dict[str, int]:
    """刷新记忆重要性，标记过期记录并清理旧元数据。"""
    init_metadata_db()
    db = SessionLocal()
    now = _utcnow_naive()
    expired_event_ids: List[str] = []
    result = {"rescored": 0, "expired": 0, "purged": 0}
    try:
        records = db.query(MemoryRecord).filter(MemoryRecord.status == MEMORY_STATUS_ACTIVE).all()
        for record in records:
            reference_time = record.last_accessed_at or record.created_at or now
            age_days = max(0.0, (now - reference_time).total_seconds() / 86400)
            recency_days = 90.0 if record.memory_type == MEMORY_TYPE_EVENT else 365.0
            decayed_score = (
                record.base_importance_score
                * record.confidence_score
                * math.exp(-age_days / recency_days)
            )
            access_bonus = min(15.0, math.log1p(max(0, record.access_count)) * 2.5)
            record.importance_score = round(_clamp_score(decayed_score + access_bonus), 2)
            record.updated_at = now
            result["rescored"] += 1

            should_expire = (
                record.expires_at is not None and record.expires_at <= now
            ) or (
                record.importance_score < settings.MEMORY_SCORE_THRESHOLD
                and (
                    record.memory_type == MEMORY_TYPE_EVENT
                    or age_days >= settings.MEMORY_PROFILE_STALE_DAYS
                )
            )
            if should_expire:
                record.status = MEMORY_STATUS_EXPIRED
                record.updated_at = now
                result["expired"] += 1
                if record.memory_type == MEMORY_TYPE_EVENT:
                    expired_event_ids.append(record.memory_id)

        purge_before = now - timedelta(days=settings.MEMORY_PURGE_RETENTION_DAYS)
        result["purged"] = (
            db.query(MemoryRecord)
            .filter(
                MemoryRecord.status.in_(
                    [MEMORY_STATUS_EXPIRED, MEMORY_STATUS_SUPERSEDED, MEMORY_STATUS_DELETED]
                ),
                MemoryRecord.updated_at < purge_before,
            )
            .delete(synchronize_session=False)
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("长期记忆生命周期清理失败", exc_info=True)
    finally:
        db.close()

    _delete_event_vectors(expired_event_ids)
    return result


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
    user_id: str,
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
        # Milvus 使用自增主键，memory_id 也必须按版本唯一，否则重复写入同一对话
        # 会留下无法由业务 ID 区分的旧向量。
        versioned_memory_id = (
            f"{user_id}:{conversation_id}:{chunk.chunk_index}:{uuid.uuid4().hex}"
        )
        rows.append(
            {
                "text": chunk.text,
                "vector": list(vectors[chunk.chunk_index]),
                "memory_id": versioned_memory_id,
                "chunk_index": chunk.chunk_index,
                "source_name": source_name,
                "chunk_text": chunk.text,
                "user_id": user_id,
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
    user_id: Optional[str] = None,
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
        user_key = _require_user_id(user_id)
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
            user_id=user_key,
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
        register_event_memory_records(
            user_id=user_key,
            rows=rows,
            conversation_id=conversation_key,
            session_id=session_key,
        )
        logger.info(
            "✓ 长期记忆写入成功 (user_id=%s, conversation_id=%s, chunks=%s)",
            user_key,
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


def store_conversation_memory(
    messages: Sequence[Any] | str,
    user_id: Optional[str] = None,
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
        user_key = _require_user_id(user_id)
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
            get_milvus_client,
            insert_rows_with_retry,
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
            user_id=user_key,
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
        register_event_memory_records(
            user_id=user_key,
            rows=rows,
            conversation_id=conversation_key,
            session_id=session_key,
        )
        logger.info(
            "✓ 长期记忆写入成功 (user_id=%s, conversation_id=%s, chunks=%s)",
            user_key,
            conversation_key,
            len(inserted_ids),
        )
        return inserted_ids
    except Exception as error:
        logger.error(f"长期记忆写入失败: {str(error)}")
        raise VectorStoreException(f"长期记忆写入失败: {str(error)}")


def search_long_term_memory(
    query: str,
    top_k: int = 5,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
    topic: Optional[str] = None,
) -> List[RetrievalResult]:
    """按向量相似度检索长期记忆。"""
    try:
        if not query or not query.strip():
            raise RetrievalException("查询文本不能为空")
        user_key = _require_user_id(user_id)

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
            user_id=user_key,
            conversation_id=conversation_id,
            session_id=session_id,
            chunk_type=chunk_type,
            topic=topic,
        )

        search_kwargs: Dict[str, Any] = {
            "collection_name": settings.MILVUS_MEMORY_COLLECTION_NAME,
            "data": [query_vector],
            "limit": max(top_k * 3, top_k),
            "output_fields": [
                "text",
                "memory_id",
                "chunk_index",
                "source_name",
                "chunk_text",
                "user_id",
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
        hit_memory_ids = [
            str(
                _get_field(_get_hit_entity(hit), "memory_id", _get_field(hit, "memory_id"))
                or ""
            ).strip()
            for hit in hits
        ]
        active_records = _get_active_event_records(user_key, hit_memory_ids)
        results: List[RetrievalResult] = []
        for hit in hits:
            entity = _get_hit_entity(hit)
            memory_id = str(
                _get_field(entity, "memory_id", _get_field(hit, "memory_id"))
                or ""
            ).strip()
            record = active_records.get(memory_id)
            if record is None:
                continue
            content = str(
                _get_field(entity, "chunk_text")
                or _get_field(entity, "text")
                or _get_field(hit, "chunk_text")
                or _get_field(hit, "text")
                or ""
            )
            metadata = {
                "memory_id": memory_id,
                "chunk_index": _get_field(entity, "chunk_index", _get_field(hit, "chunk_index")),
                "source_name": _get_field(entity, "source_name", _get_field(hit, "source_name")),
                "user_id": _get_field(entity, "user_id", _get_field(hit, "user_id")),
                "chunk_type": _get_field(entity, "chunk_type", _get_field(hit, "chunk_type")),
                "topic": _get_field(entity, "topic", _get_field(hit, "topic")),
                "conversation_id": _get_field(entity, "conversation_id", _get_field(hit, "conversation_id")),
                "session_id": _get_field(entity, "session_id", _get_field(hit, "session_id")),
                "turn_start": _get_field(entity, "turn_start", _get_field(hit, "turn_start")),
                "turn_end": _get_field(entity, "turn_end", _get_field(hit, "turn_end")),
                "created_at": _get_field(entity, "created_at", _get_field(hit, "created_at")),
                "importance_score": record.importance_score,
                "confidence_score": record.confidence_score,
                "status": record.status,
                "access_count": record.access_count,
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

        final_results = results[:top_k]
        _mark_memory_accessed(
            user_key,
            [str(item.metadata.get("memory_id") or "") for item in final_results],
        )
        return final_results
    except RetrievalException:
        raise
    except Exception as error:
        logger.error(f"长期记忆检索失败: {str(error)}")
        raise RetrievalException(f"长期记忆检索失败: {str(error)}")


def search_conversation_memory(
    query: str,
    top_k: int = 5,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
    topic: Optional[str] = None,
) -> List[RetrievalResult]:
    """按向量相似度检索长期记忆。"""
    return search_long_term_memory(
        query=query,
        top_k=top_k,
        user_id=user_id,
        conversation_id=conversation_id,
        session_id=session_id,
        chunk_type=chunk_type,
        topic=topic,
    )


def delete_long_term_memory(conversation_id: str, user_id: Optional[str] = None) -> bool:
    """按用户和 conversation_id 删除长期记忆。"""
    try:
        if not conversation_id or not conversation_id.strip():
            raise VectorStoreException("conversation_id 不能为空")
        user_key = _require_user_id(user_id)

        from app.storage.milvus_store import _ensure_memory_collection, get_milvus_client

        client = get_milvus_client()
        if not client.has_collection(settings.MILVUS_MEMORY_COLLECTION_NAME):
            return False

        _ensure_memory_collection(client)
        filter_expr = (
            f'user_id == "{_escape_filter_value(user_key)}" and '
            f'conversation_id == "{_escape_filter_value(conversation_id.strip())}"'
        )
        client.delete(
            collection_name=settings.MILVUS_MEMORY_COLLECTION_NAME,
            filter=filter_expr,
            timeout=30,
        )
        init_metadata_db()
        db = SessionLocal()
        try:
            now = _utcnow_naive()
            (
                db.query(MemoryRecord)
                .filter(
                    MemoryRecord.user_id == user_key,
                    MemoryRecord.memory_type == MEMORY_TYPE_EVENT,
                    MemoryRecord.conversation_id == conversation_id.strip(),
                    MemoryRecord.status == MEMORY_STATUS_ACTIVE,
                )
                .update(
                    {
                        MemoryRecord.status: MEMORY_STATUS_DELETED,
                        MemoryRecord.updated_at: now,
                    },
                    synchronize_session=False,
                )
            )
            db.commit()
        finally:
            db.close()
        logger.info("✓ 长期记忆删除成功 (user_id=%s, conversation_id=%s)", user_key, conversation_id)
        return True
    except Exception as error:
        logger.error(f"长期记忆删除失败: {str(error)}")
        raise VectorStoreException(f"长期记忆删除失败: {str(error)}")


def delete_conversation_memory(conversation_id: str, user_id: Optional[str] = None) -> bool:
    """按用户和 conversation_id 删除长期记忆。"""
    return delete_long_term_memory(conversation_id, user_id=user_id)
__all__ = [
    "store_semantic_long_term_memory",
    "store_long_term_memory",
    "search_long_term_memory",
    "delete_long_term_memory",
    "store_conversation_memory",
    "search_conversation_memory",
    "delete_conversation_memory",
    "get_active_profile_memories",
    "upsert_profile_memory_candidates",
    "register_event_memory_records",
    "cleanup_memory_records",
]
