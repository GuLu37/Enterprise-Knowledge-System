"""对话长期记忆服务。"""
import json
import logging
import math
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import or_

from app.config import settings
from app.core.constants import CHAT_CHUNK_OVERLAP, CHAT_CHUNK_SIZE
from app.rag.retrieval.base import RetrievalResult
from app.rag.retrieval.reranker import normalize_text
from app.storage.sqlite_metadata import (
    MemoryConsolidationRecord,
    MemoryRecord,
    SessionLocal,
    init_metadata_db,
)
from app.utils.chunking import (
    ConversationChunk,
    LongTermMemoryChunk,
    build_conversation_chunks,
    build_semantic_memory_chunks,
    format_long_term_memory_chunk,
    split_conversation_chunks,
    split_conversation_turns,
)
from app.utils.exceptions import RetrievalException, VectorStoreException

logger = logging.getLogger(__name__)

_MEMORY_ACCESS_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="memory-access",
)

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
    chunk_types: Optional[Sequence[str]] = None,
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
    if chunk_types:
        escaped_types = ", ".join(
            f'"{_escape_filter_value(item)}"' for item in chunk_types if str(item or "").strip()
        )
        if escaped_types:
            clauses.append(f"chunk_type in [{escaped_types}]")
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


def _normalize_event_content(value: Any) -> str:
    """保留对话角色换行，便于后续摘要合并重新解析用户/助手消息。"""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text)


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
            content = _normalize_event_content(row.get("chunk_text") or row.get("text"))
            if not memory_id or not content:
                continue
            base_score, confidence = _calculate_event_importance(content)
            row_conversation_id = str(
                conversation_id or row.get("conversation_id") or ""
            ).strip() or None
            row_session_id = str(session_id or row.get("session_id") or "").strip() or None
            chunk_index = row.get("chunk_index")
            row_chunk_type = str(row.get("chunk_type") or "").strip()
            if row_chunk_type == "dialogue":
                logical_key = (
                    f"event:{user_key}:{row_conversation_id or 'unknown'}:"
                    f"dialogue:{memory_id}"
                )
            else:
                logical_key = (
                    f"event:{user_key}:{row_conversation_id or 'unknown'}:"
                    f"{row_chunk_type or 'event'}:"
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
                turn_start=(
                    int(row.get("turn_start"))
                    if row.get("turn_start") is not None
                    else None
                ),
                turn_end=(
                    int(row.get("turn_end"))
                    if row.get("turn_end") is not None
                    else None
                ),
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


def _mark_memory_accessed_async(user_id: str, memory_ids: Sequence[str]) -> None:
    """后台更新访问统计，不阻塞当前记忆检索结果返回。"""
    normalized_ids = [str(item).strip() for item in memory_ids if str(item).strip()]
    if not normalized_ids:
        return

    def update() -> None:
        try:
            _mark_memory_accessed(user_id, normalized_ids)
        except Exception:
            logger.warning("异步更新长期记忆访问统计失败", exc_info=True)

    try:
        _MEMORY_ACCESS_EXECUTOR.submit(update)
    except Exception:
        logger.warning("提交长期记忆访问统计任务失败", exc_info=True)


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
    turn_offset: int = 0,
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
                "chunk_index": int(chunk.chunk_index) + int(turn_offset),
                "source_name": source_name,
                "chunk_text": chunk.text,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "session_id": session_id,
                "chunk_type": chunk_type,
                "topic": chunk_topic,
                "turn_start": int(chunk.turn_start) + int(turn_offset),
                "turn_end": int(chunk.turn_end) + int(turn_offset),
                "created_at": created_at,
            }
        )
    return rows


def _prepare_memory_collection(
    *,
    require_loaded: bool = False,
    create: bool = False,
) -> Optional[Any]:
    """统一准备 memory collection，返回不可用时的空客户端。"""
    from app.storage.milvus_store import (
        _ensure_memory_collection,
        _validate_vector_dimension,
        get_milvus_client,
        is_collection_loaded,
    )

    client = get_milvus_client()
    collection_name = settings.MILVUS_MEMORY_COLLECTION_NAME
    if not create and not client.has_collection(collection_name):
        return None
    _ensure_memory_collection(client)
    _validate_vector_dimension(client, collection_name)
    if require_loaded and not is_collection_loaded(client, collection_name):
        logger.warning("长期记忆 collection 尚未加载完成，跳过本次操作")
        return None
    return client


def _write_memory_chunks(
    *,
    chunks: Sequence[Any],
    user_id: str,
    conversation_id: str,
    session_id: str,
    source_name: str,
    chunk_type: str,
    topic: str,
    turn_offset: int = 0,
    normalize_dialogue_turns: bool = False,
) -> List[str]:
    """统一完成记忆块 embedding、Milvus 写入和 SQL 生命周期登记。"""
    if not chunks:
        return []

    from app.core.embeddings import get_default_embeddings
    from app.storage.milvus_store import insert_rows_with_retry

    client = _prepare_memory_collection(create=True)
    if client is None:
        raise VectorStoreException("长期记忆 collection 不可用")
    vectors = get_default_embeddings().embed_documents([chunk.text for chunk in chunks])
    rows = _build_insert_rows(
        chunks=chunks,
        vectors=vectors,
        user_id=user_id,
        conversation_id=conversation_id,
        session_id=session_id,
        source_name=source_name,
        chunk_type=chunk_type,
        topic=topic,
        turn_offset=turn_offset,
    )
    if normalize_dialogue_turns and chunk_type == "dialogue":
        for row, chunk in zip(rows, chunks):
            row["turn_start"] = turn_offset + int(getattr(chunk, "turn_start", 0))
            row["turn_end"] = turn_offset + int(getattr(chunk, "turn_end", 0))

    inserted_ids = insert_rows_with_retry(
        client=client,
        collection_name=settings.MILVUS_MEMORY_COLLECTION_NAME,
        rows=rows,
    )
    register_event_memory_records(
        user_id=user_id,
        rows=rows,
        conversation_id=conversation_id,
        session_id=session_id,
    )
    logger.info(
        "✓ 长期记忆写入成功 (user_id=%s, conversation_id=%s, chunks=%s)",
        user_id,
        conversation_id,
        len(inserted_ids),
    )
    return inserted_ids


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
    turn_offset: int = 0,
    detect_topic_change: bool = True,
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
                should_split_topic=(
                    _build_topic_change_detector(llm)
                    if detect_topic_change
                    else lambda _current_text, _next_text: False
                ),
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

        return _write_memory_chunks(
            chunks=chunks,
            user_id=user_key,
            conversation_id=conversation_key,
            session_id=session_key,
            source_name=source_name,
            chunk_type=chunk_type,
            topic=topic_value,
            turn_offset=turn_offset,
            normalize_dialogue_turns=True,
        )
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
        # 每次 API 请求通常只写入一个 user/assistant 对话块；把本批次的
        # 局部轮次偏移到会话全局，后续摘要才能按 turn_start/turn_end 精确展开原文。
        turn_offset = _count_dialogue_turns(user_key, conversation_key)

        return _write_memory_chunks(
            chunks=chunks,
            user_id=user_key,
            conversation_id=conversation_key,
            session_id=session_key,
            source_name=source_name,
            chunk_type=chunk_type,
            topic=topic_value,
            turn_offset=turn_offset,
        )
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
    chunk_types: Optional[Sequence[str]] = None,
    topic: Optional[str] = None,
) -> List[RetrievalResult]:
    """按向量相似度检索长期记忆。"""
    try:
        if not query or not query.strip():
            raise RetrievalException("查询文本不能为空")
        user_key = _require_user_id(user_id)

        from app.core.embeddings import get_default_embeddings

        client = _prepare_memory_collection(require_loaded=True)
        if client is None:
            return []

        embeddings = get_default_embeddings()
        query_vector = embeddings.embed_query(query.strip())

        filter_expr = _build_memory_filter(
            user_id=user_key,
            conversation_id=conversation_id,
            session_id=session_id,
            chunk_type=chunk_type,
            chunk_types=chunk_types,
            topic=topic,
        )

        search_kwargs: Dict[str, Any] = {
            "collection_name": settings.MILVUS_MEMORY_COLLECTION_NAME,
            "data": [query_vector],
            "limit": max(top_k * 3, top_k),
            "output_fields": _memory_output_fields(),
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
        _mark_memory_accessed_async(
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


def _memory_output_fields() -> List[str]:
    """返回读取记忆向量时需要的业务字段。"""
    return [
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
    ]


def _query_memory_entities(
    client: Any,
    filter_expr: str,
    limit: int = 100,
) -> List[Any]:
    """按元数据读取记忆向量实体，兼容 Milvus Lite 与服务端返回格式。"""
    try:
        result = client.query(
            collection_name=settings.MILVUS_MEMORY_COLLECTION_NAME,
            filter=filter_expr,
            output_fields=_memory_output_fields(),
            limit=max(1, min(int(limit), 1000)),
        )
    except TypeError:
        # 部分旧 pymilvus 版本不接受 limit 关键字，保留兼容路径。
        result = client.query(
            collection_name=settings.MILVUS_MEMORY_COLLECTION_NAME,
            filter=filter_expr,
            output_fields=_memory_output_fields(),
        )
    return list(result or [])


def _serialize_memory_entity(entity: Any, active_records: Dict[str, MemoryRecord]) -> Optional[Dict[str, Any]]:
    """把 Milvus 实体转成已通过 SQL 生命周期校验的结构。"""
    memory_id = str(_get_field(entity, "memory_id", "") or "").strip()
    record = active_records.get(memory_id)
    if not memory_id or record is None:
        return None
    return {
        "memory_id": memory_id,
        "content": str(
            _get_field(entity, "chunk_text")
            or _get_field(entity, "text")
            or ""
        ),
        "chunk_index": _get_field(entity, "chunk_index"),
        "source_name": _get_field(entity, "source_name"),
        "user_id": _get_field(entity, "user_id"),
        "conversation_id": _get_field(entity, "conversation_id") or record.conversation_id,
        "session_id": _get_field(entity, "session_id") or record.session_id,
        "chunk_type": _get_field(entity, "chunk_type"),
        "topic": _get_field(entity, "topic"),
        "turn_start": _get_field(entity, "turn_start"),
        "turn_end": _get_field(entity, "turn_end"),
        "created_at": _get_field(entity, "created_at"),
        "status": record.status,
    }


def retrieve_original_conversation(
    user_id: Optional[str],
    *,
    memory_ids: Optional[Sequence[str]] = None,
    query: Optional[str] = None,
    conversation_id: Optional[str] = None,
    turn_start: Optional[int] = None,
    turn_end: Optional[int] = None,
    include_neighbors: int = 0,
    include_all_conversation: bool = False,
    top_k: int = 8,
    max_chars: int = 12000,
) -> Dict[str, Any]:
    """按摘要引用渐进式展开当前用户的原始对话。"""
    user_key = _require_user_id(user_id)
    normalized_ids = [str(item).strip() for item in (memory_ids or []) if str(item).strip()]
    normalized_query = str(query or "").strip()
    normalized_conversation = str(conversation_id or "").strip()
    if not normalized_ids and not normalized_query and not normalized_conversation:
        return {
            "mode": "original_conversation",
            "summary": "请提供摘要 memory_id、检索提示或 conversation_id。",
            "summary_references": [],
            "original_chunks": [],
        }

    top_k = max(1, min(int(top_k or 8), 50))
    max_chars = max(500, min(int(max_chars or 12000), 50000))
    include_neighbors = max(0, min(int(include_neighbors or 0), 5))
    if turn_start is not None:
        turn_start = max(0, int(turn_start))
    if turn_end is not None:
        turn_end = max(0, int(turn_end))
    if turn_start is not None and turn_end is not None and turn_end < turn_start:
        turn_start, turn_end = turn_end, turn_start

    client = _prepare_memory_collection(require_loaded=True)
    if client is None:
        return {
            "mode": "original_conversation",
            "summary": "长期记忆向量库当前不可用或尚未加载完成。",
            "summary_references": [],
            "original_chunks": [],
        }

    summary_references: List[Dict[str, Any]] = []
    if normalized_ids:
        escaped_ids = ", ".join(f'"{_escape_filter_value(item)}"' for item in normalized_ids)
        entities = _query_memory_entities(
            client,
            filter_expr=(
                f'user_id == "{_escape_filter_value(user_key)}" '
                f"and memory_id in [{escaped_ids}]"
            ),
            limit=len(normalized_ids),
        )
        active_records = _get_active_event_records(user_key, normalized_ids)
        for entity in entities:
            serialized = _serialize_memory_entity(entity, active_records)
            if serialized is not None:
                summary_references.append(serialized)
    elif include_all_conversation and normalized_conversation:
        raw_entities = _query_memory_entities(
            client,
            filter_expr=(
                f'user_id == "{_escape_filter_value(user_key)}" '
                f'and conversation_id == "{_escape_filter_value(normalized_conversation)}" '
                'and chunk_type == "dialogue"'
            ),
            limit=1000,
        )
        raw_ids = [
            str(_get_field(item, "memory_id", "") or "").strip()
            for item in raw_entities
        ]
        active_raw_records = _get_active_event_records(user_key, raw_ids)
        raw_chunks: List[Dict[str, Any]] = []
        for entity in raw_entities:
            serialized = _serialize_memory_entity(entity, active_raw_records)
            if serialized is not None:
                raw_chunks.append(serialized)
        raw_chunks.sort(
            key=lambda item: (
                int(item.get("turn_start") or 0),
                int(item.get("chunk_index") or 0),
            )
        )
        total_chars = 0
        bounded_chunks: List[Dict[str, Any]] = []
        for chunk in raw_chunks:
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            content = str(chunk.get("content") or "")
            if len(content) > remaining:
                chunk = {**chunk, "content": content[:remaining]}
            bounded_chunks.append(chunk)
            total_chars += len(str(chunk.get("content") or ""))
        _mark_memory_accessed_async(
            user_key,
            [str(item.get("memory_id") or "") for item in bounded_chunks],
        )
        return {
            "mode": "original_conversation",
            "summary": f"已展开当前会话的 {len(bounded_chunks)} 个原始对话块。",
            "summary_references": [],
            "original_chunks": bounded_chunks,
            "truncated": total_chars >= max_chars,
        }
    elif normalized_query:
        summary_hits = search_long_term_memory(
            query=normalized_query,
            top_k=top_k,
            user_id=user_key,
            conversation_id=normalized_conversation or None,
            chunk_types=["semantic_memory_l3", "semantic_memory_l2", "semantic_memory_l1"],
        )
        for hit in summary_hits:
            metadata = dict(hit.metadata or {})
            summary_references.append(
                {
                    "memory_id": str(metadata.get("memory_id") or ""),
                    "content": str(hit.content or ""),
                    "conversation_id": metadata.get("conversation_id"),
                    "session_id": metadata.get("session_id"),
                    "chunk_type": metadata.get("chunk_type"),
                    "topic": metadata.get("topic"),
                    "turn_start": metadata.get("turn_start"),
                    "turn_end": metadata.get("turn_end"),
                    "score": float(hit.score or 0.0),
                }
            )

    # 若用户直接指定会话但没有摘要 ID，仍允许从该会话的原文块中取回上下文。
    if not summary_references and normalized_query:
        raw_hits = search_long_term_memory(
            query=normalized_query,
            top_k=top_k,
            user_id=user_key,
            conversation_id=normalized_conversation or None,
            chunk_type="dialogue",
        )
        raw_chunks = []
        for hit in raw_hits:
            metadata = dict(hit.metadata or {})
            raw_chunks.append(
                {
                    "memory_id": metadata.get("memory_id"),
                    "conversation_id": metadata.get("conversation_id"),
                    "turn_start": metadata.get("turn_start"),
                    "turn_end": metadata.get("turn_end"),
                    "content": str(hit.content or ""),
                    "score": float(hit.score or 0.0),
                }
            )
        return {
            "mode": "original_conversation",
            "summary": f"已按语义找到 {len(raw_chunks)} 个原始对话块。",
            "summary_references": [],
            "original_chunks": raw_chunks,
        }

    original_chunks: List[Dict[str, Any]] = []
    seen_raw_ids: set[str] = set()
    for reference in summary_references[:top_k]:
        reference_conversation = str(reference.get("conversation_id") or "").strip()
        if (
            normalized_conversation
            and reference_conversation
            and reference_conversation != normalized_conversation
        ):
            continue
        ref_conversation = str(reference_conversation or normalized_conversation).strip()
        if not ref_conversation:
            continue
        ref_start = reference.get("turn_start") if turn_start is None else turn_start
        ref_end = reference.get("turn_end") if turn_end is None else turn_end
        try:
            ref_start = int(ref_start) if ref_start is not None else None
        except (TypeError, ValueError):
            ref_start = None
        try:
            ref_end = int(ref_end) if ref_end is not None else None
        except (TypeError, ValueError):
            ref_end = None
        if ref_start is not None:
            ref_start = max(0, ref_start - include_neighbors)
        if ref_end is not None:
            ref_end += include_neighbors

        clauses = [
            f'user_id == "{_escape_filter_value(user_key)}"',
            f'conversation_id == "{_escape_filter_value(ref_conversation)}"',
            'chunk_type == "dialogue"',
        ]
        if ref_start is not None:
            clauses.append(f"turn_end >= {ref_start}")
        if ref_end is not None:
            clauses.append(f"turn_start <= {ref_end}")
        raw_entities = _query_memory_entities(
            client,
            filter_expr=" and ".join(clauses),
            limit=top_k * 4,
        )
        raw_ids = [str(_get_field(item, "memory_id", "") or "").strip() for item in raw_entities]
        active_raw_records = _get_active_event_records(user_key, raw_ids)
        for entity in raw_entities:
            serialized = _serialize_memory_entity(entity, active_raw_records)
            if serialized is None or serialized["memory_id"] in seen_raw_ids:
                continue
            seen_raw_ids.add(serialized["memory_id"])
            serialized["source_summary_memory_id"] = reference.get("memory_id")
            serialized["score"] = reference.get("score", 0.0)
            original_chunks.append(serialized)

        # 兼容历史摘要：早期记录的 dialogue 坐标可能不完整，但摘要向量本身
        # 持有【原文】段，仍可在不扩大召回范围的前提下恢复对应上下文。
        if not raw_entities:
            summary_content = str(reference.get("content") or "")
            if "【原文】" in summary_content:
                transcript = summary_content.split("【原文】", 1)[1].strip()
                if transcript:
                    original_chunks.append(
                        {
                            "memory_id": reference.get("memory_id"),
                            "conversation_id": ref_conversation,
                            "session_id": reference.get("session_id"),
                            "chunk_type": "dialogue_recovered",
                            "turn_start": ref_start,
                            "turn_end": ref_end,
                            "content": transcript,
                            "source_summary_memory_id": reference.get("memory_id"),
                            "score": reference.get("score", 0.0),
                        }
                    )

    original_chunks.sort(
        key=lambda item: (
            str(item.get("conversation_id") or ""),
            int(item.get("turn_start") or 0),
            int(item.get("chunk_index") or 0),
        )
    )
    total_chars = 0
    bounded_chunks: List[Dict[str, Any]] = []
    for chunk in original_chunks:
        content = str(chunk.get("content") or "")
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        if len(content) > remaining:
            chunk = {**chunk, "content": content[:remaining]}
        bounded_chunks.append(chunk)
        total_chars += len(str(chunk.get("content") or ""))

    _mark_memory_accessed_async(
        user_key,
        [str(item.get("memory_id") or "") for item in summary_references]
        + [str(item.get("memory_id") or "") for item in bounded_chunks],
    )
    return {
        "mode": "original_conversation",
        "summary": f"已展开 {len(bounded_chunks)} 个原始对话块。",
        "summary_references": [
            {
                "memory_id": item.get("memory_id"),
                "conversation_id": item.get("conversation_id"),
                "chunk_type": item.get("chunk_type"),
                "topic": item.get("topic"),
                "turn_start": item.get("turn_start"),
                "turn_end": item.get("turn_end"),
            }
            for item in summary_references[:top_k]
        ],
        "original_chunks": bounded_chunks,
        "truncated": total_chars >= max_chars,
    }


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


def delete_profile_memories(
    user_id: Optional[str],
    memory_keys: Optional[Sequence[str]] = None,
    keywords: Optional[Sequence[str]] = None,
    max_deletes: int = 50,
) -> Dict[str, Any]:
    """按 memory_key 精确或按关键词模糊删除用户的个性化记忆。

    与 LLM 抽取路径（action=delete）不同，这里是确定性的主动删除入口，
    供「用户明确要求删除记忆」的 skill 调用。删除采用软删除（标记
    DELETED），保留版本历史以便追溯。
    """
    user_key = _require_user_id(user_id)
    normalized_keys = [
        str(item).strip().lower()
        for item in (memory_keys or [])
        if str(item or "").strip()
    ]
    normalized_keywords = [
        normalize_text(str(item))
        for item in (keywords or [])
        if str(item or "").strip()
    ]
    if not normalized_keys and not normalized_keywords:
        return {
            "deleted_count": 0,
            "matched": [],
            "summary": "未提供任何删除条件，请指定 memory_key 或 keywords。",
        }

    db = SessionLocal()
    deleted: List[Dict[str, Any]] = []
    try:
        candidates = (
            db.query(MemoryRecord)
            .filter(
                MemoryRecord.user_id == user_key,
                MemoryRecord.memory_type == MEMORY_TYPE_PROFILE,
                MemoryRecord.status == MEMORY_STATUS_ACTIVE,
            )
            .all()
        )
        if not candidates:
            return {
                "deleted_count": 0,
                "matched": [],
                "summary": "当前没有可删除的个性化记忆。",
            }

        matched: List[MemoryRecord] = []
        for record in candidates:
            record_key = str(record.memory_key or "").strip().lower()
            record_content = normalize_text(record.content or "")
            if normalized_keys and record_key in normalized_keys:
                matched.append(record)
                continue
            if normalized_keywords and any(
                keyword and (keyword in record_key or keyword in record_content)
                for keyword in normalized_keywords
            ):
                matched.append(record)

        if not matched:
            return {
                "deleted_count": 0,
                "matched": [],
                "summary": "没有匹配到符合条件的个性化记忆。",
            }

        if len(matched) > max_deletes:
            return {
                "deleted_count": 0,
                "matched": [
                    {
                        "memory_key": str(record.memory_key or ""),
                        "content": str(record.content or "")[:200],
                    }
                    for record in matched[:max_deletes]
                ],
                "summary": (
                    f"匹配到 {len(matched)} 条记忆，超过单次删除上限 {max_deletes}，"
                    "为避免误删请用更精确的 memory_key 或关键词重试。"
                ),
            }

        now = _utcnow_naive()
        for record in matched:
            record.status = MEMORY_STATUS_DELETED
            record.updated_at = now
            deleted.append(
                {
                    "memory_id": record.memory_id,
                    "memory_key": record.memory_key,
                    "content": str(record.content or "")[:400],
                    "confidence_score": float(record.confidence_score or 0.0),
                }
            )
        db.commit()
        return {
            "deleted_count": len(deleted),
            "matched": deleted,
            "summary": f"已删除 {len(deleted)} 条个性化记忆。",
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()



# ==================== 递进摘要合并（memory consolidation）====================

# 三层递进摘要阈值：每满 10 轮合并 L1，满 20 轮合并 L2，满 40 轮合并 L3。
# 每层是上一层的 2 倍，L2 输入恰好 2 个 L1，L3 输入恰好 2 个 L2，层级自洽。
CONSOLIDATION_LEVELS: tuple[tuple[int, int], ...] = ((10, 1), (20, 2), (40, 3))

LOW_INFORMATION_DIALOGUE = {
    "好",
    "好的",
    "收到",
    "明白",
    "了解",
    "知道了",
    "可以",
    "行",
    "嗯",
    "嗯嗯",
    "谢谢",
    "感谢",
    "ok",
    "okay",
    "thanks",
    "thankyou",
}
_CONSOLIDATION_LOCK = threading.Lock()
_CONSOLIDATION_IN_FLIGHT: set[tuple[str, str]] = set()
_CONSOLIDATION_RETRY_STATE: Dict[tuple[str, str, int], tuple[float, int]] = {}
_CONSOLIDATION_RETRY_DELAYS = (60.0, 300.0, 1800.0)
MERGE_SUMMARY_SYSTEM_PROMPT = (
    "你是长期记忆递进摘要器。把以下多个摘要块压缩成一段更精炼的高层摘要。"
    "保留跨块共有的主题、结论、偏好和待办；丢弃单块琐碎细节。"
    "只输出严格 JSON，格式：{\"topic\":\"主题\",\"summary\":\"一段摘要\"}"
)


def _is_substantive_dialogue(messages: Sequence[Dict[str, str]]) -> bool:
    """判断一段对话是否包含值得生成长期摘要的用户信息。"""
    for message in messages:
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        normalized = re.sub(r"\s+", "", str(message.get("content") or "").strip().lower())
        normalized = re.sub(
            r"[，。！？、,.!?：:；;‘’“”\"'（）()\[\]{}]",
            "",
            normalized,
        )
        if normalized and normalized not in LOW_INFORMATION_DIALOGUE:
            return True
    return False


def _require_user_key(value: Optional[str]) -> str:
    return _require_user_id(value)


def _get_consolidation_cursor(
    user_id: str,
    conversation_id: str,
    level: int,
) -> int:
    """读取某一层的合并游标（已合并到的轮次），无记录返回 0。"""
    db = SessionLocal()
    try:
        record = (
            db.query(MemoryConsolidationRecord)
            .filter(
                MemoryConsolidationRecord.user_id == user_id,
                MemoryConsolidationRecord.conversation_id == conversation_id,
                MemoryConsolidationRecord.level == level,
            )
            .first()
        )
        return int(record.last_merged_turn or 0) if record else 0
    finally:
        db.close()


def _update_consolidation_cursor(
    user_id: str,
    conversation_id: str,
    level: int,
    new_turn: int,
) -> None:
    """更新某一层的合并游标；不存在则创建。"""
    db = SessionLocal()
    try:
        record = (
            db.query(MemoryConsolidationRecord)
            .filter(
                MemoryConsolidationRecord.user_id == user_id,
                MemoryConsolidationRecord.conversation_id == conversation_id,
                MemoryConsolidationRecord.level == level,
            )
            .first()
        )
        if record is None:
            record = MemoryConsolidationRecord(
                user_id=user_id,
                conversation_id=conversation_id,
                level=level,
                last_merged_turn=0,
            )
            db.add(record)
        record.last_merged_turn = max(0, int(new_turn))
        record.last_merged_at = _utcnow_naive()
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _count_dialogue_turns(user_id: str, conversation_id: str) -> int:
    """统计当前会话已覆盖的逻辑轮次，兼容旧的无轮次元数据记录。"""
    prefix = f"event:{user_id}:{conversation_id}:dialogue:"
    db = SessionLocal()
    try:
        records = (
            db.query(MemoryRecord.turn_start, MemoryRecord.turn_end)
            .filter(
                MemoryRecord.user_id == user_id,
                MemoryRecord.conversation_id == conversation_id,
                MemoryRecord.memory_type == MEMORY_TYPE_EVENT,
                MemoryRecord.memory_key.like(prefix + "%"),
                MemoryRecord.status == MEMORY_STATUS_ACTIVE,
            )
            .all()
        )
        if not records:
            return 0

        known_end = [
            int(turn_end)
            for _turn_start, turn_end in records
            if turn_end is not None
        ]
        unknown_count = len(records) - len(known_end)
        return (max(known_end) + 1 if known_end else 0) + unknown_count
    except Exception:
        return 0
    finally:
        db.close()


def _load_summary_chunks(
    user_id: str,
    conversation_id: str,
    level: int,
    start_turn: int,
    end_turn: int,
) -> List[Dict[str, str]]:
    """读取某一层在轮次区间内生成的摘要块，返回 [{topic, summary}] 列表。"""
    prefix = f"event:{user_id}:{conversation_id}:semantic_memory_l{level}:"
    db = SessionLocal()
    try:
        records = (
            db.query(MemoryRecord)
            .filter(
                MemoryRecord.user_id == user_id,
                MemoryRecord.conversation_id == conversation_id,
                MemoryRecord.memory_type == MEMORY_TYPE_EVENT,
                MemoryRecord.memory_key.like(prefix + "%"),
                MemoryRecord.status == MEMORY_STATUS_ACTIVE,
                or_(MemoryRecord.turn_end.is_(None), MemoryRecord.turn_end >= start_turn),
                or_(MemoryRecord.turn_start.is_(None), MemoryRecord.turn_start < end_turn),
            )
            .order_by(MemoryRecord.turn_start.asc(), MemoryRecord.id.asc())
            .all()
        )
        chunks: List[Dict[str, str]] = []
        for record in records:
            content = str(record.content or "")
            topic = ""
            summary = ""
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("【主题】"):
                    topic = line.removeprefix("【主题】").strip()
                elif line.startswith("【摘要】"):
                    summary = line.removeprefix("【摘要】").strip()
            if summary:
                chunks.append({"topic": topic, "summary": summary})
        return chunks
    finally:
        db.close()


def _load_dialogue_turn_range_by_prefix(
    user_id: str,
    conversation_id: str,
    start_turn: int,
    end_turn: int,
) -> List[Dict[str, str]]:
    """按 memory_key 前缀读取 dialogue 原文（兼容 register 时的 key 规则）。"""
    db = SessionLocal()
    try:
        prefix = f"event:{user_id}:{conversation_id}:dialogue:"
        records = (
            db.query(MemoryRecord)
            .filter(
                MemoryRecord.user_id == user_id,
                MemoryRecord.conversation_id == conversation_id,
                MemoryRecord.memory_type == MEMORY_TYPE_EVENT,
                MemoryRecord.memory_key.like(prefix + "%"),
                MemoryRecord.status == MEMORY_STATUS_ACTIVE,
                or_(MemoryRecord.turn_end.is_(None), MemoryRecord.turn_end >= start_turn),
                or_(MemoryRecord.turn_start.is_(None), MemoryRecord.turn_start < end_turn),
            )
            .order_by(MemoryRecord.turn_start.asc(), MemoryRecord.id.asc())
            .all()
        )
        messages: List[Dict[str, str]] = []
        for record in records:
            content = str(record.content or "")
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("用户:"):
                    messages.append({"role": "user", "content": line[3:].strip()})
                elif line.startswith("助手:"):
                    messages.append({"role": "assistant", "content": line[3:].strip()})
        return messages
    finally:
        db.close()


def merge_summary_level(
    chunks: Sequence[Dict[str, str]],
    llm: Any,
    user_id: str,
    conversation_id: str,
    level: int,
    turn_start: int,
    turn_end: int,
) -> int:
    """把低层摘要块压缩成高层摘要块，入库为 semantic_memory_l{level}。

    输入 chunks 只含低层摘要的 topic/summary，不含原文——高层只负责
    跨段浓缩，细节下沉在低层。返回写入的块数。
    """
    if not chunks:
        return 0
    if llm is None:
        return 0

    texts = []
    for chunk in chunks:
        topic = str(chunk.get("topic") or "").strip()
        summary = str(chunk.get("summary") or "").strip()
        if topic:
            texts.append(f"[{topic}] {summary}")
        else:
            texts.append(summary)

    response = llm.invoke(
        [
            SystemMessage(content=MERGE_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(texts)),
        ]
    )
    payload = _safe_json_loads(_extract_text(response))
    topic = str(payload.get("topic") or "对话摘要").strip()
    summary = str(payload.get("summary") or "").strip()
    if not summary:
        summary = (texts[0] if texts else "")[:120]

    block_text = format_long_term_memory_chunk(
        summary=summary,
        transcript="\n".join(texts),
        topic=topic,
    )

    try:
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
        vector = embeddings.embed_documents([block_text])[0]

        memory_id = (
            f"{user_id}:{conversation_id}:L{level}:"
            f"{turn_start}:{turn_end}:{uuid.uuid4().hex}"
        )
        chunk_index = _count_dialogue_turns(user_id, conversation_id) + turn_start
        row = {
            "text": block_text,
            "vector": vector,
            "memory_id": memory_id,
            "chunk_index": chunk_index,
            "source_name": "conversation",
            "chunk_text": block_text,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "session_id": conversation_id,
            "chunk_type": f"semantic_memory_l{level}",
            "topic": topic,
            "turn_start": turn_start,
            "turn_end": turn_end,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        inserted_ids = insert_rows_with_retry(
            client=client,
            collection_name=settings.MILVUS_MEMORY_COLLECTION_NAME,
            rows=[row],
        )
        register_event_memory_records(
            user_id=user_id,
            rows=[row],
            conversation_id=conversation_id,
        )
        logger.info(
            "✓ 递进摘要 L%d 写入成功 (user_id=%s, conversation_id=%s, turns=%s-%s)",
            level,
            user_id,
            conversation_id,
            turn_start,
            turn_end,
        )
        return len(inserted_ids)
    except Exception as error:
        logger.error(f"递进摘要 L{level} 写入失败: {str(error)}")
        return 0


def _consolidate_level(
    user_id: str,
    conversation_id: str,
    level: int,
    threshold: int,
    llm: Any,
) -> int:
    """执行某一层的增量合并：只处理上次合并后的新段。"""
    user_key = _require_user_key(user_id)
    cursor = _get_consolidation_cursor(user_key, conversation_id, level)
    segment_start = (cursor // threshold) * threshold

    if level == 1:
        # L1：输入新增 dialogue 原文，按固定轮次批量摘要并保留原文。
        messages = _load_dialogue_turn_range_by_prefix(
            user_key,
            conversation_id,
            start_turn=segment_start,
            end_turn=segment_start + threshold,
        )
        if not messages:
            return 0
        if not _is_substantive_dialogue(messages):
            logger.info(
                "L1 摘要跳过低信息对话 (user_id=%s, conversation_id=%s, turns=%s-%s)",
                user_key,
                conversation_id,
                segment_start,
                segment_start + threshold - 1,
            )
            # 这是有意跳过而非失败；推进游标，避免低信息段阻塞后续摘要窗口。
            _update_consolidation_cursor(
                user_key,
                conversation_id,
                level,
                segment_start + threshold,
            )
            return 0
        try:
            inserted = store_semantic_long_term_memory(
                messages=messages,
                user_id=user_key,
                conversation_id=conversation_id,
                chunk_type="semantic_memory_l1",
                llm=llm,
                token_limit=700,
                include_system=False,
                # 合并的就是窗口外历史，不能再排除最近 N 轮，否则全部轮次被丢弃
                short_window_n=0,
                # dialogue 记录按会话全局轮次编号，摘要引用沿用同一坐标系。
                # L1 的 chunks 是本段内局部轮次，需要平移到 segment_start。
                turn_offset=segment_start,
                # L1 已按固定轮次批量处理，避免每个相邻轮次额外调用话题判断 LLM。
                detect_topic_change=False,
            )
        except Exception as exc:
            logger.warning("L1 合并失败，下次触发点重试: %s", exc)
            return 0
    else:
        # L2 / L3：输入是低层摘要块 → LLM 递进压缩
        lower_chunks = _load_summary_chunks(
            user_key,
            conversation_id,
            level=level - 1,
            start_turn=segment_start,
            end_turn=segment_start + threshold,
        )
        inserted = merge_summary_level(
            chunks=lower_chunks,
            llm=llm,
            user_id=user_key,
            conversation_id=conversation_id,
            level=level,
            turn_start=segment_start,
            turn_end=segment_start + threshold,
        )

    # store_semantic_long_term_memory 返回 List，merge_summary_level 返回 int，
    # 统一归一为写入块数，避免 consolidate_check 里 merged += list 报错。
    inserted_count = len(inserted) if isinstance(inserted, (list, tuple)) else int(inserted or 0)
    if inserted_count <= 0:
        return 0

    _update_consolidation_cursor(
        user_key,
        conversation_id,
        level,
        segment_start + threshold,
    )
    return inserted_count


def consolidate_check(
    user_id: Optional[str],
    conversation_id: Optional[str],
    llm: Optional[Any] = None,
) -> int:
    """每轮写入后调用：逐层检查是否达到合并阈值。

    纯计数检查，未到阈值零成本返回。触发顺序按 L1 → L2 → L3，
    保证低层先于高层执行（L2 依赖 L1，L3 依赖 L2）。
    """
    user_key = _require_user_id(user_id)
    normalized_conversation = (conversation_id or "").strip()
    if not normalized_conversation:
        return 0

    lock_key = (user_key, normalized_conversation)
    with _CONSOLIDATION_LOCK:
        if lock_key in _CONSOLIDATION_IN_FLIGHT:
            return 0
        _CONSOLIDATION_IN_FLIGHT.add(lock_key)

    try:
        current_turn = _count_dialogue_turns(user_key, normalized_conversation)
        if current_turn <= 0:
            return 0

        merged = 0
        now = time.monotonic()
        for threshold, level in CONSOLIDATION_LEVELS:
            if current_turn < threshold:
                continue
            cursor = _get_consolidation_cursor(user_key, normalized_conversation, level)
            if current_turn // threshold <= cursor // threshold:
                continue

            retry_key = (user_key, normalized_conversation, level)
            with _CONSOLIDATION_LOCK:
                retry_state = _CONSOLIDATION_RETRY_STATE.get(retry_key)
            if retry_state is not None and now < retry_state[0]:
                continue

            inserted = _consolidate_level(
                user_key,
                normalized_conversation,
                level=level,
                threshold=threshold,
                llm=llm,
            )
            updated_cursor = _get_consolidation_cursor(
                user_key,
                normalized_conversation,
                level,
            )
            if inserted > 0 or updated_cursor > cursor:
                merged += inserted
                with _CONSOLIDATION_LOCK:
                    _CONSOLIDATION_RETRY_STATE.pop(retry_key, None)
            else:
                with _CONSOLIDATION_LOCK:
                    previous_attempts = (
                        retry_state[1] if retry_state is not None else 0
                    )
                    delay_index = min(previous_attempts, len(_CONSOLIDATION_RETRY_DELAYS) - 1)
                    _CONSOLIDATION_RETRY_STATE[retry_key] = (
                        time.monotonic() + _CONSOLIDATION_RETRY_DELAYS[delay_index],
                        previous_attempts + 1,
                    )
        return merged
    finally:
        with _CONSOLIDATION_LOCK:
            _CONSOLIDATION_IN_FLIGHT.discard(lock_key)


def consolidate_conversation_memory(
    user_id: Optional[str],
    conversation_id: Optional[str],
    llm: Optional[Any] = None,
) -> int:
    """会话结束兜底：对整段历史做一次全量三层合并。"""
    user_key = _require_user_id(user_id)
    normalized_conversation = (conversation_id or "").strip()
    if not normalized_conversation:
        return 0

    merged = 0
    for threshold, level in CONSOLIDATION_LEVELS:
        merged += _consolidate_level(
            user_key,
            normalized_conversation,
            level=level,
            threshold=threshold,
            llm=llm,
        )
    return merged


__all__ = [
    "store_semantic_long_term_memory",
    "store_long_term_memory",
    "search_long_term_memory",
    "delete_long_term_memory",
    "store_conversation_memory",
    "search_conversation_memory",
    "retrieve_original_conversation",
    "delete_conversation_memory",
    "get_active_profile_memories",
    "upsert_profile_memory_candidates",
    "register_event_memory_records",
    "delete_profile_memories",
    "cleanup_memory_records",
    "consolidate_check",
    "consolidate_conversation_memory",
    "merge_summary_level",
]
