"""用户对话记录的服务端持久化。"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.storage.sqlite_metadata import SessionLocal, UserConversationRecord
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def _utcnow() -> datetime:
    return datetime.utcnow()


def _parse_client_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except (TypeError, ValueError):
        return None


def _normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    """过滤临时占位消息，避免把加载动画或半成品状态写进历史。"""
    if not isinstance(messages, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in messages[:500]:
        if not isinstance(item, dict):
            continue
        if item.get("ephemeral"):
            continue
        role = str(item.get("role") or "").strip()
        if role not in {"user", "assistant"}:
            continue
        normalized.append(item)
    return normalized


def _serialize_record(record: UserConversationRecord) -> Dict[str, Any]:
    try:
        messages = json.loads(record.payload or "[]")
    except (TypeError, json.JSONDecodeError):
        messages = []
    return {
        "conversation_id": record.conversation_id,
        "session_id": record.session_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "messages": messages if isinstance(messages, list) else [],
    }


def list_user_conversations(
    user_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """返回某个账号按更新时间倒序的对话记录。"""
    if not user_id:
        return []
    resolved_limit = max(1, min(int(limit or 100), 500))
    db = SessionLocal()
    try:
        records = (
            db.query(UserConversationRecord)
            .filter(UserConversationRecord.user_id == user_id)
            .order_by(UserConversationRecord.updated_at.desc())
            .limit(resolved_limit)
            .all()
        )
        return [_serialize_record(record) for record in records]
    finally:
        db.close()


def list_user_conversation_ids(user_id: str) -> List[str]:
    """返回某个账号的全部服务端对话 ID。"""
    if not user_id:
        return []
    db = SessionLocal()
    try:
        records = (
            db.query(UserConversationRecord.conversation_id)
            .filter(UserConversationRecord.user_id == user_id)
            .all()
        )
        return [str(row[0]) for row in records if row[0]]
    finally:
        db.close()


def upsert_user_conversation(
    user_id: str,
    conversation_id: str,
    session_id: Optional[str] = None,
    messages: Optional[Any] = None,
    created_at: Optional[Any] = None,
    updated_at: Optional[Any] = None,
) -> Dict[str, Any]:
    """新增或覆盖一个用户的对话记录。"""
    normalized_conversation_id = (conversation_id or "").strip()
    if not user_id or not normalized_conversation_id:
        raise ValueError("user_id 和 conversation_id 不能为空")

    payload = json.dumps(
        _normalize_messages(messages),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    now = _utcnow()
    db = SessionLocal()
    try:
        record = (
            db.query(UserConversationRecord)
            .filter(
                UserConversationRecord.user_id == user_id,
                UserConversationRecord.conversation_id == normalized_conversation_id,
            )
            .first()
        )
        if record is None:
            record = UserConversationRecord(
                user_id=user_id,
                conversation_id=normalized_conversation_id,
                session_id=(session_id or "").strip() or None,
                payload=payload,
                created_at=_parse_client_datetime(created_at) or now,
                updated_at=now,
            )
            db.add(record)
        else:
            record.session_id = (session_id or "").strip() or None
            record.payload = payload
            record.updated_at = now
        db.commit()
        db.refresh(record)
        return _serialize_record(record)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def clear_user_conversations(user_id: str) -> int:
    """清空某个账号的全部服务端对话记录。"""
    if not user_id:
        return 0
    db = SessionLocal()
    try:
        deleted = (
            db.query(UserConversationRecord)
            .filter(UserConversationRecord.user_id == user_id)
            .delete(synchronize_session=False)
        )
        db.commit()
        if deleted:
            logger.info("用户对话记录已清空 (user_id=%s, deleted=%s)", user_id, deleted)
        return int(deleted or 0)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_user_conversation(
    user_id: str,
    conversation_id: str,
) -> bool:
    """删除一个用户的对话记录。"""
    normalized_conversation_id = (conversation_id or "").strip()
    if not user_id or not normalized_conversation_id:
        return False
    db = SessionLocal()
    try:
        record = (
            db.query(UserConversationRecord)
            .filter(
                UserConversationRecord.user_id == user_id,
                UserConversationRecord.conversation_id == normalized_conversation_id,
            )
            .first()
        )
        if record is None:
            return False
        db.delete(record)
        db.commit()
        logger.info(
            "用户对话记录已删除 (user_id=%s, conversation_id=%s)",
            user_id,
            normalized_conversation_id,
        )
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
