"""内置 Skill：热点词统计、记忆查询。"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable as IterableABC
from datetime import date, datetime
from typing import Any, Dict, List, Sequence

from sqlalchemy import or_

from app.rag.retrieval.reranker import normalize_text
from app.services.memory_service import get_active_profile_memories, search_long_term_memory
from app.skills.registry import SkillDefinition, SkillExecutionContext, register_skill
from app.storage.sqlite_metadata import DocumentChunkRecord, DocumentRecord, SessionLocal

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,}")
_NOISY_TERMS = {
    "文档",
    "文件",
    "内容",
    "信息",
    "资料",
    "数据",
    "记录",
    "系统",
    "知识库",
    "查询",
    "统计",
    "热点",
    "热点词",
    "全部",
    "所有",
    "其他",
    "相关",
    "说明",
    "问题",
    "回答",
    "表格",
    "附件",
    "任务",
    "记录",
    "内容",
    "document",
    "file",
    "content",
    "data",
}


def _make_json_safe(value: Any) -> Any:
    """把 ORM 对象和时间字段转成可直接 JSON 编码的结构。"""
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, IterableABC):
        items = list(value)
    else:
        items = [value]
    normalized: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
    return normalized


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = default
    return max(minimum, min(maximum, resolved))


def _is_noisy_term(term: str) -> bool:
    normalized = normalize_text(term)
    if len(normalized) < 2:
        return True
    if normalized in _NOISY_TERMS:
        return True
    if normalized.isdigit():
        return True
    if re.fullmatch(r"[\W_]+", normalized):
        return True
    return False


def _extract_term_frequencies_from_text(text: str) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    if not text:
        return frequencies

    for match in _TOKEN_PATTERN.finditer(text):
        cleaned = normalize_text(match.group(0))
        if _is_noisy_term(cleaned):
            continue
        frequencies[cleaned] += 1

    return frequencies


def _apply_document_filters(
    query,
    document_ids: Sequence[str],
    file_types: Sequence[str],
    statuses: Sequence[str],
    file_name: str = "",
):
    if document_ids:
        query = query.filter(DocumentRecord.document_id.in_(list(document_ids)))
    if file_types:
        query = query.filter(DocumentRecord.file_type.in_(list(file_types)))
    if statuses:
        query = query.filter(DocumentRecord.status.in_(list(statuses)))
    if file_name:
        pattern = f"%{file_name.strip()}%"
        query = query.filter(
            or_(
                DocumentRecord.original_filename.ilike(pattern),
                DocumentRecord.stored_filename.ilike(pattern),
            )
        )
    return query


def _serialize_document_record(record: DocumentRecord) -> Dict[str, Any]:
    return {
        "document_id": record.document_id,
        "original_filename": record.original_filename,
        "stored_filename": record.stored_filename,
        "file_type": record.file_type,
        "status": record.status,
        "file_size": int(record.file_size or 0),
        "chunk_count": int(record.chunk_count or 0),
        "created_at": _make_json_safe(record.created_at),
        "updated_at": _make_json_safe(record.updated_at),
    }


def _serialize_profile_memory(record: Any) -> Dict[str, Any]:
    return {
        "memory_id": getattr(record, "memory_id", ""),
        "memory_type": getattr(record, "memory_type", "profile"),
        "memory_key": getattr(record, "memory_key", None),
        "content": getattr(record, "content", ""),
        "status": getattr(record, "status", ""),
        "importance_score": float(getattr(record, "importance_score", 0.0) or 0.0),
        "confidence_score": float(getattr(record, "confidence_score", 0.0) or 0.0),
        "access_count": int(getattr(record, "access_count", 0) or 0),
        "conversation_id": getattr(record, "conversation_id", None),
        "session_id": getattr(record, "session_id", None),
        "created_at": _make_json_safe(getattr(record, "created_at", None)),
        "updated_at": _make_json_safe(getattr(record, "updated_at", None)),
        "expires_at": _make_json_safe(getattr(record, "expires_at", None)),
    }


def _serialize_retrieval_memory(record: Any) -> Dict[str, Any]:
    metadata = getattr(record, "metadata", {}) or {}
    return {
        "memory_id": metadata.get("memory_id"),
        "content": getattr(record, "content", ""),
        "score": float(getattr(record, "score", 0.0) or 0.0),
        "source": getattr(record, "source", "conversation"),
        "metadata": _make_json_safe(metadata),
    }


def _skill_document_hot_words(context: SkillExecutionContext, arguments: Dict[str, Any]) -> Dict[str, Any]:
    top_k = _coerce_int(arguments.get("top_k", 10), default=10, minimum=1, maximum=50)
    document_ids = _normalize_string_list(arguments.get("document_ids"))
    file_types = _normalize_string_list(arguments.get("file_types"))
    statuses = _normalize_string_list(arguments.get("statuses") or arguments.get("status"))
    file_name = str(arguments.get("file_name") or arguments.get("filename") or "").strip()
    include_samples = bool(arguments.get("include_samples", True))

    db = SessionLocal()
    try:
        docs_query = db.query(DocumentRecord)
        docs_query = _apply_document_filters(docs_query, document_ids, file_types, statuses, file_name)
        documents = docs_query.order_by(DocumentRecord.updated_at.desc()).all()
        if not documents:
            return {
                "scope": "documents",
                "filters": {
                    "document_ids": document_ids,
                    "file_types": file_types,
                    "statuses": statuses,
                    "file_name": file_name,
                },
                "summary": {
                    "document_count": 0,
                    "chunk_count": 0,
                },
                "top_words": [],
            }

        doc_map = {record.document_id: record for record in documents}
        chunk_query = (
            db.query(DocumentChunkRecord.document_id, DocumentChunkRecord.chunk_text)
            .filter(DocumentChunkRecord.document_id.in_(list(doc_map.keys())))
            .order_by(DocumentChunkRecord.document_id, DocumentChunkRecord.chunk_index)
        )
        chunks_by_document: Dict[str, List[str]] = defaultdict(list)
        for document_id, chunk_text in chunk_query.all():
            text = str(chunk_text or "").strip()
            if text:
                chunks_by_document[str(document_id)].append(text)

        term_documents: Dict[str, List[str]] = defaultdict(list)
        term_frequency: Counter[str] = Counter()

        for record in documents:
            combined_parts = [record.file_type, record.status]
            combined_parts.extend(chunks_by_document.get(record.document_id, []))
            combined_text = " ".join(part for part in combined_parts if part)[:12000]
            document_term_frequency = _extract_term_frequencies_from_text(combined_text)
            term_frequency.update(document_term_frequency)
            if include_samples:
                for term in document_term_frequency:
                    if record.document_id not in term_documents[term]:
                        term_documents[term].append(record.document_id)

        ranked_terms = sorted(
            term_frequency.items(),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )

        top_words = []
        for term, count in ranked_terms[:top_k]:
            sample_documents = [
                _serialize_document_record(doc_map[document_id])
                for document_id in term_documents.get(term, [])[:3]
                if document_id in doc_map
            ]
            top_words.append(
                {
                    "term": term,
                    "document_count": count,
                    "sample_documents": sample_documents,
                }
            )

        return {
            "scope": "documents",
            "filters": {
                "document_ids": document_ids,
                "file_types": file_types,
                "statuses": statuses,
                "file_name": file_name,
            },
            "summary": {
                "document_count": len(documents),
                "chunk_count": sum(int(record.chunk_count or 0) for record in documents),
            },
            "top_words": _make_json_safe(top_words),
        }
    finally:
        db.close()


def _skill_memory_query(context: SkillExecutionContext, arguments: Dict[str, Any]) -> Dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    top_k = _coerce_int(arguments.get("top_k", 5), default=5, minimum=1, maximum=50)
    memory_scope = str(arguments.get("memory_scope", "all") or "all").strip().lower()
    include_profile = memory_scope in {"all", "profile"}
    include_event = memory_scope in {"all", "event"}

    if not include_profile and not include_event:
        memory_scope = "all"
        include_profile = True
        include_event = True

    result: Dict[str, Any] = {
        "query": query,
        "memory_scope": memory_scope,
        "filters": {
            "conversation_id": context.conversation_id,
            "session_id": context.session_id,
            "top_k": top_k,
        },
        "profile_memories": [],
        "matched_memories": [],
    }

    if include_profile:
        profile_memories = get_active_profile_memories(context.user_id, limit=top_k)
        result["profile_memories"] = [_serialize_profile_memory(record) for record in profile_memories]

    if include_event and query:
        matched_memories = search_long_term_memory(
            query=query,
            top_k=top_k,
            user_id=context.user_id,
            conversation_id=context.conversation_id,
            session_id=context.session_id,
        )
        result["matched_memories"] = [_serialize_retrieval_memory(record) for record in matched_memories]

    result["summary"] = {
        "profile_count": len(result["profile_memories"]),
        "matched_count": len(result["matched_memories"]),
    }
    return result


def register_builtin_skills() -> None:
    """注册当前阶段支持的本地技能。"""
    register_skill(
        SkillDefinition(
            name="document_hot_word_statistics",
            description="按文件名定位文档，统计对应文档内容中的高频热词，并返回热词、频次和示例文档。",
            category="document",
            kind="local",
            tags=("statistics", "documents", "keywords"),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                        "description": "返回前多少个热点词。",
                    },
                    "file_name": {
                        "type": "string",
                        "description": "可选，按原始文件名或存储文件名模糊定位目标文档。",
                    },
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选，只统计指定文档。",
                    },
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选，只统计指定文件类型。",
                    },
                    "statuses": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["processing", "ready", "failed", "deleting", "deleted"],
                        },
                        "description": "可选，只统计指定状态。",
                    },
                    "include_samples": {
                        "type": "boolean",
                        "default": True,
                        "description": "是否返回示例文档。",
                    },
                },
            },
            executor=_skill_document_hot_words,
            metadata={"returns": "hot_word_statistics"},
        )
    )
    register_skill(
        SkillDefinition(
            name="memory_query",
            description="查询用户的长期记忆或画像记忆，支持按文本召回和当前画像查看。",
            category="memory",
            kind="local",
            tags=("memory", "profile", "search"),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "可选，按语义召回长期记忆时使用。",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 5,
                        "description": "最多返回多少条记忆。",
                    },
                    "memory_scope": {
                        "type": "string",
                        "enum": ["all", "profile", "event"],
                        "default": "all",
                        "description": "记忆范围：画像记忆、事件记忆或两者都查。",
                    },
                },
            },
            executor=_skill_memory_query,
            metadata={"returns": "memory_query_result"},
        )
    )
