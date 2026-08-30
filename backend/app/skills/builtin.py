"""内置 Skill：热点词统计、记忆查询。"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable as IterableABC
from datetime import date, datetime
from typing import Any, Dict, List, Sequence

from sqlalchemy import or_

from app.rag.retrieval.reranker import normalize_text
from app.services.memory_service import (
    delete_profile_memories,
    get_active_profile_memories,
    search_long_term_memory,
)
from app.skills.registry import SkillDefinition, SkillExecutionContext, register_skill
from app.storage.sqlite_metadata import DocumentChunkRecord, DocumentRecord, SessionLocal

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,}")
_NOISY_TERMS = {
    "文档",
    "文件",
    "章节",
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


def _chinese_ngrams(token: str) -> List[str]:
    """中文串按字符切分生成二元/三元 gram。

    中文没有空格分隔，若直接把整段连续汉字当成一个词，会丢失『密码』『账号』
    这类有意义的短词，且整句作为 term 对关键词统计毫无价值。改为切分 2/3 元
    gram 是无分词模型时最稳妥的轻量关键词提取方式。
    """
    grams: List[str] = []
    for n in (2, 3):
        for start in range(len(token) - n + 1):
            grams.append(token[start : start + n])
    return grams


def _extract_term_frequencies_from_text(text: str) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    if not text:
        return frequencies

    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        # 纯英文/数字词按整词提取；中文串走 n-gram，避免整句被当成一个无用语。
        if re.fullmatch(r"[A-Za-z0-9]+", token):
            cleaned = normalize_text(token)
            if _is_noisy_term(cleaned):
                continue
            frequencies[cleaned] += 1
        else:
            for gram in _chinese_ngrams(token):
                cleaned = normalize_text(gram)
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


def _skill_delete_profile_memory(context: SkillExecutionContext, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """按 memory_key 或关键词删除用户个性化记忆。"""
    memory_keys = _normalize_string_list(arguments.get("memory_keys"))
    keywords = _normalize_string_list(arguments.get("keywords"))
    max_deletes = _coerce_int(arguments.get("max_deletes", 50), default=50, minimum=1, maximum=200)

    if not memory_keys and not keywords:
        return {
            "deleted_count": 0,
            "matched": [],
            "summary": "未提供删除条件，请指定 memory_keys 或 keywords。",
        }

    result = delete_profile_memories(
        user_id=context.user_id,
        memory_keys=memory_keys,
        keywords=keywords,
        max_deletes=max_deletes,
    )
    return {
        "deleted_count": int(result.get("deleted_count") or 0),
        "matched": result.get("matched", []),
        "summary": str(result.get("summary") or ""),
    }


def _skill_list_knowledge_base_files(context: SkillExecutionContext, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """列出知识库已上传文档，并可概括其内容（章节大纲、内容预览、主题关键词）。"""
    document_ids = _normalize_string_list(arguments.get("document_ids"))
    file_types = _normalize_string_list(arguments.get("file_types"))
    statuses = _normalize_string_list(arguments.get("statuses") or arguments.get("status"))
    file_name = str(arguments.get("file_name") or arguments.get("filename") or "").strip()
    summarize = bool(arguments.get("summarize", True))
    preview_chars = _coerce_int(arguments.get("preview_chars", 1500), default=1500, minimum=200, maximum=6000)
    limit = _coerce_int(arguments.get("limit", 20), default=20, minimum=1, maximum=100)

    # 默认只列已入库(ready)的文档，符合「已上传的文件」语义；可按 statuses 覆盖。
    if not statuses:
        statuses = ["ready"]

    db = SessionLocal()
    try:
        docs_query = db.query(DocumentRecord)
        docs_query = _apply_document_filters(docs_query, document_ids, file_types, statuses, file_name)
        documents = docs_query.order_by(DocumentRecord.original_filename).limit(limit).all()
        filters = {
            "document_ids": document_ids,
            "file_types": file_types,
            "statuses": statuses,
            "file_name": file_name,
            "limit": limit,
        }
        if not documents:
            return {
                "summary": {
                    "document_count": 0,
                    "message": "未找到匹配的已上传文档，请确认文件名或状态。",
                },
                "documents": [],
                "filters": filters,
            }

        doc_map = {record.document_id: record for record in documents}
        chunks_query = (
            db.query(
                DocumentChunkRecord.document_id,
                DocumentChunkRecord.chunk_text,
                DocumentChunkRecord.chunk_index,
            )
            .filter(DocumentChunkRecord.document_id.in_(list(doc_map.keys())))
            .order_by(DocumentChunkRecord.document_id, DocumentChunkRecord.chunk_index)
        )
        chunks_by_document: Dict[str, List[str]] = defaultdict(list)
        for document_id, chunk_text, _chunk_index in chunks_query.all():
            text = str(chunk_text or "").strip()
            if text:
                chunks_by_document[str(document_id)].append(text)

        documents_out: List[Dict[str, Any]] = []
        for record in documents:
            doc = _serialize_document_record(record)
            if summarize:
                chunks = chunks_by_document.get(record.document_id, [])
                combined = "\n".join(chunks)
                outline = [
                    line.strip()[len("[章节]"):].strip(" :")
                    for line in combined.splitlines()
                    if line.strip().startswith("[章节]")
                ]
                preview = combined[:preview_chars]
                keywords = sorted(
                    _extract_term_frequencies_from_text(combined).items(),
                    key=lambda item: (-item[1], item[0]),
                )[:10]
                doc["outline"] = outline
                doc["content_preview"] = preview
                doc["topic_keywords"] = [term for term, _ in keywords]
            documents_out.append(doc)

        return {
            "summary": {
                "document_count": len(documents_out),
                "chunk_count": sum(int(record.chunk_count or 0) for record in documents),
            },
            "documents": _make_json_safe(documents_out),
            "filters": filters,
        }
    finally:
        db.close()


def register_builtin_skills() -> None:
    """注册当前阶段支持的本地技能。"""
    register_skill(
        SkillDefinition(
            name="document_hot_word_statistics",
            description=(
                "按文件名或文档 ID 定位文档并统计高频热词。适用于用户询问具体文件中的热词、"
                "关键词或出现频率；不用于回答文档正文、制度流程或内部事实。"
            ),
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
                        "description": "可选，按原始文件名或存储文件名模糊定位目标文档；用户提到文件名时优先填写。",
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
            description=(
                "查询当前用户自己的长期记忆或画像记忆。适用于‘你记得什么’、‘我之前说过什么’、"
                "‘我的偏好是什么’等问题；不用于查询企业文档，也不能跨用户读取记忆。"
            ),
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
    register_skill(
        SkillDefinition(
            name="delete_profile_memory",
            description=(
                "删除用户的个性化记忆（画像记忆）。当用户明确要求删除、遗忘、清除"
                "某条个人信息（如名字、偏好、地点、习惯等）时使用。可指定 memory_key "
                "精确删除，或提供 keywords 按内容模糊匹配删除；用户表达模糊时先查询并确认，"
                "不能删除事件记忆或其他用户数据。"
            ),
            category="memory",
            kind="local",
            tags=("memory", "profile", "delete", "privacy"),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "memory_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选，精确删除的 memory_key 列表，如 profile.location、profile.name。",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选，按内容关键词模糊匹配要删除的记忆，如 ['北京']。",
                    },
                    "max_deletes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "default": 50,
                        "description": "单次最多删除条数，超过会拒绝并要求更精确条件。",
                    },
                },
            },
            executor=_skill_delete_profile_memory,
            metadata={"returns": "delete_profile_memory_result"},
        )
    )
    register_skill(
        SkillDefinition(
            name="list_knowledge_base_files",
            description=(
                "列出知识库中已上传的文档，可按文件名、文档 ID、文件类型或状态筛选；"
                "并可概括指定文档的内容（章节大纲、内容预览、主题关键词）。"
                "适用于『知识库里有哪些文件』『帮我看看 04_产品与账号常见问题 这个文档讲了什么』"
                "『列出全部已上传文档』『这份文档包含哪些章节』等需求；不用于按内容检索具体答案，"
                "那应使用 search_knowledge_base。"
            ),
            category="document",
            kind="local",
            tags=("documents", "list", "summary", "overview"),
            parameters={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "可选，按文件名模糊筛选；用户提到具体文件名时填写，如『产品与账号常见问题』。",
                    },
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选，只列出指定 document_id 的文档。",
                    },
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选，只列出指定文件类型，如 ['pdf','docx','md']。",
                    },
                    "statuses": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["processing", "ready", "failed", "deleting", "deleted"],
                        },
                        "description": "可选，按状态筛选，默认只列已入库(ready)的文档。",
                    },
                    "summarize": {
                        "type": "boolean",
                        "default": True,
                        "description": "是否概括内容（章节大纲、内容预览、主题关键词）。",
                    },
                    "preview_chars": {
                        "type": "integer",
                        "minimum": 200,
                        "maximum": 6000,
                        "default": 1500,
                        "description": "每个文档内容预览的最大字符数。",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20,
                        "description": "最多返回多少条文档。",
                    },
                },
            },
            executor=_skill_list_knowledge_base_files,
            metadata={"returns": "knowledge_base_files"},
        )
    )
