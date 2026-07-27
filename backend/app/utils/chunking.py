"""文本切块工具。"""
from functools import lru_cache
from typing import List

from app.core.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


@lru_cache(maxsize=1)
def _get_document_text_splitter(chunk_size: int, chunk_overlap: int):
    """获取文档切块器。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )


def split_document_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """把文档长文本切成适合入库的 chunk 列表。"""
    chunks = _get_document_text_splitter(chunk_size, chunk_overlap).split_text(text or "")
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def split_conversation_chunks(conversation_text: str) -> List[str]:
    """对话 chunk 切块占位符，后续按对话轮次规则实现。"""
    return []
