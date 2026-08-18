"""稀疏检索器 (BM25)"""
import math
import re
from typing import Any, Dict, List, Optional

from .base import BaseRetriever, RetrievalResult
from app.config import settings
from app.storage.milvus_store import get_milvus_client, is_collection_loaded
from app.rag.retrieval.reranker import (
    extract_identifier_terms,
    normalize_text,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover

    class BM25Okapi:  # type: ignore
        def __init__(self, corpus, k1: float = 1.5, b: float = 0.75):
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.doc_len = [len(doc) for doc in corpus]
            self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
            self.doc_freqs = []
            self.df = {}
            for doc in corpus:
                freq: Dict[str, int] = {}
                for token in doc:
                    freq[token] = freq.get(token, 0) + 1
                self.doc_freqs.append(freq)
                for token in freq:
                    self.df[token] = self.df.get(token, 0) + 1
            self.doc_count = len(corpus)

        def get_scores(self, query_tokens):
            scores = []
            for idx, freq in enumerate(self.doc_freqs):
                score = 0.0
                dl = self.doc_len[idx] or 1
                for token in query_tokens:
                    tf = freq.get(token, 0)
                    if not tf:
                        continue
                    df = self.df.get(token, 0)
                    idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
                    denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                    score += idf * (tf * (self.k1 + 1)) / denom
                scores.append(score)
            return scores


def _escape_like_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_searchable_row_text(row: Dict[str, Any]) -> str:
    """组合文件名与切片正文，让文件名也参与 BM25 排序。"""
    source_name = str(row.get("source_name") or "")
    content = str(row.get("chunk_text") or row.get("text") or "")
    return "\n".join(part for part in (source_name, content) if part)


def _build_filter_expression(filter_terms: List[str]) -> str:
    """同时在正文和来源文件名中筛选稀疏检索候选。"""
    predicates = []
    for term in filter_terms[:12]:
        if not term:
            continue
        escaped_term = _escape_like_value(term)
        predicates.append(
            f'(chunk_text like "%{escaped_term}%" or source_name like "%{escaped_term}%")'
        )
    return " or ".join(predicates)


def _build_filter_terms(query: str, max_terms: int = 12) -> List[str]:
    """生成低噪声候选过滤词，保留编号、语义片段和必要短 ngram。"""
    query_terms = extract_identifier_terms(query)
    normalized_query = (query or "").strip().lower()
    normalized_query = re.sub(
        r"(根据|基于|结合|请问|麻烦|帮我|帮忙|查询|检索|搜索|查一下|查找|"
        r"公司|企业|每个|各个|所有|全部|分别|哪些|哪几|是谁|是什么|在哪里|在哪|"
        r"多少|如何|怎么|为什么|是否|有没有|可以吗|能否|吗|呢|吧|啊)",
        " ",
        normalized_query,
    )
    normalized_query = re.sub(r"[的之与和及在是有为从到对把将]", " ", normalized_query)
    semantic_parts = re.findall(r"[A-Za-z0-9]+|[一-鿿]{2,}", normalized_query)
    for part in semantic_parts:
        if part:
            query_terms.append(part)
        if re.fullmatch(r"[一-鿿]{4,}", part):
            query_terms.extend(
                part[index:index + 3]
                for index in range(len(part) - 2)
                if not any(char in "的之与和及在是有为从到对把将" for char in part[index:index + 3])
            )

    if not query_terms:
        return []

    filter_terms: List[str] = []
    seen = set()
    for term in query_terms:
        normalized_term = term.strip()
        if len(normalized_term) < 2:
            continue
        if normalized_term in seen:
            continue
        seen.add(normalized_term)
        filter_terms.append(normalized_term)
        if len(filter_terms) >= max_terms:
            break

    if filter_terms:
        return filter_terms

    fallback_terms: List[str] = []
    for term in query_terms:
        normalized_term = term.strip()
        if not normalized_term or normalized_term in seen:
            continue
        seen.add(normalized_term)
        fallback_terms.append(normalized_term)
        if len(fallback_terms) >= max_terms:
            break
    return fallback_terms


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for part in re.findall(r"[A-Za-z0-9]+|[一-鿿]+", (text or "").lower()):
        if re.fullmatch(r"[一-鿿]+", part):
            if len(part) <= 2:
                tokens.append(part)
                continue
            tokens.append(part)
            tokens.extend(part[index:index + 2] for index in range(len(part) - 1))
            if len(part) >= 4:
                tokens.extend(part[index:index + 3] for index in range(len(part) - 2))
        else:
            tokens.append(part)
    return tokens


def _lexical_match_score(query_tokens: List[str], row_tokens: List[str], query: str, row_text: str) -> float:
    """BM25 可能出现负分，用词覆盖分保证明确关键词命中不会被丢弃。"""
    unique_query_tokens = []
    seen = set()
    for token in query_tokens:
        if token and token not in seen:
            seen.add(token)
            unique_query_tokens.append(token)
    if not unique_query_tokens:
        return 0.0

    row_token_set = set(row_tokens)
    max_weight = 0.0
    hit_weight = 0.0
    for token in unique_query_tokens:
        weight = min(max(len(token), 1), 8) / 8.0
        max_weight += weight
        if token in row_token_set:
            hit_weight += weight

    score = hit_weight / max_weight if max_weight > 0.0 else 0.0
    compact_query = normalize_text(query)
    compact_row = normalize_text(row_text)
    if compact_query and compact_query in compact_row:
        score += 0.35
    return score


class SparseRetriever(BaseRetriever):
    """最简单的 BM25 稀疏检索。"""

    def __init__(self, top_k: int = 5, vector_store: Optional[Any] = None):
        super().__init__(top_k)
        self.vector_store = vector_store
        self._collection_ready: Optional[bool] = None

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        try:
            if top_k is None:
                top_k = self.top_k

            query = (query or "").strip()
            if not query:
                return []

            client = self.vector_store or get_milvus_client()
            collection_name = settings.MILVUS_DOC_COLLECTION_NAME
            if self._collection_ready is None:
                self._collection_ready = (
                    client.has_collection(collection_name)
                    and is_collection_loaded(client, collection_name)
                )
                if not self._collection_ready:
                    logger.warning(f"collection {collection_name} 尚未加载完成，跳过本次稀疏检索")
            if not self._collection_ready:
                return []

            filter_terms = _build_filter_terms(query)
            if not filter_terms:
                return []

            filter_expr = _build_filter_expression(filter_terms)
            if not filter_expr:
                return []

            rows = client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=[
                    "text",
                    "document_id",
                    "chunk_index",
                    "source_name",
                    "chunk_text",
                    "file_type",
                    "content_type",
                ],
                limit=max(top_k * 30, 80),
            )

            if not rows and len(filter_terms) > 1:
                fallback_filter = _build_filter_expression(filter_terms[:4])
                if fallback_filter and fallback_filter != filter_expr:
                    rows = client.query(
                        collection_name=collection_name,
                        filter=fallback_filter,
                        output_fields=[
                            "text",
                            "document_id",
                            "chunk_index",
                            "source_name",
                            "chunk_text",
                            "file_type",
                            "content_type",
                        ],
                        limit=max(top_k * 30, 80),
                    )

            if not rows:
                return []

            corpus = [_tokenize(_build_searchable_row_text(row)) for row in rows]
            bm25 = BM25Okapi(corpus)
            query_tokens = _tokenize(query)
            bm25_scores = bm25.get_scores(query_tokens)

            ranked_rows = []
            for bm25_score, row, row_tokens in zip(bm25_scores, rows, corpus):
                row_text = _build_searchable_row_text(row)
                lexical_score = _lexical_match_score(query_tokens, row_tokens, query, row_text)
                raw_bm25_score = float(bm25_score)
                final_score = max(0.0, raw_bm25_score) + lexical_score
                if final_score > 0.0:
                    ranked_rows.append((final_score, row))
            ranked_rows.sort(
                key=lambda item: (
                    -item[0],
                    item[1].get("chunk_index", 0),
                )
            )

            results: List[RetrievalResult] = []
            for score, row in ranked_rows[:top_k]:
                results.append(
                    RetrievalResult(
                        content=str(row.get("chunk_text") or row.get("text") or ""),
                        metadata={
                            "document_id": row.get("document_id"),
                            "chunk_index": row.get("chunk_index"),
                            "source_name": row.get("source_name"),
                            "file_type": row.get("file_type"),
                            "content_type": row.get("content_type"),
                        },
                        score=score,
                        source="sparse",
                    )
                )

            if not results:
                return []

            return results[:top_k]
        except Exception as e:
            logger.error(f"稀疏检索失败: {str(e)}")
            raise
