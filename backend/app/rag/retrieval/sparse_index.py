"""进程内持久稀疏索引（BM25）。

动机: 旧路径每次查询都要 1) 对 Milvus 发 12 个前缀 LIKE 的 OR 全表扫，
2) 拉回最多 max(top_k*30, 80) 行，3) 在 Python 里对每行做 CJK n-gram 分词，
4) 从零构建 BM25Okapi 索引 —— 且 3 个 Multi-Query 变体并行重复此过程。

本模块把 child 行一次性载入进程内，分词与 BM25 统计量只算一次；
查询直接进程内打分取 top-N，文档增删后由失效钩子触发下次懒重建。
超过行数上限或构建失败时自动回退旧 LIKE 路径，保证功能不中断。
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

_SPARSE_OUTPUT_FIELDS = [
    "child_id",
    "document_id",
    "parent_id",
    "parent_index",
    "chunk_index",
    "source_name",
    "file_type",
    "content_type",
    "child_text",
]


def _tokenize(text: str) -> List[str]:
    """与 sparse_retriever._tokenize 保持一致的 CJK 2/3-gram 切分。"""
    import re

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


class _IndexSnapshot:
    """不可变快照: 构建完成后整体替换引用，读路径完全无锁。"""

    __slots__ = ("rows", "tokens", "bm25", "built_at")

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokens: List[List[str]],
        bm25: Any,
        built_at: float,
    ):
        self.rows = rows
        self.tokens = tokens
        self.bm25 = bm25
        self.built_at = built_at


class CachedSparseIndex:
    """全量 child 行的进程内 BM25 索引，懒构建 + 版本失效。"""

    def __init__(self) -> None:
        self._snapshot: Optional[_IndexSnapshot] = None
        self._version = 0
        self._build_lock = threading.Lock()
        self._building = False

    # ---------- 生命周期 ----------

    def invalidate(self) -> None:
        """文档增删后调用: 清空快照，下次检索时懒重建。"""
        self._version += 1
        self._snapshot = None
        logger.info("稀疏持久索引已失效，将在下次检索时重建")

    def warm_build(self, client: Any, collection_name: str) -> bool:
        """启动预热时后台构建; 返回是否成功建立。"""
        try:
            snapshot = self._ensure_snapshot(client, collection_name)
            if snapshot is None:
                return False
            logger.info(
                "稀疏持久索引预热完成: rows=%d elapsed=%.2fs",
                len(snapshot.rows),
                time.time() - snapshot.built_at,
            )
            return True
        except Exception as exc:
            logger.warning("稀疏持久索引预热失败，检索将回退 LIKE 路径: %s", exc)
            return False

    # ---------- 查询 ----------

    def search(
        self,
        client: Any,
        collection_name: str,
        query_tokens: Sequence[str],
        limit: int,
    ) -> Optional[List[Tuple[Dict[str, Any], float, List[str]]]]:
        """返回 (行, 切片内 BM25 分, 行 token) top-N; 索引不可用时返回 None。

        两段式设计: 全语料 BM25 只负责候选选择（等价旧路径的 LIKE 预筛，
        但无全表扫）；选出切片后在切片内重建 BM25 重打分，与旧路径的
        子集统计语义保持一致，避免 IDF 口径变化引起排序回归。
        """
        snapshot = self._ensure_snapshot(client, collection_name)
        if snapshot is None:
            return None
        if not snapshot.rows:
            return []

        global_scores = snapshot.bm25.get_scores(list(query_tokens))
        order = sorted(range(len(global_scores)), key=lambda index: -global_scores[index])
        slice_indexes: List[int] = []
        for row_index in order[: max(1, limit)]:
            if global_scores[row_index] <= 0.0:
                break
            slice_indexes.append(row_index)
        if not slice_indexes:
            return []

        from rank_bm25 import BM25Okapi

        slice_bm25 = BM25Okapi([snapshot.tokens[index] for index in slice_indexes])
        slice_scores = slice_bm25.get_scores(list(query_tokens))
        return [
            (snapshot.rows[index], float(slice_scores[position]), snapshot.tokens[index])
            for position, index in enumerate(slice_indexes)
        ]

    # ---------- 内部 ----------

    def _ensure_snapshot(
        self, client: Any, collection_name: str
    ) -> Optional[_IndexSnapshot]:
        snapshot = self._snapshot
        if snapshot is not None:
            return snapshot
        if self._building:
            # 其他线程正在重建时不重复构建，直接回退旧路径。
            return None
        with self._build_lock:
            snapshot = self._snapshot
            if snapshot is not None:
                return snapshot
            self._building = True
            try:
                snapshot = self._build_snapshot(client, collection_name)
                if snapshot is not None:
                    self._snapshot = snapshot
                return snapshot
            finally:
                self._building = False

    def _build_snapshot(
        self, client: Any, collection_name: str
    ) -> Optional[_IndexSnapshot]:
        from rank_bm25 import BM25Okapi

        max_rows = max(1, int(settings.SPARSE_INDEX_MAX_ROWS))
        started_at = time.time()
        rows = _fetch_all_rows(client, collection_name, max_rows + 1)
        if len(rows) > max_rows:
            logger.warning(
                "child 行数 %d 超过稀疏持久索引上限 %d，继续使用 LIKE 路径",
                len(rows),
                max_rows,
            )
            return None

        corpus = []
        for row in rows:
            source_name = str(row.get("source_name") or "")
            content = str(row.get("child_text") or "")
            corpus.append(_tokenize("\n".join(part for part in (source_name, content) if part)))
        bm25 = BM25Okapi(corpus) if corpus else None
        logger.info(
            "稀疏持久索引构建完成: rows=%d elapsed=%.2fs",
            len(rows),
            time.time() - started_at,
        )
        return _IndexSnapshot(
            rows=rows,
            tokens=corpus,
            bm25=bm25,
            built_at=time.time(),
        )


def _fetch_all_rows(
    client: Any,
    collection_name: str,
    hard_limit: int,
    batch_size: int = 2000,
) -> List[Dict[str, Any]]:
    """无过滤分页拉取全部 child 行（比前缀 LIKE 全表扫便宜得多）。"""
    all_rows: List[Dict[str, Any]] = []
    offset = 0
    while len(all_rows) < hard_limit:
        batch = client.query(
            collection_name=collection_name,
            filter='child_id != ""',
            output_fields=_SPARSE_OUTPUT_FIELDS,
            limit=batch_size,
            offset=offset,
        )
        all_rows.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size
    return all_rows[:hard_limit]


_INDEX = CachedSparseIndex()


def get_sparse_index() -> CachedSparseIndex:
    """全局稀疏持久索引单例。"""
    return _INDEX


def invalidate_sparse_index() -> None:
    """文档入库/删除后的失效钩子。"""
    _INDEX.invalidate()


def warm_build_sparse_index() -> bool:
    """启动预热入口: 供 main.py 的 Milvus 预热线程调用。"""
    if (settings.SPARSE_INDEX_MODE or "cached").strip().lower() != "cached":
        return False
    if not settings.USE_SPARSE_RETRIEVER:
        return False
    try:
        from app.storage.milvus_store import (
            get_child_collection_name,
            get_milvus_client,
        )

        client = get_milvus_client()
        collection_name = get_child_collection_name(client)
        return _INDEX.warm_build(client, collection_name)
    except Exception as exc:
        logger.warning("稀疏持久索引预热异常: %s", exc)
        return False
