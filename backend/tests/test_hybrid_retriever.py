"""混合检索器测试。"""

from app.rag.retrieval.base import RetrievalResult
from app.rag.retrieval.hybrid_retriever import HybridRetriever


class _FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query: str, top_k: int = None):
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_hybrid_retriever_fuses_dense_and_sparse_results():
    dense_results = [
        RetrievalResult(
            content="A",
            metadata={"document_id": "doc-1", "chunk_index": 0},
            score=0.9,
            source="dense",
        ),
        RetrievalResult(
            content="B",
            metadata={"document_id": "doc-1", "chunk_index": 1},
            score=0.8,
            source="dense",
        ),
    ]
    sparse_results = [
        RetrievalResult(
            content="B",
            metadata={"document_id": "doc-1", "chunk_index": 1},
            score=3.0,
            source="sparse",
        ),
        RetrievalResult(
            content="C",
            metadata={"document_id": "doc-2", "chunk_index": 0},
            score=2.0,
            source="sparse",
        ),
    ]

    retriever = HybridRetriever(
        dense_retriever=_FakeRetriever(dense_results),
        sparse_retriever=_FakeRetriever(sparse_results),
        top_k=2,
        dense_weight=0.6,
        sparse_weight=0.4,
    )

    results = retriever.retrieve("高考志愿", top_k=2)

    assert len(results) == 2
    assert results[0].content == "B"
    assert results[0].source == "hybrid"
    assert results[1].content == "A"
