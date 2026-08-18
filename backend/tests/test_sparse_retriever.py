"""稀疏检索器测试。"""

from typing import Any, Dict, List

from app.rag.retrieval.sparse_retriever import SparseRetriever
from app.rag.retrieval.sparse_retriever import _build_filter_terms


class _FakeClient:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: List[Dict[str, Any]] = []

    def has_collection(self, collection_name: str) -> bool:
        return True

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return self.rows


def test_sparse_retriever_ranks_by_keyword_overlap(monkeypatch):
    client = _FakeClient(
        rows=[
            {
                "text": "高考志愿填报和专业选择",
                "document_id": "doc-1",
                "chunk_index": 1,
                "source_name": "a.txt",
                "chunk_text": "高考志愿填报和专业选择",
                "file_type": "txt",
                "content_type": "text",
            },
            {
                "text": "志愿填报",
                "document_id": "doc-2",
                "chunk_index": 2,
                "source_name": "b.txt",
                "chunk_text": "志愿填报",
                "file_type": "txt",
                "content_type": "text",
            },
        ]
    )
    monkeypatch.setattr("app.rag.retrieval.sparse_retriever.is_collection_loaded", lambda *args, **kwargs: True)

    retriever = SparseRetriever(top_k=1, vector_store=client)
    results = retriever.retrieve("高考 志愿", top_k=1)

    assert len(results) == 1
    assert results[0].metadata["document_id"] == "doc-1"
    assert client.calls
    assert 'chunk_text like "%高考%"' in client.calls[0]["filter"]
    assert 'source_name like "%高考%"' in client.calls[0]["filter"]


def test_sparse_filter_terms_remove_question_noise_but_keep_structured_fields():
    terms = _build_filter_terms("公司每个部门的负责人是谁")

    assert terms[:2] == ["部门", "负责人"]
    assert "每个部" not in terms
    assert "部门负" in terms or "负责人" in terms


class _FakeDenseClient:
    def has_collection(self, collection_name: str) -> bool:
        return True

    def search(self, **kwargs):
        return [[
            {
                "distance": 0.42,
                "entity": {
                    "chunk_text": "低于默认阈值但可能相关的内容",
                    "document_id": "doc-1",
                    "chunk_index": 1,
                    "source_name": "data.xlsx",
                    "file_type": "xlsx",
                    "content_type": "text",
                },
            }
        ]]


def test_dense_candidate_mode_keeps_low_similarity_hit(monkeypatch):
    from app.rag.retrieval.dense_retriever import DenseRetriever

    monkeypatch.setattr(
        "app.rag.retrieval.dense_retriever.is_collection_loaded",
        lambda *args, **kwargs: True,
    )
    retriever = DenseRetriever(
        embeddings=type("_Embeddings", (), {"embed_query": lambda self, query: [0.1, 0.2]})(),
        top_k=10,
        vector_store=_FakeDenseClient(),
        enforce_similarity_threshold=False,
    )

    results = retriever.retrieve("相关问题", top_k=10)

    assert len(results) == 1
    assert results[0].metadata["chunk_index"] == 1
