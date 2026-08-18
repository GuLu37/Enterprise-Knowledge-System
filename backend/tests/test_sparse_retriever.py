"""稀疏检索器测试。"""

from typing import Any, Dict, List

from app.rag.retrieval.sparse_retriever import SparseRetriever


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
