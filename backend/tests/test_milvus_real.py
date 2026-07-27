"""使用 MilvusClient 直接测试 doc_chunks 写入和查询。"""

import uuid

from pymilvus import MilvusClient


MILVUS_URI = "http://localhost:19530"
MILVUS_DATABASE = "rag_db"
COLLECTION_NAME = "doc_chunks"
VECTOR_DIMENSION = 768


def test_milvus():
    client = MilvusClient(
        uri=MILVUS_URI,
        db_name=MILVUS_DATABASE,
    )

    assert client.has_collection(COLLECTION_NAME)

    document_id = f"milvus-native-test-{uuid.uuid4().hex}"
    text = "这是一条通过 MilvusClient 原生接口写入的测试内容。"

    insert_result = client.insert(
        collection_name=COLLECTION_NAME,
        data=[
            {
                "text": text,
                "vector": [0.01] * VECTOR_DIMENSION,
                "document_id": document_id,
                "chunk_index": 0,
                "source_name": "test_milvus_real.py",
                "chunk_text": text,
                "file_type": "text/plain",
                "content_type": "text",
            }
        ],
    )
    print(f"写入结果: {insert_result}")

    client.flush(collection_name=COLLECTION_NAME)
    client.load_collection(collection_name=COLLECTION_NAME)

    query_result = client.query(
        collection_name=COLLECTION_NAME,
        filter=f'document_id == "{document_id}"',
        output_fields=[
            "pk",
            "text",
            "document_id",
            "chunk_index",
            "source_name",
            "chunk_text",
            "file_type",
            "content_type",
        ],
        limit=10,
    )
    print(f"查询结果: {query_result}")

    assert len(query_result) == 1
    assert query_result[0]["document_id"] == document_id
    assert query_result[0]["chunk_text"] == text
