"""初始化 Milvus 的 rag_db 以及文档/记忆两个 collection。"""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.config import settings  # noqa: E402
from app.storage.milvus_store import (  # noqa: E402
    _ensure_document_collection,
    _ensure_memory_collection,
    get_milvus_client,
)


def main() -> None:
    client = get_milvus_client()

    print(f"Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    print(f"Database: {settings.MILVUS_DB_NAME}")

    if settings.MILVUS_DB_NAME and settings.MILVUS_DB_NAME != "default":
        try:
            client.use_database(settings.MILVUS_DB_NAME)
            print(f"✓ 已切换到数据库: {settings.MILVUS_DB_NAME}")
        except Exception:
            client.create_database(settings.MILVUS_DB_NAME)
            client.use_database(settings.MILVUS_DB_NAME)
            print(f"✓ 已创建数据库: {settings.MILVUS_DB_NAME}")

    _ensure_document_collection(client)
    _ensure_memory_collection(client)

    print(f"✓ 已确保 collection: {settings.MILVUS_DOC_COLLECTION_NAME}")
    print(f"✓ 已确保 collection: {settings.MILVUS_MEMORY_COLLECTION_NAME}")


if __name__ == "__main__":
    main()
