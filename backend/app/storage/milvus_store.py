"""Milvus 向量块存储"""
from typing import Any, Dict, List, Optional

from pymilvus import DataType, MilvusClient

from app.config import settings
from app.utils.exceptions import VectorStoreException
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

_milvus_client: Optional[MilvusClient] = None

_VARCHAR_MAX_LENGTH = 65535


def _build_connection_args() -> Dict[str, Any]:
    """生成 MilvusClient 连接参数。"""
    args: Dict[str, Any] = {
        "uri": f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
    }
    if settings.MILVUS_DB_NAME:
        args["db_name"] = settings.MILVUS_DB_NAME
    return args


def get_milvus_client() -> MilvusClient:
    """获取 MilvusClient 单例，避免每次写入都重复创建连接。"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(**_build_connection_args())
        if settings.MILVUS_DB_NAME and settings.MILVUS_DB_NAME != "default":
            try:
                _milvus_client.use_database(settings.MILVUS_DB_NAME)
            except Exception:
                _milvus_client.create_database(settings.MILVUS_DB_NAME)
                _milvus_client.use_database(settings.MILVUS_DB_NAME)
    return _milvus_client


def _add_common_fields(schema, id_field: str) -> None:
    """给 collection schema 添加文本、向量和业务 ID 等通用字段。"""
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=_VARCHAR_MAX_LENGTH)
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=settings.EMBEDDING_DIMENSION,
    )
    schema.add_field(field_name=id_field, datatype=DataType.VARCHAR, max_length=_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
    schema.add_field(field_name="source_name", datatype=DataType.VARCHAR, max_length=_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="chunk_text", datatype=DataType.VARCHAR, max_length=_VARCHAR_MAX_LENGTH)


def _create_index_params():
    """创建向量索引参数，使用 COSINE 距离匹配归一化后的 BGE 向量。"""
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    return index_params


def _ensure_index(client: MilvusClient, collection_name: str) -> None:
    """若 collection 缺少向量索引则补建，避免 load_collection 报 index not found。"""
    indexes = client.list_indexes(collection_name)
    if not indexes:
        logger.warning(f"⚠ collection {collection_name} 缺少索引，正在补建...")
        client.create_index(
            collection_name=collection_name,
            index_params=_create_index_params(),
        )
        logger.info(f"✓ collection {collection_name} 索引补建完成")


def _ensure_document_collection(client: MilvusClient) -> None:
    """确保文档向量 collection 存在，不存在时按当前 schema 创建；存在时确保索引完整。"""
    collection_name = settings.MILVUS_DOC_COLLECTION_NAME
    if client.has_collection(collection_name):
        _ensure_index(client, collection_name)
        return

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    _add_common_fields(schema, "document_id")
    schema.add_field(field_name="file_type", datatype=DataType.VARCHAR, max_length=_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=_VARCHAR_MAX_LENGTH)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=_create_index_params(),
    )
    logger.info(f"✓ Milvus 文档 collection 创建完成 ({collection_name})")


def _ensure_memory_collection(client: MilvusClient) -> None:
    """确保长期记忆向量 collection 存在，不存在时按当前 schema 创建。"""
    collection_name = settings.MILVUS_MEMORY_COLLECTION_NAME
    if client.has_collection(collection_name):
        return

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    _add_common_fields(schema, "memory_id")

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=_create_index_params(),
    )
    logger.info(f"✓ Milvus 长期记忆 collection 创建完成 ({collection_name})")


def _validate_vector_dimension(client: MilvusClient, collection_name: str) -> None:
    """校验 collection 向量维度，避免 BGE 输出维度和 Milvus schema 不一致。"""
    description = client.describe_collection(collection_name)
    for field in description.get("fields", []):
        if field.get("name") == "vector":
            dimension = int(field.get("params", {}).get("dim", 0))
            if dimension != settings.EMBEDDING_DIMENSION:
                raise VectorStoreException(
                    f"Milvus collection 维度不匹配: {collection_name} "
                    f"当前为 {dimension}, 配置为 {settings.EMBEDDING_DIMENSION}"
                )
            return
