"""Milvus 向量块存储"""
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pymilvus import DataType, MilvusClient

from app.config import settings
from app.utils.exceptions import VectorStoreException
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

_milvus_client: Optional[MilvusClient] = None

_VARCHAR_MAX_LENGTH = 65535
_DEFAULT_INSERT_BATCH_SIZE = 16
_DEFAULT_INSERT_RETRY_LIMIT = 6
_DEFAULT_INSERT_INITIAL_DELAY = 0.5
_DEFAULT_INSERT_MAX_DELAY = 5.0
_RATE_LIMIT_MARKERS = (
    "rate limit exceeded",
    "reach the limit of request",
    "please slowdown and retry later",
)


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


def _is_rate_limited_error(error: Exception) -> bool:
    """判断异常是否属于 Milvus 写入限流。"""
    message = str(error).lower()
    return any(marker in message for marker in _RATE_LIMIT_MARKERS)


def _iter_batches(rows: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    """把插入行按固定大小拆分。"""
    if batch_size <= 0:
        batch_size = len(rows) or 1
    for start in range(0, len(rows), batch_size):
        yield list(rows[start:start + batch_size])


def insert_rows_with_retry(
    client: MilvusClient,
    collection_name: str,
    rows: Sequence[Dict[str, Any]],
    batch_size: int = _DEFAULT_INSERT_BATCH_SIZE,
    max_retries: int = _DEFAULT_INSERT_RETRY_LIMIT,
    initial_delay: float = _DEFAULT_INSERT_INITIAL_DELAY,
    max_delay: float = _DEFAULT_INSERT_MAX_DELAY,
) -> List[str]:
    """分批写入 Milvus，并在限流时做指数退避重试。"""
    if not rows:
        return []

    inserted_ids: List[str] = []
    for batch_index, batch in enumerate(_iter_batches(rows, batch_size), start=1):
        attempt = 0
        while True:
            try:
                result = client.insert(
                    collection_name=collection_name,
                    data=batch,
                )
                inserted_ids.extend(str(item) for item in result.get("ids", []))
                break
            except Exception as error:
                if not _is_rate_limited_error(error) or attempt >= max_retries:
                    raise

                delay = min(initial_delay * (2 ** attempt), max_delay)
                logger.warning(
                    "Milvus 写入触发限流，准备重试 (collection={}, batch={}, attempt={}/{}, delay={:.2f}s): {}",
                    collection_name,
                    batch_index,
                    attempt + 1,
                    max_retries,
                    delay,
                    error,
                )
                time.sleep(delay)
                attempt += 1

    return inserted_ids


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


def _collection_field_names(client: MilvusClient, collection_name: str) -> set[str]:
    """读取 collection 当前字段名集合。"""
    description = client.describe_collection(collection_name)
    return {str(field.get("name")) for field in description.get("fields", []) if field.get("name")}


def is_collection_loaded(client: MilvusClient, collection_name: str) -> bool:
    """检查 collection 是否已经加载完成。"""
    try:
        load_state = client.get_load_state(collection_name=collection_name)
    except Exception:
        return False

    state = load_state.get("state") if isinstance(load_state, dict) else None
    state_name = getattr(state, "name", None)
    if state_name:
        return state_name.lower() == "loaded"
    return str(state).lower().endswith("loaded")


def warmup_collections(collection_names: Optional[Sequence[str]] = None) -> None:
    """在启动阶段一次性预热 Milvus collection。"""
    client = get_milvus_client()
    names = list(collection_names or (
        settings.MILVUS_DOC_COLLECTION_NAME,
        settings.MILVUS_MEMORY_COLLECTION_NAME,
    ))

    for collection_name in names:
        if not client.has_collection(collection_name):
            logger.warning(f"启动预热跳过不存在的 collection: {collection_name}")
            continue

        if is_collection_loaded(client, collection_name):
            logger.info(f"collection 已加载完成，跳过预热: {collection_name}")
            continue

        logger.info(f"开始预热 collection: {collection_name}")
        client.load_collection(
            collection_name=collection_name,
            replica_number=1,
            timeout=60,
        )
        logger.info(f"✓ collection 预热完成: {collection_name}")


def _create_document_collection(client: MilvusClient) -> None:
    """创建文档向量 collection。"""
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    _add_common_fields(schema, "document_id")
    schema.add_field(field_name="file_type", datatype=DataType.VARCHAR, max_length=_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=_VARCHAR_MAX_LENGTH)

    client.create_collection(
        collection_name=settings.MILVUS_DOC_COLLECTION_NAME,
        schema=schema,
        index_params=_create_index_params(),
    )
    logger.info(f"✓ Milvus 文档 collection 创建完成 ({settings.MILVUS_DOC_COLLECTION_NAME})")


def _create_memory_collection(client: MilvusClient) -> None:
    """创建长期记忆向量 collection。"""
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    _add_common_fields(schema, "memory_id")
    schema.add_field(field_name="conversation_id", datatype=DataType.VARCHAR, max_length=_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="session_id", datatype=DataType.VARCHAR, max_length=_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="chunk_type", datatype=DataType.VARCHAR, max_length=_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="topic", datatype=DataType.VARCHAR, max_length=_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="turn_start", datatype=DataType.INT64)
    schema.add_field(field_name="turn_end", datatype=DataType.INT64)
    schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=_VARCHAR_MAX_LENGTH)

    client.create_collection(
        collection_name=settings.MILVUS_MEMORY_COLLECTION_NAME,
        schema=schema,
        index_params=_create_index_params(),
    )
    logger.info(f"✓ Milvus 长期记忆 collection 创建完成 ({settings.MILVUS_MEMORY_COLLECTION_NAME})")


def _ensure_document_collection(client: MilvusClient) -> None:
    """确保文档向量 collection 存在，不存在时按当前 schema 创建；存在时确保索引完整。"""
    collection_name = settings.MILVUS_DOC_COLLECTION_NAME
    expected_fields = {
        "pk",
        "text",
        "vector",
        "document_id",
        "chunk_index",
        "source_name",
        "chunk_text",
        "file_type",
        "content_type",
    }
    if client.has_collection(collection_name):
        existing_fields = _collection_field_names(client, collection_name)
        if not expected_fields.issubset(existing_fields):
            logger.warning(f"⚠ collection {collection_name} schema 不匹配，准备重建...")
            client.drop_collection(collection_name=collection_name)
            _create_document_collection(client)
            return
        _ensure_index(client, collection_name)
        return

    _create_document_collection(client)


def _ensure_memory_collection(client: MilvusClient) -> None:
    """确保长期记忆向量 collection 存在，不存在时按当前 schema 创建。"""
    collection_name = settings.MILVUS_MEMORY_COLLECTION_NAME
    expected_fields = {
        "pk",
        "text",
        "vector",
        "memory_id",
        "chunk_index",
        "source_name",
        "chunk_text",
        "conversation_id",
        "session_id",
        "chunk_type",
        "topic",
        "turn_start",
        "turn_end",
        "created_at",
    }
    if client.has_collection(collection_name):
        existing_fields = _collection_field_names(client, collection_name)
        if not expected_fields.issubset(existing_fields):
            logger.warning(f"⚠ collection {collection_name} schema 不匹配，准备重建...")
            client.drop_collection(collection_name=collection_name)
            _create_memory_collection(client)
            return
        _ensure_index(client, collection_name)
        return

    _create_memory_collection(client)


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
