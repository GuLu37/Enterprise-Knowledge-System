"""文档入库与管理服务"""
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, UploadFile
from pymilvus import MilvusClient
from sqlalchemy.orm import Session

from app.config import settings
from app.storage.sqlite_metadata import DocumentRecord, SessionLocal, init_metadata_db
from app.storage.milvus_store import is_collection_loaded
from app.utils.chunking import split_document_text
from app.utils.exceptions import DocumentException
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def _normalize_allowed_types() -> set[str]:
    """把配置里的允许文件类型清洗成可比较的集合。"""
    return {item.strip().lower().strip('"') for item in settings.ALLOWED_FILE_TYPES}


def _save_uploaded_file(file: UploadFile, document_id: str, original_name: str) -> Path:
    """保存上传文件到本地目录，并返回落盘后的路径。"""
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{document_id}{Path(original_name).suffix.lower()}"
    stored_path = upload_dir / stored_filename

    file.file.seek(0)
    with stored_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return stored_path


def _get_file_size(file_path: Path) -> int:
    """获取文件大小，便于做上传限制校验。"""
    return file_path.stat().st_size


def _extract_text_from_txt(file_path: Path) -> str:
    """从 TXT/MD 文件中提取文本，按常见编码做容错读取。"""
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentException("TXT 文件编码无法识别")


def _extract_text_from_pdf(file_path: Path) -> str:
    """从 PDF 文件中抽取每一页的文本。"""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except Exception as e:
        logger.warning(f"pypdf 解析失败，尝试 pypdfium2 兜底: {e}")

    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(file_path))
        pages = []
        for page in pdf:
            textpage = page.get_textpage()
            pages.append(textpage.get_text_range() or "")
            textpage.close()
            page.close()
        pdf.close()
        return "\n".join(pages)
    except Exception as e:
        logger.warning(f"pypdfium2 解析失败，尝试 unstructured 兜底: {e}")
        return _extract_text_with_unstructured(file_path)


def _extract_text_from_docx(file_path: Path) -> str:
    """从 DOCX 文件中抽取段落文本。"""
    from docx import Document as DocxDocument

    doc = DocxDocument(str(file_path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())


def _extract_text_from_xlsx(file_path: Path) -> str:
    """从 Excel 文件中抽取工作表内容，转成可切块的纯文本。"""
    import pandas as pd

    sheets = pd.read_excel(str(file_path), sheet_name=None)
    blocks = []
    for sheet_name, df in sheets.items():
        blocks.append(f"[Sheet] {sheet_name}")
        blocks.append(df.fillna("").to_csv(index=False))
    return "\n".join(blocks)


def _extract_text_from_pptx(file_path: Path) -> str:
    """从 PPTX 文件中抽取每页幻灯片文本。"""
    from pptx import Presentation

    presentation = Presentation(str(file_path))
    blocks = []
    for slide in presentation.slides:
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text.append(shape.text.strip())
        if slide_text:
            blocks.append("\n".join(slide_text))
    return "\n\n".join(blocks)


def _extract_text_with_unstructured(file_path: Path) -> str:
    """用 unstructured 作为兜底解析器，覆盖更多文件格式。"""
    try:
        from unstructured.partition.auto import partition

        elements = partition(filename=str(file_path))
        return "\n".join(str(element).strip() for element in elements if str(element).strip())
    except Exception as e:
        raise DocumentException(f"文档解析失败: {e}")


def extract_text_from_file(file_path: Path) -> str:
    """根据文件后缀选择对应解析器，提取可向量化的文本内容。"""
    suffix = file_path.suffix.lower().lstrip(".")

    try:
        if suffix in {"txt", "md"}:
            return _extract_text_from_txt(file_path)
        if suffix == "pdf":
            return _extract_text_from_pdf(file_path)
        if suffix == "docx":
            return _extract_text_from_docx(file_path)
        if suffix in {"xlsx", "xls"}:
            return _extract_text_from_xlsx(file_path)
        if suffix == "pptx":
            return _extract_text_from_pptx(file_path)
        return _extract_text_with_unstructured(file_path)
    except DocumentException:
        raise
    except Exception as e:
        logger.error(f"提取文本失败: {str(e)}")
        raise DocumentException(f"提取文本失败: {str(e)}")


def _serialize_record(record: DocumentRecord) -> Dict:
    """把 ORM 记录转成接口可直接返回的字典。"""
    return {
        "document_id": record.document_id,
        "original_filename": record.original_filename,
        "stored_filename": record.stored_filename,
        "file_path": record.file_path,
        "content_type": record.content_type,
        "file_type": record.file_type,
        "file_size": record.file_size,
        "status": record.status,
        "chunk_count": record.chunk_count,
        "error_message": record.error_message,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
    }


def _update_document_record(document_id: str, **fields) -> None:
    """更新文档元数据中的状态字段，后台任务里复用。"""
    db: Session = SessionLocal()
    try:
        record = db.query(DocumentRecord).filter(DocumentRecord.document_id == document_id).first()
        if not record:
            return

        for key, value in fields.items():
            setattr(record, key, value)
        db.commit()
    finally:
        db.close()


def _insert_document_chunks_to_milvus(
    document_id: str,
    chunks: List[str],
    source_name: str,
    file_type: str,
    content_type: Optional[str],
) -> List[str]:
    """使用 MilvusClient 原生写入文档切块。"""
    from app.core.embeddings import get_default_embeddings

    logger.info(f"开始初始化 BGE Embedding (document_id={document_id}, chunks={len(chunks)})")
    embeddings = get_default_embeddings()
    logger.info(f"BGE Embedding 就绪，开始向量化 (document_id={document_id}, chunks={len(chunks)})")
    vectors = embeddings.embed_documents(chunks)
    logger.info(f"BGE 向量化完成 (document_id={document_id}, vectors={len(vectors)})")
    client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        db_name=settings.MILVUS_DB_NAME,
    )

    rows = []
    for index, chunk in enumerate(chunks):
        rows.append(
            {
                "text": chunk,
                "vector": vectors[index],
                "document_id": document_id,
                "chunk_index": index,
                "source_name": source_name,
                "chunk_text": chunk,
                "file_type": file_type,
                "content_type": content_type or "",
            }
        )

    result = client.insert(
        collection_name=settings.MILVUS_DOC_COLLECTION_NAME,
        data=rows,
    )
    client.flush(collection_name=settings.MILVUS_DOC_COLLECTION_NAME)
    ids = [str(item) for item in result.get("ids", [])]
    logger.info(f"✓ Milvus 文档切块写入成功 (document_id={document_id}, chunks={len(ids)})")
    return ids


def _delete_document_chunks_from_milvus(
    document_id: str,
    chunk_count: int,
) -> bool:
    """使用 MilvusClient 原生删除文档切块。"""

    if chunk_count <= 0:
        logger.warning(f"文档 chunk_count 为空，无需删除切块 (document_id={document_id})")
        return False

    collection_name = settings.MILVUS_DOC_COLLECTION_NAME
    client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        db_name=settings.MILVUS_DB_NAME,
    )
    if not client.has_collection(collection_name):
        logger.warning(f"Milvus collection 不存在，无需删除切块 ({collection_name})")
        return False

    client.delete(
        collection_name=collection_name,
        filter=f'document_id == "{document_id}"',
    )
    logger.info(f"✓ Milvus 文档切块删除成功 (document_id={document_id})")
    return True


def _document_delete_requested(document_id: str) -> bool:
    """后台任务检查文档是否已经被用户删除。"""
    db: Session = SessionLocal()
    try:
        record = db.query(DocumentRecord).filter(DocumentRecord.document_id == document_id).first()
        return record is None or record.status in {"deleting", "deleted"}
    finally:
        db.close()


def _validate_document_delete_prerequisites(db: Session, document_id: str) -> tuple[DocumentRecord, Path]:
    """删除前校验 SQLite 与本地文件是否存在。"""

    record = db.query(DocumentRecord).filter(DocumentRecord.document_id == document_id).first()
    if not record:
        raise DocumentException("文档不存在")
    if record.status != "ready":
        raise DocumentException(f"文档当前状态为 {record.status}，仅 ready 状态允许删除")

    file_path = Path(record.file_path)
    if not file_path.exists():
        raise DocumentException("本地文件不存在，无法执行删除")

    return record, file_path


def _process_document_upload(
    document_id: str,
    stored_path: Path,
    original_name: str,
    file_type: str,
    content_type: Optional[str],
) -> None:
    """后台执行文档解析、切块与向量写入。"""
    db: Session = SessionLocal()
    try:
        logger.info(
            f"后台处理开始 (document_id={document_id}, file={original_name}, path={stored_path})"
        )

        logger.info(f"开始抽取文本 (document_id={document_id})")
        text = extract_text_from_file(stored_path)
        logger.info(
            f"文本抽取完成 (document_id={document_id}, text_len={len(text)})"
        )
        if not text.strip():
            raise DocumentException("文档内容为空，无法生成向量")

        logger.info(f"开始切块 (document_id={document_id})")
        chunks = split_document_text(text)
        logger.info(
            f"切块完成 (document_id={document_id}, chunk_count={len(chunks)})"
        )
        if not chunks:
            raise DocumentException("文档切块失败，未生成有效内容")

        if _document_delete_requested(document_id):
            logger.info(f"文档已请求删除，跳过后台向量写入 (document_id={document_id})")
            return

        logger.info(f"开始写入 Milvus (document_id={document_id}, chunks={len(chunks)})")
        _insert_document_chunks_to_milvus(
            document_id=document_id,
            chunks=chunks,
            source_name=original_name,
            file_type=file_type,
            content_type=content_type,
        )
        logger.info(f"Milvus 写入完成 (document_id={document_id})")

        logger.info(f"开始更新文档状态为 ready (document_id={document_id})")
        record = db.query(DocumentRecord).filter(DocumentRecord.document_id == document_id).first()
        if record:
            record.status = "ready"
            record.chunk_count = len(chunks)
            record.error_message = None
            db.commit()
            logger.info(f"文档状态更新完成 (document_id={document_id}, status=ready)")
        logger.info(f"✓ 文档后台处理完成 (document_id={document_id}, chunks={len(chunks)})")
    except Exception as e:
        db.rollback()
        try:
            logger.info(f"开始更新文档状态为 failed (document_id={document_id})")
            record = db.query(DocumentRecord).filter(DocumentRecord.document_id == document_id).first()
            if record:
                record.status = "failed"
                record.error_message = str(e)
                db.commit()
                logger.info(f"文档状态更新完成 (document_id={document_id}, status=failed)")
        except Exception:
            db.rollback()
        logger.error(f"文档后台处理失败: {str(e)}")
    finally:
        db.close()


def ingest_document(file: UploadFile, background_tasks: Optional[BackgroundTasks] = None) -> Dict:
    """上传文档，解析文本，切块后写入 SQLite 元数据和 Milvus 向量库。"""
    if not file.filename:
        raise DocumentException("文件名不能为空")

    # 1) 先做后缀校验，避免不支持的格式进入后续链路
    original_name = Path(file.filename).name
    file_type = Path(original_name).suffix.lower().lstrip(".")
    allowed_types = _normalize_allowed_types()
    if file_type not in allowed_types:
        raise DocumentException(f"不支持的文件类型: {file_type}")

    # 2) 给文件生成内部 ID，保证系统内文件名唯一且可追踪
    document_id = uuid4().hex
    stored_path = _save_uploaded_file(file, document_id, original_name)
    file_size = _get_file_size(stored_path)

    # 3) 先按字节大小拦截超限文件，避免后续解析和向量化浪费资源
    if file_size > settings.MAX_UPLOAD_SIZE:
        stored_path.unlink(missing_ok=True)
        raise DocumentException("文件超过最大上传限制")

    # 4) 确保 SQLite 元数据表已创建，然后写入一条 processing 记录
    init_metadata_db()
    db: Session = SessionLocal()

    record = DocumentRecord(
        document_id=document_id,
        original_filename=original_name,
        stored_filename=stored_path.name,
        file_path=str(stored_path.resolve()),
        content_type=file.content_type,
        file_type=file_type,
        file_size=file_size,
        status="processing",
        chunk_count=0,
    )

    try:
        db.add(record)
        db.commit()

        if background_tasks is not None:
            background_tasks.add_task(
                _process_document_upload,
                document_id=document_id,
                stored_path=stored_path,
                original_name=original_name,
                file_type=file_type,
                content_type=file.content_type,
            )
            db.refresh(record)
            return _serialize_record(record)

        text = extract_text_from_file(stored_path)
        if not text.strip():
            raise DocumentException("文档内容为空，无法生成向量")

        chunks = split_document_text(text)
        if not chunks:
            raise DocumentException("文档切块失败，未生成有效内容")

        _insert_document_chunks_to_milvus(
            document_id=document_id,
            chunks=chunks,
            source_name=original_name,
            file_type=file_type,
            content_type=file.content_type,
        )

        record.status = "ready"
        record.chunk_count = len(chunks)
        record.error_message = None
        db.commit()
        db.refresh(record)

        return _serialize_record(record)
    except Exception as e:
        db.rollback()
        try:
            record.status = "failed"
            record.error_message = str(e)
            db.add(record)
            db.commit()
        except Exception:
            db.rollback()
        logger.error(f"文档入库失败: {str(e)}")
        raise
    finally:
        db.close()


def list_documents(skip: int = 0, limit: int = 10) -> Dict:
    """使用 MilvusClient 原生查询文档切块，并按 document_id 汇总。"""
    client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        db_name=settings.MILVUS_DB_NAME,
    )
    collection_name = settings.MILVUS_DOC_COLLECTION_NAME

    if not client.has_collection(collection_name) or not is_collection_loaded(client, collection_name):
        return {
            "documents": [],
            "total": 0,
            "skip": skip,
            "limit": limit,
        }

    rows = client.query(
        collection_name=collection_name,
        filter="",
        output_fields=[
            "document_id",
            "source_name",
            "file_type",
            "content_type",
            "chunk_index",
        ],
        limit=10000,
    )

    documents = {}
    for row in rows:
        document_id = row["document_id"]
        document = documents.setdefault(
            document_id,
            {
                "document_id": document_id,
                "original_filename": row.get("source_name"),
                "source_name": row.get("source_name"),
                "file_type": row.get("file_type"),
                "content_type": row.get("content_type"),
                "chunk_count": 0,
                "status": "ready",
            },
        )
        document["chunk_count"] += 1

    document_list = list(documents.values())
    start = max(skip, 0)
    end = start + max(limit, 0)

    return {
        "documents": document_list[start:end],
        "total": len(document_list),
        "skip": skip,
        "limit": limit,
    }


def delete_document(document_id: str) -> Dict:
    """删除 SQLite 元数据、Milvus 向量块和本地原始文件。"""
    init_metadata_db()
    db: Session = SessionLocal()
    try:
        record, file_path = _validate_document_delete_prerequisites(db, document_id)

        record.status = "deleting"
        db.commit()

        milvus_deleted = _delete_document_chunks_from_milvus(document_id, record.chunk_count)
        file_path.unlink()

        payload = _serialize_record(record)
        db.delete(record)
        db.commit()
        payload["milvus_deleted"] = milvus_deleted
        payload["file_deleted"] = True

        return payload
    except Exception as e:
        db.rollback()
        logger.error(f"删除文档失败: {str(e)}")
        raise
    finally:
        db.close()
