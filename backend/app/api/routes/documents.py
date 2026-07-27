"""文档管理路由"""
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
import logging
from app.services.document_service import ingest_document
from app.services.document_service import list_documents as list_documents_service
from app.services.document_service import delete_document as delete_document_service

from app.utils.exceptions import DocumentException

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    上传文档

    - **file**: 要上传的文档文件
    """
    try:
        return ingest_document(file, background_tasks=background_tasks)
    except DocumentException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"文档上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_documents(skip: int = 0, limit: int = 10):
    """
    列出所有文档

    - **skip**: 跳过的文档数
    - **limit**: 返回的最大文档数
    """
    try:
        return list_documents_service(skip=skip, limit=limit)
    except DocumentException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{document_id}")
async def delete_document(document_id: str):
    """
    删除文档

    - **document_id**: 文档ID
    """
    try:
        return await run_in_threadpool(delete_document_service, document_id)
    except DocumentException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"文档删除失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
