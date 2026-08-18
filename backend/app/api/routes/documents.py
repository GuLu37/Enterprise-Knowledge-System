"""文档管理路由"""

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
import logging

from app.api.schemas import (
    DocumentContentResponse,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.utils.exceptions import DocumentException

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/content/{document_id}", response_model=DocumentContentResponse)
async def get_document_content(document_id: str):
    """获取文档正文预览内容。"""
    try:
        from app.services.document_service import get_document_content as get_document_content_service

        return DocumentContentResponse(**get_document_content_service(document_id))
    except DocumentException as e:
        status_code = 404 if "不存在" in str(e) else 400
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        logger.error(f"获取文档内容失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    上传文档

    - **file**: 要上传的文档文件
    """
    try:
        from app.services.document_service import ingest_document

        return ingest_document(file, background_tasks=background_tasks)
    except DocumentException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"文档上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
):
    """
    列出所有文档

    - **skip**: 跳过的文档数
    - **limit**: 返回的最大文档数
    """
    try:
        from app.services.document_service import list_documents as list_documents_service

        return list_documents_service(skip=skip, limit=limit)
    except DocumentException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/delete/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def delete_document(document_id: str, background_tasks: BackgroundTasks):
    """
    删除文档

    - **document_id**: 文档ID
    """
    try:
        from app.services.document_service import (
            delete_document as delete_document_service,
            request_document_deletion,
        )

        payload = await run_in_threadpool(request_document_deletion, document_id)
        background_tasks.add_task(delete_document_service, document_id, True)
        return payload
    except DocumentException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"文档删除失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
