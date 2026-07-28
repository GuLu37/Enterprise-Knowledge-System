"""检索路由"""
from fastapi import APIRouter, HTTPException, Query
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


def _build_hybrid_response(query: str, top_k: int, results):
    return {
        "query": query,
        "retrieval_method": "hybrid",
        "results": results,
        "top_k": top_k,
    }


@router.post("/search/hybrid")
async def hybrid_search(
    query: str,
    top_k: int = Query(5, ge=1, le=50),
):
    """
    混合检索文档

    - **query**: 查询文本
    - **top_k**: 返回的最大结果数
    """
    try:
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="查询文本不能为空")

        from app.services.retrieval_service import (
            retrieve_documents,
            serialize_retrieval_results,
        )

        results = retrieve_documents(query, top_k=top_k)
        return _build_hybrid_response(query, top_k, results)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/dense")
async def dense_search(query: str, top_k: int = Query(5, ge=1, le=50)):
    """
    密集向量检索 (语义相似度)

    - **query**: 查询文本
    - **top_k**: 返回的最大结果数
    """
    try:
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="查询文本不能为空")

        from app.core.embeddings import get_default_embeddings
        from app.rag.retrieval.dense_retriever import DenseRetriever

        embeddings = get_default_embeddings()
        dense_retriever = DenseRetriever(embeddings=embeddings, top_k=top_k)

        results = dense_retriever.retrieve(query, top_k=top_k)
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"密集检索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/sparse")
async def sparse_search(query: str, top_k: int = Query(5, ge=1, le=50)):
    """
    稀疏检索 (BM25 关键词)

    - **query**: 查询文本
    - **top_k**: 返回的最大结果数
    """
    try:
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="查询文本不能为空")

        from app.rag.retrieval.sparse_retriever import SparseRetriever

        sparse_retriever = SparseRetriever(top_k=top_k)
        return sparse_retriever.retrieve(query, top_k=top_k)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"稀疏检索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
