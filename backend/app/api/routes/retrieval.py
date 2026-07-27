"""检索路由"""
from fastapi import APIRouter, HTTPException
import logging

from app.api.schemas import RetrievalRequest, RetrievalResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _build_response(query: str, top_k: int, method: str, results):
    return {
        "query": query,
        "retrieval_method": method,
        "results": results,
        "top_k": top_k,
    }


@router.post("/search/hybrid", response_model=RetrievalResponse)
async def hybrid_search(request: RetrievalRequest):
    """
    混合检索文档

    - **query**: 查询文本
    - **top_k**: 返回的最大结果数
    """
    try:
        from app.services.retrieval_service import (
            retrieve_documents,
            serialize_retrieval_results,
        )

        results = serialize_retrieval_results(
            retrieve_documents(request.query, top_k=request.top_k)
        )
        return _build_response(request.query, request.top_k, "hybrid", results)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/dense", response_model=RetrievalResponse)
async def dense_search(request: RetrievalRequest):
    """
    密集向量检索 (语义相似度)

    - **query**: 查询文本
    - **top_k**: 返回的最大结果数
    """
    try:
        from app.core.embeddings import get_default_embeddings
        from app.rag.retrieval.dense_retriever import DenseRetriever
        from app.services.retrieval_service import serialize_retrieval_results

        embeddings = get_default_embeddings()
        dense_retriever = DenseRetriever(embeddings=embeddings, top_k=request.top_k)

        results = dense_retriever.retrieve(request.query, top_k=request.top_k)
        return _build_response(
            request.query,
            request.top_k,
            "dense",
            serialize_retrieval_results(results),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"密集检索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/sparse", response_model=RetrievalResponse)
async def sparse_search(request: RetrievalRequest):
    """
    稀疏检索 (BM25 关键词)

    - **query**: 查询文本
    - **top_k**: 返回的最大结果数
    """
    try:
        from app.rag.retrieval.sparse_retriever import SparseRetriever
        from app.services.retrieval_service import serialize_retrieval_results

        sparse_retriever = SparseRetriever(top_k=request.top_k)
        results = sparse_retriever.retrieve(request.query, top_k=request.top_k)
        return _build_response(
            request.query,
            request.top_k,
            "sparse",
            serialize_retrieval_results(results),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"稀疏检索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
