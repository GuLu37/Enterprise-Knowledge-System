"""提供给 LLM 的知识库工具。"""
from typing import Any, Dict, MutableSequence, Optional

from app.services.retrieval_service import (
    run_rag_workflow,
    serialize_retrieval_results,
)


def run_rag_tool(
    query: str,
    top_k: Optional[int] = None,
    default_top_k: int = 5,
    max_top_k: int = 50,
    retrieval_method: str = "hybrid",
    sources_sink: Optional[MutableSequence[Dict[str, Any]]] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """执行知识库 RAG 工具，并返回可用于 Prompt 组装的结构化结果。"""
    resolved_top_k = max(1, min(top_k or default_top_k, max_top_k))
    resolved_method = (retrieval_method or "hybrid").strip().lower()
    workflow = run_rag_workflow(
        query=query,
        top_k=resolved_top_k,
        retrieval_method=resolved_method,
        llm=llm,
    )
    results = serialize_retrieval_results(workflow.results)

    if sources_sink is not None:
        sources_sink.extend(results)

    return {
        "query": workflow.query,
        "retrieval_method": workflow.retrieval_method,
        "expanded_queries": workflow.expanded_queries,
        "results": results,
        "context": workflow.context,
        "message": workflow.message,
    }
