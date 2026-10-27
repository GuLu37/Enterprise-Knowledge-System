"""提供给 LLM 的知识库工具。"""
from typing import Any, Dict, MutableSequence, Optional

from app.services.retrieval_service import (
    run_rag_workflow,
    serialize_retrieval_results,
)
from app.tools.trace import log_tool_call, log_tool_error, log_tool_result


def run_rag_tool(
    query: str,
    top_k: Optional[int] = None,
    default_top_k: int = 5,
    max_top_k: int = 50,
    retrieval_method: str = "hybrid",
    sources_sink: Optional[MutableSequence[Dict[str, Any]]] = None,
    llm: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """执行知识库 RAG 工具，并返回可用于 Prompt 组装的结构化结果。"""
    resolved_top_k = max(1, min(top_k or default_top_k, max_top_k))
    resolved_method = (retrieval_method or "hybrid").strip().lower()
    log_tool_call(
        kind="tool",
        name="search_knowledge_base",
        arguments={
            "query": query,
            "top_k": resolved_top_k,
            "retrieval_method": resolved_method,
        },
        context=context or {},
    )
    try:
        workflow = run_rag_workflow(
            query=query,
            top_k=resolved_top_k,
            retrieval_method=resolved_method,
            llm=llm,
        )
        results = serialize_retrieval_results(workflow.results)
        context_metadata = dict((context or {}).get("metadata") or {})
        retrieval_plan = context_metadata.get("retrieval_plan")

        if sources_sink is not None:
            sources_sink.extend(results)

        payload = {
            "query": workflow.query,
            "retrieval_method": workflow.retrieval_method,
            "expanded_queries": workflow.expanded_queries,
            "results": results,
            "context": workflow.context,
            "message": workflow.message,
        }
        if retrieval_plan:
            payload["retrieval_plan"] = retrieval_plan
        log_tool_result(
            kind="tool",
            name="search_knowledge_base",
            result={
                "retrieval_method": payload["retrieval_method"],
                "expanded_queries": payload["expanded_queries"],
                "result_count": len(results),
                "message": payload["message"],
                "retrieval_plan": retrieval_plan or {},
            },
            context=context or {},
        )
        return payload
    except Exception as exc:
        log_tool_error(
            kind="tool",
            name="search_knowledge_base",
            error=exc,
            context=context or {},
        )
        raise
