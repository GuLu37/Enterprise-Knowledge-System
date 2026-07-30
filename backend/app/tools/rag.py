"""提供给 LLM 的知识库工具。"""
import json
from typing import Any, Dict, List, MutableSequence, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.services.retrieval_service import (
    run_rag_workflow,
    serialize_retrieval_results,
)


class KnowledgeBaseSearchInput(BaseModel):
    """知识库搜索工具参数。"""

    query: str = Field(description="要在企业知识库中检索的问题或关键词")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="最多返回的知识片段数量；未传时使用当前请求的默认值",
    )
    retrieval_method: Optional[str] = Field(
        default=None,
        description="检索方式，可选 hybrid、dense、sparse；未传时使用当前请求的默认值",
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
        "expanded_queries": workflow.expanded_queries,
        "results": results,
        "context": workflow.context,
        "message": workflow.message,
    }


def create_rag_tools(
    sources_sink: Optional[MutableSequence[Dict[str, Any]]] = None,
    default_top_k: int = 5,
    max_top_k: int = 50,
    retrieval_method: str = "hybrid",
    llm: Optional[Any] = None,
) -> List[StructuredTool]:
    """创建带有本次请求来源收集器的 RAG 工具。"""
    default_retrieval_method = (retrieval_method or "hybrid").strip().lower()

    def search_knowledge_base(
        query: str,
        top_k: Optional[int] = None,
        retrieval_method: Optional[str] = None,
    ) -> str:
        """检索企业知识库中的文档片段，用于回答企业内部知识问题。"""
        result = run_rag_tool(
            query=query,
            top_k=top_k,
            default_top_k=default_top_k,
            max_top_k=max_top_k,
            retrieval_method=retrieval_method or default_retrieval_method,
            sources_sink=sources_sink,
            llm=llm,
        )
        return json.dumps(result, ensure_ascii=False)

    return [
        StructuredTool.from_function(
            func=search_knowledge_base,
            name="search_knowledge_base",
            description=(
                "搜索企业知识库中的内部文档。当用户询问公司制度、业务流程、"
                "项目资料、上传文件或其他可能存在于企业文档中的事实时调用。"
                "闲聊、创作和不依赖企业文档的一般问题不需要调用。"
            ),
            args_schema=KnowledgeBaseSearchInput,
        )
    ]
