"""提供给 LLM 的知识库工具。"""
import json
from typing import Any, Dict, List, MutableSequence, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.services.retrieval_service import (
    retrieve_documents,
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


def create_rag_tools(
    sources_sink: Optional[MutableSequence[Dict[str, Any]]] = None,
    default_top_k: int = 5,
    max_top_k: int = 50,
) -> List[StructuredTool]:
    """创建带有本次请求来源收集器的 RAG 工具。"""

    def search_knowledge_base(query: str, top_k: Optional[int] = None) -> str:
        """检索企业知识库中的文档片段，用于回答企业内部知识问题。"""
        resolved_top_k = max(1, min(top_k or default_top_k, max_top_k))
        results = serialize_retrieval_results(
            retrieve_documents(query=query, top_k=resolved_top_k)
        )

        if sources_sink is not None:
            sources_sink.extend(results)

        return json.dumps(
            {
                "query": query,
                "results": results,
                "message": "未检索到相关内容" if not results else "检索完成",
            },
            ensure_ascii=False,
        )

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
