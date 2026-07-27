"""对话和生成路由"""
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """聊天消息模型"""

    role: str  # system, user, assistant
    content: str


class ChatRequest(BaseModel):
    """对话请求"""
    query: str
    history: Optional[List[ChatMessage]] = None
    top_k: int = 5
    use_retrieval: bool = True
    temperature: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None


class ChatResponse(BaseModel):
    """对话响应"""
    query: str
    response: str
    sources: Optional[List[dict]] = None
    model: str


def _build_messages(request: ChatRequest):
    """将请求转换为 LangChain 消息列表。"""
    messages = []

    for message in request.history or []:
        if message.role == "system":
            messages.append(SystemMessage(content=message.content))
        elif message.role == "user":
            messages.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            messages.append(AIMessage(content=message.content))
        else:
            raise HTTPException(status_code=400, detail=f"不支持的消息角色: {message.role}")

    messages.append(HumanMessage(content=request.query.strip()))
    return messages


def _extract_text(chunk) -> str:
    """提取流式 chunk 的纯文本。"""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    """
    生成对话响应 (基于 RAG)

    - **query**: 用户查询
    - **history**: 对话历史
    - **top_k**: 检索结果数
    - **use_retrieval**: 是否使用检索
    - **temperature**: 生成温度
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="查询文本不能为空")

        messages = _build_messages(request)
        from app.core.llm import get_llm

        llm = get_llm(
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
        )
        response = llm.invoke(messages)
        text = _extract_text(response)

        return ChatResponse(
            query=request.query,
            response=text,
            sources=None,
            model=request.model or "default",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成响应失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_response(request: ChatRequest):
    """
    流式生成对话响应

    - **query**: 用户查询
    - **history**: 对话历史
    - **top_k**: 检索结果数
    - **use_retrieval**: 是否使用检索
    - **temperature**: 生成温度
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="查询文本不能为空")

        messages = _build_messages(request)

        def event_stream():
            try:
                from app.core.llm import get_llm

                # 构建LLM
                llm = get_llm(
                    provider=request.provider,
                    model=request.model,
                    temperature=request.temperature,
                )

                yield _sse_event(
                    "start",
                    {
                        "query": request.query,
                        "provider": request.provider,
                        "model": request.model,
                        "use_retrieval": request.use_retrieval,
                    },
                )

                for chunk in llm.stream(messages):
                    text = _extract_text(chunk)
                    if text:
                        logger.info(f"流式输出片段: {text}")
                        yield _sse_event("message", {"content": text})

                logger.info("流式生成完成")
                yield _sse_event("done", {"query": request.query})
            except Exception as e:
                logger.error(f"流式生成失败: {str(e)}")
                yield _sse_event("error", {"message": str(e)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"流式生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
