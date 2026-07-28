"""对话 HTTP 路由。"""
import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class ChatMessage(BaseModel):
    """聊天消息模型。"""

    role: str
    content: str


class ChatRequest(BaseModel):
    """对话请求。"""

    query: str
    history: Optional[List[ChatMessage]] = None
    top_k: int = 5
    use_retrieval: bool = True
    temperature: Optional[float] = None
    provider: Optional[str] = None
    model: Optional[str] = None


class ChatResponse(BaseModel):
    """对话响应。"""

    query: str
    response: str
    sources: Optional[List[dict]] = None
    model: str


def _sse_event(event: str, data: dict) -> str:
    """把 service 事件转换成 SSE 格式。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    """生成一次支持 LLM 自主调用 RAG 工具的回答。"""
    try:
        from app.services.chat_service import generate_chat

        result = await run_in_threadpool(
            generate_chat,
            request.query,
            request.history,
            request.top_k,
            request.use_retrieval,
            request.provider,
            request.model,
            request.temperature,
        )
        return ChatResponse(
            query=request.query,
            response=result.text,
            sources=result.sources,
            model=result.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/stream")
async def stream_response(request: ChatRequest):
    """流式生成回答，并转发工具调用及来源事件。"""

    def event_stream():
        try:
            from app.services.chat_service import stream_chat_events

            yield _sse_event(
                "start",
                {
                    "query": request.query,
                    "provider": request.provider,
                    "model": request.model,
                    "use_retrieval": request.use_retrieval,
                },
            )

            events = stream_chat_events(
                query=request.query,
                history=request.history,
                top_k=request.top_k,
                use_retrieval=request.use_retrieval,
                provider=request.provider,
                model=request.model,
                temperature=request.temperature,
            )
            for event in events:
                yield _sse_event(event["event"], event["data"])
        except ValueError as exc:
            yield _sse_event("error", {"message": str(exc)})
        except Exception as exc:
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
