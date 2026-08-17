"""对话 HTTP 路由。"""
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatRuntimeStatusResponse,
    ChatSettingsResponse,
    ConversationDeleteResponse,
    ChatWarmupResponse,
)
from app.config import settings
from app.services.auth_service import require_current_user
from app.utils.exceptions import VectorStoreException

router = APIRouter()


def _sse_event(event: str, data: dict) -> str:
    """把 service 事件转换成 SSE 格式。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/settings", response_model=ChatSettingsResponse)
async def get_chat_settings():
    """返回前端需要遵守的聊天配置。"""
    return ChatSettingsResponse(max_conversations=settings.CHAT_MAX_CONVERSATIONS)


@router.get("/runtime-status", response_model=ChatRuntimeStatusResponse)
async def get_chat_runtime_status(request: Request):
    """返回聊天模型和向量模型的后台初始化状态。"""
    runtime_warmup = getattr(request.app.state, "runtime_warmup", {}) or {}
    llm_warmed = bool(runtime_warmup.get("llm_warmed"))
    embedding_warmed = bool(runtime_warmup.get("embedding_warmed"))
    ready = llm_warmed and embedding_warmed
    status = "ready" if ready else str(runtime_warmup.get("status") or "initializing")

    return ChatRuntimeStatusResponse(
        ready=ready,
        status=status,
        llm_warmed=llm_warmed,
        embedding_warmed=embedding_warmed,
        provider=runtime_warmup.get("provider"),
    )


@router.post("/warmup", response_model=ChatWarmupResponse)
async def warmup_chat_runtime():
    """预热聊天相关的 LLM 和 embedding 运行时。"""
    try:
        from app.services.chat_service import warmup_chat_runtime

        result = await run_in_threadpool(warmup_chat_runtime)
        return ChatWarmupResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/conversations/{conversation_id}", response_model=ConversationDeleteResponse)
async def delete_conversation(conversation_id: str, current_user=Depends(require_current_user)):
    """删除一个 conversation_id 关联的长期记忆。"""
    try:
        from app.services.memory_service import delete_long_term_memory

        memory_deleted = await run_in_threadpool(
            delete_long_term_memory,
            conversation_id,
            current_user["user_id"],
        )
        return ConversationDeleteResponse(
            conversation_id=conversation_id,
            memory_deleted=memory_deleted,
        )
    except VectorStoreException as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest, current_user=Depends(require_current_user)):
    """生成一次先做意图路由、再按需调用 RAG 工具的回答。"""
    try:
        from app.services.chat_service import generate_chat

        result = await run_in_threadpool(
            generate_chat,
            query=request.query,
            history=request.history,
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            top_k=request.top_k,
            use_retrieval=request.use_retrieval,
            retrieval_method=request.retrieval_method,
            user_id=current_user["user_id"],
            short_memory_strategy=request.short_memory_strategy,
            short_memory_n=request.short_memory_n,
            short_memory_m=request.short_memory_m,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
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
async def stream_response(request: ChatRequest, current_user=Depends(require_current_user)):
    """流式生成回答，并转发 RAG 工具调用及来源事件。"""

    def event_stream():
        try:
            from app.services.chat_service import stream_chat_events

            yield _sse_event(
                "start",
                {
                    "query": request.query,
                    "conversation_id": request.conversation_id,
                    "session_id": request.session_id,
                    "user_id": current_user["user_id"],
                    "provider": request.provider,
                    "model": request.model,
                    "use_retrieval": request.use_retrieval,
                    "retrieval_method": request.retrieval_method,
                    "short_memory_strategy": request.short_memory_strategy,
                    "short_memory_n": request.short_memory_n,
                    "short_memory_m": request.short_memory_m,
                },
            )

            events = stream_chat_events(
                query=request.query,
                history=request.history,
                conversation_id=request.conversation_id,
                session_id=request.session_id,
                top_k=request.top_k,
                use_retrieval=request.use_retrieval,
                retrieval_method=request.retrieval_method,
                user_id=current_user["user_id"],
                short_memory_strategy=request.short_memory_strategy,
                short_memory_n=request.short_memory_n,
                short_memory_m=request.short_memory_m,
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
