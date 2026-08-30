"""Skill 目录与 Function Calling 路由。"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import (
    SkillDefinitionResponse,
    SkillExecuteRequest,
    SkillExecuteResponse,
)
from app.services.auth_service import require_current_user
from app.skills.registry import (
    SkillExecutionContext,
    SkillExecutionError,
    SkillNotFoundError,
    execute_skill,
    list_agent_capabilities,
    list_agent_function_calling_tools,
    list_skill_definitions,
)

router = APIRouter()


@router.get("", response_model=list[SkillDefinitionResponse])
async def get_skills(current_user=Depends(require_current_user)):
    """返回当前可用的技能目录。"""
    return list_skill_definitions()


@router.get("/function-calling")
async def get_function_calling_tools(current_user=Depends(require_current_user)):
    """返回标准 Function Calling 工具定义。"""
    return {"tools": list_agent_function_calling_tools()}


@router.get("/capabilities")
async def get_capabilities(current_user=Depends(require_current_user)):
    """返回统一能力目录，Function Calling 工具列表是 Skill 的发现入口。"""
    tools = list_agent_function_calling_tools()
    return {
        "capabilities": list_agent_capabilities(),
        "tools": tools,
        "function_calling": tools,
    }


@router.post("/execute", response_model=SkillExecuteResponse)
async def execute_skill_route(request: SkillExecuteRequest, current_user=Depends(require_current_user)):
    """执行一个标准化 Skill。"""
    try:
        result = await run_in_threadpool(
            execute_skill,
            request.name,
            request.arguments,
            SkillExecutionContext(
                user_id=current_user["user_id"],
                conversation_id=request.context.conversation_id,
                session_id=request.context.session_id,
                query=request.context.query,
                trace_id=request.context.trace_id,
                metadata=request.context.metadata,
            ),
        )
        return SkillExecuteResponse(**result)
    except SkillNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到技能: {exc}")
    except SkillExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
