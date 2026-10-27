from types import SimpleNamespace

from app.skills import builtin
from app.skills.registry import SkillExecutionContext, execute_skill
from app.tools.rag import run_rag_tool


def test_execute_skill_emits_trace_events(monkeypatch):
    events = []

    monkeypatch.setattr(
        "app.skills.registry.log_tool_call",
        lambda **kwargs: events.append(("call", kwargs)),
    )
    monkeypatch.setattr(
        "app.skills.registry.log_tool_result",
        lambda **kwargs: events.append(("result", kwargs)),
    )
    monkeypatch.setattr(
        "app.skills.registry.log_tool_error",
        lambda **kwargs: events.append(("error", kwargs)),
    )
    monkeypatch.setattr(
        builtin,
        "get_active_profile_memories",
        lambda user_id, limit=20: [],
    )
    monkeypatch.setattr(
        builtin,
        "search_long_term_memory",
        lambda **kwargs: [],
    )

    execute_skill(
        "memory_query",
        {"query": "记忆查询"},
        SkillExecutionContext(user_id="u1", trace_id="trace-1"),
    )

    assert events[0][0] == "call"
    assert events[0][1]["kind"] == "skill"
    assert events[1][0] == "result"
    assert events[1][1]["kind"] == "skill"


def test_list_available_tools_exposes_agent_function_calling_registry():
    result = execute_skill(
        "list_available_tools",
        {},
        SkillExecutionContext(user_id="u1"),
    )["result"]

    names = {item["name"] for item in result["tools"]}
    assert result["summary"]["tool_count"] == len(result["tools"])
    assert "search_knowledge_base" in names
    assert "list_available_tools" in names
    assert "memory_query" in names
    assert all("parameters" in item for item in result["tools"])


def test_run_rag_tool_emits_trace_events(monkeypatch):
    events = []

    monkeypatch.setattr(
        "app.tools.rag.log_tool_call",
        lambda **kwargs: events.append(("call", kwargs)),
    )
    monkeypatch.setattr(
        "app.tools.rag.log_tool_result",
        lambda **kwargs: events.append(("result", kwargs)),
    )
    monkeypatch.setattr(
        "app.tools.rag.run_rag_workflow",
        lambda **kwargs: SimpleNamespace(
            query=kwargs["query"],
            retrieval_method=kwargs["retrieval_method"],
            expanded_queries=["q1"],
            results=[],
            context="ctx",
            message="ok",
        ),
    )
    monkeypatch.setattr(
        "app.tools.rag.serialize_retrieval_results",
        lambda results: [],
    )

    run_rag_tool(query="差旅报销", top_k=3, retrieval_method="hybrid")

    assert events[0][0] == "call"
    assert events[0][1]["kind"] == "tool"
    assert events[1][0] == "result"
    assert events[1][1]["kind"] == "tool"
