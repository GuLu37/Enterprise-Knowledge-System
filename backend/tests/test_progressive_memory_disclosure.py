import pytest
from types import SimpleNamespace

from app.services import chat_service, memory_service
from app.utils.chunking import build_conversation_chunks


def test_original_memory_requires_current_user_and_expands_summary_range(monkeypatch):
    summary_entity = {
        "memory_id": "summary-1",
        "chunk_text": "【主题】项目 Alpha\n【摘要】用户确认了上线计划。",
        "chunk_index": 10,
        "source_name": "conversation",
        "user_id": "u1",
        "conversation_id": "c1",
        "session_id": "c1",
        "chunk_type": "semantic_memory_l1",
        "topic": "项目 Alpha",
        "turn_start": 2,
        "turn_end": 3,
    }
    raw_entity = {
        "memory_id": "raw-1",
        "chunk_text": "用户: 请确认 Alpha 上线计划\n助手: 已确认本周五上线。",
        "chunk_index": 2,
        "source_name": "conversation",
        "user_id": "u1",
        "conversation_id": "c1",
        "session_id": "c1",
        "chunk_type": "dialogue",
        "topic": "",
        "turn_start": 2,
        "turn_end": 3,
    }

    class FakeClient:
        def has_collection(self, name):
            return True

        def query(self, **kwargs):
            if "memory_id in" in kwargs["filter"]:
                return [summary_entity]
            return [raw_entity]

    monkeypatch.setattr(
        "app.storage.milvus_store.get_milvus_client",
        lambda: FakeClient(),
    )
    monkeypatch.setattr("app.storage.milvus_store._ensure_memory_collection", lambda client: None)
    monkeypatch.setattr("app.storage.milvus_store._validate_vector_dimension", lambda client, name: None)
    monkeypatch.setattr("app.storage.milvus_store.is_collection_loaded", lambda client, name: True)
    monkeypatch.setattr(
        memory_service,
        "_get_active_event_records",
        lambda user_id, ids: {
            item: SimpleNamespace(
                memory_id=item,
                status="active",
                conversation_id="c1",
                session_id="c1",
                importance_score=50,
                confidence_score=0.8,
                access_count=0,
            )
            for item in ids
            if item in {"summary-1", "raw-1"}
        },
    )
    monkeypatch.setattr(memory_service, "_mark_memory_accessed", lambda *args: None)

    result = memory_service.retrieve_original_conversation(
        "u1",
        memory_ids=["summary-1"],
        include_neighbors=0,
    )

    assert result["summary_references"][0]["memory_id"] == "summary-1"
    assert result["original_chunks"][0]["memory_id"] == "raw-1"
    assert "Alpha 上线计划" in result["original_chunks"][0]["content"]


def test_summary_history_injects_progressive_disclosure_reference():
    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(content="用户已经确认了项目 Alpha 的上线计划。")

    messages = chat_service._build_summary_history(
        history=[
            {"role": "user", "content": "请讨论项目 Alpha"},
            {"role": "assistant", "content": "好的"},
            {"role": "user", "content": "确认周五上线"},
            {"role": "assistant", "content": "已确认"},
        ],
        window_n=1,
        summary_m=2,
        llm=FakeLLM(),
        conversation_id="c1",
    )

    summary_message = next(message for message in messages if "递进摘要" in message.content)
    assert "memory_retrieve_original" in summary_message.content
    assert '"conversation_id": "c1"' in summary_message.content


def test_original_conversation_request_routes_to_disclosure_skill():
    intent = chat_service._rule_route_skill_intent("请展开之前摘要对应的完整对话")

    assert intent is not None
    assert intent.skill_name == "memory_retrieve_original"


@pytest.mark.parametrize(
    "query",
    ["列出你的工具", "你有什么技能吗", "有哪些可用能力"],
)
def test_tool_catalog_queries_route_to_listing_skill(query):
    intent = chat_service._rule_route_skill_intent(query)

    assert intent is not None
    assert intent.skill_name == "list_available_tools"


def test_other_user_summary_reference_is_not_exposed(monkeypatch):
    class FakeClient:
        def has_collection(self, name):
            return True

        def query(self, **kwargs):
            return [
                {
                    "memory_id": "summary-other",
                    "chunk_text": "其他用户的摘要",
                    "user_id": "u2",
                    "conversation_id": "c2",
                    "chunk_type": "semantic_memory_l1",
                    "turn_start": 0,
                    "turn_end": 0,
                }
            ]

    monkeypatch.setattr("app.storage.milvus_store.get_milvus_client", lambda: FakeClient())
    monkeypatch.setattr("app.storage.milvus_store._ensure_memory_collection", lambda client: None)
    monkeypatch.setattr("app.storage.milvus_store._validate_vector_dimension", lambda client, name: None)
    monkeypatch.setattr("app.storage.milvus_store.is_collection_loaded", lambda client, name: True)
    monkeypatch.setattr(memory_service, "_get_active_event_records", lambda user_id, ids: {})

    result = memory_service.retrieve_original_conversation("u1", memory_ids=["summary-other"])

    assert result["summary_references"] == []
    assert result["original_chunks"] == []


def test_memory_context_reuses_profile_and_vector_search(monkeypatch):
    profile_calls = []
    search_calls = []

    monkeypatch.setattr(
        chat_service,
        "get_active_profile_memories",
        lambda user_id, limit=20: profile_calls.append((user_id, limit)) or [],
    )
    monkeypatch.setattr(
        chat_service,
        "search_long_term_memory",
        lambda **kwargs: search_calls.append(kwargs) or [],
    )

    context = chat_service.MemoryContext(
        query="我之前说过项目 Alpha 吗",
        user_id="u1",
        conversation_id="c1",
        session_id="c1",
    )

    first = context.get_messages(route="rag")
    second = context.get_messages(route="rag")

    assert first == second == []
    assert len(profile_calls) == 1
    assert len(search_calls) == 1


def test_memory_query_merges_recent_history(monkeypatch):
    from app.skills import builtin

    class FakeMemoryContext:
        def __init__(self):
            self.recent_history = [
                {"role": "user", "content": "我刚才说过项目 Alpha 的上线计划"},
                {"role": "assistant", "content": "已记录在案"},
            ]

        def get_profile_memories(self, limit=20):
            return []

        def search(self, **kwargs):
            return []

    result = builtin._skill_memory_query(
        chat_service.SkillExecutionContext(
            user_id="u1",
            conversation_id="c1",
            session_id="c1",
            metadata={"memory_context": FakeMemoryContext()},
        ),
        {"query": "我刚才说过什么", "top_k": 3, "memory_scope": "event"},
    )

    texts = [item["content"] for item in result["matched_memories"]]
    assert any("项目 Alpha" in text for text in texts)


def test_tool_history_does_not_serialize_runtime_memory_context():
    context = chat_service.MemoryContext(query="记忆查询", user_id="u1")
    execution = {
        "tool": "memory_query",
        "arguments": {"query": "记忆查询"},
        "context": {"metadata": {"memory_context": context}},
        "result": {"summary": {"profile_count": 0, "matched_count": 0}},
    }

    messages = chat_service._append_tool_result_to_history(
        [],
        "memory_query",
        execution["arguments"],
        execution,
    )

    assert "MemoryContext" not in messages[0].content
    assert '"matched_count": 0' in messages[0].content


def test_memory_context_skips_unrelated_query(monkeypatch):
    calls = []

    monkeypatch.setattr(
        chat_service,
        "get_active_profile_memories",
        lambda user_id, limit=20: calls.append("profile") or [],
    )
    monkeypatch.setattr(
        chat_service,
        "search_long_term_memory",
        lambda **kwargs: calls.append("search") or [],
    )

    messages = chat_service._build_long_term_memory_messages(
        query="如何申请差旅报销",
        user_id="u1",
        conversation_id="c1",
        route="rag",
    )

    assert messages == []
    assert calls == []


def test_tool_defaults_are_shared_by_direct_and_agent_paths():
    query = "请展开之前摘要对应的完整对话"

    direct = chat_service._normalize_action_arguments(
        "memory_retrieve_original",
        {},
        query,
    )
    agent = chat_service._default_action_arguments(
        "memory_retrieve_original",
        query,
    )

    assert direct == agent
    assert direct["top_k"] == 8
    assert direct["max_chars"] == 12000


def test_memory_write_entrypoints_share_common_writer(monkeypatch):
    calls = []

    def fake_writer(**kwargs):
        calls.append(kwargs)
        return ["memory-1"]

    monkeypatch.setattr(memory_service, "_write_memory_chunks", fake_writer)
    monkeypatch.setattr(memory_service, "_count_dialogue_turns", lambda user_id, conversation_id: 3)

    semantic_ids = memory_service.store_semantic_long_term_memory(
        "一段历史对话",
        user_id="u1",
        conversation_id="c1",
    )
    dialogue_ids = memory_service.store_conversation_memory(
        "一段原始对话",
        user_id="u1",
        conversation_id="c1",
    )

    assert semantic_ids == dialogue_ids == ["memory-1"]
    assert len(calls) == 2
    assert calls[0]["chunk_type"] == "semantic_memory"
    assert calls[1]["chunk_type"] == "dialogue"
    assert calls[1]["turn_offset"] == 3


def test_summary_consolidation_cursor_waits_for_successful_write(monkeypatch):
    updates = []
    monkeypatch.setattr(
        memory_service,
        "_load_summary_chunks",
        lambda *args, **kwargs: [{"topic": "Alpha", "summary": "上线计划"}],
    )
    monkeypatch.setattr(
        memory_service,
        "merge_summary_level",
        lambda **kwargs: 0,
    )
    monkeypatch.setattr(
        memory_service,
        "_update_consolidation_cursor",
        lambda *args: updates.append(args),
    )

    assert memory_service._consolidate_level("u1", "c1", 2, 2, None) == 0
    assert updates == []

    monkeypatch.setattr(
        memory_service,
        "merge_summary_level",
        lambda **kwargs: 1,
    )
    assert memory_service._consolidate_level("u1", "c1", 2, 2, None) == 1
    assert updates == [("u1", "c1", 2, 2)]


def test_summary_consolidation_thresholds_are_10_20_40():
    assert memory_service.CONSOLIDATION_LEVELS == ((10, 1), (20, 2), (40, 3))


def test_l1_batch_summary_skips_topic_detection(monkeypatch):
    calls = []

    class FakeLLM:
        def invoke(self, messages):
            calls.append(messages)
            return SimpleNamespace(content='{"topic":"Alpha","summary":"已确认 Alpha 计划。"}')

    monkeypatch.setattr(
        memory_service,
        "_build_topic_change_detector",
        lambda llm: (_ for _ in ()).throw(AssertionError("L1 不应调用话题判断 LLM")),
    )
    monkeypatch.setattr(
        memory_service,
        "_write_memory_chunks",
        lambda **kwargs: ["l1-1"],
    )

    result = memory_service.store_semantic_long_term_memory(
        messages=[
            {"role": "user", "content": "讨论 Alpha 项目"},
            {"role": "assistant", "content": "已记录"},
            {"role": "user", "content": "确认上线计划"},
            {"role": "assistant", "content": "已确认"},
        ],
        user_id="u1",
        conversation_id="c1",
        llm=FakeLLM(),
        detect_topic_change=False,
        short_window_n=0,
    )

    assert result == ["l1-1"]
    assert len(calls) == 1


def test_long_message_chunks_keep_one_logical_turn():
    chunks = build_conversation_chunks(
        [
            {"role": "user", "content": "请处理这段很长的内容。" * 80},
            {"role": "assistant", "content": "已收到。"},
        ],
        chunk_size=80,
        chunk_overlap=0,
    )

    assert len(chunks) > 1
    assert all(chunk.turn_start == 0 and chunk.turn_end == 0 for chunk in chunks)


def test_summary_strategy_requires_conversation_id():
    with pytest.raises(ValueError, match="conversation_id"):
        chat_service._validate_summary_conversation_id(
            chat_service.SHORT_MEMORY_STRATEGY_SUMMARY,
            None,
        )


def _history_with_named_turns(count=6):
    history = []
    for index in range(1, count + 1):
        history.extend(
            [
                {"role": "user", "content": f"用户问题-{index}"},
                {"role": "assistant", "content": f"助手回答-{index}"},
            ]
        )
    return history


def test_intent_and_agent_selection_use_short_memory_window():
    history = _history_with_named_turns()

    intent_messages = chat_service._build_intent_router_messages(
        "继续刚才的问题",
        history=history,
        short_memory_n=5,
    )
    agent_messages = chat_service._build_agent_selection_messages(
        "继续刚才的问题",
        history=history,
        short_memory_n=3,
    )

    intent_prompt = "\n".join(str(message.content) for message in intent_messages)
    agent_prompt = "\n".join(str(message.content) for message in agent_messages)
    assert "用户问题-1" not in intent_prompt
    assert "用户问题-2" in intent_prompt
    assert "用户问题-6" in intent_prompt
    assert "用户问题-1" not in agent_prompt
    assert "用户问题-3" not in agent_prompt
    assert "用户问题-4" in agent_prompt
    assert "用户问题-6" in agent_prompt


def test_chat_entrypoints_pass_short_memory_n_to_action_router(monkeypatch):
    captured = []

    def fake_route_query_action(*args, **kwargs):
        captured.append(kwargs["short_memory_n"])
        return chat_service.QueryIntent(
            route="direct",
            needs_retrieval=False,
            reason="test",
            source="test",
        )

    monkeypatch.setattr(chat_service, "route_query_action", fake_route_query_action)
    monkeypatch.setattr(chat_service, "_invoke_llm_safely", lambda llm, messages: "ok")
    monkeypatch.setattr(chat_service, "_persist_long_term_memory_async", lambda **kwargs: None)

    chat_service.run_chat("测试", short_memory_n=7, llm=object())
    list(chat_service.stream_chat("测试", short_memory_n=4, llm=object()))


def test_recent_history_is_attached_to_memory_context(monkeypatch):
    captured = []

    def fake_run_skill_action(**kwargs):
        captured.append(kwargs["memory_context"].recent_history)
        return chat_service.ChatRunResult(text="ok")

    monkeypatch.setattr(chat_service, "route_query_action", lambda *args, **kwargs: chat_service.QueryIntent(route="skill", needs_retrieval=False, reason="test", source="test", skill_name="memory_query", skill_arguments={"memory_scope": "event"}))
    monkeypatch.setattr(chat_service, "_run_skill_action", fake_run_skill_action)

    history = [
        {"role": "user", "content": "我说过项目 Alpha"},
        {"role": "assistant", "content": "已记录"},
    ]

    chat_service.run_chat("查询记忆", history=history, llm=object())

    assert captured and captured[0] == history
