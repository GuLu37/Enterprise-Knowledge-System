"""Chat service 的工具调用编排测试。"""
from app.services.chat_service import build_chat_messages, run_chat


class _Response:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.additional_kwargs = {}


class _Tool:
    name = "search_knowledge_base"

    def __init__(self, sources):
        self.sources = sources

    def invoke(self, args):
        self.sources.append(
            {
                "content": "制度要求提前提交申请。",
                "metadata": {"document_id": "doc-1", "chunk_index": 0},
                "score": 0.9,
                "source": "hybrid",
            }
        )
        return "知识库返回：制度要求提前提交申请。"


class _ToolCallingLLM:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return _Response(
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "search_knowledge_base",
                        "args": {"query": "请假制度"},
                    }
                ]
            )
        return _Response(content="根据制度，需要提前提交申请。")


class _PlainLLM:
    def __init__(self):
        self.bound_tools = False

    def bind_tools(self, tools):
        self.bound_tools = True
        return self

    def invoke(self, messages):
        return _Response(content="你好。")


class _SummaryLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _Response(content=f"摘要{len(self.calls)}")


class _MemoryAwareLLM:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return _Response(content="综合长期记忆后的回答。")


def test_run_chat_executes_rag_tool_only_when_the_llm_requests_it(monkeypatch):
    def fake_create_rag_tools(sources, **kwargs):
        return [_Tool(sources)]

    monkeypatch.setattr(
        "app.services.chat_service.create_rag_tools",
        fake_create_rag_tools,
    )

    result = run_chat(
        query="请假制度是什么？",
        llm=_ToolCallingLLM(),
    )

    assert result.text == "根据制度，需要提前提交申请。"
    assert len(result.sources) == 1
    assert result.sources[0]["metadata"]["document_id"] == "doc-1"


def test_run_chat_does_not_bind_rag_tools_when_retrieval_is_disabled():
    llm = _PlainLLM()

    result = run_chat(
        query="你好",
        use_retrieval=False,
        llm=llm,
    )

    assert result.text == "你好。"
    assert llm.bound_tools is False


def test_build_chat_messages_sliding_window_keeps_recent_turns_only():
    history = []
    for index in range(1, 5):
        history.append({"role": "user", "content": f"用户{index}"})
        history.append({"role": "assistant", "content": f"助手{index}"})

    messages = build_chat_messages(
        history=history,
        query="当前问题",
        short_memory_strategy="window",
        short_memory_n=2,
    )

    contents = [getattr(message, "content", "") for message in messages]
    assert "用户1" not in contents
    assert "助手1" not in contents
    assert "用户3" in contents
    assert "助手4" in contents
    assert contents[-1] == "当前问题"


def test_build_chat_messages_summary_strategy_uses_llm_to_compress_older_history():
    history = []
    for index in range(1, 7):
        history.append({"role": "user", "content": f"用户{index}"})
        history.append({"role": "assistant", "content": f"助手{index}"})

    llm = _SummaryLLM()
    messages = build_chat_messages(
        history=history,
        query="当前问题",
        short_memory_strategy="summary",
        short_memory_n=2,
        short_memory_m=4,
        llm=llm,
    )

    contents = [getattr(message, "content", "") for message in messages]
    assert len(llm.calls) == 2
    assert any("摘要2" in content for content in contents)
    assert "用户5" in contents
    assert "助手6" in contents


def test_run_chat_injects_long_term_memory_and_persists_it(monkeypatch):
    captured_store_args = {}

    class _MemoryHit:
        def __init__(self, content):
            self.content = content

    def fake_search_long_term_memory(**kwargs):
        return [
            _MemoryHit("【主题】假期安排\n【摘要】用户想确认年假流程。\n【原文】用户：年假怎么申请？"),
        ]

    def fake_store_semantic_long_term_memory(**kwargs):
        captured_store_args.update(kwargs)
        return ["memory-1"]

    monkeypatch.setattr(
        "app.services.chat_service.search_long_term_memory",
        fake_search_long_term_memory,
    )
    monkeypatch.setattr(
        "app.services.chat_service.store_semantic_long_term_memory",
        fake_store_semantic_long_term_memory,
    )

    llm = _MemoryAwareLLM()
    result = run_chat(
        query="年假怎么申请？",
        history=[
            {"role": "user", "content": "我在看年假流程"},
            {"role": "assistant", "content": "好的"},
        ],
        conversation_id="conv-1",
        session_id="sess-1",
        use_retrieval=False,
        llm=llm,
    )

    assert result.text == "综合长期记忆后的回答。"
    assert len(llm.calls) == 1

    first_call_contents = [getattr(message, "content", "") for message in llm.calls[0]]
    assert any("长期记忆" in content for content in first_call_contents)
    assert any("假期安排" in content for content in first_call_contents)
    assert captured_store_args["conversation_id"] == "conv-1"
    assert captured_store_args["session_id"] == "sess-1"
    assert captured_store_args["short_window_n"] == 5
    assert captured_store_args["messages"][-2]["role"] == "user"
    assert captured_store_args["messages"][-1]["role"] == "assistant"
