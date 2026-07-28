"""Chat service 的工具调用编排测试。"""
from app.services.chat_service import run_chat


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
