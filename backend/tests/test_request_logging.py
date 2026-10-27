from types import SimpleNamespace

from app import main


def test_log_requests_uses_debug_level(monkeypatch):
    calls = []

    monkeypatch.setattr(main.logger, "debug", lambda message: calls.append(("debug", message)))
    monkeypatch.setattr(main.logger, "info", lambda message: calls.append(("info", message)))
    monkeypatch.setattr(main.logger, "exception", lambda message: calls.append(("exception", message)))

    class FakeRequest:
        method = "GET"

        class _Url:
            path = "/api/v1/chat/runtime-status"
            query = ""

        url = _Url()
        client = SimpleNamespace(host="127.0.0.1")
        state = SimpleNamespace(renewed_access_token=None)

    async def fake_call_next(_request):
        return SimpleNamespace(status_code=200, headers={})

    import asyncio

    asyncio.run(main.log_requests(FakeRequest(), fake_call_next))

    assert any(level == "debug" and "请求开始" in message for level, message in calls)
    assert any(level == "debug" and "请求完成" in message for level, message in calls)
    assert all(level != "info" for level, _ in calls)
