from app.utils.logger import _should_show_in_console


def test_console_filter_hides_uvicorn_access():
    assert _should_show_in_console({"extra": {"source_logger": "app.main"}})
    assert not _should_show_in_console({"extra": {"source_logger": "uvicorn.access"}})
