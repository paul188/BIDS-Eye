"""Every request gets an ID (request_log.py): it is in the X-Request-ID header
of every answer, including errors and the plain 500, and on every log line
written while the request is handled, also from a worker thread. LOG_FORMAT
chooses text or JSON lines."""
from __future__ import annotations

import asyncio
import io
import json
import logging

import httpx
import pytest
from pydantic import ValidationError
from fastapi import FastAPI

from config.settings import Settings, settings_obj
import request_log
from exceptions.domain import DomainError
from exceptions.handlers import UNEXPECTED_DETAIL, add_error_handlers
from request_log import HEADER, JsonFormatter, LogFormat, RequestIdMiddleware, TextFormatter

log = logging.getLogger("tests.request_log")


class _TeapotError(DomainError, status_code=418, detail="No coffee here."):
    pass


def _app() -> FastAPI:
    app = FastAPI()
    add_error_handlers(app)
    app.add_middleware(RequestIdMiddleware)

    @app.get("/id")
    async def the_id():
        return {"id": request_log.current()}

    @app.get("/thread")
    async def in_a_thread():
        return {"id": await asyncio.to_thread(request_log.current)}

    @app.get("/log")
    async def writes_a_line():
        log.info("inside the request", extra={"datasets": 3})
        return {}

    @app.get("/teapot")
    async def expected_failure():
        raise _TeapotError("no coffee")

    @app.get("/bug")
    async def bug():
        raise RuntimeError("secret detail")

    return app


async def _get_async(path: str, headers: dict | None = None) -> httpx.Response:
    # As in production: an exception no handler planned for becomes the 500.
    transport = httpx.ASGITransport(app=_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path, headers=headers)


def _get(path: str, headers: dict | None = None) -> httpx.Response:
    return asyncio.run(_get_async(path, headers))


@pytest.fixture
def lines():
    """What the root logger writes during the test, as JSON lines."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # Without the test client's own lines.
    yield lambda: [line for line in map(json.loads, stream.getvalue().splitlines())
                   if line["logger"] != "httpx"]
    root.removeHandler(handler)
    root.setLevel(level)


# ── The ID ────────────────────────────────────────────────────────────────────

def test_the_answer_names_the_id_the_request_ran_under():
    reply = _get("/id")
    assert len(reply.headers[HEADER]) == 32
    assert reply.json() == {"id": reply.headers[HEADER]}


def test_two_requests_get_different_ids():
    assert _get("/id").headers[HEADER] != _get("/id").headers[HEADER]


def test_a_well_formed_incoming_id_is_kept():
    assert _get("/id", {HEADER: "nginx-7f3a.42_b"}).headers[HEADER] == "nginx-7f3a.42_b"


@pytest.mark.parametrize("incoming", ["", "a b", "x" * 65, 'id"} {"level": "ERROR', "über"])
def test_any_other_incoming_id_is_replaced(incoming):
    answered = _get("/id", {HEADER: incoming.encode("latin-1")}).headers[HEADER]
    assert answered != incoming
    assert len(answered) == 32


def test_a_worker_thread_sees_the_id():
    reply = _get("/thread")
    assert reply.json() == {"id": reply.headers[HEADER]}


def test_no_id_is_left_after_the_request():
    async def ask_then_look():
        await _get_async("/id")
        return request_log.current()

    assert asyncio.run(ask_then_look()) is None


@pytest.mark.parametrize("path, status", [("/teapot", 418), ("/no-such-route", 404)])
def test_an_error_answer_carries_the_id(path, status):
    reply = _get(path)
    assert reply.status_code == status
    assert len(reply.headers[HEADER]) == 32


def test_the_plain_500_carries_the_id(lines):
    reply = _get("/bug")
    assert reply.status_code == 500
    assert reply.json() == {"detail": UNEXPECTED_DETAIL}
    failed = [line for line in lines() if line["logger"] == "request_log"]
    assert [line["request_id"] for line in failed] == [reply.headers[HEADER]]
    assert failed[0]["level"] == "ERROR"
    assert failed[0]["status"] == 500
    assert "RuntimeError: secret detail" in failed[0]["traceback"]


def test_the_500_handler_adds_no_header_without_the_middleware():
    app = FastAPI()
    add_error_handlers(app)

    @app.get("/bug")
    async def bug():
        raise RuntimeError("x")

    async def ask():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/bug")

    reply = asyncio.run(ask())
    assert reply.status_code == 500
    assert HEADER not in reply.headers


# ── The log lines ─────────────────────────────────────────────────────────────

def test_every_line_of_a_request_carries_its_id(lines):
    reply = _get("/log?code=secret")
    inside, done = lines()
    assert inside["request_id"] == done["request_id"] == reply.headers[HEADER]
    assert inside["message"] == "inside the request"
    assert inside["datasets"] == 3  # passed as extra=
    assert done["logger"] == "request_log"
    assert (done["method"], done["path"], done["status"]) == ("GET", "/log", 200)
    assert isinstance(done["duration_ms"], int)
    # The query string is not logged.
    assert "secret" not in json.dumps(done)


def test_a_line_outside_a_request_has_no_id(lines):
    log.info("startup")
    assert lines()[0]["request_id"] is None


def _record(message: str, **extra) -> logging.LogRecord:
    record = logging.LogRecord("some.logger", logging.WARNING, "file.py", 1, message, (), None)
    record.__dict__.update(extra)
    return record


def test_a_json_line_is_one_line_with_the_standard_fields():
    line = JsonFormatter().format(_record("two\nlines", level="mine", n=1))
    assert "\n" not in line
    parsed = json.loads(line)
    assert parsed["message"] == "two\nlines"
    assert parsed["logger"] == "some.logger"
    assert parsed["level"] == "WARNING"  # an extra field cannot replace a standard one
    assert parsed["n"] == 1
    assert parsed["time"].endswith("+00:00")


def test_a_json_line_leaves_out_uvicorns_coloured_copy():
    assert "color_message" not in JsonFormatter().format(_record("x", color_message="\x1b[36mx"))


def test_a_json_line_holds_the_traceback():
    try:
        raise ValueError("boom")
    except ValueError as exc:
        record = _record("failed")
        record.exc_info = (type(exc), exc, exc.__traceback__)
    parsed = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in parsed["traceback"]


def test_a_json_line_takes_a_value_json_cannot_hold():
    parsed = json.loads(JsonFormatter().format(_record("x", when={1, 2}.__class__)))
    assert parsed["when"] == "<class 'set'>"


def test_a_text_line_marks_no_request_with_a_dash():
    assert "  some.logger  [-]  hello" in TextFormatter().format(_record("hello"))


# ── LOG_FORMAT ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, formatter", [
    (None, TextFormatter), ("text", TextFormatter), ("json", JsonFormatter),
])
def test_log_format_chooses_the_lines(monkeypatch, value, formatter):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    if value is not None:
        monkeypatch.setenv("LOG_FORMAT", value)
    monkeypatch.setattr(settings_obj, "LOG_FORMAT", Settings().LOG_FORMAT)
    assert type(LogFormat.from_settings().formatter()) is formatter


@pytest.fixture
def _loggers_put_back():
    """configure() changes loggers of the whole process; undo that."""
    names = ("uvicorn", "uvicorn.access", "sqlalchemy.engine.Engine")
    before = {name: (logging.getLogger(name).handlers[:], logging.getLogger(name).propagate)
              for name in names}
    yield
    for name, (handlers, propagate) in before.items():
        logging.getLogger(name).handlers = handlers
        logging.getLogger(name).propagate = propagate


@pytest.mark.parametrize("name", ["uvicorn", "sqlalchemy.engine.Engine"])
def test_a_logger_with_its_own_handler_writes_json_once(monkeypatch, _loggers_put_back, name):
    # As SQLAlchemy does for an engine that echoes its SQL.
    logging.getLogger(name).addHandler(logging.StreamHandler())
    monkeypatch.setattr(settings_obj, "LOG_FORMAT", "json")

    request_log.configure()

    logger = logging.getLogger(name)
    assert [type(handler.formatter) for handler in logger.handlers] == [JsonFormatter]
    assert logger.propagate is False  # or the root handler would write the line again


def test_uvicorns_access_line_is_dropped(_loggers_put_back):
    logging.getLogger("uvicorn.access").addHandler(logging.StreamHandler())
    request_log.configure()
    assert not logging.getLogger("uvicorn.access").hasHandlers()


@pytest.mark.parametrize("value", ["JSON", "true", ""])
def test_any_other_log_format_stops_the_start(monkeypatch, value):
    monkeypatch.setenv("LOG_FORMAT", value)
    with pytest.raises(ValidationError, match="LOG_FORMAT"):
        Settings()
