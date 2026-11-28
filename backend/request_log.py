"""
request_log.py
--------------
One ID per HTTP request, on every log line written while it is handled.

- RequestIdMiddleware gives each request an ID: the X-Request-ID it came
  with, when that is well formed, or a new one. The ID goes back in the
  response's X-Request-ID header, and the middleware writes one line per
  request: method, path, status, duration. The path is logged without its
  query string (the GitHub callback carries the sign-in code there).
- The ID is kept in a context variable, so code that never sees the request
  logs it too, including the translation in asyncio.to_thread.
- configure() sets up logging for the whole process. LOG_FORMAT chooses the
  lines: "text" (the default) or "json", one object per line. uvicorn's own
  messages and SQLAlchemy's echoed SQL get the same format; uvicorn's access
  line is dropped for the middleware's.

Lines written outside a request (startup, the crawler) have no ID.
The logs name no user.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from config.settings import settings_obj

HEADER = "X-Request-ID"

# An incoming ID is kept only in this shape; anything else is replaced.
_WELL_FORMED = re.compile(r"[A-Za-z0-9._-]{1,64}")

log = logging.getLogger(__name__)

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def current() -> str | None:
    """The ID of the request being handled, or None outside a request."""
    return _request_id.get()


def id_header(scope: Scope) -> dict[str, str]:
    """The response header with the request's ID, for an answer that is sent
    outside the middleware (the plain 500 of exceptions/handlers.py)."""
    request_id = scope.get("state", {}).get("request_id")
    return {HEADER: request_id} if request_id else {}


# ── Log lines ─────────────────────────────────────────────────────────────────

class TextFormatter(logging.Formatter):
    """One readable line; "-" stands for no request."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s  %(levelname)-8s  %(name)s  [%(request_id)s]  %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        record.request_id = current() or "-"
        return super().format(record)


# What every log record has; anything else on a record came in through extra=.
# color_message is uvicorn's copy of its message with terminal colours.
_RECORD_ATTRIBUTES = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName", "request_id", "color_message",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time (UTC), level, logger, request_id,
    message, the fields passed as extra=, and the traceback if there is one."""

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": current(),
            "message": record.getMessage(),
        }
        for name, value in record.__dict__.items():
            if name not in _RECORD_ATTRIBUTES and name not in line:
                line[name] = value
        if record.exc_info:
            line["traceback"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str, ensure_ascii=False)


class LogFormat(str, Enum):
    """How a log line is written; the value is what LOG_FORMAT is set to."""
    TEXT = "text"
    JSON = "json"

    def __str__(self) -> str:
        # The value, on every Python version (format() of a str enum changed in 3.11).
        return self.value

    @classmethod
    def from_settings(cls) -> LogFormat:
        """LOG_FORMAT (config/settings.py, which refuses any other value)."""
        return cls(settings_obj.LOG_FORMAT)

    def formatter(self) -> logging.Formatter:
        return JsonFormatter() if self is LogFormat.JSON else TextFormatter()


def configure() -> None:
    """Set up logging for the process in the format LOG_FORMAT names.

    Called by main.py at import. The uvicorn command sets up its loggers
    before it imports main, so they can be changed here.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(LogFormat.from_settings().formatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    # These write through a handler of their own (SQLAlchemy's when the
    # engine echoes its SQL); they get ours instead, so each line is written
    # once and in our format.
    for name in ("uvicorn", "sqlalchemy.engine.Engine"):
        own = logging.getLogger(name)
        own.handlers = [handler]
        own.propagate = False
    # The middleware writes the line per request.
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False


# ── Middleware ────────────────────────────────────────────────────────────────

class RequestIdMiddleware:
    """Gives each HTTP request its ID and logs the request when it is done.

    Added last in main.py, so it runs first and the lines of every other
    middleware carry the ID.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_id(scope) or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        token = _request_id.set(request_id)
        started = time.perf_counter()
        status: int | None = None

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)[HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            # Answered with the plain 500 after this (exceptions/handlers.py).
            fields = _fields(scope, 500, started)
            log.exception("%s %s -> failed after %d ms",
                          fields["method"], fields["path"], fields["duration_ms"], extra=fields)
            raise
        else:
            fields = _fields(scope, status, started)
            log.info("%s %s -> %s in %d ms",
                     fields["method"], fields["path"], status or "no response",
                     fields["duration_ms"], extra=fields)
        finally:
            _request_id.reset(token)


def _incoming_id(scope: Scope) -> str | None:
    value = Headers(scope=scope).get(HEADER, "")
    return value if _WELL_FORMED.fullmatch(value) else None


def _fields(scope: Scope, status: int | None, started: float) -> dict[str, Any]:
    return {
        "method": scope["method"],
        "path": scope["path"],
        "status": status,
        "duration_ms": round((time.perf_counter() - started) * 1000),
    }
