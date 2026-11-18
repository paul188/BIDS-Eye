"""
exceptions/domain.py
--------------------
Errors of the search that end the request with a fixed HTTP answer.

A service raises its error (query_guard.SearchBusy, search.SearchExpired)
without knowing about FastAPI; each error class names the status and text
the client gets, and the one handler in main.py turns every DomainError into
that response. The routers catch none of them.

The exception's own message (``str(exc)``) is for logs and may name
internals; the client sees only ``detail``.
"""
from __future__ import annotations


class DomainError(Exception):
    """An expected failure with a fixed answer for the client."""

    status_code: int = 500
    detail: str = "The request could not be completed."
