"""
exceptions/expected.py
----------------------
The base of every failure the backend expects.

An expected failure can happen while the code is right: the database is
unreachable, a generated query is refused, a search expired. Its class is
defined next to the code that raises it, wraps the low-level error it comes
from (``raise WorkerDbError(...) from exc``), and is the only thing its
callers catch. Everything else that is raised is a bug and is not caught.

Two kinds:

    DomainError (exceptions/domain.py)   the client gets its own status and
                                         text (SearchBusyError, 503, ...)
    ExpectedError directly               internal; a caller handles it and
                                         the client never sees it
                                         (WorkerDbError, QueryRefusedError)

An internal one that escapes its callers reaches the client as the plain 500
of exceptions/handlers.py, like a bug: nobody planned an answer for it.

A boundary that has to outlive any expected failure (a background loop)
catches ExpectedError, not Exception, so bugs still show.
tests/test_expected_errors.py checks that every exception class in the
backend derives from ExpectedError.
"""
from __future__ import annotations


class ExpectedError(Exception):
    """A failure that can happen while the code is right."""
