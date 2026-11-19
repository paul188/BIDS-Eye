"""
exceptions/domain.py
--------------------
Every error the API answers on purpose.

An expected failure the client should hear about is a subclass of
DomainError, itself an ExpectedError (exceptions/expected.py; an internal
expected failure derives from ExpectedError directly). It is defined next to
the code that raises it, without knowing about FastAPI:

    services/query_guard.py   SearchBusyError           503
    services/search.py        SearchExpiredError        410
    routers/auth.py           NotSignedInError          401
                              OAuthStateError           400
                              GitHubTokenError          400
                              GitHubLoginMissingError   502
                              NotInGitHubOrgError       403
    rate_limits.py            RateLimitError            429

Each class names the status and text the client gets on its class line,

    class SearchExpiredError(DomainError, status_code=410, detail="..."):

and the one handler in
exceptions/handlers.py turns it into that response, so no router catches
any of them and nothing raises HTTPException. (Any other exception gets the
handlers' own 500; see handlers.py.)

The exception's own message (``str(exc)``) is for logs and may name
internals; the client sees only ``detail``. A class whose answer depends on
the case (RateLimitError: which limit, when to retry) passes ``detail`` and
``headers`` when it is raised.
"""
from __future__ import annotations

from typing import Any

from exceptions.expected import ExpectedError


class DomainError(ExpectedError):
    """An expected failure with a fixed answer for the client.

    A subclass gives its answer on the class line (``status_code=...,
    detail=...``); leaving either out is a TypeError when the module is
    imported, not a wrong answer at the first request. DomainError itself
    has no answer.
    """

    status_code: int
    detail: Any
    headers: dict[str, str] | None = None

    def __init_subclass__(cls, *, status_code: int, detail: Any, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.status_code = status_code
        cls.detail = detail

    def __init__(
        self,
        message: str = "",
        *,
        detail: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        if detail is not None:
            self.detail = detail
        if headers is not None:
            self.headers = headers
