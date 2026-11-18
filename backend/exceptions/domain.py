"""
exceptions/domain.py
--------------------
Every error the API answers on purpose.

Code that can fail in an expected way raises a subclass of DomainError,
defined next to that code and without knowing about FastAPI:

    services/query_guard.py   SearchBusyError           503
    services/search.py        SearchExpiredError        410
    routers/auth.py           NotSignedInError          401
                              OAuthStateError           400
                              GitHubTokenError          400
                              GitHubLoginMissingError   502
                              NotInGitHubOrgError       403
    rate_limits.py            RateLimitError            429

Each class names the status and text the client gets; the one handler in
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


class DomainError(Exception):
    """An expected failure with a fixed answer for the client.

    The class attributes are the answer. DomainError itself has none: each
    subclass sets ``status_code`` and ``detail``; ``headers`` is optional.
    """

    status_code: int
    detail: Any
    headers: dict[str, str] | None = None

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
