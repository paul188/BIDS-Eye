"""
exceptions/handlers.py
----------------------
Where an error becomes an HTTP response. main.py adds these to the app;
tests add them to their own small apps.

- DomainError (exceptions/domain.py): the error's own status, ``detail`` and
  headers.
- Any other exception: a bug or an outage nobody planned an answer for. The
  client gets UNEXPECTED_STATUS and UNEXPECTED_DETAIL below, never the
  exception's message. Starlette still raises the exception again after the
  response is sent, so the server logs it with its traceback as before.

FastAPI's own answers stay as they are: 404/405 for unknown routes and 422
for a request body that fails validation.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import request_log
from exceptions.domain import DomainError

UNEXPECTED_STATUS = 500
UNEXPECTED_DETAIL = "The request could not be completed."


async def domain_error_response(request: Request, exc: DomainError) -> JSONResponse:
    """The error's answer, in the same ``{"detail": ...}`` body that
    FastAPI's own errors give, so the frontend reads both alike."""
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers
    )


async def unexpected_error_response(request: Request, exc: Exception) -> JSONResponse:
    """The answer for an exception no class planned for."""
    # Sent outside the request-ID middleware, so the header is added here.
    return JSONResponse(
        status_code=UNEXPECTED_STATUS,
        content={"detail": UNEXPECTED_DETAIL},
        headers=request_log.id_header(request.scope),
    )


def add_error_handlers(app: FastAPI) -> None:
    # Starlette types every handler as taking any Exception; it calls this one
    # only for a DomainError.
    app.add_exception_handler(DomainError, domain_error_response)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unexpected_error_response)
