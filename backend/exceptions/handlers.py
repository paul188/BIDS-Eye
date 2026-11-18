"""
exceptions/handlers.py
----------------------
Where an error becomes an HTTP response. main.py installs these on the app;
tests install them on their own small apps.

- DomainError (exceptions/domain.py): the error's own status, ``detail`` and
  headers.
- Any other exception: a bug or an outage nobody planned an answer for. The
  client gets DomainError's default 500 and text, never the exception's
  message. Starlette still raises the exception again after the response is
  sent, so the server logs it with its traceback as before.

FastAPI's own answers stay as they are: 404/405 for unknown routes and 422
for a request body that fails validation.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from exceptions.domain import DomainError


async def domain_error_response(request: Request, exc: DomainError) -> JSONResponse:
    """The error's answer, in the same ``{"detail": ...}`` body that
    FastAPI's own errors give, so the frontend reads both alike."""
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers
    )


async def unexpected_error_response(request: Request, exc: Exception) -> JSONResponse:
    """DomainError's default answer for an exception no class planned for."""
    return JSONResponse(
        status_code=DomainError.status_code, content={"detail": DomainError.detail}
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DomainError, domain_error_response)
    app.add_exception_handler(Exception, unexpected_error_response)
