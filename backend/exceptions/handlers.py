"""
exceptions/handlers.py
----------------------
The one place a DomainError (exceptions/domain.py) becomes an HTTP response.
main.py installs it on the app; tests install it on their own small apps.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from exceptions.domain import DomainError


async def domain_error_response(request: Request, exc: DomainError) -> JSONResponse:
    """The error's fixed answer, in the same ``{"detail": ...}`` body that
    HTTPException gives, so the frontend reads both alike."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def install_domain_error_handler(app: FastAPI) -> None:
    app.add_exception_handler(DomainError, domain_error_response)
