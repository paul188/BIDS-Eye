"""The API's security middleware, built in one place for main.py and the tests.

Requests pass through, outermost first:

    CORS -> CSRF -> Session -> RequireLogin -> routes

- CORS: only an explicit FRONTEND_URL origin may make credentialed
  cross-origin calls. With FRONTEND_URL unset or "*" there is no CORS at all;
  the frontend calls /api on its own origin through nginx and needs none.
- CSRF (starlette-csrf, double-submit cookie): once a browser holds a session,
  every POST must repeat the readable CSRF cookie in the x-csrftoken header.
  Another site can make the browser send cookies but cannot read them, so it
  cannot produce the header. This is on top of the session's SameSite=Lax.
- Session: the signed HttpOnly login cookie (see routers/auth.py).
- RequireLogin: deny by default. Every /api path needs a signed-in session
  unless it is listed in PUBLIC_API_PATHS, so a new router is protected
  without anyone having to remember Depends(require_auth).
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette_csrf import CSRFMiddleware

from routers.auth import (
    SESSION_COOKIE,
    AuthSettings,
    NotSignedInError,
    session_middleware_options,
    session_user,
)

CSRF_COOKIE = "bids_eye_csrf"
CSRF_HEADER = "x-csrftoken"

API_PREFIX = "/api"
# Reachable without a session: starting and finishing the GitHub login, and
# logging out (harmless without a session, and the frontend may call it after
# the session already expired).
PUBLIC_API_PATHS = frozenset({
    "/api/auth/github",
    "/api/auth/github/callback",
    "/api/auth/logout",
})


class RequireLoginMiddleware:
    """401 for every /api request without a signed-in session, except PUBLIC_API_PATHS.

    Must sit inside SessionMiddleware, which provides scope["session"]. The user
    is left in request.state.user for require_auth.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _needs_login(scope["path"]):
            await self.app(scope, receive, send)
            return
        user = session_user(scope["session"])
        if user is None:
            # Before routing, so exception handlers do not reach here: send
            # NotSignedInError's answer directly instead of raising it.
            response = JSONResponse({"detail": NotSignedInError.detail}, status_code=NotSignedInError.status_code)
            await response(scope, receive, send)
            return
        scope.setdefault("state", {})["user"] = user
        await self.app(scope, receive, send)


def _needs_login(path: str) -> bool:
    in_api = path == API_PREFIX or path.startswith(API_PREFIX + "/")
    return in_api and path not in PUBLIC_API_PATHS


def install_security_middleware(app: FastAPI) -> None:
    """Add the middleware stack. Raises when SESSION_SECRET is missing or too short."""
    secret = AuthSettings().SESSION_SECRET.get_secret_value()

    # add_middleware wraps the app: the last one added runs first.
    app.add_middleware(RequireLoginMiddleware)
    app.add_middleware(SessionMiddleware, **session_middleware_options(secret))
    app.add_middleware(
        CSRFMiddleware,
        secret=secret,
        cookie_name=CSRF_COOKIE,
        header_name=CSRF_HEADER,
        # Checked only when the browser holds a session: a signed-out POST has
        # nothing to forge and gets its 401 from RequireLogin instead.
        sensitive_cookies={SESSION_COOKIE},
        cookie_path="/",  # read by the page's script, so not HttpOnly by design
        cookie_secure=True,
        cookie_samesite="strict",
    )

    frontend_url = os.getenv("FRONTEND_URL", "").strip().rstrip("/")
    if frontend_url and frontend_url != "*":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[frontend_url],
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", CSRF_HEADER],
        )
