"""
sessions.py
-----------
The signed-in session: what its cookie holds, how long it lasts, and how to
read the user from it. api_security.py builds the middleware from it,
deps.py hands the user to routes, routers/auth.py writes it at sign-in.
"""

from __future__ import annotations

import time
from collections.abc import MutableMapping

from exceptions.domain import DomainError

# The login lives in Starlette's SessionMiddleware (see api_security.py): a
# signed, HttpOnly cookie that page scripts cannot read. The browser sends it on
# every /api request; the frontend never sees the value.
#
# Signing in stores nothing on the server: the cookie holds the GitHub login,
# GitHub's number for the account and the sign-in time, and nothing else is
# asked from GitHub or stored (no e-mail address, no users table). The site's
# privacy notice says so (frontend/src/components/CookieBanner.vue);
# tests/test_auth_session.py checks it. The account number is what a user's
# saved searches are kept under, once they turn that on
# (services/stores/search_history.py).
#
# Nothing is kept on the server, so a session cannot be withdrawn one by one:
# - GITHUB_ORG membership is checked at sign-in only. Someone removed from the
#   organization keeps access until their cookie expires, at most SESSION_DAYS.
# - Logging out deletes the cookie in that browser; a copy of it stays valid
#   until it expires.
# Shorten SESSION_DAYS if that is too long. To sign everyone out at once,
# change SESSION_SECRET and restart the backend (README, "Sign-in").
SESSION_COOKIE = "bids_eye_session"
SESSION_DAYS = 7
_SESSION_MAX_AGE = SESSION_DAYS * 24 * 3600


class NotSignedInError(DomainError, status_code=401, detail="Not signed in"):
    """No signed-in session. RequireLoginMiddleware (api_security.py) sends the
    same answer before routing, where exception handlers do not reach."""


def session_middleware_options(secret: str) -> dict:
    """Keyword arguments for SessionMiddleware."""
    return {
        "secret_key": secret,
        "session_cookie": SESSION_COOKIE,
        "max_age": _SESSION_MAX_AGE,
        "path": "/api",
        "same_site": "lax",
        "https_only": True,
    }


def session_user(session: MutableMapping) -> dict | None:
    """The signed-in user stored in a session, or None.

    A session ends SESSION_DAYS after the login however much it is used, so a
    copied cookie stops working then. SessionMiddleware already ends it: it
    signs the cookie only when the session changes (Starlette 1.x) and refuses
    a signature older than max_age. The login time stored in the session keeps
    the cap should the middleware ever re-sign on every response, as older
    Starlette did. An expired session is emptied, which makes the middleware
    delete the cookie.
    """
    user = session.get("user")
    if not isinstance(user, dict) or not user.get("sub"):
        return None
    logged_in_at = user.get("iat")
    if not isinstance(logged_in_at, (int, float)) or time.time() - logged_in_at > _SESSION_MAX_AGE:
        session.clear()
        return None
    return user
