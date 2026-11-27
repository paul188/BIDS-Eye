from __future__ import annotations

import logging
import secrets
import time
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from deps import CurrentUserDep
from exceptions.domain import DomainError
from security.rate_limits import Category, rate_limit
from security.settings import AuthSettings
from services import github_oauth

logger = logging.getLogger(__name__)


# ── Settings ──────────────────────────────────────────────────────────────────

# What the session cookie holds and how long it lasts: security/sessions.py.
_OAUTH_STATE_SECONDS = 10 * 60


def _settings(request: Request) -> AuthSettings:
    """The settings install_security_middleware (security/middleware.py) read at startup."""
    return request.app.state.auth_settings


auth_router = APIRouter(prefix="/auth", tags=["auth"])


# ── Errors (answered by exceptions/handlers.py) ───────────────────────────────

class OAuthStateError(DomainError, status_code=400, detail="Invalid or expired OAuth state."):
    """GitHub returned a state this browser's session was not given, or it expired."""


# ── GitHub login (the GitHub calls: services/github_oauth.py) ─────────────────

# Where the browser lands when a sign-in ends at GitHub without a code:
# /?login=<reason>. LoginView.vue has the sentence for each reason.
LOGIN_CANCELLED = "cancelled"
LOGIN_REFUSED = "refused"


def _github_redirect_uri(request: Request) -> str:
    return _settings(request).GITHUB_REDIRECT_URI or str(request.url_for("github_callback"))


@auth_router.get("/github", dependencies=[rate_limit(Category.LOGIN)])
async def github_login(request: Request):
    redirect_uri = _github_redirect_uri(request)

    # The OAuth state lives in the session cookie, and cookies belong to one
    # host. If the login starts on another host than the one GitHub returns to
    # (www.bidseye.xyz vs bidseye.xyz), restart it on the callback's host first,
    # or the callback would not see the state. One hop at most: behind a proxy
    # that rewrites Host every login takes the hop, and no more than that.
    target = urlsplit(redirect_uri)
    if (
        target.netloc
        and target.netloc.lower() != request.headers.get("host", "").lower()
        and "hop" not in request.query_params
    ):
        return RedirectResponse(f"{target.scheme}://{target.netloc}{request.url.path}?hop=1")

    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = {"value": state, "exp": time.time() + _OAUTH_STATE_SECONDS}

    return RedirectResponse(github_oauth.authorize_url(_settings(request), redirect_uri, state))


def _take_oauth_state(request: Request, returned: str) -> bool:
    """Whether GitHub returned the state this browser's session was given.

    Single use: the state leaves the session whatever the outcome. Binding it to
    the session also means a callback link made in someone else's browser
    cannot sign this browser into their account.
    """
    expected = request.session.pop("oauth_state", None)
    if not isinstance(expected, dict) or not returned:
        return False
    value, exp = expected.get("value"), expected.get("exp")
    if not isinstance(value, str) or not isinstance(exp, (int, float)) or time.time() > exp:
        return False
    return secrets.compare_digest(value, returned)


@auth_router.get("/github/callback", name="github_callback", dependencies=[rate_limit(Category.LOGIN)])
async def github_callback(request: Request):
    settings = _settings(request)
    params = request.query_params
    state_ok = _take_oauth_state(request, params.get("state", ""))

    # GitHub returns ?error=... and no code when the user pressed "Cancel" on
    # its consent screen (access_denied), or when it refuses the app's request.
    # Nobody is signed in either way; back to the login page with the reason.
    error = params.get("error", "")
    if error:
        if error != "access_denied":
            logger.warning("GitHub refused the sign-in: %r", error[:80])
        reason = LOGIN_CANCELLED if error == "access_denied" else LOGIN_REFUSED
        return RedirectResponse(f"/?login={reason}")

    if not state_ok:
        raise OAuthStateError()

    account = await github_oauth.sign_in(
        settings, params.get("code", ""), _github_redirect_uri(request)
    )

    # A fresh session for the new login: nothing from before it carries over.
    request.session.clear()
    user = {"sub": account.login, "iat": int(time.time())}
    if account.account_id is not None:
        user["uid"] = account.account_id
    request.session["user"] = user

    # Relative on purpose: the cookie was just set on this host, so the app
    # must open on this host too.
    return RedirectResponse("/")


# ── Signed-in user ────────────────────────────────────────────────────────────

@auth_router.get("/me", dependencies=[rate_limit(Category.LIGHT)])
async def me(user: CurrentUserDep):
    return {"login": user["sub"]}


@auth_router.post("/logout", dependencies=[rate_limit(Category.LIGHT)])
async def logout(request: Request):
    request.session.clear()
    return {"ok": True}
