from __future__ import annotations

import asyncio
import os
import secrets
import time
from collections.abc import MutableMapping
from urllib.parse import quote, urlsplit

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.dialects.postgresql import insert as pg_insert

from exceptions.domain import DomainError
from rate_limits import Category, rate_limit


# ── Settings ──────────────────────────────────────────────────────────────────

# The login lives in Starlette's SessionMiddleware (see api_security.py): a
# signed, HttpOnly cookie that page scripts cannot read. The browser sends it on
# every /api request; the frontend never sees the value.
SESSION_COOKIE = "bids_eye_session"
SESSION_DAYS = 7
_SESSION_MAX_AGE = SESSION_DAYS * 24 * 3600
_OAUTH_STATE_SECONDS = 10 * 60

_PLACEHOLDER_SECRETS = ("change-me", "changeme")

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Errors (answered by exceptions/handlers.py) ───────────────────────────────

class NotSignedInError(DomainError):
    """No signed-in session. RequireLoginMiddleware (api_security.py) sends the
    same answer before routing, where exception handlers do not reach."""

    status_code = 401
    detail = "Not signed in"


class OAuthStateError(DomainError):
    """GitHub returned a state this browser's session was not given, or it expired."""

    status_code = 400
    detail = "Invalid or expired OAuth state."


class GitHubTokenError(DomainError):
    """GitHub did not exchange the login code for an access token."""

    status_code = 400
    detail = "Failed to obtain access token from GitHub."


class GitHubLoginMissingError(DomainError):
    """GitHub's /user answer had no login name."""

    status_code = 502
    detail = "GitHub did not return a login name."


class NotInGitHubOrgError(DomainError):
    """The user is not a member of GITHUB_ORG."""

    status_code = 403
    detail = "Access restricted to members of the required GitHub organization."


# ── Session ───────────────────────────────────────────────────────────────────

def session_secret() -> str:
    """The key that signs the session and CSRF cookies. Refuses to run without a real one.

    A missing or placeholder key would let anyone who reads the repo sign a
    session for any user, so the app must not start instead of falling back.
    """
    secret = os.getenv("JWT_SECRET", "").strip()
    if not secret or secret.lower().startswith(_PLACEHOLDER_SECRETS):
        raise RuntimeError(
            "JWT_SECRET is missing or still a placeholder. Set it to a long random value, "
            'e.g. python3 -c "import secrets; print(secrets.token_urlsafe(64))"'
        )
    return secret


def session_middleware_options() -> dict:
    """Keyword arguments for SessionMiddleware."""
    return {
        "secret_key": session_secret(),
        "session_cookie": SESSION_COOKIE,
        "max_age": _SESSION_MAX_AGE,
        "path": "/api",
        "same_site": "lax",
        "https_only": True,
    }


def _now() -> float:
    return time.time()


def session_user(session: MutableMapping) -> dict | None:
    """The signed-in user stored in a session, or None.

    SessionMiddleware re-issues the cookie on every response, so its own max_age
    only limits idle time. The login time stored in the session caps the total
    lifetime: a copied cookie stops working SESSION_DAYS after the login. An
    expired session is emptied, which makes the middleware delete the cookie.
    """
    user = session.get("user")
    if not isinstance(user, dict) or not user.get("sub"):
        return None
    logged_in_at = user.get("iat")
    if not isinstance(logged_in_at, (int, float)) or _now() - logged_in_at > _SESSION_MAX_AGE:
        session.clear()
        return None
    return user


def require_auth(request: Request) -> dict:
    """The signed-in user, or 401.

    RequireLoginMiddleware already turns away every /api request without a
    session and leaves the user in request.state; this hands it to the route.
    It checks the session itself as well, so a route stays protected even if it
    is ever mounted outside the middleware's reach.
    """
    user = getattr(request.state, "user", None) or session_user(request.session)
    if user is None:
        raise NotSignedInError()
    return user


# ── GitHub login ──────────────────────────────────────────────────────────────

def _github_redirect_uri(request: Request) -> str:
    return os.getenv("GITHUB_REDIRECT_URI") or str(request.url_for("github_callback"))


@router.get("/github", dependencies=[rate_limit(Category.LOGIN)])
async def github_login(request: Request):
    redirect_uri = _github_redirect_uri(request)

    # The OAuth state lives in the session cookie, and cookies belong to one
    # host. If the login starts on another host than the one GitHub returns to
    # (www.bidseye.xyz vs bidseye.xyz), restart it on the callback's host first,
    # or the callback would not see the state. One hop at most.
    target = urlsplit(redirect_uri)
    if (
        target.netloc
        and target.netloc.lower() != request.headers.get("host", "").lower()
        and "hop" not in request.query_params
    ):
        return RedirectResponse(f"{target.scheme}://{target.netloc}{request.url.path}?hop=1")

    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = {"value": state, "exp": _now() + _OAUTH_STATE_SECONDS}

    auth_url = (
        "https://github.com/login/oauth/authorize"
        f"?client_id={quote(os.getenv('GITHUB_CLIENT_ID', ''))}"
        f"&redirect_uri={quote(redirect_uri, safe='')}"
        f"&scope={quote('read:org user:email')}"
        f"&state={state}"
    )
    return RedirectResponse(auth_url)


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
    if not isinstance(value, str) or not isinstance(exp, (int, float)) or _now() > exp:
        return False
    return secrets.compare_digest(value, returned)


def _json(response) -> object:
    """The answer's JSON body, or None when it is not JSON."""
    try:
        return response.json()
    except ValueError:
        return None


def _github_login(user_resp) -> str:
    """The account's login from GitHub's /user answer; '' when GitHub answered with an error.

    An error answer ({"message": "Bad credentials"}) has no login.
    """
    data = _json(user_resp) if user_resp.status_code == 200 else None
    login = data.get("login") if isinstance(data, dict) else None
    return login.strip() if isinstance(login, str) else ""


def _primary_email(emails_resp) -> str | None:
    """The verified primary address from GitHub's /user/emails answer, else None.

    Without the user:email scope, or on an error, GitHub answers with an
    object instead of a list; the sign-in then goes on without an address.
    """
    data = _json(emails_resp) if emails_resp.status_code == 200 else None
    if not isinstance(data, list):
        return None
    return next(
        (e["email"] for e in data
         if isinstance(e, dict) and e.get("primary") and e.get("verified")
         and isinstance(e.get("email"), str)),
        None,
    )


@router.get("/github/callback", name="github_callback", dependencies=[rate_limit(Category.LOGIN)])
async def github_callback(request: Request):
    code = request.query_params.get("code", "")

    if not _take_oauth_state(request, request.query_params.get("state", "")):
        raise OAuthStateError()

    # Exchange code for GitHub access token
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": os.getenv("GITHUB_CLIENT_ID"),
                "client_secret": os.getenv("GITHUB_CLIENT_SECRET"),
                "code": code,
                "redirect_uri": _github_redirect_uri(request),
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
    token_data = _json(token_resp)
    access_token = token_data.get("access_token") if isinstance(token_data, dict) else None
    if not access_token:
        raise GitHubTokenError(f"token exchange answered {token_resp.status_code}")

    gh_headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    async with httpx.AsyncClient() as client:
        user_resp, emails_resp = await asyncio.gather(
            client.get("https://api.github.com/user", headers=gh_headers, timeout=10),
            client.get("https://api.github.com/user/emails", headers=gh_headers, timeout=10),
        )

    github_login_name = _github_login(user_resp)
    if not github_login_name:
        raise GitHubLoginMissingError(f"/user answered {user_resp.status_code}")
    primary_email = _primary_email(emails_resp)

    github_org = os.getenv("GITHUB_ORG", "").strip()
    if github_org:
        async with httpx.AsyncClient() as client:
            membership = await client.get(
                f"https://api.github.com/orgs/{github_org}/members/{github_login_name}",
                headers=gh_headers,
                timeout=10,
            )
        if membership.status_code != 204:
            raise NotInGitHubOrgError(f"{github_login_name} not in {github_org}")

    await _remember_user(github_login_name, primary_email)

    # A fresh session for the new login: nothing from before it carries over.
    request.session.clear()
    request.session["user"] = {
        "sub": github_login_name,
        "email": primary_email or "",
        "iat": int(_now()),
    }

    # Relative on purpose: the cookie was just set on this host, so the app
    # must open on this host too.
    return RedirectResponse("/")


async def _remember_user(github_login: str, email: str | None) -> None:
    from db.db import async_session_maker
    from models import User

    async with async_session_maker() as session:
        await session.execute(
            pg_insert(User)
            .values(github_login=github_login, github_email=email)
            .on_conflict_do_update(
                index_elements=["github_login"],
                set_={"github_email": email},
            )
        )
        await session.commit()


# ── Signed-in user ────────────────────────────────────────────────────────────

@router.get("/me", dependencies=[rate_limit(Category.LIGHT)])
async def me(user: dict = Depends(require_auth)):
    return {"login": user["sub"]}


@router.post("/logout", dependencies=[rate_limit(Category.LIGHT)])
async def logout(request: Request):
    request.session.clear()
    return {"ok": True}
