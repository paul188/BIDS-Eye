from __future__ import annotations

import logging
import secrets
import time
from collections.abc import MutableMapping
from typing import Annotated
from urllib.parse import quote, urlsplit

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from exceptions.domain import DomainError
from rate_limits import Category, rate_limit

logger = logging.getLogger(__name__)


# ── Settings ──────────────────────────────────────────────────────────────────

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
_OAUTH_STATE_SECONDS = 10 * 60


class AuthSettings(BaseSettings):
    """Read from the environment once, when api_security.py builds the middleware.

    A missing or contradictory value stops the app from starting, instead of
    showing up at the first sign-in or, worse, not at all.
    """

    model_config = SettingsConfigDict(str_strip_whitespace=True, hide_input_in_errors=True)

    # Signs the session and CSRF cookies. With a missing or guessable key anyone
    # who reads the repo could sign a session for any user, so the app refuses
    # to start without one of at least 32 characters (.env.example's
    # placeholder is shorter). Generate one with
    #   python3 -c "import secrets; print(secrets.token_urlsafe(64))"
    SESSION_SECRET: SecretStr = Field(min_length=32)

    # The GitHub OAuth App the sign-in goes through.
    GITHUB_CLIENT_ID: str = Field(min_length=1)
    GITHUB_CLIENT_SECRET: SecretStr = Field(min_length=1)
    # The callback URL registered in the OAuth App. Set it in production: behind
    # a proxy that ends TLS or rewrites Host, the URL built from the request
    # can come out as http:// or on the wrong host.
    GITHUB_REDIRECT_URI: str = ""

    # Who may sign in: members of GITHUB_ORG, or every GitHub account. The
    # second has to be asked for by name, so that a lost or misspelt GITHUB_ORG
    # stops the start instead of quietly opening the site to everyone.
    GITHUB_ORG: str = ""
    ALLOW_ANY_GITHUB_USER: bool = False

    @field_validator("GITHUB_CLIENT_SECRET")
    @classmethod
    def _secret_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _org_or_explicit_open(self) -> AuthSettings:
        if not self.GITHUB_ORG and not self.ALLOW_ANY_GITHUB_USER:
            raise ValueError(
                "Set GITHUB_ORG, or ALLOW_ANY_GITHUB_USER=true to let every GitHub account sign in"
            )
        if self.GITHUB_ORG and self.ALLOW_ANY_GITHUB_USER:
            raise ValueError(
                "GITHUB_ORG and ALLOW_ANY_GITHUB_USER=true contradict each other; set only one"
            )
        return self


def _settings(request: Request) -> AuthSettings:
    """The settings install_security_middleware (api_security.py) read at startup."""
    return request.app.state.auth_settings


auth_router = APIRouter(prefix="/auth", tags=["auth"])


# ── Errors (answered by exceptions/handlers.py) ───────────────────────────────

class NotSignedInError(DomainError, status_code=401, detail="Not signed in"):
    """No signed-in session. RequireLoginMiddleware (api_security.py) sends the
    same answer before routing, where exception handlers do not reach."""


class OAuthStateError(DomainError, status_code=400, detail="Invalid or expired OAuth state."):
    """GitHub returned a state this browser's session was not given, or it expired."""


class GitHubTokenError(
    DomainError, status_code=400, detail="Failed to obtain access token from GitHub."
):
    """GitHub did not exchange the login code for an access token."""


class GitHubLoginMissingError(
    DomainError, status_code=502, detail="GitHub did not return a login name."
):
    """GitHub's /user answer had no login name."""


class NotInGitHubOrgError(
    DomainError,
    status_code=403,
    detail="Access restricted to members of the required GitHub organization.",
):
    """The user is not a member of GITHUB_ORG."""


class GitHubOrgCheckError(
    DomainError,
    status_code=502,
    detail="GitHub could not confirm the organization membership. Please try again later.",
):
    """GitHub answered the membership check with neither yes nor no.

    An outage (5xx), a rate limit, or a 403 because the organization restricts
    third-party apps and has not approved this OAuth App (see .env.example).
    """


class GitHubUnreachableError(
    DomainError, status_code=502, detail="GitHub could not be reached. Please try again."
):
    """A call to GitHub timed out or the connection failed."""


# ── Session ───────────────────────────────────────────────────────────────────

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


def user_id(user: dict) -> int | None:
    """GitHub's number for the signed-in account; None in a session from
    before sign-in kept it (at most SESSION_DAYS old).

    Unlike the login it never changes hands: a login can be renamed, and a
    released one can be registered by someone else.
    """
    uid = user.get("uid")
    return uid if isinstance(uid, int) and not isinstance(uid, bool) else None


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


# What a route writes to get the signed-in user: ``user: CurrentUserDep``.
CurrentUserDep = Annotated[dict, Depends(require_auth)]


# ── GitHub login ──────────────────────────────────────────────────────────────

GITHUB_SCOPE = "read:org"
_GITHUB_TIMEOUT_SECONDS = 15

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

    auth_url = (
        "https://github.com/login/oauth/authorize"
        f"?client_id={quote(_settings(request).GITHUB_CLIENT_ID, safe='')}"
        f"&redirect_uri={quote(redirect_uri, safe='')}"
        # read:org only for the GITHUB_ORG membership check; the login name
        # itself needs no scope.
        f"&scope={quote(GITHUB_SCOPE)}"
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
    if not isinstance(value, str) or not isinstance(exp, (int, float)) or time.time() > exp:
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


def _github_account_id(user_resp) -> int | None:
    """The account's number from GitHub's /user answer, or None."""
    data = _json(user_resp) if user_resp.status_code == 200 else None
    account_id = data.get("id") if isinstance(data, dict) else None
    return account_id if isinstance(account_id, int) and not isinstance(account_id, bool) else None


async def _exchange_code(
    github: httpx.AsyncClient, settings: AuthSettings, code: str, redirect_uri: str
) -> str:
    """The access token GitHub gives for the login code."""
    token_resp = await github.post(
        "https://github.com/login/oauth/access_token",
        json={
            "client_id": settings.GITHUB_CLIENT_ID,
            "client_secret": settings.GITHUB_CLIENT_SECRET.get_secret_value(),
            "code": code,
            "redirect_uri": redirect_uri,
        },
        headers={"Accept": "application/json"},
    )
    token_data = _json(token_resp)
    access_token = token_data.get("access_token") if isinstance(token_data, dict) else None
    if not isinstance(access_token, str) or not access_token:
        raise GitHubTokenError(f"token exchange answered {token_resp.status_code}")
    return access_token


async def _require_org_member(
    github: httpx.AsyncClient, gh_headers: dict, org: str, login: str
) -> None:
    """Raise unless GitHub says ``login`` is a member of ``org``.

    204 is yes. 404 is no; so is 302, GitHub's answer when the account asking
    is itself outside the organization. Anything else is no answer at all, and
    is not held against the user.
    """
    membership = await github.get(
        f"https://api.github.com/orgs/{quote(org, safe='')}/members/{quote(login, safe='')}",
        headers=gh_headers,
    )
    if membership.status_code == 204:
        return
    if membership.status_code in (302, 404):
        raise NotInGitHubOrgError(f"{login} not in {org}")
    raise GitHubOrgCheckError(f"membership check for {org} answered {membership.status_code}")


async def _revoke_token(
    github: httpx.AsyncClient, settings: AuthSettings, access_token: str
) -> None:
    """Have GitHub delete the access token: the sign-in is done with it.

    Otherwise it would stay valid at GitHub with read:org although nothing
    here keeps it. A failure is logged and does not undo the sign-in.
    """
    try:
        revoked = await github.request(
            "DELETE",
            f"https://api.github.com/applications/{quote(settings.GITHUB_CLIENT_ID, safe='')}/token",
            auth=(settings.GITHUB_CLIENT_ID, settings.GITHUB_CLIENT_SECRET.get_secret_value()),
            json={"access_token": access_token},
            headers={"Accept": "application/vnd.github+json"},
        )
    except httpx.HTTPError as exc:
        logger.warning("GitHub access token not revoked: %s", type(exc).__name__)
        return
    if revoked.status_code != 204:
        logger.warning("GitHub access token not revoked: answered %s", revoked.status_code)


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

    try:
        async with httpx.AsyncClient(timeout=_GITHUB_TIMEOUT_SECONDS) as github:
            access_token = await _exchange_code(
                github, settings, params.get("code", ""), _github_redirect_uri(request)
            )
            try:
                gh_headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                }
                user_resp = await github.get("https://api.github.com/user", headers=gh_headers)
                github_login_name = _github_login(user_resp)
                github_account_id = _github_account_id(user_resp)
                if not github_login_name:
                    raise GitHubLoginMissingError(f"/user answered {user_resp.status_code}")
                if settings.GITHUB_ORG:
                    await _require_org_member(
                        github, gh_headers, settings.GITHUB_ORG, github_login_name
                    )
            finally:
                await _revoke_token(github, settings, access_token)
    except httpx.HTTPError as exc:
        raise GitHubUnreachableError(type(exc).__name__) from exc

    # A fresh session for the new login: nothing from before it carries over.
    request.session.clear()
    user = {"sub": github_login_name, "iat": int(time.time())}
    if github_account_id is not None:
        user["uid"] = github_account_id
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
