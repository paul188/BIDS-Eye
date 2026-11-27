"""
services/github_oauth.py
------------------------
The GitHub side of the sign-in: the authorize URL, then for the code GitHub
returns, the access token, the account's login and number, the GITHUB_ORG
membership check, and revoking the token again. Every answer from GitHub is
checked before it is used; a missing or odd one becomes one of the errors
below. routers/auth.py keeps the routes, the OAuth state and the session.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from exceptions.domain import DomainError
from security.settings import AuthSettings

logger = logging.getLogger(__name__)

GITHUB_SCOPE = "read:org"
_GITHUB_TIMEOUT_SECONDS = 15


# ── Errors (answered by exceptions/handlers.py) ───────────────────────────────

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


# ── Signing in ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GitHubAccount:
    login: str
    account_id: int | None  # GitHub's number for the account; None if it sent none


def authorize_url(settings: AuthSettings, redirect_uri: str, state: str) -> str:
    """Where the browser goes to sign in at GitHub."""
    return (
        "https://github.com/login/oauth/authorize"
        f"?client_id={quote(settings.GITHUB_CLIENT_ID, safe='')}"
        f"&redirect_uri={quote(redirect_uri, safe='')}"
        # read:org only for the GITHUB_ORG membership check; the login name
        # itself needs no scope.
        f"&scope={quote(GITHUB_SCOPE)}"
        f"&state={state}"
    )


async def sign_in(settings: AuthSettings, code: str, redirect_uri: str) -> GitHubAccount:
    """The account behind the login ``code``, once GitHub has vouched for it.

    Raises GitHubTokenError, GitHubLoginMissingError, NotInGitHubOrgError,
    GitHubOrgCheckError or GitHubUnreachableError. The access token is revoked
    whatever the outcome once it was obtained.
    """
    try:
        async with httpx.AsyncClient(timeout=_GITHUB_TIMEOUT_SECONDS) as github:
            access_token = await _exchange_code(github, settings, code, redirect_uri)
            try:
                gh_headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                }
                user_resp = await github.get("https://api.github.com/user", headers=gh_headers)
                login = _github_login(user_resp)
                if not login:
                    raise GitHubLoginMissingError(f"/user answered {user_resp.status_code}")
                if settings.GITHUB_ORG:
                    await _require_org_member(github, gh_headers, settings.GITHUB_ORG, login)
                return GitHubAccount(login, _github_account_id(user_resp))
            finally:
                await _revoke_token(github, settings, access_token)
    except httpx.HTTPError as exc:
        raise GitHubUnreachableError(type(exc).__name__) from exc


# ── GitHub's answers ──────────────────────────────────────────────────────────

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
