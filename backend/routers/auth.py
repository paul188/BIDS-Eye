from __future__ import annotations

import asyncio
import os
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.dialects.postgresql import insert as pg_insert

_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
_ALGORITHM = "HS256"
_EXPIRE_DAYS = 7

router = APIRouter(prefix="/auth", tags=["auth"])
_bearer = HTTPBearer()

# In-memory state store for OAuth CSRF protection.
# Maps state token → expiry timestamp. Single-instance only (fine for this deployment).
_oauth_states: dict[str, float] = {}
_STATE_TTL_MINUTES = 10


def _make_token(sub: str, email: str = "") -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=_EXPIRE_DAYS)
    payload: dict = {"sub": sub, "exp": exp}
    if email:
        payload["email"] = email
    return jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)


def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    try:
        payload = jwt.decode(
            credentials.credentials, _SECRET, algorithms=[_ALGORITHM]
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload


def _new_state() -> str:
    """Generate a fresh CSRF state token and register it with a TTL."""
    # Prune expired states to keep the dict small
    now = datetime.now(timezone.utc).timestamp()
    expired = [k for k, exp in _oauth_states.items() if exp < now]
    for k in expired:
        del _oauth_states[k]

    state = secrets.token_urlsafe(24)
    _oauth_states[state] = (
        datetime.now(timezone.utc) + timedelta(minutes=_STATE_TTL_MINUTES)
    ).timestamp()
    return state


def _consume_state(state: str) -> bool:
    """Validate and remove a state token. Returns False if invalid or expired."""
    expiry = _oauth_states.pop(state, None)
    if expiry is None:
        return False
    return datetime.now(timezone.utc).timestamp() <= expiry


@router.get("/github")
async def github_login(request: Request):
    client_id = os.getenv("GITHUB_CLIENT_ID", "")
    redirect_uri = os.getenv(
        "GITHUB_REDIRECT_URI",
        str(request.url_for("github_callback")),
    )
    state = _new_state()
    auth_url = (
        "https://github.com/login/oauth/authorize"
        f"?client_id={quote(client_id)}"
        f"&redirect_uri={quote(redirect_uri, safe='')}"
        f"&scope={quote('read:org user:email')}"
        f"&state={state}"
    )
    return RedirectResponse(auth_url)


@router.get("/github/callback", name="github_callback")
async def github_callback(request: Request):
    state = request.query_params.get("state", "")
    code = request.query_params.get("code", "")

    if not _consume_state(state):
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state.")

    # Exchange code for GitHub access token
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": os.getenv("GITHUB_CLIENT_ID"),
                "client_secret": os.getenv("GITHUB_CLIENT_SECRET"),
                "code": code,
                "redirect_uri": os.getenv(
                    "GITHUB_REDIRECT_URI",
                    str(request.url_for("github_callback")),
                ),
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )
    access_token = token_resp.json().get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="Failed to obtain access token from GitHub.")

    gh_headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    async with httpx.AsyncClient() as client:
        user_resp, emails_resp = await asyncio.gather(
            client.get("https://api.github.com/user", headers=gh_headers, timeout=10),
            client.get("https://api.github.com/user/emails", headers=gh_headers, timeout=10),
        )

    user_data = user_resp.json()
    github_login_name: str = user_data.get("login", "unknown")
    primary_email = next(
        (e["email"] for e in emails_resp.json() if e.get("primary") and e.get("verified")),
        None,
    )

    github_org = os.getenv("GITHUB_ORG", "").strip()
    if github_org:
        async with httpx.AsyncClient() as client:
            membership = await client.get(
                f"https://api.github.com/orgs/{github_org}/members/{github_login_name}",
                headers=gh_headers,
                timeout=10,
            )
        if membership.status_code != 204:
            raise HTTPException(
                status_code=403,
                detail="Access restricted to members of the required GitHub organization.",
            )

    from db.db import async_session_maker
    from models import User

    async with async_session_maker() as session:
        await session.execute(
            pg_insert(User)
            .values(github_login=github_login_name, github_email=primary_email)
            .on_conflict_do_update(
                index_elements=["github_login"],
                set_={"github_email": primary_email},
            )
        )
        await session.commit()

    jwt_token = _make_token(sub=github_login_name, email=primary_email or "")

    frontend_url = os.getenv("FRONTEND_URL", "").strip().rstrip("/")
    if not frontend_url or frontend_url == "*":
        frontend_url = str(request.base_url).rstrip("/")

    return RedirectResponse(f"{frontend_url}/?token={jwt_token}")


@router.post("/logout")
async def logout():
    return {"ok": True}
