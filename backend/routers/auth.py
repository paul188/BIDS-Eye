from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
_ALGORITHM = "HS256"
_EXPIRE_DAYS = 7

router = APIRouter(prefix="/auth", tags=["auth"])
_bearer = HTTPBearer()

oauth = OAuth()
oauth.register(
    name="github",
    client_id=os.getenv("GITHUB_CLIENT_ID"),
    client_secret=os.getenv("GITHUB_CLIENT_SECRET"),
    authorize_url="https://github.com/login/oauth/authorize",
    access_token_url="https://github.com/login/oauth/access_token",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "read:org user:email"},
)


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


@router.get("/github")
async def github_login(request: Request):
    redirect_uri = os.getenv(
        "GITHUB_REDIRECT_URI",
        str(request.url_for("github_callback")),
    )
    return await oauth.github.authorize_redirect(request, redirect_uri)


@router.get("/github/callback", name="github_callback")
async def github_callback(request: Request):
    token = await oauth.github.authorize_access_token(request)

    user_resp = await oauth.github.get("user", token=token)
    user_data = user_resp.json()
    github_login_name: str = user_data.get("login", "unknown")

    emails_resp = await oauth.github.get("user/emails", token=token)
    emails = emails_resp.json()
    primary_email = next(
        (e["email"] for e in emails if e.get("primary") and e.get("verified")),
        None,
    )

    github_org = os.getenv("GITHUB_ORG", "").strip()
    if github_org:
        membership_resp = await oauth.github.get(
            f"orgs/{github_org}/members/{github_login_name}", token=token
        )
        if membership_resp.status_code != 204:
            raise HTTPException(
                status_code=403,
                detail="Access restricted to members of the required GitHub organization.",
            )

    jwt_token = _make_token(sub=github_login_name, email=primary_email or "")

    frontend_url = os.getenv("FRONTEND_URL", "").strip().rstrip("/")
    if not frontend_url or frontend_url == "*":
        frontend_url = str(request.base_url).rstrip("/")

    return RedirectResponse(f"{frontend_url}/?token={jwt_token}")


@router.post("/logout")
async def logout():
    return {"ok": True}
