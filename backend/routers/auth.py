from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
_ALGORITHM = "HS256"
_EXPIRE_DAYS = 7

router = APIRouter(prefix="/auth", tags=["auth"])
_bearer = HTTPBearer()


class LoginRequest(BaseModel):
    password: str


def _make_token() -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=_EXPIRE_DAYS)
    return jwt.encode({"sub": "user", "exp": exp}, _SECRET, algorithm=_ALGORITHM)


def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> None:
    try:
        payload = jwt.decode(
            credentials.credentials, _SECRET, algorithms=[_ALGORITHM]
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if payload.get("sub") != "user":
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/login")
async def login(body: LoginRequest):
    expected = os.getenv("AUTH_PASSWORD", "")
    if not expected or body.password != expected:
        raise HTTPException(status_code=401, detail="Incorrect password")
    return {"token": _make_token()}


@router.post("/logout")
async def logout():
    return {"ok": True}
