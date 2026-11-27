"""
deps.py
-------
FastAPI dependency providers shared across routers: the database session and
the signed-in user.
"""

from typing import Annotated, AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from db.db import async_session_maker
from security.sessions import NotSignedInError, session_user


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLAlchemy session, rolling back on error."""
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# What a route writes to get a session: ``session: AsyncSessionDep``.
AsyncSessionDep = Annotated[AsyncSession, Depends(get_db)]


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


def github_id(user: CurrentUserDep) -> int | None:
    """The signed-in user's GitHub account number (user_id); None for a
    session from before sign-in kept it."""
    return user_id(user)


# What a route writes to get the account number: ``github_id: GithubIdDep``.
# It goes through require_auth, so overriding that in a test covers it too.
GithubIdDep = Annotated[int | None, Depends(github_id)]
