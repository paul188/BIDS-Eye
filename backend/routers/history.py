"""
routers/history.py
------------------
A user's saved searches: GET /history lists them and says whether saving is
on, PUT /history/enabled turns it on or off (off deletes them),
GET /history/{id} gives one as it was answered, DELETE /history/{id} removes
one. The rules are in services/stores/search_history.py; this module does the
HTTP side. A search is saved where it is answered (routers/query.py).

Every route acts on the signed-in user's own rows only: the user comes from
the session, never from the request.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from deps import get_db
from exceptions.domain import DomainError
from rate_limits import Category, rate_limit
from routers.auth import require_auth, user_id
from schemas import HistoryEnabled, HistoryEntry, HistoryList
from services.stores import search_history

router = APIRouter(prefix="/history", tags=["history"])


class SignInAgainError(
    DomainError,
    status_code=409,
    detail="Please sign out and sign in again, then turn this on.",
):
    """The session predates saved searches and has no GitHub account number."""


@router.get("", dependencies=[rate_limit(Category.LIGHT)])
async def list_history(
    session: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[dict, Depends(require_auth)],
) -> HistoryList:
    """Whether the user's searches are saved, and the saved ones, newest first."""
    entries = await search_history.list_entries(session, user_id(user))
    return HistoryList(
        enabled=entries is not None,
        retention_days=search_history.RETENTION_DAYS,
        entries=entries or [],
    )


@router.put("/enabled", dependencies=[rate_limit(Category.LIGHT)])
async def set_history_enabled(
    body: HistoryEnabled,
    session: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[dict, Depends(require_auth)],
) -> HistoryEnabled:
    """Turn saving on, or off: off deletes everything saved for the user."""
    github_id = user_id(user)
    if github_id is None:
        if body.enabled:
            raise SignInAgainError(user.get("sub", ""))
        return HistoryEnabled(enabled=False)
    if body.enabled:
        await search_history.enable(session, github_id)
    else:
        await search_history.disable(session, github_id)
    return body


@router.get("/{entry_id}", dependencies=[rate_limit(Category.LIGHT)])
async def get_history_entry(
    entry_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[dict, Depends(require_auth)],
) -> HistoryEntry:
    """One saved search as it was answered then."""
    return await search_history.get_entry(session, user_id(user), entry_id)


@router.delete("/{entry_id}", status_code=204, dependencies=[rate_limit(Category.LIGHT)])
async def delete_history_entry(
    entry_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    user: Annotated[dict, Depends(require_auth)],
) -> None:
    """Delete one saved search."""
    await search_history.delete_entry(session, user_id(user), entry_id)
