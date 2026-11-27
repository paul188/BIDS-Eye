"""
routers/history.py
------------------
A user's saved searches: GET /history lists them and says whether saving is
on, PUT /history/enabled turns it on or off (off deletes them),
GET /history/{id} gives one as it was answered, DELETE /history/{id} removes
one. The rules are in services/stores/search_history.py; this module does the
HTTP side. A search is saved where it is answered (routers/search.py).

Every route acts on the signed-in user's own rows only: the user comes from
the session, never from the request.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

from deps import AsyncSessionDep, CurrentUserDep, GithubIdDep
from exceptions.domain import DomainError
from rate_limits import Category, rate_limit
from schemas import HistoryEnabled, HistoryEntry, HistoryList
from services.stores import search_history

history_router = APIRouter(prefix="/history", tags=["history"])


class SignInAgainError(
    DomainError,
    status_code=409,
    detail="Please sign out and sign in again, then turn this on.",
):
    """The session predates saved searches and has no GitHub account number."""


@history_router.get("", dependencies=[rate_limit(Category.LIGHT)])
async def list_history(
    session: AsyncSessionDep,
    github_id: GithubIdDep,
) -> HistoryList:
    """Whether the user's searches are saved, and the saved ones, newest first."""
    entries = await search_history.list_entries(session, github_id)
    return HistoryList(
        enabled=entries is not None,
        retention_days=search_history.RETENTION_DAYS,
        entries=entries or [],
    )


@history_router.put("/enabled", dependencies=[rate_limit(Category.LIGHT)])
async def set_history_enabled(
    body: HistoryEnabled,
    session: AsyncSessionDep,
    user: CurrentUserDep,
    github_id: GithubIdDep,
) -> HistoryEnabled:
    """Turn saving on, or off: off deletes everything saved for the user."""
    if github_id is None:
        if body.enabled:
            raise SignInAgainError(user.get("sub", ""))
        return HistoryEnabled(enabled=False)
    if body.enabled:
        await search_history.enable(session, github_id)
    else:
        await search_history.disable(session, github_id)
    return body


@history_router.get("/{entry_id}", dependencies=[rate_limit(Category.LIGHT)])
async def get_history_entry(
    entry_id: UUID,
    session: AsyncSessionDep,
    github_id: GithubIdDep,
) -> HistoryEntry:
    """One saved search as it was answered then."""
    return await search_history.get_entry(session, github_id, entry_id)


@history_router.delete("/{entry_id}", status_code=204, dependencies=[rate_limit(Category.LIGHT)])
async def delete_history_entry(
    entry_id: UUID,
    session: AsyncSessionDep,
    github_id: GithubIdDep,
) -> None:
    """Delete one saved search."""
    await search_history.delete_entry(session, github_id, entry_id)
