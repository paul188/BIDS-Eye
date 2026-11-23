"""
services/stores/search_history.py
---------------------------------
The searches of users who turned on "Remember my searches and search results".

Off for everyone until they turn it on: without a search_history_consents row
nothing about a user is stored, as before. With one, every search they run is
kept in search_history as it was answered (question, translation, chips,
first page of results), so their history is the same in every browser they
sign in from. The site's privacy notice describes this
(frontend/src/components/CookieBanner.vue); keep the two in line.

The rules, all enforced here:
- Consent first: record() saves only for a user with a consent row.
- Turning it off deletes the user's rows and the consent row (disable()).
- RETENTION_DAYS: a search is deleted that long after it was asked, and a
  consent that long after the user last saved or read anything.
- A user is GitHub's account number (routers/auth.py user_id), never the
  login, and a row is only ever read or deleted through its owner's number.

query_log (models.QueryLog) stays anonymous: nothing here points at it.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Optional
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions.domain import DomainError
from models import SearchHistoryConsent, SearchHistoryEntry
from schemas import HistoryEntry, HistoryEntrySummary, QueryResponse

log = logging.getLogger(__name__)

RETENTION_DAYS = 365


class HistoryUnavailableError(
    DomainError,
    status_code=503,
    detail="Your search history is not available right now. Please try again.",
):
    """The database failed while reading or changing a user's history."""


class HistoryEntryNotFoundError(
    DomainError, status_code=404, detail="This saved search no longer exists."
):
    """No such entry among this user's: deleted, expired, or someone else's."""


def _cutoff() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)


@asynccontextmanager
async def _database(session: AsyncSession) -> AsyncIterator[None]:
    """Turn a database failure into HistoryUnavailableError, session rolled back."""
    try:
        yield
    except SQLAlchemyError as exc:
        await session.rollback()
        raise HistoryUnavailableError(type(exc).__name__) from exc


async def _delete_expired(session: AsyncSession) -> None:
    """Drop everything past RETENTION_DAYS, whoever it belongs to.

    Runs with every save and every listing, so the rows of a user who never
    comes back go when anyone else uses the history. A consent's last use is
    never older than its newest search, so no search is left without one.
    """
    cutoff = _cutoff()
    await session.execute(delete(SearchHistoryEntry).where(SearchHistoryEntry.asked_at < cutoff))
    await session.execute(
        delete(SearchHistoryConsent).where(SearchHistoryConsent.last_used_at < cutoff)
    )


async def _touch(session: AsyncSession, github_id: int) -> bool:
    """Note that the user's history was used now; False when they have not turned it on."""
    touched = await session.execute(
        update(SearchHistoryConsent)
        .where(
            SearchHistoryConsent.github_id == github_id,
            SearchHistoryConsent.last_used_at >= _cutoff(),
        )
        .values(last_used_at=func.now())
    )
    return touched.rowcount == 1


async def enable(session: AsyncSession, github_id: int) -> None:
    """Record the user's consent; searches from now on are saved."""
    statement = (
        insert(SearchHistoryConsent)
        .values(github_id=github_id)
        .on_conflict_do_update(
            index_elements=[SearchHistoryConsent.github_id],
            set_={"last_used_at": func.now()},
        )
    )
    async with _database(session):
        await session.execute(statement)
        await session.commit()


async def disable(session: AsyncSession, github_id: int) -> None:
    """Withdraw the consent: the user's saved searches and the consent row go."""
    async with _database(session):
        await session.execute(
            delete(SearchHistoryEntry).where(SearchHistoryEntry.github_id == github_id)
        )
        await session.execute(
            delete(SearchHistoryConsent).where(SearchHistoryConsent.github_id == github_id)
        )
        await session.commit()


async def record(
    session: AsyncSession, github_id: Optional[int], question: str, response: QueryResponse
) -> Optional[str]:
    """Save a search that was just answered; the new entry's id, or None.

    None for a user who has not turned the history on (or whose session has
    no account number). A database error is logged and gives None too: the
    search answers anyway, as with the query log.
    """
    if github_id is None:
        return None
    entry_id: Optional[UUID] = None
    try:
        # A savepoint keeps a failed insert from breaking the response; the
        # commit makes the row last (get_db never commits).
        async with session.begin_nested():
            if await _touch(session, github_id):
                entry_id = uuid4()
                session.add(SearchHistoryEntry(
                    id=entry_id, github_id=github_id, question=question,
                    total=response.total, answer=response.model_dump_json(),
                ))
            await _delete_expired(session)
        await session.commit()
    except SQLAlchemyError:
        log.warning("search history: save failed", exc_info=True)
        await session.rollback()
        return None
    return str(entry_id) if entry_id else None


async def list_entries(
    session: AsyncSession, github_id: Optional[int]
) -> Optional[list[HistoryEntrySummary]]:
    """The user's saved searches, newest first; None when they have not turned it on."""
    if github_id is None:
        return None
    async with _database(session):
        enabled = await _touch(session, github_id)
        await _delete_expired(session)
        rows = []
        if enabled:
            rows = (await session.execute(
                select(SearchHistoryEntry.id, SearchHistoryEntry.asked_at,
                       SearchHistoryEntry.question, SearchHistoryEntry.total)
                .where(SearchHistoryEntry.github_id == github_id)
                .order_by(SearchHistoryEntry.asked_at.desc())
            )).all()
        await session.commit()
    if not enabled:
        return None
    return [
        HistoryEntrySummary(id=str(row.id), asked_at=row.asked_at, question=row.question,
                            total=row.total)
        for row in rows
    ]


async def get_entry(session: AsyncSession, github_id: Optional[int], entry_id: UUID) -> HistoryEntry:
    """One of the user's saved searches. Raises HistoryEntryNotFoundError."""
    if github_id is None:
        raise HistoryEntryNotFoundError(str(entry_id))
    async with _database(session):
        row = (await session.execute(
            select(SearchHistoryEntry).where(
                SearchHistoryEntry.id == entry_id,
                SearchHistoryEntry.github_id == github_id,
                SearchHistoryEntry.asked_at >= _cutoff(),
            )
        )).scalar_one_or_none()
    if row is None:
        raise HistoryEntryNotFoundError(str(entry_id))
    return HistoryEntry(
        id=str(row.id), asked_at=row.asked_at, question=row.question,
        answer=QueryResponse.model_validate_json(row.answer).model_copy(
            update={"history_id": str(row.id)}
        ),
    )


async def delete_entry(session: AsyncSession, github_id: Optional[int], entry_id: UUID) -> None:
    """Delete one of the user's saved searches; nothing happens when it is already gone."""
    if github_id is None:
        return
    async with _database(session):
        await session.execute(
            delete(SearchHistoryEntry).where(
                SearchHistoryEntry.id == entry_id,
                SearchHistoryEntry.github_id == github_id,
            )
        )
        await session.commit()
