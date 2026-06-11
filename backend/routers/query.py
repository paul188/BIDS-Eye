"""
routers/query.py
----------------
Natural-language dataset query endpoint.

Accepts a plain-text question, passes it through the Text-To-SQL layer, executes
the resulting SQL against the BIDS database, and returns matching datasets.

The expensive translation (Gemini intent + RAG + SQL generation) runs once per
question; the resulting base SELECT is cached under an opaque ``query_id`` so
subsequent pages reuse it. Relevance ordering/weighting and LIMIT/OFFSET are
applied per page in postprocessing (see services.text_to_sql.build_page_sql).
"""

from __future__ import annotations

import asyncio
import os
from typing import Annotated, Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from deps import get_db
from schemas import (
    DatasetSchema,
    ParticipantSchema,
    QueryPageRequest,
    QueryRequest,
    QueryResponse,
)
from services.query_cache import CachedQuery, query_cache
from services.text_to_sql import (
    _correct_sql_with_gemini,
    build_count_sql,
    build_page_sql,
    text_to_sql,
)

router = APIRouter(prefix="/query", tags=["query"])

# Capped here as well as in the cache so a page can't request an unbounded slice.
_MAX_PAGE_SIZE = 100
_DEFAULT_PAGE_SIZE = 20
_MAX_PARTICIPANTS_PER_DATASET = 50


async def _participants_for(
    session: AsyncSession, dataset_ids: list[UUID]
) -> dict[UUID, list[ParticipantSchema]]:
    """Fetch participants grouped by dataset_id."""
    if not dataset_ids:
        return {}

    from db.models import BIDSParticipant
    from sqlalchemy import select

    rows = await session.execute(
        select(BIDSParticipant).where(BIDSParticipant.dataset_id.in_(dataset_ids))
    )
    result: dict[UUID, list[ParticipantSchema]] = {}
    for p in rows.scalars():
        result.setdefault(p.dataset_id, []).append(ParticipantSchema.model_validate(p))
    return result


async def _count_with_base_correction(
    session: AsyncSession, query_id: str, cached: CachedQuery, question: str
) -> int:
    """Count total matching datasets, self-correcting the cached *base* SQL once
    via Gemini if it raises a DB error.

    Running the COUNT validates the base SQL before any page is fetched. A fix is
    persisted back to the cache so later page turns reuse the corrected base.
    """
    try:
        result = await session.execute(
            text(build_count_sql(cached.base_sql)), cached.params
        )
        return int(result.scalar() or 0)
    except Exception as exc:
        # Roll back the aborted transaction before any retry — without this,
        # every subsequent execute on the same session fails with
        # InFailedSQLTransactionError.
        await session.rollback()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        if not api_key:
            raise
        corrected = await asyncio.to_thread(
            _correct_sql_with_gemini, cached.base_sql, str(exc), question, api_key
        )
        if corrected == cached.base_sql:
            raise
        cached.base_sql = corrected
        cached.self_corrected = True
        query_cache.update(query_id, cached)
        result = await session.execute(
            text(build_count_sql(cached.base_sql)), cached.params
        )
        return int(result.scalar() or 0)


async def _rows_to_datasets(session: AsyncSession, raw: Sequence) -> list[DatasetSchema]:
    """Build DatasetSchema objects (with capped participants) from result rows."""
    dataset_ids = [UUID(str(r["id"])) for r in raw]
    participants_map = await _participants_for(session, dataset_ids)
    return [
        DatasetSchema(
            id=r["id"],
            name=r["name"],
            accession_id=r.get("accession_id"),
            bids_version=r.get("bids_version"),
            dataset_type=r["dataset_type"],
            source_type=r["source_type"],
            remote_url=r.get("remote_url"),
            validation_status=r.get("validation_status"),
            subject_count=r.get("subject_count"),
            participants=participants_map.get(UUID(str(r["id"])), [])[
                :_MAX_PARTICIPANTS_PER_DATASET
            ],
        )
        for r in raw
    ]


def _clamp_page(page: int, page_size: int) -> tuple[int, int]:
    page = max(1, page)
    page_size = max(1, min(page_size or _DEFAULT_PAGE_SIZE, _MAX_PAGE_SIZE))
    return page, page_size


async def _run_page(
    session: AsyncSession,
    query_id: str,
    cached: CachedQuery,
    question: str,
    page: int,
    page_size: int,
) -> tuple[list[DatasetSchema], int]:
    """Fetch one page for a cached query: total count + the page's datasets.

    The COUNT runs first (validating / self-correcting the base SQL); the page
    query then reuses the now-validated base, so it only applies ordering +
    LIMIT/OFFSET.
    """
    total = await _count_with_base_correction(session, query_id, cached, question)

    offset = (page - 1) * page_size
    page_sql, extra_params = build_page_sql(
        cached.base_sql,
        cached.scored_filters,
        cached.apply_relevance,
        page_size,
        offset,
    )
    result = await session.execute(
        text(page_sql), {**cached.params, **extra_params}
    )
    datasets = await _rows_to_datasets(session, result.mappings().all())
    return datasets, total


@router.post("", response_model=QueryResponse)
async def query_datasets(
    body: QueryRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Submit a natural-language question and receive the first page of matching
    BIDS datasets, plus a ``query_id`` for paging through the rest.
    """
    page, page_size = _clamp_page(1, _DEFAULT_PAGE_SIZE)
    translation = await text_to_sql(body.question)

    cached = CachedQuery(
        base_sql=translation.sql,
        params=translation.params,
        scored_filters=translation.scored_filters,
        apply_relevance=translation.apply_relevance,
        explanation=translation.explanation,
        self_corrected=translation.self_corrected,
    )
    query_id = query_cache.put(cached)

    datasets, total = await _run_page(
        session, query_id, cached, body.question, page, page_size
    )
    # Surface any base-SQL self-correction that happened during the count.
    translation = translation.model_copy(
        update={"sql": cached.base_sql, "self_corrected": cached.self_corrected}
    )

    return QueryResponse(
        message=f"Found {total} dataset(s) matching your query.",
        translation=translation,
        datasets=datasets,
        query_id=query_id,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.post("/page", response_model=QueryResponse)
async def query_page(
    body: QueryPageRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
):
    """Fetch another page of an already-translated query, reusing its cached base
    SQL (no new LLM call)."""
    cached = query_cache.get(body.query_id)
    if cached is None:
        raise HTTPException(
            status_code=410,
            detail="This search has expired. Please run the query again.",
        )

    page, page_size = _clamp_page(body.page, body.page_size)
    # The question is only needed for self-correction, which already happened on
    # the first page; an empty string is fine here.
    datasets, total = await _run_page(
        session, body.query_id, cached, "", page, page_size
    )

    return QueryResponse(
        message=f"Found {total} dataset(s) matching your query.",
        translation=None,
        datasets=datasets,
        query_id=body.query_id,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )
