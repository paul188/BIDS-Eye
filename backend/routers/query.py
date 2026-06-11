"""
routers/query.py
----------------
Natural-language dataset query endpoint.

Accepts a plain-text question, passes it through the Text-To-SQL layer
(placeholder until the LLM implementation is wired in), executes the
resulting SQL against the BIDS database, and returns matching datasets.
"""

from __future__ import annotations

import asyncio
import os
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from deps import get_db
from schemas import DatasetSchema, ParticipantSchema, QueryRequest, QueryResponse
from services.text_to_sql import _correct_sql_with_gemini, text_to_sql

router = APIRouter(prefix="/query", tags=["query"])


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


@router.post("", response_model=QueryResponse)
async def query_datasets(
    body: QueryRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Submit a natural-language question and receive matching BIDS datasets.

    The question is translated to SQL via the Text-To-SQL layer.
    Currently uses a placeholder that returns all datasets.
    """
    translation = await text_to_sql(body.question)

    try:
        rows = await session.execute(text(translation.sql), translation.params)
    except Exception as exc:
        # Roll back the aborted transaction before any retry — without this,
        # every subsequent execute on the same session fails with
        # InFailedSQLTransactionError.
        await session.rollback()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        if api_key:
            corrected_sql = await asyncio.to_thread(
                _correct_sql_with_gemini,
                translation.sql, str(exc), body.question, api_key,
            )
            if corrected_sql != translation.sql:
                translation = translation.model_copy(
                    update={"sql": corrected_sql, "self_corrected": True}
                )
                rows = await session.execute(text(translation.sql), translation.params)
            else:
                raise
        else:
            raise

    raw = rows.mappings().all()

    # Bound the response. Broad queries can match >1000 datasets and tens of
    # thousands of participants; the full eager-loaded payload (10-15 MB, 30 s+)
    # gets dropped by Cloudflare's ~100 s origin timeout, surfacing in the browser
    # as "NetworkError when attempting to fetch resource". Cap both the dataset
    # count and the participants per dataset. subject_count stays exact (it comes
    # from the SQL COUNT), and the UI only uses participants for the distinct
    # diagnosis tags, so a capped list renders the same for all but mega-datasets.
    _MAX_DATASETS = 200
    _MAX_PARTICIPANTS_PER_DATASET = 50

    total = len(raw)
    raw = raw[:_MAX_DATASETS]

    dataset_ids = [UUID(str(r["id"])) for r in raw]
    participants_map = await _participants_for(session, dataset_ids)

    datasets = [
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
            participants=participants_map.get(UUID(str(r["id"])), [])[:_MAX_PARTICIPANTS_PER_DATASET],
        )
        for r in raw
    ]

    if total > len(datasets):
        message = f"Found {total} dataset(s); showing the first {len(datasets)}. Refine your query to narrow the results."
    else:
        message = f"Found {total} dataset(s) matching your query."

    return QueryResponse(
        message=message,
        translation=translation,
        datasets=datasets,
    )
