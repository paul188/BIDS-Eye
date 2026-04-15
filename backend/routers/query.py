"""
routers/query.py
----------------
Natural-language dataset query endpoint.

Accepts a plain-text question, passes it through the Text-To-SQL layer
(placeholder until the LLM implementation is wired in), executes the
resulting SQL against the BIDS database, and returns matching datasets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Text-To-SQL"))

from deps import get_db
from placeholder import text_to_sql
from schemas import DatasetSchema, ParticipantSchema, QueryRequest, QueryResponse

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

    rows = await session.execute(
        text(translation.sql), translation.params
    )
    raw = rows.mappings().all()

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
            participants=participants_map.get(UUID(str(r["id"])), []),
        )
        for r in raw
    ]

    return QueryResponse(
        message=f"Found {len(datasets)} dataset(s) matching your query.",
        translation=translation,
        datasets=datasets,
    )
