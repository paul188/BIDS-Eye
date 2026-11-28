"""
services/results/display/dataset_cards.py
-----------------------------------------
The datasets of one result page as the cards show them: the catalogue fields
of the generated query's rows, plus what is read with queries of their own
(the participant summary, deprecation notes, the generated summary
and the linked paper). Those extra reads leave the generated SQL unchanged,
and databases without the newer columns keep working.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from participant_codes import sex_code  # RAG/ is on the path wherever the backend runs

from routers.schemas import DatasetSchema, LinkedPaperSchema, ParticipantSchema, ParticipantSummarySchema
from services.indexing import dataset_description
from services.schema.tables import datasets as _datasets, papers as _papers

log = logging.getLogger(__name__)

# Names OpenNeuro shows when the uploader set none (the crawler's
# remote.openneuro_metadata.clean_dataset_name skips the same ones).
_PLACEHOLDER_NAMES = frozenset({"unnamed dataset", "untitled", "untitled dataset", "n/a"})


def display_name(name: Optional[str], accession_id: Optional[str]) -> str:
    """The card title: the dataset's name, or its accession id when the name
    is blank or a placeholder (a card without a title has nothing to click)."""
    cleaned = " ".join((name or "").split())
    if cleaned and cleaned.lower() not in _PLACEHOLDER_NAMES:
        return cleaned
    return accession_id or "Untitled dataset"


async def participants_for(
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


def summarize_participants(participants: Sequence[ParticipantSchema]) -> Optional[ParticipantSummarySchema]:
    """Group, sex and age counts over every participant."""
    if not participants:
        return None
    diagnoses: dict[str, int] = {}
    female = male = 0
    ages: list[float] = []
    for p in participants:
        label = (p.diagnosis or "").strip()
        if label:
            diagnoses[label] = diagnoses.get(label, 0) + 1
        # The column holds standard codes (the crawler's remote/participant_codes.py);
        # a row crawled since the last rewrite is read through the vocabulary.
        sex = sex_code(p.sex or "")
        if sex == "female":
            female += 1
        elif sex == "male":
            male += 1
        if p.age is not None and p.age > 0:
            ages.append(p.age)
    return ParticipantSummarySchema(
        total=len(participants), diagnoses=diagnoses, female=female, male=male,
        age_min=min(ages) if ages else None, age_max=max(ages) if ages else None,
    )


async def deprecations_for(
    session: AsyncSession, dataset_ids: list[UUID]
) -> dict[UUID, str]:
    """Return {dataset_id: reason} for datasets marked deprecated.

    Read with its own query instead of through the generated SQL, so the
    projection the LLM pipeline relies on stays unchanged.
    """
    if not dataset_ids:
        return {}
    rows = await session.execute(
        text(
            f"SELECT id, deprecated_reason FROM {_datasets.name} "
            "WHERE id = ANY(:ids) AND deprecated_reason IS NOT NULL"
        ),
        {"ids": dataset_ids},
    )
    return {row[0]: row[1] for row in rows}


async def rows_to_datasets(session: AsyncSession, raw: Sequence) -> list[DatasetSchema]:
    """Build DatasetSchema objects (with the participant summary) from result rows."""
    dataset_ids = [UUID(str(r["id"])) for r in raw]
    participants_map = await participants_for(session, dataset_ids)
    deprecations = await deprecations_for(session, dataset_ids)
    return [
        DatasetSchema(
            id=r["id"],
            name=display_name(r["name"], r.get("accession_id")),
            accession_id=r.get("accession_id"),
            bids_version=r.get("bids_version"),
            dataset_type=r["dataset_type"],
            source_type=r["source_type"],
            remote_url=r.get("remote_url"),
            validation_status=r.get("validation_status"),
            authors=r.get("authors"),
            institutions=r.get("institutions"),
            description_text=r.get("description_text"),
            deprecated_reason=deprecations.get(UUID(str(r["id"]))),
            uncertain_level=r.get("uncertain_level"),
            uncertain_reasons=r.get("uncertain_reasons"),
            subject_count=r.get("subject_count"),
            participant_summary=summarize_participants(participants_map.get(UUID(str(r["id"])), [])),
        )
        for r in raw
    ]


def _paper_url(doi, pmid, pmcid, arxiv_id) -> Optional[str]:
    if doi:
        return f"https://doi.org/{doi}"
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if pmcid:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return None


async def attach_descriptions(session: AsyncSession, datasets: list[DatasetSchema]) -> None:
    """The generated summary of a thin README, and the linked paper when it was
    checked to report the dataset (services/indexing/dataset_description.py). Its own
    query, like deprecations_for; a failure leaves the page without them."""
    if not datasets:
        return
    sql = (
        "SELECT d.id, d.dataset_summary, d.paper_relation->>'relation', "
        "       p.title, p.year, p.doi, p.pmid, p.pmcid, p.arxiv_id "
        f"FROM {_datasets.name} d "
        f"LEFT JOIN {_papers.name} p ON p.paper_key = d.paper_relation->>'paper_key' "
        "     AND (d.paper_relation->>'shown')::boolean "
        "WHERE d.id = ANY(:ids) "
        "  AND (d.dataset_summary IS NOT NULL OR (d.paper_relation->>'shown')::boolean)"
    )
    try:
        async with session.begin_nested():
            rows = (await session.execute(text(sql), {"ids": [d.id for d in datasets]})).all()
    except SQLAlchemyError:
        log.warning("Dataset descriptions unavailable for this page", exc_info=True)
        return
    found = {row[0]: row for row in rows}
    for dataset in datasets:
        row = found.get(dataset.id)
        if row is None:
            continue
        dataset.generated_summary = row[1]
        relation, title = row[2], row[3]
        if title and relation in dataset_description.SHOWN_RELATIONS:
            dataset.linked_paper = LinkedPaperSchema(
                title=title, year=row[4], url=_paper_url(row[5], row[6], row[7], row[8]),
                relation=relation, relation_label=dataset_description.RELATION_LABELS[relation])
