"""
remote/graphql.py
-----------------
GraphQL fallback ingestion for restricted OpenNeuro datasets.

Used when S3 access is denied (dataset is embargoed/restricted).
Fetches dataset-level metadata from the public OpenNeuro GraphQL API
and populates the DB with what is available:

  BIDSDataset   — name, authors, license, README, study metadata, size, etc.
  BIDSParticipant — one row per subject the API lists (age, sex, group)
  BIDSObject    — one stub row per task (suffix=bold, datatype=func)
                  so task-level SQL queries still work

All rows are marked source_type="openneuro_graphql" so they can be
distinguished from fully-indexed datasets.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.db import async_session_maker
from db.models import BIDSDataset, BIDSObject, BIDSParticipant
from remote.openneuro_metadata import (
    METADATA_SELECTION,
    OpenNeuroAPIError,
    apply_fields,
    deprecation_reason,
    extract_fields,
    post_graphql,
    set_deprecation,
)

log = logging.getLogger(__name__)

# Field names follow the live OpenNeuro schema (types Metadata, Snapshot,
# Description, Summary). METADATA_SELECTION adds README, dxStatus, studyDomain,
# studyDesign, ages and size; GraphQL merges it with the selections below.
_QUERY = """
query Dataset($id: ID!) {
  dataset(id: $id) {
    id
    %s
    metadata {
      datasetName
      species
      modalities
      dataProcessed
      tasksCompleted
      grantFunderName
    }
    latestSnapshot {
      description {
        BIDSVersion
        Name
        Authors
        License
        DatasetDOI
        Funding
        ReferencesAndLinks
        Acknowledgements
      }
      summary {
        modalities
        tasks
        subjectMetadata { participantId age sex group }
      }
    }
  }
}
""" % METADATA_SELECTION


# ── GraphQL fetch ─────────────────────────────────────────────────────────────

def _fetch_graphql(accession_id: str) -> dict | None:
    """Query OpenNeuro GraphQL. Returns the dataset dict or None on failure."""
    try:
        body = post_graphql(_QUERY, {"id": accession_id})
    except OpenNeuroAPIError as exc:
        log.warning("  [graphql] Request failed for %s: %s", accession_id, exc)
        return None
    if body.get("errors"):
        log.warning("  [graphql] API errors for %s: %s", accession_id, body["errors"])
    return body["data"].get("dataset")


def _as_list(value) -> list:
    if not value:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _build_description(accession_id: str, ds: dict, snapshot_desc: dict) -> dict:
    """
    Build the JSON blob stored in bids_datasets.description.
    Only keeps fields that don't have dedicated columns — promoted fields
    (Authors, License, DatasetDOI, ReferencesAndLinks, Funding, README, study
    metadata) are stored in their own columns, not duplicated here.
    """
    meta = ds.get("metadata") or {}
    summary = (ds.get("latestSnapshot") or {}).get("summary") or {}
    return {
        "Name":            meta.get("datasetName") or snapshot_desc.get("Name") or accession_id,
        "BIDSVersion":     snapshot_desc.get("BIDSVersion") or "unknown",
        "Acknowledgements": snapshot_desc.get("Acknowledgements") or "",
        "Species":         meta.get("species") or "",
        # Extra fields not in standard dataset_description.json
        "_graphql": {
            "modalities":          _as_list(summary.get("modalities") or meta.get("modalities")),
            "tasks":               _as_list(summary.get("tasks") or meta.get("tasksCompleted")),
            "dataProcessed":       meta.get("dataProcessed"),
            "numberOfParticipants": len(summary.get("subjectMetadata") or []) or None,
        },
    }


def _extract_promoted(accession_id: str, ds: dict, snapshot_desc: dict) -> dict:
    """Values for the first-class columns defined on the BIDSDataset ORM model.

    README, dx_status, study_domain, study_design, age range and size are
    written afterwards with remote.openneuro_metadata.apply_fields.
    """
    meta = ds.get("metadata") or {}
    funding = _as_list(snapshot_desc.get("Funding")) or _as_list(meta.get("grantFunderName"))
    return {
        "authors":          _as_list(snapshot_desc.get("Authors")) or None,
        "license":          snapshot_desc.get("License") or None,
        "doi":              snapshot_desc.get("DatasetDOI") or None,
        "paper_references": _as_list(snapshot_desc.get("ReferencesAndLinks")) or None,
        "funding":          funding or None,
        "description_text": None,   # dataset_description.json has no Description on OpenNeuro
    }


# ── DB upsert helpers ─────────────────────────────────────────────────────────

async def _upsert_dataset(
    accession_id: str,
    description: dict,
    promoted: dict,
    remote_url: str,
) -> BIDSDataset | None:
    """Insert or update a BIDSDataset row for a graphql-sourced dataset."""
    root_path = f"__graphql__/{accession_id}"
    name = (description.get("Name") or accession_id)[:300]
    bids_version = description.get("BIDSVersion", "unknown")

    stmt = (
        pg_insert(BIDSDataset)
        .values(
            id=uuid4(),
            root_path=root_path,
            description=description,
            name=name,
            bids_version=bids_version,
            dataset_type="raw",
            source_type="openneuro_graphql",
            accession_id=accession_id,
            remote_url=remote_url,
            **promoted,
        )
        .on_conflict_do_update(
            index_elements=["root_path"],
            set_={
                "description":      pg_insert(BIDSDataset).excluded.description,
                "name":             pg_insert(BIDSDataset).excluded.name,
                "bids_version":     pg_insert(BIDSDataset).excluded.bids_version,
                "source_type":      pg_insert(BIDSDataset).excluded.source_type,
                "remote_url":       pg_insert(BIDSDataset).excluded.remote_url,
                "authors":          pg_insert(BIDSDataset).excluded.authors,
                "license":          pg_insert(BIDSDataset).excluded.license,
                "doi":              pg_insert(BIDSDataset).excluded.doi,
                "paper_references": pg_insert(BIDSDataset).excluded.paper_references,
                "funding":          pg_insert(BIDSDataset).excluded.funding,
                "description_text": pg_insert(BIDSDataset).excluded.description_text,
            },
        )
    )
    async with async_session_maker() as session:
        await session.execute(stmt)
        await session.commit()
        return await session.scalar(
            select(BIDSDataset).where(BIDSDataset.root_path == root_path)
        )


async def _upsert_participants(
    dataset: BIDSDataset,
    subjects: list[dict],
) -> int:
    """
    Create one participant row per subject the API lists (age, sex, group).

    Values are stored as OpenNeuro reports them; the DB preprocessing scripts
    normalise sex codes afterwards, as they do for participants.tsv rows.
    The group label is kept in `extra` — it is free text, not a canonical
    diagnosis code.
    """
    rows = []
    seen: set[str] = set()
    for subject in subjects or []:
        participant_id = str(subject.get("participantId") or "").strip()
        if not participant_id or participant_id in seen:
            continue
        seen.add(participant_id)
        extra = {"_source": "openneuro_graphql"}
        if subject.get("group"):
            extra["group"] = str(subject["group"])
        age = subject.get("age")
        rows.append(dict(
            id=uuid4(),
            dataset_id=dataset.id,
            participant_id=participant_id,
            age=float(age) if isinstance(age, (int, float)) and not isinstance(age, bool) else None,
            sex=(str(subject["sex"])[:50] if subject.get("sex") else None),
            handedness=None,
            diagnosis=None,
            extra=extra,
        ))

    if not rows:
        return 0

    stmt = (
        pg_insert(BIDSParticipant)
        .values(rows)
        .on_conflict_do_update(
            constraint="uq_participant_per_dataset",
            set_={
                "age":   pg_insert(BIDSParticipant).excluded.age,
                "sex":   pg_insert(BIDSParticipant).excluded.sex,
                "extra": pg_insert(BIDSParticipant).excluded.extra,
            },
        )
    )
    async with async_session_maker() as session:
        await session.execute(stmt)
        await session.commit()
    return len(rows)


async def _upsert_stub_objects(
    dataset: BIDSDataset,
    tasks: list[str],
    modalities: list[str],
) -> int:
    """
    Create one stub BIDSObject per task so task-level SQL queries work.
    Marked is_remote=True and path encodes that this is synthetic.
    """
    if not tasks:
        return 0

    # Map GraphQL modality names to BIDS datatypes
    _modality_to_datatype = {
        "MRI": "func", "fMRI": "func", "T1w": "anat", "T2w": "anat",
        "EEG": "eeg", "MEG": "meg", "iEEG": "ieeg",
        "DWI": "dwi", "PET": "pet", "behavioral": "beh",
    }
    datatype = next(
        (_modality_to_datatype[m] for m in modalities if m in _modality_to_datatype),
        "func",  # default
    )

    rows = []
    for task in tasks:
        path = f"__graphql__/{dataset.accession_id}/task-{task}_stub"
        rows.append(dict(
            id=uuid4(),
            path=path,
            filename=f"task-{task}_stub",   # NOT NULL on bids_objects
            subject=None,
            subject_index=None,
            session=None,
            session_index=None,
            task=task,
            run=None,
            suffix="bold",
            datatype=datatype,
            extension=".nii.gz",
            other_entities={"_synthetic": True},
            is_remote=True,
            dataset_id=dataset.id,
            class_="image_file",
        ))

    stmt = (
        pg_insert(BIDSObject)
        .values(rows)
        .on_conflict_do_update(
            index_elements=["path"],
            set_={"task": pg_insert(BIDSObject).excluded.task},
        )
    )
    async with async_session_maker() as session:
        await session.execute(stmt)
        await session.commit()
    return len(rows)


# ── Main entry point ──────────────────────────────────────────────────────────

async def index_via_graphql(accession_id: str) -> None:
    """
    Ingest a restricted OpenNeuro dataset via the GraphQL API.

    Populates:
      - BIDSDataset      (description fields, README, study metadata, size)
      - BIDSParticipant  (one row per subject the API lists)
      - BIDSObject       (one stub row per task)
    """
    log.info("  [graphql] Fetching metadata for %s ...", accession_id)
    ds = await asyncio.to_thread(_fetch_graphql, accession_id)

    if ds is None:
        log.warning("  [graphql] No data returned for %s — skipping", accession_id)
        return

    meta = ds.get("metadata") or {}
    snapshot = ds.get("latestSnapshot") or {}
    snapshot_desc = snapshot.get("description") or {}
    summary = snapshot.get("summary") or {}

    description = _build_description(accession_id, ds, snapshot_desc)
    promoted = _extract_promoted(accession_id, ds, snapshot_desc)
    remote_url = f"s3://openneuro.org/{accession_id}"

    # 1. Upsert dataset row
    dataset = await _upsert_dataset(accession_id, description, promoted, remote_url)
    if dataset is None:
        log.error("  [graphql] Failed to upsert dataset row for %s", accession_id)
        return
    log.info("  [graphql] Dataset row upserted: %s (%s)", accession_id, dataset.name)

    # 2. README, dx_status, study_domain, study_design, age range, size
    async with async_session_maker() as session:
        await apply_fields(session, accession_id, extract_fields(ds))
        await set_deprecation(session, accession_id, deprecation_reason(ds))
        await session.commit()

    # 3. Participants from the per-subject metadata
    n_participants = await _upsert_participants(dataset, summary.get("subjectMetadata") or [])
    log.info("  [graphql] Upserted %d participant rows", n_participants)

    # 4. Stub objects per task
    tasks = _as_list(summary.get("tasks") or meta.get("tasksCompleted"))
    modalities = _as_list(summary.get("modalities") or meta.get("modalities"))
    n_objects = await _upsert_stub_objects(dataset, tasks, modalities)
    log.info("  [graphql] Created %d stub object rows for tasks: %s", n_objects, tasks)

    log.info("  [graphql] Done: %s", accession_id)
