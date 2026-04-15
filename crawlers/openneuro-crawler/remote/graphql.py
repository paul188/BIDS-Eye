"""
remote/graphql.py
-----------------
GraphQL fallback ingestion for restricted OpenNeuro datasets.

Used when S3 access is denied (dataset is embargoed/restricted).
Fetches dataset-level metadata from the public OpenNeuro GraphQL API
and populates the DB with what is available:

  BIDSDataset   — name, description, modalities, tasks, BIDS version, etc.
  BIDSParticipant — synthetic rows from sex counts (age=NULL)
  BIDSObject    — one stub row per task (suffix=bold, datatype=func)
                  so task-level SQL queries still work

All rows are marked source_type="openneuro_graphql" so they can be
distinguished from fully-indexed datasets.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import requests
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.db import async_session_maker
from db.models import BIDSDataset, BIDSObject, BIDSParticipant

log = logging.getLogger(__name__)

GRAPHQL_URL = "https://openneuro.org/crn/graphql"

_QUERY = """
query Dataset($id: ID!) {
  dataset(id: $id) {
    id
    metadata {
      datasetName
      datasetDOI
      license
      authors
      acknowledgements
      fundingSource
      species
      modalities
      dataProcessed
      tasks
      ages { min max }
      sex { maleCount femaleCount otherCount }
      numberOfParticipants
      studyDesign
      studyDomain
    }
    latestSnapshot {
      description {
        BIDSVersion
        Name
        Authors
        License
        DatasetDOI
      }
    }
  }
}
"""


# ── GraphQL fetch ─────────────────────────────────────────────────────────────

def _fetch_graphql(accession_id: str) -> dict | None:
    """Query OpenNeuro GraphQL. Returns the dataset dict or None on failure."""
    try:
        resp = requests.post(
            GRAPHQL_URL,
            json={"query": _QUERY, "variables": {"id": accession_id}},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        errors = data.get("errors")
        if errors:
            log.warning("  [graphql] API errors for %s: %s", accession_id, errors)
        return data.get("data", {}).get("dataset")
    except Exception as exc:
        log.warning("  [graphql] Request failed for %s: %s", accession_id, exc)
        return None


def _build_description(accession_id: str, ds: dict, snapshot_desc: dict) -> dict:
    """
    Build the JSON blob stored in bids_datasets.description.
    Only keeps fields that don't have dedicated columns — promoted fields
    (Authors, License, DatasetDOI, ReferencesAndLinks, Funding, Description)
    are stored in their own columns, not duplicated here.
    """
    meta = ds.get("metadata") or {}
    return {
        "Name":            meta.get("datasetName") or snapshot_desc.get("Name") or accession_id,
        "BIDSVersion":     snapshot_desc.get("BIDSVersion") or "unknown",
        "Acknowledgements": meta.get("acknowledgements") or "",
        "Species":         meta.get("species") or "",
        # Extra fields not in standard dataset_description.json
        "_graphql": {
            "modalities":          meta.get("modalities") or [],
            "tasks":               meta.get("tasks") or [],
            "dataProcessed":       meta.get("dataProcessed"),
            "studyDesign":         meta.get("studyDesign") or "",
            "studyDomain":         meta.get("studyDomain") or "",
            "numberOfParticipants": meta.get("numberOfParticipants"),
            "ages":                meta.get("ages") or {},
            "sex":                 meta.get("sex") or {},
        },
    }


def _extract_promoted(accession_id: str, ds: dict, snapshot_desc: dict) -> dict:
    """Extract values for the first-class columns that are NOT stored in the blob."""
    meta = ds.get("metadata") or {}
    raw_authors = meta.get("authors") or snapshot_desc.get("Authors") or []
    authors = raw_authors if isinstance(raw_authors, list) else ([raw_authors] if raw_authors else None)
    raw_fund = meta.get("fundingSource") or ""
    funding = [raw_fund] if isinstance(raw_fund, str) and raw_fund else (raw_fund if isinstance(raw_fund, list) else None)
    return {
        "authors":          authors or None,
        "license":          meta.get("license") or snapshot_desc.get("License") or None,
        "doi":              meta.get("datasetDOI") or snapshot_desc.get("DatasetDOI") or None,
        "paper_references": None,   # not available from GraphQL
        "funding":          funding,
        "description_text": None,   # not available from GraphQL metadata
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
    meta: dict,
) -> int:
    """
    Create synthetic participant rows from aggregate sex counts.
    Individual ages are unknown — stored as NULL.
    Age range is stored in the dataset description JSON.
    """
    sex_counts = meta.get("sex") or {}
    male   = int(sex_counts.get("maleCount")   or 0)
    female = int(sex_counts.get("femaleCount")  or 0)
    other  = int(sex_counts.get("otherCount")   or 0)

    rows = []
    counter = 1
    for sex_label, count in [("M", male), ("F", female), ("O", other)]:
        for _ in range(count):
            rows.append(dict(
                id=uuid4(),
                dataset_id=dataset.id,
                participant_id=f"sub-{counter:03d}",
                age=None,   # only aggregate min/max available
                sex=sex_label,
                handedness=None,
                diagnosis=None,
                extra={"_synthetic": True},
            ))
            counter += 1

    if not rows:
        return 0

    # Deduplicate
    seen: set[tuple] = set()
    deduped = []
    for r in rows:
        key = (r["dataset_id"], r["participant_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    stmt = (
        pg_insert(BIDSParticipant)
        .values(deduped)
        .on_conflict_do_update(
            constraint="uq_participant_per_dataset",
            set_={
                "sex":   pg_insert(BIDSParticipant).excluded.sex,
                "extra": pg_insert(BIDSParticipant).excluded.extra,
            },
        )
    )
    async with async_session_maker() as session:
        await session.execute(stmt)
        await session.commit()
    return len(deduped)


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
      - BIDSDataset  (full metadata from API)
      - BIDSParticipant  (synthetic rows from sex counts)
      - BIDSObject   (one stub row per task)
    """
    log.info("  [graphql] Fetching metadata for %s ...", accession_id)
    ds = _fetch_graphql(accession_id)

    if ds is None:
        log.warning("  [graphql] No data returned for %s — skipping", accession_id)
        return

    meta = ds.get("metadata") or {}
    snapshot = ds.get("latestSnapshot") or {}
    snapshot_desc = snapshot.get("description") or {}

    description = _build_description(accession_id, ds, snapshot_desc)
    promoted = _extract_promoted(accession_id, ds, snapshot_desc)
    remote_url = f"s3://openneuro.org/{accession_id}"

    # 1. Upsert dataset row
    dataset = await _upsert_dataset(accession_id, description, promoted, remote_url)
    if dataset is None:
        log.error("  [graphql] Failed to upsert dataset row for %s", accession_id)
        return
    log.info("  [graphql] Dataset row upserted: %s (%s)", accession_id, dataset.name)

    # 2. Synthetic participants from sex counts
    n_participants = await _upsert_participants(dataset, meta)
    log.info("  [graphql] Created %d synthetic participant rows", n_participants)

    # 3. Stub objects per task
    tasks = meta.get("tasks") or []
    modalities = meta.get("modalities") or []
    n_objects = await _upsert_stub_objects(dataset, tasks, modalities)
    log.info("  [graphql] Created %d stub object rows for tasks: %s", n_objects, tasks)

    log.info("  [graphql] Done: %s", accession_id)
