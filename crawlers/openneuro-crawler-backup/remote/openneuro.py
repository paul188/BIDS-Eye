"""
remote/openneuro.py
-------------------
Ghost-indexer for public OpenNeuro datasets.

Crawls s3://openneuro.org/{accession_id}/ using anonymous S3 access.
Only .json and .tsv files are downloaded (metadata).
All files — including imaging files (.nii.gz) — are stored as BIDSObject rows,
but with is_remote=True so callers know the content is not local.

The S3 URL for any object can be reconstructed as:
    dataset.remote_url + "/" + object.path

Requires boto3:
    pip install boto3

Usage:
    from remote.openneuro import index_openneuro
    await index_openneuro("ds000001")
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import PurePosixPath
from typing import Iterator
from uuid import uuid4

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.db import async_session_maker, create_all_tables
from db.models import BIDSDataset, BIDSObject

log = logging.getLogger(__name__)

OPENNEURO_BUCKET = "openneuro.org"

# Entities stored as first-class columns; everything else → other_entities
_FIRST_CLASS_ENTITIES = frozenset(
    {"subject", "session", "task", "run", "suffix", "datatype", "extension"}
)

_BIDS_DATATYPES = frozenset(
    {"anat", "func", "dwi", "fmap", "beh", "meg", "eeg", "ieeg", "perf", "pet", "micr"}
)

# Short key → canonical entity name (matches pybids naming)
_ENTITY_MAP = {
    "sub": "subject", "ses": "session", "task": "task", "run": "run",
    "acq": "acquisition", "ce": "ceagent", "rec": "reconstruction",
    "dir": "direction", "echo": "echo", "part": "part", "space": "space",
    "res": "resolution", "hemi": "hemisphere", "label": "label",
    "desc": "description", "trc": "tracer", "stain": "stain",
}

_ENTITY_RE = re.compile(r"([a-zA-Z]+)-([a-zA-Z0-9]+)")


# ─── S3 helpers ───────────────────────────────────────────────────────────────

def _s3_client():
    """Create an anonymous boto3 S3 client (no credentials required)."""
    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )


def _list_objects(client, accession_id: str) -> list[dict]:
    """Return all S3 object metadata dicts under the accession prefix."""
    paginator = client.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=OPENNEURO_BUCKET, Prefix=f"{accession_id}/"):
        objects.extend(page.get("Contents", []))
    return objects


def _fetch_text(client, key: str) -> str:
    resp = client.get_object(Bucket=OPENNEURO_BUCKET, Key=key)
    return resp["Body"].read().decode("utf-8")


# ─── BIDS entity parsing ───────────────────────────────────────────────────────

def _parse_entities(s3_key: str) -> dict:
    """
    Parse BIDS entities from an S3 key such as:
        ds000001/sub-01/func/sub-01_task-rest_bold.nii.gz

    Returns a dict with canonical entity names, including
    subject, session, task, run, suffix, datatype, extension.
    """
    path = PurePosixPath(s3_key)
    # Drop the accession_id prefix
    parts = path.parts[1:]  # e.g. ('sub-01', 'func', 'sub-01_task-rest_bold.nii.gz')

    entities: dict = {}

    # Infer subject/session/datatype from directory components
    for part in parts[:-1]:
        if part.startswith("sub-"):
            entities["subject"] = part[4:]
        elif part.startswith("ses-"):
            entities["session"] = part[4:]
        elif part in _BIDS_DATATYPES:
            entities["datatype"] = part

    # Parse the filename
    filename = path.name

    # Split compound extension (.nii.gz before anything else)
    if filename.endswith(".nii.gz"):
        stem, extension = filename[:-7], ".nii.gz"
    elif "." in filename:
        dot = filename.index(".")
        stem, extension = filename[:dot], filename[dot:]
    else:
        stem, extension = filename, ""
    entities["extension"] = extension

    # Extract entity-value pairs from the stem
    for m in _ENTITY_RE.finditer(stem):
        key, val = m.group(1), m.group(2)
        canonical = _ENTITY_MAP.get(key, key)
        entities[canonical] = val

    # Suffix = last underscore token in stem that has no entity key prefix
    stem_parts = stem.split("_")
    if stem_parts:
        last = stem_parts[-1]
        if "-" not in last:
            entities["suffix"] = last

    return entities


def _choose_class(extension: str) -> str:
    if extension in {".nii", ".nii.gz", ".mgz", ".mgh"}:
        return "image_file"
    if extension in {".tsv", ".tsv.gz", ".csv"}:
        return "data_file"
    if extension == ".json":
        return "json_file"
    return "file"


# ─── Sidecar inheritance resolver ─────────────────────────────────────────────

class SidecarResolver:
    """
    Resolves BIDS JSON sidecar inheritance without pybids.

    BIDS spec: a JSON sidecar applies to a target file when all entities
    present in the sidecar's filename are a subset of the target file's
    entities (and they match).  More specific sidecars (more entity key/value
    pairs) override less specific ones.

    Load all fetched JSON files with load_json(), then call resolve(s3_key)
    to get the fully-merged metadata dict for any file.
    """

    def __init__(self, accession_id: str) -> None:
        self._prefix = accession_id + "/"
        # {relative_s3_key: parsed_json}  e.g. {"sub-01/func/sub-01_bold.json": {...}}
        self._sidecars: dict[str, dict] = {}

    def load_json(self, s3_key: str, content: str) -> None:
        rel = s3_key.removeprefix(self._prefix)
        try:
            self._sidecars[rel] = json.loads(content)
        except json.JSONDecodeError:
            log.warning("Invalid JSON at %s — skipped", s3_key)

    def resolve(self, s3_key: str) -> dict:
        """Return merged sidecar metadata for `s3_key`, most-specific wins."""
        target = _parse_entities(s3_key)

        candidates: list[tuple[int, dict]] = []
        for rel_key, data in self._sidecars.items():
            sidecar_entities = _parse_entities(self._prefix + rel_key)
            # Scope entities: subject/session/task/run only (not suffix/extension/datatype)
            scope_keys = {"subject", "session", "task", "run", "acquisition"}
            sidecar_scope = {k: v for k, v in sidecar_entities.items() if k in scope_keys}
            target_scope  = {k: v for k, v in target.items()           if k in scope_keys}

            # Sidecar applies if every entity in its scope matches the target
            if all(target_scope.get(k) == v for k, v in sidecar_scope.items()):
                candidates.append((len(sidecar_scope), data))

        # Merge least → most specific (most specific overwrites)
        merged: dict = {}
        for _, data in sorted(candidates, key=lambda x: x[0]):
            merged.update(data)
        return merged


# ─── Main ingester ─────────────────────────────────────────────────────────────

async def index_openneuro(accession_id: str, batch_size: int = 500) -> None:
    """
    Ghost-index an OpenNeuro dataset.

    - Lists all objects under s3://openneuro.org/{accession_id}/
    - Downloads only .json and .tsv files (sidecar metadata)
    - Upserts ALL files as BIDSObject rows (is_remote=True)
    - Resolves sidecar inheritance for each file
    - Stores imaging files as ghost rows — they are never downloaded

    Args:
        accession_id: OpenNeuro accession ID (e.g. "ds000001")
        batch_size:   Number of rows to upsert per DB round-trip
    """
    log.info("=== OpenNeuro: indexing %s ===", accession_id)
    client = _s3_client()

    # ── Phase 1: list all S3 objects ─────────────────────────────────────────
    log.info("  Listing objects for %s …", accession_id)
    all_objects = _list_objects(client, accession_id)
    if not all_objects:
        raise ValueError(
            f"No objects found for '{accession_id}'. "
            "Check the accession ID (format: ds000001)."
        )
    log.info("  Found %d object(s) in S3", len(all_objects))

    # ── Phase 2: fetch JSON sidecars ──────────────────────────────────────────
    resolver = SidecarResolver(accession_id)
    json_keys = [o["Key"] for o in all_objects if o["Key"].endswith(".json")]
    log.info("  Fetching %d JSON sidecar(s) …", len(json_keys))
    for key in json_keys:
        try:
            resolver.load_json(key, _fetch_text(client, key))
        except Exception as exc:
            log.warning("  [skip-json] %s — %s", key, exc)

    # ── Phase 3: parse dataset_description ───────────────────────────────────
    description = resolver._sidecars.get("dataset_description.json")
    if not description:
        raise ValueError(
            f"No dataset_description.json found for '{accession_id}'. "
            "This dataset may not be BIDS-formatted."
        )

    await create_all_tables()
    remote_url = f"s3://{OPENNEURO_BUCKET}/{accession_id}"

    async with async_session_maker() as session:
        # ── Phase 4: upsert BIDSDataset row ──────────────────────────────────
        # root_path uses a synthetic key so it never collides with local datasets
        root_path = f"openneuro:{accession_id}"
        stmt = (
            pg_insert(BIDSDataset)
            .values(
                id=uuid4(),
                root_path=root_path,
                description=description,
                name=description.get("Name", accession_id),
                bids_version=description.get("BIDSVersion", "unknown"),
                dataset_type="raw",
                parent_id=None,
                source_type="openneuro",
                accession_id=accession_id,
                remote_url=remote_url,
            )
            .on_conflict_do_update(
                index_elements=["root_path"],
                set_={
                    "description":  pg_insert(BIDSDataset).excluded.description,
                    "name":         pg_insert(BIDSDataset).excluded.name,
                    "bids_version": pg_insert(BIDSDataset).excluded.bids_version,
                    "remote_url":   pg_insert(BIDSDataset).excluded.remote_url,
                },
            )
        )
        await session.execute(stmt)
        await session.commit()

        dataset = await session.scalar(
            select(BIDSDataset).where(BIDSDataset.root_path == root_path)
        )

        # ── Phase 5: upsert one BIDSObject per S3 object ─────────────────────
        log.info("  Building object rows for %d file(s) …", len(all_objects))
        batch: list[dict] = []
        total = 0

        for s3_obj in all_objects:
            key = s3_obj["Key"]
            try:
                entities = _parse_entities(key)
                extension = entities.get("extension", "")

                # Only resolve sidecar for non-JSON files (JSON files are sidecars)
                metadata = resolver.resolve(key) if extension != ".json" else {}

                subject_label = entities.get("subject")
                session_label = entities.get("session")
                run_raw       = entities.get("run")

                overflow = {k: str(v) for k, v in entities.items()
                            if k not in _FIRST_CLASS_ENTITIES}
                other_entities = {**overflow, **metadata}

                # Synthetic path: "openneuro:ds000001/sub-01/func/..."
                # The relative part is everything after the accession prefix
                rel_path = key.removeprefix(accession_id + "/")

                batch.append(dict(
                    id=uuid4(),
                    dataset_id=dataset.id,
                    path=f"openneuro:{key}",
                    subject=subject_label,
                    subject_index=(
                        int(subject_label)
                        if subject_label and subject_label.isdigit()
                        else None
                    ),
                    session=session_label,
                    session_index=(
                        int(session_label)
                        if session_label and session_label.isdigit()
                        else None
                    ),
                    task=entities.get("task"),
                    run=int(run_raw) if run_raw and str(run_raw).isdigit() else None,
                    suffix=entities.get("suffix", ""),
                    datatype=entities.get("datatype"),
                    extension=extension,
                    other_entities=other_entities,
                    filename=PurePosixPath(key).name,
                    class_=_choose_class(extension),
                    is_remote=True,
                ))

                if len(batch) >= batch_size:
                    await _flush_objects(session, batch)
                    total += len(batch)
                    batch = []

            except Exception as exc:
                log.warning("  [skip] %s — %s", key, exc)

        if batch:
            await _flush_objects(session, batch)
            total += len(batch)

        await session.commit()

    log.info("  [db] Upserted %d object(s) for %s", total, accession_id)
    log.info("=== OpenNeuro: done (%s) ===", accession_id)


async def _flush_objects(session, batch: list[dict]) -> None:
    stmt = (
        pg_insert(BIDSObject)
        .values(batch)
        .on_conflict_do_update(
            index_elements=["path"],
            set_={
                "other_entities": pg_insert(BIDSObject).excluded.other_entities,
                "suffix":         pg_insert(BIDSObject).excluded.suffix,
                "datatype":       pg_insert(BIDSObject).excluded.datatype,
                "extension":      pg_insert(BIDSObject).excluded.extension,
                "is_remote":      pg_insert(BIDSObject).excluded.is_remote,
            },
        )
    )
    await session.execute(stmt)
