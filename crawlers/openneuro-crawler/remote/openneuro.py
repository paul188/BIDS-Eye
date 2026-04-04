"""
remote/openneuro.py
-------------------
Ghost-indexer for public OpenNeuro datasets.

Strategy (mirrors datalad.py):
  1. List all objects under s3://openneuro.org/{accession_id}/ (anonymous boto3).
  2. Build a local mirror dir (~/.bids-sql/openneuro/{accession_id}/):
       - Real content for .json and .tsv files (metadata).
       - Empty stub files for imaging files (.nii.gz etc.) so pybids can see
         the full directory tree and extract entities from filenames.
  3. Delegate to input_pipeline.run_pipeline() for all entity extraction
     and DB insertion — reuses the battle-tested pybids logic.
  4. Stamp the resulting BIDSDataset row with source_type="openneuro",
     accession_id, and the canonical S3 remote_url.
  5. Mark all BIDSObject rows as is_remote=True (content lives on S3).

Usage:
    from remote.openneuro import index_openneuro
    await index_openneuro("ds000001")
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from sqlalchemy import select, update

from db.db import async_session_maker
from db.models import BIDSDataset, BIDSObject

log = logging.getLogger(__name__)

OPENNEURO_BUCKET = "openneuro.org"
_CACHE_DIR = Path.home() / ".bids-sql" / "openneuro"

# Extensions whose content we actually download (metadata only)
_METADATA_EXTENSIONS = frozenset({".json", ".tsv"})


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


def _fetch_bytes(client, key: str) -> bytes:
    resp = client.get_object(Bucket=OPENNEURO_BUCKET, Key=key)
    return resp["Body"].read()


# ─── Mirror builder ────────────────────────────────────────────────────────────

def _build_mirror(
    client,
    accession_id: str,
    all_objects: list[dict],
    mirror_dir: Path,
) -> None:
    """
    Populate mirror_dir to look like a local BIDS dataset:
      - Real file content for .json / .tsv (metadata pybids needs).
      - Empty stub files for all other types so pybids sees the full tree
        and can extract entities from filenames.
    """
    log.info("  [mirror] Building mirror at %s", mirror_dir)
    downloaded = skipped = stubbed = 0

    for obj in all_objects:
        key: str = obj["Key"]
        # Strip the accession_id prefix to get the relative path
        rel = key[len(accession_id) + 1:]  # "sub-01/func/sub-01_bold.nii.gz"
        if not rel:
            continue

        dest = mirror_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Determine file extension (handle .nii.gz)
        low = rel.lower()
        if low.endswith(".nii.gz"):
            ext = ".nii.gz"
        else:
            ext = Path(low).suffix

        if dest.exists():
            skipped += 1
            continue

        if ext in _METADATA_EXTENSIONS:
            try:
                dest.write_bytes(_fetch_bytes(client, key))
                downloaded += 1
            except Exception as exc:
                log.warning("  [mirror] Failed to download %s: %s", key, exc)
                dest.touch()   # stub so pybids still sees the file
                stubbed += 1
        else:
            dest.touch()
            stubbed += 1

    log.info(
        "  [mirror] Done — %d downloaded, %d stubbed, %d skipped (already present)",
        downloaded, stubbed, skipped,
    )


# ─── Post-pipeline stamps ─────────────────────────────────────────────────────

async def _stamp_dataset(
    mirror_root_path: str,
    accession_id: str,
    remote_url: str,
) -> BIDSDataset | None:
    """Set source_type, accession_id, and remote_url on the indexed dataset row."""
    async with async_session_maker() as session:
        dataset = await session.scalar(
            select(BIDSDataset).where(BIDSDataset.root_path == mirror_root_path)
        )
        if dataset is None:
            log.error(
                "Dataset not found in DB after pipeline (root_path=%s). "
                "The mirror may not be a valid BIDS dataset.",
                mirror_root_path,
            )
            return None

        await session.execute(
            update(BIDSDataset)
            .where(BIDSDataset.id == dataset.id)
            .values(
                source_type="openneuro",
                accession_id=accession_id,
                remote_url=remote_url,
            )
        )
        await session.commit()
        return dataset


async def _mark_objects_remote(dataset_id: uuid4) -> int:
    """
    Mark all BIDSObject rows for this dataset as is_remote=True.

    The pipeline sees real (but empty) stub files locally, so it sets
    is_remote=False by default.  This step corrects that — the actual
    content lives on S3, not in the local mirror.
    """
    async with async_session_maker() as session:
        result = await session.execute(
            update(BIDSObject)
            .where(BIDSObject.dataset_id == dataset_id)
            .values(is_remote=True)
        )
        await session.commit()
        return result.rowcount


# ─── Main entry point ─────────────────────────────────────────────────────────

async def index_openneuro(
    accession_id: str,
    force_rebuild: bool = False,
) -> None:
    """
    Ghost-index an OpenNeuro dataset via the standard input_pipeline.

    Args:
        accession_id:  OpenNeuro accession ID (e.g. "ds000001").
        force_rebuild: Discard and re-download the local mirror even if it
                       already exists (useful after upstream dataset updates).
    """
    log.info("=== OpenNeuro: indexing %s ===", accession_id)

    mirror_dir = _CACHE_DIR / accession_id
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: list all S3 objects ───────────────────────────────────────────
    client = _s3_client()
    log.info("  Listing objects for %s …", accession_id)
    all_objects = _list_objects(client, accession_id)
    if not all_objects:
        raise ValueError(
            f"No objects found for '{accession_id}'. "
            "Check the accession ID (format: ds000001)."
        )
    log.info("  Found %d object(s) in S3", len(all_objects))

    # ── Step 2: build (or reuse) local mirror ─────────────────────────────────
    if mirror_dir.exists() and force_rebuild:
        import shutil
        log.info("  [mirror] Removing existing mirror at %s", mirror_dir)
        shutil.rmtree(mirror_dir)

    mirror_dir.mkdir(exist_ok=True)
    _build_mirror(client, accession_id, all_objects, mirror_dir)

    # ── Step 3: run standard indexing pipeline ────────────────────────────────
    # Import here to keep the module importable without bids-sql installed.
    from input_pipeline import run_pipeline

    log.info("  [pipeline] Running local indexing pipeline on %s", mirror_dir)
    await run_pipeline(str(mirror_dir), skip_validation=True)

    # ── Step 4: stamp the dataset row with OpenNeuro provenance ───────────────
    remote_url = f"s3://{OPENNEURO_BUCKET}/{accession_id}"
    dataset = await _stamp_dataset(str(mirror_dir.resolve()), accession_id, remote_url)
    if dataset is None:
        return

    # ── Step 5: mark all objects as remote ────────────────────────────────────
    n = await _mark_objects_remote(dataset.id)
    log.info("  [db] Marked %d object(s) as remote for %s", n, accession_id)
    log.info("=== OpenNeuro: done (%s) ===", accession_id)
