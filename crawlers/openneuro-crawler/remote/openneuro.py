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
import os
from pathlib import Path
from uuid import uuid4

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from sqlalchemy import select, update

from db.db import async_session_maker
from db.models import BIDSDataset, BIDSObject

log = logging.getLogger(__name__)


class DatasetAccessDeniedError(Exception):
    """Raised when the dataset's core metadata is inaccessible on S3."""


OPENNEURO_BUCKET = "openneuro.org"
_CACHE_DIR = Path(os.environ.get("BIDS_SQL_CACHE", Path.home() / ".bids-sql")) / "openneuro"

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

        # Skip macOS resource fork files (._filename) — they are binary AppleDouble
        # metadata, never valid JSON, and pybids will choke trying to parse them.
        basename = rel.split("/")[-1]
        if basename.startswith("._"):
            skipped += 1
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
            # Re-try dataset_description.json if it's a 0-byte stub from a
            # previous failed/restricted crawl — it needs real content
            if rel == "dataset_description.json" and dest.stat().st_size == 0:
                pass  # fall through and re-download
            elif ext == ".json":
                # Validate cached JSON — re-download if corrupt (e.g. written
                # before JSON validation was added, or has invalid escapes)
                import json as _json
                try:
                    _json.loads(dest.read_bytes())
                    skipped += 1
                    continue
                except Exception:
                    dest.unlink()  # corrupt — fall through to re-download
            else:
                skipped += 1
                continue

        if ext in _METADATA_EXTENSIONS:
            try:
                raw = _fetch_bytes(client, key)
                if ext == ".json":
                    # Ensure clean UTF-8: fix latin-1 encoding and strip UTF-8 BOM
                    try:
                        raw.decode("utf-8")
                    except UnicodeDecodeError:
                        raw = raw.decode("latin-1").encode("utf-8")
                    raw = raw.lstrip(b"\xef\xbb\xbf")  # strip UTF-8 BOM if present
                    # Validate JSON — stub out malformed files so pybids doesn't crash
                    import json as _json
                    try:
                        _json.loads(raw)
                    except _json.JSONDecodeError as je:
                        # Try a sequence of repairs before giving up.
                        import re as _re
                        text = raw.decode("utf-8", errors="replace")
                        # Normalise Windows line endings so \n patterns work reliably
                        repaired = text.replace("\r\n", "\n").replace("\r", "\n")

                        # Repair 1: trailing commas before } or ]
                        repaired = _re.sub(r",(\s*[}\]])", r"\1", repaired)

                        # Repair 2: missing commas between properties
                        # (closing } immediately followed by whitespace then a new key)
                        repaired = _re.sub(r"(\})([ \t]*\n[ \t]*)(\"|')", r"\1,\2\3", repaired)

                        # Repair 3: invalid backslash escapes (e.g. \' or \p)
                        # JSON only allows: \" \\ \/ \b \f \n \r \t \uXXXX
                        _VALID_ESCAPES = set('"\\\/bfnrtu')
                        def _fix_escapes(m):
                            ch = m.group(1)
                            return '\\' + ch if ch in _VALID_ESCAPES else ch
                        repaired = _re.sub(r'\\(.)', _fix_escapes, repaired)

                        # Repair 4: raw control characters (ASCII 0x00-0x1F except
                        # tab/newline) embedded inside string values crash the parser
                        repaired = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', repaired)

                        # Repair 5: single-quoted keys → double-quoted
                        # Only replace when the key is at the start of a JSON property
                        # position (preceded by { or , and optional whitespace)
                        repaired = _re.sub(r"(?<=[{,])\s*'([^']+)'\s*:", r' "\1":', repaired)

                        try:
                            _json.loads(repaired)
                            log.warning("  [mirror] Repaired JSON in %s (%s)", key, je)
                            raw = repaired.encode("utf-8")
                        except _json.JSONDecodeError:
                            log.warning("  [mirror] Malformed JSON in %s (%s) — stubbing", key, je)
                            raw = b"{}"
                dest.write_bytes(raw)
                downloaded += 1
            except Exception as exc:
                is_access_denied = "AccessDenied" in str(exc)
                log.warning("  [mirror] Failed to download %s: %s", key, exc)
                # Any AccessDenied means the dataset is restricted — bail immediately
                # and let the GraphQL fallback handle it rather than wasting time
                # trying to download hundreds more files.
                if is_access_denied:
                    raise DatasetAccessDeniedError(
                        f"{accession_id}: access denied on {rel}"
                    )
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

        # Extract first-class fields from description JSON (already stored by indexer)
        desc = dataset.description or {}
        raw_authors = desc.get("Authors")
        authors    = raw_authors if isinstance(raw_authors, list) else ([raw_authors] if raw_authors else None)
        raw_refs   = desc.get("ReferencesAndLinks")
        paper_references = raw_refs if isinstance(raw_refs, list) else ([raw_refs] if raw_refs else None)
        raw_fund   = desc.get("Funding")
        funding    = raw_fund if isinstance(raw_fund, list) else ([raw_fund] if raw_fund else None)
        description_text = desc.get("Description") or None
        if description_text:
            description_text = description_text.strip() or None

        # Strip promoted fields from the JSON blob — they live in dedicated columns now
        _PROMOTED = {"Authors", "License", "DatasetDOI", "ReferencesAndLinks", "Funding", "Description"}
        clean_desc = {k: v for k, v in desc.items() if k not in _PROMOTED}

        await session.execute(
            update(BIDSDataset)
            .where(BIDSDataset.id == dataset.id)
            .values(
                source_type="openneuro",
                accession_id=accession_id,
                remote_url=remote_url,
                authors=authors,
                license=desc.get("License") or None,
                doi=desc.get("DatasetDOI") or None,
                paper_references=paper_references,
                funding=funding,
                description_text=description_text,
                description=clean_desc,
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
    try:
        _build_mirror(client, accession_id, all_objects, mirror_dir)
    except DatasetAccessDeniedError:
        log.warning(
            "  Dataset %s is restricted (S3 access denied). "
            "Falling back to GraphQL ingestion.", accession_id
        )
        from remote.graphql import index_via_graphql
        await index_via_graphql(accession_id)
        return

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
