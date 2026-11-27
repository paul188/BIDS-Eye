"""
remote/datalad.py
-----------------
Ghost-indexer for public DataLad repositories.

Strategy:
  1. Clone the repository into a persistent local cache (~/.bids-sql/datalad/<name>/).
     The clone contains the full directory tree but imaging files are broken symlinks —
     only metadata files (.json, .tsv) are present as real content.
  2. Run `datalad get **/*.json **/*.tsv` to download metadata.
  3. Delegate to the local pipeline (input_pipeline.run_pipeline) for entity extraction
     and DB insertion — this reuses the battle-tested pybids logic.
  4. Stamp the resulting BIDSDataset row with source_type="datalad" and the original URL.
  5. Mark all BIDSObject rows as is_remote=True because imaging content is not local.

Public DataLad repositories:
  - OpenNeuro via DataLad:    https://github.com/OpenNeuroDatasets/<accession_id>
  - GIN (G-Node):             https://gin.g-node.org/<org>/<repo>
  - datasets.datalad.org:     https://datasets.datalad.org/?dir=/<path>
  - OSF:                      osf:///<project_id>  (requires datalad-osf)

Requires datalad and git-annex:
    pip install datalad
    # git-annex must be installed separately — see https://www.datalad.org/get_datalad.html

Usage:
    from remote.datalad import index_datalad
    await index_datalad("https://github.com/OpenNeuroDatasets/ds000001")
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, update

from db.db import async_session_maker
from db.models import BIDSDataset, BIDSObject

log = logging.getLogger(__name__)

# Local cache directory for datalad clones
_CACHE_DIR = Path.home() / ".bids-sql" / "datalad"


def _datalad_available() -> bool:
    return shutil.which("datalad") is not None


def _git_annex_available() -> bool:
    return shutil.which("git-annex") is not None


def _safe_dirname(url: str) -> str:
    """Convert a URL into a filesystem-safe directory name."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", url.rstrip("/").split("/")[-1])


def _clone(url: str, dest: Path) -> None:
    log.info("  [datalad] Cloning %s → %s", url, dest)
    subprocess.run(
        ["datalad", "clone", url, str(dest)],
        check=True, text=True, capture_output=True,
    )


def _get_metadata(clone_dir: Path) -> None:
    """Download only metadata files inside the clone."""
    log.info("  [datalad] Fetching metadata files (*.json, *.tsv) …")
    subprocess.run(
        ["datalad", "get", "--jobs", "4",
         "**/*.json", "**/*.tsv", "dataset_description.json", "participants.tsv"],
        check=True, text=True, capture_output=True,
        cwd=str(clone_dir),
    )


async def _stamp_dataset(clone_root_path: str, source_url: str) -> BIDSDataset | None:
    """Set source_type and remote_url on the dataset row that was just indexed."""
    async with async_session_maker() as session:
        dataset = await session.scalar(
            select(BIDSDataset).where(BIDSDataset.root_path == clone_root_path)
        )
        if dataset is None:
            log.error("Dataset not found in DB after local indexing (root_path=%s)", clone_root_path)
            return None

        await session.execute(
            update(BIDSDataset)
            .where(BIDSDataset.id == dataset.id)
            .values(source_type="datalad", remote_url=source_url)
        )
        await session.commit()
        return dataset


async def _mark_objects_remote(dataset_id: uuid4) -> int:
    """
    Mark all BIDSObject rows for this dataset as is_remote=True.

    After datalad get, only .json/.tsv files were downloaded; imaging files
    are broken symlinks.  Marking them all remote reflects the fact that
    content cannot be read from the local clone path.
    """
    async with async_session_maker() as session:
        result = await session.execute(
            update(BIDSObject)
            .where(BIDSObject.dataset_id == dataset_id)
            .values(is_remote=True)
        )
        await session.commit()
        return result.rowcount


async def index_datalad(
    url: str,
    skip_validation: bool = True,
    force_reclone: bool = False,
) -> None:
    """
    Ghost-index a public DataLad repository.

    Args:
        url:              DataLad/git repository URL.
        skip_validation:  Skip bids-validator (default True — the clone may
                          have broken symlinks that confuse the validator).
        force_reclone:    Delete and re-clone even if a local clone exists.
    """
    if not _datalad_available():
        raise RuntimeError(
            "datalad is not installed.\n"
            "  pip install datalad\n"
            "  (git-annex is also required — see https://www.datalad.org/get_datalad.html)"
        )
    if not _git_annex_available():
        raise RuntimeError(
            "git-annex is not found on PATH.\n"
            "  Install it via your system package manager or conda:\n"
            "    conda install -c conda-forge git-annex"
        )

    # ── Step 1: clone into persistent cache ───────────────────────────────────
    clone_dir = _CACHE_DIR / _safe_dirname(url)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if clone_dir.exists() and force_reclone:
        log.info("  [datalad] Removing existing clone at %s", clone_dir)
        shutil.rmtree(clone_dir)

    if not clone_dir.exists():
        try:
            _clone(url, clone_dir)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"datalad clone failed for {url!r}:\n{exc.stderr}"
            ) from exc
    else:
        log.info("  [datalad] Reusing existing clone at %s", clone_dir)

    # ── Step 2: get metadata files ─────────────────────────────────────────────
    try:
        _get_metadata(clone_dir)
    except subprocess.CalledProcessError as exc:
        # Non-fatal: some repos may not have all the globs; log and continue.
        log.warning("  [datalad] get returned non-zero: %s", exc.stderr.strip())

    # ── Step 3: run local pipeline ─────────────────────────────────────────────
    # Import here to avoid circular imports (input_pipeline imports from models).
    from input_pipeline import run_pipeline

    log.info("  [datalad] Running local indexing pipeline on %s", clone_dir)
    await run_pipeline(str(clone_dir), skip_validation=skip_validation)

    # ── Step 4: stamp the dataset row ─────────────────────────────────────────
    dataset = await _stamp_dataset(str(clone_dir.resolve()), url)
    if dataset is None:
        return

    # ── Step 5: mark all objects as remote ────────────────────────────────────
    n = await _mark_objects_remote(dataset.id)
    log.info("  [db] Marked %d object(s) as remote for %s", n, dataset.name)
    log.info("=== DataLad: done (%s) ===", url)
