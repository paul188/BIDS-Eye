"""
crawler_service.py
------------------
Background crawler service for OpenNeuro datasets.

Each cycle (APScheduler, every CRAWL_INTERVAL_HOURS):
  1. Takes the catalogue: every dataset in OpenNeuro's public S3 bucket plus
     every dataset already in the database (CRAWL_CATALOGUE=s3, the default),
     or accessions.yaml (CRAWL_CATALOGUE=file), or OPENNEURO_ACCESSIONS.
  2. Asks the OpenNeuro API for each dataset's metadata and latest snapshot,
     a dozen datasets per request, and decides which datasets to index again
     (remote/snapshot_plan.py): new ones, those with a new snapshot, and a few
     more. All others keep their files; their API metadata (README, uploader
     fields, downloads, deprecation) is written from the same answers.
  3. Indexes those datasets one at a time (remote/openneuro.py), with
     CRAWL_DELAY_SECONDS between them, at most CRAWL_MAX_PER_CYCLE.
  4. Runs the post-crawl steps (post_crawl.py).

Configuration (environment variables):
    OPENNEURO_ACCESSIONS   Comma-separated accession IDs; only these are crawled
    CRAWL_CATALOGUE        "s3" (default) or "file" (accessions.yaml)
    CRAWL_INTERVAL_HOURS   How often a cycle starts (default: 1)
    CRAWL_DELAY_SECONDS    Sleep after each indexed dataset (default: 30)
    CRAWL_MAX_PER_CYCLE    Most datasets indexed per cycle; 0 = all (default: 0)
    CRAWL_MAX_AGE_DAYS     Index a dataset again after this many days (default: 30)
    CRAWL_FORCE            "true": index every dataset again this cycle
    CRAWL_MAX_FILES        Skip datasets with more files (remote/openneuro.py)

    python crawler_service.py          run as a service, a cycle per interval
    python crawler_service.py --once   run one cycle and exit
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # imported where it is used, so tests need no APScheduler
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

log = logging.getLogger(__name__)

# ─── Crawl state (progress counters for the log) ──────────────────────────────

crawler_state: dict = {
    "running": False,
    "current_accession": None,
    "queue": [],
    "last_run_started": None,
    "last_run_finished": None,
    "last_error": None,
    "indexed_count": 0,
    "error_count": 0,
    "deferred_count": 0,
}

_crawl_semaphore = asyncio.Semaphore(1)


# ─── Configuration ─────────────────────────────────────────────────────────────

def _env_number(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        log.warning("[crawler] %s is not a number; using %s", name, default)
        return default


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


def _load_accessions_file() -> list[str]:
    """Accession IDs from accessions.yaml; empty when it is missing or broken."""
    yaml_path = Path(__file__).parent / "accessions.yaml"
    if yaml_path.exists():
        import yaml
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as exc:
            log.error("Failed to load accessions.yaml, nothing is crawled: %s", exc)
            return []
        if isinstance(data, list):
            return [str(a) for a in data]
        if isinstance(data, dict):
            return [str(a) for a in data.get("accessions", [])]
    log.warning("No accessions.yaml next to the crawler.")
    return []


def _load_accessions(known: List[str]) -> list[str]:
    """The datasets of this cycle; *known* are those in the database.

    OPENNEURO_ACCESSIONS wins. Otherwise CRAWL_CATALOGUE: "s3" lists the
    public bucket and adds the known datasets (so a dataset removed from S3
    gets its deprecation mark from the API); "file" reads accessions.yaml.
    """
    env_val = os.getenv("OPENNEURO_ACCESSIONS", "").strip()
    if env_val:
        return [a.strip() for a in env_val.split(",") if a.strip()]
    if os.getenv("CRAWL_CATALOGUE", "s3").strip().lower() == "file":
        return _load_accessions_file()
    from fetch_accessions import list_all_accessions
    try:
        listed = list_all_accessions()
    except Exception as exc:
        log.error("[crawler] Could not list the OpenNeuro bucket: %s", exc)
        listed = []
    if not listed:
        # Without the listing only the known datasets are checked this cycle.
        log.warning("[crawler] Empty OpenNeuro catalogue; checking the %d known datasets", len(known))
    return sorted(set(listed) | set(known))


# ─── Crawl logic ───────────────────────────────────────────────────────────────

async def _crawl_one(accession_id: str, delay_seconds: float,
                     record: Optional[dict] = None) -> bool:
    """Index a single dataset, honouring the concurrency semaphore.
    True when it was indexed."""
    from remote.openneuro import DatasetTooLargeError, index_openneuro

    indexed = False
    async with _crawl_semaphore:
        crawler_state["current_accession"] = accession_id
        log.info("[crawler] Indexing %s …", accession_id)
        try:
            await index_openneuro(accession_id, record=record)
            indexed = True
            crawler_state["indexed_count"] += 1
            print(
                f"[DONE] {accession_id} | "
                f"indexed={crawler_state['indexed_count']} "
                f"errors={crawler_state['error_count']} | "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
                flush=True,
            )
        except DatasetTooLargeError as exc:
            crawler_state["deferred_count"] += 1
            print(f"[DEFER] {exc}", flush=True)
            log.warning("[crawler] Deferred: %s", exc)
        except Exception as exc:
            crawler_state["error_count"] += 1
            crawler_state["last_error"] = f"{accession_id}: {exc}"
            print(
                f"[FAIL] {accession_id} | {exc} | "
                f"indexed={crawler_state['indexed_count']} "
                f"errors={crawler_state['error_count']} | "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
                flush=True,
            )
            log.error("[crawler] Error indexing %s: %s", accession_id, exc, exc_info=True)
        finally:
            crawler_state["current_accession"] = None

        if indexed and delay_seconds > 0:
            log.debug("[crawler] Sleeping %ss before next dataset …", delay_seconds)
            await asyncio.sleep(delay_seconds)
    return indexed


async def _refresh_metadata(plan, stored, skip: set) -> List[str]:
    """Write the API metadata of the datasets that are not indexed this cycle.

    Returns those whose README or uploader fields changed: the post-crawl text
    steps (profiles, descriptions) run for them.
    """
    from db.db import async_session_maker
    from remote.openneuro_metadata import apply_fields, deprecation_reason, extract_fields, \
        set_deprecation
    from remote.snapshot_plan import Action, texts_changed

    changed: List[str] = []
    async with async_session_maker() as session:
        for accession, action in plan.actions.items():
            if accession in skip or accession not in stored:
                continue
            record = plan.records.get(accession)
            if action is Action.REMOVED:
                await set_deprecation(session, accession, plan.unavailable[accession])
            elif record is not None:
                if texts_changed(stored[accession], record):
                    changed.append(accession)
                await apply_fields(session, accession, extract_fields(record))
                await set_deprecation(session, accession, deprecation_reason(record))
        await session.commit()
    return changed


async def run_crawl_cycle() -> None:
    """
    Plan the cycle, index what changed (one dataset at a time), refresh the
    API metadata of the rest, then run the post-crawl steps.
    Called by APScheduler on the configured interval.
    """
    if crawler_state["running"]:
        log.info("[crawler] Previous cycle still running — skipping this trigger.")
        return

    from db.db import engine
    from remote.snapshot_plan import fetch_records, load_stored, make_plan

    crawler_state["running"] = True
    crawler_state["last_run_started"] = datetime.now(timezone.utc).isoformat()
    try:
        async with engine.connect() as conn:
            stored = await load_stored(conn)
        accessions = await asyncio.to_thread(_load_accessions, sorted(stored))
        if not accessions:
            return
        log.info("[crawler] Checking %d dataset(s) against the OpenNeuro API …", len(accessions))
        records, unavailable, failed = await asyncio.to_thread(fetch_records, accessions)
        plan = make_plan(
            accessions, records, unavailable, stored, datetime.now(timezone.utc),
            max_age=timedelta(days=_env_number("CRAWL_MAX_AGE_DAYS", 30)),
            force=_flag("CRAWL_FORCE"))
        to_index = plan.to_index(int(_env_number("CRAWL_MAX_PER_CYCLE", 0)))
        log.info("[crawler] Plan: %s; API failed for %d; indexing %d this cycle",
                 plan.counts(), len(failed), len(to_index))

        texts = await _refresh_metadata(plan, stored, set(to_index))

        delay = _env_number("CRAWL_DELAY_SECONDS", 30)
        crawler_state["queue"] = list(to_index)
        indexed: list[str] = []
        for accession_id in to_index:
            if accession_id in crawler_state["queue"]:
                crawler_state["queue"].remove(accession_id)
            if await _crawl_one(accession_id, delay, plan.records.get(accession_id)):
                indexed.append(accession_id)

        # Papers and dataset descriptions for what changed. Never raises.
        from post_crawl import run_post_crawl
        # The paper step skips datasets checked within PAPER_RECHECK_DAYS itself.
        checked = sorted((set(accessions) & set(stored)) | set(indexed))
        await run_post_crawl(sorted(set(indexed) | set(texts)), files=indexed, papers=checked)
    finally:
        crawler_state["running"] = False
        crawler_state["queue"] = []
        crawler_state["last_run_finished"] = datetime.now(timezone.utc).isoformat()
        log.info("[crawler] Crawl cycle complete: indexed=%d errors=%d deferred=%d",
                 crawler_state["indexed_count"], crawler_state["error_count"],
                 crawler_state["deferred_count"])


# ─── Service entry point ───────────────────────────────────────────────────────

def create_scheduler() -> AsyncIOScheduler:
    """
    Build and return a configured APScheduler.
    Call scheduler.start() inside a running asyncio event loop.
    """
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    interval_hours = float(os.getenv("CRAWL_INTERVAL_HOURS", "1"))
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_crawl_cycle,
        trigger="interval",
        hours=interval_hours,
        id="openneuro_crawl",
        name="OpenNeuro crawl cycle",
        # Run immediately on startup in addition to the interval
        next_run_time=datetime.now(timezone.utc),
    )
    log.info(
        "[crawler] Scheduled crawl every %.1f hour(s), starting immediately.",
        interval_hours,
    )
    return scheduler


async def main() -> None:
    """Standalone entry point: run the crawler service forever, or one cycle (--once)."""
    parser = argparse.ArgumentParser(description="OpenNeuro crawler")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    if args.once:
        await run_crawl_cycle()
        return
    scheduler = create_scheduler()
    scheduler.start()
    log.info("[crawler] Service started. Press Ctrl-C to stop.")
    try:
        # Keep the event loop alive
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        log.info("[crawler] Shutting down …")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())
