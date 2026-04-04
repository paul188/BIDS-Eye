"""
crawler_service.py
------------------
Background crawler service for OpenNeuro datasets.

Runs continuously, cycling through a list of accession IDs and re-indexing
each one on a configurable schedule.  Resource usage is deliberately throttled:
  - One dataset at a time (asyncio.Semaphore(1))
  - Configurable inter-dataset sleep (default 30 s)
  - APScheduler triggers the main crawl loop on a fixed interval (default 1 h)

Configuration (environment variables):
    OPENNEURO_ACCESSIONS   Comma-separated accession IDs, e.g. "ds000001,ds000002"
                           Falls back to accessions.yaml in the same directory.
    CRAWL_INTERVAL_HOURS   How often to restart the full cycle (default: 1)
    CRAWL_DELAY_SECONDS    Sleep between consecutive datasets (default: 30)

The crawler state dict is module-level so FastAPI can import and expose it:
    from crawler_service import crawler_state
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler

log = logging.getLogger(__name__)

# ─── Shared state (read by FastAPI /api/crawler/status) ───────────────────────

crawler_state: dict = {
    "running": False,
    "current_accession": None,
    "queue": [],
    "last_run_started": None,
    "last_run_finished": None,
    "last_error": None,
    "indexed_count": 0,
    "error_count": 0,
}

_crawl_semaphore = asyncio.Semaphore(1)


# ─── Configuration ─────────────────────────────────────────────────────────────

def _load_accessions() -> list[str]:
    """
    Load accession IDs from OPENNEURO_ACCESSIONS env var or accessions.yaml.
    Returns an empty list if neither is available (crawler will be a no-op).
    """
    env_val = os.getenv("OPENNEURO_ACCESSIONS", "").strip()
    if env_val:
        return [a.strip() for a in env_val.split(",") if a.strip()]

    yaml_path = Path(__file__).parent / "accessions.yaml"
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, list):
                return [str(a) for a in data]
            if isinstance(data, dict):
                return [str(a) for a in data.get("accessions", [])]
        except Exception as exc:
            log.warning("Failed to load accessions.yaml: %s", exc)

    log.warning(
        "No accession IDs configured. Set OPENNEURO_ACCESSIONS or create accessions.yaml."
    )
    return []


# ─── Crawl logic ───────────────────────────────────────────────────────────────

async def _crawl_one(accession_id: str, delay_seconds: int) -> None:
    """Index a single dataset, honouring the concurrency semaphore."""
    async with _crawl_semaphore:
        crawler_state["current_accession"] = accession_id
        log.info("[crawler] Indexing %s …", accession_id)
        try:
            from remote.openneuro import index_openneuro
            await index_openneuro(accession_id)
            crawler_state["indexed_count"] += 1
            log.info("[crawler] Finished %s", accession_id)
        except Exception as exc:
            crawler_state["error_count"] += 1
            crawler_state["last_error"] = f"{accession_id}: {exc}"
            log.error("[crawler] Error indexing %s: %s", accession_id, exc, exc_info=True)
        finally:
            crawler_state["current_accession"] = None

        if delay_seconds > 0:
            log.debug("[crawler] Sleeping %ds before next dataset …", delay_seconds)
            await asyncio.sleep(delay_seconds)


async def run_crawl_cycle() -> None:
    """
    Index all configured accession IDs sequentially (one at a time).
    Called by APScheduler on the configured interval.
    """
    if crawler_state["running"]:
        log.info("[crawler] Previous cycle still running — skipping this trigger.")
        return

    accessions = _load_accessions()
    if not accessions:
        return

    delay = int(os.getenv("CRAWL_DELAY_SECONDS", "30"))
    crawler_state["running"] = True
    crawler_state["queue"] = list(accessions)
    crawler_state["last_run_started"] = datetime.now(timezone.utc).isoformat()
    log.info("[crawler] Starting crawl cycle: %d dataset(s)", len(accessions))

    try:
        for accession_id in accessions:
            if accession_id in crawler_state["queue"]:
                crawler_state["queue"].remove(accession_id)
            await _crawl_one(accession_id, delay)
    finally:
        crawler_state["running"] = False
        crawler_state["queue"] = []
        crawler_state["last_run_finished"] = datetime.now(timezone.utc).isoformat()
        log.info("[crawler] Crawl cycle complete.")


# ─── Service entry point ───────────────────────────────────────────────────────

def create_scheduler() -> AsyncIOScheduler:
    """
    Build and return a configured APScheduler.
    Call scheduler.start() inside a running asyncio event loop.
    """
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
    """Standalone entry point: run the crawler service forever."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
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
