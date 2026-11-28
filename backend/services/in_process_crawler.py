"""
services/in_process_crawler.py
------------------------------
The OpenNeuro crawler run inside the API process.

For development with a single container: RUN_CRAWLER_IN_PROCESS=true. In
production the crawler runs as its own Docker service, and the backend image
does not contain it.

main.py wraps the time the API serves requests in running():

    async with in_process_crawler.running():
        yield
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from config.settings import settings_obj

log = logging.getLogger(__name__)

# <repo>/ingest/openneuro-crawler; this file is <repo>/backend/services/….
CRAWLER_DIR = Path(__file__).resolve().parents[2] / "ingest" / "openneuro-crawler"


def wanted() -> bool:
    """True when RUN_CRAWLER_IN_PROCESS=true."""
    return settings_obj.RUN_CRAWLER_IN_PROCESS


def start() -> Any | None:
    """Start the crawler's scheduler on the running event loop.

    Returns the scheduler, or None when the crawler could not be started; the
    API then runs without it.
    """
    # The directory name has a hyphen, so it is no package: import from its path.
    if str(CRAWLER_DIR) not in sys.path:
        sys.path.insert(0, str(CRAWLER_DIR))
    try:
        from crawler_service import create_scheduler
        scheduler = create_scheduler()
        scheduler.start()
    except (ImportError, ValueError) as exc:  # crawler not importable, bad CRAWL_INTERVAL_HOURS
        log.warning("Could not start crawler in-process: %s", exc)
        return None
    log.info("Background crawler started in-process.")
    return scheduler


@asynccontextmanager
async def running() -> AsyncGenerator[None, None]:
    """Run the crawler for as long as the block lasts, when it is wanted()."""
    scheduler = start() if wanted() else None
    try:
        yield
    finally:
        if scheduler is not None:
            # Do not wait for a crawl cycle that is still running.
            scheduler.shutdown(wait=False)
