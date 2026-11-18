"""
routers/crawler.py
------------------
Crawler status endpoint.
Reads the shared state dict that crawler_service.py maintains when the
crawler runs in this process (RUN_CRAWLER_IN_PROCESS=true, see main.py).
"""

from __future__ import annotations

import sys

from fastapi import APIRouter

from rate_limits import Category, rate_limit
from schemas import CrawlerStatusResponse

router = APIRouter(prefix="/crawler", tags=["crawler"])


@router.get("/status", response_model=CrawlerStatusResponse, dependencies=[rate_limit(Category.LIGHT)])
async def crawler_status():
    """Return the current state of the background crawler service."""
    # Only report state if main.py already loaded the crawler in this process.
    # Importing it here would load a fresh copy whose default values look like
    # real status. When the crawler runs in its own container, its state is not
    # visible from here.
    crawler_service = sys.modules.get("crawler_service")
    if crawler_service is None:
        return CrawlerStatusResponse(running=False)
    return CrawlerStatusResponse(**crawler_service.crawler_state)
