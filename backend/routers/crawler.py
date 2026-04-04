"""
routers/crawler.py
------------------
Crawler status endpoint.
Reads the shared state dict that crawler_service.py maintains.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "crawlers" / "openneuro-crawler-backup"))

from schemas import CrawlerStatusResponse

router = APIRouter(prefix="/crawler", tags=["crawler"])


@router.get("/status", response_model=CrawlerStatusResponse)
async def crawler_status():
    """Return the current state of the background crawler service."""
    try:
        from crawler_service import crawler_state
        return CrawlerStatusResponse(**crawler_state)
    except ImportError:
        # Crawler service not running in this process (e.g. separate container)
        return CrawlerStatusResponse(running=False)
