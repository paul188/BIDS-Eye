"""
main.py
-------
BIDS-Eye FastAPI application.

Startup:
  - Creates all DB tables (idempotent).
  - Starts the background OpenNeuro crawler scheduler (if this process
    should run it; in Docker the crawler runs as a separate service).

Routes:
  GET  /api/auth/github          — redirect to GitHub OAuth consent screen
  GET  /api/auth/github/callback — OAuth callback; sets JWT and redirects to frontend
  POST /api/auth/logout          — stateless logout
  POST /api/query                — natural-language dataset search (requires JWT)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import Depends, FastAPI

import db.models  # noqa: F401 — registers models with Base.metadata before create_all
import institution_index
import models     # noqa: F401 — registers backend-only models (query_log, etc.)
import place_index
import worker_db
from db.db import create_all_tables, engine
from services.catalogue.catalogue import catalogue

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Ensure DB schema is up to date
    await create_all_tables()
    log.info("Database tables verified.")

    # Stop when a migration is missing, then count what the catalogue holds:
    # prompts mention only the columns enough datasets fill
    # (services/catalogue/catalogue.py).
    await catalogue.start(engine)

    # Distinct bids_datasets.institutions values, for resolving institution
    # names in questions (RAG/institution_index.py). Falls back to
    # RAG/name_index.json when the column is missing.
    await institution_index.refresh_from_db(engine)
    # Cities, regions and countries of the sites (bids_site_places), for
    # place names in questions (RAG/place_index.py).
    await place_index.refresh_from_db(engine)

    # The translation worker thread asks the database for word counts
    # (services/fulltext/word_frequencies.py) and word forms
    # (services/catalogue/word_form_store.py); its queries run here, on this
    # event loop, with the pooled engine.
    worker_db.install(worker_db.PooledWorkerDb(engine, asyncio.get_running_loop()))

    # Optionally run the crawler in-process (useful for single-container dev).
    # In production the crawler runs as its own Docker service.
    crawler_scheduler = None
    if os.getenv("RUN_CRAWLER_IN_PROCESS", "false").lower() == "true":
        _crawler_path = Path(__file__).resolve().parents[1] / "crawlers" / "openneuro-crawler"
        if str(_crawler_path) not in sys.path:
            sys.path.insert(0, str(_crawler_path))
        try:
            from crawler_service import create_scheduler
            crawler_scheduler = create_scheduler()
            crawler_scheduler.start()
            log.info("Background crawler started in-process.")
        except (ImportError, ValueError) as exc:  # crawler not importable, bad CRAWL_INTERVAL_HOURS
            log.warning("Could not start crawler in-process: %s", exc)

    yield

    if crawler_scheduler is not None:
        crawler_scheduler.shutdown(wait=False)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BIDS-Eye",
    description="Natural-language search over BIDS neuroimaging datasets.",
    version="0.1.0",
    lifespan=lifespan,
)

from api_security import install_security_middleware               # noqa: E402

# CORS, CSRF, the signed HttpOnly session cookie and login-by-default for /api
# (see api_security.py). Raises when SESSION_SECRET is missing or too short.
install_security_middleware(app)

from exceptions.handlers import add_error_handlers                 # noqa: E402

# Every expected failure (exceptions/domain.py) answers with its own status
# and text, so the routers catch none of them; anything else gets a plain 500.
add_error_handlers(app)

# ── Routers ───────────────────────────────────────────────────────────────────

from routers.auth import require_auth, router as auth_router      # noqa: E402
from routers.query import router as query_router                   # noqa: E402

app.include_router(auth_router, prefix="/api")
app.include_router(query_router, prefix="/api", dependencies=[Depends(require_auth)])


@app.get("/health")
async def health():
    return {"status": "ok"}
