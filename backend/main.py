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

# ── Make BIDS-SQL importable ──────────────────────────────────────────────────
_BIDS_SQL = Path(__file__).resolve().parents[1] / "BIDS-SQL"
if str(_BIDS_SQL) not in sys.path:
    sys.path.insert(0, str(_BIDS_SQL))

import db.models  # noqa: E402, F401 — registers models with Base.metadata before create_all
import models     # noqa: E402, F401 — registers backend-only models (message_usage, etc.)
from db.db import create_all_tables, engine  # noqa: E402
from services.catalogue import schema_capabilities, word_form_store  # noqa: E402
from services.fulltext import text_search  # noqa: E402

_RAG = Path(__file__).resolve().parents[1] / "RAG"
if str(_RAG) not in sys.path:
    sys.path.insert(0, str(_RAG))
import institution_index  # noqa: E402
import place_index  # noqa: E402
from services.catalogue.schema_capabilities import has_fts, refresh_available_columns  # noqa: E402

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

    # Record which optional dataset columns (README, OpenNeuro metadata,
    # institutions) this database has filled; prompts only mention those.
    await refresh_available_columns(engine)

    # Distinct bids_datasets.institutions values, for resolving institution
    # names in questions (RAG/institution_index.py). Falls back to
    # RAG/name_index.json when the column is missing.
    await institution_index.refresh_from_db(engine)
    # Cities, regions and countries of the sites (bids_site_places), for
    # place names in questions (RAG/place_index.py).
    await place_index.refresh_from_db(engine)

    # Multi-word free-text terms filter on their rarest word; the count comes
    # from the database and is asked for from the translation worker thread.
    if has_fts():
        text_search.set_frequency_lookup(
            text_search.make_frequency_lookup(engine, asyncio.get_running_loop())
        )

    # The catalogue's word forms, asked for from the same worker thread: which
    # other words share a searched word's stem (services/catalogue/word_form_store.py).
    if schema_capabilities.has_word_forms():
        word_form_store.set_lookup(
            word_form_store.make_lookup(engine, asyncio.get_running_loop())
        )

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
        except Exception as exc:
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
# (see api_security.py). Raises when JWT_SECRET is missing or a placeholder.
install_security_middleware(app)

from exceptions.handlers import install_error_handlers             # noqa: E402

# Every expected failure (exceptions/domain.py) answers with its own status
# and text, so the routers catch none of them; anything else gets a plain 500.
install_error_handlers(app)

# ── Routers ───────────────────────────────────────────────────────────────────

from routers.auth import require_auth, router as auth_router      # noqa: E402
from routers.query import router as query_router                   # noqa: E402

app.include_router(auth_router, prefix="/api")
app.include_router(query_router, prefix="/api", dependencies=[Depends(require_auth)])


@app.get("/health")
async def health():
    return {"status": "ok"}
