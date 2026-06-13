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
  GET  /api/crawler/status       — crawler background service state (requires JWT)
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

# ── Make BIDS-SQL importable ──────────────────────────────────────────────────
_BIDS_SQL = Path(__file__).resolve().parents[1] / "BIDS-SQL"
if str(_BIDS_SQL) not in sys.path:
    sys.path.insert(0, str(_BIDS_SQL))

import db.models  # noqa: E402, F401 — registers models with Base.metadata before create_all
import models     # noqa: E402, F401 — registers backend-only models (message_usage, etc.)
from db.db import create_all_tables  # noqa: E402

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

app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "change-me"))

_frontend_url = os.getenv("FRONTEND_URL", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend_url] if _frontend_url != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

from routers.auth import require_auth, router as auth_router      # noqa: E402
from routers.query import router as query_router                   # noqa: E402

app.include_router(auth_router, prefix="/api")
app.include_router(query_router, prefix="/api", dependencies=[Depends(require_auth)])

if os.getenv("CRAWLER_ENABLED", "false").lower() == "true":
    from routers.crawler import router as crawler_router           # noqa: E402
    app.include_router(crawler_router, prefix="/api", dependencies=[Depends(require_auth)])


@app.get("/health")
async def health():
    return {"status": "ok"}
