"""
main.py
-------
BIDS-Eye FastAPI application.

Startup (lifespan):
  - Creates missing DB tables.
  - Stops when a migration is missing; reads what the catalogue database
    holds into memory (services/database_stats/database_stats.py).
  - Gives the translation worker thread its database access (worker_db.py).
  - Starts the OpenNeuro crawler in this process when
    RUN_CRAWLER_IN_PROCESS=true (services/in_process_crawler.py); in Docker
    it runs as a separate service.

Routes (every /api route needs the signed session cookie, except the three
marked public; see api_security.py):
  GET  /api/auth/github          — public; redirect to GitHub's consent screen
  GET  /api/auth/github/callback — public; signs the user in (session cookie)
                                   and redirects to the frontend
  POST /api/auth/logout          — public; clears the session
  GET  /api/auth/me              — the signed-in user's GitHub login
  POST /api/query                — natural-language dataset search
  POST /api/query/page           — another page of a search's results
  GET  /api/history              — the user's saved searches, if they turned that on
  PUT  /api/history/enabled      — turn saving on, or off (deletes them)
  GET, DELETE /api/history/{id}  — one saved search
  GET  /health                   — liveness check, no login
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI

import db.models  # noqa: F401 — registers models with Base.metadata before create_all
import models     # noqa: F401 — registers backend-only models (query_log, etc.)
import request_log
import worker_db
from api_security import install_security_middleware
from db.db import create_all_tables, engine
from exceptions.handlers import add_error_handlers
from routers.auth import auth_router, require_auth
from routers.history import history_router
from routers.search import search_router
from services import in_process_crawler, switches
from services.database_stats.database_stats import database_stats
from services.translate import optional_columns

log = logging.getLogger(__name__)
# Text or JSON lines (LOG_FORMAT), each with its request's ID (request_log.py).
request_log.configure()

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Stop when a feature switch is neither true nor false.
    switches.check()

    # Ensure DB schema is up to date
    await create_all_tables()
    log.info("Database tables verified.")

    # Stop when a migration is missing, then read what the catalogue database
    # holds: counts per column and coded value, and the institution and place
    # names that questions are matched against
    # (services/database_stats/database_stats.py). A search reads them again
    # when they are 10 minutes old.
    await database_stats.start(engine)
    # Prompts mention only the columns enough datasets fill
    # (services/translate/optional_columns.py).
    optional_columns.log_changes(database_stats.coverage)

    # The translation worker thread asks the database for word counts
    # (services/fulltext/word_frequencies.py) and word forms
    # (services/fulltext/word_form_counts.py); its queries run here, on this
    # event loop, with the pooled engine.
    worker_db.install(worker_db.PooledWorkerDb(engine, asyncio.get_running_loop()))

    # The crawler in this process, when RUN_CRAWLER_IN_PROCESS=true
    # (services/in_process_crawler.py).
    async with in_process_crawler.running():
        yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BIDS-Eye",
    description="Natural-language search over BIDS neuroimaging datasets.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS, CSRF, the signed HttpOnly session cookie and login-by-default for /api
# (see api_security.py). Raises when SESSION_SECRET or a GitHub sign-in setting
# is missing or unusable (routers/auth.py, AuthSettings).
install_security_middleware(app)

# Outermost: an ID per request, on its log lines and in the X-Request-ID
# response header, and one log line per request (request_log.py).
app.add_middleware(request_log.RequestIdMiddleware)

# Every expected failure (exceptions/domain.py) answers with its own status
# and text, so the routers catch none of them; anything else gets a plain 500.
add_error_handlers(app)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(auth_router, prefix="/api")
app.include_router(search_router, prefix="/api", dependencies=[Depends(require_auth)])
app.include_router(history_router, prefix="/api", dependencies=[Depends(require_auth)])


@app.get("/health")
async def health():
    return {"status": "ok"}
