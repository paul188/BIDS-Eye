"""
post_crawl.py
-------------
Steps that run after a crawl cycle, for the datasets the cycle indexed:

  -. summaries  per-dataset summaries of the file table that the search
                reads (remote/dataset_facets.refresh). No network.
  -. groups     participant diagnoses from a group column and participants.json
                (remote/participant_groups.run). OpenNeuro's public S3 only.
  0. links      derivatives (OpenNeuro's preprocessing bucket and the
                dataset's own derivatives/ folders) and scanner-header sites
                (remote/dataset_links.refresh_links). No LLM, public sources only.
  1. papers     find and look up the papers each dataset names
                (remote/paper_store.enrich: Europe PMC, bioRxiv/medRxiv, arXiv,
                OpenAlex). Only datasets whose papers were never checked, or
                were checked more than PAPER_RECHECK_DAYS ago (default 30).
  -. affiliations  institutions of the first and last author of each primary
                paper, from OpenAlex (remote/paper_affiliations.refresh).
  -. places    city, region and country of each new site name, from the
                Research Organization Registry (remote/site_places.match_new).
  2. descriptions  how each primary paper relates to its dataset, and a
                generated summary where the README is thin
                (backend/services/dataset_description.py, the same code as
                scripts/describe_datasets.py). Only what is missing or stale.
                Needs GEMINI_API_KEY; the text goes to Gemini only.
  3. word forms  rebuild bids_word_forms, the catalogue's word forms per stem
                (backend/services/word_form_store.py). No LLM, no network;
                the whole catalogue, not only the indexed datasets.
  4. vacuum     VACUUM (ANALYZE) of the file table, bids_objects. A search
                counts subjects from an index alone
                (crawlers/migrate_add_object_indexes.sql); for the rows a
                crawl wrote that only works again after a vacuum, and
                autovacuum starts only once a fifth of the table changed.

Environment:
    CRAWL_PARTICIPANT_GROUPS "false" skips decoding participant groups (default on)
    CRAWL_LINKS             "false" skips step 0 (default on)
    CRAWL_SITE_PLACES       "false" skips matching site names to places (default on)
    CRAWL_PAPER_AFFILIATIONS "false" skips the lead authors' institutions as sites (default on)
    CRAWL_PAPERS            "false" skips step 1 (default on)
    CRAWL_DESCRIPTIONS      "false" skips step 2 (default on)
    CRAWL_WORD_FORMS        "false" skips step 3 (default on)
    CRAWL_VACUUM            "false" skips step 4 (default on)
    PAPER_RECHECK_DAYS      how old a paper lookup may get (default 30)
    DATASET_DESCRIPTION_MODEL  Gemini model for step 2 (default gemini-2.5-flash)

The steps create the tables and columns they write, skip what is already
done, and never fail the crawl: an error is logged and the next cycle tries
again. The paper lookup makes blocking HTTP calls, so with
RUN_CRAWLER_IN_PROCESS=true it pauses the API while it runs; the crawler is
meant to run as its own service.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from sqlalchemy import text

log = logging.getLogger(__name__)

# backend/ and LLM_preprocessor/ sit in the repository root, or next to the
# crawler in its Docker image (/app/crawler, /app/backend, /app/LLM_preprocessor).
_HERE = Path(__file__).resolve()
_ROOTS = (_HERE.parents[2], _HERE.parents[1])


def _flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


async def _needs_paper_check(conn, accessions: Sequence[str]) -> List[str]:
    days = float(os.getenv("PAPER_RECHECK_DAYS", "30"))
    rows = await conn.execute(text(
        "SELECT accession_id FROM bids_datasets WHERE accession_id = ANY(:acc) "
        "AND (papers_checked_at IS NULL "
        "     OR papers_checked_at < now() - make_interval(secs => :secs))"),
        {"acc": list(accessions), "secs": days * 86400})
    return [row[0] for row in rows]


async def lookup_papers(engine, accessions: Sequence[str]) -> Dict[str, int]:
    """Step 1. Returns the lookup's counts ({} when nothing was due)."""
    from remote import paper_store

    async with engine.begin() as conn:
        await paper_store.ensure_tables(conn)
        due = await _needs_paper_check(conn, accessions)
    if not due:
        return {}
    async with engine.begin() as conn:
        return await paper_store.enrich(conn, due)


def _descriptions_available() -> bool:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        log.info("[post-crawl] No GEMINI_API_KEY: dataset descriptions skipped.")
        return False
    for name in ("backend", "LLM_preprocessor"):
        path = next((root / name for root in _ROOTS if (root / name).is_dir()), None)
        if path is not None and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import llm_client  # noqa: F401
        from google import genai  # noqa: F401  (llm_client imports it only when called)
        from services import dataset_description  # noqa: F401
    except ImportError as exc:
        log.warning("[post-crawl] Dataset descriptions skipped: %s", exc)
        return False
    return True


async def describe(engine, accessions: Sequence[str]) -> Dict[str, int]:
    """Step 2. Returns the counts of dataset_description.run."""
    from remote import paper_store
    from services import dataset_description

    async with engine.begin() as conn:
        await paper_store.ensure_tables(conn)
    return dict(await dataset_description.run(engine, list(accessions)))


def _backend_services_available() -> bool:
    path = next((root / "backend" for root in _ROOTS if (root / "backend").is_dir()), None)
    if path is not None and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        from services import word_form_store  # noqa: F401
    except ImportError as exc:
        log.warning("[post-crawl] Word forms skipped: %s", exc)
        return False
    return True


async def refresh_word_forms(engine) -> Dict[str, int]:
    """Step 3. Returns the counts of word_form_store.refresh."""
    from services import word_form_store

    async with engine.begin() as conn:
        return await word_form_store.refresh(conn)


async def vacuum_file_table(engine) -> None:
    """Step 4. Vacuum and analyze bids_objects.

    VACUUM cannot run inside a transaction, hence the autocommit connection.
    PARALLEL 0: parallel workers need more shared memory than the postgres
    container's 64 MB /dev/shm ("could not resize shared memory segment").
    The role may carry a statement timeout (300 s on production); it is
    lifted for this statement and restored for the pooled connection.
    """
    async with engine.connect() as conn:
        conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
        await conn.execute(text("SET statement_timeout = 0"))
        try:
            await conn.execute(text("VACUUM (PARALLEL 0, ANALYZE) bids_objects"))
        finally:
            await conn.execute(text("RESET statement_timeout"))


async def run_post_crawl(accessions: Sequence[str]) -> None:
    """The steps for *accessions*; logs and swallows every error."""
    accessions = [a for a in accessions if a]
    if not accessions:
        return
    from db.db import engine

    # The search reads per-dataset summaries of bids_objects, not the file
    # table: a dataset whose files changed is found by its new data types
    # only after this (remote/dataset_facets.py). Always on.
    try:
        from remote import dataset_facets

        async with engine.begin() as conn:
            stats = await dataset_facets.refresh(conn, accessions)
        log.info("[post-crawl] Dataset summaries: %s", stats)
    except Exception as exc:
        log.error("[post-crawl] Dataset summaries failed: %s", exc, exc_info=True)

    # Participant diagnoses from a group column and its participants.json
    # Levels ("F" = "Frontotemporal Dementia Group"; remote/participant_groups.py).
    # Only datasets without any coded diagnosis. Reads OpenNeuro's public S3.
    if _flag("CRAWL_PARTICIPANT_GROUPS"):
        try:
            from remote import participant_groups

            async with engine.begin() as conn:
                reports = await participant_groups.run(conn, accessions, apply=True)
            log.info("[post-crawl] Participant groups decoded: %d datasets, %d rows",
                     len(reports), sum(r["written"] for r in reports))
        except Exception as exc:
            log.error("[post-crawl] Participant groups failed: %s", exc, exc_info=True)

    if _flag("CRAWL_LINKS"):
        try:
            from remote import dataset_links

            async with engine.begin() as conn:
                stats = await dataset_links.refresh_links(conn, accessions)
            log.info("[post-crawl] Derivatives and sites: %s", stats)
        except Exception as exc:
            log.error("[post-crawl] Derivatives and sites failed: %s", exc, exc_info=True)

    if _flag("CRAWL_PAPERS"):
        try:
            stats = await lookup_papers(engine, accessions)
            log.info("[post-crawl] Papers looked up: %s", stats or "none due")
        except Exception as exc:
            log.error("[post-crawl] Paper lookup failed: %s", exc, exc_info=True)

    # The institutions of the first and last author of each primary paper
    # (OpenAlex, with their ROR id) are sites too (remote/paper_affiliations.py).
    if _flag("CRAWL_PAPER_AFFILIATIONS"):
        try:
            from remote import paper_affiliations

            stats = await paper_affiliations.refresh(engine, accessions)
            log.info("[post-crawl] Paper affiliations: %s", stats)
        except Exception as exc:
            log.error("[post-crawl] Paper affiliations failed: %s", exc, exc_info=True)

    # Where the sites are: names not matched yet go to ROR once
    # (remote/site_places.py); questions that name a place read the result.
    if _flag("CRAWL_SITE_PLACES"):
        try:
            from remote import site_places

            async with engine.begin() as conn:
                stats = await site_places.match_new(conn)
            log.info("[post-crawl] Site places: %s", stats)
        except Exception as exc:
            log.error("[post-crawl] Site places failed: %s", exc, exc_info=True)

    if _flag("CRAWL_DESCRIPTIONS") and _descriptions_available():
        try:
            stats = await describe(engine, accessions)
            log.info("[post-crawl] Dataset descriptions: %s", stats)
        except Exception as exc:
            log.error("[post-crawl] Dataset descriptions failed: %s", exc, exc_info=True)

    if _flag("CRAWL_WORD_FORMS") and _backend_services_available():
        try:
            stats = await refresh_word_forms(engine)
            log.info("[post-crawl] Word forms: %s", stats)
        except Exception as exc:
            log.error("[post-crawl] Word forms failed: %s", exc, exc_info=True)

    if _flag("CRAWL_VACUUM"):
        try:
            await vacuum_file_table(engine)
            log.info("[post-crawl] File table vacuumed.")
        except Exception as exc:
            log.error("[post-crawl] Vacuum of the file table failed: %s", exc, exc_info=True)
