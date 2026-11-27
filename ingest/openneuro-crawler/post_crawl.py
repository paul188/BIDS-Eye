"""
post_crawl.py
-------------
Steps that run after a crawl cycle, for the datasets the cycle indexed:

  -. summaries  per-dataset summaries of the file table that the search
                reads (remote/dataset_facets.refresh). No network.
  -. groups     participant diagnoses from a group column and participants.json
                (remote/participant_groups.run). OpenNeuro's public S3 only.
  -. codes      participant sex and handedness as typed ('F', 'Female',
                'Ambidextrous') rewritten to the vocabulary's standard codes
                (remote/participant_codes.run).
                participants.json from OpenNeuro's public S3, only for values
                the vocabulary does not list.
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
  -. profiles  what each README says: a short profile, the conditions with
                their role (patients, controls, excluded, ...) and the
                paradigms, each with a checked quote
                (backend/services/indexing/readme_profile_runner.py, the same
                code as ingest/backfill/extract_readme_profiles.py). Only datasets
                whose README or study fields changed. Needs GEMINI_API_KEY;
                the text goes to Gemini only.
  2. descriptions  how each primary paper relates to its dataset, and a
                generated summary where the README is thin
                (backend/services/indexing/dataset_description.py, the same code as
                ingest/backfill/describe_datasets.py). Only what is missing or stale.
                Needs GEMINI_API_KEY; the text goes to Gemini only.
  3. word forms  rebuild bids_word_forms, the catalogue's word forms per stem
                (backend/services/fulltext/word_form_counts.py). No LLM, no network;
                the whole catalogue, not only the indexed datasets.
  4. vacuum     VACUUM (ANALYZE) of the file table, bids_objects. A search
                counts subjects from an index alone
                (backend/migrations/README.md); for the rows a
                crawl wrote that only works again after a vacuum, and
                autovacuum starts only once a fifth of the table changed.

Environment:
    CRAWL_PARTICIPANT_GROUPS "false" skips decoding participant groups (default on)
    CRAWL_PARTICIPANT_CODES "false" skips rewriting participant sex and handedness to standard codes (default on)
    CRAWL_LINKS             "false" skips step 0 (default on)
    CRAWL_SITE_PLACES       "false" skips matching site names to places (default on)
    CRAWL_PAPER_AFFILIATIONS "false" skips the lead authors' institutions as sites (default on)
    CRAWL_PAPERS            "false" skips step 1 (default on)
    CRAWL_README_PROFILES   "false" skips the README profiles (default on)
    CRAWL_DESCRIPTIONS      "false" skips step 2 (default on)
    CRAWL_WORD_FORMS        "false" skips step 3 (default on)
    CRAWL_VACUUM            "false" skips step 4 (default on)
    PAPER_RECHECK_DAYS      how old a paper lookup may get (default 30)
    DATASET_DESCRIPTION_MODEL  Gemini model for step 2 (default gemini-2.5-flash)
    README_PROFILE_MODEL    Gemini model for the README profiles

The same steps over datasets that are already indexed, without crawling
them (from this folder, with the crawler's environment):

    python post_crawl.py --all                          # every dataset
    python post_crawl.py --accessions ds000171,ds004627
    python post_crawl.py --all --only profiles,descriptions

--only runs the named steps and ignores the CRAWL_* switches; the names are
those in STEPS. Each step skips what is already up to date.

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
from typing import Collection, Dict, List, Optional, Sequence

from sqlalchemy import text

log = logging.getLogger(__name__)

# backend/ and backend/llm/ sit in the repository root, or next to the
# crawler in its Docker image (/app/crawler, /app/backend, /app/backend/llm).
_HERE = Path(__file__).resolve()
_ROOTS = (_HERE.parents[2], _HERE.parents[1])


# The steps in the order they run, by the names --only takes.
STEPS = ("summaries", "groups", "codes", "links", "papers", "affiliations", "places",
         "profiles", "descriptions", "word_forms", "vacuum")


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
        due = await _needs_paper_check(conn, accessions)
    if not due:
        return {}
    async with engine.begin() as conn:
        return await paper_store.enrich(conn, due)


def _add_to_path(*names: str) -> None:
    for name in names:
        path = next((root / name for root in _ROOTS if (root / name).is_dir()), None)
        if path is not None and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _descriptions_available() -> bool:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        log.info("[post-crawl] No GEMINI_API_KEY: dataset descriptions skipped.")
        return False
    _add_to_path("backend", "backend/llm")
    try:
        import llm_client  # noqa: F401
        from google import genai  # noqa: F401  (llm_client imports it only when called)
        from services.indexing import dataset_description  # noqa: F401
    except ImportError as exc:
        log.warning("[post-crawl] Dataset descriptions skipped: %s", exc)
        return False
    return True


def _profiles_available() -> bool:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        log.info("[post-crawl] No GEMINI_API_KEY: README profiles skipped.")
        return False
    # backend/vocabulary/ holds the diagnosis resolver that codes the conditions.
    _add_to_path("backend", "backend/llm", "backend/vocabulary")
    try:
        import llm_client  # noqa: F401
        import retriever  # noqa: F401
        from google import genai  # noqa: F401  (llm_client imports it only when called)
        from services.indexing import readme_profile_runner  # noqa: F401
    except ImportError as exc:
        log.warning("[post-crawl] README profiles skipped: %s", exc)
        return False
    return True


async def profile_readmes(engine, accessions: Sequence[str]) -> Dict[str, int]:
    """The counts of readme_profile_runner.run."""
    from services.indexing import readme_profile_runner

    return dict(await readme_profile_runner.run(engine, list(accessions)))


async def describe(engine, accessions: Sequence[str]) -> Dict[str, int]:
    """Step 2. Returns the counts of dataset_description.run."""
    from services.indexing import dataset_description

    return dict(await dataset_description.run(engine, list(accessions)))


def _backend_services_available() -> bool:
    _add_to_path("backend")
    try:
        from services.fulltext import word_form_counts  # noqa: F401
    except ImportError as exc:
        log.warning("[post-crawl] Word forms skipped: %s", exc)
        return False
    return True


async def refresh_word_forms(engine) -> Dict[str, int]:
    """Step 3. Returns the counts of word_form_counts.refresh."""
    from services.fulltext import word_form_counts

    async with engine.begin() as conn:
        return await word_form_counts.refresh(conn)


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


async def run_post_crawl(accessions: Sequence[str],
                         only: Optional[Collection[str]] = None) -> None:
    """The steps for *accessions*; logs and swallows every error.

    *only* runs just the named steps (STEPS) and ignores the CRAWL_* switches.
    """
    accessions = [a for a in accessions if a]
    if not accessions:
        return
    unknown = set(only or ()) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown post-crawl steps: {', '.join(sorted(unknown))}")
    from db.db import engine

    def wanted(step: str, switch: Optional[str] = None) -> bool:
        if only is not None:
            return step in only
        return switch is None or _flag(switch)

    # The search reads per-dataset summaries of bids_objects, not the file
    # table: a dataset whose files changed is found by its new data types
    # only after this (remote/dataset_facets.py). Always on in a crawl.
    if wanted("summaries"):
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
    if wanted("groups", "CRAWL_PARTICIPANT_GROUPS"):
        try:
            from remote import participant_groups

            async with engine.begin() as conn:
                reports = await participant_groups.run(conn, accessions, apply=True)
            log.info("[post-crawl] Participant groups decoded: %d datasets, %d rows",
                     len(reports), sum(r["written"] for r in reports))
        except Exception as exc:
            log.error("[post-crawl] Participant groups failed: %s", exc, exc_info=True)

    # Participant sex and handedness as typed ('F', 'Female', 'Ambidextrous')
    # become the vocabulary's standard codes (remote/participant_codes.py). A
    # crawl writes the typed value again, so this runs after every cycle.
    if wanted("codes", "CRAWL_PARTICIPANT_CODES"):
        try:
            from remote import participant_codes

            async with engine.begin() as conn:
                reports = await participant_codes.run(conn, accessions, apply=True)
            log.info("[post-crawl] Participant codes: %d rows rewritten in %d datasets, "
                     "%d of them values nothing names",
                     sum(r["written"] for r in reports), len({r["accession"] for r in reports}),
                     sum(sum(r["unknown"].values()) for r in reports))
        except Exception as exc:
            log.error("[post-crawl] Participant codes failed: %s", exc, exc_info=True)

    if wanted("links", "CRAWL_LINKS"):
        try:
            from remote import dataset_links

            async with engine.begin() as conn:
                stats = await dataset_links.refresh_links(conn, accessions)
            log.info("[post-crawl] Derivatives and sites: %s", stats)
        except Exception as exc:
            log.error("[post-crawl] Derivatives and sites failed: %s", exc, exc_info=True)

    if wanted("papers", "CRAWL_PAPERS"):
        try:
            stats = await lookup_papers(engine, accessions)
            log.info("[post-crawl] Papers looked up: %s", stats or "none due")
        except Exception as exc:
            log.error("[post-crawl] Paper lookup failed: %s", exc, exc_info=True)

    # The institutions of the first and last author of each primary paper
    # (OpenAlex, with their ROR id) are sites too (remote/paper_affiliations.py).
    if wanted("affiliations", "CRAWL_PAPER_AFFILIATIONS"):
        try:
            from remote import paper_affiliations

            stats = await paper_affiliations.refresh(engine, accessions)
            log.info("[post-crawl] Paper affiliations: %s", stats)
        except Exception as exc:
            log.error("[post-crawl] Paper affiliations failed: %s", exc, exc_info=True)

    # Where the sites are: names not matched yet go to ROR once
    # (remote/site_places.py); questions that name a place read the result.
    if wanted("places", "CRAWL_SITE_PLACES"):
        try:
            from remote import site_places

            async with engine.begin() as conn:
                stats = await site_places.match_new(conn)
            log.info("[post-crawl] Site places: %s", stats)
        except Exception as exc:
            log.error("[post-crawl] Site places failed: %s", exc, exc_info=True)

    # What each README says (services/indexing/readme_profile_runner.py): the
    # search indexes the profile and the studied conditions, and the
    # re-ranker reads the profile.
    if wanted("profiles", "CRAWL_README_PROFILES") and _profiles_available():
        try:
            stats = await profile_readmes(engine, accessions)
            log.info("[post-crawl] README profiles: %s", stats)
        except Exception as exc:
            log.error("[post-crawl] README profiles failed: %s", exc, exc_info=True)

    if wanted("descriptions", "CRAWL_DESCRIPTIONS") and _descriptions_available():
        try:
            stats = await describe(engine, accessions)
            log.info("[post-crawl] Dataset descriptions: %s", stats)
        except Exception as exc:
            log.error("[post-crawl] Dataset descriptions failed: %s", exc, exc_info=True)

    if wanted("word_forms", "CRAWL_WORD_FORMS") and _backend_services_available():
        try:
            stats = await refresh_word_forms(engine)
            log.info("[post-crawl] Word forms: %s", stats)
        except Exception as exc:
            log.error("[post-crawl] Word forms failed: %s", exc, exc_info=True)

    if wanted("vacuum", "CRAWL_VACUUM"):
        try:
            await vacuum_file_table(engine)
            log.info("[post-crawl] File table vacuumed.")
        except Exception as exc:
            log.error("[post-crawl] Vacuum of the file table failed: %s", exc, exc_info=True)


async def _indexed_accessions(engine) -> List[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(text(
            "SELECT DISTINCT accession_id FROM bids_datasets "
            "WHERE accession_id IS NOT NULL ORDER BY 1"))
        return [row[0] for row in rows]


async def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the post-crawl steps over datasets that are already indexed.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="every indexed dataset")
    target.add_argument("--accessions", help="comma-separated accession ids")
    parser.add_argument("--only", help=f"comma-separated steps of: {', '.join(STEPS)}")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "google_genai", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    only = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    unknown = set(only or ()) - set(STEPS)
    if unknown:
        parser.error(f"unknown steps: {', '.join(sorted(unknown))} (steps: {', '.join(STEPS)})")
    from db.db import engine

    if args.all:
        accessions = await _indexed_accessions(engine)
    else:
        accessions = [a.strip() for a in args.accessions.split(",") if a.strip()]
    log.info("[post-crawl] %d datasets, steps: %s", len(accessions), ", ".join(only or STEPS))
    try:
        await run_post_crawl(accessions, only=only)
    finally:
        await engine.dispose()
    return 0


if __name__ == "__main__":
    import asyncio

    sys.exit(asyncio.run(_main()))
