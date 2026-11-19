#!/usr/bin/env python3
"""Read every dataset's README with an LLM once and store what it says.

For each dataset the model returns a short profile, the conditions the text
names with their role (patients, model, controls, excluded, measured,
mentioned), species, age groups, topics and paradigms, each condition and
paradigm with a verbatim quote. Quotes are checked against the input; see
backend/services/indexing/readme_profile.py.

Writes bids_datasets.llm_profile / llm_conditions / llm_profile_meta and
bids_dataset_facts. A dataset is skipped when its stored profile was made from
the same input with the same prompt version, so the script can be stopped and
rerun at any time, and a rerun after a README changed redoes only that one.

Usage (inside the backend container, which has the keys and the database):

    python scripts/extract_readme_profiles.py --dry-run --limit 20 --save-jsonl /out/sample.jsonl
    python scripts/extract_readme_profiles.py                     # all datasets that need it
    python scripts/extract_readme_profiles.py --accessions ds000171,ds004627 --force

Run crawlers/migrate_add_readme_profiles.sql first. Needs GEMINI_API_KEY.
Database connection: DATABASE_URL, or the METADATA_DB_* variables.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO / "backend", _REPO / "LLM_preprocessor", _REPO / "RAG"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import yaml  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from services.indexing import readme_profile as rp  # noqa: E402

log = logging.getLogger("extract_readme_profiles")

DEFAULT_MODEL = os.getenv("README_PROFILE_MODEL", "gemini-3.8-flash")

_REQUIRED_COLUMNS = ("llm_profile", "llm_conditions", "llm_profile_meta")

_SELECT_SQL = """
    SELECT id, accession_id, name, readme_text, dx_status, study_domain,
           study_design, description_text, llm_profile_meta
    FROM bids_datasets
    {where}
    ORDER BY accession_id
"""

_UPDATE_SQL = text("""
    UPDATE bids_datasets
    SET llm_profile = :profile, llm_conditions = :conditions,
        llm_profile_meta = CAST(:meta AS jsonb)
    WHERE id = :id
""")
_DELETE_FACTS_SQL = text("DELETE FROM bids_dataset_facts WHERE dataset_id = :id")
_INSERT_FACT_SQL = text("""
    INSERT INTO bids_dataset_facts (dataset_id, kind, term, role, code, score, evidence, verified)
    VALUES (:dataset_id, :kind, :term, :role, :code, :score, :evidence, :verified)
""")


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        if url.startswith("postgresql://"):
            url = "postgresql+asyncpg://" + url[len("postgresql://"):]
        return url
    try:
        return (
            "postgresql+asyncpg://"
            f"{os.environ['METADATA_DB_USER']}:{os.environ['METADATA_DB_PASSWORD']}"
            f"@{os.environ['METADATA_DB_HOST']}:{os.environ.get('METADATA_DB_PORT', '5432')}"
            f"/{os.environ['METADATA_DB_NAME']}"
        )
    except KeyError as exc:
        raise SystemExit(
            f"Database settings missing ({exc}). Set DATABASE_URL or the METADATA_DB_* variables."
        )


async def _check_schema(conn) -> List[str]:
    rows = await conn.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = 'bids_datasets'"
    ))
    present = {row[0] for row in rows}
    missing = [c for c in _REQUIRED_COLUMNS if c not in present]
    facts = await conn.execute(text("SELECT to_regclass('bids_dataset_facts') IS NOT NULL"))
    if not facts.scalar():
        missing.append("table bids_dataset_facts")
    return missing


# What loading the resolver raises when its files are missing or broken:
# importing the retriever reads RAG/value_mappings.yaml, and it reads
# name_index.json (FileNotFoundError is an OSError, JSONDecodeError a ValueError).
_RESOLVER_ERRORS = (ImportError, OSError, ValueError, yaml.YAMLError)


def _code_lookup():
    """Diagnosis resolver from the search pipeline. Raises one of
    _RESOLVER_ERRORS when it cannot load."""
    from retriever import MetadataRetriever
    retriever = MetadataRetriever(str(_REPO / "RAG" / "name_index.json"))

    def lookup(term: str):
        scored = retriever.retrieve_for_field_multi_scored("diagnosis", [term])
        return [pair for pairs in scored.values() for pair in pairs]

    return lookup


def _ask_model(source: rp.ProfileSource, model: str, thinking: Optional[str]) -> str:
    from llm_client import llm_generate

    return llm_generate(
        rp.build_prompt(source),
        system=rp.SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=4096,
        models=[model],
        json_schema=rp.RESPONSE_SCHEMA,
        thinking_level=thinking,
        call_timeout=120,
        budget_seconds=300,
        use_fallback=False,
    )


class ProfileAnswerError(Exception):
    """The model's answer could not be used, twice."""


def _extract(source: rp.ProfileSource, model: str, thinking: Optional[str]) -> rp.Profile:
    """Ask the model, retrying once when the answer is not usable JSON."""
    last_error: Optional[ValueError] = None
    for _attempt in range(2):
        raw = _ask_model(source, model, thinking)
        try:
            return rp.parse_response(raw, source)
        except ValueError as exc:
            last_error = exc
    raise ProfileAnswerError(f"unusable answer twice: {last_error}") from last_error


def _extract_safe(source: rp.ProfileSource, model: str, thinking: Optional[str]):
    """(source, profile, error) — runs in a worker thread. A failed model call
    or an unusable answer is returned as the error, counted and skipped; a bug
    (in build_prompt, say) is raised and stops the run, instead of reading
    as "model call failed" for every dataset."""
    from llm_client import LLMAllFailedError

    try:
        return source, _extract(source, model, thinking), None
    except (LLMAllFailedError, ProfileAnswerError) as exc:
        return source, None, exc


async def _write(conn, dataset_id: Any, profile: rp.Profile, source: rp.ProfileSource,
                 model: str) -> None:
    meta = rp.meta_record(profile, source, model, datetime.now(timezone.utc).isoformat())
    async with conn.begin():
        await conn.execute(_UPDATE_SQL, {
            "id": dataset_id,
            "profile": profile.profile_text(),
            "conditions": rp.studied_conditions_text(profile),
            "meta": json.dumps(meta),
        })
        await conn.execute(_DELETE_FACTS_SQL, {"id": dataset_id})
        rows = rp.fact_rows(profile)
        if rows:
            await conn.execute(_INSERT_FACT_SQL, [{**row, "dataset_id": dataset_id} for row in rows])


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="ask the model, write nothing")
    parser.add_argument("--limit", type=int, default=0, help="at most this many datasets")
    parser.add_argument("--accessions", default="", help="comma-separated accession ids")
    parser.add_argument("--force", action="store_true", help="redo datasets that are up to date")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking", default=os.getenv("README_PROFILE_THINKING") or None,
                        help="Gemini 3+ thinking level: minimal, low, medium, high")
    parser.add_argument("--workers", type=int, default=6, help="parallel model calls")
    parser.add_argument("--no-codes", action="store_true", help="skip diagnosis code mapping")
    parser.add_argument("--save-jsonl", type=Path, help="append every result here for review")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        log.error("GEMINI_API_KEY is not set")
        return 2

    engine = create_async_engine(_database_url())
    try:
        async with engine.connect() as conn:
            missing = await _check_schema(conn)
            if missing:
                log.error("Missing %s — run crawlers/migrate_add_readme_profiles.sql first",
                          ", ".join(missing))
                return 2
            where, params = "", {}
            accessions = [a.strip() for a in args.accessions.split(",") if a.strip()]
            if accessions:
                where, params = "WHERE accession_id = ANY(:acc)", {"acc": accessions}
            rows = (await conn.execute(text(_SELECT_SQL.format(where=where)), params)).mappings().all()
        ids = {(r["accession_id"] or str(r["id"])): r["id"] for r in rows}
        todo = rp.sources_needing_work([dict(r) for r in rows], force=args.force)
        if args.limit:
            todo = todo[: args.limit]
        log.info("%d datasets, %d need a profile (model %s, thinking %s)",
                 len(rows), len(todo), args.model, args.thinking or "default")
        if not todo:
            return 0

        # Facts stored without codes count as done (sources_needing_work
        # compares only the input), so a broken resolver must stop the run
        # instead of leaving every dataset without codes until --force.
        try:
            lookup = None if args.no_codes else _code_lookup()
        except _RESOLVER_ERRORS as exc:
            log.error("Diagnosis resolver unavailable (%s). Fix it, or run with --no-codes "
                      "to store the facts without codes.", exc)
            return 2
        stats: Counter = Counter()
        roles: Counter = Counter()
        loop = asyncio.get_running_loop()
        started = time.monotonic()
        out = args.save_jsonl.open("a") if args.save_jsonl else None

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [
                loop.run_in_executor(pool, _extract_safe, source, args.model, args.thinking)
                for source in todo
            ]
            async with engine.connect() as conn:
                for done in asyncio.as_completed(futures):
                    source, profile, error = await done
                    if error is not None:
                        stats["failed"] += 1
                        log.warning("%s: model call failed: %s", source.accession_id, str(error)[:300])
                        continue
                    rp.attach_codes(profile, lookup)
                    for fact in profile.facts:
                        if fact.kind == "condition":
                            roles[(fact.role, fact.verified)] += 1
                    stats["unverified"] += sum(not f.verified for f in profile.facts)
                    if out:
                        out.write(json.dumps({
                            "accession_id": source.accession_id,
                            "profile": profile.profile_text(),
                            "conditions_studied": profile.studied_conditions,
                            "facts": rp.fact_rows(profile),
                        }) + "\n")
                        out.flush()
                    if not args.dry_run:
                        await _write(conn, ids[source.accession_id], profile, source, args.model)
                    stats["done"] += 1
                    if stats["done"] % 50 == 0:
                        log.info("%d / %d done (%.0f s)", stats["done"], len(todo),
                                 time.monotonic() - started)
        if out:
            out.close()
    finally:
        await engine.dispose()

    log.info("Done: %d profiled, %d failed, %d unverified facts, %.0f s%s",
             stats["done"], stats["failed"], stats["unverified"], time.monotonic() - started,
             " (dry run, nothing written)" if args.dry_run else "")
    log.info("Condition roles (role, quote verified): %s",
             ", ".join(f"{r}/{'ok' if v else 'unverified'}={n}" for (r, v), n in sorted(roles.items())))
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
