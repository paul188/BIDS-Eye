"""
services/indexing/readme_profile_runner.py
------------------------------------------
Ask the model for the README profile of each dataset that needs one, and
store it (services/indexing/readme_profile.py builds the prompt and reads
the answer; this module does the model calls and the database).

Writes bids_datasets.llm_profile / llm_conditions / llm_profile_meta and
bids_dataset_facts. A dataset is skipped when its stored profile was made
from the same input with the same prompt version, so a run can be stopped and
repeated at any time, and a changed README redoes only that dataset.

Two callers: the crawler after each cycle, for the datasets it indexed
(ingest/openneuro-crawler/post_crawl.py), and
ingest/backfill/extract_readme_profiles.py, for the whole catalogue or a sample to
review. Needs GEMINI_API_KEY; the text goes to Gemini only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, List, Optional, Sequence

from exceptions.expected import ExpectedError
from services.indexing import readme_profile as rp

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("README_PROFILE_MODEL", "gemini-3.8-flash")
DEFAULT_THINKING = os.getenv("README_PROFILE_THINKING") or None

_REQUIRED_COLUMNS = ("llm_profile", "llm_conditions", "llm_profile_meta")

_SELECT_SQL = """
    SELECT id, accession_id, name, readme_text, dx_status, study_domain,
           study_design, description_text, llm_profile_meta
    FROM bids_datasets
    {where}
    ORDER BY accession_id
"""

_UPDATE_SQL = """
    UPDATE bids_datasets
    SET llm_profile = :profile, llm_conditions = :conditions,
        llm_profile_meta = CAST(:meta AS jsonb)
    WHERE id = :id
"""
_DELETE_FACTS_SQL = "DELETE FROM bids_dataset_facts WHERE dataset_id = :id"
_INSERT_FACT_SQL = """
    INSERT INTO bids_dataset_facts (dataset_id, kind, term, role, code, score, evidence, verified)
    VALUES (:dataset_id, :kind, :term, :role, :code, :score, :evidence, :verified)
"""


class SchemaMissingError(ExpectedError):
    """The profile columns or bids_dataset_facts do not exist yet."""


class ResolverUnavailableError(ExpectedError):
    """The diagnosis resolver could not load, so no codes could be attached."""


class ProfileAnswerError(ExpectedError):
    """The model's answer could not be used, twice."""


async def missing_schema(conn: Any) -> List[str]:
    """The columns and tables the run writes that the database lacks."""
    from sqlalchemy import text

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


def code_lookup() -> rp.CodeLookup:
    """Diagnosis resolver from the search pipeline (backend/vocabulary/retriever.py).

    Raises ResolverUnavailableError when its files or modules are missing:
    importing the retriever reads backend/vocabulary/value_mappings.yaml, and it reads
    name_index.json next to itself.
    """
    import yaml

    try:
        import retriever as retriever_module

        name_index = Path(retriever_module.__file__).resolve().parent / "name_index.json"
        retriever = retriever_module.MetadataRetriever(str(name_index))
    except (ImportError, OSError, ValueError, yaml.YAMLError) as exc:
        raise ResolverUnavailableError(str(exc)) from exc

    def lookup(term: str):
        scored = retriever.retrieve_for_field_multi_scored("diagnosis", [term])
        return [pair for pairs in scored.values() for pair in pairs]

    return lookup


def _ask_model(source: rp.ProfileSource, model: str, thinking: Optional[str]) -> str:
    from llm_client import llm_generate

    return llm_generate(
        rp.build_prompt(source),
        profile="readme_profile",
        system=rp.SYSTEM_PROMPT,
        models=[model],
        json_schema=rp.RESPONSE_SCHEMA,
        thinking_level=thinking,
    )


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


def extract_safe(source: rp.ProfileSource, model: str, thinking: Optional[str]):
    """(source, profile, error) — runs in a worker thread. A failed model call
    or an unusable answer is returned as the error, counted and skipped; a bug
    (in build_prompt, say) is raised and stops the run, instead of reading
    as "model call failed" for every dataset."""
    from llm_client import LLMAllFailedError

    try:
        return source, _extract(source, model, thinking), None
    except (LLMAllFailedError, ProfileAnswerError) as exc:
        return source, None, exc


async def _write(conn: Any, dataset_id: Any, profile: rp.Profile, source: rp.ProfileSource,
                 model: str) -> None:
    from sqlalchemy import text

    meta = rp.meta_record(profile, source, model, datetime.now(timezone.utc).isoformat())
    async with conn.begin():
        await conn.execute(text(_UPDATE_SQL), {
            "id": dataset_id,
            "profile": profile.profile_text(),
            "conditions": rp.studied_conditions_text(profile),
            "meta": json.dumps(meta),
        })
        await conn.execute(text(_DELETE_FACTS_SQL), {"id": dataset_id})
        rows = rp.fact_rows(profile)
        if rows:
            await conn.execute(text(_INSERT_FACT_SQL),
                               [{**row, "dataset_id": dataset_id} for row in rows])


async def run(
    engine: Any,
    accessions: Optional[Sequence[str]] = None,
    *,
    model: str = DEFAULT_MODEL,
    thinking: Optional[str] = DEFAULT_THINKING,
    workers: int = 6,
    force: bool = False,
    dry_run: bool = False,
    limit: int = 0,
    codes: bool = True,
    out: Optional[IO[str]] = None,
) -> Counter:
    """Profile the datasets whose profile is missing or stale (default: every
    dataset). Commits after each dataset. Returns counts: done, failed,
    skipped (up to date), unverified (facts whose quote was not found), and
    role:<role>/<ok|unverified> for the conditions.

    Raises SchemaMissingError before any model call when the tables are not
    migrated, and ResolverUnavailableError when codes are wanted but the
    resolver cannot load: facts stored without codes count as done, so a
    broken resolver must stop the run instead of leaving every dataset without
    codes until a forced rerun.
    """
    from sqlalchemy import text

    stats: Counter = Counter()
    accessions = [a for a in (accessions or []) if a]
    where, params = "", {}
    if accessions:
        where, params = "WHERE accession_id = ANY(:acc)", {"acc": list(accessions)}
    async with engine.connect() as conn:
        missing = await missing_schema(conn)
        if missing:
            raise SchemaMissingError(
                f"missing {', '.join(missing)}: run `alembic upgrade head` in backend/ first")
        rows = (await conn.execute(text(_SELECT_SQL.format(where=where)), params)).mappings().all()
    ids = {(r["accession_id"] or str(r["id"])): r["id"] for r in rows}
    todo = rp.sources_needing_work([dict(r) for r in rows], force=force)
    stats["skipped"] = len(rows) - len(todo)
    if limit:
        todo = todo[:limit]
    log.info("README profiles: %d datasets, %d need a profile (model %s, thinking %s)",
             len(rows), len(todo), model, thinking or "default")
    if not todo:
        return stats

    lookup = code_lookup() if codes else None
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [loop.run_in_executor(pool, extract_safe, source, model, thinking)
                   for source in todo]
        async with engine.connect() as conn:
            for done in asyncio.as_completed(futures):
                source, profile, error = await done
                if error is not None:
                    stats["failed"] += 1
                    log.warning("README profile %s: model call failed: %s",
                                source.accession_id, str(error)[:300])
                    continue
                rp.attach_codes(profile, lookup)
                for fact in profile.facts:
                    if fact.kind == "condition":
                        stats[f"role:{fact.role}/{'ok' if fact.verified else 'unverified'}"] += 1
                stats["unverified"] += sum(not f.verified for f in profile.facts)
                if out is not None:
                    out.write(json.dumps({
                        "accession_id": source.accession_id,
                        "profile": profile.profile_text(),
                        "conditions_studied": profile.studied_conditions,
                        "facts": rp.fact_rows(profile),
                    }) + "\n")
                    out.flush()
                if not dry_run:
                    await _write(conn, ids[source.accession_id], profile, source, model)
                stats["done"] += 1
                if stats["done"] % 50 == 0:
                    log.info("README profiles: %d / %d done", stats["done"], len(todo))
    return stats
