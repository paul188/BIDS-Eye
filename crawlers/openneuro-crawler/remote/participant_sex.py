"""
remote/participant_sex.py
-------------------------
Participant sex as the vocabulary's standard codes.

participants.tsv values reach bids_participants.sex as typed: 'F', 'Female',
'FEMALE', 'woman', '2'. This module rewrites each to the standard code of its
entry in RAG/value_mappings.yaml (sex: male, female, other_diverse), so that
a search compares one value and no other code needs its own list of
spellings. A sex that is not known is stored as NULL, never as a code: the
search reads "sex IS NOT NULL" as "the dataset records it".

Rules:
  - a value is matched against the `codes` and the standard code of each sex
    entry, ignoring case and outer spaces;
  - a number is never read by itself: datasets use 1/2 and 0/1 both ways
    round. It counts only through the Levels text of the dataset's
    participants.json ("1": "male");
  - any other value the vocabulary does not list is tried through Levels the
    same way;
  - a value of the vocabulary's unknown entry ('U', '-', 'unknown') becomes
    NULL;
  - so does a value nothing names. On the live database (2026-11-23) these
    were ages and participant ids from shifted columns, 736 rows in 11
    datasets. They are reported, so that a real spelling can be added to
    the vocabulary.

A rewritten row keeps the typed value in extra._sex_raw, so the change can be
listed and undone:

    UPDATE bids_participants SET sex = extra ->> '_sex_raw',
           extra = (extra::jsonb - '_sex_raw')::json
    WHERE extra::jsonb ? '_sex_raw';

A crawl writes the typed value again; post_crawl.py runs this after each
cycle for the datasets it indexed.

    python -m remote.participant_sex                  # dry run, all datasets
    python -m remote.participant_sex --apply          # write
    python -m remote.participant_sex ds004504 --apply # some datasets
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from remote.participant_groups import ParticipantSidecarError, fetch_sidecar, level_texts

log = logging.getLogger(__name__)

RAW_KEY = "_sex_raw"
# The vocabulary's entry for a sex that is not known: its spellings are read,
# its code is never stored.
UNKNOWN_CODE = "unknown_missing_sex"
# The participants.tsv columns the crawler reads the sex from.
_SIDECAR_COLUMNS = ("sex", "gender")
_NUMBER = re.compile(r"[-+]?\d+(\.\d+)?")


class SexVocabularyError(Exception):
    """RAG/value_mappings.yaml was not found or has no sex entries."""


def _vocabulary_file() -> Path:
    here = Path(__file__).resolve()
    # The repository root, or /app in the crawler and backend images.
    for root in (*here.parents, Path("/app")):
        path = root / "RAG" / "value_mappings.yaml"
        if path.exists():
            return path
    raise SexVocabularyError("RAG/value_mappings.yaml not found")


@lru_cache(maxsize=1)
def sex_codes() -> Tuple[Dict[str, str], Tuple[str, ...]]:
    """(typed value in lower case → standard code, the codes that are stored).

    The table also reads the spellings of UNKNOWN_CODE; that code is not
    among the stored ones.
    """
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with _vocabulary_file().open(encoding="utf-8") as handle:
        entries = (yaml.load(handle, Loader=loader) or {}).get("sex") or {}
    table: Dict[str, str] = {}
    standard: List[str] = []
    for entry in entries.values():
        code = (entry or {}).get("standard_code")
        if not code:
            continue
        standard.append(code)
        for value in (code, *(entry.get("codes") or ())):
            key = str(value).strip().lower()
            if key and not _NUMBER.fullmatch(key):
                table.setdefault(key, code)
    if not standard:
        raise SexVocabularyError("value_mappings.yaml has no sex entries")
    return table, tuple(code for code in standard if code != UNKNOWN_CODE)


def standard_code(value: str, levels: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The standard code of a typed value (UNKNOWN_CODE for 'U', '-' and the
    like), or None when nothing names it.

    *levels*: value (lower case) → Levels text of the dataset's sex column.
    """
    table, _ = sex_codes()
    key = str(value).strip().lower()
    if key in table:
        return table[key]
    text = (levels or {}).get(key)
    return table.get(text.strip().lower()) if text else None


def sex_levels(sidecar: Optional[dict]) -> Dict[str, str]:
    """value (lower case) → Levels text of the sex column in participants.json."""
    for column in _SIDECAR_COLUMNS:
        levels = level_texts(sidecar, column)
        if levels:
            return levels
    return {}


# ── Database ──────────────────────────────────────────────────────────────────

_VALUES = """
SELECT d.id, d.accession_id, p.sex, COUNT(*)
FROM bids_participants p JOIN bids_datasets d ON d.id = p.dataset_id
WHERE p.sex IS NOT NULL AND BTRIM(p.sex) <> ''
  AND NOT (p.sex = ANY(CAST(:standard AS text[])))
"""
_WRITE = """
UPDATE bids_participants
SET sex = CAST(:code AS text),
    extra = (CASE WHEN json_typeof(extra) = 'object' THEN extra::jsonb ELSE '{}'::jsonb END
             || jsonb_build_object(CAST(:raw_key AS text), CAST(:raw AS text)))::json
WHERE dataset_id = :id AND sex = CAST(:raw AS text)
"""


async def run(conn, accessions: Optional[Sequence[str]] = None, apply: bool = False,
              fetch: Callable[[str], Optional[dict]] = fetch_sidecar) -> List[dict]:
    """Plan (and with *apply*, write) the standard code of every typed value.

    Runs in the caller's transaction. Returns one report per dataset with a
    value to rewrite:
    {"accession", "values": {typed: [stored, rows]}, "unknown": {typed: rows}, "written"}.
    "values" are the ones the vocabulary or participants.json names; stored
    is None for a sex that is not known. "unknown" are the values nothing
    names; they become NULL as well.
    """
    from sqlalchemy import text

    _, standard = sex_codes()
    sql = _VALUES + ("" if accessions is None else " AND d.accession_id = ANY(:accessions)")
    params: dict = {"standard": list(standard)}
    if accessions is not None:
        params["accessions"] = list(accessions)
    found = (await conn.execute(text(
        sql + " GROUP BY d.id, d.accession_id, p.sex ORDER BY d.accession_id, p.sex"), params)).all()

    by_dataset: Dict[Tuple[str, str], Dict[str, int]] = {}
    for dataset_id, accession, value, rows in found:
        by_dataset.setdefault((dataset_id, accession), {})[value] = rows

    reports = []
    for (dataset_id, accession), counts in by_dataset.items():
        codes = {value: standard_code(value) for value in counts}
        if any(code is None for code in codes.values()):
            try:
                levels = sex_levels(await asyncio.to_thread(fetch, accession))
            except ParticipantSidecarError as exc:
                # Only the values the vocabulary lists are written. The
                # others stay as typed, not NULL, so the next run asks again.
                log.warning("participant sex: %s read without participants.json (%s)", accession, exc)
                codes = {value: code for value, code in codes.items() if code is not None}
            else:
                codes = {value: code or standard_code(value, levels) for value, code in codes.items()}
        stored = {value: code if code in standard else None for value, code in codes.items()}
        written = 0
        if apply:
            for value, code in stored.items():
                result = await conn.execute(text(_WRITE), {
                    "code": code, "raw_key": RAW_KEY, "raw": value, "id": dataset_id})
                written += result.rowcount or 0
        unknown = {value: counts[value] for value, code in codes.items() if code is None}
        if unknown:
            log.warning("participant sex: %s has values nothing names, stored as NULL: %s",
                        accession, sorted(unknown)[:10])
        if not codes:
            continue
        reports.append({
            "accession": accession,
            "values": {value: [stored[value], counts[value]]
                       for value, code in codes.items() if code is not None},
            "unknown": unknown,
            "written": written,
        })
    return reports


if __name__ == "__main__":  # pragma: no cover
    async def _main() -> int:
        from db.db import engine

        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        apply = "--apply" in sys.argv
        async with (engine.begin() if apply else engine.connect()) as conn:
            if not apply:
                from sqlalchemy import text
                await conn.execute(text("SET TRANSACTION READ ONLY"))
            reports = await run(conn, args or None, apply=apply)
        rewrites: Dict[str, int] = {}
        unknown: Dict[str, int] = {}
        for report in reports:
            for value, (code, rows) in report["values"].items():
                line = f"{value!r} -> {code or 'NULL (not known)'}"
                rewrites[line] = rewrites.get(line, 0) + rows
            for value, rows in report["unknown"].items():
                unknown[value] = unknown.get(value, 0) + rows
        for line, rows in sorted(rewrites.items(), key=lambda item: -item[1]):
            print(f"{rows:7d}  {line}")
        for value, rows in sorted(unknown.items(), key=lambda item: -item[1]):
            print(f"{rows:7d}  {value!r} -> NULL (nothing names it)")
        if "--datasets" in sys.argv:
            for report in reports:
                print(json.dumps(report, ensure_ascii=False))
        print(f"{len(reports)} datasets, {sum(rewrites.values())} rows read from the vocabulary, "
              f"{sum(unknown.values())} rows nothing names "
              + ("written: " + str(sum(r["written"] for r in reports)) if apply else "(dry run)"))
        return 0

    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(_main()))
