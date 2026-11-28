"""
remote/participant_codes.py
---------------------------
Participant sex and handedness as the vocabulary's standard codes.

participants.tsv values reach bids_participants as typed: 'F', 'Female',
'FEMALE', 'woman', '2'; 'Ambidextrous', 'R'. This module rewrites each to the
standard code of its entry in backend/vocabulary/value_mappings.yaml (sections `sex` and
`handedness`), so that a search compares one value and no other code needs
its own list of spellings.

Rules, the same for both columns (FIELDS says where they differ):
  - a value is matched against the `codes` and the standard code of each
    entry of the section, ignoring case and outer spaces;
  - a value the vocabulary does not list is tried through the Levels text of
    the dataset's participants.json ("1": "male");
  - a value nothing names becomes NULL. On the live database (2026-11-23)
    these were ages and participant ids in the sex column, from shifted
    columns: 736 rows in 11 datasets. They are reported, so that a real
    spelling can be added to the vocabulary.

  - a value that says "not known" ('U', '-', 'unknown': the vocabulary's
    unknown_missing_sex and unknown_handedness entries) is stored as NULL,
    never as a code: the search reads "sex IS NOT NULL" as "the dataset
    records it".

Sex only:
  - a number is never read by itself, although the vocabulary lists some:
    datasets use 1/2 and 0/1 both ways round. It counts only through Levels.

A rewritten row keeps the typed value in extra (_sex_raw, _handedness_raw),
so the change can be listed and undone:

    UPDATE bids_participants SET sex = extra ->> '_sex_raw',
           extra = (extra::jsonb - '_sex_raw')::json
    WHERE extra::jsonb ? '_sex_raw';

A crawl writes the typed value again; post_crawl.py runs this after each
cycle for the datasets it indexed.

    python -m remote.participant_codes                  # dry run, all datasets
    python -m remote.participant_codes --apply          # write
    python -m remote.participant_codes ds004504 --apply # some datasets
    python -m remote.participant_codes --datasets       # also one line per dataset
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from remote.participant_groups import ParticipantSidecarError, fetch_sidecar, level_texts

log = logging.getLogger(__name__)

_NUMBER = re.compile(r"[-+]?\d+(\.\d+)?")


@dataclass(frozen=True)
class Field:
    column: str                          # of bids_participants; also the vocabulary's section
    sidecar_columns: Tuple[str, ...]     # the participants.tsv columns the crawler reads it from
    unknown_code: Optional[str] = None   # the entry whose values are stored as NULL
    numbers: bool = True                 # read a bare number the vocabulary lists

    @property
    def raw_key(self) -> str:
        return f"_{self.column}_raw"


SEX = Field("sex", ("sex", "gender"), unknown_code="unknown_missing_sex", numbers=False)
# The vocabulary lists laterality quotients as codes ('0.83' right-handed,
# '-0.5' left-handed); they are read as listed.
HANDEDNESS = Field("handedness", ("handedness", "hand", "handed"), unknown_code="unknown_handedness")
FIELDS = (SEX, HANDEDNESS)


class VocabularyError(Exception):
    """backend/vocabulary/value_mappings.yaml was not found or lacks a section."""


def _vocabulary_file() -> Path:
    here = Path(__file__).resolve()
    # The repository root, or /app in the crawler and backend images.
    for root in (*here.parents, Path("/app")):
        path = root / "backend" / "vocabulary" / "value_mappings.yaml"
        if path.exists():
            return path
    raise VocabularyError("backend/vocabulary/value_mappings.yaml not found")


@lru_cache(maxsize=1)
def _vocabulary() -> dict:
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with _vocabulary_file().open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=loader) or {}


@lru_cache(maxsize=None)
def codes(field: Field) -> Tuple[Dict[str, str], Tuple[str, ...]]:
    """(typed value in lower case → standard code, the codes that are stored).

    The table also reads the spellings of the field's unknown code; that code
    is not among the stored ones.
    """
    entries = _vocabulary().get(field.column) or {}
    table: Dict[str, str] = {}
    standard: List[str] = []
    for entry in entries.values():
        code = (entry or {}).get("standard_code")
        if not code:
            continue
        standard.append(code)
        for value in (code, *(entry.get("codes") or ())):
            key = str(value).strip().lower()
            if key and (field.numbers or not _NUMBER.fullmatch(key)):
                table.setdefault(key, code)
    if not standard:
        raise VocabularyError(f"value_mappings.yaml has no {field.column} entries")
    return table, tuple(code for code in standard if code != field.unknown_code)


def standard_code(field: Field, value: str, levels: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The standard code of a typed value (the field's unknown code for 'U',
    '-' and the like), or None when nothing names it.

    *levels*: value (lower case) → Levels text of the field's column.
    """
    table, _ = codes(field)
    key = str(value).strip().lower()
    if key in table:
        return table[key]
    text = (levels or {}).get(key)
    return table.get(text.strip().lower()) if text else None


def field_levels(field: Field, sidecar: Optional[dict]) -> Dict[str, str]:
    """value (lower case) → Levels text of the field's column in participants.json."""
    for column in field.sidecar_columns:
        levels = level_texts(sidecar, column)
        if levels:
            return levels
    return {}


# ── Database ──────────────────────────────────────────────────────────────────
# {column} is one of the FIELDS' column names, never a value from outside.

_VALUES = """
SELECT d.id, d.accession_id, p.{column}, COUNT(*)
FROM bids_participants p JOIN bids_datasets d ON d.id = p.dataset_id
WHERE p.{column} IS NOT NULL AND BTRIM(p.{column}) <> ''
  AND NOT (p.{column} = ANY(CAST(:standard AS text[])))
"""
_WRITE = """
UPDATE bids_participants
SET {column} = CAST(:code AS text),
    extra = (CASE WHEN json_typeof(extra) = 'object' THEN extra::jsonb ELSE '{{}}'::jsonb END
             || jsonb_build_object(CAST(:raw_key AS text), CAST(:raw AS text)))::json
WHERE dataset_id = :id AND {column} = CAST(:raw AS text)
"""


async def run(conn, accessions: Optional[Sequence[str]] = None, apply: bool = False,
              fetch: Callable[[str], Optional[dict]] = fetch_sidecar,
              fields: Sequence[Field] = FIELDS) -> List[dict]:
    """Plan (and with *apply*, write) the standard code of every typed value.

    Runs in the caller's transaction. Returns one report per field and
    dataset with a value to rewrite:
    {"field", "accession", "values": {typed: [stored, rows]}, "unknown": {typed: rows}, "written"}.
    "values" are the ones the vocabulary or participants.json names; stored
    is None for a value of the field's unknown code. "unknown" are the values
    nothing names; they become NULL as well.
    """
    from sqlalchemy import text

    sidecars: Dict[str, Optional[dict]] = {}

    async def sidecar_of(accession: str) -> Optional[dict]:
        if accession not in sidecars:  # one request per dataset, whatever the number of fields
            sidecars[accession] = await asyncio.to_thread(fetch, accession)
        return sidecars[accession]

    reports = []
    for field in fields:
        _, standard = codes(field)
        sql = _VALUES.format(column=field.column) + (
            "" if accessions is None else " AND d.accession_id = ANY(:accessions)")
        params: dict = {"standard": list(standard)}
        if accessions is not None:
            params["accessions"] = list(accessions)
        found = (await conn.execute(text(
            sql + f" GROUP BY d.id, d.accession_id, p.{field.column} "
                  f"ORDER BY d.accession_id, p.{field.column}"), params)).all()

        by_dataset: Dict[Tuple[str, str], Dict[str, int]] = {}
        for dataset_id, accession, value, rows in found:
            by_dataset.setdefault((dataset_id, accession), {})[value] = rows

        for (dataset_id, accession), counts in by_dataset.items():
            read = {value: standard_code(field, value) for value in counts}
            if any(code is None for code in read.values()):
                try:
                    levels = field_levels(field, await sidecar_of(accession))
                except ParticipantSidecarError as exc:
                    # Only the values the vocabulary lists are written. The
                    # others stay as typed, not NULL, so the next run asks again.
                    log.warning("participant %s: %s read without participants.json (%s)",
                                field.column, accession, exc)
                    read = {value: code for value, code in read.items() if code is not None}
                else:
                    read = {value: code or standard_code(field, value, levels)
                            for value, code in read.items()}
            if not read:
                continue
            stored = {value: code if code in standard else None for value, code in read.items()}
            written = 0
            if apply:
                for value, code in stored.items():
                    result = await conn.execute(text(_WRITE.format(column=field.column)), {
                        "code": code, "raw_key": field.raw_key, "raw": value, "id": dataset_id})
                    written += result.rowcount or 0
            unknown = {value: counts[value] for value, code in read.items() if code is None}
            if unknown:
                log.warning("participant %s: %s has values nothing names, stored as NULL: %s",
                            field.column, accession, sorted(unknown)[:10])
            reports.append({
                "field": field.column,
                "accession": accession,
                "values": {value: [stored[value], counts[value]]
                           for value, code in read.items() if code is not None},
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
                line = f"{report['field']}: {value!r} -> {code or 'NULL (not known)'}"
                rewrites[line] = rewrites.get(line, 0) + rows
            for value, rows in report["unknown"].items():
                line = f"{report['field']}: {value!r} -> NULL (nothing names it)"
                unknown[line] = unknown.get(line, 0) + rows
        for line, rows in sorted(rewrites.items(), key=lambda item: -item[1]):
            print(f"{rows:7d}  {line}")
        for line, rows in sorted(unknown.items(), key=lambda item: -item[1]):
            print(f"{rows:7d}  {line}")
        if "--datasets" in sys.argv:
            for report in reports:
                print(json.dumps(report, ensure_ascii=False))
        print(f"{len({r['accession'] for r in reports})} datasets, "
              f"{sum(rewrites.values())} rows read from the vocabulary, "
              f"{sum(unknown.values())} rows nothing names "
              + ("written: " + str(sum(r["written"] for r in reports)) if apply else "(dry run)"))
        return 0

    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(_main()))
