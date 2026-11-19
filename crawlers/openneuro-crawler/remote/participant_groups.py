"""
remote/participant_groups.py
----------------------------
Participant diagnoses from a dataset's group column and its participants.json.

Many datasets record each participant's group in participants.tsv as a short
code and explain it in participants.json:

    participants.tsv   Group = F
    participants.json  "Group": {"Levels": {"F": "Frontotemporal Dementia Group"}}

The crawler keeps such a column in bids_participants.extra and leaves
diagnosis empty (ds004504, ds006036: 23 FTD participants each, found only by
their titles). This module reads the group column, turns each value into its
Levels text, and writes the diagnosis code when that text names a diagnosis
in the vocabulary (RAG/value_mappings.yaml) exactly.

Only datasets whose participants have no diagnosis at all are touched, and
only rows whose diagnosis is empty are written. A written row gets
extra._diagnosis_from = "<column>", so the change can be listed and undone:

    UPDATE bids_participants SET diagnosis = NULL,
           extra = (extra::jsonb - '_diagnosis_from')::json
    WHERE extra::jsonb ? '_diagnosis_from';

Rules (checked on the live database, 2026-11-17; fuzzy matching turned "1"
into neurofibromatosis type 1 and "placebo" into a control code):
  - the column name says group / diagnosis / condition / type …, and it has
    2 to 12 distinct values;
  - a value counts only when its text matches a vocabulary synonym exactly,
    after "participants with", "group" and the like are removed (also tried
    without course words: "First Episode Psychosis" → psychosis);
  - yes/no columns and negated texts ("No ADHD diagnosis") are not decoded;
  - codes outside diagnoses (age groups, lifestyle, metadata) do not count,
    and a column is used only when at least one value is a diagnosis; of
    several such columns the one coding the most participants wins.

    python -m remote.participant_groups                  # dry run, all datasets
    python -m remote.participant_groups --apply          # write
    python -m remote.participant_groups ds004504 --apply # some datasets
"""

from __future__ import annotations

import asyncio
import http.client
import json
import logging
import re
import sys
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

SOURCE_KEY = "_diagnosis_from"
SIDECAR_URL = "https://s3.amazonaws.com/openneuro.org/{accession}/participants.json"

_GROUP_KEY = re.compile(
    r"group|grp|(^|_)dx($|_)|diag|condition|cohort|population|patient|participant_?type|"
    r"subject_?type|^type$|status|clinical|disease|pathology|category|^class$|^arm$",
    re.IGNORECASE)
# Columns about something else that only share a word with a group column.
_NOT_GROUP_KEY = re.compile(
    r"age|sex|gender|hand|edu|hormone|mood|route|duration|year|session|order|game|"
    r"date|time|score|inclusion|exclu|form|panel|scores|caffeine|drug|medic|"
    r"other|comorbid|secondary|history",  # a second diagnosis, not the group (ds007182)
    re.IGNORECASE)
_MIN_VALUES, _MAX_VALUES = 2, 12

_YES_NO = re.compile(r"^(yes|no|y|n|true|false|0|1|present|absent|not present|none|"
                     r"n/?a|-|unknown|other|missing)$", re.IGNORECASE)
_NEGATION = re.compile(r"\b(not|no|without|never|non)\b", re.IGNORECASE)
_WITH_PHRASE = re.compile(
    r"\b(participants?|patients?|subjects?|people|individuals?|persons?|children|adults?)"
    r"\s+(with|who\s+have|having|diagnosed\s+with)\b", re.IGNORECASE)
_SCAFFOLD_WORDS = frozenset({
    "group", "groups", "grp", "participant", "participants", "patient", "patients",
    "subject", "subjects", "cohort", "arm", "volunteer", "volunteers", "individual",
    "individuals", "people", "sample", "comparator", "comparators",
})
# Control labels that may be written; other codes under healthy_control are
# age groups or task variants ("young adult", "experimental").
_CONTROL_CODES = frozenset({"healthy_control", "healthy_volunteer", "neurotypical",
                            "typically_developing"})
_NOT_DIAGNOSIS = frozenset({"healthy_control", "behavioral_and_lifestyle",
                            "exclusion_and_metadata"})


# ── Vocabulary ────────────────────────────────────────────────────────────────

def _add_rag_to_path() -> None:
    here = Path(__file__).resolve()
    for root in (here.parents[3], here.parents[2], Path("/app")):
        rag = root / "RAG"
        if (rag / "value_mappings.yaml").exists():
            if str(rag) not in sys.path:
                sys.path.insert(0, str(rag))
            return


@lru_cache(maxsize=1)
def _vocabulary() -> Tuple[Dict[str, str], Dict[str, Tuple[str, ...]], frozenset]:
    """(synonym → code, code → broader groups, course words) for diagnoses."""
    _add_rag_to_path()
    import yaml
    from yaml_to_llamaindex import leaf_db
    from text_terms import DIAGNOSIS_COURSE_WORDS

    rag = Path(sys.modules["yaml_to_llamaindex"].__file__).resolve().parent
    entries = (yaml.safe_load((rag / "value_mappings.yaml").read_text()) or {}).get("diagnosis", {})
    broader = {code: tuple((entry or {}).get("broader") or ()) for code, entry in entries.items()}
    return dict(leaf_db.get("diagnosis", {})), broader, frozenset(DIAGNOSIS_COURSE_WORDS)


def code_kind(code: str, broader: Optional[Dict[str, Tuple[str, ...]]] = None) -> Optional[str]:
    """'diagnosis', 'control' or None (age group, lifestyle, metadata)."""
    if broader is None:
        broader = _vocabulary()[1]
    if code in _CONTROL_CODES:
        return "control"
    parents = set(broader.get(code, ()))
    if code in _NOT_DIAGNOSIS or parents & _NOT_DIAGNOSIS:
        return None
    return "diagnosis"


# ── Decoding ──────────────────────────────────────────────────────────────────

def _forms(text: str, course_words: Iterable[str]) -> List[str]:
    text = _WITH_PHRASE.sub(" ", text.lower().replace("_", " "))
    words = [w for w in re.split(r"[\s,;:()]+", text) if w and w not in _SCAFFOLD_WORDS]
    base = " ".join(words)
    forms = [base, base.replace("-", " "), base.replace(" ", "-")]
    if base.endswith("s"):
        forms.append(base[:-1])                       # "controls"
    without_course = " ".join(w for w in words if w not in set(course_words))
    if without_course and without_course != base:
        forms.append(without_course)                  # "first episode psychosis"
    return [f for f in dict.fromkeys(forms) if f]


def label_code(text: str, leaf: Optional[Dict[str, str]] = None,
               course_words: Optional[Iterable[str]] = None) -> Optional[str]:
    """The diagnosis code a group label names exactly, or None."""
    if leaf is None or course_words is None:
        leaf, _broader, course_words = _vocabulary()
    text = (text or "").strip()
    if not text or _YES_NO.match(text) or _NEGATION.search(text) or not re.search(r"[a-zA-Z]{2}", text):
        return None
    parts = [p for p in re.split(r"\s*/\s*", text) if p] if "/" in text else [text]
    codes = []
    for part in parts:
        code = next((leaf[f] for f in _forms(part, course_words) if f in leaf), None)
        if code is None:
            return None
        codes.append(code)
    # "Schizophrenia/Schizoaffective" names two diagnoses: only one code if they agree.
    return codes[0] if len(set(codes)) == 1 else None


def level_texts(sidecar: Optional[dict], column: str) -> Dict[str, str]:
    """value (lower case) → Levels text of *column* in participants.json."""
    if not isinstance(sidecar, dict):
        return {}
    wanted = column.strip().lower()
    entry = next((v for k, v in sidecar.items() if str(k).strip().lower() == wanted), None)
    if not isinstance(entry, dict):
        return {}
    levels = entry.get("Levels") or entry.get("levels")
    if not isinstance(levels, dict):
        return {}
    return {str(k).strip().lower(): str(v).strip() for k, v in levels.items()}


def group_columns(rows: Sequence[Tuple[str, Any]]) -> List[str]:
    """Columns of extra that can hold a participant group."""
    values: Dict[str, set] = {}
    for _pid, extra in rows:
        if not isinstance(extra, dict):
            continue
        for key, value in extra.items():
            if key.startswith("_") or value is None or str(value).strip() == "":
                continue
            values.setdefault(key, set()).add(str(value).strip())
    return [key for key, seen in values.items()
            if _GROUP_KEY.search(key.strip()) and not _NOT_GROUP_KEY.search(key.strip())
            and _MIN_VALUES <= len(seen) <= _MAX_VALUES]


def plan_dataset(rows: Sequence[Tuple[str, Any]], sidecar: Optional[dict],
                 decode: Callable[[str], Optional[str]] = label_code,
                 kind: Callable[[str], Optional[str]] = code_kind) -> Optional[dict]:
    """Which column to read and the code of each participant; None when nothing names a diagnosis.

    Returns {"column", "codes": {value: code}, "labels": {value: text},
             "assign": {participant_id: code}}.
    """
    best: Optional[dict] = None
    for column in group_columns(rows):
        texts = level_texts(sidecar, column)
        codes: Dict[str, str] = {}
        labels: Dict[str, str] = {}
        for _pid, extra in rows:
            value = str((extra or {}).get(column, "") or "").strip()
            if not value or value in codes or value in labels:
                continue
            text = texts.get(value.lower(), value)
            labels[value] = text
            code = decode(text)
            if code and kind(code):
                codes[value] = code
        if not any(kind(code) == "diagnosis" for code in codes.values()):
            continue
        assign = {}
        for pid, extra in rows:
            value = str((extra or {}).get(column, "") or "").strip()
            if value in codes:
                assign[pid] = codes[value]
        if best is None or len(assign) > len(best["assign"]):
            best = {"column": column, "codes": codes, "labels": labels, "assign": assign}
    return best


# ── Database ──────────────────────────────────────────────────────────────────

class ParticipantSidecarError(Exception):
    """participants.json exists or may exist, but could not be read."""


def fetch_sidecar(accession: str) -> Optional[dict]:
    """participants.json of an OpenNeuro dataset (latest snapshot), or None
    when the dataset has none (404). Raises ParticipantSidecarError when it
    could not be read: a dataset decoded without the sidecar's Levels would
    get raw values as codes, and once it has any it is never looked at again."""
    if not re.fullmatch(r"ds\d{6}", accession or ""):
        return None
    url = SIDECAR_URL.format(accession=accession)
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise ParticipantSidecarError(f"{url}: {exc}") from exc
    # URLError and timeouts are OSErrors, IncompleteRead an HTTPException,
    # JSONDecodeError a ValueError.
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise ParticipantSidecarError(f"{url}: {exc}") from exc


_DATASETS = """
SELECT d.id, d.accession_id FROM bids_datasets d
WHERE EXISTS (SELECT 1 FROM bids_participants p WHERE p.dataset_id = d.id)
  AND NOT EXISTS (SELECT 1 FROM bids_participants p WHERE p.dataset_id = d.id
                  AND p.diagnosis IS NOT NULL AND p.diagnosis <> '')
"""
_ROWS = "SELECT participant_id, extra FROM bids_participants WHERE dataset_id = :id"
_WRITE = """
UPDATE bids_participants
SET diagnosis = CAST(:code AS text),
    extra = (CASE WHEN json_typeof(extra) = 'object' THEN extra::jsonb ELSE '{}'::jsonb END
             || jsonb_build_object(CAST(:source_key AS text), CAST(:column AS text)))::json
WHERE dataset_id = :id AND participant_id = ANY(CAST(:pids AS text[]))
  AND (diagnosis IS NULL OR diagnosis = '')
"""


def _as_dict(extra: Any) -> Any:
    if isinstance(extra, str):
        try:
            return json.loads(extra)
        except ValueError:
            return None
    return extra


async def run(conn, accessions: Optional[Sequence[str]] = None, apply: bool = False,
              fetch: Callable[[str], Optional[dict]] = fetch_sidecar) -> List[dict]:
    """Plan (and with *apply*, write) the diagnoses for datasets without any.

    Runs in the caller's transaction. Returns one report per decoded dataset.
    """
    from sqlalchemy import text

    sql = _DATASETS + ("" if accessions is None else " AND d.accession_id = ANY(:accessions)")
    params = {} if accessions is None else {"accessions": list(accessions)}
    datasets = (await conn.execute(text(sql + " ORDER BY d.accession_id"), params)).all()
    reports = []
    for dataset_id, accession in datasets:
        rows = [(pid, _as_dict(extra)) for pid, extra in
                (await conn.execute(text(_ROWS), {"id": dataset_id})).all()]
        if not group_columns(rows):
            continue
        try:
            sidecar = await asyncio.to_thread(fetch, accession)
        except ParticipantSidecarError as exc:
            # Skipped, not decoded without it: the dataset still has no
            # diagnosis, so the next run tries again.
            log.warning("participant groups: %s skipped, sidecar not read (%s)", accession, exc)
            continue
        plan = plan_dataset(rows, sidecar)
        if plan is None:
            continue
        written = 0
        if apply:
            by_code: Dict[str, List[str]] = {}
            for pid, code in plan["assign"].items():
                by_code.setdefault(code, []).append(pid)
            for code, pids in by_code.items():
                result = await conn.execute(text(_WRITE), {
                    "code": code, "source_key": SOURCE_KEY, "column": plan["column"],
                    "id": dataset_id, "pids": pids})
                written += result.rowcount or 0
        counts: Dict[str, int] = {}
        for code in plan["assign"].values():
            counts[code] = counts.get(code, 0) + 1
        reports.append({"accession": accession, "column": plan["column"],
                        "values": {v: (plan["labels"].get(v), c) for v, c in plan["codes"].items()},
                        "participants": counts, "written": written})
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
        for report in reports:
            print(json.dumps(report, ensure_ascii=False))
        total = sum(sum(r["participants"].values()) for r in reports)
        print(f"{len(reports)} datasets, {total} participants "
              + ("written: " + str(sum(r["written"] for r in reports)) if apply else "(dry run)"))
        return 0

    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(_main()))
