"""
collect_response.py — Parse teacher-LLM responses into validated training data.

Usage:
    # After pasting a prompt into Gemini/GPT and copying the JSON response:
    python collect_response.py \\
        --response response_001.json \\
        --db-url "postgresql://user@localhost:5429/bids_sql" \\
        --out training.jsonl

The script:
  1. Parses the JSON array from the teacher response
  2. Runs each SQL against the DB (dry-run: ROLLBACK) to verify it executes
  3. Appends valid pairs to training.jsonl in instruction-tuning format
  4. Routes timeout / DB-data issues into a separate needs-repair bucket
  5. Reports invalid pairs with the error so you can fix them manually
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent))
from value_mappings import (
    SEX_LABEL, HANDEDNESS_LABEL,
    FIELD_STANDARD_CODE_LABEL, FIELD_CONCEPT_EXPANSION,
)
from sql_expander import expand_sql

# All canonical codes from value_mappings (keys of the reverse maps)
_VALID_SEX_VALUES        = set(SEX_LABEL.keys())
_VALID_HANDEDNESS_VALUES = set(HANDEDNESS_LABEL.keys())

# Valid values for normalized DB columns: standard_codes (leaf nodes) ∪ concept keys
# (intermediate nodes that get expanded at inference time).
# All are lowercase — the DB stores only lowercase standard_codes, and concept keys
# are also lowercase.  A literal like 'EEG' or 'Bold' is invalid even if 'eeg'/'bold'
# would be correct.
def _valid_column_values(column: str) -> frozenset[str]:
    sc_labels  = FIELD_STANDARD_CODE_LABEL.get(column, {})
    expansions = FIELD_CONCEPT_EXPANSION.get(column, {})
    return frozenset(sc_labels.keys()) | frozenset(expansions.keys())

_VALID_SUFFIX_VALUES   = _valid_column_values("suffix")
_VALID_DATATYPE_VALUES = _valid_column_values("datatype")
_VALID_TASK_VALUES     = _valid_column_values("task")
_VALID_DIAGNOSIS_VALUES = _valid_column_values("diagnosis")

# Regex to extract literal string values used in sex/handedness filters
_SEX_FILTER_RE        = re.compile(r"p\.sex\s+(?:=|ILIKE|LIKE)\s+'([^']*)'", re.IGNORECASE)
_HANDEDNESS_FILTER_RE = re.compile(r"p\.handedness\s+(?:=|ILIKE|LIKE)\s+'([^']*)'", re.IGNORECASE)
_STRIP_WILDCARDS      = re.compile(r"^%?(.*?)%?$")

# Regex for normalized columns that must use exact standard_codes (= and IN/NOT IN,
# not ILIKE — free-text ILIKE on description_text is fine but not on normalized fields).
# Matches all table aliases: o.suffix, bo.suffix, o1.task, bo2.datatype, p.diagnosis, etc.
_ALIAS_OBJ = r"(?:bo\d*|o\d*)"
_SUFFIX_EQ_RE    = re.compile(rf"{_ALIAS_OBJ}\.suffix\s*=\s*'([^']*)'",    re.IGNORECASE)
_DATATYPE_EQ_RE  = re.compile(rf"{_ALIAS_OBJ}\.datatype\s*=\s*'([^']*)'",  re.IGNORECASE)
_TASK_EQ_RE      = re.compile(rf"{_ALIAS_OBJ}\.task\s*=\s*'([^']*)'",      re.IGNORECASE)
_DIAGNOSIS_EQ_RE = re.compile(r"p\.diagnosis\s*=\s*'([^']*)'",              re.IGNORECASE)

# Same but for IN (...) forms — captures the full comma-separated quoted value list.
_SUFFIX_IN_RE    = re.compile(rf"{_ALIAS_OBJ}\.suffix\s+(?:NOT\s+)?IN\s*\(([^)]+)\)",    re.IGNORECASE)
_DATATYPE_IN_RE  = re.compile(rf"{_ALIAS_OBJ}\.datatype\s+(?:NOT\s+)?IN\s*\(([^)]+)\)",  re.IGNORECASE)
_TASK_IN_RE      = re.compile(rf"{_ALIAS_OBJ}\.task\s+(?:NOT\s+)?IN\s*\(([^)]+)\)",      re.IGNORECASE)
_DIAGNOSIS_IN_RE = re.compile(r"p\.diagnosis\s+(?:NOT\s+)?IN\s*\(([^)]+)\)",             re.IGNORECASE)

# extension is a file-format column ('.nii.gz', '.json', …).  Any attempt to cast
# it to text and search for a keyword is a hallucination — extension never contains
# metadata field names.
_EXT_CAST_RE = re.compile(r"extension::text\s+(?:=|ILIKE|LIKE)", re.IGNORECASE)


def _parse_in_values(values_str: str) -> list[str]:
    """Parse a SQL IN-list string like "'foo', 'bar'" into ['foo', 'bar']."""
    return [tok.strip().strip("'\"") for tok in values_str.split(",") if tok.strip().strip("'\"")]


def _check_semantic_values(sql: str) -> Optional[str]:
    """
    Return an error string if the SQL uses implausible or unrecognized literal values
    for normalized DB columns.  Returns None if the SQL looks clean.

    Validates:
      - sex, handedness — against known reverse-map codes
      - suffix, datatype, task, diagnosis — against known standard_codes + concept keys
        (= and IN/NOT IN comparisons; ILIKE is left unchecked as it's used for free-text search)
    """
    for m in _SEX_FILTER_RE.finditer(sql):
        val = _STRIP_WILDCARDS.match(m.group(1)).group(1).strip().lower()
        if val and val not in _VALID_SEX_VALUES:
            return f"implausible sex filter value: '{m.group(1)}'"

    for m in _HANDEDNESS_FILTER_RE.finditer(sql):
        val = _STRIP_WILDCARDS.match(m.group(1)).group(1).strip().lower()
        if val and val not in _VALID_HANDEDNESS_VALUES:
            return f"implausible handedness filter value: '{m.group(1)}'"

    _normalized_checks = [
        (_SUFFIX_EQ_RE,    _SUFFIX_IN_RE,    _VALID_SUFFIX_VALUES,    "suffix"),
        (_DATATYPE_EQ_RE,  _DATATYPE_IN_RE,  _VALID_DATATYPE_VALUES,  "datatype"),
        (_TASK_EQ_RE,      _TASK_IN_RE,      _VALID_TASK_VALUES,      "task"),
        (_DIAGNOSIS_EQ_RE, _DIAGNOSIS_IN_RE, _VALID_DIAGNOSIS_VALUES, "diagnosis"),
    ]
    for eq_pattern, in_pattern, valid_set, col_name in _normalized_checks:
        if not valid_set:
            continue  # value_mappings.yaml has no entries for this column — skip

        # Check = 'value' form
        for m in eq_pattern.finditer(sql):
            val = m.group(1).strip()
            if val not in valid_set:
                hint = f" (did you mean '{val.lower()}'?)" if val.lower() in valid_set else ""
                return (
                    f"unrecognized {col_name} value: '{val}'{hint} — "
                    f"use an exact standard_code or concept key from value_mappings.yaml"
                )

        # Check IN ('v1', 'v2', ...) and NOT IN (...) forms
        for m in in_pattern.finditer(sql):
            for val in _parse_in_values(m.group(1)):
                if val not in valid_set:
                    hint = f" (did you mean '{val.lower()}'?)" if val.lower() in valid_set else ""
                    return (
                        f"unrecognized {col_name} value in IN list: '{val}'{hint} — "
                        f"use an exact standard_code or concept key from value_mappings.yaml"
                    )

    if _EXT_CAST_RE.search(sql):
        return "extension::text used as metadata search (extension is file format only)"

    return None


INSTRUCTION = (
    "You are a SQL expert for a BIDS neuroimaging database. "
    "Given a natural language question about brain imaging datasets, "
    "write a valid PostgreSQL query."
)


# Errors that indicate a DB/data issue rather than a definitely wrong SQL query.
# These should not be promoted to the gold training set; keep them in a separate
# bucket for later repair or re-validation.
_DATA_ERRORS = (
    "UntranslatableCharacter",   # stored JSON contains raw unicode escapes
    "unsupported Unicode escape", # same, different message variant
)

_TIMEOUT_ERROR = "canceling statement due to statement timeout"


def norm_sql_for_bundle(sql: str) -> str:
    """Normalise SQL for bundle-consistency comparison (whitespace + case only)."""
    return re.sub(r"\s+", " ", sql.strip().lower().rstrip(";")).strip()


def check_bundle_consistency(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Verify that every paraphrase bundle has identical SQL across all its members.

    If a bundle's members disagree on SQL, their paraphrase_bundle_id is set to
    None so they are kept as standalone pairs rather than discarded.  The data
    itself is fine — only the grouping was wrong.

    Returns (cleaned_records, split_bundle_reports) where each report is a dict
    describing the offending bundle.
    """
    from collections import defaultdict

    # Group by bundle id (ignore null / standalone entries)
    bundles: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        bid = rec.get("paraphrase_bundle_id")
        if bid is not None:
            bundles[bid].append(rec)

    split_reports = []
    bad_bundle_ids: set[str] = set()

    for bid, members in bundles.items():
        norm_sqls = [norm_sql_for_bundle(m["output"]) for m in members]
        if len(set(norm_sqls)) > 1:
            bad_bundle_ids.add(bid)
            split_reports.append({
                "bundle_id": bid,
                "member_count": len(members),
                "distinct_sqls": len(set(norm_sqls)),
                "questions": [m["input"] for m in members],
                "sqls": [m["output"] for m in members],
            })

    # Nullify bundle ids for bad bundles
    cleaned = []
    for rec in records:
        if rec.get("paraphrase_bundle_id") in bad_bundle_ids:
            rec = {**rec, "paraphrase_bundle_id": None}
        cleaned.append(rec)

    return cleaned, split_reports


def validate_sql(session: Session, sql: str) -> Tuple[str, Optional[str]]:
    """
    Run the SQL inside a transaction that is always rolled back.
    Returns:
      - ("valid", None) for execution-clean SQL
      - ("needs_repair", reason) for SQL blocked by DB/data issues
      - ("invalid", error) for SQL that should not enter the dataset

    Data-encoding errors and statement timeouts are treated as infrastructure
    issues. These pairs should be reviewed separately, not written into the
    supervised gold set.
    """
    try:
        with session.begin_nested():
            session.execute(text(sql))
            session.rollback()
        return "valid", None
    except Exception as e:
        msg = str(e)
        if any(tag in msg for tag in _DATA_ERRORS):
            return "needs_repair", msg
        if _TIMEOUT_ERROR in msg:
            return "needs_repair", msg
        return "invalid", msg


def load_response(path: Path) -> list:
    """
    Load a teacher response. Accepts:
      - A file containing a raw JSON array
      - A file containing the JSON array wrapped in markdown fences
    """
    raw = path.read_text(encoding="utf-8").strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True,
                        help="Path to the teacher's JSON response file")
    parser.add_argument("--db-url", required=False, default=None,
                        help='e.g. "postgresql://user@localhost:5429/bids_sql"')
    parser.add_argument("--out", type=Path, default=Path("training.jsonl"),
                        help="Output JSONL file (appended, not overwritten)")
    parser.add_argument(
        "--repair-out",
        type=Path,
        default=Path("needs_repair.jsonl"),
        help="Output JSONL file for pairs blocked by DB/data issues",
    )
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override the instruction field written into training.jsonl")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip SQL validation (useful if DB is offline)")
    args = parser.parse_args()

    pairs = load_response(args.response)
    print(f"Loaded {len(pairs)} pairs from {args.response}")

    instruction = args.instruction if args.instruction else INSTRUCTION

    engine = (
        create_engine(args.db_url,
                      connect_args={"options": "-c statement_timeout=30000"})
        if not args.no_validate else None
    )

    needs_repair, invalid = 0, 0
    valid_buffer: list[dict] = []   # accumulate before bundle check
    repair_records: list[tuple[dict, str]] = []

    with (
        Session(engine) if engine else _null_ctx() as session,
    ):
        for i, pair in enumerate(pairs):
            question = pair.get("question", "").strip()
            sql = pair.get("sql", "").strip()
            pattern = pair.get("pattern", "?")

            if not question or not sql:
                print(f"  [{i+1}] SKIP  — missing question or sql")
                invalid += 1
                continue

            record = {
                "instruction": instruction,
                "input": question,
                "output": sql,
                "pattern": pattern,
                "source": args.response.stem,
            }
            for optional_key in ("family", "sql_structure", "paraphrase_bundle_id"):
                if optional_key in pair:
                    record[optional_key] = pair.get(optional_key)

            semantic_err = _check_semantic_values(sql)
            if semantic_err:
                print(f"  [{i+1}] FAIL   (semantic) — {semantic_err}")
                print(f"         Q: {question[:80]}")
                invalid += 1
                continue

            if session is not None:
                # Expand concept keys (e.g. 'epilepsy_spectrum' → IN (...)) for
                # validation only — the training record keeps the unexpanded form
                # so the LLM learns to emit concept keys at inference time.
                sql_for_validation = expand_sql(sql)
                status, detail = validate_sql(session, sql_for_validation)
                if status == "invalid":
                    print(f"  [{i+1}] FAIL   (pattern={pattern}) — {detail[:120]}")
                    print(f"         Q: {question[:80]}")
                    invalid += 1
                    continue
                if status == "needs_repair":
                    repair_records.append((record, detail))
                    print(f"  [{i+1}] REPAIR (pattern={pattern}) {question[:70]}")
                    needs_repair += 1
                    continue

            valid_buffer.append(record)
            print(f"  [{i+1}] OK    (pattern={pattern}) {question[:70]}")

    # ── Bundle consistency check ───────────────────────────────────────────────
    # Must happen after all pairs are validated so we see the full bundle.
    valid_buffer, split_reports = check_bundle_consistency(valid_buffer)
    if split_reports:
        print(f"\n  BUNDLE WARNING — {len(split_reports)} bundle(s) had mismatched SQL "
              f"and were split into standalone pairs:")
        for r in split_reports:
            print(f"    bundle '{r['bundle_id']}': {r['member_count']} members, "
                  f"{r['distinct_sqls']} distinct SQLs")
            for q in r["questions"]:
                print(f"      - {q[:80]}")

    # ── Write outputs ──────────────────────────────────────────────────────────
    with (
        open(args.out, "a", encoding="utf-8") as out_f,
        open(args.repair_out, "a", encoding="utf-8") as repair_f,
    ):
        for record in valid_buffer:
            out_f.write(json.dumps(record) + "\n")
        for record, detail in repair_records:
            repair_f.write(json.dumps({**record, "repair_reason": detail}) + "\n")

    valid = len(valid_buffer)
    print(f"\n{valid} valid pairs appended to {args.out}")
    print(f"{needs_repair} pairs written to {args.repair_out} for later repair")
    print(f"{invalid} pairs rejected")
    if split_reports:
        print(f"{len(split_reports)} bundle(s) split into standalones (SQL mismatch — check above)")
    if needs_repair or invalid:
        print("Review needs-repair pairs separately; fix rejected pairs manually and re-run if needed.")


class _null_ctx:
    """Context manager that yields None (for --no-validate mode)."""
    def __enter__(self): return None
    def __exit__(self, *_): pass


if __name__ == "__main__":
    main()
