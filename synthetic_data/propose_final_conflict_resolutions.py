#!/usr/bin/env python3
"""
propose_final_conflict_resolutions.py — Auto-resolve remaining SQL conflicts.

Takes the conflict review JSON from review_sql_conflicts.py and resolves ALL
remaining conflicts (including NEEDS_MANUAL groups) using heuristic scoring:

  1. Prefer the SQL that passes EXPLAIN syntax-check (if --db-url given).
  2. Prefer the SQL with more SELECT columns (more complete output).
  3. Prefer the SQL that contains a JOIN (richer queries).
  4. Tie-break: prefer the longer SQL.

For each conflict group the best-scoring SQL becomes the canonical answer for
all rows in that group.  Rows that had an inferior SQL variant are removed —
only one question-SQL pair survives per conflict group.

This is the last step before the dataset is used for training.

Expected inputs:
  --conflict-review   JSON produced by review_sql_conflicts.py
  --input             The same JSONL that was fed to review_sql_conflicts.py

Output:
  --write-proposed-final  JSONL, fully conflict-resolved and ready for training
  --report                JSON with all resolution decisions
  --summary               Markdown summary
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set


# ── helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            line = line.strip()
            if line:
                row = json.loads(line)
                row["_row_id"] = idx
                rows.append(row)
    return rows


def get_question(row: dict) -> str:
    return (row.get("input") or row.get("question") or "").strip()


def get_sql(row: dict) -> str:
    return (row.get("output") or row.get("sql") or "").strip()


def norm_sql(sql: str) -> str:
    sql = sql.strip().lower().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql


# ── SQL scoring heuristics ─────────────────────────────────────────────────────

def sql_syntax_ok(sql: str, db_url: Optional[str]) -> bool:
    if not db_url:
        return True
    try:
        from sqlalchemy import create_engine, text as sa_text
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(sa_text(f"EXPLAIN {sql}"))
        return True
    except Exception:
        return False


def count_select_columns(sql: str) -> int:
    lower = sql.lower()
    select_idx = lower.find("select")
    from_idx = lower.find(" from ")
    if select_idx == -1 or from_idx == -1:
        return 0
    select_clause = sql[select_idx + 6: from_idx]
    depth = 0
    commas = 0
    for ch in select_clause:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            commas += 1
    return commas + 1


def score_sql(sql: str, db_url: Optional[str]) -> tuple:
    syntax = 1 if sql_syntax_ok(sql, db_url) else 0
    cols = count_select_columns(sql)
    has_join = 1 if re.search(r"\bjoin\b", sql, re.IGNORECASE) else 0
    return (syntax, cols, has_join, len(sql))


def pick_best_sql(variants: List[str], db_url: Optional[str]) -> str:
    return max(variants, key=lambda s: score_sql(s, db_url))


# ── resolution ────────────────────────────────────────────────────────────────

def resolve_all_conflicts(
    conflict_review: dict,
    rows: List[dict],
    db_url: Optional[str],
) -> tuple[Set[int], Dict[int, str], List[dict]]:
    """
    Returns:
      drop_ids          — row ids to remove entirely
      override_sql      — row_id → canonical SQL (for rows that survive but get a new SQL)
      resolution_log    — list of decision records for the report
    """
    drop_ids: Set[int] = set()
    override_sql: Dict[int, str] = {}
    resolution_log: List[dict] = []

    for group in conflict_review.get("conflict_groups", []):
        resolution = group.get("resolution", "")
        row_ids = group.get("keep_row_ids", []) + group.get("drop_row_ids", [])
        variants = group.get("sql_variants", [])

        # Groups already auto-resolved by review_sql_conflicts: honour their drop list.
        if resolution in {"AUTO_KEEP_MAJORITY", "AUTO_KEEP_LONGEST"}:
            drop_ids.update(group.get("drop_row_ids", []))
            resolution_log.append(
                {
                    "question": group["question"],
                    "resolution": resolution,
                    "canonical_sql": group.get("canonical_sql"),
                    "dropped_row_ids": group.get("drop_row_ids", []),
                    "notes": group.get("notes", ""),
                }
            )
            continue

        # NEEDS_MANUAL: resolve via heuristic scoring
        if not variants:
            continue
        best_sql = pick_best_sql(variants, db_url)
        best_norm = norm_sql(best_sql)
        sc = score_sql(best_sql, db_url)

        # Keep rows whose SQL matches the winner; drop the rest
        kept = []
        dropped = []
        for rid in row_ids:
            row = next((r for r in rows if r["_row_id"] == rid), None)
            if row is None:
                continue
            if norm_sql(get_sql(row)) == best_norm:
                kept.append(rid)
            else:
                dropped.append(rid)
                drop_ids.add(rid)

        resolution_log.append(
            {
                "question": group["question"],
                "resolution": "AUTO_HEURISTIC",
                "canonical_sql": best_sql,
                "score": {
                    "syntax_ok": bool(sc[0]),
                    "select_columns": sc[1],
                    "has_join": bool(sc[2]),
                    "sql_length": sc[3],
                },
                "kept_row_ids": kept,
                "dropped_row_ids": dropped,
                "notes": (
                    f"Resolved from {len(variants)} variants via heuristic scoring; "
                    f"{len(dropped)} row(s) removed."
                ),
            }
        )

    return drop_ids, override_sql, resolution_log


# ── output helpers ─────────────────────────────────────────────────────────────

def render_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# Final Conflict Resolutions",
        "",
        "## Summary",
        "",
        f"- Input rows:              {s['input_rows']}",
        f"- Conflict groups:         {s['conflict_groups']}",
        f"  - AUTO_KEEP_MAJORITY:    {s['auto_majority']}",
        f"  - AUTO_KEEP_LONGEST:     {s['auto_longest']}",
        f"  - AUTO_HEURISTIC:        {s['auto_heuristic']}",
        f"- Total rows dropped:      {s['rows_dropped']}",
        f"- Proposed-final rows:     {s['proposed_final_rows']}",
        "",
        "## Resolution Details",
        "",
    ]

    for idx, r in enumerate(report["resolutions"][:30], start=1):
        lines += [
            f"### {idx}. [{r['resolution']}] {r['question']!r}",
            f"- {r['notes']}",
        ]
        if r.get("canonical_sql"):
            lines.append(f"- Canonical: `{r['canonical_sql'][:140]}`")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conflict-review", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--write-proposed-final", type=Path, required=True)
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    conflict_review = json.loads(args.conflict_review.read_text(encoding="utf-8"))
    rows = load_jsonl(args.input)

    drop_ids, override_sql, resolution_log = resolve_all_conflicts(
        conflict_review, rows, args.db_url
    )

    for path in (args.report, args.summary, args.write_proposed_final):
        path.parent.mkdir(parents=True, exist_ok=True)

    final_count = 0
    with args.write_proposed_final.open("w", encoding="utf-8") as fh:
        for row in rows:
            if row["_row_id"] in drop_ids:
                continue
            payload = {k: v for k, v in row.items() if not k.startswith("_")}
            if row["_row_id"] in override_sql:
                if "output" in payload:
                    payload["output"] = override_sql[row["_row_id"]]
                elif "sql" in payload:
                    payload["sql"] = override_sql[row["_row_id"]]
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            final_count += 1

    auto_majority = sum(1 for r in resolution_log if r["resolution"] == "AUTO_KEEP_MAJORITY")
    auto_longest = sum(1 for r in resolution_log if r["resolution"] == "AUTO_KEEP_LONGEST")
    auto_heuristic = sum(1 for r in resolution_log if r["resolution"] == "AUTO_HEURISTIC")

    report = {
        "summary": {
            "input_rows": len(rows),
            "conflict_groups": len(resolution_log),
            "auto_majority": auto_majority,
            "auto_longest": auto_longest,
            "auto_heuristic": auto_heuristic,
            "rows_dropped": len(drop_ids),
            "proposed_final_rows": final_count,
        },
        "resolutions": resolution_log,
    }

    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.summary.write_text(render_markdown(report), encoding="utf-8")

    print(f"Input rows:          {len(rows)}")
    print(f"Conflict groups:     {len(resolution_log)}")
    print(f"  AUTO_MAJORITY:     {auto_majority}")
    print(f"  AUTO_LONGEST:      {auto_longest}")
    print(f"  AUTO_HEURISTIC:    {auto_heuristic}")
    print(f"Rows dropped:        {len(drop_ids)}")
    print(f"Proposed-final rows: {final_count} → {args.write_proposed_final}")


if __name__ == "__main__":
    main()
