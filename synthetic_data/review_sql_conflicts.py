#!/usr/bin/env python3
"""
review_sql_conflicts.py — Detect and auto-resolve SQL conflicts in training data.

A SQL conflict occurs when the same (or very similar) natural-language question
maps to two or more different SQL queries.  This can happen when:
  - Gemini is inconsistent across batches for the same intent.
  - A question was slightly rephrased but the SQL intent was accidentally changed.

Three resolution tiers (applied in order):
  1. AUTO_KEEP_MAJORITY  — If ≥60% of variants agree on one SQL, keep that SQL
                           for all rows and mark the minority rows for removal.
  2. AUTO_KEEP_LONGEST   — If all variants are structurally similar (high SQL
                           similarity) keep the longest (most complete) one.
  3. NEEDS_MANUAL        — Otherwise flag for human review. All rows are kept
                           in the auto-resolved output; the report lists them.

Expected input: JSONL with at least:
  - input or question
  - output or sql

Output:
  - auto-resolved JSONL  (--write-auto-resolved)
  - JSON report          (--report)
  - Markdown summary     (--summary)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Set, Tuple


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


def norm_question(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\"'`?!.,;]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def norm_sql(sql: str) -> str:
    sql = sql.strip().lower().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def sql_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_sql(a), norm_sql(b)).ratio()


# ── conflict detection ────────────────────────────────────────────────────────

def detect_conflicts(rows: List[dict]) -> Dict[str, List[dict]]:
    """
    Group rows by normalised question.  Return only groups where ≥2 distinct
    normalised SQL strings exist (i.e. there is a conflict).
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        key = norm_question(get_question(row))
        groups[key].append(row)

    return {
        key: group
        for key, group in groups.items()
        if len({norm_sql(get_sql(r)) for r in group}) > 1
    }


# ── resolution ────────────────────────────────────────────────────────────────

def resolve_conflict(group: List[dict]) -> dict:
    """
    Try to resolve a conflict group.  Returns a dict with keys:
      resolution  — AUTO_KEEP_MAJORITY | AUTO_KEEP_LONGEST | NEEDS_MANUAL
      canonical_sql — the chosen SQL (or None for NEEDS_MANUAL)
      drop_row_ids  — row ids to remove from the resolved output
      keep_row_ids  — row ids to keep
      notes         — human-readable explanation
    """
    norm_sqls = [norm_sql(get_sql(r)) for r in group]
    counts = Counter(norm_sqls)
    most_common_sql, most_common_count = counts.most_common(1)[0]
    majority_threshold = 0.6 * len(group)

    # Tier 1: majority vote
    if most_common_count >= majority_threshold and most_common_count > 1:
        keep_ids = [r["_row_id"] for r in group if norm_sql(get_sql(r)) == most_common_sql]
        drop_ids = [r["_row_id"] for r in group if norm_sql(get_sql(r)) != most_common_sql]
        canonical = get_sql(next(r for r in group if norm_sql(get_sql(r)) == most_common_sql))
        return {
            "resolution": "AUTO_KEEP_MAJORITY",
            "canonical_sql": canonical,
            "keep_row_ids": keep_ids,
            "drop_row_ids": drop_ids,
            "notes": (
                f"{most_common_count}/{len(group)} rows agree on the majority SQL; "
                f"{len(drop_ids)} minority row(s) removed."
            ),
        }

    # Tier 2: all SQL variants are structurally similar — keep longest
    sql_variants = list(counts.keys())
    all_similar = all(
        sql_sim(sql_variants[i], sql_variants[j]) >= 0.80
        for i in range(len(sql_variants))
        for j in range(i + 1, len(sql_variants))
    )
    if all_similar:
        # Keep the row with the longest (most complete) SQL
        best_row = max(group, key=lambda r: len(norm_sql(get_sql(r))))
        canonical = get_sql(best_row)
        canonical_norm = norm_sql(canonical)
        keep_ids = [r["_row_id"] for r in group if norm_sql(get_sql(r)) == canonical_norm]
        drop_ids = [r["_row_id"] for r in group if norm_sql(get_sql(r)) != canonical_norm]
        return {
            "resolution": "AUTO_KEEP_LONGEST",
            "canonical_sql": canonical,
            "keep_row_ids": keep_ids,
            "drop_row_ids": drop_ids,
            "notes": (
                f"All {len(sql_variants)} SQL variants are ≥80% similar; "
                f"kept the longest variant, dropped {len(drop_ids)} shorter row(s)."
            ),
        }

    # Tier 3: needs manual review — keep all rows
    return {
        "resolution": "NEEDS_MANUAL",
        "canonical_sql": None,
        "keep_row_ids": [r["_row_id"] for r in group],
        "drop_row_ids": [],
        "notes": (
            f"{len(sql_variants)} distinct SQL variants with low mutual similarity; "
            "manual review required."
        ),
    }


# ── output helpers ─────────────────────────────────────────────────────────────

def write_resolved(rows: List[dict], drop_row_ids: Set[int], path: Path) -> int:
    kept = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if row["_row_id"] in drop_row_ids:
                continue
            payload = {k: v for k, v in row.items() if not k.startswith("_")}
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def render_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# SQL Conflict Review",
        "",
        "## Summary",
        "",
        f"- Input rows:                {s['input_rows']}",
        f"- Conflict groups found:     {s['conflict_groups']}",
        f"  - AUTO_KEEP_MAJORITY:      {s['auto_majority']}",
        f"  - AUTO_KEEP_LONGEST:       {s['auto_longest']}",
        f"  - NEEDS_MANUAL:            {s['needs_manual']}",
        f"- Rows dropped (auto):       {s['rows_dropped']}",
        f"- Auto-resolved output rows: {s['resolved_rows']}",
        "",
    ]

    needs_manual = [g for g in report["conflict_groups"] if g["resolution"] == "NEEDS_MANUAL"]
    if needs_manual:
        lines += ["## Groups Needing Manual Review", ""]
        for idx, g in enumerate(needs_manual[:20], start=1):
            lines += [
                f"### {idx}. question: {g['question']!r}",
                f"- {g['notes']}",
            ]
            for v in g["sql_variants"][:4]:
                lines.append(f"  - `{v[:120]}`")
            lines.append("")

    auto = [g for g in report["conflict_groups"] if g["resolution"] != "NEEDS_MANUAL"]
    if auto:
        lines += ["## Auto-Resolved Groups", ""]
        for idx, g in enumerate(auto[:20], start=1):
            lines += [
                f"### {idx}. [{g['resolution']}] {g['question']!r}",
                f"- {g['notes']}",
                f"- Canonical: `{(g['canonical_sql'] or '')[:120]}`",
                "",
            ]

    return "\n".join(lines).strip() + "\n"


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--write-auto-resolved", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    conflicts = detect_conflicts(rows)

    all_drop_ids: Set[int] = set()
    conflict_groups = []
    auto_majority = auto_longest = needs_manual = 0

    for question_key, group in conflicts.items():
        resolution = resolve_conflict(group)
        all_drop_ids.update(resolution["drop_row_ids"])

        if resolution["resolution"] == "AUTO_KEEP_MAJORITY":
            auto_majority += 1
        elif resolution["resolution"] == "AUTO_KEEP_LONGEST":
            auto_longest += 1
        else:
            needs_manual += 1

        conflict_groups.append(
            {
                "question": get_question(group[0]),
                "group_size": len(group),
                "sql_variants": list({get_sql(r) for r in group}),
                **resolution,
            }
        )

    conflict_groups.sort(key=lambda g: (-g["group_size"], g["resolution"]))

    for path in (args.report, args.summary, args.write_auto_resolved):
        path.parent.mkdir(parents=True, exist_ok=True)

    resolved_count = write_resolved(rows, all_drop_ids, args.write_auto_resolved)

    report = {
        "summary": {
            "input_rows": len(rows),
            "conflict_groups": len(conflicts),
            "auto_majority": auto_majority,
            "auto_longest": auto_longest,
            "needs_manual": needs_manual,
            "rows_dropped": len(all_drop_ids),
            "resolved_rows": resolved_count,
        },
        "conflict_groups": conflict_groups,
    }

    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.summary.write_text(render_markdown(report), encoding="utf-8")

    print(f"Input rows:       {len(rows)}")
    print(f"Conflict groups:  {len(conflicts)}")
    print(f"  AUTO_MAJORITY:  {auto_majority}")
    print(f"  AUTO_LONGEST:   {auto_longest}")
    print(f"  NEEDS_MANUAL:   {needs_manual}")
    print(f"Rows dropped:     {len(all_drop_ids)}")
    print(f"Resolved output:  {resolved_count} rows → {args.write_auto_resolved}")


if __name__ == "__main__":
    main()
