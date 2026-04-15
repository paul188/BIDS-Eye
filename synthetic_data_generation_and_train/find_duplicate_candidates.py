#!/usr/bin/env python3
"""
find_duplicate_candidates.py — Find exact and near-duplicate (question, SQL) pairs.

Two kinds of duplicates are detected:
  1. Exact duplicates — identical normalised question string; only the first occurrence
     is kept in the output JSONL (--write-exact-dedup).
  2. Near-duplicate candidates — questions with high lexical similarity but different
     SQL.  These are flagged in the report for human review; they are NOT removed
     automatically because the SQL difference might be intentional.

Expected input: JSONL with at least:
  - input or question  (the NL question)
  - output or sql      (the SQL answer)

Output:
  - exact-dedup JSONL   (--write-exact-dedup)
  - JSON report         (--report)
  - Markdown summary    (--summary)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
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


def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def jaccard(a: str, b: str, stopwords: Set[str]) -> float:
    tokens_a = {t for t in re.findall(r"[a-z0-9]+", a) if t not in stopwords and len(t) > 1}
    tokens_b = {t for t in re.findall(r"[a-z0-9]+", b) if t not in stopwords and len(t) > 1}
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 0.0


STOPWORDS = {
    "a", "an", "all", "and", "any", "are", "as", "at", "by", "data", "dataset",
    "datasets", "do", "does", "find", "for", "from", "has", "have", "in", "is",
    "list", "of", "or", "show", "that", "the", "their", "what", "which", "with",
}

# Threshold above which two questions are considered near-duplicates
NEAR_DUP_SEQ = 0.90
NEAR_DUP_JAC = 0.85


# ── analysis ───────────────────────────────────────────────────────────────────

def find_exact_duplicates(rows: List[dict]) -> Tuple[List[int], List[dict]]:
    """Return (drop_row_ids, dedup_rows). Keeps the first occurrence of each question."""
    seen: Dict[str, int] = {}   # norm_question → first row_id
    drop_ids: List[int] = []
    dedup_rows: List[dict] = []

    for row in rows:
        key = norm_question(get_question(row))
        if key in seen:
            drop_ids.append(row["_row_id"])
        else:
            seen[key] = row["_row_id"]
            dedup_rows.append(row)

    return drop_ids, dedup_rows


def find_near_duplicate_candidates(
    rows: List[dict],
) -> List[dict]:
    """
    Find pairs of rows where the questions are near-duplicates but the SQL differs.
    Returns a list of candidate records for the report.
    Only checks consecutive windows for efficiency (O(n) rather than O(n²)).
    """
    # Group by normalised question prefix (first 6 tokens) for a fast filter
    from collections import defaultdict

    bucket: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        nq = norm_question(get_question(row))
        prefix = " ".join(nq.split()[:6])
        bucket[prefix].append(row)

    candidates = []
    seen_pairs: Set[Tuple[int, int]] = set()

    # Within each bucket check all pairs
    for group in bucket.values():
        if len(group) < 2:
            continue
        for i, left in enumerate(group):
            for right in group[i + 1:]:
                pair = (min(left["_row_id"], right["_row_id"]), max(left["_row_id"], right["_row_id"]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                ql = norm_question(get_question(left))
                qr = norm_question(get_question(right))
                sr = seq_ratio(ql, qr)
                jac = jaccard(ql, qr, STOPWORDS)

                if sr >= NEAR_DUP_SEQ or (sr >= 0.85 and jac >= NEAR_DUP_JAC):
                    sql_l = norm_sql(get_sql(left))
                    sql_r = norm_sql(get_sql(right))
                    if sql_l != sql_r:
                        candidates.append(
                            {
                                "row_ids": [left["_row_id"], right["_row_id"]],
                                "questions": [get_question(left), get_question(right)],
                                "sql_a": get_sql(left),
                                "sql_b": get_sql(right),
                                "seq_ratio": round(sr, 4),
                                "token_jaccard": round(jac, 4),
                                "verdict": "near_dup_different_sql",
                            }
                        )

    return sorted(candidates, key=lambda c: -c["seq_ratio"])


# ── output helpers ─────────────────────────────────────────────────────────────

def write_dedup(rows: List[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = {k: v for k, v in row.items() if not k.startswith("_")}
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def render_markdown(report: dict) -> str:
    s = report["summary"]
    lines = [
        "# Duplicate Candidates Report",
        "",
        "## Summary",
        "",
        f"- Input rows:              {s['input_rows']}",
        f"- Exact duplicates found:  {s['exact_duplicate_count']}  (dropped from dedup output)",
        f"- Dedup output rows:       {s['dedup_rows']}",
        f"- Near-dup candidates:     {s['near_dup_candidate_count']}  (flagged, not removed)",
        "",
    ]

    if report["exact_duplicates"]:
        lines += ["## Exact Duplicates (dropped)", ""]
        for item in report["exact_duplicates"][:30]:
            lines.append(f"- row {item['dropped_row_id']} duplicates row {item['kept_row_id']}: {item['question']!r}")
        if len(report["exact_duplicates"]) > 30:
            lines.append(f"  … and {len(report['exact_duplicates']) - 30} more")
        lines.append("")

    if report["near_dup_candidates"]:
        lines += ["## Near-Duplicate Candidates (different SQL — manual review)", ""]
        for idx, c in enumerate(report["near_dup_candidates"][:20], start=1):
            lines += [
                f"### {idx}. seq_ratio={c['seq_ratio']}  jaccard={c['token_jaccard']}",
                f"- Row {c['row_ids'][0]}: {c['questions'][0]}",
                f"- Row {c['row_ids'][1]}: {c['questions'][1]}",
                f"- SQL A: `{c['sql_a'][:120]}`",
                f"- SQL B: `{c['sql_b'][:120]}`",
                "",
            ]

    return "\n".join(lines).strip() + "\n"


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--write-exact-dedup", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.input)

    drop_ids, dedup_rows = find_exact_duplicates(rows)

    # Build exact-dup report entries
    norm_to_first: Dict[str, int] = {}
    for row in rows:
        key = norm_question(get_question(row))
        if key not in norm_to_first:
            norm_to_first[key] = row["_row_id"]
    exact_dup_entries = []
    for row in rows:
        if row["_row_id"] in set(drop_ids):
            key = norm_question(get_question(row))
            exact_dup_entries.append(
                {
                    "dropped_row_id": row["_row_id"],
                    "kept_row_id": norm_to_first[key],
                    "question": get_question(row),
                }
            )

    near_dup_candidates = find_near_duplicate_candidates(dedup_rows)

    report = {
        "summary": {
            "input_rows": len(rows),
            "exact_duplicate_count": len(drop_ids),
            "dedup_rows": len(dedup_rows),
            "near_dup_candidate_count": len(near_dup_candidates),
        },
        "exact_duplicates": exact_dup_entries,
        "near_dup_candidates": near_dup_candidates,
    }

    for path in (args.report, args.summary, args.write_exact_dedup):
        path.parent.mkdir(parents=True, exist_ok=True)

    write_dedup(dedup_rows, args.write_exact_dedup)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.summary.write_text(render_markdown(report), encoding="utf-8")

    print(f"Input rows:          {len(rows)}")
    print(f"Exact duplicates:    {len(drop_ids)} dropped")
    print(f"Dedup output:        {len(dedup_rows)} rows → {args.write_exact_dedup}")
    print(f"Near-dup candidates: {len(near_dup_candidates)} flagged → {args.report}")


if __name__ == "__main__":
    main()
