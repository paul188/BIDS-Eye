#!/usr/bin/env python3
"""
question_diversity_filter.py — Filter weak or risky paraphrase bundles.

This script is meant for generated Text-to-SQL data where multiple questions may
share the same SQL intent. It helps keep only a few strong paraphrases per SQL
and flags:
  - near-duplicate wording
  - likely semantic drift
  - weak style diversity inside a paraphrase bundle

Expected input: JSONL with at least:
  - input or question
  - output or sql
Optional fields:
  - paraphrase_bundle_id
  - source

Output:
  - filtered JSONL
  - review JSON report
  - Markdown summary
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


STOPWORDS = {
    "a", "an", "all", "and", "any", "are", "as", "at", "by", "data", "dataset",
    "datasets", "do", "does", "find", "for", "from", "has", "have", "in", "is",
    "list", "of", "or", "show", "that", "the", "their", "what", "which", "with",
}

OPENING_STYLE_HINTS = {
    "find": "imperative",
    "list": "imperative",
    "show": "imperative",
    "identify": "formal",
    "determine": "formal",
    "which": "question",
    "what": "question",
    "are": "question",
    "is": "question",
    "i'm": "casual",
    "im": "casual",
}

SEMANTIC_MARKERS = [
    "at least", "at most", "more than", "less than", "fewer than", "exactly",
    "only", "all", "every", "without", "no ", "not ", "both", "either",
    "between", "average", "mean", "highest", "largest", "smallest", "top ",
]


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_row_id"] = idx
            rows.append(row)
    return rows


def get_question(row: dict) -> str:
    return (row.get("input") or row.get("question") or "").strip()


def get_sql(row: dict) -> str:
    return (row.get("output") or row.get("sql") or "").strip()


def normalize_question(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"\bn[-\s]?back\b", "nback", text)
    text = text.replace("resting-state", "resting state")
    text = re.sub(r"\belectroencephalography\b", "eeg", text)
    text = re.sub(r"\bfunctional magnetic resonance imaging\b", "fmri", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_sql(sql: str) -> str:
    sql = sql.strip().lower().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def token_set(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)


def semantic_marker_set(text: str) -> set[str]:
    lowered = text.lower()
    return {marker.strip() for marker in SEMANTIC_MARKERS if marker in lowered}


def guess_style(question: str) -> str:
    lowered = normalize_question(question)
    first = lowered.split(" ", 1)[0] if lowered else ""
    return OPENING_STYLE_HINTS.get(first, "other")


@dataclass
class PairAssessment:
    left_row_id: int
    right_row_id: int
    seq_ratio: float
    token_jaccard: float
    same_numbers: bool
    same_semantic_markers: bool
    verdict: str
    reasons: List[str]


def assess_pair(left: dict, right: dict) -> PairAssessment:
    q1 = normalize_question(get_question(left))
    q2 = normalize_question(get_question(right))
    t1 = token_set(q1)
    t2 = token_set(q2)
    sr = seq_ratio(q1, q2)
    tj = jaccard(t1, t2)

    nums1 = extract_numbers(q1)
    nums2 = extract_numbers(q2)
    same_numbers = nums1 == nums2
    markers1 = semantic_marker_set(q1)
    markers2 = semantic_marker_set(q2)
    same_markers = markers1 == markers2

    reasons: List[str] = []
    verdict = "ok"

    if sr >= 0.94 or (sr >= 0.9 and tj >= 0.88):
        verdict = "near_duplicate"
        reasons.append("wording is too similar")
    elif not same_numbers:
        verdict = "semantic_drift"
        reasons.append("numerical constraints differ")
    elif not same_markers and (markers1 or markers2):
        verdict = "semantic_drift"
        reasons.append("semantic markers differ")
    elif tj < 0.2 and sr < 0.45:
        verdict = "manual_review"
        reasons.append("phrasing is very different; meaning may have drifted")
    elif sr < 0.6 and tj < 0.35:
        reasons.append("good lexical spread")

    return PairAssessment(
        left_row_id=left["_row_id"],
        right_row_id=right["_row_id"],
        seq_ratio=round(sr, 4),
        token_jaccard=round(tj, 4),
        same_numbers=same_numbers,
        same_semantic_markers=same_markers,
        verdict=verdict,
        reasons=reasons,
    )


def bundle_key(row: dict) -> Tuple[str, str]:
    bundle = row.get("paraphrase_bundle_id")
    sql = normalize_sql(get_sql(row))
    if bundle is not None:
        return str(bundle), sql
    return f"sql::{hash(sql)}", sql


def build_bundle_report(rows: Sequence[dict], max_keep_per_sql: int) -> dict:
    bundles: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        bundles[bundle_key(row)].append(row)

    kept_row_ids: set[int] = set()
    dropped_row_ids: set[int] = set()
    bundle_reports = []

    for (_, sql), group in bundles.items():
        if len(group) == 1:
            kept_row_ids.add(group[0]["_row_id"])
            bundle_reports.append(
                {
                    "sql": sql,
                    "bundle_size": 1,
                    "styles": [guess_style(get_question(group[0]))],
                    "decision": "keep_singleton",
                    "keep_row_ids": [group[0]["_row_id"]],
                    "drop_row_ids": [],
                    "pair_assessments": [],
                    "rows": [
                        {
                            "row_id": group[0]["_row_id"],
                            "question": get_question(group[0]),
                            "style": guess_style(get_question(group[0])),
                        }
                    ],
                }
            )
            continue

        assessments: List[PairAssessment] = []
        near_dup_rows: set[int] = set()
        risky_rows: set[int] = set()

        for i, left in enumerate(group):
            for right in group[i + 1:]:
                assessment = assess_pair(left, right)
                assessments.append(assessment)
                if assessment.verdict == "near_duplicate":
                    near_dup_rows.add(right["_row_id"])
                elif assessment.verdict in {"semantic_drift", "manual_review"}:
                    risky_rows.add(left["_row_id"])
                    risky_rows.add(right["_row_id"])

        style_buckets: Dict[str, List[dict]] = defaultdict(list)
        for row in group:
            style_buckets[guess_style(get_question(row))].append(row)

        ordered_rows: List[dict] = []
        for style in sorted(style_buckets):
            style_rows = sorted(
                style_buckets[style],
                key=lambda r: (r["_row_id"] in risky_rows, r["_row_id"] in near_dup_rows, r["_row_id"]),
            )
            ordered_rows.extend(style_rows)

        keep_rows: List[dict] = []
        seen_styles: set[str] = set()
        for row in ordered_rows:
            row_id = row["_row_id"]
            style = guess_style(get_question(row))
            if row_id in risky_rows:
                continue
            if row_id in near_dup_rows:
                continue
            if style not in seen_styles or len(keep_rows) < max_keep_per_sql:
                keep_rows.append(row)
                seen_styles.add(style)
            if len(keep_rows) >= max_keep_per_sql:
                break

        if not keep_rows:
            keep_rows = [min(group, key=lambda r: r["_row_id"])]

        keep_ids = {row["_row_id"] for row in keep_rows}
        drop_ids = {row["_row_id"] for row in group if row["_row_id"] not in keep_ids}
        kept_row_ids.update(keep_ids)
        dropped_row_ids.update(drop_ids)

        decision = "keep_diverse_subset"
        if risky_rows:
            decision = "keep_subset_flagged_review"
        elif near_dup_rows and len(keep_rows) == 1:
            decision = "collapse_near_duplicates"

        bundle_reports.append(
            {
                "sql": sql,
                "bundle_size": len(group),
                "styles": sorted({guess_style(get_question(row)) for row in group}),
                "decision": decision,
                "keep_row_ids": sorted(keep_ids),
                "drop_row_ids": sorted(drop_ids),
                "pair_assessments": [assessment.__dict__ for assessment in assessments],
                "rows": [
                    {
                        "row_id": row["_row_id"],
                        "question": get_question(row),
                        "style": guess_style(get_question(row)),
                        "kept": row["_row_id"] in keep_ids,
                        "flagged_risky": row["_row_id"] in risky_rows,
                        "flagged_near_duplicate": row["_row_id"] in near_dup_rows,
                    }
                    for row in sorted(group, key=lambda r: r["_row_id"])
                ],
            }
        )

    return {
        "summary": {
            "input_rows": len(rows),
            "bundle_count": len(bundle_reports),
            "kept_rows": len(kept_row_ids),
            "dropped_rows": len(dropped_row_ids),
            "bundles_flagged_for_review": sum(
                1 for bundle in bundle_reports if bundle["decision"] == "keep_subset_flagged_review"
            ),
        },
        "bundles": sorted(bundle_reports, key=lambda bundle: (-bundle["bundle_size"], bundle["decision"])),
        "drop_row_ids": sorted(dropped_row_ids),
    }


def write_filtered(rows: Sequence[dict], drop_row_ids: Iterable[int], out_path: Path) -> int:
    drop_ids = set(drop_row_ids)
    kept = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["_row_id"] in drop_ids:
                continue
            payload = {k: v for k, v in row.items() if not k.startswith("_")}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            kept += 1
    return kept


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Question Diversity Filter",
        "",
        "## Summary",
        "",
        f"- Input rows: {summary['input_rows']}",
        f"- Bundle count: {summary['bundle_count']}",
        f"- Kept rows: {summary['kept_rows']}",
        f"- Dropped rows: {summary['dropped_rows']}",
        f"- Bundles flagged for review: {summary['bundles_flagged_for_review']}",
        "",
        "## Largest Bundles",
        "",
    ]

    for idx, bundle in enumerate(report["bundles"][:20], start=1):
        lines.extend(
            [
                f"### {idx}. bundle_size={bundle['bundle_size']} decision={bundle['decision']}",
                "",
                f"- Styles: {', '.join(bundle['styles'])}",
                f"- Keep row ids: {bundle['keep_row_ids']}",
                f"- Drop row ids: {bundle['drop_row_ids']}",
                "",
            ]
        )
        for row in bundle["rows"][:8]:
            marker = "KEEP" if row["kept"] else "DROP"
            flags = []
            if row["flagged_risky"]:
                flags.append("risky")
            if row["flagged_near_duplicate"]:
                flags.append("near-dup")
            suffix = f" [{' / '.join(flags)}]" if flags else ""
            lines.append(f"- {marker} row {row['row_id']} ({row['style']}): {row['question']}{suffix}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--filtered-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-keep-per-sql", type=int, default=3)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    report = build_bundle_report(rows, max_keep_per_sql=args.max_keep_per_sql)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.filtered_out.parent.mkdir(parents=True, exist_ok=True)

    kept = write_filtered(rows, report["drop_row_ids"], args.filtered_out)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.summary.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote filtered dataset to {args.filtered_out} ({kept} rows kept)")
    print(f"Wrote JSON report to {args.report}")
    print(f"Wrote Markdown summary to {args.summary}")


if __name__ == "__main__":
    main()
