#!/usr/bin/env python3
"""
post_process_pipeline.py — Validate raw Gemini results and audit duplication.

Pipeline stages:
  1. Parse raw `result_*.txt` files with the repo collector
  2. Write execution-clean pairs into the gold dataset
  3. Write timeout / DB-data-blocked pairs into a repair bucket
  4. Run duplicate and conflict analysis on the gold dataset
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constants import SYSTEM

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
COLLECT_SCRIPT = PIPELINE_DIR / "collect_response.py"
DUP_SCRIPT = PIPELINE_DIR / "find_duplicate_candidates.py"
CONFLICT_SCRIPT = PIPELINE_DIR / "review_sql_conflicts.py"
FINAL_PROPOSAL_SCRIPT = PIPELINE_DIR / "propose_final_conflict_resolutions.py"
DIVERSITY_FILTER_SCRIPT = PIPELINE_DIR / "question_diversity_filter.py"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--gold-out", type=Path, required=True)
    parser.add_argument("--repair-out", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--overwrite-gold", action="store_true")
    parser.add_argument("--max-keep-per-sql", type=int, default=3)
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite_gold:
        args.gold_out.write_text("", encoding="utf-8")
        args.repair_out.write_text("", encoding="utf-8")

    result_files = sorted(args.result_dir.glob("result_*.txt"))
    if not result_files:
        raise SystemExit(f"No result files found in {args.result_dir}")

    for result_path in result_files:
        run(
            [
                sys.executable,
                str(COLLECT_SCRIPT),
                "--response",
                str(result_path),
                "--db-url",
                args.db_url,
                "--out",
                str(args.gold_out),
                "--repair-out",
                str(args.repair_out),
                "--instruction",
                SYSTEM,
            ]
        )

    run(
        [
            sys.executable,
            str(DIVERSITY_FILTER_SCRIPT),
            "--input",
            str(args.gold_out),
            "--filtered-out",
            str(args.report_dir / "training.diversity_filtered.jsonl"),
            "--report",
            str(args.report_dir / "question_diversity_filter.json"),
            "--summary",
            str(args.report_dir / "question_diversity_filter.md"),
            "--max-keep-per-sql",
            str(args.max_keep_per_sql),
        ]
    )

    filtered_input = args.report_dir / "training.diversity_filtered.jsonl"

    run(
        [
            sys.executable,
            str(DUP_SCRIPT),
            "--input",
            str(filtered_input),
            "--report",
            str(args.report_dir / "duplicate_candidates.json"),
            "--summary",
            str(args.report_dir / "duplicate_candidates.md"),
            "--write-exact-dedup",
            str(args.report_dir / "training.exact_dedup.jsonl"),
        ]
    )

    dedup_input = args.report_dir / "training.exact_dedup.jsonl"

    run(
        [
            sys.executable,
            str(CONFLICT_SCRIPT),
            "--input",
            str(dedup_input),
            "--report",
            str(args.report_dir / "sql_conflict_review.json"),
            "--summary",
            str(args.report_dir / "sql_conflict_review.md"),
            "--write-auto-resolved",
            str(args.report_dir / "training.auto_resolved.jsonl"),
        ]
    )

    run(
        [
            sys.executable,
            str(FINAL_PROPOSAL_SCRIPT),
            "--conflict-review",
            str(args.report_dir / "sql_conflict_review.json"),
            "--input",
            str(dedup_input),
            "--report",
            str(args.report_dir / "final_conflict_proposals.json"),
            "--summary",
            str(args.report_dir / "final_conflict_proposals.md"),
            "--write-proposed-final",
            str(args.report_dir / "training.proposed_final.jsonl"),
        ]
    )

    print("Post-processing complete.")
    print(f"Gold dataset: {args.gold_out}")
    print(f"Repair bucket: {args.repair_out}")
    print(f"Reports: {args.report_dir}")


if __name__ == "__main__":
    main()
