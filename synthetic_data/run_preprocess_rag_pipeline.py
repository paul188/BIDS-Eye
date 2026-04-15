#!/usr/bin/env python3
"""
run_preprocess_rag_pipeline.py
-------------------------------
Apply the LLM_preprocessor → RAG pipeline to questions sampled from
training_data_generation/results/ and write enriched records to
training_data_generation/results_RAG/.

Each output record contains the original fields (question, sql, pattern,
family) plus:
  query_plan        — structured QueryPlan from the LLM preprocessor
  rag_resolved      — {field: {term: [canonical_codes]}} from the RAG
  augmented_question — question with resolved codes injected as context

Authentication:
  export GEMINI_API_KEY=...   or   export GOOGLE_API_KEY=...

Requires:
  pip install google-genai pydantic sentence-transformers numpy

Usage:
  # Build the metadata index first (needs DB access):
  python modal_app/build_metadata_index.py --db-url <url> --out /tmp/metadata_index.json

  # Run the pipeline (default: 50 questions):
  python training_data_generation/run_preprocess_rag_pipeline.py \\
      --index /tmp/metadata_index.json

  # Without RAG (preprocessor only):
  python training_data_generation/run_preprocess_rag_pipeline.py --no-rag

  # Custom sample size / output dir:
  python training_data_generation/run_preprocess_rag_pipeline.py \\
      --index /tmp/metadata_index.json --n 100 --out-dir results_RAG
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR  = Path(__file__).parent / "results"
DEFAULT_OUT  = Path(__file__).parent / "results_RAG"
DEFAULT_INDEX = REPO / "LLM_preprocessor" / "metadata_index.json"

sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# Sample questions from result files
# ---------------------------------------------------------------------------

def load_questions(results_dir: Path, n: int, seed: int = 42) -> list[dict]:
    """
    Load up to `n` unique (question, sql, pattern, family) records from
    the result JSON files, sampled evenly across all files.
    """
    all_records: list[dict] = []
    for path in sorted(results_dir.glob("result_*.txt")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for rec in data:
                    if rec.get("question") and rec.get("sql"):
                        all_records.append({
                            "question": rec["question"],
                            "sql":      rec["sql"],
                            "pattern":  rec.get("pattern", ""),
                            "family":   rec.get("family", ""),
                            "source_file": path.name,
                        })
        except Exception as exc:
            print(f"  [WARN] could not read {path.name}: {exc}", file=sys.stderr)

    if not all_records:
        raise SystemExit(f"No records found in {results_dir}")

    # Deduplicate by question text
    seen: set[str] = set()
    unique = []
    for rec in all_records:
        if rec["question"] not in seen:
            seen.add(rec["question"])
            unique.append(rec)

    rng = random.Random(seed)
    sample = rng.sample(unique, min(n, len(unique)))
    print(f"Sampled {len(sample)} questions from {len(unique)} unique across "
          f"{len(list(results_dir.glob('result_*.txt')))} files.")
    return sample


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline_on_batch(
    records: list[dict],
    retriever,          # LocalMetadataRetriever or None
    api_key: str,
    out_dir: Path,
    delay: float = 2.0, # seconds between Gemini calls
) -> None:
    from LLM_preprocessor.preprocess import preprocess_query, build_rag_requests, augment_with_rag

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    errors: list[dict] = []

    for i, rec in enumerate(records, 1):
        question = rec["question"]
        print(f"\n[{i}/{len(records)}] {question[:90]}{'…' if len(question)>90 else ''}")

        try:
            # 1. Preprocessor
            plan = preprocess_query(question, api_key=api_key)
            print(f"  family={plan.query_family.value}  "
                  f"groups={len(plan.groups)}  "
                  f"scan_reqs={len(plan.scan_requirements)}  "
                  f"rag_reqs={len(plan.rag_requests)}")

            # 2. RAG
            rag_resolved: dict = {}
            augmented_question = plan.natural_language_summary

            if retriever is not None:
                rag_requests = build_rag_requests(plan)
                raw_rag: dict[str, dict[str, list[str]]] = {}

                for req in rag_requests:
                    if req.field.value == "scan":
                        # Multi-field lookup: datatype + suffix + task
                        if hasattr(retriever, "retrieve_scan_terms"):
                            scan_resolved = retriever.retrieve_scan_terms(req.terms)
                        else:
                            scan_resolved = {}
                            for field in ("datatype", "suffix", "task"):
                                if hasattr(retriever, "retrieve_for_field_multi"):
                                    tm = retriever.retrieve_for_field_multi(field, req.terms)
                                else:
                                    cat_key = field + "s"
                                    tm = {}
                                    for term in req.terms:
                                        hints = retriever.retrieve(f"{field}: {term}")
                                        codes = hints.get(cat_key, [])[:10]
                                        if codes:
                                            tm[term] = codes
                                if tm:
                                    scan_resolved[field] = tm
                        for field, tm in scan_resolved.items():
                            if tm:
                                raw_rag.setdefault(field, {}).update(tm)
                    else:
                        # Single-field lookup
                        if hasattr(retriever, "retrieve_for_field_multi"):
                            term_map = retriever.retrieve_for_field_multi(
                                req.field.value, req.terms,
                            )
                        else:
                            cat_key = req.field.value + "s"
                            term_map = {}
                            for term in req.terms:
                                hints = retriever.retrieve(f"{req.field.value}: {term}")
                                codes = hints.get(cat_key, [])[:10]
                                if codes:
                                    term_map[term] = codes
                        if term_map:
                            raw_rag[req.field.value] = term_map

                augmented_plan = augment_with_rag(plan, raw_rag)
                rag_resolved = raw_rag
                augmented_question = augmented_plan.augmented_question

                resolved_summary = {
                    f: list({c for codes in tmap.values() for c in codes})
                    for f, tmap in raw_rag.items()
                }
                print(f"  RAG resolved: {resolved_summary}")

            # 3. Build output record
            out_record = {
                **rec,  # question, sql, pattern, family, source_file
                "query_plan":          plan.model_dump(),
                "rag_resolved":        rag_resolved,
                "augmented_question":  augmented_question,
            }
            results.append(out_record)

        except Exception as exc:
            print(f"  [ERROR] {exc}", file=sys.stderr)
            errors.append({**rec, "error": str(exc)})

        # Respect Gemini rate limits
        if i < len(records):
            time.sleep(delay)

    # Write outputs
    out_path = out_dir / "pipeline_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(results)} records to {out_path}")

    if errors:
        err_path = out_dir / "pipeline_errors.json"
        err_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(errors)} errors to {err_path}")

    # Per-family breakdown
    from collections import Counter
    families = Counter(r["query_plan"]["query_family"] for r in results)
    print("\nQuery families in output:")
    for fam, count in families.most_common():
        print(f"  {count:3d}  {fam}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--index", metavar="PATH",
        default=str(DEFAULT_INDEX),
        help=f"Path to metadata_index.json (default: {DEFAULT_INDEX}). "
             "Build it first with build_metadata_index_job.sh. "
             "Ignored when --no-rag is set.",
    )
    parser.add_argument(
        "--no-rag", action="store_true",
        help="Skip the RAG step (preprocessor only).",
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of questions to process (default: 50).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help=f"Source results directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds to wait between Gemini calls (default: 2.0).",
    )
    args = parser.parse_args()

    # API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running.")

    # RAG retriever
    retriever = None
    if not args.no_rag:
        if not Path(args.index).exists():
            raise SystemExit(
                f"Metadata index not found: {args.index}\n"
                "Build it first:\n"
                "  sbatch training_data_generation/build_metadata_index_job.sh\n"
                "Or skip RAG with --no-rag."
            )
        from LLM_preprocessor.rag import LocalMetadataRetriever
        print(f"Loading metadata index from {args.index} …")
        retriever = LocalMetadataRetriever(args.index)
        print("Index loaded.")

    # Sample questions
    records = load_questions(args.results_dir, args.n, seed=args.seed)

    # Run pipeline
    run_pipeline_on_batch(
        records,
        retriever=retriever,
        api_key=api_key,
        out_dir=args.out_dir,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
