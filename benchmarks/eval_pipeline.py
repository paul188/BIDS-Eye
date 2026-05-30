"""
End-to-end pipeline benchmark for BIDS-Eye.

Reuses QUERIES from RAG/run_query_eval.py (no duplication).
For each query: POST /api/query, score against expected_min, must_contain, must_not.

Usage:
    python benchmarks/eval_pipeline.py --api-url http://localhost:8000 \
        --out benchmarks/results/pipeline_metrics.json --rate-limit 8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

_KNOWN_TABLES = {"bids_datasets", "bids_objects", "bids_participants"}

def _check_schema_hallucination(sql: str) -> bool:
    """Return True if SQL references a table name outside the known BIDS schema."""
    sql_lower = sql.lower()
    # Strip quoted identifiers to avoid false positives from string values
    sql_stripped = re.sub(r"'[^']*'", "''", sql_lower)
    table_refs = re.findall(r'\bfrom\s+(\w+)|\bjoin\s+(\w+)', sql_stripped)
    for grp in table_refs:
        tbl = next((t for t in grp if t), None)
        if tbl and tbl not in _KNOWN_TABLES:
            return True
    return False

sys.path.insert(0, str(Path(__file__).parent.parent))

from RAG.run_query_eval import QUERIES, evaluate as _score_query


_CATEGORY_PATTERNS = [
    ("Participant count",  re.compile(r"\b(more than|at least|over|fewer than|≥|≤|\d+\s+participant|\d+\s+subject)", re.I)),
    ("Combined",          re.compile(r"\b(and|with both|combined|simultaneous)\b.*\b(fmri|eeg|mri|meg|mdd|adhd|autism|epilepsy)", re.I)),
    ("Developmental",     re.compile(r"\b(pediatric|adolescent|elderly|aging|children|female|male|older adult|young adult)", re.I)),
    ("Modality",          re.compile(r"\b(fmri|eeg|meg|mri|dti|dwi|ieeg|fnirs|pet|asl|spectroscopy|t1|structural|functional|diffusion|near.infrared)\b", re.I)),
    ("Diagnosis",         re.compile(r"\b(autism|adhd|schizophreni|psychosis|depression|depressed|mdd|bipolar|parkinson|alzheimer|epilepsy|seizure|dyslexia|ptsd|ocd|anxiety|fibromyalgia|tbi|stroke|mci|mci|als|healthy control|typically developing)\b", re.I)),
    ("Task",              re.compile(r"\b(resting.state|rest|n.back|nback|working memory|attention|face|emotion|motor|task|imagery|movie|music|theory of mind|reward|language|reading|navigation|pain|sleep|oddball|bci|gambling)\b", re.I)),
    ("Edge case",         re.compile(r".*", re.I)),  # catch-all
]


def _categorize(q: str) -> str:
    for cat, pattern in _CATEGORY_PATTERNS:
        if pattern.search(q):
            return cat
    return "Other"


def _post_query(api_url: str, question: str, timeout: int = 90) -> dict:
    data = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/query",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def _evaluate_one(spec: dict, response: dict) -> dict[str, Any]:
    issues = _score_query(spec, response)

    translation_block = response.get("translation") or {}
    sql = translation_block.get("sql", "") or ""
    result_count = len(response.get("datasets", []))
    has_error = "error" in response

    zero_result = "ZERO_RESULTS" in " ".join(issues)
    spurious_filter = any("FORBIDDEN_IN_SQL" in i for i in issues)
    missing_filter = any("MISSING_IN_SQL" in i for i in issues)
    fallback_triggered = any("POSSIBLE_FALLBACK" in i for i in issues)
    sql_valid = bool(sql) and not has_error
    schema_hallucination = bool(sql) and _check_schema_hallucination(sql)
    self_corrected = bool(translation_block.get("self_corrected", False))
    passed = not issues

    return {
        "question": spec["q"],
        "category": _categorize(spec["q"]),
        "result_count": result_count,
        "sql_snippet": sql[:200] if sql else "",
        "pass": passed,
        "issues": issues,
        "zero_result": zero_result,
        "spurious_filter": spurious_filter,
        "missing_filter": missing_filter,
        "fallback_triggered": fallback_triggered,
        "sql_valid": sql_valid,
        "schema_hallucination": schema_hallucination,
        "self_corrected": self_corrected,
    }


def _cat_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "pass_pct": round(sum(r["pass"] for r in rows) / n, 3),
        "zero_result_pct": round(sum(r["zero_result"] for r in rows) / n, 3),
        "spurious_filter_pct": round(sum(r["spurious_filter"] for r in rows) / n, 3),
        "missing_filter_pct": round(sum(r["missing_filter"] for r in rows) / n, 3),
        "sql_valid_pct": round(sum(r["sql_valid"] for r in rows) / n, 3),
        "fallback_pct": round(sum(r["fallback_triggered"] for r in rows) / n, 3),
        "schema_hallucination_pct": round(sum(r["schema_hallucination"] for r in rows) / n, 3),
        "self_correction_pct": round(sum(r["self_corrected"] for r in rows) / n, 3),
    }


def _compute_table(results: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    table = {}
    order = ["Participant count", "Diagnosis", "Task", "Modality", "Combined", "Developmental", "Edge case", "Other"]
    for cat in order:
        rows = by_cat.get(cat)
        if rows:
            table[cat] = _cat_metrics(rows)
    for cat, rows in by_cat.items():
        if cat not in table:
            table[cat] = _cat_metrics(rows)

    table["Overall"] = _cat_metrics(results)
    return table


def _print_table(table: dict[str, dict]) -> None:
    W = 106
    print(f"\n{'='*W}")
    print("  Pipeline Performance by Category")
    print(f"{'='*W}")
    header = (
        f"{'Category':<22} {'N':>4} {'Pass%':>7} {'Zero%':>7} "
        f"{'Spur%':>7} {'Miss%':>7} {'SQL✓%':>7} {'Fall%':>7} {'Halluc%':>8} {'SelfFix%':>9}"
    )
    print(header)
    print("-" * W)
    for cat, m in table.items():
        if cat == "Overall":
            print("-" * W)
        print(
            f"{cat:<22} {m['n']:>4} "
            f"{m['pass_pct']:>7.1%} "
            f"{m['zero_result_pct']:>7.1%} "
            f"{m['spurious_filter_pct']:>7.1%} "
            f"{m['missing_filter_pct']:>7.1%} "
            f"{m['sql_valid_pct']:>7.1%} "
            f"{m['fallback_pct']:>7.1%} "
            f"{m['schema_hallucination_pct']:>8.1%} "
            f"{m['self_correction_pct']:>9.1%}"
        )
    print()
    print("  Pass%=no issues  Zero%=0 results when data expected  Spur%=forbidden SQL pattern")
    print("  Miss%=expected SQL fragment absent  SQL✓%=SQL generated  Fall%=unfiltered result set")
    print("  Halluc%=SQL references unknown table  SelfFix%=Gemini self-corrected a failed query")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end BIDS-Eye pipeline benchmark")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--out", default="benchmarks/results/pipeline_metrics.json")
    parser.add_argument("--rate-limit", type=float, default=8.0,
                        help="Seconds between requests (avoid overloading the API)")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Limit number of queries (0 = all)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-failed", action="store_true",
                        help="Continue even if API is unreachable")
    args = parser.parse_args()

    queries = QUERIES
    if args.max_queries > 0:
        queries = queries[: args.max_queries]

    print(f"Running {len(queries)} queries against {args.api_url}")
    print(f"Rate limit: {args.rate_limit}s between requests\n")

    # Quick connectivity check
    probe = _post_query(args.api_url, "test", timeout=5)
    if "error" in probe and not args.skip_failed:
        print(f"ERROR: API unreachable at {args.api_url}: {probe['error']}", file=sys.stderr)
        print("Start the backend with:  docker-compose up  or  uvicorn backend.main:app", file=sys.stderr)
        sys.exit(1)

    results = []
    for i, spec in enumerate(queries, 1):
        start = time.perf_counter()
        response = _post_query(args.api_url, spec["q"])
        elapsed = time.perf_counter() - start

        result = _evaluate_one(spec, response)
        result["latency_s"] = round(elapsed, 2)
        results.append(result)

        status = "PASS" if result["pass"] else f"FAIL({', '.join(result['issues'][:2])})"
        n_datasets = result["result_count"]
        print(f"[{i:3d}/{len(queries)}] {status:<40} {n_datasets:>5} results  {elapsed:.1f}s  {spec['q'][:60]}")

        if args.verbose and not result["pass"]:
            for issue in result["issues"]:
                print(f"         ↳ {issue}")
            if result["sql_snippet"]:
                print(f"         SQL: {result['sql_snippet'][:120]}")

        if i < len(queries):
            time.sleep(args.rate_limit)

    table = _compute_table(results)
    _print_table(table)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "summary": table,
                "per_query": results,
                "n_queries": len(results),
                "api_url": args.api_url,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )
    print(f"\nDetailed results written to {out_path}")

    overall = table.get("Overall", {})
    pass_pct = overall.get("pass_pct", 0)
    print(f"Overall Pass%: {pass_pct:.1%}")
    if pass_pct < 0.85:
        print("WARNING: Overall Pass% below 0.85 target.")
        sys.exit(1)


if __name__ == "__main__":
    main()
