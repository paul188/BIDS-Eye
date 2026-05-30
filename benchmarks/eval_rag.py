"""
RAG resolver benchmark — evaluates MetadataRetriever against ground-truth term→code entries.

Usage:
    python benchmarks/eval_rag.py \
        --yaml RAG/value_mappings.yaml \
        --name-index RAG/name_index.json \
        --ground-truth benchmarks/ground_truth/rag_terms.jsonl \
        --out benchmarks/results/rag_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from RAG.retriever import MetadataRetriever


_FIELD_TO_RETRIEVER_METHOD = {
    "diagnosis": "retrieve_for_field",
    "task": "retrieve_for_field",
    "datatype": "retrieve_for_field",
    "suffix": "retrieve_for_field",
    "name": "retrieve_for_field",
}

_RESOLUTION_PATHS = ("exact", "fuzzy", "embedding_biolord", "embedding_sapbert", "unresolved")


def _call_retriever(retriever: MetadataRetriever, field: str, term: str) -> list[str]:
    try:
        return retriever.retrieve_for_field(field, term, n_results=20) or []
    except Exception:
        return []



def _evaluate_entry(
    retriever: MetadataRetriever,
    entry: dict[str, Any],
) -> dict[str, Any]:
    field = entry["field"]
    term = entry["term"]
    expected = set(entry["expected_codes"])

    start = time.perf_counter()
    returned = _call_retriever(retriever, field, term)
    elapsed_ms = (time.perf_counter() - start) * 1000

    returned_set = set(returned)
    hit_any = bool(expected & returned_set)
    top1_hit = bool(returned) and returned[0] in expected
    recall_at_3 = bool(expected & set(returned[:3]))

    # Resolution path: introspect retriever internals when possible
    path = "unresolved"
    if returned:
        path = _infer_path(retriever, field, term, returned[0])

    return {
        "term": term,
        "field": field,
        "variant_type": entry.get("variant_type", "unknown"),
        "expected_codes": list(expected),
        "returned_codes": returned[:5],
        "pass": hit_any,
        "top1_hit": top1_hit,
        "recall_at_3": recall_at_3,
        "resolution_path": path,
        "latency_ms": round(elapsed_ms, 1),
    }


def _infer_path(
    retriever: MetadataRetriever, field: str, term: str, top_code: str
) -> str:
    """Heuristic to classify which tier resolved the term."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "RAG"))
    from yaml_to_llamaindex import leaf_db, group_db, display_db, group_display_db

    leaves = leaf_db.get(field, {})
    groups = group_db.get(field, {})
    display = display_db.get(field, {})
    group_display = group_display_db.get(field, {})

    term_lower = term.lower()
    candidates = [
        term_lower,
        term_lower.replace(" ", "_"),
        term_lower.replace("-", "_"),
        term_lower.replace("-", " "),
    ]
    for c in candidates:
        if c in leaves or c in groups or c in display or c in group_display:
            return "exact"

    # Use RapidFuzz WRatio to decide if fuzzy could have fired
    try:
        from rapidfuzz import process as rfp, fuzz as rfz
        all_keys = list(display.keys()) + list(group_display.keys())
        best = rfp.extractOne(term_lower, all_keys, scorer=rfz.WRatio, score_cutoff=75) if all_keys else None
        if best:
            return "fuzzy"
    except ImportError:
        pass

    # Check if embedding index was loaded (means embedding fallback fired)
    from RAG.retriever import _emb_indices
    biolord_loaded = _emb_indices.get("FremyCompany/BioLORD-2023-C") is not None
    sapbert_loaded = _emb_indices.get("cambridgeltl/SapBERT-from-PubMedBERT-fulltext") is not None

    if biolord_loaded:
        return "embedding_biolord"
    if sapbert_loaded:
        return "embedding_sapbert"
    return "fuzzy"


def _compute_table(results: list[dict]) -> dict[str, dict]:
    by_field: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_field[r["field"]].append(r)

    table = {}
    for field, rows in sorted(by_field.items()):
        n = len(rows)
        path_counts: dict[str, int] = defaultdict(int)
        for r in rows:
            path_counts[r["resolution_path"]] += 1

        table[field] = {
            "n": n,
            "p_at_1": round(sum(r["top1_hit"] for r in rows) / n, 3),
            "recall_at_3": round(sum(r["recall_at_3"] for r in rows) / n, 3),
            "pass_rate": round(sum(r["pass"] for r in rows) / n, 3),
            "exact_pct": round(path_counts["exact"] / n, 3),
            "fuzzy_pct": round(path_counts["fuzzy"] / n, 3),
            "embed_pct": round(
                (path_counts["embedding_biolord"] + path_counts["embedding_sapbert"]) / n, 3
            ),
            "unresolved_pct": round(path_counts["unresolved"] / n, 3),
        }

    all_n = len(results)
    all_path: dict[str, int] = defaultdict(int)
    for r in results:
        all_path[r["resolution_path"]] += 1

    table["overall"] = {
        "n": all_n,
        "p_at_1": round(sum(r["top1_hit"] for r in results) / all_n, 3),
        "recall_at_3": round(sum(r["recall_at_3"] for r in results) / all_n, 3),
        "pass_rate": round(sum(r["pass"] for r in results) / all_n, 3),
        "exact_pct": round(all_path["exact"] / all_n, 3),
        "fuzzy_pct": round(all_path["fuzzy"] / all_n, 3),
        "embed_pct": round(
            (all_path["embedding_biolord"] + all_path["embedding_sapbert"]) / all_n, 3
        ),
        "unresolved_pct": round(all_path["unresolved"] / all_n, 3),
    }
    return table


def _compute_variant_subtable(results: list[dict]) -> dict[str, dict]:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_variant[r["variant_type"]].append(r)

    table = {}
    for vtype, rows in sorted(by_variant.items()):
        n = len(rows)
        table[vtype] = {
            "n": n,
            "p_at_1": round(sum(r["top1_hit"] for r in rows) / n, 3),
            "pass_rate": round(sum(r["pass"] for r in rows) / n, 3),
        }
    return table


def _print_table(table: dict[str, dict], title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    header = f"{'Field':<16} {'N':>5} {'P@1':>7} {'R@3':>7} {'Pass%':>7} {'Exact%':>7} {'Fuzzy%':>7} {'Embed%':>7} {'Miss%':>7}"
    print(header)
    print("-" * 70)
    for field, m in table.items():
        if field == "overall":
            print("-" * 70)
        embed = m.get("embed_pct", 0)
        print(
            f"{field:<16} {m['n']:>5} "
            f"{m['p_at_1']:>7.1%} "
            f"{m.get('recall_at_3', 0):>7.1%} "
            f"{m['pass_rate']:>7.1%} "
            f"{m.get('exact_pct', 0):>7.1%} "
            f"{m.get('fuzzy_pct', 0):>7.1%} "
            f"{embed:>7.1%} "
            f"{m.get('unresolved_pct', 0):>7.1%}"
        )


def _print_variant_table(table: dict[str, dict]) -> None:
    print(f"\n{'='*50}")
    print("  Synonym robustness (by variant type)")
    print(f"{'='*50}")
    print(f"{'Variant':<16} {'N':>5} {'P@1':>7} {'Pass%':>7}")
    print("-" * 50)
    for vtype, m in table.items():
        print(f"{vtype:<16} {m['n']:>5} {m['p_at_1']:>7.1%} {m['pass_rate']:>7.1%}")


_CONF_BUCKETS = [
    ("0.0–0.5", 0.0, 0.5),
    ("0.5–0.7", 0.5, 0.7),
    ("0.7–0.9", 0.7, 0.9),
    ("0.9–1.0", 0.9, 1.01),
]


def _compute_confidence_calibration(
    results: list[dict], yaml_path: str
) -> dict[str, dict]:
    """
    Correlate synonym confidence weights (from value_mappings.yaml) with P@1.

    For each eval entry, look up the maximum confidence weight assigned to
    any synonym of the expected code in the YAML. Bucket by weight range and
    report P@1 per bucket. A well-calibrated dictionary should show monotonically
    increasing P@1 as confidence increases.
    """
    import yaml

    try:
        with open(yaml_path, encoding="utf-8") as fh:
            mappings = yaml.safe_load(fh)
    except Exception as exc:
        return {"error": str(exc)}

    # Build: {field: {canonical_code: max_confidence}}
    code_conf: dict[str, dict[str, float]] = {}
    for field, concepts in (mappings or {}).items():
        if not isinstance(concepts, dict):
            continue
        code_conf[field] = {}
        for code, concept in concepts.items():
            if not isinstance(concept, dict):
                continue
            synonyms = concept.get("synonyms", [])
            max_conf = 0.0
            for syn in synonyms:
                if isinstance(syn, dict):
                    max_conf = max(max_conf, float(syn.get("confidence", 0.0)))
                elif isinstance(syn, str):
                    max_conf = max(max_conf, 1.0)
            code_conf[field][str(code)] = max_conf

    # Assign each result its confidence weight
    bucket_rows: dict[str, list[dict]] = {b[0]: [] for b in _CONF_BUCKETS}
    unmatched = 0
    for r in results:
        field = r["field"]
        expected = r["expected_codes"]
        # Use the max confidence among all expected codes
        confs = [
            code_conf.get(field, {}).get(c, -1.0)
            for c in expected
        ]
        best_conf = max((c for c in confs if c >= 0), default=None)
        if best_conf is None:
            unmatched += 1
            continue
        for label, lo, hi in _CONF_BUCKETS:
            if lo <= best_conf < hi:
                bucket_rows[label].append(r)
                break

    calibration: dict[str, dict] = {}
    for label, rows in bucket_rows.items():
        if not rows:
            continue
        n = len(rows)
        calibration[label] = {
            "n": n,
            "p_at_1": round(sum(r["top1_hit"] for r in rows) / n, 3),
            "pass_rate": round(sum(r["pass"] for r in rows) / n, 3),
        }
    if unmatched:
        calibration["_unmatched_in_yaml"] = {"n": unmatched}
    return calibration


def _print_calibration_table(calibration: dict[str, dict]) -> None:
    if "error" in calibration:
        print(f"\n  Confidence calibration unavailable: {calibration['error']}")
        return
    print(f"\n{'='*55}")
    print("  Synonym confidence calibration (weight → P@1)")
    print(f"{'='*55}")
    print(f"{'Confidence':<14} {'N':>5} {'P@1':>7} {'Pass%':>7}")
    print("-" * 55)
    for label, m in calibration.items():
        if label.startswith("_"):
            continue
        print(f"{label:<14} {m['n']:>5} {m['p_at_1']:>7.1%} {m['pass_rate']:>7.1%}")
    if "_unmatched_in_yaml" in calibration:
        print(f"  ({calibration['_unmatched_in_yaml']['n']} entries not found in YAML — excluded)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG resolver against ground truth")
    parser.add_argument("--yaml", default="RAG/value_mappings.yaml")
    parser.add_argument("--name-index", default="RAG/name_index.json")
    parser.add_argument("--ground-truth", default="benchmarks/ground_truth/rag_terms.jsonl")
    parser.add_argument("--out", default="benchmarks/results/rag_metrics.json")
    parser.add_argument("--verbose", action="store_true", help="Print per-entry results")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"ERROR: ground truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    entries = [json.loads(line) for line in gt_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(entries)} ground-truth entries from {gt_path}")

    print("Loading MetadataRetriever...")
    retriever = MetadataRetriever(args.name_index)
    print("Retriever loaded.")

    results = []
    failed_entries = []
    for i, entry in enumerate(entries, 1):
        try:
            r = _evaluate_entry(retriever, entry)
            results.append(r)
            if args.verbose:
                status = "PASS" if r["pass"] else "FAIL"
                print(
                    f"[{i:3d}/{len(entries)}] {status} {entry['field']:12} "
                    f"{entry['term']:40} → {r['returned_codes'][:3]}"
                )
        except Exception as exc:
            failed_entries.append({"entry": entry, "error": str(exc)})
            print(f"  ERROR on {entry}: {exc}", file=sys.stderr)

    if not results:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    table = _compute_table(results)
    variant_table = _compute_variant_subtable(results)
    calibration = _compute_confidence_calibration(results, args.yaml)

    _print_table(table, "RAG Resolution Performance by Field")
    _print_variant_table(variant_table)
    _print_calibration_table(calibration)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "summary": table,
                "variant_breakdown": variant_table,
                "calibration": calibration,
                "per_entry": results,
                "failed": failed_entries,
                "n_entries": len(entries),
                "n_evaluated": len(results),
            },
            indent=2,
        )
    )
    print(f"\nDetailed results written to {out_path}")

    overall = table.get("overall", {})
    p1 = overall.get("p_at_1", 0)
    miss = overall.get("unresolved_pct", 0)
    print(f"\nOverall P@1: {p1:.1%}  |  Unresolved: {miss:.1%}")
    if p1 < 0.80:
        print("WARNING: Overall P@1 below 0.80 target.")
        sys.exit(1)


if __name__ == "__main__":
    main()
