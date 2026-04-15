#!/usr/bin/env python3
"""
modal_app/build_few_shot.py
---------------------------
Curate high-quality few-shot examples from training_data_generation/results/
and write them to modal_app/few_shot_examples.json.

Selection strategy per query family:
  1. De-duplicate paraphrase bundles (keep one representative per bundle)
  2. Round-robin across distinct patterns so no single SQL structure dominates
  3. Prefer shorter SQL (cleaner, fits in context)
  4. Keep up to MAX_PER_FAMILY examples

Re-run whenever new training data is generated:
    python modal_app/build_few_shot.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[1]
RESULTS_DIR  = REPO_ROOT / "training_data_generation" / "results"
OUT_FILE     = Path(__file__).parent / "few_shot_examples.json"

MAX_PER_FAMILY = 10   # stored; 5 are picked at inference time


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_all_examples() -> List[dict]:
    examples = []
    for f in sorted(RESULTS_DIR.glob("result_*.txt")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                examples.extend(data)
        except Exception as exc:
            print(f"  [WARN] {f.name}: {exc}", file=sys.stderr)
    return examples


def _sql_is_valid(sql: str) -> bool:
    """Reject examples with obviously wrong SQL."""
    s = sql.strip().upper()
    if not s.startswith("SELECT"):
        return False
    required = ["BIDS_DATASETS", "GROUP BY"]
    return all(kw in s for kw in required)


def select_for_family(examples: List[dict]) -> List[dict]:
    """
    From a list of examples for one family, pick up to MAX_PER_FAMILY
    diverse, non-duplicate representatives.
    """
    # 1. Drop examples with invalid SQL
    valid = [e for e in examples if _sql_is_valid(e.get("sql", ""))]

    # 2. De-duplicate paraphrase bundles: keep shortest SQL per bundle
    seen_bundles: dict[str, dict] = {}
    unbundled: List[dict] = []
    for e in valid:
        bid = e.get("paraphrase_bundle_id")
        if bid is None:
            unbundled.append(e)
        else:
            if bid not in seen_bundles or len(e["sql"]) < len(seen_bundles[bid]["sql"]):
                seen_bundles[bid] = e
    deduped = unbundled + list(seen_bundles.values())
    # Sort ascending by SQL length: shorter = cleaner
    deduped.sort(key=lambda e: len(e["sql"]))

    # 3. Group by pattern for round-robin diversity
    by_pattern: Dict[str, List[dict]] = defaultdict(list)
    for e in deduped:
        by_pattern[e.get("pattern", "unknown")].append(e)

    # 4. Round-robin pick
    selected: List[dict] = []
    pattern_iters = [iter(v) for v in by_pattern.values()]
    while len(selected) < MAX_PER_FAMILY and pattern_iters:
        exhausted = []
        for it in pattern_iters:
            if len(selected) >= MAX_PER_FAMILY:
                break
            try:
                selected.append(next(it))
            except StopIteration:
                exhausted.append(it)
        for it in exhausted:
            pattern_iters.remove(it)

    return [
        {"question": e["question"], "sql": e["sql"], "pattern": e.get("pattern", "")}
        for e in selected
    ]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading examples from {RESULTS_DIR} ...")
    all_examples = load_all_examples()
    print(f"  {len(all_examples)} total examples loaded")

    # Group by family
    by_family: Dict[str, List[dict]] = defaultdict(list)
    for e in all_examples:
        fam = e.get("family", "unknown")
        by_family[fam].append(e)

    print(f"  {len(by_family)} distinct families\n")

    few_shot: Dict[str, List[dict]] = {}
    for family in sorted(by_family):
        selected = select_for_family(by_family[family])
        few_shot[family] = selected
        patterns = {e["pattern"] for e in selected}
        print(f"  {family:35s}  {len(selected):2d} examples  "
              f"({len(by_family[family])} total)  patterns: {len(patterns)}")

    OUT_FILE.write_text(json.dumps(few_shot, indent=2))
    total = sum(len(v) for v in few_shot.values())
    print(f"\nWrote {total} examples across {len(few_shot)} families → {OUT_FILE}")


if __name__ == "__main__":
    main()
