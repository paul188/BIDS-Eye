"""Score a run of the demand questions against the datasets the papers used.

    python3 benchmarks/demand/score_known_item.py run.json [--held-out] [--misses 30]

For a question written from a reuse paper the known answer is the dataset the
authors chose. Reported per group:
  found     the search returned the known dataset at all (for a paper that
            pooled several: the share of them it returned)
  top 20    it is on the first page
  top 5     it is among the first five
  MRR       mean of 1 / rank of the first known dataset (0 when missing)
  returned  median number of datasets returned
A dataset the authors did not use may fit the need just as well, so this
measures "would this search have led them to the data they used", not
precision.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--queries", type=Path, default=HERE / "demand_queries.yaml")
    parser.add_argument("--held-out", action="store_true", help="score the held-out questions (averages only)")
    parser.add_argument("--misses", type=int, default=0, help="list this many development questions that missed")
    args = parser.parse_args()

    queries = {q["id"]: q for q in yaml.safe_load(args.queries.read_text())["queries"]}
    data = json.loads(args.run.read_text())
    print("run:", (data.get("meta") or {}).get("label"), "|", (data.get("meta") or {}).get("started"))
    rows = []
    for r in data["results"]:
        q = queries.get(r["id"])
        if q is None or not q.get("known_items") or r["id"].startswith("h") != args.held_out:
            continue
        returned = r.get("returned") or []
        known = set(q["known_items"])
        ranks = [i for i, a in enumerate(returned, 1) if a in known]
        rows.append({
            "q": q, "error": r.get("error"), "n_returned": len(returned),
            "found": len(known & set(returned)) / len(known),
            "top20": float(bool(ranks) and ranks[0] <= 20), "top5": float(bool(ranks) and ranks[0] <= 5),
            "rr": 1 / ranks[0] if ranks else 0.0, "rank": ranks[0] if ranks else None,
        })
    if not rows:
        raise SystemExit("no scored questions in this run for the chosen split")

    def line(label: str, subset: list) -> None:
        if not subset:
            return
        mean = lambda k: sum(x[k] for x in subset) / len(subset)
        print(f"{label:34} {len(subset):5d} {mean('found'):6.0%} {mean('top20'):7.0%} {mean('top5'):6.0%} "
              f"{mean('rr'):6.2f} {statistics.median(x['n_returned'] for x in subset):9.0f} "
              f"{sum(1 for x in subset if x['n_returned'] == 0) / len(subset):6.0%}")

    print(f"\n{'HELD-OUT' if args.held_out else 'DEVELOPMENT'} questions with a known answer: {len(rows)}\n")
    print(f"{'group':34} {'n':>5} {'found':>6} {'top 20':>7} {'top 5':>6} {'MRR':>6} {'returned':>9} {'empty':>6}")
    line("all", rows)
    for form in ("short", "need"):
        line(f"form: {form}", [x for x in rows if x["q"]["form"] == form])
    line("one known dataset", [x for x in rows if len(x["q"]["known_items"]) == 1])
    line("several known datasets", [x for x in rows if len(x["q"]["known_items"]) > 1])
    line("writer confidence high", [x for x in rows if x["q"].get("confidence") == "high"])
    line("writer confidence low", [x for x in rows if x["q"].get("confidence") != "high"])
    # the catalogue records none of the known datasets with the recording type asked for
    line("modality not in catalogue", [x for x in rows if x["q"].get("modality_not_in_catalogue")])
    line("all without those", [x for x in rows if not x["q"].get("modality_not_in_catalogue")])
    line("paper not linked in catalogue", [x for x in rows if not x["q"].get("paper_linked_in_catalogue")])
    line("paper linked in catalogue", [x for x in rows if x["q"].get("paper_linked_in_catalogue")])
    by_criterion = defaultdict(list)
    for x in rows:
        for c in x["q"].get("criteria", []):
            by_criterion[c].append(x)
    print()
    for c, subset in sorted(by_criterion.items(), key=lambda kv: -len(kv[1])):
        if len(subset) >= 5:
            line(f"criterion: {c}", subset)
    # every dataset counts once, however many papers reused it
    per_dataset = defaultdict(list)
    for x in rows:
        for a in x["q"]["known_items"]:
            per_dataset[a].append(x["found"])
    macro = sum(sum(v) / len(v) for v in per_dataset.values()) / len(per_dataset)
    print(f"\nfound, each of the {len(per_dataset)} datasets counted once: {macro:.0%}")
    errors = [x for x in rows if x["error"]]
    if errors:
        print(f"{len(errors)} question(s) raised an error")
    if args.misses and not args.held_out:
        print("\nmissed (development only):")
        for x in [x for x in rows if x["found"] == 0][:args.misses]:
            print(f"  {x['q']['id']:30} returned {x['n_returned']:4d}  wanted {','.join(x['q']['known_items'])}  | {x['q']['question'][:110]}")


if __name__ == "__main__":
    main()
