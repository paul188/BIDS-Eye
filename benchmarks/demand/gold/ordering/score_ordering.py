#!/usr/bin/env python3
"""Score a run of the ordering gold requests (truth.json).

Recall over the whole answer set would punish a correct top 10 ("10 of 469
found"), so these requests are scored on their first N results only
(N = the limit, or the first page of 20 without one):

  top      share of the first N that may stand there: the N best by the sort
           key, ties included. Without a sort key, any correct answer counts.
  order    of the pairs among the first N that are correct answers, the share
           in the requested order (ties count as in order). Only with a sort key.
  filter   share of the first N that are correct answers at all
  extra    results after the limit; listed, not penalised (a limit without a
           sort key keeps the relevance order and is not cut, by design)
  new      returned datasets the gold snapshot never judged (crawled since);
           left out of the shares

    python3 benchmarks/recall/eval_recall.py run --queries benchmarks/demand/gold/ordering/requests.yaml \\
        --split all --out <run.json> ...
    python3 benchmarks/demand/gold/ordering/score_ordering.py <run.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent


def pair_order(ranked: list[str], values: dict, descending: bool) -> Optional[float]:
    """Share of pairs (i before j) whose values are in the requested order."""
    known = [values[a] for a in ranked if values.get(a) is not None]
    pairs = [(x, y) for i, x in enumerate(known) for y in known[i + 1:]]
    if not pairs:
        return None
    good = sum((x >= y) if descending else (x <= y) for x, y in pairs)
    return good / len(pairs)


def score_one(truth: dict, returned: list[str]) -> dict:
    members, judged = set(truth["members"]), set(truth["judged"])
    n = truth["n"]
    first = [a for a in returned[:n] if a in judged]
    new = [a for a in returned[:n] if a not in judged]
    row = {"returned": len(returned), "new": new,
           "extra": max(0, len(returned) - truth["limit"]) if truth.get("limit") else 0}
    wanted = min(n, len(members))
    row["filter"] = sum(a in members for a in first) / len(first) if first else None
    if truth.get("sort_key"):
        top = set(truth["acceptable_top"])
        row["top"] = sum(a in top for a in first) / wanted if wanted else None
        row["order"] = pair_order([a for a in first if a in members], truth["values"],
                                  truth["direction"] == "DESC")
    else:
        row["top"] = sum(a in members for a in first) / wanted if wanted else None
        row["order"] = None
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--truth", type=Path, default=HERE / "truth.json")
    args = parser.parse_args()
    truth = json.loads(args.truth.read_text())["queries"]
    run = {r["id"]: r for r in json.loads(args.run.read_text())["results"]}

    def fmt(x: Optional[float]) -> str:
        return "   —" if x is None else f"{x:4.0%}"

    print(f"{'query':36} {'n':>3} {'ret':>5} {'top':>5} {'order':>5} {'filter':>6} {'extra':>5} {'new':>4}  note")
    rows = []
    for qid, t in truth.items():
        r = run.get(qid)
        if r is None:
            continue
        row = score_one(t, r.get("returned") or [])
        rows.append(row)
        print(f"{qid:36} {t['n']:3d} {row['returned']:5d} {fmt(row['top']):>5} {fmt(row['order']):>5} "
              f"{fmt(row['filter']):>6} {row['extra']:5d} {len(row['new']):4d}  {(r.get('error') or '')[:40]}")
    for key in ("top", "order", "filter"):
        vals = [row[key] for row in rows if row[key] is not None]
        if vals:
            print(f"mean {key}: {sum(vals) / len(vals):.0%} over {len(vals)} requests")
    missing = sorted(set(truth) - set(run))
    if missing:
        print(f"not in the run: {', '.join(missing)}")


if __name__ == "__main__":
    main()
