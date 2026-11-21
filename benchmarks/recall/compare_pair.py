"""Compare the two orderings of a paired text-rank run against the judged set.

    python3 compare_pair.py rank_on.json rank_off.json ground_truth.json [--held-out]

Scores the development questions. With --held-out it scores the questions in
held_out_questions.txt instead and prints averages only; see the rules there.
"""
from __future__ import annotations

import json
import random
import sys

from eval_recall import load_held_out, ndcg

PAGE = 20


def _rank_text(rank):
    """The rank of a result file as text: a RankQuery dict, or the old "stemmed#exact" string."""
    if isinstance(rank, dict):
        return " # ".join(f"{name}: {query}" for name, query in rank.items() if query) or None
    return rank


def metrics(returned, truth):
    relevant = set(truth["relevant"])
    partial = set(truth.get("partially_relevant", []))
    top = returned[:PAGE]
    return {
        "r20": len(set(top) & relevant) / min(len(relevant), PAGE),
        "ndcg20": ndcg(returned, relevant, partial, depth=PAGE),
    }


def boot(diffs, n=10000, seed=7):
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n)
    )
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> None:
    held_out = "--held-out" in sys.argv[1:]
    paths = [a for a in sys.argv[1:] if a != "--held-out"]
    on = {r["id"]: r for r in json.load(open(paths[0]))["results"]}
    off = {r["id"]: r for r in json.load(open(paths[1]))["results"]}
    truth = json.load(open(paths[2]))["queries"]
    held_ids = load_held_out()

    rows = []
    for qid, t in truth.items():
        a, b = on.get(qid), off.get(qid)
        if a is None or b is None or not t["relevant"] or (qid in held_ids) != held_out:
            continue
        m_on, m_off = metrics(a["returned"], t), metrics(b["returned"], t)
        rows.append({
            "id": qid,
            "total": a.get("total"),
            "tsq": _rank_text(a.get("rank_query") or a.get("rank_tsquery")),
            "fallback": a.get("fallback"),
            "error": a.get("error"),
            "same_set": set(a["returned"]) == set(b["returned"]),
            "order_changed": a["returned"][:PAGE] != b["returned"][:PAGE],
            "on": m_on, "off": m_off,
        })

    print("HELD-OUT SET, averages only" if held_out else "DEVELOPMENT SET")
    print(f"{'query':30} {'total':>5} {'R@20 off':>8} {'R@20 on':>8} {'nDCG off':>8} {'nDCG on':>8}  rank query")
    for r in ([] if held_out else rows):
        note = "FALLBACK" if r["fallback"] else (r["error"] or (r["tsq"] or "—"))
        print(f"{r['id']:30} {str(r['total']):>5} {r['off']['r20']:8.0%} {r['on']['r20']:8.0%} "
              f"{r['off']['ndcg20']:8.2f} {r['on']['ndcg20']:8.2f}  {str(note)[:60]}")

    def summary(name, subset):
        if not subset:
            print(f"\n{name}: no questions")
            return
        print(f"\n{name}: {len(subset)} questions")
        for key, label in (("r20", "recall@20"), ("ndcg20", "nDCG@20")):
            off_mean = sum(r["off"][key] for r in subset) / len(subset)
            on_mean = sum(r["on"][key] for r in subset) / len(subset)
            diffs = [r["on"][key] - r["off"][key] for r in subset]
            lo, hi = boot(diffs)
            better = sum(d > 1e-9 for d in diffs)
            worse = sum(d < -1e-9 for d in diffs)
            print(f"  {label:10} off {off_mean:.3f}  on {on_mean:.3f}  diff {on_mean - off_mean:+.3f} "
                  f"(95% bootstrap {lo:+.3f} to {hi:+.3f}); better {better}, worse {worse}, same {len(diffs) - better - worse}")

    usable = [r for r in rows if not r["fallback"] and not r["error"]]
    summary("All scorable questions", usable)
    summary("Questions with a rank query", [r for r in usable if r["tsq"]])
    summary("Questions with a rank query and more than 20 results",
            [r for r in usable if r["tsq"] and (r["total"] or 0) > PAGE])
    print(f"\nsame result set in both orderings: {sum(r['same_set'] for r in rows)} of {len(rows)}; "
          f"first page changed: {sum(r['order_changed'] for r in rows)}; "
          f"fallback: {sum(bool(r['fallback']) for r in rows)}; errors: {sum(bool(r['error']) for r in rows)}")


if __name__ == "__main__":
    main()
