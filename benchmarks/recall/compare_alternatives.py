"""Compare the variants of a paired_alternatives_run against the judged set.

    python3 compare_alternatives.py <run dir> <ground_truth.json> <recall dir> [--table | --held-out]

Scores the development questions; --table adds one line per question with
alternatives. With --held-out it scores the questions in
held_out_questions.txt instead, averages only; see the rules there.
"""
import json
import random
import sys
from pathlib import Path

run_dir, truth_path, recall_dir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, recall_dir)
from eval_recall import load_held_out, ndcg  # noqa: E402

truth = json.loads(truth_path.read_text())["queries"]
held = load_held_out()
runs = {p.stem: {r["id"]: r for r in json.loads(p.read_text())["results"]}
        for p in sorted(run_dir.glob("alt_*.json"))}
order = ["alt_off"] + sorted((v for v in runs if v.startswith("alt_cap")),
                             key=lambda v: int(v[7:])) \
    + sorted((v for v in runs if v.startswith("alt_ratio")), key=lambda v: int(v[9:])) \
    + [v for v in ("alt_flat", "alt_on") if v in runs]


def metrics(row, t):
    relevant, partial = set(t["relevant"]), set(t.get("partially_relevant", []))
    judged = relevant | partial | set(t.get("not_relevant", [])) | set(t.get("unsure", []))
    returned = row["returned"]
    levels = row.get("levels") or {}
    confirmed = [a for a in returned if levels.get(a, 0) == 0]
    judged_returned = [a for a in returned if a in judged]
    return {
        "n": len(returned),
        "n_confirmed": len(confirmed),
        "recall": len(set(returned) & relevant) / len(relevant),
        "r20": len(set(returned[:20]) & relevant) / min(len(relevant), 20),
        "ndcg10": ndcg(returned, relevant, partial, 10),
        "ndcg20": ndcg(returned, relevant, partial, 20),
        "p20": (len(set(returned[:20]) & relevant) / len(returned[:20])) if returned else 0.0,
        "precision": (len(set(judged_returned) & relevant) / len(judged_returned)) if judged_returned else None,
        "unjudged20": len([a for a in returned[:20] if a not in judged]),
    }


def boot(diffs, n=10000, seed=7):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


on = runs["alt_on"]
held_out = "--held-out" in sys.argv
ids = [q for q, t in truth.items() if q in on and (q in held) == held_out and t["relevant"]
       and not on[q].get("error") and t.get("method") != "structural"]
affected = [q for q in ids if on[q].get("alternatives")]
errors = [q for q in on if on[q].get("error")]
print(f"scorable {'held-out' if held_out else 'development'} questions: {len(ids)}; with alternatives: {len(affected)}; "
      f"errors/fallbacks: {len(errors)}")
m = {v: {q: metrics(runs[v][q], truth[q]) for q in ids} for v in order}

for name, subset in (("QUESTIONS WITH ALTERNATIVES", affected), ("ALL SCORABLE QUESTIONS", ids)):
    print(f"\n{name} ({len(subset)})")
    print(f"{'variant':10} {'rows':>6} {'R@20':>6} {'nDCG@10':>8} {'nDCG@20':>8} {'P@20':>6} {'recall':>7} {'prec':>6} {'unjdg@20':>8}")
    for v in order:
        def avg(key):
            vals = [m[v][q][key] for q in subset if m[v][q][key] is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        print(f"{v:10} {avg('n'):6.0f} {avg('r20'):6.3f} {avg('ndcg10'):8.3f} {avg('ndcg20'):8.3f} "
              f"{avg('p20'):6.3f} {avg('recall'):7.3f} {avg('precision'):6.3f} {avg('unjudged20'):8.2f}")
    if name.startswith("QUESTIONS"):
        pairs = [(v, "alt_off") for v in order[1:]] + ([("alt_on", "alt_flat")] if "alt_flat" in runs else []) \
            + [(v, "alt_on") for v in order if v.startswith("alt_ratio")]
        for v, base in pairs:
            print(f"  {v} vs {base}:")
            for key in ("r20", "ndcg10", "ndcg20"):
                diffs = [m[v][q][key] - m[base][q][key] for q in subset]
                lo, hi = boot(diffs)
                print(f"    {key:7} diff {sum(diffs) / len(diffs):+.3f} (95% bootstrap {lo:+.3f} to {hi:+.3f}); "
                      f"better {sum(d > 1e-9 for d in diffs)}, worse {sum(d < -1e-9 for d in diffs)}, "
                      f"same {sum(abs(d) <= 1e-9 for d in diffs)}")

if "--table" in sys.argv and not held_out:
    print("\nper question (nDCG@20 / rows):")
    for q in affected:
        alts = on[q]["alternatives"]
        cells = "  ".join(f"{v[4:]}={m[v][q]['ndcg20']:.2f}/{m[v][q]['n']}" for v in order)
        print(f"{q:28} {cells}  own={on[q].get('own')} alts={alts}")
