"""Compare repeated runs of two search versions on the real requests.

    python3 benchmarks/demand/compare_runs.py --base run1.json run2.json --cand runA.json runB.json

The search is not repeatable (the same version gave different results for 19
of 69 requests in two runs), so one run per version proves nothing. Rules:
  per request   a change is real only if every candidate value lies outside the
                range of the baseline values, all on the same side
  on average    each request's mean over the candidate runs minus its mean over
                the baseline runs; mean of these differences with a 95% bootstrap
                interval over requests
  empty pages   share of (request, run) pairs that returned nothing although the
                catalogue holds a correct answer
"""
import argparse, json, random, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def metrics(run_path, truth):
    run = {r["id"]: r for r in json.loads(Path(run_path).read_text())["results"]}
    out = {}
    for qid, t in truth.items():
        r = run.get(qid)
        if r is None:
            continue
        rel, ret = set(t["relevant"]), r.get("returned") or []
        hits = len(rel & set(ret))
        out[qid] = {
            "returned": len(ret),
            "recall": hits / len(rel) if rel else None,
            "precision": hits / len(ret) if ret else None,
            "recall20": len(rel & set(ret[:20])) / min(len(rel), 20) if rel else None,
            "empty": bool(rel) and not ret,
            "relaxed": bool(r.get("relaxed")),
        }
    return out


def interval(diffs, rounds=4000, seed=7):
    rnd = random.Random(seed)
    means = sorted(statistics.mean(rnd.choices(diffs, k=len(diffs))) for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", nargs="+", required=True)
    p.add_argument("--cand", nargs="+", required=True)
    p.add_argument("--truth", default=str(HERE / "gold" / "requests_truth.json"))
    args = p.parse_args()
    truth = json.loads(Path(args.truth).read_text())["queries"]
    base = [metrics(x, truth) for x in args.base]
    cand = [metrics(x, truth) for x in args.cand]
    ids = [q for q in truth if all(q in m for m in base + cand)]

    for name, runs in (("baseline", base), ("candidate", cand)):
        pairs = [m[q] for m in runs for q in ids if truth[q]["relevant"]]
        print(f"{name:10} runs={len(runs)}  empty pages {sum(x['empty'] for x in pairs)}/{len(pairs)} "
              f"({100 * sum(x['empty'] for x in pairs) / len(pairs):.0f}%)  relaxed {sum(x['relaxed'] for x in pairs)}")
    print()
    for key in ("recall", "recall20", "precision"):
        diffs, b_all, c_all = [], [], []
        for q in ids:
            b = [m[q][key] for m in base if m[q][key] is not None]
            c = [m[q][key] for m in cand if m[q][key] is not None]
            if key == "precision":             # an empty page has no precision: count it as 0 when answers exist
                b = [m[q][key] if m[q][key] is not None else 0.0 for m in base] if truth[q]["relevant"] else b
                c = [m[q][key] if m[q][key] is not None else 0.0 for m in cand] if truth[q]["relevant"] else c
            if b and c:
                diffs.append(statistics.mean(c) - statistics.mean(b)); b_all.append(statistics.mean(b)); c_all.append(statistics.mean(c))
        lo, hi = interval(diffs)
        print(f"{key:10} baseline {100 * statistics.mean(b_all):5.1f}%  candidate {100 * statistics.mean(c_all):5.1f}%  "
              f"difference {100 * statistics.mean(diffs):+5.1f} points (95% interval {100 * lo:+.1f} to {100 * hi:+.1f}), n={len(diffs)}")
    print("\nper request, recall: real changes only (all candidate runs outside the baseline range, same side)")
    better, worse = [], []
    for q in ids:
        b = [m[q]["recall"] for m in base]; c = [m[q]["recall"] for m in cand]
        if None in b or None in c:
            continue
        line = (f"  {q[:38]:38} baseline {' '.join(f'{100 * v:3.0f}%' for v in b)} ({' '.join(str(m[q]['returned']) for m in base)} returned)"
                f"  candidate {' '.join(f'{100 * v:3.0f}%' for v in c)} ({' '.join(str(m[q]['returned']) for m in cand)} returned)")
        if min(c) > max(b): better.append(line)
        elif max(c) < min(b): worse.append(line)
    print(f" better: {len(better)}"); print("\n".join(better))
    print(f" worse: {len(worse)}"); print("\n".join(worse))
    print("\nper request, results returned: real changes where recall did not change")
    for q in ids:
        b = [m[q]["returned"] for m in base]; c = [m[q]["returned"] for m in cand]
        br = [m[q]["recall"] for m in base]; cr = [m[q]["recall"] for m in cand]
        if None in br or None in cr or min(cr) > max(br) or max(cr) < min(br):
            continue
        if min(c) > 1.5 * max(b) + 5 or max(c) * 1.5 + 5 < min(b):
            print(f"  {q[:38]:38} baseline {b}  candidate {c}  precision "
                  f"{[round(100 * (m[q]['precision'] or 0)) for m in base]} -> {[round(100 * (m[q]['precision'] or 0)) for m in cand]}")


if __name__ == "__main__":
    main()
