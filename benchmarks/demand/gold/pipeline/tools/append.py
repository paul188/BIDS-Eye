"""Append judgments to a batch's output file. Reads JSON lines from stdin.

    python3 tools/append.py BATCH <<'JSONL'
    {"query": "qf04_eeg_autism", "acc": "ds000001", "label": "not_relevant", "reason": "..."}
    JSONL

Each line is checked (the pair belongs to the batch, the values are valid) and
written at once. A pair that is already judged is skipped, never overwritten.
"""
import json, sys, os, fcntl
from common import G, LABELS, POPS, YNU, load_batch, done_pairs
name = sys.argv[1]
b, need = load_batch(name)
need_set = set(need)
done = done_pairs(name)
is_pop = "questions" not in b
ok = skipped = 0
errors = []
os.makedirs(f"{G}/judged", exist_ok=True)
with open(f"{G}/judged/{name}.jsonl", "a") as out:
    fcntl.flock(out, fcntl.LOCK_EX)
    for i, line in enumerate(sys.stdin, 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception as e:
            errors.append(f"line {i}: not JSON ({e})"); continue
        key = ("pop" if is_pop else r.get("query"), r.get("acc"))
        if key not in need_set:
            errors.append(f"line {i}: {key} is not a pair of this batch"); continue
        if key in done:
            skipped += 1; continue
        if is_pop:
            bad = [k for k, allowed in (("population", POPS), ("healthy_adults_5", YNU), ("healthy_children_5", YNU), ("includes_men", YNU)) if r.get(k) not in allowed]
            if bad or not (r.get("n_healthy") is None or isinstance(r.get("n_healthy"), int)):
                errors.append(f"line {i}: invalid value for {bad or 'n_healthy'}"); continue
        elif r.get("label") not in LABELS:
            errors.append(f"line {i}: label must be one of {LABELS}"); continue
        if not str(r.get("reason", "")).strip():
            errors.append(f"line {i}: reason missing"); continue
        out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush(); os.fsync(out.fileno())
        done[key] = r; ok += 1
left = len(need) - len([p for p in need if p in done])
print(f"appended {ok}, skipped {skipped} already judged, {len(errors)} rejected; {left} of {len(need)} still to judge")
for e in errors:
    print("  REJECTED", e)
