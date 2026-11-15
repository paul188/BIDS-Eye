"""Datasets a run returned that no judge or review read, per judged request (top N of the result list).

    python3 tools/unjudged.py <run.json> [N=20]
They count as not relevant in requests_truth.json; judge them before trusting precision.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lib, spec
D, ON = lib.load()
run = {r["id"]: r for r in json.load(open(sys.argv[1]))["results"]}
top = int(sys.argv[2]) if len(sys.argv) > 2 else 20
read = {}
for l in open(f"{lib.G}/judgments_draft.jsonl"):
    r = json.loads(l)
    read.setdefault(r["query"], set()).add(r["acc"])
total = 0
for q in spec.Q:
    if q["method"] not in ("mixed", "judged") or q["id"] not in run:
        continue
    ret = run[q["id"]].get("returned") or []
    un = [a for a in ret[:top] if a not in read.get(q["id"], set())]
    total += len(un)
    if un:
        print(f"{q['id'][:40]:40} returned {len(ret):4}  unread in top {top}: {' '.join(un)}")
print("unread in total:", total)
