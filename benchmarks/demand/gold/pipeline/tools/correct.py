"""Append reviewed labels: python3 tools/correct.py <query-prefix> <label> "<reason>" acc[,acc...]
A later line for the same (query, acc) replaces an earlier one (assemble.py reads in order)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lib, spec
qp, label, reason, accs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
label = {"rel": "relevant", "part": "partially_relevant", "not": "not_relevant", "uns": "unsure"}.get(label, label)
assert label in ("relevant", "partially_relevant", "not_relevant", "unsure"), label
qids = [q["id"] for q in spec.Q if q["id"].startswith(qp)]
assert len(qids) == 1, qids
D, ON = lib.load()
with open(f"{lib.G}/corrections.jsonl", "a") as f:
    for a in accs:
        assert a in D or a in ON, a
        f.write(json.dumps({"query": qids[0], "acc": a, "label": label, "reason": reason}, ensure_ascii=False) + "\n")
print(qids[0], label, len(accs))
