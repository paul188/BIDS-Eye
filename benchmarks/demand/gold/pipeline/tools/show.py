"""Print draft judgments compactly: python3 tools/show.py <query-prefix> [labels] [acc,acc,...]"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lib
D, ON = lib.load()
qp = sys.argv[1]
labels = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "-" else None
accs = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
SHORT = {"relevant": "REL", "partially_relevant": "PART", "not_relevant": "NOT", "unsure": "UNS"}
for l in open(f"{lib.G}/judgments_draft.jsonl"):
    r = json.loads(l)
    if not r["query"].startswith(qp): continue
    if labels and SHORT[r["label"]] not in labels: continue
    if accs and r["acc"] not in accs: continue
    name = (D.get(r["acc"]) or {}).get("name") or ((ON.get(r["acc"]) or {}).get("name") or "")
    print(f'{r["query"][:4]} {r["acc"]} {SHORT[r["label"]]:4} {r["source"][:1]} | {name[:60]} | {r["reason"][:170]}')
