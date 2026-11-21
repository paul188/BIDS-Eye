"""For each labelled form: datasets it occurs in, and the neighbour words that
account for most of them (a form that lives inside one fixed phrase)."""
import json, sys
from collections import Counter, defaultdict

S = sys.argv[1]
docs = json.load(open(f"{S}/docs.json"))
lab = json.load(open(f"{S}/raw_labels.json"))
pairs = {p["id"]: p for p in json.load(open(sys.argv[2]))["pairs"]}
STOP = set("a an and as at by for from in of on or the to with is are was were be been this that these those it its we our".split())

forms = set()
for pid in lab:
    forms.update(pairs[pid]["a"]); forms.update(pairs[pid]["b"])

df = Counter(); left = defaultdict(Counter); right = defaultdict(Counter)
for ds, toks in docs.items():
    seen = set(); L = defaultdict(set); R = defaultdict(set)
    for i, t in enumerate(toks):
        if t in forms:
            seen.add(t)
            j = i - 1
            if j >= 0: L[t].add(toks[j])
            if i + 1 < len(toks): R[t].add(toks[i + 1])
    for t in seen:
        df[t] += 1
        for w in L[t]: left[t][w] += 1
        for w in R[t]: right[t][w] += 1

res = {}
for f in sorted(forms):
    n = df[f]
    if not n: continue
    top = [(f"{w} {f}", c) for w, c in left[f].most_common(6) if w not in STOP][:3] + \
          [(f"{f} {w}", c) for w, c in right[f].most_common(6) if w not in STOP][:3]
    top.sort(key=lambda x: -x[1])
    res[f] = {"df": n, "top": top[:4]}
json.dump(res, open(f"{S}/neigh.json", "w"), indent=0)
for f, v in sorted(res.items(), key=lambda kv: -kv[1]["df"]):
    print(f"{f:18} {v['df']:4}  " + "; ".join(f"{p} {c}" for p, c in v["top"]))
