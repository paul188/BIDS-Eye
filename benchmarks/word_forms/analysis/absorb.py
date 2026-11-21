"""For each labelled direction A->B: datasets that say B but not A (what stemming
adds to a search for A). How many of them use B only inside a fixed phrase
(auto-mined bigrams, not tuned on the labels)?"""
import json, sys
from collections import Counter

S = sys.argv[1]
docs = json.load(open(f"{S}/docs.json"))
lab = json.load(open(f"{S}/raw_labels.json"))
pairs = {p["id"]: p for p in json.load(open(sys.argv[2]))["pairs"]}
big = json.load(open(f"{S}/bigrams.json"))
F = {tuple(r[0].split()) for r in big if r[1] / r[2] >= 0.7 and r[5] > 1.0}
D = {"both": (1, 1), "a_to_b": (1, 0), "b_to_a": (0, 1), "neither": (0, 0)}

sets = [(ds, toks, set(toks)) for ds, toks in docs.items()]

def added(A, B):
    A, B = set(A), set(B); n = absorbed = 0; phr = Counter()
    for ds, toks, s in sets:
        if s & A or not (s & B): continue
        n += 1
        inside = True; used = []
        for i, t in enumerate(toks):
            if t in B:
                l = (toks[i-1], t) if i else None
                r = (t, toks[i+1]) if i + 1 < len(toks) else None
                if l in F: used.append(" ".join(l))
                elif r in F: used.append(" ".join(r))
                else: inside = False; break
        if inside: absorbed += 1; phr.update(set(used))
    return n, absorbed, phr

tot = {"want_no": [0, 0], "want_yes": [0, 0]}
detail = []
for pid, v in lab.items():
    w = D.get(v["label"])
    if not w: continue
    p = pairs[pid]
    for (src, dst), want in zip([("a", "b"), ("b", "a")], w):
        n, ab, phr = added(p[src], p[dst])
        key = "want_yes" if want else "want_no"
        tot[key][0] += n; tot[key][1] += ab
        if not want and n:
            detail.append((n, ab, pid, "/".join(p[src][:2]), "/".join(p[dst][:2]), phr.most_common(2)))
for k, (n, ab) in tot.items():
    print(f"{k:9} datasets stemming adds {n:5}; only via a fixed phrase {ab:5} ({100*ab/max(n,1):.0f}%)")
print("\nlargest unwanted additions (search A -> datasets with only B):")
for n, ab, pid, a, b, phr in sorted(detail, reverse=True)[:30]:
    print(f"  {pid} {a:22} -> {b:24} adds {n:4}, via phrase {ab:4}  {phr}")
