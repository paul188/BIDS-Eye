"""How are the words of a term arranged when both occur in one sentence?

For every sentence that has all words of a term, the instance is put in one class:
  adjacent      the words stand in order, joined by space or hyphen
  coord         modifier and head are linked in the parse, with a conjunction/comma/slash between ("temporal and frontal cortex")
  stacked       linked in the parse, other modifiers between ("temporal association cortex")
  head_coord    modifier belongs to a noun the head is coordinated with ("temporal lobe and cortex")
  postmod       the head comes first and governs the modifier ("cortex of the temporal lobe")
  unlinked_near not linked in the parse, at most 3 words apart
  unlinked_far  not linked, further apart
Terms of three or more words: adjacent / loose (in order within n+3 words) / apart.
"""
import gzip, json, random, sys
from collections import Counter, defaultdict
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from termlist import terms
MOD = {"amod", "compound", "nmod", "npadvmod", "nummod", "poss", "advmod"}
def matches(tok, forms):
    low = tok[0].lower()
    return tok[1] in forms or low in forms or (low.endswith("s") and low[:-1] in forms) or (low.endswith("'s") and low[:-2] in forms)
def governor(toks, i):
    """Index of the noun token i modifies (through conjunctions), and whether a conjunction was passed."""
    via = False
    for _ in range(6):
        t = toks[i]
        if t[3] == "conj": i = t[4]; via = True; continue
        if t[3] in MOD: return t[4], via
        return None, via
    return None, via
def conj_family(toks, i):
    fam = {i}
    for _ in range(4):
        fam |= {k for k, t in enumerate(toks) if t[3] == "conj" and t[4] in fam}
        fam |= {toks[k][4] for k in list(fam) if toks[k][3] == "conj"}
    return fam
def ancestors(toks, i, n=5):
    out = []
    for _ in range(n):
        h = toks[i][4]
        if h == i: break
        out.append(h); i = h
    return out
def classify2(toks, wpos, ms, hs):
    best = None
    order = ["adjacent", "coord", "stacked", "head_coord", "postmod", "unlinked_near", "unlinked_far"]
    for m in ms:
        for h in hs:
            if m == h: continue
            gap = wpos[h] - wpos[m]
            between = toks[m + 1:h] if h > m else []
            if gap == 1 and all(t[0] in "-‐‑–" for t in between): cls = "adjacent"
            else:
                g, via = governor(toks, m)
                if g == h and h > m:
                    cls = "coord" if via or any(t[2] == "CCONJ" or t[0] in ",/" for t in between) else "stacked"
                elif g is not None and h in conj_family(toks, g) and h != g: cls = "head_coord"
                elif h in ancestors(toks, m) and h < m: cls = "postmod"
                else: cls = "unlinked_near" if abs(gap) <= 3 else "unlinked_far"
            cand = (order.index(cls), abs(gap), cls, gap, m, h)
            if best is None or cand < best: best = cand
    return best[2], best[3], best[4], best[5]
def main(path, out):
    ts = terms()
    first = defaultdict(list)
    for k, (kind, name, words) in enumerate(ts): first[words[0][0]].append(k)
    allforms = {f for _, _, words in ts for w in words for f in w}
    counts = defaultdict(Counter); gaps = defaultdict(Counter); samples = defaultdict(lambda: defaultdict(list)); seen = Counter()
    papers = defaultdict(lambda: defaultdict(set)); n = 0
    rnd = random.Random(3)
    for line in gzip.open(path, "rt"):
        paper, part, toks = json.loads(line); n += 1
        present = defaultdict(list)
        for i, t in enumerate(toks):
            low = t[0].lower()
            for f in (t[1], low, low[:-1] if low.endswith("s") else low):
                if f in allforms: present[f].append(i)
        if len(present) < 2: continue
        wpos, c = [], 0
        for t in toks:
            wpos.append(c)
            if t[2] not in ("PUNCT", "SYM", "SPACE"): c += 1
        for k, (kind, name, words) in enumerate(ts):
            idx = [sorted({i for f in w for i in present.get(f, ())}) for w in words]
            if not all(idx): continue
            if len(words) == 2:
                cls, gap, m, h = classify2(toks, wpos, idx[0], idx[1])
                gaps[name][(max(-8, min(8, gap)), cls.startswith("unlinked"))] += 1
                lo, hi = min(m, h), max(m, h)
            else:
                cls, lo, hi = "apart", idx[0][0], idx[-1][-1]
                for a in idx[0]:
                    pos, ok, tight = a, True, True
                    for w in idx[1:]:
                        nxt = [j for j in w if j > pos]
                        if not nxt: ok = False; break
                        if wpos[nxt[0]] - wpos[pos] != 1: tight = False
                        pos = nxt[0]
                    if ok and tight: cls, lo, hi = "adjacent", a, pos; break
                    if ok and wpos[pos] - wpos[a] <= len(words) + 2 and cls != "adjacent": cls, lo, hi = "loose", a, pos
            counts[name][cls] += 1; papers[name][cls].add(paper); seen[(name, cls)] += 1
            if cls != "adjacent":
                s = samples[name][cls]
                snip = "".join(t[0] + (" " if t[5] else "") for t in toks[max(0, lo - 9):hi + 8]).strip()
                if len(s) < 12: s.append(snip)
                elif rnd.random() < 12 / seen[(name, cls)]: s[rnd.randrange(12)] = snip
    json.dump({"sentences": n, "terms": [{"kind": kind, "term": name, "counts": counts[name],
               "papers": {c: len(p) for c, p in papers[name].items()},
               "docs_adjacent": len(papers[name].get("adjacent", ())),
               "docs_only_coord_stacked": len((papers[name].get("coord", set()) | papers[name].get("stacked", set())) - papers[name].get("adjacent", set())),
               "docs_only_near": len((papers[name].get("unlinked_near", set()) | papers[name].get("postmod", set()) | papers[name].get("loose", set())) - papers[name].get("adjacent", set()) - papers[name].get("coord", set()) - papers[name].get("stacked", set())),
               "docs_only_far": len((papers[name].get("unlinked_far", set()) | papers[name].get("head_coord", set()) | papers[name].get("apart", set())) - set().union(*[papers[name].get(c, set()) for c in ("adjacent", "coord", "stacked", "unlinked_near", "postmod", "loose")])),
               "gaps": [[g, int(u), v] for (g, u), v in sorted(gaps[name].items())],
               "samples": samples[name]} for kind, name, _ in ts]}, open(out, "w"), indent=1)
    print("sentences", n)
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
