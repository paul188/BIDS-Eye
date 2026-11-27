"""Term-independent: how do premodifiers attach to their noun in this text? (distance, stacking, coordination, 'of' phrases)"""
import gzip, json, sys
from collections import Counter
def main(path):
    dist = Counter(); kind = Counter(); nmods = Counter(); ofp = Counter(); n = 0
    for line in gzip.open(path, "rt"):
        paper, part, toks = json.loads(line); n += 1
        wpos, c = [], 0
        for t in toks:
            wpos.append(c)
            if t[2] not in ("PUNCT", "SYM", "SPACE"): c += 1
        kids = {}
        for i, t in enumerate(toks):
            if t[4] != i: kids.setdefault(t[4], []).append(i)
        for h, t in enumerate(toks):
            if t[2] not in ("NOUN", "PROPN"): continue
            direct = [m for m in kids.get(h, []) if m < h and toks[m][3] in ("amod", "compound", "nmod") and toks[m][2] in ("ADJ", "NOUN", "PROPN", "VERB")]
            if not direct:
                nmods[0] += 1
            else:
                mods = []
                for m in direct:
                    mods.append((m, False)); stack = [m]
                    while stack:
                        x = stack.pop()
                        for k in kids.get(x, []):
                            if toks[k][3] == "conj" and k < h and toks[k][2] in ("ADJ", "NOUN", "PROPN", "VERB"): mods.append((k, True)); stack.append(k)
                nmods[min(len(mods), 5)] += 1
                coordinated = any(v for _, v in mods)
                for m, via in mods:
                    g = wpos[h] - wpos[m]; pos = "ADJ" if toks[m][2] in ("ADJ", "VERB") else "NOUN"
                    dist[(pos, min(g, 6))] += 1
                    between = toks[m + 1:h]
                    if g == 1: kind[(pos, "adjacent")] += 1
                    elif coordinated and any(b[2] == "CCONJ" or b[0] in ",/" for b in between): kind[(pos, "list before the noun")] += 1
                    else: kind[(pos, "other modifier between")] += 1
            if any(toks[k][3] == "prep" and toks[k][0].lower() == "of" for k in kids.get(h, [])): ofp["noun + of-phrase"] += 1
            ofp["nouns"] += 1
    print("sentences", n)
    tot = sum(v for k, v in nmods.items())
    print("premodifiers per noun:", {k: round(v / tot, 3) for k, v in sorted(nmods.items())})
    for pos in ("ADJ", "NOUN"):
        t = sum(v for (p, _), v in kind.items() if p == pos)
        print(pos, "modifier links", t, {k: round(v / t, 3) for (p, k), v in kind.items() if p == pos},
              "| by distance", {g: round(v / t, 3) for (p, g), v in sorted(dist.items()) if p == pos})
    print("nouns followed by an of-phrase:", round(ofp["noun + of-phrase"] / ofp["nouns"], 3))
if __name__ == "__main__": main(sys.argv[1])
