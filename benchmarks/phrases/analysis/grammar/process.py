"""Verb/noun rewordings: two word stems that occur within 5 words of each other, by order, gap and form."""
import gzip, json, random, sys
from collections import Counter, defaultdict
PAIRS = """motion correct; slice tim; spatial smooth; spatial normali; skull strip; brain extract; distortion correct; bias correct;
nuisance regress; band filter; high filter; function connect; eye track; source locali; source reconstruct; motor imag; fear condition;
sleep depriv; finger tap; movie watch; decision mak|made; emotion regulat; face perce; speech perce; response inhibit; mental rotat;
eye clos; eye open; reinforc learn; memor encod; memor retriev; data acqui; head mov; stimul present; word read; picture nam;
brain stimulat; reward anticipat; language comprehen; sentence comprehen; imag regist; cortical thick|thin; brain activ; pain stimul;
music listen; story listen; video view|watch; image view; rest scan; task switch; attention shift; error monitor; object recogni;
face recogni; emotion recogni; word generat; verb generat; risk tak|took; reward learn; motor learn; sequence learn; language learn""".replace("\n", " ")
def main(path, out):
    pairs = [(a.split("|"), b.split("|")) for a, b in (p.split() for p in PAIRS.split(";") if p.strip())]
    cls = defaultdict(Counter); forms = defaultdict(Counter); samples = defaultdict(lambda: defaultdict(list)); rnd = random.Random(5); seen = Counter()
    for line in gzip.open(path, "rt"):
        paper, part, toks = json.loads(line)
        low = [t[0].lower() for t in toks]
        wpos, c = [], 0
        for t in toks:
            wpos.append(c)
            if t[2] not in ("PUNCT", "SYM", "SPACE"): c += 1
        for a, b in pairs:
            ia = [i for i, w in enumerate(low) if any(w.startswith(s) for s in a)]
            if not ia: continue
            ib = [i for i, w in enumerate(low) if any(w.startswith(s) for s in b)]
            if not ib: continue
            name = a[0] + "+" + b[0]; used = set()
            for i in ia:
                cand = [(abs(wpos[j] - wpos[i]), j) for j in ib if j != i and j not in used and abs(wpos[j] - wpos[i]) <= 5]
                if not cand: continue
                d, j = min(cand); used.add(j); gap = wpos[j] - wpos[i]
                bt = toks[j]; nominal = bt[2] in ("NOUN", "PROPN") or low[j].endswith(("ion", "ions", "ing", "ment", "ity", "ery", "ness", "al"))
                if gap == 1: k = "AB adjacent, nominal" if nominal else "AB adjacent, verbal"
                elif gap > 1: k = "AB with words between"
                else: k = "BA (reversed)"
                cls[name][k] += 1; forms[name][(k, low[i], low[j])] += 1; seen[(name, k)] += 1
                if k != "AB adjacent, nominal":
                    lo, hi = min(i, j), max(i, j); s = samples[name][k]
                    snip = "".join(t[0] + (" " if t[5] else "") for t in toks[max(0, lo - 6):hi + 6]).strip()
                    if len(s) < 8: s.append(snip)
                    elif rnd.random() < 8 / seen[(name, k)]: s[rnd.randrange(8)] = snip
    json.dump({n: {"classes": cls[n], "forms": [[k, a, b, v] for (k, a, b), v in forms[n].most_common(25)], "samples": samples[n]} for n in cls},
              open(out, "w"), indent=1)
    tot = Counter()
    for n in cls: tot.update(cls[n])
    print(tot)
if __name__ == "__main__": main(sys.argv[1], sys.argv[2])
