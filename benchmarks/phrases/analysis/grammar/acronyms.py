"""Abbreviations defined in the papers ("working memory (WM)") and how often each form is used afterwards (Schwartz-Hearst matching)."""
import json, re, sys
from collections import Counter, defaultdict
DEF = re.compile(r"((?:[A-Za-z0-9][\w'’./-]*[ ‐-]){1,10})\(\s*([A-Za-z][A-Za-z0-9-]{1,9}?)s?\s*[);,]")
def long_form(before, short):
    """Schwartz & Hearst 2003: shortest long form whose characters cover the short form, first letter starting a word."""
    s, l = len(short) - 1, len(before) - 1
    while s >= 0:
        c = short[s].lower()
        if not c.isalnum(): s -= 1; continue
        while l >= 0 and (before[l].lower() != c or (s == 0 and l > 0 and before[l - 1].isalnum())): l -= 1
        if l < 0: return None
        s -= 1; l -= 1
    lf = before[l + 1:].strip()
    lf = lf[lf.find(" ") + 1:] if l + 1 > 0 and before[l].isalnum() else lf
    words = re.split(r"[ ‐-]+", lf)
    if not (2 <= len(words) <= min(len(short) + 5, len(short) * 2)): return None
    if short.lower() in (w.lower() for w in words): return None
    return lf
def norm(lf): return re.sub(r"[\s‐-]+", " ", lf.lower()).rstrip("s")
def main(src, out):
    papers = defaultdict(list)
    for line in open(src):
        d = json.loads(line); papers[d["paper"]].append(d["text"])
    longs = defaultdict(Counter); use_long = Counter(); use_short = Counter(); npapers = Counter(); once = Counter()
    for pid, paras in papers.items():
        text = "\n".join(paras); defs = {}
        for m in DEF.finditer(text):
            short = m.group(2)
            if sum(ch.isupper() for ch in short) < 2 and not short.isupper(): continue
            lf = long_form(m.group(1), short)
            if lf and short not in defs: defs[short] = lf
        for short, lf in defs.items():
            pat = r"\b" + r"[\s‐-]+".join(re.escape(w) for w in re.split(r"[ ‐-]+", lf)) + r"(?:s|es)?\b"
            nl = len(re.findall(pat, text, flags=re.I)); ns = len(re.findall(r"(?<![\w-])" + re.escape(short) + r"s?(?![\w])", text))
            key = (short, norm(lf))
            longs[short][norm(lf)] += 1; use_long[key] += nl; use_short[key] += max(ns - 1, 0); npapers[key] += 1
            if nl <= 1: once[key] += 1
    rows = [{"short": k[0], "long": k[1], "papers": npapers[k], "long_uses": use_long[k], "short_uses": use_short[k],
             "papers_long_only_at_definition": once[k]} for k in npapers]
    rows.sort(key=lambda r: -r["papers"])
    amb = {s: c.most_common(6) for s, c in longs.items() if len(c) > 1 and sum(c.values()) >= 6}
    json.dump({"papers": len(papers), "pairs": rows, "ambiguous": amb}, open(out, "w"), indent=1)
    tl, tsh = sum(use_long.values()), sum(use_short.values())
    print("papers", len(papers), "defined pairs", len(rows), "paper-definitions", sum(npapers.values()))
    print("uses of long form", tl, "uses of short form", tsh, "short share", round(tsh / (tl + tsh), 3))
    print("definitions where the long form appears only once in the paper:", sum(once.values()), "of", sum(npapers.values()))
    for r in rows[:45]: print(r["papers"], r["short"], "|", r["long"], "| long", r["long_uses"], "short", r["short_uses"])
    print("\nAMBIGUOUS")
    for s, c in sorted(amb.items(), key=lambda kv: -sum(v for _, v in kv[1]))[:40]: print(s, c)
if __name__ == "__main__": main(sys.argv[1], sys.argv[2])
