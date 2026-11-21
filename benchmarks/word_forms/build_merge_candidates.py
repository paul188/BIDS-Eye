"""
benchmarks/word_forms/build_merge_candidates.py
-----------------------------------------------
Words that mean the same but that PostgreSQL's stemmer keeps apart:
"stimulus"/"stimuli", "behaviour"/"behavior", "children"/"child". A search for
one form misses every dataset that writes only the other.

    DATABASE_URL=postgresql://... python3 benchmarks/word_forms/build_merge_candidates.py \
        [--parsed DS.parsed.jsonl.gz ...]

Candidates come from
  - the parser's dictionary forms (spaCy lemmas in parse.py output): younger -> young,
    mice -> mouse, analyses -> analysis, began -> begin;
  - Latin and Greek plural endings: -us/-i, -um/-a, -on/-a, -is/-es, -ex/-ices, -a/-ae, -ma/-mata;
  - British and American spellings: -our/-or, -re/-er, -ise/-ize, -yse/-yze, ae/e, oe/e, -ll-/-l-,
    -ogue/-og, -ence/-ense, grey/gray, programme/program;
  - a short list of irregular English plurals;
  - noun/adjective endings: epilepsy/epileptic, cortex/cortical, hippocampus/hippocampal,
    anxiety/anxious, dopamine/dopaminergic.
A pair is kept when both forms occur in the catalogue and their stems differ.
Pairs are joined into families; each family lists every catalogue form of its
stems, with dataset counts and an example per form.

Writes merge_candidates.json and merge_review.html (from merge_template.html)
next to this file. The database is read in a read-only transaction.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent

TEXT_SQL = ("translate(coalesce(name,'') || ' ' || coalesce(dx_status,'') || ' ' || coalesce(study_domain,'') || ' ' || "
            "coalesce(study_design,'') || ' ' || coalesce(description_text,'') || ' ' || coalesce(llm_conditions,'') || ' ' || "
            "coalesce(llm_profile,'') || ' ' || coalesce(readme_text,''), '/', ' ')")

IRREGULAR = [("man", "men"), ("woman", "women"), ("child", "children"), ("mouse", "mice"), ("foot", "feet"),
             ("tooth", "teeth"), ("person", "people"), ("goose", "geese"), ("louse", "lice"), ("ox", "oxen"),
             ("die", "dice"), ("leaf", "leaves"), ("half", "halves"), ("knife", "knives"), ("life", "lives"),
             ("wife", "wives"), ("self", "selves"), ("calf", "calves"), ("wolf", "wolves")]

# (singular ending, plural ending)
LATIN = [("us", "i"), ("um", "a"), ("on", "a"), ("is", "es"), ("ex", "ices"), ("ix", "ices"), ("a", "ae"),
         ("ma", "mata"), ("en", "ina"), ("x", "ces")]

# (British pattern, American replacement); regexes over the whole word
SPELLING = [
    (r"(?<=[a-z]{3})our(s|ed|ing|al|ally|ite|ites|less)?$", r"or\1"),
    (r"(?<=[^aeiou])(t|b|c)re(s|d)?$", r"\1er\2"),
    (r"is(e|es|ed|ing|er|ers|ation|ations|ational)$", r"iz\1"),
    (r"ys(e|es|ed|ing|er|ers)$", r"yz\1"),
    # ae/oe only in the medical stems that have them: haem-, paed-, oedema, oestrogen, foetal ...
    (r"(h|an|p|isch|leuk|gyn|orthop|f|arch|pal|glyc|septic)ae(?=[mdscot])", r"\1e"),
    (r"^aet", "et"),
    (r"^oe(?=[a-z])", "e"),
    (r"foet", "fet"),
    (r"oea$", "ea"),
    (r"(?<=[eao])ll(ed|ing|er|ers)$", r"l\1"),
    (r"ogue(s)?$", r"og\1"),
    (r"ence(s)?$", r"ense\1"),
    (r"^grey", "gray"),
    (r"programme(s)?$", r"program\1"),
]

# (noun ending, adjective ending): the condition or structure and its adjective
DERIVATION = [("sy", "tic"), ("ia", "ic"), ("ia", "iac"), ("ism", "istic"), ("ism", "ic"), ("sis", "tic"),
              ("iety", "ious"), ("y", "ic"), ("um", "ar"), ("um", "al"), ("us", "al"), ("us", "ic"),
              ("a", "ar"), ("ex", "ical"), ("on", "al"), ("on", "onal"), ("sion", "sual"), ("ma", "matic"),
              ("ine", "inergic"), ("in", "ergic"), ("ol", "olic"), ("es", "etic"), ("ice", "ical"), ("le", "lar"),
              ("ia", "ial")]
# Not used: -a/-al (signa/signal, genera/general), -e/-al, -e/-ic, -ion/-ive: far more false pairs than real ones.
MIN_FORM_DATASETS = 2

SKIP_WORDS = {"data"}       # datum/data: nobody writes datum


def run_sql(sql: str) -> str:
    url = os.environ["DATABASE_URL"]
    return subprocess.run(["psql", url, "-X", "-At", "-v", "ON_ERROR_STOP=1", "-c",
                           "set default_transaction_read_only = on; " + sql],
                          capture_output=True, text=True, check=True).stdout


def load_catalogue():
    out = run_sql(f"copy (select accession_id, array_to_string(tsvector_to_array(to_tsvector('simple', {TEXT_SQL})), ' ') "
                  f"from bids_datasets order by 1) to stdout")
    docs = {}
    for line in out.splitlines()[1:] if out.startswith("SET") else out.splitlines():
        if "\t" not in line:
            continue
        acc, words = line.split("\t", 1)
        docs[acc] = set(w for w in words.split() if re.fullmatch(r"[a-z]+", w))
    vocab = Counter()
    for ws in docs.values():
        vocab.update(ws)
    words = sorted(vocab)
    stems = {}
    for k in range(0, len(words), 5000):
        chunk = words[k:k + 5000]
        arr = "array[" + ",".join("'" + w + "'" for w in chunk) + "]"
        res = run_sql(f"copy (select w, array_to_string(tsvector_to_array(to_tsvector('english', w)), ',') "
                      f"from unnest({arr}) w) to stdout")
        for line in res.splitlines():
            if "\t" in line:
                w, st = line.split("\t", 1)
                stems[w] = st
    return docs, vocab, stems


NAMES: set = set()      # words the parser tags as a name most of the time (filled by lemma_pairs)


def lemma_pairs(paths):
    seen = defaultdict(Counter)
    pos_of = defaultdict(Counter)
    tag = defaultdict(Counter)
    for p in paths:
        for line in gzip.open(p, "rt"):
            _, _, toks = json.loads(line)
            for t in toks:
                w, lem, pos = t[0].lower(), t[1].lower(), t[2]
                tag[w][pos == "PROPN"] += 1
                if pos in ("NOUN", "ADJ", "VERB", "ADV", "PROPN") and w.isalpha() and lem.isalpha():
                    seen[w][lem] += 1
                    pos_of[(w, lem)][pos] += 1
    NAMES.update(w for w, c in tag.items() if c[True] > c[False])
    pairs = {}
    for w, lems in seen.items():
        lem, n = lems.most_common(1)[0]
        if lem == w or n < 2 or n / sum(lems.values()) < 0.5:
            continue
        pos = pos_of[(w, lem)].most_common(1)[0][0]
        # irregular verb forms are left out: searches name things, and "run" (a scan run) is not "ran"
        if pos == "ADJ" and w.endswith(("er", "est")):
            pairs[(lem, w)] = "comparative"
        elif pos in ("NOUN", "PROPN"):
            pairs[(lem, w)] = "plural"
    return pairs


def rule_pairs(vocab):
    pairs = {}
    for a, b in IRREGULAR:
        if a in vocab and b in vocab:
            pairs[(a, b)] = "irregular plural"
    for w in vocab:
        for sg, pl in LATIN:
            if w.endswith(sg) and len(w) - len(sg) >= 3:
                other = w[: -len(sg)] + pl
                if other in vocab and other != w:
                    pairs[(w, other)] = "Latin/Greek plural"
        for noun, adj in DERIVATION:
            if w.endswith(noun) and len(w) - len(noun) >= (2 if noun == "sion" else 3):
                other = w[: -len(noun)] + adj
                if other in vocab and other != w:
                    pairs[(w, other)] = "noun/adjective"
        for pat, rep in SPELLING:
            other = re.sub(pat, rep, w)
            if other != w and other in vocab:
                pairs[(other, w)] = "British/American spelling"
    return pairs


def examples(docs_text, word, n=2):
    out = []
    pat = re.compile(r"(?i)(?<![a-z])" + re.escape(word) + r"(?![a-z])")
    for acc, text in docs_text:
        m = pat.search(text)
        if m:
            a, b = max(0, m.start() - 70), min(len(text), m.end() + 70)
            out.append({"dataset": acc, "before": text[a:m.start()], "match": text[m.start():m.end()],
                        "after": text[m.end():b]})
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", nargs="*", default=[])
    ap.add_argument("--min-impact", type=int, default=1)
    a = ap.parse_args()

    docs, vocab, stems = load_catalogue()
    print("datasets", len(docs), "words", len(vocab))
    links = {}
    lemmas = lemma_pairs(a.parsed)
    for (x, y), kind in list(rule_pairs(vocab).items()) + list(lemmas.items()):
        if x in SKIP_WORDS or y in SKIP_WORDS or x not in vocab or y not in vocab:
            continue
        if x in NAMES or y in NAMES or min(len(x), len(y)) < 3:
            continue          # France/Francis, Maria/Marion; pi/pis
        if max(vocab[x], vocab[y]) < MIN_FORM_DATASETS:
            continue          # a form in one dataset still counts when the other is common (schizophrenic)
        sx, sy = stems.get(x, ""), stems.get(y, "")
        if not sx or not sy or sx == sy:
            continue
        links.setdefault((x, y), set()).add(kind)
    print("links", len(links))

    # families: the linked forms, each with its regular plural/singular of the same stem
    import sys
    sys.path.insert(0, str(HERE.parent.parent / "RAG"))
    from word_forms import inflections

    parent = {}

    def find(w):
        parent.setdefault(w, w)
        while parent[w] != w:
            parent[w] = parent[parent[w]]
            w = parent[w]
        return w

    def union(a_, b_):
        parent[find(a_)] = find(b_)
    for (x, y) in links:
        union(x, y)
    for w in list(parent):
        for v in inflections(w):
            if v in vocab and stems.get(v) == stems[w]:
                union(w, v)
    fams = defaultdict(set)
    for w in parent:
        fams[find(w)].add(w)

    docs_of = defaultdict(set)
    for acc, ws in docs.items():
        for w in ws:
            docs_of[w].add(acc)
    stem_group = defaultdict(set)
    for w, st in stems.items():
        if st:
            stem_group[st].add(w)

    text = run_sql(f"copy (select accession_id, regexp_replace({TEXT_SQL}, '\\s+', ' ', 'g') from bids_datasets order by 1) to stdout")
    docs_text = [tuple(l.split("\t", 1)) for l in text.splitlines() if "\t" in l]

    families = []
    for forms in fams.values():
        fam_docs = set().union(*(docs_of[w] for w in forms))
        by_st = defaultdict(list)
        for w in forms:
            by_st[stems[w]].append(w)
        groups = []
        for st, ws in by_st.items():
            today = set().union(*(docs_of[w] for w in stem_group[st]))     # what a search for this stem finds now
            own = set().union(*(docs_of[w] for w in ws))
            ws = sorted(ws, key=lambda w: -vocab[w])
            groups.append({"stem": st, "datasets": len(own), "missed_without_others": len(fam_docs - today),
                           "stem_also_matches": sorted(stem_group[st] - set(ws), key=lambda w: -vocab[w])[:6],
                           "forms": [{"form": w, "datasets": vocab[w], "examples": examples(docs_text, w)} for w in ws[:8]],
                           "more_forms": max(0, len(ws) - 8)})
        groups.sort(key=lambda g: -g["datasets"])
        impact = max(g["missed_without_others"] for g in groups)
        if impact < a.min_impact:
            continue
        fam_links = [{"a": x, "b": y, "kinds": sorted(k)} for (x, y), k in links.items() if x in forms]
        kinds = sorted({k for l in fam_links for k in l["kinds"]})
        families.append({"kinds": kinds, "impact": impact, "datasets": len(fam_docs),
                         "links": fam_links, "groups": groups})
    families.sort(key=lambda f: (-f["impact"], f["groups"][0]["stem"]))
    for k, f in enumerate(families):
        f["id"] = f"m{k:04d}"
        f["title"] = " / ".join(g["forms"][0]["form"] for g in f["groups"])
    set_id = "merge-" + hashlib.sha1("|".join(f["title"] for f in families).encode()).hexdigest()[:12]
    data = {"set_id": set_id, "datasets": len(docs), "families": families}
    (HERE / "merge_candidates.json").write_text(json.dumps(data, indent=1))
    tpl = (HERE / "merge_template.html").read_text()
    page = tpl.replace("__SET_ID__", set_id).replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    (HERE / "merge_review.html").write_text(page)
    kinds = Counter(k for f in families for k in f["kinds"])
    print("families", len(families), "set", set_id)
    print("by kind", dict(kinds))
    print("impact >= 20:", sum(f["impact"] >= 20 for f in families), " >= 5:", sum(f["impact"] >= 5 for f in families))


if __name__ == "__main__":
    main()
