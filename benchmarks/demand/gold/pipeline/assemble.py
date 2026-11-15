"""Merge recorded facts, participant classifications, judge labels and reviewed corrections into the gold truth.

Order of authority per (question, dataset):
  1. corrections.jsonl   labels changed after review (always win)
  2. judge label         for judged / mixed questions
  3. rule from facts     exact questions, and the fact part of mixed ones
Datasets outside a question's pool are not relevant.
"""
import json, re, glob, os, collections, sys
import lib, spec
from population import RULES as POP_RULE

G = lib.G
D, ON = lib.load()
POOLS = json.load(open(f"{G}/pools.json"))
TREES = {}
for l in open(f"{G}/on_trees.jsonl"):
    r = json.loads(l); TREES[r["id"]] = r
TEXT = {k: lib.our_text(d) for k, d in D.items()}
PREPROC_FILES = json.load(open(f"{G}/desc_preproc_bold.json")) if os.path.exists(f"{G}/desc_preproc_bold.json") else {}

# What our crawl lacks but OpenNeuro records: the licence, and participant ages.
def _num(x):
    try: return float(x)
    except Exception: return None
for _acc, _d in D.items():
    _o = ON.get(_acc) or {}
    _snap = _o.get("latestSnapshot") or {}
    if not (_d.get("license") or "").strip():
        _d["license"] = ((_snap.get("description") or {}).get("License") or "")
    _ages = [_num(x.get("age")) for x in ((_snap.get("summary") or {}).get("subjectMetadata") or [])]
    _d["on_ages"] = [a for a in _ages if a is not None]
CHILD_WORDS = re.compile(r"child|infant|toddler|adolescen|pediatric|paediatric|\bkids?\b|\bteen|school-?age|newborn|neonat|\bboys\b|\bgirls\b|juvenile|developmental", re.I)
ADULT_WORDS = re.compile(r"\badults?\b|undergraduate|university students|college students|young adult|older adult|elderly|\bmen\b|\bwomen\b", re.I)

# ── judge output ──────────────────────────────────────────────────────────────
JUDGED = collections.defaultdict(dict)          # query -> acc -> (label, reason)
POP_JUDGED = {}
for path in sorted(glob.glob(f"{G}/judged/*.jsonl")):
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if "population" in r:
            POP_JUDGED[r["acc"]] = r
        elif "query" in r:
            JUDGED[r["query"]][r["acc"]] = (r["label"], r.get("reason", ""))
CORR = collections.defaultdict(dict)
if os.path.exists(f"{G}/corrections.jsonl"):
    for line in open(f"{G}/corrections.jsonl"):
        if line.strip().startswith("{"):
            r = json.loads(line)
            CORR[r["query"]][r["acc"]] = (r["label"], r.get("reason", ""))

# ── file listings from OpenNeuro ──────────────────────────────────────────────
def deriv_dirs(acc):
    t = TREES.get(acc) or {}
    out = []
    for k, v in (t.get("sub") or {}).items():
        if k.lower() in ("derivatives", "derivative"):
            out += [x.rstrip("/").lower() for x in v if x.endswith("/")]
    return out

def tree_has(acc, pattern):
    t = TREES.get(acc) or {}
    rx = re.compile(pattern, re.I)
    return any(rx.search(x) for v in (t.get("sub") or {}).values() for x in v) or any(rx.search(x) for x in t.get("root") or [])

# ── participants ──────────────────────────────────────────────────────────────
def population(acc):
    """healthy_only | patients_and_healthy | patients_only | special | not_human | unclear, and where it comes from."""
    j = POP_JUDGED.get(acc)
    rule = POP_RULE.get(acc, "")
    if j:
        return {"special_non_clinical": "special"}.get(j["population"], j["population"]), "judged", j
    if rule.startswith("mixed"): return "patients_and_healthy", "participants.tsv diagnoses", None
    if rule.startswith("healthy_only:label"): return "healthy_only", "uploader label", None
    if rule.startswith("healthy_only:dx"): return "healthy_only", "participants.tsv diagnoses", None
    if rule == "presumed_healthy": return "healthy_only", "no clinical term in the text", None
    if rule == "not_human": return "not_human", "species label", None
    return "unclear", "not classified", None

def healthy_t1(acc, d, child=False):
    """At least 5 healthy adults (or children aged 2-12) with a T1w image?"""
    pop, source, j = population(acc)
    n_t1 = d.get("n_sub_t1") or 0
    if pop in ("not_human", "patients_only") or n_t1 < 5:
        return None
    key = "healthy_children_5" if child else "healthy_adults_5"
    ages_known = (d.get("n_t1_joined") or 0) > 0 and (d.get("n_t1_no_age") or 0) < (d.get("n_t1_joined") or 0)
    if pop == "healthy_only":
        n = d.get("n_child_t1" if child else "n_adult_t1") or 0
        if ages_known:
            return "relevant" if n >= 5 else None
        on_ages = d.get("on_ages") or []
        if len(on_ages) >= 5:                       # OpenNeuro's own participant summary
            n_on = sum(1 for a in on_ages if (2 <= a < 13 if child else a >= 18))
            return "relevant" if n_on >= 5 and n_t1 >= 5 else None
        lo, hi = d.get("age_min"), d.get("age_max")
        if lo is not None and hi is not None:
            if not child and lo >= 18: return "relevant"
            if not child and hi < 18: return None
            if child and (lo > 12.99 or hi < 2): return None
        if j and j.get(key) in ("yes", "no"):
            return "relevant" if j[key] == "yes" else None
        text = TEXT[acc]
        if child:                                   # no ages anywhere: children only if the text speaks of them
            return "unknown" if CHILD_WORDS.search(text) else None
        if ADULT_WORDS.search(text) and not CHILD_WORDS.search(text): return "relevant"
        if not CHILD_WORDS.search(text): return "presumed"      # no ages anywhere, children never mentioned
        return "unknown"
    if pop == "patients_and_healthy":
        n = d.get("n_child_hc_t1" if child else "n_adult_hc_t1") or 0
        if n >= 5: return "relevant"
        if j and j.get(key) in ("yes", "no"):
            return "relevant" if j[key] == "yes" else None
        if child and not CHILD_WORDS.search(TEXT[acc]) and not (d.get("n_child_t1") or 0): return None
        return "unknown"
    if pop == "special":
        return "unknown"
    return "unknown"

# ── rules that need more than one dataset record ──────────────────────────────
def exact_label(q, acc, d):
    qid = q["id"]
    e = q.get("exact")
    lab = None
    if callable(e):
        lab = e(d, TEXT[acc]) if q.get("exact_text") else e(d)
    if qid.startswith("qf05"):
        tr = spec.L(d, "bold_tr_task")
        if not lab and tr and all(t < 0.1 for t in tr): lab = "unknown"       # excitation TR of 3D-EPI / line scans, not the volume TR
        if acc == "ds006947": lab = "unknown"                                 # "short-TR" stated, no value recorded
        if not lab and spec.TASK_BOLD(d) and not tr: lab = "unknown"          # task runs without a recorded TR
    if qid.startswith("qf11") and not lab and any("fmriprep" in x for x in deriv_dirs(acc)): lab = "relevant"
    if qid.startswith("qx06") and not lab and any("mriqc" in x for x in deriv_dirs(acc)): lab = "relevant"
    if qid.startswith(("qf18", "qf25")) and not lab:
        if q["must"](d) and any(re.search(r"fmriprep|xcp|cpac|c-pac|halfpipe|tedana|aroma", x) for x in deriv_dirs(acc)): lab = "relevant"
    if qid.startswith("qx11"):
        lab = "relevant" if (d.get("n_desc_preproc_bold") or PREPROC_FILES.get(acc) or tree_has(acc, r"desc-preproc_bold\.nii")) else None
    if qid.startswith("qf06"):
        if spec.REST_BOLD(d):
            pop = population(acc)[0]
            lab = {"healthy_only": "relevant", "patients_and_healthy": "relevant", "special": "unknown", "unclear": "unknown"}.get(pop)
    if qid.startswith("qp02"): lab = healthy_t1(acc, d)
    if qid.startswith("qp03"): lab = healthy_t1(acc, d, child=True)
    if qid.startswith("qp12"):
        lab = "relevant" if spec.MRI(d) and population(acc)[0] == "healthy_only" else None
    return lab

LABELS = ("relevant", "partially_relevant", "not_relevant", "unsure")
NORM = {"partial": "partially_relevant", "unknown": "unsure", "presumed": "relevant"}
PRESUMED = "healthy sample with T1w images; no ages recorded and children never mentioned in the text: adults presumed"

def outside_label(q, acc):
    """Exact questions that OpenNeuro's own summary and file listing can decide for datasets our catalogue lacks."""
    qid = q["id"]
    snap = (ON.get(acc) or {}).get("latestSnapshot") or {}
    mods = [str(m).lower() for m in ((snap.get("summary") or {}).get("modalities") or [])]
    licence = ((snap.get("description") or {}).get("License") or "").upper().replace("-", "").replace(" ", "")
    if qid.startswith("qx11"):
        return "relevant" if (PREPROC_FILES.get(acc) or tree_has(acc, r"desc-preproc_bold\.nii")) else None
    if qid.startswith("qx07"):
        if "eeg" in mods or "meg" in mods: return "relevant"
        return "partial" if "ieeg" in mods else None
    if qid.startswith("qf34") and "mri" in mods:
        if licence.startswith(("CC0", "PDDL")): return "relevant"
        return "partial" if licence.startswith("CCBY") and "NC" not in licence and "ND" not in licence else None
    if qid.startswith("qf26"):
        return "relevant" if tree_has(acc, r"_bold\.(nii|json)") else None
    return None

def build():
    truth, rows, conflicts = {}, [], []
    for q in spec.Q:
        qid = q["id"]
        labels = {}                                   # acc -> (label, reason, source)
        for acc, d in D.items():
            lab = exact_label(q, acc, d)
            if lab:
                labels[acc] = (NORM.get(lab, lab), PRESUMED if lab == "presumed" else "recorded facts", "presumed" if lab == "presumed" else "facts")
        if q["method"] == "exact":
            for acc in ON:
                lab = None if acc in D else outside_label(q, acc)
                if lab:
                    labels[acc] = (NORM.get(lab, lab), "OpenNeuro summary and file listing", "facts")
        if q["method"] in ("mixed", "judged"):
            for acc, (lab, reason) in JUDGED.get(qid, {}).items():
                prev = labels.get(acc)
                if prev and prev[0] == "relevant" and lab in ("not_relevant", "partially_relevant"):
                    conflicts.append((qid, acc, "facts say relevant", lab, reason))
                if prev and prev[0] == "relevant" and lab == "unsure":
                    continue                           # facts decide when the judge cannot
                if prev and prev[0] == "partially_relevant" and lab == "not_relevant" and qid.startswith("qf41"):
                    continue                           # an electrodes file exists: stays partial
                labels[acc] = (lab, reason, "judged")
        if q.get("never_relevant"):
            labels = {a: (("partially_relevant" if v[0] == "relevant" else v[0]), v[1], v[2]) for a, v in labels.items()}
        for acc, (lab, reason) in CORR.get(qid, {}).items():
            labels[acc] = (lab, reason, "reviewed")
        ours = {a: v for a, v in labels.items() if a in D}
        outside = {a: v for a, v in labels.items() if a not in D}
        entry = {"method": "structural" if q["method"] in ("exact", "population") else "judged",
                 "source": "real_request", "type": q["method"], "scored": q.get("scored", True)}
        for name in LABELS:
            entry[name if name != "partially_relevant" else "partially_relevant"] = sorted(a for a, v in ours.items() if v[0] == name)
        entry["relevant_not_in_catalogue"] = sorted(a for a, v in outside.items() if v[0] == "relevant")
        entry["partially_relevant_not_in_catalogue"] = sorted(a for a, v in outside.items() if v[0] == "partially_relevant")
        if q["method"] in ("mixed", "judged"):
            entry["n_judged"] = sum(1 for v in ours.values() if v[2] in ("judged", "reviewed"))
            entry["not_relevant"] = sorted(set(entry["not_relevant"]) | (set(D) - set(ours)))
        truth[qid] = entry
        for acc, (lab, reason, source) in sorted(labels.items()):
            if source not in ("facts", "presumed") or lab != "not_relevant":
                rows.append({"query": qid, "acc": acc, "label": lab, "source": source, "reason": reason, "in_catalogue": acc in D})
    return truth, rows, conflicts

if __name__ == "__main__":
    truth, rows, conflicts = build()
    print(f"{'question':42} {'method':10} {'rel':>5} {'part':>5} {'not':>5} {'unsure':>6} {'rel_out':>7}  judged/pool")
    for q in spec.Q:
        t = truth[q["id"]]
        need = [a for a in POOLS[q["id"]] if a in D] if q["method"] in ("mixed", "judged") else []
        done = len(JUDGED.get(q["id"], {}))
        print(f"{q['id'][:42]:42} {q['method']:10} {len(t['relevant']):5} {len(t['partially_relevant']):5} {len(t['not_relevant']):5} {len(t['unsure']):6} {len(t['relevant_not_in_catalogue']):7}  {done}")
    print("\nconflicts between facts and judges:", len(conflicts))
    for c in conflicts[:60]:
        print("  ", c)
    json.dump(truth, open(f"{G}/truth_draft.json", "w"), indent=1)
    with open(f"{G}/judgments_draft.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("population judged:", len(POP_JUDGED))
