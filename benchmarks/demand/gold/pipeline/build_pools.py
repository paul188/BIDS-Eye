"""Candidate pools per question: our facts and text, OpenNeuro's text and search, and what the live search returned."""
import json, re, sys, collections
import lib, spec, onsearch
D, ON = lib.load()
TEXT = {k: lib.our_text(d) for k, d in D.items()}
ONTEXT = {k: lib.on_text(o) for k, o in ON.items()}
ALLTEXT = {k: TEXT.get(k, "") + "\n" + ONTEXT.get(k, "") for k in set(D) | set(ON)}
run = {r["id"]: r for r in json.load(open(f"{lib.G}/run_partial.json"))["results"]}
ON_CAP = 80
RUN_TOP = 30

def on_modalities(o):
    s = (o.get("latestSnapshot") or {}).get("summary") or {}
    return " ".join((s.get("modalities") or []) + (s.get("secondaryModalities") or [])).lower()

def pool(q):
    why = collections.defaultdict(list)
    rx = [re.compile(p, re.I) for p in q.get("text", [])]
    mod_rx = re.compile(q["modality_text"], re.I) if q.get("modality_text") else None
    must = q.get("must")
    for k in ALLTEXT:
        d = D.get(k)
        text = ALLTEXT[k]
        in_ours = d is not None
        ok_must = True
        if must and in_ours:
            ok_must = bool(must(d))
            if not ok_must and mod_rx and mod_rx.search(text):
                ok_must = "text"          # modality only named in the text
        elif not in_ours:
            if q.get("on_skip") or q["method"] in ("exact", "population"):
                continue
            if q.get("on_mod") and not re.search(spec.ON_MOD[q["on_mod"]], on_modalities(ON[k])):
                continue
            ok_must = "unknown"           # not in our catalogue: only OpenNeuro's summary and text
        if not ok_must:
            continue
        if in_ours and callable(q.get("exact")):
            e = q["exact"](d, text) if q.get("exact_text") else q["exact"](d)
            if e in ("relevant", "partial", "unknown"):
                why[k].append(f"facts:{e}")
        if in_ours and q.get("pool_all_must") and ok_must is True:
            why[k].append("precondition")
        if in_ours and q.get("names") and q["names"](d):
            why[k].append("names")
        hits = [p.pattern[:30] for p in rx if p.search(text)]
        if q.get("need_all_text") and len(hits) < len(rx):
            hits = []
        if hits and not (q.get("review_only_unmatched") and any(w.startswith("facts:relevant") for w in why[k])):
            why[k].append("text:" + "; ".join(hits[:3]))
        if in_ours and q.get("dx"):
            dx = " ".join((d.get("participants") or {}).get("dx_counts") or {}) + " " + (d.get("dx_status") or "")
            if re.search(q["dx"], dx, re.I):
                why[k].append("diagnosis")
        if in_ours and q.get("extra"):
            keys = " ".join(((d.get("participants") or {}).get("extra_keys") or []) + (d.get("phenotype_files") or []))
            if re.search(q["extra"], keys, re.I):
                why[k].append("participant columns")
        if in_ours and q.get("review") and q["review"](d):
            if any(re.search(p, text, re.I) for p in q.get("review_text", [".*"])):
                why[k].append("review:facts missing")
    for s in q.get("on", []):
        try:
            ids = onsearch.search(**s)
        except Exception as e:
            print("  on search failed", s, e, file=sys.stderr); continue
        if len(ids) <= ON_CAP:
            for k in ids:
                if k in D or (k in ON and not q.get("on_skip") and q["method"] not in ("exact", "population")):
                    why[k].append("openneuro search:" + json.dumps(s, sort_keys=True)[:50])
    r = run.get(q["id"])
    if r and not r.get("fallback"):
        for k in (r.get("returned") or [])[:RUN_TOP]:
            why[k].append("returned by BIDS-Eye")
    return {k: v for k, v in why.items() if v}

if __name__ == "__main__":
    pools = {}
    for q in spec.Q:
        p = pool(q)
        pools[q["id"]] = p
        ours = [k for k in p if k in D]; on_only = [k for k in p if k not in D]
        facts_rel = sum(1 for k in ours if any(w == "facts:relevant" for w in p[k]))
        only_facts = sum(1 for k in ours if all(w.startswith("facts:") for w in p[k]))
        to_judge = len(ours) - only_facts if q["method"] in ("mixed", "judged") else 0
        print(f"{q['id'][:40]:40} {q['method']:6} pool={len(ours):5} facts_rel={facts_rel:5} to_judge={to_judge:5} on_only={len(on_only):4}")
    json.dump(pools, open(f"{lib.G}/pools.json", "w"))
    print("questions:", len(spec.Q))
