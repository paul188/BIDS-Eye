"""Per request: how many datasets and how many correct answers pass each condition of the search."""
import json, collections, sys
import conds
run, truth = conds.load()
meta = json.load(open("blame_meta.json"))
bits = collections.defaultdict(dict)
for l in open("blame.out"):
    p = l.rstrip("\n").split("|")
    if len(p) == 3: bits[p[0]][p[1]] = p[2]
only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
out = {}
for qid, r in run.items():
    t = truth.get(qid)
    if not t: continue
    rel, ret = set(t["relevant"]), r.get("returned") or []
    m, b = meta.get(qid) or {"where": [], "having": []}, bits.get(qid) or {}
    cs = m["where"] + m["having"]
    rows = []
    for i, c in enumerate(cs):
        n_all = sum(1 for v in b.values() if v[i] == "1")
        n_rel = sum(1 for a in rel if b.get(a, "0" * len(cs))[i] == "1")
        only_this = sum(1 for a in rel if a in b and b[a].count("0") == 1 and b[a][i] == "0")     # correct answers lost to this condition alone
        rows.append({"cond": c, "all": n_all, "rel": n_rel, "sole_loss": only_this})
    passed = {a for a, v in b.items() if "0" not in v} if b else None
    out[qid] = {"n_rel": len(rel), "returned": len(ret), "hits": len(rel & set(ret)), "passed_all": None if passed is None else len(passed),
                "rel_passed_all": None if passed is None else len(passed & rel), "conds": rows, "fallback": r.get("fallback"), "self_corrected": r.get("self_corrected"),
                "evaluated": bool(b)}
json.dump(out, open("blame.json", "w"), indent=1)
for qid, o in out.items():
    if only and not qid.startswith(tuple(only)): continue
    print(f"## {qid[:38]} rel={o['n_rel']} returned={o['returned']} hits={o['hits']} pass_all={o['passed_all']} rel_pass_all={o['rel_passed_all']}{' FALLBACK' if o['fallback'] else ''}{'' if o['evaluated'] else ' (not evaluated)'}")
    for c in o["conds"]:
        print(f"   all={c['all']:5} rel={c['rel']:4} sole_loss={c['sole_loss']:3} | {c['cond'][:int(sys.argv[2]) if len(sys.argv) > 2 else 120]}")
