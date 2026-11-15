"""Write the gold test set for the 70 real requests to benchmarks/demand/gold/.

    python3 assemble.py && python3 export_gold.py

Reads truth_draft.json and judgments_draft.jsonl (assemble.py), the criteria
(spec.py) and the requests (../../requests.yaml). Also copies the scripts and
the inputs that cannot be regenerated (judge output, corrections) to gold/pipeline/.
"""
import collections, glob, json, os, shutil
import yaml
import lib, spec, assemble

G = lib.G
DEMAND = os.path.dirname(os.path.dirname(G))
OUT = os.path.join(DEMAND, "gold")
SNAPSHOT = "2026-11-14"

# What the recorded facts and the dataset texts cannot settle, per question.
NOT_CHECKED = {
    "qf05": "A recorded TR below 0.1 s is the excitation TR of 3D-EPI or line-scan data, not the volume TR; those datasets are unsure.",
    "qf07": "Whether a task is run in blocks is often not stated; category localizers and the HCP task battery count as block designs by convention.",
    "qf09": "The ASL labelling scheme is often not stated; such datasets are unsure.",
    "qf12": "The order of runs inside a session is known only where the README or a paper states it. Rest - task - rest that nobody describes is not found.",
    "qf19": "Films that are not named or described cannot be checked for visible talking people; such datasets are unsure.",
    "qf20": "OpenNeuro holds NIfTI, not PNG or JPG: nothing is fully relevant.",
    "qf29": "The 30 trials and the randomisation named in the request are not checked.",
    "qf34": "Datasets without a recorded licence are unsure.",
    "qf39": "A healthy adult sample of 10 or more without recorded sex counts as including men.",
    "qf41": "An electrodes file whose origin is not stated counts as partial (template or measured unknown).",
    "qf44": "The threshold of 50 subjects is ours; the request names none.",
    "qp02": "Where no ages are recorded and the text never mentions children, adults are presumed (source 'presumed' in judgments.jsonl). 'Healthy' rests on diagnoses, uploader labels or the absence of any clinical term.",
    "qp03": "Datasets without recorded ages whose text mentions children are unsure.",
    "qp04": "Prospective motion correction is known only from sequence names and README text.",
    "qp05": "Voxel size and number of volumes are not recorded and are not checked.",
    "qp10": "Whether the publication reports group-level activation is not checked.",
    "qp12": "Not scored: 'normative database' implies a size and curation the request does not define.",
    "qp13": "The partially relevant class (any event-related EEG experiment) is listed only for the datasets that were judged.",
    "qp14": "Datasets without a recorded licence are unsure.",
    "qx08": "Electrode type comes from the channel types of three subjects per dataset where the README is silent.",
    "qx09": "Sex not recorded counts as not relevant; numeric sex codes are not trusted without text.",
}
METHOD = {
    "exact": "rule on recorded facts (files, sidecars, participants.tsv, licence)",
    "population": "rule on recorded facts plus a classification of who the participants are (rules, 616 datasets read by a judge)",
    "mixed": "recorded facts where they decide; every other candidate read by an LLM judge and reviewed",
    "judged": "every candidate read by an LLM judge and reviewed",
}


def compact(obj):
    """JSON with one line per list, so the file stays readable and small."""
    out = ["{"]
    items = list(obj.items())
    for i, (k, v) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        if isinstance(v, dict):
            inner = ",\n".join(f"   {json.dumps(a)}: {json.dumps(b, ensure_ascii=False)}" for a, b in v.items())
            out.append(f"  {json.dumps(k)}: {{\n{inner}\n  }}{comma}")
        else:
            out.append(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}{comma}")
    out.append("}")
    return "\n".join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    truth = json.load(open(f"{G}/truth_draft.json"))
    rows = [json.loads(l) for l in open(f"{G}/judgments_draft.jsonl")]
    requests = {r["id"]: r for r in yaml.safe_load(open(f"{DEMAND}/requests.yaml"))["requests"]}
    by_source = collections.defaultdict(collections.Counter)
    for r in rows:
        by_source[r["query"]][r["source"]] += 1

    # 1. truth in the format of benchmarks/recall/ground_truth.json
    scored = {q["id"]: truth[q["id"]] for q in spec.Q if q.get("scored", True)}
    not_scored = {q["id"]: truth[q["id"]] for q in spec.Q if not q.get("scored", True)}
    head = {
        "snapshot_date": SNAPSHOT,
        "judged_by": "Rules on recorded facts; LLM judges (Claude) per criteria in requests_gold.yaml; reviewed, changes in pipeline/corrections.jsonl",
        "note": "For judged questions every catalogue dataset that was not a candidate is listed as not_relevant (n_judged = how many a judge or the review read). "
                "relevant_not_in_catalogue: datasets on OpenNeuro that our catalogue lacks; they do not count in recall.",
    }
    with open(f"{OUT}/requests_truth.json", "w") as f:
        body = compact(scored)
        f.write("{\n" + "".join(f" {json.dumps(k)}: {json.dumps(v)},\n" for k, v in head.items())
                + ' "queries": ' + body.replace("\n", "\n ") + ',\n "not_scored": ' + compact(not_scored).replace("\n", "\n ") + "\n}\n")
    json.load(open(f"{OUT}/requests_truth.json"))                      # must parse

    # 2. one line per judged or rule-decided pair
    with open(f"{OUT}/judgments.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 3. who the participants are, per dataset
    with open(f"{OUT}/population.jsonl", "w") as f:
        for acc in sorted(assemble.D):
            pop, source, j = assemble.population(acc)
            rec = {"acc": acc, "population": pop, "source": source}
            if j:
                rec.update({k: j[k] for k in ("n_healthy", "healthy_adults_5", "healthy_children_5", "includes_men", "reason") if k in j})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 4. the question + answer sheet
    sheet = []
    for q in spec.Q:
        t, r = truth[q["id"]], requests[q["id"]]
        entry = {
            "id": q["id"], "ref": r.get("ref"), "url": r.get("url"),
            "question": r.get("question"), "quote": r.get("quote"),
            "relevant_if": q["relevant_if"],
        }
        if q.get("partially_if"):
            entry["partially_if"] = q["partially_if"]
        entry["method"] = METHOD[q["method"]]
        if NOT_CHECKED.get(q["id"][:4]):
            entry["not_checked"] = NOT_CHECKED[q["id"][:4]]
        if not q.get("scored", True):
            entry["scored"] = False
        entry["counts"] = {
            "relevant": len(t["relevant"]), "partially_relevant": len(t["partially_relevant"]), "unsure": len(t["unsure"]),
            "relevant_not_in_catalogue": len(t["relevant_not_in_catalogue"]),
        }
        if "n_judged" in t:
            entry["counts"]["candidates_read"] = t["n_judged"]
            entry["counts"]["changed_in_review"] = by_source[q["id"]]["reviewed"]
        if len(t["relevant"]) <= 40:
            entry["relevant"] = t["relevant"]
        if t["relevant_not_in_catalogue"]:
            entry["relevant_not_in_catalogue"] = t["relevant_not_in_catalogue"][:40]
        sheet.append(entry)
    with open(f"{OUT}/requests_gold.yaml", "w") as f:
        f.write("# Gold answers for the 70 real requests of requests.yaml. Built by pipeline/export_gold.py; do not edit by hand.\n"
                f"# Catalogue and OpenNeuro snapshot: {SNAPSHOT}. Full answer lists: requests_truth.json; reasons: judgments.jsonl.\n"
                "# 'relevant' is listed here when a question has at most 40 relevant datasets.\n")
        yaml.safe_dump({"requests": sheet}, f, allow_unicode=True, sort_keys=False, width=110)

    # 5. scripts and the inputs that cannot be regenerated
    B = f"{OUT}/pipeline"
    for sub in ("", "/tools", "/judged"):
        os.makedirs(B + sub, exist_ok=True)
    keep = ["lib.py", "spec.py", "build_pools.py", "cards.py", "population.py", "assemble.py", "export_gold.py",
            "fetch_on.py", "fetch_trees.py", "walk_preproc.py", "channel_types.py", "onsearch.py",
            "JUDGE_INSTRUCTIONS.md", "POPULATION_INSTRUCTIONS.md", "corrections.jsonl", "population_rules.json", "channel_types.json",
            "desc_preproc_bold.json"]
    keep += [os.path.basename(p) for p in glob.glob(f"{G}/f*.sql")]
    for name in keep:
        if os.path.exists(f"{G}/{name}"):
            shutil.copy2(f"{G}/{name}", f"{B}/{name}")
    for p in glob.glob(f"{G}/tools/*.py"):
        shutil.copy2(p, f"{B}/tools/")
    for p in glob.glob(f"{G}/judged/*.jsonl"):
        shutil.copy2(p, f"{B}/judged/")
    print("written:", OUT)
    for name in sorted(os.listdir(OUT)):
        p = f"{OUT}/{name}"
        print(f"  {name:24} {os.path.getsize(p) // 1000 if os.path.isfile(p) else sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(p) for x in fs) // 1000:6} kB")


if __name__ == "__main__":
    main()
