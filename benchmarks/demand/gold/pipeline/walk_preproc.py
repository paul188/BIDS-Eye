"""Which datasets hold files ending in _desc-preproc_bold.nii.gz? Walk OpenNeuro's file tree where preprocessed outputs can sit."""
import json, re, time, urllib.request, os
import lib
D, ON = lib.load()
G = lib.G
Q = 'query($id:ID!,$tag:String!,$tree:String){snapshot(datasetId:$id,tag:$tag){files(tree:$tree){id filename directory}}}'
def ls(acc, tag, tree):
    req = urllib.request.Request("https://openneuro.org/crn/graphql", json.dumps({"query": Q, "variables": {"id": acc, "tag": tag, "tree": tree}}).encode(), {"content-type": "application/json", "user-agent": "Mozilla/5.0 (research script)"})
    for a in range(4):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=120))
            return ((d.get("data") or {}).get("snapshot") or {}).get("files") or []
        except Exception:
            time.sleep(2 * (a + 1))
    return []
def find(acc, tag, tree, depth, budget):
    """Depth-first into the first subject / session / func folder of each level."""
    if budget[0] <= 0 or depth > 6: return None
    budget[0] -= 1
    files = ls(acc, tag, tree)
    names = [f["filename"] for f in files]
    hit = [n for n in names if re.search(r"desc-preproc_bold\.nii", n)]
    if hit: return hit[0]
    dirs = [f for f in files if f["directory"]]
    order = [f for f in dirs if f["filename"] == "func"] or [f for f in dirs if re.match(r"(sub|ses)-", f["filename"])][:1] or \
            [f for f in dirs if re.search(r"deriv|fmriprep|preproc|xcp|pipeline|output", f["filename"], re.I)][:6]
    for f in order:
        r = find(acc, tag, f["id"], depth + 1, budget)
        if r: return f["filename"] + "/" + r
    return None
trees = {json.loads(l)["id"]: json.loads(l) for l in open(f"{G}/on_trees.jsonl")}
cands = []
for acc, t in trees.items():
    o = ON.get(acc) or {}
    s = o.get("latestSnapshot") or {}
    name = ((s.get("description") or {}).get("Name") or "") + " " + ((s.get("description") or {}).get("DatasetType") or "")
    text = (s.get("readme") or "")
    subdirs = " ".join(x for k, v in (t.get("sub") or {}).items() if k.lower().startswith("deriv") for x in v) + " " + " ".join(k for k in (t.get("sub") or {}))
    if re.search(r"fmriprep|preproc|xcp|halfpipe|cpac|c-pac", subdirs, re.I) or re.search(r"fmriprep|derivative|preproc", name, re.I) or re.search(r"desc-preproc", text):
        cands.append(acc)
print("candidates", len(cands), flush=True)
out = {}
for i, acc in enumerate(sorted(cands)):
    budget = [14]
    r = find(acc, trees[acc]["tag"], None, 0, budget)
    out[acc] = r
    if r: print(acc, r, flush=True)
json.dump(out, open(f"{G}/desc_preproc_bold.json", "w"))
print("done; with such files:", sum(1 for v in out.values() if v))
