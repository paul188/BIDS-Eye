"""Root file listing of every public OpenNeuro dataset, plus one level below derivatives/, stimuli/, sourcedata/, phenotype/ (resumable)."""
import json, os, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
G = os.path.dirname(os.path.abspath(__file__))
OUT = f"{G}/on_trees.jsonl"
Q = 'query($id:ID!,$tag:String!,$tree:String){snapshot(datasetId:$id,tag:$tag){files(tree:$tree){id filename size directory}}}'
def post(v):
    req = urllib.request.Request("https://openneuro.org/crn/graphql", json.dumps({"query": Q, "variables": v}).encode(), {"content-type": "application/json", "user-agent": "Mozilla/5.0 (research script)"})
    for attempt in range(4):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=120))
            return ((d.get("data") or {}).get("snapshot") or {}).get("files") or []
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None
def one(o):
    tag = (o.get("latestSnapshot") or {}).get("tag")
    if not tag: return {"id": o["id"], "error": "no snapshot"}
    root = post({"id": o["id"], "tag": tag, "tree": None})
    if root is None: return {"id": o["id"], "error": "root failed"}
    rec = {"id": o["id"], "tag": tag, "root": [f["filename"] + ("/" if f["directory"] else "") for f in root], "sub": {}}
    for f in root:
        if f["directory"] and f["filename"].lower() in ("derivatives", "derivative", "stimuli", "stimulus", "sourcedata", "phenotype", "code"):
            sub = post({"id": o["id"], "tag": tag, "tree": f["id"]})
            if sub is not None:
                rec["sub"][f["filename"]] = [s["filename"] + ("/" if s["directory"] else "") for s in sub][:80]
                if f["filename"].lower().startswith("deriv"):
                    for s in sub[:12]:
                        if s["directory"]:
                            deeper = post({"id": o["id"], "tag": tag, "tree": s["id"]})
                            if deeper is not None:
                                rec["sub"][f["filename"] + "/" + s["filename"]] = [x["filename"] + ("/" if x["directory"] else "") for x in deeper][:40]
    return rec
done = {json.loads(l)["id"] for l in open(OUT)} if os.path.exists(OUT) else set()
todo = [json.loads(l) for l in open(f"{G}/openneuro_all.jsonl")]
todo = [o for o in todo if o["id"] not in done]
print("todo", len(todo), flush=True)
with ThreadPoolExecutor(4) as ex, open(OUT, "a") as f:
    for i, rec in enumerate(ex.map(one, todo)):
        f.write(json.dumps(rec) + "\n"); f.flush()
        if i % 100 == 0: print(i, flush=True)
print("done")
