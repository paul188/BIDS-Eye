"""List the func folders of the first subjects of given datasets on OpenNeuro: python3 tools/check_preproc.py ds007088 ..."""
import json, sys, time, urllib.request, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lib
D, ON = lib.load()
Q = 'query($id:ID!,$tag:String!,$tree:String){snapshot(datasetId:$id,tag:$tag){files(tree:$tree){id filename directory}}}'
def ls(acc, tag, tree):
    req = urllib.request.Request("https://openneuro.org/crn/graphql", json.dumps({"query": Q, "variables": {"id": acc, "tag": tag, "tree": tree}}).encode(), {"content-type": "application/json"})
    for a in range(3):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=40))
            return ((d.get("data") or {}).get("snapshot") or {}).get("files") or []
        except Exception:
            time.sleep(2 * (a + 1))
    return []
for acc in sys.argv[1:]:
    tag = (ON[acc].get("latestSnapshot") or {}).get("tag")
    root = ls(acc, tag, None)
    subs = [f for f in root if f["directory"] and f["filename"].startswith("sub-")]
    print(acc, tag, "dirs", [f["filename"] for f in root if f["directory"]][:5], "subjects", len(subs), flush=True)
    found = None
    for s in subs[:3]:
        lvl = ls(acc, tag, s["id"])
        ses = [f for f in lvl if f["directory"] and f["filename"].startswith("ses-")]
        for node in (ses[:2] or [s]):
            inner = lvl if node is s else ls(acc, tag, node["id"])
            func = [f for f in inner if f["directory"] and f["filename"] == "func"]
            if not func:
                print("   ", s["filename"], node["filename"], "no func; has", [f["filename"] for f in inner][:4], flush=True); continue
            fn = [f["filename"] for f in ls(acc, tag, func[0]["id"])]
            hit = [n for n in fn if "desc-preproc_bold.nii" in n]
            print("   ", s["filename"], node["filename"], "func files", len(fn), "preproc", len(hit), [n[-60:] for n in (hit or fn)[:2]], flush=True)
            found = found or (hit[0] if hit else None)
        if found: break
    print("  =>", acc, found or "none found in the first subjects", flush=True)
