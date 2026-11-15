"""Channel types (SEEG / ECOG / …) from up to 3 subjects' channels.tsv per dataset, read from OpenNeuro."""
import json, re, time, urllib.request, collections, sys
import lib
D, ON = lib.load()
Q = 'query($id:ID!,$tag:String!,$tree:String){snapshot(datasetId:$id,tag:$tag){files(tree:$tree){id filename directory urls}}}'
UA = {"content-type": "application/json", "user-agent": "Mozilla/5.0 (research script)"}
def ls(acc, tag, tree):
    req = urllib.request.Request("https://openneuro.org/crn/graphql", json.dumps({"query": Q, "variables": {"id": acc, "tag": tag, "tree": tree}}).encode(), UA)
    for a in range(4):
        try:
            return ((json.load(urllib.request.urlopen(req, timeout=120)).get("data") or {}).get("snapshot") or {}).get("files") or []
        except Exception:
            time.sleep(2 * (a + 1))
    return []
def find_channels(acc, tag, tree, depth=0):
    files = ls(acc, tag, tree)
    hit = [f for f in files if f["filename"].endswith("_channels.tsv")]
    if hit: return hit[0]
    for f in files:
        if f["directory"] and (f["filename"] == "ieeg" or (depth < 2 and re.match(r"ses-", f["filename"]))):
            r = find_channels(acc, tag, f["id"], depth + 1)
            if r: return r
    return None
def types(acc):
    tag = (ON.get(acc, {}).get("latestSnapshot") or {}).get("tag")
    if not tag: return None
    root = ls(acc, tag, None)
    out = collections.Counter(); groups = collections.Counter(); n = 0
    for f in [x for x in root if x["directory"] and x["filename"].startswith("sub-")][:3]:
        ch = find_channels(acc, tag, f["id"])
        if not ch or not ch.get("urls"): continue
        try:
            text = urllib.request.urlopen(urllib.request.Request(ch["urls"][0], headers={"user-agent": UA["user-agent"]}), timeout=120).read().decode("utf-8", "replace")
        except Exception as e:
            continue
        lines = [l.split("\t") for l in text.splitlines() if l.strip()]
        if not lines: continue
        head = [h.strip().lower() for h in lines[0]]
        n += 1
        if "type" in head:
            i = head.index("type")
            out.update(l[i].strip().upper() for l in lines[1:] if len(l) > i)
        if "group" in head:
            i = head.index("group")
            groups.update(re.sub(r"\d+", "", l[i].strip())[:14] for l in lines[1:] if len(l) > i)
    return {"subjects_read": n, "types": dict(out), "groups": dict(groups.most_common(8))}
if __name__ == "__main__":
    res = {}
    for acc in sys.argv[1:]:
        res[acc] = types(acc); print(acc, json.dumps(res[acc]), flush=True)
    json.dump(res, open(f"{lib.G}/channel_types.json", "w"), indent=1)
