"""OpenNeuro's own search (the search page's advancedSearch), cached on disk."""
import json, time, urllib.request, urllib.error, os
G = os.path.dirname(os.path.abspath(__file__))
CACHE = f"{G}/onsearch_cache.json"
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
Q = """query($q: DatasetSearchInput!, $after: String) {
  advancedSearch(query: $q, allDatasets: false, datasetType: "All Public", datasetStatus: "All", first: 100, after: $after) {
    edges { id } pageInfo { count hasNextPage endCursor } } }"""
def _post(query, variables):
    req = urllib.request.Request("https://openneuro.org/crn/graphql", json.dumps({"query": query, "variables": variables}).encode(),
        {"content-type": "application/json", "user-agent": "Mozilla/5.0 (research script)"})
    for attempt in range(5):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=120))
            if d.get("errors"): raise RuntimeError(str(d["errors"])[:300])
            return d["data"]
        except urllib.error.HTTPError as e:
            err = f"{e.code} {e.read().decode()[:300]}"
        except Exception as e:
            err = repr(e)[:300]
        if attempt == 4: raise RuntimeError(err)
        time.sleep(3 * (attempt + 1))
def search(**q):
    """search(keywords=["autism"], modality="eeg") -> sorted list of accession ids."""
    key = json.dumps(q, sort_keys=True)
    if key in _cache: return _cache[key]
    ids, after = [], None
    while True:
        page = _post(Q, {"q": q, "after": after})["advancedSearch"]
        ids += [e["id"] for e in page["edges"] if e]
        if not page["pageInfo"]["hasNextPage"] or not page["edges"]: break
        after = page["pageInfo"]["endCursor"]
    _cache[key] = sorted(set(ids))
    json.dump(_cache, open(CACHE, "w"))
    return _cache[key]
if __name__ == "__main__":
    r = _post('{__type(name:"DatasetSearchInput"){inputFields{name type{name kind ofType{name kind ofType{name}}}}}}', {})
    print([(f["name"], f["type"]["name"] or f["type"]["ofType"]["name"] or f["type"]["ofType"]["ofType"]["name"]) for f in r["__type"]["inputFields"]])
    for q in [dict(keywords=["autism"], modality="eeg"), dict(keywords=["autism"]), dict(keywords=["dream"]), dict(keywords=["spinal"]), dict(keywords=["alzheimer"], modality="eeg"), dict(keywords=["eating disorder"]), dict(keywords=["eating", "disorder"])]:
        ids = search(**q); print(q, len(ids), ids[:12])
