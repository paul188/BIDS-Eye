"""Fetch every public OpenNeuro dataset's metadata, README and summary into openneuro_all.jsonl (resumable)."""
import json, time, urllib.request, urllib.error, sys, os
OUT = sys.argv[1]
Q = """
query($after: String) {
  datasets(first: 25, after: $after, filterBy: {public: true}, orderBy: {created: ascending}) {
    edges { node { id created publishDate onBrainlife
      derivatives { name }
      metadata { modalities species dxStatus studyDomain studyDesign studyLongitudinal ages tasksCompleted trialCount seniorAuthor datasetName associatedPaperDOI openneuroPaperDOI dataProcessed }
      latestSnapshot { tag created readme size
        deprecated { reason }
        description { Name Authors License DatasetDOI ReferencesAndLinks DatasetType }
        summary { modalities primaryModality secondaryModalities sessions subjects tasks size totalFiles dataProcessed subjectMetadata { age sex group } pet { TracerName TracerRadionuclide } }
      } } }
    pageInfo { hasNextPage endCursor count }
  }
}"""
def post(v):
    req = urllib.request.Request("https://openneuro.org/crn/graphql", json.dumps({"query": Q, "variables": v}).encode(),
        {"content-type": "application/json", "user-agent": "Mozilla/5.0 (research script)"})
    for attempt in range(6):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=180))
            if d.get("data") and d["data"].get("datasets"):
                return d
            err = str(d.get("errors"))[:300]
        except urllib.error.HTTPError as e:
            err = f"{e.code} {e.read().decode()[:300]}"
        except Exception as e:
            err = repr(e)[:300]
        print("retry", attempt, err, flush=True); time.sleep(5 * (attempt + 1))
    raise SystemExit("gave up")
state = OUT + ".cursor"
after = open(state).read().strip() if os.path.exists(state) else None
n = sum(1 for _ in open(OUT)) if os.path.exists(OUT) else 0
while True:
    d = post({"after": after})
    page = d["data"]["datasets"]
    with open(OUT, "a") as f:
        for e in page["edges"]:
            if e and e.get("node"):
                f.write(json.dumps(e["node"]) + "\n"); n += 1
    after = page["pageInfo"]["endCursor"]
    open(state, "w").write(after or "")
    print(n, "of", page["pageInfo"]["count"], flush=True)
    if not page["pageInfo"]["hasNextPage"]:
        break
    time.sleep(0.5)
print("done", n)
