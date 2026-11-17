"""
benchmarks/openneuro_facets/fetch_openneuro.py
----------------------------------------------
Ask OpenNeuro's own search (the GraphQL advancedSearch the search page uses)
for every filter selection in facets.yaml and store the dataset ids.

    python benchmarks/openneuro_facets/fetch_openneuro.py --out openneuro_results.json

Same parameters as the search page: public datasets, all statuses, synthetic
datasets excluded (the server adds that filter itself).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ENDPOINT = "https://openneuro.org/crn/graphql"
QUERY = """
query($q: DatasetSearchInput!, $after: String) {
  advancedSearch(query: $q, allDatasets: false, datasetType: "All Public",
                 datasetStatus: "All", first: 100, after: $after) {
    edges { id }
    pageInfo { count hasNextPage endCursor }
  }
}"""


def post(variables: dict) -> dict:
    body = json.dumps({"query": QUERY, "variables": variables}).encode()
    request = urllib.request.Request(ENDPOINT, body, {"content-type": "application/json",
                                                      "user-agent": "bids-eye-benchmark"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.load(response)
            if data.get("errors"):
                raise RuntimeError(data["errors"])
            return data["data"]["advancedSearch"]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    raise AssertionError


def fetch(query: dict) -> dict:
    ids, after, count = [], None, None
    while True:
        page = post({"q": query, "after": after})
        count = page["pageInfo"]["count"]
        ids += [edge["id"] for edge in page["edges"]]
        if not page["pageInfo"]["hasNextPage"] or not page["edges"]:
            break
        after = page["pageInfo"]["endCursor"]
    return {"count": count, "ids": sorted(set(ids))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=HERE / "facets.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    queries = yaml.safe_load(args.queries.read_text())["queries"]
    wanted = set(filter(None, args.only.split(",")))
    out = json.loads(args.out.read_text()) if args.out.exists() else {"fetched": None, "results": {}}
    for q in queries:
        if wanted and q["id"] not in wanted:
            continue
        # A list is the union of several selections: the search page has one
        # species at a time, so "mice or rats" is Mouse ∪ Rat.
        selections = q["openneuro"] if isinstance(q["openneuro"], list) else [q["openneuro"]]
        parts = [fetch(selection) for selection in selections]
        ids = sorted(set().union(*(part["ids"] for part in parts)))
        result = {"count": len(ids), "ids": ids}
        out["results"][q["id"]] = {"openneuro": q["openneuro"], **result}
        print(f"{q['id']:24} count={result['count']:5} ids={len(result['ids'])}", flush=True)
    out["fetched"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
