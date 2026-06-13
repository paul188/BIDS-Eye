#!/usr/bin/env python3
"""
Generate a proposed mapping between BIDS-Eye task codes and Cognitive Atlas tasks.

Run AFTER import_cognitive_atlas.py so that cognitive_atlas_constructs.yaml exists.

Run from the repo root:
    python scripts/generate_task_bridge.py

Output: RAG/proposed_task_mappings.json

Next step (manual):
    Review the file.  Correct wrong matches, remove noise, add missing entries.
    Then rename / copy to RAG/bids_to_ca_map.json and commit it.
    yaml_to_llamaindex.py loads that file at startup to build the measures_db
    reverse index (cognitive construct → task codes).

Requires: requests, rapidfuzz  (pip install requests rapidfuzz)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import yaml
from rapidfuzz import fuzz, process as rfprocess

try:
    import requests as _requests

    def _get_json(url: str, retries: int = 5) -> Optional[dict | list]:
        for attempt in range(retries):
            try:
                r = _requests.get(url, timeout=30)
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 2 ** (attempt + 1)))
                    print(f"  [429] rate limited — waiting {wait}s …", file=sys.stderr)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                if attempt == retries - 1:
                    print(f"  [WARN] {url}: {exc}", file=sys.stderr)
                    return None
                time.sleep(2 ** attempt)
        return None

except ImportError:
    import json as _json
    import urllib.request
    import urllib.error

    def _get_json(url: str, retries: int = 5) -> Optional[dict | list]:  # type: ignore[misc]
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    return _json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    wait = int(exc.headers.get("Retry-After", 2 ** (attempt + 1)))
                    print(f"  [429] rate limited — waiting {wait}s …", file=sys.stderr)
                    time.sleep(wait)
                    continue
                if attempt == retries - 1:
                    print(f"  [WARN] {url}: {exc}", file=sys.stderr)
                    return None
                time.sleep(2 ** attempt)
            except Exception as exc:
                if attempt == retries - 1:
                    print(f"  [WARN] {url}: {exc}", file=sys.stderr)
                    return None
                time.sleep(2 ** attempt)
        return None


_CA_BASE = "https://www.cognitiveatlas.org"
_TASK_LIST = f"{_CA_BASE}/api/v-alpha/task"
_TASK_JSON = f"{_CA_BASE}/task/json/{{id}}/"

# Only accept matches above this threshold (0-100 WRatio scale)
_FUZZY_THRESHOLD = 72


def _key(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _strip_task_suffix(name: str) -> str:
    """'Color-Word Stroop Task' → 'Color-Word Stroop'."""
    for suffix in (" task", " paradigm", " test", " procedure", " assessment"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_bids_tasks(yaml_path: str) -> Dict[str, str]:
    """Return {standard_code: label}."""
    with open(yaml_path) as fh:
        schema = yaml.safe_load(fh)
    tasks = schema.get("task", {})
    return {k: v.get("label", k) for k, v in tasks.items() if isinstance(v, dict)}


def load_ca_constructs(ca_yaml_path: str) -> Dict[str, str]:
    """Return {ca_id: construct_key}."""
    if not Path(ca_yaml_path).exists():
        return {}
    with open(ca_yaml_path) as fh:
        data = yaml.safe_load(fh)
    constructs = data.get("cognitive_construct", {})
    return {v.get("ca_id", ""): k for k, v in constructs.items() if isinstance(v, dict) and v.get("ca_id")}


def main(yaml_path: str, ca_yaml_path: str, out_path: str) -> None:
    # ── Load BIDS tasks ──────────────────────────────────────────────────────
    print("Loading BIDS task codes …")
    bids_tasks = load_bids_tasks(yaml_path)
    print(f"  {len(bids_tasks)} task codes")

    # Build fuzzy pool: multiple surface forms → standard_code
    fuzzy_pool: Dict[str, str] = {}
    for code, label in bids_tasks.items():
        fuzzy_pool[label.lower()] = code
        fuzzy_pool[code.replace("_", " ")] = code
        # Also stripped suffix variant
        stripped = _strip_task_suffix(label).lower()
        if stripped != label.lower():
            fuzzy_pool[stripped] = code

    # ── Load CA constructs for measures lookup ───────────────────────────────
    print("Loading CA constructs …")
    ca_id_to_key = load_ca_constructs(ca_yaml_path)
    if not ca_id_to_key:
        print("  [WARN] cognitive_atlas_constructs.yaml not found — measures will be empty")
        print("         Run scripts/import_cognitive_atlas.py first.")

    # ── Fetch CA task list ───────────────────────────────────────────────────
    print("Fetching CA task list …")
    ca_tasks_raw = _get_json(_TASK_LIST)
    if not ca_tasks_raw or not isinstance(ca_tasks_raw, list):
        print("ERROR: could not fetch CA task list", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(ca_tasks_raw)} CA tasks")

    results: Dict[str, dict] = {}

    print("Matching tasks and fetching concept links …")
    for i, ca_task in enumerate(ca_tasks_raw):
        ca_id = ca_task.get("id", "")
        ca_name = ca_task.get("name", "")
        if not ca_id or not ca_name:
            continue

        if i % 100 == 0:
            print(f"  {i}/{len(ca_tasks_raw)}: {ca_name}")

        # Build candidate strings to fuzzy-match
        alias_raw = ca_task.get("alias") or ""
        candidates = [ca_name, _strip_task_suffix(ca_name)]
        for a in alias_raw.split(","):
            a = a.strip()
            if a:
                candidates += [a, _strip_task_suffix(a)]
        candidates = list(dict.fromkeys(c.lower() for c in candidates if c))

        # Best fuzzy match across all candidates
        best_code: Optional[str] = None
        best_score = 0.0
        best_via = ca_name

        for cand in candidates:
            hits = rfprocess.extract(cand, fuzzy_pool.keys(), scorer=fuzz.WRatio, limit=3)
            if hits and hits[0][1] > best_score:
                best_score = hits[0][1]
                best_code = fuzzy_pool[hits[0][0]]
                best_via = cand

        if best_code is None or best_score < _FUZZY_THRESHOLD:
            continue

        # Fetch full task JSON for concept links (via contrasts)
        task_full = _get_json(_TASK_JSON.format(id=ca_id))
        time.sleep(0.05)

        measures_set: set[str] = set()
        if task_full and isinstance(task_full, dict):
            # Top-level concepts array: field is "concept_id" (CA schema)
            for concept in task_full.get("concepts") or []:
                ck = ca_id_to_key.get(concept.get("concept_id", ""))
                if ck:
                    measures_set.add(ck)
        measures = sorted(measures_set)

        # Skip low-confidence matches with no concept links (not useful)
        if not measures and best_score < 85:
            continue

        entry = {
            "ca_task_id": ca_id,
            "ca_task_name": ca_name,
            "matched_via": best_via,
            "confidence": round(best_score / 100.0, 3),
            "measures": measures,
        }

        # Keep best match per BIDS code (highest confidence wins)
        prev = results.get(best_code)
        if prev is None or prev["confidence"] < entry["confidence"]:
            results[best_code] = entry

    # ── Write output ─────────────────────────────────────────────────────────
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    n_with = sum(1 for v in results.values() if v["measures"])
    print(f"\nWrote {len(results)} mappings → {out_file}")
    print(f"  {n_with} with concept links (measures)")
    print(f"  {len(results) - n_with} without concept links")
    print(
        f"\nNext: review {out_file}, correct errors, "
        "then rename to RAG/bids_to_ca_map.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", default="RAG/value_mappings.yaml")
    parser.add_argument("--ca-constructs", default="RAG/cognitive_atlas_constructs.yaml")
    parser.add_argument("--out", default="RAG/proposed_task_mappings.json")
    args = parser.parse_args()
    main(args.yaml, args.ca_constructs, args.out)
