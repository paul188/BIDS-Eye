#!/usr/bin/env python3
"""
Fill CA bridge coverage gaps for under-covered cognitive constructs.

Targets constructs that have ZERO reachable BIDS tasks even after
traversing the full CA child hierarchy.  For each target construct (and
all its CA children), fetches CA tasks that measure them and fuzzy-matches
those CA tasks to BIDS task codes.

Run from repo root:
    python backend/vocabulary/tools/fill_bridge_gaps.py

Merges results directly into backend/vocabulary/bids_to_ca_map.json.
Requires: rapidfuzz
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

import yaml
from rapidfuzz import fuzz, process as rfprocess

from cognitive_atlas_api import CognitiveAtlasFetchError, get_json as _get_json


_CA_BASE   = "https://www.cognitiveatlas.org"
_TASK_LIST = f"{_CA_BASE}/api/v-alpha/task"
_TASK_JSON = f"{_CA_BASE}/task/json/{{id}}/"

# Accept fuzzier matches when we know the CA task measures a target construct
_FUZZY_THRESHOLD_ANCHORED = 68  # lower than normal 72 since construct link is already known

# Constructs whose children have 0 bridge coverage
TARGET_ROOTS = [
    "temporal_cognition",
    "explicit_learning",
    "cardinal_direction_judgment",
    "episodic_future_thinking",
]


def _key(name: str) -> str:
    s = name.lower().strip()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _strip_suffix(name: str) -> str:
    for suffix in (" task", " paradigm", " test", " procedure", " assessment"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _all_descendants(root: str, children: Dict[str, List[str]]) -> Set[str]:
    visited: Set[str] = set()
    queue = [root]
    while queue:
        k = queue.pop()
        if k in visited:
            continue
        visited.add(k)
        queue.extend(children.get(k, []))
    return visited


def load_constructs(ca_yaml_path: str):
    with open(ca_yaml_path) as f:
        data = yaml.safe_load(f)
    cc = data.get("cognitive_construct", {})
    # Build children map
    children: Dict[str, List[str]] = defaultdict(list)
    for k, v in cc.items():
        for p in (v.get("broader") or []) + (v.get("part_of") or []):
            children[p].append(k)
    return cc, children


def load_bids_tasks(yaml_path: str) -> Dict[str, str]:
    with open(yaml_path) as f:
        schema = yaml.safe_load(f)
    tasks = schema.get("task", {})
    return {k: v.get("label", k) for k, v in tasks.items() if isinstance(v, dict)}


def main(yaml_path: str, ca_yaml_path: str, bridge_path: str) -> None:
    # ── Load existing bridge ──────────────────────────────────────────────────
    with open(bridge_path) as f:
        bridge: Dict[str, dict] = json.load(f)
    existing_ca_ids = {v["ca_task_id"] for v in bridge.values()}
    print(f"Existing bridge: {len(bridge)} entries  ({len(existing_ca_ids)} unique CA task IDs)")

    # ── Build target construct set ────────────────────────────────────────────
    cc, ca_children = load_constructs(ca_yaml_path)
    target_keys: Set[str] = set()
    for root in TARGET_ROOTS:
        target_keys.update(_all_descendants(root, ca_children))

    target_ca_ids: Dict[str, str] = {
        cc[k]["ca_id"]: k
        for k in target_keys
        if cc.get(k, {}).get("ca_id")
    }
    print(f"Target constructs: {len(target_keys)} keys  ({len(target_ca_ids)} CA IDs)")

    # ── Build BIDS fuzzy pool ─────────────────────────────────────────────────
    bids_tasks = load_bids_tasks(yaml_path)
    fuzzy_pool: Dict[str, str] = {}
    for code, label in bids_tasks.items():
        if not label:
            continue
        fuzzy_pool[label.lower()] = code
        fuzzy_pool[code.replace("_", " ")] = code
        stripped = _strip_suffix(label).lower()
        if stripped != label.lower():
            fuzzy_pool[stripped] = code

    # ── Fetch CA task list ────────────────────────────────────────────────────
    print("Fetching CA task list …")
    try:
        ca_tasks_raw = _get_json(_TASK_LIST)
    except CognitiveAtlasFetchError as exc:
        sys.exit(f"ERROR: could not fetch CA task list: {exc}")
    if not ca_tasks_raw or not isinstance(ca_tasks_raw, list):
        print("ERROR: could not fetch CA task list", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(ca_tasks_raw)} CA tasks total")

    # ── Scan CA tasks for target construct links ──────────────────────────────
    new_entries: Dict[str, dict] = {}
    scanned = 0
    failed: List[str] = []

    for i, ca_task in enumerate(ca_tasks_raw):
        ca_id   = ca_task.get("id", "")
        ca_name = ca_task.get("name", "")
        if not ca_id or not ca_name:
            continue

        # Skip tasks already in the bridge
        if ca_id in existing_ca_ids:
            continue

        if i % 100 == 0:
            print(f"  {i}/{len(ca_tasks_raw)}: {ca_name}")

        # Fetch full task JSON to check its concepts
        try:
            task_full = _get_json(_TASK_JSON.format(id=ca_id))
        except CognitiveAtlasFetchError as exc:
            print(f"  [WARN] {exc}", file=sys.stderr)
            failed.append(ca_id)
            continue
        time.sleep(0.05)
        scanned += 1

        if not task_full or not isinstance(task_full, dict):
            continue

        # Which target constructs does this CA task measure?
        measures_set: Set[str] = set()
        for concept in task_full.get("concepts") or []:
            ck = target_ca_ids.get(concept.get("concept_id", ""))
            if ck:
                measures_set.add(ck)

        if not measures_set:
            continue  # not relevant to our gap constructs

        measures = sorted(measures_set)
        print(f"  ← '{ca_name}' measures {measures}")

        # Fuzzy-match to BIDS
        alias_raw = ca_task.get("alias") or ""
        candidates = [ca_name, _strip_suffix(ca_name)]
        for a in alias_raw.split(","):
            a = a.strip()
            if a:
                candidates += [a, _strip_suffix(a)]
        candidates = list(dict.fromkeys(c.lower() for c in candidates if c))

        best_code: Optional[str] = None
        best_score = 0.0
        best_via = ca_name

        for cand in candidates:
            hits = rfprocess.extract(cand, fuzzy_pool.keys(), scorer=fuzz.WRatio, limit=3)
            if hits and hits[0][1] > best_score:
                best_score = hits[0][1]
                best_code  = fuzzy_pool[hits[0][0]]
                best_via   = cand

        if best_code is None or best_score < _FUZZY_THRESHOLD_ANCHORED:
            print(f"    [no match] best={best_score:.0f}")
            continue

        print(f"    → {best_code} (score={best_score:.0f}, via={best_via!r})")
        entry = {
            "ca_task_id":   ca_id,
            "ca_task_name": ca_name,
            "matched_via":  best_via,
            "confidence":   round(best_score / 100.0, 3),
            "measures":     measures,
        }
        prev = new_entries.get(best_code) or bridge.get(best_code)
        if prev is None or prev["confidence"] < entry["confidence"]:
            new_entries[best_code] = entry

    print(f"\nScanned {scanned} new CA tasks  → {len(new_entries)} new BIDS matches")
    # The bridge only gains entries here, so what was found is still written;
    # the exit status says that a run again may find more.
    if failed:
        print(f"WARNING: {len(failed)} CA task(s) could not be read and were not scanned: "
              f"{', '.join(failed[:10])}{' …' if len(failed) > 10 else ''}. Run again.",
              file=sys.stderr)

    if not new_entries:
        if failed:
            sys.exit(1)
        print("No new entries found — bridge is already as complete as the CA API allows.")
        return

    # ── Merge into bridge ─────────────────────────────────────────────────────
    updated = 0
    for code, entry in new_entries.items():
        if code not in bridge:
            bridge[code] = entry
            updated += 1
            print(f"  + {code}: {entry['ca_task_name']}  measures={entry['measures']}")
        else:
            # Extend measures of existing entry
            existing_measures = set(bridge[code].get("measures") or [])
            new_measures = set(entry["measures"]) - existing_measures
            if new_measures:
                bridge[code]["measures"] = sorted(existing_measures | new_measures)
                updated += 1
                print(f"  ~ {code}: added measures {sorted(new_measures)}")

    with open(bridge_path, "w", encoding="utf-8") as f:
        json.dump(bridge, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(bridge)} entries to {bridge_path}  ({updated} changed)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml",          default="backend/vocabulary/value_mappings.yaml")
    parser.add_argument("--ca-constructs", default="backend/vocabulary/cognitive_atlas_constructs.yaml")
    parser.add_argument("--bridge",        default="backend/vocabulary/bids_to_ca_map.json")
    args = parser.parse_args()
    main(args.yaml, args.ca_constructs, args.bridge)
