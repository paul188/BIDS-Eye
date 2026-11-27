#!/usr/bin/env python3
"""
Fetch all Cognitive Atlas concepts and write backend/vocabulary/cognitive_atlas_constructs.yaml.

Run from the repo root:
    python backend/vocabulary/tools/import_cognitive_atlas.py

Output: backend/vocabulary/cognitive_atlas_constructs.yaml  (~900 concepts, ~1 800 hierarchy edges)

The file is a drop-in for yaml_to_llamaindex.py which loads it alongside
value_mappings.yaml.  Re-run any time the CA API is updated.

Requires: nothing beyond PyYAML (the API is read with urllib)
"""
import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import yaml

from cognitive_atlas_api import CognitiveAtlasFetchError, get_json as _get_json


_CA_BASE = "https://www.cognitiveatlas.org"
_CONCEPT_LIST = f"{_CA_BASE}/api/v-alpha/concept"
_CONCEPT_JSON = f"{_CA_BASE}/concept/json/{{id}}/"


def _key(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def main(out_path: str) -> None:
    # ── 1. Fetch concept list ────────────────────────────────────────────────
    print("Fetching concept list …")
    try:
        raw = _get_json(_CONCEPT_LIST)
    except CognitiveAtlasFetchError as exc:
        sys.exit(f"ERROR: could not fetch concept list: {exc}")
    if not raw or not isinstance(raw, list):
        print("ERROR: could not fetch concept list", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(raw)} concepts")

    # Build id → slug-key mapping before fetching details so parent references
    # can be resolved when we process each concept.
    id_to_key: Dict[str, str] = {}
    for c in raw:
        cid, name = c.get("id", ""), c.get("name", "")
        if cid and name:
            id_to_key[cid] = _key(name)

    # Detect and disambiguate key collisions (rare in practice).
    key_counts = Counter(id_to_key.values())
    duplicated = {k for k, n in key_counts.items() if n > 1}
    if duplicated:
        # Append CA id suffix to distinguish collisions.
        seen: Dict[str, int] = {}
        for cid, k in list(id_to_key.items()):
            if k in duplicated:
                idx = seen.get(k, 0)
                id_to_key[cid] = f"{k}_{idx}" if idx else k
                seen[k] = idx + 1

    # ── 2. Fetch per-concept JSON and build YAML nodes ───────────────────────
    print("Fetching concept details …")
    constructs: Dict[str, dict] = {}
    failed: List[str] = []

    for i, c in enumerate(raw):
        cid = c.get("id", "")
        name = c.get("name", "")
        if not cid or not name:
            continue

        key = id_to_key[cid]
        if i % 50 == 0:
            print(f"  {i}/{len(raw)}: {name}")

        try:
            full = _get_json(_CONCEPT_JSON.format(id=cid))
        except CognitiveAtlasFetchError as exc:
            print(f"  [WARN] {exc}", file=sys.stderr)
            failed.append(cid)
            continue
        time.sleep(0.05)

        definition_text: str = c.get("definition_text") or ""
        alias_raw: str = c.get("alias") or ""

        if full and isinstance(full, dict):
            definition_text = full.get("definition_text") or definition_text
            alias_raw = full.get("alias") or alias_raw

        # Aliases → weighted synonyms (CA community-contributed, so weight 0.7)
        aliases = [a.strip() for a in alias_raw.split(",") if a.strip()]
        synonyms = [
            {"term": a, "weight": 0.7}
            for a in aliases
            if a.lower() != name.lower()
        ]

        # Relation edges from full JSON
        broader: List[str] = []   # KINDOF parents  (is-a)
        part_of: List[str] = []   # PARTOF parents  (mereological)

        if full and isinstance(full, dict):
            for rel in full.get("relationships") or []:
                if rel.get("direction") != "parent":
                    continue
                rel_type = rel.get("relationship", "")
                pid = rel.get("id", "")
                parent_key = id_to_key.get(pid) or _key(rel.get("name", ""))
                if not parent_key:
                    continue
                if rel_type == "KINDOF" and parent_key not in broader:
                    broader.append(parent_key)
                elif rel_type == "PARTOF" and parent_key not in part_of:
                    part_of.append(parent_key)

        node: dict = {
            "label": name,
            "standard_code": key,
            "ca_id": cid,
        }
        if definition_text:
            node["definition_text"] = definition_text
        if broader:
            node["broader"] = broader
        if part_of:
            node["part_of"] = part_of
        if synonyms:
            node["synonyms"] = synonyms

        constructs[key] = node

    # A concept whose JSON could not be read would be written without its
    # broader / part_of hierarchy, into a file that replaces the old one.
    if failed:
        sys.exit(f"ERROR: {len(failed)} concept(s) could not be read ({', '.join(failed[:10])}"
                 f"{' …' if len(failed) > 10 else ''}); {out_path} is unchanged. Run again.")

    # ── 3. Write YAML ────────────────────────────────────────────────────────
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        yaml.dump(
            {"cognitive_construct": constructs},
            fh,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    n_broader  = sum(1 for v in constructs.values() if v.get("broader"))
    n_part_of  = sum(1 for v in constructs.values() if v.get("part_of"))
    n_synonyms = sum(1 for v in constructs.values() if v.get("synonyms"))
    print(f"\nWrote {len(constructs)} concepts → {out_file}")
    print(f"  KINDOF (broader) edges : {n_broader}")
    print(f"  PARTOF (part_of) edges : {n_part_of}")
    print(f"  Nodes with aliases     : {n_synonyms}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="backend/vocabulary/cognitive_atlas_constructs.yaml")
    args = parser.parse_args()
    main(args.out)
