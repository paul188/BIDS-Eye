"""
RAG/fold_general_leaves.py
--------------------------
Restructure the 11 audited "general-only" leaves (general_leaves_to_merge.txt)
into their broader category, per the agreed design:

  A) 7 single-leaf groups -> the GROUP becomes a "dual node": it absorbs the
     leaf's standard_code + codes + synonyms (+label as a synonym) and the leaf
     entry is deleted. The standard_code VALUE is preserved on the group, so the
     DB rows still resolve and NO database change is needed.

  B) memory: fold `memory_task_general` into the retrieval sub-group
     `memory_retrieval_general` (which is kept, it parents 36 recall tasks).
     memory_task_general is deleted; its 2 DB rows must be remapped to
     memory_retrieval_general  ->  PRODUCTION DB MERGE (printed at the end).

  C) ALS -> MND > ALS > subtypes:
       - motor_neuron_disease_general becomes the umbrella GROUP (keeps its 1 DB
         row as a dual node).
       - als_general is merged INTO amyotrophic_lateral_sclerosis (the phantom
         empty group): that node gets standard_code=als_general (preserving the 1
         DB row, no remap) + als_general's synonyms/codes, broader=[MND].
       - als_upper_limb_dominant re-parents under amyotrophic_lateral_sclerosis.
       - the now-empty als_spectrum group is deleted.

Load/dump mirrors RAG/merge_proposals.py (yaml.safe_load -> yaml.dump width=80)
so the git diff stays localized. Run with --dry-run first.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# A) (section, leaf, group)  group -> dual node, leaf deleted, NO db change
SINGLE_FOLDS: List[Tuple[str, str, str]] = [
    ("task",   "recognition_task_general", "recognition"),
    ("task",   "learning_task_general",    "associative"),
    ("task",   "perception_general",       "sensory_and_perception"),
    ("task",   "social_task_general",      "social_cognition"),
    ("task",   "spatial_memory",           "spatial_memory_group"),
    ("task",   "theory_of_mind",           "theory_of_mind_group"),
    ("suffix", "field_map_general",        "field_maps"),
]


def _pairs(raw: Any) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for s in raw or []:
        if isinstance(s, str):
            t = s.strip()
            if t:
                out.append((t, 1.0))
        elif isinstance(s, dict) and "term" in s:
            t = str(s["term"]).strip()
            if t:
                out.append((t, float(s.get("weight", 1.0))))
    return out


def _fmt(term: str, weight: float) -> Any:
    return term if weight >= 1.0 else {"term": term, "weight": round(weight, 2)}


def _merge_synonyms(target: dict, leaf: dict) -> None:
    """Append leaf's label (as a synonym) + leaf's synonyms onto target, deduped."""
    existing = _pairs(target.get("synonyms"))
    seen = {t.lower() for t, _ in existing}
    incoming: List[Tuple[str, float]] = []
    label = (leaf.get("label") or "").strip()
    if label:
        incoming.append((label, 1.0))
    incoming.extend(_pairs(leaf.get("synonyms")))
    merged = list(existing)
    for t, w in incoming:
        if t.lower() not in seen:
            merged.append((t, w))
            seen.add(t.lower())
    if merged:
        target["synonyms"] = [_fmt(t, w) for t, w in merged]


def _merge_codes(target: dict, leaf: dict) -> None:
    codes = list(target.get("codes") or [])
    for c in leaf.get("codes") or []:
        if c not in codes:
            codes.append(c)
    if codes:
        target["codes"] = codes
    if leaf.get("dataset_codes") and not target.get("dataset_codes"):
        target["dataset_codes"] = leaf["dataset_codes"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", default=str(Path(__file__).parent / "value_mappings.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    yp = Path(args.yaml)
    schema = yaml.safe_load(open(yp, encoding="utf-8"))
    diag, task, suf = schema["diagnosis"], schema["task"], schema["suffix"]

    # ── A) single-leaf dual-node folds ────────────────────────────────────────
    for section, leaf_key, grp_key in SINGLE_FOLDS:
        sec = schema[section]
        leaf, grp = sec[leaf_key], sec[grp_key]
        _merge_synonyms(grp, leaf)
        _merge_codes(grp, leaf)
        grp["standard_code"] = leaf["standard_code"]      # group becomes dual node
        if not grp.get("label") and leaf.get("label"):
            grp["label"] = leaf["label"]
        del sec[leaf_key]
        print(f"  A  {section}/{leaf_key:28s} -> dual-node {grp_key}  (no DB change)")

    # ── B) memory: fold memory_task_general into memory_retrieval_general ──────
    mtg = task["memory_task_general"]
    mrg = task["memory_retrieval_general"]              # kept as the retrieval sub-group
    _merge_synonyms(mrg, mtg)
    _merge_codes(mrg, mtg)
    del task["memory_task_general"]
    print("  B  task/memory_task_general       -> folded into memory_retrieval_general"
          "  (DB MERGE: 2 rows)")

    # ── C) ALS: MND > ALS > subtypes ──────────────────────────────────────────
    mnd = diag["motor_neuron_disease_general"]
    als = diag["amyotrophic_lateral_sclerosis"]
    alsg = diag["als_general"]
    # MND becomes the umbrella group (dual node keeps its own std_code + 1 DB row)
    mnd["is_group"] = True
    mnd_broader = list(mnd.get("broader") or [])
    if "spinal_neuromuscular" not in mnd_broader:
        mnd_broader.append("spinal_neuromuscular")
    mnd["broader"] = mnd_broader
    # merge als_general INTO amyotrophic_lateral_sclerosis (preserve als_general std/row)
    _merge_synonyms(als, alsg)
    _merge_codes(als, alsg)
    als["standard_code"] = alsg["standard_code"]        # = 'als_general' (no DB remap)
    als["is_group"] = True
    als["broader"] = ["motor_neuron_disease_general"]
    if not als.get("description") and alsg.get("description"):
        als["description"] = alsg["description"]
    del diag["als_general"]
    # re-parent ALS subtypes from als_spectrum -> amyotrophic_lateral_sclerosis
    for k, v in diag.items():
        if isinstance(v, dict) and "als_spectrum" in (v.get("broader") or []):
            v["broader"] = ["amyotrophic_lateral_sclerosis" if b == "als_spectrum" else b
                            for b in v["broader"]]
    del diag["als_spectrum"]                            # now empty
    print("  C  ALS -> MND(group) > amyotrophic_lateral_sclerosis(std=als_general) > subtypes"
          "  (no DB change)")

    print("\n==> PRODUCTION DB MERGE to apply on Hetzner later:")
    print("    UPDATE bids_objects SET task='memory_retrieval_general' "
          "WHERE task='memory_task_general';   -- 2 rows")

    if args.dry_run:
        print("\n(dry-run — no file written)")
        return

    fd, tmp = tempfile.mkstemp(dir=yp.parent, prefix=".fold_tmp_", suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.dump(schema, fh, default_flow_style=False, allow_unicode=True,
                      sort_keys=False, width=80)
        os.replace(tmp, yp)
        print(f"\nWritten: {yp}")
    except Exception as exc:
        os.unlink(tmp)
        sys.exit(f"Error writing file: {exc}")


if __name__ == "__main__":
    main()
