#!/usr/bin/env python3
"""
auto_map_unmapped.py
---------------------
LLM-assisted mapping of unmapped_db_values.yaml → value_mappings.yaml.

For each unmapped DB value the LLM returns one of five decisions:
  filter               — junk / noise (numeric group codes, internal IDs, punctuation, etc.)
  add_to_codes         — unambiguous alias for an existing entry; add to its 'codes' list
  add_to_dataset_codes — ambiguous tag whose meaning in these specific datasets maps to an
                         existing entry; adds a dataset-scoped entry to 'dataset_codes'
  new_entry            — genuinely new concept; create a new leaf under the given group
  new_entry_scoped     — new concept that only applies in specific datasets; create a new
                         leaf with a dataset-scoped 'dataset_codes' entry

When a tag means different things in different datasets, the LLM should emit
MULTIPLE decisions for the same raw value (one per meaning), using the scoped actions.
apply_canonical_codes.py will then update the DB with dataset-specific WHERE clauses.

Continuous-update mode (default)
---------------------------------
After each Gemini batch, decisions are immediately applied to value_mappings.yaml
and the updated tree is passed to subsequent batches.  This means Gemini always
sees the current state of the YAML — newly added entries are visible to the next batch.

Integration log
---------------
Every applied decision is appended to mapping_integration_log.jsonl (one JSON
object per line) so you can audit what was added and when.

Two-step workflow
-----------------
Step 1 — generate + apply interleaved (default, recommended):
    python auto_map_unmapped.py \\
        --unmapped unmapped_db_values.yaml \\
        --mappings value_mappings.yaml \\
        --checkpoint decisions.json

Step 2 — OR: generate only, review checkpoint, then apply separately:
    python auto_map_unmapped.py --generate-only …   # writes checkpoint
    python auto_map_unmapped.py --apply …           # reads checkpoint, updates YAML

Optional flags:
  --dataset-context dataset_context.json      # name+description per dataset (from find_unmapped)
  --sections diagnosis task suffix datatype   # limit to specific fields
  --batch-size 15                             # values per LLM call (default 10)
  --model gemini-3.1-pro                      # deprecated compatibility flag; canonical cascade is used
  --resume                                    # skip values already in checkpoint
  --generate-only                             # skip applying after each batch
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ── Gemini helpers ─────────────────────────────────────────────────────────────

MODEL_CASCADE = [
    # Das aktuelle Flaggschiff für Geschwindigkeit & Intelligenz (GA-Status)
    ("gemini-3.1-pro",         "Primary",    2, [30, 60]),

    # Der hocheffiziente Allrounder (stabil und kostengünstig)
    ("gemini-3.1-flash",       "Fallback 1", 2, [30, 60]),

    # Die Ultra-Leichtgewicht-Variante für einfache Aufgaben/Massendaten
    ("gemini-3.1-flash-lite",  "Fallback 2", 3, [60, 120, 240]),

    # Optional: Spezialmodell für komplexe Logik (Reasoning-Fokus)
    ("gemini-3-deep-think",    "Fallback 3", 5, [120, 240, 480]),
]

_MODEL_IDS = [model_id for model_id, *_ in MODEL_CASCADE]

def _gemini_client():
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
    return genai.Client(api_key=api_key)


def _call_gemini(client, _model: str, prompt: str) -> str:
    from google.genai import types
    model_cascade = list(_MODEL_IDS)

    for mdl in model_cascade:
        for attempt in range(5):
            try:
                resp = client.models.generate_content(
                    model=mdl,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                    contents=prompt,
                )
                if resp.text:
                    return resp.text
            except Exception as exc:
                err = str(exc)
                if "404" in err or "NOT_FOUND" in err:
                    break
                if "429" in err or "503" in err or "RESOURCE_EXHAUSTED" in err or "UNAVAILABLE" in err:
                    wait = 30.0 * (2 ** attempt)
                    print(f"    rate-limited, waiting {wait:.0f}s …")
                    time.sleep(wait)
                else:
                    print(f"    unexpected error: {err[:100]}")
                    break
    raise RuntimeError("Gemini call failed for all models in cascade.")


# ── YAML helpers ───────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, indent=2)


# ── Full tree summary for prompts ──────────────────────────────────────────────

_LEAF_KEYS = frozenset({
    "label", "description", "codes", "synonyms", "canonical_code",
    "standard_code", "extra_codes", "dataset_codes",
})


def _full_tree_summary(section_data: dict) -> str:
    """
    Return a compact hierarchy-only summary of the section, showing:
      - Group/sub-group names and their descriptions (no synonyms, no codes)
      - Leaf entry names and their labels/descriptions

    This is deliberately minimal: Gemini only needs to know what concepts already
    exist and where they live, not the full synonym/codes lists.  Keeping the
    summary small leaves token budget for the dataset context.

    Leaf nodes are identified by having 'standard_code'.
    Group nodes have no 'standard_code'.
    """
    lines: list[str] = []

    def walk(node: dict, indent: int = 0) -> None:
        pad = "  " * indent
        for key, val in node.items():
            if key in _LEAF_KEYS or not isinstance(val, dict):
                continue
            label = val.get("label", "")
            desc = val.get("description", "")
            hint = label or desc[:80] or ""
            if "standard_code" in val:
                # Leaf entry — show key, standard_code (if it differs from the key), and label
                sc = val.get("standard_code", "")
                sc_suffix = f" [standard_code={sc}]" if sc != key else ""
                lines.append(f"{pad}{key}{sc_suffix}: {hint}")
            else:
                # Group or sub-group — show key name with description if available
                lines.append(f"{pad}[group: {key}]" + (f"  — {hint}" if hint else ""))
                walk(val, indent + 1)

    walk(section_data)
    return "\n".join(lines) if lines else "(empty)"


def _extract_groups(section_data: dict) -> list[str]:
    """Return the top-level group names from a section (nodes without standard_code)."""
    return [
        k for k, v in section_data.items()
        if k not in _LEAF_KEYS and isinstance(v, dict) and "standard_code" not in v
    ]


def _collect_all_codes(section_data: dict) -> set[str]:
    """
    Return a flat set of every raw value already mapped in this section —
    i.e. everything in any 'codes' list or 'dataset_codes[].raw' field.
    Used to skip already-covered values before sending a batch to Gemini.
    """
    covered: set[str] = set()

    def walk(node: dict) -> None:
        if not isinstance(node, dict):
            return
        for c in node.get("codes", []):
            covered.add(str(c))
        for dc in node.get("dataset_codes", []):
            if isinstance(dc, dict) and dc.get("raw"):
                covered.add(str(dc["raw"]))
        for k, v in node.items():
            if k not in _LEAF_KEYS and isinstance(v, dict):
                walk(v)

    walk(section_data)
    return covered


# ── Prompt builder ─────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are a neuroimaging data curation expert. Your task is to classify raw database \
values for the '{section}' field and integrate them into a controlled vocabulary YAML.

═══ STRUCTURAL CONVENTIONS ═══
- Groups (concept categories) MUST NOT have 'codes' directly.
  If a group needs a catch-all / general entry, create a sub-entry named
  '<group>_general' (e.g. 'epilepsy_general' under 'epilepsy_spectrum').
- Every leaf entry MUST have: standard_code (snake_case), label (human-readable),
  and synonyms (3–6 natural-language terms a researcher might type when searching).
- standard_codes are lowercase snake_case, globally unique within this section.
- Group nodes have no standard_code or codes — they only organise leaf entries.
- Synonyms must NOT duplicate terms that exist in nearby entries.
  Use the existing tree to understand what already exists and avoid overlap.

═══ EXISTING VOCABULARY TREE (hierarchy + labels only) ═══
(Groups: [group: name] — description; Leaves: key: label — description)
{tree_summary}

Available top-level group names for new entries: {groups}

═══ RAW VALUES TO CLASSIFY ═══
Each entry includes: count (DB rows), and for each dataset that uses this value:
  accession_id, dataset name, and a description excerpt from the dataset README.
Use the dataset descriptions to understand what the raw value means in context.
{batch_json}

═══ TASK ═══
For EACH raw value, return one OR MORE JSON objects.
Return MULTIPLE objects for the SAME raw value when it has DIFFERENT meanings in
different datasets — one object per distinct meaning, each scoped to its datasets.

Each object must have:
  "raw": <the raw value, exactly as given>
  "action": one of:
    "filter"               — junk: opaque IDs, punctuation, uninterpretable abbreviations
    "add_to_codes"         — unambiguous global alias for an existing entry
    "add_to_dataset_codes" — alias whose meaning only applies in specific datasets
    "new_entry"            — new concept, appears consistently across datasets
    "new_entry_scoped"     — new concept, only meaningful in specific datasets
  "reasoning": one sentence

  If action == "add_to_codes":
    "target_standard_code": <the standard_code value of the existing entry — use the
      value shown in [standard_code=...] in the tree above, NOT the YAML key name>

  If action == "add_to_dataset_codes":
    (Only use this when the raw value is genuinely ambiguous — it means DIFFERENT things
    in different datasets. A value that appears in only one dataset but has a clear,
    unambiguous meaning should use "add_to_codes" instead.)
    "target_standard_code": <the standard_code value of the existing entry — use the
      value shown in [standard_code=...] in the tree above, NOT the YAML key name>
    "datasets": [<accession_ids where this mapping applies>]

  If action == "new_entry":
    "group": <MUST be one of the available group names above, or "other">
    "standard_code": <new unique snake_case code>
    "label": <short human-readable label>
    "description": <one sentence>
    "synonyms": [<3–6 natural-language search terms; no overlap with existing entries>]

  If action == "new_entry_scoped":
    "group": <MUST be one of the available group names above, or "other">
    "standard_code": <new unique snake_case code>
    "label": <short human-readable label>
    "description": <one sentence>
    "synonyms": [<3–6 natural-language search terms>]
    "datasets": [<accession_ids where this concept applies>]

FILTERING RULES — use "filter" for:
  - Purely numeric codes ("1", "2", "3") with no consistent cross-dataset meaning
  - Single letters or punctuation with no interpretable meaning
  - Internal pipeline / study codes interpretable only within one specific dataset
    UNLESS the dataset description makes the meaning clear and worth preserving
  - Values you cannot interpret even with dataset context

SYNONYM RULES:
  - Synonyms are natural-language phrases a researcher would type in a search box
  - Do NOT use the raw code itself or the standard_code as a synonym
  - Avoid redundancy with labels and descriptions already in the tree above

Return ONLY a JSON array, no markdown fences.
"""


def build_prompt(
    section: str,
    section_data: dict,
    batch: list[dict],
    dataset_context: dict[str, dict],
) -> str:
    """Build the Gemini classification prompt using the hierarchy tree and dataset context."""
    DESC_LIMIT = 400  # chars of description_text per dataset

    enriched_batch = []
    for item in batch:
        datasets_info = []
        for acc_id in item.get("datasets", []):
            ctx = dataset_context.get(acc_id, {})
            desc = ctx.get("description", "")
            entry: dict = {"accession_id": acc_id}
            if ctx.get("name"):
                entry["name"] = ctx["name"]
            if desc:
                entry["description"] = desc[:DESC_LIMIT] + ("…" if len(desc) > DESC_LIMIT else "")
            datasets_info.append(entry)
        enriched_batch.append({
            "raw": item["raw"],
            "count": item["count"],
            "datasets": datasets_info,
        })

    tree_summary = _full_tree_summary(section_data)
    groups = _extract_groups(section_data)
    groups_str = ", ".join(f'"{g}"' for g in groups) if groups else '"other"'

    return _PROMPT_TEMPLATE.format(
        section=section,
        tree_summary=tree_summary,
        groups=groups_str,
        batch_json=json.dumps(enriched_batch, indent=2),
    )


# ── Apply decisions to value_mappings.yaml ─────────────────────────────────────

def apply_decisions(
    decisions: dict[str, list[dict]],
    mappings: dict,
    log_path: Path | None = None,
) -> tuple[dict, dict[str, int]]:
    """
    Apply decisions to a copy of mappings.
    Returns (updated_mappings, stats).
    Optionally appends applied decisions to a JSONL integration log.
    """
    stats = {
        "filtered": 0, "add_to_codes": 0, "add_to_dataset_codes": 0,
        "new_entry": 0, "new_entry_scoped": 0, "skipped": 0,
    }

    log_entries: list[dict] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def find_entry_by_sc(section_data: dict, target_sc: str) -> dict | None:
        """Recursively find a leaf node with the given standard_code."""
        if not isinstance(section_data, dict):
            return None
        if section_data.get("standard_code") == target_sc:
            return section_data
        for k, v in section_data.items():
            if k not in _LEAF_KEYS and isinstance(v, dict):
                found = find_entry_by_sc(v, target_sc)
                if found is not None:
                    return found
        return None

    def find_sc_by_key(node: dict, key_target: str) -> str | None:
        """Find what standard_code a YAML key has (to diagnose key/standard_code mismatches)."""
        for k, v in node.items():
            if k not in _LEAF_KEYS and isinstance(v, dict):
                if k == key_target and "standard_code" in v:
                    return v["standard_code"]
                found = find_sc_by_key(v, key_target)
                if found:
                    return found
        return None

    def _append_dataset_code(entry: dict, raw: str, datasets: list[str]) -> None:
        """Append/merge a dataset-scoped raw→code mapping."""
        dc_list: list[dict] = entry.setdefault("dataset_codes", [])
        for dc in dc_list:
            if dc.get("raw") == raw:
                existing = set(dc.get("datasets", []))
                dc["datasets"] = sorted(existing | set(datasets))
                return
        dc_list.append({"raw": raw, "datasets": sorted(datasets)})

    for section, section_decisions in decisions.items():
        section_data = mappings.setdefault(section, {})

        for dec in section_decisions:
            raw = dec.get("raw", "")
            action = dec.get("action", "")
            log_base = {"section": section, "raw": raw, "action": action, "ts": ts}

            if action == "filter":
                stats["filtered"] += 1
                log_entries.append({**log_base, "reasoning": dec.get("reasoning", "")})

            elif action == "add_to_codes":
                target_sc = dec.get("target_standard_code", "")
                entry = find_entry_by_sc(section_data, target_sc)
                if entry is None:
                    hint = ""
                    actual_sc = find_sc_by_key(section_data, target_sc)
                    if actual_sc:
                        hint = f" (YAML key '{target_sc}' exists but its standard_code is '{actual_sc}')"
                    print(f"  WARN [{section}] add_to_codes: standard_code '{target_sc}' not found "
                          f"(raw={raw!r}){hint} — skipping")
                    stats["skipped"] += 1
                    continue
                codes: list = entry.setdefault("codes", [])
                if raw not in codes:
                    codes.append(raw)
                stats["add_to_codes"] += 1
                log_entries.append({**log_base, "target_standard_code": target_sc})

            elif action == "add_to_dataset_codes":
                target_sc = dec.get("target_standard_code", "")
                datasets = dec.get("datasets", [])
                if not datasets:
                    print(f"  WARN [{section}] add_to_dataset_codes: no datasets listed "
                          f"(raw={raw!r}) — skipping")
                    stats["skipped"] += 1
                    continue
                entry = find_entry_by_sc(section_data, target_sc)
                if entry is None:
                    hint = ""
                    actual_sc = find_sc_by_key(section_data, target_sc)
                    if actual_sc:
                        hint = f" (YAML key '{target_sc}' exists but its standard_code is '{actual_sc}')"
                    print(f"  WARN [{section}] add_to_dataset_codes: standard_code '{target_sc}' "
                          f"not found (raw={raw!r}){hint} — skipping")
                    stats["skipped"] += 1
                    continue
                _append_dataset_code(entry, raw, datasets)
                stats["add_to_dataset_codes"] += 1
                log_entries.append({**log_base, "target_standard_code": target_sc,
                                    "datasets": datasets})

            elif action == "new_entry":
                group = dec.get("group", "other")
                sc = dec.get("standard_code", "")
                label = dec.get("label", raw)
                description = dec.get("description", "")
                synonyms = dec.get("synonyms", [])
                if not sc:
                    print(f"  WARN [{section}] new_entry missing standard_code (raw={raw!r}) — skipping")
                    stats["skipped"] += 1
                    continue
                # Check if standard_code already exists anywhere in this section
                if find_entry_by_sc(section_data, sc) is not None:
                    print(f"  WARN [{section}] new_entry standard_code '{sc}' already exists "
                          f"(raw={raw!r}) — skipping")
                    stats["skipped"] += 1
                    continue
                group_node = section_data.setdefault(group, {})
                node: dict = {
                    "label": label,
                    "standard_code": sc,
                    "codes": [raw],
                }
                if description:
                    node["description"] = description
                if synonyms:
                    node["synonyms"] = synonyms
                group_node[sc] = node
                stats["new_entry"] += 1
                log_entries.append({**log_base, "standard_code": sc, "group": group,
                                    "label": label})

            elif action == "new_entry_scoped":
                group = dec.get("group", "other")
                sc = dec.get("standard_code", "")
                label = dec.get("label", raw)
                description = dec.get("description", "")
                synonyms = dec.get("synonyms", [])
                datasets = dec.get("datasets", [])
                if not sc:
                    print(f"  WARN [{section}] new_entry_scoped missing standard_code "
                          f"(raw={raw!r}) — skipping")
                    stats["skipped"] += 1
                    continue
                if not datasets:
                    print(f"  WARN [{section}] new_entry_scoped missing datasets "
                          f"(raw={raw!r}) — skipping")
                    stats["skipped"] += 1
                    continue
                group_node = section_data.setdefault(group, {})
                existing = find_entry_by_sc(section_data, sc)
                if existing is not None:
                    # Entry already exists — just add the dataset_codes entry
                    _append_dataset_code(existing, raw, datasets)
                else:
                    node = {
                        "label": label,
                        "standard_code": sc,
                        "dataset_codes": [{"raw": raw, "datasets": sorted(datasets)}],
                    }
                    if description:
                        node["description"] = description
                    if synonyms:
                        node["synonyms"] = synonyms
                    group_node[sc] = node
                stats["new_entry_scoped"] += 1
                log_entries.append({**log_base, "standard_code": sc, "group": group,
                                    "label": label, "datasets": datasets})

            else:
                print(f"  WARN [{section}] unknown action '{action}' for raw={raw!r} — skipping")
                stats["skipped"] += 1

    # Append to integration log
    if log_path is not None and log_entries:
        with log_path.open("a", encoding="utf-8") as lf:
            for entry in log_entries:
                lf.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return mappings, stats


# ── Decision generation ────────────────────────────────────────────────────────

def generate_decisions(
    unmapped: dict,
    mappings: dict,
    mappings_path: Path,
    checkpoint_path: Path,
    log_path: Path,
    sections: list[str],
    batch_size: int,
    model: str,
    resume: bool,
    generate_only: bool,
    dataset_context: dict[str, dict],
) -> dict[str, list[dict]]:
    """
    For each section × batch: call Gemini, immediately apply to mappings (unless
    generate_only=True), and save both checkpoint and mappings after every batch.

    Continuous-apply mode (default) means the YAML tree passed to later batches
    already contains entries added by earlier batches — Gemini always sees current state.
    """
    client = _gemini_client()

    # Load existing checkpoint
    decisions: dict[str, list[dict]] = {}
    if checkpoint_path.exists():
        decisions = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    for section in sections:
        section_values = unmapped.get(section, {})
        if not section_values:
            print(f"  [{section}] nothing unmapped — skipping")
            continue

        already_done: set[str] = set()
        if resume:
            already_done = {d["raw"] for d in decisions.get(section, [])}

        items = [
            {"raw": raw, "count": info["count"], "datasets": info["datasets"][:5]}
            for raw, info in section_values.items()
            if raw not in already_done
        ]

        if not items:
            print(f"  [{section}] all {len(section_values)} values already in checkpoint")
            continue

        print(f"  [{section}] {len(items)} values to classify "
              f"({len(already_done)} already in checkpoint)")

        decisions.setdefault(section, [])

        for i in range(0, len(items), batch_size):
            batch = items[i: i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(items) + batch_size - 1) // batch_size
            print(f"    batch {batch_num}/{total_batches}: {len(batch)} values …",
                  end=" ", flush=True)

            # Always use the CURRENT state of the section (updated by previous batches)
            section_data = mappings.get(section, {})
            prompt = build_prompt(section, section_data, batch, dataset_context)

            try:
                raw_resp = _call_gemini(client, model, prompt)
                raw_resp = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_resp.strip())
                batch_decisions = json.loads(raw_resp)
                if not isinstance(batch_decisions, list):
                    raise ValueError("expected JSON array")
            except Exception as exc:
                print(f"FAILED ({exc})")
                print("      Skipping batch — re-run with --resume to retry.")
                continue

            decisions[section].extend(batch_decisions)
            print(f"ok ({len(batch_decisions)} decisions)", end="")

            # Save checkpoint after every batch
            checkpoint_path.write_text(
                json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            if not generate_only:
                # Immediately apply this batch to the live mappings
                batch_dec = {section: batch_decisions}
                mappings, batch_stats = apply_decisions(batch_dec, mappings, log_path)
                save_yaml(mappings_path, mappings)
                applied = (batch_stats["add_to_codes"] + batch_stats["add_to_dataset_codes"]
                           + batch_stats["new_entry"] + batch_stats["new_entry_scoped"])
                print(f"  → applied {applied} "
                      f"(+codes={batch_stats['add_to_codes']}, "
                      f"+dc={batch_stats['add_to_dataset_codes']}, "
                      f"+new={batch_stats['new_entry']}, "
                      f"+scoped={batch_stats['new_entry_scoped']}, "
                      f"filtered={batch_stats['filtered']}, "
                      f"skipped={batch_stats['skipped']})")
            else:
                print()

    return decisions


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unmapped",   type=Path,
                        default=Path(__file__).with_name("unmapped_db_values.yaml"))
    parser.add_argument("--mappings",   type=Path,
                        default=Path(__file__).with_name("value_mappings.yaml"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).with_name("mapping_decisions.json"))
    parser.add_argument("--log",        type=Path,
                        default=Path(__file__).with_name("mapping_integration_log.jsonl"),
                        help="JSONL file to append applied decisions to (audit log)")
    parser.add_argument("--apply",      action="store_true",
                        help="Apply all decisions from checkpoint to value_mappings.yaml "
                             "(use this after --generate-only to apply in one go)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Call Gemini but do NOT apply to value_mappings.yaml after each "
                             "batch. Use --apply afterwards to apply from checkpoint.")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip values already present in checkpoint")
    parser.add_argument("--sections",   nargs="+",
                        default=["diagnosis", "task", "suffix", "datatype", "sex", "handedness"],
                        help="Which sections to process")
    parser.add_argument("--dataset-context", type=Path,
                        default=Path(__file__).with_name("dataset_context.json"),
                        help="JSON: accession_id → {name, description} "
                             "(written by find_unmapped_db_values.py)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Values per LLM call (default 10)")
    parser.add_argument(
        "--model",
        default="gemini-3.1-pro",
        help="Deprecated compatibility flag; the canonical MODEL_CASCADE is always used.",
    )
    args = parser.parse_args()

    unmapped = load_yaml(args.unmapped)
    mappings = load_yaml(args.mappings)

    # Load dataset context
    dataset_context: dict[str, dict] = {}
    if args.dataset_context.exists():
        dataset_context = json.loads(args.dataset_context.read_text(encoding="utf-8"))
        print(f"Loaded dataset context for {len(dataset_context)} dataset(s) "
              f"from {args.dataset_context}")
    else:
        print(f"Note: no dataset context file at {args.dataset_context} — "
              f"run find_unmapped_db_values.py first for richer Gemini context.")

    total_unmapped = sum(len(unmapped.get(s, {})) for s in args.sections)
    print(f"Unmapped values to process: {total_unmapped} across {args.sections}")

    if args.apply:
        # ── Apply-only mode ───────────────────────────────────────────────────
        if not args.checkpoint.exists():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
        decisions = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        total_decisions = sum(len(v) for v in decisions.values())
        print(f"Applying {total_decisions} decisions from {args.checkpoint} …")

        updated, stats = apply_decisions(decisions, mappings, args.log)
        save_yaml(args.mappings, updated)

        print(f"\nDone.")
        print(f"  filtered:             {stats['filtered']}")
        print(f"  add_to_codes:         {stats['add_to_codes']}")
        print(f"  add_to_dataset_codes: {stats['add_to_dataset_codes']}")
        print(f"  new_entry:            {stats['new_entry']}")
        print(f"  new_entry_scoped:     {stats['new_entry_scoped']}")
        print(f"  skipped:              {stats['skipped']}")
        print(f"\nvalue_mappings.yaml updated. Integration log: {args.log}")
        print(f"Re-run find_unmapped_db_values.py to verify coverage.")

    else:
        # ── Generate (+ continuous apply) mode ────────────────────────────────
        mode = "generate-only" if args.generate_only else "generate+apply"
        print(f"Mode: {mode} → checkpoint: {args.checkpoint}")
        if args.resume and args.checkpoint.exists():
            print(f"  Resuming from existing checkpoint.")
        if not args.generate_only:
            print(f"  Changes written to {args.mappings} after each batch.")
            print(f"  Integration log: {args.log}")

        decisions = generate_decisions(
            unmapped=unmapped,
            mappings=mappings,
            mappings_path=args.mappings,
            checkpoint_path=args.checkpoint,
            log_path=args.log,
            sections=args.sections,
            batch_size=args.batch_size,
            model=args.model,
            resume=args.resume,
            generate_only=args.generate_only,
            dataset_context=dataset_context,
        )

        total = sum(len(v) for v in decisions.values())
        print(f"\nDone. {total} decisions written to {args.checkpoint}")

        if args.generate_only:
            print(f"\nNext steps:")
            print(f"  1. Review {args.checkpoint} (edit any 'action' fields as needed)")
            print(f"  2. python auto_map_unmapped.py --apply")
        else:
            print(f"\nvalue_mappings.yaml is fully up to date.")
            print(f"Re-run find_unmapped_db_values.py to verify coverage.")


if __name__ == "__main__":
    main()
