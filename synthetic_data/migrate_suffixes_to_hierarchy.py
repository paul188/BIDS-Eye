#!/usr/bin/env python3
"""
migrate_suffixes_to_hierarchy.py

Iteratively classifies unmapped BIDS suffix values (from unmapped_db_values.yaml)
into the canonical suffix hierarchy in value_mappings.yaml by asking Gemini.

For each batch Gemini returns one of three actions per raw value:
  add_code   — append the raw value to an existing leaf's 'codes' list
  new_entry  — create a new leaf (in an existing or new category)
  skip       — non-portable / dataset-specific; remove from unmapped list

Authentication:
  export GEMINI_API_KEY=...   or   export GOOGLE_API_KEY=...

Requires:
  pip install google-genai pyyaml

Usage:
  python migrate_suffixes_to_hierarchy.py [--batch 50] [--dry-run] [--loop]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent
MAPPINGS_YAML   = HERE / "value_mappings.yaml"
UNMAPPED_YAML   = HERE / "unmapped_db_values.yaml"
SKIP_SIDECAR    = HERE / "confirmed_skip_suffixes.json"

# ---------------------------------------------------------------------------
# Model cascade
# ---------------------------------------------------------------------------

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

SYSTEM_INSTRUCTION = (
    "You are a neuroimaging data expert helping to organise BIDS file suffixes "
    "into a canonical hierarchy. You will receive the existing hierarchy and a "
    "list of raw suffix strings found in a database. "
    "For each raw value decide one of three actions:\n"
    "  1. add_code   — the value is an alias/variant of an existing leaf; "
    "append it to that leaf's codes list.\n"
    "  2. new_entry  — the value represents a genuinely new concept; create a "
    "new leaf under an existing category, or under a new category if nothing fits.\n"
    "  3. skip       — the value is dataset-specific, auto-generated, "
    "non-portable, or otherwise not useful (e.g. subject IDs, session labels, "
    "numeric indices).\n"
    "Return ONLY a JSON array — no markdown fences, no prose. "
    "Each element must follow the schema described in the prompt."
)

# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

class _NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, _data: object) -> bool:
        return True


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(
            data,
            fh,
            Dumper=_NoAliasDumper,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            width=10000,
        )


def backup_once(path: Path) -> None:
    bak = path.with_suffix(".yaml.bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  Backup → {bak}")


# ---------------------------------------------------------------------------
# Sidecar for confirmed skips
# ---------------------------------------------------------------------------

def load_skips(path: Path) -> set[str]:
    return set(json.loads(path.read_text())) if path.exists() else set()


def save_skips(path: Path, skips: set[str]) -> None:
    path.write_text(json.dumps(sorted(skips), indent=2))


# ---------------------------------------------------------------------------
# Hierarchy helpers
# ---------------------------------------------------------------------------

def build_hierarchy_summary(suffix_section: dict) -> str:
    """Compact text representation of the existing suffix hierarchy for the prompt."""
    lines = ["Existing suffix hierarchy (category_key > leaf_key | label | codes):"]
    for cat_key, cat_val in suffix_section.items():
        if not isinstance(cat_val, dict):
            continue
        lines.append(f"\n{cat_key}:")
        for leaf_key, leaf in cat_val.items():
            if not isinstance(leaf, dict):
                continue
            label = leaf.get("label", leaf_key)
            codes = leaf.get("codes", [])
            lines.append(f"  {leaf_key} | {label} | codes: {codes}")
    return "\n".join(lines)


def get_or_create_category(suffix_section: dict, cat_key: str, cat_label: str | None) -> dict:
    if cat_key not in suffix_section:
        suffix_section[cat_key] = {}
        if cat_label:
            suffix_section[cat_key]["label"] = cat_label
    return suffix_section[cat_key]


def find_leaf(suffix_section: dict, dot_path: str) -> dict | None:
    """Resolve 'category.leaf_key' to the leaf dict, or None."""
    parts = dot_path.split(".", 1)
    if len(parts) != 2:
        return None
    cat, leaf = parts
    cat_dict = suffix_section.get(cat)
    if not isinstance(cat_dict, dict):
        return None
    leaf_dict = cat_dict.get(leaf)
    return leaf_dict if isinstance(leaf_dict, dict) else None


# ---------------------------------------------------------------------------
# Batch collection
# ---------------------------------------------------------------------------

def collect_batch(
    unmapped_suffix: dict,
    skips: set[str],
    batch_size: int,
) -> list[tuple[str, dict]]:
    """Return up to batch_size unprocessed (raw_value, entry_dict) pairs."""
    result = []
    for key, val in unmapped_suffix.items():
        if key in skips:
            continue
        result.append((key, val))
        if len(result) >= batch_size:
            break
    return result


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(hierarchy_summary: str, batch: list[tuple[str, dict]]) -> str:
    items = []
    for raw_val, entry in batch:
        items.append({
            "raw_value": raw_val,
            "count": entry.get("count"),
            "example_datasets": (entry.get("datasets") or [])[:3],
        })

    schema = (
        "Each element must be ONE of:\n"
        '  {"raw_value":"...", "action":"add_code",  "target":"<category_key>.<leaf_key>"}\n'
        '  {"raw_value":"...", "action":"new_entry",  "category":"<existing_or_new_category_key>", '
        '"category_label":"<label if new category, else omit>", '
        '"label":"<human label>", "standard_code":"<snake_case_key>"}\n'
        '  {"raw_value":"...", "action":"skip"}\n'
    )

    return (
        f"{hierarchy_summary}\n\n"
        f"Raw suffix values to classify:\n"
        f"{json.dumps(items, indent=2, ensure_ascii=False)}\n\n"
        f"Return a JSON array. {schema}"
    )


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def _try_model(client: Any, model: str, prompt: str,
               max_attempts: int, wait_seconds: list[int], types: Any) -> list[dict]:
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,
                ),
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
            result = json.loads(raw)
            if not isinstance(result, list):
                raise ValueError(f"Expected JSON list, got {type(result)}")
            return result
        except Exception as exc:
            print(f"  [{model}] attempt {attempt}/{max_attempts} failed: {exc}", file=sys.stderr)
            if attempt < max_attempts:
                wait = wait_seconds[attempt - 1]
                print(f"  Retrying in {wait}s…", file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"All {max_attempts} attempts failed for {model}")


def call_gemini(prompt: str, api_key: str) -> list[dict]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    last_exc: Exception | None = None
    for model, role, max_attempts, waits in MODEL_CASCADE:
        print(f"\n  Using {role}: {model}", file=sys.stderr)
        try:
            return _try_model(client, model, prompt, max_attempts, waits, types)
        except Exception as exc:
            print(f"  {role} exhausted: {exc}", file=sys.stderr)
            last_exc = exc
    raise RuntimeError(f"All models failed. Last: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Apply decisions
# ---------------------------------------------------------------------------

def apply_decisions(
    decisions: list[dict],
    batch_lookup: dict[str, dict],
    suffix_mappings: dict,   # suffix section of value_mappings.yaml
    unmapped_suffix: dict,   # suffix section of unmapped_db_values.yaml
    skips: set[str],
) -> tuple[int, int, int, int]:  # added, created, skipped, errors
    added = created = skipped = errors = 0

    for dec in decisions:
        raw = dec.get("raw_value")
        action = dec.get("action")

        if not raw or not action:
            print(f"  [WARN] Malformed decision: {dec}", file=sys.stderr)
            errors += 1
            continue

        if raw not in batch_lookup:
            print(f"  [WARN] Unknown raw_value in response: {raw!r}", file=sys.stderr)
            errors += 1
            continue

        # ---- skip ----
        if action == "skip":
            skips.add(raw)
            unmapped_suffix.pop(raw, None)
            print(f"  [SKIP]    {raw!r}")
            skipped += 1

        # ---- add_code ----
        elif action == "add_code":
            target = dec.get("target", "")
            leaf = find_leaf(suffix_mappings, target)
            if leaf is None:
                print(f"  [ERROR]   add_code target not found: {target!r} (for {raw!r})",
                      file=sys.stderr)
                errors += 1
                continue
            codes: list = leaf.setdefault("codes", [])
            if raw not in codes:
                codes.append(raw)
            unmapped_suffix.pop(raw, None)
            print(f"  [ADD]     {raw!r}  →  {target}")
            added += 1

        # ---- new_entry ----
        elif action == "new_entry":
            cat_key   = dec.get("category", "")
            cat_label = dec.get("category_label")
            label     = dec.get("label", raw)
            std_code  = dec.get("standard_code", re.sub(r"\W+", "_", raw).lower())

            if not cat_key:
                print(f"  [ERROR]   new_entry missing 'category' for {raw!r}", file=sys.stderr)
                errors += 1
                continue

            cat_dict = get_or_create_category(suffix_mappings, cat_key, cat_label)
            if std_code in cat_dict:
                # Leaf already exists — just add the code
                leaf = cat_dict[std_code]
                if isinstance(leaf, dict):
                    codes = leaf.setdefault("codes", [])
                    if raw not in codes:
                        codes.append(raw)
                    unmapped_suffix.pop(raw, None)
                    print(f"  [ADD→existing] {raw!r}  →  {cat_key}.{std_code}")
                    added += 1
            else:
                cat_dict[std_code] = {
                    "label": label,
                    "standard_code": std_code,
                    "codes": [raw],
                }
                unmapped_suffix.pop(raw, None)
                marker = " [NEW CAT]" if cat_label else ""
                print(f"  [NEW]{marker}  {raw!r}  →  {cat_key}.{std_code} ({label!r})")
                created += 1

        else:
            print(f"  [WARN] Unknown action {action!r} for {raw!r}", file=sys.stderr)
            errors += 1

    return added, created, skipped, errors


# ---------------------------------------------------------------------------
# Main batch function
# ---------------------------------------------------------------------------

def migrate_batch(
    mappings_path: Path,
    unmapped_path: Path,
    skip_sidecar: Path,
    batch_size: int,
    api_key: str,
    dry_run: bool,
) -> int:
    mappings = load_yaml(mappings_path)
    unmapped = load_yaml(unmapped_path)

    suffix_mappings: dict = mappings.get("suffix", {})
    unmapped_suffix: dict = unmapped.get("suffix", {})

    skips = load_skips(skip_sidecar)
    batch = collect_batch(unmapped_suffix, skips, batch_size)

    if not batch:
        print("No unprocessed suffix entries remain. Done!")
        return 0

    print(f"Collected {len(batch)} suffix entries.")
    for raw, entry in batch:
        print(f"  {raw!r}  (count={entry.get('count')}, "
              f"datasets={entry.get('datasets', [])[:2]}…)")

    hierarchy_summary = build_hierarchy_summary(suffix_mappings)
    prompt = build_prompt(hierarchy_summary, batch)

    if dry_run:
        print("\n--- PROMPT (dry-run, not sent) ---")
        print(prompt[:3000], "…" if len(prompt) > 3000 else "")
        print("--- END PROMPT ---")
        return len(batch)

    print(f"\nSending to Gemini (cascade: "
          f"{' → '.join(m for m,*_ in MODEL_CASCADE)})…")
    decisions = call_gemini(prompt, api_key)

    batch_lookup = {raw: entry for raw, entry in batch}
    added, created, skipped, errors = apply_decisions(
        decisions, batch_lookup, suffix_mappings, unmapped_suffix, skips
    )

    print(f"\nSummary: {added} codes added, {created} new entries, "
          f"{skipped} skipped, {errors} errors.")

    save_skips(skip_sidecar, skips)
    backup_once(mappings_path)
    backup_once(unmapped_path)
    save_yaml(mappings_path, mappings)
    save_yaml(unmapped_path, unmapped)
    print(f"Saved {mappings_path.name} and {unmapped_path.name}")

    return len(batch)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mappings", default=str(MAPPINGS_YAML),
                        help="Path to value_mappings.yaml")
    parser.add_argument("--unmapped", default=str(UNMAPPED_YAML),
                        help="Path to unmapped_db_values.yaml")
    parser.add_argument("--batch", type=int, default=50,
                        help="Number of suffix entries per Gemini call (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompt without calling the API or modifying files")
    parser.add_argument("--loop", action="store_true",
                        help="Keep running batches until no entries remain")
    args = parser.parse_args()

    mappings_path = Path(args.mappings)
    unmapped_path = Path(args.unmapped)

    for p in (mappings_path, unmapped_path):
        if not p.exists():
            raise SystemExit(f"File not found: {p}")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running.")

    if args.loop:
        batch_num = 0
        while True:
            batch_num += 1
            print(f"\n{'='*60}\n  Batch {batch_num}\n{'='*60}")
            processed = migrate_batch(
                mappings_path, unmapped_path, SKIP_SIDECAR,
                args.batch, api_key, args.dry_run,
            )
            if processed == 0 or args.dry_run:
                break
            time.sleep(2)
    else:
        migrate_batch(
            mappings_path, unmapped_path, SKIP_SIDECAR,
            args.batch, api_key, args.dry_run,
        )


if __name__ == "__main__":
    main()
