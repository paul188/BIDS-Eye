#!/usr/bin/env python3
"""
migrate_tasks_to_hierarchy.py

Iteratively migrates tasks from `learning_memory` and `other` into the
canonical task hierarchy by asking Gemini to classify them.

Authentication:
  export GEMINI_API_KEY=...
or
  export GOOGLE_API_KEY=...

Requires:
  pip install google-genai pyyaml

Usage:
  python migrate_tasks_to_hierarchy.py [--yaml PATH] [--batch 50] [--dry-run]

Run repeatedly (or in a loop) until all tasks are processed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Category tree — these are the valid destination categories Gemini may choose.
# "other" is a special sentinel meaning "doesn't fit anywhere; leave as-is."
# The keys here are the dot-separated paths used in the JSON response.
# ---------------------------------------------------------------------------
CATEGORY_PATHS: list[str] = [
    "resting_state",
    "naturalistic",
    "memory.working_memory",
    "memory.episodic_memory",
    "memory.episodic_memory.encoding",
    "memory.retrieval",
    "memory.recognition",
    "memory.associative",
    "memory.Autobiographical",
    "memory.temporal",
    "memory.spatial_memory",
    "attention_and_control",
    "motor_somasensory",
    "language_speech",
    "sensory_and_perception",
    "social_emotional_reward",
    "physiological_neurostimulation",
    "calibration_baseline",
    "other",  # sentinel — task stays in `other`, marked confirmed
]


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class _NoAliasDumper(yaml.Dumper):
    """YAML Dumper that never emits anchors or aliases.

    PyYAML reuses Python object identity to decide whether to emit an anchor
    (&id001) and a back-reference alias (*id001).  After a round-trip load the
    same list/dict may be referenced from multiple keys, producing anchors in
    output that are technically correct but ugly and confusing.  This dumper
    simply ignores the alias machinery so every node is written out in full.
    """

    def ignore_aliases(self, _data: object) -> bool:  # noqa: D401
        return True


def save_yaml(path: Path, data: dict) -> None:
    """Write back to YAML.

    Guarantees:
    - Key insertion order is preserved (sort_keys=False).
    - No YAML anchors / aliases in the output.
    - 2-space indentation at every nesting level.
    - Block-sequence items (list entries) are indented 2 spaces relative to
      their parent key, matching the style of the original file.
    - Unicode characters are written as-is (allow_unicode=True).
    """
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(
            data,
            fh,
            Dumper=_NoAliasDumper,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            width=10000,  # avoid wrapping long description strings mid-sentence
        )


# ---------------------------------------------------------------------------
# Sidecar file for tasks confirmed to stay in `other`
# ---------------------------------------------------------------------------

def load_confirmed_others(sidecar: Path) -> set[str]:
    if sidecar.exists():
        return set(json.loads(sidecar.read_text()))
    return set()


def save_confirmed_others(sidecar: Path, keys: set[str]) -> None:
    sidecar.write_text(json.dumps(sorted(keys), indent=2))


# ---------------------------------------------------------------------------
# Task collection
# ---------------------------------------------------------------------------

def is_task_entry(value: Any) -> bool:
    """A task entry is a dict that has at least a 'label' or 'description' key."""
    return isinstance(value, dict) and ("label" in value or "description" in value)


def collect_source_tasks(
    task_section: dict,
    confirmed_others: set[str],
    batch_size: int,
) -> list[tuple[str, str, dict]]:
    """
    Return up to `batch_size` unprocessed tasks as (source_category, task_key, task_dict).
    Sources: 'learning_memory' (all tasks) and 'other' (excluding confirmed_others).
    """
    results: list[tuple[str, str, dict]] = []

    for source_name in ("learning_memory", "other"):
        source = task_section.get(source_name, {})
        if not isinstance(source, dict):
            continue
        for key, val in source.items():
            if not is_task_entry(val):
                continue  # skip meta-keys like label/codes
            if source_name == "other" and key in confirmed_others:
                continue
            results.append((source_name, key, val))
            if len(results) >= batch_size:
                return results

    return results


# ---------------------------------------------------------------------------
# Category destination helpers
# ---------------------------------------------------------------------------

def resolve_category(task_section: dict, dot_path: str) -> dict | None:
    """
    Walk the dot-path (e.g. 'memory.episodic_memory.encoding') inside task_section
    and return the dict at that location, or None if not found.
    """
    parts = dot_path.split(".")
    node = task_section
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if not isinstance(node, dict):
        return None
    return node


# ---------------------------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are a neuroscience domain expert helping to organise fMRI task paradigms "
    "into a canonical hierarchy. You will receive a JSON list of tasks. "
    "For each task, choose exactly one category from the provided list. "
    "Return ONLY a JSON array — no markdown fences, no prose, no extra keys. "
    "Each element must be: {\"task_key\": \"...\", \"category\": \"<dot_path>\"}. "
    "Use the 'other' category only when the task genuinely does not fit any other category."
)


def build_prompt(tasks: list[tuple[str, str, dict]], categories: list[str]) -> str:
    task_list = []
    for _src, key, val in tasks:
        entry = {"task_key": key}
        if "label" in val:
            entry["label"] = val["label"]
        if "description" in val:
            entry["description"] = val["description"]
        task_list.append(entry)

    cat_lines = "\n".join(f"  - {c}" for c in categories)
    tasks_json = json.dumps(task_list, indent=2, ensure_ascii=False)

    return (
        f"Available categories (use the exact dot-path string):\n{cat_lines}\n\n"
        f"Tasks to classify:\n{tasks_json}\n\n"
        "Return a JSON array where each element is:\n"
        '  {"task_key": "<key>", "category": "<dot_path>"}\n'
        "Include one entry per task. Use 'other' only when nothing fits."
    )


# Model cascade: primary -> fallback 1 -> fallback 2 -> fallback 3
# Each model is tried with its own retry budget before falling through to the next.
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


def _try_model(
    client: Any,
    model: str,
    prompt: str,
    max_attempts: int,
    wait_seconds: list[int],
    types: Any,
) -> list[dict]:
    """Try a single model with retries. Raises on exhaustion."""
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
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
            result = json.loads(raw)
            if not isinstance(result, list):
                raise ValueError(f"Expected a JSON list, got: {type(result)}")
            return result
        except Exception as exc:
            print(f"  [{model}] attempt {attempt}/{max_attempts} failed: {exc}", file=sys.stderr)
            if attempt < max_attempts:
                wait = wait_seconds[attempt - 1]
                print(f"  Retrying in {wait}s…", file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"All {max_attempts} attempts failed for {model}")


def call_gemini(prompt: str, api_key: str) -> list[dict]:
    """Call Gemini using the canonical model cascade."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)

    last_exc: Exception | None = None
    for model, role, max_attempts, wait_seconds in MODEL_CASCADE:
        print(f"\n  Using {role}: {model}", file=sys.stderr)
        try:
            return _try_model(client, model, prompt, max_attempts, wait_seconds, types)
        except Exception as exc:
            print(f"  {role} ({model}) exhausted: {exc}", file=sys.stderr)
            last_exc = exc

    raise RuntimeError(f"All models in cascade failed. Last error: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Main migration logic
# ---------------------------------------------------------------------------

def migrate_batch(
    yaml_path: Path,
    sidecar_path: Path,
    batch_size: int,
    api_key: str,
    dry_run: bool,
) -> int:
    """
    Process one batch of up to `batch_size` tasks.
    Returns the number of tasks processed (0 means nothing left to do).
    """
    data = load_yaml(yaml_path)
    task_section: dict = data.get("task", {})

    confirmed_others = load_confirmed_others(sidecar_path)
    batch = collect_source_tasks(task_section, confirmed_others, batch_size)

    if not batch:
        print("No unprocessed tasks remain. Done!")
        return 0

    print(f"Collected {len(batch)} tasks from sources: "
          f"{set(src for src, _, _ in batch)}")
    for src, key, val in batch:
        label = val.get("label", key)
        print(f"  [{src}] {key} — {label}")

    models = " → ".join(m for m, _, _, _ in MODEL_CASCADE)
    print(f"\nSending to Gemini (cascade: {models})…")
    prompt = build_prompt(batch, CATEGORY_PATHS)

    if dry_run:
        print("\n--- PROMPT (dry-run, not sent) ---")
        print(prompt)
        print("--- END PROMPT ---")
        return len(batch)

    assignments: list[dict] = call_gemini(prompt, api_key)

    # Build a lookup of the batch tasks by key for quick access
    batch_lookup: dict[str, tuple[str, dict]] = {
        key: (src, val) for src, key, val in batch
    }

    moved = 0
    errors = 0
    newly_confirmed_others: set[str] = set()

    for assignment in assignments:
        task_key = assignment.get("task_key")
        category = assignment.get("category")

        if not task_key or not category:
            print(f"  [WARN] Malformed assignment: {assignment}", file=sys.stderr)
            errors += 1
            continue

        if task_key not in batch_lookup:
            print(f"  [WARN] Unknown task key in response: {task_key!r}", file=sys.stderr)
            errors += 1
            continue

        if category not in CATEGORY_PATHS:
            print(f"  [WARN] Unknown category {category!r} for {task_key!r}", file=sys.stderr)
            errors += 1
            continue

        source_name, task_dict = batch_lookup[task_key]

        if category == "other":
            # Mark as confirmed other — do not move, do not re-process
            newly_confirmed_others.add(task_key)
            print(f"  [CONFIRMED OTHER] {task_key}")
            continue

        # Resolve destination
        dest = resolve_category(task_section, category)
        if dest is None:
            print(f"  [ERROR] Category path {category!r} not found in YAML for {task_key!r}",
                  file=sys.stderr)
            errors += 1
            continue

        # Check for key collision at destination
        if task_key in dest:
            print(f"  [WARN] Key {task_key!r} already exists in {category!r}; skipping move",
                  file=sys.stderr)
            errors += 1
            continue

        # Remove from source
        source_dict = task_section[source_name]
        del source_dict[task_key]

        # Insert at destination
        dest[task_key] = task_dict

        print(f"  [MOVED] {task_key}  →  {category}")
        moved += 1

    print(f"\nSummary: {moved} moved, {len(newly_confirmed_others)} confirmed as 'other', "
          f"{errors} errors.")

    # Persist changes
    confirmed_others |= newly_confirmed_others
    save_confirmed_others(sidecar_path, confirmed_others)

    # Backup original before writing
    backup = yaml_path.with_suffix(".yaml.bak")
    if not backup.exists():
        import shutil
        shutil.copy2(yaml_path, backup)
        print(f"Backup created at {backup}")

    save_yaml(yaml_path, data)
    print(f"Saved {yaml_path}")

    return len(batch)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml",
        default=str(Path(__file__).parent / "value_mappings.yaml"),
        help="Path to value_mappings.yaml",
    )
    parser.add_argument("--batch", type=int, default=50, help="Tasks per Gemini call")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompt but do not call the API or modify files",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep running batches until no tasks remain",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        raise SystemExit(f"File not found: {yaml_path}")

    sidecar_path = yaml_path.with_name("confirmed_others.json")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running.")

    if args.loop:
        batch_num = 0
        while True:
            batch_num += 1
            print(f"\n{'='*60}")
            print(f"  Batch {batch_num}")
            print(f"{'='*60}")
            processed = migrate_batch(
                yaml_path, sidecar_path, args.batch, api_key, args.dry_run
            )
            if processed == 0:
                break
            if args.dry_run:
                break  # dry-run shows one batch then stops
            time.sleep(2)  # brief pause between batches
    else:
        migrate_batch(
            yaml_path, sidecar_path, args.batch, api_key, args.dry_run
        )


if __name__ == "__main__":
    main()
