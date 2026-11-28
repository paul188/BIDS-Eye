#!/usr/bin/env python3
"""
Write reviewed task → Cognitive Atlas links into value_mappings.yaml.

A task entry in value_mappings.yaml names what it measures next to its
synonyms:

    sternberg_task:
      label: Sternberg task
      standard_code: sternberg_task
      ca_id: tsk_4a57abb949e0c        # the Cognitive Atlas task it was matched to
      measures:                       # cognitive_construct codes
      - maintenance
      - working_memory_retrieval

The retriever reads `measures` to turn a construct ("working memory studies")
into the tasks that measure it (yaml_to_llamaindex.measures_db).

generate_task_bridge.py and fill_bridge_gaps.py write proposals as JSON,
{task_code: {"ca_task_id": ..., "measures": [...], ...}}. Review that file
(delete wrong matches, fix measures), then apply it:

    python backend/vocabulary/tools/apply_task_measures.py backend/vocabulary/proposed_task_mappings.json

Applying only adds: a task keeps the measures it has and gains the proposed
ones, and keeps its ca_id. To remove a link, edit value_mappings.yaml.

The file is edited line by line, not re-dumped, so its layout and every
other entry stay exactly as they were; the result is read back and checked
against that before it is written.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

_VOCABULARY = Path(__file__).resolve().parents[1]
_ENTRY = re.compile(r"^  (\S[^:]*):\s*$")
_STANDARD_CODE = re.compile(r"^    standard_code: (\S+)\s*$")


def _task_section(lines: List[str]) -> Tuple[int, int]:
    """Index of the first and one past the last line of the task: section."""
    start = lines.index("task:\n") + 1
    end = next((i for i in range(start, len(lines)) if lines[i][:1] not in ("", " ", "\n")),
               len(lines))
    return start, end


def _entries(lines: List[str], start: int, end: int) -> Dict[str, Tuple[int, int]]:
    """standard_code → (first, one past last) line of its entry."""
    starts = [i for i in range(start, end) if _ENTRY.match(lines[i])] + [end]
    found: Dict[str, Tuple[int, int]] = {}
    for first, last in zip(starts, starts[1:]):
        for i in range(first + 1, last):
            m = _STANDARD_CODE.match(lines[i])
            if m:
                found[m.group(1)] = (first, last)
                break
    return found


def _rewrite_entry(body: List[str], ca_id: str | None, measures: List[str]) -> List[str]:
    """The entry's lines without its ca_id/measures lines, then those written fresh
    right after standard_code."""
    kept: List[str] = []
    skipping = False
    for line in body:
        if skipping and line.startswith("    - "):
            continue
        skipping = False
        if line.startswith("    ca_id:"):
            continue
        if line.startswith("    measures:"):
            skipping = line.strip() == "measures:"
            continue
        kept.append(line)
    added = ([f"    ca_id: {ca_id}\n"] if ca_id else []) + (
        ["    measures:\n"] + [f"    - {m}\n" for m in measures] if measures else [])
    at = next(i for i, line in enumerate(kept) if _STANDARD_CODE.match(line)) + 1
    return kept[:at] + added + kept[at:]


def apply(text: str, proposals: Dict[str, dict], constructs: set[str]) -> Tuple[str, List[str]]:
    """value_mappings.yaml text with *proposals* merged in, and a note per proposal left out.

    A proposal is left out when its code is no task with a standard_code (a
    group has no datasets of its own: its key would reach the search as a code
    that matches nothing), and a measure when it is no construct code.
    """
    lines = text.splitlines(keepends=True)
    start, end = _task_section(lines)
    entries = _entries(lines, start, end)
    data = yaml.safe_load(text)
    expected = copy.deepcopy(data)
    by_code = {v["standard_code"]: k for k, v in data["task"].items()
               if isinstance(v, dict) and v.get("standard_code")}

    notes: List[str] = []
    changes: Dict[str, Tuple[str | None, List[str]]] = {}
    for code, proposal in proposals.items():
        if code not in entries or code not in by_code:
            notes.append(f"{code}: no task with that standard_code (a group?), left out")
            continue
        unknown = [m for m in proposal.get("measures") or [] if m not in constructs]
        if unknown:
            notes.append(f"{code}: no such construct {unknown}, left out")
        entry = expected["task"][by_code[code]]
        ca_id = entry.get("ca_id") or proposal.get("ca_task_id") or None
        measures = sorted(set(entry.get("measures") or [])
                          | {m for m in proposal.get("measures") or [] if m in constructs})
        if ca_id:
            entry["ca_id"] = ca_id
        if measures:
            entry["measures"] = measures
        changes[code] = (ca_id, measures)

    # Bottom-up, so the line numbers of the entries still to do stay valid.
    for code in sorted(changes, key=lambda c: entries[c][0], reverse=True):
        first, last = entries[code]
        lines[first:last] = _rewrite_entry(lines[first:last], *changes[code])

    result = "".join(lines)
    if yaml.safe_load(result) != expected:
        raise RuntimeError("value_mappings.yaml would change beyond the proposed links; nothing written")
    return result, notes


def construct_codes(path: Path) -> set[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {v["standard_code"] for v in (data.get("cognitive_construct") or {}).values()
            if isinstance(v, dict) and v.get("standard_code")}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("proposals", type=Path, help="reviewed JSON from generate_task_bridge.py / fill_bridge_gaps.py")
    parser.add_argument("--yaml", type=Path, default=_VOCABULARY / "value_mappings.yaml")
    parser.add_argument("--ca-constructs", type=Path, default=_VOCABULARY / "cognitive_atlas_constructs.yaml")
    args = parser.parse_args(argv)

    proposals = json.loads(args.proposals.read_text(encoding="utf-8"))
    text = args.yaml.read_text(encoding="utf-8")
    result, notes = apply(text, proposals, construct_codes(args.ca_constructs))
    for note in notes:
        print(f"  {note}", file=sys.stderr)
    if result == text:
        print("value_mappings.yaml already has every proposed link.")
        return 0
    args.yaml.write_text(result, encoding="utf-8")
    print(f"Wrote the proposed task links into {args.yaml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
