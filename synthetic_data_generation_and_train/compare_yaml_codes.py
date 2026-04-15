#!/usr/bin/env python3
"""
compare_yaml_codes.py
----------------------
Checks whether value_mappings_2.yaml contains all codes present in
value_mappings.yaml.

Reports:
  - Codes in original but MISSING from new file (these would be lost)
  - Codes in new file but NOT in original (newly added)
  - Summary counts per section

Usage:
  python training_data_generation/compare_yaml_codes.py
  python training_data_generation/compare_yaml_codes.py \\
      --original training_data_generation/value_mappings.yaml \\
      --new      value_mappings_2.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# Keys that mark a leaf node — do not recurse into them
_LEAF_KEYS = {"label", "description", "codes", "synonyms", "canonical_code",
              "standard_code", "extra_codes"}


def collect_codes(node: dict, path: str = "") -> dict[str, list[str]]:
    """
    Recursively collect all codes from both 'codes' and 'extra_codes' lists.
    Returns {code_lowercase: [dotted_path, ...]} so we know where each code came from.
    """
    result: dict[str, list[str]] = {}

    def walk(n, p):
        if not isinstance(n, dict):
            return
        for key in ("codes", "extra_codes"):
            if key in n:
                for c in (n[key] or []):
                    code = str(c).lower().strip()
                    result.setdefault(code, []).append(f"{p}.{key}" if p else key)
        for key, val in n.items():
            if key not in _LEAF_KEYS and isinstance(val, dict):
                child = f"{p}.{key}" if p else key
                walk(val, child)

    walk(node, path)
    return result


def load_all_codes(yaml_path: Path) -> dict[str, dict[str, list[str]]]:
    """
    Load a YAML file and return {section: {code: [paths]}} for every top-level section.
    """
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    sections: dict[str, dict[str, list[str]]] = {}
    for section, content in data.items():
        if isinstance(content, dict):
            sections[section] = collect_codes(content, section)
        # non-dict sections (e.g. simple scalars) have no codes to check
    return sections


def main() -> None:
    default_orig = Path(__file__).with_name("value_mappings.yaml")
    default_new  = Path(__file__).parent.parent / "value_mappings_2.yaml"

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", default=str(default_orig),
                        help="Path to original value_mappings.yaml")
    parser.add_argument("--new", default=str(default_new),
                        help="Path to new / candidate YAML file")
    parser.add_argument("--show-present", action="store_true",
                        help="Also print codes that ARE present in both files")
    args = parser.parse_args()

    orig_path = Path(args.original)
    new_path  = Path(args.new)

    print(f"Original : {orig_path}")
    print(f"New      : {new_path}\n")

    orig_sections = load_all_codes(orig_path)
    new_sections  = load_all_codes(new_path)

    # Union of all section names
    all_sections = sorted(set(orig_sections) | set(new_sections))

    grand_missing  = 0
    grand_added    = 0
    grand_total    = 0

    for section in all_sections:
        orig_codes = orig_sections.get(section, {})
        new_codes  = new_sections.get(section, {})

        missing = {c: orig_codes[c] for c in orig_codes if c not in new_codes}
        added   = {c: new_codes[c]  for c in new_codes  if c not in orig_codes}
        shared  = {c for c in orig_codes if c in new_codes}

        grand_missing += len(missing)
        grand_added   += len(added)
        grand_total   += len(orig_codes)

        bar = "─" * 62
        status = "✓ FULLY COVERED" if not missing else f"✗ {len(missing)} MISSING"
        print(f"\n{bar}")
        print(f"  {section.upper()}")
        print(f"  Original codes: {len(orig_codes)}  |  New codes: {len(new_codes)}  |  {status}")
        print(bar)

        if missing:
            print(f"\n  MISSING from new file ({len(missing)} codes):")
            for code, paths in sorted(missing.items()):
                print(f"    {code!r:40s}  ← {paths[0]}")

        if added:
            print(f"\n  NEW codes not in original ({len(added)} codes):")
            for code, paths in sorted(added.items()):
                print(f"    {code!r:40s}  ← {paths[0]}")

        if args.show_present and shared:
            print(f"\n  Present in both ({len(shared)} codes):")
            for code in sorted(shared):
                print(f"    {code!r}")

    # ── Summary ──────────────────────────────────────────────────────────────────
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  SUMMARY")
    print(sep)
    print(f"  Original total codes : {grand_total}")
    print(f"  Missing from new     : {grand_missing}")
    print(f"  Added in new         : {grand_added}")

    if grand_missing == 0:
        print(f"\n  ✓ All original codes are present in the new file.")
    else:
        print(f"\n  ✗ {grand_missing} code(s) from the original are NOT in the new file.")
        print(f"    Add --show-present to also list matching codes.")

    print(sep)


if __name__ == "__main__":
    main()
