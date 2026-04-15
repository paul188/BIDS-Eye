#!/usr/bin/env python3
"""
clean_unmapped_values.py

Removes auto-generated / non-portable keys from the `suffix` section of
unmapped_db_values.yaml:

  1. f{number}   e.g. f1, f193, f210  — auto-generated feature keys
  2. grot{a-i}   e.g. grota … groti   — FSL dual-regression temporaries (ds006391)
  3. starts with a digit               — e.g. 008, 03092021

Runs as a dry-run by default (prints what would be deleted).
Pass --apply to write the cleaned file in-place (a .bak backup is made first).

Usage:
  python clean_unmapped_values.py [--yaml PATH] [--apply]
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Patterns to delete (applied only to the `suffix` section)
# ---------------------------------------------------------------------------

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("f{number}",    re.compile(r"^f\d+$")),
    ("grot{a-i}",    re.compile(r"^grot[a-i]$")),
    ("starts-digit", re.compile(r"^\d")),
    ("vol*",         re.compile(r"^vol", re.IGNORECASE)),
    ("t{number}",    re.compile(r"^t\d+$")),
    ("sub{number}",  re.compile(r"^sub\d+$")),
]


def match_pattern(key: str) -> str | None:
    """Return the matched pattern name if the key should be deleted, else None."""
    for name, pat in PATTERNS:
        if pat.match(key):
            return name
    return None


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

class _NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, _data: object) -> bool:
        return True


def load_yaml(path: Path) -> tuple[str, dict]:
    """Return (header_comment, parsed_data)."""
    raw = path.read_text(encoding="utf-8")
    header = "".join(
        line for line in raw.splitlines(keepends=True)
        if line.startswith("#")
    )
    # Only keep leading comment block
    body_lines = raw.splitlines(keepends=True)
    in_header = True
    header_lines = []
    for line in body_lines:
        if in_header and line.startswith("#"):
            header_lines.append(line)
        else:
            in_header = False
    header = "".join(header_lines)
    data = yaml.safe_load(raw)
    return header, data


def save_yaml(path: Path, header: str, data: dict) -> None:
    body = yaml.dump(
        data,
        Dumper=_NoAliasDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        indent=2,
        width=10000,
    )
    path.write_text(header + body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def clean(yaml_path: Path, apply: bool) -> None:
    header, data = load_yaml(yaml_path)

    suffix_section: dict = data.get("suffix", {})
    if not suffix_section:
        print("No `suffix` section found in the file.")
        return

    to_remove: dict[str, str] = {}  # key → pattern_name
    for key in suffix_section:
        reason = match_pattern(str(key))
        if reason:
            to_remove[key] = reason

    if not to_remove:
        print("Nothing to remove in `suffix` section.")
        return

    # Group by pattern for the report
    by_pattern: dict[str, list[str]] = {}
    for key, reason in to_remove.items():
        by_pattern.setdefault(reason, []).append(key)

    mode = "REMOVED" if apply else "WOULD REMOVE (dry-run — pass --apply to write)"
    print(f"\n{mode} from `suffix`:\n")
    for pattern, keys in sorted(by_pattern.items()):
        examples = keys[:8]
        tail = f", … (+{len(keys) - 8} more)" if len(keys) > 8 else ""
        print(f"  {pattern:<15} {len(keys):>4}x   e.g. {', '.join(examples)}{tail}")

    print(f"\nTotal: {len(to_remove)} keys removed from `suffix`.")

    if apply:
        for key in to_remove:
            del suffix_section[key]

        backup = yaml_path.with_suffix(".yaml.bak")
        if not backup.exists():
            shutil.copy2(yaml_path, backup)
            print(f"Backup → {backup}")
        save_yaml(yaml_path, header, data)
        print(f"Saved  → {yaml_path}")
    else:
        print("\nRe-run with --apply to write changes.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--yaml",
        default=str(Path(__file__).parent / "unmapped_db_values.yaml"),
        help="Path to unmapped_db_values.yaml (default: same directory as this script)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes in-place (a .bak backup is created first). "
             "Without this flag the script only prints what would be deleted.",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        raise SystemExit(f"File not found: {yaml_path}")

    clean(yaml_path, args.apply)


if __name__ == "__main__":
    main()
