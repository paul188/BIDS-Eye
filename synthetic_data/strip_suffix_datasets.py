#!/usr/bin/env python3
"""
strip_suffix_datasets.py

Removes the `datasets:` list from every entry in the `suffix` section of
unmapped_db_values.yaml, keeping only the `count:` field.
All other sections are left untouched.

Dry-run by default; pass --apply to write in-place (backup created first).

Usage:
  python strip_suffix_datasets.py [--yaml PATH] [--apply]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


class _NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, _data: object) -> bool:
        return True


def load_yaml_with_header(path: Path) -> tuple[str, dict]:
    raw = path.read_text(encoding="utf-8")
    header_lines = []
    for line in raw.splitlines(keepends=True):
        if line.startswith("#"):
            header_lines.append(line)
        else:
            break
    return "".join(header_lines), yaml.safe_load(raw)


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


def strip(yaml_path: Path, apply: bool) -> None:
    header, data = load_yaml_with_header(yaml_path)
    suffix_section: dict = data.get("suffix", {})

    if not suffix_section:
        print("No `suffix` section found.")
        return

    removed_count = 0
    for key, entry in suffix_section.items():
        if isinstance(entry, dict) and "datasets" in entry:
            removed_count += 1
            if apply:
                del entry["datasets"]

    mode = "REMOVED" if apply else "WOULD REMOVE (dry-run — pass --apply to write)"
    print(f"{mode} `datasets:` from {removed_count} entries in `suffix`.")

    if apply:
        bak = yaml_path.with_suffix(".yaml.bak")
        if not bak.exists():
            shutil.copy2(yaml_path, bak)
            print(f"Backup → {bak}")
        save_yaml(yaml_path, header, data)
        print(f"Saved  → {yaml_path}")
    else:
        print("Re-run with --apply to write changes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--yaml",
        default=str(Path(__file__).parent / "unmapped_db_values.yaml"),
        help="Path to unmapped_db_values.yaml",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes in-place (backup created first).",
    )
    args = parser.parse_args()

    path = Path(args.yaml)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    strip(path, args.apply)


if __name__ == "__main__":
    main()
