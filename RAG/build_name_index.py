#!/usr/bin/env python3
"""
RAG/build_name_index.py
-----------------------
Build a flat index of author names and funding sources from the database
for use by RAG/retriever.py (RapidFuzz token-similarity matching).

No embeddings — just the raw string lists.  Runs in seconds.
The output replaces the role of modal_app/build_metadata_index.py for
the author and funding fields.  Tasks, diagnoses, suffixes, and datatypes
are handled directly by value_mappings.yaml via yaml_to_llamaindex.py.

Usage
-----
    python RAG/build_name_index.py \\
        --db-url "postgresql://user:password@localhost:5432/bids_sql" \\
        --out    RAG/name_index.json

Re-run whenever the database grows significantly so new authors and
funding sources are picked up by the retriever.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _get_column_type(conn, table_name: str, column_name: str) -> str | None:
    from sqlalchemy import text
    sql = text("""
        SELECT pg_catalog.format_type(a.atttypid, a.atttypmod)
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = :table AND a.attname = :col
          AND a.attnum > 0 AND NOT a.attisdropped
        ORDER BY CASE WHEN n.nspname = current_schema() THEN 0 ELSE 1 END
        LIMIT 1
    """)
    return conn.execute(sql, {"table": table_name, "col": column_name}).scalar_one_or_none()


def _distinct_array_values(
    conn, column_name: str, value_alias: str, column_type: str
) -> List[str]:
    from sqlalchemy import text
    if column_type.endswith("[]"):
        source = f"unnest({column_name})"
        where  = f"{column_name} IS NOT NULL AND cardinality({column_name}) > 0"
    elif column_type in {"json", "jsonb"}:
        source = f"jsonb_array_elements_text({column_name}::jsonb)"
        where  = (
            f"{column_name} IS NOT NULL "
            f"AND jsonb_typeof({column_name}::jsonb) = 'array' "
            f"AND jsonb_array_length({column_name}::jsonb) > 0"
        )
    else:
        raise ValueError(
            f"Unsupported column type for bids_datasets.{column_name}: {column_type}"
        )
    sql = text(f"""
        SELECT DISTINCT {value_alias}
        FROM (
            SELECT {source} AS {value_alias}
            FROM bids_datasets
            WHERE {where}
        ) t
        WHERE {value_alias} IS NOT NULL AND {value_alias} != ''
        ORDER BY {value_alias}
    """)
    return [row[0] for row in conn.execute(sql).fetchall()]


def fetch_names(db_url: str) -> Dict[str, List[str]]:
    from sqlalchemy import create_engine
    engine = create_engine(db_url)

    columns = {
        "authors":         ("authors",  "author"),
        "funding_sources": ("funding",  "funding_source"),
    }
    names: Dict[str, List[str]] = {key: [] for key in columns}

    with engine.connect() as conn:
        for key, (col_name, alias) in columns.items():
            try:
                col_type = _get_column_type(conn, "bids_datasets", col_name)
                if not col_type:
                    raise ValueError(f"Column bids_datasets.{col_name} not found in schema")
                names[key] = _distinct_array_values(conn, col_name, alias, col_type)
                print(f"  {key}: {len(names[key])} distinct values")
            except Exception as exc:
                print(f"  [WARN] {key} failed: {exc} — skipping", file=sys.stderr)
                conn.rollback()

    return names


# ── File I/O ───────────────────────────────────────────────────────────────────

def save_atomic(path: Path, data: Dict) -> None:
    """Write *data* to *path* atomically via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url", required=True,
        help="PostgreSQL connection string, e.g. postgresql://user:pw@host/db"
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "name_index.json",
        help="Output path (default: RAG/name_index.json)"
    )
    args = parser.parse_args()

    print("Querying DB for author and funding names...")
    names = fetch_names(args.db_url)

    total = sum(len(v) for v in names.values())
    if total == 0:
        print("WARNING: no values retrieved — check DB connection and schema.", file=sys.stderr)

    save_atomic(args.out, names)
    size_kb = args.out.stat().st_size / 1000
    print(f"Saved name index to {args.out} ({size_kb:.1f} KB, {total} total entries)")
    print("Done.")
    print()
    print("Next steps:")
    print("  The retriever loads this file automatically:")
    print("    from RAG.retriever import MetadataRetriever")
    print(f"    r = MetadataRetriever('{args.out}')")


if __name__ == "__main__":
    main()
