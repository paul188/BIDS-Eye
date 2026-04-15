#!/usr/bin/env python3
"""
modal_app/build_metadata_index.py
----------------------------------
Build a vector index of known BIDS DB values and upload it to the
"bids-eye-metadata" Modal Volume.

Run this script on the HPC (where PostgreSQL is accessible) after
the crawl job has populated the database.  Re-run it whenever the DB
grows significantly so the retriever sees new tasks / diagnoses / etc.

What it does
────────────
1. Queries all distinct values for: tasks, datatypes, suffixes, diagnoses.
2. Embeds them with sentence-transformers/all-MiniLM-L6-v2 (L2-normalised).
3. Saves everything as a single JSON file (metadata_index.json).
4. Uploads the file to the "bids-eye-metadata" Modal Volume via modal.Volume.

Usage
─────
    module load Python/3.11.3-GCCcore-12.3.0
    source $SCRATCH/train_venv/bin/activate   # or any venv with sentence-transformers + sqlalchemy

    python modal_app/build_metadata_index.py \\
        --db-url "postgresql://user:password@localhost:5432/bids_sql" \\
        --out    /tmp/metadata_index.json \\
        --upload                           # push to Modal Volume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


# ── DB queries ─────────────────────────────────────────────────────────────────

def _get_column_type(conn, table_name: str, column_name: str) -> str | None:
    from sqlalchemy import text

    sql = text(
        """
        SELECT pg_catalog.format_type(a.atttypid, a.atttypmod) AS column_type
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c
          ON c.oid = a.attrelid
        JOIN pg_catalog.pg_namespace n
          ON n.oid = c.relnamespace
        WHERE c.relname = :table_name
          AND a.attname = :column_name
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY CASE WHEN n.nspname = current_schema() THEN 0 ELSE 1 END,
                 n.nspname
        LIMIT 1
        """
    )
    return conn.execute(
        sql,
        {"table_name": table_name, "column_name": column_name},
    ).scalar_one_or_none()


def _build_distinct_list_query(column_name: str, value_alias: str, column_type: str) -> str:
    if column_type.endswith("[]"):
        source = f"unnest({column_name})"
        where_clause = f"{column_name} IS NOT NULL AND cardinality({column_name}) > 0"
    elif column_type in {"json", "jsonb"}:
        source = f"jsonb_array_elements_text({column_name}::jsonb)"
        where_clause = (
            f"{column_name} IS NOT NULL "
            f"AND jsonb_typeof({column_name}::jsonb) = 'array' "
            f"AND jsonb_array_length({column_name}::jsonb) > 0"
        )
    else:
        raise ValueError(
            f"Unsupported type for bids_datasets.{column_name}: {column_type}"
        )

    return f"""
        SELECT DISTINCT {value_alias}
        FROM (
            SELECT {source} AS {value_alias}
            FROM bids_datasets
            WHERE {where_clause}
        ) t
        WHERE {value_alias} IS NOT NULL AND {value_alias} != ''
        ORDER BY {value_alias}
    """


def fetch_db_values(db_url: str) -> Dict[str, List[str]]:
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    queries = {
        "tasks": "SELECT DISTINCT task FROM bids_objects WHERE task IS NOT NULL AND task != '' ORDER BY task",
        "datatypes": "SELECT DISTINCT datatype FROM bids_objects WHERE datatype IS NOT NULL AND datatype != '' ORDER BY datatype",
        "suffixes": "SELECT DISTINCT suffix FROM bids_objects WHERE suffix IS NOT NULL AND suffix != '' ORDER BY suffix",
        "diagnoses": "SELECT DISTINCT diagnosis FROM bids_participants WHERE diagnosis IS NOT NULL AND diagnosis != '' ORDER BY diagnosis",
        "dataset_names": "SELECT DISTINCT name FROM bids_datasets WHERE name IS NOT NULL AND name != '' ORDER BY name LIMIT 500",
    }
    dataset_list_columns = {
        "authors": ("authors", "author"),
        "funding_sources": ("funding", "funding_source"),
    }

    values: Dict[str, List[str]] = {}
    with engine.connect() as conn:
        for category, sql in queries.items():
            try:
                rows = conn.execute(text(sql)).fetchall()
                values[category] = [row[0] for row in rows]
                print(f"  {category}: {len(values[category])} distinct values")
            except Exception as exc:
                print(f"  [WARN] {category} query failed: {exc} — skipping", file=sys.stderr)
                conn.rollback()
                values[category] = []

        for category, (column_name, value_alias) in dataset_list_columns.items():
            try:
                column_type = _get_column_type(conn, "bids_datasets", column_name)
                if not column_type:
                    raise ValueError(
                        f"Could not determine type for bids_datasets.{column_name}"
                    )
                sql = _build_distinct_list_query(column_name, value_alias, column_type)
                rows = conn.execute(text(sql)).fetchall()
                values[category] = [row[0] for row in rows]
                print(
                    f"  {category}: {len(values[category])} distinct values "
                    f"(from bids_datasets.{column_name} {column_type})"
                )
            except Exception as exc:
                print(f"  [WARN] {category} query failed: {exc} — skipping", file=sys.stderr)
                conn.rollback()
                values[category] = []

    return values


# ── Embedding ─────────────────────────────────────────────────────────────────

def build_embeddings(values: Dict[str, List[str]]) -> Dict[str, list]:
    """Embed all values; returns {category: [[float, ...], ...]}."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers.\n"
            "Activate a venv that includes sentence-transformers, sqlalchemy, and psycopg2.\n"
            "For example:\n"
            "  source $SCRATCH/train_venv/bin/activate\n"
            "or install it into the active environment with:\n"
            "  pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # We enrich each value with a short natural-language description so the
    # embedding captures semantic meaning rather than just the bare string.
    # E.g. "rest" → "BIDS task name: rest" so it aligns better with
    # question embeddings like "resting-state fMRI".
    category_prefix = {
        "tasks": "BIDS task name:",
        "datatypes": "BIDS datatype folder:",
        "suffixes": "BIDS file suffix:",
        "diagnoses": "participant clinical diagnosis:",
        "dataset_names": "neuroimaging dataset name:",
        "authors": "neuroimaging dataset author:",
        "funding_sources": "neuroimaging dataset funding source:",
    }

    embeddings: Dict[str, list] = {}
    for category, vals in values.items():
        if not vals:
            embeddings[category] = []
            continue
        prefix = category_prefix.get(category, "")
        sentences = [f"{prefix} {v}" for v in vals]
        embs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)
        embeddings[category] = embs.tolist()
        print(f"  embedded {len(vals)} {category} values → shape {embs.shape}")

    return embeddings


# ── Upload to Modal Volume ─────────────────────────────────────────────────────

def save_index_atomic(path: Path, index: Dict[str, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(index, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


# ── Upload to Modal Volume ─────────────────────────────────────────────────────

def upload_to_modal(local_path: Path) -> None:
    try:
        import modal
    except ImportError:
        print("modal not installed — skipping upload. Install with: pip install modal")
        return

    volume = modal.Volume.from_name("bids-eye-metadata", create_if_missing=True)
    remote_path = "/metadata_index.json"

    with volume.batch_upload() as batch:
        batch.put_file(str(local_path), remote_path)

    print(f"Uploaded {local_path} → Modal Volume 'bids-eye-metadata':{remote_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", required=True,
                        help="PostgreSQL connection string")
    parser.add_argument("--out", type=Path, default=Path("/tmp/metadata_index.json"),
                        help="Local path to write the index file")
    parser.add_argument("--upload", action="store_true",
                        help="Upload the index to the Modal Volume after building")
    args = parser.parse_args()

    print("Querying DB for distinct values...")
    values = fetch_db_values(args.db_url)

    print("Building embeddings...")
    embeddings = build_embeddings(values)

    index = {"values": values, "embeddings": embeddings}

    save_index_atomic(args.out, index)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Saved index to {args.out} ({size_mb:.1f} MB)")

    if args.upload:
        print("Uploading to Modal Volume...")
        upload_to_modal(args.out)

    print("Done.")
    print()
    print("Next steps:")
    print("  1. modal deploy modal_app/app.py")
    print("  2. The inference app will load the index automatically.")


if __name__ == "__main__":
    main()
