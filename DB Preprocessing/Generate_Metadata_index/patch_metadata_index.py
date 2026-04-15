#!/usr/bin/env python3
"""
training_data_generation/patch_metadata_index.py
-------------------------------------------------
Extend an existing metadata_index.json with two new categories:
  authors         — all distinct author strings from bids_datasets.authors
  funding_sources — all distinct funding strings from bids_datasets.funding

Use this when you already have a built index (tasks/diagnoses/datatypes/suffixes)
and only want to add the author and funding vectors without rebuilding everything.

Usage (run on HPC with PostgreSQL running):
  python training_data_generation/patch_metadata_index.py \\
      --db-url "postgresql://user:password@localhost:5429/bids_sql" \\
      --index  /path/to/metadata_index.json

The file is updated in-place; a .bak backup is created first.

SLURM usage:
  This script needs DB access — run it inside a SLURM job that follows the
  standard pattern (copy DB from Lustre, start pg_ctl, run script, stop pg_ctl).
  See training_data_generation/build_metadata_index_job.sh for the template.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


# Mirrors build_metadata_index.py and rag.py
_CATEGORY_PREFIX = {
    "authors":         "neuroimaging dataset author:",
    "funding_sources": "neuroimaging dataset funding source:",
}

_NEW_CATEGORIES = list(_CATEGORY_PREFIX.keys())


# ── DB extraction ──────────────────────────────────────────────────────────────

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


def fetch_authors_and_funding(db_url: str) -> dict[str, list[str]]:
    """
    Extract all distinct author and funding strings from bids_datasets.
    Supports both TEXT[] and JSON/JSONB array storage.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)

    column_specs = {
        "authors": ("authors", "author"),
        "funding_sources": ("funding", "funding_source"),
    }

    values: dict[str, list[str]] = {}
    with engine.connect() as conn:
        for category, (column_name, value_alias) in column_specs.items():
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
                print(f"  [WARN] {category} query failed: {exc} — skipping",
                      file=sys.stderr)
                conn.rollback()
                values[category] = []

    return values


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_categories(values: dict[str, list[str]]) -> dict[str, list]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers.\n"
            "Activate a venv that includes it before running this script.\n"
            "For HPC jobs, use a venv like $SCRATCH/train_venv or install with:\n"
            "  pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings: dict[str, list] = {}

    for category, vals in values.items():
        if not vals:
            embeddings[category] = []
            print(f"  {category}: empty — skipped")
            continue
        prefix = _CATEGORY_PREFIX.get(category, "")
        sentences = [f"{prefix} {v}".strip() for v in vals]
        embs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)
        embeddings[category] = embs.tolist()
        print(f"  {category}: embedded {len(vals)} values → shape {embs.shape}")

    return embeddings


# ── Patch index file ──────────────────────────────────────────────────────────

def _load_existing_index(index_path: Path) -> tuple[dict, Path]:
    backup_path = index_path.with_suffix(".json.bak")
    candidates = [index_path, backup_path]
    failures: list[str] = []

    for candidate in candidates:
        if not candidate.exists():
            failures.append(f"{candidate}: missing")
            continue
        if candidate.stat().st_size == 0:
            failures.append(f"{candidate}: empty")
            continue

        try:
            with open(candidate, encoding="utf-8") as fh:
                idx = json.load(fh)
            if not isinstance(idx, dict):
                raise ValueError("top-level JSON value must be an object")
            return idx, candidate
        except Exception as exc:
            failures.append(f"{candidate}: {type(exc).__name__}: {exc}")

    failure_text = "\n".join(f"  - {item}" for item in failures)
    raise SystemExit(
        f"Index is not valid JSON: {index_path}\n"
        "Tried the main file and backup, but neither could be loaded:\n"
        f"{failure_text}\n\n"
        "Rebuild a fresh base index first with:\n"
        "  sbatch training_data_generation/build_metadata_index_job.sh\n"
        "Then rerun this patch script."
    )


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(payload, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def patch_index(index_path: Path, idx: dict, source_path: Path,
                new_values: dict[str, list[str]],
                new_embeddings: dict[str, list]) -> None:
    bak = index_path.with_suffix(".json.bak")

    if source_path == index_path and (not bak.exists() or bak.stat().st_size == 0):
        shutil.copy2(index_path, bak)
        print(f"Backup → {bak}")
    elif source_path == bak:
        print(f"Recovered base index from backup → {bak}")

    idx.setdefault("values", {})
    idx.setdefault("embeddings", {})

    for cat in _NEW_CATEGORIES:
        old_count = len(idx["values"].get(cat, []))
        idx["values"][cat]     = new_values.get(cat, [])
        idx["embeddings"][cat] = new_embeddings.get(cat, [])
        new_count = len(idx["values"][cat])
        verb = "Added" if old_count == 0 else "Updated"
        print(f"  {verb} '{cat}': {old_count} → {new_count} values")

    _write_json_atomic(index_path, idx)
    size_mb = index_path.stat().st_size / 1e6
    print(f"Saved patched index → {index_path} ({size_mb:.1f} MB)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db-url", required=True,
        help="PostgreSQL connection string, e.g. postgresql://user:pw@localhost:5429/bids_sql",
    )
    parser.add_argument(
        "--index", type=Path,
        default=Path(__file__).resolve().parents[1] / "LLM_preprocessor" / "metadata_index.json",
        help="Path to the metadata_index.json to patch (default: LLM_preprocessor/metadata_index.json)",
    )
    args = parser.parse_args()

    if not args.index.exists():
        raise SystemExit(
            f"Index not found: {args.index}\n"
            "Build it first with:\n"
            "  sbatch training_data_generation/build_metadata_index_job.sh"
        )

    print(f"Index: {args.index}")
    print(f"DB:    {args.db_url}\n")

    print("Preflight: validating base index...")
    idx, source_path = _load_existing_index(args.index)
    print(f"  Base index loaded from: {source_path}")

    print("Step 1: Extracting author and funding values from DB...")
    values = fetch_authors_and_funding(args.db_url)

    print("\nStep 2: Building embeddings...")
    embeddings = embed_categories(values)

    print("\nStep 3: Patching index...")
    patch_index(args.index, idx, source_path, values, embeddings)

    print("\nDone. The following categories were added/updated:")
    for cat in _NEW_CATEGORIES:
        n = len(values.get(cat, []))
        print(f"  {cat}: {n} values")
    print("\nRe-deploy the Modal app if using Modal inference:")
    print("  modal deploy modal_app/app.py")


if __name__ == "__main__":
    main()
