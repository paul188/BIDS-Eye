"""
script.py
---------
Local smoke-test for the full BIDS-Eye query pipeline.

Exercises the same path the FastAPI backend uses:
  1. Gemini preprocessing  → structured QueryPlan
  2. RAG resolution        → canonical DB codes
  3. Gemini SQL generation → PostgreSQL SELECT
  4. (optional) DB execution against the local postgres_data instance

Usage:
  # SQL generation only (no DB):
  python script.py

  # With DB execution (requires postgres_data running):
  python script.py --db-url "postgresql://user:password@localhost:5432/bids_sql"

  # Custom question:
  python script.py --question "Find fMRI datasets with n-back tasks"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# ── Make backend package importable ───────────────────────────────────────────
_BACKEND = Path(__file__).resolve().parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.text_to_sql import text_to_sql, _run_pipeline_sync  # noqa: E402


async def run(question: str, db_url: str | None = None) -> None:
    print(f"\nQuestion: {question}\n")

    result = await text_to_sql(question)

    print("── Generated SQL ──────────────────────────────────────────────────")
    print(result.sql)

    if result.explanation:
        print("\n── Plan / explanation ─────────────────────────────────────────────")
        print(result.explanation)

    if db_url:
        print("\n── Executing against DB ───────────────────────────────────────────")
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(db_url)
            with engine.connect() as conn:
                rows = conn.execute(text(result.sql)).fetchall()
            print(f"Returned {len(rows)} row(s)")
            if rows:
                cols = list(rows[0]._fields) if hasattr(rows[0], "_fields") else list(range(len(rows[0])))
                col_w = {c: max(len(str(c)), max(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)}
                header = "  ".join(str(c).ljust(col_w[c]) for c in cols)
                print(header)
                print("-" * len(header))
                for row in rows[:50]:
                    print("  ".join(str(v).ljust(col_w[cols[i]]) for i, v in enumerate(row)))
                if len(rows) > 50:
                    print(f"  … and {len(rows) - 50} more rows")
        except Exception as exc:
            print(f"DB execution failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIDS-Eye local pipeline test")
    parser.add_argument(
        "--question",
        default="Find fMRI datasets with n-back tasks from NIH-funded studies",
        help="Natural-language search question",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="PostgreSQL URL to execute SQL against (optional). "
             "Example: postgresql://user:password@localhost:5432/bids_sql",
    )
    args = parser.parse_args()
    asyncio.run(run(args.question, args.db_url))
