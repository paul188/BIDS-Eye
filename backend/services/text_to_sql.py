"""
services/text_to_sql.py
-----------------------
Text-to-SQL service for BIDS-Eye.

Full production pipeline (requires Modal deployment + GPU):
  1. Gemini preprocessing  → structured QueryPlan (intent extraction)
  2. RAG resolution        → canonical DB codes (diagnoses, tasks, scan types)
  3. SQLCoder + LoRA       → PostgreSQL SELECT from augmented question + schema
  4. Gemini error corrector → fixes any SQL that fails at runtime

Local development fallback (no GPU / no Modal):
  Steps 1–2 run locally. Step 3 is replaced by Gemini SQL generation.
  Step 4 (error correction) is still used on failed queries.

Which mode is used:
  - If MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are set → full pipeline via Modal
  - Otherwise                                        → local Gemini fallback
  - If GEMINI_API_KEY is also absent                 → returns all datasets

Set environment variables in .env (see .env.example).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Project-root modules on sys.path ──────────────────────────────────────────
# Works both locally (BIDS-Eye/) and inside Docker (/app/)
_ROOT = Path(__file__).resolve().parents[2]
for _mod_dir in (
    _ROOT / "LLM_preprocessor",
    _ROOT / "RAG",
    _ROOT / "synthetic_data_generation_and_train",
):
    if _mod_dir.exists() and str(_mod_dir) not in sys.path:
        sys.path.insert(0, str(_mod_dir))

from schemas import TextToSQLResult  # noqa: E402

# ── Fallback SQL: return all datasets when no API key is configured ────────────
_FALLBACK_SQL = (
    "SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
    "       d.source_type, d.remote_url, d.validation_status,\n"
    "       COUNT(DISTINCT o.subject) AS subject_count\n"
    "FROM bids_datasets d\n"
    "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
    "GROUP BY d.id\n"
    "ORDER BY d.name"
)

_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)


# ── SQL helpers ────────────────────────────────────────────────────────────────

def _extract_sql(raw: str) -> str:
    """Pull the first SELECT statement out of an LLM response."""
    m = re.search(r"\[SQL\]\s*(.*?)(?:\[/SQL\]|$)", raw, re.DOTALL | re.IGNORECASE)
    if m and re.match(r"(?i)select\b", m.group(1).strip()):
        return m.group(1).strip()
    m = _SQL_FENCE.search(raw)
    if m and re.match(r"(?i)select\b", m.group(1).strip()):
        return m.group(1).strip()
    stripped = raw.strip()
    if re.match(r"(?i)select\b", stripped):
        return stripped.split("\n\n")[0].strip()
    return _FALLBACK_SQL


def _correct_sql_with_gemini(sql: str, db_error: str, question: str, api_key: str) -> str:
    """
    Stage 4 — ask Gemini to fix a SQL query that failed with a PostgreSQL error.
    Returns the corrected SQL, or the original if correction fails.
    """
    try:
        from google import genai
        from google.genai import types
        from constants import SCHEMA_DDL
    except ImportError as exc:
        log.warning("Gemini correction unavailable: %s", exc)
        return sql

    prompt = (
        "A PostgreSQL query failed. Fix it and return ONLY the corrected SQL — "
        "no explanation, no markdown fences.\n\n"
        f"Original question:\n{question}\n\n"
        f"Database schema:\n{SCHEMA_DDL}\n\n"
        f"Failed SQL:\n{sql}\n\n"
        f"PostgreSQL error:\n{db_error}"
    )

    _MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]
    client = genai.Client(api_key=api_key)
    for model in _MODELS:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            corrected = _extract_sql(resp.text.strip())
            return corrected if corrected != _FALLBACK_SQL else sql
        except Exception as exc:
            log.warning("Gemini model %s failed during SQL correction: %s", model, exc)
    return sql


# ── Stage 3 local fallback: Gemini SQL generation ─────────────────────────────

def _gemini_sql_generation(augmented_question: str, api_key: str) -> str:
    """
    Local replacement for SQLCoder + LoRA when Modal is not available.
    Uses Gemini to generate SQL from the augmented question + schema.
    """
    try:
        from google import genai
        from google.genai import types
        from constants import SCHEMA_DDL
    except ImportError as exc:
        log.warning("Gemini SQL generation unavailable: %s", exc)
        return _FALLBACK_SQL

    instructions = (
        "- Only use tables and columns present in the schema.\n"
        "- Always SELECT: d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, "
        "d.source_type, d.remote_url, d.validation_status, COUNT(DISTINCT o.subject) AS subject_count\n"
        "- Always include: LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
        "- Always include: GROUP BY d.id\n"
        "- If [Resolved DB filters] are in the question, copy each EXISTS subquery VERBATIM "
        "into the WHERE clause.\n"
        "- Use ILIKE '%%term%%' for text search.\n"
        "- Do NOT add LIMIT unless the question asks for 'top N'.\n"
        "- Return ONLY the SQL — no explanation, no markdown fences.\n"
    )

    prompt = textwrap.dedent(f"""\
        ### Task
        Generate a SQL query to answer [QUESTION]{augmented_question}[/QUESTION]

        ### Instructions
        {instructions}
        ### Database Schema
        {SCHEMA_DDL}

        ### Answer
        [SQL]
    """)

    _MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]
    client = genai.Client(api_key=api_key)
    for model in _MODELS:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            sql = _extract_sql(resp.text.strip())
            log.info("Gemini SQL generation succeeded with model %s", model)
            return sql
        except Exception as exc:
            log.warning("Gemini model %s failed for SQL generation: %s", model, exc)

    return _FALLBACK_SQL


# ── Full Modal pipeline (stages 1–4) ──────────────────────────────────────────

def _run_via_modal(question: str) -> TextToSQLResult:
    """
    Full pipeline via Modal: SQLCoder + LoRA inference with Gemini correction.
    Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET to be set.
    """
    try:
        import modal
    except ImportError:
        raise RuntimeError("modal package not installed — cannot use Modal pipeline")

    Model = modal.Cls.from_name("bids-eye", "TextToSQLModel")
    model = Model()
    result = model.run.remote(question)

    sql = result.get("sql") or _FALLBACK_SQL
    explanation = result.get("query_plan", {}) and str(result["query_plan"])
    error = result.get("error")
    if error:
        log.warning("Modal pipeline returned error: %s", error)

    return TextToSQLResult(sql=sql, explanation=explanation or None)


# ── Local pipeline (stages 1–2 + Gemini SQL as stage 3) ───────────────────────

def _run_local(question: str, api_key: str) -> TextToSQLResult:
    """
    Local fallback pipeline:
      Stage 1: Gemini preprocessing  → QueryPlan
      Stage 2: RAG resolution        → augmented question with DB codes
      Stage 3: Gemini SQL generation → SELECT statement (replaces SQLCoder+LoRA)
    """
    try:
        from preprocess import run_pipeline
        from retriever import MetadataRetriever
    except ImportError as exc:
        log.warning("Pipeline modules not importable (%s) — using fallback SQL", exc)
        return TextToSQLResult(sql=_FALLBACK_SQL, explanation=str(exc))

    # Stage 2 retriever (author / funding name resolution)
    name_index_path = _ROOT / "RAG" / "name_index.json"
    retriever: Optional[MetadataRetriever] = None
    if name_index_path.exists():
        try:
            retriever = MetadataRetriever(str(name_index_path))
        except Exception as exc:
            log.warning("Could not load RAG name index: %s", exc)

    # Stages 1 + 2
    try:
        augmented_plan = run_pipeline(question, retriever=retriever, api_key=api_key)
        augmented_question = augmented_plan.augmented_question
        explanation = augmented_plan.plan.natural_language_summary
    except Exception as exc:
        log.warning("Preprocessing/RAG failed: %s — falling back to raw question", exc)
        augmented_question = question
        explanation = f"Preprocessing failed: {exc}"

    # Stage 3: Gemini SQL generation (local stand-in for SQLCoder+LoRA)
    sql = _gemini_sql_generation(augmented_question, api_key)

    return TextToSQLResult(sql=sql, explanation=explanation)


# ── Public entry point ─────────────────────────────────────────────────────────

async def text_to_sql(question: str) -> TextToSQLResult:
    """
    Translate a natural-language question into a PostgreSQL SELECT statement.

    Chooses the appropriate backend automatically:
      - Modal (full SQLCoder+LoRA pipeline) when Modal credentials are set
      - Local Gemini fallback otherwise

    Always runs in a thread pool to avoid blocking the FastAPI event loop.
    """
    has_modal = bool(os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if has_modal:
        log.info("Using Modal pipeline (SQLCoder + LoRA)")
        return await asyncio.to_thread(_run_via_modal, question)

    if api_key:
        log.info("Using local Gemini pipeline (no Modal credentials found)")
        return await asyncio.to_thread(_run_local, question, api_key)

    log.warning("No GEMINI_API_KEY or Modal credentials set — returning all datasets")
    return TextToSQLResult(
        sql=_FALLBACK_SQL,
        explanation="No API credentials configured — returning all datasets.",
    )
