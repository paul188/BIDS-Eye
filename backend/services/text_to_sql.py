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
import json
import logging
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ── Few-shot pool (loaded once at import time) ────────────────────────────────
_FEW_SHOT_POOL: Dict[str, List[dict]] = {}
_FEW_SHOT_PATH = Path(__file__).parent.parent / "few_shot_examples.json"
if _FEW_SHOT_PATH.exists():
    try:
        with open(_FEW_SHOT_PATH) as _f:
            _FEW_SHOT_POOL = json.load(_f)
        log.info("Loaded %d few-shot families from %s",
                 len(_FEW_SHOT_POOL), _FEW_SHOT_PATH)
    except Exception as _e:
        log.warning("Could not load few_shot_examples.json: %s", _e)

_FAMILY_ALIASES: Dict[str, str] = {
    "subject_multimodal": "subject_multimodal_query",
    "metadata_query":     "json_query",
    "json_numeric_query": "json_numeric_query",
}
_FALLBACK_FAMILIES = ["combined_filter", "concept_query", "scan_filter"]


def _pick_examples(pool: Dict[str, List[dict]], query_family: str,
                   n: int = 8, seed: str = "") -> List[dict]:
    """Select the N most relevant examples using keyword search across ALL families.

    Strategy:
    1. Score every example in the pool by keyword overlap with `seed` (the question).
    2. Break ties by preferring the matched query family, then fallback families.
    3. Return the top-N, deduplicated by question text.

    This lets the LLM see examples from across all 20 families rather than
    being limited to a single family — a 'fullsearch' over the example pool.
    """
    if not pool or not seed:
        return []

    # Build a flat list of all examples tagged with their family
    all_examples: List[tuple] = []
    family_key = _FAMILY_ALIASES.get(query_family, query_family)
    for fam, examples in pool.items():
        for ex in examples:
            all_examples.append((fam, ex))

    # Keyword overlap score: content words from the question
    _STOPWORDS = {
        "a","an","the","and","or","in","of","to","for","with","that","this",
        "is","are","do","does","me","my","i","all","any","no","not","from",
        "by","be","at","on","as","it","its","have","has","had","give","show",
        "find","list","which","what","how","get","tell","want","give","please",
        "can","could","would","datasets","dataset","data","participants",
    }
    seed_words = {w for w in seed.lower().split() if w not in _STOPWORDS and len(w) > 2}

    family_priority = {family_key: 2}
    for fb in _FALLBACK_FAMILIES:
        family_priority.setdefault(fb, 1)

    def score(item: tuple) -> tuple:
        fam, ex = item
        ex_words = {w for w in ex["question"].lower().split() if w not in _STOPWORDS and len(w) > 2}
        overlap = len(seed_words & ex_words)
        prio = family_priority.get(fam, 0)
        return (overlap, prio)

    ranked = sorted(all_examples, key=score, reverse=True)

    # Deduplicate by question and pick top-n
    seen_questions: set = set()
    picked: List[dict] = []
    for _, ex in ranked:
        q = ex["question"]
        if q not in seen_questions:
            seen_questions.add(q)
            picked.append(ex)
        if len(picked) >= n:
            break
    return picked


def _format_examples(examples: List[dict]) -> str:
    if not examples:
        return ""
    lines = ["### Examples", "Here are similar queries and their correct SQL:\n"]
    for ex in examples:
        lines.append(f"[QUESTION]{ex['question']}[/QUESTION]")
        lines.append(f"[SQL]\n{ex['sql'].strip()}\n[/SQL]\n")
    return "\n".join(lines)

# ── Project-root modules on sys.path ──────────────────────────────────────────
# Works both locally (BIDS-Eye/) and inside Docker (/app/)
_ROOT = Path(__file__).resolve().parents[2]
for _mod_dir in (
    _ROOT / "backend",
    _ROOT / "LLM_preprocessor",
    _ROOT / "RAG",
):
    if _mod_dir.exists() and str(_mod_dir) not in sys.path:
        sys.path.insert(0, str(_mod_dir))

from schemas import TextToSQLResult  # noqa: E402
from llm_client import llm_generate, LLMAllFailedError  # noqa: E402

# ── Fallback SQL: return a bounded sample of datasets when SQL generation is
# unavailable (no API key, or every LLM tier failed). The LIMIT is essential:
# an unbounded all-datasets response (1700+ datasets + every participant) is huge
# and slow, and gets dropped by Cloudflare's ~100s origin timeout / the browser,
# surfacing as "NetworkError when attempting to fetch resource". Keep it small.
_FALLBACK_SQL = (
    "SELECT d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
    "       d.source_type, d.remote_url, d.validation_status,\n"
    "       COUNT(DISTINCT o.subject) AS subject_count\n"
    "FROM bids_datasets d\n"
    "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
    "GROUP BY d.id\n"
    "ORDER BY d.name\n"
    "LIMIT 200"
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

    try:
        corrected = _extract_sql(llm_generate(prompt, temperature=0.1, api_key=api_key))
        return corrected if corrected != _FALLBACK_SQL else sql
    except LLMAllFailedError as exc:
        log.warning("All LLM tiers failed during SQL correction: %s", exc)
        return sql


# ── Stage 3 local fallback: Gemini SQL generation ─────────────────────────────

def _gemini_sql_generation(augmented_question: str, api_key: str,
                           examples_block: str = "") -> str:
    """
    Primary SQL generation using Gemini with few-shot examples.
    Previously a fallback; now the main pipeline.
    """
    try:
        from constants import SCHEMA_DDL
    except ImportError as exc:
        log.warning("Gemini SQL generation unavailable: %s", exc)
        return _FALLBACK_SQL

    instructions = (
        "## CRITICAL rules — violations produce wrong results\n"
        "1. suffix and datatype hold normalized codes, NOT raw BIDS values. "
        "Use ONLY these values — "
        "datatype: 'anatomical_mri' (structural MRI / T1w / T2w), 'functional_mri' (fMRI / BOLD), "
        "'diffusion_mri' (DWI), 'electroencephalography' (EEG), 'magnetoencephalography' (MEG), "
        "'intracranial_eeg' (iEEG/SEEG), 'behavioural_data', "
        "'positron_emission_tomography' (PET), 'field_maps', 'perfusion_asl', 'fnirs'. "
        "suffix: 'fmri_bold', 't1_weighted_mri', 't2_weighted_mri', 'diffusion_mri_dwi', "
        "'eeg', 'meg', 'intracranial_eeg', 'pet'. "
        "Never use raw BIDS values such as 'T1w', 'T2w', 'bold', 'dwi', 'anat', 'func', "
        "'eeg'/'meg' as datatype, 'beh', 'ieeg', 'fmap', 'perf', 'nirs'. "
        "Filter modality by datatype ONLY — do NOT combine a datatype filter "
        "with a suffix IN list for the same modality "
        "(e.g. structural MRI → just `datatype = 'anatomical_mri'`, never add suffix ORs).\n"
        "2. Disease/condition completeness: bids_participants.diagnosis is sparsely populated "
        "— most datasets about a disease have no structured diagnosis rows. "
        "For ANY disease filter, ALWAYS combine the EXISTS block with a name/description fallback: "
        "(EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id "
        "AND p2.diagnosis IN ('...')) OR d.name ILIKE '%<term>%' OR d.description_text ILIKE '%<term>%'). "
        "Derive <term> by stripping trailing '_disease', '_disorder', '_syndrome' from the code "
        "(e.g. 'parkinsons_disease' → 'parkinson', 'alzheimers_disease' → 'alzheimer', "
        "'autism_spectrum_disorder' → 'autism'). "
        "Never emit a bare diagnosis EXISTS without this OR fallback.\n"
        "3. Participant diagnosis: only add a diagnosis filter if one is explicitly present in "
        "[Resolved DB filters]. Words like 'patients', 'participants', 'subjects' refer to "
        "study participants in general — NOT a diagnosis constraint. "
        "Never generate `p.diagnosis != 'healthy_volunteer'` based only on those words.\n"
        "4. VOCABULARY MISS: If [VOCABULARY MISS] appears, those terms have no canonical code. "
        "Use ONLY the ILIKE clause shown in that block (copy it verbatim). "
        "Never assign a VOCABULARY MISS term to o.task, o.datatype, o.suffix, or p.diagnosis.\n"
        "## Important rules\n"
        "- [Resolved DB filters]: copy each EXISTS subquery VERBATIM into WHERE (AND them together). "
        "'scan X:' and 'scan task X:' for the same term are two separate EXISTS — keep both. "
        "If a 'subject count: HAVING ...' line is present, add that HAVING clause after GROUP BY d.id.\n"
        "- ILIKE: use only on bids_datasets.name or bids_datasets.description_text "
        "(for keyword/title searches). Never on diagnosis, task, datatype, or suffix "
        "(they hold canonical codes). Use the single most distinctive keyword "
        "(e.g. for 'drinking alcohol' use '%alcohol%', not '%drinking alcohol%') — "
        "never multi-word phrases when a single word suffices.\n"
        "- Do NOT add LIMIT unless the question asks for 'top N'.\n"
        "## Required query structure\n"
        "- Only use tables/columns present in the schema.\n"
        "- Always SELECT: d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, "
        "d.source_type, d.remote_url, d.validation_status, COUNT(DISTINCT o.subject) AS subject_count\n"
        "- Always include: LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
        "- Always include: GROUP BY d.id\n"
        "- Return ONLY the SQL — no explanation, no markdown fences.\n"
    )

    examples_section = f"\n{examples_block}\n" if examples_block else ""

    prompt = textwrap.dedent(f"""\
        ### Role
        You are an expert in neuroimaging research and the BIDS (Brain Imaging Data Structure) format.
        You generate precise PostgreSQL queries against a neuroimaging dataset catalog.
        Modality reference: fMRI → functional_mri, structural MRI / T1w / T2w → anatomical_mri,
        EEG → electroencephalography, MEG → magnetoencephalography, DWI → diffusion_mri,
        PET → positron_emission_tomography, iEEG / SEEG → intracranial_eeg.
        Clinical note: participant diagnoses (Alzheimer's, Parkinson's, epilepsy, schizophrenia, etc.)
        are sparsely recorded in structured rows — most datasets only mention them in name or description text.

        ### Task
        Generate a SQL query to answer [QUESTION]{augmented_question}[/QUESTION]

        ### Instructions
        {instructions}{examples_section}
        ### Database Schema
        {SCHEMA_DDL}

        ### Answer
        [SQL]
    """)

    # Cascade + retries + cross-provider fallback handled by llm_client; raises
    # LLMAllFailedError if every tier fails (caught upstream for graceful degrade).
    return _extract_sql(llm_generate(prompt, temperature=0.0, api_key=api_key))


# ── SQL safety: replace bad outer scan filters with VOCABULARY MISS ILIKE ───────

_BAD_OUTER_FILTER = re.compile(
    r'(?:AND\s+)?o\d*\.(task|datatype|suffix)\s*=\s*\'[^\']*\'',
    re.IGNORECASE,
)
# Extracts the ILIKE suggestion the preprocessor already computed for each miss
_VOCAB_MISS_ILIKE_RE = re.compile(
    r'For free-text search use:\s*(.+?)(?=\s*\])',
    re.IGNORECASE | re.DOTALL,
)


def _strip_outer_scan_filters(sql: str) -> tuple[str, bool]:
    """Remove o.task/datatype/suffix equality filters from the outer SELECT query.

    Returns (cleaned_sql, any_stripped).
    """
    out: list[str] = []
    depth = 0
    pos = 0
    stripped_any = False
    while pos < len(sql):
        ch = sql[pos]
        if ch == '(':
            depth += 1
            out.append(ch)
            pos += 1
        elif ch == ')':
            depth -= 1
            out.append(ch)
            pos += 1
        elif depth == 0:
            m = _BAD_OUTER_FILTER.match(sql, pos)
            if m:
                log.warning("Stripped bad outer scan filter: %s", m.group(0).strip())
                pos = m.end()
                stripped_any = True
            else:
                out.append(ch)
                pos += 1
        else:
            out.append(ch)
            pos += 1
    result = ''.join(out)
    # Clean up an empty WHERE clause
    result = re.sub(r'\bWHERE\s+(GROUP|ORDER|HAVING|LIMIT)\b', r'\1', result, flags=re.IGNORECASE)
    return result, stripped_any


def _fix_vocab_miss_sql(sql: str, augmented_question: str) -> str:
    """Strip hallucinated outer scan filters and replace with the VOCABULARY MISS
    ILIKE suggestion the preprocessor already computed.

    If the LLM writes `WHERE o.task = 'complex_motor_task'` instead of following
    the [VOCABULARY MISS] hint, this function:
      1. Removes the bad filter.
      2. Injects the ILIKE on d.name / d.description_text from the VOCABULARY MISS
         block — so the query still narrows results rather than returning everything.
    """
    # Step 1: strip any bad outer o.task/datatype/suffix = '...' filters
    cleaned, _ = _strip_outer_scan_filters(sql)

    # Step 2: extract VOCABULARY MISS ILIKE suggestions from the augmented question
    ilike_parts = [m.group(1).strip() for m in _VOCAB_MISS_ILIKE_RE.finditer(augmented_question)]
    if not ilike_parts:
        return cleaned  # No VOCABULARY MISS in this query — nothing to inject

    # Step 3: if the SQL already has an ILIKE the LLM wrote it correctly — leave it
    if re.search(r'\bILIKE\b', cleaned, re.IGNORECASE):
        return cleaned

    # Step 4: inject unconditionally — the LLM produced no filter at all for these terms
    combined = ' AND '.join(f'({p})' for p in ilike_parts)
    log.info("Injecting VOCABULARY MISS ILIKE (LLM produced no filter): %s", combined)

    if re.search(r'\bWHERE\b', cleaned, re.IGNORECASE):
        cleaned = re.sub(r'\bWHERE\b', f'WHERE {combined} AND', cleaned, count=1, flags=re.IGNORECASE)
    else:
        cleaned = re.sub(r'\bGROUP\s+BY\b', f'WHERE {combined}\nGROUP BY', cleaned, count=1, flags=re.IGNORECASE)

    return cleaned


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

    try:
        Model = modal.Cls.from_name("bids-eye", "TextToSQLModel")
    except Exception as exc:
        raise RuntimeError(f"Modal app 'bids-eye' not found or not deployed: {exc}") from exc

    model = Model()
    result = model.run.remote(question)

    raw_sql = result.get("sql") or _FALLBACK_SQL
    augmented = result.get("augmented_question", question)
    sql = _fix_vocab_miss_sql(raw_sql, augmented)
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

    # Stage 2 retriever (author / funding name resolution + optional DB back-check)
    name_index_path = _ROOT / "RAG" / "name_index.json"
    retriever: Optional[MetadataRetriever] = None
    if name_index_path.exists():
        try:
            retriever = MetadataRetriever(
                str(name_index_path),
                db_url=os.getenv("DATABASE_URL"),
            )
        except Exception as exc:
            log.warning("Could not load RAG name index: %s", exc)

    # Stages 1 + 2
    augmented_plan = run_pipeline(question, retriever=retriever, api_key=api_key)
    augmented_question = augmented_plan.augmented_question
    query_family = augmented_plan.plan.query_family.value
    explanation = augmented_plan.plan.natural_language_summary

    # Stage 3: Gemini SQL generation with few-shot examples
    examples = _pick_examples(_FEW_SHOT_POOL, query_family, n=5, seed=question)
    examples_block = _format_examples(examples)
    if examples:
        log.info("Injecting %d few-shot examples for family '%s'", len(examples), query_family)

    sql = _fix_vocab_miss_sql(
        _gemini_sql_generation(augmented_question, api_key, examples_block),
        augmented_question,
    )

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
        modal_timeout = int(os.getenv("MODAL_REQUEST_TIMEOUT", "120"))
        log.info("Using Modal pipeline (SQLCoder + LoRA), timeout=%ds", modal_timeout)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run_via_modal, question),
                timeout=modal_timeout,
            )
        except asyncio.TimeoutError:
            log.warning("Modal timed out after %ds — falling back to local Gemini", modal_timeout)
        except Exception as exc:
            log.warning("Modal failed (%s) — falling back to local Gemini", exc)

    if api_key:
        log.info("Using local Gemini pipeline")
        try:
            return await asyncio.to_thread(_run_local, question, api_key)
        except LLMAllFailedError as exc:
            # Every LLM tier (all Gemini models + the Claude fallback) is down.
            # Degrade gracefully instead of 500ing the request.
            log.warning("All LLM providers failed (%s) — returning all datasets", exc)
            return TextToSQLResult(
                sql=_FALLBACK_SQL,
                explanation="All LLM providers are currently unavailable — showing a sample of datasets. Please try again shortly.",
            )

    log.warning("No GEMINI_API_KEY or Modal credentials set — returning all datasets")
    return TextToSQLResult(
        sql=_FALLBACK_SQL,
        explanation="No API credentials configured — showing a sample of datasets.",
    )
