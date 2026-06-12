"""
services/text_to_sql.py
-----------------------
Text-to-SQL service for BIDS-Eye.

Pipeline:
  1. Gemini preprocessing  → structured QueryPlan (intent extraction)
  2. RAG resolution        → canonical DB codes (diagnoses, tasks, scan types)
 3. Gemini SQL generation → raw PostgreSQL SELECT from augmented question +
                             schema + few-shot examples
  4. SQL rewrite service    → projection normalization, filter cleanup, and
                             VOCABULARY MISS fallback predicates
  5. Gemini error corrector → fixes any SQL that fails at runtime

Which mode is used:
  - If GEMINI_API_KEY is set → the pipeline above runs
  - If it is absent          → returns a bounded sample of datasets

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


_STOPWORDS = {
    "a","an","the","and","or","in","of","to","for","with","that","this",
    "is","are","do","does","me","my","i","all","any","no","not","from",
    "by","be","at","on","as","it","its","have","has","had","give","show",
    "find","list","which","what","how","get","tell","want","give","please",
    "can","could","would","datasets","dataset","data","participants",
}


def _extract_resolved_codes(augmented_question: str) -> Dict[str, List[str]]:
    """Parse resolved ontology codes from the augmented question string.

    Extracts canonical DB codes from lines like:
      o2.datatype = 'functional_mri'
      o2.task IN ('resting_state', 'n_back')
      p.diagnosis IN ('autism_spectrum_disorder')
    Returns a dict with keys 'datatype', 'task', 'diagnosis', 'suffix'.
    """
    codes: Dict[str, List[str]] = {"datatype": [], "task": [], "diagnosis": [], "suffix": []}
    for field in codes:
        for m in re.finditer(rf"\.{field}\s*=\s*'([^']+)'", augmented_question):
            v = m.group(1)
            if v not in codes[field]:
                codes[field].append(v)
        for m in re.finditer(rf"\.{field}\s+IN\s*\(([^)]+)\)", augmented_question, re.IGNORECASE):
            for v in re.findall(r"'([^']+)'", m.group(1)):
                if v not in codes[field]:
                    codes[field].append(v)
    return codes


def _pick_examples(pool: Dict[str, List[dict]], query_family: str,
                   n: int = 8, seed: str = "",
                   resolved_codes: Optional[Dict[str, List[str]]] = None) -> List[dict]:
    """Select the N most relevant few-shot examples from the pool.

    Scoring is a 3-tuple (code_overlap, keyword_overlap, family_priority):
    - code_overlap: # of resolved ontology codes (datatype/task/diagnosis/suffix) that
      appear in the example's meta tags. This is the primary signal when the RAG
      pipeline has resolved codes; when it hasn't, it contributes 0 and the
      selector degrades gracefully to keyword + family priority.
    - keyword_overlap: content-word overlap between the user question and the
      example question.
    - family_priority: +2 for the matched query family, +1 for fallback families.
    """
    if not pool or not seed:
        return []

    family_key = _FAMILY_ALIASES.get(query_family, query_family)

    all_examples: List[tuple] = []
    for fam, examples in pool.items():
        for ex in examples:
            all_examples.append((fam, ex))

    resolved_set: set = set()
    if resolved_codes:
        for field in ("datatype", "task", "diagnosis", "suffix"):
            resolved_set.update(resolved_codes.get(field, []))

    seed_words = {w for w in seed.lower().split() if w not in _STOPWORDS and len(w) > 2}

    family_priority = {family_key: 2}
    for fb in _FALLBACK_FAMILIES:
        family_priority.setdefault(fb, 1)

    def score(item: tuple) -> tuple:
        fam, ex = item
        meta = ex.get("meta", {})
        ex_codes: set = set()
        for field in ("datatype", "task", "diagnosis", "suffix"):
            ex_codes.update(meta.get(field, []))
        code_overlap = len(resolved_set & ex_codes)
        ex_words = {w for w in ex["question"].lower().split()
                    if w not in _STOPWORDS and len(w) > 2}
        keyword_overlap = len(seed_words & ex_words)
        prio = family_priority.get(fam, 0)
        return (code_overlap, keyword_overlap, prio)

    ranked = sorted(all_examples, key=score, reverse=True)

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
from services.sql_rewriter import FALLBACK_SQL as _FALLBACK_SQL  # noqa: E402
from services.sql_rewriter import extract_sql as _extract_sql  # noqa: E402
from services.sql_rewriter import rewrite_generated_sql as _rewrite_generated_sql  # noqa: E402


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
        raw = llm_generate(prompt, temperature=0.1, api_key=api_key)
        corrected = _rewrite_generated_sql(raw, question).sql
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
        "- Begin the query with EXACTLY: SELECT {{COLS}}\n"
        "  Do NOT list any column names after SELECT — write the literal placeholder "
        "{{COLS}}. The full column list is injected automatically afterwards. The "
        "alias `subject_count` is part of that injected list, so you may still "
        "reference it in HAVING / ORDER BY (e.g. HAVING COUNT(DISTINCT o.subject) > 30, "
        "ORDER BY subject_count DESC).\n"
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


# ── Pipeline (stages 1–2 + Gemini SQL as stage 3) ─────────────────────────────

def _run_local(question: str, api_key: str) -> TextToSQLResult:
    """
    Translation pipeline:
      Stage 1: Gemini preprocessing  → QueryPlan
      Stage 2: RAG resolution        → augmented question with DB codes
      Stage 3: Gemini SQL generation → SELECT statement (+ few-shot examples)
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
    resolved_codes = _extract_resolved_codes(augmented_question)
    examples = _pick_examples(_FEW_SHOT_POOL, query_family, n=5, seed=question,
                              resolved_codes=resolved_codes)
    examples_block = _format_examples(examples)
    if examples:
        log.info("Injecting %d few-shot examples for family '%s' (resolved codes: %s)",
                 len(examples), query_family,
                 {k: v for k, v in resolved_codes.items() if v})

    # Keep LLM generation and SQL rewriting separate: the model emits raw SQL,
    # then the rewrite service handles projection normalization, scan-filter
    # cleanup, and VOCABULARY MISS fallbacks.
    rewritten = _rewrite_generated_sql(
        _gemini_sql_generation(augmented_question, api_key, examples_block),
        augmented_question,
    )

    # Relevance ranking is applied later by services.sql_rewriter.build_page_sql
    # so the cached base SELECT can be reused across pages without rebaking the
    # order clause. We still decide *whether* to rank: skip for explicit top-N
    # / user-ordered queries so we don't override the order the user asked for.
    plan = augmented_plan.plan
    apply_relevance = plan.result_limit is None and plan.order_by is None
    scored_filters = [
        {"field": f.field, "code": f.code, "score": f.score}
        for f in augmented_plan.scored_filters
    ]

    return TextToSQLResult(
        sql=rewritten.sql,
        explanation=explanation,
        scored_filters=scored_filters,
        apply_relevance=apply_relevance,
        params=rewritten.params,
    )


# ── Public entry point ─────────────────────────────────────────────────────────

async def text_to_sql(question: str) -> TextToSQLResult:
    """
    Translate a natural-language question into a PostgreSQL SELECT statement via
    the Gemini pipeline. Falls back to a bounded sample of datasets when no
    GEMINI_API_KEY is configured or every LLM tier is unavailable.

    Always runs in a thread pool to avoid blocking the FastAPI event loop.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if api_key:
        log.info("Using Gemini pipeline")
        # Hard wall-clock deadline for the whole LLM pipeline (intent + RAG + SQL
        # generation). Cloudflare drops the request at ~100s; if the LLMs are
        # pathologically slow (Gemini 503 storm + slow fallback) we must return
        # *something* fast rather than let the browser see a NetworkError.
        deadline = float(os.getenv("PIPELINE_DEADLINE_SECONDS", "80"))
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run_local, question, api_key),
                timeout=deadline,
            )
        except LLMAllFailedError as exc:
            # Every LLM tier (all Gemini models + the Claude fallback) is down.
            # Degrade gracefully instead of 500ing the request.
            log.warning("All LLM providers failed (%s) — returning a sample of datasets", exc)
            return TextToSQLResult(
                sql=_FALLBACK_SQL,
                explanation="All LLM providers are currently unavailable — showing a sample of datasets. Please try again shortly.",
            )
        except asyncio.TimeoutError:
            log.warning("LLM pipeline exceeded %.0fs deadline — returning a sample of datasets", deadline)
            return TextToSQLResult(
                sql=_FALLBACK_SQL,
                explanation="That query is taking too long right now — showing a sample of datasets. Please try again or narrow your query.",
            )

    log.warning("No GEMINI_API_KEY set — returning a sample of datasets")
    return TextToSQLResult(
        sql=_FALLBACK_SQL,
        explanation="No API credentials configured — showing a sample of datasets.",
    )
