"""
services/translate/text_to_sql.py
---------------------------------
Text-to-SQL service for BIDS-Eye.

Pipeline:
  1. Gemini preprocessing  → structured QueryPlan (intent extraction)
  2. RAG resolution        → canonical DB codes (diagnoses, tasks, scan types)
 3. SQL construction      → deterministic compiler for supported typed plans,
                             otherwise Gemini SQL generation with examples
  4. SQL rewrite service    → projection normalization, filter cleanup,
                             VOCABULARY MISS fallback predicates, and expansion
                             of text_match() into full-text search
  5. Gemini error corrector → fixes any SQL that fails at runtime

Which mode is used:
  - If GEMINI_API_KEY is set → the pipeline above runs
  - If it is absent          → returns a bounded sample of datasets

Set environment variables in .env (see .env.example).
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
import json
import logging
import math
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ── Few-shot pool (loaded once at import time) ────────────────────────────────
_FEW_SHOT_POOL: Dict[str, List[dict]] = {}
_FEW_SHOT_PATH = Path(__file__).resolve().parents[2] / "few_shot_examples.json"
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
# Whole prompt character guard, including system policy and generated schema.
# This is deliberately a conservative growth safeguard, not a token counter.
_SQL_PROMPT_CHAR_BUDGET = 48_000


_STOPWORDS = {
    "a","an","the","and","or","in","of","to","for","with","that","this",
    "is","are","do","does","me","my","i","all","any","no","not","from",
    "by","be","at","on","as","it","its","have","has","had","give","show",
    "find","list","which","what","how","get","tell","want","give","please",
    "can","could","would","datasets","dataset","data",
}

_EXAMPLE_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TOKEN_ALIASES = {
    "aged": "age", "ages": "age", "younger": "young", "youngest": "young",
    "older": "old", "oldest": "old", "children": "child", "participants": "participant",
    "subjects": "subject", "datasets": "dataset", "recently": "recent",
    "highest": "high", "lowest": "low", "studies": "study",
    # Common irregular plurals that the conservative suffix rules below cannot
    # reduce without conflating unrelated words (e.g. disease/diseases).
    "analyses": "analysis", "axes": "axis", "bases": "basis",
    "crises": "crisis", "diagnoses": "diagnosis", "hypotheses": "hypothesis",
    "metastases": "metastasis", "psychoses": "psychosis", "syntheses": "synthesis",
    "theses": "thesis",
}


def _example_tokens(text: str) -> List[str]:
    """Normalise words for deterministic few-shot retrieval (no model call)."""
    tokens = []
    for token in _EXAMPLE_TOKEN_RE.findall(text.lower().replace("_", " ")):
        if token in _STOPWORDS or len(token) <= 2:
            continue
        token = _TOKEN_ALIASES.get(token, token)
        # Small, conservative normalisation catches plural/tense variants
        # without adding a stemmer dependency or collapsing technical terms.
        if token.endswith("ies") and len(token) > 5:
            token = token[:-3] + "y"
        elif token.endswith("ing") and len(token) > 6:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 4:
            token = token[:-1]
        tokens.append(token)
    return tokens


def _intent_features(text: str, family: str = "", meta: Optional[dict] = None) -> set:
    """Typed, rule-based intent cues shared by the question and example."""
    # Only populated metadata contributes evidence. Stringifying an all-empty
    # `requested` dict used to inject the words "diagnosis", "task", etc. and
    # falsely require those categories for otherwise unrelated questions.
    meta_text = " ".join(
        f"{key} {value}" for key, value in (meta or {}).items() if value
    ) if isinstance(meta, dict) else str(meta or "")
    value = f"{text} {family} {meta_text}".lower().replace("_", " ")
    words = set(_EXAMPLE_TOKEN_RE.findall(value))
    intents = set()
    if words & {"age", "aged", "young", "younger", "old", "older", "child", "children",
                "adolescent", "adult", "infant", "pediatric", "paediatric"}:
        intents.add("age")
    if words & {"participant", "participants", "subject", "subjects", "sample", "cohort"}:
        intents.add("participants")
    if words & {"diagnosis", "diagnosed", "patient", "patients", "schizophrenia", "autism",
                "epilepsy", "parkinson", "depression", "bipolar", "healthy", "control", "controls"}:
        intents.add("clinical")
    if words & {"both", "versus", "vs", "compare", "comparison", "control", "controls"}:
        intents.add("case_control")
    if words & {"without", "exclude", "excluding", "absent", "lack", "lacking", "neither", "no", "not"}:
        intents.add("absence")
    if words & {"functional", "fmri", "eeg", "meg", "dwi", "diffusion", "anatomical", "mri",
                "pet", "fnirs", "asl", "bold", "modality", "datatype"}:
        intents.add("modality")
    if words & {"task", "paradigm", "rest", "resting", "nback", "stimulus", "stimuli"}:
        intents.add("task")
    if words & {"tr", "repetition", "echo", "tesla", "strength", "voxel", "scanner", "sequence",
                "sampling", "bandwidth", "manufacturer", "acquisition"}:
        intents.add("acquisition")
    if words & {"recent", "recently", "newest", "date", "year", "last", "past", "months", "days"}:
        intents.add("recency")
    if (words & {"recent", "recently", "newest", "date", "year", "last", "past", "months", "days"}
            and words & {"dataset", "datasets", "openneuro"}
            and not words & {"paper", "papers", "article", "journal", "pubmed"}):
        intents.add("dataset_publication_date")
    if (words & {"date", "year", "last", "past", "months", "days", "recent", "recently"}
            and words & {"paper", "papers", "article", "journal", "pubmed", "publication"}):
        intents.add("paper_publication_date")
    if words & {"top", "popular", "highest", "lowest", "largest", "smallest", "most", "least", "downloads"}:
        intents.add("ranking")
    if words & {"tr", "repetition"}:
        intents.add("repetition_time")
    if words & {"te", "echo"}:
        intents.add("echo_time")
    if words & {"tesla"} or {"field", "strength"}.issubset(words):
        intents.add("field_strength")
    if words & {"manufacturer", "vendor", "siemens", "philips", "ge"}:
        intents.add("scanner_vendor")
    if words & {"paper", "papers", "publication", "doi", "pubmed", "reference", "references", "article"}:
        intents.add("paper_reference")
    if words & {"author", "authors", "lab", "laboratory", "institution", "institutions", "site", "sites"}:
        intents.add("provenance")
    if words & {"license", "licence", "redistributable", "permission", "share"}:
        intents.add("license")
    if words & {"session", "sessions", "longitudinal", "visit", "visits"}:
        intents.add("sessions")
    if meta:
        if meta.get("datatype"):
            intents.add("modality")
        if meta.get("task"):
            intents.add("task")
        if meta.get("diagnosis"):
            intents.add("clinical")
    return intents


def _query_structure(question: str) -> set:
    """Infer SQL-shape cues from wording; this only helps choose examples."""
    raw = question.lower()
    words = set(_EXAMPLE_TOKEN_RE.findall(raw))
    structures = set()
    if words & {"without", "no", "not", "neither", "exclude", "excluding", "absent", "lack", "lacking"}:
        structures.add("absence")
    elif re.search(r"\b(any|with|contain|include|have|featur)\w*\b", raw):
        structures.add("presence")
    if words & {"both", "versus", "vs", "compare", "comparison"}:
        structures.add("multi_condition")
    if (words & {"count", "average", "mean", "minimum", "maximum", "fewer", "least", "more", "greater"}
            or re.search(r"\b(at least|at most|more than|fewer than|less than|greater than)\b", raw)):
        structures.add("aggregate")
        structures.add("threshold")
    if re.search(r"\b(top|popular|recent|recently|highest|lowest|largest|smallest|newest|most|least)\b", raw):
        structures.add("ranking")
    if (words & {"age", "aged", "young", "younger", "old", "older", "child", "children", "adolescent"}
            or re.search(r"\b\d+\s*(years?\s*old|or younger|or older)\b", raw)):
        structures.add("age_filter")
    if words & {"echo", "repetition", "spacing", "field", "strength", "scanner", "sequence", "sampling", "tesla"}:
        structures.add("metadata_filter")
    if words & {"title", "named", "name", "called"}:
        structures.add("name_filter")
    if words & {"task", "paradigm", "rest", "resting"}:
        structures.add("task_filter")
    return structures


@lru_cache(maxsize=512)
def _sql_shape(sql: str) -> frozenset:
    """Extract logical SQL features from an example, without executing it."""
    text = sql.lower().replace("{{cols}}", "d.id")
    features = set()
    if re.search(r"\bnot\s+exists\s*\(", text):
        features.add("absence")
    if re.search(r"(?<!not )\bexists\s*\(", text):
        features.add("presence")
    if re.search(r"\bhaving\b|\b(count|avg|average|min|max)\s*\(", text):
        features.add("aggregate")
    if re.search(r"\border\s+by\b", text):
        features.add("ranking")
    if re.search(r"\bbetween\b|>=|<=|(?<!\w)>(?!\w)|(?<!\w)<(?!\w)", text):
        features.add("threshold")
    if re.search(r"\b(age|age_min|age_max)\b", text):
        features.add("age_filter")
    if re.search(r"\bdiagnosis\b", text):
        features.add("clinical_filter")
    if re.search(r"\bsex\b", text):
        features.add("sex_filter")
    if re.search(r"other_entities|->>|jsonb", text):
        features.add("metadata_filter")
    if "text_match(" in text or "readme_text" in text or "description_text" in text:
        features.add("text_search")
    if "paper_references" in text or "pubmed" in text or "doi" in text:
        features.add("paper_reference")
    if "bids_participants" in text:
        features.add("participants")
    if "bids_objects" in text:
        features.add("scan_data")
    if re.search(r"\bjoin\b", text):
        features.add("join")
    if len(re.findall(r"\bexists\s*\(", text)) > 1:
        features.add("multi_condition")
    return frozenset(features)


def _bm25_scores(query_tokens: List[str], documents: List[List[str]]) -> List[float]:
    """BM25 over the small, fixed example pool; stable and dependency-free."""
    if not query_tokens or not documents:
        return [0.0] * len(documents)
    n_docs = len(documents)
    lengths = [len(doc) for doc in documents]
    avg_len = sum(lengths) / max(n_docs, 1)
    document_frequency: Dict[str, int] = {}
    term_frequencies: List[Dict[str, int]] = []
    for document in documents:
        counts: Dict[str, int] = {}
        for token in document:
            counts[token] = counts.get(token, 0) + 1
        term_frequencies.append(counts)
        for token in counts:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    query_frequency: Dict[str, int] = {}
    for token in query_tokens:
        query_frequency[token] = query_frequency.get(token, 0) + 1
    scores = []
    for length, counts in zip(lengths, term_frequencies):
        score = 0.0
        for token, qf in query_frequency.items():
            tf = counts.get(token, 0)
            if not tf:
                continue
            df = document_frequency[token]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + 1.2 * (1.0 - 0.75 + 0.75 * length / max(avg_len, 1.0))
            score += idf * (tf * 2.2 / denom) * min(qf, 2)
        scores.append(score)
    return scores


@lru_cache(maxsize=512)
def _static_guard_reason(sql: str) -> Optional[str]:
    """Cached static guard check for an example's SQL template."""
    from services.query_guard import check_static
    return check_static(sql.replace("{{COLS}}", "d.id"))


def _family_key(query_family: str, pool: Dict[str, List[dict]]) -> str:
    """Resolve a legacy family alias only when the exact pool key is absent."""
    if query_family in pool:
        return query_family
    alias = _FAMILY_ALIASES.get(query_family)
    return alias if alias in pool else query_family


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
                   n: int = 5, seed: str = "",
                   resolved_codes: Optional[Dict[str, List[str]]] = None) -> List[dict]:
    """Select safe, relevant and varied examples with deterministic retrieval.

    Relevance combines typed ontology-code coverage, BM25 over question and
    example metadata, query-family match, and SQL-shape cues. Greedy MMR then
    reduces near-duplicates so a prompt teaches several useful patterns. It
    never calls an LLM. The weights and diversity penalties below are an
    **ASSISTANT PROPOSAL — REVIEW THESE VALUES**; keep the selection method
    deterministic so they can be tuned against held-out search results.
    """
    if not pool or not seed:
        return []

    family_key = _family_key(query_family, pool)

    all_examples: List[dict] = []
    for fam, examples in pool.items():
        for ex in examples:
            # Don't teach the generator an unavailable-column query or SQL the
            # static safety guard would reject. Runtime EXPLAIN is not repeated
            # here: it depends on the live database and would add latency.
            if _uses_unavailable_column(ex["sql"]):
                continue
            guard_reason = _static_guard_reason(ex["sql"])
            if guard_reason:
                log.debug("Skipping unsafe few-shot example %r: %s", ex["question"], guard_reason)
                continue
            all_examples.append({"family": fam, "example": ex, "order": len(all_examples)})
    if not all_examples:
        return []

    fields = ("datatype", "task", "diagnosis", "suffix")
    requested = {
        field: set((resolved_codes or {}).get(field, []))
        for field in fields
    }
    has_codes = any(requested.values())
    query_tokens = _example_tokens(seed)
    query_shape = _query_structure(seed)
    query_categories = _intent_features(seed, family_key, requested)
    specific_categories = query_categories & {
        "age", "clinical", "acquisition", "recency", "paper_reference",
        "provenance", "license", "sessions", "dataset_publication_date",
        "paper_publication_date",
    }
    technical_topics = specific_categories & {
        "repetition_time", "echo_time", "field_strength", "scanner_vendor",
    }

    documents: List[List[str]] = []
    for item in all_examples:
        ex = item["example"]
        meta = ex.get("meta", {})
        codes = {field: set(meta.get(field, [])) for field in fields}
        item["codes"] = codes
        item["question_tokens"] = set(_example_tokens(ex.get("question", "")))
        item["structure"] = (set(ex.get("structure", [])) |
                              set(_sql_shape(ex.get("sql", ""))) |
                              _query_structure(ex.get("question", "")))
        item["categories"] = _intent_features(ex.get("question", ""), item["family"], meta)
        item["pattern"] = ex.get("pattern", "")
        # BM25 is deliberately based on the natural-language example only;
        # exact values and SQL categories are scored in separate dimensions.
        item["tokens"] = _example_tokens(ex.get("question", ""))
        documents.append(item["tokens"])

    bm25 = _bm25_scores(query_tokens, documents)
    max_bm25 = max(bm25, default=0.0)
    for item, lexical_score in zip(all_examples, bm25):
        # A query may ask for several independent specific dimensions (e.g.
        # institution AND linked paper). A single example need only teach one
        # requested dimension; requiring every category in every example can
        # eliminate the entire pool when no combined example exists.
        if ((specific_categories and not specific_categories.intersection(item["categories"])) or
                (technical_topics and not technical_topics.intersection(item["categories"]))):
            item["relevance"] = 0.0
            continue
        matched = sum(len(requested[field] & item["codes"][field]) for field in fields)
        requested_n = sum(len(requested[field]) for field in fields)
        example_n = sum(len(item["codes"][field]) for field in fields)
        if requested_n:
            recall = matched / requested_n
            precision = matched / max(example_n, 1)
            code_score = 0.8 * recall + 0.2 * precision
        else:
            code_score = 0.0
        lexical = lexical_score / max_bm25 if max_bm25 else 0.0
        family = (1.0 if item["family"] == family_key else
                  0.4 if item["family"] in _FALLBACK_FAMILIES else 0.0)
        shape = (0.8 * len(query_shape & item["structure"]) / len(query_shape) +
                 0.2 * len(query_shape & item["structure"]) / max(len(item["structure"]), 1)
                 if query_shape else 0.0)
        category = (0.8 * len(query_categories & item["categories"]) / len(query_categories) +
                    0.2 * len(query_categories & item["categories"]) / max(len(item["categories"]), 1)
                    if query_categories else 0.0)
        # REVIEW (assistant-proposed): values get only one quarter of the
        # relevance weight. Topic, logic, category and family signals together
        # carry three quarters, to avoid choosing examples just because they
        # share a diagnosis/task code while teaching the wrong SQL structure.
        if has_codes:
            relevance = (0.25 * code_score + 0.27 * lexical + 0.26 * shape +
                         0.14 * category + 0.08 * family)
        else:
            relevance = 0.34 * lexical + 0.30 * shape + 0.22 * category + 0.14 * family
        item["relevance"] = relevance
        item["features"] = {
            "question": item["question_tokens"],
            "codes": {f"{field}:{code}" for field in fields for code in item["codes"][field]},
            "categories": item["categories"],
            "structure": item["structure"],
            "pattern": {item["pattern"]} if item["pattern"] else set(),
            "family": {item["family"]},
        }

    def overlap(left: dict, right: dict) -> float:
        """Weighted feature Jaccard used as the MMR redundancy penalty."""
        weights = (("question", 0.30), ("codes", 0.15), ("categories", 0.20),
                   ("structure", 0.25), ("pattern", 0.06), ("family", 0.04))
        total = 0.0
        for key, weight in weights:
            a, b = left["features"][key], right["features"][key]
            if a or b:
                total += weight * len(a & b) / len(a | b)
        return total

    # Keep diversity soft: topic/logical matches remain more important than
    # forcing an unrelated family into the prompt.
    family_cap = max(2, math.ceil(n * 0.8))
    pattern_cap = max(2, math.ceil(n * 0.6))
    min_relevance = max(0.28, 0.55 * max(item["relevance"] for item in all_examples))
    # REVIEW (assistant proposal): the absolute and relative cutoffs avoid
    # padding a prompt with weak matches; validate the values on a disjoint set.
    picked_items: List[dict] = []
    family_counts: Dict[str, int] = {}
    pattern_counts: Dict[str, int] = {}
    remaining = list(all_examples)
    while remaining and len(picked_items) < n:
        best = None
        best_rank = (-math.inf, -math.inf, -math.inf)
        for item in remaining:
            if item["relevance"] < min_relevance:
                continue
            redundancy = max((overlap(item, prior) for prior in picked_items), default=0.0)
            adjusted = item["relevance"] - 0.12 * redundancy
            if family_counts.get(item["family"], 0) >= family_cap:
                adjusted -= 0.12
            if item["pattern"] and pattern_counts.get(item["pattern"], 0) >= pattern_cap:
                adjusted -= 0.08
            # Deterministic ties: relevance first, then original pool order.
            rank = (adjusted, item["relevance"], -item["order"])
            if rank > best_rank:
                best, best_rank = item, rank
        if best is None:
            break
        remaining.remove(best)
        question_key = " ".join(_example_tokens(best["example"].get("question", "")))
        if any(question_key == " ".join(_example_tokens(p["example"].get("question", "")))
               for p in picked_items):
            continue
        picked_items.append(best)
        family_counts[best["family"]] = family_counts.get(best["family"], 0) + 1
        if best["pattern"]:
            pattern_counts[best["pattern"]] = pattern_counts.get(best["pattern"], 0) + 1
    return [item["example"] for item in picked_items]


def _format_examples(examples: List[dict], max_chars: Optional[int] = None) -> str:
    if not examples:
        return ""
    lines = ["### Examples", "Similar query and correct SQL:\n"]
    if max_chars is not None and len("\n".join(lines)) > max_chars:
        return ""
    for ex in examples:
        item = (f"[QUESTION]{ex['question']}[/QUESTION]\n"
                f"[SQL]\n{ex['sql'].strip()}\n[/SQL]")
        candidate = "\n".join([*lines, item])
        if max_chars is not None and len(candidate) > max_chars:
            break  # preserve rank order and keep only complete examples
        lines.append(item)
    return "\n".join(lines)

# ── Project-root modules on sys.path ──────────────────────────────────────────
# Works both locally (BIDS-Eye/) and inside Docker (/app/)
_ROOT = Path(__file__).resolve().parents[3]
for _mod_dir in (
    _ROOT / "backend",
    _ROOT / "LLM_preprocessor",
    _ROOT / "RAG",
):
    if _mod_dir.exists() and str(_mod_dir) not in sys.path:
        sys.path.insert(0, str(_mod_dir))

from schemas import TextToSQLResult  # noqa: E402
from llm_client import llm_generate, LLMAllFailedError  # noqa: E402
from services.translate.sql_rewriter import FALLBACK_SQL as _FALLBACK_SQL  # noqa: E402
from services.translate.sql_rewriter import extract_sql as _extract_sql  # noqa: E402
from services.translate.sql_rewriter import rewrite_generated_sql as _rewrite_generated_sql  # noqa: E402
from services.stores import translation_store  # noqa: E402
from services.catalogue.schema_capabilities import (  # noqa: E402
    available_columns as _available_columns,
    uses_unavailable_column as _uses_unavailable_column,
)
from services.translate.participant_query_compiler import (  # noqa: E402
    compile_plan as _compile_plan,
)


def _correct_sql_with_gemini(sql: str, db_error: str, question: str, api_key: str) -> str:
    """
    Stage 4 — ask Gemini to fix a SQL query that failed with a PostgreSQL error.
    Returns the corrected SQL, or the original if correction fails.
    """
    try:
        from constants import build_schema_ddl
    except ImportError as exc:
        log.warning("Gemini correction unavailable: %s", exc)
        return sql
    schema_ddl = build_schema_ddl(_available_columns())

    prompt = (
        "A PostgreSQL query failed. Fix it and return ONLY the corrected SQL — "
        "no explanation, no markdown fences.\n\n"
        f"Original question:\n{question}\n\n"
        f"Database schema:\n{schema_ddl}\n\n"
        f"Failed SQL:\n{sql}\n\n"
        f"PostgreSQL error:\n{db_error}"
    )

    try:
        raw = llm_generate(prompt, temperature=0.0, api_key=api_key,
                           purpose="sql_correction")
        corrected = _rewrite_generated_sql(raw, question).sql
        return corrected if corrected != _FALLBACK_SQL else sql
    except LLMAllFailedError as exc:
        log.warning("All LLM tiers failed during SQL correction: %s", exc)
        return sql


# ── Stage 3 local fallback: Gemini SQL generation ─────────────────────────────

def _gemini_sql_generation(augmented_question: str, api_key: str,
                           examples: Optional[List[dict]] = None) -> str:
    """
    Primary SQL generation using Gemini with few-shot examples.
    Previously a fallback; now the main pipeline.
    """
    try:
        from constants import build_schema_ddl
    except ImportError as exc:
        log.warning("Gemini SQL generation unavailable: %s", exc)
        return _FALLBACK_SQL
    schema_ddl = build_schema_ddl(_available_columns())

    # The model never writes a text predicate itself. It copies or writes
    # text_match('...') placeholders, which services/fulltext/text_search.py expands
    # into full-text search (or ILIKE on a database without search_tsv).
    long_text_cols = ["d.description_text"]
    if "readme_text" in _available_columns():
        long_text_cols.append("d.readme_text")
    long_text_phrase = " or ".join(long_text_cols)

    instructions = (
        "## 1. Hard correctness constraints\n"
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
        "Each 'diagnosis:' line in [Resolved DB filters] therefore already pairs the EXISTS "
        "block with a text fallback: "
        "(EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id "
        "AND p2.diagnosis IN ('...')) OR text_match('...', 'short')). "
        "Copy the whole parenthesised expression verbatim, including every term inside "
        "text_match and an AND text_match(...) that some lines add after it. "
        "Never emit the diagnosis EXISTS without its OR text_match(...) part.\n"
        "3. Participant diagnosis: only add a diagnosis filter if one is explicitly present in "
        "[Resolved DB filters]. Words like 'patients', 'participants', 'subjects' refer to "
        "study participants in general — NOT a diagnosis constraint. "
        "Never generate `p.diagnosis != 'healthy_volunteer'` based only on those words. "
        "'Healthy', 'normal', 'controls' add NO condition at all (no diagnosis filter, no "
        "text_match): most datasets include healthy participants and few record it. "
        "If a 'participant sex:', 'participant age:' or 'participant sex and age:' line is present, copy it verbatim as its "
        "own AND condition and write no other condition on sex or age; never combine sex, age "
        "and diagnosis in one EXISTS (that demands one participants row with all three, and "
        "most datasets record only some of them).\n"
        "4. VOCABULARY MISS: If [VOCABULARY MISS] appears, those terms have no canonical code. "
        "Use ONLY the text_match(...) clause shown in that block (copy it verbatim). "
        "Never assign a VOCABULARY MISS term to o.task, o.datatype, o.suffix, or p.diagnosis.\n"
        "## 2. Build filters from the current request\n"
        "- [Resolved DB filters]: copy each filter expression VERBATIM into WHERE (AND them together). "
        "'scan X:' and 'scan task X:' for the same term are two separate EXISTS — keep both. "
        "If a 'subject count: HAVING ...' line is present, add that HAVING clause after GROUP BY d.id. "
        "If a 'dataset size: ...' line is present, AND that condition into WHERE verbatim.\n"
        "- Text search: text_match('<keyword>') is a search macro that the system expands into "
        "a full-text predicate over the dataset's name, description, study fields and README. "
        "It is a boolean condition for WHERE and combines with AND / OR / NOT. Use it only for "
        "a subject the DATA must be about and that no line in [Resolved DB filters] already "
        "covers. Add NO text_match (and no other condition) for: what the asker wants to do with "
        "the data ('for machine learning', 'for classification', 'to test my pipeline'); wishes "
        "about quality, difficulty or documentation ('standard', 'simple', 'non-trivial', "
        "'well documented', 'labeled', 'complete', 'sufficient sample size', 'images included'); "
        "file formats; what a publication reports; a dataset named only as an example or "
        "comparison ('like ABIDE', 'similar to DEAP'). Such words describe the request, not the "
        "dataset text, and a condition on them returns nothing. Use the single most distinctive keyword "
        "(e.g. for 'drinking alcohol' write text_match('alcohol')) — several words in one "
        "term must ALL occur, so add words only when each one is essential. "
        "Separate alternatives with ' | ': text_match('anaesthesia | anesthesia'). "
        "Word forms are matched automatically ('depression' also finds 'depressive'). "
        f"Never write ILIKE on {long_text_phrase} to look for a word; the one exception is "
        "a URL fragment such as '%doi.org%', which text_match cannot find. "
        "Use d.name ILIKE '%word%' only when the question asks for a word in the dataset "
        "name or title. ILIKE remains correct for short fields: d.license, "
        "unnest(d.paper_references), scanner strings in other_entities. "
        "Never use text_match or ILIKE on diagnosis, task, datatype, or suffix "
        "(they hold canonical codes).\n"
        "- Alternatives: data types or groups listed as options ('fMRI, EEG or fNIRS', "
        "'brain signals (fMRI, EEG, fNIRS)', 'MI, ERP and SSVEP paradigms' as kinds of dataset) "
        "mean ANY of them: join their conditions with OR. Use AND only when one dataset must "
        "contain all of them ('simultaneous EEG and fMRI', 'T1w and BOLD for every subject'). "
        "A modality mentioned only in a comparison ('EEG data, similar to the fMRI collection X') "
        "is not a filter.\n"
        "- Do NOT add LIMIT unless the question asks for 'top N'.\n"
        "## 3. Required SQL structure\n"
        "- Only use tables/columns present in the schema.\n"
        "- Begin the query with EXACTLY: SELECT {{COLS}}\n"
        "  Do NOT list any column names after SELECT — write the literal placeholder "
        "{{COLS}}. The full column list is injected automatically afterwards. The "
        "alias `subject_count` is part of that injected list, so you may still "
        "reference it in HAVING / ORDER BY (e.g. HAVING COUNT(DISTINCT o.subject) > 30, "
        "ORDER BY subject_count DESC).\n"
        "- Always include: LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
        "- Always include: GROUP BY d.id\n"
    )

    system_prompt = textwrap.dedent(f"""\
        You generate PostgreSQL queries for the BIDS-Eye neuroimaging dataset catalog.

        {instructions}
        Return only the SQL query, with no explanation or markdown fences.
    """)

    # Keep live context together in the user message, with the actionable request
    # last. Drop complete lower-ranked examples if the full request grows beyond
    # the character guard; never truncate an SQL example mid-statement.
    schema_section = (
        '<database_schema authority="only available tables and columns">\n'
        f"{schema_ddl}\n</database_schema>"
    )
    request_section = f"<current_request>\n{augmented_question}\n</current_request>"
    base_prompt = f"{schema_section}\n{request_section}"
    # "Illustrative only" made the model drop parts of a matching example
    # (the text fallback of "7 Tesla" in 2 of 4 live runs, 2026-11-15).
    examples_open = ('<examples purpose="correct SQL for similar requests; when one matches '
                     'the current request, keep every part of its WHERE clause">\n')
    examples_close = "\n</examples>\n"
    example_budget = max(
        0,
        _SQL_PROMPT_CHAR_BUDGET - len(system_prompt) - len(base_prompt)
        - len(examples_open) - len(examples_close),
    )
    examples_block = _format_examples(examples or [], max_chars=example_budget)
    examples_section = (
        f"{examples_open}{examples_block}{examples_close}" if examples_block else ""
    )
    prompt = f"{schema_section}\n{examples_section}{request_section}"
    if len(system_prompt) + len(prompt) > _SQL_PROMPT_CHAR_BUDGET:
        log.warning(
            "SQL prompt exceeds %d-character guard before examples (system=%d, user=%d)",
            _SQL_PROMPT_CHAR_BUDGET, len(system_prompt), len(prompt),
        )

    # Cascade + retries + cross-provider fallback handled by llm_client; raises
    # LLMAllFailedError if every tier fails (caught upstream for graceful degrade).
    return _extract_sql(llm_generate(
        prompt, system=system_prompt, temperature=0.0, api_key=api_key, purpose="sql"))


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

    # A question translated before answers stages 1-3 from the store; the
    # compiler and the rewriter below run as for a new question.
    stored, augmented_plan = _stored_translation(question)
    model_usage = (_start_usage_collection()
                   if translation_store.enabled() or translation_store.capture_file() else [])

    if augmented_plan is None:
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
        augmented_plan = run_pipeline(
            question, retriever=retriever, api_key=api_key,
            available_columns=_available_columns(),
        )
    augmented_question = augmented_plan.augmented_question
    query_family = augmented_plan.plan.query_family.value
    explanation = augmented_plan.plan.natural_language_summary

    # Stage 3: compile cohort plans (diagnosis groups, optionally with scan
    # requirements) deterministically. The LLM has already interpreted the
    # query and RAG resolved each group; asking another model to reconstruct
    # their AND/OR semantics as SQL adds variance without adding information.
    # Unsupported plans still use the established SQL-generation path below.
    resolved_codes = _extract_resolved_codes(augmented_question)
    plan = augmented_plan.plan
    compiled = _compile_plan(plan, getattr(augmented_plan, "resolved", None) or [],
                             augmented_question)
    raw_sql = compiled.sql
    model_sql: Optional[str] = None
    if (stored is not None and stored.model_sql
            and stored.status == translation_store.VERIFIED):
        # A reviewer's SQL stands, also where the plan could be compiled.
        raw_sql = model_sql = stored.model_sql
        log.info("SQL path: stored, verified (no model call)")
    elif raw_sql:
        log.info("SQL path: compiled (%d participant groups, no SQL generation)",
                 len(plan.groups))
    elif stored is not None and stored.model_sql:
        raw_sql = model_sql = stored.model_sql
        log.info("SQL path: stored (no model call)")
    else:
        # One line per search: which plans the compiler declines, and why,
        # shows what to support next.
        log.info("SQL path: model (compiler declined: %s)", compiled.reason)
        examples = _pick_examples(_FEW_SHOT_POOL, query_family, n=5, seed=question,
                                  resolved_codes=resolved_codes)
        if examples:
            log.info("Injecting %d few-shot examples for family '%s' (resolved codes: %s)",
                     len(examples), query_family,
                     {k: v for k, v in resolved_codes.items() if v})
        raw_sql = model_sql = _gemini_sql_generation(augmented_question, api_key, examples)

    # Keep LLM generation and SQL rewriting separate: the model emits raw SQL,
    # or the supported typed-plan compiler emits it; the rewrite service then
    # handles projection normalization, scan-filter cleanup, and text search.
    rewritten = _rewrite_generated_sql(
        raw_sql,
        augmented_question,
        # Word forms decided for this question (RAG/word_forms.py); {} mostly.
        getattr(augmented_plan, "word_forms", None) or None,
    )

    # Relevance ranking is applied later by services.translate.sql_rewriter.build_page_sql
    # so the cached base SELECT can be reused across pages without rebaking the
    # order clause. We still decide *whether* to rank: skip for explicit top-N
    # / user-ordered queries so we don't override the order the user asked for.
    plan = augmented_plan.plan
    apply_relevance = plan.result_limit is None and plan.order_by is None
    scored_filters = [
        {"field": f.field, "code": f.code, "score": f.score}
        for f in augmented_plan.scored_filters
    ]

    if stored is None and not _fallback_models(model_usage):
        translation_store.capture(question, augmented_plan, model_sql)

    return TextToSQLResult(
        sql=rewritten.sql,
        explanation=explanation,
        scored_filters=scored_filters,
        apply_relevance=apply_relevance,
        params=rewritten.params,
        # Without this the router ranks by code matches only: the text rank
        # (ts_rank_cd over search_tsv) is silently left out.
        rank_tsquery=rewritten.rank_tsquery,
        filter_sql=rewritten.filter_sql,
        relaxed_sql=rewritten.relaxed_sql,
        value_fixes=rewritten.value_fixes,
        from_store=stored is not None and not model_usage,
        store_entry=_store_entry(question, stored, augmented_plan, model_sql, model_usage),
    )


def _start_usage_collection() -> list:
    from llm_client import start_usage_collection
    return start_usage_collection()


def _store_entry(question: str, stored, augmented_plan, model_sql: Optional[str],
                 model_usage: list):
    """The entry to keep for this translation, or None.

    None for a hit that needed no model (nothing new), and for a translation
    in which a fallback model answered: that one should not stand for a week.
    """
    if stored is not None and not model_usage:
        return None
    fallback = _fallback_models(model_usage)
    if fallback:
        log.info("translation store: not kept, answered by fallback model %s", fallback)
        return None
    return translation_store.candidate(question, augmented_plan, model_sql)


def _fallback_models(model_usage: list) -> list:
    """Models other than the first of the cascade that answered in this translation."""
    from llm_client import _build_cascade
    cascade = _build_cascade()
    primary = cascade[0].model if cascade else None
    return sorted({u["model"] for u in model_usage if u["model"] != primary})


def _stored_translation(question: str):
    """(entry, its plan) for a question translated before, else (None, None)."""
    stored = translation_store.lookup(question)
    if stored is None:
        return None, None
    try:
        from preprocess import AugmentedQueryPlan
        return stored, AugmentedQueryPlan.model_validate(stored.plan)
    except Exception as exc:  # noqa: BLE001 — an unreadable entry is a miss
        log.warning("translation store: stored plan not readable (%s)", exc)
        return None, None


# ── Public entry point ─────────────────────────────────────────────────────────

def keyword_fallback(question: str, explanation: str) -> TextToSQLResult:
    """Plain keyword search when no language model answered.

    Matches any content word of the question in the dataset text and ranks by
    full-text rank, so the best matches come first. Without the full-text
    column, or without usable words, the bounded sample as before.
    """
    from services.fulltext import text_search
    from services.catalogue.schema_capabilities import has_fts
    from services.translate.sql_rewriter import DEFAULT_DATASET_PROJECTION

    query = text_search.keyword_query(question) if has_fts() else None
    if not query:
        return TextToSQLResult(sql=_FALLBACK_SQL, explanation=explanation + " Showing a sample of datasets.")
    sql = (
        f"SELECT {DEFAULT_DATASET_PROJECTION}\n"
        "FROM bids_datasets d\n"
        "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
        f"WHERE d.{text_search.FTS_COLUMN} @@ to_tsquery('{text_search.TS_CONFIG}', '{query}')\n"
        "GROUP BY d.id"
    )
    return TextToSQLResult(
        sql=sql, explanation=explanation + " Showing a plain keyword search for your question instead.",
        apply_relevance=True, rank_tsquery=query, filter_sql=sql,
    )


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
            log.warning("All LLM providers failed (%s) — falling back to a keyword search", exc)
            return keyword_fallback(
                question, "The language model is unavailable right now; please try again shortly.")
        except asyncio.TimeoutError:
            log.warning("LLM pipeline exceeded %.0fs deadline — falling back to a keyword search", deadline)
            return keyword_fallback(
                question, "Understanding that question took too long right now.")

    log.warning("No GEMINI_API_KEY set — returning a sample of datasets")
    return TextToSQLResult(
        sql=_FALLBACK_SQL,
        explanation="No API credentials configured — showing a sample of datasets.",
    )
