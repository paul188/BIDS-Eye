"""
LLM_preprocessor/preprocess.py
--------------------------------
Stage 1 of the Text-to-SQL pipeline.

Flow:
  User query
      │
      ▼
  preprocess_query()          ← this module
      │  returns QueryPlan (structured Pydantic object)
      │
      ▼
  RAG  (RAG/retriever.py MetadataRetriever)
      │  receives rag_requests from QueryPlan
      │  returns resolved canonical codes per term
      │  scan terms are searched across datatype + suffix + task simultaneously
      │
      ▼
  augment_with_rag()          ← this module
      │  returns AugmentedQueryPlan with canonical codes filled in
      │
      ▼
  SQL generation LLM (SQLCoder via modal_app/app.py)

The LLM's job is STRUCTURAL ONLY:
  - Identify which terms are scan-related (put in ScanRequirement.scan_terms)
  - Identify participant filters (diagnosis, sex, age, handedness)
  - Identify dataset-level filters (author, funding, license, DOI)
  - Identify sidecar / JSONB filters (RepetitionTime > 2.0)

The RAG layer handles ALL cardinality decisions using threshold-based fuzzy
matching.  The LLM never sets expected code counts or classifies scan terms
into datatype vs suffix vs task.

Authentication:
  export GEMINI_API_KEY=...   or   export GOOGLE_API_KEY=...

Requires:
  pip install google-genai pydantic
"""

from __future__ import annotations

import json
import os
import textwrap
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class QueryFamily(str, Enum):
    # Participant / cohort focused
    CONCEPT_QUERY         = "concept_query"         # "datasets with schizophrenia patients"
    PARTICIPANT_FILTER    = "participant_filter"     # "≥15 male ambidextrous with ADHD"
    COMPARISON_QUERY      = "comparison_query"       # "more male than female participants"
    AGE_QUERY             = "age_query"              # "participants over 65 / neonates"

    # Scan / modality focused
    SCAN_FILTER           = "scan_filter"            # "fMRI datasets with n-back task"
    MULTIMODAL_QUERY      = "multimodal_query"       # "has both anat and fMRI"
    ABSENCE_QUERY         = "absence_query"          # "no EEG data"
    SUBJECT_MULTIMODAL    = "subject_multimodal"     # "same subject has T1w AND did n-back"

    # Dataset metadata focused
    AUTHOR_QUERY          = "author_query"           # "datasets by Jane Doe"
    DESCRIPTION_SEARCH    = "description_search"     # "datasets with 'network' in name"
    DOI_QUERY             = "doi_query"              # "datasets with / without a DOI"
    LICENSE_QUERY         = "license_query"          # "datasets under CC0"
    FUNDING_QUERY         = "funding_query"          # "funded by NIH"

    # Sidecar / JSONB metadata
    JSON_NUMERIC_QUERY    = "json_numeric_query"     # "RepetitionTime > 2.0"
    METADATA_QUERY        = "metadata_query"         # "has / lacks AcquisitionTime field"

    # Structural / aggregate
    SESSION_QUERY         = "session_query"          # "≥4 sessions per subject"
    RANKING_QUERY         = "ranking_query"          # "top 10 by subject count"
    AGGREGATE_QUERY       = "aggregate_query"        # "average age per dataset"
    COMBINED_FILTER       = "combined_filter"        # mix of the above


class RAGField(str, Enum):
    DIAGNOSIS = "diagnosis"
    TASK      = "task"
    SUFFIX    = "suffix"
    DATATYPE  = "datatype"
    AUTHOR    = "author"
    FUNDING   = "funding"
    SCAN      = "scan"      # internal: multi-field lookup across datatype+suffix+task


class RAGRequest(BaseModel):
    """
    Instruction to the RAG layer to resolve a natural-language term into
    canonical DB codes.

    The RAG layer uses threshold-based fuzzy matching — no n_results needed.
    For SCAN field requests, the retriever searches datatype, suffix, and task
    simultaneously and returns all matches above the similarity threshold.
    """
    field: RAGField
    terms: List[str] = Field(
        description="Raw user terms to look up, e.g. ['schizophrenia', 'psychosis']"
    )


class MetadataFilter(BaseModel):
    """A filter on bids_objects.other_entities (JSONB sidecar fields)."""
    json_field: str = Field(
        description="Exact sidecar key, e.g. 'RepetitionTime', 'AcquisitionTime'"
    )
    operator: Literal["=", ">", "<", ">=", "<=", "!=", "exists", "not_exists"]
    value: Optional[Union[float, int, str]] = Field(
        default=None,
        description="Omit for exists / not_exists operators"
    )


class ParticipantGroup(BaseModel):
    """
    One cohort of participants, e.g. 'patients' or 'controls'.
    For a simple single-population query there will be exactly one group.
    For a schizophrenia-vs-controls query there will be two.
    """
    group_name: str = Field(
        description="Short label, e.g. 'patient_group', 'control_group', 'all_participants'"
    )

    # --- fields resolved by RAG ---
    diagnosis_terms: List[str] = Field(
        default_factory=list,
        description="Raw diagnosis strings from the query. RAG resolves these to canonical codes."
    )
    task_terms: List[str] = Field(
        default_factory=list,
        description="Task strings tied to this participant group (if group-specific)."
    )

    # --- directly usable fields ---
    sex: List[str] = Field(
        default_factory=list,
        description="e.g. ['male'], ['female'], ['M', 'F']"
    )
    handedness: List[str] = Field(
        default_factory=list,
        description="e.g. ['right'], ['left'], ['ambidextrous']"
    )
    age_min: Optional[float] = Field(default=None, description="Minimum participant age")
    age_max: Optional[float] = Field(default=None, description="Maximum participant age")
    min_subjects: Optional[int] = Field(
        default=None,
        description="Minimum number of participants in this group that the dataset must have"
    )
    extra_participant_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Non-standard participants.tsv columns (bids_participants.extra JSONB), "
                    "e.g. {'group': 'control', 'bmi': '>25'}"
    )


class ScanRequirement(BaseModel):
    """
    A scan-level constraint: the dataset must (or must not) contain files
    matching the specified scan / modality / task terms.

    Put ALL scan-related terms here — do NOT classify them into datatype vs
    suffix vs task.  The RAG layer searches all three fields simultaneously
    and returns whichever canonical codes match above the similarity threshold.
    """
    scan_terms: List[str] = Field(
        default_factory=list,
        description=(
            "All terms describing scan types, modalities, or tasks in this requirement. "
            "Include everything scan-related: 'fMRI', 'BOLD', 'T1w', 'n-back', "
            "'resting state', 'EEG', 'diffusion tensor imaging', 'bold'. "
            "Do NOT classify into datatype / suffix / task — the RAG layer does that."
        )
    )
    required: bool = Field(
        default=True,
        description="True = dataset MUST contain matching files; False = must NOT contain them"
    )
    same_subject: bool = Field(
        default=False,
        description="True = the same subject must have ALL of these scan types "
                    "(used for within-subject multi-modal queries)"
    )


class QueryPlan(BaseModel):
    """
    Complete structured representation of a user query.
    Passed to the RAG layer, then to the SQL generation LLM.
    """

    # ── Intent ────────────────────────────────────────────────────────────────
    query_family: QueryFamily
    natural_language_summary: str = Field(
        description="One sentence restating what the user wants in precise terms"
    )

    # ── Participant cohorts ───────────────────────────────────────────────────
    groups: List[ParticipantGroup] = Field(default_factory=list)
    group_operator: Literal["AND", "OR"] = Field(
        default="AND",
        description="AND = dataset must contain ALL groups; OR = any one suffices"
    )

    # ── Scan / modality requirements ─────────────────────────────────────────
    scan_requirements: List[ScanRequirement] = Field(
        default_factory=list,
        description="Each entry is one modality/task constraint on the dataset's files"
    )

    # ── Dataset-level text filters ────────────────────────────────────────────
    author_names: List[str] = Field(
        default_factory=list,
        description="Author names as mentioned by the user — RAG resolves these to exact DB strings"
    )
    name_search: Optional[str] = Field(
        default=None,
        description="Substring to ILIKE-search in bids_datasets.name"
    )
    description_search: Optional[str] = Field(
        default=None,
        description="Substring to ILIKE-search in bids_datasets.description_text"
    )
    license_search: Optional[str] = Field(
        default=None,
        description="License string to search for, e.g. 'CC0', 'PDDL'"
    )
    doi_required: Optional[bool] = Field(
        default=None,
        description="True = must have a DOI; False = must NOT have a DOI; None = no constraint"
    )
    funding_terms: List[str] = Field(
        default_factory=list,
        description="Funding source strings as mentioned by the user — RAG resolves these to exact DB strings"
    )

    # ── Sidecar / JSONB metadata filters ─────────────────────────────────────
    metadata_filters: List[MetadataFilter] = Field(
        default_factory=list,
        description="Filters on bids_objects.other_entities JSONB sidecar fields"
    )

    # ── Session / longitudinal ────────────────────────────────────────────────
    min_sessions: Optional[int] = Field(
        default=None,
        description="Minimum number of distinct sessions a subject must have"
    )

    # ── Comparison / aggregate ────────────────────────────────────────────────
    count_comparison: Optional[str] = Field(
        default=None,
        description="Free-text comparison to resolve, e.g. 'male_count > female_count', "
                    "'female_count > male_count'"
    )

    # ── Result config ─────────────────────────────────────────────────────────
    result_limit: Optional[int] = Field(
        default=None,
        description="Top-N limit if the user asked for a specific number; None otherwise"
    )
    order_by: Optional[str] = Field(
        default=None,
        description="Field to order by, e.g. 'subject_count', 'd.name'"
    )
    order_direction: Literal["ASC", "DESC"] = Field(default="DESC")

    # ── RAG requests (leave empty — auto-derived by build_rag_requests) ───────
    rag_requests: List[RAGRequest] = Field(
        default_factory=list,
        description="Leave empty. Auto-derived from the plan fields by build_rag_requests(). "
                    "Only set if you need to add a lookup not covered by the standard fields."
    )


# ---------------------------------------------------------------------------
# Post-RAG augmented plan
# ---------------------------------------------------------------------------

class ResolvedTerms(BaseModel):
    """RAG output for one field: mapping from raw user term to canonical DB codes."""
    field: RAGField
    term_to_codes: Dict[str, List[str]] = Field(
        description="e.g. {'schizophrenia': ['schizophrenia_spectrum', 'schizophrenia']}"
    )

    @property
    def all_codes(self) -> List[str]:
        seen: set[str] = set()
        result = []
        for codes in self.term_to_codes.values():
            for c in codes:
                if c not in seen:
                    seen.add(c)
                    result.append(c)
        return result


class AugmentedQueryPlan(BaseModel):
    """
    QueryPlan + resolved canonical codes from the RAG layer.
    This is what the SQL generation LLM receives.
    """
    plan: QueryPlan
    resolved: List[ResolvedTerms] = Field(default_factory=list)
    augmented_question: str = Field(
        description="The original question with canonical DB values injected as context"
    )

    def codes_for(self, field: RAGField) -> List[str]:
        for r in self.resolved:
            if r.field == field:
                return r.all_codes
        return []


# ---------------------------------------------------------------------------
# LLM call — preprocess a user query into a QueryPlan
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are the preprocessing stage of a BIDS neuroimaging dataset search engine.
    Your job is to parse a natural-language user query into a strict JSON structure
    (a QueryPlan) that downstream components will use to build a SQL query.

    The database contains neuroimaging datasets with:
      - bids_datasets: name, authors, license, doi, funding, description_text
      - bids_objects:  task, suffix, datatype, session, subject,
                       other_entities (JSONB sidecar metadata)
      - bids_participants: diagnosis, sex, handedness, age, extra (JSONB)

    YOUR JOB IS STRUCTURAL ONLY.  Extract what the user is asking for and put
    each piece of information in the right field.  Do NOT try to guess canonical
    DB codes — a downstream RAG layer handles all code resolution using
    threshold-based fuzzy matching.

    ── Scan / modality terms ───────────────────────────────────────────────────
    Put ALL scan-related terms in ScanRequirement.scan_terms.
    Do NOT try to classify them as datatype vs suffix vs task — the RAG searches
    all three fields simultaneously and returns whatever matches.

    Examples:
      "fMRI datasets"                → scan_terms: ["fMRI"]
      "bold T1w scans"               → scan_terms: ["bold", "T1w"]
      "functional MRI with n-back"   → scan_terms: ["functional MRI", "n-back"]
      "resting state fMRI"           → scan_terms: ["resting state", "fMRI"]
      "EEG and eye-tracking"         → scan_terms: ["EEG", "eye-tracking"]
      "has both anat and func data"  → two ScanRequirements:
                                         [{scan_terms: ["anat"]}, {scan_terms: ["func"]}]

    ── Participant terms ───────────────────────────────────────────────────────
    Put diagnosis / condition strings in ParticipantGroup.diagnosis_terms.
    These are resolved by the RAG layer (e.g. "depression" → canonical code).

    ── Dataset-level terms ─────────────────────────────────────────────────────
    Put author names in author_names; funding agencies in funding_terms.
    The RAG resolves these to exact DB strings.

    ── Counts and n_results ────────────────────────────────────────────────────
    Do NOT set any n_results, expected_*_codes, or n_*_codes fields.
    The RAG layer decides how many codes to return based on similarity threshold.

    ── rag_requests ────────────────────────────────────────────────────────────
    Leave rag_requests as an empty list [].  It is auto-derived from the plan.

    ── query_family guidelines ─────────────────────────────────────────────────
      concept_query        — asks for datasets containing a concept (diagnosis/modality)
      participant_filter   — filters by participant properties (sex, age, handedness, count)
      comparison_query     — compares counts between groups (male vs female)
      age_query            — filters by participant age range
      scan_filter          — filters by scan type + task
      multimodal_query     — requires presence/absence of multiple datatypes
      absence_query        — "does NOT contain X"
      subject_multimodal   — same subject must have multiple scan types
      author_query         — filter by author name
      description_search   — ILIKE search on dataset name or description
      doi_query            — has / lacks a DOI
      license_query        — filter by license
      funding_query        — filter by funding source
      json_numeric_query   — numeric filter on JSONB sidecar field
      metadata_query       — existence check on JSONB sidecar field
      session_query        — filter by session count
      ranking_query        — order by some metric, top N
      aggregate_query      — compute aggregate (average, count) across datasets
      combined_filter      — combination of two or more of the above

    Return ONLY valid JSON matching the QueryPlan schema — no prose, no fences.
""")


# Model cascade — pinned version IDs required; unpinned aliases like
# "gemini-2.0-flash" have been deprecated for newer API accounts.
_MODEL_CASCADE = [
    # Das aktuelle Flaggschiff für Geschwindigkeit & Intelligenz (GA-Status)
    ("gemini-3.1-pro",         "Primary",    2, [30, 60]),
    
    # Der hocheffiziente Allrounder (stabil und kostengünstig)
    ("gemini-3.1-flash",       "Fallback 1", 2, [30, 60]),
    
    # Die Ultra-Leichtgewicht-Variante für einfache Aufgaben/Massendaten
    ("gemini-3.1-flash-lite",  "Fallback 2", 3, [60, 120, 240]),
    
    # Optional: Spezialmodell für komplexe Logik (Reasoning-Fokus)
    ("gemini-3-deep-think",    "Fallback 3", 5, [120, 240, 480]),
]


def _call_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini with the model cascade; return raw text response."""
    import time

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    failures: list[str] = []

    for model, role, max_attempts, waits in _MODEL_CASCADE:
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_SYSTEM_PROMPT,
                        temperature=0.1,
                    ),
                )
                return response.text.strip()
            except Exception as exc:
                err_str = str(exc)
                if attempt < max_attempts:
                    time.sleep(waits[attempt - 1])
                else:
                    failures.append(f"{model} ({role}): {err_str}")
    raise RuntimeError(
        f"All {len(failures)} model(s) in cascade failed:\n"
        + "\n".join(f"  • {f}" for f in failures)
    )


def preprocess_query(
    query: str,
    api_key: Optional[str] = None,
) -> QueryPlan:
    """
    Send the user query to Gemini and parse the response into a QueryPlan.

    Parameters
    ----------
    query   : raw natural-language question from the user
    api_key : GEMINI_API_KEY or GOOGLE_API_KEY; falls back to environment variables

    Returns
    -------
    QueryPlan  (Pydantic model, all fields validated)
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before calling preprocess_query().")

    schema_json = json.dumps(QueryPlan.model_json_schema(), indent=2)

    prompt = textwrap.dedent(f"""\
        Parse this user query into a QueryPlan JSON object.

        User query: {query}

        QueryPlan JSON schema:
        {schema_json}

        Return only the JSON object — no markdown fences, no explanation.
    """)

    raw = _call_gemini(prompt, key)

    # Strip markdown fences if the model adds them anyway
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    return QueryPlan.model_validate_json(raw)


# ---------------------------------------------------------------------------
# RAG integration helpers
# ---------------------------------------------------------------------------

def build_rag_requests(plan: QueryPlan) -> List[RAGRequest]:
    """
    Derive the full list of RAGRequests from a QueryPlan.

    Scan terms from all ScanRequirements are collected into a single
    RAGRequest with field=RAGField.SCAN.  The retriever handles multi-field
    lookup (datatype + suffix + task) for these terms.

    All other terms (diagnosis, participant task, author, funding) get
    per-field RAGRequests as before.
    """
    requests: Dict[RAGField, RAGRequest] = {}

    def _merge(field: RAGField, terms: List[str]) -> None:
        if not terms:
            return
        if field not in requests:
            requests[field] = RAGRequest(field=field, terms=list(terms))
        else:
            for t in terms:
                if t not in requests[field].terms:
                    requests[field].terms.append(t)

    # Scan requirements → single SCAN request (multi-field RAG)
    all_scan_terms: List[str] = []
    for sr in plan.scan_requirements:
        all_scan_terms.extend(sr.scan_terms)
    _merge(RAGField.SCAN, all_scan_terms)

    # Participant groups
    for grp in plan.groups:
        _merge(RAGField.DIAGNOSIS, grp.diagnosis_terms)
        _merge(RAGField.TASK,      grp.task_terms)

    # Dataset-level text fields
    _merge(RAGField.AUTHOR,  plan.author_names)
    _merge(RAGField.FUNDING, plan.funding_terms)

    # Merge any extra requests the LLM explicitly set
    for req in plan.rag_requests:
        if req.field not in requests:
            requests[req.field] = req.model_copy()
        else:
            for t in req.terms:
                if t not in requests[req.field].terms:
                    requests[req.field].terms.append(t)

    return list(requests.values())


def augment_with_rag(
    plan: QueryPlan,
    rag_results: Dict[str, Dict[str, List[str]]],
) -> AugmentedQueryPlan:
    """
    Combine a QueryPlan with RAG results to produce an AugmentedQueryPlan.

    Parameters
    ----------
    plan        : output of preprocess_query()
    rag_results : {field: {term: [canonical_code, ...]}}
                  e.g. {"diagnosis": {"schizophrenia": ["schizophrenia_spectrum"]},
                         "datatype":  {"fMRI": ["func"]},
                         "suffix":    {"fMRI": ["bold"]}}

    Returns
    -------
    AugmentedQueryPlan  ready to pass to the SQL generation LLM
    """
    resolved: List[ResolvedTerms] = []
    for field_str, term_map in rag_results.items():
        try:
            field = RAGField(field_str)
        except ValueError:
            continue
        if field == RAGField.SCAN:
            continue  # SCAN is internal; resolved results use specific field names
        resolved.append(ResolvedTerms(field=field, term_to_codes=term_map))

    aug = AugmentedQueryPlan(
        plan=plan,
        resolved=resolved,
        augmented_question=_build_augmented_question(plan, resolved),
    )
    return aug


def _build_augmented_question(
    plan: QueryPlan,
    resolved: List[ResolvedTerms],
) -> str:
    """
    Inject resolved canonical codes into the natural-language summary.

    Scan-related fields (datatype / suffix / task) are grouped by the source
    user term and emitted as pre-built EXISTS subqueries so the SQL model can
    copy them verbatim.  All three fields that matched the same user term
    (e.g. "fMRI" → datatype=functional_mri AND suffix=mri_functional) are
    combined with AND inside a single EXISTS.

    Non-scan fields (diagnosis, author, funding) are listed as code names.
    """
    if not resolved:
        return plan.natural_language_summary

    _SCAN_FIELDS = {RAGField.DATATYPE, RAGField.SUFFIX, RAGField.TASK}
    _FIELD_ORDER  = (RAGField.DATATYPE, RAGField.SUFFIX, RAGField.TASK)

    # Group scan codes by source term: term → {RAGField: [codes]}
    scan_by_term: Dict[str, Dict[RAGField, List[str]]] = {}
    non_scan_lines: List[str] = []

    label_map = {
        RAGField.DIAGNOSIS: "diagnosis codes",
        RAGField.AUTHOR:    "author names (exact DB strings)",
        RAGField.FUNDING:   "funding sources (exact DB strings)",
    }

    for r in resolved:
        if r.field in _SCAN_FIELDS:
            for term, codes in r.term_to_codes.items():
                if codes:
                    scan_by_term.setdefault(term, {})[r.field] = codes
        else:
            if r.all_codes:
                label = label_map.get(r.field, r.field.value)
                non_scan_lines.append(f"  {label}: {', '.join(r.all_codes)}")

    context_lines: List[str] = []

    # Build one EXISTS hint per source term, combining all matched fields with AND
    for term, field_codes in scan_by_term.items():
        conditions: List[str] = []
        for field in _FIELD_ORDER:
            if field not in field_codes:
                continue
            codes = field_codes[field]
            col   = field.value  # "datatype", "suffix", or "task"
            if len(codes) == 1:
                conditions.append(f"o2.{col} = '{codes[0]}'")
            else:
                vals = ", ".join(f"'{c}'" for c in codes)
                conditions.append(f"o2.{col} IN ({vals})")
        if conditions:
            where_part = " AND ".join(conditions)
            exists_sql = (
                f"EXISTS (SELECT 1 FROM bids_objects o2 "
                f"WHERE o2.dataset_id = d.id AND {where_part})"
            )
            context_lines.append(f'  scan "{term}": {exists_sql}')

    context_lines.extend(non_scan_lines)

    if not context_lines:
        return plan.natural_language_summary

    header = (
        "[Resolved DB filters — copy each EXISTS subquery verbatim into the SQL WHERE clause]"
    )
    context = header + "\n" + "\n".join(context_lines)
    return f"{context}\n\n{plan.natural_language_summary}"


# ---------------------------------------------------------------------------
# Convenience: full pipeline (preprocess + RAG)
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    retriever: Any = None,
    api_key: Optional[str] = None,
) -> AugmentedQueryPlan:
    """
    End-to-end: parse query → RAG lookup → return AugmentedQueryPlan.

    `retriever` should be a MetadataRetriever (RAG/retriever.py).
    If None, the RAG step is skipped.

    Scan terms are resolved across datatype + suffix + task simultaneously
    using retrieve_scan_terms() when available, falling back to per-field
    retrieve_for_field_multi() calls otherwise.
    """
    plan = preprocess_query(query, api_key=api_key)

    if retriever is None:
        return AugmentedQueryPlan(
            plan=plan,
            resolved=[],
            augmented_question=plan.natural_language_summary,
        )

    rag_requests = build_rag_requests(plan)
    rag_results: Dict[str, Dict[str, List[str]]] = {}

    for req in rag_requests:
        if req.field == RAGField.SCAN:
            # Multi-field lookup: datatype + suffix + task in one pass
            if hasattr(retriever, "retrieve_scan_terms"):
                scan_resolved = retriever.retrieve_scan_terms(req.terms)
            else:
                # Fallback for retrievers that don't yet have retrieve_scan_terms
                scan_resolved: Dict[str, Dict[str, List[str]]] = {}
                for field in ("datatype", "suffix", "task"):
                    if hasattr(retriever, "retrieve_for_field_multi"):
                        tm = retriever.retrieve_for_field_multi(field, req.terms)
                    else:
                        cat_key = field + "s"
                        tm = {}
                        for term in req.terms:
                            hints = retriever.retrieve(f"{field}: {term}")
                            codes = hints.get(cat_key, [])[:10]
                            if codes:
                                tm[term] = codes
                    if tm:
                        scan_resolved[field] = tm
            # Spread per-field results into rag_results
            for field, term_map in scan_resolved.items():
                if term_map:
                    rag_results.setdefault(field, {}).update(term_map)

        else:
            # Single-field lookup
            if hasattr(retriever, "retrieve_for_field_multi"):
                term_map = retriever.retrieve_for_field_multi(
                    req.field.value, req.terms
                )
            else:
                cat_key = req.field.value + "s"
                term_map: Dict[str, List[str]] = {}
                for term in req.terms:
                    hints = retriever.retrieve(f"{req.field.value}: {term}")
                    codes = hints.get(cat_key, [])[:10]
                    if codes:
                        term_map[term] = codes
            if term_map:
                rag_results[req.field.value] = term_map

    return augment_with_rag(plan, rag_results)


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_queries = [
        "Find datasets with at least 15 male ambidextrous participants diagnosed with schizophrenia",
        "Which datasets include fMRI data with an n-back working memory task?",
        "Show me datasets authored by Karl Friston",
        "Find datasets with more female than male participants",
        "Which datasets lack any EEG data but have at least 50 subjects?",
        "List datasets where the same subject has both a T1w scan and did a resting state fMRI",
        "Find datasets where RepetitionTime is greater than 2.0 seconds",
        "Show the top 10 datasets by number of subjects",
        "Datasets funded by the NIH with a CC0 license",
    ]

    query = sys.argv[1] if len(sys.argv) > 1 else test_queries[0]
    print(f"Query: {query}\n")

    result = preprocess_query(query)
    print(result.model_dump_json(indent=2))
