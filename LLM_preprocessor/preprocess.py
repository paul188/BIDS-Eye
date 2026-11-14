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
  SQL generation LLM (Gemini, in backend/services/text_to_sql.py)

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
import logging
import os
import sys
import textwrap
from enum import Enum
from pathlib import Path
from typing import Any, Collection, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field

# Make RAG/ importable from this module regardless of working directory.
_RAG_DIR = str(Path(__file__).parent.parent / "RAG")
if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from join_registry import (
    build_join_context_block as _build_join_context_block,
    check_contradictions as _check_contradictions,
    field_to_db_col as _field_to_db_col,
)
from text_terms import (
    MODE_LOOSE as _MODE_LOOSE,
    MODE_NEAR as _MODE_NEAR,
    MODE_SHORT as _MODE_SHORT,
    diagnosis_alternatives as _diagnosis_alternatives,
    format_text_match as _format_text_match,
    format_task_text_match as _format_task_text_match,
    split_words as _split_words,
)
from institution_index import text_alternatives as _institution_text_alternatives
import senses as _senses

log = logging.getLogger(__name__)

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
    INSTITUTION = "institution"
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


class Topic(BaseModel):
    """
    What the datasets should be about, with the words that fix its meaning.

    "social conflict" is not "conflict": the qualifier "social" selects one
    meaning of the head word "conflict". A topic keeps the two together, so
    the search never looks for the head word alone, and lists the other
    meanings so that datasets about them move down.
    """
    phrase: str = Field(description="The subject in standard words, abbreviations and clipped "
                                    "words written out: 'social conflict', 'emotion'")
    typed: str = Field(default="", description="The words as the user typed them: 'social conflicts', 'emo'")
    head: str = Field(description="The noun the qualifiers modify: 'conflict'")
    qualifiers: List[str] = Field(default_factory=list,
                                  description="Words that select the meaning of the head: ['social']")
    gloss: str = Field(default="", description="One line saying which meaning is meant")
    alternatives: List[str] = Field(
        default_factory=list,
        description="Up to 8 phrases a dataset description would use for exactly this meaning: "
                    "synonyms, narrower terms, named paradigms. Never the head word alone.")
    not_meant: List[str] = Field(
        default_factory=list,
        description="Other meanings of the head word that the qualifiers rule out, as words a "
                    "dataset description would use for them. Empty when the head has one meaning.")


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

    # ── Subject matter ────────────────────────────────────────────────────────
    topics: List[Topic] = Field(
        default_factory=list,
        description="Subjects whose meaning depends on a qualifier ('social conflict') or that "
                    "the user abbreviated ('emo'). See the Topics section of the instructions.",
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
        description="Keyword for full-text search over the dataset's name, description, "
                    "study fields and README"
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
    institution_terms: List[str] = Field(
        default_factory=list,
        description="Institutions, universities, hospitals, labs or imaging centres where the data "
                    "was collected, as written by the user, one per entry, e.g. ['Stanford'], "
                    "['MIT', 'Max Planck Institute']. Keep abbreviations as typed."
    )
    excluded_institution_terms: List[str] = Field(
        default_factory=list,
        description="Institutions the data must NOT come from ('not collected at Stanford' → "
                    "['Stanford']). Never also list them in institution_terms."
    )

    # ── Sidecar / JSONB metadata filters ─────────────────────────────────────
    metadata_filters: List[MetadataFilter] = Field(
        default_factory=list,
        description="Filters on bids_objects.other_entities JSONB sidecar fields"
    )

    # ── Dataset size ──────────────────────────────────────────────────────────
    max_size_gb: Optional[float] = Field(
        default=None,
        description="Upper bound on the total dataset size, in gigabytes "
                    "('no larger than 5 GB' → 5, 'under 500 MB' → 0.5, 'below 2 TB' → 2048)"
    )
    min_size_gb: Optional[float] = Field(
        default=None,
        description="Lower bound on the total dataset size, in gigabytes"
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


class ScoredFilter(BaseModel):
    """A resolved canonical code with its RAG match score (0.0–1.0).

    Used downstream to rank result datasets by how well they match the query:
    text_to_sql injects these as a `matched_codes` CTE so each dataset gets a
    relevance_score. Only fields that map to a DB column (diagnosis, task,
    datatype, suffix) are included — see _build_scored_filters().
    """
    field: str
    code: str
    score: float


class ResolvedTopic(BaseModel):
    """A Topic after the sense inventory, the vocabulary and the catalogue checked it."""
    topic: Topic
    sense: Optional[str] = None             # sense picked in RAG/senses.yaml
    codes: List[str] = Field(default_factory=list)          # task codes
    scores: Dict[str, float] = Field(default_factory=dict)  # code → match score
    alternatives: List[str] = Field(default_factory=list)   # searched, best first
    dropped: Dict[str, str] = Field(default_factory=dict)   # alternative → why not searched
    not_meant: List[str] = Field(default_factory=list)
    condition: Optional[str] = None         # the filter line handed to the SQL model


class AugmentedQueryPlan(BaseModel):
    """
    QueryPlan + resolved canonical codes from the RAG layer.
    This is what the SQL generation LLM receives.
    """
    plan: QueryPlan
    resolved: List[ResolvedTerms] = Field(default_factory=list)
    scored_filters: List[ScoredFilter] = Field(default_factory=list)
    topics: List[ResolvedTopic] = Field(default_factory=list)
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

# The example concepts in the "Broad task/paradigm concepts" paragraph are on
# purpose not questions of the recall benchmark (benchmarks/recall/queries.yaml).
# A prompt that names the benchmark's own questions makes its score say nothing
# about other questions. Keep it that way when adding examples.
_SYSTEM_PROMPT = textwrap.dedent("""\
    You are the preprocessing stage of a BIDS neuroimaging dataset search engine.
    Your job is to parse a natural-language user query into a strict JSON structure
    (a QueryPlan) that downstream components will use to build a SQL query.

    The database contains neuroimaging datasets with:
      - bids_datasets: name, authors, license, doi, funding, description_text,
                       README text
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

    Broad task/paradigm concepts count as scan-related, even when the user says
    "assessments" or "studies" rather than "task": reading, arithmetic, spatial
    navigation, mental rotation, time perception, empathy, meditation. Keep the
    distinctive concept in scan_terms (e.g. "assessments of reading" →
    ["reading"]). Do not put the only requested task/topic in
    description_search; task codes and a text search for the same phrase are
    combined downstream.

    Examples:
      "fMRI datasets"                → scan_terms: ["fMRI"]
      "bold T1w scans"               → scan_terms: ["bold", "T1w"]
      "functional MRI with n-back"   → scan_terms: ["functional MRI", "n-back"]
      "resting state fMRI"           → scan_terms: ["resting state", "fMRI"]
      "EEG and eye-tracking"         → scan_terms: ["EEG", "eye-tracking"]
      "has both anat and func data"  → two ScanRequirements:
                                         [{scan_terms: ["anat"]}, {scan_terms: ["func"]}]

    ── Compound scan requirements (conjunction splitting) ──────────────────────
    When the query uses conjunctions like "and", "combined with", "together with",
    "as well as", or phrasing like "combine X and Y tasks" to join two DISTINCT
    task or modality types, split them into SEPARATE ScanRequirements — one per
    concept. Do NOT bundle different task types into a single scan_terms list.
    A modality shared by both (e.g. "fMRI") should appear in each requirement.

    Examples:
      "combine motor and cognitive tasks"
          → [{scan_terms: ["motor"]}, {scan_terms: ["cognitive"]}]
      "fMRI with both resting state and n-back"
          → [{scan_terms: ["fMRI", "resting state"]}, {scan_terms: ["fMRI", "n-back"]}]
      "EEG and fMRI" (two modalities, multimodal intent)
          → [{scan_terms: ["EEG"]}, {scan_terms: ["fMRI"]}]

    Contrast — do NOT split when terms describe the same scan:
      "resting state fMRI"           → single ScanRequirement: ["resting state", "fMRI"]
      "bold T1w"                     → single ScanRequirement: ["bold", "T1w"]

    ── Qualitative / adjectival modifiers → description_search ─────────────────
    Some words describe a *property* of tasks or data rather than naming a
    specific task type or modality. These have no DB code and must NOT go into
    scan_terms — the RAG layer cannot resolve them and they will block results.
    Instead, place the single most distinctive keyword in description_search so
    the pipeline runs a full-text search over the dataset's name, description
    and README.

    Route a word/phrase to description_search when it is:
      • A difficulty or complexity adjective: "complex", "simple", "basic",
        "advanced", "demanding", "high-load", "challenging", "easy"
      • A paradigm-style label not tied to a specific task code: "longitudinal",
        "clinical", "ecological", "naturalistic" (when not used as a task name)
      • A population qualifier: "pediatric", "geriatric", "bilingual", "elderly"
      • Any other descriptor that tells you *how* or *what kind* rather than
        *which specific task or modality* is present

    Leave description_search null when every content word maps to a known field
    (scan modality, diagnosis, author, etc.).

    Examples:
      "complex motor and cognitive tasks"
          → scan_requirements: [{scan_terms:["motor"]},{scan_terms:["cognitive"]}]
            description_search: "complex"
      "demanding attention tasks"
          → scan_requirements: [{scan_terms:["attention"]}]
            description_search: "demanding"
      "pediatric resting state fMRI"
          → scan_requirements: [{scan_terms:["resting state","fMRI"]}]
            description_search: "pediatric"
      "fMRI datasets with n-back"
          → scan_requirements: [{scan_terms:["fMRI","n-back"]}]
            description_search: null   ← no qualitative modifier

    ── Topics: a word whose meaning a qualifier fixes ──────────────────────────
    Many words mean different things in different studies. When the user
    qualifies such a word so that only one meaning fits, put the subject in
    `topics` and NOT in scan_terms or description_search. The qualifier stays
    bound to its word: it is never searched apart from it, and the word is
    never searched without it. Also use a topic when the user abbreviates or
    clips the subject's word; write it out in `phrase`, keep the typed form
    in `typed`.

    A plain task, paradigm or modality without such a qualifier ("n-back",
    "resting state", "EEG") stays in scan_terms. A property word ("complex",
    "naturalistic", "pediatric") stays in description_search. A diagnosis
    stays in diagnosis_terms.

    Fill every topic:
      phrase        the subject in standard words, abbreviations written out
      typed         the words as typed
      head          the noun the qualifiers modify
      qualifiers    the words that select the meaning
      gloss         one line naming the meaning
      alternatives  up to 8 phrases a dataset description would use for
                    exactly this meaning: synonyms, narrower terms, named
                    paradigms or tests. Never the head word alone, never a
                    phrase that also fits another meaning.
      not_meant     the other meanings of the head word in neuroscience that
                    the qualifier rules out, as the words a description would
                    use for them (technical terms, task names); [] if none

    Examples:
      "prism adaptation experiments"
          → topics: [{phrase: "prism adaptation", typed: "prism adaptation",
                      head: "adaptation", qualifiers: ["prism"],
                      gloss: "adapting reaching or pointing to a visual shift
                              induced by prism goggles",
                      alternatives: ["prism adaptation", "prismatic adaptation",
                                     "prism goggles", "visuomotor adaptation to prisms"],
                      not_meant: ["repetition suppression", "fMRI adaptation",
                                  "contrast adaptation", "dark adaptation"]}]
            scan_requirements: []
      "circ rhythm fMRI"
          → topics: [{phrase: "circadian rhythm", typed: "circ rhythm",
                      head: "rhythm", qualifiers: ["circadian"],
                      gloss: "the roughly 24-hour cycle of sleep, wakefulness and physiology",
                      alternatives: ["circadian", "diurnal variation", "time of day",
                                     "sleep-wake cycle", "melatonin"],
                      not_meant: ["theta rhythm", "alpha rhythm", "musical rhythm",
                                  "cardiac rhythm"]}]
            scan_requirements: [{scan_terms: ["fMRI"]}]

    ── Participant terms ───────────────────────────────────────────────────────
    Put diagnosis / condition strings in ParticipantGroup.diagnosis_terms.
    These are resolved by the RAG layer (e.g. "depression" → canonical code).
    IMPORTANT: Do NOT put generic demographic words such as "patients",
    "participants", "subjects", "people", "individuals", "volunteers",
    "adults", "children", "healthy" into diagnosis_terms — these are not
    diagnoses.  Only extract actual medical or psychiatric condition labels.

    ── Dataset-level terms ─────────────────────────────────────────────────────
    Put author names in author_names; funding agencies in funding_terms.
    The RAG resolves these to exact DB strings.
    Put the place where data was acquired (university, hospital, institute,
    lab, imaging centre) in institution_terms, e.g. "scanned at Stanford",
    "data from MIT", "collected at the Max Planck Institute". A funder is not
    an institution: "funded by the NIH" → funding_terms, "acquired at the NIH"
    → institution_terms. Do not also put the institution in description_search.
    "not from Stanford", "excluding MIT data" → excluded_institution_terms.
    Leave both lists empty when an institution name is part of something else:
      a scale, atlas or template:  "Stanford Sleepiness Scale", "Harvard-Oxford
          atlas", "MNI space", "Montreal Cognitive Assessment" → description_search
      a study or dataset name:     "Human Connectome Project", "Cam-CAN",
          "UK Biobank", "ABIDE" → name_search or description_search
      a person or their lab:       "the Poldrack lab" → author_names ["Poldrack"]
      a country, region or city:   "datasets from Germany" → description_search
      a generic word:              "multi-site", "hospital data", "clinical sites"
          → no institution (multi-site is a property, not a place)

    ── Counts and n_results ────────────────────────────────────────────────────
    Do NOT set any n_results, expected_*_codes, or n_*_codes fields.
    The RAG layer decides how many codes to return based on similarity threshold.

    ── rag_requests ────────────────────────────────────────────────────────────
    Leave rag_requests as an empty list [].  It is auto-derived from the plan.

    ── Scanner hardware metadata ────────────────────────────────────────────────
    Queries about MRI scanner hardware go into metadata_filters (MetadataFilter),
    NOT into scan_terms.  Use these exact BIDS sidecar key names:
      Manufacturer           → string, use operator "=" (the SQL layer uses ILIKE)
                               Values: 'Siemens', 'Philips', 'GE', 'Bruker', 'Canon'
      ManufacturerModelName  → string, use operator "="
                               Values: 'Prisma', 'Prisma_fit', 'Skyra', 'Trio',
                                       'Achieva', 'Ingenia', 'MAGNETOM Vida', etc.
      MagneticFieldStrength  → numeric Tesla, use >, <, >=, <=, = operators
                               Common values: 1.5, 3.0, 7.0
      SoftwareVersions       → string, use operator "=" or "exists"

    Examples:
      "3 Tesla datasets"         → metadata_filters: [{"json_field": "MagneticFieldStrength", "operator": "=", "value": 3.0}]
      "above 3T"                 → metadata_filters: [{"json_field": "MagneticFieldStrength", "operator": ">", "value": 3.0}]
      "Siemens scanners"         → metadata_filters: [{"json_field": "Manufacturer", "operator": "=", "value": "Siemens"}]
      "Prisma scanner"           → metadata_filters: [{"json_field": "ManufacturerModelName", "operator": "=", "value": "Prisma"}]
      "has software version info"→ metadata_filters: [{"json_field": "SoftwareVersions", "operator": "exists"}]

    ── Dataset size ────────────────────────────────────────────────────────────
    A limit on how big the dataset is on disk goes into max_size_gb / min_size_gb,
    always converted to gigabytes (1 GB = 1024 MB, 1 TB = 1024 GB). This is about
    download size, NOT the number of subjects.

    Examples:
      "fMRI datasets no larger than 5 GB"   → max_size_gb: 5
      "small datasets under 500 MB"         → max_size_gb: 0.5
      "datasets of at least 1 TB"           → min_size_gb: 1024
      "datasets with at least 50 subjects"  → no size field (that is min_subjects)

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
      description_search   — keyword search on dataset name, description or README
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


def _call_gemini(prompt: str, api_key: str) -> str:
    """Run the intent-extraction prompt through the shared LLM cascade.

    Cascade + retries + cross-provider fallback live in llm_client.llm_generate;
    raises llm_client.LLMAllFailedError if every tier fails.
    """
    from llm_client import llm_generate
    return llm_generate(prompt, system=_SYSTEM_PROMPT, temperature=0.1, api_key=api_key)


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

    # Participant groups — strip generic demographic words before RAG resolution
    # so that "patients" / "participants" never resolves to unknown_diagnosis
    _GENERIC_WORDS = {
        "patients", "patient", "participants", "participant",
        "subjects", "subject", "people", "person",
        "individuals", "individual", "volunteers", "volunteer",
        "adults", "adult", "children", "child",
        "healthy", "normal",
    }
    for grp in plan.groups:
        real_diagnoses = [t for t in grp.diagnosis_terms
                          if t.lower().strip() not in _GENERIC_WORDS]
        _merge(RAGField.DIAGNOSIS, real_diagnoses)
        _merge(RAGField.TASK,      grp.task_terms)

    # Dataset-level text fields
    _merge(RAGField.AUTHOR,  plan.author_names)
    _merge(RAGField.FUNDING, plan.funding_terms)
    _merge(RAGField.INSTITUTION, plan.institution_terms + plan.excluded_institution_terms)

    # Merge any extra requests the LLM explicitly set
    for req in plan.rag_requests:
        if req.field not in requests:
            requests[req.field] = req.model_copy()
        else:
            for t in req.terms:
                if t not in requests[req.field].terms:
                    requests[req.field].terms.append(t)

    return list(requests.values())


# Fields whose codes can be scored per-dataset (they map to a queryable DB
# column via field_to_db_col). author/funding live on bids_datasets as arrays
# (exists_alias 'none') and are excluded from relevance scoring.
_SCORABLE_FIELDS = {"diagnosis", "task", "datatype", "suffix"}


def _build_scored_filters(
    rag_results: Dict[str, Dict[str, List[str]]],
    scores: Dict[str, Dict[str, float]],
) -> List[ScoredFilter]:
    """Flatten RAG results + scores into per-(field, code) ScoredFilters.

    Codes are taken from rag_results (already DB-pruned, so DB-miss codes are
    excluded automatically). Only fields that map to a DB column are kept.
    Returns [] when no scores were captured (e.g. an older retriever without the
    *_scored methods) so downstream ranking degrades gracefully.
    """
    if not scores:
        return []
    out: List[ScoredFilter] = []
    seen: set[tuple[str, str]] = set()
    for field, term_map in rag_results.items():
        if field not in _SCORABLE_FIELDS or _field_to_db_col(field) is None:
            continue
        field_scores = scores.get(field, {})
        for codes in term_map.values():
            for code in codes:
                if (field, code) in seen:
                    continue
                seen.add((field, code))
                out.append(
                    ScoredFilter(field=field, code=code,
                                 score=field_scores.get(code, 1.0))
                )
    return out


# ---------------------------------------------------------------------------
# Topics: a qualified subject, searched as one unit
# ---------------------------------------------------------------------------

# Alternatives taken from the language model, and searched in total.
_TOPIC_MAX_LLM_ALTERNATIVES = 8
_TOPIC_MAX_ALTERNATIVES = 16
_TOPIC_MAX_CODES = 20


def _topic_max_matches() -> int:
    """An alternative from the language model that more datasets match is too broad.

    90 is about 5% of the 1,721 datasets of 2026-11. Alternatives from
    RAG/senses.yaml are curated and exempt.
    """
    try:
        return int(os.getenv("TOPIC_MAX_ALTERNATIVE_MATCHES", "90"))
    except ValueError:
        return 90


def _stem(part: str) -> str:
    if len(part) > 4 and part.endswith("ies"):
        return part[:-3] + "y"
    if len(part) > 3 and part.endswith("s") and not part.endswith("ss"):
        return part[:-1]
    return part


def _word_set(text: str) -> frozenset:
    """Content words of *text*, plural folded: "social conflicts" → {social, conflict}."""
    return frozenset(_stem(part) for word in _split_words(text or "") for part in word)


def _count_topic_matches(alternatives: List[str]) -> Optional[List[int]]:
    """Datasets matching each alternative, from the backend's full-text search; None if unknown."""
    try:
        from services import text_search  # backend/services; absent outside the API
    except Exception:
        return None
    try:
        return text_search.count_matches(alternatives, _MODE_NEAR)
    except Exception as exc:  # a check must never break a search
        log.warning("Topic alternative check failed (%s) — searching them unchecked", exc)
        return None


def _is_ambiguity_topic(topic: Topic) -> bool:
    """A topic the search treats as one: its head has senses, or the user abbreviated it.

    On the 167 development questions of 2026-11-14 the language model also
    made topics of plain subjects ("sign language processing", "gambling
    choices") and listed unwanted meanings for nearly all of them; searching
    those as topics lost recall (47% → 38% on the 24 such questions). The
    gains on the ambiguity questions came from heads in RAG/senses.yaml and
    from abbreviations ("emo").
    """
    phrase = topic.phrase or topic.typed
    if _senses.senses_of(topic.head or phrase):
        return True
    return bool(topic.typed) and bool(_word_set(topic.typed)) \
        and _word_set(topic.typed) != _word_set(phrase)


def _keep_ambiguity_topics(plan: QueryPlan) -> None:
    """Hand every other topic back to the scan-term path it had before topics existed."""
    kept: List[Topic] = []
    for topic in plan.topics:
        if _is_ambiguity_topic(topic):
            kept.append(topic)
            continue
        term = topic.phrase or topic.typed
        if term and not any(term in r.scan_terms for r in plan.scan_requirements):
            plan.scan_requirements.append(ScanRequirement(scan_terms=[term]))
    plan.topics = kept


def _detach_topic_terms(plan: QueryPlan) -> None:
    """Remove scan terms and description words that only repeat a topic.

    The topic searches them as one unit; a scan term "conflict" next to the
    topic "social conflict" would bring the head word back on its own.
    """
    names = [_word_set(name) for t in plan.topics for name in (t.phrase, t.typed) if name]
    names = [n for n in names if n]
    if not names:
        return

    def covered(term: str) -> bool:
        words = _word_set(term)
        return bool(words) and any(words <= name for name in names)

    for requirement in plan.scan_requirements:
        requirement.scan_terms = [t for t in requirement.scan_terms if not covered(t)]
    plan.scan_requirements = [r for r in plan.scan_requirements if r.scan_terms]
    if plan.description_search and covered(plan.description_search):
        plan.description_search = None


def _topic_codes(
    topic: Topic, pick: Optional["_senses.SensePick"], not_meant: List[str], retriever: Any,
) -> Dict[str, float]:
    """Task codes for the whole phrase, never for the head word alone.

    The vocabulary also matches a phrase through one of its words: "cortical
    inhibition" reached the construct inhibition and with it stop-signal and
    go/no-go tasks. With qualifiers: when the phrase gives everything the
    head word alone gives, those codes came through the head and are
    dropped; what the qualifier added stays. A phrase that gives fewer or
    other codes than the head ("spatial memory" vs "memory") was narrowed by
    its qualifier and keeps them. The picked sense's own codes come after.
    """
    codes: Dict[str, float] = {}
    phrase = topic.phrase or topic.typed

    def lookup(term: str) -> Dict[str, float]:
        if retriever is None or not hasattr(retriever, "retrieve_for_field_multi_scored"):
            return {}
        try:
            found = retriever.retrieve_for_field_multi_scored("task", [term])
        except Exception as exc:
            log.warning("Topic vocabulary lookup failed for %r (%s)", term, exc)
            return {}
        return {code: float(score) for pairs in found.values() for code, score in pairs}

    codes = lookup(phrase)
    head = (topic.head or "").strip()
    if codes and topic.qualifiers and head and _word_set(head) != _word_set(phrase):
        head_codes = lookup(head)
        if head_codes and set(head_codes) <= set(codes):
            codes = {code: score for code, score in codes.items() if code not in head_codes}
    if pick is not None:
        for code in pick.tasks:
            codes.setdefault(code, 1.0)
    excluded = set(pick.excluded_tasks) if pick is not None else set()
    # "simon" among the unwanted meanings removes simon_task.
    unwanted_words = {next(iter(w)) for w in map(_word_set, not_meant) if len(w) == 1}
    codes = {
        code: score for code, score in codes.items()
        if code not in excluded and not (set(code.lower().split("_")) & unwanted_words)
    }
    codes = dict(list(codes.items())[:_TOPIC_MAX_CODES])
    if codes and retriever is not None and hasattr(retriever, "db_verify_codes"):
        live, _miss = retriever.db_verify_codes("task", list(codes))
        codes = {code: codes[code] for code in live}
    return codes


def _topic_phrase_terms(topic: Topic) -> List[str]:
    """The user's own subject: the phrase, and the typed form when it differs ("emo")."""
    terms = [topic.phrase or topic.typed]
    if topic.typed and _word_set(topic.typed) and _word_set(topic.typed) != _word_set(terms[0]):
        terms.append(topic.typed)
    return [t for t in terms if t]


def _topic_condition(resolved: ResolvedTopic) -> Optional[str]:
    """One filter for a topic:

        EXISTS (task codes)
        OR text_match('<phrase>', 'near')
        OR ((text_match('<alternatives of several words>', 'near')
             OR text_match('<one-word alternatives>', 'short'))
            AND NOT text_match('<not meant>', 'near'))

    The phrase comes first: the uncertain tier relaxes only the first text
    search of a topic (services/uncertain_tier.py, relax_text_condition).

    The phrase is the user's own words and needs no guard. An alternative can
    also occur in a text about another meaning ("conflict resolution task" in
    a Stroop study); such a dataset fails this condition and the uncertain
    tier keeps it as probable (services/uncertain_tier.py).
    """
    parts: List[str] = []
    if resolved.codes:
        values = ", ".join(f"'{c}'" for c in resolved.codes)
        task = f"o2.task = {values}" if len(resolved.codes) == 1 else f"o2.task IN ({values})"
        parts.append(f"EXISTS (SELECT 1 FROM bids_objects o2 WHERE o2.dataset_id = d.id AND {task})")
    phrase_match = _format_text_match(_topic_phrase_terms(resolved.topic), _MODE_NEAR)
    if phrase_match:
        parts.append(phrase_match)
    # A one-word alternative is searched in the short fields only: anywhere in
    # a README, "aggression" also matched "the defacing was quite aggressive"
    # and an aggression questionnaire. Several words must stand together.
    phrases = [a for a in resolved.alternatives if len(_split_words(a)) > 1]
    words = [a for a in resolved.alternatives if len(_split_words(a)) == 1]
    alternative_parts = [m for m in (_format_text_match(phrases, _MODE_NEAR),
                                     _format_text_match(words, _MODE_SHORT)) if m]
    alternative_match = None
    if len(alternative_parts) == 1:
        alternative_match = alternative_parts[0]
    elif alternative_parts:
        alternative_match = "(" + " OR ".join(alternative_parts) + ")"
    if alternative_match:
        guard = _format_text_match(resolved.not_meant, _MODE_NEAR)
        parts.append(f"({alternative_match} AND NOT {guard})" if guard else alternative_match)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"


def resolve_topic(
    topic: Topic,
    retriever: Any = None,
    count_matches: Any = _count_topic_matches,
) -> ResolvedTopic:
    """Check a topic against the sense inventory, the vocabulary and the catalogue.

    - RAG/senses.yaml: a qualifier that picks a sense of the head adds that
      sense's phrases and task codes and excludes the other senses.
    - Alternatives are dropped when they are only the head word or a
      qualifier, name an excluded meaning, match no dataset, or (when the
      language model proposed them) match too many to say anything.
    - Task codes come from the whole phrase and the picked sense.
    """
    phrase = (topic.phrase or topic.typed).strip()
    qualifiers = list(topic.qualifiers)
    pick = _senses.pick_sense(topic.head or phrase, qualifiers, phrase=phrase)
    not_meant = list(dict.fromkeys(
        [n for n in topic.not_meant if n.strip()] + (pick.not_meant if pick else [])))
    not_meant_sets = [w for w in map(_word_set, not_meant) if w]

    base = _word_set(topic.head)
    for qualifier in qualifiers:
        base |= _word_set(qualifier)
    own = {_word_set(t) for t in _topic_phrase_terms(topic)}

    dropped: Dict[str, str] = {}
    kept: List[tuple] = []          # (alternative, curated)
    seen: set = set()
    candidates = [(a, True) for a in (pick.alternatives if pick else [])]
    candidates += [(a, False) for a in topic.alternatives[:_TOPIC_MAX_LLM_ALTERNATIVES]]
    for alternative, curated in candidates:
        words = _word_set(alternative)
        if not words or words in seen or words in own:
            continue
        seen.add(words)
        # Never the head word alone; a qualifier alone ("social") only when
        # senses.yaml lists it as a phrase of the sense ("GABA").
        if words <= base and (not curated or words <= _word_set(topic.head)):
            dropped[alternative] = "only the head word or a qualifier"
        elif any(n <= words for n in not_meant_sets):
            dropped[alternative] = "names a meaning the search excludes"
        else:
            kept.append((alternative, curated))

    counts = count_matches([a for a, _ in kept]) if (kept and count_matches) else None
    if counts is not None and len(counts) == len(kept):
        limit = _topic_max_matches()
        checked = []
        for (alternative, curated), count in zip(kept, counts):
            if count == 0:
                dropped[alternative] = "no dataset uses it"
            elif count > limit and not curated:
                dropped[alternative] = f"too common ({count} datasets)"
            else:
                checked.append((alternative, curated))
        kept = checked

    codes = _topic_codes(topic, pick, not_meant, retriever)
    resolved = ResolvedTopic(
        topic=topic,
        sense=pick.sense.name if (pick and pick.sense) else None,
        codes=list(codes),
        scores=codes,
        alternatives=[a for a, _ in kept][:_TOPIC_MAX_ALTERNATIVES],
        dropped=dropped,
        not_meant=not_meant,
    )
    resolved.condition = _topic_condition(resolved)
    log.info(
        "Topic %r: sense=%s codes=%s alternatives=%s dropped=%s not_meant=%s",
        phrase, resolved.sense, resolved.codes, resolved.alternatives, dropped, not_meant,
    )
    return resolved


def resolve_topics(plan: QueryPlan, retriever: Any = None,
                   count_matches: Any = _count_topic_matches) -> List[ResolvedTopic]:
    return [resolve_topic(t, retriever, count_matches)
            for t in plan.topics if (t.phrase or t.typed).strip()]


def augment_with_rag(
    plan: QueryPlan,
    rag_results: Dict[str, Dict[str, List[str]]],
    unresolved_terms: Optional[Dict[str, List[str]]] = None,
    db_miss_terms: Optional[Dict[str, List[str]]] = None,
    scores: Optional[Dict[str, Dict[str, float]]] = None,
    available_columns: Collection[str] = (),
    topics: Optional[List[ResolvedTopic]] = None,
) -> AugmentedQueryPlan:
    """
    Combine a QueryPlan with RAG results to produce an AugmentedQueryPlan.

    Parameters
    ----------
    plan             : output of preprocess_query()
    rag_results      : {field: {term: [canonical_code, ...]}}
                       e.g. {"diagnosis": {"schizophrenia": ["schizophrenia_spectrum"]},
                              "datatype":  {"fMRI": ["func"]},
                              "suffix":    {"fMRI": ["bold"]}}
    unresolved_terms : {field: [term, ...]} for terms that RAG could not map.
                       These are surfaced as an anti-ILIKE warning in the augmented
                       question so the SQL model does not fall back to broken text
                       searches on canonical-code columns.
    db_miss_terms    : {field: [code, ...]} for codes that resolved in the YAML
                       vocabulary but have zero rows in the live database.  These
                       are emitted as a DB_MISS warning so the SQL model omits
                       those filters rather than generating always-false EXISTS.

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

    topics = list(topics or [])
    scored_filters = _build_scored_filters(rag_results, scores or {})
    seen = {(f.field, f.code) for f in scored_filters}
    for topic in topics:
        for code in topic.codes:
            if ("task", code) not in seen:
                seen.add(("task", code))
                scored_filters.append(
                    ScoredFilter(field="task", code=code, score=topic.scores.get(code, 1.0)))

    aug = AugmentedQueryPlan(
        plan=plan,
        resolved=resolved,
        scored_filters=scored_filters,
        topics=topics,
        augmented_question=_build_augmented_question(
            plan, resolved, unresolved_terms or {}, db_miss_terms or {},
            available_columns, topics,
        ),
    )
    return aug


def _institution_lines(
    plan: QueryPlan,
    resolved: List[ResolvedTerms],
    unresolved: List[str],
    available_columns: Collection[str],
) -> List[str]:
    """One filter line per institution the user named.

    Resolved names (exact bids_datasets.institutions strings from
    RAG/institution_index.py) become an array overlap, which the GIN index on
    the column serves:
        (d.institutions && ARRAY['Stanford University', ...]::text[]
         OR text_match('Stanford', 'short'))
    The text part catches datasets that name the site in their title,
    description or study fields but whose scans carry no InstitutionName.
    README text only ranks here: README mentions of an institution are often
    citations or tools ('Harvard-Oxford atlas', 'Stanford Sleepiness Scale').

    An institution the stored names do not know (or a database without the
    column) falls back to a full-text search that includes the README, where
    labs usually describe where they scanned.

    An excluded institution removes datasets with any of its stored names:
        NOT COALESCE(d.institutions && ARRAY[...]::text[], false)
    (COALESCE keeps datasets without any recorded institution). Without
    stored names it removes datasets that name it in their short text fields;
    a README mention alone does not exclude.
    """
    resolved_names: Dict[str, List[str]] = {}
    for r in resolved:
        if r.field == RAGField.INSTITUTION:
            for term, names in r.term_to_codes.items():
                resolved_names.setdefault(term, []).extend(names)

    excluded = list(dict.fromkeys(t for t in plan.excluded_institution_terms if t and t.strip()))
    terms = list(dict.fromkeys(
        [t for t in plan.institution_terms if t and t.strip()] + list(resolved_names) + list(unresolved)
    ))
    terms = [t for t in terms if t not in excluded]
    use_column = "institutions" in available_columns

    def _array(names: List[str]) -> str:
        return "ARRAY[" + ", ".join("'" + n.replace("'", "''") + "'" for n in names) + "]::text[]"

    lines: List[str] = []
    for term in excluded:
        names = list(dict.fromkeys(resolved_names.get(term, []))) if use_column else []
        if names:
            condition = f"NOT COALESCE(d.institutions && {_array(names)}, false)"
        else:
            text_part = _format_text_match(_institution_text_alternatives(term) or [term], _MODE_SHORT)
            if not text_part:
                continue
            condition = f"NOT {text_part}"
        label = term.replace('"', "'")
        lines.append(f'  not institution "{label}": {condition}')

    for term in terms:
        names = list(dict.fromkeys(resolved_names.get(term, []))) if use_column else []
        alternatives = _institution_text_alternatives(term)
        if names:
            array = ", ".join("'" + name.replace("'", "''") + "'" for name in names)
            condition = f"d.institutions && ARRAY[{array}]::text[]"
            text_part = _format_text_match(alternatives, _MODE_SHORT)
            if text_part:
                condition = f"({condition} OR {text_part})"
        else:
            condition = _format_text_match(alternatives or [term])
            if not condition:
                continue
        label = term.replace('"', "'")
        lines.append(f'  institution "{label}": {condition}')
    return lines


_BYTES_PER_GB = 1024 ** 3


def _size_conditions(plan: QueryPlan) -> List[str]:
    """SQL conditions on d.size_bytes for the size limits in the plan."""
    conds: List[str] = []
    if plan.max_size_gb is not None and plan.max_size_gb > 0:
        conds.append(f"d.size_bytes <= {int(plan.max_size_gb * _BYTES_PER_GB)}")
    if plan.min_size_gb is not None and plan.min_size_gb > 0:
        conds.append(f"d.size_bytes >= {int(plan.min_size_gb * _BYTES_PER_GB)}")
    return conds


def _build_augmented_question(
    plan: QueryPlan,
    resolved: List[ResolvedTerms],
    unresolved_terms: Optional[Dict[str, List[str]]] = None,
    db_miss_terms: Optional[Dict[str, List[str]]] = None,
    available_columns: Collection[str] = (),
    topics: Sequence[ResolvedTopic] = (),
) -> str:
    """
    Inject resolved canonical codes into the natural-language summary.

    *available_columns* names the optional bids_datasets columns the database
    has populated (backend/services/schema_capabilities.py). A filter on a
    column that is not listed is left out rather than emitted and failing.

    Scan-related fields (datatype / suffix / task) are grouped by the source
    user term and emitted as pre-built EXISTS subqueries so the SQL model can
    copy them verbatim.  All three fields that matched the same user term
    (e.g. "fMRI" → datatype=functional_mri AND suffix=fmri_bold) are
    combined with AND inside a single EXISTS.

    A diagnosis becomes an EXISTS on the structured rows OR a text_match()
    placeholder on the short text fields, because most datasets about a
    disease have no diagnosis rows.  Free-text terms (description_search,
    unresolved vocabulary) become text_match() placeholders as well; the
    backend expands them into full-text predicates (RAG/text_terms.py).

    Author and funding are listed as exact DB strings.
    """
    size_conds = _size_conditions(plan) if "size_bytes" in available_columns else []
    # Unresolved terms count too: their text filter is the only thing that
    # keeps a question like "eye tracking datasets" from returning everything.
    if (not resolved and not plan.description_search and not plan.name_search
            and not size_conds and not plan.institution_terms
            and not plan.excluded_institution_terms
            and not unresolved_terms and not db_miss_terms
            and not any(t.condition for t in topics)):
        return plan.natural_language_summary

    _SCAN_FIELDS = {RAGField.DATATYPE, RAGField.SUFFIX, RAGField.TASK}
    _FIELD_ORDER  = (RAGField.DATATYPE, RAGField.SUFFIX, RAGField.TASK)

    # Group scan codes by source term: term → {RAGField: [codes]}
    scan_by_term: Dict[str, Dict[RAGField, List[str]]] = {}
    non_scan_lines: List[str] = []
    context_lines: List[str] = []

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
        elif r.field == RAGField.DIAGNOSIS:
            if r.all_codes:
                vals = ", ".join(f"'{c}'" for c in r.all_codes)
                exists_sql = (
                    f"EXISTS (SELECT 1 FROM bids_participants p2 "
                    f"WHERE p2.dataset_id = d.id AND p2.diagnosis IN ({vals}))"
                )
                # Structured diagnosis rows are sparse: back them up with the
                # user's words and the vocabulary synonyms in name,
                # description and study fields.
                text_fallback = _format_text_match(
                    _diagnosis_alternatives(r.term_to_codes), _MODE_SHORT
                )
                if text_fallback:
                    exists_sql = f"({exists_sql} OR {text_fallback})"
                context_lines.append(f'  diagnosis: {exists_sql}')
        elif r.field == RAGField.INSTITUTION:
            continue  # see _institution_lines()
        else:
            if r.all_codes:
                label = label_map.get(r.field, r.field.value)
                non_scan_lines.append(f"  {label}: {', '.join(r.all_codes)}")

    # Contradiction detection: build flat {field_name: [codes]} from all resolved data.
    _flat: Dict[str, List[str]] = {}
    for _term_field_codes in scan_by_term.values():
        for _rag_field, _codes in _term_field_codes.items():
            _flat.setdefault(_rag_field.value, []).extend(_codes)
    for _r in resolved:
        if _r.field == RAGField.DIAGNOSIS and _r.all_codes:
            _flat.setdefault(_r.field.value, []).extend(_r.all_codes)

    _contradictions = _check_contradictions(_flat)
    if _contradictions:
        _hard = [w.message for w in _contradictions if w.severity == "hard"]
        _soft = [w.message for w in _contradictions if w.severity == "soft"]
        _parts: List[str] = []
        if _hard:
            _parts.append("HARD CONTRADICTION — " + " | ".join(_hard))
        if _soft:
            _parts.append("WARNING — " + " | ".join(_soft))
        context_lines.append("[Query note: " + " ".join(_parts) + "]")

    # Build EXISTS hints per source term.
    # Modality (datatype + suffix) uses OR — a file qualifies if it matches
    # the modality by EITHER its folder (datatype) OR its file suffix.
    # Task gets a separate EXISTS so compound terms like "resting state fMRI"
    # correctly emit both a modality filter and a task filter.
    for term, field_codes in scan_by_term.items():
        modality_conds: List[str] = []   # datatype + suffix codes
        task_conds: List[str] = []       # task codes

        for field in _FIELD_ORDER:
            if field not in field_codes:
                continue
            codes = field_codes[field]
            col   = field.value
            if len(codes) == 1:
                cond = f"o2.{col} = '{codes[0]}'"
            else:
                vals = ", ".join(f"'{c}'" for c in codes)
                cond = f"o2.{col} IN ({vals})"
            if field == RAGField.TASK:
                task_conds.append(cond)
            else:
                modality_conds.append(cond)

        # If a term resolves to BOTH modality codes AND task codes, combine
        # everything into one OR EXISTS — matching any of the codes increases
        # confidence that the term is present, so OR is correct within a term.
        # Example: "attention" → (suffix='att' OR task IN (attention_tasks))
        #
        # If the term resolved to ONLY modality codes (e.g. "fMRI", "EEG"),
        # emit a single modality EXISTS with OR between datatype and suffix.
        # If the term resolved to ONLY task codes (e.g. "resting state",
        # "working memory"), emit a separate task EXISTS.
        all_conds = modality_conds + task_conds
        if modality_conds and task_conds:
            # Single OR EXISTS across all fields for this term
            or_part = " OR ".join(all_conds)
            if len(all_conds) > 1:
                or_part = f"({or_part})"
            condition = f"EXISTS (SELECT 1 FROM bids_objects o2 WHERE o2.dataset_id = d.id AND {or_part})"
            text_fallback = _format_task_text_match([term])
            if text_fallback:
                condition = f"({condition} OR {text_fallback})"
            context_lines.append(
                f'  scan "{term}": {condition}'
            )
        else:
            # Modality EXISTS: OR between datatype and suffix so partially-annotated
            # files still match (e.g. file has suffix=fmri_bold but no explicit datatype).
            if modality_conds:
                or_part = " OR ".join(modality_conds)
                if len(modality_conds) > 1:
                    or_part = f"({or_part})"
                context_lines.append(
                    f'  scan "{term}": '
                    f"EXISTS (SELECT 1 FROM bids_objects o2 WHERE o2.dataset_id = d.id AND {or_part})"
                )

            # Task EXISTS: separate from modality so both constraints apply independently.
            # e.g. "resting state fMRI" → fMRI modality EXISTS AND resting_state task EXISTS
            if task_conds:
                where_part = " OR ".join(task_conds)
                if len(task_conds) > 1:
                    where_part = f"({where_part})"
                condition = f"EXISTS (SELECT 1 FROM bids_objects o2 WHERE o2.dataset_id = d.id AND {where_part})"
                text_fallback = _format_task_text_match([term])
                if text_fallback:
                    condition = f"({condition} OR {text_fallback})"
                context_lines.append(
                    f'  scan "{term}": {condition}'
                )

    # Topics: the qualified subject as one condition (see _topic_condition).
    for topic in topics:
        if topic.condition:
            label = (topic.topic.phrase or topic.topic.typed).replace('"', "").replace("'", "")
            context_lines.append(f'  topic "{label}": {topic.condition}')

    context_lines.extend(non_scan_lines)

    # Institutions: an exact filter on d.institutions, or a text search when
    # the stored names do not know the institution. Unresolved institution
    # terms are handled there, not in the VOCABULARY MISS block below.
    unresolved_terms = dict(unresolved_terms or {})
    context_lines.extend(_institution_lines(
        plan, resolved,
        unresolved_terms.pop(RAGField.INSTITUTION.value, []),
        available_columns,
    ))

    # Inject free-text hints for qualitative/adjectival terms that the
    # preprocessing LLM routed to description_search / name_search because they
    # can't be resolved to a DB code (e.g. "complex", "pediatric", "demanding").
    # description_search covers name, description, study fields and README.
    description_match = _format_text_match([plan.description_search or ""])
    if description_match:
        context_lines.append(f"  dataset text: {description_match}")
    if plan.name_search:
        context_lines.append(
            f"  dataset name: d.name ILIKE '%{plan.name_search}%'"
        )

    # Dataset size limit — a plain condition on bids_datasets, no EXISTS needed.
    if size_conds:
        context_lines.append("  dataset size: " + " AND ".join(size_conds))

    # Emit min_subjects as a HAVING hint using the total subject count.
    # "at least N participants with X" → HAVING COUNT(DISTINCT o.subject) >= N
    # (subject_count in SELECT is computed over all subjects, not just diagnosed ones —
    #  which matches what users see in the UI and expect as "dataset size").
    min_subjects_values = [
        g.min_subjects for g in plan.groups if g.min_subjects is not None
    ]
    if min_subjects_values:
        n = max(min_subjects_values)
        context_lines.append(
            f"  subject count: HAVING COUNT(DISTINCT o.subject) >= {n}"
            " [apply this HAVING clause on the outer query, not in a subquery]"
        )

    # Warn about terms that RAG could not map — prevent SQL model from generating
    # broken text filters on canonical-code columns (diagnosis, task, datatype, suffix).
    if unresolved_terms:
        missing_parts = []
        for field, terms in unresolved_terms.items():
            missing_parts.append(f"{field}: {', '.join(repr(t) for t in terms)}")
        # One 'loose' text_match() per term: the dataset must contain the
        # term's most distinctive word anywhere in its text (README included);
        # the other words only move it up in the ranking.
        text_suggestions = []
        for field, terms in unresolved_terms.items():
            for term in terms:
                suggestion = _format_text_match([term], _MODE_LOOSE)
                if suggestion and suggestion not in text_suggestions:
                    text_suggestions.append(suggestion)
        ilike_hint = ""
        if text_suggestions:
            ilike_hint = (
                " For free-text search use: "
                + " AND ".join(text_suggestions)
            )
        context_lines.append(
            "[VOCABULARY MISS — these terms are not in the known concept vocabulary: "
            + "; ".join(missing_parts)
            + ". Do NOT filter bids_participants.diagnosis or bids_objects.task/datatype/suffix "
            "for these terms — those columns store canonical codes, not free text."
            + ilike_hint + "]"
        )

    # Warn about codes that resolved in the vocabulary but have zero DB rows.
    # The SQL model must omit these from EXISTS filters (they always return empty).
    if db_miss_terms:
        miss_parts = []
        for field, codes in db_miss_terms.items():
            unique_codes = list(dict.fromkeys(codes))  # deduplicate
            miss_parts.append(
                f"{field}: {', '.join(repr(c) for c in unique_codes)}"
            )
        context_lines.append(
            "[DB MISS — resolved codes with zero rows in the database: "
            + "; ".join(miss_parts)
            + ". These terms resolve in the vocabulary but have no patient/scan data yet. "
            "Omit these codes from EXISTS filters entirely — do not include them in "
            "IN (...) lists or = comparisons. If a broader parent code is available "
            "(e.g. 'anxiety_disorder' when children like 'panic_disorder' are missing), "
            "use the parent code instead.]"
        )

    if not context_lines:
        return plan.natural_language_summary

    header = (
        "[Resolved DB filters — copy each filter expression verbatim into the SQL WHERE clause]"
    )
    join_block = _build_join_context_block()
    context = join_block + "\n\n" + header + "\n" + "\n".join(context_lines)
    return f"{context}\n\n{plan.natural_language_summary}"


# ---------------------------------------------------------------------------
# Convenience: full pipeline (preprocess + RAG)
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    retriever: Any = None,
    api_key: Optional[str] = None,
    available_columns: Collection[str] = (),
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
    _keep_ambiguity_topics(plan)
    _detach_topic_terms(plan)
    topics = resolve_topics(plan, retriever)

    if retriever is None:
        return augment_with_rag(plan, {}, available_columns=available_columns, topics=topics)

    rag_requests = build_rag_requests(plan)
    rag_results: Dict[str, Dict[str, List[str]]] = {}
    # {field: {code: best_score}} — captured when the retriever exposes the
    # *_scored methods; drives relevance ranking downstream.
    rag_scores: Dict[str, Dict[str, float]] = {}

    def _record_scores(field: str, pairs: List[tuple]) -> None:
        cur = rag_scores.setdefault(field, {})
        for code, score in pairs:
            if score > cur.get(code, 0.0):
                cur[code] = score

    for req in rag_requests:
        if req.field == RAGField.SCAN:
            # Multi-field lookup: datatype + suffix + task in one pass
            if hasattr(retriever, "retrieve_scan_terms_scored"):
                scan_scored = retriever.retrieve_scan_terms_scored(req.terms)
                for field, term_map in scan_scored.items():
                    for term, pairs in term_map.items():
                        rag_results.setdefault(field, {})[term] = [c for c, _ in pairs]
                        _record_scores(field, pairs)
                continue
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
            if hasattr(retriever, "retrieve_for_field_multi_scored"):
                scored_map = retriever.retrieve_for_field_multi_scored(
                    req.field.value, req.terms
                )
                term_map = {}
                for term, pairs in scored_map.items():
                    term_map[term] = [c for c, _ in pairs]
                    _record_scores(req.field.value, pairs)
                if term_map:
                    rag_results[req.field.value] = term_map
                continue
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

    # Optional DB back-check: remove codes that have zero rows in the live DB
    # and surface them as DB_MISS warnings so the SQL model omits those filters.
    db_miss_terms: Dict[str, List[str]] = {}
    if retriever is not None and hasattr(retriever, "db_verify_codes"):
        for field in list(rag_results.keys()):
            term_map = rag_results[field]
            new_term_map: Dict[str, List[str]] = {}
            for term, codes in term_map.items():
                live, miss = retriever.db_verify_codes(field, codes)
                if miss:
                    db_miss_terms.setdefault(field, []).extend(miss)
                if live:
                    new_term_map[term] = live
                elif not live and codes:
                    # All codes for this term are DB misses — term is unresolvable
                    db_miss_terms.setdefault(field, []).extend(codes)
            rag_results[field] = new_term_map

    # Collect terms that were requested but returned no codes — surface as warnings
    # so the SQL model does not fall back to ILIKE on canonical-code columns.
    unresolved: Dict[str, List[str]] = {}
    for req in rag_requests:
        if req.field == RAGField.SCAN:
            resolved_scan = set()
            for sf in ("datatype", "suffix", "task"):
                resolved_scan.update(rag_results.get(sf, {}).keys())
            missed = [t for t in req.terms if t not in resolved_scan]
            if missed:
                unresolved["scan"] = missed
        else:
            resolved_terms = set(rag_results.get(req.field.value, {}).keys())
            missed = [t for t in req.terms if t not in resolved_terms]
            if missed:
                unresolved[req.field.value] = missed

    return augment_with_rag(
        plan, rag_results, unresolved_terms=unresolved,
        db_miss_terms=db_miss_terms, scores=rag_scores,
        available_columns=available_columns, topics=topics,
    )


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
