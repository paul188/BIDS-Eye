"""The plan the model returns (QueryPlan) and the plan after RAG (AugmentedQueryPlan)."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
from criteria import SPECIES_CODES as _SPECIES_CODES, normalise_species as _normalise_species
from criteria import AGE_GROUP_CODES as _AGE_GROUP_CODES, \
    normalise_age_groups as _normalise_age_groups


log = logging.getLogger("preprocess")


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
    any_of: bool = Field(
        default=False,
        description="True = the user offers these terms as alternatives ('X or Y', "
                    "'X, Y or Z'): a dataset with any ONE of them matches. "
                    "False = the dataset must match every term."
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
                    "study fields and README. Alternatives and exclusions as a formula: "
                    "'drawings OR art', 'memory AND NOT sleep'"
    )
    description_alternatives: List[str] = Field(
        default_factory=list,
        description="Up to 8 other words or phrases a dataset description would use for "
                    "description_search, when researchers call it something else than the "
                    "user did. [] when description_search is null or already the usual word."
    )
    license_search: Optional[str] = Field(
        default=None,
        description="License string to search for, e.g. 'CC0', 'PDDL'"
    )
    request_wishes: List[str] = Field(
        default_factory=list,
        description=(
            "Phrases of the request that say what the asker wants or intends, not what the "
            "data contain, and that no dataset text would state: purpose ('for machine "
            "learning'), quality or difficulty ('standard manner', 'non-trivial', 'very "
            "simple'), documentation ('labeled', 'well documented', 'images included'), what "
            "a publication reports, datasets named only for comparison ('like ABIDE'). "
            "They are shown to the user as not searched. A phrase listed here must not "
            "appear in any other field."
        ),
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
    location_terms: List[str] = Field(
        default_factory=list,
        description="Cities, regions or countries where the data was collected, as written by "
                    "the user, one per entry, e.g. ['Bonn'], ['Germany'], ['California']. "
                    "Not an institution ('University of Bonn' → institution_terms)."
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
    study_structure: Optional[Literal["longitudinal", "cross-sectional"]] = Field(
        default=None,
        description="Study design the user asks for: 'longitudinal' (the same participants "
                    "followed over time) or 'cross-sectional'; None when not asked"
    )

    # ── Species (RAG/criteria.py) ─────────────────────────────────────────────
    species: List[str] = Field(
        default_factory=list,
        description="Species the user asks for, as codes: "
                    + ", ".join(_SPECIES_CODES)
                    + ". 'animal studies' / 'non-human' → any_animal; 'rodents' → mouse and rat; "
                    "monkeys, macaques, marmosets → primate; 'human' only when the user "
                    "explicitly limits the search to humans ('human participants', 'in humans', "
                    "'human fMRI'). Empty when the user does not name a species."
    )

    @field_validator("species", mode="before")
    @classmethod
    def _species_codes(cls, value: Any) -> List[str]:
        # Unknown words are dropped, not an error: a bad code must not fail the search.
        if value is None:
            return []
        return _normalise_species([value] if isinstance(value, str) else value)

    # ── Age group (RAG/criteria.py) ───────────────────────────────────────────
    age_groups: List[str] = Field(
        default_factory=list,
        description="Age groups the user names in words, as codes: "
                    + ", ".join(_AGE_GROUP_CODES)
                    + ". children, kids, pediatric, youth, school-age → children (includes "
                    "adolescents); adolescents, teenagers → adolescents; infants, babies, "
                    "newborns, toddlers → infants; elderly, older adults, seniors → older_adults. "
                    "Empty when not asked, for 'adults', and for numeric ages (those go to "
                    "ParticipantGroup.age_min / age_max)."
    )

    @field_validator("age_groups", mode="before")
    @classmethod
    def _age_group_codes(cls, value: Any) -> List[str]:
        # Unknown words are dropped, not an error: a bad code must not fail the search.
        if value is None:
            return []
        return _normalise_age_groups([value] if isinstance(value, str) else value)

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
    # Words of the question whose stem also matches unrelated words, with the
    # forms that may match ("ageing": ["ageing", "aging"]); RAG/word_forms.py.
    # The SQL rewriter passes them to the text search.
    word_forms: Dict[str, List[str]] = Field(default_factory=dict)

    def codes_for(self, field: RAGField) -> List[str]:
        for r in self.resolved:
            if r.field == field:
                return r.all_codes
        return []
