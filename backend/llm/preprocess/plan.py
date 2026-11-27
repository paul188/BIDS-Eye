"""The plan the model returns (QueryPlan) and the plan after RAG (AugmentedQueryPlan)."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from services.schema.coded_fields import CodedField
from criteria import SPECIES_CODES as _SPECIES_CODES, normalise_species as _normalise_species
from criteria import AGE_GROUP_CODES as _AGE_GROUP_CODES, \
    normalise_age_groups as _normalise_age_groups


log = logging.getLogger("preprocess")


class RAGField(str, Enum):
    """A field the RAG layer resolves user terms for.

    The value is the field's key in backend/vocabulary/join_paths.yaml, which
    says where it is stored (join_registry.field_column); the coded ones are
    the CodedField of the same column. tests/services/schema/test_schema_tables.py
    checks both, for every member but SCAN.
    """
    DIAGNOSIS = CodedField.DIAGNOSIS.value
    TASK      = CodedField.TASK.value
    SUFFIX    = CodedField.SUFFIX.value
    DATATYPE  = CodedField.DATATYPE.value
    AUTHOR    = "author"
    FUNDING   = "funding"
    INSTITUTION = "institution"
    SCAN      = "scan"      # internal: multi-field lookup across datatype+suffix+task; no column


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


class ParticipantAttribute(str, Enum):
    """A property one participant has: a column of bids_participants."""
    DIAGNOSIS  = "diagnosis"
    SEX        = "sex"
    AGE        = "age"
    HANDEDNESS = "handedness"


class ParticipantCriterion(BaseModel):
    """One property a participant of the group has ("ADHD", "female", "aged 8 to 12")."""
    attribute: ParticipantAttribute
    terms: List[str] = Field(
        default_factory=list,
        description="The words of the request for this property: diagnoses as written "
                    "(['ADHD'], ['healthy controls']; several are alternatives), sex "
                    "(['female']), handedness (['left-handed']). Empty for age."
    )
    min: Optional[float] = Field(default=None, description="age only: lowest age in years")
    max: Optional[float] = Field(default=None, description="age only: highest age in years")


# The keys a ParticipantGroup had before it held criteria. Recorded plans
# (benchmarks/*/results) and tests still use them; _from_flat_fields reads them.
_FLAT_GROUP_KEYS = ("diagnosis_terms", "sex", "handedness", "age_min", "age_max")


class ParticipantGroup(BaseModel):
    """
    A set of participants the dataset must have, described by what each of
    them is: "20 female ADHD patients aged 8 to 12" is one group with three
    criteria, all of which hold for the same 20 people. Properties of the
    dataset itself (scans, tasks, species, text) are not group criteria.

    quantifier says how many of the dataset's participants are meant:
    some  — at least one, or at least min_subjects ("with ADHD patients")
    all   — every participant ("only female participants")
    none  — no participant ("without epilepsy patients")
    """
    group_name: str = Field(
        description="Short label, e.g. 'patients', 'controls', 'all_participants'"
    )
    quantifier: Literal["some", "all", "none"] = Field(
        default="some",
        description="some = the dataset has such participants; all = every participant is "
                    "like this ('only', 'exclusively', 'all-female'); none = the dataset has no "
                    "such participant ('without', 'excluding', 'no ... patients')"
    )
    criteria: List[ParticipantCriterion] = Field(
        default_factory=list,
        description="What each participant of the group is. All criteria hold for the same "
                    "participant. Empty for a plain count ('at least 50 subjects')."
    )
    min_subjects: Optional[int] = Field(
        default=None,
        description="Minimum number of participants of this group the dataset must have"
    )

    @model_validator(mode="before")
    @classmethod
    def _from_flat_fields(cls, data: Any) -> Any:
        # {"diagnosis_terms": [...], "sex": [...], "age_min": 8} → criteria.
        if not isinstance(data, dict) or not any(k in data for k in _FLAT_GROUP_KEYS):
            return data
        data = dict(data)
        criteria = list(data.get("criteria") or [])
        for key, attribute in (("diagnosis_terms", "diagnosis"), ("sex", "sex"),
                               ("handedness", "handedness")):
            terms = data.pop(key, None) or []
            if terms:
                criteria.append({"attribute": attribute, "terms": list(terms)})
        low, high = data.pop("age_min", None), data.pop("age_max", None)
        if low is not None or high is not None:
            criteria.append({"attribute": "age", "min": low, "max": high})
        data["criteria"] = criteria
        return data

    @model_validator(mode="after")
    def _no_count_of_nobody(self) -> "ParticipantGroup":
        # "At least 20 of no participant" has no meaning: the minimum is dropped.
        if self.quantifier == "none":
            self.min_subjects = None
        return self

    # Read access by attribute. The model writes criteria; the code reads these.
    def terms(self, attribute: ParticipantAttribute) -> List[str]:
        return [t for c in self.criteria if c.attribute == attribute for t in c.terms]

    def attributes(self) -> set:
        return {c.attribute for c in self.criteria}

    @property
    def diagnosis_terms(self) -> List[str]:
        return self.terms(ParticipantAttribute.DIAGNOSIS)

    @property
    def sex(self) -> List[str]:
        return self.terms(ParticipantAttribute.SEX)

    @property
    def handedness(self) -> List[str]:
        return self.terms(ParticipantAttribute.HANDEDNESS)

    @property
    def age_min(self) -> Optional[float]:
        # Several age criteria all hold: the highest lower bound.
        lows = [c.min for c in self.criteria if c.attribute == ParticipantAttribute.AGE and c.min is not None]
        return max(lows) if lows else None

    @property
    def age_max(self) -> Optional[float]:
        highs = [c.max for c in self.criteria if c.attribute == ParticipantAttribute.AGE and c.max is not None]
        return min(highs) if highs else None

    def add_terms(self, attribute: ParticipantAttribute, terms: List[str]) -> None:
        """Add words to the group's criterion for *attribute* (a new one if there is none)."""
        for criterion in self.criteria:
            if criterion.attribute == attribute:
                criterion.terms.extend(t for t in terms if t not in criterion.terms)
                return
        self.criteria.append(ParticipantCriterion(attribute=attribute, terms=list(terms)))


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


class TermAlternatives(BaseModel):
    """Other words a dataset description uses for one scan term."""
    term: str = Field(description="The scan term exactly as in scan_terms")
    alternatives: List[str] = Field(
        default_factory=list,
        description="Up to 8 words or short phrases a dataset description writes for exactly "
                    "this thing or a specific kind of it: named paradigms, tasks, stimuli, "
                    "stimulus sets, measures. Never broader terms or merely related topics.")


class SortKey(str, Enum):
    """What the user can order datasets by. Each key is a column the search
    has for every dataset (services/translate/participant_query_compiler.py
    ORDER_COLUMNS)."""
    SUBJECTS  = "subjects"    # number of subjects: "largest", "most participants"
    SIZE      = "size"        # download size: "biggest in GB"
    PUBLISHED = "published"   # publication date: "newest", "most recent"
    DOWNLOADS = "downloads"   # "most downloaded", "most popular"
    VIEWS     = "views"       # "most viewed"
    NAME      = "name"        # "alphabetically"


class Ordering(BaseModel):
    """The order and number of results the user asked for.

    "the 10 largest fMRI datasets" → {sort_key: subjects, direction: DESC, limit: 10};
    "newest datasets first" → {sort_key: published, direction: DESC};
    "5 datasets on autism" → {limit: 5}: no order named, so the results keep
    their relevance order and are not cut (cutting before the relevance
    ranking would keep five arbitrary ones).
    """
    sort_key: Optional[SortKey] = Field(
        default=None,
        description="What to order by; None when the user names only a number of results"
    )
    direction: Literal["ASC", "DESC"] = Field(
        default="DESC",
        description="DESC = largest / newest / most first; ASC = smallest / oldest / fewest "
                    "first, and A to Z for name"
    )
    limit: Optional[int] = Field(
        default=None,
        description="The number of results the user asked for ('top 10' → 10); None otherwise"
    )

    @field_validator("limit")
    @classmethod
    def _positive_limit(cls, value: Optional[int]) -> Optional[int]:
        # "top 0" or a negative number is no limit the search can honour.
        return value if value is None or value > 0 else None

    @field_validator("sort_key", mode="before")
    @classmethod
    def _known_key(cls, value: Any) -> Optional[str]:
        # An unknown key is dropped, not an error: a bad key must not fail the search.
        values = {key.value for key in SortKey}
        return value if value is None or value in values else None


class QueryPlan(BaseModel):
    """
    Complete structured representation of a user query.
    Passed to the RAG layer, then to the SQL generation LLM.
    """

    # ── Intent ────────────────────────────────────────────────────────────────
    natural_language_summary: str = Field(
        description="One sentence restating what the user wants in precise terms"
    )

    # ── Participant cohorts ───────────────────────────────────────────────────
    groups: List[ParticipantGroup] = Field(
        default_factory=list,
        description="The participants the dataset must (or must not) have. See section 1.2."
    )
    group_operator: Literal["AND", "OR"] = Field(
        default="AND",
        description="Between the groups whose quantifier is some or all: AND = the dataset "
                    "has every group; OR = any one suffices. Groups with quantifier none "
                    "always hold."
    )

    @model_validator(mode="before")
    @classmethod
    def _from_flat_fields(cls, data: Any) -> Any:
        # Plans written before groups held criteria (recordings, tests):
        # excluded_diagnosis_terms becomes a group with quantifier none, and
        # a group's task_terms a scan requirement (a task is done by the
        # dataset's subjects, it does not tell one group from another).
        if not isinstance(data, dict):
            return data
        excluded = data.get("excluded_diagnosis_terms")
        tasks = [t for g in data.get("groups") or [] if isinstance(g, dict)
                 for t in g.get("task_terms") or []]
        if not excluded and not tasks and "excluded_diagnosis_terms" not in data:
            return data
        data = dict(data)
        groups = [{k: v for k, v in g.items() if k != "task_terms"} if isinstance(g, dict) else g
                  for g in data.get("groups") or []]
        for term in data.pop("excluded_diagnosis_terms", None) or []:
            groups.append({"group_name": "excluded", "quantifier": "none",
                           "criteria": [{"attribute": "diagnosis", "terms": [term]}]})
        data["groups"] = groups
        if tasks:
            data["scan_requirements"] = list(data.get("scan_requirements") or []) + [
                {"scan_terms": tasks}]
        return data

    @property
    def cohorts(self) -> List[ParticipantGroup]:
        """The groups the dataset must have (quantifier some or all)."""
        return [g for g in self.groups if g.quantifier != "none"]

    @property
    def excluded_diagnosis_terms(self) -> List[str]:
        """Diagnoses no participant may have: the groups with quantifier none and
        only a diagnosis. "No epilepsy patients over 60" (diagnosis and age) is no
        plain exclusion of epilepsy; excluded_groups() has it."""
        return [t for g in self.groups
                if g.quantifier == "none" and g.attributes() == {ParticipantAttribute.DIAGNOSIS}
                for t in g.diagnosis_terms]

    def excluded_groups(self) -> List[ParticipantGroup]:
        """Groups with quantifier none that excluded_diagnosis_terms does not cover."""
        return [g for g in self.groups
                if g.quantifier == "none" and g.attributes() != {ParticipantAttribute.DIAGNOSIS}]

    # ── Scan / modality requirements ─────────────────────────────────────────
    scan_requirements: List[ScanRequirement] = Field(
        default_factory=list,
        description="Each entry is one modality/task constraint on the dataset's files"
    )

    scan_term_alternatives: List[TermAlternatives] = Field(
        default_factory=list,
        description="For scan terms that name a task, paradigm, stimulus or measure: the words "
                    "dataset descriptions use for it instead. See section 1.3.3.",
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

    # ── Species (backend/vocabulary/criteria.py) ─────────────────────────────────────────────
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

    # ── Age group (backend/vocabulary/criteria.py) ───────────────────────────────────────────
    age_groups: List[str] = Field(
        default_factory=list,
        description="Age groups the user names in words, as codes: "
                    + ", ".join(_AGE_GROUP_CODES)
                    + ". children, kids, pediatric, youth, school-age → children (includes "
                    "adolescents); adolescents, teenagers → adolescents; infants, babies, "
                    "newborns, toddlers → infants; elderly, older adults, seniors → older_adults. "
                    "Empty when not asked, for 'adults', and for numeric ages (those go to "
                    "an age criterion of a participant group)."
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

    # ── Order and number of results ───────────────────────────────────────────
    ordering: Optional[Ordering] = Field(
        default=None,
        description="Only when the user asks for an order ('largest first', 'most recent') "
                    "or a number of results ('top 10', 'the 5 largest'); None otherwise. "
                    "Without a sort_key the results stay in relevance order."
    )

    @field_validator("ordering")
    @classmethod
    def _drop_empty_ordering(cls, value: Optional[Ordering]) -> Optional[Ordering]:
        return value if value is not None and (value.sort_key or value.limit) else None

    @property
    def user_ordered(self) -> bool:
        """True when the user named what to order by, so the search keeps that
        order (and the limit) instead of ranking by relevance."""
        return self.ordering is not None and self.ordering.sort_key is not None

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
    sense: Optional[str] = None             # sense picked in backend/vocabulary/senses.yaml
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
    # forms that may match ("ageing": ["ageing", "aging"]); backend/vocabulary/stem_collisions.py.
    # The SQL rewriter passes them to the text search.
    allowed_forms: Dict[str, List[str]] = Field(default_factory=dict)
    # Phrases of description_search left out of the search because fewer than
    # three datasets say them ("postictal"). The explanation names them.
    unmatched_text: List[str] = Field(default_factory=list)
