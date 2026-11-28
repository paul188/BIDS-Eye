"""
routers/schemas.py
----------
Pydantic response models for the BIDS-Eye API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from typing_extensions import Annotated

from services.fulltext.ranking import RankQuery


class ParticipantSummarySchema(BaseModel):
    """Counts over all of a dataset's participants."""
    total: int
    diagnoses: Dict[str, int] = {}   # raw diagnosis value → participants
    female: int = 0
    male: int = 0
    age_min: Optional[float] = None
    age_max: Optional[float] = None


class LinkedPaperSchema(BaseModel):
    """The dataset's primary paper, shown when it was checked to report the
    dataset or a larger sample that includes it (services/indexing/dataset_description.py)."""
    title: str
    year: Optional[int] = None
    url: Optional[str] = None
    relation: str               # reports | larger_sample
    relation_label: str         # "reports this dataset", …


class DatasetSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    accession_id: Optional[str] = None
    bids_version: Optional[str] = None
    dataset_type: str
    source_type: str
    remote_url: Optional[str] = None
    validation_status: Optional[str] = None
    authors: Optional[List[str]] = None
    institutions: Optional[List[str]] = None
    description_text: Optional[str] = None
    # Set when the dataset is gone from, or flagged deprecated on, its source
    # repository; None for datasets that are fine. Shown as a tag in the UI.
    deprecated_reason: Optional[str] = None
    # Uncertain tier (services/uncertain/tier.py): 0 or None = every filter
    # confirmed, 1 = probably relevant, 2 = possibly relevant; with the reason
    # (e.g. "ages not recorded"). Set only when UNCERTAIN_TIER is on.
    uncertain_level: Optional[int] = None
    uncertain_reasons: Optional[str] = None
    # Computed fields (not direct ORM columns)
    subject_count: Optional[int] = None
    participant_summary: Optional[ParticipantSummarySchema] = None
    # A passage of the README: the part that matches the search when it has
    # text terms, else the start. Matched words sit between "⟦" and "⟧".
    snippet: Optional[str] = None
    # Generated 2-3 sentences, only where the README is thin; shown labelled
    # as generated (services/indexing/dataset_description.py).
    generated_summary: Optional[str] = None
    linked_paper: Optional[LinkedPaperSchema] = None
    # Why the dataset matched: for each filter that can match in several
    # ways, the ways that hold ("Age: 12 of 40 participants aged 4–23
    # months"). services/results/display/match_reasons.py.
    match_reasons: List[str] = []


class ScoredFilter(BaseModel):
    """A code the question asks for and how well it matched (0..1); datasets
    that have the code rank higher (sql_rewriter._inject_relevance_cte)."""
    field: str
    code: str
    score: float


class TextToSQLResult(BaseModel):
    """Output contract for the Text-To-SQL translation layer."""
    sql: str
    params: Dict[str, Any] = {}
    explanation: Optional[str] = None
    self_corrected: bool = False
    # Ranking inputs, applied in postprocessing (not baked into `sql`) so the
    # base SELECT can be cached and reused across pages.
    scored_filters: List[ScoredFilter] = []
    apply_relevance: bool = False
    # Full-text queries that rank the result (services/fulltext/ranking.py); None
    # when the question has no text terms.
    rank_query: Optional[RankQuery] = None
    # Number of first candidates an LLM judged and reordered; None = not re-ranked.
    reranked: Optional[int] = None
    # The query before the uncertain tier wraps it, read for the filter chips
    # (services/results/display/chips.py). Server-side only.
    filter_sql: Optional[str] = Field(default=None, exclude=True)
    # The query without its free-text-only conditions (empty-result fallback).
    relaxed_sql: Optional[str] = Field(default=None, exclude=True)
    # datatype / suffix / task values no dataset has, corrected or dropped
    # before the query ran (services/translate/value_check.py).
    value_fixes: List[str] = []
    # Who wrote the SQL: "compiled", "model: <why the compiler declined>" or
    # "keyword". For query_log only; holds no words of the question.
    sql_path: Optional[str] = Field(default=None, exclude=True)


class ChipKind(str, Enum):
    """What a chip describes (QueryFilterSchema.kind; frontend api/index.ts)."""
    DIAGNOSIS = "diagnosis"
    MODALITY = "modality"
    TASK = "task"
    AGE = "age"
    SEX = "sex"
    SUBJECTS = "subjects"   # the dataset's subject count
    COUNT = "count"         # another count: sessions, runs, patients with a diagnosis
    TEXT = "text"           # only words of the dataset text
    TOPIC = "topic"         # a qualified subject searched as one unit
    SENSE = "sense"         # the meanings a topic excludes
    COLUMN = "column"       # another catalogue column: species, size, location, …
    SORT = "sort"           # ORDER BY / LIMIT
    OTHER = "other"         # a condition with no summary of its own

    @classmethod
    def _missing_(cls, value: object) -> Optional[ChipKind]:
        # Searches saved before 2026-11-26 call COLUMN "field".
        return cls.COLUMN if value == "field" else None

    @property
    def named(self) -> Optional[str]:
        """The chip's field when its kind decides it ("Diagnosis"); None when
        the chip names it (a count, a column, a sort)."""
        return _KIND_NAMES.get(self)


_KIND_NAMES = {
    ChipKind.DIAGNOSIS: "Diagnosis", ChipKind.MODALITY: "Modality", ChipKind.TASK: "Task",
    ChipKind.AGE: "Age", ChipKind.SEX: "Sex", ChipKind.SUBJECTS: "Subjects",
    ChipKind.TEXT: "Text", ChipKind.TOPIC: "Topic", ChipKind.SENSE: "Not meant",
}


class QueryFilterSchema(BaseModel):
    """One condition of the query, shown as a chip (services/results/display/chips.py)."""
    # Position in the query: "w0" WHERE, "h0" HAVING, "o0" ORDER BY, "w0n" a
    # topic's excluded meanings.
    id: str
    kind: ChipKind
    field: str = ""     # what the chip is about: "Diagnosis"; ChipKind.named by default
    label: str          # the asked value: "Parkinson's disease"
    negated: bool = False   # the dataset must NOT match
    # Words of the dataset text (README, description) the condition also
    # searches; the page marks them. A "text" chip is only these.
    text_words: List[str] = []
    detail: str = ""    # longer description for a tooltip
    # Of a label that lists values: all of them, and how many the label lists;
    # the page shows "+N" for the rest. [] and 0 for any other label.
    values: List[str] = []
    shown: int = 0

    @model_validator(mode="after")
    def _named_by_kind(self) -> QueryFilterSchema:
        if not self.field:
            if self.kind.named is None:
                raise ValueError(f"a {self.kind.value} chip needs its field")
            self.field = self.kind.named
        return self


class QueryRequest(BaseModel):
    # The frontend sends nothing blank and at most 2000 characters; the same
    # rules here, so no search (and no daily search) is spent on whitespace.
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class QueryPageRequest(BaseModel):
    """Fetch a different page of an already-translated query, reusing the cached
    base SQL (no new LLM call)."""
    # The id is uuid4().hex (32 characters, services/stores/translated_queries.py);
    # anything much longer is not one of ours.
    query_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    page: int = 1
    page_size: int = 20


class QueryResponse(BaseModel):
    message: str
    translation: Optional[TextToSQLResult] = None  # exposed for debugging
    # How the question was understood: one chip per condition. Empty when the
    # query has no readable conditions (sample fallback, self-corrected SQL).
    filters: List[QueryFilterSchema] = []
    datasets: List[DatasetSchema]
    # Pagination metadata (shared by the initial /query and /query/page replies).
    query_id: Optional[str] = None
    total: int = 0
    page: int = 1
    page_size: int = 20
    has_more: bool = False
    # The search_history row this answer was saved as; None for a user who
    # has not turned the history on (services/stores/search_history.py).
    history_id: Optional[str] = None


class HistoryEntrySummary(BaseModel):
    id: str
    asked_at: datetime
    question: str
    total: int


class HistoryList(BaseModel):
    """A user's saved searches, newest first; none while ``enabled`` is false."""
    enabled: bool
    retention_days: int
    entries: List[HistoryEntrySummary] = []


class HistoryEntry(BaseModel):
    id: str
    asked_at: datetime
    question: str
    # The answer as it was given then. Its query_id has usually expired.
    answer: QueryResponse


class HistoryEnabled(BaseModel):
    enabled: bool

