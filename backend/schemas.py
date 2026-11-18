"""
schemas.py
----------
Pydantic response models for the BIDS-Eye API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated


class ParticipantSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    participant_id: str
    age: Optional[float] = None
    sex: Optional[str] = None
    handedness: Optional[str] = None
    diagnosis: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class ParticipantSummarySchema(BaseModel):
    """Counts over all of a dataset's participants (the participants list
    itself is capped)."""
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
    # Uncertain tier (services/fulltext/uncertain_tier.py): 0 or None = every filter
    # confirmed, 1 = probably relevant, 2 = possibly relevant; with the reason
    # (e.g. "ages not recorded"). Set only when UNCERTAIN_TIER is on.
    uncertain_level: Optional[int] = None
    uncertain_reasons: Optional[str] = None
    # Computed fields (not direct ORM columns)
    subject_count: Optional[int] = None
    participants: List[ParticipantSchema] = []
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
    # months"). services/results/match_reasons.py.
    match_reasons: List[str] = []


class DatasetListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[DatasetSchema]


class TextToSQLResult(BaseModel):
    """Output contract for the Text-To-SQL translation layer."""
    sql: str
    params: Dict[str, Any] = {}
    explanation: Optional[str] = None
    self_corrected: bool = False
    # Ranking inputs, applied in postprocessing (not baked into `sql`) so the
    # base SELECT can be cached and reused across pages.
    scored_filters: List[Dict[str, Any]] = []
    apply_relevance: bool = False
    # Full-text query that ranks the result (services/fulltext/text_search.py); None
    # when the question has no text terms or the database has no search_tsv.
    rank_tsquery: Optional[str] = None
    # Number of first candidates an LLM judged and reordered; None = not re-ranked.
    reranked: Optional[int] = None
    # The query before the uncertain tier wraps it, read for the filter chips
    # (services/results/query_filters.py). Server-side only.
    filter_sql: Optional[str] = Field(default=None, exclude=True)
    # The query without its free-text-only conditions (empty-result fallback).
    relaxed_sql: Optional[str] = Field(default=None, exclude=True)
    # datatype / suffix / task values no dataset has, corrected or dropped
    # before the query ran (services/catalogue/catalogue_values.py).
    value_fixes: List[str] = []
    # True when no model was asked: the question was translated before
    # (services/stores/translation_store.py).
    from_store: bool = False
    # A new translation as a store entry; the router keeps it once the search
    # ran without correction. Server-side only.
    store_entry: Optional[Any] = Field(default=None, exclude=True)


class QueryFilterSchema(BaseModel):
    """One condition of the query, shown as a chip the user can remove."""
    id: str
    kind: str
    field: str
    label: str
    negated: bool = False
    text_words: List[str] = []
    detail: str = ""
    # The full list behind a label shortened to "+N"; [] when nothing is hidden.
    values: List[str] = []


class QueryRequest(BaseModel):
    # The frontend sends nothing blank and at most 2000 characters; the same
    # rules here, so no search (and no daily search) is spent on whitespace.
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class QueryPageRequest(BaseModel):
    """Fetch a different page of an already-translated query, reusing the cached
    base SQL (no new LLM call)."""
    query_id: str
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


class CrawlerStatusResponse(BaseModel):
    running: bool
    current_accession: Optional[str] = None
    queue: List[str] = []
    last_run_started: Optional[str] = None
    last_run_finished: Optional[str] = None
    last_error: Optional[str] = None
    indexed_count: int = 0
    error_count: int = 0
