"""
schemas.py
----------
Pydantic response models for the BIDS-Eye API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ParticipantSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    participant_id: str
    age: Optional[float] = None
    sex: Optional[str] = None
    handedness: Optional[str] = None
    diagnosis: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


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
    # Computed fields (not direct ORM columns)
    subject_count: Optional[int] = None
    participants: List[ParticipantSchema] = []


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


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=2000)


class QueryPageRequest(BaseModel):
    """Fetch a different page of an already-translated query, reusing the cached
    base SQL (no new LLM call)."""
    query_id: str
    page: int = 1
    page_size: int = 20


class QueryResponse(BaseModel):
    message: str
    translation: Optional[TextToSQLResult] = None  # exposed for debugging
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
