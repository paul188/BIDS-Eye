"""
schemas.py
----------
Pydantic response models for the BIDS-Eye API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


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


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    message: str
    translation: Optional[TextToSQLResult] = None  # exposed for debugging
    datasets: List[DatasetSchema]


class CrawlerStatusResponse(BaseModel):
    running: bool
    current_accession: Optional[str] = None
    queue: List[str] = []
    last_run_started: Optional[str] = None
    last_run_finished: Optional[str] = None
    last_error: Optional[str] = None
    indexed_count: int = 0
    error_count: int = 0
