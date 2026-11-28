"""
services/results/display/result_page.py
---------------------------------------
The API's answer for one page of a search.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from routers.schemas import DatasetSchema, QueryFilterSchema, QueryResponse, TextToSQLResult


def found_message(total: int) -> str:
    return f"Found {total} dataset{'' if total == 1 else 's'} matching your query."


def response(
    datasets: List[DatasetSchema], total: int, query_id: str, page: int, page_size: int, *,
    translation: Optional[TextToSQLResult] = None, filters: Sequence[QueryFilterSchema] = (),
) -> QueryResponse:
    """One page of results; the first page also carries its translation and chips."""
    return QueryResponse(
        message=found_message(total),
        translation=translation,
        filters=list(filters),
        datasets=datasets,
        query_id=query_id,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )
