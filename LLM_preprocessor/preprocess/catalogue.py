"""The backend's catalogue lookups that preprocessing uses.

The only place this package reaches into backend/services. Each accessor
raises ImportError outside the backend (no services/ on the path), as the
import itself did before, so the callers' fallbacks are unchanged. tables()
has no fallback: the SQL this package writes needs the table names.
verify_codes() falls back by itself.
"""

from __future__ import annotations

from typing import Any


def tables():
    """backend/services/schema/tables: the catalogue's tables, for their names."""
    from services.schema import tables
    return tables


def tsquery():
    """backend/services/fulltext/tsquery: full-text match counts of search terms."""
    from services.fulltext import tsquery
    return tsquery


def vectors():
    """backend/services/fulltext/vectors: the exact-search switch."""
    from services.fulltext import vectors
    return vectors


def word_form_counts():
    """backend/services/fulltext/word_form_counts: the catalogue's word forms."""
    from services.fulltext import word_form_counts
    return word_form_counts


def value_counts():
    """The coded values of the backend's catalogue and their dataset counts
    (catalogue.values, services/catalogue/catalogue.py)."""
    from services.catalogue.catalogue import catalogue
    return catalogue.values


def verify_codes(field: str, codes: list[str], retriever: Any = None) -> tuple[list[str], list[str]]:
    """(codes some dataset has, codes no dataset has) for vocabulary codes of *field*.

    Asks the backend's counts of coded values (catalogue.values) when they hold
    *field*. Elsewhere (scripts, benchmarks without the backend) the
    retriever's own database check answers, if it has a database URL. When
    neither can tell, every code counts as present, as before.
    """
    try:
        values = value_counts()
    except ImportError:  # outside the backend
        values = None
    if values is not None and values.loaded(field):
        present = [code for code in codes if values.dataset_count(field, code) > 0]
        return present, [code for code in codes if code not in present]
    if retriever is not None and hasattr(retriever, "db_verify_codes"):
        return retriever.db_verify_codes(field, codes)
    return list(codes), []
