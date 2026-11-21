"""The backend's catalogue lookups that preprocessing uses.

The only place this package reaches into backend/services. Each accessor
raises ImportError outside the backend (no services/ on the path), as the
import itself did before, so the callers' fallbacks are unchanged.
verify_codes() falls back by itself.
"""

from __future__ import annotations

from typing import Any


def tsquery():
    """backend/services/fulltext/tsquery: full-text match counts of search terms."""
    from services.fulltext import tsquery
    return tsquery


def vectors():
    """backend/services/fulltext/vectors: the exact-search switch."""
    from services.fulltext import vectors
    return vectors


def word_form_store():
    """backend/services/catalogue/word_form_store: the catalogue's word forms."""
    from services.catalogue import word_form_store
    return word_form_store


def catalogue_values():
    """backend/services/catalogue/catalogue_values: coded values and their dataset counts."""
    from services.catalogue import catalogue_values
    return catalogue_values


def verify_codes(field: str, codes: list[str], retriever: Any = None) -> tuple[list[str], list[str]]:
    """(codes some dataset has, codes no dataset has) for vocabulary codes of *field*.

    Asks the backend's cache of coded values (catalogue_values) when it holds
    *field*. Elsewhere (scripts, benchmarks without the backend) the
    retriever's own database check answers, if it has a database URL. When
    neither can tell, every code counts as present, as before.
    """
    try:
        values = catalogue_values()
    except ImportError:  # outside the backend
        values = None
    if values is not None and values.loaded(field):
        present = [code for code in codes if values.dataset_count(field, code) > 0]
        return present, [code for code in codes if code not in present]
    if retriever is not None and hasattr(retriever, "db_verify_codes"):
        return retriever.db_verify_codes(field, codes)
    return list(codes), []
