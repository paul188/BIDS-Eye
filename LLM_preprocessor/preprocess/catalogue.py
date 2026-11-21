"""The backend's catalogue lookups that preprocessing uses.

The only place this package reaches into backend/services. Each function
raises ImportError outside the backend (no services/ on the path), as the
import itself did before, so the callers' fallbacks are unchanged.
"""


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
