"""
services/fulltext/vectors.py
----------------------------
The stored full-text vectors a search reads, and the switches that choose them.

bids_datasets.search_tsv is a stored tsvector over
    weight A  name
    weight B  dx_status, study_domain, study_design, description_text,
              llm_conditions (studied conditions read from the README)
    weight C  llm_profile (LLM-written profile, services/indexing/readme_profile.py)
    weight D  readme_text
(see crawlers/migrate_add_search_tsv.sql).

search_tsv_papers is search_tsv plus, at weight D, the title, abstract and
keywords of the papers the dataset names as its own (TEXT_SEARCH_PAPERS,
papers_enabled). search_tsv_exact and search_tsv_exact_papers hold the same
text without stemming, for words that must match in the form asked for
(TEXT_SEARCH_EXACT, exact_enabled; crawlers/migrate_add_exact_search.sql).
How a search uses them: services/fulltext/text_search.py.
"""

from __future__ import annotations

import os

from services.catalogue import schema_capabilities

TS_CONFIG = "english"
FTS_COLUMN = "search_tsv"
PAPER_FTS_COLUMN = schema_capabilities.PAPER_FTS_COLUMN
PAPERS_ENV = "TEXT_SEARCH_PAPERS"
# The vectors without stemming, for words that must match in the form asked for.
EXACT_TS_CONFIG = "simple"
EXACT_FTS_COLUMN = schema_capabilities.EXACT_FTS_COLUMN
EXACT_PAPER_FTS_COLUMN = schema_capabilities.EXACT_PAPER_FTS_COLUMN
EXACT_ENV = "TEXT_SEARCH_EXACT"


def exact_enabled() -> bool:
    """True when words with decided forms must match in those forms.

    Read on every call, so a one-off container can set the variable.
    """
    flag = os.getenv(EXACT_ENV, "false").strip().lower() in ("1", "true", "yes", "on")
    return flag and schema_capabilities.has_exact_fts()


def papers_enabled() -> bool:
    """True when positive text matches should include paper text.

    Read on every call, so a one-off container can set the variable.
    """
    flag = os.getenv(PAPERS_ENV, "false").strip().lower() in ("1", "true", "yes", "on")
    return flag and schema_capabilities.has_paper_fts()
