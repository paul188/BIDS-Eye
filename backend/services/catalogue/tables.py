"""
services/catalogue/tables.py
----------------------------
The catalogue tables the search reads, declared once.

BIDS-SQL's models (db/models.py) declare the tables the crawler creates with
create_all_tables(): bids_datasets with its core columns, bids_objects,
bids_participants. The migrations in crawlers/ add more: the search
vectors, the OpenNeuro fields, the per-dataset summaries. This module
declares those, so code names them through these objects instead of
writing table and column names out as strings.

Each added column and table names the migration that creates it, and
check_schema (services/catalogue/catalogue.py) compares the declarations
with the database at startup.

A flag is an environment variable that turns an optional search feature
on. There are two, PAPERS_FLAG and EXACT_FLAG. A column or table that
names a flag is needed only while that flag is on; the others are always
needed.
"""

from __future__ import annotations

import os
from typing import Any, cast

from sqlalchemy import (BigInteger, Boolean, Column, DateTime, Double, Integer, MetaData,
                        Table, Text)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.types import TypeEngine

from db.models import BIDSDataset, BIDSObject, BIDSParticipant

# The keys under which a declared column or table stores its migration and
# its flag, in SQLAlchemy's Column.info and Table.info.
MIGRATION = "migration"
FLAG = "flag"

# The flags: names of environment variables, read by is_flag.
# Text matches also search the text of the papers a dataset names.
PAPERS_FLAG = "TEXT_SEARCH_PAPERS"
# Words whose form is decided must match in that form (unstemmed vectors,
# bids_word_forms).
EXACT_FLAG = "TEXT_SEARCH_EXACT"

# The migrations in crawlers/ that create the declarations below. These are
# file paths, not flags.
_SEARCH_TSV = "crawlers/migrate_add_search_tsv.sql"
_PAPER_SEARCH = "crawlers/migrate_add_paper_search.sql"
_EXACT_SEARCH = "crawlers/migrate_add_exact_search.sql"
_FACETS = "crawlers/migrate_add_dataset_facets.sql"
_OPENNEURO = "crawlers/migrate_add_openneuro_metadata.sql"
_INSTITUTIONS = "crawlers/migrate_add_institutions.sql"
_PROFILES = "crawlers/migrate_add_readme_profiles.sql"
_DESCRIPTIONS = "crawlers/migrate_add_dataset_descriptions.sql"


def is_flag(flag: str) -> bool:
    """True when the flag *flag* (PAPERS_FLAG or EXACT_FLAG) is on.

    On means the environment variable is 1, true, yes or on; unset or
    anything else is off. Read on every call, so a one-off container or a
    test can set the variable after import.
    """
    return os.getenv(flag, "false").strip().lower() in ("1", "true", "yes", "on")


metadata = MetaData()

objects = cast(Table, BIDSObject.__table__)
participants = cast(Table, BIDSParticipant.__table__)


def _migration_column(name: str, type_: TypeEngine[Any] | type[TypeEngine[Any]], migration: str,
                      flag: str | None = None) -> Column[Any]:
    """Declare a bids_datasets column that the SQL file *migration* creates.

    With a *flag*, check_schema requires the column only while that flag is on.
    """
    info = {MIGRATION: migration}
    if flag:
        info[FLAG] = flag
    return Column(name, type_, info=info)


# The columns of bids_datasets that the migrations add, plus its key.
datasets = Table(
    BIDSDataset.__tablename__, metadata,
    # The dataset's key, declared in db/models.py; repeated so code can name
    # datasets.c.id.
    Column("id", UUID(as_uuid=True), primary_key=True),

    # Full-text search vectors. All four are GENERATED columns: PostgreSQL
    # recomputes them on every write.
    # The dataset's text, stemmed with 'english', weighted A name, B uploader
    # study fields and LLM-found conditions, C llm_profile, D readme_text.
    # Text matches and the ranking use it.
    _migration_column("search_tsv", TSVECTOR, _SEARCH_TSV),
    # search_tsv plus the title, abstract and keywords of the dataset's
    # primary papers at weight D. Used only to match, never to rank.
    _migration_column("search_tsv_papers", TSVECTOR, _PAPER_SEARCH, PAPERS_FLAG),
    # search_tsv without stemming ('simple'), so "ageing" does not also match
    # "age". Searched together with search_tsv for words whose form is decided.
    _migration_column("search_tsv_exact", TSVECTOR, _EXACT_SEARCH, EXACT_FLAG),
    # search_tsv_papers without stemming.
    _migration_column("search_tsv_exact_papers", TSVECTOR, _EXACT_SEARCH, EXACT_FLAG),

    # Number of distinct subjects in the dataset's files: a copy of
    # COUNT(DISTINCT subject) over bids_objects, refreshed after each crawl
    # (crawlers/openneuro-crawler/remote/dataset_facets.py).
    _migration_column("n_subjects", Integer, _FACETS),

    # Fields from the OpenNeuro API, filled by the crawler and by
    # scripts/backfill_openneuro_metadata.py. The ones in
    # constants.OPTIONAL_DATASET_COLUMNS are shown to the SQL model once
    # enough datasets fill them. "Uploader-entered" means free text the
    # uploader typed on OpenNeuro, often empty.
    # Sanitised README of the latest snapshot.
    _migration_column("readme_text", Text, _OPENNEURO),
    # Uploader-entered diagnosis status, e.g. 'Healthy / Control'.
    _migration_column("dx_status", Text, _OPENNEURO),
    # Uploader-entered research topic.
    _migration_column("study_domain", Text, _OPENNEURO),
    # Uploader-entered study design, e.g. 'Longitudinal'.
    _migration_column("study_design", Text, _OPENNEURO),
    # Uploader-entered species, e.g. 'Human', 'Mouse'.
    _migration_column("species", Text, _OPENNEURO),
    # Uploader-entered 'Cross-Sectional', 'Longitudinal', …
    _migration_column("study_longitudinal", Text, _OPENNEURO),
    # Uploader-entered senior author.
    _migration_column("senior_author", Text, _OPENNEURO),
    # When the dataset went public on OpenNeuro.
    _migration_column("publish_date", DateTime(timezone=True), _OPENNEURO),
    # Whether the dataset has been processed on brainlife.io.
    _migration_column("on_brainlife", Boolean, _OPENNEURO),
    # OpenNeuro download count of the latest snapshot.
    _migration_column("downloads", Integer, _OPENNEURO),
    # OpenNeuro view count of the latest snapshot.
    _migration_column("views", Integer, _OPENNEURO),
    # Youngest and oldest participant age OpenNeuro reports, in years. Ages
    # outside 0-120 are dropped as entry errors; NULL when none are left.
    _migration_column("age_min", Double, _OPENNEURO),
    _migration_column("age_max", Double, _OPENNEURO),
    # Total size of the latest snapshot in bytes.
    _migration_column("size_bytes", BigInteger, _OPENNEURO),
    # NULL while the dataset is fine on OpenNeuro; otherwise why OpenNeuro
    # marked it deprecated.
    _migration_column("deprecated_reason", Text, _OPENNEURO),

    # Every InstitutionName found in the dataset's sidecar JSON files.
    _migration_column("institutions", ARRAY(Text), _INSTITUTIONS),

    # Written by an LLM; read by the API itself, never by generated SQL.
    # Short profile of the README in standard terms
    # (scripts/extract_readme_profiles.py). Part of search_tsv at weight C,
    # and read by the re-ranker.
    _migration_column("llm_profile", Text, _PROFILES),
    # How the dataset's primary paper relates to it (reports, larger_sample,
    # related_study, tool, unclear), with the evidence quote and whether the
    # paper is shown (services/dataset_description.py).
    _migration_column("paper_relation", JSONB, _DESCRIPTIONS),
    # Two or three generated sentences, only for datasets whose README and
    # description together are under 500 characters.
    _migration_column("dataset_summary", Text, _DESCRIPTIONS),
)

# One row per dataset and (datatype, suffix, task) it has files for: a
# summary of bids_objects (2.7 million rows, this one about 21,000) that the
# executed SQL reads instead (sql_rewriter._use_summaries). Refreshed after
# each crawl, like n_subjects.
dataset_facets = Table(
    "bids_dataset_facets", metadata,
    # bids_datasets.id.
    Column("dataset_id", UUID(as_uuid=True)),
    # BIDS datatype folder, e.g. 'anat', 'func', 'eeg'.
    Column("datatype", Text),
    # BIDS suffix, e.g. 'T1w', 'bold', 'events'.
    Column("suffix", Text),
    # BIDS task label, e.g. 'rest'; NULL for files without a task.
    Column("task", Text),
    info={MIGRATION: _FACETS},
)

# Every word form in the catalogue's text with its stem, filled by
# scripts/refresh_word_forms.py and after each crawl. It shows which forms a
# stem merges, which decides whether a searched word must match in the form
# asked for.
word_forms = Table(
    "bids_word_forms", metadata,
    # The word as it appears in the text, e.g. 'ageing'.
    Column("form", Text, primary_key=True),
    # Its 'english' stem, e.g. 'age'.
    Column("stem", Text),
    # How many datasets contain the form.
    Column("ndoc", Integer),
    info={MIGRATION: _EXACT_SEARCH, FLAG: EXACT_FLAG},
)
