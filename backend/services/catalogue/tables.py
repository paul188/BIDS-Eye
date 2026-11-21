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
    """Declare a column of bids_datasets that a migration in crawlers/ adds.

    *migration* is the SQL file that creates the column; check_schema names
    it in the error when the column is missing. *flag* is PAPERS_FLAG,
    EXACT_FLAG or None: with a flag, the column is required only while that
    flag is on, without one always.

    Both are stored in the column's info dict. SQLAlchemy ignores that dict,
    so they change nothing in the SQL the column is used in.
    """
    info = {MIGRATION: migration}
    if flag:
        info[FLAG] = flag
    return Column(name, type_, info=info)


# The columns of bids_datasets that the migrations add, plus its key.
datasets = Table(
    BIDSDataset.__tablename__, metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    # Full-text vectors. The paper ones add the text of the papers a dataset
    # names; the exact ones are not stemmed.
    _migration_column("search_tsv", TSVECTOR, _SEARCH_TSV),
    _migration_column("search_tsv_papers", TSVECTOR, _PAPER_SEARCH, PAPERS_FLAG),
    _migration_column("search_tsv_exact", TSVECTOR, _EXACT_SEARCH, EXACT_FLAG),
    _migration_column("search_tsv_exact_papers", TSVECTOR, _EXACT_SEARCH, EXACT_FLAG),
    # The stored subject count (crawlers/openneuro-crawler/remote/dataset_facets.py).
    _migration_column("n_subjects", Integer, _FACETS),
    # OpenNeuro fields; the ones in constants.OPTIONAL_DATASET_COLUMNS are
    # shown to the SQL model once enough datasets fill them.
    _migration_column("readme_text", Text, _OPENNEURO),
    _migration_column("dx_status", Text, _OPENNEURO),
    _migration_column("study_domain", Text, _OPENNEURO),
    _migration_column("study_design", Text, _OPENNEURO),
    _migration_column("species", Text, _OPENNEURO),
    _migration_column("study_longitudinal", Text, _OPENNEURO),
    _migration_column("senior_author", Text, _OPENNEURO),
    _migration_column("publish_date", DateTime(timezone=True), _OPENNEURO),
    _migration_column("on_brainlife", Boolean, _OPENNEURO),
    _migration_column("downloads", Integer, _OPENNEURO),
    _migration_column("views", Integer, _OPENNEURO),
    _migration_column("age_min", Double, _OPENNEURO),
    _migration_column("age_max", Double, _OPENNEURO),
    _migration_column("size_bytes", BigInteger, _OPENNEURO),
    _migration_column("deprecated_reason", Text, _OPENNEURO),
    _migration_column("institutions", ARRAY(Text), _INSTITUTIONS),
    # Read by the API itself, never by generated SQL.
    _migration_column("llm_profile", Text, _PROFILES),
    _migration_column("paper_relation", JSONB, _DESCRIPTIONS),
    _migration_column("dataset_summary", Text, _DESCRIPTIONS),
)

# One row per dataset and (datatype, suffix, task) it has files for: a
# summary of bids_objects that the executed SQL reads instead
# (sql_rewriter._use_summaries).
dataset_facets = Table(
    "bids_dataset_facets", metadata,
    Column("dataset_id", UUID(as_uuid=True)),
    Column("datatype", Text),
    Column("suffix", Text),
    Column("task", Text),
    info={MIGRATION: _FACETS},
)

# Every word form in the catalogue's text, its stem and how many datasets have it.
word_forms = Table(
    "bids_word_forms", metadata,
    Column("form", Text, primary_key=True),
    Column("stem", Text),
    Column("ndoc", Integer),
    info={MIGRATION: _EXACT_SEARCH, FLAG: EXACT_FLAG},
)
