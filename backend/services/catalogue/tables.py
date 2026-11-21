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
with the database at startup. A column or table marked with a FLAG is
needed only while that environment switch is on.
"""

from __future__ import annotations

import os
from typing import Any, cast

from sqlalchemy import (BigInteger, Boolean, Column, DateTime, Double, Integer, MetaData,
                        Table, Text)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.types import TypeEngine

from db.models import BIDSDataset, BIDSObject, BIDSParticipant

# Keys of Column.info and Table.info.
MIGRATION = "migration"
FLAG = "flag"

PAPERS_FLAG = "TEXT_SEARCH_PAPERS"
EXACT_FLAG = "TEXT_SEARCH_EXACT"


def switched_on(flag: str) -> bool:
    """True when the environment switch *flag* is on.

    Read on every call, so a one-off container can set the variable.
    """
    return os.getenv(flag, "false").strip().lower() in ("1", "true", "yes", "on")

_SEARCH_TSV = "crawlers/migrate_add_search_tsv.sql"
_PAPER_SEARCH = "crawlers/migrate_add_paper_search.sql"
_EXACT_SEARCH = "crawlers/migrate_add_exact_search.sql"
_FACETS = "crawlers/migrate_add_dataset_facets.sql"
_OPENNEURO = "crawlers/migrate_add_openneuro_metadata.sql"
_INSTITUTIONS = "crawlers/migrate_add_institutions.sql"
_PROFILES = "crawlers/migrate_add_readme_profiles.sql"
_DESCRIPTIONS = "crawlers/migrate_add_dataset_descriptions.sql"


def _added(name: str, type_: TypeEngine[Any] | type[TypeEngine[Any]], migration: str,
           flag: str | None = None) -> Column[Any]:
    info = {MIGRATION: migration}
    if flag:
        info[FLAG] = flag
    return Column(name, type_, info=info)


metadata = MetaData()

objects = cast(Table, BIDSObject.__table__)
participants = cast(Table, BIDSParticipant.__table__)

# The columns of bids_datasets that the migrations add, plus its key.
datasets = Table(
    BIDSDataset.__tablename__, metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    # Full-text vectors. The paper ones add the text of the papers a dataset
    # names; the exact ones are not stemmed.
    _added("search_tsv", TSVECTOR, _SEARCH_TSV),
    _added("search_tsv_papers", TSVECTOR, _PAPER_SEARCH, PAPERS_FLAG),
    _added("search_tsv_exact", TSVECTOR, _EXACT_SEARCH, EXACT_FLAG),
    _added("search_tsv_exact_papers", TSVECTOR, _EXACT_SEARCH, EXACT_FLAG),
    # The stored subject count (crawlers/openneuro-crawler/remote/dataset_facets.py).
    _added("n_subjects", Integer, _FACETS),
    # OpenNeuro fields; the ones in constants.OPTIONAL_DATASET_COLUMNS are
    # shown to the SQL model once enough datasets fill them.
    _added("readme_text", Text, _OPENNEURO),
    _added("dx_status", Text, _OPENNEURO),
    _added("study_domain", Text, _OPENNEURO),
    _added("study_design", Text, _OPENNEURO),
    _added("species", Text, _OPENNEURO),
    _added("study_longitudinal", Text, _OPENNEURO),
    _added("senior_author", Text, _OPENNEURO),
    _added("publish_date", DateTime(timezone=True), _OPENNEURO),
    _added("on_brainlife", Boolean, _OPENNEURO),
    _added("downloads", Integer, _OPENNEURO),
    _added("views", Integer, _OPENNEURO),
    _added("age_min", Double, _OPENNEURO),
    _added("age_max", Double, _OPENNEURO),
    _added("size_bytes", BigInteger, _OPENNEURO),
    _added("deprecated_reason", Text, _OPENNEURO),
    _added("institutions", ARRAY(Text), _INSTITUTIONS),
    # Read by the API itself, never by generated SQL.
    _added("llm_profile", Text, _PROFILES),
    _added("paper_relation", JSONB, _DESCRIPTIONS),
    _added("dataset_summary", Text, _DESCRIPTIONS),
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
