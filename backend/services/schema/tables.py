"""
services/schema/tables.py
-------------------------
The catalogue tables the search reads, declared once, so code names them
through these objects instead of writing table and column names as strings.

The tables come from two places:

  BIDS-SQL's models (db/models.py), created by create_all_tables():
  bids_datasets with its core columns, bids_objects, bids_participants.
  They are taken from there (datasets, objects, participants).

  The migrations in crawlers/, run by hand: more columns on bids_datasets
  (search vectors, OpenNeuro fields, summaries) and the tables
  bids_dataset_facets and bids_word_forms. The columns are added to
  BIDS-SQL's bids_datasets table with db.extensions.extend, so datasets has
  both kinds; the tables are declared here (dataset_facets, word_forms).
  Each names the migration that creates it, and check_schema
  (services/schema/check.py) compares them with the database at
  startup.

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

from db.extensions import extend
from db.models import BIDSDataset, BIDSObject, BIDSParticipant


# ── Flags ────────────────────────────────────────────────────────────────────

# Text matches also search the text of the papers a dataset names.
PAPERS_FLAG = "TEXT_SEARCH_PAPERS"
# Words whose form is decided must match in that form.
EXACT_FLAG = "TEXT_SEARCH_EXACT"


def is_flag(flag: str) -> bool:
    """True when the flag *flag* (PAPERS_FLAG or EXACT_FLAG) is on.

    On means the environment variable is 1, true, yes or on; unset or
    anything else is off. Read on every call, so a one-off container or a
    test can set the variable after import.
    """
    return os.getenv(flag, "false").strip().lower() in ("1", "true", "yes", "on")


# ── What each declaration records for check_schema ───────────────────────────

# Keys of Column.info and Table.info, dicts SQLAlchemy keeps but never reads.
MIGRATION = "migration"
FLAG = "flag"

# The SQL files in crawlers/ that create the declarations below.
_SEARCH_TSV = "crawlers/migrate_add_search_tsv.sql"
_PAPER_SEARCH = "crawlers/migrate_add_paper_search.sql"
_EXACT_SEARCH = "crawlers/migrate_add_exact_search.sql"
_FACETS = "crawlers/migrate_add_dataset_facets.sql"
_OPENNEURO = "crawlers/migrate_add_openneuro_metadata.sql"
_INSTITUTIONS = "crawlers/migrate_add_institutions.sql"
_PROFILES = "crawlers/migrate_add_readme_profiles.sql"
_DESCRIPTIONS = "crawlers/migrate_add_dataset_descriptions.sql"


def _created_by(migration: str, flag: str | None = None) -> dict[str, str]:
    """The info of a column or table that the SQL file *migration* creates.

    With a *flag*, check_schema requires it only while that flag is on.
    """
    info = {MIGRATION: migration}
    if flag:
        info[FLAG] = flag
    return info


def _migration_column(name: str, type_: TypeEngine[Any] | type[TypeEngine[Any]], migration: str,
                      flag: str | None = None) -> Column[Any]:
    """A column to add to bids_datasets; *migration* and *flag* as in _created_by."""
    return Column(name, type_, info=_created_by(migration, flag))


# ── Tables from db/models.py, unchanged ──────────────────────────────────────

objects = cast(Table, BIDSObject.__table__)
participants = cast(Table, BIDSParticipant.__table__)


# ── Columns the migrations add to bids_datasets ──────────────────────────────

# bids_datasets: the columns of db/models.py plus the ones added here.
datasets = extend(
    BIDSDataset,
    # Full-text search vectors, GENERATED: PostgreSQL recomputes them on every write.
    # Stemmed dataset text: A name, B study fields, C llm_profile, D readme_text.
    _migration_column("search_tsv", TSVECTOR, _SEARCH_TSV),
    # search_tsv plus the primary papers' title, abstract and keywords; only to match.
    _migration_column("search_tsv_papers", TSVECTOR, _PAPER_SEARCH, PAPERS_FLAG),
    # search_tsv unstemmed, so "ageing" does not also match "age".
    _migration_column("search_tsv_exact", TSVECTOR, _EXACT_SEARCH, EXACT_FLAG),
    # search_tsv_papers unstemmed.
    _migration_column("search_tsv_exact_papers", TSVECTOR, _EXACT_SEARCH, EXACT_FLAG),

    # Distinct subjects in bids_objects, refreshed after each crawl (dataset_facets.py).
    _migration_column("n_subjects", Integer, _FACETS),

    # From the OpenNeuro API; "uploader" fields are free text, often empty.
    # Sanitised README of the latest snapshot.
    _migration_column("readme_text", Text, _OPENNEURO),
    # Uploader's diagnosis status, e.g. 'Healthy / Control'.
    _migration_column("dx_status", Text, _OPENNEURO),
    # Uploader's research topic.
    _migration_column("study_domain", Text, _OPENNEURO),
    # Uploader's study design, e.g. 'Longitudinal'.
    _migration_column("study_design", Text, _OPENNEURO),
    # Uploader's species, e.g. 'Human', 'Mouse'.
    _migration_column("species", Text, _OPENNEURO),
    # Uploader's 'Cross-Sectional', 'Longitudinal', …
    _migration_column("study_longitudinal", Text, _OPENNEURO),
    # Uploader's senior author.
    _migration_column("senior_author", Text, _OPENNEURO),
    # When the dataset went public.
    _migration_column("publish_date", DateTime(timezone=True), _OPENNEURO),
    # Whether it has been processed on brainlife.io.
    _migration_column("on_brainlife", Boolean, _OPENNEURO),
    # Download count of the latest snapshot.
    _migration_column("downloads", Integer, _OPENNEURO),
    # View count of the latest snapshot.
    _migration_column("views", Integer, _OPENNEURO),
    # Youngest reported participant age in years; ages outside 0-120 dropped.
    _migration_column("age_min", Double, _OPENNEURO),
    # Oldest reported participant age in years; ages outside 0-120 dropped.
    _migration_column("age_max", Double, _OPENNEURO),
    # Size of the latest snapshot in bytes.
    _migration_column("size_bytes", BigInteger, _OPENNEURO),
    # Why OpenNeuro deprecated the dataset; NULL while it is fine.
    _migration_column("deprecated_reason", Text, _OPENNEURO),

    # Every InstitutionName in the dataset's sidecar JSON files.
    _migration_column("institutions", ARRAY(Text), _INSTITUTIONS),

    # Written by an LLM; read by the API itself, never by generated SQL.
    # Short README profile in standard terms; in search_tsv and read by the re-ranker.
    _migration_column("llm_profile", Text, _PROFILES),
    # How the primary paper relates to the dataset, with evidence quote and shown flag.
    _migration_column("paper_relation", JSONB, _DESCRIPTIONS),
    # Two or three sentences, only where README and description are under 500 chars.
    _migration_column("dataset_summary", Text, _DESCRIPTIONS),
)


# ── Tables the migrations add ────────────────────────────────────────────────

metadata = MetaData()

# One row per dataset and (datatype, suffix, task) it has files for: a small
# summary of bids_objects that the executed SQL reads instead
# (sql_rewriter._use_summaries). Refreshed after each crawl.
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
    info=_created_by(_FACETS),
)

# Every word form in the catalogue's text with its stem: shows which forms a
# stem merges. Filled by scripts/refresh_word_forms.py and after each crawl.
word_forms = Table(
    "bids_word_forms", metadata,
    # The word as it appears in the text, e.g. 'ageing'.
    Column("form", Text, primary_key=True),
    # Its 'english' stem, e.g. 'age'.
    Column("stem", Text),
    # How many datasets contain the form.
    Column("ndoc", Integer),
    info=_created_by(_EXACT_SEARCH, EXACT_FLAG),
)

# What check_schema checks: in each table, the columns a migration creates.
MIGRATED_TABLES = (datasets, dataset_facets, word_forms)
