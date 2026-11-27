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
  bids_dataset_facets, bids_word_forms, bids_papers, bids_dataset_papers,
  bids_derivatives, bids_dataset_sites and bids_site_places. The columns
  are added to BIDS-SQL's bids_datasets table with db.extensions.extend, so
  datasets has both kinds; the tables are declared here (dataset_facets,
  word_forms, papers, dataset_papers, derivatives, dataset_sites,
  site_places).
  Each names the migration that creates it, and check_schema
  (services/schema/check.py) compares them with the database at
  startup.

A flag is a switch (services/switches.py) that turns an optional search
feature on. Two matter here, PAPERS_FLAG and EXACT_FLAG. A column or table that
names a flag is needed only while that flag is on; the others are always
needed.

Each table also records what its rows describe (About), and each text column
which text it holds (TextKind). Code that needs a word or an evidence source
per table reads those instead of listing the tables again.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, cast

from sqlalchemy import (BigInteger, Boolean, Column, DateTime, Double, Integer, MetaData,
                        Table, Text)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.types import TypeEngine

from db.extensions import extend
from db.models import BIDSDataset, BIDSObject, BIDSParticipant

from services.switches import Switch


# ── Flags ────────────────────────────────────────────────────────────────────

# Text matches also search the text of the papers a dataset names.
PAPERS_FLAG = Switch.TEXT_SEARCH_PAPERS
# Words whose form is decided must match in that form.
EXACT_FLAG = Switch.TEXT_SEARCH_EXACT


# ── What each declaration records ────────────────────────────────────────────

# Keys of Column.info and Table.info, dicts SQLAlchemy keeps but never reads.
# For check_schema:
MIGRATION = "migration"
FLAG = "flag"
# For the evidence and the chips:
ABOUT = "about"   # of a table: its About
TEXT = "text"     # of a column: its TextKind
LABEL_IN_TEXT = "label_in_text"   # of a column: an uploader label that search_tsv also holds


class About(str, Enum):
    """What the rows of a table describe."""
    DATASET = "dataset"
    PARTICIPANTS = "participants"
    FILES = "files"
    DERIVATIVES = "derivatives"
    SITES = "sites"
    PAPERS = "papers"
    WORD_FORMS = "word_forms"


class TextKind(str, Enum):
    """Which text a text column of bids_datasets holds."""
    TEXT = "text"                        # name, description, study fields and README together
    TEXT_AND_PAPERS = "text_and_papers"  # TEXT plus the dataset's papers
    README = "readme"
    SHORT_TEXT = "short_text"            # the name or the description


# The SQL files in crawlers/ that create the declarations below.
_SEARCH_TSV = "crawlers/migrate_add_search_tsv.sql"
_PAPER_SEARCH = "crawlers/migrate_add_paper_search.sql"
_EXACT_SEARCH = "crawlers/migrate_add_exact_search.sql"
_FACETS = "crawlers/migrate_add_dataset_facets.sql"
_OPENNEURO = "crawlers/migrate_add_openneuro_metadata.sql"
_INSTITUTIONS = "crawlers/migrate_add_institutions.sql"
_PROFILES = "crawlers/migrate_add_readme_profiles.sql"
_DESCRIPTIONS = "crawlers/migrate_add_dataset_descriptions.sql"
_DATASET_LINKS = "crawlers/migrate_add_dataset_links.sql"
_SITE_PLACES = "crawlers/migrate_add_site_places.sql"


def _created_by(migration: str, flag: str | None = None) -> dict[str, str]:
    """The info of a column or table that the SQL file *migration* creates.

    With a *flag*, check_schema requires it only while that flag is on.
    """
    info = {MIGRATION: migration}
    if flag:
        info[FLAG] = flag
    return info


def _migration_column(name: str, type_: TypeEngine[Any] | type[TypeEngine[Any]], migration: str,
                      flag: str | None = None, text: TextKind | None = None,
                      label_in_text: bool = False) -> Column[Any]:
    """A column to add to bids_datasets; *migration* and *flag* as in _created_by."""
    info = _created_by(migration, flag)
    if text:
        info[TEXT] = text
    if label_in_text:
        info[LABEL_IN_TEXT] = True
    return Column(name, type_, info=info)


# ── Tables from db/models.py, columns unchanged ──────────────────────────────

objects = cast(Table, BIDSObject.__table__)
objects.info[ABOUT] = About.FILES
participants = cast(Table, BIDSParticipant.__table__)
participants.info[ABOUT] = About.PARTICIPANTS


# ── Columns the migrations add to bids_datasets ──────────────────────────────

# bids_datasets: the columns of db/models.py plus the ones added here.
datasets = extend(
    BIDSDataset,
    # Full-text search vectors, GENERATED: PostgreSQL recomputes them on every write.
    # Stemmed dataset text: A name, B study fields, C llm_profile, D readme_text.
    _migration_column("search_tsv", TSVECTOR, _SEARCH_TSV, text=TextKind.TEXT),
    # search_tsv plus the primary papers' title, abstract and keywords; only to match.
    _migration_column("search_tsv_papers", TSVECTOR, _PAPER_SEARCH, PAPERS_FLAG, TextKind.TEXT_AND_PAPERS),
    # search_tsv unstemmed, so "ageing" does not also match "age".
    _migration_column("search_tsv_exact", TSVECTOR, _EXACT_SEARCH, EXACT_FLAG, TextKind.TEXT),
    # search_tsv_papers unstemmed.
    _migration_column("search_tsv_exact_papers", TSVECTOR, _EXACT_SEARCH, EXACT_FLAG, TextKind.TEXT_AND_PAPERS),

    # Distinct subjects in bids_objects, refreshed after each crawl (dataset_facets.py).
    _migration_column("n_subjects", Integer, _FACETS),

    # From the OpenNeuro API; "uploader" fields are free text, often empty.
    # Sanitised README of the latest snapshot.
    _migration_column("readme_text", Text, _OPENNEURO, text=TextKind.README),
    # Uploader's diagnosis status, e.g. 'Healthy / Control'.
    _migration_column("dx_status", Text, _OPENNEURO, label_in_text=True),
    # Uploader's research topic.
    _migration_column("study_domain", Text, _OPENNEURO, label_in_text=True),
    # Uploader's study design, e.g. 'Longitudinal'.
    _migration_column("study_design", Text, _OPENNEURO, label_in_text=True),
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
datasets.info[ABOUT] = About.DATASET
# BIDS-SQL's own text columns.
for _column in (datasets.c.name, datasets.c.description_text):
    _column.info[TEXT] = TextKind.SHORT_TEXT


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
    info=_created_by(_FACETS) | {ABOUT: About.FILES},
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
    info=_created_by(_EXACT_SEARCH, EXACT_FLAG) | {ABOUT: About.WORD_FORMS},
)

# One row per paper a dataset names, from Europe PMC, bioRxiv, arXiv or
# OpenAlex. Filled by scripts/backfill_papers.py.
papers = Table(
    "bids_papers", metadata,
    Column("id", BigInteger, primary_key=True),
    # doi:…, pmid:…, pmcid:… or arxiv:…; unique.
    Column("paper_key", Text),
    # The published version's DOI; preprint_doi for the preprint.
    Column("doi", Text),
    Column("preprint_doi", Text),
    Column("pmid", Text),
    Column("pmcid", Text),
    Column("arxiv_id", Text),
    Column("title", Text),
    Column("abstract", Text),
    Column("year", Integer),
    Column("journal", Text),
    # True when no published version is known.
    Column("is_preprint", Boolean),
    Column("authors", ARRAY(Text)),
    # Medical Subject Headings; mesh_major the main ones.
    Column("mesh_terms", ARRAY(Text)),
    Column("mesh_major", ARRAY(Text)),
    Column("keywords", ARRAY(Text)),
    Column("is_open_access", Boolean),
    Column("license", Text),
    # Where the record came from: europepmc, biorxiv, medrxiv, arxiv, openalex.
    Column("source", Text),
    Column("fetched_at", DateTime(timezone=True)),
    info=_created_by(_PAPER_SEARCH) | {ABOUT: About.PAPERS},
)

# Which dataset names which paper, and in which role.
dataset_papers = Table(
    "bids_dataset_papers", metadata,
    # bids_datasets.id.
    Column("dataset_id", UUID(as_uuid=True), primary_key=True),
    # bids_papers.id.
    Column("paper_id", BigInteger, primary_key=True),
    # 'primary', 'related', 'tool' or 'reuse'.
    Column("role", Text),
    # Why the link has that role.
    Column("role_reason", Text),
    # Where the link was read: associated, references, acknowledge, readme, reuse_field.
    Column("sources", ARRAY(Text)),
    # The reference string the link was read from.
    Column("raw_ref", Text),
    # Order within the dataset, most trusted first.
    Column("position", Integer),
    info=_created_by(_PAPER_SEARCH) | {ABOUT: About.PAPERS},
)

# One row per processed version of a dataset. Filled by
# scripts/backfill_dataset_links.py and the crawler.
derivatives = Table(
    "bids_derivatives", metadata,
    Column("id", BigInteger, primary_key=True),
    # bids_datasets.id.
    Column("dataset_id", UUID(as_uuid=True)),
    # Normalised pipeline: 'fmriprep', 'mriqc', 'freesurfer', … 'unspecified'.
    Column("pipeline", Text),
    # The pipeline as the source names it, and its version (GeneratedBy).
    Column("pipeline_name", Text),
    Column("pipeline_version", Text),
    # 'openneuro_derivatives' or 'in_dataset'.
    Column("source", Text),
    # s3://openneuro-derivatives/… or derivatives/<folder>.
    Column("location", Text),
    # sub-* folders in the derivative.
    Column("n_subjects", Integer),
    Column("fetched_at", DateTime(timezone=True)),
    info=_created_by(_DATASET_LINKS) | {ABOUT: About.DERIVATIVES},
)

# One row per collection site name a dataset gives, and its evidence.
dataset_sites = Table(
    "bids_dataset_sites", metadata,
    Column("id", BigInteger, primary_key=True),
    # bids_datasets.id.
    Column("dataset_id", UUID(as_uuid=True)),
    # The name or statement as found.
    Column("name_raw", Text),
    # 'dicom' (scanner headers) or 'ethics_board' (approvals).
    Column("source", Text),
    # Research Organization Registry id, once resolved.
    Column("ror_id", Text),
    # The quote, where the source is prose.
    Column("evidence", Text),
    info=_created_by(_DATASET_LINKS) | {ABOUT: About.SITES},
)

# Where the names of bids_dataset_sites are: each raw name matched to the
# Research Organization Registry (ROR), one row per matched organisation.
# Kept current by the crawler (remote/site_places.py).
site_places = Table(
    "bids_site_places", metadata,
    # bids_dataset_sites.name_raw, as stored.
    Column("name_raw", Text, primary_key=True),
    # https://ror.org/…; '' when checked and not matched.
    Column("ror_id", Text, primary_key=True),
    # ROR display name.
    Column("institution", Text),
    Column("city", Text),
    # State, province or Land, e.g. 'North Rhine-Westphalia'.
    Column("region", Text),
    Column("country", Text),
    # ISO 3166-1 alpha-2.
    Column("country_code", Text),
    Column("lat", Double),
    Column("lng", Double),
    Column("matched_at", DateTime(timezone=True)),
    info=_created_by(_SITE_PLACES) | {ABOUT: About.SITES},
)

# What check_schema checks: in each table, the columns a migration creates.
MIGRATED_TABLES = (datasets, dataset_facets, word_forms, papers, dataset_papers, derivatives,
                   dataset_sites, site_places)

# Every table declared here or taken from BIDS-SQL.
CATALOGUE_TABLES = (objects, participants, *MIGRATED_TABLES)

# Table name -> what its rows describe. A table without an About fails here.
TABLE_ABOUT: dict[str, About] = {table.name: table.info[ABOUT] for table in CATALOGUE_TABLES}

# Text column of bids_datasets -> which text it holds.
COLUMN_TEXT: dict[str, TextKind] = {column.name: column.info[TEXT] for column in datasets.c
                                    if TEXT in column.info}

# Uploader labels of bids_datasets that search_tsv also holds
# (crawlers/openneuro-crawler/remote/openneuro_metadata.py).
LABELS_IN_TEXT: frozenset[str] = frozenset(column.name for column in datasets.c
                                           if column.info.get(LABEL_IN_TEXT))

# The full-text search vectors of bids_datasets.
SEARCH_VECTORS = frozenset(column.name for column in datasets.c
                           if isinstance(column.type, TSVECTOR))
