"""What services/schema/tables.py records about its tables and text columns,
and that the maps built from it, the schema shown to the SQL model and the
fields (services/schema/fields.py) agree with it."""

from __future__ import annotations

import re

from services.translate.prompt import schema_ddl
from sqlalchemy import Table

from services.evidence import branches
from services.results import query_filters
from services.schema import tables
from services.schema import fields
from services.schema.fields import CatalogueField
from services.schema.tables import About, TextKind


def test_every_declared_table_says_what_it_is_about() -> None:
    declared = {value for value in vars(tables).values() if isinstance(value, Table)}
    assert declared == set(tables.CATALOGUE_TABLES)
    assert set(tables.TABLE_ABOUT) == {table.name for table in declared}


def test_every_search_vector_says_which_text_it_holds() -> None:
    assert tables.SEARCH_VECTORS == {"search_tsv", "search_tsv_exact",
                                     "search_tsv_papers", "search_tsv_exact_papers"}
    assert tables.SEARCH_VECTORS <= set(tables.COLUMN_TEXT)


def test_every_about_has_a_chip_word_and_a_source_or_is_no_source() -> None:
    assert set(query_filters._ABOUT_WORDS) == set(About)
    # A new About must be given a source, or be added to the two that have none.
    assert set(branches._ABOUT_SOURCES) == set(About) - {About.DATASET, About.WORD_FORMS}


def test_every_text_kind_has_a_source() -> None:
    assert set(branches._KIND_SOURCES) == set(TextKind)


# ── The schema shown to the SQL model (schema_ddl.py) ─────────────────────────

def test_the_prompt_schema_names_declared_columns_with_their_types() -> None:
    declared = {table.name: table for table in tables.CATALOGUE_TABLES}
    ddl = schema_ddl.build_schema_ddl(tables.OPTIONAL_DATASET_COLUMNS)
    shown = re.findall(r"CREATE TABLE (\w+) \(.*?\n(.*?)\n\);", ddl, re.S)
    assert [name for name, _ in shown] == ["bids_datasets", "bids_objects", "bids_participants",
                                           "bids_derivatives", "bids_dataset_papers"]
    for name, body in shown:
        columns = re.findall(r"^    (\w+) +([A-Z]+(?:\[\])?)", body, re.M)
        assert columns
        for column, type_ in columns:
            assert type_ == schema_ddl.prompt_type(declared[name].c[column]), (name, column)


def test_the_prompt_schema_links_every_other_table_to_its_dataset() -> None:
    """With a foreign key (BIDS-SQL's models) or without (the tables declared here)."""
    ddl = schema_ddl.build_schema_ddl()
    assert ddl.count("CREATE TABLE") == 5
    assert len(re.findall(r"^    dataset_id +UUID REFERENCES bids_datasets\(id\),$", ddl, re.M)) == 4
    assert len(re.findall(r"^    id +UUID PRIMARY KEY,$", ddl, re.M)) == 3
    # A composite key is not shown as the key of one column.
    assert "PRIMARY KEY REFERENCES" not in ddl


# ── Fields (services/schema/fields.py) ────────────────────────────────────────

def test_every_field_is_stored_in_a_declared_column() -> None:
    """fields.py names tables and columns as strings; these are the declared ones."""
    declared = {table.name: table for table in tables.CATALOGUE_TABLES}
    for field in CatalogueField:
        assert field.stored.column in declared[field.stored.table].c, field
        # The outer alias d is bids_datasets, and only bids_datasets.
        assert (field.stored.table == tables.datasets.name) == (not field.stored.needs_exists), field


def test_the_coded_field_groups_are_where_they_say() -> None:
    assert set(fields.CONCEPT_FIELDS) <= set(fields.CODED_FIELDS)
    for name in fields.FILE_FIELDS:
        assert CatalogueField(name).stored.table == tables.objects.name
        assert name in tables.dataset_facets.c
    for name in fields.PARTICIPANT_FIELDS:
        assert CatalogueField(name).stored.table == tables.participants.name


def test_a_card_names_a_column_by_its_label_or_its_name() -> None:
    assert tables.card_label("dx_status") == "diagnosis label"
    assert tables.card_label("species") == "species label"
    assert tables.card_label("age_min") == "age min"
    assert tables.card_label("not_a_column") == "not a column"


def test_the_optional_columns_are_declared_on_bids_datasets_with_notes() -> None:
    """schema_ddl shows an optional column only with its notes, after the
    columns that are always shown; it fails to import otherwise."""
    assert "readme_text" in tables.OPTIONAL_DATASET_COLUMNS
    assert "name" not in tables.OPTIONAL_DATASET_COLUMNS
    shown = [name for name, _, _ in schema_ddl._OPTIONAL_DATASET_COLUMNS]
    assert sorted(shown) == sorted(tables.OPTIONAL_DATASET_COLUMNS)
