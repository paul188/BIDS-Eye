"""What services/schema/tables.py records about its tables and text columns,
and that the maps built from it, the schema shown to the SQL model and the
coded fields agree with it."""

from __future__ import annotations

import re

import constants
import join_registry
from sqlalchemy import Table

from services.evidence import branches
from services.results import query_filters
from services.schema import tables
from services.schema import coded_fields
from services.schema.coded_fields import CodedField
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


# ── The schema shown to the SQL model (constants.py) ─────────────────────────

# bids_objects.run is an INTEGER; the prompt has always said TEXT.
_SHOWN_DIFFERENTLY = {("bids_objects", "run")}


def test_the_prompt_schema_names_declared_columns_with_their_types() -> None:
    declared = {table.name: table for table in tables.CATALOGUE_TABLES}
    ddl = constants.build_schema_ddl(constants.OPTIONAL_DATASET_COLUMNS)
    shown = re.findall(r"CREATE TABLE (\w+) \(.*?\n(.*?)\n\);", ddl, re.S)
    assert [name for name, _ in shown] == ["bids_datasets", "bids_objects", "bids_participants",
                                           "bids_derivatives", "bids_dataset_papers"]
    for name, body in shown:
        columns = re.findall(r"^    (\w+) +([A-Z]+(?:\[\])?)", body, re.M)
        assert columns
        for column, type_ in columns:
            actual = constants.prompt_type(declared[name].c[column])
            assert (type_ == actual) != ((name, column) in _SHOWN_DIFFERENTLY), (name, column)


# ── Coded fields (services/schema/coded_fields.py) ───────────────────────────

def test_coded_fields_are_columns_where_the_groups_say() -> None:
    assert set(coded_fields.CODED_FIELDS) == {field.value for field in CodedField}
    assert set(coded_fields.CONCEPT_FIELDS) <= set(coded_fields.CODED_FIELDS)
    for name in coded_fields.FILE_FIELDS:
        assert name in tables.objects.c and name in tables.dataset_facets.c
    for name in coded_fields.PARTICIPANT_FIELDS:
        assert name in tables.participants.c


def test_join_paths_agrees_with_the_coded_fields() -> None:
    for table, names in ((tables.objects, coded_fields.FILE_FIELDS),
                         (tables.participants, coded_fields.PARTICIPANT_FIELDS)):
        for name in names:
            assert join_registry.field_to_db_col(name) == (table.name, name)
    declared = {table.name: table for table in tables.CATALOGUE_TABLES}
    for name, info in join_registry._load()["fields"].items():
        assert info["column"] in declared[info["table"]].c, name
    codes = {name for name, info in join_registry._load()["fields"].items()
             if info["value_type"] == "canonical_code"}
    assert codes == set(coded_fields.CONCEPT_FIELDS)
