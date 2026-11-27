"""What services/schema/tables.py records about its tables and text columns,
and that the maps built from it, the schema shown to the SQL model and the
coded fields agree with it."""

from __future__ import annotations

import re

from services.translate.prompt import schema_ddl
import join_registry
from preprocess import RAGField
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


# ── The schema shown to the SQL model (schema_ddl.py) ─────────────────────────

def test_the_prompt_schema_names_declared_columns_with_their_types() -> None:
    declared = {table.name: table for table in tables.CATALOGUE_TABLES}
    ddl = schema_ddl.build_schema_ddl(schema_ddl.OPTIONAL_DATASET_COLUMNS)
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


# ── RAG fields (backend/llm/preprocess/plan.py) ───────────────────────────────

def test_every_rag_field_says_where_it_is_stored() -> None:
    """The bridge from a RAGField to its column: its join_paths.yaml entry."""
    declared = {table.name: table for table in tables.CATALOGUE_TABLES}
    for field in RAGField:
        if field is RAGField.SCAN:  # a lookup across datatype, suffix and task
            continue
        info = join_registry.get_field_info(field.value)
        assert info is not None, f"{field.value} has no entry in join_paths.yaml"
        assert info["rag_backed"], field.value
        stored = join_registry.field_column(field.value)
        assert stored.column in declared[stored.table].c, field.value


def test_the_coded_rag_fields_are_the_coded_fields() -> None:
    coded = {field.value for field in RAGField} & set(coded_fields.CODED_FIELDS)
    assert coded == set(coded_fields.CONCEPT_FIELDS)
    rag_backed = {name for name, info in join_registry._load()["fields"].items() if info["rag_backed"]}
    assert rag_backed == {field.value for field in RAGField} - {RAGField.SCAN.value}
