"""What services/schema/tables.py records about its tables and text columns,
and that the maps built from it cover every value."""

from __future__ import annotations

from sqlalchemy import Table

from services.evidence import branches
from services.results import query_filters
from services.schema import tables
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
