"""The full-text rank query must travel from the SQL rewriter to the page query.

The rewriter computes ``rank_tsquery`` and the router passes
``translation.rank_tsquery`` to ``build_page_sql``. ``_run_local`` sits between
the two; when it dropped the value, results were filtered by full-text search
but never ordered by it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import preprocess
import retriever
from services.catalogue import schema_capabilities as caps
from services.fulltext import word_frequencies
from services.translate import text_to_sql
from services.translate.sql_rewriter import build_page_sql

BASE = (
    "SELECT {{COLS}} FROM bids_datasets d "
    "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL "
)


@pytest.fixture(autouse=True)
def _full_text_available():
    caps.set_fts(True)
    caps.set_available_columns(())
    word_frequencies.set_frequency_lookup(None)
    yield
    caps.set_fts(False)
    caps.set_available_columns(())
    word_frequencies.set_frequency_lookup(None)


def _fake_pipeline(monkeypatch, generated_sql: str, **plan_overrides) -> None:
    """Replace the two LLM stages and the resolver; keep the real rewriter."""
    plan = SimpleNamespace(
        query_family=SimpleNamespace(value="description_search"),
        natural_language_summary="demo",
        result_limit=None,
        order_by=None,
        groups=[],
    )
    for name, value in plan_overrides.items():
        setattr(plan, name, value)
    augmented = SimpleNamespace(
        plan=plan, augmented_question="datasets mentioning ketamine", scored_filters=[],
        resolved=[],
    )
    monkeypatch.setattr(preprocess, "run_pipeline", lambda *args, **kwargs: augmented)
    monkeypatch.setattr(retriever, "MetadataRetriever", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        text_to_sql, "_gemini_sql_generation", lambda *args, **kwargs: generated_sql
    )


def test_run_local_passes_the_rank_query_on(monkeypatch) -> None:
    _fake_pipeline(monkeypatch, BASE + "WHERE text_match('ketamine') GROUP BY d.id")

    result = text_to_sql._run_local("datasets mentioning ketamine", "dummy-key")

    assert "search_tsv @@" in result.sql
    assert result.rank_tsquery == "ketamine"
    # What the router does with it: the page query orders by the text rank.
    # Compared in lower case: with sqlglot installed the rewriter re-renders
    # function names and keywords in upper case.
    page_sql, _ = build_page_sql(
        result.sql, result.scored_filters, result.apply_relevance, 20, 0,
        rank_tsquery=result.rank_tsquery,
    )
    assert "ts_rank_cd" in page_sql.lower()
    assert "ranked.relevance_score + ranked.text_rank desc" in page_sql.lower()


def test_run_local_has_no_rank_query_without_text_terms(monkeypatch) -> None:
    _fake_pipeline(monkeypatch, BASE + "WHERE d.license ILIKE '%CC0%' GROUP BY d.id")

    result = text_to_sql._run_local("CC0 datasets", "dummy-key")

    assert result.rank_tsquery is None


def test_run_local_has_no_rank_query_without_the_search_column(monkeypatch) -> None:
    caps.set_fts(False)
    _fake_pipeline(monkeypatch, BASE + "WHERE text_match('ketamine') GROUP BY d.id")

    result = text_to_sql._run_local("datasets mentioning ketamine", "dummy-key")

    assert result.rank_tsquery is None
    assert "search_tsv" not in result.sql
