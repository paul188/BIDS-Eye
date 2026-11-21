"""A translation without any condition is never served as a search result.

"bonn" and "datasets related to drawings or art" got SQL without a WHERE and
returned all 1,721 datasets (live, 2026-11-17/18).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import preprocess
import retriever
from services.catalogue.catalogue import ColumnCoverage, catalogue
from services.fulltext import word_frequencies
from services.translate import text_to_sql
from services.translate.text_to_sql import keeps_every_dataset

BASE = (
    "SELECT {{COLS}} FROM bids_datasets d "
    "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL "
)


@pytest.fixture(autouse=True)
def _full_text_available():
    catalogue.coverage = ColumnCoverage.full(())
    word_frequencies.set_frequency_lookup(None)
    yield
    catalogue.coverage = ColumnCoverage.full(())
    word_frequencies.set_frequency_lookup(None)


def _fake_pipeline(monkeypatch, question: str, generated_sql: str, **plan_overrides) -> None:
    """Replace the two LLM stages and the resolver; keep the real rewriter."""
    plan = SimpleNamespace(
        query_family=SimpleNamespace(value="description_search"),
        natural_language_summary="Datasets related to Bonn.",
        result_limit=None, order_by=None, groups=[],
    )
    for name, value in plan_overrides.items():
        setattr(plan, name, value)
    augmented = SimpleNamespace(plan=plan, augmented_question=question,
                                scored_filters=[], resolved=[])
    monkeypatch.setattr(preprocess, "run_pipeline", lambda *args, **kwargs: augmented)
    monkeypatch.setattr(retriever, "MetadataRetriever", lambda *args, **kwargs: None)
    monkeypatch.setattr(text_to_sql, "_gemini_sql_generation", lambda *args, **kwargs: generated_sql)


def test_no_condition_becomes_a_keyword_search(monkeypatch) -> None:
    _fake_pipeline(monkeypatch, "bonn", BASE + "GROUP BY d.id")

    result = text_to_sql._run_local("bonn", "dummy-key")

    assert "search_tsv @@" in result.sql
    assert "bonn" in result.rank_query.stemmed
    assert not keeps_every_dataset(result.sql)
    assert result.explanation.startswith("Datasets related to Bonn. No search condition")
    assert "keyword search" in result.explanation


def test_a_ranking_question_may_keep_every_dataset(monkeypatch) -> None:
    sql = BASE + "GROUP BY d.id ORDER BY subject_count DESC LIMIT 10"
    _fake_pipeline(monkeypatch, "the 10 largest datasets", sql, result_limit=10)

    result = text_to_sql._run_local("the 10 largest datasets", "dummy-key")

    assert "search_tsv" not in result.sql
    assert "LIMIT 10" in result.sql.upper()


def test_a_filtered_search_is_left_alone(monkeypatch) -> None:
    _fake_pipeline(monkeypatch, "CC0 datasets", BASE + "WHERE d.license ILIKE '%CC0%' GROUP BY d.id")

    result = text_to_sql._run_local("CC0 datasets", "dummy-key")

    assert "CC0" in result.sql
    assert "keyword search" not in result.explanation


@pytest.mark.parametrize("sql, every", [
    ("SELECT d.id FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id GROUP BY d.id", True),
    ("SELECT d.id FROM bids_datasets d", True),
    ("SELECT d.id FROM bids_datasets d WHERE d.n_subjects > 5", False),
    ("SELECT d.id FROM bids_datasets d GROUP BY d.id HAVING COUNT(*) > 1", False),
    ("SELECT d.id FROM bids_datasets d JOIN bids_participants p ON p.dataset_id = d.id GROUP BY d.id", False),
    ("SELECT d.id FROM bids_datasets d UNION SELECT d.id FROM bids_datasets d", False),
    ("not sql at all (", False),
])
def test_keeps_every_dataset(sql: str, every: bool) -> None:
    assert keeps_every_dataset(sql) is every
