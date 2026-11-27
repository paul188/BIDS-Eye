"""Preprocessing's DB back-check answered from the backend's cache of coded values
(backend/llm/preprocess/catalogue.py verify_codes, services/database_stats/database_stats.py)."""

from __future__ import annotations

from preprocess import catalogue

from services.database_stats.database_stats import ValueCounts
from services.database_stats.database_stats import database_stats


class _Retriever:
    """The retriever's own database check; records that it was asked."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, list[str]]] = []

    def db_verify_codes(self, field, codes):
        self.asked.append((field, list(codes)))
        return [], list(codes)


def test_the_cache_answers_when_it_holds_the_field(monkeypatch) -> None:
    monkeypatch.setattr(database_stats, "values", ValueCounts({"diagnosis": {"epilepsy": 7, "healthy_control": 155}}))
    retriever = _Retriever()
    live, miss = catalogue.verify_codes(
        "diagnosis", ["epilepsy", "frontotemporal_dementia", "healthy_control"], retriever)
    assert live == ["epilepsy", "healthy_control"]
    assert miss == ["frontotemporal_dementia"]
    assert retriever.asked == []


def test_the_retriever_answers_for_a_field_the_cache_does_not_hold(monkeypatch) -> None:
    monkeypatch.setattr(database_stats, "values", ValueCounts({"diagnosis": {"epilepsy": 7}}))
    retriever = _Retriever()
    assert catalogue.verify_codes("author", ["smith"], retriever) == ([], ["smith"])
    assert retriever.asked == [("author", ["smith"])]


def test_without_cache_or_retriever_every_code_counts_as_present() -> None:
    # conftest empties the cache before each test.
    assert catalogue.verify_codes("task", ["n_back", "made_up"]) == (["n_back", "made_up"], [])


def test_the_pipeline_reports_a_cached_miss_as_db_miss(monkeypatch) -> None:
    """A task code the vocabulary knows but no dataset has becomes a DB MISS,
    with no retriever database at all (the live API)."""
    import preprocess as p

    monkeypatch.setattr(database_stats, "values", ValueCounts({"task": {"n_back": 4}, "suffix": {}, "datatype": {}}))

    class Retriever:
        def retrieve_for_field_multi_scored(self, field, terms):
            return {t: [("never_recorded_task", 1.0)] for t in terms}

    plan = p.QueryPlan(query_family="concept_query", natural_language_summary="a task",
                       scan_requirements=[], rag_requests=[])
    monkeypatch.setattr(p.pipeline, "preprocess_query", lambda question, api_key=None: plan)
    monkeypatch.setattr(p.pipeline, "build_rag_requests",
                        lambda plan, *args, **kwargs: [p.RAGRequest(field=p.RAGField.TASK,
                                                                    terms=["odd task"])])
    augmented = p.run_pipeline("odd task", retriever=Retriever()).augmented_question
    assert "DB MISS" in augmented and "never_recorded_task" in augmented
