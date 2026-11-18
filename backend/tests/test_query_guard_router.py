"""The search endpoints run generated SQL only behind services/query_guard.py.

Needs the backend image (routers.query imports the db package)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

import routers.query as query_router
from services import dataset_cards, query_guard, readme_snippets, search_runner
from services.query_cache import CachedQuery, query_cache
from services.sql_rewriter import FALLBACK_SQL

_SAFE = ("SELECT d.id FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id "
         "AND NOT o.subject IS NULL GROUP BY d.id")
_EXPLODING = ("SELECT d.id FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id "
              "LEFT JOIN bids_participants AS p ON p.dataset_id = d.id WHERE p.age <= 12 GROUP BY d.id")


class _Result:
    def __init__(self, value=None, rows=()):
        self.value, self.rows = value, list(rows)

    def scalar(self):
        return self.value

    def mappings(self):
        return self

    def all(self):
        return self.rows


class _Savepoint:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    """Answers COUNT with 5 (the fallback with 200); EXPLAIN with *plan_rows*."""

    def __init__(self, plan_rows: float = 100):
        self.plan_rows = plan_rows
        self.statements = []
        self.rollbacks = 0

    async def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append(sql)
        if sql.startswith("EXPLAIN"):
            return _Result([{"Plan": {"Node Type": "Seq Scan", "Plan Rows": self.plan_rows}}])
        if "reltuples" in sql:
            return _Result(2_700_000)
        if sql.startswith("SELECT COUNT(*)"):
            return _Result(200 if "ORDER BY d.name LIMIT 200" in " ".join(sql.split()) else 5)
        return _Result(None, rows=[])

    def begin_nested(self):
        return _Savepoint()

    async def rollback(self):
        self.rollbacks += 1


@pytest.fixture(autouse=True)
def _fresh_guard_state(monkeypatch):
    monkeypatch.setattr(query_guard, "_max_plan_rows", None)
    monkeypatch.setattr(query_guard, "_semaphore", None)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")


def _count(session, sql, corrector):
    cached = CachedQuery(base_sql=sql, params={}, scored_filters=[], apply_relevance=False, explanation="")
    query_id = query_cache.put(cached)
    total = asyncio.run(search_runner.count_with_base_correction(session, query_id, cached, "question"))
    return total, cached


def test_exploding_query_never_reaches_the_database(monkeypatch) -> None:
    seen = []

    def corrector(sql, error, question, api_key):
        seen.append(error)
        return _SAFE

    monkeypatch.setattr(search_runner, "_correct_sql_with_gemini", corrector)
    session = _Session()
    total, cached = _count(session, _EXPLODING, corrector)
    assert total == 5
    assert cached.base_sql == _SAFE and cached.self_corrected
    assert "bids_participants and bids_objects" in seen[0]
    assert not any("bids_participants AS p ON" in s and not s.startswith("EXPLAIN") for s in session.statements)


def test_refused_twice_gives_the_fallback_sample(monkeypatch) -> None:
    monkeypatch.setattr(search_runner, "_correct_sql_with_gemini", lambda sql, error, q, key: _EXPLODING + " ")
    session = _Session()
    total, cached = _count(session, _EXPLODING, None)
    assert cached.base_sql == FALLBACK_SQL
    assert total == 200


def test_plan_explosion_is_refused_too(monkeypatch) -> None:
    errors = []
    monkeypatch.setattr(search_runner, "_correct_sql_with_gemini",
                        lambda sql, error, q, key: errors.append(error) or sql)
    session = _Session(plan_rows=50_000_000)
    total, cached = _count(session, _SAFE, None)
    assert "rows at one step" in errors[0]
    assert cached.base_sql == FALLBACK_SQL


def test_limits_are_set_before_every_count(monkeypatch) -> None:
    session = _Session()
    _count(session, _SAFE, None)
    first_count = next(i for i, s in enumerate(session.statements) if s.startswith("SELECT COUNT(*)"))
    before = session.statements[:first_count]
    assert any(s.startswith("SET LOCAL statement_timeout") for s in before)
    assert any(s.startswith("SET LOCAL temp_file_limit") for s in before)
    assert any(s.startswith("EXPLAIN") for s in before)


def test_page_query_runs_with_limits(monkeypatch) -> None:
    async def no_rows(session, raw):
        return []

    async def nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(dataset_cards, "rows_to_datasets", no_rows)
    monkeypatch.setattr(readme_snippets, "attach_snippets", nothing)
    monkeypatch.setattr(dataset_cards, "attach_descriptions", nothing)
    cached = CachedQuery(base_sql=_SAFE, params={}, scored_filters=[], apply_relevance=False, explanation="",
                         rerank_done=True)
    cached.total = 5
    query_id = query_cache.put(cached)
    session = _Session()
    asyncio.run(search_runner.run_page(session, query_id, cached, "", 1, 20))
    page = next(i for i, s in enumerate(session.statements) if "page_result" in s or "LIMIT" in s)
    assert any(s.startswith("SET LOCAL statement_timeout") for s in session.statements[:page])


def test_busy_search_answers_503(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_MAX_CONCURRENT", "1")
    monkeypatch.setenv("SEARCH_SLOT_WAIT_SECONDS", "0.05")
    cached = CachedQuery(base_sql=_SAFE, params={}, scored_filters=[], apply_relevance=False, explanation="",
                         rerank_done=True)
    query_id = query_cache.put(cached)

    async def scenario():
        async with query_guard.slot():
            with pytest.raises(HTTPException) as caught:
                await query_router.query_page(
                    query_router.QueryPageRequest(query_id=query_id), _Session())
            assert caught.value.status_code == 503

    asyncio.run(scenario())


class QueryCanceledError(Exception):
    pass


class _SlowRankingSession(_Session):
    """The relevance-ranked page (matched_codes) runs into the timeout."""

    async def execute(self, statement, params=None):
        if "matched_codes" in str(statement):
            self.statements.append(str(statement))
            raise QueryCanceledError("canceling statement due to statement timeout")
        return await super().execute(statement, params)


def _ranked_cached():
    cached = CachedQuery(base_sql=_SAFE, params={}, explanation="", apply_relevance=True, rerank_done=True,
                         scored_filters=[{"field": "datatype", "code": "anatomical_mri", "score": 1.0}])
    cached.total = 1200
    return cached, query_cache.put(cached)


def _no_side_queries(monkeypatch):
    async def no_rows(session, raw):
        return []

    async def nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(dataset_cards, "rows_to_datasets", no_rows)
    monkeypatch.setattr(readme_snippets, "attach_snippets", nothing)
    monkeypatch.setattr(dataset_cards, "attach_descriptions", nothing)


def test_slow_ranking_serves_the_unranked_page(monkeypatch) -> None:
    _no_side_queries(monkeypatch)
    cached, query_id = _ranked_cached()
    session = _SlowRankingSession()
    asyncio.run(search_runner.run_page(session, query_id, cached, "", 1, 20))
    ranked = next(i for i, s in enumerate(session.statements) if "matched_codes" in s)
    assert session.statements[ranked - 1] == "SET LOCAL statement_timeout = '25s'"
    assert any("LIMIT" in s and "matched_codes" not in s for s in session.statements[ranked + 1:])
    assert cached.ranking_skipped


def test_later_pages_of_a_slow_query_stay_unranked(monkeypatch) -> None:
    _no_side_queries(monkeypatch)
    cached, query_id = _ranked_cached()
    cached.ranking_skipped = True
    session = _SlowRankingSession()
    asyncio.run(search_runner.run_page(session, query_id, cached, "", 2, 20))
    assert not any("matched_codes" in s for s in session.statements)


def test_fast_ranking_keeps_the_ranked_page_and_restores_the_timeout(monkeypatch) -> None:
    _no_side_queries(monkeypatch)
    cached, query_id = _ranked_cached()
    session = _Session()
    asyncio.run(search_runner.run_page(session, query_id, cached, "", 1, 20))
    ranked = next(i for i, s in enumerate(session.statements) if "matched_codes" in s)
    assert session.statements[ranked + 1].startswith("SET LOCAL statement_timeout = '120s'")
    assert not cached.ranking_skipped


def test_other_page_errors_are_not_hidden(monkeypatch) -> None:
    _no_side_queries(monkeypatch)
    cached, query_id = _ranked_cached()

    class _Broken(_Session):
        async def execute(self, statement, params=None):
            if "matched_codes" in str(statement):
                raise RuntimeError("relation does not exist")
            return await super().execute(statement, params)

    with pytest.raises(RuntimeError):
        asyncio.run(search_runner.run_page(_Broken(), query_id, cached, "", 1, 20))
