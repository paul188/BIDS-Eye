from __future__ import annotations

import asyncio
from uuid import UUID

from sqlalchemy.exc import ProgrammingError

import routers.search as query_router
from routers.schemas import DatasetSchema, TextToSQLResult
from services.results import dataset_cards
from services import search, search_runner
from services.stores.translated_queries import query_cache


class _FakeMappingsResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeResult:
    def __init__(self, *, scalar_value=None, rows=None):
        self._scalar_value = scalar_value
        self._rows = rows or []

    def scalar(self):
        return self._scalar_value

    def mappings(self):
        return _FakeMappingsResult(self._rows)

    # The page's own queries (README excerpts, deprecations, descriptions): no rows.
    def __iter__(self):
        return iter(())

    def all(self):
        return []


class _FakeSession:
    """Minimal AsyncSession stand-in for the router integration test."""

    def __init__(self):
        self.rollback_count = 0
        self.added = []
        self.execute_calls = 0
        self.count_calls = 0
        self.statements = []

    def begin_nested(self):
        session = self

        class _Savepoint:
            async def __aenter__(self):
                return session

            async def __aexit__(self, *exc):
                return False

        return _Savepoint()

    async def execute(self, statement, params=None):
        self.execute_calls += 1
        sql = str(statement)
        self.statements.append(sql)
        if sql.startswith("SET LOCAL"):
            return _FakeResult()
        if sql.startswith("EXPLAIN"):
            return _FakeResult(scalar_value=[{"Plan": {"Node Type": "Aggregate", "Plan Rows": 1}}])
        if "reltuples" in sql:
            return _FakeResult(scalar_value=2_700_000)
        if "COUNT(*)" in sql:
            self.count_calls += 1
        if "COUNT(*)" in sql and self.count_calls == 1:
            raise ProgrammingError("SELECT COUNT(*)", {}, Exception("synthetic database error"))
        if "COUNT(*)" in sql:
            return _FakeResult(scalar_value=2)
        if "page_result" in sql:
            rows = [
                {
                    "id": UUID("00000000-0000-0000-0000-000000000001"),
                    "name": "Dataset one",
                    "accession_id": "ds000001",
                    "bids_version": "1.8.0",
                    "dataset_type": "raw",
                    "source_type": "openneuro",
                    "remote_url": None,
                    "validation_status": "passed",
                    "authors": ["A. Researcher"],
                    "description_text": "First dataset",
                    "subject_count": 7,
                }
            ]
            return _FakeResult(rows=rows)
        return _FakeResult(rows=[])

    async def rollback(self):
        self.rollback_count += 1

    def add(self, row):
        self.added.append(row)

    async def commit(self):
        pass


async def _fake_rows_to_datasets(session, raw):
    """Return stable DatasetSchema objects without touching the real database."""
    return [
        DatasetSchema(
            id=row["id"],
            name=row["name"],
            accession_id=row["accession_id"],
            bids_version=row["bids_version"],
            dataset_type=row["dataset_type"],
            source_type=row["source_type"],
            remote_url=row["remote_url"],
            validation_status=row["validation_status"],
            authors=row["authors"],
            description_text=row["description_text"],
            subject_count=row["subject_count"],
            participants=[],
        )
        for row in raw
    ]


def test_query_and_page_flow_reuses_cached_base_sql(monkeypatch):
    async def run():
        async def fake_text_to_sql(question):
            return TextToSQLResult(
                sql="SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL GROUP BY d.id",
                params={},
                explanation="demo",
                scored_filters=[{"field": "task", "code": "resting_state", "score": 0.9}],
                apply_relevance=True,
            )

        monkeypatch.setattr(search, "text_to_sql", fake_text_to_sql)
        monkeypatch.setattr(dataset_cards, "rows_to_datasets", _fake_rows_to_datasets)
        monkeypatch.setattr(
            search_runner,
            "_correct_sql",
            lambda sql, db_error, question: sql.replace("{{COLS}}", "d.id"),
        )
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")

        session = _FakeSession()
        first = await query_router.query_datasets(
            query_router.QueryRequest(question="find datasets"), session, None
        )

        assert first.total == 2
        assert first.query_id is not None
        assert first.translation is not None
        assert first.translation.self_corrected is True
        assert first.translation.sql.startswith("SELECT")
        assert len(first.datasets) == 1

        cached = query_cache.get(first.query_id)
        assert cached is not None
        assert cached.self_corrected is True
        assert "d.id" in cached.base_sql

        second = await query_router.query_page(
            query_router.QueryPageRequest(query_id=first.query_id, page=2, page_size=1),
            session,
        )

        assert second.translation is None
        assert second.total == 2
        assert second.page == 2
        assert second.page_size == 1
        assert second.has_more is False
        assert len(second.datasets) == 1

    asyncio.run(run())


def test_deprecated_reason_is_read_with_its_own_query() -> None:
    dataset_id = UUID("00000000-0000-0000-0000-000000000001")
    other_id = UUID("00000000-0000-0000-0000-000000000002")

    class _Session:
        def __init__(self):
            self.statements = []

        async def execute(self, statement, params=None):
            self.statements.append(str(statement))
            return [(dataset_id, "No longer available on OpenNeuro")]

    async def scenario() -> None:
        session = _Session()
        found = await dataset_cards.deprecations_for(session, [dataset_id, other_id])
        assert found == {dataset_id: "No longer available on OpenNeuro"}
        assert "deprecated_reason IS NOT NULL" in session.statements[0]

    asyncio.run(scenario())
