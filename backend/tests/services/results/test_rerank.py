from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from uuid import UUID

import pytest

from llm_client import LLMAllFailedError
from config.settings import settings_obj
from services.results import rerank
from services.database_stats.database_stats import database_stats
from tests.helpers import full_coverage
from services.translate.sql_rewriter import SqlRewriteService, build_page_sql
from services.fulltext.ranking import RankQuery

IDS = [str(UUID(int=i)) for i in range(1, 7)]
BASE = (
    "SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o "
    "ON o.dataset_id = d.id AND o.subject IS NOT NULL GROUP BY d.id"
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(settings_obj, "RERANK_ENABLED", True)
    database_stats.coverage = full_coverage({"readme_text", "dx_status"})
    yield
    database_stats.coverage = full_coverage(())


def test_order_puts_good_matches_first_and_misses_last() -> None:
    scores = {IDS[0]: 1, IDS[1]: 3, IDS[2]: 0, IDS[3]: 3, IDS[4]: 2}
    result = rerank.order_from_scores(IDS, scores)
    assert result.ordered_ids == [IDS[1], IDS[3], IDS[4], IDS[0]]   # ties keep SQL order
    assert result.demoted_ids == [IDS[2]]
    assert IDS[5] not in result.scores                               # not judged → middle


def test_scores_out_of_range_or_malformed_are_ignored() -> None:
    raw = json.dumps({"judgments": [{"n": 1, "score": 3}, {"n": 9, "score": 3},
                                    {"n": 2, "score": 7}, {"n": "x"}, {"n": 3, "score": 0}]})
    assert rerank.parse_scores(raw, 3) == {1: 3, 3: 0}


def test_candidate_description_prefers_the_profile() -> None:
    with_profile = rerank.describe(
        {"name": "Reward in BD", "llm_profile": "EEG study of reward in bipolar disorder.",
         "dx_status": "Bipolar", "readme_excerpt": "raw readme"},
        {"datatypes": ["eeg"], "tasks": ["mid"], "subjects": 60},
    )
    assert "Description: EEG study" in with_profile and "raw readme" not in with_profile
    assert "Modalities: eeg" in with_profile and "Subjects: 60" in with_profile
    without = rerank.describe({"name": "X", "dx_status": "Epilepsy", "readme_excerpt": "We  recorded\niEEG"}, None)
    assert "Diagnosis field: Epilepsy" in without and "README opening: We recorded iEEG" in without


def test_should_rerank_respects_order_requests_fallback_and_switch(monkeypatch) -> None:
    assert rerank.should_rerank(True, 10, False)
    assert not rerank.should_rerank(False, 10, False)     # user asked for an order / top N
    assert not rerank.should_rerank(True, 10, True)       # fallback sample
    assert not rerank.should_rerank(True, 1, False)
    monkeypatch.setattr(settings_obj, "RERANK_ENABLED", False)
    assert not rerank.should_rerank(True, 10, False)
    monkeypatch.setattr(settings_obj, "RERANK_ENABLED", False)
    assert not rerank.should_rerank(True, 10, False)   # off unless switched on


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _Session:
    def __init__(self):
        self.sql = []

    async def execute(self, statement, params=None):
        sql = str(statement)
        self.sql.append(sql)
        if "FROM bids_objects" in sql:
            return _Result([{"dataset_id": UUID(i), "datatypes": ["eeg"], "tasks": [], "subjects": 5}
                            for i in IDS])
        return _Result([{"id": UUID(i), "accession_id": f"ds{n}", "name": f"Dataset {n}",
                         "llm_profile": f"profile {n}", "readme_excerpt": "readme"}
                        for n, i in enumerate(IDS)])

    def begin_nested(self):
        return contextlib.nullcontext()


def test_rerank_judges_in_parallel_batches(monkeypatch) -> None:
    monkeypatch.setattr(settings_obj, "RERANK_BATCH", 4)
    prompts = []

    def fake_judge(question, interpretation, descriptions, cfg):
        prompts.append(descriptions)
        # Score by the dataset number in the profile: higher number, better.
        return {n: (int(re.search(r"profile (\d)", d).group(1)) % 4)
                for n, d in enumerate(descriptions, start=1)}

    monkeypatch.setattr(rerank, "_judge_batch", fake_judge)
    session = _Session()
    result = asyncio.run(rerank.rerank_candidates(session, IDS, "bipolar EEG", "EEG in bipolar"))
    assert [len(p) for p in prompts] == [4, 2]
    # profiles 0..5 → scores 0,1,2,3,0,1
    assert result.ordered_ids == [IDS[3], IDS[2], IDS[1], IDS[5]]
    assert result.demoted_ids == [IDS[0], IDS[4]]
    assert "d.llm_profile" in session.sql[0] and "left(d.readme_text, 700)" in session.sql[0]


def test_rerank_gives_up_quietly_on_errors_and_timeouts(monkeypatch) -> None:
    def broken(*args):
        raise LLMAllFailedError("gemini/x: 503 UNAVAILABLE")

    monkeypatch.setattr(rerank, "_judge_batch", broken)
    assert asyncio.run(rerank.rerank_candidates(_Session(), IDS, "q")) is None

    monkeypatch.setattr(settings_obj, "RERANK_TIMEOUT_SECONDS", 0.2)

    def slow(*args):
        time.sleep(1)
        return {1: 3}

    monkeypatch.setattr(rerank, "_judge_batch", slow)
    started = time.monotonic()
    assert asyncio.run(rerank.rerank_candidates(_Session(), IDS, "q")) is None
    assert time.monotonic() - started < 1.5


def test_page_sql_applies_the_reranked_order_after_sqlglot() -> None:
    base = SqlRewriteService().rewrite_generated_sql(BASE, "q").sql
    sql, params = build_page_sql(base, [], True, 20, 0, rank_query=RankQuery("bipolar"),
                                 rerank_ordered=[IDS[2], IDS[0]], rerank_demoted=[IDS[1]])
    assert params["_rr_top"] == [IDS[2], IDS[0]] and params["_rr_low"] == [IDS[1]]
    assert ":_rr_top" in sql and "%(" not in sql
    assert re.search(r"ORDER BY CASE WHEN page_result.id = ANY", sql)
    assert "page_result.relevance_score + page_result.text_rank DESC" in sql

    # Without relevance columns the order falls back to subject count.
    plain, _ = build_page_sql(base, [], True, 20, 0, rerank_ordered=[IDS[0]])
    assert "relevance_score" not in plain and "subject_count DESC" in plain
    # No re-ranking: no ORDER BY added.
    untouched, params = build_page_sql(base, [], True, 20, 0)
    assert "_rr_top" not in untouched and set(params) == {"_limit", "_offset"}


def test_a_bug_in_judging_a_batch_is_raised(monkeypatch) -> None:
    def broken(*args):
        raise KeyError("score")

    monkeypatch.setattr(rerank, "_judge_batch", broken)
    with pytest.raises(KeyError):
        asyncio.run(rerank.rerank_candidates(_Session(), IDS, "q"))


def test_candidates_that_cannot_be_read_skip_the_reranking_in_a_savepoint() -> None:
    from sqlalchemy.exc import OperationalError

    class _Failing(_Session):
        def __init__(self):
            super().__init__()
            self.savepoints = []

        def begin_nested(self):
            session = self

            class _Savepoint:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    session.savepoints.append("rolled back" if exc_type else "released")
                    return False
            return _Savepoint()

        async def execute(self, statement, params=None):
            raise OperationalError("SELECT", {}, Exception("connection reset"))

    session = _Failing()
    assert asyncio.run(rerank.rerank_candidates(session, IDS, "q")) is None
    assert session.savepoints == ["rolled back"]
