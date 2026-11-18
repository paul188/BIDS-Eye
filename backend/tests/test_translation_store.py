"""Questions asked before are answered from the store: exact matches only, and
only while the pipeline, the schema and the catalogue are what they were."""

from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for _p in ("LLM_preprocessor", "RAG"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

import preprocess  # noqa: E402
from services.translate import text_to_sql as t2s  # noqa: E402
from services.stores import translation_store as ts  # noqa: E402

MODEL_SQL = ("SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o "
             "ON o.dataset_id = d.id AND o.subject IS NOT NULL "
             "WHERE d.name ILIKE '%motor%' GROUP BY d.id")


@pytest.fixture(autouse=True)
def _store_on(monkeypatch):
    monkeypatch.setenv("TRANSLATION_STORE", "true")
    monkeypatch.delenv("TRANSLATION_STORE_TTL_DAYS", raising=False)
    ts.store.clear()
    ts.set_data_stamp("datasets=100")
    yield
    ts.store.clear()
    ts.set_data_stamp(None)


def _plan(question: str = "motor datasets") -> preprocess.AugmentedQueryPlan:
    return preprocess.AugmentedQueryPlan(
        plan=preprocess.QueryPlan(query_family="metadata_query",
                                  natural_language_summary="Datasets about motor tasks."),
        augmented_question=f"Question: {question}",
    )


class _Pipeline:
    """Stands in for the two model stages and counts how often each was asked."""

    def __init__(self, monkeypatch, compiled_sql=None):
        self.plans = self.generations = 0
        monkeypatch.setattr(preprocess, "run_pipeline", self._run_pipeline)
        monkeypatch.setattr(t2s, "_gemini_sql_generation", self._generate)
        monkeypatch.setattr(
            t2s, "_compile_plan",
            lambda *a, **k: type("C", (), {"sql": compiled_sql, "reason": "test"})())

    def _run_pipeline(self, question, **kwargs):
        self.plans += 1
        return _plan(question)

    def _generate(self, *args, **kwargs):
        self.generations += 1
        return MODEL_SQL


def _translate_and_keep(question: str):
    result = t2s._run_local(question, "key")
    if result.store_entry is not None:
        ts.store.put(result.store_entry)
    return result


# ── the key ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a, b", [
    ("Datasets with fMRI", "datasets  with fmri"),
    ("datasets with fMRI?", "datasets with fMRI"),
    ("  Parkinson’s patients ", "parkinson's patients"),
    ("datasets with fMRI.", "datasets with fMRI !"),
])
def test_same_question(a, b):
    assert ts.normalize_question(a) == ts.normalize_question(b)


@pytest.mark.parametrize("a, b", [
    ("datasets with fMRI", "datasets without fMRI"),
    ("at least 40 patients", "at most 40 patients"),
    ("at least 40 patients", "at least 50 patients"),
    ("datasets with fMRI", "dataset with fMRI"),
    ("fMRI and EEG", "fMRI or EEG"),
    ("no controls", "controls"),
])
def test_different_questions_never_share_a_key(a, b):
    assert ts.question_key(a) != ts.question_key(b)


# ── hits and misses ─────────────────────────────────────────────────────────────

def test_second_time_no_model_is_asked_and_the_result_is_the_same(monkeypatch):
    pipeline = _Pipeline(monkeypatch)

    first = _translate_and_keep("Motor datasets")
    second = _translate_and_keep("motor datasets?")

    assert (pipeline.plans, pipeline.generations) == (1, 1)
    assert not first.from_store and second.from_store
    assert second.store_entry is None
    same = ("sql", "params", "explanation", "scored_filters", "apply_relevance",
            "rank_tsquery", "filter_sql")
    assert {k: getattr(second, k) for k in same} == {k: getattr(first, k) for k in same}


def test_a_compiled_plan_is_stored_without_model_sql(monkeypatch):
    pipeline = _Pipeline(monkeypatch, compiled_sql=MODEL_SQL)

    first = _translate_and_keep("motor datasets")
    second = _translate_and_keep("motor datasets")

    assert first.store_entry.model_sql is None
    assert (pipeline.plans, pipeline.generations) == (1, 0)
    assert second.from_store and second.sql == first.sql


def test_another_question_is_translated_anew(monkeypatch):
    pipeline = _Pipeline(monkeypatch)
    _translate_and_keep("motor datasets")
    result = _translate_and_keep("motor datasets without fMRI")
    assert pipeline.plans == 2 and not result.from_store


def test_switched_off_nothing_is_stored_or_read(monkeypatch):
    pipeline = _Pipeline(monkeypatch)
    monkeypatch.setenv("TRANSLATION_STORE", "false")
    first = _translate_and_keep("motor datasets")
    assert first.store_entry is None
    monkeypatch.setenv("TRANSLATION_STORE", "true")
    ts.store.put(ts.candidate("motor datasets", _plan(), MODEL_SQL))
    monkeypatch.setenv("TRANSLATION_STORE", "false")
    _translate_and_keep("motor datasets")
    assert pipeline.plans == 2


@pytest.mark.parametrize("change", ["pipeline", "schema", "catalogue", "age"])
def test_an_entry_is_not_used_after_what_it_depends_on_changed(monkeypatch, change):
    pipeline = _Pipeline(monkeypatch)
    _translate_and_keep("motor datasets")

    if change == "pipeline":
        monkeypatch.setenv("GEMINI_MODEL_CASCADE", "some-other-model")
    elif change == "schema":
        monkeypatch.setattr(ts, "schema_fingerprint", lambda: "other")
    elif change == "catalogue":
        ts.set_data_stamp("datasets=101")
    else:
        entry = ts.store._entries[ts.question_key("motor datasets")]
        entry.created_at -= timedelta(days=8)

    result = _translate_and_keep("motor datasets")
    assert pipeline.plans == 2 and not result.from_store
    # The new translation replaced the stale entry.
    assert _translate_and_keep("motor datasets").from_store and pipeline.plans == 2


def test_a_verified_entry_does_not_expire(monkeypatch):
    entry = ts.candidate("motor datasets", _plan(), MODEL_SQL)
    entry.status = ts.VERIFIED
    entry.created_at -= timedelta(days=400)
    ts.store.put(entry)
    assert ts.lookup("motor datasets") is entry


def test_pipeline_fingerprint_follows_the_files(monkeypatch, tmp_path):
    (tmp_path / "RAG").mkdir()
    vocabulary = tmp_path / "RAG" / "value_mappings.yaml"
    vocabulary.write_text("a: 1\n")
    monkeypatch.setattr(ts, "_ROOT", tmp_path)
    monkeypatch.setattr(ts, "_PIPELINE_DIRS", (tmp_path / "RAG",))
    monkeypatch.setattr(ts, "_PIPELINE_FILES", ())

    ts._pipeline_files_digest.cache_clear()
    before = ts.pipeline_fingerprint()
    vocabulary.write_text("a: 2\n")
    ts._pipeline_files_digest.cache_clear()
    after = ts.pipeline_fingerprint()
    ts._pipeline_files_digest.cache_clear()

    assert before != after


def test_an_answer_of_a_fallback_model_is_not_kept(monkeypatch):
    _Pipeline(monkeypatch)
    usage: list = []
    monkeypatch.setattr(t2s, "_start_usage_collection", lambda: usage)
    monkeypatch.setenv("GEMINI_MODEL_CASCADE", "primary,second")
    usage.append({"model": "second", "purpose": "sql"})

    assert t2s._run_local("motor datasets", "key").store_entry is None

    usage[:] = [{"model": "primary", "purpose": "intent"}]
    assert t2s._run_local("motor datasets", "key").store_entry is not None


def test_a_stored_plan_that_cannot_be_read_is_a_miss(monkeypatch):
    pipeline = _Pipeline(monkeypatch)
    entry = ts.candidate("motor datasets", _plan(), MODEL_SQL)
    entry.plan = {"not": "a plan"}
    ts.store.put(entry)
    result = t2s._run_local("motor datasets", "key")
    assert pipeline.plans == 1 and not result.from_store


# ── the router keeps only searches that worked ──────────────────────────────────

class _Session:
    def __init__(self):
        self.merged, self.committed = [], 0

    def begin_nested(self):
        class _Savepoint:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False
        return _Savepoint()

    async def merge(self, row):
        self.merged.append(row)

    def add(self, row):
        pass

    async def commit(self):
        self.committed += 1


def _search(monkeypatch, total: int, self_corrected: bool = False):
    import routers.query as router
    from services import search, search_runner
    from schemas import QueryRequest, TextToSQLResult

    entry = ts.candidate("motor datasets", _plan(), MODEL_SQL)

    async def nothing(*args, **kwargs):
        return None

    async def translate(question):
        return TextToSQLResult(sql="SELECT 1", explanation="", store_entry=entry)

    async def page(session, query_id, cached, *args, **kwargs):
        cached.self_corrected = self_corrected
        return [], total

    monkeypatch.setattr(ts, "refresh", nothing)
    monkeypatch.setattr(ts, "count_hit", nothing)
    monkeypatch.setattr(search, "text_to_sql", translate)
    monkeypatch.setattr(search_runner, "run_page", page)
    monkeypatch.setattr(search, "_filters_for", lambda cached: [])
    session = _Session()
    asyncio.run(router.query_datasets(
        QueryRequest(question="motor datasets"), session, {"sub": "login"}))
    return session


def test_a_search_that_worked_is_kept_and_written(monkeypatch):
    session = _search(monkeypatch, total=12)
    assert ts.lookup("motor datasets") is not None
    assert [row.question_norm for row in session.merged] == ["motor datasets"]
    assert session.merged[0].model_sql == MODEL_SQL


@pytest.mark.parametrize("total, self_corrected", [(0, False), (12, True)])
def test_an_empty_or_corrected_search_is_not_kept(monkeypatch, total, self_corrected):
    session = _search(monkeypatch, total=total, self_corrected=self_corrected)
    assert ts.lookup("motor datasets") is None
    assert not session.merged


# ── review status ───────────────────────────────────────────────────────────────

def _entry(status: str, question: str = "motor datasets"):
    entry = ts.candidate(question, _plan(question), MODEL_SQL)
    entry.status = status
    return entry


def test_a_verified_entry_survives_a_new_prompt_and_a_crawl(monkeypatch):
    ts.store.put(_entry(ts.VERIFIED))
    monkeypatch.setenv("GEMINI_MODEL_CASCADE", "some-other-model")
    ts.set_data_stamp("datasets=101")
    assert ts.lookup("motor datasets") is not None


@pytest.mark.parametrize("fingerprint", ["schema_fingerprint", "vocabulary_fingerprint"])
def test_a_verified_entry_is_not_used_after_schema_or_vocabulary_changed(monkeypatch, fingerprint):
    ts.store.put(_entry(ts.VERIFIED))
    monkeypatch.setattr(ts, fingerprint, lambda: "other")
    assert ts.lookup("motor datasets") is None


@pytest.mark.parametrize("status", [ts.REJECTED, ts.NEEDS_REVIEW])
def test_rejected_and_waiting_entries_are_not_used(status):
    ts.store.put(_entry(status))
    assert ts.lookup("motor datasets") is None


@pytest.mark.parametrize("status", [ts.VERIFIED, ts.REJECTED, ts.NEEDS_REVIEW])
def test_a_search_never_replaces_a_reviewed_entry(status):
    reviewed = _entry(status)
    ts.store.put(reviewed)
    session = _Session()
    asyncio.run(ts.remember(session, _entry(ts.UNREVIEWED)))
    assert ts.store._entries[reviewed.key] is reviewed
    assert not session.merged


def test_the_sql_of_a_verified_entry_is_used_even_when_the_plan_compiles(monkeypatch):
    other_sql = MODEL_SQL.replace("motor", "compiled")
    pipeline = _Pipeline(monkeypatch, compiled_sql=other_sql)
    ts.store.put(_entry(ts.VERIFIED))
    verified = t2s._run_local("motor datasets", "key")
    assert "motor" in verified.sql and "compiled" not in verified.sql
    assert verified.from_store and pipeline.plans == 0

    ts.store.clear()
    ts.store.put(_entry(ts.UNREVIEWED))
    assert "compiled" in t2s._run_local("motor datasets", "key").sql


def test_the_router_records_what_the_reviewer_needs(monkeypatch):
    session = _search(monkeypatch, total=12)
    row = session.merged[0]
    assert (row.status, row.result_total, row.final_sql) == (ts.UNREVIEWED, 12, "SELECT 1")


# ── the table: marking for review, and reviewing ────────────────────────────────

class _Connection:
    """A synchronous SQLite connection behind the async calls the store makes."""

    def __init__(self):
        from sqlalchemy import create_engine
        from models import QueryTranslation
        self.engine = create_engine("sqlite://")
        QueryTranslation.__table__.create(self.engine)
        self.raw = self.engine.connect()

    async def execute(self, statement, *args):
        return self.raw.execute(statement, *args)

    def insert(self, entry, **extra):
        from models import QueryTranslation
        self.raw.execute(QueryTranslation.__table__.insert().values(
            question_key=entry.key, question_norm=entry.question_norm, status=entry.status,
            plan=entry.plan, model_sql=entry.model_sql, pipeline_fp=entry.pipeline_fp,
            vocabulary_fp=entry.vocabulary_fp, schema_fp=entry.schema_fp,
            data_stamp=entry.data_stamp, created_at=entry.created_at, hits=0, **extra))

    def status(self, entry):
        return asyncio.run(ts.find(self, entry.key[:12])).status


def test_a_schema_change_marks_verified_entries_for_review_and_only_those(monkeypatch):
    connection = _Connection()
    verified, unreviewed = _entry(ts.VERIFIED, "a"), _entry(ts.UNREVIEWED, "b")
    rejected = _entry(ts.REJECTED, "c")
    for entry in (verified, unreviewed, rejected):
        connection.insert(entry)

    assert asyncio.run(ts.mark_for_review(connection)) == 0

    monkeypatch.setattr(ts, "schema_fingerprint", lambda: "new schema")
    assert asyncio.run(ts.mark_for_review(connection)) == 1
    row = asyncio.run(ts.find(connection, verified.key[:12]))
    assert (row.status, row.review_reason) == (ts.NEEDS_REVIEW, "schema changed")
    assert connection.status(unreviewed) == ts.UNREVIEWED
    assert connection.status(rejected) == ts.REJECTED

    # Read back, the marked entry is not used until it is approved again ...
    ts.store.replace_all(asyncio.run(ts._read_entries(connection)))
    assert ts.lookup("a") is None
    # ... and approving ties it to the schema of now.
    asyncio.run(ts.review(connection, verified.key[:12], ts.VERIFIED, "paul"))
    ts.store.replace_all(asyncio.run(ts._read_entries(connection)))
    assert ts.lookup("a") is not None
    row = asyncio.run(ts.find(connection, verified.key[:12]))
    assert (row.reviewed_by, row.review_reason, row.schema_fp) == ("paul", None, "new schema")


def test_a_review_can_correct_the_sql_or_reject(monkeypatch):
    connection = _Connection()
    first, second = _entry(ts.UNREVIEWED, "a"), _entry(ts.UNREVIEWED, "b")
    connection.insert(first)
    connection.insert(second)

    asyncio.run(ts.review(connection, first.key[:12], ts.VERIFIED, "paul", model_sql="SELECT 2"))
    asyncio.run(ts.review(connection, second.key[:12], ts.REJECTED, "paul"))

    row = asyncio.run(ts.find(connection, first.key[:12]))
    assert (row.status, row.model_sql) == (ts.VERIFIED, "SELECT 2")
    assert connection.status(second) == ts.REJECTED
    with pytest.raises(LookupError):
        asyncio.run(ts.find(connection, "no-such-key"))
    with pytest.raises(ValueError):
        asyncio.run(ts.review(connection, first.key[:12], ts.UNREVIEWED, "paul"))


def test_the_review_list_puts_the_most_used_first():
    connection = _Connection()
    rare, common = _entry(ts.UNREVIEWED, "rare"), _entry(ts.UNREVIEWED, "common")
    connection.insert(rare)
    connection.insert(common)
    asyncio.run(ts.count_hit(_SessionOver(connection), "common"))
    rows = asyncio.run(ts.list_for_review(connection, ts.UNREVIEWED))
    assert [(r.question_norm, r.hits) for r in rows] == [("common", 1), ("rare", 0)]


class _SessionOver(_Session):
    def __init__(self, connection):
        super().__init__()
        self.connection = connection

    async def execute(self, statement, *args):
        return await self.connection.execute(statement, *args)


# ── capture file and import ─────────────────────────────────────────────────────

def test_every_new_translation_is_captured_also_with_the_store_off(monkeypatch, tmp_path):
    captured = tmp_path / "captured.jsonl"
    monkeypatch.setenv("TRANSLATION_CAPTURE_FILE", str(captured))
    monkeypatch.setenv("TRANSLATION_STORE", "false")
    _Pipeline(monkeypatch)

    t2s._run_local("Motor datasets", "key")
    t2s._run_local("EEG datasets", "key")

    items = ts.read_translations(captured)
    assert all(i["code_commit"] == ts.code_commit() for i in items)
    assert [i["question"] for i in items] == ["Motor datasets", "EEG datasets"]
    assert items[0]["model_sql"] == MODEL_SQL
    preprocess.AugmentedQueryPlan.model_validate(items[0]["plan"])


def test_a_stored_answer_and_a_fallback_model_answer_are_not_captured(monkeypatch, tmp_path):
    captured = tmp_path / "captured.jsonl"
    monkeypatch.setenv("TRANSLATION_CAPTURE_FILE", str(captured))
    _Pipeline(monkeypatch)
    _translate_and_keep("motor datasets")
    _translate_and_keep("motor datasets")          # from the store
    assert len(captured.read_text().splitlines()) == 1

    monkeypatch.setenv("GEMINI_MODEL_CASCADE", "primary,second")
    monkeypatch.setattr(t2s, "_start_usage_collection", lambda: [{"model": "second"}])
    t2s._run_local("EEG datasets", "key")
    assert len(captured.read_text().splitlines()) == 1


def test_a_recording_of_record_run_is_read_the_same_way(tmp_path):
    import json
    plan = _plan().model_dump(mode="json", exclude={"topics"})
    recording = tmp_path / "recording.json"
    recording.write_text(json.dumps({
        "motor datasets": {**plan, "model_sql": MODEL_SQL, "usage": [{"model": "m"}],
                           "code_commit": "abc1234"},
        "EEG datasets": {**plan},
    }))
    items = ts.read_translations(recording)
    assert [(i["question"], i["model_sql"], i["code_commit"]) for i in items] == [
        ("motor datasets", MODEL_SQL, "abc1234"), ("EEG datasets", None, None)]
    assert "usage" not in items[0]["plan"] and "model_sql" not in items[0]["plan"]
    preprocess.AugmentedQueryPlan.model_validate(items[0]["plan"])


def test_import_stores_what_runs_and_finds_something_and_spares_reviewed_entries():
    from sqlalchemy import text
    connection = _Connection()
    connection.raw.execute(text("CREATE TABLE bids_datasets (id INTEGER)"))
    reviewed = _entry(ts.VERIFIED, "reviewed question")
    old = _entry(ts.UNREVIEWED, "asked before")
    connection.insert(reviewed)
    connection.insert(old, result_total=3)
    plan = _plan().model_dump(mode="json", exclude={"topics"})
    items = [{"question": q, "plan": plan, "model_sql": MODEL_SQL} for q in (
        "good one", "finds nothing", "sql is broken", "reviewed question", "Asked before?")]

    async def validate(question, plan, model_sql):
        if question == "sql is broken":
            raise RuntimeError("column d.nope does not exist")
        return "SELECT final", 0 if question == "finds nothing" else 7

    outcome = asyncio.run(ts.import_translations(connection, items, validate))

    assert outcome["stored"] == ["good one", "Asked before?"]
    assert outcome["empty"] == ["finds nothing"]
    assert outcome["kept_reviewed"] == ["reviewed question"]
    assert outcome["failed"][0].startswith("sql is broken  [column d.nope")
    rows = {r.question_norm: r for r in asyncio.run(ts.list_for_review(connection))}
    assert set(rows) == {"reviewed question", "asked before", "good one"}
    assert (rows["good one"].status, rows["good one"].final_sql, rows["good one"].result_total) == (
        ts.UNREVIEWED, "SELECT final", 7)
    assert rows["asked before"].result_total == 7
    assert rows["reviewed question"].status == ts.VERIFIED
    # What was imported is usable at once.
    ts.store.replace_all(asyncio.run(ts._read_entries(connection)))
    ts.set_data_stamp(rows["good one"].data_stamp)
    assert ts.lookup("Good one") is not None


# ── the commit of the code that made an entry ───────────────────────────────────

def test_the_commit_comes_from_the_deploy_then_git(monkeypatch, tmp_path):
    monkeypatch.setattr(ts, "_ROOT", tmp_path)
    monkeypatch.delenv("BIDSEYE_COMMIT", raising=False)
    ts.code_commit.cache_clear()
    assert ts.code_commit() == "unknown"                    # no file, not a checkout

    (tmp_path / "DEPLOYED_COMMIT").write_text("87fe942\n")
    ts.code_commit.cache_clear()
    assert ts.code_commit() == "87fe942"

    monkeypatch.setenv("BIDSEYE_COMMIT", "abc1234-dirty")
    ts.code_commit.cache_clear()
    assert ts.code_commit() == "abc1234-dirty"
    ts.code_commit.cache_clear()


def test_every_stored_entry_names_its_commit(monkeypatch):
    monkeypatch.setenv("BIDSEYE_COMMIT", "abc1234")
    ts.code_commit.cache_clear()
    try:
        session = _search(monkeypatch, total=12)
        assert session.merged[0].code_commit == "abc1234"

        from sqlalchemy import text
        connection = _Connection()
        connection.raw.execute(text("CREATE TABLE bids_datasets (id INTEGER)"))
        old = _entry(ts.VERIFIED, "from an older commit")
        connection.insert(old, code_commit="0ld0ld0")
        connection.insert(_entry(ts.UNREVIEWED, "before commits were recorded"))

        async def validate(question, plan, model_sql):
            return "SELECT final", 3
        plan = _plan().model_dump(mode="json", exclude={"topics"})
        # An import keeps the commit its file names; the importing code's
        # commit (abc1234 here) is never put on a translation it did not make.
        asyncio.run(ts.import_translations(connection, [
            {"question": "recorded by fff9999", "plan": plan, "model_sql": None, "code_commit": "fff9999"},
            {"question": "recorded without a commit", "plan": plan, "model_sql": None},
        ], validate))

        by_commit = {(r.code_commit, r.status): r.entries
                     for r in asyncio.run(ts.count_by_commit(connection))}
        assert by_commit == {("fff9999", ts.UNREVIEWED): 1, ("0ld0ld0", ts.VERIFIED): 1,
                             (ts.PRE_TRACKING, ts.UNREVIEWED): 2}
        listed = asyncio.run(ts.list_for_review(connection, commit="0ld"))
        assert [r.question_norm for r in listed] == ["from an older commit"]
        untracked = asyncio.run(ts.list_for_review(connection, commit=ts.PRE_TRACKING))
        assert sorted(r.question_norm for r in untracked) == [
            "before commits were recorded", "recorded without a commit"]
    finally:
        ts.code_commit.cache_clear()
