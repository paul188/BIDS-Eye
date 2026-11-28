import ast
from pathlib import Path

import pytest

from services.stores import interpretations as store
from services.stores.translated_queries import CachedQuery

_BACKEND = Path(store.__file__).resolve().parents[2]
_ROOT = _BACKEND.parent
# Imported by the translation, but they only define the shape of rows and
# responses (see interpretations._VERSION_FILES).
_NOT_TRANSLATION = {_BACKEND / "database" / "app_tables.py", _BACKEND / "routers" / "schemas.py",
                    _BACKEND / "exceptions" / "domain.py",
                    _BACKEND / "exceptions" / "expected.py"}


def test_questions_that_differ_only_in_form_share_a_key():
    assert store.normalize("  Resting   state fMRI? ") == "resting state fmri"
    assert store._key("Resting state fMRI?") == store._key("resting state  fmri")
    assert store._key("resting state fmri") != store._key("resting state eeg")


@pytest.mark.parametrize("a, b", [
    ("Parkinson’s patients", "parkinson's patients"),        # curly apostrophe
    ("children 6–12 years", "children 6-12 years"),          # en dash
    ("resting state fMRI", "resting state fmri"),        # non-breaking space
    ("resting state f​MRI", "resting state fmri"),       # zero-width space
    ("Straße", "STRASSE"),                               # casefold, not lower
])
def test_forms_of_one_question_share_a_key(a, b):
    assert store._key(a) == store._key(b)


@pytest.mark.parametrize("a, b", [
    ("datasets with fMRI", "datasets without fMRI"),
    ("at least 40 patients", "at most 40 patients"),
    ("fMRI and EEG", "fMRI or EEG"),
    ("no controls", "controls"),
])
def test_different_questions_never_share_a_cache_key(a, b):
    assert store._key(a) != store._key(b)


def test_cache_is_off_by_default(monkeypatch):
    monkeypatch.setenv("INTERPRETATION_CACHE", "false")
    assert not store.enabled()
    monkeypatch.setenv("INTERPRETATION_CACHE", "true")
    assert store.enabled()


def _local_module(name: str):
    """The repository file of an imported module name, or None for a library."""
    parts = name.split(".")
    candidates = [_BACKEND.joinpath(*parts).with_suffix(".py")]
    if len(parts) == 1:
        candidates += [directory / f"{name}.py" for directory in store._VERSION_DIRS]
    return next((c for c in candidates if c.is_file()), None)


def _imports(path: Path) -> set:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
            # "from services import text_search" names a module too.
            names |= {f"{node.module}.{alias.name}" for alias in node.names}
    return names


def test_version_covers_every_backend_module_the_translation_imports():
    """A module the translation reaches but the version ignores would let a
    deploy keep serving translations made by the old code."""
    listed = set(store.version_paths())
    todo = [_BACKEND / "services" / "translate" / "text_to_sql.py"] + [
        p for p in store.version_paths() if p.suffix == ".py"]
    reached: set = set()
    while todo:
        path = todo.pop()
        if path in reached:
            continue
        reached.add(path)
        todo.extend(m for name in _imports(path) if (m := _local_module(name)))
    missing = sorted(str(p.relative_to(_ROOT)) for p in reached - listed - _NOT_TRANSLATION)
    assert missing == []


def test_version_reads_the_files_the_old_list_missed():
    names = {f"{p.parent.name}/{p.name}" for p in store.version_paths()}
    assert {"translate/participant_query_compiler.py", "vocabulary/criteria.py",
            "vocabulary/cohort_count.py", "vocabulary/senses.yaml", "vocabulary/stem_collisions.yaml"} <= names
    assert not any("__pycache__" in str(p) for p in store.version_paths())


def test_a_new_vocabulary_file_changes_the_version(monkeypatch, tmp_path):
    rag = tmp_path / "vocabulary"
    rag.mkdir()
    (rag / "value_mappings.yaml").write_text("a: 1\n")
    monkeypatch.setattr(store, "_VERSION_DIRS", (rag,))
    monkeypatch.setattr(store, "_VERSION_FILES", ())

    def version() -> str:
        monkeypatch.setattr(store, "_version", None)
        return store.pipeline_version()

    before = version()
    (rag / "senses.yaml").write_text("b: 2\n")
    with_senses = version()
    assert with_senses != before
    # Neither notes nor bytecode are part of the pipeline.
    (rag / "notes.txt").write_text("not part of the pipeline")
    (rag / "__pycache__").mkdir()
    (rag / "__pycache__" / "senses.cpython-312.pyc").write_bytes(b"\0")
    assert version() == with_senses


@pytest.mark.parametrize("total, fallback, self_corrected, relaxed, expected", [
    (25, False, False, False, True),   # ran as written, real result
    (3, False, False, False, True),
    (2, False, False, False, False),   # nearly empty: the relaxed re-run's case
    (0, False, False, False, False),
    (25, True, False, False, False),   # emergency fallback
    (25, False, True, False, False),   # model-repaired SQL
    (25, False, False, True, False),   # relaxed: depends on that day's counts
])
def test_only_a_translation_that_ran_as_written_is_stored(total, fallback, self_corrected,
                                                          relaxed, expected):
    from services.translate.sql_rewriter import FALLBACK_SQL

    cached = CachedQuery(base_sql=FALLBACK_SQL if fallback else "SELECT 1",
                         params={}, scored_filters=[], apply_relevance=False)
    cached.self_corrected = self_corrected
    cached.relaxed = relaxed
    assert store.worth_storing(cached, total) is expected


def test_version_reads_every_preprocessing_module():
    """preprocess/ is a package: each of its modules decides the translation,
    and the version reads folders without recursing into them."""
    modules = {p for p in (_ROOT / "backend" / "llm").rglob("*.py") if "__pycache__" not in p.parts}
    assert modules and modules <= set(store.version_paths())


class _Rows:
    def __init__(self, row):
        self.row = row

    def scalar_one_or_none(self):
        return self.row


class _StoredRow:
    key = "k"
    review_verdict = None
    payload = '{"sql": "SELECT 1"}'


class _Session:
    """Answers the lookup with *row*; the hit UPDATE raises *update_error*."""

    def __init__(self, row=None, select_error=None, update_error=None):
        self.row, self.select_error, self.update_error = row, select_error, update_error
        self.rollbacks = self.commits = 0

    async def execute(self, statement, params=None):
        if str(statement).lstrip().upper().startswith("UPDATE"):
            if self.update_error:
                raise self.update_error
            return None
        if self.select_error:
            raise self.select_error
        return _Rows(self.row)

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


def _get(session):
    import asyncio
    return asyncio.run(store.get(session, "fMRI datasets"))


def test_a_failed_hit_count_keeps_the_stored_translation(monkeypatch):
    from sqlalchemy.exc import OperationalError

    monkeypatch.setenv("INTERPRETATION_CACHE", "true")
    session = _Session(row=_StoredRow(), update_error=OperationalError("UPDATE", {}, Exception("lock timeout")))
    assert _get(session) == store.StoredTranslation(sql="SELECT 1")
    assert session.rollbacks == 1 and session.commits == 0


def test_a_translation_reviewed_as_wrong_translates_anew(monkeypatch):
    from database.app_tables import ReviewVerdict

    monkeypatch.setenv("INTERPRETATION_CACHE", "true")
    row = _StoredRow()
    row.review_verdict = "wrong"  # as the database returns it
    assert ReviewVerdict.WRONG.value == "wrong"
    assert _get(_Session(row=row)) is None
    row.review_verdict = ReviewVerdict.OK.value
    assert _get(_Session(row=row)) == store.StoredTranslation(sql="SELECT 1")


@pytest.mark.parametrize("payload", [
    "not json",
    '{"params": {}}',                                   # no SQL
    '{"sql": "SELECT 1", "scored_filters": [{"field": "task"}]}',
])
def test_a_stored_payload_that_is_not_a_translation_translates_anew(monkeypatch, payload):
    monkeypatch.setenv("INTERPRETATION_CACHE", "true")
    row = _StoredRow()
    row.payload = payload
    assert _get(_Session(row=row)) is None


def test_a_translation_reads_back_as_it_was_written():
    from services.fulltext.ranking import RankQuery

    written = store.StoredTranslation(
        sql="SELECT 1", params={"n": 3}, apply_relevance=True,
        scored_filters=[{"field": "task", "code": "resting_state", "score": 0.9}],
        rank_query=RankQuery(stemmed="memory", exact="ageing"), sql_path="compiled")
    assert store.StoredTranslation.model_validate_json(written.to_json()) == written


def test_an_unsafe_stored_tsquery_is_dropped():
    read = store.StoredTranslation.model_validate_json(
        '{"sql": "SELECT 1", "rank_query": {"stemmed": "memory", "exact": "x\'; DROP"}}')
    assert read.rank_query.stemmed == "memory" and read.rank_query.exact is None


def test_a_failed_lookup_translates_anew(monkeypatch):
    from sqlalchemy.exc import OperationalError

    monkeypatch.setenv("INTERPRETATION_CACHE", "true")
    session = _Session(select_error=OperationalError("SELECT", {}, Exception("connection reset")))
    assert _get(session) is None
    assert session.rollbacks == 1


def test_a_bug_in_the_lookup_is_raised(monkeypatch):
    monkeypatch.setenv("INTERPRETATION_CACHE", "true")
    with pytest.raises(TypeError):
        _get(_Session(select_error=TypeError("a bug")))


class _WriteSession:
    """Records statements and ORM rows; *commit_error* is raised by commit()."""

    def __init__(self, execute_error=None, commit_error=None):
        self.execute_error, self.commit_error = execute_error, commit_error
        self.statements, self.added = [], []
        self.commits = self.rollbacks = 0

    async def execute(self, statement, params=None):
        if self.execute_error:
            raise self.execute_error
        from sqlalchemy.dialects import postgresql
        self.statements.append(str(statement.compile(dialect=postgresql.dialect())))

    def add(self, row):
        self.added.append(row)

    def begin_nested(self):
        import contextlib
        return contextlib.nullcontext()

    async def commit(self):
        if self.commit_error:
            raise self.commit_error
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


def _put(session, payload=None):
    import asyncio
    cached = CachedQuery(base_sql="SELECT 1", params={}, scored_filters=[], apply_relevance=False)
    asyncio.run(store.put(session, "fMRI datasets",
                          store.StoredTranslation(sql="SELECT 1", **(payload or {})), cached, 25))


def test_a_question_stored_twice_at_once_keeps_the_first(monkeypatch):
    monkeypatch.setenv("INTERPRETATION_CACHE", "true")
    session = _WriteSession()
    _put(session)
    assert "ON CONFLICT (key) DO NOTHING" in session.statements[0]
    assert session.commits == 1


def test_a_failed_store_is_logged_and_the_search_answers(monkeypatch):
    from sqlalchemy.exc import OperationalError

    monkeypatch.setenv("INTERPRETATION_CACHE", "true")
    session = _WriteSession(execute_error=OperationalError("INSERT", {}, Exception("disk full")))
    _put(session)
    assert session.rollbacks == 1


def test_a_payload_json_cannot_encode_is_a_bug(monkeypatch):
    monkeypatch.setenv("INTERPRETATION_CACHE", "true")
    with pytest.raises(TypeError):
        _put(_WriteSession(), payload={"params": {"since": object()}})


def test_a_failed_query_log_commit_rolls_back_and_the_search_answers():
    import asyncio
    from sqlalchemy.exc import OperationalError

    session = _WriteSession(commit_error=OperationalError("COMMIT", {}, Exception("connection reset")))
    asyncio.run(store.log_query(session, "q", store.StoredTranslation(sql="SELECT 1"), 3, from_cache=False,
                                relaxed=False, self_corrected=False, fallback=False))
    assert len(session.added) == 1 and session.rollbacks == 1
