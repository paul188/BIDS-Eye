import ast
from pathlib import Path

import pytest

from services.stores import interpretation_store as store

_BACKEND = Path(store.__file__).resolve().parents[2]
_ROOT = _BACKEND.parent
# Imported by the translation, but they only define the shape of rows and
# responses (see interpretation_store._VERSION_FILES).
_NOT_TRANSLATION = {_BACKEND / "models.py", _BACKEND / "schemas.py"}


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
    monkeypatch.delenv("INTERPRETATION_CACHE", raising=False)
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
    assert {"translate/participant_query_compiler.py", "RAG/criteria.py", "RAG/cohort_count.py",
            "RAG/senses.yaml", "RAG/word_forms.yaml"} <= names
    assert not any("__pycache__" in str(p) for p in store.version_paths())


def test_a_new_vocabulary_file_changes_the_version(monkeypatch, tmp_path):
    rag = tmp_path / "RAG"
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
    from services import search
    from services.translate.sql_rewriter import FALLBACK_SQL
    from services.stores.query_cache import CachedQuery

    cached = CachedQuery(base_sql=FALLBACK_SQL if fallback else "SELECT 1",
                         params={}, scored_filters=[], apply_relevance=False)
    cached.self_corrected = self_corrected
    cached.relaxed = relaxed
    assert search._worth_storing(cached, total) is expected
