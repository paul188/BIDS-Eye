"""Word forms: the catalogue's forms per stem and which of them a search keeps
(RAG/word_forms.py, services/fulltext/word_form_counts.py, crawlers/migrate_add_exact_search.sql)."""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
_CRAWLER = ROOT / "crawlers" / "openneuro-crawler"
if str(_CRAWLER) not in sys.path:
    sys.path.insert(0, str(_CRAWLER))

import word_forms  # noqa: E402
from services.fulltext import word_form_counts as store  # noqa: E402
from services.fulltext import vectors  # noqa: E402
from worker_db import WorkerDbError  # noqa: E402
from word_forms import WordEvidence  # noqa: E402

AGEING = WordEvidence("ageing", "age", 291, (
    ("age", 220), ("aged", 44), ("aging", 34), ("ages", 30), ("ageing", 4), ("ageings", 1)))
SCANS = WordEvidence("scans", "scan", 502, (
    ("scans", 280), ("scan", 252), ("scanning", 170), ("scanned", 85)))
DEPRESSED = WordEvidence("depressed", "depress", 36, (
    ("depression", 33), ("depressive", 11), ("depressed", 4)))


@pytest.fixture
def reviewed_list(tmp_path, monkeypatch):
    """Point word_forms at a list of its own; returns a function that writes it."""
    path = tmp_path / "word_forms.yaml"

    def write(content: dict) -> None:
        path.write_text(yaml.safe_dump(content))
        word_forms._reviewed.cache_clear()

    monkeypatch.setattr(word_forms, "_PATH", path)
    write({})
    yield write
    word_forms._reviewed.cache_clear()


# ── Collisions ────────────────────────────────────────────────────────────────

def test_question_words_are_plain_words_without_repeats() -> None:
    assert word_forms.question_words("Quantitative MRI datasets on ageing, T1 MRI in 7T") == [
        "quantitative", "mri", "datasets", "ageing"]


def test_inflections_cover_plural_and_singular() -> None:
    assert {"weight", "weights"} <= word_forms.inflections("weight")
    assert {"study", "studies"} <= word_forms.inflections("studies")
    assert "injury" in word_forms.inflections("injuries")
    assert "loss" in word_forms.inflections("losses")
    assert "clas" not in word_forms.inflections("class")


def test_a_stem_collides_when_other_forms_bring_most_datasets() -> None:
    assert word_forms.family_docs(AGEING) == 4
    assert word_forms.is_collision(AGEING)
    # Plural and singular are the word's own forms: no collision.
    assert word_forms.family_docs(SCANS) == 280
    assert not word_forms.is_collision(SCANS)
    # Too few extra datasets to matter.
    rare = WordEvidence("dystonic", "dyston", 9, (("dystonia", 8), ("dystonic", 1)))
    assert not word_forms.is_collision(rare)


# ── Deciding ──────────────────────────────────────────────────────────────────

def test_prompt_block_lists_forms_with_counts(reviewed_list) -> None:
    block = word_forms.prompt_block(word_forms.needs_decision({"ageing": AGEING, "scans": SCANS}))
    assert "ageing: age (220), aged (44), aging (34), ages (30), ageing (4), ageings (1)" in block
    assert "scans" not in block
    assert word_forms.prompt_block([]) == ""


def test_decide_keeps_chosen_forms_and_their_inflections(reviewed_list) -> None:
    decided = word_forms.decide({"ageing": AGEING}, [("ageing", ["ageing", "aging"])])
    # Asked word first, then by count; "ageings" comes with "ageing".
    assert decided == {"ageing": ("ageing", "aging", "ageings")}


def test_decide_is_lenient_with_the_models_answer(reviewed_list) -> None:
    evidence = {"ageing": AGEING, "scans": SCANS, "depressed": DEPRESSED}
    decided = word_forms.decide(evidence, [
        ("Ageing", ["Aging", "elderly", "senescence"]),   # unknown forms dropped, case folded
        ("scans", ["scans"]),                              # not a collision: not asked
        ("mri", ["mri"]),                                  # not in the evidence
        ("depressed", ["depressed", "depression", "depressive"]),   # all kept: no restriction
        ("ageing", ["age"]),                               # a second answer is ignored
    ])
    assert decided == {"ageing": ("ageing", "aging", "ageings")}
    assert word_forms.decide(evidence, []) == {}


def test_reviewed_list_wins_and_covers_inflections(reviewed_list) -> None:
    reviewed_list({
        "exact": [{"forms": ["severe", "severely", "severity"], "excludes": ["several"]}],
        "stemmed": [["depressed", "depression", "depressive"]],
    })
    assert word_forms.forms_for("severity") == ("severity", "severe", "severely")
    assert word_forms.forms_for("severe", {"severe": ("severe", "several")}) == (
        "severe", "severely", "severity")
    # Listed as stemmed: no forms, and the model is not asked.
    assert word_forms.forms_for("depressed", {"depressed": ("depressed",)}) is None
    assert word_forms.needs_decision({"depressed": DEPRESSED}) == []
    # Not listed: the model's choice, or nothing.
    assert word_forms.forms_for("ageing", {"ageing": ("ageing", "aging")}) == ("ageing", "aging")
    assert word_forms.forms_for("ageing") is None


def test_forms_are_safe_for_a_tsquery_literal(reviewed_list) -> None:
    reviewed_list({"exact": [{"forms": ["sign", "sign's", "sign | x", "signs"]}]})
    assert word_forms.forms_for("sign") == ("sign", "signs")
    assert word_forms.forms_for("x", {"x": ("x", "y'); DROP")}) == ("x",)


def test_the_shipped_list_is_consistent() -> None:
    word_forms._reviewed.cache_clear()
    data = yaml.safe_load((ROOT / "RAG" / "word_forms.yaml").read_text()) or {}
    seen = set()
    for entry in data.get("exact") or []:
        forms = entry["forms"]
        assert len(forms) >= 1 and entry.get("why"), entry
        assert all(re.fullmatch(r"[a-z]+", form) for form in forms), entry
        assert not set(forms) & set(entry.get("excludes") or []), entry
        assert not seen & set(forms), f"listed twice: {seen & set(forms)}"
        seen |= set(forms)
    for group in data.get("stemmed") or []:
        assert all(re.fullmatch(r"[a-z]+", word) for word in group), group
        assert not seen & set(group), f"listed twice: {seen & set(group)}"
        seen |= set(group)


# ── Store ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def word_forms_table(monkeypatch):
    # Words are looked up only while exact search is on (it needs the table).
    monkeypatch.setenv(vectors.EXACT_ENV, "true")
    yield
    store.reset_lookup()


def test_lookup_parses_rows_and_skips_what_cannot_be_a_word(word_forms_table) -> None:
    asked = []

    def fake(words):
        asked.append(words)
        return [("ageing", "age", 291, ["age:220", "aging:34", "ageing:4"])]

    store.set_lookup(fake)
    found = store.lookup(["ageing", "t1w", "ageing", "MRI", "x"])
    assert asked == [["ageing"]]
    assert found == {"ageing": WordEvidence("ageing", "age", 291,
                                            (("age", 220), ("aging", 34), ("ageing", 4)))}


def test_lookup_never_breaks_a_search(word_forms_table, monkeypatch) -> None:
    def broken(words):
        raise WorkerDbError("database gone")

    store.set_lookup(broken)
    assert store.lookup(["ageing"]) == {}
    store.set_lookup(lambda words: None)
    assert store.lookup(["ageing"]) == {}
    store.set_lookup(None)
    assert store.lookup(["ageing"]) == {}
    # With exact search off nothing is asked at all.
    monkeypatch.delenv(vectors.EXACT_ENV)
    store.set_lookup(lambda words: pytest.fail("looked up with exact search off"))
    assert store.lookup(["ageing"]) == {}


# ── Schema ────────────────────────────────────────────────────────────────────

def _squash(sql: str) -> str:
    return re.sub(r"\s+", "", sql)


def test_migration_and_crawler_define_the_same_exact_vectors() -> None:
    from remote import openneuro_metadata as meta
    from remote import paper_store

    migration = _squash((ROOT / "crawlers" / "migrate_add_exact_search.sql").read_text())
    # The exact vectors are the stemmed ones with another configuration, nothing else.
    assert meta.EXACT_SEARCH_VECTOR_EXPRESSION == meta.SEARCH_VECTOR_EXPRESSION.replace(
        "'english'", "'simple'")
    assert "'english'" not in meta.EXACT_SEARCH_VECTOR_EXPRESSION
    assert meta.EXACT_SEARCH_VECTOR_EXPRESSION.count("'simple'::regconfig") == 4
    assert paper_store.PAPER_EXACT_SEARCH_VECTOR_EXPRESSION == (
        paper_store.PAPER_SEARCH_VECTOR_EXPRESSION.replace("'english'", "'simple'"))
    assert _squash(meta.EXACT_SEARCH_VECTOR_EXPRESSION) in migration
    assert _squash(paper_store.PAPER_EXACT_SEARCH_VECTOR_EXPRESSION) in migration

    column, index = meta.exact_search_vector_sql()
    assert "ADD COLUMN IF NOT EXISTS search_tsv_exact tsvector" in column
    assert "CREATE INDEX IF NOT EXISTS idx_bids_datasets_search_tsv_exact " in index
    assert "idx_bids_datasets_search_tsv_exact" in migration
    assert "idx_bids_datasets_search_tsv_exact_papers" in migration


def test_migration_file_matches_the_word_forms_table() -> None:
    sql = (ROOT / "crawlers" / "migrate_add_exact_search.sql").read_text()
    code = " ".join(store.TABLES_SQL)

    def columns(text_: str) -> set:
        body = text_[text_.index("CREATE TABLE IF NOT EXISTS bids_word_forms"):]
        body = body[:body.index(")")]
        return set(re.findall(r"^\s+(\w+)\s+[A-Z]", body, re.M))

    assert columns(code) == columns(sql) == {"form", "stem", "ndoc"}
    for statement in store.TABLES_SQL:
        assert _squash(statement) in _squash(sql)
    assert store.SOURCE_COLUMN in store._REFRESH_SQL and "ts_stat" in store._REFRESH_SQL


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def mappings(self):
        return self

    def one(self):
        return self._rows[0]

    def scalar_one(self):
        return self._rows[0][0]

    def all(self):
        return self._rows

    def scalar(self):
        row = self._rows[0]
        return row[0] if isinstance(row, tuple) else None


def test_a_bug_in_a_word_form_lookup_is_raised(word_forms_table) -> None:
    def buggy(words):
        raise TypeError("a bug")

    store.set_lookup(buggy)
    with pytest.raises(TypeError):
        store.lookup(["ageing"])


class _AutocommitConnection:
    """Every statement is a transaction of its own; records what it ran."""

    def __init__(self):
        self.statements, self._xact = [], 0

    async def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append(sql)
        if "pg_current_xact_id" in sql:
            self._xact += 1
            return _Result([(self._xact,)])
        return _Result([(1,)])


def test_refresh_refuses_a_connection_without_a_transaction() -> None:
    conn = _AutocommitConnection()
    with pytest.raises(RuntimeError, match="needs a transaction"):
        asyncio.run(store.refresh(conn))
    assert not any("DELETE" in sql for sql in conn.statements)
