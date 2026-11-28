"""Word forms: the catalogue's forms per stem and which of them a search keeps
(backend/vocabulary/stem_collisions.py, services/fulltext/word_form_counts.py)."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
_CRAWLER = ROOT / "ingest" / "openneuro-crawler"
if str(_CRAWLER) not in sys.path:
    sys.path.insert(0, str(_CRAWLER))

import stem_collisions  # noqa: E402
from config.settings import settings_obj  # noqa: E402
from services.fulltext import word_form_counts as store  # noqa: E402
from services.fulltext import vectors  # noqa: E402
from database.worker import WorkerDbError  # noqa: E402
from stem_collisions import StemCounts  # noqa: E402

AGEING = StemCounts("ageing", "age", 291, (
    ("age", 220), ("aged", 44), ("aging", 34), ("ages", 30), ("ageing", 4), ("ageings", 1)))
SCANS = StemCounts("scans", "scan", 502, (
    ("scans", 280), ("scan", 252), ("scanning", 170), ("scanned", 85)))
DEPRESSED = StemCounts("depressed", "depress", 36, (
    ("depression", 33), ("depressive", 11), ("depressed", 4)))


@pytest.fixture
def reviewed_list(tmp_path, monkeypatch):
    """Point word_forms at a list of its own; returns a function that writes it."""
    path = tmp_path / "stem_collisions.yaml"

    def write(content: dict) -> None:
        path.write_text(yaml.safe_dump(content))
        stem_collisions._reviewed.cache_clear()

    monkeypatch.setattr(stem_collisions, "_PATH", path)
    write({})
    yield write
    stem_collisions._reviewed.cache_clear()


# ── Collisions ────────────────────────────────────────────────────────────────

def test_question_words_are_plain_words_without_repeats() -> None:
    assert stem_collisions.question_words("Quantitative MRI datasets on ageing, T1 MRI in 7T") == [
        "quantitative", "mri", "datasets", "ageing"]


def test_question_words_skip_tokens_that_are_not_plain_words_whole() -> None:
    # No fragments: "Sjögren" is not "gren", "MP2RAGE" is not "rage".
    assert stem_collisions.question_words("Sjögren syndrome with MP2RAGE and T1w scans") == [
        "syndrome", "scans"]


def test_question_words_leave_out_stopwords_before_the_cap() -> None:
    filler = " ".join(["which", "about", "between", "through", "these"] * 10)
    assert stem_collisions.question_words(f"{filler} ageing") == ["ageing"]


def test_inflections_cover_plural_and_singular() -> None:
    assert {"weight", "weights"} <= stem_collisions.inflections("weight")
    assert {"study", "studies"} <= stem_collisions.inflections("studies")
    assert "injury" in stem_collisions.inflections("injuries")
    assert "loss" in stem_collisions.inflections("losses")
    assert "clas" not in stem_collisions.inflections("class")


def test_a_stem_collides_when_other_forms_bring_most_datasets() -> None:
    assert stem_collisions.family_docs(AGEING) == 4
    assert stem_collisions.is_collision(AGEING)
    # Plural and singular are the word's own forms: no collision.
    assert stem_collisions.family_docs(SCANS) == 280
    assert not stem_collisions.is_collision(SCANS)
    # Too few extra datasets to matter.
    rare = StemCounts("dystonic", "dyston", 9, (("dystonia", 8), ("dystonic", 1)))
    assert not stem_collisions.is_collision(rare)


# ── Deciding ──────────────────────────────────────────────────────────────────

def test_prompt_block_lists_the_colliding_forms_with_counts(reviewed_list) -> None:
    block = stem_collisions.prompt_block(stem_collisions.needs_decision({"ageing": AGEING, "scans": SCANS}))
    # The pairs to judge, then the word itself; "ageings" is the word's own plural.
    assert "ageing: age (220), aged (44), aging (34), ages (30), ageing (4)\n" in block + "\n"
    assert "scans" not in block
    assert stem_collisions.prompt_block([]) == ""


def test_decide_keeps_chosen_forms_and_their_inflections(reviewed_list) -> None:
    decided = stem_collisions.decide({"ageing": AGEING}, [("ageing", ["ageing", "aging"])])
    # Asked word first, then by count; "ageings" comes with "ageing".
    assert decided == {"ageing": ("ageing", "aging", "ageings")}


def test_decide_is_lenient_with_the_models_answer(reviewed_list) -> None:
    stem_counts = {"ageing": AGEING, "scans": SCANS, "depressed": DEPRESSED}
    decided = stem_collisions.decide(stem_counts, [
        ("Ageing", ["Aging", "elderly", "senescence"]),   # unknown forms dropped, case folded
        ("scans", ["scans"]),                              # not a collision: not asked
        ("mri", ["mri"]),                                  # not in the counts
        ("depressed", ["depressed", "depression", "depressive"]),   # all kept: no restriction
        ("ageing", ["age"]),                               # a second answer is ignored
    ])
    assert decided == {"ageing": ("ageing", "aging", "ageings")}
    assert stem_collisions.decide(stem_counts, []) == {}


def test_reviewed_list_wins_and_covers_inflections(reviewed_list) -> None:
    reviewed_list({
        "exact": [{"forms": ["severe", "severely", "severity"], "excludes": ["several"]}],
        "stemmed": [["depressed", "depression", "depressive"]],
    })
    # Without the catalogue, the forms the group names.
    assert stem_collisions.forms_for("severity") == ("severity", "severe", "severely")
    # The list's pair wins over a model's choice.
    assert stem_collisions.forms_for("severe", {"severe": ("severe", "several")}) == ("severe",)
    # Listed as stemmed: no forms, and the model is not asked.
    assert stem_collisions.forms_for("depressed", {"depressed": ("depressed",)}) is None
    assert stem_collisions.needs_decision({"depressed": DEPRESSED}) == []
    # Not listed: the model's choice, or nothing.
    assert stem_collisions.forms_for("ageing", {"ageing": ("ageing", "aging")}) == ("ageing", "aging")
    assert stem_collisions.forms_for("ageing") is None


def test_forms_are_safe_for_a_tsquery_literal(reviewed_list) -> None:
    reviewed_list({"exact": [{"forms": ["sign", "sign's", "sign | x", "signs"], "excludes": ["signal"]}]})
    assert stem_collisions.forms_for("sign") == ("sign", "signs")
    assert stem_collisions.forms_for("x", {"x": ("x", "y'); DROP")}) == ("x",)


def test_the_shipped_list_is_consistent() -> None:
    stem_collisions._reviewed.cache_clear()
    data = yaml.safe_load((ROOT / "backend" / "vocabulary" / "stem_collisions.yaml").read_text()) or {}
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
    monkeypatch.setattr(settings_obj, vectors.EXACT_ENV, True)
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
    assert found == {"ageing": StemCounts("ageing", "age", 291,
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
    monkeypatch.setattr(settings_obj, vectors.EXACT_ENV, False)
    store.set_lookup(lambda words: pytest.fail("looked up with exact search off"))
    assert store.lookup(["ageing"]) == {}


# ── Schema ────────────────────────────────────────────────────────────────────

def _squash(sql: str) -> str:
    return re.sub(r"\s+", "", sql)


def test_the_exact_vectors_are_the_stemmed_ones_without_stemming() -> None:
    from remote import openneuro_metadata as meta
    from remote import paper_store

    assert meta.EXACT_SEARCH_VECTOR_EXPRESSION == meta.SEARCH_VECTOR_EXPRESSION.replace(
        "'english'", "'simple'")
    assert "'english'" not in meta.EXACT_SEARCH_VECTOR_EXPRESSION
    assert meta.EXACT_SEARCH_VECTOR_EXPRESSION.count("'simple'::regconfig") == 4
    assert paper_store.PAPER_EXACT_SEARCH_VECTOR_EXPRESSION == (
        paper_store.PAPER_SEARCH_VECTOR_EXPRESSION.replace("'english'", "'simple'"))


def test_the_word_forms_are_read_from_the_exact_vector() -> None:
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


def test_word_forms_stem_like_the_search_vector() -> None:
    from remote import openneuro_metadata as meta
    from remote import paper_store
    from services.fulltext.text_search import TS_CONFIG

    def configs(sql: str) -> set:
        return set(re.findall(r"to_ts(?:vector|query)\('(\w+)'", sql))

    vectors = [meta.SEARCH_VECTOR_EXPRESSION, paper_store.PAPER_SEARCH_VECTOR_EXPRESSION]
    for sql in vectors:
        assert configs(sql) == {TS_CONFIG}
    for sql in (store._STEM_SQL, store._REFRESH_SQL, store._LOOKUP_SQL):
        assert configs(sql) == {TS_CONFIG}


# ── Pairs ─────────────────────────────────────────────────────────────────────

WEIGHT = StemCounts("weight", "weight", 1400, (
    ("weighted", 1300), ("weight", 120), ("weights", 40), ("weighting", 30), ("weightings", 12)))


def test_a_form_nobody_judged_keeps_matching(reviewed_list) -> None:
    rare = StemCounts("ageing", "age", 300, AGEING.forms + (("ageless", 3),))
    # "ageless" is too rare to be shown, so nobody judged it: it stays.
    assert "ageless" not in stem_collisions.open_forms(rare)
    decided = stem_collisions.decide({"ageing": rare}, [("ageing", ["ageing", "aging"])])
    assert decided == {"ageing": ("ageing", "aging", "ageings", "ageless")}


def test_a_pair_goes_one_way(reviewed_list) -> None:
    reviewed_list({"exact": [{"forms": ["ageing", "aging"], "excludes": ["age", "ages", "aged"],
                              "why": "x"}]})
    assert stem_collisions.excluded_forms("ageing") == {"age", "ages", "aged"}
    # "aged" against "ageing" was not judged: a search for "aged" stems.
    assert stem_collisions.excluded_forms("aged") == frozenset()
    assert stem_collisions.forms_for("aged") is None


def test_a_reviewed_word_takes_its_stem_minus_the_excluded_forms(reviewed_list) -> None:
    reviewed_list({"exact": [{"forms": ["weight", "weights"], "excludes": ["weighted", "weighting"],
                              "why": "x"}]})
    asked = []

    def catalogue(word):
        asked.append(word)
        return WEIGHT

    assert stem_collisions.forms_for("weight", None, catalogue) == ("weight", "weights", "weightings")
    assert asked == ["weight"]
    # A word without pairs never asks the catalogue.
    assert stem_collisions.forms_for("memory", None, catalogue) is None
    assert asked == ["weight"]
    # The question's words get the same from decide(), with no model answer.
    assert stem_collisions.decide({"weight": WEIGHT}, []) == {"weight": ("weight", "weights", "weightings")}


def test_a_reviewed_word_with_an_open_pair_goes_to_the_model(reviewed_list) -> None:
    reviewed_list({"exact": [{"forms": ["weight", "weights"], "excludes": ["weighted"], "why": "x"}]})
    # "weighted" is judged; "weighting" (30 datasets, against 120) is not.
    assert stem_collisions.open_forms(WEIGHT) == ()
    many = StemCounts("weight", "weight", 1600, WEIGHT.forms[:3] + (("weighting", 200),))
    assert stem_collisions.open_forms(many) == ("weighting",)
    decided = stem_collisions.decide({"weight": many}, [("weight", ["weight", "weights"])])
    assert decided == {"weight": ("weight", "weights")}
    # Listed as stemmed: every pair matches, nothing is open.
    reviewed_list({"stemmed": [["weight", "weighted", "weighting"]]})
    assert stem_collisions.open_forms(many) == ()


def test_excluded_forms_cover_both_spellings(reviewed_list) -> None:
    reviewed_list({"exact": [{"forms": ["organ", "organs"], "excludes": ["organized"], "why": "x"}]})
    assert {"organized", "organised"} <= stem_collisions.excluded_forms("organ")


def test_cached_form_counts_asks_once(word_forms_table) -> None:
    asked = []

    def fake(words):
        asked.append(words)
        return [("weight", "weight", 1400, ["weighted:1300", "weight:120"])]

    store.set_lookup(fake)
    assert store.cached_stem_counts("weight").forms == (("weighted", 1300), ("weight", 120))
    assert store.cached_stem_counts("weight") is not None
    assert asked == [["weight"]]
    # A missing answer is kept too.
    store.set_lookup(lambda words: None)
    assert store.cached_stem_counts("weight") is None
    assert store.cached_stem_counts("weight") is None


def test_the_guard_of_a_reviewed_word_lists_its_stem_minus_the_excluded(
        reviewed_list, word_forms_table, monkeypatch) -> None:
    from services.fulltext import tsquery, vectors

    monkeypatch.setattr(settings_obj, vectors.EXACT_ENV, True)
    reviewed_list({"exact": [{"forms": ["weight", "weights"], "excludes": ["weighted", "weighting"],
                              "why": "x"}]})
    store.set_lookup(lambda words: [("weight", "weight", 1400,
                                     [f"{form}:{n}" for form, n in WEIGHT.forms])])
    assert tsquery.exact_guard((("weight",),)) == "(weight | weights | weightings)"
    # Without the catalogue: the group's forms.
    store.set_lookup(None)
    assert tsquery.exact_guard((("weight",),)) == "(weight | weights)"


# ── Stored pairs (word_form_pairs.json) ───────────────────────────────────────

@pytest.fixture
def stored_pairs(tmp_path, monkeypatch, reviewed_list):
    """Point word_forms at a pairs file of its own; returns a function that writes it."""
    path = tmp_path / "word_form_pairs.json"

    def write(content: dict) -> None:
        path.write_text(json.dumps(content))
        stem_collisions._stored_pairs.cache_clear()

    monkeypatch.setattr(stem_collisions, "_PAIRS_PATH", path)
    write({})
    yield write
    stem_collisions._stored_pairs.cache_clear()


TIMING = StemCounts("timing", "time", 500, (
    ("time", 489), ("times", 168), ("timing", 126), ("timed", 40), ("timer", 12)))


def test_stored_pairs_exclude_and_close_their_forms(stored_pairs) -> None:
    stored_pairs({"different": {"timing": ["time", "times"]}, "stem": {"timing": ["timed"]}})
    assert stem_collisions.excluded_forms("timing") == {"time", "times"}
    # Plural of the asked word finds the same pairs.
    assert stem_collisions.excluded_forms("timings") == {"time", "times"}
    # Judged pairs, either way, are not asked about again.
    assert stem_collisions.open_forms(TIMING) == ()
    assert stem_collisions.forms_for("timing", None, lambda word: TIMING) == ("timing", "timed", "timer")
    # Without the catalogue a stored pair cannot say what else the stem holds: stem.
    assert stem_collisions.forms_for("timing") is None
    assert stem_collisions.decide({"timing": TIMING}, []) == {"timing": ("timing", "timed", "timer")}


def test_the_reviewed_list_wins_over_stored_pairs(stored_pairs, reviewed_list) -> None:
    stored_pairs({"different": {"timing": ["time", "timed"]}})
    reviewed_list({"exact": [{"forms": ["timing", "timed"], "excludes": ["timer"], "why": "x"}]})
    assert stem_collisions.excluded_forms("timing") == {"time", "timer"}
    reviewed_list({"stemmed": [["timing", "time"]]})
    assert stem_collisions.excluded_forms("timing") == frozenset()


def test_a_broken_pairs_file_is_ignored(stored_pairs, caplog) -> None:
    stem_collisions._PAIRS_PATH.write_text("{not json")
    stem_collisions._stored_pairs.cache_clear()
    assert stem_collisions.excluded_forms("timing") == frozenset()
    assert "could not be read" in caplog.text


def test_the_shipped_pairs_are_consistent() -> None:
    stem_collisions._stored_pairs.cache_clear()
    data = json.loads((ROOT / "backend" / "vocabulary" / "word_form_pairs.json").read_text())
    for kind in ("different", "stem"):
        for word, forms in data[kind].items():
            assert re.fullmatch(r"[a-z]+", word) and all(re.fullmatch(r"[a-z]+", f) for f in forms), word
            assert not set(forms) & stem_collisions.own_forms(word), word
    for word in set(data["different"]) & set(data["stem"]):
        assert not set(data["different"][word]) & set(data["stem"][word]), word
