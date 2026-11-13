"""Filter chips read from generated SQL, and removing them (services/query_filters.py)."""

from __future__ import annotations

import asyncio
from uuid import UUID

import sqlglot

from services import query_filters as qf

_HEAD = (
    "SELECT d.id, d.name, COUNT(DISTINCT o.subject) AS subject_count "
    "FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id "
    "AND NOT o.subject IS NULL "
)

# Shapes taken from the recall benchmark (benchmarks/recall/results).
DIAGNOSIS = (
    "(EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
    "AND p2.diagnosis IN ('parkinsons_disease')) OR d.search_tsv @@ "
    "to_tsquery('english', 'parkinson:AB | parkinsons:AB | (paralysis:AB & agitans:AB)'))"
)
MODALITY = (
    "EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
    "AND (o2.datatype = 'functional_mri' OR o2.suffix = 'fmri_bold'))"
)
NO_EEG = (
    "NOT EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
    "AND (o2.datatype = 'electroencephalography' OR o2.suffix = 'eeg'))"
)
TASK = (
    "(EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id AND o2.task IN "
    "('auditory_oddball', 'deviant_detection_task', 'oddball_task')) OR d.search_tsv @@ "
    "to_tsquery('english', 'auditory:AB & oddball:AB'))"
)
AGE = (
    "(EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
    "AND p2.age BETWEEN 10 AND 19) OR (d.age_min <= 19 AND d.age_max >= 10))"
)
ADULTS = (
    "(EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
    "AND p2.age >= 18) OR d.age_max >= 18)"
)
TEXT = "d.search_tsv @@ to_tsquery('english', 'sleep & deprivation')"


def _sql(*conditions: str, having: str = "") -> str:
    sql = _HEAD + "WHERE " + " AND ".join(conditions) + " GROUP BY d.id"
    return sql + (f" HAVING {having}" if having else "")


def _by_id(sql: str) -> dict:
    return {chip.id: chip for chip in qf.describe(sql)}


def test_each_top_level_condition_becomes_one_chip():
    chips = _by_id(_sql(DIAGNOSIS, MODALITY, TASK, AGE, TEXT,
                        having="COUNT(DISTINCT o.subject) >= 40"))

    assert [c.kind for c in chips.values()] == [
        "diagnosis", "modality", "task", "age", "text", "subjects"]

    assert chips["w0"].field == "Diagnosis"
    assert chips["w0"].label == "Parkinson's disease"
    assert chips["w0"].text_words == ["parkinson", "parkinsons", "paralysis agitans"]
    assert "Or the dataset text mentions" in chips["w0"].detail

    assert chips["w1"].label == "fMRI"
    # Task codes expand to many; the words the user typed read better.
    assert chips["w2"].label == "auditory oddball"
    assert "Any of 3 coded tasks" in chips["w2"].detail
    assert chips["w3"].label == "10–19"
    assert chips["w4"].label == "“sleep deprivation”"
    assert chips["h0"].field == "Subjects"
    assert chips["h0"].label == "≥ 40"


def test_negated_condition_and_one_sided_age():
    chips = _by_id(_sql(MODALITY, NO_EEG, ADULTS))
    assert chips["w1"].negated is True
    assert chips["w1"].label == "EEG"
    assert chips["w2"].label == "18+"


def test_query_without_conditions_has_no_chips():
    assert qf.describe(_HEAD + "GROUP BY d.id ORDER BY d.name LIMIT 200") == []
    assert qf.describe("not sql at all") == []
    assert qf.describe(None) == []


def test_remove_drops_only_the_named_conditions():
    sql = _sql(DIAGNOSIS, MODALITY, TEXT, having="COUNT(DISTINCT o.subject) >= 40")

    refined = qf.remove(sql, ["w0", "h0"])
    chips = qf.describe(refined)
    assert [c.kind for c in chips] == ["modality", "text"]
    assert "HAVING" not in refined.upper()
    assert "parkinsons_disease" not in refined
    sqlglot.parse_one(refined, read="postgres")


def test_remove_keeps_or_conditions_grouped():
    refined = qf.remove(_sql(MODALITY, DIAGNOSIS, TEXT), ["w0"])
    # The diagnosis OR text-match stays one condition, ANDed with the rest.
    assert [c.kind for c in qf.describe(refined)] == ["diagnosis", "text"]


def test_removing_everything_returns_every_dataset():
    refined = qf.remove(_sql(MODALITY, TEXT), ["w0", "w1"])
    assert "WHERE" not in refined.upper()


def test_remove_with_nothing_or_unknown_ids_keeps_the_query():
    sql = _sql(MODALITY, TEXT)
    assert qf.remove(sql, []) == sql
    assert [c.kind for c in qf.describe(qf.remove(sql, ["w9"]))] == ["modality", "text"]


def test_tier_still_applies_after_a_chip_is_removed(monkeypatch):
    from services import uncertain_tier

    refined = qf.remove(_sql(MODALITY, ADULTS, TEXT), ["w2"])
    tiered = uncertain_tier.apply(refined, papers=False)
    assert uncertain_tier.LEVEL_COLUMN in tiered
    sqlglot.parse_one(tiered, read="postgres")


# ── /query/refine ─────────────────────────────────────────────────────────────

class _Result:
    def __init__(self, scalar=None):
        self._scalar = scalar

    def scalar(self):
        return self._scalar

    def mappings(self):
        return self

    def all(self):
        return []


class _Session:
    def __init__(self):
        self.statements = []

    async def execute(self, statement, params=None):
        self.statements.append(str(statement))
        return _Result(scalar=3)

    async def rollback(self):
        pass


def test_refine_reruns_the_cached_query_without_the_removed_conditions(monkeypatch):
    import routers.query as query_router
    from schemas import QueryRefineRequest
    from services.query_cache import CachedQuery, query_cache

    async def no_rows(session, raw):
        return []

    monkeypatch.setattr(query_router, "_rows_to_datasets", no_rows)
    monkeypatch.delenv("UNCERTAIN_TIER", raising=False)

    def llm_must_not_run(*args, **kwargs):
        raise AssertionError("refine must not call a language model")

    monkeypatch.setattr(query_router, "_correct_sql_with_gemini", llm_must_not_run)
    monkeypatch.setattr(query_router, "rerank_candidates", llm_must_not_run)

    sql = _sql(DIAGNOSIS, MODALITY, TEXT)
    original_id = query_cache.put(CachedQuery(
        base_sql=sql, params={}, scored_filters=[], apply_relevance=False,
        filter_sql=sql, rerank_done=True,
        rerank_ordered=[str(UUID(int=1))],
    ))

    session = _Session()
    response = asyncio.run(query_router.query_refine(
        QueryRefineRequest(query_id=original_id, remove=["w0"]), session, {"sub": "tester"}))

    assert response.total == 3
    assert response.query_id != original_id
    # The chips are the original's, so the page can show the removed one struck out.
    assert [f.id for f in response.filters] == ["w0", "w1", "w2"]

    refined = query_cache.get(response.query_id)
    assert response.translation.sql == refined.base_sql
    assert "parkinsons_disease" not in refined.base_sql
    assert "functional_mri" in refined.base_sql
    assert refined.rerank_ordered == [str(UUID(int=1))]
    # The original stays usable, so chips can be put back.
    assert query_cache.get(original_id).base_sql == sql
    assert all("parkinsons_disease" not in s for s in session.statements)


def test_refine_is_refused_for_self_corrected_queries():
    import pytest
    from fastapi import HTTPException

    import routers.query as query_router
    from schemas import QueryRefineRequest
    from services.query_cache import CachedQuery, query_cache

    sql = _sql(MODALITY)
    query_id = query_cache.put(CachedQuery(
        base_sql=sql, params={}, scored_filters=[], apply_relevance=False,
        filter_sql=sql, self_corrected=True,
    ))
    with pytest.raises(HTTPException) as raised:
        asyncio.run(query_router.query_refine(
            QueryRefineRequest(query_id=query_id, remove=["w0"]), _Session(), {"sub": "t"}))
    assert raised.value.status_code == 400


# ── README excerpts ───────────────────────────────────────────────────────────

def test_highlight_query_takes_words_of_text_and_coded_chips():
    query = qf.highlight_tsquery(_sql(DIAGNOSIS, MODALITY, NO_EEG, ADULTS, TEXT))
    words = query.split(" | ")
    assert {"parkinson", "parkinsons", "paralysis", "agitans", "fmri", "sleep", "deprivation"} <= set(words)
    # Negated chips and ages have nothing to mark.
    assert "eeg" not in words and "18" not in words
    assert qf.highlight_tsquery(_HEAD + "GROUP BY d.id") is None
    assert qf.highlight_tsquery(None) is None


def test_clean_snippet_strips_markdown():
    import routers.query as query_router

    raw = ("# Dataset ⟦Parkinson⟧ study\n![logo](img.png) See [the paper](https://doi.org/x) "
           "and https://openneuro.org.\n| col | col |\n**Bold** text with sub_01 and __init__")
    assert query_router.clean_snippet(raw) == (
        "Dataset ⟦Parkinson⟧ study See the paper and col col Bold text with sub_01 and init")


class _SnippetSession:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def begin_nested(self):
        class _Nested:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *exc):
                return False
        return _Nested()

    async def execute(self, statement, params=None):
        self.calls.append((str(statement), params))
        return self.rows


def _dataset(n: int):
    from schemas import DatasetSchema
    return DatasetSchema(id=UUID(int=n), name=f"ds{n}", dataset_type="raw",
                         source_type="openneuro", participants=[])


def test_attach_snippets_marks_matches_and_falls_back_to_the_start():
    import routers.query as query_router
    from services import schema_capabilities as caps

    long_start = "Plain README start. " * 30
    session = _SnippetSession([
        (UUID(int=1), "patients with ⟦Parkinson's⟧ disease … scanned with ⟦fMRI⟧"),
        (UUID(int=2), long_start),
    ])
    datasets = [_dataset(1), _dataset(2), _dataset(3)]
    caps.set_display_columns({"readme_text"})
    try:
        asyncio.run(query_router._attach_snippets(session, datasets, "parkinson | fmri"))
    finally:
        caps.set_display_columns(())

    sql, params = session.calls[0]
    assert "ts_headline" in sql and params["q"] == "parkinson | fmri"
    # A passage from mid-sentence is marked as an excerpt at both ends.
    assert datasets[0].snippet == "… patients with ⟦Parkinson's⟧ disease … scanned with ⟦fMRI⟧ …"
    # No match in the README: ts_headline returns its start, cut at a word.
    assert datasets[1].snippet.endswith(" …") and len(datasets[1].snippet) <= 330
    assert "⟦" not in datasets[1].snippet
    assert datasets[2].snippet is None


def test_attach_snippets_without_the_column_does_nothing():
    import routers.query as query_router

    session = _SnippetSession([])
    datasets = [_dataset(1)]
    asyncio.run(query_router._attach_snippets(session, datasets, "x"))
    assert session.calls == [] and datasets[0].snippet is None
