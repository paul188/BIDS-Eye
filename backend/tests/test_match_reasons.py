"""Why a result matched: the venues of each chip (services/results/match_reasons.py)."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
import sqlglot

from criteria import age_condition, species_condition
from schemas import DatasetSchema, ParticipantSummarySchema
from services.fulltext import text_search
from services.results import match_reasons as mr

_HEAD = (
    "SELECT d.id, d.name, COUNT(DISTINCT o.subject) AS subject_count "
    "FROM bids_datasets AS d LEFT JOIN bids_objects AS o ON o.dataset_id = d.id "
    "AND NOT o.subject IS NULL "
)
DIAGNOSIS = (
    "(EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
    "AND p2.diagnosis IN ('parkinsons_disease')) OR d.search_tsv @@ "
    "to_tsquery('english', 'parkinson:AB | parkinsons:AB'))"
)
MODALITY = (
    "EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
    "AND o2.datatype = 'functional_mri')"
)
TASK = (
    "(EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id AND o2.task IN "
    "('auditory_oddball')) OR d.search_tsv @@ to_tsquery('english', 'auditory:AB & oddball:AB'))"
)


def _sql(*conditions: str) -> str:
    sql = _HEAD + "WHERE " + " AND ".join(conditions) + " GROUP BY d.id"
    return text_search.expand_text_search(sql, papers=False).sql


def _plans(*conditions: str):
    alias, plans = mr.plan(_sql(*conditions))
    assert alias == "d"
    return {p.field: p for p in plans}


def _kinds(plan) -> list:
    return [venue.kind for _, venue in plan.branches]


# ── Reading the venues ────────────────────────────────────────────────────────

def test_infant_age_group_has_four_venues():
    plans = _plans(age_condition(["infants"]))
    age = plans["Age"]
    assert _kinds(age) == [mr.PARTICIPANTS, mr.FIELD, mr.FIELD, mr.SHORT_TEXT]
    participants = age.branches[0][1]
    assert participants.participant_rows is not None and participants.age_column is not None
    # Either age bound alone is read as the whole range.
    assert age.branches[1][1].columns[:2] == ("age_min", "age_max")
    # The species guard reads one column only: nothing to tell apart.
    assert "Species" not in plans


def test_diagnosis_reads_the_participant_list_and_the_text():
    diagnosis = _plans(DIAGNOSIS)["Diagnosis"]
    # 'parkinson:AB' reaches the name and description only (weights A, B).
    assert _kinds(diagnosis) == [mr.PARTICIPANTS, mr.SHORT_TEXT]
    assert _kinds(_plans(DIAGNOSIS.replace(":AB", ""))["Diagnosis"]) == [mr.PARTICIPANTS, mr.TEXT]
    assert diagnosis.branches[0][1].reads_diagnosis
    assert diagnosis.branches[0][1].age_column is None


def test_task_reads_files_and_text_species_reads_label_and_text():
    assert _kinds(_plans(TASK)["Task"]) == [mr.FILES, mr.SHORT_TEXT]
    assert _kinds(_plans(species_condition(["mouse"]))["Species"]) == [mr.FIELD, mr.SHORT_TEXT]


def test_single_venue_negated_and_having_chips_are_not_explained():
    # A modality only ever matches files; a negated chip matched by not holding.
    assert _plans(MODALITY, f"NOT {DIAGNOSIS}") == {}
    sql = (_HEAD + "WHERE " + MODALITY + " GROUP BY d.id HAVING "
           "COUNT(DISTINCT o.subject) >= 20")
    assert mr.plan(sql)[1] == []


def test_a_branch_on_the_outer_join_is_left_out():
    # o is the outer query's join: the branch cannot be evaluated per dataset,
    # and what is left is one venue.
    sql = _sql("(o.task = 'rest' OR d.search_tsv @@ to_tsquery('english', 'rest:AB'))")
    assert mr.plan(sql)[1] == []


def test_paper_and_readme_venues():
    paper = ("(d.search_tsv @@ to_tsquery('english', 'sleep') OR "
             "d.search_tsv_papers @@ to_tsquery('english', 'sleep'))")
    assert _kinds(_plans(paper)["Text"]) == [mr.TEXT, mr.PAPER]
    readme = "(d.readme_text ILIKE '%sleep%' OR d.description_text ILIKE '%sleep%')"
    assert _kinds(_plans(readme)["Text"]) == [mr.README, mr.SHORT_TEXT]


@pytest.mark.parametrize(("tsquery", "short"), [
    ("parkinson:AB", True),
    ("(alzheimer:AB & disease:AB) | ad:A", True),
    ("auditory:AB <-> oddball:AB", True),
    ("sleep:AB <2> deprivation:AB", True),
    ("child:*AB", True),
    ("parkinson", False),           # every weight, README included
    ("parkinson:AB | tremor", False),
    ("infant:ABD", False),
    ("", False),
])
def test_short_fields_only(tsquery, short):
    assert mr.short_fields_only(tsquery) is short


def test_a_word_form_guard_does_not_decide_the_text_venue():
    # A search and its exact word forms (services/fulltext/text_search.py):
    # the guard reads 'simple' and has no labels; the search decides.
    guarded = ("((d.search_tsv @@ to_tsquery('english', 'ageing:AB') AND "
               "d.search_tsv_exact @@ to_tsquery('simple', 'ageing | aging')) OR "
               "EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
               "AND p2.age >= 60))")
    plans = mr.plan(_HEAD + "WHERE " + guarded + " GROUP BY d.id")[1]
    assert [v.kind for p in plans for _, v in p.branches] == [mr.SHORT_TEXT, mr.PARTICIPANTS]


def test_participants_in_an_in_subquery_are_not_counted():
    condition = ("(d.id IN (SELECT dataset_id FROM bids_participants WHERE diagnosis = 'adhd') "
                 "OR d.search_tsv @@ to_tsquery('english', 'adhd:AB'))")
    branch = _plans(condition)
    venue = next(iter(branch.values())).branches[0][1]
    assert venue.kind == mr.PARTICIPANTS and venue.participant_rows is None


def test_unreadable_sql_has_no_plan():
    assert mr.plan("not sql") == (None, [])
    assert mr.plan(None) == (None, [])
    assert mr.plan("SELECT 1") == (None, [])


def test_built_query_is_valid_sql_with_counts_and_ages():
    alias, plans = mr.plan(_sql(age_condition(["infants"]), DIAGNOSIS))
    sql = mr.build_sql(alias, plans)
    tree = sqlglot.parse_one(sql.replace(":ids", "'{}'"), read="postgres")
    names = {e.alias_or_name for e in tree.expressions}
    assert {"dataset_id", "h0_0", "n0_0", "lo0_0", "hi0_0", "h1_0", "n1_0",
            "h1_1", "f_age_min", "f_age_max"} <= names
    assert "lo1_0" not in names   # the diagnosis branch reads no age
    assert "MIN(pa.age)" in sql and "COUNT(*)" in sql
    assert sql.endswith("WHERE d.id = ANY(:ids)")


# ── Wording ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("low", "high", "expected"), [
    (0.33, 1.92, "4–23 months"),
    (0.5, 0.5, "6 months"),
    (3, 11.54, "3–11.5 years"),
    (0.5, 2, "0.5–2 years"),
    (25, 25, "25 years"),
])
def test_age_span(low, high, expected):
    assert mr.age_span(low, high) == expected


def _age_plans():
    return mr.plan(_sql(age_condition(["infants"]), DIAGNOSIS, species_condition(["mouse"])))[1]


def test_reason_for_recorded_ages_and_coded_diagnosis():
    row = {"h0_0": True, "n0_0": 12, "lo0_0": 0.33, "hi0_0": 1.92,
           "h1_0": True, "n1_0": 42, "h1_1": True}
    assert mr.reasons_for(_age_plans(), row, total=60) == [
        "Age: 12 of 60 participants aged 4–23 months",
        "Diagnosis: coded for 42 of 60 participants; named in the title or description",
    ]


def test_reason_without_a_participant_total():
    row = {"h1_0": True, "n1_0": 42}
    assert mr.reasons_for(_age_plans(), row, total=None) == ["Diagnosis: coded for 42 participants"]
    # A total below the count (an inconsistent summary) is not shown.
    assert mr.reasons_for(_age_plans(), row, total=10) == ["Diagnosis: coded for 42 participants"]


def test_reason_for_the_dataset_age_range_appears_once():
    row = {"h0_1": True, "h0_2": True, "f_age_min": "0.5", "f_age_max": "2"}
    assert mr.reasons_for(_age_plans(), row, total=None) == ["Age: dataset age range 0.5–2 years"]
    one_bound = {"h0_1": True, "f_age_min": "0.33", "f_age_max": None}
    assert mr.reasons_for(_age_plans(), one_bound, None) == ["Age: dataset age range 4 months"]


def test_reason_for_text_and_label_venues():
    row = {"h0_3": True, "h2_0": True, "f_species": "Mouse"}
    assert mr.reasons_for(_age_plans(), row, None) == [
        "Age: named in the title or description",
        "Species: species label “Mouse”",
    ]
    long_label = {"h2_0": True, "f_species": "x" * 80}
    assert mr.reasons_for(_age_plans(), long_label, None) == [f"Species: species label “{'x' * 60}…”"]


def test_no_reason_when_no_venue_holds():
    # An uncertain-tier row: the age is only probable (README), its own line says so.
    assert mr.reasons_for(_age_plans(), {"h0_0": False, "n0_0": 0}, total=5) == []


# ── Attaching to a page ───────────────────────────────────────────────────────

class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _Nested:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, rows=None, error=None):
        self.rows, self.error, self.calls = rows or [], error, []

    def begin_nested(self):
        return _Nested()

    async def execute(self, statement, params=None):
        self.calls.append((str(statement), params))
        if self.error:
            raise self.error
        return _Result(self.rows)


def _dataset(total=None) -> DatasetSchema:
    summary = ParticipantSummarySchema(total=total) if total else None
    return DatasetSchema(id=uuid4(), name="ds", dataset_type="raw", source_type="openneuro",
                         participant_summary=summary)


def test_attach_sets_reasons_from_one_query():
    first, second = _dataset(total=40), _dataset()
    rows = [{"dataset_id": first.id, "h0_0": True, "n0_0": 12, "lo0_0": 0.33, "hi0_0": 1.92}]
    session = _Session(rows)
    asyncio.run(mr.attach_match_reasons(session, [first, second], _sql(age_condition(["infants"]))))
    assert len(session.calls) == 1
    assert session.calls[0][1] == {"ids": [first.id, second.id]}
    assert first.match_reasons == ["Age: 12 of 40 participants aged 4–23 months"]
    assert second.match_reasons == []


def test_attach_skips_the_query_when_nothing_needs_explaining():
    session = _Session()
    dataset = _dataset()
    asyncio.run(mr.attach_match_reasons(session, [dataset], _sql(MODALITY)))
    asyncio.run(mr.attach_match_reasons(session, [dataset], None))
    asyncio.run(mr.attach_match_reasons(session, [], _sql(DIAGNOSIS)))
    assert session.calls == [] and dataset.match_reasons == []


def test_a_failing_query_leaves_the_page_without_reasons():
    dataset = _dataset()
    session = _Session(error=RuntimeError("statement timeout"))
    asyncio.run(mr.attach_match_reasons(session, [dataset], _sql(DIAGNOSIS)))
    assert dataset.match_reasons == []
