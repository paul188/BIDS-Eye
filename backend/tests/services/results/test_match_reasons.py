"""Why a result matched: the venues of each chip (services/results/match_reasons.py)."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
import sqlglot

from criteria import age_condition, species_condition
from routers.schemas import DatasetSchema, ParticipantSummarySchema
from services.fulltext import text_search
from services import sql_tree
from services.evidence.confirmed import (ConfirmedByDefaultEvidence, ConfirmedEvidence, ConfirmedInDatasetColumnEvidence,
                                         ConfirmedInFileTablesEvidence, age_span)
from services.evidence.base import Source
from services.evidence import branches
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
    return [evidence.source for evidence in plan.branches]


# ── Reading the venues ────────────────────────────────────────────────────────

def test_infant_age_group_has_four_venues():
    plans = _plans(age_condition(["infants"]))
    age = plans["Age"]
    assert _kinds(age) == [Source.PARTICIPANTS, Source.FIELD, Source.FIELD, Source.SHORT_TEXT]
    participants = age.branches[0]
    assert participants.participant_rows is not None and participants.age_column is not None
    # Either age bound alone is read as the whole range.
    assert age.branches[1].columns[:2] == ("age_min", "age_max")
    # The species guard of the age group is not explained: nobody asked for a species.
    assert "Species" not in plans


@pytest.mark.parametrize("order", ["species first", "age first"])
def test_a_species_asked_for_is_explained_next_to_an_age_group(order):
    # "human infants": the species condition the search asked for is its
    # own chip (a human label, or no label and the default); the age
    # group's guard, the same SQL in parentheses with the ages, is not.
    conditions = [species_condition(["human"]), age_condition(["infants"])]
    if order == "age first":
        conditions.reverse()
    plans = mr.plan(_sql(*conditions))[1]
    assert [p.field for p in plans] == (
        ["Species", "Age"] if order == "species first" else ["Age", "Species"])
    species = next(p for p in plans if p.field == "Species")
    assert _kinds(species) == [Source.FIELD, Source.ASSUMED]


def test_diagnosis_reads_the_participant_list_and_the_text():
    diagnosis = _plans(DIAGNOSIS)["Diagnosis"]
    # 'parkinson:AB' reaches the name and description only (weights A, B).
    assert _kinds(diagnosis) == [Source.PARTICIPANTS, Source.SHORT_TEXT]
    assert _kinds(_plans(DIAGNOSIS.replace(":AB", ""))["Diagnosis"]) == [Source.PARTICIPANTS, Source.TEXT]
    assert diagnosis.branches[0].reads_diagnosis
    assert diagnosis.branches[0].age_column is None


def test_task_reads_files_and_text_species_reads_label_and_text():
    assert _kinds(_plans(TASK)["Task"]) == [Source.FILES, Source.SHORT_TEXT]
    assert _kinds(_plans(species_condition(["mouse"]))["Species"]) == [Source.FIELD, Source.SHORT_TEXT]


def test_single_venue_negated_and_having_chips_are_not_explained():
    # A modality only ever matches files; a negated chip matched by not holding
    # (an excluded diagnosis has its own reason, see the default test below).
    assert _plans(MODALITY, f"NOT {TASK}") == {}
    sql = (_HEAD + "WHERE " + MODALITY + " GROUP BY d.id HAVING "
           "COUNT(DISTINCT o.subject) >= 20")
    assert mr.plan(sql)[1] == []


def test_a_branch_on_the_outer_join_is_left_out():
    # o is the outer query's join: the branch cannot be evaluated per dataset,
    # and what is left is one venue.
    sql = _sql("(o.task = 'rest' OR d.search_tsv @@ to_tsquery('english', 'rest:AB'))")
    assert mr.plan(sql)[1] == []


@pytest.mark.parametrize("branch, kinds", [
    ("NOT d.search_tsv @@ to_tsquery('english', 'mouse')", []),
    ("NOT d.search_tsv_papers @@ to_tsquery('english', 'mouse')", []),
    ("NOT EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
     "AND p2.diagnosis = 'x')", []),
    ("NOT d.id IN (SELECT dataset_id FROM bids_derivatives)", []),
    ("d.species IS NULL", [ConfirmedByDefaultEvidence]),
    ("d.species IS NOT NULL", [ConfirmedInDatasetColumnEvidence]),   # sqlglot: NOT (species IS NULL)
    ("NOT d.species = 'Mouse'", [ConfirmedInDatasetColumnEvidence]),
    ("d.on_brainlife", [ConfirmedInDatasetColumnEvidence]),          # the branch is the column itself
])
def test_a_branch_that_is_a_not_confirms_nothing_it_denies(branch, kinds):
    node = sqlglot.parse_one(f"SELECT 1 FROM bids_datasets AS d WHERE {branch}",
                             read="postgres").args["where"].this
    assert [type(way) for way in branches.confirming_evidence(text_search.Clause.read(node), "d")] == kinds


_POSITIVE = {
    "vector": "d.search_tsv @@ to_tsquery('english', 'mouse')",
    "short vector": "d.search_tsv @@ to_tsquery('english', 'mouse:AB')",
    "paper vector": "d.search_tsv_papers @@ to_tsquery('english', 'mouse')",
    "name": "d.name ILIKE '%mouse%'",
    "lower name": "LOWER(d.name) LIKE '%mouse%'",
    "name pattern": "d.name ~* 'mouse'",
    "description": "d.description_text LIKE '%mouse%'",
    "readme": "d.readme_text ILIKE '%mouse%'",
    "participants": ("EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
                     "AND p2.diagnosis = 'parkinsons_disease')"),
    "participants in": "d.id IN (SELECT dataset_id FROM bids_participants WHERE age < 3)",
    "files": ("EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
              "AND o2.task = 'rest')"),
    "species default": "d.species IS NULL",
}
_DENIALS = {
    "NOT": "NOT {}", "NOT ()": "NOT ({})", "NOT (OR)": "NOT ({} OR d.id IS NULL)",
    "IS FALSE": "({}) IS FALSE", "= FALSE": "({}) = FALSE", "IS NOT TRUE": "({}) IS NOT TRUE",
    "<> TRUE": "({}) <> TRUE", "IS DISTINCT FROM TRUE": "({}) IS DISTINCT FROM TRUE", "IS NULL": "({}) IS NULL",
    "NOT LIKE": " NOT LIKE ", "NOT ILIKE": " NOT ILIKE ", "NOT IN": " NOT IN ", "!~*": " !~* ",
}


def _denials():
    for shape, sql in _POSITIVE.items():
        for how, form in _DENIALS.items():
            if "{}" in form:
                yield pytest.param(form.format(sql), id=f"{shape}, {how}")
            else:
                operator = " ~* " if how == "!~*" else form.replace("NOT ", "")
                if operator in sql:
                    yield pytest.param(sql.replace(operator, form), id=f"{shape}, {how}")


def _confirms(way) -> bool:
    """Not a bare column value: the denied shapes still read one."""
    return isinstance(way, ConfirmedEvidence) and not isinstance(way, ConfirmedInDatasetColumnEvidence)


def _ways(condition: str) -> tuple:
    tree = sql_tree.parse_select(f"SELECT 1 FROM bids_datasets AS d WHERE {condition}")
    return branches.confirming_evidence(text_search.Clause.read(tree.args["where"].this), "d")


@pytest.mark.parametrize("condition", _POSITIVE.values(), ids=_POSITIVE)
def test_each_shape_confirms(condition):
    assert any(_confirms(way) for way in _ways(condition))


@pytest.mark.parametrize("condition", _denials())
def test_a_denied_shape_confirms_nothing(condition):
    assert not [way for way in _ways(condition) if _confirms(way)]


def test_an_empty_text_column_names_nothing():
    assert _ways("d.readme_text IS NULL") == ()
    assert _ways("d.search_tsv IS NULL") == ()


def test_an_unclassified_branch_is_logged_without_its_values(caplog):
    _ways("d.name = 'secret search'")
    assert "d.name = ?" in caplog.text and "secret" not in caplog.text
    caplog.clear()
    _ways("NOT d.search_tsv @@ to_tsquery('english', 'mouse')")
    _ways("o.task = 'rest'")
    assert caplog.text == ""


def test_paper_and_readme_venues():
    paper = ("(d.search_tsv @@ to_tsquery('english', 'sleep') OR "
             "d.search_tsv_papers @@ to_tsquery('english', 'sleep'))")
    assert _kinds(_plans(paper)["Text"]) == [Source.TEXT, Source.PAPERS]
    # search_tsv_papers alone holds both texts: in the own text, or only in a paper.
    for guard in ("", " AND d.search_tsv_exact_papers @@ to_tsquery('simple', 'sleep')"):
        both = _plans(f"(d.search_tsv_papers @@ to_tsquery('english', 'sleep'){guard})")["Text"]
        own, only_paper = both.branches
        assert (own.source, only_paper.source) == (Source.TEXT, Source.PAPERS)
        assert "_papers" not in own.sql() and "search_tsv" in own.sql()
        assert only_paper.sql().endswith(f" AND NOT ({own.sql()})")
    # Limited to the name and description, a search never reaches a paper.
    assert _plans("d.search_tsv_papers @@ to_tsquery('english', 'sleep:AB')") == {}
    # A URL fragment stays a substring search (text_search.convert_long_text_ilike).
    readme = "(d.readme_text ILIKE '%doi.org%' OR d.description_text ILIKE '%doi.org%')"
    assert _kinds(_plans(readme)["Text"]) == [Source.README, Source.SHORT_TEXT]


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
    assert branches.short_fields_only(tsquery) is short


def test_a_word_form_guard_does_not_decide_the_text_venue():
    # A search and its exact word forms (services/fulltext/text_search.py):
    # the guard reads 'simple' and has no labels; the search decides.
    guarded = ("((d.search_tsv @@ to_tsquery('english', 'ageing:AB') AND "
               "d.search_tsv_exact @@ to_tsquery('simple', 'ageing | aging')) OR "
               "EXISTS(SELECT 1 FROM bids_participants AS p2 WHERE p2.dataset_id = d.id "
               "AND p2.age >= 60))")
    plans = mr.plan(_HEAD + "WHERE " + guarded + " GROUP BY d.id")[1]
    assert [e.source for p in plans for e in p.branches] == [Source.SHORT_TEXT, Source.PARTICIPANTS]


def test_participants_in_an_in_subquery_are_not_counted():
    condition = ("(d.id IN (SELECT dataset_id FROM bids_participants WHERE diagnosis = 'adhd') "
                 "OR d.search_tsv @@ to_tsquery('english', 'adhd:AB'))")
    branch = _plans(condition)
    venue = next(iter(branch.values())).branches[0]
    assert venue.source == Source.PARTICIPANTS and venue.participant_rows is None


def test_unreadable_sql_has_no_plan():
    assert mr.plan("not sql") == (None, [])
    assert mr.plan(None) == (None, [])
    assert mr.plan("SELECT 1") == (None, [])


def test_built_query_is_valid_sql_with_counts_and_ages():
    alias, plans = mr.plan(_sql(age_condition(["infants"], has_species_column=False), DIAGNOSIS))
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
    assert age_span(low, high) == expected


def _age_plans():
    # Without the species guard, so the chips are age (0), diagnosis (1), species (2).
    return mr.plan(_sql(age_condition(["infants"], has_species_column=False), DIAGNOSIS,
                        species_condition(["mouse"])))[1]


def test_reason_for_the_species_default():
    plans = mr.plan(_sql(species_condition(["human"])))[1]
    assert [p.field for p in plans] == ["Species"]
    assert _kinds(plans[0]) == [Source.FIELD, Source.ASSUMED]
    assert mr.reasons_for(plans, {"h0_0": True, "f_species": "Human"}, None) == [
        "Species: species label “Human”"]
    assert mr.reasons_for(plans, {"h0_1": True, "f_species": None}, None) == [
        "Species: no species label, assumed human"]
    # The negated animal words only say when the default applies: no tsvector value.
    assert "f_search_tsv" not in mr.build_sql("d", plans)


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
    # An age of 0 is an age (newborns), not a missing bound.
    from_birth = {"h0_1": True, "f_age_min": "0", "f_age_max": "2"}
    assert mr.reasons_for(_age_plans(), from_birth, None) == ["Age: dataset age range 0–2 years"]
    # 0 to 0 is a placeholder and says nothing.
    assert mr.reasons_for(_age_plans(), {"h0_1": True, "f_age_min": "0", "f_age_max": "0"}, None) == []


def test_a_kind_without_a_card_text_cannot_be_built():
    with pytest.raises(ValueError):
        ConfirmedInFileTablesEvidence("TRUE", source=Source.PARTICIPANTS)
    with pytest.raises(ValueError):
        ConfirmedByDefaultEvidence("TRUE", columns=("age_min",))
    with pytest.raises(ValueError):
        ConfirmedByDefaultEvidence("TRUE")


def test_participants_joined_to_another_table_are_not_counted():
    condition = ("(EXISTS(SELECT 1 FROM bids_participants AS p2 JOIN bids_objects AS o2 "
                 "ON o2.dataset_id = p2.dataset_id WHERE p2.dataset_id = d.id AND p2.diagnosis = 'adhd') "
                 "OR d.search_tsv @@ to_tsquery('english', 'adhd:AB'))")
    branch = next(iter(_plans(condition).values())).branches[0]
    assert branch.source == Source.PARTICIPANTS and branch.participant_rows is None


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
    from sqlalchemy.exc import OperationalError
    session = _Session(error=OperationalError("SELECT", {}, Exception("canceling statement due to statement timeout")))
    asyncio.run(mr.attach_match_reasons(session, [dataset], _sql(DIAGNOSIS)))
    assert dataset.match_reasons == []


def test_reason_for_an_excluded_diagnosis_that_holds_by_default():
    from criteria import without_diagnosis

    plans = mr.plan(_sql(without_diagnosis(DIAGNOSIS)))[1]
    assert len(plans) == 1 and plans[0].field.startswith("No ")
    assert _kinds(plans[0]) == [Source.ASSUMED_ABSENT]
    sql = mr.build_sql("d", plans)
    assert "dx_any.diagnosis IS NOT NULL" in sql and "d.dx_status" in sql
    field = plans[0].field
    # Nothing recorded: absent by default. Something else recorded: no reason needed.
    assert mr.reasons_for(plans, {"h0_0": True}, None) == [
        f"{field}: no diagnosis recorded, assumed absent"]
    assert mr.reasons_for(plans, {"h0_0": False}, None) == []
