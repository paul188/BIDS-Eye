"""The age-group criterion (RAG/criteria.py): "children" is an age filter, not a word.

"impulse control in children" searched the word "children" (2026-11-15) and
missed datasets that say "youth" or "adolescents" or only record ages.
"""

from __future__ import annotations

import pytest

from criteria import AGE_GROUPS, age_condition, normalise_age_groups


def _plan(**fields):
    from preprocess import QueryPlan

    return QueryPlan(query_family="description_search",
                     natural_language_summary="Find datasets.", **fields)


def _augmented(available=("species",), **fields) -> str:
    from preprocess import _build_augmented_question

    return _build_augmented_question(_plan(**fields), [], available_columns=set(available))


def _expanded(condition: str) -> str:
    from services.fulltext import text_search

    base = ("SELECT d.id, d.name FROM bids_datasets AS d WHERE "
            "EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
            "AND o2.datatype = 'functional_mri') AND {cond} GROUP BY d.id")
    return text_search.expand_text_search(base.format(cond=condition), fts=True, papers=True).sql


# ── Codes ─────────────────────────────────────────────────────────────────────

def test_planner_words_become_codes() -> None:
    assert normalise_age_groups(["kids", "Elderly", "unicorn"]) == ["children", "older_adults"]
    assert normalise_age_groups(["pediatric", "youth"]) == ["children"]
    assert normalise_age_groups(["teenagers"]) == ["adolescents"]
    assert normalise_age_groups(["older adults"]) == ["older_adults"]
    assert normalise_age_groups(None) == []


def test_children_cover_infants_and_adolescents() -> None:
    # Paul, 2026-11-17: a request for children includes adolescents.
    assert normalise_age_groups(["adolescents", "children", "babies"]) == ["children"]
    assert AGE_GROUPS["children"].high == 18
    assert "adolescents" in AGE_GROUPS["children"].words


def test_plan_accepts_a_bad_age_group_without_failing() -> None:
    assert _plan(age_groups=["adults", "kids"]).age_groups == ["children"]
    assert _plan(age_groups="seniors").age_groups == ["older_adults"]


# ── Condition ─────────────────────────────────────────────────────────────────

def test_recorded_ages_decide_and_zero_is_no_age() -> None:
    cond = age_condition(["children"])
    assert "pa.age > 0 AND pa.age < 18" in cond
    assert "(d.age_min > 0 AND d.age_min < 18) OR (d.age_max > 0 AND d.age_max < 18)" in cond
    # Without any age above 0, the short fields decide.
    assert "COALESCE(d.age_min, 0) <= 0 AND COALESCE(d.age_max, 0) <= 0 AND NOT EXISTS" in cond
    assert "'short')" in cond and "youth" in cond and "pediatric" in cond


def test_adolescents_stop_below_18() -> None:
    # With 19, every adult study of 18-year-old students counted (401 datasets).
    cond = age_condition(["adolescents"])
    assert "pa.age >= 12 AND pa.age < 18" in cond
    assert "d.age_min < 18 AND d.age_max >= 12" in cond


def test_animals_are_never_an_age_group() -> None:
    from criteria import HUMAN_SPECIES

    # Human by label or by the species default (RAG/criteria.py).
    cond = age_condition(["children"])
    assert cond.endswith(f" AND {HUMAN_SPECIES})")
    assert age_condition(["children"], has_species_column=False).count("species") == 0


def test_older_adults_have_no_upper_bound() -> None:
    cond = age_condition(["older_adults"])
    assert "pa.age >= 60)" in cond
    assert "(d.age_max >= 60 OR d.age_min >= 60)" in cond


# ── Plan → filter line ────────────────────────────────────────────────────────

def test_age_group_alone_is_a_filter_line() -> None:
    augmented = _augmented(age_groups=["children"])
    assert "  age group children (complete, no other age condition): ((EXISTS (" in augmented


def test_pediatric_left_in_description_search_becomes_the_age_group() -> None:
    augmented = _augmented(description_search="pediatric")
    assert "dataset text:" not in augmented
    assert "age group children (complete" in augmented
    augmented = _augmented(description_search="older adults")
    assert "age group older adults (complete" in augmented


def test_other_words_stay_a_text_search() -> None:
    augmented = _augmented(description_search="bilingual children", age_groups=["children"])
    assert "dataset text: text_match('bilingual children')" in augmented
    assert "age group children (complete" in augmented


# ── What reads the SQL afterwards ─────────────────────────────────────────────

def test_uncertain_tier_keeps_readme_mentions_as_probable() -> None:
    pytest.importorskip("sqlglot")
    import sqlglot
    from services.fulltext import tier_classify, uncertain_tier

    sql = _expanded(age_condition(["children"]))
    tree = sqlglot.parse_one(sql, read="postgres")
    kinds = []
    for node in uncertain_tier._flatten_and(tree.args["where"].this):
        condition = tier_classify.classify(node, "d", papers=True)
        kinds.append(condition.kind if condition else None)
        if condition and condition.kind == "age":
            assert "children" in condition.probable and "adolescents" in condition.probable
            assert condition.reason_probable.startswith("ages not recorded")
            # Own text only: a paper's child words were mostly about other studies.
            assert "search_tsv_papers" not in condition.probable
    # The modality and the species guard stay hard; the age part is soft.
    assert kinds == [None, "age", None]


def test_filter_chips_say_age_and_species() -> None:
    pytest.importorskip("sqlglot")
    from criteria import species_condition
    from services.results import query_filters

    chips = [(c.field, c.label) for c in query_filters.describe(_expanded(age_condition(["children"])))]
    assert ("Age", "under 18") in chips and ("Species", "human") in chips
    chips = [(c.field, c.label) for c in query_filters.describe(_expanded(age_condition(["older_adults"])))]
    assert ("Age", "60+") in chips
    chips = [(c.field, c.label) for c in query_filters.describe(_expanded(species_condition(["mouse"])))]
    assert ("Species", "mouse") in chips
    chips = [(c.field, c.label) for c in query_filters.describe(_expanded(species_condition(["any_animal"])))]
    assert ("Species", "any animal") in chips


def test_compiler_takes_the_age_line() -> None:
    from preprocess import ParticipantGroup, RAGField, ResolvedTerms, QueryPlan
    from services.translate import participant_query_compiler as pqc

    plan = QueryPlan(query_family="description_search", natural_language_summary="x",
                     groups=[ParticipantGroup(group_name="g0", diagnosis_terms=["ADHD"])],
                     age_groups=["children"])
    resolved = [ResolvedTerms(field=RAGField.DIAGNOSIS, term_to_codes={"ADHD": ["adhd"]})]
    line = f"  age group children (complete, no other age condition): {age_condition(['children'])}"
    augmented = ("[Resolved DB filters — copy each filter expression verbatim into the SQL WHERE clause]\n"
                 + line + "\n\nFind datasets.")
    result = pqc.compile_plan(plan, resolved, augmented)
    assert result.sql is not None, result.reason
    assert age_condition(["children"]) in result.sql
    assert "p_cohort.diagnosis IN ('adhd')" in result.sql
