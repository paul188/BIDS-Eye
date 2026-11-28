"""The species criterion (backend/vocabulary/criteria.py): one fixed filter line per question."""

from __future__ import annotations

import pytest

from criteria import (ANY_ANIMAL, HUMAN, HUMAN_SPECIES, SPECIES, SPECIES_ASSUMED_HUMAN,
                      normalise_species, species_condition)


def _augmented(species, available=("species",), description_search=None) -> str:
    from preprocess import QueryPlan, _build_augmented_question

    plan = QueryPlan(
        natural_language_summary="Find animal datasets.",
        species=species,
        description_search=description_search,
    )
    return _build_augmented_question(plan, [], available_columns=set(available))


def test_planner_words_become_codes_and_unknown_ones_are_dropped() -> None:
    assert normalise_species(["mice", "rodents", "Monkeys", "unicorn"]) == ["mouse", "rat", "primate"]
    assert normalise_species(["mouse", "animals"]) == [ANY_ANIMAL]   # any animal covers the rest
    # ... but not humans; and "non-human" is no human.
    assert normalise_species(["humans", "mouse", "animals"]) == [HUMAN, ANY_ANIMAL]
    assert normalise_species(["non-human"]) == [ANY_ANIMAL]
    assert normalise_species(None) == []


def test_plan_accepts_a_bad_species_value_without_failing() -> None:
    from preprocess import QueryPlan

    plan = QueryPlan(natural_language_summary="x",
                     species=["Homo sapiens", "macaques", "unicorn"])
    assert plan.species == [HUMAN, "primate"]
    assert QueryPlan(natural_language_summary="x",
                     species="mice").species == ["mouse"]


def test_mouse_reads_the_label_and_the_title_of_unlabelled_datasets() -> None:
    cond = species_condition(["mouse"])
    assert cond.startswith("(d.species ~* '\\m(mouse|mice|mus)\\M' OR ")
    # 'Other' / 'n/a' / empty labels say nothing, like NULL.
    assert "(d.species IS NULL OR d.species ~* '^\\s*(other|n/?a|unknown)?\\s*$')" in cond
    assert "text_match('mouse | mice | murine', 'short')" in cond


def test_primate_fallback_reads_the_whole_text() -> None:
    assert "text_match('macaque | macaca | monkey | monkeys | rhesus | marmoset | baboon')" \
        in species_condition(["primate"])


def test_any_animal_excludes_human_and_phantom_labels() -> None:
    cond = species_condition([ANY_ANIMAL])
    assert "d.species !~* '(human|homo sapiens|phantom|synthetic|printed)'" in cond
    assert "'short')" in cond


def test_several_species_are_alternatives() -> None:
    cond = species_condition(["mouse", "rat"])
    assert "(mouse|mice|mus)" in cond and "(rats?|rattus)" in cond
    assert ") OR d.species ~* '\\m(rats?|rattus)\\M'" in cond


def test_without_the_column_only_the_text_is_searched() -> None:
    assert species_condition(["mouse"], has_column=False) == "(text_match('mouse | mice | murine', 'short'))"


def test_human_is_the_default_for_unlabelled_datasets() -> None:
    assert SPECIES.default == HUMAN
    assert species_condition([HUMAN]) == f"({HUMAN_SPECIES})"
    # A human label, anchored: 'Non-Human Primates' and 'Phantom Human' are not human.
    assert HUMAN_SPECIES.startswith("(d.species ~* '^\\s*(humans?|homo sapiens)\\s*$' OR ")
    # No label: human unless the name or description names an animal species.
    assert SPECIES_ASSUMED_HUMAN.startswith(
        "((d.species IS NULL OR d.species ~* '^\\s*(other|n/?a|unknown)?\\s*$') AND NOT text_match('mouse | ")
    assert "macaque" in SPECIES_ASSUMED_HUMAN and "piglet" in SPECIES_ASSUMED_HUMAN
    assert SPECIES_ASSUMED_HUMAN.endswith("'short'))")
    # 'animal' shares the stem of 'animated' (ds000228, children watching a film);
    # 'cat' is in "CAT snacks", a human study.
    assert "animal" not in SPECIES_ASSUMED_HUMAN and " cat " not in SPECIES_ASSUMED_HUMAN


def test_human_without_the_column_is_no_condition() -> None:
    # Every dataset counts as human by default.
    assert species_condition([HUMAN], has_column=False) is None
    assert species_condition([HUMAN, "mouse"], has_column=False) is None


def test_human_and_animals_are_alternatives() -> None:
    cond = species_condition([HUMAN, ANY_ANIMAL])
    assert cond.startswith(f"({HUMAN_SPECIES} OR (d.species !~* ")


def test_human_alone_is_a_filter_line_and_not_a_text_search() -> None:
    augmented = _augmented([HUMAN], description_search="human")
    assert "dataset text:" not in augmented
    assert f"  species (complete, no other species condition): ({HUMAN_SPECIES})" in augmented


def test_species_alone_is_a_filter_line() -> None:
    # "mouse datasets": nothing else in the plan, the line must still be there.
    augmented = _augmented(["mouse"])
    assert "  species (complete, no other species condition): (d.species ~* " in augmented


def test_animal_studies_is_not_searched_as_text() -> None:
    # "animal studies done at Harvard" searched 'animal & studies' in the README
    # and paper text, and returned human studies that cite animal work.
    augmented = _augmented([ANY_ANIMAL], description_search="animal studies")
    assert "dataset text:" not in augmented
    assert "species (complete" in augmented


def test_species_condition_stays_hard_in_the_uncertain_tier() -> None:
    pytest.importorskip("sqlglot")
    from services.fulltext import text_search
    from services.uncertain import tier as uncertain_tier

    base = ("SELECT d.id, d.name FROM bids_datasets AS d WHERE "
            "EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
            "AND o2.datatype = 'functional_mri') AND {cond}")
    expanded = text_search.expand_text_search(base.format(cond=species_condition(["mouse"])),
                                              papers=True).sql
    assert uncertain_tier.apply(expanded, papers=True) == expanded


def test_human_condition_is_hard_in_the_tier_and_a_human_chip() -> None:
    pytest.importorskip("sqlglot")
    from services.fulltext import text_search
    from services.uncertain import tier as uncertain_tier
    from services.results import query_filters

    base = ("SELECT d.id, d.name FROM bids_datasets AS d WHERE "
            "EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
            "AND o2.datatype = 'functional_mri') AND {cond}")
    expanded = text_search.expand_text_search(base.format(cond=species_condition([HUMAN])),
                                              papers=True).sql
    # The animal words stay negated on the name and description.
    assert "AND NOT d.search_tsv @@ to_tsquery('english', 'mouse:AB | mice:AB" in expanded
    assert uncertain_tier.apply(expanded, papers=True) == expanded
    assert ("Species", "human") in [(c.field, c.label) for c in query_filters.describe(expanded)]


# ── Diagnosis: not stated means not there ─────────────────────────────────────

EPILEPSY_STATED = ("(EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id "
                   "AND p2.diagnosis IN ('epilepsy'))")


def _without(excluded, resolved_terms=None, unresolved=None, groups=()):
    from preprocess import (ParticipantGroup, QueryPlan, RAGField, ResolvedTerms,
                            _build_augmented_question)

    plan = QueryPlan(
        natural_language_summary="Find datasets without these patients.",
        groups=[ParticipantGroup(group_name=f"g{i}", diagnosis_terms=list(terms))
                for i, terms in enumerate(groups)],
        excluded_diagnosis_terms=excluded,
    )
    resolved = ([ResolvedTerms(field=RAGField.DIAGNOSIS, term_to_codes=resolved_terms)]
                if resolved_terms else [])
    return plan, resolved, _build_augmented_question(plan, resolved, unresolved_terms=unresolved)


def test_an_unstated_diagnosis_is_absent_without_a_named_default() -> None:
    from criteria import DIAGNOSIS, diagnosis_unknown, without_diagnosis

    # The general rule, not a named default like species = human.
    assert DIAGNOSIS.default is None and SPECIES.default == "human"
    assert without_diagnosis("text_match('epilepsy', 'short')") == "NOT (text_match('epilepsy', 'short'))"
    # No coded participant and no dx_status: nothing is recorded.
    assert "dx_any.diagnosis IS NOT NULL" in diagnosis_unknown("x") and "x.dx_status" in diagnosis_unknown("x")


def test_an_excluded_diagnosis_is_the_negated_request_for_it() -> None:
    from preprocess import RAGField, build_rag_requests

    plan, _, augmented = _without(["epilepsy"], {"epilepsy": ["epilepsy"]})
    assert [r.terms for r in build_rag_requests(plan) if r.field == RAGField.DIAGNOSIS] == [["epilepsy"]]
    line = next(row for row in augmented.splitlines() if row.startswith('  not diagnosis "epilepsy"'))
    # A dataset that records no diagnosis fails every part of the inner
    # condition, so the NOT keeps it.
    assert line.split(": ", 1)[1].startswith(f"NOT ({EPILEPSY_STATED} OR text_match('")
    assert "'short')" in line and "IS NULL" not in line
    # Not also asked for.
    assert not any(row.startswith("  diagnosis") for row in augmented.splitlines())


def test_asked_and_excluded_diagnoses_get_separate_lines() -> None:
    _, _, augmented = _without(["bipolar disorder"],
                               {"depression": ["major_depressive_disorder"],
                                "bipolar disorder": ["bipolar_disorder"]},
                               groups=[["depression"]])
    lines = augmented.splitlines()
    asked = next(row for row in lines if row.startswith("  diagnosis"))
    excluded = next(row for row in lines if row.startswith("  not diagnosis"))
    assert "major_depressive_disorder" in asked and "bipolar_disorder" not in asked
    assert "bipolar_disorder" in excluded and "major_depressive_disorder" not in excluded


def test_an_excluded_diagnosis_outside_the_vocabulary_is_excluded_by_its_words() -> None:
    _, _, augmented = _without(["moyamoya"], unresolved={"diagnosis": ["moyamoya"]})
    assert ('not diagnosis "moyamoya" (complete, datasets that record no diagnosis stay): '
            "NOT (text_match('moyamoya', 'short'))") in augmented
    assert "VOCABULARY MISS" not in augmented   # not searched for


def test_excluding_healthy_people_is_no_condition() -> None:
    plan, _, augmented = _without(["healthy controls"])
    assert "not diagnosis" not in augmented


def test_an_excluded_diagnosis_compiles() -> None:
    from services.translate import participant_query_compiler as pqc

    plan, resolved, augmented = _without(["epilepsy"], {"epilepsy": ["epilepsy"]})
    compiled = pqc.compile_plan(plan, resolved, augmented)
    assert compiled.sql is not None, compiled.reason
    assert f"WHERE NOT ({EPILEPSY_STATED} OR text_match('" in compiled.sql


def _sex_augmented(species, summary="Datasets with only female participants.") -> str:
    from preprocess import ParticipantGroup, QueryPlan, _build_augmented_question

    plan = QueryPlan(
        query_family="participant_filter",
        natural_language_summary=summary,
        groups=[ParticipantGroup(group_name="all_participants", sex=["female"])],
        species=species,
    )
    return _build_augmented_question(plan, [], available_columns={"species"})


def test_a_sex_without_a_species_asks_for_people() -> None:
    # Gold request qx09 "only female participants": 5 of 41 results were
    # female mice, rats, sheep and macaques.
    augmented = _sex_augmented([])
    assert "  participant sex: " in augmented
    assert f"  species (complete, no other species condition): ({HUMAN_SPECIES})" in augmented


def test_a_sex_with_a_species_keeps_that_species() -> None:
    augmented = _sex_augmented(["rat"], summary="Datasets with only female rats.")
    assert "  species (complete, no other species condition): (d.species ~* " in augmented
    assert HUMAN_SPECIES not in augmented
