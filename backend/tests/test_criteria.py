"""The species criterion (RAG/criteria.py): one fixed filter line per question."""

from __future__ import annotations

import pytest

from criteria import ANY_ANIMAL, normalise_species, species_condition


def _augmented(species, available=("species",), description_search=None) -> str:
    from preprocess import QueryPlan, _build_augmented_question

    plan = QueryPlan(
        query_family="description_search",
        natural_language_summary="Find animal datasets.",
        species=species,
        description_search=description_search,
    )
    return _build_augmented_question(plan, [], available_columns=set(available))


def test_planner_words_become_codes_and_unknown_ones_are_dropped() -> None:
    assert normalise_species(["mice", "rodents", "Monkeys", "human", "unicorn"]) == ["mouse", "rat", "primate"]
    assert normalise_species(["mouse", "animals"]) == [ANY_ANIMAL]   # any animal covers the rest
    assert normalise_species(None) == []


def test_plan_accepts_a_bad_species_value_without_failing() -> None:
    from preprocess import QueryPlan

    plan = QueryPlan(query_family="description_search", natural_language_summary="x",
                     species=["Homo sapiens", "macaques"])
    assert plan.species == ["primate"]
    assert QueryPlan(query_family="description_search", natural_language_summary="x",
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
    from services.fulltext import text_search, uncertain_tier

    base = ("SELECT d.id, d.name FROM bids_datasets AS d WHERE "
            "EXISTS(SELECT 1 FROM bids_objects AS o2 WHERE o2.dataset_id = d.id "
            "AND o2.datatype = 'functional_mri') AND {cond}")
    expanded = text_search.expand_text_search(base.format(cond=species_condition(["mouse"])),
                                              fts=True, papers=True).sql
    assert uncertain_tier.apply(expanded, papers=True) == expanded
