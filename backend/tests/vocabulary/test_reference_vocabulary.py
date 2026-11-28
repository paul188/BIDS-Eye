"""Diseases and tasks no dataset has yet (backend/vocabulary/reference_vocabulary.py), and names.

"rasmussen" alone was read as an author and the name lookup picked "Rasmus
Sinn" (2026-11-18). These tests pin the three parts of the fix: the official
vocabulary (MeSH, Cognitive Atlas) knows the disease, an author must carry
the name the user typed, and a disease name without "by"/"lab" is not a person.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import preprocess as pp
import reference_vocabulary as rv
import retriever
from retriever import MetadataRetriever, _resolve_yaml_term, _same_person
from text_terms import _lookup_strings, diagnosis_alternatives

NAME_INDEX = Path(retriever.__file__).parent / "name_index.json"


@pytest.fixture(autouse=True)
def _no_embedding_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(field: str, term: str):
        raise AssertionError(f"{field}/{term!r} fell through to the embedding fallback")

    monkeypatch.setattr(retriever, "_embedding_fallback_scored", fail)


# ── The official vocabulary ───────────────────────────────────────────────────

@pytest.mark.parametrize("term", [
    "rasmussen", "Rasmussen's", "rasmussen encephalitis", "Rasmussen Syndrome",
])
def test_rasmussen_is_the_disease(term: str) -> None:
    assert [c for c, _ in rv.match("diagnosis", term)] == ["rasmussen_syndrome"]


@pytest.mark.parametrize("term,code", [
    ("huntington's disease", "huntington_disease"),
    ("Guillain-Barré syndrome", "guillain_barre_syndrome"),
    ("tourette", "tourette_syndrome"),
    ("narcolepsy", "narcolepsy"),
    ("migraine with aura", "migraine_with_aura"),
])
def test_official_terms(term: str, code: str) -> None:
    assert code in [c for c, _ in rv.match("diagnosis", term)]


@pytest.mark.parametrize("word", ["down", "pick", "limbic", "japanese", "brainstem"])
def test_ordinary_and_medical_words_are_not_surnames(word: str) -> None:
    assert rv.match("diagnosis", word) == []


def test_cognitive_atlas_tasks_are_known() -> None:
    assert rv.knows("task", "Iowa gambling task")
    assert rv.knows("task", "stop signal")
    assert not rv.knows("task", "rasmussen")


def test_curated_vocabulary_decides_first() -> None:
    assert "schizophrenia" in _resolve_yaml_term("diagnosis", "schizophrenia")
    assert "parkinsons_disease" in _resolve_yaml_term("diagnosis", "parkinson")


def test_unknown_to_the_curated_vocabulary_uses_the_official_one() -> None:
    assert _resolve_yaml_term("diagnosis", "rasmussen") == ["rasmussen_syndrome"]


def test_official_terms_widen_the_text_search() -> None:
    strings = [s for s, _ in _lookup_strings("huntington_disease")]
    assert strings[0] == "huntington disease"
    assert "huntington's chorea" in strings
    assert "huntington" in diagnosis_alternatives({"huntington": ["huntington_disease"]})


# ── Author names ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("typed,author,same", [
    ("rasmussen", "Rasmus Sinn", False),
    ("Friston", "Karl J. Friston", True),
    ("K. Friston", "Karl J. Friston", True),
    ("J. Friston", "Karl Friston", False),
    ("Poldrak", "Russell A. Poldrack", True),       # one letter
    ("williams", "Laura Gwilliams", False),         # first letter differs
    ("the Poldrack lab", "Russell A. Poldrack", True),
    ("Sinn", "Rasmus Sinn", True),
])
def test_same_person(typed: str, author: str, same: bool) -> None:
    assert _same_person(typed, author) is same


def test_rasmussen_finds_no_author() -> None:
    names = MetadataRetriever(NAME_INDEX)
    assert names.retrieve_for_field("author", "rasmussen") == []
    assert names.retrieve_for_field("author", "Rasmus Sinn") == ["Rasmus Sinn"]


# ── A disease name is not a person ────────────────────────────────────────────

def _plan(authors, summary="Datasets by an author named X."):
    return pp.QueryPlan(natural_language_summary=summary, author_names=list(authors))


@pytest.fixture(scope="module")
def names() -> MetadataRetriever:
    return MetadataRetriever(NAME_INDEX)


def test_bare_disease_name_becomes_the_diagnosis(names: MetadataRetriever) -> None:
    plan = _plan(["rasmussen"], "The user is searching for datasets by an author named Rasmussen.")
    pp._names_that_are_conditions(plan, "rasmussen", names)
    assert plan.author_names == []
    assert plan.groups[0].diagnosis_terms == ["rasmussen"]
    assert "author named" not in plan.natural_language_summary
    assert "Rasmussen syndrome" in plan.natural_language_summary


def test_misspelt_disease_surname_without_such_an_author(names: MetadataRetriever) -> None:
    # Logged on the live site, 2026-11-18: searched as the author 'Rasmussehn'.
    plan = _plan(["rasmussehn"])
    pp._names_that_are_conditions(plan, "rasmussehn", names)
    assert plan.author_names == []
    assert plan.groups[0].diagnosis_terms == ["rasmussen"]


def test_near_surname_needs_one_clear_candidate() -> None:
    assert rv.near_surname("rasmussehn") == "rasmussen"
    assert rv.near_surname("huntingtn") == "huntington"
    assert rv.near_surname("poldrack") is None
    assert rv.near_surname("wilso") is None          # too short to guess


def test_disease_name_without_such_an_author_moves_even_after_by(names: MetadataRetriever) -> None:
    plan = _plan(["huntington"])
    pp._names_that_are_conditions(plan, "datasets by huntington", names)
    assert plan.author_names == []


def test_named_person_stays_an_author(names: MetadataRetriever) -> None:
    plan = _plan(["williams"])
    pp._names_that_are_conditions(plan, "datasets by williams", names)
    assert plan.author_names == ["williams"]
    assert plan.groups == []


def test_ordinary_author_is_untouched(names: MetadataRetriever) -> None:
    plan = _plan(["Poldrack"])
    pp._names_that_are_conditions(plan, "poldrack", names)
    assert plan.author_names == ["Poldrack"]


def test_bare_task_name_becomes_a_scan_term(names: MetadataRetriever) -> None:
    plan = _plan(["Iowa gambling task"])
    pp._names_that_are_conditions(plan, "iowa gambling task", names)
    assert plan.author_names == []
    assert plan.scan_requirements[0].scan_terms == ["Iowa gambling task"]


def test_reference_files_are_part_of_the_pipeline_version() -> None:
    from services.stores import interpretations
    names_ = {p.name for p in interpretations.version_paths()}
    assert {"reference_vocabulary.py", "reference_vocabulary.yaml"} <= names_
