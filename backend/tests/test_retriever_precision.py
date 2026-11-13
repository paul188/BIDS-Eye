"""Precision of the vocabulary resolver (RAG/retriever.py).

Each case is a term that used to resolve to its own concept plus an unrelated
one. The tests run against the real value_mappings.yaml; none of these terms
reaches the embedding fallback.
"""

from __future__ import annotations

import pytest

import retriever
from retriever import _plural_variants, _resolve_yaml_term


@pytest.fixture(autouse=True)
def _no_embedding_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly instead of downloading models if a term misses fuzzy matching."""

    def fail(field: str, term: str):
        raise AssertionError(f"{field}/{term!r} fell through to the embedding fallback")

    monkeypatch.setattr(retriever, "_embedding_fallback_scored", fail)


FMRI_SUFFIXES = {"fmri_bold", "bold_map", "bold_reference", "single_band_reference"}


def test_structural_mri_does_not_resolve_to_fmri_suffixes() -> None:
    suffixes = set(_resolve_yaml_term("suffix", "structural MRI"))

    assert "t1_weighted_mri" in suffixes
    assert not suffixes & FMRI_SUFFIXES
    assert "cerebral_blood_flow" not in suffixes
    assert _resolve_yaml_term("datatype", "structural MRI") == ["anatomical_mri"]


def test_fmri_does_not_include_perfusion_asl() -> None:
    assert _resolve_yaml_term("datatype", "fMRI") == ["functional_mri"]
    assert _resolve_yaml_term("suffix", "fMRI") == ["fmri_bold"]


@pytest.mark.parametrize("term", ["Parkinson's", "Parkinson’s", "parkinsons", "Parkinson's disease"])
def test_parkinsons_excludes_animal_models(term: str) -> None:
    codes = _resolve_yaml_term("diagnosis", term)

    assert codes[0] == "parkinsons_disease"
    assert not [code for code in codes if "knockdown" in code]
    assert set(codes) == {"parkinsons_disease", "parkinsons_disease_mci", "de_novo_parkinsons"}


def test_shared_word_matches_do_not_join_stronger_matches() -> None:
    # "disorder" alone used to pull in depression, OCD, Alzheimer's, ALS, ...
    assert _resolve_yaml_term("diagnosis", "hyperactivity disorder") == ["adhd"]
    # "mri" / "tomography" alone used to pull in the fMRI suffix group.
    assert _resolve_yaml_term("suffix", "computed tomography") == ["ct_scan"]


def test_scan_terms_end_to_end() -> None:
    # MetadataRetriever needs a name index only for author/funding lookups.
    resolver = retriever.MetadataRetriever.__new__(retriever.MetadataRetriever)

    resolved = resolver.retrieve_scan_terms(["structural MRI", "fMRI"])

    assert resolved["datatype"] == {
        "structural MRI": ["anatomical_mri"],
        "fMRI": ["functional_mri"],
    }
    assert not set(resolved["suffix"]["structural MRI"]) & FMRI_SUFFIXES
    assert "task" not in resolved


def test_possessive_variant_is_tried_for_exact_match() -> None:
    assert "parkinson" in _plural_variants("parkinson's")
    assert _plural_variants("epilepsy")[0] == "epilepsy"


@pytest.mark.parametrize(
    "term",
    [
        "Down syndrome",       # used to give west_syndrome_history ("syndrome")
        "Rett syndrome",
        "conduct disorder",    # used to give unipolar_depression ("disorder")
        "cerebral palsy",      # used to give ALS ("als" inside "palsy")
        "phobia",              # used to give panic_disorder ("agoraphobia")
    ],
)
def test_no_code_rather_than_a_wrong_one(term: str) -> None:
    # An unresolved term falls back to a free-text filter downstream; a wrong
    # code silently returns datasets about another disease.
    with pytest.raises(AssertionError, match="embedding fallback"):
        _resolve_yaml_term("diagnosis", term)


def test_abbreviation_label_must_match_a_whole_word() -> None:
    # "chi" (closed head injury) is a prefix of "children".
    assert _resolve_yaml_term("diagnosis", "TD children") == ["typically_developing"]


def test_filler_words_do_not_steer_fuzzy_matching() -> None:
    # Full phrase is one letter from "MEG recording".
    assert _resolve_yaml_term("suffix", "iEEG recording") == ["intracranial_eeg"]


def test_hyphen_and_space_spellings_are_equal() -> None:
    assert _resolve_yaml_term("datatype", "near infrared spectroscopy") == ["fnirs"]
    # Identical to a low-weight synonym once separators are normalised: an
    # exact match, not a reason to pull in the whole resting-state group.
    assert _resolve_yaml_term("task", "resting-state eyes open") == ["resting_state_eyes_open"]


def test_tie_prefers_label_made_of_the_users_words() -> None:
    # Ties with "resting-state tms-eeg", which adds a word the user did not
    # type; that label used to win and return the TMS code alone.
    codes = _resolve_yaml_term("task", "resting state EEG")
    assert codes[0] == "resting_state"
    assert len(codes) > 1


def test_animal_models_need_an_animal_word() -> None:
    assert not [c for c in _resolve_yaml_term("diagnosis", "parkinsonism") if "model" in c]
    assert "mir5_atp10b_knockdown_model" in _resolve_yaml_term("diagnosis", "Parkinson's mouse model")


def test_autism_is_not_the_sibling_cohort() -> None:
    assert _resolve_yaml_term("diagnosis", "autism") == ["autism_spectrum_disorder"]


@pytest.mark.parametrize(
    "term",
    ["reward processing", "emotion processing", "reward", "memory", "language", "music"],
)
def test_task_terms_resolve_to_no_suffix(term: str) -> None:
    # "reward processing" used to match the suffix bucket
    # entities_and_processing_keys through "processing"; that code is in almost
    # every dataset, so "Datasets with reward processing tasks" returned all.
    with pytest.raises(AssertionError, match="embedding fallback"):
        _resolve_yaml_term("suffix", term)


@pytest.mark.parametrize("term", ["roi", "atlas", "channels", "entities and processing keys"])
def test_bucket_suffix_is_never_returned(term: str) -> None:
    assert "entities_and_processing_keys" not in _resolve_yaml_term("suffix", term)
