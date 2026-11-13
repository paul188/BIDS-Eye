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
