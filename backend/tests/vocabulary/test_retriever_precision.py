"""Precision of the vocabulary resolver (backend/vocabulary/retriever.py).

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
    "term,official",
    [
        ("Down syndrome", "down_syndrome"),         # used to give west_syndrome_history ("syndrome")
        ("Rett syndrome", "rett_syndrome"),
        ("conduct disorder", "conduct_disorder"),   # used to give unipolar_depression ("disorder")
        ("cerebral palsy", "cerebral_palsy"),       # used to give ALS ("als" inside "palsy")
        ("phobia", "phobic_disorders"),             # used to give panic_disorder ("agoraphobia")
    ],
)
def test_no_curated_code_of_another_disease(term: str, official: str) -> None:
    # A wrong code silently returns datasets about another disease. These
    # diseases are not in the curated vocabulary: they resolve to their own
    # official concept (backend/vocabulary/reference_vocabulary.py, MeSH), whose code no
    # participant row carries; the diagnosis line then searches its terms.
    assert _resolve_yaml_term("diagnosis", term) == [official]


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


# ── Task terms: the same result however the request is phrased ────────────────

@pytest.mark.parametrize(
    "phrasings",
    [
        # Was 45 codes (construct) vs 1 code (synonym of decision_task_general).
        ["decision making", "decision making tasks", "decision making task",
         "decision-making", "decision-making paradigm"],
        # Was 1 code (a raw dataset label) vs 22 codes (construct).
        ["language", "language tasks", "language task"],
        # Was 166 vs 38 vs 52 codes.
        ["memory", "memory tasks", "memory assessments"],
        # Was 21 codes vs 1 (synonym of the monetary incentive delay task).
        ["reward processing", "reward processing tasks"],
        ["reward", "reward tasks"],
        # Was 51 codes vs 4.
        ["social cognition", "social cognition tasks"],
        ["working memory", "working memory tasks"],
        ["lexical decision", "lexical decision task"],
        ["n-back", "n back", "n-back task"],
    ],
)
def test_task_phrasings_resolve_identically(phrasings: list) -> None:
    results = {term: frozenset(_resolve_yaml_term("task", term)) for term in phrasings}
    assert len(set(results.values())) == 1, {t: len(codes) for t, codes in results.items()}
    assert results[phrasings[0]]


def test_task_core_is_combined_from_construct_and_task_synonym() -> None:
    codes = _resolve_yaml_term("task", "decision making tasks")
    assert "decision_task_general" in codes      # synonym "decision making task"
    assert "columbia_card_task" in codes         # through the construct
    codes = _resolve_yaml_term("task", "language")
    assert "language_control" in codes           # raw dataset label "language"
    assert "picture_naming_task" in codes        # through the construct


def test_named_task_is_not_widened_by_its_task_suffix() -> None:
    # "theory of mind task" is also a synonym of the social-cognition group.
    assert len(_resolve_yaml_term("task", "theory of mind task")) == \
        len(_resolve_yaml_term("task", "theory of mind")) < 10
    # The specific task named with its suffix is still found.
    assert "lexical_decision_task" in _resolve_yaml_term("task", "lexical decision")


def test_construct_match_needs_more_than_the_word_task() -> None:
    # "stroop task" matched the construct "task set reconfiguration" through
    # "task" alone and returned 51 unrelated tasks.
    assert _resolve_yaml_term("task", "stroop task") == _resolve_yaml_term("task", "stroop")
    assert len(_resolve_yaml_term("task", "stroop task")) < 5
    assert _resolve_yaml_term("task", "tasks") == []


# ── Ambiguous terms keep every reading ────────────────────────────────────────

def test_loader_keeps_every_code_of_a_shared_term() -> None:
    from yaml_to_llamaindex import leaf_codes_db, leaf_db

    codes = leaf_codes_db["diagnosis"]["typically developing"]
    assert set(codes) == {"typically_developing", "neurotypical"}
    assert leaf_db["diagnosis"]["typically developing"] in codes


def test_term_listed_under_two_concepts_resolves_to_both() -> None:
    # The loader used to keep only the concept listed last in the vocabulary,
    # so "PNT" and "SST" resolved to one unrelated task each.
    assert "picture_naming_task" in _resolve_yaml_term("task", "PNT")
    assert "stop_signal_task" in _resolve_yaml_term("task", "SST")
    assert set(_resolve_yaml_term("diagnosis", "typically developing")) == {
        "typically_developing", "neurotypical",
    }


def test_shared_term_splits_the_score_between_its_concepts() -> None:
    scored = dict(retriever._resolve_yaml_term_scored("diagnosis", "typically developing"))
    assert scored == {"typically_developing": 0.5, "neurotypical": 0.5}
    # A term with one concept keeps the full score.
    assert dict(retriever._resolve_yaml_term_scored("diagnosis", "schizophrenia"))[
        "schizophrenia"
    ] == 1.0


def test_term_inside_labels_of_two_concepts_resolves_to_both() -> None:
    # "fmri bold signal" (fmri_bold) and "bold signal map" (bold_map) contain
    # the term equally well; so do the PTSD and brain-injury labels for "trauma".
    assert {"fmri_bold", "bold_map"} <= set(_resolve_yaml_term("suffix", "BOLD signal"))
    codes = _resolve_yaml_term("diagnosis", "trauma")
    assert "post_traumatic_stress_disorder" in codes
    assert "traumatic_brain_injury" in codes


def test_tied_labels_split_the_score_by_how_many_lead_to_a_code() -> None:
    scored = dict(retriever._resolve_yaml_term_scored("suffix", "BOLD signal"))
    assert scored["fmri_bold"] == scored["bold_map"]
    assert 0.4 < scored["fmri_bold"] < 0.5      # half of the 0.95 match


def test_short_term_does_not_match_the_start_of_a_construct(monkeypatch: pytest.MonkeyPatch) -> None:
    # "emo" scored 90 against the construct "emotion", whose only linked task
    # is meditation_general: "emo studies coauthored by Morgenroth" filtered on
    # meditation and missed both Emo-FilM datasets.
    monkeypatch.setattr(retriever, "_embedding_fallback_scored", lambda field, term: [])
    assert retriever._resolve_via_constructs("emo") == []
    assert retriever._match_task_term("emo") is None
    assert _resolve_yaml_term("task", "emo") == []
    # Five letters and more still find the start of a word.
    assert "directed_forgetting_task" in _resolve_yaml_term("task", "forgetting")


def test_construct_with_few_tasks_counts_only_next_to_a_named_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retriever, "_embedding_fallback_scored", lambda field, term: [])
    # "instinct" is linked to pavlovian_conditioning alone; alone it is no filter.
    assert len(retriever._resolve_via_constructs("instinct", fuzzy=False)) < retriever._MIN_CONSTRUCT_TASKS
    assert _resolve_yaml_term("task", "instinct") == []
    # "emotion" names the task emotion_general; the thin link only widens it.
    assert "emotion_general" in _resolve_yaml_term("task", "emotion")
    # Constructs linked to many tasks are unchanged.
    assert len(_resolve_yaml_term("task", "decision making")) > 20


@pytest.mark.parametrize("term", ["brain trauma", "neurotrauma", "head trauma", "acquired brain injuries"])
def test_brain_trauma_is_traumatic_brain_injury(term: str) -> None:
    # These were synonyms of the group sensory_metabolic_trauma, which also
    # holds the blindness and perceptual-distortion codes: "datasets containing
    # brain trauma" confirmed cannabis users and bilinguals (live, 2026-11-15).
    assert _resolve_yaml_term("diagnosis", term) == ["traumatic_brain_injury", "brain_contusion"]


@pytest.mark.parametrize("term", ["glioblastoma", "GBM"])
def test_glioblastoma_is_grade_4_glioma(term: str) -> None:
    # Was a synonym of tumors_lesions: meningioma, CNS lymphoma, brain abscess.
    assert _resolve_yaml_term("diagnosis", term) == ["grade_4_glioma"]


@pytest.mark.parametrize("term", ["TLE", "temporal lobe epilepsy", "mTLE", "mesial temporal lobe epilepsy",
                                  "immune-mediated temporal lobe epilepsy"])
def test_temporal_lobe_epilepsy_is_not_every_epilepsy(term: str) -> None:
    # TLE was a synonym of epilepsy and searched all 13 epilepsy codes:
    # "20 patients with TLE" confirmed an FCD and a paediatric dataset (2026-11-18).
    assert _resolve_yaml_term("diagnosis", term) == ["temporal_lobe_epilepsy", "hippocampal_sclerosis"]


def test_epilepsy_still_covers_temporal_lobe_epilepsy() -> None:
    codes = _resolve_yaml_term("diagnosis", "epilepsy")
    assert codes[0] == "epilepsy"
    assert {"temporal_lobe_epilepsy", "hippocampal_sclerosis", "focal_cortical_dysplasia"} <= set(codes)


ASL_FILES = ["arterial_spin_labelling", "cerebral_blood_flow", "m0_scan", "asl_reference", "asl_context", "att"]


@pytest.mark.parametrize("term", ["ASL", "arterial spin labelling", "pCASL"])
def test_asl_includes_the_files_only_asl_has(term: str) -> None:
    # "dataset with ASL" missed ds005529, which has only M0 scans and CBF maps (2026-11-21).
    assert _resolve_yaml_term("suffix", term) == ASL_FILES


def test_cbf_and_m0_alone_stay_narrow() -> None:
    assert _resolve_yaml_term("suffix", "CBF") == ["cerebral_blood_flow"]
    assert _resolve_yaml_term("suffix", "M0") == ["m0_scan"]


@pytest.mark.parametrize("term", ["APT", "APTw", "amide proton transfer"])
def test_apt_is_the_scan_type(term: str) -> None:
    # "Dataset with epilepsy patients and APT" asked for the task tactile
    # stimulation, through its synonym "APTS" (live, 2026-11-23).
    assert _resolve_yaml_term("suffix", term) == ["apt"]
    if retriever.is_shorthand(term):
        assert _resolve_yaml_term("task", term) == []


@pytest.mark.parametrize("term", ["EUPD", "emotionally unstable personality disorder"])
def test_eupd_is_borderline_personality_disorder(term: str) -> None:
    # ds000214 "EUPD Cyberball" was missing from a borderline personality search.
    assert _resolve_yaml_term("diagnosis", term) == ["borderline_personality_disorder"]


# ── Shorthands resolve through a listing of exactly that shorthand ─────────────

@pytest.mark.parametrize("field, term, codes", [
    ("task", "APT", []),                   # was tactile_stimulation, through the plural "APTS"
    ("task", "ASL", []),                   # was two statistical-learning tasks
    ("diagnosis", "MRI", []),              # was two epilepsy codes
    ("suffix", "CEST", []),                # no code; backend/vocabulary/shorthands.yaml writes it out
    ("suffix", "apt", ["apt"]),            # typed in small letters, still the shorthand
    ("suffix", "CTs", ["ct_scan"]),        # plural of a capitalised shorthand
    ("diagnosis", "TBI", ["traumatic_brain_injury"]),
    # Time-of-flight images are angiograms or venograms; the gold request for
    # "TOF or SWI images" lost one of eight datasets as a text search.
    ("suffix", "TOF", ["mr_venography", "mr_angiography"]),
    ("diagnosis", "MS", ["multiple_sclerosis"]),
])
def test_a_shorthand_resolves_only_through_its_listing(field: str, term: str, codes: list) -> None:
    assert _resolve_yaml_term(field, term) == codes


def test_a_shorthand_with_two_listings_keeps_both() -> None:
    # Parkinson's disease and the prisoner's dilemma: nothing in "PD" says which.
    assert "parkinsons_disease" in _resolve_yaml_term("diagnosis", "PD")
    assert _resolve_yaml_term("task", "PD") == ["strategic_interaction_game"]
    assert retriever.shorthand_meanings("PD") == [
        "Parkinson's disease (diagnosis)", "Strategic interaction game (task)"]


def test_an_english_word_in_small_letters_is_not_a_shorthand() -> None:
    assert retriever.is_shorthand("ALL") and not retriever.is_shorthand("all")
    assert retriever.is_shorthand("tle") and retriever.is_shorthand("mTLE")
    assert not retriever.is_shorthand("resting state") and not retriever.is_shorthand("memory")


DBS_TASKS = ["deep_brain_stimulation_high_frequency", "deep_brain_stimulation_low_frequency",
             "deep_brain_stimulation_variable_frequency"]


@pytest.mark.parametrize("term", ["DBS", "dbs", "deep brain stimulation"])
def test_dbs_is_deep_brain_stimulation(term: str) -> None:
    # "DBS" was a synonym of the whole neuromodulation group: 42 tasks,
    # neurofeedback and TMS among them (2026-11-23).
    assert _resolve_yaml_term("task", term) == DBS_TASKS
    assert set(DBS_TASKS) <= set(_resolve_yaml_term("task", "neuromodulation"))
    assert retriever.shorthand_meanings("DBS") == ["Deep brain stimulation (task)"]


def test_meanings_come_from_the_listings() -> None:
    assert retriever.shorthand_meanings("APT") == ["amide proton transfer imaging (APT) (scan type)"]
    assert retriever.shorthand_meanings("ERP") == ["event-related potential"]
    assert retriever.shorthand_meanings("XQZ") == []


TMS_EEG_TASKS = ["resting_state_tms_eeg", "tms_eeg_stimulation_variant1", "tms_eeg_stimulation_variant2"]


@pytest.mark.parametrize("term", ["TMS", "tms", "transcranial magnetic stimulation"])
def test_tms_is_the_tms_tasks(term: str) -> None:
    # "TMS" was a synonym of the whole neuromodulation group: 42 tasks,
    # neurofeedback and deep brain stimulation among them (2026-11-23).
    codes = _resolve_yaml_term("task", term)
    assert codes[0] == "transcranial_magnetic_stimulation" and len(codes) == 10
    assert set(TMS_EEG_TASKS) < set(codes) and "tms_training_block_1" in codes
    assert not any("neurofeedback" in c or "deep_brain" in c for c in codes)
    assert set(codes) <= set(_resolve_yaml_term("task", "neuromodulation"))


def test_tms_eeg_is_the_three_tms_eeg_tasks() -> None:
    # Was a synonym of "physiological or neurostimulation task": 67 tasks, meditation among them.
    assert _resolve_yaml_term("task", "TMS-EEG") == TMS_EEG_TASKS


@pytest.mark.parametrize("term", ["tDCS", "tdcs", "transcranial direct current stimulation"])
def test_tdcs_is_electrical_stimulation(term: str) -> None:
    # Was a synonym of two groups: 80 tasks. No task code is tDCS alone; it
    # is electrical stimulation, with tES, tACS and tRNS.
    assert _resolve_yaml_term("task", term) == ["electrical_stimulation"]
