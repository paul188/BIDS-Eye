"""Other spellings and numbers of a searched word (RAG/word_forms.py spelling_variants,
plural_variants, search_variants)."""

from __future__ import annotations

import pytest

import word_forms
from services.database_stats.database_stats import ColumnCoverage, database_stats
from services.fulltext import tsquery, vectors, word_frequencies
from word_forms import plural_variants, search_variants, spelling_variants

# Both directions; the pairs seen in the catalogue on 2026-11-21.
PAIRS = [
    ("behaviour", "behavior"), ("behavioural", "behavioral"), ("behaviours", "behaviors"),
    ("colour", "color"), ("tumour", "tumor"), ("neighbouring", "neighboring"), ("unfavourable", "unfavorable"),
    ("centre", "center"), ("fibre", "fiber"), ("litres", "liters"), ("millimetre", "millimeter"),
    ("multicentre", "multicenter"), ("randomised", "randomized"), ("normalisation", "normalization"),
    ("characterising", "characterizing"), ("analysed", "analyzed"), ("analyse", "analyze"),
    ("labelled", "labeled"), ("modelling", "modeling"), ("signalled", "signaled"), ("travelled", "traveled"),
    ("catalogue", "catalog"), ("analogue", "analog"), ("grey", "gray"), ("greyscale", "grayscale"),
    ("programme", "program"), ("licence", "license"), ("defence", "defense"), ("sulphate", "sulfate"),
    ("haemodynamic", "hemodynamic"), ("haemorrhage", "hemorrhage"), ("paediatric", "pediatric"),
    ("anaesthesia", "anesthesia"), ("anaemia", "anemia"), ("aetiology", "etiology"), ("ischaemic", "ischemic"),
    ("oedema", "edema"), ("lymphoedema", "lymphedema"), ("oestrogen", "estrogen"), ("foetal", "fetal"),
    ("faecal", "fecal"), ("diarrhoea", "diarrhea"), ("hypoglycaemia", "hypoglycemia"),
    ("localiser", "localizer"),
]

# Words that have no other spelling. A variant of these would be junk at best
# and another word at worst ("parametres" stems to "parametr", as "parametric").
UNCHANGED = """
motor factor error doctor monitor water after number filter letter better enter later order cluster
parameter parameters diameter acre size prize exercise promise advise revise rise raise noise surprise
compromise otherwise praise cruise demise hemisphere chemotherapy theme orchestra pedestrian restriction fetch planes called
installed filled controlled revealed smaller analyses analyzes backlog hedgehog four hour detour data
""".split()


@pytest.mark.parametrize("uk, us", PAIRS)
def test_british_and_american_find_each_other(uk: str, us: str) -> None:
    assert us in spelling_variants(uk)
    assert uk in spelling_variants(us)


@pytest.mark.parametrize("word", UNCHANGED)
def test_words_without_another_spelling_stay_alone(word: str) -> None:
    assert spelling_variants(word) == ()


def test_two_changes_in_one_word() -> None:
    assert "anesthetized" in spelling_variants("anaesthetised")
    assert "anaesthetised" in spelling_variants("anesthetized")


@pytest.fixture
def full_text():
    database_stats.coverage = ColumnCoverage.full(())
    word_frequencies.set_frequency_lookup(None)
    yield
    word_frequencies.set_frequency_lookup(None)


def test_a_search_word_matches_in_both_spellings(full_text) -> None:
    terms = [[("behavioural",), ("problems",)]]
    assert tsquery.build_filter_tsquery(terms, "all") == "(behavioural | behavioral) & problems"
    near = tsquery.build_filter_tsquery(terms, "near")
    assert "(behavioural | behavioral) <1> problems" in near
    assert "problems <1> (behavioural | behavioral)" in near
    assert tsquery.build_filter_tsquery([[("tumour",)]], "short") == "(tumour:AB | tumor:AB)"
    assert tsquery.build_filter_tsquery([[("colour", "coded")]], "all") == "((colour | color) <-> coded)"
    assert tsquery.is_safe_tsquery(near)


def test_words_without_another_spelling_keep_their_query(full_text) -> None:
    assert tsquery.build_filter_tsquery([[("motor",), ("task",)]], "all") == "motor & task"


def test_the_reviewed_list_applies_to_the_other_spelling(tmp_path, monkeypatch) -> None:
    path = tmp_path / "word_forms.yaml"
    path.write_text("exact:\n  - forms: [localizer, localizers]\n    excludes: [local]\n")
    monkeypatch.setattr(word_forms, "_PATH", path)
    word_forms._reviewed.cache_clear()
    try:
        assert word_forms.reviewed("localiser") == (True, ("localiser", "localizer", "localizers"))
    finally:
        word_forms._reviewed.cache_clear()


def test_the_exact_guard_accepts_both_spellings(tmp_path, monkeypatch, full_text) -> None:
    path = tmp_path / "word_forms.yaml"
    path.write_text("exact:\n  - forms: [localizer, localizers]\n    excludes: [local]\n")
    monkeypatch.setattr(word_forms, "_PATH", path)
    monkeypatch.setenv(vectors.EXACT_ENV, "true")
    word_forms._reviewed.cache_clear()
    try:
        guard = tsquery.exact_guard([("localizer",)])
        assert guard == "(localizer | localiser | localizers | localisers)"
    finally:
        word_forms._reviewed.cache_clear()


# Latin and Greek plurals, both directions.
PLURALS = [
    ("stimulus", "stimuli"), ("gyrus", "gyri"), ("sulcus", "sulci"), ("nucleus", "nuclei"),
    ("hippocampus", "hippocampi"), ("locus", "loci"), ("focus", "foci"), ("radius", "radii"),
    ("spectrum", "spectra"), ("maximum", "maxima"), ("stratum", "strata"), ("criterion", "criteria"),
    ("phenomenon", "phenomena"), ("ganglion", "ganglia"), ("matrix", "matrices"), ("vertebra", "vertebrae"),
    ("amygdala", "amygdalae"), ("schema", "schemata"), ("analysis", "analyses"), ("hypothesis", "hypotheses"),
    ("diagnosis", "diagnoses"),
]
# Only from the plural: the stem of the -ices form also matches indicated, vertical, cortical.
PLURAL_TO_SINGULAR_ONLY = [("cortices", "cortex"), ("indices", "index"), ("vertices", "vertex")]

NO_PLURAL = """
status virus campus census corpus various previous anxious minus multi semi anti mini peri hemi data
datum media medium complex reflex dementia insomnia alpha beta gamma theta delta area trauma basis bases
axis processes responses cases phases index vertex cortex fmri rsfmri dmri smri rhesus tinnitus bolus
plexus globus pallidus coeruleus nifti afni helsinki biosemi gorgolewski interstimulus
""".split()


@pytest.mark.parametrize("singular, plural", PLURALS)
def test_latin_singular_and_plural_find_each_other(singular: str, plural: str) -> None:
    assert plural in plural_variants(singular)
    assert singular in plural_variants(plural)


@pytest.mark.parametrize("plural, singular", PLURAL_TO_SINGULAR_ONLY)
def test_ices_plurals_find_their_singular(plural: str, singular: str) -> None:
    assert singular in plural_variants(plural)
    assert plural not in plural_variants(singular)


@pytest.mark.parametrize("word", NO_PLURAL)
def test_words_without_a_latin_plural_stay_alone(word: str) -> None:
    assert plural_variants(word) == ()


def test_search_variants_combine_spelling_and_number() -> None:
    assert set(search_variants("haematoma")) == {"hematoma"}
    assert {"behavior"} == set(search_variants("behaviour"))
    assert {"stimuli"} == set(search_variants("stimulus"))


def test_a_search_word_matches_in_both_numbers(full_text) -> None:
    assert tsquery.build_filter_tsquery([[("visual",), ("stimuli",)]], "all") == "visual & (stimuli | stimulus)"


IRREGULAR = [("man", "men"), ("woman", "women"), ("child", "children"), ("grandchild", "grandchildren"),
             ("mouse", "mice"), ("foot", "feet"), ("tooth", "teeth"), ("calf", "calves"), ("half", "halves")]


@pytest.mark.parametrize("singular, plural", IRREGULAR)
def test_irregular_singular_and_plural_find_each_other(singular: str, plural: str) -> None:
    assert plural in plural_variants(singular)
    assert singular in plural_variants(plural)


@pytest.mark.parametrize("word", "human humans person people life lives leaf leaves manual germany".split())
def test_no_irregular_plural_for_lookalikes(word: str) -> None:
    assert plural_variants(word) == ()


def test_a_search_for_children_finds_child(full_text) -> None:
    assert tsquery.build_filter_tsquery([[("children",)]], "all") == "(children | child)"


def test_a_word_list_names_each_form_once(full_text) -> None:
    query = tsquery.build_filter_tsquery([[("mouse",)], [("mice",)], [("rat",)]], "short")
    assert query == "mouse:AB | mice:AB | rat:AB"
