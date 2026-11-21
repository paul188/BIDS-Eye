"""Hand cases for expander.py. Needs spaCy with en_core_web_sm on PYTHONPATH:

    PYTHONPATH=~/.cache/bidseye_grammar/nlplib python3 -m pytest benchmarks/phrases/test_expander.py
"""

import sys
from pathlib import Path

import pytest

spacy = pytest.importorskip("spacy")
sys.path.insert(0, str(Path(__file__).parent))
from expander import abbreviations_in, expand_sentence, index_text  # noqa: E402

NLP = spacy.load("en_core_web_sm", disable=["ner"])
ATTESTED = {("temporal", "pole"), ("healthy", "adult"), ("functional", "network"), ("visual", "stimulus"),
            ("temporal", "sulcus"), ("word", "naming")}
COMPOUNDS = {("movie", "watch"), ("music", "listen"), ("motion", "correct"), ("stimulus", "present")}


def toks(text):
    doc = NLP(text)
    return [[t.text, t.lemma_.lower(), t.pos_, t.dep_, t.head.i, 1 if t.whitespace_ else 0] for t in doc]


def texts(sentence, kind=None, **kw):
    return {e.text.lower() for e in expand_sentence(toks(sentence), **kw) if kind is None or e.kind == kind}


@pytest.mark.parametrize("sentence, expected", [
    ("Activity in the temporal and frontal cortices was reduced.", {"temporal cortices"}),
    ("Rethinking model-based and model-free influences on effort.", {"model-based influences"}),
    ("Volumes of the temporal, parietal, and occipital lobes were measured.",
     {"temporal lobes", "parietal lobes"}),
    ("We found effects in the superior and middle temporal gyri.", {"superior temporal gyri"}),
    ("Both structural and functional connectivity were computed.", {"structural connectivity"}),
    ("We segmented gray and white matter.", {"gray matter"}),
])
def test_lists_with_a_shared_noun(sentence, expected):
    assert expected <= texts(sentence, "list")


def test_a_list_of_whole_nouns_is_not_a_list_of_modifiers():
    got = texts("Masks of the amygdala, hippocampus and temporal lobe were drawn.", "list", attested=ATTESTED)
    assert "amygdala lobe" not in got and "hippocampus lobe" not in got
    got = texts("The file contains age, gender, body mass index and performance.", "list", attested=ATTESTED)
    assert "age index" not in got and "gender index" not in got


@pytest.mark.parametrize("sentence, expected", [
    ("We scanned five healthy, young adults.", "healthy adults"),
    ("Segmentation of grey + white matter was done.", "grey matter"),
])
def test_list_variants(sentence, expected):
    assert expected in texts(sentence, "list", attested=ATTESTED | {("healthy", "adult"), ("grey", "matter")})


def test_noun_items_need_an_attested_pair():
    s = "Both word and picture naming were tested."
    assert "word naming" in texts(s, "list", attested=ATTESTED)
    assert texts(s, "list") == set()


def test_paths_and_numbers_are_not_expanded():
    assert texts("Files are in s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R2 and func/sub-01_bold.nii.gz.") == set()
    assert texts("An interval between 1.5 and 4 seconds.", "list") == set()


def test_shared_modifier_only_when_attested():
    s = "The temporal lobe and pole were excluded."
    assert "temporal pole" in texts(s, "shared_modifier", attested=ATTESTED)
    assert texts(s, "shared_modifier") == set()
    assert "temporal hippocampus" not in texts("The temporal lobe and hippocampus were excluded.",
                                               "shared_modifier", attested=ATTESTED)


def test_skipped_modifier_only_when_attested():
    s = "We recruited 40 healthy young adults."
    assert "healthy adults" in texts(s, "skip", attested=ATTESTED | {("verbal", "memory")})
    assert texts(s, "skip") == set()
    assert "verbal memory" not in texts("A verbal working memory task.", "skip", attested=ATTESTED | {("verbal", "memory")})


def test_no_skip_over_a_region_word():
    got = texts("Responses in the left superior temporal gyrus.", "skip",
                attested={("superior", "gyrus"), ("left", "gyrus"), ("temporal", "gyrus")})
    assert "superior gyrus" not in got and "left gyrus" not in got


def test_inner_word_of_a_list_item_must_belong_to_the_noun():
    got = texts("Scores in vocabulary and mental imagery tasks are available.", "list",
                attested={("mental", "task"), ("vocabulary", "task")})
    assert "mental tasks" not in got


def test_noun_compound_is_not_read_as_verb_and_object():
    assert "phase encoding" not in texts("Phase 1 (encoding phase): 1s", "verb_object", compounds={("phase", "encod")})


def test_possessive_and_parenthesis():
    assert "alzheimer disease" in texts("Patients with Alzheimer's disease were scanned.", "possessive")
    assert "ventral temporal cortex" in texts("Decoding in the ventral temporal (VT) cortex worked.", "parenthesis")


def test_postposed_and_verb_object():
    assert "eyes closed" in texts("Data were recorded with eyes open or closed.", "postposed")
    got = texts("Participants watched four short movies and listened to music.", "verb_object", compounds=COMPOUNDS)
    assert {"movies watched", "music listened"} <= got
    assert "head motion corrected" in texts("The data were corrected for head motion.", "verb_object",
                                            compounds=COMPOUNDS)
    assert "stimuli presented" in texts("Visual stimuli were presented on a screen.", "verb_object",
                                        compounds=COMPOUNDS)
    assert texts("These will be shared with upcoming publications.", "verb_object", compounds=COMPOUNDS) == set()
    assert texts("Each run started and ended with fixation.", "postposed") == set()


def test_fused_spellings():
    assert "frontoparietal" in texts("The fronto-parietal network was active.", "fused")
    assert "fronto-parietal" in texts("The frontoparietal network was active.", "fused")


def test_abbreviations_defined_in_the_text():
    text = "We measured white matter (WM) integrity. WM volume was lower in patients."
    abbr = abbreviations_in(text)
    assert abbr == {"WM": "white matter"}
    assert "white matter volume" in texts("WM volume was lower in patients.", "abbreviation", abbreviations=abbr)


def test_index_text_keeps_expansions_apart():
    from expander import Expansion
    assert index_text([Expansion("list", "temporal cortices", 0, 1), Expansion("list", "frontal cortices", 0, 1)],
                      gap=2) == "temporal cortices the the frontal cortices"
