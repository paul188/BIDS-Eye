"""A phrase is never shortened to one broad word (search-quality review 2026-11-17, point 1)."""
import preprocess
import text_terms
from services.fulltext import tsquery, word_frequencies
from services.uncertain import text as tier_text


def _counts(table):
    """A frequency lookup answering from *table* (tsquery → datasets)."""
    def lookup(queries, guards=None):
        return [table[q] for q in queries]
    return lookup


# ── uncertain tier: the rarer half must be rare ─────────────────────────────

def test_a_common_remainder_is_no_weak_evidence(monkeypatch):
    # "visual & cortex" → "cortex" (142 datasets) said 'probable' for any README with "cortex".
    monkeypatch.setattr(word_frequencies, "_lookup_counts",
                        _counts({"visual": 600, "cortex": 142}))
    assert tier_text.relax_tsquery("visual & cortex") is None
    # The scope still widens when the words stay.
    assert tier_text.relax_tsquery("visual:AB & cortex:AB") == "visual & cortex"


def test_a_rare_remainder_stands_for_the_term(monkeypatch):
    monkeypatch.setattr(word_frequencies, "_lookup_counts",
                        _counts({"memory": 400, "consolidation": 10}))
    assert tier_text.relax_tsquery("memory & consolidation") == "consolidation"


def test_dropping_only_general_words_needs_no_rare_remainder(monkeypatch):
    monkeypatch.setattr(word_frequencies, "_lookup_counts",
                        _counts({"emotional": 94, "state": 900}))
    assert tier_text.relax_tsquery("emotional & state") == "emotional"


def test_without_counts_the_shortening_stays():
    word_frequencies.set_frequency_lookup(None)
    try:
        assert tier_text.relax_tsquery("memory & consolidation") == "consolidation"
    finally:
        word_frequencies.reset_frequency_lookup()


# ── loose text: a general word is never the one kept ────────────────────────

def test_loose_core_keeps_the_specific_word(monkeypatch):
    # In the catalogue "syndrome" (11) is rarer than "bipolar" (43); together 1.
    monkeypatch.setattr(word_frequencies, "_lookup_counts",
                        _counts({"bipolar": 43, "syndrome": 11, "bipolar & syndrome": 1}))
    assert tsquery._loose_core([["bipolar"], ["syndrome"]]) == [["bipolar"]]


def test_loose_core_without_counts_skips_the_general_word():
    word_frequencies.set_frequency_lookup(None)
    try:
        assert tsquery._loose_core([["bipolar"], ["syndrome"]]) == [["bipolar"]]
    finally:
        word_frequencies.reset_frequency_lookup()


# ── diagnosis fallback: strings are searched whole ───────────────────────────

def test_diagnosis_strings_keep_all_their_words():
    assert text_terms._label_words("eating disorders") == [["eating"], ["disorders"]]
    assert text_terms._label_words("Parkinson's disease (MCI)") == [["parkinson"], ["disease"]]
    assert text_terms._label_words("epileptic patients") == [["epileptic"], ["patients"]]


def test_population_labels_are_not_shortened_to_their_adjective():
    # "healthy" alone matched 508 datasets, "young" also "Young Children".
    strings = {"healthy_volunteer": [("healthy volunteer", 1.0)],
               "young_adult": [("Young Adult", 1.0)],
               "trained_participant": [("Trained Participant", 1.0)]}
    for term, code, words in (("healthy volunteer", "healthy_volunteer", "healthy volunteer"),
                              ("young adults", "young_adult", "young adult"),
                              ("trained participants", "trained_participant", "trained participant")):
        found = text_terms.diagnosis_alternatives({term: [code]}, lookup=strings.get)
        assert words in found
        assert not any(len(alternative.split()) == 1 for alternative in found), found


def test_eating_disorder_alternatives_do_not_search_eating_alone():
    strings = {"eating_disorder": [("eating disorder", 1.0), ("anorexia nervosa", 0.9),
                                   ("anorexia", 0.8), ("bulimia", 0.8)]}
    alternatives = text_terms.diagnosis_alternatives(
        {"eating disorders": ["eating_disorder"]}, lookup=lambda code: strings.get(code, []))
    assert "eating" not in alternatives
    assert "eating disorders" in alternatives or "eating disorder" in alternatives
    assert "anorexia" in alternatives


# ── request words: concept words inside a phrase stay ──────────────────────

def test_request_words_inside_a_concept_stay():
    assert preprocess._content_words("deep brain stimulation") == "deep brain stimulation"
    assert preprocess._content_words("reward learning") == "reward learning"
    assert preprocess._content_words("sleep quality") == "sleep quality"
    assert preprocess._content_words("eyes open") == "eyes open"
    assert preprocess._content_words("free recall") == "free recall"
    assert preprocess._content_words("cognitive training") == "cognitive training"


def test_wish_phrases_still_go():
    assert preprocess._content_words("preprocessed machine learning") == "preprocessed"
    assert preprocess._content_words("good quality open data") == ""
    assert preprocess._content_words("classification") == ""
    assert preprocess._content_words("deep learning") == ""
    assert preprocess._content_words("very simple randomized") == "randomized"
