"""Other words for a scan term: an ERP study that says only "oddball" or "P300".

The language model lists, per scan term that names a task, paradigm or
stimulus, the words dataset descriptions use for it or for a specific kind of
it (QueryPlan.scan_term_alternatives). preprocess/ checks them against the
catalogue and searches them next to the term's own text search, as 'also':
a dataset that has only them ranks after those with the user's words.
"""

from __future__ import annotations

import pytest

sqlglot = pytest.importorskip("sqlglot")

import preprocess as pp  # noqa: E402
from preprocess import RAGField, ResolvedTerms, ScanRequirement, TermAlternatives  # noqa: E402
from services.fulltext import text_search, word_frequencies  # noqa: E402


def _plan(terms, alternatives) -> pp.QueryPlan:
    return pp.QueryPlan(natural_language_summary="Summary.",
                        scan_requirements=[ScanRequirement(scan_terms=terms)],
                        scan_term_alternatives=[TermAlternatives(term=t, alternatives=a)
                                                for t, a in alternatives.items()])


def _counts(table, default=5):
    return lambda alternatives: [table.get(a, default) for a in alternatives]


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    monkeypatch.setattr(word_frequencies, "_lookup_counts", lambda queries: None)


def test_alternatives_are_checked_against_the_catalogue() -> None:
    plan = _plan(["EEG", "ERP"], {
        "ERP": ["oddball", "P300", "ERPs", "event-related potential", "brain", "N2pc"],
        "not a scan term": ["anything"],
    })
    dropped = pp.check_scan_term_alternatives(plan, _counts({"brain": 900, "N2pc": 0}))
    assert [(e.term, e.alternatives) for e in plan.scan_term_alternatives] == [
        # The listed written-out form (backend/vocabulary/shorthands.yaml) first, then the model's.
        ("ERP", ["event-related potential", "oddball", "P300"])]
    assert dropped == {"ERP: brain": "too common (900 datasets)", "ERP: N2pc": "no dataset uses it"}


def test_a_shorthand_gets_its_listed_forms_without_the_model() -> None:
    # The model listed nothing for "CEST"; shorthands.yaml says what it stands for.
    plan = _plan(["CEST"], {})
    pp.check_scan_term_alternatives(plan, _counts({}))
    assert [(e.term, e.alternatives) for e in plan.scan_term_alternatives] == [
        ("CEST", ["chemical exchange saturation transfer"])]


def test_a_listed_form_is_never_too_common() -> None:
    plan = _plan(["ERP"], {"ERP": ["oddball"]})
    pp.check_scan_term_alternatives(plan, _counts({"event-related potential": 400, "oddball": 400}))
    assert plan.scan_term_alternatives[0].alternatives == ["event-related potential"]


def test_what_the_model_proposes_for_an_unexplained_shorthand_is_dropped() -> None:
    # No listing knows "XQZ": the model's words for it are guesses, as
    # "Arterial Perfusion Tagging" was for "APT" (2026-11-23).
    plan = _plan(["XQZ"], {"XQZ": ["cross quantile zoning", "quantile maps"]})
    dropped = pp.check_scan_term_alternatives(plan, _counts({}))
    assert plan.scan_term_alternatives == []
    assert set(dropped.values()) == {"no listing says what the shorthand stands for"}


def test_a_resolved_task_term_searches_its_alternatives_too() -> None:
    plan = _plan(["ERP"], {"ERP": ["oddball", "P300"]})
    tasks = ResolvedTerms(field=RAGField.TASK, term_to_codes={"ERP": ["erp_task"]})
    augmented = pp._build_augmented_question(plan, [tasks])
    line = next(line for line in augmented.splitlines() if line.startswith('  scan "ERP"'))
    assert "EXISTS" in line and "task = 'erp_task'" in line
    assert "text_match('oddball | p300', 'also')" in line


def test_an_unknown_term_searches_its_alternatives_too() -> None:
    plan = _plan(["isolated words"], {"isolated words": ["single words", "lexical decision"]})
    augmented = pp._build_augmented_question(plan, [], unresolved_terms={"scan": ["isolated words"]})
    assert ("(text_match('isolated words', 'loose') OR "
            "text_match('single words | lexical decision', 'also'))") in augmented


def test_terms_without_alternatives_are_written_as_before() -> None:
    plan = _plan(["isolated words"], {})
    augmented = pp._build_augmented_question(plan, [], unresolved_terms={"scan": ["isolated words"]})
    assert "text_match('isolated words', 'loose')" in augmented and "'also'" not in augmented


def test_a_dataset_with_only_the_alternatives_ranks_after_the_users_term() -> None:
    expanded = text_search.expand_text_search(
        "SELECT d.id FROM bids_datasets AS d WHERE "
        "(text_match('isolated words', 'loose') OR text_match('single words | lexical decision', 'also')) "
        "GROUP BY d.id", papers=False)
    assert expanded.rank_query.alternative is not None
    assert "lexical" not in expanded.rank_query.alternative
