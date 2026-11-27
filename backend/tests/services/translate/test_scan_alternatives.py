"""Scan terms the user offers as alternatives ("X, Y or Z") become one OR
condition built in code, so the SQL model does not decide between AND and OR."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
for _p in ("backend/llm", "backend/vocabulary"):
    if str(_ROOT / _p) not in sys.path:
        sys.path.insert(0, str(_ROOT / _p))

from preprocess import (  # noqa: E402
    QueryPlan, RAGField, ResolvedTerms, ScanRequirement, _SYSTEM_PROMPT,
    _build_augmented_question, _gemini_output_schema,
)

EEG = ResolvedTerms(field=RAGField.DATATYPE, term_to_codes={"EEG": ["electroencephalography"]})
MEG = ResolvedTerms(field=RAGField.DATATYPE, term_to_codes={"MEG": ["magnetoencephalography"]})
TASKS = ResolvedTerms(field=RAGField.TASK, term_to_codes={
    "grasping": ["grasping_task"], "reaching": ["reaching_task", "reach_to_target"]})


def _plan(*requirements: ScanRequirement) -> QueryPlan:
    return QueryPlan(query_family="scan_filter", natural_language_summary="Summary.",
                     scan_requirements=list(requirements))


def _scan_lines(augmented: str) -> list:
    return [line.strip() for line in augmented.splitlines() if line.startswith("  scan")]


def test_alternatives_become_one_line_joined_with_or():
    plan = _plan(ScanRequirement(scan_terms=["EEG"]),
                 ScanRequirement(scan_terms=["grasping", "reaching"], any_of=True))
    lines = _scan_lines(_build_augmented_question(plan, [EEG, TASKS]))

    assert len(lines) == 2
    assert lines[0].startswith('scan "EEG": EXISTS')
    assert lines[1].startswith('scan any of "grasping", "reaching" (alternatives, one is enough): (')
    condition = lines[1].split(": ", 1)[1]
    assert condition.count("EXISTS (SELECT 1 FROM bids_objects") == 2
    assert "o2.task = 'grasping_task'" in condition
    assert "o2.task IN ('reaching_task', 'reach_to_target')" in condition
    # The two terms are alternatives at the top level of the one condition.
    assert re.search(r"grasping_task'\)( OR text_match\([^)]*\)\))? OR \(?EXISTS", condition)
    assert condition.count("(") == condition.count(")")


def test_without_the_flag_every_term_keeps_its_own_required_line():
    plan = _plan(ScanRequirement(scan_terms=["EEG", "grasping", "reaching"]))
    lines = _scan_lines(_build_augmented_question(plan, [EEG, TASKS]))
    assert [line.split(":")[0] for line in lines] == [
        'scan "EEG"', 'scan "grasping"', 'scan "reaching"']


def test_two_modalities_as_alternatives():
    plan = _plan(ScanRequirement(scan_terms=["EEG", "MEG"], any_of=True))
    both = ResolvedTerms(field=RAGField.DATATYPE, term_to_codes={
        "EEG": ["electroencephalography"], "MEG": ["magnetoencephalography"]})
    lines = _scan_lines(_build_augmented_question(plan, [both]))
    assert len(lines) == 1
    assert "o2.datatype = 'electroencephalography') OR EXISTS" in lines[0]
    assert "o2.datatype = 'magnetoencephalography')" in lines[0]


def test_an_alternative_the_vocabulary_does_not_know_joins_the_or_as_text():
    plan = _plan(ScanRequirement(scan_terms=["EEG"]),
                 ScanRequirement(scan_terms=["grasping", "juggling"], any_of=True))
    grasping = ResolvedTerms(field=RAGField.TASK, term_to_codes={"grasping": ["grasping_task"]})
    augmented = _build_augmented_question(plan, [EEG, grasping], unresolved_terms={"scan": ["juggling"]})
    lines = _scan_lines(augmented)

    assert lines[1].startswith('scan any of "grasping", "juggling"')
    assert re.search(r" OR text_match\('juggling', 'loose'\)\)$", lines[1])
    # Not also required on its own.
    assert "VOCABULARY MISS" not in augmented


def test_a_miss_outside_the_list_stays_required():
    plan = _plan(ScanRequirement(scan_terms=["grasping", "reaching"], any_of=True),
                 ScanRequirement(scan_terms=["juggling"]))
    augmented = _build_augmented_question(plan, [TASKS], unresolved_terms={"scan": ["juggling"]})
    assert "VOCABULARY MISS" in augmented and "'juggling'" in augmented.split("VOCABULARY MISS")[1]


def test_one_resolved_alternative_is_a_plain_line():
    plan = _plan(ScanRequirement(scan_terms=["grasping", "juggling"], any_of=True))
    only = ResolvedTerms(field=RAGField.TASK, term_to_codes={"grasping": ["grasping_task"]})
    lines = _scan_lines(_build_augmented_question(plan, [only]))
    assert len(lines) == 1 and lines[0].startswith('scan "grasping": ')


def test_a_negated_list_keeps_one_line_per_term():
    # "no grasping or reaching" excludes both (test_plan_logic.py).
    plan = _plan(ScanRequirement(scan_terms=["grasping", "reaching"], any_of=True, required=False))
    augmented = _build_augmented_question(plan, [TASKS])
    assert _scan_lines(augmented) == []
    assert [line.split(":")[0].strip() for line in augmented.splitlines() if "not scan" in line] == [
        'not scan "grasping" (complete)', 'not scan "reaching" (complete)']


def test_terms_are_matched_whatever_their_case():
    plan = _plan(ScanRequirement(scan_terms=["eeg", "Meg"], any_of=True))
    lines = _scan_lines(_build_augmented_question(plan, [EEG, MEG]))
    assert len(lines) == 1 and lines[0].startswith("scan any of")


def test_the_model_is_told_about_the_flag():
    schema = _gemini_output_schema()
    assert "any_of" in str(schema)
    assert "any_of: true" in _SYSTEM_PROMPT


def _top_level(sql: str, word: str) -> int:
    """How often *word* (AND / OR) stands outside every bracket and quote."""
    depth, count, quoted, i = 0, 0, False, 0
    while i < len(sql):
        char = sql[i]
        if char == "'":
            quoted = not quoted
        elif not quoted and char == "(":
            depth += 1
        elif not quoted and char == ")":
            depth -= 1
        elif not quoted and depth == 0 and sql.startswith(f" {word} ", i):
            count += 1
        i += 1
    return count


def test_the_or_line_is_one_bracketed_unit():
    # Copied next to another filter with AND, "A AND B OR C" must not arise.
    plan = _plan(ScanRequirement(scan_terms=["EEG"]),
                 ScanRequirement(scan_terms=["grasping", "reaching"], any_of=True))
    lines = _scan_lines(_build_augmented_question(plan, [EEG, TASKS]))
    eeg, alternatives = (line.split(": ", 1)[1] for line in lines)

    assert alternatives.startswith("(") and alternatives.endswith(")")
    assert _top_level(alternatives, "OR") == 0          # every OR is inside the brackets
    assert _top_level(alternatives[1:-1], "OR") == 1    # and the outer pair is what holds them
    where = f"{eeg} AND {alternatives}"
    assert (_top_level(where, "AND"), _top_level(where, "OR")) == (1, 0)


def test_the_compiler_takes_the_or_line_as_one_condition():
    from services.translate.participant_query_compiler import compile_plan
    from preprocess import ParticipantGroup
    plan = _plan(ScanRequirement(scan_terms=["EEG"]),
                 ScanRequirement(scan_terms=["grasping", "reaching"], any_of=True))
    plan.groups = [ParticipantGroup(group_name="patients", diagnosis_terms=["schizophrenia"])]
    diagnosis = ResolvedTerms(field=RAGField.DIAGNOSIS, term_to_codes={"schizophrenia": ["schizophrenia"]})
    resolved = [EEG, TASKS, diagnosis]
    compiled = compile_plan(plan, resolved, _build_augmented_question(plan, resolved))

    assert compiled.sql, compiled.reason
    where = " ".join(compiled.sql.split("WHERE", 1)[1].split("GROUP BY")[0].split())
    # EEG, the alternatives and the cohort: three required conditions, no OR between them.
    assert (_top_level(where, "AND"), _top_level(where, "OR")) == (2, 0)
    assert "grasping_task" in where and "reaching_task" in where and "schizophrenia" in where


def test_the_compiler_still_declines_what_is_not_a_scan_line():
    from services.translate.participant_query_compiler import _scan_conditions
    block = ("[Resolved DB filters — copy each filter expression verbatim into the SQL WHERE clause]\n"
             '  scan any of "a" (alternatives, one is enough): (x)\n')
    assert _scan_conditions(block)[0] is None            # a list of one is not this line
    block = block.replace('"a"', '"a", "b"').replace("(x)", "x OR y")
    assert _scan_conditions(block)[0] is None            # not bracketed: not taken


def test_with_full_text_search_a_short_term_is_matched_as_a_word():
    # "ERP" must not match "interpretation": the text alternative is a word
    # query, not a substring search.
    from services.translate.sql_rewriter import rewrite_generated_sql
    plan = _plan(ScanRequirement(scan_terms=["grasping", "ERP"], any_of=True))
    grasping = ResolvedTerms(field=RAGField.TASK, term_to_codes={"grasping": ["grasping_task"]})
    augmented = _build_augmented_question(plan, [grasping], unresolved_terms={"scan": ["ERP"]})
    condition = _scan_lines(augmented)[0].split(": ", 1)[1]
    sql = ("SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id "
           f"AND o.subject IS NOT NULL WHERE {condition} GROUP BY d.id")
    flat = " ".join(rewrite_generated_sql(sql, augmented).sql.split())
    assert "TO_TSQUERY" in flat.upper() and "erp" in flat.lower()
    assert not re.search(r"ILIKE '%erp%'", flat, re.I)


def test_the_rewriter_keeps_the_or_between_alternatives():
    from services.translate.sql_rewriter import rewrite_generated_sql
    plan = _plan(ScanRequirement(scan_terms=["EEG"]),
                 ScanRequirement(scan_terms=["grasping", "juggling"], any_of=True))
    augmented = _build_augmented_question(plan, [EEG, TASKS], unresolved_terms={"scan": ["juggling"]})
    conditions = [line.split(": ", 1)[1] for line in _scan_lines(augmented)]
    sql = ("SELECT {{COLS}} FROM bids_datasets d LEFT JOIN bids_objects o ON o.dataset_id = d.id "
           "AND o.subject IS NOT NULL WHERE " + " AND ".join(conditions) + " GROUP BY d.id")
    rewritten = rewrite_generated_sql(sql, augmented).sql
    flat = " ".join(rewritten.split())
    assert "text_match(" not in flat
    assert "grasping_task" in flat and "juggling" in flat.lower()
    assert re.search(r"grasping_task'\).{0,400}? OR ", flat)


def test_a_modality_listed_with_its_tasks_is_required_not_an_alternative():
    # "EEG during sleep or anesthesia" planned as one list of all three.
    plan = _plan(ScanRequirement(scan_terms=["EEG", "grasping", "reaching"], any_of=True))
    lines = _scan_lines(_build_augmented_question(plan, [EEG, TASKS]))
    assert lines[0].startswith('scan "EEG": EXISTS')
    assert lines[1].startswith('scan any of "grasping", "reaching"')
    assert "electroencephalography" not in lines[1]


def test_a_modality_and_one_task_stay_alternatives():
    plan = _plan(ScanRequirement(scan_terms=["EEG", "grasping"], any_of=True))
    grasping = ResolvedTerms(field=RAGField.TASK, term_to_codes={"grasping": ["grasping_task"]})
    lines = _scan_lines(_build_augmented_question(plan, [EEG, grasping]))
    assert len(lines) == 1 and lines[0].startswith('scan any of "EEG", "grasping"')


def test_two_modalities_with_tasks_stay_alternatives():
    plan = _plan(ScanRequirement(scan_terms=["EEG", "MEG", "grasping", "reaching"], any_of=True))
    lines = _scan_lines(_build_augmented_question(plan, [EEG, MEG, TASKS]))
    assert len(lines) == 1 and lines[0].startswith('scan any of "EEG", "MEG", "grasping", "reaching"')
