"""
The instructions of the two SQL prompts, and how a generation request is put
together.

  system_prompt()      the rules the SQL model writes a query by
  user_prompt()        schema, examples and the request, within the size guard
  correction_prompt()  the request to repair a query PostgreSQL rejected

The schema text comes from schema_ddl.py and the examples from examples.py;
text_to_sql.py sends the prompts.
"""

from __future__ import annotations

import logging
import textwrap
from typing import List

from services.translate.prompt.examples import format_examples

log = logging.getLogger(__name__)

# Whole prompt character guard, including system policy and generated schema.
# This is deliberately a conservative growth safeguard, not a token counter.
SQL_PROMPT_CHAR_BUDGET = 48_000


def system_prompt(long_text_phrase: str) -> str:
    """The rules of SQL generation.

    *long_text_phrase* names the long text columns the database has
    ("d.description_text or d.readme_text"): the model is told not to search
    them with ILIKE.
    """
    instructions = (
        "## 1. Hard correctness constraints\n"
        "1. suffix and datatype hold normalized codes, NOT raw BIDS values. "
        "Use ONLY these values — "
        "datatype: 'anatomical_mri' (structural MRI / T1w / T2w), 'functional_mri' (fMRI / BOLD), "
        "'diffusion_mri' (DWI), 'electroencephalography' (EEG), 'magnetoencephalography' (MEG), "
        "'intracranial_eeg' (iEEG/SEEG), 'behavioural_data', "
        "'positron_emission_tomography' (PET), 'field_maps', 'perfusion_asl', 'fnirs'. "
        "suffix: 'fmri_bold', 't1_weighted_mri', 't2_weighted_mri', 'diffusion_mri_dwi', "
        "'eeg', 'meg', 'intracranial_eeg', 'pet'. "
        "Never use raw BIDS values such as 'T1w', 'T2w', 'bold', 'dwi', 'anat', 'func', "
        "'eeg'/'meg' as datatype, 'beh', 'ieeg', 'fmap', 'perf', 'nirs'. "
        "Filter modality by datatype ONLY — do NOT combine a datatype filter "
        "with a suffix IN list for the same modality "
        "(e.g. structural MRI → just `datatype = 'anatomical_mri'`, never add suffix ORs).\n"
        "2. Disease/condition completeness: bids_participants.diagnosis is sparsely populated "
        "— most datasets about a disease have no structured diagnosis rows. "
        "Each 'diagnosis:' line in [Resolved DB filters] therefore already pairs the EXISTS "
        "block with a text fallback: "
        "(EXISTS (SELECT 1 FROM bids_participants p2 WHERE p2.dataset_id = d.id "
        "AND p2.diagnosis IN ('...')) OR text_match('...', 'short')). "
        "Copy the whole parenthesised expression verbatim, including every term inside "
        "text_match and an AND text_match(...) that some lines add after it. "
        "Never emit the diagnosis EXISTS without its OR text_match(...) part.\n"
        "3. Participant diagnosis: only add a diagnosis filter if one is explicitly present in "
        "[Resolved DB filters]. Words like 'patients', 'participants', 'subjects' refer to "
        "study participants in general — NOT a diagnosis constraint. "
        "Never generate `p.diagnosis != 'healthy_volunteer'` based only on those words. "
        "'Healthy', 'normal', 'controls' add NO condition at all (no diagnosis filter, no "
        "text_match): most datasets include healthy participants and few record it. "
        "If a 'participant sex:', 'participant age:' or 'participant sex and age:' line is present, copy it verbatim as its "
        "own AND condition and write no other condition on sex or age; never combine sex, age "
        "and diagnosis in one EXISTS (that demands one participants row with all three, and "
        "most datasets record only some of them).\n"
        "4. VOCABULARY MISS: If [VOCABULARY MISS] appears, those terms have no canonical code. "
        "Use ONLY the text_match(...) clause shown in that block (copy it verbatim). "
        "Never assign a VOCABULARY MISS term to o.task, o.datatype, o.suffix, or p.diagnosis.\n"
        "## 2. Build filters from the current request\n"
        "- [Resolved DB filters]: copy each filter expression VERBATIM into WHERE (AND them together). "
        "'scan X:' and 'scan task X:' for the same term are two separate EXISTS — keep both. "
        "If a 'subject count: HAVING ...' line is present, add that HAVING clause after GROUP BY d.id. "
        "If a 'dataset size: ...' line is present, AND that condition into WHERE verbatim.\n"
        "- Text search: text_match('<keyword>') is a search macro that the system expands into "
        "a full-text predicate over the dataset's name, description, study fields and README. "
        "It is a boolean condition for WHERE and combines with AND / OR / NOT. Use it only for "
        "a subject the DATA must be about and that no line in [Resolved DB filters] already "
        "covers. Add NO text_match (and no other condition) for: what the asker wants to do with "
        "the data ('for machine learning', 'for classification', 'to test my pipeline'); wishes "
        "about quality, difficulty or documentation ('standard', 'simple', 'non-trivial', "
        "'well documented', 'labeled', 'complete', 'sufficient sample size', 'images included'); "
        "file formats; what a publication reports; a dataset named only as an example or "
        "comparison ('like ABIDE', 'similar to DEAP'). Such words describe the request, not the "
        "dataset text, and a condition on them returns nothing. Use the single most distinctive keyword "
        "(e.g. for 'drinking alcohol' write text_match('alcohol')) — several words in one "
        "term must ALL occur, so add words only when each one is essential. "
        "Separate alternatives with ' | ': text_match('anaesthesia | anesthesia'). "
        "Word forms are matched automatically ('depression' also finds 'depressive'). "
        f"Never write ILIKE on {long_text_phrase} to look for a word; the one exception is "
        "a URL fragment such as '%doi.org%', which text_match cannot find. "
        "Use d.name ILIKE '%word%' only when the question asks for a word in the dataset "
        "name or title. ILIKE remains correct for short fields: d.license, "
        "unnest(d.paper_references), scanner strings in other_entities. "
        "Never use text_match or ILIKE on diagnosis, task, datatype, or suffix "
        "(they hold canonical codes).\n"
        "- Alternatives: data types or groups listed as options ('fMRI, EEG or fNIRS', "
        "'brain signals (fMRI, EEG, fNIRS)', 'MI, ERP and SSVEP paradigms' as kinds of dataset) "
        "mean ANY of them: join their conditions with OR. Use AND only when one dataset must "
        "contain all of them ('simultaneous EEG and fMRI', 'T1w and BOLD for every subject'). "
        "A modality mentioned only in a comparison ('EEG data, similar to the fMRI collection X') "
        "is not a filter.\n"
        "- Do NOT add LIMIT unless the question asks for 'top N'.\n"
        "## 3. Required SQL structure\n"
        "- Only use tables/columns present in the schema.\n"
        "- Begin the query with EXACTLY: SELECT {{COLS}}\n"
        "  Do NOT list any column names after SELECT — write the literal placeholder "
        "{{COLS}}. The full column list is injected automatically afterwards. The "
        "alias `subject_count` is part of that injected list, so you may still "
        "reference it in HAVING / ORDER BY (e.g. HAVING COUNT(DISTINCT o.subject) > 30, "
        "ORDER BY subject_count DESC).\n"
        "- Always include: LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
        "- Always include: GROUP BY d.id\n"
    )

    return textwrap.dedent(f"""\
        You generate PostgreSQL queries for the BIDS-Eye neuroimaging dataset catalog.

        {instructions}
        Return only the SQL query, with no explanation or markdown fences.
    """)


def user_prompt(schema_ddl: str, augmented_question: str, examples: List[dict],
                system_prompt: str) -> str:
    """The user message of SQL generation: schema, examples, then the request."""
    # Keep live context together in the user message, with the actionable request
    # last. Drop complete lower-ranked examples if the full request grows beyond
    # the character guard; never truncate an SQL example mid-statement.
    schema_section = (
        '<database_schema authority="only available tables and columns">\n'
        f"{schema_ddl}\n</database_schema>"
    )
    request_section = f"<current_request>\n{augmented_question}\n</current_request>"
    base_prompt = f"{schema_section}\n{request_section}"
    # "Illustrative only" made the model drop parts of a matching example
    # (the text fallback of "7 Tesla" in 2 of 4 live runs, 2026-11-15).
    examples_open = ('<examples purpose="correct SQL for similar requests; when one matches '
                     'the current request, keep every part of its WHERE clause">\n')
    examples_close = "\n</examples>\n"
    example_budget = max(
        0,
        SQL_PROMPT_CHAR_BUDGET - len(system_prompt) - len(base_prompt)
        - len(examples_open) - len(examples_close),
    )
    examples_block = format_examples(examples, max_chars=example_budget)
    examples_section = (
        f"{examples_open}{examples_block}{examples_close}" if examples_block else ""
    )
    prompt = f"{schema_section}\n{examples_section}{request_section}"
    if len(system_prompt) + len(prompt) > SQL_PROMPT_CHAR_BUDGET:
        log.warning(
            "SQL prompt exceeds %d-character guard before examples (system=%d, user=%d)",
            SQL_PROMPT_CHAR_BUDGET, len(system_prompt), len(prompt),
        )
    return prompt


def correction_prompt(question: str, schema_ddl: str, sql: str, db_error: str) -> str:
    """The request to fix *sql*, which PostgreSQL rejected with *db_error*."""
    return (
        "A PostgreSQL query failed. Fix it and return ONLY the corrected SQL — "
        "no explanation, no markdown fences. Change only what the error requires; "
        "keep the SELECT list and the rest of the query as they are.\n\n"
        f"Original question:\n{question}\n\n"
        f"Database schema:\n{schema_ddl}\n\n"
        f"Failed SQL:\n{sql}\n\n"
        f"PostgreSQL error:\n{db_error}"
    )
