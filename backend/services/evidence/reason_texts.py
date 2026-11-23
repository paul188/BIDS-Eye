"""
services/evidence/reason_texts.py
---------------------------------
Every reason a result card gives, in one list: why a dataset matched a filter
(confirmed), and why a condition could not be confirmed for it (the uncertain
tier). Each piece of evidence carries one of these texts
(services/evidence/base.py); which one applies is decided where the evidence
is built.

A confirmed match (results/match_reasons.py): where the value was found
      in the participant list      participants_listed, participants_counted
      in the dataset's own fields  dataset_age_range, field_value
      everywhere else              FOUND_IN, by the source of the evidence
      assumed by default           ASSUMED

The uncertain tier (services/uncertain/); reasons of several conditions are
joined with "; " (tier.py):
  the field is not recorded
      NOT_RECORDED                 ages, diagnosis, participant sex
      patient count                PATIENT_COUNT_NOT_RECORDED
      scanner                      scanner_not_recorded
  the field is not recorded, and the dataset's own text names the asked value
      ages                         AGE_GROUP_NAMED
      participant sex              sex_named, only_sex_named
      scanner                      scanner_named
  a text search matched in a weaker form
      some of the words            words_missing
      the words, not together      words_apart
      next to an excluded meaning  excluded_meaning
      further down in the README   readme_only
      only in a paper              PAPER
"""

from __future__ import annotations

from typing import Optional, Sequence

from services.evidence import base

# ── A confirmed match: where the value was found ──────────────────────────────

# By the source of the evidence, for the sources that need no numbers.
FOUND_IN = {
    base.FILES: "in the dataset's files",
    base.DERIVATIVES: "in its derivatives",
    base.SITES: "at a listed site",
    base.PAPER: "named in a linked paper",
    base.TEXT: "named in the README or description",
    base.SHORT_TEXT: "named in the title or description",
    base.README: "named in the README",
    # An excluded diagnosis the dataset passes because it records none.
    base.ASSUMED_ABSENT: "no diagnosis recorded, assumed absent",
}

# Columns with a default (RAG/criteria.py): the column is empty and its
# default holds.
ASSUMED = {"species": "no species label, assumed human"}

# How a reason names a column of the dataset; others by their own name.
FIELD_LABELS = {
    "species": "species label",
    "dx_status": "diagnosis label",
    "study_design": "study design",
    "study_domain": "study domain",
}

PARTICIPANTS_LISTED = "recorded in the participant list"


def _number(value: float) -> str:
    return f"{round(value, 1):g}"


def age_span(low: float, high: float) -> str:
    """'4–23 months' below 2 years, else '6–11 years'; one value once."""
    if high < 2:
        months_lo, months_hi = round(low * 12), round(high * 12)
        return (f"{months_lo} months" if months_lo == months_hi
                else f"{months_lo}–{months_hi} months")
    lo, hi = _number(low), _number(high)
    return f"{lo} years" if lo == hi else f"{lo}–{hi} years"


def participants_counted(count: int, total: Optional[int], ages: Optional[str],
                         diagnosis: bool) -> str:
    """*ages*: their age span when the condition reads ages ("4–23 months")."""
    whom = f"{count} of {total} participants" if total and total >= count else f"{count} participants"
    if ages:
        return f"{whom} aged {ages}"
    return f"coded for {whom}" if diagnosis else f"recorded for {whom}"


def dataset_age_range(ages: str) -> str:
    return f"dataset age range {ages}"


def field_value(column: str, value: str) -> str:
    return f"{FIELD_LABELS.get(column, column.replace('_', ' '))} “{value}”"


# ── The field is not recorded ─────────────────────────────────────────────────

# By the kind of a condition (classify.py).
NOT_RECORDED = {
    "age": "ages not recorded",
    "diagnosis": "diagnosis not recorded",
    "sex": "participant sex not recorded",
}

# "At least N participants with a diagnosis" (RAG/cohort_count.py).
PATIENT_COUNT_NOT_RECORDED = "how many participants have the diagnosis is not recorded"


def scanner_not_recorded(label: str) -> str:
    """*label*: "field strength", "scanner manufacturer" or "scanner model"."""
    return f"{label} not in the file metadata"


# ── Not recorded, but the dataset's own text names the asked value ────────────

AGE_GROUP_NAMED = f"{NOT_RECORDED['age']}; the description names the age group"


def sex_named(sex: str) -> str:
    return f"{NOT_RECORDED['sex']}; the description names {sex} participants"


def only_sex_named(sex: str) -> str:
    return f"{NOT_RECORDED['sex']}; the description says all participants are {sex}"


def scanner_named(label: str, named: str) -> str:
    """*named*: the asked values, e.g. "3 T or 7 T", "Siemens"."""
    return f"{scanner_not_recorded(label)}; the description names {named}"


# ── A text search matched in a weaker form ────────────────────────────────────

PAPER = "found only in a paper the dataset names"


def _listed(terms: Sequence[str], limit: int) -> str:
    return ", ".join(terms[:limit]) + (" …" if len(terms) > limit else "")


def words_missing(words: Sequence[str]) -> str:
    return f"matches only part of the search; missing: {_listed(words, 4)}"


def words_apart(terms: Sequence[str]) -> str:
    return f"the words {_listed(terms, 3)} occur, but not together"


def excluded_meaning(terms: Sequence[str]) -> str:
    return f"a related term occurs next to a meaning the search excludes ({_listed(terms, 3)})"


def readme_only(terms: Sequence[str]) -> str:
    """*terms*: the words whose search was widened to the whole README; may be empty."""
    if not terms:
        return "the search term appears only further down in the README"
    shown = " / ".join(f'"{t}"' for t in terms[:2]) + (" …" if len(terms) > 2 else "")
    return f"{shown} appears only further down in the README"
