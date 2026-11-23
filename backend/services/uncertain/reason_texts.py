"""
services/uncertain/reason_texts.py
-----------------------------
Every reason a result card gives for a dataset in the uncertain tier.

A reason says why a condition could not be confirmed for the dataset. The
parts of the tier decide which one applies; the wording is here, so the whole
list can be read in one place. Reasons of several conditions are joined with
"; " (tier.py).

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

from typing import Sequence

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
