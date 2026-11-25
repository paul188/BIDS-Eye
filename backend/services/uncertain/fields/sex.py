"""
services/uncertain/fields/sex.py
------------------------------------
Participant sex as a field a dataset may not record (FIELD, at the end).

Its text form: the dataset's own text names participants of the asked sex
("22 women", "all-female sample"). Also here, because they read the same
conditions: which sex a condition asks for (asked_sex) and the repair of an
"only female participants" condition split in two (keep_only_sex_pairs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:  # sqlglot is a backend dependency; keep the module importable without it
    from sqlglot import exp
except ImportError:  # pragma: no cover
    exp = None  # type: ignore[assignment]

from services.evidence.possible import NotRecorded
from services.evidence.probable import NamedInText
from services.uncertain.conditions import own_text, sql_literal
from services.uncertain.fields.base import SoftField, participant_rows_record
from services.schema.tables import participants as _participants
from participant_codes import sex_code  # RAG/ is on the path wherever the backend runs
import text_evidence


LABEL = "participant sex"   # how a result card names the field

# The stored codes that name one sex; a condition on any other value has no
# text form.
ONE_SEX = ("female", "male")


def text_words(sex: str) -> Tuple[str, ...]:
    """The words a dataset's text uses for participants of *sex*.

    From the vocabulary (text_words of the sex entries in
    RAG/value_mappings.yaml): every common English word, none left out by
    hand. Whether a mention is about the participants is decided by its
    grammar (_PARTICIPANT_RE), the same for every word: "22 women" counts;
    generic "man" without a count ("the brain of man", "Isle of Man") and a
    possessive ("a woman's voice") do not. Not measured against recorded sex
    yet.
    """
    return text_evidence.text_words("sex").get(sex, ())


# A sex word said of participants: after a count ("22 women", "12 healthy
# young men", "a 34-year-old woman"), before one ("females: 12", "women
# (n = 14)"), or with a word for the people studied ("female participants",
# "subjects were male").
_PARTICIPANT_RE = (r"([0-9]+[ -]([a-z-]+ ){{0,3}}({w})|({w})[ :=(]+(n ?= ?)?[0-9]+"
                   r"|({w}) ({people})|({people}) (were|are) (all )?({w}))")
# "Only female participants": "only" and "all" are english stopwords, so a
# tsquery cannot say it; a regex over the dataset's own text does.
_ONLY_SEX_RE = (r"(all[- ]?({w})|({w})[- ]only|only ({w})|exclusively ({w})"
                r"|all (participants|subjects|volunteers) (were|are) ({w}))")


def asked_sex(node) -> Optional[str]:
    """'female' or 'male' when a sex condition asks for one sex, else None."""
    values = set()
    for comparison in node.find_all(exp.EQ, exp.In):
        if not any(c.name.lower() == "sex" for c in comparison.find_all(exp.Column)):
            continue
        for literal in comparison.find_all(exp.Literal):
            # Which literal stands for which sex: the vocabulary (RAG/participant_codes.py).
            if literal.is_string and sex_code(literal.this) in ONE_SEX:
                values.add(sex_code(literal.this))
    return next(iter(values)) if len(values) == 1 else None


def _only_one_sex(node) -> bool:
    """The condition requires every participant row to have the sex
    (preprocess._participant_conditions: NOT EXISTS a row without it)."""
    for not_node in node.find_all(exp.Not):
        inner = not_node.this
        if isinstance(inner, exp.Exists) and any(
                c.name.lower() == "sex" for c in inner.find_all(exp.Column)):
            return True
    return False


def _reads_participant_sex(node) -> bool:
    return any(t.name.lower() == _participants.name for t in node.find_all(exp.Table)) and any(
        c.name.lower() == "sex" for c in node.find_all(exp.Column))


def keep_only_sex_pairs(nodes: list) -> list:
    """"Only female participants" as one condition again.

    preprocess writes it as EXISTS(a female row) AND NOT EXISTS(a row that is
    not female); flattening the ANDs made two conditions, so the tier never
    saw "only" (the README text for any female counted), and the NOT EXISTS
    alone held for a dataset with no participant rows at all.
    """
    rest = []
    pairs: dict[str, exp.Expression] = {}
    for node in nodes:
        bare = node.this if isinstance(node, exp.Paren) else node
        if isinstance(bare, exp.Not) and isinstance(bare.this, exp.Exists) and _reads_participant_sex(bare):
            pairs.setdefault("not", node)
        else:
            rest.append(node)
    if "not" not in pairs:
        return nodes
    partner = next((n for n in rest if not isinstance(n, exp.Not) and _reads_participant_sex(n)
                    and asked_sex(n)), None)
    if partner is None:
        return nodes
    joined = exp.Paren(this=exp.And(this=partner.copy(), expression=pairs["not"].copy()))
    return [joined if n is partner else n for n in rest]


@dataclass(frozen=True)
class SexNamed(NamedInText):
    """The dataset's own text names participants of *sex*; with *only*, it
    says all of them are."""
    sex: str = ""
    only: bool = False

    def sql(self, alias: str) -> Optional[str]:
        words = "|".join(text_words(self.sex))
        only = _ONLY_SEX_RE.format(w=words)
        if self.only:
            return _in_own_text(alias, only)
        # "An all-female sample" names female participants too.
        people = "|".join(text_evidence.people_words())
        return _in_own_text(alias, f"({_PARTICIPANT_RE.format(w=words, people=people)}|{only})")

    def reason(self) -> str:
        not_recorded = NotRecorded(LABEL).reason()
        if self.only:
            return f"{not_recorded}; the description says all participants are {self.sex}"
        return f"{not_recorded}; the description names {self.sex} participants"


def sex_named(node) -> Optional[SexNamed]:
    """The sex a condition asks for, as a text would name it; None if the
    condition does not ask for one sex."""
    sex = asked_sex(node)
    if sex is None or not text_words(sex):
        return None
    return SexNamed(sex=sex, only=_only_one_sex(node))


_WORD_END = "($|[^[:alnum:]_'’])"


def _in_own_text(alias: str, pattern: str) -> str:
    """The pattern as whole words in the dataset's own text, any case."""
    # No apostrophe after it: "a woman's voice" names an owner, not a participant.
    return f"{own_text(alias)} ~* {sql_literal(r'(^|[^[:alnum:]_])' + pattern + _WORD_END)}"


FIELD = SoftField(
    kind="sex",
    label=LABEL,
    columns=("sex",),
    recorded=participant_rows_record("sex"),
    named_in_text=sex_named,
)
