"""
services/fulltext/tier_sex.py
-----------------------------
Participant sex in the dataset's own text: "women", "all-female" as the
probable form of a sex filter.
"""

from __future__ import annotations

from typing import Optional, Tuple

try:  # sqlglot is a backend dependency; keep the module importable without it
    from sqlglot import exp
except ImportError:  # pragma: no cover
    exp = None  # type: ignore[assignment]

from services.fulltext.tier_conditions import sql_literal
from services.schema.tables import participants as _participants


# Words a README uses for each sex (stemmed by the english configuration:
# "females" is "femal", "women" stays). Not "man"/"woman" ("Isle of Man",
# "a woman's voice" in stimuli) nor "girls"/"boys" (stimuli and children's
# books as often as participants).
SEX_WORDS = {"female": ("female", "females", "women"), "male": ("male", "males", "men")}
_SEX_CODES = {"f": "female", "female": "female", "m": "male", "male": "male"}
# "Only female participants": "only" and "all" are english stopwords, so a
# tsquery cannot say it; a regex over the dataset's own text does.
_ONLY_SEX_RE = (r"(all[- ]?({w})|({w})[- ]only|only ({w})|exclusively ({w})"
                r"|all (participants|subjects|volunteers) (were|are) ({w}))")
_ONLY_SEX_WORDS = {"female": "female|females|women", "male": "male|males|men"}


def asked_sex(node) -> Optional[str]:
    """'female' or 'male' when a sex condition asks for one sex, else None."""
    values = set()
    for comparison in node.find_all(exp.EQ, exp.In):
        if not any(c.name.lower() == "sex" for c in comparison.find_all(exp.Column)):
            continue
        for literal in comparison.find_all(exp.Literal):
            if literal.is_string and literal.this.lower() in _SEX_CODES:
                values.add(_SEX_CODES[literal.this.lower()])
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


def own_text(alias: str) -> str:
    """The dataset's own name, description and README as one text."""
    return (f"(COALESCE({alias}.name, '') || ' ' || COALESCE({alias}.description_text, '') "
            f"|| ' ' || COALESCE({alias}.readme_text, ''))")


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


def sex_text_condition(node, alias: str) -> Optional[Tuple[str, str]]:
    """(SQL, reason) for a dataset whose text names the asked sex; None if
    the condition does not ask for one sex."""
    sex = asked_sex(node)
    if sex is None:
        return None
    if _only_one_sex(node):
        pattern = _ONLY_SEX_RE.format(w=_ONLY_SEX_WORDS[sex])
        return (f"{own_text(alias)} ~* {sql_literal(r'(^|[^[:alnum:]_])' + pattern + r'($|[^[:alnum:]_])')}",
                f"participant sex not recorded; the description says all participants are {sex}")
    words = " | ".join(SEX_WORDS[sex])
    return (f"{alias}.search_tsv @@ TO_TSQUERY('english', '{words}')",
            f"participant sex not recorded; the description names {sex} participants")
