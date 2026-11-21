"""
services/fulltext/ranking.py
----------------------------
The text rank of a result: SQL for ts_rank_cd over the stored vectors, and the
rank string that carries its queries from the translation to the result page.

The rank string is one tsquery for search_tsv, or "stemmed#exact" when words
with decided forms rank on search_tsv_exact, or "stemmed#exact#own" when other
words ('also') widen a search (join_rank, split_rank). It is stored with a
translation and travels as one string.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from services.fulltext.tsquery import is_safe_tsquery
from services.fulltext import vectors
from services.fulltext.vectors import EXACT_FTS_COLUMN, EXACT_TS_CONFIG, FTS_COLUMN, TS_CONFIG

# How much a match counts, by where it is. The search vectors give each part
# of a dataset a weight label (crawlers/migrate_add_search_tsv.sql and the
# other migrate_add_*.sql); ts_rank_cd scores a label by its weight here.
WEIGHT_NAME = 1.0           # label A: the dataset's name
WEIGHT_SHORT_FIELDS = 0.6   # label B: short fields, incl. conditions the README says were studied
WEIGHT_LLM_PROFILE = 0.4    # label C: the language model's profile of the README
WEIGHT_README = 0.05        # label D: README text, and paper text when that is on

WEIGHT_BY_LABEL = {
    "A": WEIGHT_NAME,
    "B": WEIGHT_SHORT_FIELDS,
    "C": WEIGHT_LLM_PROFILE,
    "D": WEIGHT_README,
}
# ts_rank_cd takes the weights as one array in the order D, C, B, A.
RANK_WEIGHTS = "{" + ", ".join(str(WEIGHT_BY_LABEL[label]) for label in "DCBA") + "}"
# 32 scales the rank into 0..1 (rank / (rank + 1)), the range RAG scores use.
RANK_NORMALIZATION = 32

# Between the parts of a rank query: the tsquery for search_tsv, the one for
# search_tsv_exact, and the search that 'also' words widen. Not a tsquery
# character.
RANK_SEPARATOR = "#"
# Taken off the text rank of a dataset that matches only the other words
# ('also') of a search and not the search itself. Every ordering adds the
# text rank (0..1) to the score of the matched codes (a sum of at most a few
# dozen scores up to 1), so this puts such a dataset below every dataset
# that matches the user's own words, and keeps the order among them.
ALSO_ONLY_PENALTY = 100


def join_rank(stemmed: Optional[str], exact: Optional[str],
              own: Optional[str] = None) -> Optional[str]:
    """One rank string from its parts; the stemmed query alone as it is."""
    if own:
        return f"{stemmed or ''}{RANK_SEPARATOR}{exact or ''}{RANK_SEPARATOR}{own}"
    if not exact:
        return stemmed or None
    return f"{stemmed or ''}{RANK_SEPARATOR}{exact}"


def _rank_parts(rank: Optional[str]) -> List[Optional[str]]:
    parts = (rank or "").split(RANK_SEPARATOR, 2)
    parts += [""] * (3 - len(parts))
    return [part if is_safe_tsquery(part) else None for part in parts]


def split_rank(rank: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(stemmed, exact) of a rank string; a part is None when absent or unsafe."""
    stemmed, exact, _own = _rank_parts(rank)
    return stemmed, exact


def rank_own(rank: Optional[str]) -> Optional[str]:
    """The search that other words widen ('also'); None when there is none."""
    return _rank_parts(rank)[2]


def without_own(rank: Optional[str]) -> Optional[str]:
    """The rank string without the 'also' part: every match ranks alike."""
    return join_rank(*split_rank(rank))


def rank_expression(alias: str, rank: Optional[str]) -> Optional[str]:
    """SQL for the text rank of the dataset row *alias*; None without a rank.

    0..1; with both parts the raw ranks are added before scaling, as one
    query over one vector would have done. A dataset that matches only the
    other words of a search ('also') gets ALSO_ONLY_PENALTY taken off.
    """
    expression = _plain_rank_expression(alias, rank)
    own = rank_own(rank)
    if expression is None or not own:
        return expression
    return (f"({expression} - CASE WHEN {alias}.{FTS_COLUMN} @@ "
            f"to_tsquery('{TS_CONFIG}', '{own}') THEN 0 ELSE {ALSO_ONLY_PENALTY} END)")


def _plain_rank_expression(alias: str, rank: Optional[str]) -> Optional[str]:
    stemmed, exact = split_rank(rank)
    if exact and not vectors.exact_enabled():
        exact = None
    if stemmed and not exact:
        return text_rank_expression(f"{alias}.{FTS_COLUMN}", stemmed)
    if exact and not stemmed:
        return text_rank_expression(f"{alias}.{EXACT_FTS_COLUMN}", exact, EXACT_TS_CONFIG)
    if not stemmed:
        return None
    raw = (f"COALESCE(ts_rank_cd('{RANK_WEIGHTS}', {alias}.{FTS_COLUMN}, "
           f"to_tsquery('{TS_CONFIG}', '{stemmed}'), 0), 0) + "
           f"COALESCE(ts_rank_cd('{RANK_WEIGHTS}', {alias}.{EXACT_FTS_COLUMN}, "
           f"to_tsquery('{EXACT_TS_CONFIG}', '{exact}'), 0), 0)")
    # Normalization 32 of text_rank_expression, applied to the sum.
    return f"(({raw}) / (({raw}) + 1))"


def text_rank_expression(vector_sql: str, rank_tsquery: str, config: str = TS_CONFIG) -> str:
    """SQL expression for the 0..1 text rank of one dataset.

    *config*: TS_CONFIG for search_tsv, EXACT_TS_CONFIG for search_tsv_exact.
    """
    if not is_safe_tsquery(rank_tsquery):
        raise ValueError("unsafe rank tsquery")
    if config not in (TS_CONFIG, EXACT_TS_CONFIG):
        raise ValueError("unknown text search configuration")
    return (
        f"COALESCE(ts_rank_cd('{RANK_WEIGHTS}', {vector_sql}, "
        f"to_tsquery('{config}', '{rank_tsquery}'), {RANK_NORMALIZATION}), 0)"
    )
