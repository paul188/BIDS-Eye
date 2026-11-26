"""
services/fulltext/ranking.py
----------------------------
The text rank of a result: SQL for ts_rank_cd over the stored vectors, and the
RankQuery that carries its tsqueries from the translation to the result page.
A RankQuery is kept with a translation (as JSON, rank_to_json) and with every
page of it (services/stores/translated_queries.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, Optional

from services.fulltext.tsquery import is_safe_tsquery
from services.fulltext import vectors
from services.fulltext.vectors import EXACT_FTS_COLUMN, EXACT_TS_CONFIG, FTS_COLUMN, TS_CONFIG

# How much a match counts, by where it is. The search vectors give each part
# of a dataset a Postgres weight label (setweight, see fulltext/vectors.py);
# ts_rank_cd scores a label by its weight here.
TS_WEIGHT_NAME = 1.0           # label A: the dataset's name
TS_WEIGHT_SHORT_FIELDS = 0.6   # label B: short fields, incl. conditions the README says were studied
TS_WEIGHT_LLM_PROFILE = 0.4    # label C: the language model's profile of the README
TS_WEIGHT_README = 0.05        # label D: README text, and paper text when that is on

TS_WEIGHT_BY_LABEL = {
    "A": TS_WEIGHT_NAME,
    "B": TS_WEIGHT_SHORT_FIELDS,
    "C": TS_WEIGHT_LLM_PROFILE,
    "D": TS_WEIGHT_README,
}
# ts_rank_cd's weights argument: one array in the order D, C, B, A.
TS_RANK_WEIGHTS = "{" + ", ".join(str(TS_WEIGHT_BY_LABEL[label]) for label in "DCBA") + "}"
# ts_rank_cd's normalization argument, a Postgres bit mask. 0 leaves the rank
# as it is; 32 divides it by itself + 1, which puts it in 0..1, the range RAG
# scores use.
TS_RANK_NORMALIZATION_NONE = 0
TS_RANK_NORMALIZATION_RANK_PLUS_ONE = 32

# Some searches get other words added: "dead people" also searches
# "post-mortem | ex vivo", written text_match('post-mortem | ex vivo', 'also').
# Whether a dataset has only those other words, not the user's own, is a
# column of its own (other_words_only_expression), not part of its rank:
# where such datasets go is for the ranker to say (services/results/rankers.py).
OTHER_WORDS_COLUMN = "other_words_only"


@dataclass(frozen=True)
class RankQuery:
    """The tsqueries that rank a result by its text; each is None when absent.

    stemmed      ranks on search_tsv
    exact        ranks the words with decided forms on search_tsv_exact
    alternative  the search as the user wrote it, when alternative words
                 ('also') widen it: a dataset that matches only the
                 alternative words does not match this one
                 (other_words_only_expression)
    """
    stemmed: Optional[str] = None
    exact: Optional[str] = None
    alternative: Optional[str] = None

    def __post_init__(self) -> None:
        # Each part is inlined into SQL (rank_expression).
        for part in fields(self):
            query = getattr(self, part.name)
            if query is not None and not is_safe_tsquery(query):
                raise ValueError(f"unsafe {part.name} tsquery")

    def without_alternative(self) -> RankQuery:
        """The same ranking without the alternative part: every match ranks alike."""
        return replace(self, alternative=None)


def rank_to_json(rank: Optional[RankQuery]) -> Optional[Dict[str, Optional[str]]]:
    return asdict(rank) if rank is not None else None


def rank_from_json(data: Any) -> Optional[RankQuery]:
    """The RankQuery that rank_to_json wrote; None for anything else.

    A part that is not a safe tsquery is dropped, as the stored JSON is read
    back into SQL.
    """
    if not isinstance(data, dict):
        return None
    parts = {part.name: data.get(part.name) for part in fields(RankQuery)}
    safe = {name: query for name, query in parts.items() if isinstance(query, str) and is_safe_tsquery(query)}
    return RankQuery(**safe) if safe.get("stemmed") or safe.get("exact") else None


def rank_expression(alias: str, rank: Optional[RankQuery]) -> Optional[str]:
    """SQL for the text rank of the dataset row *alias*; None without a rank.

    0..1; with both parts the raw ranks are added before scaling, as one
    query over one vector would have done.
    """
    if rank is None:
        return None
    stemmed, exact = rank.stemmed, rank.exact
    if exact and not vectors.exact_enabled():
        exact = None
    if stemmed and not exact:
        return ts_rank_cd_expression(f"{alias}.{FTS_COLUMN}", stemmed)
    if exact and not stemmed:
        return ts_rank_cd_expression(f"{alias}.{EXACT_FTS_COLUMN}", exact, EXACT_TS_CONFIG)
    if not stemmed:
        return None
    raw = (f"COALESCE(ts_rank_cd('{TS_RANK_WEIGHTS}', {alias}.{FTS_COLUMN}, "
           f"to_tsquery('{TS_CONFIG}', '{stemmed}'), {TS_RANK_NORMALIZATION_NONE}), 0) + "
           f"COALESCE(ts_rank_cd('{TS_RANK_WEIGHTS}', {alias}.{EXACT_FTS_COLUMN}, "
           f"to_tsquery('{EXACT_TS_CONFIG}', '{exact}'), {TS_RANK_NORMALIZATION_NONE}), 0)")
    # TS_RANK_NORMALIZATION_RANK_PLUS_ONE, applied to the sum.
    return f"(({raw}) / (({raw}) + 1))"


def other_words_only_expression(alias: str, rank: Optional[RankQuery]) -> Optional[str]:
    """SQL that is true for a dataset row *alias* that matches only the
    words a search was widened with ('also'), not the search as the user
    wrote it. None when the search was not widened or has no text rank."""
    if rank is None or not rank.alternative or rank_expression(alias, rank) is None:
        return None
    return (f"({alias}.{FTS_COLUMN} @@ to_tsquery('{TS_CONFIG}', '{rank.alternative}')) "
            "IS NOT TRUE")


def ts_rank_cd_expression(vector_sql: str, rank_tsquery: str, config: str = TS_CONFIG) -> str:
    """SQL expression for the 0..1 text rank of one dataset.

    *config*: TS_CONFIG for search_tsv, EXACT_TS_CONFIG for search_tsv_exact.
    """
    if not is_safe_tsquery(rank_tsquery):
        raise ValueError("unsafe rank tsquery")
    if config not in (TS_CONFIG, EXACT_TS_CONFIG):
        raise ValueError("unknown text search configuration")
    return (
        f"COALESCE(ts_rank_cd('{TS_RANK_WEIGHTS}', {vector_sql}, "
        f"to_tsquery('{config}', '{rank_tsquery}'), {TS_RANK_NORMALIZATION_RANK_PLUS_ONE}), 0)"
    )
