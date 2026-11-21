"""The rank of a translation, from backends before and after RankQuery.

Older backends carry it as the string rank_tsquery ("stemmed#exact"), newer
ones as rank_query, a services/fulltext/ranking.RankQuery. These helpers read
and pass on whichever the backend on sys.path has, and write either to JSON.
"""
from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable, Dict


def rank_of(result: Any) -> Any:
    """The rank of a translation, rewrite result or CachedQuery; None without one."""
    if hasattr(result, "rank_query"):
        return result.rank_query
    return getattr(result, "rank_tsquery", None)


def rank_kwargs(target: Callable[..., Any], rank: Any) -> Dict[str, Any]:
    """The keyword argument that passes *rank* to build_page_sql or CachedQuery."""
    name = "rank_query" if "rank_query" in inspect.signature(target).parameters else "rank_tsquery"
    return {name: rank}


def rank_json(rank: Any) -> Any:
    """*rank* as JSON: a RankQuery as a dict, the old string as it is."""
    return dataclasses.asdict(rank) if dataclasses.is_dataclass(rank) else rank
