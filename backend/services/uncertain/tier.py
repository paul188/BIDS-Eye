"""
services/uncertain/tier.py
-----------------------------------
Keep datasets that a filter cannot check, and show them after the confirmed
results.

Some filters fail only because a dataset does not record the field at all:
about half the datasets have no participant ages, most have no coded
diagnosis. Some text filters fail because a dataset uses only some of the
words ("MRI data of 3-12 year old children" for "young & children"). On the
167-question recall benchmark these exclusions are most of what the search
misses (benchmarks/recall/work/failure_analysis/).

Each top-level AND condition of the generated SQL gets one of four states for
a dataset:

    0  confirmed     the condition holds
    1  probable      a weaker form holds: some of the search words, the words
                     only in a paper the dataset names (with TEXT_SEARCH_PAPERS
                     on), or, for a missing field, the README names the asked
                     group ("children" for an age filter up to 12)
    2  possible      the field is not recorded and nothing hints either way
    9  contradicted  the dataset records the field and it does not fit

Only conditions on fields that can be missing (ages, diagnosis, sex, the
field strength and scanner make or model in the sidecars) or on text can be
uncertain. For each, the dataset's own text naming the asked value is the
probable form: "women", "all-female", "7T", "Siemens". Modality, file
types, size, subject counts and the other sidecar values come from the
files themselves and stay hard filters, as does anything negated.

"At least N participants with a diagnosis" is its own case (classify.py):
confirmed when that many are coded or labelled with it, probable when the
dataset does not record who has the diagnosis, contradicted when its
participant rows show fewer.

A dataset is kept when
  - no condition is contradicted,
  - at most MAX_UNSURE conditions are not confirmed, and
  - at least one uncertain-able condition is confirmed or probable: a dataset
    that only matches the modality while everything else is unknown is not
    evidence of anything.

The result gets two columns: uncertain_level (0, 1 or 2: the weakest state)
and uncertain_reasons (why, for the result card). The page query orders by
uncertain_level first (services/translate/sql_rewriter.py). Datasets that confirm every
condition are exactly the rows the original query returned.

Switched on with UNCERTAIN_TIER=true; the benchmark compares both from one
translation (benchmarks/recall/paired_uncertain_run.py).

This module does the rewrite (apply). The parts it uses:
  classify.py    which states each condition can take
  text.py        weaker forms of a text search
  sex.py         participant sex in the dataset's own text
  scanner.py     field strength, vendor and model in the text
  conditions.py  SoftCondition and what the others share
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence

try:  # sqlglot is a backend dependency; keep the module importable without it
    import sqlglot
    import sqlglot.errors
    from sqlglot import exp
except ImportError:  # pragma: no cover
    sqlglot = None  # type: ignore[assignment]
    exp = None  # type: ignore[assignment]

from services.uncertain.conditions import PAPER_REASON, SoftCondition, possible_enabled, sql_literal
from services.uncertain.sex import keep_only_sex_pairs
from services.uncertain.classify import classify
from services.schema.tables import datasets as _datasets


ENV_FLAG = "UNCERTAIN_TIER"
# Conditions per dataset that may be unconfirmed.
MAX_UNSURE = int(os.getenv("UNCERTAIN_MAX_UNSURE", "1"))


def hoist_terms_enabled() -> bool:
    """Name each tsquery once in a one-row CTE instead of repeating its literal.

    Off unless UNCERTAIN_HOIST_TERMS=true. It shortens cohort SQL by a third,
    but the planner no longer sees the tsquery as a constant: on the live
    database (2026-11-15, ten query shapes) two text-led shapes ran 4x faster
    and one age shape 2x slower; without it every shape ran as before.
    """
    return os.getenv("UNCERTAIN_HOIST_TERMS", "false").strip().lower() in ("1", "true", "yes", "on")


LEVEL_COLUMN = "uncertain_level"
REASONS_COLUMN = "uncertain_reasons"


def enabled() -> bool:
    """True when UNCERTAIN_TIER is on. Read on every call."""
    return os.getenv(ENV_FLAG, "false").strip().lower() in ("1", "true", "yes", "on")


def _flatten_and(node) -> list:
    from services.fulltext.text_search import is_guarded_pair
    if isinstance(node, exp.And):
        return _flatten_and(node.left) + _flatten_and(node.right)
    # Any depth: "((codes OR epilepsy words) AND spatiotemporal)" in two pairs
    # of parentheses stayed one condition, whose weaker form was any one of
    # its text searches: a paper that says "spatiotemporal" was a probable
    # match for "spatiotemporal epilepsy" (live, 2026-11-20).
    inner = node
    while isinstance(inner, exp.Paren) and isinstance(inner.this, exp.Paren):
        inner = inner.this
    if isinstance(inner, exp.Paren) and isinstance(inner.this, exp.And):
        if is_guarded_pair(inner):
            return [inner]   # a text search and its word forms: one condition
        return _flatten_and(inner.this)
    return [node]


def _dataset_alias(select) -> Optional[str]:
    source = select.args.get("from") or select.args.get("from_")
    table = getattr(source, "this", None)
    if isinstance(table, exp.Table) and table.name == _datasets.name:
        return table.alias_or_name
    return None


def apply(sql: str, papers: Optional[bool] = None, hoist_terms: Optional[bool] = None) -> str:
    """The query with uncertain conditions softened and the two tier columns.

    *papers*: let paper text serve as evidence (default: TEXT_SEARCH_PAPERS
    and the column exist). Returns *sql* unchanged when it has nothing that
    can be uncertain, or when it cannot be parsed.
    """
    if papers is None:
        from services.fulltext import vectors
        papers = vectors.papers_enabled()
    if hoist_terms is None:
        hoist_terms = hoist_terms_enabled()
    if sqlglot is None or LEVEL_COLUMN in sql:
        return sql
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.SqlglotError:
        return sql
    if not isinstance(tree, exp.Select):
        return sql
    # A top-N or ordered question ("the 10 largest …") keeps its own order and
    # limit; softening the filter inside them would let rows the tier later
    # drops fill the limit and push confirmed results out.
    if tree.args.get("limit") or tree.args.get("order") or tree.args.get("offset"):
        return sql
    alias = _dataset_alias(tree)
    where = tree.args.get("where")
    if alias is None or where is None:
        return sql

    hard: List[str] = []
    soft: List[SoftCondition] = []
    for node in keep_only_sex_pairs(_flatten_and(where.this)):
        # classify has no expected failures (sqlglot and regex work on a tree
        # that parsed); it ran without one on the 1,972 conditions of the
        # benchmark runs' SQL (2026-11-19). A failure here is a bug: raised,
        # not turned silently into a hard filter as it used to be.
        condition = classify(node, alias, papers=papers)
        if condition is None:
            hard.append(node.sql(dialect="postgres"))
        else:
            soft.append(condition)
    if not soft:
        return sql

    # Keep the indexed candidate filter, then calculate each evidence channel
    # once for those candidates. The old wrapper copied the same full-text
    # expression into the candidate WHERE, the tier CASE, and the paper flag.
    inner = tree.copy()
    keep = " AND ".join([*(f"({h})" for h in hard), *(c.keep_sql() for c in soft)])
    inner.set("where", exp.Where(this=sqlglot.parse_one(keep, read="postgres")))
    inner_sql = inner.sql(dialect="postgres")

    reserved = sql.lower()
    def unique_cte_name(base: str) -> str:
        candidate = base
        suffix = 2
        while re.search(rf"\b{re.escape(candidate)}\b", reserved):
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate.lower()

    terms_name = unique_cte_name("_qf_tier_terms")
    terms_alias = unique_cte_name("_qf_q")

    evidence_columns = [f"{alias}.id"]
    state_cases = []
    paper_columns = []
    for i, condition in enumerate(soft):
        confirmed = f"confirmed_{i}"
        probable = f"probable_{i}"
        known = f"known_{i}"
        paper = f"paper_{i}"
        evidence_columns.extend((
            f"({condition.sql}) AS {confirmed}",
            f"({condition.probable}) AS {probable}" if condition.probable
            else f"FALSE AS {probable}",
            f"({condition.known}) AS {known}" if condition.known
            else f"TRUE AS {known}",
            f"({condition.paper}) AS {paper}" if condition.paper
            else f"FALSE AS {paper}",
        ))

        raw = f"ev.{confirmed}"
        weak = f"ev.{probable}"
        recorded = f"ev.{known}"
        possible = "2" if possible_enabled() else "9"
        if condition.known is None:
            state = f"CASE WHEN {raw} THEN 0 WHEN {weak} THEN 1 ELSE 9 END"
        elif condition.kind == "diagnosis":
            state = (f"CASE WHEN {raw} THEN 0 WHEN {weak} THEN 1 "
                     f"WHEN NOT {recorded} THEN {possible} ELSE 9 END"
                     if possible_enabled() else
                     f"CASE WHEN {raw} THEN 0 WHEN {weak} THEN 1 ELSE 9 END")
        else:
            unknown = f"CASE WHEN {weak} THEN 1 ELSE {possible} END"
            state = (f"CASE WHEN {raw} THEN 0 WHEN NOT {recorded} "
                     f"THEN {unknown} ELSE 9 END")
        state_cases.append(f"{state} AS u{i}")
        if condition.paper:
            paper_columns.append(f"ev.{paper} AS p{i}")

    raw_sql = (
        f"SELECT {', '.join(evidence_columns)}\n"
        f"FROM {_datasets.name} AS {alias}"
    )
    # Evidence is looked up per candidate row (LATERAL). Two CTEs joined on id
    # read the same, but the planner then sorts every file row of the
    # candidates instead of hashing them: 3 s became 5 s on the live database.
    raw_where = f"\nWHERE {alias}.id = tier_base.id"

    # Full-text predicates are used in the candidate filter and again to label
    # their evidence level. Hoist identical tsquery values into a one-row CTE;
    # both stages refer to q.qN instead of copying long synonym lists.
    tsquery_re = re.compile(
        r"TO_TSQUERY\(\s*'(?P<config>english|simple)'\s*,\s*'(?P<query>[^']*)'\s*\)",
        re.IGNORECASE,
    )
    query_values: dict[tuple[str, str], str] = {}
    for match in tsquery_re.finditer(inner_sql + "\n" + raw_sql):
        key = (match.group("config").lower(), match.group("query"))
        query_values.setdefault(key, f"q{len(query_values)}")

    terms_cte = None
    if query_values and hoist_terms:
        def replace_tsquery(match: re.Match) -> str:
            key = (match.group("config").lower(), match.group("query"))
            return f"{terms_alias}.{query_values[key]}"

        inner_sql = tsquery_re.sub(replace_tsquery, inner_sql)
        raw_sql = tsquery_re.sub(replace_tsquery, raw_sql)
        terms_select = ", ".join(
            f"TO_TSQUERY('{config}', '{query}') AS {column}"
            for (config, query), column in query_values.items()
        )
        terms_cte = f"{terms_name} AS (SELECT {terms_select})"

        # The query CTE is one row; joining it makes its named tsquery values
        # visible to the indexed candidate WHERE and the evidence expressions.
        inner_tree = sqlglot.parse_one(inner_sql, read="postgres")
        query_table = exp.Table(
            this=exp.to_identifier(terms_name),
            alias=exp.TableAlias(this=exp.to_identifier(terms_alias)),
        )
        joins = list(inner_tree.args.get("joins") or [])
        joins.append(exp.Join(this=query_table, kind="CROSS"))
        inner_tree.set("joins", joins)
        inner_sql = inner_tree.sql(dialect="postgres")
        raw_sql += f"\nCROSS JOIN {terms_name} AS {terms_alias}"
    raw_sql += raw_where

    state_sql = (
        f"SELECT ev.id, {', '.join(state_cases)}"
        + (f", {', '.join(paper_columns)}" if paper_columns else "")
        + f"\nFROM (\n{raw_sql}\n) AS ev"
    )

    names = [f"evidence.u{i}" for i in range(len(soft))]
    unsure = " + ".join(f"(CASE WHEN {name} > 0 THEN 1 ELSE 0 END)" for name in names)
    evidence = " OR ".join(f"{name} <= 1" for name in names)
    level = f"GREATEST({', '.join(names)})" if len(names) > 1 else names[0]
    reasons = []
    for i, c in enumerate(soft):
        branches = []
        if c.paper:
            branches.append(f"WHEN evidence.u{i} = 1 AND evidence.p{i} THEN {sql_literal(PAPER_REASON)}")
        if c.reason_probable:
            branches.append(f"WHEN evidence.u{i} = 1 THEN {sql_literal(c.reason_probable)}")
        if c.reason_possible:
            branches.append(f"WHEN evidence.u{i} = 2 THEN {sql_literal(c.reason_possible)}")
        if branches:
            reasons.append(f"CASE {' '.join(branches)} END")
    reasons_sql = (f"NULLIF(CONCAT_WS('; ', {', '.join(reasons)}), '')" if reasons else "NULL")

    ctes = [f"{name} AS MATERIALIZED ({select})"
            for name, select in dict.fromkeys(cte for c in soft for cte in c.ctes)]
    if terms_cte:
        ctes.insert(0, terms_cte)
    with_sql = f"WITH {', '.join(ctes)}\n" if ctes else ""
    return (
        f"{with_sql}"
        f"SELECT tier_base.*, {level} AS {LEVEL_COLUMN}, "
        f"{reasons_sql} AS {REASONS_COLUMN}\n"
        f"FROM (\n{inner_sql}\n) AS tier_base\n"
        f"CROSS JOIN LATERAL (\n{state_sql}\n) AS evidence\n"
        f"WHERE {level} < 9 AND ({unsure}) <= {MAX_UNSURE} AND ({evidence})"
    )


def confirmed_only(rows: Sequence[dict]) -> list:
    """The rows every condition confirms (what the original query returns)."""
    return [row for row in rows if not row.get(LEVEL_COLUMN)]
