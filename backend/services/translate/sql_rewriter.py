"""
sql_rewriter.py
---------------
SQL post-processing for the Text-to-SQL pipeline.

This module owns the parts of the pipeline that operate on SQL after the LLM
has already produced a draft query:
  - extracting SQL from fenced or annotated model output,
  - canonical projection replacement,
  - stripping hallucinated scan filters,
  - expanding broad vocabulary keys ('neurodevelopmental') into their codes,
  - injecting VOCABULARY MISS fallback predicates,
  - expanding text_match() placeholders into full-text predicates,
  - wrapping relevance scoring, text ranking and pagination.

The LLM-facing prompt generation stays in ``services.translate.text_to_sql`` so the
translation stack has a clean separation between "generate SQL" and "rewrite
SQL".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field as _field
from typing import Any, Iterable, Mapping, Optional, Sequence

try:  # pragma: no cover - exercised in environments that have sqlglot installed
    from sqlglot import exp, parse_one
    from sqlglot.errors import SqlglotError
except ImportError:  # pragma: no cover - keep the service usable when the dep is absent locally
    exp = None  # type: ignore[assignment]
    parse_one = None  # type: ignore[assignment]
    SqlglotError = ()  # type: ignore[assignment,misc]  # nothing is parsed, so nothing to catch

from join_registry import field_to_db_col as _field_to_db_col  # noqa: E402
from services.translate.concept_keys import expand_concept_keys as _expand_concept_keys  # noqa: E402
from text_terms import parse_alternatives as _parse_alternatives  # noqa: E402
import cohort_count as _cohort_count  # noqa: E402

from services.fulltext import ranking, text_search as _text_search  # noqa: E402
from services.fulltext import uncertain_tier as _uncertain_tier  # noqa: E402
from services.catalogue import schema_capabilities as _schema_capabilities  # noqa: E402
from services.translate import value_check as _value_check  # noqa: E402

log = logging.getLogger(__name__)

_PLACEHOLDER = "{{COLS}}"
DEFAULT_DATASET_PROJECTION = (
    "d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
    "       d.source_type, d.remote_url, d.validation_status,\n"
    "       d.authors, d.institutions, d.description_text,\n"
    "       COUNT(DISTINCT o.subject) AS subject_count"
)
_FALLBACK_SQL = (
    f"SELECT {DEFAULT_DATASET_PROJECTION}\n"
    "FROM bids_datasets d\n"
    "LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
    "GROUP BY d.id\n"
    "ORDER BY d.name\n"
    "LIMIT 200"
)
FALLBACK_SQL = _FALLBACK_SQL

_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)
_BAD_OUTER_FILTER = re.compile(
    r"(?:AND\s+)?o\d*\.(task|datatype|suffix)\s*=\s*'[^']*'",
    re.IGNORECASE,
)
_VOCAB_MISS_HINT_RE = re.compile(
    r"For free-text search use:\s*(.+?)(?=\s*\])",
    re.IGNORECASE | re.DOTALL,
)
# A topic filter line of the augmented question (preprocess/topics.py, _topic_condition).
_TOPIC_LINE_RE = re.compile(r'^\s*topic "[^"\n]*": (?P<condition>.+)$', re.MULTILINE)
# The description search with its other words (preprocess/, 'dataset text:').
_DESCRIPTION_LINE_RE = re.compile(r"^\s*dataset text: (?P<condition>\(.+\))\s*$", re.MULTILINE)
# Any 'dataset text:' line; a formula ("memory AND NOT sleep") has several parts.
_ANY_DESCRIPTION_LINE_RE = re.compile(r"^\s*dataset text: (?P<condition>.+?)\s*$", re.MULTILINE)


def _top_level_and_parts(condition: str) -> list:
    """*condition* split on the ANDs outside parentheses and quotes."""
    parts, depth, quoted, start = [], 0, False, 0
    i = 0
    while i < len(condition):
        char = condition[i]
        if char == "'":
            quoted = not quoted
        elif not quoted and char == "(":
            depth += 1
        elif not quoted and char == ")":
            depth -= 1
        elif not quoted and depth == 0:
            joiner = re.match(r"\s+AND\s+", condition[i:], re.IGNORECASE)
            if joiner:
                parts.append(condition[start:i].strip())
                i += joiner.end()
                start = i
                continue
        i += 1
    parts.append(condition[start:].strip())
    return [p for p in parts if p]


_TEXT_MATCH_CALL_RE = re.compile(
    r"\btext_match\s*\(\s*'([^']*)'\s*(?:,\s*'[^']*'\s*)?\)",
    re.IGNORECASE,
)
_CODE_RE = re.compile(r"^[A-Za-z0-9_]+$")
# A JSON text value that PostgreSQL can cast to a number (no backslashes, so
# the pattern survives sqlglot and standard_conforming_strings unchanged).
_NUMBER_PATTERN = r"^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$"
_SCORABLE_FIELDS = {"diagnosis", "task", "datatype", "suffix"}

# Task ILIKE concept expansion: maps vague category terms to canonical task codes.
# "cognitive" is not a BIDS task name — all experimental tasks are cognitive, so we
# remove the filter. "naturalistic" is a paradigm category covering movie/video/narrative
# tasks whose BIDS identifiers don't contain the word "naturalistic".
_TASK_ILIKE_CONCEPT_RE = re.compile(
    r"(o\d*)\.task\s+ILIKE\s+'%(\w+)%'",
    re.IGNORECASE,
)
_NATURALISTIC_TASKS = (
    "naturalistic_passive_viewing", "naturalistic_object_category_perception",
    "naturalistic_animal_behavior_viewing", "naturalistic_dialogue_monologue_long",
    "naturalistic_dialogue_monologue_short", "naturalistic_kandinsky_viewing",
    "naturalistic_point_of_view_viewing", "naturalistic_rorschach_viewing",
    "naturalistic_seti_viewing", "naturalistic_tunnel_viewing",
    "movie_viewing", "movie_viewing_raiders", "movie_viewing_variant_1",
    "movie_viewing_variant_2", "movie_viewing_variant_007", "movie_viewing_variant_008",
    "movie_viewing_variant_009", "budapest_movie_viewing_1r",
    "budapest_movie_viewing_3r", "budapest_movie_viewing_5r",
    "silent_movie_viewing", "laluna_movie_viewing", "partly_cloudy_movie_viewing",
    "rest_film_viewing", "film_recall", "movie_memory",
    "narrative_listening", "video_game_and_video",
    "buck_natural_language_listening", "sloth_natural_language_listening",
    "passive_listening_natural_speech", "diotic_natural_speech_cortical_attention",
    "diotic_natural_speech_subcortical_attention",
    "attention_natural_image_reconstruction",
    "attention_natural_image_reconstruction_training",
    "perception_natural_image_test",
)
# Terms that map to a task IN list
_TASK_CONCEPT_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "naturalistic": _NATURALISTIC_TASKS,
}
# Terms that are vague category labels — remove the task filter entirely (return TRUE)
_TASK_CONCEPT_DROP = {"cognitive", "social", "emotional", "linguistic", "sensory"}

# bids_participants correlated count in WHERE → rewrite to HAVING on bids_objects subjects.
# bids_participants is sparsely populated; o.subject is the reliable count source.
_PARTICIPANTS_COUNT_WHERE_RE = re.compile(
    r"AND\s*\(\s*SELECT\s+COUNT\s*\(\s*DISTINCT\s+\w+\.participant_id\s*\)\s+"
    r"FROM\s+bids_participants\s+\w+\s+WHERE\s+\w+\.dataset_id\s*=\s*d\.id\s*\)\s*"
    r"(>=|>|<=|<|=)\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)
# The same plain count (all participants of the dataset, no condition) anywhere,
# e.g. in HAVING; not when already the second argument of the GREATEST below.
_PARTICIPANTS_COUNT_SUBQUERY_RE = re.compile(
    r"(?<!GREATEST\(COUNT\(DISTINCT o\.subject\), )"
    r"\(\s*SELECT\s+COUNT\s*\(\s*(?:DISTINCT\s+\w+\.participant_id|\*)\s*\)\s+"
    r"FROM\s+bids_participants\s+(?:AS\s+)?\w+\s+WHERE\s+\w+\.dataset_id\s*=\s*d\.id\s*\)",
    re.IGNORECASE | re.DOTALL,
)


# The filter line of "at least N patients with X" (LLM_preprocessor/preprocess/).
_PATIENTS_LINE_RE = re.compile(
    r"^  patients \([^)\n]*\): (patients_with\('[a-z0-9_, ]+'\) >= \d+)\s*$", re.MULTILINE)

_SQL_END_MARKER_RE = re.compile(r"\[/SQL\]|\[SQL\]|```", re.IGNORECASE)



_HEALTHY_CODES = frozenset({"typically_developing", "neurotypical", "control", "controls", "normal", "normal_control"})


def _is_healthy_code(code: str) -> bool:
    """True for a diagnosis code that only says the participant is healthy."""
    code = code.strip().lower()
    return code.startswith("healthy") or code in _HEALTHY_CODES


def _cut_at_end_marker(sql: str) -> tuple[str, bool]:
    """Drop a closing [/SQL] tag or code fence and anything after it."""
    m = _SQL_END_MARKER_RE.search(sql)
    if m is None:
        return sql.strip(), False
    return sql[: m.start()].strip(), True


@dataclass(slots=True)
class RewriteResult:
    """Normalized SQL plus what later stages need to run and rank it."""
    sql: str
    params: dict[str, Any]
    # What ranks the result by its text (services/fulltext/text_search.rank_expression);
    # None without text terms.
    rank_tsquery: Optional[str] = None
    # The query before the uncertain tier wraps it: its WHERE conditions are
    # the filter chips (services/results/query_filters.py). Equals sql without the tier.
    filter_sql: Optional[str] = None
    # The same query without its free-text-only conditions, for the case that
    # the full query finds nothing (services/search_runner.py); None when there is no
    # such condition to drop.
    relaxed_sql: Optional[str] = None
    # Values the catalogue check corrected or dropped (services/translate/value_check.py).
    value_fixes: list[str] = _field(default_factory=list)


class SqlRewriteService:
    """SQL post-processing for generated dataset queries.

    The service keeps the LLM pipeline focused on generating intent and raw SQL.
    Structural SQL fixes live here so they can be tested and evolved independently.
    """

    def __init__(self, projection: str = DEFAULT_DATASET_PROJECTION):
        self.projection = projection
        self._projection_expressions = self._load_projection_expressions()

    def _load_projection_expressions(self):
        if parse_one is None or exp is None:
            return None
        try:
            parsed = parse_one(
                f"SELECT {self.projection} FROM bids_datasets d",
                read="postgres",
            )
        except SqlglotError as exc:  # pragma: no cover - depends on sqlglot availability
            log.warning("Could not pre-parse canonical projection: %s", exc)
            return None
        if not isinstance(parsed, exp.Select):
            return None
        return [item.copy() for item in parsed.expressions]

    @staticmethod
    def extract_sql(raw: str) -> str:
        """Pull the first SQL statement out of an LLM response.

        The LLM may return a plain ``SELECT`` or a ``WITH ... SELECT`` block,
        so both shapes are accepted before we fall back to the bounded sample.
        """
        m = re.search(r"\[SQL\]\s*(.*?)(?:\[/SQL\]|$)", raw, re.DOTALL | re.IGNORECASE)
        if m and re.match(r"(?i)(select|with)\b", m.group(1).strip()):
            return _cut_at_end_marker(m.group(1))[0]
        m = _SQL_FENCE.search(raw)
        if m and re.match(r"(?i)(select|with)\b", m.group(1).strip()):
            return _cut_at_end_marker(m.group(1))[0]
        stripped = raw.strip()
        if re.match(r"(?i)(select|with)\b", stripped):
            # The prompt ends with "[SQL]", so the model often answers with
            # bare SQL followed by "[/SQL]"; everything from there on goes.
            sql, had_marker = _cut_at_end_marker(stripped)
            # Without a marker, a blank line is the best guess for where
            # trailing prose begins.
            return sql if had_marker else sql.split("\n\n")[0].strip()
        return _FALLBACK_SQL

    def rewrite_generated_sql(self, raw_sql: str, augmented_question: str,
                              word_forms: Optional[Mapping[str, Sequence[str]]] = None,
                              ) -> RewriteResult:
        """Normalize raw LLM output into executable SQL plus bind parameters.

        The method intentionally returns a structured result instead of mutating
        the caller's state. That keeps the translation pipeline easy to cache and
        makes the rewrite behavior testable in isolation.

        *word_forms*: the forms decided for words of this question
        (RAG/word_forms.py); text search then requires them
        (services/fulltext/text_search.py, TEXT_SEARCH_EXACT).
        """
        sql = self.extract_sql(raw_sql)
        sql = self._rewrite_projection(sql)
        sql = self._strip_outer_scan_filters(sql)
        sql = self._rewrite_task_ilike_concepts(sql)
        sql = _expand_concept_keys(sql)
        # A datatype / suffix / task value no dataset has matches nothing
        # (suffix = 'bold'): correct it or drop its condition.
        sql, value_fixes = _value_check.check_sql(sql)
        sql = self._rewrite_participant_count_to_having(sql)
        sql = self._participants_join_to_subject_count(sql)
        # Any other join of bids_participants next to the file join multiplies
        # the two tables: its conditions become subqueries.
        sql = self._participants_join_to_subqueries(sql)
        sql = self._drop_healthy_filters(sql)
        sql = self._decorrelate_grouped_exists(sql)
        sql = self._unify_scanner_model_key(sql)
        sql = self._guard_json_number_casts(sql)
        sql = self._case_insensitive_sex(sql)
        sql = self._handedness_codes(sql)
        sql = self._drop_invented_text_filters(sql, augmented_question)
        sql = self._inject_vocab_miss(sql, augmented_question)
        sql = self._inject_topics(sql, augmented_question)
        sql = self._restore_description_alternatives(sql, augmented_question)
        sql = self._restore_description_formula(sql, augmented_question)
        # "At least N patients with X": patients_with(...) >= N becomes a count of
        # the participants known to have the diagnosis (RAG/cohort_count.py).
        sql = self._inject_patient_counts(sql, augmented_question)
        sql = _cohort_count.expand(sql)
        tiered = _uncertain_tier.enabled()
        relaxed_sql = None
        without_text = self._without_text_only_filters(sql)
        if without_text is not None:
            relaxed_sql = _text_search.expand_text_search(
                without_text, papers=False if tiered else None).sql
            if tiered:
                relaxed_sql = _uncertain_tier.apply(relaxed_sql)
        # With the tier on, paper text is evidence for the second tier only;
        # confirmed results match on the dataset's own text.
        expanded = _text_search.expand_text_search(
            sql, papers=False if tiered else None, exact_forms=word_forms)
        sql = expanded.sql
        filter_sql = sql
        if tiered:
            # Keep datasets a filter cannot check, as a second tier
            # (services/fulltext/uncertain_tier.py). Off unless UNCERTAIN_TIER=true.
            sql = _uncertain_tier.apply(sql)
        return RewriteResult(sql=sql, params={}, rank_tsquery=expanded.rank_tsquery,
                             filter_sql=filter_sql, relaxed_sql=relaxed_sql,
                             value_fixes=value_fixes)

    def finish_corrected_sql(self, raw: str, failed_sql: str) -> str:
        """The model's repair of *failed_sql*, ready to run; FALLBACK_SQL when
        its answer holds no query.

        *failed_sql* already went through rewrite_generated_sql, so the repair
        is not rewritten again: a second pass put the dataset columns over the
        uncertain tier's wrapper, whose outer query has no table d, and every
        repaired tiered query failed. Only the outer column list of
        *failed_sql* is put back, because the page query and the result cards
        read those columns (uncertain_level among them). The guards check the
        repair like any generated query (services/query_guard.py).

        Returns *failed_sql* itself when the model changed only the formatting.
        """
        sql = self.extract_sql(raw)
        if sql == _FALLBACK_SQL or parse_one is None or exp is None:
            return sql
        try:
            failed = parse_one(failed_sql, read="postgres")
            repaired = parse_one(sql, read="postgres")
        except SqlglotError:
            return sql  # the database gives the reason
        if isinstance(failed, exp.Select) and isinstance(repaired, exp.Select):
            repaired.set("expressions", [item.copy() for item in failed.expressions])
        repaired_sql = repaired.sql(dialect="postgres")
        if repaired_sql == failed.sql(dialect="postgres"):
            return failed_sql
        return repaired_sql

    def build_count_sql(self, base_sql: str) -> str:
        """Wrap a cached base query in a COUNT(*) query for pagination metadata."""
        base = self._use_summaries(self._normalize_sql(base_sql))
        return f"SELECT COUNT(*) FROM (\n{base}\n) AS count_result"

    def build_page_sql(
        self,
        base_sql: str,
        scored_filters: list,
        apply_relevance: bool,
        limit: int,
        offset: int,
        rank_tsquery: Optional[str] = None,
        rerank_ordered: Optional[Sequence[str]] = None,
        rerank_demoted: Optional[Sequence[str]] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Wrap a cached base query for one page of results.

        Relevance ordering is layered on only when requested so the cached base
        query can be reused across pages without rebaking the sort order.

        *rerank_ordered* / *rerank_demoted* are dataset ids from the LLM
        re-ranker (services/results/rerank.py): the first come first in that order,
        the second last, everything else keeps the relevance order between.
        Their parameters are added after the last sqlglot pass, which would
        otherwise re-render them.
        """
        ordered = (
            self._inject_relevance_cte(base_sql, scored_filters, rank_tsquery)
            if apply_relevance else base_sql
        )
        ranked = ordered != base_sql
        ordered = self._use_summaries(self._normalize_sql(ordered))
        params: dict[str, Any] = {"_limit": limit, "_offset": offset}
        # Confirmed results first, then the uncertain tier (services/fulltext/uncertain_tier.py).
        tiered = _uncertain_tier.LEVEL_COLUMN in base_sql
        tier_key = f"page_result.{_uncertain_tier.LEVEL_COLUMN}, " if tiered else ""
        order_clause = ""
        if tiered and not (rerank_ordered or rerank_demoted):
            order_clause = "ORDER BY " + tier_key + (
                "page_result.relevance_score + page_result.text_rank DESC, " if ranked else ""
            ) + "page_result.subject_count DESC NULLS LAST, page_result.id\n"
        if rerank_ordered or rerank_demoted:
            params["_rr_top"] = [str(i) for i in (rerank_ordered or [])]
            params["_rr_low"] = [str(i) for i in (rerank_demoted or [])]
            tail = (
                "page_result.relevance_score + page_result.text_rank DESC, "
                if ranked else ""
            ) + "page_result.subject_count DESC NULLS LAST, page_result.id"
            order_clause = (
                f"ORDER BY {tier_key}CASE WHEN page_result.id = ANY(CAST(:_rr_top AS uuid[])) THEN 0\n"
                "              WHEN page_result.id = ANY(CAST(:_rr_low AS uuid[])) THEN 2\n"
                "              ELSE 1 END,\n"
                "         array_position(CAST(:_rr_top AS uuid[]), page_result.id),\n"
                f"         {tail}\n"
            )
        if not order_clause:
            order_clause = self._stable_page_order(base_sql, ranked)
        sql = (
            f"SELECT * FROM (\n{ordered}\n) AS page_result\n"
            f"{order_clause}"
            "LIMIT :_limit OFFSET :_offset"
        )
        return sql, params

    @staticmethod
    def _stable_page_order(base_sql: str, ranked: bool) -> str:
        """The page's ORDER BY when neither the tier nor the re-ranker sets one.

        Postgres does not promise to keep a subquery's order, and LIMIT/OFFSET
        over an unordered or tied result can repeat or skip rows between
        pages. So the page orders itself, with the dataset id as the last key.
        A query whose own top-level ORDER BY the user asked for ("the largest
        first") keeps that order: re-sorting it here would undo the request.
        """
        if ranked:
            return ("ORDER BY page_result.relevance_score + page_result.text_rank DESC, "
                    "page_result.subject_count DESC NULLS LAST, page_result.id\n")
        if _has_top_level_order(base_sql):
            return ""
        if "subject_count" in base_sql:
            return "ORDER BY page_result.subject_count DESC NULLS LAST, page_result.id\n"
        return "ORDER BY page_result.id\n"

    # ── Executed SQL: per-dataset summaries instead of the file table ─────────
    #
    # Applied to the SQL that runs, never to the stored translation: the
    # cache, the filter chips and the logs keep the query as it was written.

    _FACET_COLUMNS = frozenset({"dataset_id", "datatype", "suffix", "task"})

    def _use_summaries(self, sql: str) -> str:
        """Read bids_dataset_facets and bids_datasets.n_subjects where that gives the same rows.

        bids_objects has 2.7 million rows; its distinct (dataset, datatype,
        suffix, task) combinations are 21,000. A search asked the file table
        once per result and vocabulary code for its ranking score (40 of 44
        seconds for "resting-state fMRI") and sorted all of it to count
        subjects (2 to 4 seconds on every search).
        """
        facets = _schema_capabilities.has_dataset_facets()
        counts = _schema_capabilities.has_subject_counts()
        if parse_one is None or exp is None or not (facets or counts) or "bids_objects" not in sql:
            return sql
        try:
            parsed = parse_one(sql, read="postgres")
            changed = False
            for select in list(parsed.find_all(exp.Select)):
                if counts and self._stored_subject_count(select):
                    changed = True
            for select in list(parsed.find_all(exp.Select)):
                if facets and self._facet_table(select):
                    changed = True
            return parsed.sql(dialect="postgres") if changed else sql
        except SqlglotError as exc:  # the unchanged query is always correct
            log.warning("Summary rewrite skipped: %s", exc)
            return sql

    @staticmethod
    def _own_nodes(select: Any, kind: Any) -> list:
        """Nodes of *kind* in *select* that are not inside a nested SELECT."""
        return [n for n in select.find_all(kind)
                if n is not select and n.find_ancestor(exp.Select) is select]

    def _stored_subject_count(self, select: Any) -> bool:
        """COUNT(DISTINCT o.subject) over the LEFT JOIN on bids_objects -> d.n_subjects.

        Only when that count is all the join is used for: no other column of
        the joined files, no other aggregate in the same SELECT.
        """
        source = select.args.get("from") or select.args.get("from_")
        table = source.this if source is not None else None
        if not isinstance(table, exp.Table) or table.name.lower() != "bids_datasets":
            return False
        d = table.alias_or_name
        joins = select.args.get("joins") or []
        target = None
        for join in joins:
            t = join.this
            if isinstance(t, exp.Table) and t.name.lower() == "bids_objects" and (join.side or "").upper() == "LEFT":
                target = join
                break
        if target is None:
            return False
        o = target.this.alias_or_name
        on = target.args.get("on")
        if on is None:
            return False
        link = False
        for part in self._flatten_and(on):
            text_ = part.sql(dialect="postgres").lower().replace(" ", "")
            if text_ in (f"{o}.dataset_id={d}.id".lower(), f"{d}.id={o}.dataset_id".lower()):
                link = True
            elif text_ not in (f"not{o}.subjectisnull".lower(), f"{o}.subjectisnotnull".lower()):
                return False
        if not link:
            return False
        # Every other use of the joined files must be COUNT(DISTINCT o.subject).
        counts = []
        for column in select.find_all(exp.Column):
            if column.table != o or column.find_ancestor(exp.Join) is target:
                continue
            distinct = column.parent
            count = distinct.parent if isinstance(distinct, exp.Distinct) else None
            if (column.name.lower() != "subject" or not isinstance(count, exp.Count)
                    or len(distinct.expressions) != 1 or count.find_ancestor(exp.Select) is not select):
                return False
            counts.append(count)
        if not counts:
            return False
        if any(a not in counts for a in self._own_nodes(select, exp.AggFunc)):
            return False
        group = select.args.get("group")
        grouped = [g.sql(dialect="postgres").lower() for g in (group.expressions if group else [])]
        if f"{d}.id".lower() not in grouped:
            return False
        stored = (
            f"COALESCE({d}.{_schema_capabilities.SUBJECT_COUNT_COLUMN}, "
            f"(SELECT COUNT(DISTINCT sc.subject) FROM bids_objects AS sc "
            f"WHERE sc.dataset_id = {d}.id AND NOT sc.subject IS NULL))"
        )
        for count in counts:
            count.replace(parse_one(stored, read="postgres"))
        select.set("joins", [j for j in joins if j is not target] or None)
        if not select.args.get("joins") and grouped == [f"{d}.id".lower()]:
            # One row per dataset already: no grouping needed, and a HAVING
            # on the stored number is an ordinary condition.
            select.set("group", None)
            having = select.args.get("having")
            if having is not None:
                where = select.args.get("where")
                condition = having.this if where is None else exp.and_(exp.paren(where.this), exp.paren(having.this))
                select.set("where", exp.Where(this=condition))
                select.set("having", None)
        return True

    def _facet_table(self, select: Any) -> bool:
        """EXISTS / IN subquery on bids_objects that looks only at datatype, suffix, task."""
        source = select.args.get("from") or select.args.get("from_")
        table = source.this if source is not None else None
        if not isinstance(table, exp.Table) or table.name.lower() != "bids_objects":
            return False
        if any(select.args.get(k) for k in ("joins", "group", "having", "limit", "order", "distinct")):
            return False
        parent = select.parent
        in_exists = isinstance(parent, exp.Exists)
        in_list = (isinstance(parent, exp.Subquery) and isinstance(parent.parent, exp.In)
                   and len(select.expressions) == 1 and isinstance(select.expressions[0], exp.Column)
                   and select.expressions[0].name.lower() == "dataset_id")
        if not (in_exists or in_list):
            return False
        alias = table.alias_or_name
        for column in select.find_all(exp.Column):
            if column.table == alias or not column.table:
                if column.name.lower() not in self._FACET_COLUMNS:
                    return False
        if any(isinstance(n, exp.Star) for n in select.find_all(exp.Star)):
            return False
        table.set("this", exp.to_identifier(_schema_capabilities.FACETS_TABLE))
        if not table.alias:
            table.set("alias", exp.TableAlias(this=exp.to_identifier("bids_objects")))
        return True

    def _normalize_sql(self, sql: str) -> str:
        sql = sql.strip().rstrip(";").strip()
        if parse_one is None:
            return sql
        try:
            parsed = parse_one(sql, read="postgres")
        except SqlglotError:
            return sql
        return parsed.sql(dialect="postgres")

    def _rewrite_projection(self, sql: str) -> str:
        """Replace the generated projection with the canonical dataset columns."""
        if _PLACEHOLDER in sql:
            sql = sql.replace(_PLACEHOLDER, self.projection)

        if parse_one is None or exp is None:
            return self._rewrite_projection_fallback(sql)

        try:
            parsed = parse_one(sql, read="postgres")
        except SqlglotError:
            return self._rewrite_projection_fallback(sql)

        if not isinstance(parsed, exp.Select):
            return self._rewrite_projection_fallback(sql)

        if self._projection_expressions is not None:
            parsed.set("expressions", [item.copy() for item in self._projection_expressions])
        return parsed.sql(dialect="postgres")

    def _rewrite_projection_fallback(self, sql: str) -> str:
        """String-based projection rewrite used when sqlglot is unavailable."""
        if _PLACEHOLDER in sql:
            return sql.replace(_PLACEHOLDER, self.projection)

        if re.match(r"\s*WITH\b", sql, re.IGNORECASE):
            return sql

        cols_start, from_start = self._outer_select_span(sql)
        if cols_start is None or from_start is None:
            return sql
        return sql[:cols_start] + self.projection + "\n" + sql[from_start:]

    def _outer_select_span(self, sql: str) -> tuple[Optional[int], Optional[int]]:
        select_start = self._find_top_level_select(sql)
        if select_start is None:
            return None, None

        m = re.match(r"\s*SELECT\s+(?:DISTINCT\s+)?", sql[select_start:], re.IGNORECASE)
        if not m:
            return None, None

        cols_start = select_start + m.end()
        depth = 0
        i = cols_start
        while i < len(sql) - 3:
            ch = sql[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif depth == 0:
                word = sql[i : i + 4].upper()
                if word == "FROM" and (i == 0 or (not sql[i - 1].isalnum() and sql[i - 1] != "_")):
                    return cols_start, i
            i += 1
        return cols_start, None

    def _find_top_level_select(self, sql: str) -> Optional[int]:
        depth = 0
        i = 0
        while i < len(sql) - 5:
            ch = sql[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif depth == 0:
                if sql[i : i + 6].upper() == "SELECT" and self._is_word_boundary(sql, i, i + 6):
                    return i
            i += 1
        return None

    @staticmethod
    def _is_word_boundary(sql: str, start: int, end: int) -> bool:
        before = sql[start - 1] if start > 0 else " "
        after = sql[end] if end < len(sql) else " "
        return (not before.isalnum() and before != "_") and (not after.isalnum() and after != "_")

    def _strip_outer_scan_filters(self, sql: str) -> str:
        """Remove hallucinated outer scan filters from the generated SQL."""
        if parse_one is None or exp is None:
            return self._strip_outer_scan_filters_fallback(sql)[0]

        try:
            parsed = parse_one(sql, read="postgres")
        except SqlglotError:
            return self._strip_outer_scan_filters_fallback(sql)[0]

        if not isinstance(parsed, exp.Select):
            return sql

        where = parsed.args.get("where")
        if where is None:
            return parsed.sql(dialect="postgres")

        predicates = self._flatten_and(where.this)
        kept: list[Any] = []
        stripped_any = False
        for predicate in predicates:
            if self._is_hallucinated_scan_filter(predicate):
                stripped_any = True
                log.warning("Stripped bad outer scan filter: %s", predicate.sql(dialect="postgres"))
                continue
            kept.append(predicate)

        if not stripped_any:
            return parsed.sql(dialect="postgres")

        if not kept:
            parsed.set("where", None)
            return parsed.sql(dialect="postgres")

        new_condition = self._combine_and(kept)
        parsed.set("where", exp.Where(this=new_condition))
        return parsed.sql(dialect="postgres")

    def _strip_outer_scan_filters_fallback(self, sql: str) -> tuple[str, bool]:
        """Legacy string-based cleanup for scan filters.

        The fallback keeps the service functional in environments where sqlglot
        is not installed yet, while preserving the previous behavior.
        """
        out: list[str] = []
        depth = 0
        pos = 0
        stripped_any = False
        while pos < len(sql):
            ch = sql[pos]
            if ch == "(":
                depth += 1
                out.append(ch)
                pos += 1
            elif ch == ")":
                depth -= 1
                out.append(ch)
                pos += 1
            elif depth == 0:
                m = _BAD_OUTER_FILTER.match(sql, pos)
                if m:
                    log.warning("Stripped bad outer scan filter: %s", m.group(0).strip())
                    pos = m.end()
                    stripped_any = True
                else:
                    out.append(ch)
                    pos += 1
            else:
                out.append(ch)
                pos += 1
        result = "".join(out)
        result = re.sub(
            r"\bWHERE\s+(GROUP|ORDER|HAVING|LIMIT)\b",
            r"\1",
            result,
            flags=re.IGNORECASE,
        )
        return result, stripped_any

    def _flatten_and(self, node: Any) -> list[Any]:
        if exp is not None and isinstance(node, exp.Paren):
            return self._flatten_and(node.this)
        if exp is not None and isinstance(node, exp.And):
            return self._flatten_and(node.this) + self._flatten_and(node.expression)
        return [node]

    def _combine_and(self, predicates: Iterable[Any]) -> Any:
        predicates = list(predicates)
        if not predicates:
            raise ValueError("cannot combine an empty predicate list")
        combined = predicates[0]
        for predicate in predicates[1:]:
            if exp is None:
                raise RuntimeError("sqlglot is unavailable")
            combined = exp.And(this=combined, expression=predicate)
        return combined

    def _is_hallucinated_scan_filter(self, predicate: Any) -> bool:
        if exp is None or not isinstance(predicate, exp.EQ):
            return False
        left = predicate.this
        right = predicate.expression
        if not isinstance(left, exp.Column):
            return False
        table = (left.table or "").lower() if getattr(left, "table", None) else ""
        column = (left.name or "").lower()
        if not table.startswith("o"):
            return False
        if column not in {"task", "datatype", "suffix"}:
            return False
        return isinstance(right, exp.Literal)

    # BIDS names the scanner model ManufacturersModelName; DICOM, and a few
    # datasets' sidecars, ManufacturerModelName (production 2026-11-18: 1,093
    # datasets the first, 52 the second, 48 of them on files without the
    # first). Either key the model writes is read as both, in the expression
    # crawlers/migrate_add_sidecar_indexes.sql indexes.
    _SCANNER_MODEL_KEYS = ("ManufacturersModelName", "ManufacturerModelName")

    def _unify_scanner_model_key(self, sql: str) -> str:
        """Read the scanner model under both of its sidecar keys.

        <alias>.other_entities ->> 'ManufacturerModelName' (or the BIDS key)
        becomes COALESCE(<alias>.other_entities ->> 'ManufacturersModelName',
        <alias>.other_entities ->> 'ManufacturerModelName').
        """
        if parse_one is None or exp is None or "ModelName" not in sql:
            return sql
        try:
            parsed = parse_one(sql, read="postgres")
        except SqlglotError:
            return sql
        changed = False
        for node in list(parsed.find_all(exp.JSONExtractScalar)):
            if self._json_key(node.expression) not in self._SCANNER_MODEL_KEYS:
                continue
            if isinstance(node.parent, exp.Coalesce):
                continue  # already the unified expression (or written that way)
            source = node.this.sql(dialect="postgres")
            node.replace(parse_one(
                "COALESCE(" + ", ".join(f"{source} ->> '{k}'" for k in self._SCANNER_MODEL_KEYS) + ")",
                read="postgres"))
            changed = True
        if not changed:
            return sql
        log.info("Read the scanner model under both sidecar keys")
        return parsed.sql(dialect="postgres")

    @staticmethod
    def _json_key(key: Any) -> Optional[str]:
        """The top-level key of ->> 'Key': a one-step JSONPath in sqlglot, a string literal in older versions."""
        if isinstance(key, exp.JSONPath):
            steps = [s for s in key.expressions if not isinstance(s, exp.JSONPathRoot)]
            if len(steps) == 1 and isinstance(steps[0], exp.JSONPathKey):
                return str(steps[0].this)
            return None
        if isinstance(key, exp.Literal) and key.is_string:
            return key.this
        return None

    def _guard_json_number_casts(self, sql: str) -> str:
        """Make number casts of JSON values safe: a non-number no longer crashes the query.

        Sidecar and participants.tsv values are free text ("n/a", "7T",
        "unknown"), so CAST(o.other_entities ->> 'MagneticFieldStrength' AS
        DOUBLE PRECISION) fails the whole query on the first such value and the
        user gets the fallback sample. Each such cast becomes
            CASE WHEN <value> ~ '<number>' THEN CAST(<value> AS <type>) END
        so a non-number is NULL and simply does not match. Casts already inside
        a CASE that tests the same value are left alone.
        """
        if parse_one is None or exp is None or "->>" not in sql:
            return sql
        try:
            parsed = parse_one(sql, read="postgres")
        except SqlglotError:
            return sql
        changed = False
        for cast in list(parsed.find_all(exp.Cast)):
            inner = cast.this.unnest() if isinstance(cast.this, exp.Paren) else cast.this
            if not isinstance(inner, exp.JSONExtractScalar):
                continue
            if not cast.to.is_type(*exp.DataType.NUMERIC_TYPES):
                continue
            value_sql = inner.sql(dialect="postgres")
            if self._already_guarded(cast, value_sql):
                continue
            guarded = parse_one(
                f"CASE WHEN ({value_sql}) ~ '{_NUMBER_PATTERN}' "
                f"THEN CAST(({value_sql}) AS {cast.to.sql(dialect='postgres')}) END",
                read="postgres",
            )
            cast.replace(guarded)
            changed = True
        if not changed:
            return sql
        log.info("Guarded number casts of JSON values")
        return parsed.sql(dialect="postgres")

    @staticmethod
    def _already_guarded(node: Any, value_sql: str) -> bool:
        parent = node.parent
        while parent is not None:
            if isinstance(parent, exp.If):
                condition = parent.this.sql(dialect="postgres")
                if value_sql in condition and "~" in condition:
                    return True
            parent = parent.parent
        return False

    # p.sex = 'female', p2.sex IN ('F', 'female'), p.sex <> 'male' — never an
    # expression already wrapped in LOWER(...) or UPPER(...).
    _SEX_EQ_RE = re.compile(r"(?<![\w.(])(\w+\.sex)\s*(=|<>|!=)\s*'([^']*)'", re.IGNORECASE)
    _SEX_IN_RE = re.compile(r"(?<![\w.(])(\w+\.sex)\s+(NOT\s+IN|IN)\s*\(([^()]*)\)", re.IGNORECASE)
    _SEX_SYNONYMS = {"f": ("female", "f"), "m": ("male", "m"), "female": ("female", "f"), "male": ("male", "m")}

    @classmethod
    def _sex_values(cls, literal: str) -> list[str]:
        value = literal.strip().lower()
        return list(cls._SEX_SYNONYMS.get(value, (value,)))

    def _case_insensitive_sex(self, sql: str) -> str:
        """Compare participant sex without regard to case or abbreviation.

        participants.tsv values reach the database as typed: 'female' for most
        datasets, but 'Female' in 48 and 'FEMALE' in 4 (2026-11-14), so
        p.sex = 'female' missed 42 of the datasets OpenNeuro's sex filter
        finds. Each equality or IN list on a sex column becomes LOWER(col)
        compared with lower-case values, 'f'/'m' and 'female'/'male' standing
        for each other.
        """
        def values_sql(values: list[str]) -> str:
            return ", ".join("'" + v.replace("'", "''") + "'" for v in dict.fromkeys(values))

        def eq(m: re.Match) -> str:
            values = self._sex_values(m.group(3))
            negated = m.group(2) != "="
            if len(values) == 1:
                return f"LOWER({m.group(1)}) {'<>' if negated else '='} {values_sql(values)}"
            return f"LOWER({m.group(1)}) {'NOT IN' if negated else 'IN'} ({values_sql(values)})"

        def in_list(m: re.Match) -> str:
            literals = re.findall(r"'((?:[^']|'')*)'", m.group(3))
            if not literals or re.sub(r"'(?:[^']|'')*'|[\s,]", "", m.group(3)):
                return m.group(0)  # not a plain list of string literals
            values = [v for lit in literals for v in self._sex_values(lit.replace("''", "'"))]
            return f"LOWER({m.group(1)}) {m.group(2).upper()} ({values_sql(values)})"

        rewritten = self._SEX_IN_RE.sub(in_list, self._SEX_EQ_RE.sub(eq, sql))
        if rewritten != sql:
            log.info("Made sex comparisons case-insensitive")
        return rewritten

    # p.handedness = 'left', p2.handedness IN ('L', 'left'), p.handedness ~* '\mleft\M',
    # p.handedness ILIKE '%left%' — never an expression already wrapped in LOWER(...).
    _HANDEDNESS_EQ_RE = re.compile(r"(?<![\w.(])(\w+\.handedness)\s*(=|<>|!=)\s*'([^']*)'", re.IGNORECASE)
    _HANDEDNESS_IN_RE = re.compile(
        r"(?<![\w.(])(\w+\.handedness)\s+(NOT\s+IN|IN)\s*\(([^()]*)\)", re.IGNORECASE)
    _HANDEDNESS_PATTERN_RE = re.compile(
        r"(?<![\w.(])(\w+\.handedness)\s+(NOT\s+I?LIKE|I?LIKE|!~\*?|~\*?)\s+(?:E)?'((?:[^']|'')*)'",
        re.IGNORECASE)
    # The stored codes (RAG/value_mappings.yaml, handedness) and what a
    # request or the SQL model says for them.
    _HANDEDNESS_CODES = ("right_handed", "left_handed", "ambidextrous", "unknown_handedness")
    _HANDEDNESS_SYNONYMS = {
        **dict.fromkeys(("l", "lh", "left", "lefthanded", "left_hander", "lefthander",
                         "left_handers", "lefthanders", "sinistral", "sinistrals", "lefty"),
                        "left_handed"),
        **dict.fromkeys(("r", "rh", "right", "righthanded", "right_hander", "righthander",
                         "right_handers", "righthanders", "dextral", "dextrals"),
                        "right_handed"),
        **dict.fromkeys(("a", "ambi", "ambidexter", "ambidextrals", "mixed", "mixed_handed",
                         "both", "either"),
                        "ambidextrous"),
        **dict.fromkeys(("unknown", "n/a", "na"), "unknown_handedness"),
    }

    @classmethod
    def _handedness_code(cls, literal: str) -> Optional[str]:
        """The stored code a handedness value stands for, or None if unclear."""
        value = re.sub(r"[\s-]+", "_", literal.strip().lower())
        if value in cls._HANDEDNESS_CODES:
            return value
        if value in cls._HANDEDNESS_SYNONYMS:
            return cls._HANDEDNESS_SYNONYMS[value]
        for prefix, code in (("left", "left_handed"), ("right", "right_handed"), ("ambi", "ambidextrous")):
            if value.startswith(prefix):
                return code
        return None

    @classmethod
    def _handedness_pattern_code(cls, pattern: str) -> Optional[str]:
        """The one code a LIKE or regex pattern on handedness is after, or None."""
        words = set(re.findall(r"[a-z]+", re.sub(r"\\[mMyYbB]", " ", pattern.lower())))
        found = {code for code in (cls._handedness_code(w) for w in words if len(w) > 1) if code}
        return found.pop() if len(found) == 1 else None

    def _handedness_codes(self, sql: str) -> str:
        """Compare participant handedness with the stored codes.

        participants.tsv values are stored as right_handed, left_handed,
        ambidextrous (a few 'Ambidextrous' / 'AMBIDEXTROUS') and
        unknown_handedness. The SQL model wrote p2.handedness = 'left' and
        p.handedness ~* '\\mleft\\M' (the underscore is a word character, so
        \\M never matches inside 'left_handed'): four real requests for
        left-handed participants found 0 datasets (2026-11-18), where 99
        datasets list left-handed participants and 16 list at least 20.

        Each equality, IN list, LIKE / ILIKE or regex match on a handedness
        column becomes LOWER(col) compared with the stored code. A value or
        pattern that names no handedness ('3.34', '%' alone) is left as it is.
        """
        def quoted(values: list[str]) -> str:
            return ", ".join(f"'{v}'" for v in dict.fromkeys(values))

        def eq(m: re.Match) -> str:
            code = self._handedness_code(m.group(3))
            if code is None:
                return m.group(0)
            return f"LOWER({m.group(1)}) {'=' if m.group(2) == '=' else '<>'} '{code}'"

        def in_list(m: re.Match) -> str:
            literals = re.findall(r"'((?:[^']|'')*)'", m.group(3))
            if not literals or re.sub(r"'(?:[^']|'')*'|[\s,]", "", m.group(3)):
                return m.group(0)  # not a plain list of string literals
            codes = [code for lit in literals
                     if (code := self._handedness_code(lit.replace("''", "'"))) is not None]
            if len(codes) < len(literals):
                return m.group(0)  # a value with no stored code
            return f"LOWER({m.group(1)}) {m.group(2).upper()} ({quoted(codes)})"

        def pattern(m: re.Match) -> str:
            code = self._handedness_pattern_code(m.group(3).replace("''", "'"))
            if code is None:
                return m.group(0)
            negated = m.group(2).upper().startswith(("NOT", "!"))
            return f"LOWER({m.group(1)}) {'<>' if negated else '='} '{code}'"

        rewritten = self._HANDEDNESS_PATTERN_RE.sub(
            pattern, self._HANDEDNESS_IN_RE.sub(in_list, self._HANDEDNESS_EQ_RE.sub(eq, sql)))
        if rewritten != sql:
            log.info("Compared handedness with the stored codes")
        return rewritten

    def _rewrite_task_ilike_concepts(self, sql: str) -> str:
        """Replace vague `task ILIKE '%term%'` patterns with precise task IN lists.

        The LLM violates its own instructions and emits task ILIKE when it can't
        find a canonical code. "naturalistic" is a paradigm category covering many
        movie/video/narrative task identifiers that don't contain the word itself.
        "cognitive" is not a BIDS task name at all — every experimental task is
        cognitive, so we drop the filter (replaced with TRUE inside EXISTS).
        """
        def _expand(m: re.Match) -> str:
            alias = m.group(1)
            term = m.group(2).lower()
            if term in _TASK_CONCEPT_EXPANSIONS:
                tasks_sql = ", ".join(
                    f"'{t}'" for t in _TASK_CONCEPT_EXPANSIONS[term]
                )
                log.info("Expanded task ILIKE '%%%s%%' to IN list (%d tasks)", term, len(_TASK_CONCEPT_EXPANSIONS[term]))
                return f"{alias}.task IN ({tasks_sql})"
            if term in _TASK_CONCEPT_DROP:
                log.info("Dropped vague task ILIKE '%%%s%%' (not a BIDS task code)", term)
                return "TRUE"
            return m.group(0)

        return _TASK_ILIKE_CONCEPT_RE.sub(_expand, sql)

    @staticmethod
    def _ancestors(node: Any, root: Any) -> list:
        """The nodes between *node* (exclusive) and *root* (inclusive)."""
        found = []
        current = node.parent
        while current is not None:
            found.append(current)
            if current is root:
                break
            current = current.parent
        return found

    def _participants_join_to_subqueries(self, sql: str) -> str:
        """Replace a JOIN of bids_participants next to bids_objects by subqueries.

        "datasets with at least 100 children aged 12 or younger" came back as
            FROM bids_datasets d LEFT JOIN bids_objects o ... LEFT JOIN bids_participants p
            WHERE p.age <= 12 GROUP BY d.id HAVING COUNT(DISTINCT p.participant_id) >= 100
        which pairs every participant with every file of the dataset. On
        2026-11-14 one such query wrote temporary files until the server's disk
        was full ("No space left on device"), failing every other write too.

        The join goes; the participant conditions of its ON and of WHERE become
            EXISTS (SELECT 1 FROM bids_participants p WHERE p.dataset_id = d.id AND <conditions>)
        and every aggregate over the participant columns (COUNT, AVG …) becomes
        the same aggregate in a correlated subquery with those conditions.
        Left alone when a condition mixes participant and other columns, or a
        participant column is used outside an aggregate in HAVING/ORDER BY.
        """
        if parse_one is None or exp is None or not re.search(r"\bJOIN\s+bids_participants\b", sql, re.IGNORECASE):
            return sql
        try:
            select = parse_one(sql, read="postgres")
        except SqlglotError:
            return sql
        if not isinstance(select, exp.Select):
            return sql
        joins = select.args.get("joins") or []
        part = [j for j in joins if isinstance(j.this, exp.Table) and j.this.name.lower() == "bids_participants"]
        if len(part) != 1 or not any(isinstance(j.this, exp.Table) and j.this.name.lower() == "bids_objects"
                                     for j in joins):
            return sql
        from_ = select.args.get("from_") or select.args.get("from")
        if from_ is None or not isinstance(from_.this, exp.Table):
            return sql
        d_alias = from_.this.alias_or_name
        join = part[0]
        p_alias = join.this.alias_or_name

        def own_columns(node) -> list:
            # Columns of this query level; a subquery may reuse the alias p for its own table.
            return [c for c in node.find_all(exp.Column)
                    if not any(isinstance(a, (exp.Subquery, exp.Exists)) for a in self._ancestors(c, node))]

        def tables(node) -> set:
            return {c.table for c in own_columns(node)}

        link = None
        conditions = []
        on = join.args.get("on")
        for conj in (self._flatten_and(on) if on is not None else []):
            sql_text = conj.sql(dialect="postgres").replace(" ", "").lower()
            if sql_text in (f"{p_alias}.dataset_id={d_alias}.id", f"{d_alias}.id={p_alias}.dataset_id"):
                link = conj
            elif tables(conj) <= {p_alias}:
                conditions.append(conj)
            else:
                return sql
        if link is None:
            return sql
        where = select.args.get("where")
        kept = []
        for conj in (self._flatten_and(where.this) if where is not None else []):
            used = tables(conj)
            if p_alias in used:
                if used != {p_alias}:
                    return sql
                conditions.append(conj)
            else:
                kept.append(conj)

        def scoped(inner: str) -> str:
            cond = " AND ".join([f"{p_alias}.dataset_id = {d_alias}.id"]
                                + [c.sql(dialect="postgres") for c in conditions])
            return f"(SELECT {inner} FROM bids_participants AS {p_alias} WHERE {cond})"

        for key in ("having", "order"):
            part_node = select.args.get(key)
            if part_node is None:
                continue
            for agg in [a for a in part_node.find_all(exp.AggFunc)
                        if not any(isinstance(x, (exp.Subquery, exp.Exists)) for x in self._ancestors(a, part_node))]:
                if tables(agg) == {p_alias}:
                    agg.replace(parse_one(scoped(agg.sql(dialect="postgres")), read="postgres"))
            if p_alias in tables(part_node):
                return sql
        if any(p_alias in tables(projection) for projection in select.expressions):
            return sql

        select.set("joins", [j for j in joins if j is not join])
        if conditions:
            kept.append(parse_one(f"EXISTS {scoped('1')}", read="postgres"))
        if kept:
            select.set("where", exp.Where(this=self._combine_and(kept)))
        else:
            select.set("where", None)
        log.info("Replaced the bids_participants join next to bids_objects by subqueries")
        return select.sql(dialect="postgres")

    def _rewrite_participant_count_to_having(self, sql: str) -> str:
        """Move correlated bids_participants COUNT from WHERE to HAVING on o.subject.

        bids_participants is sparsely populated — many datasets have rows in
        bids_objects but zero rows in bids_participants. The reliable subject
        count is COUNT(DISTINCT o.subject) which is already joined in the query.
        """
        m = _PARTICIPANTS_COUNT_WHERE_RE.search(sql)
        if not m:
            return self._participant_count_in_having(sql)
        op = m.group(1)
        n = m.group(2)
        log.info(
            "Rewrote bids_participants COUNT subquery to HAVING COUNT(DISTINCT o.subject) %s %s",
            op, n,
        )
        # Remove the subquery from WHERE
        sql = sql[: m.start()] + sql[m.end() :]
        having_clause = f"HAVING COUNT(DISTINCT o.subject) {op} {n}"
        if re.search(r"\bHAVING\b", sql, re.IGNORECASE):
            # AND into existing HAVING
            sql = re.sub(
                r"\bHAVING\b",
                f"HAVING COUNT(DISTINCT o.subject) {op} {n} AND",
                sql,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            # Insert after GROUP BY clause, before ORDER BY / LIMIT / end
            sql = re.sub(
                r"(\bGROUP\s+BY\b[^;]*?)(\s*(?:ORDER\s+BY|LIMIT|;|$))",
                lambda mm: mm.group(1) + f"\n{having_clause}" + mm.group(2),
                sql,
                count=1,
                flags=re.IGNORECASE | re.DOTALL,
            )
        return sql

    def _inject_patient_counts(self, sql: str, augmented_question: str) -> str:
        """Put the patient count of the filter lines into the query, whatever the model wrote.

        The filter line `patients (…): patients_with('<codes>') >= N` is the
        whole condition. The model usually writes its own count instead
        (a subquery counting bids_participants rows with the diagnosis), which
        only sees coded rows: its comparison is removed, the placeholder is
        ANDed in. A changed prompt would shift the SQL of unrelated questions;
        this does not.
        """
        wanted = list(dict.fromkeys(
            m.group(1) for m in _PATIENTS_LINE_RE.finditer(augmented_question or "")))
        if not wanted or parse_one is None or exp is None:
            return sql
        try:
            parsed = parse_one(sql, read="postgres")
            if not isinstance(parsed, exp.Select):
                return sql
            for clause, node_type in (("where", exp.Where), ("having", exp.Having)):
                holder = parsed.args.get(clause)
                if holder is None:
                    continue
                kept = [part for part in self._flatten_and(holder.this)
                        if not self._is_diagnosis_count(part)]
                if len(kept) == len(self._flatten_and(holder.this)):
                    continue
                if kept:
                    parsed.set(clause, node_type(this=exp.and_(*[exp.paren(k.copy()) for k in kept])))
                else:
                    parsed.set(clause, None)
            sql = parsed.sql(dialect="postgres")
        except SqlglotError as exc:
            log.warning("Could not remove the model's patient count: %s", exc)
        present = {" ".join(m.group(0).lower().split()) for m in _cohort_count.PLACEHOLDER_RE.finditer(sql)}
        for condition in wanted:
            if " ".join(condition.lower().split()) not in present:
                sql = self._and_into_where(sql, condition)
        return sql

    @staticmethod
    def _is_diagnosis_count(node: Any) -> bool:
        """A comparison of a count of bids_participants rows that have a diagnosis."""
        while isinstance(node, exp.Paren):
            node = node.this
        if not isinstance(node, (exp.GTE, exp.GT, exp.EQ, exp.LTE, exp.LT)):
            return False
        for side in (node.this, node.expression):
            while isinstance(side, exp.Paren):
                side = side.this
            if not isinstance(side, exp.Subquery):
                continue
            inner = side.this
            if not isinstance(inner, exp.Select) or inner.find(exp.Count) is None:
                continue
            tables = {t.name.lower() for t in inner.find_all(exp.Table)}
            columns = {c.name.lower() for c in inner.find_all(exp.Column)}
            if tables == {"bids_participants"} and "diagnosis" in columns:
                return True
        return False

    def _participants_join_to_subject_count(self, sql: str) -> str:
        """JOIN bids_participants only to count them -> the dataset's subject count.

        "Datasets with at least 100 participants": the model joins
        bids_participants next to the file join and writes HAVING
        COUNT(DISTINCT p.participant_id) >= 100. The two joins multiply (every
        file row times every participant row): on production the sort filled
        the disk after 70 seconds. The size of a dataset is its subject count,
        as in the other count rewrite above; a third of the datasets has no
        participant rows at all. Left alone when the query reads any other
        column of the joined participants (a count of filtered participants).
        """
        if parse_one is None or exp is None or "bids_participants" not in sql.lower():
            return sql
        try:
            parsed = parse_one(sql, read="postgres")
            select = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
            if select is None:  # e.g. TABLE x; the guard refuses it
                return sql
            source = select.args.get("from") or select.args.get("from_")
            table = source.this if source is not None else None
            if not isinstance(table, exp.Table) or table.name.lower() != "bids_datasets":
                return sql
            d = table.alias_or_name.lower()
            joins = list(select.args.get("joins") or [])
            target = next((j for j in joins if isinstance(j.this, exp.Table)
                           and j.this.name.lower() == "bids_participants"), None)
            if target is None:
                return sql
            p = target.this.alias_or_name.lower()
            on = target.args.get("on")
            link = on.sql(dialect="postgres").lower().replace(" ", "") if on is not None else ""
            if link not in (f"{p}.dataset_id={d}.id", f"{d}.id={p}.dataset_id"):
                return sql
            counts = []
            for column in select.find_all(exp.Column):
                if (column.table or "").lower() != p or column.find_ancestor(exp.Join) is target:
                    continue
                if column.find_ancestor(exp.Select) is not select:
                    return sql  # a subquery reads the joined participants
                count = column.find_ancestor(exp.Count)
                if column.name.lower() != "participant_id" or count is None:
                    return sql
                counts.append(count)
            if not counts:
                return sql
            files = next((j.this.alias_or_name for j in joins if isinstance(j.this, exp.Table)
                          and j.this.name.lower() == "bids_objects"), None)
            subjects = (f"COUNT(DISTINCT {files}.subject)" if files else
                        f"(SELECT COUNT(DISTINCT cnt_o.subject) FROM bids_objects AS cnt_o "
                        f"WHERE cnt_o.dataset_id = {table.alias_or_name}.id)")
            for count in {id(c): c for c in counts}.values():
                count.replace(parse_one(subjects, read="postgres"))
            select.set("joins", [j for j in joins if j is not target])
            log.info("Rewrote the count over joined bids_participants to the subject count")
            return parsed.sql(dialect="postgres")
        except SqlglotError as exc:  # the unchanged query is still the model's
            log.warning("Participants-join rewrite skipped: %s", exc)
            return sql

    def _drop_healthy_filters(self, sql: str) -> str:
        """Remove conditions that require a recorded healthy diagnosis.

        1,331 of the 1,718 classified datasets include healthy participants,
        and 175 record it as a diagnosis. Requiring the record (or the word
        "healthy" in the text) removed 77 of 230 correct answers of one real
        request and all 15 of another; not filtering keeps them and lets in
        the few patient-only datasets. So "healthy" never filters. A list
        that mixes healthy codes with a disease stays as it is.
        """
        if parse_one is None or exp is None or "diagnosis" not in sql.lower():
            return sql
        try:
            parsed = parse_one(sql, read="postgres")
            changed = False
            for node in list(parsed.find_all(exp.EQ, exp.In)):
                column = node.this
                if not isinstance(column, exp.Column) or column.name.lower() != "diagnosis":
                    continue
                if isinstance(node, exp.EQ):
                    values = [node.expression]
                else:
                    values = node.expressions
                    if not values or node.args.get("query") is not None:
                        continue
                if not all(isinstance(v, exp.Literal) and v.is_string and _is_healthy_code(v.this)
                           for v in values):
                    continue
                if node.find_ancestor(exp.Not) is not None:
                    continue  # "no healthy participants" is a real restriction
                node.replace(exp.true())
                changed = True
            if not changed:
                return sql

            def simplify(node):
                inner = node.unnest() if isinstance(node, exp.Paren) else node
                if isinstance(inner, exp.And):
                    left, right = simplify(inner.left), simplify(inner.right)
                    if isinstance(left, exp.Boolean) and left.this:
                        return right
                    if isinstance(right, exp.Boolean) and right.this:
                        return left
                    return exp.and_(left, right)
                if isinstance(inner, exp.Or):
                    left, right = simplify(inner.left), simplify(inner.right)
                    if any(isinstance(side, exp.Boolean) and side.this for side in (left, right)):
                        return exp.true()
                    return exp.paren(exp.or_(left, right))
                if isinstance(inner, exp.Exists):
                    sub = inner.this
                    where = sub.args.get("where") if isinstance(sub, exp.Select) else None
                    if where is not None and not sub.args.get("group"):
                        rest = simplify(where.this)
                        # Only the link to the dataset is left: "has any
                        # participants row" is not what was asked.
                        if isinstance(rest, exp.Boolean) and rest.this or (
                                isinstance(rest, exp.EQ)
                                and all(isinstance(side, exp.Column) for side in (rest.this, rest.expression))):
                            return exp.true()
                        where.set("this", rest)
                    return inner
                return node

            for where in list(parsed.find_all(exp.Where)):
                if where.find_ancestor(exp.Exists) is not None:
                    continue  # handled through its EXISTS
                rest = simplify(where.this)
                if isinstance(rest, exp.Boolean) and rest.this:
                    where.pop()
                else:
                    where.set("this", rest)
            log.info("Dropped the healthy-diagnosis requirement")
            return parsed.sql(dialect="postgres")
        except SqlglotError:
            return sql

    def _decorrelate_grouped_exists(self, sql: str) -> str:
        """Turn EXISTS (... WHERE x.dataset_id = d.id GROUP BY ... HAVING ...) into d.id IN (...).

        "At least 3 sessions per subject" was written as a NOT EXISTS with a
        GROUP BY that the database evaluates once per dataset: it ran into the
        300 s limit and the request fell back to an unfiltered sample. The
        same test as one grouped subquery over the whole table takes seconds.
        """
        if parse_one is None or exp is None or "having" not in sql.lower():
            return sql
        try:
            parsed = parse_one(sql, read="postgres")
            changed = False
            for node in list(parsed.find_all(exp.Exists)):
                sub = node.this
                if not isinstance(sub, exp.Select) or not sub.args.get("group") or not sub.args.get("having"):
                    continue
                where = sub.args.get("where")
                if where is None or sub.args.get("joins"):
                    continue
                conjuncts = self._flatten_and(where.this)
                link = None
                for c in conjuncts:
                    if (isinstance(c, exp.EQ) and isinstance(c.this, exp.Column)
                            and isinstance(c.expression, exp.Column)):
                        names = {(c.this.table.lower(), c.this.name.lower()),
                                 (c.expression.table.lower(), c.expression.name.lower())}
                        inner = [x for x in (c.this, c.expression) if x.name.lower() == "dataset_id"]
                        if ("d", "id") in names and inner:
                            link, inner_col = c, inner[0]
                            break
                if link is None:
                    continue
                rest = [c for c in conjuncts if c is not link]
                if any(col.table.lower() == "d" for c in rest for col in c.find_all(exp.Column)):
                    continue
                if rest:
                    where.set("this", self._combine_and(rest))
                else:
                    where.pop()
                group = sub.args["group"]
                group.set("expressions", [inner_col.copy()] + list(group.expressions))
                sub.set("expressions", [inner_col.copy()])
                replacement = exp.In(this=exp.column("id", table="d"), query=exp.Subquery(this=sub.copy()))
                node.replace(replacement)
                changed = True
            if not changed:
                return sql
            log.info("Rewrote a per-dataset grouped EXISTS into one grouped subquery")
            return parsed.sql(dialect="postgres")
        except SqlglotError:
            return sql

    def _without_text_only_filters(self, sql: str) -> Optional[str]:
        """*sql* without the top-level AND parts that are free-text searches alone.

        The fallback for an empty result: a free-text condition (words the
        vocabulary does not know, a descriptive phrase) is the weakest part of
        a query, and the one that most often matches nothing. Conditions on
        recorded facts, and text searches that back up such a condition
        (diagnosis, task, topic), stay. None when nothing can be dropped or
        nothing would remain.
        """
        if parse_one is None or exp is None or "text_match" not in sql.lower():
            return None
        try:
            parsed = parse_one(sql, read="postgres")
            where = parsed.args.get("where") if isinstance(parsed, exp.Select) else None
            if where is None:
                return None

            def parts(node):
                node = node.unnest() if isinstance(node, exp.Paren) else node
                if isinstance(node, exp.And):
                    return parts(node.left) + parts(node.right)
                return [node]

            def text_only(node) -> bool:
                node = node.unnest() if isinstance(node, exp.Paren) else node
                if isinstance(node, exp.Or):
                    return text_only(node.left) and text_only(node.right)
                return isinstance(node, exp.Anonymous) and node.name.lower() == "text_match"

            conjuncts = parts(where.this)
            kept = [c for c in conjuncts if not text_only(c)]
            if len(kept) == len(conjuncts) or not kept:
                return None
            merged = kept[0]
            for c in kept[1:]:
                merged = exp.and_(merged, c)
            parsed.set("where", exp.Where(this=merged))
            return parsed.sql(dialect="postgres")
        except SqlglotError:
            return None

    def _drop_invented_text_filters(self, sql: str, augmented_question: str) -> str:
        """Remove text conditions the SQL model added on its own.

        Preprocessing decides which words are searched: every text_match() it
        wants is in the augmented question. The SQL model adds more for words
        that describe the request ("standard manner", "30 trials per task"),
        or repeats as text what it already filtered by a table ("mriqc &
        derivatives" next to the bids_derivatives lookup); such a condition
        only empties the result. A top-level AND part is dropped when it
        consists of text_match() calls alone and none of them comes from
        preprocessing. Mixed parts (a column test OR a text search, as in the
        species recipes) and negated searches stay, and so does the last
        remaining condition.
        """
        if parse_one is None or exp is None or "text_match" not in sql.lower():
            return sql

        def key(argument: str) -> tuple:
            return tuple(sorted(str(term) for term in _parse_alternatives(argument)))

        allowed = {key(m.group(1)) for m in _TEXT_MATCH_CALL_RE.finditer(augmented_question)}
        try:
            parsed = parse_one(sql, read="postgres")
            where = parsed.args.get("where") if isinstance(parsed, exp.Select) else None
            if where is None:
                return sql

            def parts(node):
                node = node.unnest() if isinstance(node, exp.Paren) else node
                if isinstance(node, exp.And):
                    return parts(node.left) + parts(node.right)
                return [node]

            def text_only(node) -> bool:
                node = node.unnest() if isinstance(node, exp.Paren) else node
                if isinstance(node, exp.Or):
                    return text_only(node.left) and text_only(node.right)
                return isinstance(node, exp.Anonymous) and node.name.lower() == "text_match"

            def invented(node) -> bool:
                calls = [c for c in node.find_all(exp.Anonymous) if c.name.lower() == "text_match"]
                arguments = [c.expressions[0].this for c in calls
                             if c.expressions and isinstance(c.expressions[0], exp.Literal)]
                return bool(calls) and len(arguments) == len(calls) and all(
                    key(a) not in allowed for a in arguments)

            conjuncts = parts(where.this)
            kept = [c for c in conjuncts if not (text_only(c) and invented(c))]
            if len(kept) == len(conjuncts) or not kept:
                return sql
            dropped = [c.sql(dialect="postgres") for c in conjuncts if c not in kept]
            log.info("Dropped text filters the SQL model invented: %s", "; ".join(dropped))
            merged = kept[0]
            for c in kept[1:]:
                merged = exp.and_(merged, c)
            parsed.set("where", exp.Where(this=merged))
            return parsed.sql(dialect="postgres")
        except SqlglotError:
            return sql
    @staticmethod
    def _participant_count_in_having(sql: str) -> str:
        """Count a dataset's participants from both sources in HAVING.

        "at least 100 participants" came back as
            HAVING (SELECT COUNT(DISTINCT p.participant_id) FROM bids_participants AS p
                    WHERE p.dataset_id = d.id) >= 100
        which counts only participants.tsv rows: ds000158 (217 subject folders,
        no participants rows) counted as 0, and 22 of the 156 datasets
        OpenNeuro's subject-count filter finds were missing. The larger of
        the two counts is the dataset's size.
        """
        having = re.search(r"\bHAVING\b", sql, re.IGNORECASE)
        if not having:
            return sql
        tail = _PARTICIPANTS_COUNT_SUBQUERY_RE.sub(
            lambda m: f"GREATEST(COUNT(DISTINCT o.subject), {m.group(0)})", sql[having.start():])
        if tail == sql[having.start():]:
            return sql
        log.info("Counted participants from participants.tsv and subject folders in HAVING")
        return sql[:having.start()] + tail

    def _inject_vocab_miss(self, sql: str, augmented_question: str) -> str:
        """Add the free-text filter for unresolved vocabulary terms if the model left it out.

        The [VOCABULARY MISS] block of the augmented question carries one
        text_match() placeholder per unresolved term. A placeholder the model
        already copied (same search terms) is not added twice.
        """
        cleaned, _ = self._strip_outer_scan_filters_fallback(sql)
        hinted: list[str] = []
        for hint in _VOCAB_MISS_HINT_RE.finditer(augmented_question):
            hinted.extend(m.group(0) for m in _TEXT_MATCH_CALL_RE.finditer(hint.group(1)))
        if not hinted:
            return cleaned

        def key(call: str) -> tuple:
            argument = _TEXT_MATCH_CALL_RE.match(call).group(1)  # type: ignore[union-attr]
            return tuple(sorted(str(term) for term in _parse_alternatives(argument)))

        present = {key(m.group(0)) for m in _TEXT_MATCH_CALL_RE.finditer(cleaned)}
        missing = [call for call in dict.fromkeys(hinted) if key(call) not in present]
        if not missing:
            return cleaned

        combined = " AND ".join(missing)
        log.info("Injecting VOCABULARY MISS text filter: %s", combined)
        return self._and_into_where(cleaned, combined)

    def _inject_topics(self, sql: str, augmented_question: str) -> str:
        """Add a topic's filter if the model left it out.

        A topic is copied when its first text_match() (the user's own phrase)
        is in the SQL. Otherwise the whole condition is ANDed in: without it
        the search would ignore the subject the question is about.
        """
        present = {
            tuple(sorted(str(t) for t in _parse_alternatives(m.group(1))))
            for m in _TEXT_MATCH_CALL_RE.finditer(sql)
        }
        for line in _TOPIC_LINE_RE.finditer(augmented_question):
            condition = line.group("condition").strip()
            first = _TEXT_MATCH_CALL_RE.search(condition)
            if first is None:
                continue
            key = tuple(sorted(str(t) for t in _parse_alternatives(first.group(1))))
            if key in present:
                continue
            log.info("Injecting topic filter the model left out: %s", condition)
            sql = self._and_into_where(sql, condition)
            present.add(key)
        return sql

    def _restore_description_alternatives(self, sql: str, augmented_question: str) -> str:
        """Put back the other words of a description search the model left out.

        The 'dataset text:' line can read
            (text_match('dead people') OR text_match('post-mortem | ex vivo', 'also'))
        (preprocess/topics.py, check_description_alternatives). A model that copied
        only the first search would find what the user's own words find,
        which for "dead people" is nothing.

        A model that left the line out is left alone: it usually said the
        same with a table ("preprocessed using fMRIPrep" → bids_derivatives,
        "with a paper" → bids_dataset_papers). ANDing the line in anyway
        (deployed a069533 to 049e939) cut those results on the gold requests:
        fMRIPrep 345 → 38 datasets, recall 94% → 10%.
        """
        line = _DESCRIPTION_LINE_RE.search(augmented_question)
        if line is None:
            return sql
        condition = line.group("condition").strip()
        calls = list(_TEXT_MATCH_CALL_RE.finditer(condition))
        if len(calls) != 2:
            return sql

        def key(match: re.Match) -> tuple:
            return tuple(sorted(str(t) for t in _parse_alternatives(match.group(1))))

        own, others = key(calls[0]), key(calls[1])
        in_sql = list(_TEXT_MATCH_CALL_RE.finditer(sql))
        if any(key(m) == others for m in in_sql):
            return sql
        for match in in_sql:
            if key(match) != own:
                continue
            if re.search(r"\bNOT\s*$", sql[:match.start()], re.IGNORECASE):
                continue
            log.info("Restoring description alternatives the model left out: %s", condition)
            return sql[:match.start()] + condition + sql[match.end():]
        return sql

    def _restore_description_formula(self, sql: str, augmented_question: str) -> str:
        """AND back the exclusions of a description formula the model left out.

        A formula line (RAG/text_formula.py) reads, for "memory but not sleep",
            dataset text: text_match('memory') AND NOT text_match('sleep')
        Its top-level AND parts are separate conditions. A model that copied
        only the first would search for memory and keep the sleep studies.

        Only the NOT parts are put back. A positive part the model left out
        is usually covered by a table it chose instead (fMRIPrep →
        bids_derivatives); ANDing the text back in emptied such results
        (gold requests, 2026-11-17: recall 70% → 62–64% with the same
        fallback for description alternatives). An exclusion cannot replace
        a table, and putting it back only removes what the user ruled out.
        """
        line = _ANY_DESCRIPTION_LINE_RE.search(augmented_question)
        if line is None:
            return sql
        parts = [p for p in _top_level_and_parts(line.group("condition").strip())
                 if re.match(r"NOT\s+text_match\s*\(", p, re.IGNORECASE)]
        if not parts:
            return sql

        def key(argument: str) -> tuple:
            return tuple(sorted(str(t) for t in _parse_alternatives(argument)))

        def negated_before(text: str, position: int) -> bool:
            return bool(re.search(r"\bNOT\s*\(?\s*$", text[:position], re.IGNORECASE))

        in_sql = {(key(m.group(1)), negated_before(sql, m.start()))
                  for m in _TEXT_MATCH_CALL_RE.finditer(sql)}
        for part in parts:
            wanted = {(key(m.group(1)), negated_before(part, m.start()))
                      for m in _TEXT_MATCH_CALL_RE.finditer(part)}
            if not wanted or wanted <= in_sql:
                continue
            log.info("Injecting the exclusion of the description formula the model left out: %s", part)
            sql = self._and_into_where(sql, part)
        return sql

    def _and_into_where(self, sql: str, condition: str) -> str:
        """AND *condition* onto the outer WHERE, keeping the existing filter intact.

        Into every branch of a UNION / INTERSECT / EXCEPT. SQL sqlglot cannot
        parse is returned unchanged: the guard refuses it (query_guard.
        check_static), so self-correction rewrites it. The string edit used
        before put the condition in front of the old filter without brackets
        (with a top-level OR it bound one branch only), into the first WHERE
        even inside a subquery, and dropped it when there was neither WHERE
        nor GROUP BY.
        """
        if parse_one is None or exp is None:
            return sql
        try:
            parsed = parse_one(sql, read="postgres")
            extra = exp.condition(condition, dialect="postgres")
        except SqlglotError as exc:
            log.warning("Condition not added to unparsable SQL (%s): %s", exc, condition)
            return sql

        def add(node) -> bool:
            if isinstance(node, exp.Subquery):
                return add(node.this)
            if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
                return add(node.left) & add(node.right)
            if not isinstance(node, exp.Select):
                return False
            where = node.args.get("where")
            # Parenthesise the old filter: it may contain a top-level OR.
            merged = exp.and_(extra.copy(), exp.paren(where.this)) if where else extra.copy()
            node.set("where", exp.Where(this=merged))
            return True

        if not add(parsed):
            log.warning("Condition not added to a %s query: %s", type(parsed).__name__, condition)
            return sql
        return parsed.sql(dialect="postgres")

    def _inject_relevance_cte(
        self, sql: str, scored_filters: list, rank_tsquery: Optional[str] = None
    ) -> str:
        """Wrap the base SQL so each row gets a score and the result is ordered by it.

        relevance_score  sum of the RAG scores of the canonical codes the
                         dataset really has (structured match)
        text_rank        0..1 full-text rank of the dataset text against the
                         question's free-text terms (services/fulltext/text_search.py);
                         far below 0 for a dataset that matches only the
                         other words of a search, not the user's own

        Datasets are ordered by the sum, so a dataset with a diagnosis row and
        the term in its title beats one with only either, and a dataset that
        merely mentions the term in its README comes last.
        """
        rows: list[str] = []
        field_cols: dict[str, tuple[str, str]] = {}
        for scored_filter in scored_filters or []:
            if isinstance(scored_filter, dict):
                field = scored_filter.get("field")
                code = scored_filter.get("code")
                score = scored_filter.get("score")
            else:
                field = getattr(scored_filter, "field", None)
                code = getattr(scored_filter, "code", None)
                score = getattr(scored_filter, "score", None)

            if field not in _SCORABLE_FIELDS or not code or score is None:
                continue
            if not _CODE_RE.match(str(code)):
                continue

            col_info = _field_to_db_col(field)
            if col_info is None:
                continue
            field_cols[field] = col_info
            rows.append(f"('{field}', '{code}', {float(score):.4f})")

        text_rank = ranking.rank_expression("ft", rank_tsquery)
        use_text_rank = text_rank is not None
        if not rows and not use_text_rank:
            return sql

        base_sql = self._normalize_sql(sql)
        with_clause = ""
        relevance_expr = "0"
        if rows:
            values_block = ",\n    ".join(rows)
            exists_clauses = [
                f"(mc.field = '{field}' AND EXISTS (SELECT 1 FROM {table} mt "
                f"WHERE mt.dataset_id = base.id AND mt.{col} = mc.code))"
                for field, (table, col) in field_cols.items()
            ]
            where_expr = "\n           OR ".join(exists_clauses)
            with_clause = (
                "WITH matched_codes (field, code, score) AS (\n"
                f"  VALUES\n    {values_block}\n"
                ")\n"
            )
            relevance_expr = (
                "COALESCE((\n"
                "    SELECT SUM(mc.score) FROM matched_codes mc\n"
                f"    WHERE {where_expr}\n"
                "  ), 0)"
            )

        text_rank_expr = "0"
        text_join = ""
        if use_text_rank:
            text_rank_expr = text_rank  # type: ignore[assignment]
            text_join = "\n  LEFT JOIN bids_datasets ft ON ft.id = base.id"

        # ORDER BY cannot add two output aliases in the same SELECT, hence the
        # extra level.
        return (
            f"{with_clause}"
            "SELECT * FROM (\n"
            "  SELECT base.*,\n"
            f"  {relevance_expr} AS relevance_score,\n"
            f"  {text_rank_expr} AS text_rank\n"
            f"  FROM (\n{base_sql}\n) AS base{text_join}\n"
            ") AS ranked\n"
            "ORDER BY ranked.relevance_score + ranked.text_rank DESC, "
            "ranked.subject_count DESC NULLS LAST, ranked.id"
        )


def _has_top_level_order(sql: str) -> bool:
    """True when the outermost SELECT has its own ORDER BY."""
    if parse_one is not None:
        try:
            return parse_one(sql, read="postgres").args.get("order") is not None
        except SqlglotError:
            pass
    # Without a parse: an ORDER BY after the last closing parenthesis.
    return re.search(r"\bORDER\s+BY\b", sql[sql.rfind(")") + 1:], re.IGNORECASE) is not None


sql_rewrite_service = SqlRewriteService()


def extract_sql(raw: str) -> str:
    return sql_rewrite_service.extract_sql(raw)


def rewrite_generated_sql(raw_sql: str, augmented_question: str,
                          word_forms: Optional[Mapping[str, Sequence[str]]] = None,
                          ) -> RewriteResult:
    return sql_rewrite_service.rewrite_generated_sql(raw_sql, augmented_question, word_forms)


def finish_corrected_sql(raw: str, failed_sql: str) -> str:
    return sql_rewrite_service.finish_corrected_sql(raw, failed_sql)


def build_count_sql(base_sql: str) -> str:
    return sql_rewrite_service.build_count_sql(base_sql)


def build_page_sql(
    base_sql: str,
    scored_filters: list,
    apply_relevance: bool,
    limit: int,
    offset: int,
    rank_tsquery: Optional[str] = None,
    rerank_ordered: Optional[Sequence[str]] = None,
    rerank_demoted: Optional[Sequence[str]] = None,
) -> tuple[str, dict[str, Any]]:
    return sql_rewrite_service.build_page_sql(
        base_sql, scored_filters, apply_relevance, limit, offset, rank_tsquery=rank_tsquery,
        rerank_ordered=rerank_ordered, rerank_demoted=rerank_demoted,
    )
