"""
sql_rewriter.py
---------------
SQL post-processing for the Text-to-SQL pipeline.

This module owns the parts of the pipeline that operate on SQL after the LLM
has already produced a draft query:
  - extracting SQL from fenced or annotated model output,
  - canonical projection replacement,
  - stripping hallucinated scan filters,
  - injecting VOCABULARY MISS fallback predicates,
  - expanding text_match() placeholders into full-text predicates,
  - wrapping relevance scoring, text ranking and pagination.

The LLM-facing prompt generation stays in ``services.text_to_sql`` so the
translation stack has a clean separation between "generate SQL" and "rewrite
SQL".
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# Make project-local modules importable regardless of working directory.
_ROOT = Path(__file__).resolve().parents[2]
for _mod_dir in (
    _ROOT / "backend",
    _ROOT / "LLM_preprocessor",
    _ROOT / "RAG",
):
    if _mod_dir.exists() and str(_mod_dir) not in sys.path:
        sys.path.insert(0, str(_mod_dir))

try:  # pragma: no cover - exercised in environments that have sqlglot installed
    from sqlglot import exp, parse_one
except ImportError:  # pragma: no cover - keep the service usable when the dep is absent locally
    exp = None  # type: ignore[assignment]
    parse_one = None  # type: ignore[assignment]

from join_registry import field_to_db_col as _field_to_db_col  # noqa: E402
from text_terms import parse_alternatives as _parse_alternatives  # noqa: E402

from services import text_search as _text_search  # noqa: E402

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


_SQL_END_MARKER_RE = re.compile(r"\[/SQL\]|\[SQL\]|```", re.IGNORECASE)


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
    # Full-text query that ranks the result; None without text terms.
    rank_tsquery: Optional[str] = None


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
        except Exception as exc:  # pragma: no cover - depends on sqlglot availability
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

    def rewrite_generated_sql(self, raw_sql: str, augmented_question: str) -> RewriteResult:
        """Normalize raw LLM output into executable SQL plus bind parameters.

        The method intentionally returns a structured result instead of mutating
        the caller's state. That keeps the translation pipeline easy to cache and
        makes the rewrite behavior testable in isolation.
        """
        sql = self.extract_sql(raw_sql)
        sql = self._rewrite_projection(sql)
        sql = self._strip_outer_scan_filters(sql)
        sql = self._rewrite_task_ilike_concepts(sql)
        sql = self._rewrite_participant_count_to_having(sql)
        sql = self._guard_json_number_casts(sql)
        sql = self._inject_vocab_miss(sql, augmented_question)
        expanded = _text_search.expand_text_search(sql)
        return RewriteResult(sql=expanded.sql, params={}, rank_tsquery=expanded.rank_tsquery)

    def build_count_sql(self, base_sql: str) -> str:
        """Wrap a cached base query in a COUNT(*) query for pagination metadata."""
        base = self._normalize_sql(base_sql)
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
        re-ranker (services/rerank.py): the first come first in that order,
        the second last, everything else keeps the relevance order between.
        Their parameters are added after the last sqlglot pass, which would
        otherwise re-render them.
        """
        ordered = (
            self._inject_relevance_cte(base_sql, scored_filters, rank_tsquery)
            if apply_relevance else base_sql
        )
        ranked = ordered != base_sql
        ordered = self._normalize_sql(ordered)
        params: dict[str, Any] = {"_limit": limit, "_offset": offset}
        order_clause = ""
        if rerank_ordered or rerank_demoted:
            params["_rr_top"] = [str(i) for i in (rerank_ordered or [])]
            params["_rr_low"] = [str(i) for i in (rerank_demoted or [])]
            tail = (
                "page_result.relevance_score + page_result.text_rank DESC, "
                if ranked else ""
            ) + "page_result.subject_count DESC NULLS LAST, page_result.id"
            order_clause = (
                "ORDER BY CASE WHEN page_result.id = ANY(CAST(:_rr_top AS uuid[])) THEN 0\n"
                "              WHEN page_result.id = ANY(CAST(:_rr_low AS uuid[])) THEN 2\n"
                "              ELSE 1 END,\n"
                "         array_position(CAST(:_rr_top AS uuid[]), page_result.id),\n"
                f"         {tail}\n"
            )
        sql = (
            f"SELECT * FROM (\n{ordered}\n) AS page_result\n"
            f"{order_clause}"
            "LIMIT :_limit OFFSET :_offset"
        )
        return sql, params

    def _normalize_sql(self, sql: str) -> str:
        sql = sql.strip().rstrip(";").strip()
        if parse_one is None:
            return sql
        try:
            parsed = parse_one(sql, read="postgres")
        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
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

    def _rewrite_participant_count_to_having(self, sql: str) -> str:
        """Move correlated bids_participants COUNT from WHERE to HAVING on o.subject.

        bids_participants is sparsely populated — many datasets have rows in
        bids_objects but zero rows in bids_participants. The reliable subject
        count is COUNT(DISTINCT o.subject) which is already joined in the query.
        """
        m = _PARTICIPANTS_COUNT_WHERE_RE.search(sql)
        if not m:
            return sql
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

    def _and_into_where(self, sql: str, condition: str) -> str:
        """AND *condition* onto the outer WHERE, keeping the existing filter intact."""
        if parse_one is not None and exp is not None:
            try:
                parsed = parse_one(sql, read="postgres")
                extra = exp.condition(condition, dialect="postgres")
                if isinstance(parsed, exp.Select):
                    where = parsed.args.get("where")
                    # Parenthesise the old filter: it may contain a top-level OR.
                    merged = exp.and_(extra, exp.paren(where.this)) if where else extra
                    parsed.set("where", exp.Where(this=merged))
                    return parsed.sql(dialect="postgres")
            except Exception:
                pass  # fall through to the string-based edit

        if re.search(r"\bWHERE\b", sql, re.IGNORECASE):
            return re.sub(
                r"\bWHERE\b", f"WHERE {condition} AND", sql, count=1, flags=re.IGNORECASE
            )
        return re.sub(
            r"\bGROUP\s+BY\b", f"WHERE {condition}\nGROUP BY", sql, count=1, flags=re.IGNORECASE
        )

    def _inject_relevance_cte(
        self, sql: str, scored_filters: list, rank_tsquery: Optional[str] = None
    ) -> str:
        """Wrap the base SQL so each row gets a score and the result is ordered by it.

        relevance_score  sum of the RAG scores of the canonical codes the
                         dataset really has (structured match)
        text_rank        0..1 full-text rank of the dataset text against the
                         question's free-text terms (services/text_search.py)

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

        use_text_rank = _text_search.is_safe_tsquery(rank_tsquery)
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
            text_rank_expr = _text_search.text_rank_expression(
                f"ft.{_text_search.FTS_COLUMN}", rank_tsquery  # type: ignore[arg-type]
            )
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
            "ranked.subject_count DESC NULLS LAST"
        )


sql_rewrite_service = SqlRewriteService()


def extract_sql(raw: str) -> str:
    return sql_rewrite_service.extract_sql(raw)


def rewrite_generated_sql(raw_sql: str, augmented_question: str) -> RewriteResult:
    return sql_rewrite_service.rewrite_generated_sql(raw_sql, augmented_question)


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
