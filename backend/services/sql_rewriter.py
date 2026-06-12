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
  - wrapping relevance scoring and pagination.

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
from typing import Any, Iterable, Optional

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

log = logging.getLogger(__name__)

_PLACEHOLDER = "{{COLS}}"
DEFAULT_DATASET_PROJECTION = (
    "d.id, d.name, d.accession_id, d.bids_version, d.dataset_type,\n"
    "       d.source_type, d.remote_url, d.validation_status,\n"
    "       d.authors, d.description_text,\n"
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
_VOCAB_MISS_ILIKE_RE = re.compile(
    r"For free-text search use:\s*(.+?)(?=\s*\])",
    re.IGNORECASE | re.DOTALL,
)
_LITERAL_QUOTE_RE = re.compile(r"'(%[^']*%)'")
_CODE_RE = re.compile(r"^[A-Za-z0-9_]+$")
_SCORABLE_FIELDS = {"diagnosis", "task", "datatype", "suffix"}


@dataclass(slots=True)
class RewriteResult:
    """Normalized SQL plus any bind parameters introduced during rewriting."""
    sql: str
    params: dict[str, Any]


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
            return m.group(1).strip()
        m = _SQL_FENCE.search(raw)
        if m and re.match(r"(?i)(select|with)\b", m.group(1).strip()):
            return m.group(1).strip()
        stripped = raw.strip()
        if re.match(r"(?i)(select|with)\b", stripped):
            return stripped.split("\n\n")[0].strip()
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
        sql, params = self._inject_vocab_miss(sql, augmented_question)
        return RewriteResult(sql=sql, params=params)

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
    ) -> tuple[str, dict[str, int]]:
        """Wrap a cached base query for one page of results.

        Relevance ordering is layered on only when requested so the cached base
        query can be reused across pages without rebaking the sort order.
        """
        ordered = self._inject_relevance_cte(base_sql, scored_filters) if apply_relevance else base_sql
        ordered = self._normalize_sql(ordered)
        sql = (
            f"SELECT * FROM (\n{ordered}\n) AS page_result\n"
            "LIMIT :_limit OFFSET :_offset"
        )
        return sql, {"_limit": limit, "_offset": offset}

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

    def _inject_vocab_miss(self, sql: str, augmented_question: str) -> tuple[str, dict[str, Any]]:
        """Inject free-text fallback filters for unresolved vocabulary terms."""
        cleaned, _ = self._strip_outer_scan_filters_fallback(sql)
        ilike_parts = [m.group(1).strip() for m in _VOCAB_MISS_ILIKE_RE.finditer(augmented_question)]
        if not ilike_parts:
            return cleaned, {}

        if re.search(r"\bILIKE\b", cleaned, re.IGNORECASE):
            return cleaned, {}

        params: dict[str, Any] = {}
        literal_to_param: dict[str, str] = {}
        parameterized_parts: list[str] = []
        for part in ilike_parts:
            matches = _LITERAL_QUOTE_RE.findall(part)
            for match in matches:
                param_name = literal_to_param.get(match)
                if param_name is None:
                    param_name = f"vocab_miss_{len(params)}"
                    literal_to_param[match] = param_name
                    params[param_name] = match
                part = part.replace(f"'{match}'", f":{param_name}")
            parameterized_parts.append(part)

        combined = " AND ".join(f"({part})" for part in parameterized_parts)
        log.info("Injecting VOCABULARY MISS ILIKE: %s", combined)

        if re.search(r"\bWHERE\b", cleaned, re.IGNORECASE):
            return (
                re.sub(
                    r"\bWHERE\b",
                    f"WHERE {combined} AND",
                    cleaned,
                    count=1,
                    flags=re.IGNORECASE,
                ),
                params,
            )

        return (
            re.sub(
                r"\bGROUP\s+BY\b",
                f"WHERE {combined}\nGROUP BY",
                cleaned,
                count=1,
                flags=re.IGNORECASE,
            ),
            params,
        )

    def _inject_relevance_cte(self, sql: str, scored_filters: list) -> str:
        """Wrap the base SQL so relevance scoring can be applied per row."""
        if not scored_filters:
            return sql

        rows: list[str] = []
        field_cols: dict[str, tuple[str, str]] = {}
        for scored_filter in scored_filters:
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

        if not rows:
            return sql

        values_block = ",\n    ".join(rows)
        exists_clauses = [
            f"(mc.field = '{field}' AND EXISTS (SELECT 1 FROM {table} mt "
            f"WHERE mt.dataset_id = base.id AND mt.{col} = mc.code))"
            for field, (table, col) in field_cols.items()
        ]
        where_expr = "\n           OR ".join(exists_clauses)
        base_sql = self._normalize_sql(sql)

        return (
            "WITH matched_codes (field, code, score) AS (\n"
            f"  VALUES\n    {values_block}\n"
            ")\n"
            "SELECT base.*,\n"
            "  COALESCE((\n"
            "    SELECT SUM(mc.score) FROM matched_codes mc\n"
            f"    WHERE {where_expr}\n"
            "  ), 0) AS relevance_score\n"
            f"FROM (\n{base_sql}\n) AS base\n"
            "ORDER BY relevance_score DESC, base.subject_count DESC NULLS LAST"
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
) -> tuple[str, dict[str, int]]:
    return sql_rewrite_service.build_page_sql(base_sql, scored_filters, apply_relevance, limit, offset)
