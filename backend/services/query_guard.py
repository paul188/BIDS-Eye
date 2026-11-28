"""
Guards that keep one generated query from taking the server down.

The search turns a question into SQL that nobody reviews before it runs. On
2026-11-14 "datasets with at least 100 children aged 12 or younger" joined
every participant with every file of each dataset; the sort wrote temporary
files until the server's disk was full, and every other write failed with it.
Fixing that one query shape is not enough, so every generated query passes
four independent guards:

1. check_static(): the SQL must be one plain SELECT that reads only the
   dataset catalogue (bids_* tables), never the app's own tables such as
   users or query_log, which share the database; no CROSS or comma joins
   (a LATERAL subquery excepted), no join without a condition, no recursive
   CTE, no functions that sleep, generate rows, build huge strings or touch
   files and server state, and never bids_participants and bids_objects
   joined in the same query level (each is per-file or per-participant, so
   the join multiplies them).
2. check_plan(): EXPLAIN (no execution) must not expect more rows at any step
   than a few times the largest table. Normal searches stay at or below one
   scan of bids_objects (2.7 million rows in 2026-11); the join above was
   estimated at 12 million and was far larger in reality.
3. apply_limits(): each transaction that runs generated SQL gets a statement
   timeout, a cap on temporary file space and a bounded sort memory, so a
   query that slips through fails alone instead of starving the server. JIT
   compilation is off: it can take longer than the search itself.
4. slot(): at most SEARCH_MAX_CONCURRENT generated queries run at the same
   time (the server has 2 cores); a request that waits too long gets a
   "busy" error instead of piling up.

A refused query raises QueryRefusedError. The router treats it like a database
error: the SQL goes to self-correction with the reason, and if that fails
too the user gets the fallback sample.

Settings (environment):
  SEARCH_STATEMENT_TIMEOUT   default '120s' (sidecar scans take up to ~6 min today)
  SEARCH_TEMP_FILE_LIMIT     default '1GB'   (needs a superuser role; skipped otherwise)
  SEARCH_WORK_MEM            default '32MB'
  SEARCH_RANKED_PAGE_TIMEOUT default '25s' (then the page is served unranked)
  SEARCH_MAX_CONCURRENT      default 2
  SEARCH_SLOT_WAIT_SECONDS   default 30
  SEARCH_MAX_PLAN_ROWS       default: 3 × the row estimate of the largest table
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from typing import Any, AsyncIterator, Optional

from config.settings import Settings, settings_obj
from exceptions.domain import DomainError
from exceptions.expected import ExpectedError

from sqlglot import exp, parse
from sqlglot.errors import SqlglotError

from services.schema.tables import datasets as _datasets, objects as _objects, participants as _participants

log = logging.getLogger(__name__)


class QueryRefusedError(ExpectedError):
    """A generated query that must not run; the message says why."""


class SearchBusyError(
    DomainError, status_code=503, detail="The search is busy. Please try again in a minute."
):
    """No slot for a generated query became free in time."""


# ── 1. static check ───────────────────────────────────────────────────────────

# Functions a dataset search never needs and that can sleep, produce rows or
# bytes without bound, or reach server state and files.
_FORBIDDEN_FUNCTIONS = frozenset({
    "pg_sleep", "pg_sleep_for", "pg_sleep_until", "generate_series", "repeat", "lpad", "rpad",
    "dblink", "dblink_exec", "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "lo_get", "set_config", "pg_terminate_backend", "pg_cancel_backend",
    "pg_reload_conf", "pg_advisory_lock", "query_to_xml", "query_to_json", "pg_notify", "nextval",
    "setval", "txid_current", "pg_logical_emit_message", "current_setting",
    # these run the SQL text they are given, or dump whole tables
    "ts_stat", "query_to_xml_and_xmlschema", "query_to_xmlschema", "cursor_to_xml", "cursor_to_xmlschema",
    "table_to_xml", "table_to_xmlschema", "table_to_xml_and_xmlschema", "schema_to_xml",
    "schema_to_xmlschema", "schema_to_xml_and_xmlschema", "database_to_xml", "database_to_xmlschema",
    "database_to_xml_and_xmlschema", "dblink_connect", "dblink_open", "dblink_fetch", "dblink_send_query",
    "pg_file_write", "lo_from_bytea", "lo_put",
})
_PER_ROW_TABLES = (_objects.name, _participants.name)

# The only tables generated SQL may read: the dataset catalogue, which is the
# bids_* tables plus object_associations. The app keeps its own tables in the
# same database (other users' questions in query_log, stored translations;
# database/app_tables.py), and a question can steer the
# model into a subquery on them. tests/test_query_guard.py checks that none of
# the app's tables match this rule.
_CATALOGUE_PREFIX = "bids_"
_CATALOGUE_TABLES = frozenset({"object_associations"})


def _is_catalogue_table(name: str) -> bool:
    name = name.lower()
    return name.startswith(_CATALOGUE_PREFIX) or name in _CATALOGUE_TABLES


def _is_visible_cte(table: Any) -> bool:
    """The table is a CTE of this query or of one that encloses it. A CTE of
    another subquery does not count: it cannot stand for a real table here."""
    name = table.name.lower()
    node = table.parent
    while node is not None:
        for value in node.args.values():
            if isinstance(value, exp.With) and any(cte.alias_or_name.lower() == name for cte in value.expressions):
                return True
        node = node.parent
    return False


def _outside_catalogue(root: Any) -> Optional[str]:
    """The first table this query reads that is not part of the dataset catalogue."""
    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            continue  # a function in FROM (jsonb_to_recordset(...)); _FORBIDDEN_FUNCTIONS covers those
        schema, catalog = table.db, table.catalog
        if catalog or schema.lower() not in ("", "public"):
            return ".".join(part for part in (catalog, schema, table.name) if part)
        if not schema and _is_visible_cte(table):
            continue
        if not _is_catalogue_table(table.name):
            return table.name
    return None


def _function_name(node: Any) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return (node.sql_name() if hasattr(node, "sql_name") else type(node).__name__).lower()


def _scope_tables(select: Any) -> list:
    """Tables read directly by this SELECT level (FROM and JOINs, not subqueries)."""
    tables = []
    from_ = select.args.get("from_") or select.args.get("from")
    sources = ([from_.this] if from_ is not None else []) + [j.this for j in select.args.get("joins") or []]
    for source in sources:
        if isinstance(source, exp.Table):
            tables.append(source.name.lower())
    return tables


_SUBJECT_COLUMNS = ("subject", "participant_id")


def _conjuncts(node: Any) -> list:
    """The AND-ed parts of a condition (an OR stays one part)."""
    node = node.unnest() if isinstance(node, exp.Paren) else node
    if isinstance(node, exp.And):
        return _conjuncts(node.this) + _conjuncts(node.expression)
    return [node]


def _column_equalities(join: Any) -> list:
    """(left, right) columns of each `a.x = b.y` that must hold for the join."""
    on = join.args.get("on")
    pairs = []
    for part in (_conjuncts(on) if on is not None else []):
        if isinstance(part, exp.EQ) and isinstance(part.this, exp.Column) and isinstance(part.expression, exp.Column):
            pairs.append((part.this, part.expression))
    return pairs


def _links_tables(join: Any) -> bool:
    """The join condition ties a column of the joined table to another table's.
    ON TRUE, ON 1 = 1 or ON a.x = 1 OR ... join (nearly) every row with every row."""
    if join.args.get("using"):
        return True
    return any(left.table != right.table for left, right in _column_equalities(join))


def _matches_subject(join: Any) -> bool:
    """The join pairs rows of the same subject ("MEG and rest in the same
    subject", a file with its participant row): bounded by one subject's rows."""
    return any(left.name.lower() in _SUBJECT_COLUMNS and right.name.lower() in _SUBJECT_COLUMNS
               for left, right in _column_equalities(join))


def _singleton_tsquery_ctes(root: Any) -> set[str]:
    """Names of top-level CTEs guaranteed to return exactly one safe row.

    The uncertainty tier factors literal TO_TSQUERY calls into a SELECT with
    no FROM clause, then joins that single row to candidate/evidence rows. The
    generic CROSS JOIN rule must still reject arbitrary CTEs and row sources.
    """
    with_clause = root.args.get("with_") or root.args.get("with")
    if with_clause is None:
        return set()
    safe = set()
    for cte in with_clause.expressions:
        query = cte.this
        if not isinstance(query, exp.Select):
            continue
        if any(query.args.get(key) for key in (
            "from", "from_", "joins", "where", "group", "having", "qualify",
            "order", "limit", "offset", "distinct",
        )):
            continue
        projections = query.expressions
        if not projections:
            continue
        valid = True
        for projection in projections:
            expression = projection.this if isinstance(projection, exp.Alias) else projection
            if not isinstance(expression, exp.Anonymous) or str(expression.this).upper() != "TO_TSQUERY":
                valid = False
                break
            arguments = expression.expressions
            if (len(arguments) != 2 or any(not isinstance(arg, exp.Literal) or not arg.is_string
                                          for arg in arguments)
                    or str(arguments[0].this).lower() not in {"english", "simple"}):
                valid = False
                break
        if valid:
            safe.add(cte.alias_or_name.lower())
    return safe


def check_static(sql: str) -> Optional[str]:
    """The reason this SQL must not run, or None.

    SQL that sqlglot cannot parse is refused: the rules below cannot be
    checked on it, and Postgres accepts some of it (``EXISTS (TABLE users)``
    reads a table outside the catalogue). sqlglot parsed every one of the
    2,867 distinct queries of the benchmark runs and the 319 few-shot
    examples (2026-11-18), so this refuses nothing a search writes today."""
    try:
        statements = [s for s in parse(sql, read="postgres") if s is not None]
    except SqlglotError as exc:
        return f"the SQL could not be checked ({exc}); write one plain PostgreSQL SELECT"
    if len(statements) != 1:
        return "more than one SQL statement"
    root = statements[0]
    if not isinstance(root, (exp.Select, exp.Union, exp.Subquery)):
        return f"not a SELECT ({type(root).__name__})"
    singleton_tsquery_ctes = _singleton_tsquery_ctes(root)
    for node in root.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Command)):
            return f"writes or commands are not allowed ({type(node).__name__})"
        if isinstance(node, exp.Select) and node.args.get("into"):
            return "SELECT INTO creates a table"
        if isinstance(node, exp.Select) and node.args.get("locks"):
            return "row locks (FOR UPDATE / FOR SHARE) are not allowed"
        if isinstance(node, exp.With) and node.args.get("recursive"):
            return "recursive CTE"
        if isinstance(node, exp.GenerateSeries):  # not an exp.Func in sqlglot
            return "function generate_series() is not allowed"
        if isinstance(node, exp.Pad):  # lpad / rpad, which sqlglot names PAD
            return "function lpad()/rpad() is not allowed"
        if isinstance(node, (exp.Func, exp.Anonymous)) and _function_name(node) in _FORBIDDEN_FUNCTIONS:
            return f"function {_function_name(node)}() is not allowed"
    for select in root.find_all(exp.Select):
        joins = select.args.get("joins") or []
        for join in joins:
            lateral = isinstance(join.this, exp.Lateral) or bool(join.this.find(exp.Lateral)) \
                or bool(join.args.get("lateral"))
            if lateral:
                continue
            kind = (join.args.get("kind") or "").upper()
            if kind == "CROSS":
                source = join.this
                if (isinstance(source, exp.Table) and not source.args.get("db")
                        and not source.args.get("catalog")
                        and source.name.lower() in singleton_tsquery_ctes):
                    continue
                return "CROSS JOIN"
            if (join.args.get("method") or "").upper() == "NATURAL":
                return "NATURAL JOIN"
            if join.args.get("on") is None and not join.args.get("using"):
                return "join without a join condition"
            if not _links_tables(join):
                return "join condition does not tie the two tables together (e.g. ON TRUE)"
        tables = _scope_tables(select)
        # Rows of bids_objects (files) and bids_participants multiply when the
        # two meet in one query level, unless every such join is per subject.
        per_row = [j for j in joins if isinstance(j.this, exp.Table) and j.this.name.lower() in _PER_ROW_TABLES]
        per_row_count = sum(tables.count(t) for t in _PER_ROW_TABLES)
        if per_row_count > 1 and not all(_matches_subject(j) for j in per_row):
            if all(t in tables for t in _PER_ROW_TABLES):
                return ("bids_participants and bids_objects joined in the same query: the join pairs every "
                        "participant with every file; use EXISTS or a correlated subquery for participants")
            return ("bids_objects or bids_participants joined more than once in the same query without "
                    "matching the subject; use EXISTS")
    foreign = _outside_catalogue(root)
    if foreign is not None:
        return f"table {foreign} is not part of the dataset catalogue; read only the bids_* tables"
    return None


# ── 2. plan check ─────────────────────────────────────────────────────────────

_max_plan_rows: Optional[float] = None
_PLAN_ROWS_FACTOR = 3.0
_PLAN_ROWS_DEFAULT = 1.0e7


async def max_plan_rows(session: Any) -> float:
    """Most rows any plan step may expect: 3 × the largest table's estimate.

    The estimate is read once, in a savepoint (*session* is the search's: a
    failed read must not abort it). After a failure this search uses
    _PLAN_ROWS_DEFAULT and the next one reads again."""
    global _max_plan_rows
    configured = settings_obj.SEARCH_MAX_PLAN_ROWS
    if configured:
        return configured
    if _max_plan_rows is None:
        from sqlalchemy import text
        from sqlalchemy.exc import DBAPIError
        try:
            async with session.begin_nested():
                biggest = (await session.execute(text(
                    "SELECT max(reltuples) FROM pg_class WHERE relname = ANY(:tables)"),
                    {"tables": [_objects.name, _participants.name, _datasets.name]})).scalar()
        except DBAPIError as exc:
            log.warning("Table sizes not read (%s); plan limit %.0f rows for this search",
                        exc.orig or exc, _PLAN_ROWS_DEFAULT)
            return _PLAN_ROWS_DEFAULT
        _max_plan_rows = max(float(biggest or 0) * _PLAN_ROWS_FACTOR, 1.0e6)
    return _max_plan_rows


def largest_step(plan_json: Any) -> tuple[float, str]:
    """(rows, node type) of the plan step that expects the most rows."""
    if isinstance(plan_json, str):
        plan_json = json.loads(plan_json)
    root = plan_json[0]["Plan"] if isinstance(plan_json, list) else plan_json["Plan"]
    best = (0.0, "")
    stack = [root]
    while stack:
        node = stack.pop()
        rows = float(node.get("Plan Rows") or 0)
        if rows > best[0]:
            best = (rows, node.get("Node Type", ""))
        stack.extend(node.get("Plans") or [])
    return best


async def check_plan(session: Any, sql: str, params: Optional[dict] = None) -> Optional[str]:
    """EXPLAIN the query (nothing runs) and refuse an expected row explosion.

    A query Postgres rejects passes here: the real execution reports the
    error, and that message is what self-correction needs. The EXPLAIN runs
    in a savepoint, since an error outside one aborts the search's
    transaction, and the next statement would report only "current
    transaction is aborted"."""
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError
    try:
        async with session.begin_nested():
            plan = (await session.execute(text(f"EXPLAIN (FORMAT JSON) {sql}"), params or {})).scalar()
    except DBAPIError as exc:
        log.info("EXPLAIN failed, the query runs unchecked by the plan guard: %s", exc.orig or exc)
        return None
    rows, node = largest_step(plan)
    limit = await max_plan_rows(session)
    if rows > limit:
        return (f"the query plan expects {rows:,.0f} rows at one step ({node}), more than the "
                f"{limit:,.0f} a dataset search may use; it probably multiplies files or participants")
    return None


# ── 3. limits per transaction ─────────────────────────────────────────────────

_SETTING_RE = re.compile(r"^[0-9]+\s*(?:ms|s|min|h|kB|MB|GB)?$", re.IGNORECASE)


def _setting(name: str) -> str:
    """Setting *name* of settings_obj; its default when it is no Postgres size or time."""
    value = str(getattr(settings_obj, name)).strip()
    return value if _SETTING_RE.match(value) else str(Settings.model_fields[name].default)


async def apply_limits(session: Any) -> None:
    """Statement timeout, temp file cap, sort memory and no JIT for this transaction."""
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError
    await session.execute(text(f"SET LOCAL statement_timeout = '{_setting('SEARCH_STATEMENT_TIMEOUT')}'"))
    await session.execute(text(f"SET LOCAL work_mem = '{_setting('SEARCH_WORK_MEM')}'"))
    # Compiling one search took 4 s, running it 0.1 s (2026-11-25).
    await session.execute(text("SET LOCAL jit = off"))
    # temp_file_limit can only be set by a superuser; a savepoint keeps a
    # refusal from aborting the transaction.
    try:
        async with session.begin_nested():
            await session.execute(text(
                f"SET LOCAL temp_file_limit = '{_setting('SEARCH_TEMP_FILE_LIMIT')}'"))
    except DBAPIError as exc:  # pragma: no cover - depends on the role
        log.warning("temp_file_limit not set: %s", exc)


def ranked_page_timeout() -> str:
    """Timeout of a relevance-ranked page; past it the router serves the page
    unranked (SEARCH_RANKED_PAGE_TIMEOUT, default '25s')."""
    return _setting("SEARCH_RANKED_PAGE_TIMEOUT")


def is_timeout(exc: BaseException) -> bool:
    """A statement cancelled by statement_timeout (asyncpg QueryCanceledError,
    possibly wrapped by SQLAlchemy)."""
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if "QueryCanceled" in type(current).__name__ or "statement timeout" in str(current):
            return True
        current = getattr(current, "orig", None) or current.__cause__
    return False


# ── 4. concurrency ────────────────────────────────────────────────────────────

_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max(1, settings_obj.SEARCH_MAX_CONCURRENT))
    return _semaphore


@contextlib.asynccontextmanager
async def slot() -> AsyncIterator[None]:
    """Hold one of the few places for running generated SQL."""
    semaphore = _get_semaphore()
    wait = settings_obj.SEARCH_SLOT_WAIT_SECONDS
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=wait)
    except asyncio.TimeoutError:
        raise SearchBusyError(f"no search slot free within {wait:.0f}s") from None
    try:
        yield
    finally:
        semaphore.release()


# ── all together ──────────────────────────────────────────────────────────────

async def vet(session: Any, base_sql: str, count_sql: str, params: Optional[dict] = None) -> None:
    """Raise QueryRefusedError when the base query must not run (guards 1 and 2)."""
    reason = check_static(base_sql)
    if reason is None:
        reason = await check_plan(session, count_sql, params)
    if reason is not None:
        log.warning("Refused generated SQL: %s", reason)
        raise QueryRefusedError(reason)

