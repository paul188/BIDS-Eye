"""Guards against generated queries that could take the server down
(services/query_guard.py).

On 2026-11-14 one search ("datasets with at least 100 children aged 12 or
younger") joined every participant with every file and filled the server's
disk. These tests pin each guard against the families of queries that can do
that — row explosions, functions without bounds, writes and locks — and pin
that the real queries users get still pass. Generated SQL also may read only
the dataset catalogue, not the app's own tables in the same database.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from services import query_guard
from services.query_guard import QueryRefusedError, SearchBusyError, check_static, largest_step
from services.translate.sql_rewriter import SqlRewriteService

REWRITER = SqlRewriteService()
REAL_QUERIES = json.loads((Path(__file__).parent / "data" / "real_search_sql.json").read_text())["queries"]

_HEAD = ("SELECT d.id FROM bids_datasets AS d "
         "LEFT JOIN bids_objects AS o ON o.dataset_id = d.id AND NOT o.subject IS NULL ")
_GOOD = _HEAD + ("WHERE EXISTS (SELECT 1 FROM bids_participants AS p WHERE p.dataset_id = d.id AND p.age < 12) "
                 "GROUP BY d.id")


# ── what must keep working ────────────────────────────────────────────────────

@pytest.mark.parametrize("query", REAL_QUERIES, ids=[q["id"] for q in REAL_QUERIES])
def test_real_generated_queries_pass(query: dict) -> None:
    assert check_static(query["sql"]) is None


def test_real_query_sample_covers_the_known_shapes() -> None:
    joined = "\n".join(q["sql"] for q in REAL_QUERIES)
    for shape in ("CROSS JOIN LATERAL", "bids_participants", "HAVING", "search_tsv_papers", "other_entities"):
        assert shape in joined, shape
    assert len(REAL_QUERIES) >= 50


@pytest.mark.parametrize("sql", [
    _GOOD,
    # the uncertain tier's wrapper
    "SELECT tb.*, ut.lvl FROM (" + _GOOD + ") AS tb CROSS JOIN LATERAL (SELECT 1 AS lvl FROM bids_datasets AS d "
    "WHERE d.id = tb.id) AS ut WHERE ut.lvl < 9",
    "SELECT tb.* FROM (" + _GOOD + ") AS tb, LATERAL (SELECT 1 AS x) AS ut",
    # same subject has MEG and rest
    "SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM bids_objects AS a JOIN bids_objects AS b "
    "ON a.dataset_id = b.dataset_id AND a.subject = b.subject WHERE a.dataset_id = d.id)",
    # a file joined with its own participant row
    "SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM bids_objects AS o2 JOIN bids_participants AS p2 "
    "ON p2.dataset_id = o2.dataset_id AND p2.participant_id = o2.subject WHERE o2.dataset_id = d.id AND p2.age < 12)",
    # participants alone, joined (no files next to them)
    "SELECT d.id FROM bids_datasets AS d JOIN bids_participants AS p ON p.dataset_id = d.id WHERE p.age < 12 "
    "GROUP BY d.id",
    # USING and a join to a small table
    "SELECT d.id FROM bids_datasets AS d JOIN bids_derivatives AS x USING (dataset_id)",
    "SELECT d.id FROM bids_datasets AS d LEFT JOIN bids_dataset_sites AS s ON s.dataset_id = d.id",
    # functions a search does use
    "SELECT d.id FROM bids_datasets AS d WHERE LOWER(d.name) LIKE '%rest%' AND COALESCE(d.species, '') !~* 'human' "
    "AND d.publish_date >= NOW() - INTERVAL '1 year' AND d.search_tsv @@ TO_TSQUERY('english', 'memory')",
    "SELECT d.id FROM bids_datasets AS d WHERE GREATEST(d.age_min, 0) < 12 AND UNNEST(d.institutions) IS NOT NULL",
    # a CTE that is not recursive
    "WITH matched AS (SELECT 'task' AS field) SELECT d.id FROM bids_datasets AS d CROSS JOIN LATERAL "
    "(SELECT 1 FROM matched) AS m",
])
def test_safe_shapes_pass(sql: str) -> None:
    assert check_static(sql) is None


def test_unparsable_sql_is_left_to_the_database() -> None:
    assert check_static("SELEC broken (") is None


def test_cross_join_to_literal_tsquery_cte_is_safe() -> None:
    sql = (
        "WITH terms AS (SELECT TO_TSQUERY('english', 'memory:AB') AS q0, "
        "TO_TSQUERY('simple', 'memory') AS q1) "
        "SELECT d.id FROM bids_datasets AS d CROSS JOIN terms AS q "
        "WHERE d.search_tsv @@ q.q0"
    )
    assert check_static(sql) is None


def test_cross_join_to_data_cte_is_still_refused() -> None:
    sql = (
        "WITH candidates AS (SELECT o.dataset_id FROM bids_objects AS o) "
        "SELECT d.id FROM bids_datasets AS d CROSS JOIN candidates"
    )
    assert check_static(sql) == "CROSS JOIN"


# ── row explosions: files × participants ──────────────────────────────────────

_PARTICIPANT_JOINS = {
    "left join, the original": _HEAD + "LEFT JOIN bids_participants AS p ON p.dataset_id = d.id "
    "WHERE p.age <= 12 GROUP BY d.id HAVING COUNT(DISTINCT p.participant_id) >= 100",
    "inner join": _HEAD + "JOIN bids_participants AS p ON p.dataset_id = d.id GROUP BY d.id",
    "inner join keyword": _HEAD + "INNER JOIN bids_participants p ON p.dataset_id = d.id GROUP BY d.id",
    "right join": _HEAD + "RIGHT JOIN bids_participants AS p ON p.dataset_id = d.id GROUP BY d.id",
    "full join": _HEAD + "FULL OUTER JOIN bids_participants AS p ON p.dataset_id = d.id GROUP BY d.id",
    "participants first": "SELECT d.id FROM bids_datasets AS d JOIN bids_participants AS p ON p.dataset_id = d.id "
    "JOIN bids_objects AS o ON o.dataset_id = d.id GROUP BY d.id",
    "other aliases": "SELECT x.id FROM bids_datasets x LEFT JOIN bids_objects f ON f.dataset_id = x.id "
    "LEFT JOIN bids_participants q ON q.dataset_id = x.id GROUP BY x.id",
    "joined on dataset only, in EXISTS": "SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM "
    "bids_objects AS o2 JOIN bids_participants AS p2 ON p2.dataset_id = o2.dataset_id WHERE o2.dataset_id = d.id)",
    "in a CTE": "WITH x AS (SELECT o.dataset_id FROM bids_objects AS o JOIN bids_participants AS p "
    "ON p.dataset_id = o.dataset_id) SELECT d.id FROM bids_datasets AS d JOIN x ON x.dataset_id = d.id",
    "in a derived table": "SELECT d.id FROM bids_datasets AS d JOIN (SELECT o.dataset_id FROM bids_objects AS o "
    "JOIN bids_participants AS p ON p.dataset_id = o.dataset_id) AS x ON x.dataset_id = d.id",
    "in a UNION branch": "SELECT d.id FROM bids_datasets AS d UNION SELECT o.dataset_id FROM bids_objects AS o "
    "JOIN bids_participants AS p ON p.dataset_id = o.dataset_id",
    "in a scalar subquery": "SELECT d.id, (SELECT COUNT(*) FROM bids_objects AS o JOIN bids_participants AS p "
    "ON p.dataset_id = o.dataset_id WHERE o.dataset_id = d.id) AS n FROM bids_datasets AS d",
    "in HAVING": _HEAD + "GROUP BY d.id HAVING (SELECT COUNT(*) FROM bids_objects AS o3 JOIN bids_participants AS p3 "
    "ON p3.dataset_id = o3.dataset_id WHERE o3.dataset_id = d.id) > 10",
    "subject only partly matched": _HEAD + "JOIN bids_participants AS p ON p.dataset_id = d.id "
    "OR p.participant_id = o.subject GROUP BY d.id",
}


@pytest.mark.parametrize("sql", list(_PARTICIPANT_JOINS.values()), ids=list(_PARTICIPANT_JOINS))
def test_files_times_participants_is_refused(sql: str) -> None:
    reason = check_static(sql)
    assert reason is not None
    assert "bids_participants" in reason or "join" in reason.lower()


_SELF_JOINS = {
    "files twice": _HEAD + "JOIN bids_objects AS o2 ON o2.dataset_id = d.id GROUP BY d.id",
    "files twice in EXISTS": "SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM bids_objects AS a "
    "JOIN bids_objects AS b ON a.dataset_id = b.dataset_id WHERE a.dataset_id = d.id)",
    "participants twice": "SELECT d.id FROM bids_datasets AS d JOIN bids_participants AS a ON a.dataset_id = d.id "
    "JOIN bids_participants AS b ON b.dataset_id = d.id",
    "three file joins": _HEAD + "JOIN bids_objects AS o2 ON o2.dataset_id = d.id "
    "JOIN bids_objects AS o3 ON o3.dataset_id = d.id",
}


@pytest.mark.parametrize("sql", list(_SELF_JOINS.values()), ids=list(_SELF_JOINS))
def test_self_joins_without_the_subject_are_refused(sql: str) -> None:
    assert "more than once" in (check_static(sql) or "")


_UNBOUNDED_JOINS = {
    "CROSS JOIN": ("SELECT d.id FROM bids_datasets AS d CROSS JOIN bids_objects AS o", "CROSS JOIN"),
    "comma join": ("SELECT d.id FROM bids_datasets AS d, bids_objects AS o WHERE o.dataset_id = d.id",
                   "without a join condition"),
    "NATURAL JOIN": ("SELECT d.id FROM bids_datasets AS d NATURAL JOIN bids_objects AS o", "NATURAL"),
    "ON TRUE": ("SELECT d.id FROM bids_datasets AS d JOIN bids_objects AS o ON TRUE", "tie"),
    "ON 1 = 1": ("SELECT d.id FROM bids_datasets AS d JOIN bids_objects AS o ON 1 = 1", "tie"),
    "ON a constant": ("SELECT d.id FROM bids_datasets AS d JOIN bids_objects AS o ON o.subject = '01'", "tie"),
    "ON link OR TRUE": ("SELECT d.id FROM bids_datasets AS d JOIN bids_objects AS o ON o.dataset_id = d.id OR TRUE",
                        "tie"),
    "ON same table only": ("SELECT d.id FROM bids_datasets AS d JOIN bids_objects AS o ON o.dataset_id = o.dataset_id",
                           "tie"),
    "cross join in a subquery": ("SELECT d.id FROM bids_datasets AS d WHERE d.id IN (SELECT o.dataset_id FROM "
                                 "bids_objects AS o CROSS JOIN bids_participants AS p)", "CROSS JOIN"),
}


@pytest.mark.parametrize("sql, words", list(_UNBOUNDED_JOINS.values()), ids=list(_UNBOUNDED_JOINS))
def test_joins_without_a_real_condition_are_refused(sql: str, words: str) -> None:
    reason = check_static(sql)
    assert reason is not None and words.lower() in reason.lower(), reason


# ── functions that sleep, generate rows or bytes, or touch the server ─────────

@pytest.mark.parametrize("call", [
    "pg_sleep(100)", "PG_SLEEP(1)", "pg_sleep_for('5 minutes')", "pg_sleep_until('2030-01-01')",
    "repeat('x', 1000000000)", "lpad('x', 1000000000, 'y')", "rpad('x', 1000000000)",
    "pg_read_file('/etc/passwd')", "pg_read_binary_file('/etc/passwd')", "pg_ls_dir('/')", "pg_stat_file('/')",
    "dblink('host=evil', 'select 1')", "set_config('statement_timeout', '0', false)",
    "current_setting('data_directory')", "pg_terminate_backend(1)", "pg_cancel_backend(1)", "pg_reload_conf()",
    "pg_advisory_lock(1)", "query_to_xml('select 1', true, true, '')", "nextval('s')", "lo_import('/etc/passwd')",
    "pg_notify('c', 'x')",
    "ts_stat('SELECT search_tsv FROM bids_datasets')", "database_to_xml(true, true, '')",
    "table_to_xml('bids_objects', true, true, '')", "schema_to_xml('public', true, true, '')",
    "query_to_xmlschema('select 1', true, true, '')", "cursor_to_xml('c', 1, true, true, '')",
])
def test_dangerous_functions_are_refused_anywhere(call: str) -> None:
    for sql in (f"SELECT {call}",
                f"SELECT d.id FROM bids_datasets AS d WHERE d.name = {call}",
                f"SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT {call})",
                f"WITH x AS (SELECT {call} AS v) SELECT d.id FROM bids_datasets AS d, LATERAL (SELECT 1) AS l"):
        reason = check_static(sql)
        assert reason is not None and "not allowed" in reason, (sql, reason)


@pytest.mark.parametrize("sql", [
    "SELECT generate_series(1, 1000000000)",
    "SELECT d.id FROM bids_datasets AS d WHERE d.id IN (SELECT generate_series(1, 1000000000))",
    "SELECT d.id FROM bids_datasets AS d, LATERAL generate_series(1, 1000000) AS g",
])
def test_generate_series_is_refused(sql: str) -> None:
    assert "generate_series" in (check_static(sql) or "")


def test_recursive_cte_is_refused() -> None:
    sql = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT n FROM r"
    assert check_static(sql) == "recursive CTE"


# ── statements that write, lock or run more than once ────────────────────────

@pytest.mark.parametrize("sql, words", [
    ("SELECT 1; SELECT 2", "more than one"),
    ("SELECT d.id FROM bids_datasets AS d; DROP TABLE bids_datasets", "more than one"),
    ("DELETE FROM bids_datasets", "SELECT"),
    ("UPDATE bids_datasets SET name = 'x'", "SELECT"),
    ("INSERT INTO bids_datasets (name) VALUES ('x')", "SELECT"),
    ("DROP TABLE bids_datasets", "SELECT"),
    ("CREATE TABLE x AS SELECT * FROM bids_objects", "SELECT"),
    ("TRUNCATE bids_objects", "SELECT"),
    ("SELECT d.* INTO copy_of_datasets FROM bids_datasets AS d", "INTO"),
    ("SELECT d.id FROM bids_datasets AS d FOR UPDATE", "lock"),
    ("SELECT d.id FROM bids_datasets AS d FOR SHARE", "lock"),
    ("WITH gone AS (DELETE FROM bids_datasets RETURNING id) SELECT * FROM gone", "not allowed"),
])
def test_writes_locks_and_batches_are_refused(sql: str, words: str) -> None:
    reason = check_static(sql)
    assert reason is not None and words.lower() in reason.lower(), (sql, reason)


# ── only the dataset catalogue ────────────────────────────────────────────────
# The app's own tables share the database: users (with e-mail addresses),
# query_log and user_queries (other users' questions). A question can steer the
# model into reading them, e.g. one character of an address at a time through
# EXISTS.

def _app_tables() -> list[str]:
    import models  # backend/models.py: the app's own tables, not the catalogue
    return sorted({cls.__tablename__ for cls in vars(models).values()
                   if isinstance(cls, type) and cls.__module__ == "models" and hasattr(cls, "__tablename__")})


def test_the_app_tables_are_known() -> None:
    assert {"users", "query_log", "user_queries", "message_usage"} <= set(_app_tables())


@pytest.mark.parametrize("table", _app_tables())
def test_no_app_table_is_readable(table: str) -> None:
    sql = (f"SELECT d.id FROM bids_datasets AS d "
           f"WHERE EXISTS (SELECT 1 FROM {table} AS t WHERE t.github_login LIKE 'a%')")
    reason = check_static(sql)
    assert reason is not None and f"table {table} is not part" in reason, reason


@pytest.mark.parametrize("sql, table", [
    ("SELECT d.id FROM bids_datasets AS d WHERE d.name IN (SELECT github_email FROM users)", "users"),
    ("SELECT d.id FROM bids_datasets AS d JOIN users AS u ON u.github_login = d.name", "users"),
    ("SELECT d.id FROM bids_datasets AS d WHERE d.id = (SELECT count(*) FROM public.users)", "users"),
    ('SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM "users" AS u)', "users"),
    ("WITH u AS (SELECT github_email FROM users) "
     "SELECT d.id FROM bids_datasets AS d WHERE d.name IN (SELECT github_email FROM u)", "users"),
    ("SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM pg_authid AS a)", "pg_authid"),
    ("SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM pg_catalog.pg_authid AS a)",
     "pg_catalog.pg_authid"),
    ("SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM information_schema.tables AS t)",
     "information_schema.tables"),
    ("SELECT d.id FROM other.public.bids_datasets AS d", "other.public.bids_datasets"),
    # A CTE named like the table counts only where it is visible: this one is
    # defined in the first subquery, so the second reads the real users table.
    ("SELECT d.id FROM bids_datasets AS d "
     "WHERE d.id IN (WITH users AS (SELECT o.dataset_id FROM bids_objects AS o) SELECT dataset_id FROM users) "
     "AND EXISTS (SELECT 1 FROM users AS u WHERE u.github_email LIKE 'a%')", "users"),
])
def test_tables_outside_the_catalogue_are_refused(sql: str, table: str) -> None:
    reason = check_static(sql)
    assert reason is not None and f"table {table} is not part" in reason, (sql, reason)


@pytest.mark.parametrize("sql", [
    "SELECT d.id FROM public.bids_datasets AS d",
    "SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM bids_derivatives AS dv "
    "WHERE dv.dataset_id = d.id AND dv.pipeline = 'fmriprep')",
    "SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM bids_dataset_sites AS s "
    "JOIN bids_site_places AS p ON p.site_id = s.id WHERE s.dataset_id = d.id)",
    "SELECT d.id FROM bids_datasets AS d WHERE EXISTS (SELECT 1 FROM object_associations AS a "
    "JOIN bids_objects AS o ON o.id = a.object_id WHERE o.dataset_id = d.id)",
    # The query's own CTE, even one named like an app table, is not that table.
    "WITH users AS (SELECT o.dataset_id FROM bids_objects AS o) "
    "SELECT d.id FROM bids_datasets AS d WHERE d.id IN (SELECT dataset_id FROM users)",
    "SELECT d.id FROM bids_datasets AS d, LATERAL jsonb_array_elements(d.authors_json) AS e(v)",
    "SELECT r.a FROM jsonb_to_recordset('[]') AS r(a int)",
])
def test_catalogue_tables_and_own_ctes_pass(sql: str) -> None:
    assert check_static(sql) is None, sql


# ── the rewriter and the guard together ───────────────────────────────────────

@pytest.mark.parametrize("sql", [
    _PARTICIPANT_JOINS["left join, the original"],
    _PARTICIPANT_JOINS["inner join"],
    _HEAD + "LEFT JOIN bids_participants AS p ON p.dataset_id = d.id AND p.sex = 'female' "
    "WHERE p.age > 65 GROUP BY d.id",
    _HEAD + "LEFT JOIN bids_participants p ON p.dataset_id = d.id GROUP BY d.id ORDER BY AVG(p.age) DESC",
])
def test_rewriter_repairs_the_participant_join(sql: str) -> None:
    raw = sql.replace("SELECT d.id", "SELECT {{COLS}}", 1)
    rewritten = REWRITER.rewrite_generated_sql(raw, "question").sql
    assert "JOIN bids_participants" not in rewritten.upper().replace("JOIN BIDS_PARTICIPANTS", "JOIN bids_participants")
    assert check_static(rewritten) is None


@pytest.mark.parametrize("sql", list(_PARTICIPANT_JOINS.values()) + list(_SELF_JOINS.values())
                         + [s for s, _ in _UNBOUNDED_JOINS.values()], ids=lambda s: s[:60])
def test_no_dangerous_join_survives_rewrite_and_guard(sql: str) -> None:
    """Whatever the language model writes: after the rewriter, the query is
    either repaired or refused, never run with the join."""
    raw = sql.replace("SELECT d.id", "SELECT {{COLS}}", 1)
    rewritten = REWRITER.rewrite_generated_sql(raw, "question").sql
    if check_static(rewritten) is None:
        assert check_static(sql) is None or "JOIN bids_participants" not in rewritten, rewritten


def test_rewrite_keeps_the_participant_conditions() -> None:
    rewritten = REWRITER._participants_join_to_subqueries(_PARTICIPANT_JOINS["left join, the original"])
    assert rewritten.count("p.age <= 12") == 2  # in the EXISTS and in the HAVING count
    assert "COUNT(DISTINCT p.participant_id) FROM bids_participants AS p" in rewritten


def test_rewrite_turns_participant_aggregates_in_order_by_into_subqueries() -> None:
    sql = _HEAD + "LEFT JOIN bids_participants p ON p.dataset_id = d.id GROUP BY d.id ORDER BY AVG(p.age) DESC"
    rewritten = REWRITER._participants_join_to_subqueries(sql)
    assert "ORDER BY (SELECT AVG(p.age) FROM bids_participants AS p WHERE p.dataset_id = d.id) DESC" in rewritten


def test_rewrite_leaves_participant_columns_outside_aggregates_to_the_guard() -> None:
    sql = _HEAD + "LEFT JOIN bids_participants p ON p.dataset_id = d.id GROUP BY d.id, p.sex ORDER BY p.sex"
    assert REWRITER._participants_join_to_subqueries(sql) == sql
    assert check_static(sql) is not None


# ── 2. plan check ─────────────────────────────────────────────────────────────

def test_largest_step_walks_the_plan() -> None:
    plan = [{"Plan": {"Node Type": "Aggregate", "Plan Rows": 1, "Plans": [
        {"Node Type": "Hash Join", "Plan Rows": 12_300_000, "Plans": [
            {"Node Type": "Seq Scan", "Plan Rows": 2_700_000}]}]}}]
    assert largest_step(json.dumps(plan)) == (12_300_000.0, "Hash Join")
    assert largest_step(plan) == (12_300_000.0, "Hash Join")


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value


class _Nested:
    def __init__(self, fail: bool):
        self.fail = fail

    async def __aenter__(self):
        if self.fail:
            raise RuntimeError("permission denied to set parameter temp_file_limit")
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    """Records statements; answers EXPLAIN with a plan of *rows* rows."""

    def __init__(self, rows: float = 1000, biggest_table=2_700_000, explain_fails=False, savepoint_fails=False):
        self.rows, self.biggest_table = rows, biggest_table
        self.explain_fails, self.savepoint_fails = explain_fails, savepoint_fails
        self.statements = []

    async def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append(sql)
        if sql.startswith("EXPLAIN"):
            if self.explain_fails:
                raise RuntimeError("syntax error")
            return _Result([{"Plan": {"Node Type": "Hash Join", "Plan Rows": self.rows}}])
        if "reltuples" in sql:
            if isinstance(self.biggest_table, Exception):
                raise self.biggest_table
            return _Result(self.biggest_table)
        return _Result(None)

    def begin_nested(self):
        return _Nested(self.savepoint_fails)


@pytest.fixture(autouse=True)
def _fresh_guard_state(monkeypatch):
    monkeypatch.setattr(query_guard, "_max_plan_rows", None)
    monkeypatch.setattr(query_guard, "_semaphore", None)
    for name in ("SEARCH_MAX_PLAN_ROWS", "SEARCH_STATEMENT_TIMEOUT", "SEARCH_TEMP_FILE_LIMIT", "SEARCH_WORK_MEM",
                 "SEARCH_MAX_CONCURRENT", "SEARCH_SLOT_WAIT_SECONDS"):
        monkeypatch.delenv(name, raising=False)


def _vet(session, sql=_GOOD):
    asyncio.run(query_guard.vet(session, sql, "SELECT COUNT(*) FROM (" + sql + ") c"))


def test_the_estimate_of_the_disk_filling_query_is_refused() -> None:
    # EXPLAIN on production estimated 12.3 million rows for it (2026-11-14);
    # no other benchmark query exceeded one scan of bids_objects, 2.7 million.
    with pytest.raises(QueryRefusedError, match="12,300,000 rows"):
        _vet(_Session(rows=12_300_000))


def test_one_scan_of_the_largest_table_passes() -> None:
    _vet(_Session(rows=2_700_000))


def test_limit_is_three_times_the_largest_table() -> None:
    _vet(_Session(rows=8_000_000, biggest_table=2_700_000))
    with pytest.raises(QueryRefusedError):
        _vet(_Session(rows=8_200_000, biggest_table=2_700_000))


def test_small_database_still_allows_a_million_rows() -> None:
    _vet(_Session(rows=900_000, biggest_table=1000))


def test_configured_plan_limit_wins(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_MAX_PLAN_ROWS", "500")
    with pytest.raises(QueryRefusedError):
        _vet(_Session(rows=501))


def test_unknown_table_size_falls_back_to_ten_million() -> None:
    _vet(_Session(rows=9_000_000, biggest_table=RuntimeError("no pg_class")))
    query_guard._max_plan_rows = None
    with pytest.raises(QueryRefusedError):
        _vet(_Session(rows=11_000_000, biggest_table=RuntimeError("no pg_class")))


def test_failing_explain_leaves_the_error_to_the_real_run() -> None:
    _vet(_Session(explain_fails=True))


def test_static_refusal_skips_the_plan() -> None:
    session = _Session()
    with pytest.raises(QueryRefusedError, match="pg_sleep"):
        _vet(session, "SELECT pg_sleep(1)")
    assert not any(s.startswith("EXPLAIN") for s in session.statements)


def test_explain_never_executes_the_query() -> None:
    session = _Session()
    _vet(session)
    ran = [s for s in session.statements if not s.startswith("EXPLAIN") and "reltuples" not in s]
    assert ran == []


# ── 3. limits per transaction ─────────────────────────────────────────────────

def test_default_limits() -> None:
    session = _Session()
    asyncio.run(query_guard.apply_limits(session))
    assert session.statements == [
        "SET LOCAL statement_timeout = '120s'",
        "SET LOCAL work_mem = '32MB'",
        "SET LOCAL temp_file_limit = '1GB'",
    ]


@pytest.mark.parametrize("value, expected", [
    ("45s", "45s"), ("500ms", "500ms"), ("2min", "2min"), ("30000", "30000"),
    ("0", "0"), ("45s'; DROP TABLE bids_datasets; --", "120s"), ("", "120s"), ("1 hour", "120s"),
])
def test_statement_timeout_setting_is_validated(monkeypatch, value: str, expected: str) -> None:
    monkeypatch.setenv("SEARCH_STATEMENT_TIMEOUT", value)
    session = _Session()
    asyncio.run(query_guard.apply_limits(session))
    assert f"SET LOCAL statement_timeout = '{expected}'" in session.statements


@pytest.mark.parametrize("value, expected", [("500MB", "500MB"), ("2GB", "2GB"), ("-1", "1GB"), ("lots", "1GB")])
def test_temp_file_limit_setting_is_validated(monkeypatch, value: str, expected: str) -> None:
    monkeypatch.setenv("SEARCH_TEMP_FILE_LIMIT", value)
    session = _Session()
    asyncio.run(query_guard.apply_limits(session))
    assert f"SET LOCAL temp_file_limit = '{expected}'" in session.statements


def test_role_without_temp_file_permission_keeps_the_other_limits() -> None:
    session = _Session(savepoint_fails=True)
    asyncio.run(query_guard.apply_limits(session))
    assert "SET LOCAL statement_timeout = '120s'" in session.statements
    assert "SET LOCAL work_mem = '32MB'" in session.statements


# ── 4. concurrency ────────────────────────────────────────────────────────────

def test_slot_reports_busy_instead_of_queueing_forever(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_MAX_CONCURRENT", "1")
    monkeypatch.setenv("SEARCH_SLOT_WAIT_SECONDS", "0.05")

    async def scenario():
        async with query_guard.slot():
            with pytest.raises(SearchBusyError):
                async with query_guard.slot():
                    pass
        async with query_guard.slot():  # free again
            pass

    asyncio.run(scenario())


def test_at_most_the_configured_number_run_at_once(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_MAX_CONCURRENT", "2")
    running, peak = 0, 0

    async def search():
        nonlocal running, peak
        async with query_guard.slot():
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

    async def scenario():
        await asyncio.gather(*(search() for _ in range(10)))

    asyncio.run(scenario())
    assert peak == 2


def test_a_failing_search_releases_its_slot(monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_MAX_CONCURRENT", "1")
    monkeypatch.setenv("SEARCH_SLOT_WAIT_SECONDS", "0.05")

    async def scenario():
        with pytest.raises(RuntimeError):
            async with query_guard.slot():
                raise RuntimeError("statement timeout")
        async with query_guard.slot():
            pass

    asyncio.run(scenario())


# ── slow ranking falls back to the unranked page ──────────────────────────────

class QueryCanceledError(Exception):
    pass


class _WrappedError(Exception):
    def __init__(self, orig):
        super().__init__("(sqlalchemy.dialects.postgresql.asyncpg.Error)")
        self.orig = orig


def test_timeouts_are_recognised_also_when_wrapped() -> None:
    assert query_guard.is_timeout(QueryCanceledError("canceling statement due to statement timeout"))
    assert query_guard.is_timeout(_WrappedError(QueryCanceledError("x")))
    assert query_guard.is_timeout(RuntimeError("canceling statement due to statement timeout"))
    assert not query_guard.is_timeout(RuntimeError("syntax error at or near FROM"))
    assert not query_guard.is_timeout(_WrappedError(RuntimeError("relation does not exist")))


def test_ranked_page_timeout_setting(monkeypatch) -> None:
    assert query_guard.ranked_page_timeout() == "25s"
    monkeypatch.setenv("SEARCH_RANKED_PAGE_TIMEOUT", "10s")
    assert query_guard.ranked_page_timeout() == "10s"
    monkeypatch.setenv("SEARCH_RANKED_PAGE_TIMEOUT", "10s'; SELECT 1")
    assert query_guard.ranked_page_timeout() == "25s"
