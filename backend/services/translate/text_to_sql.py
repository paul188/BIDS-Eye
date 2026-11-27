"""
services/translate/text_to_sql.py
---------------------------------
Text-to-SQL service for BIDS-Eye.

Pipeline:
  1. LLM preprocessing     → structured QueryPlan (intent extraction)
  2. RAG resolution        → canonical DB codes (diagnoses, tasks, scan types)
 3. SQL construction      → deterministic compiler for supported typed plans,
                             otherwise LLM SQL generation with examples
                             (prompt text, schema and examples: prompt/)
  4. SQL rewrite service    → projection normalization, filter cleanup,
                             VOCABULARY MISS fallback predicates, and expansion
                             of text_match() into full-text search
  5. LLM error corrector   → fixes any SQL that fails at runtime

The models: profiles intent and sql in backend/llm/llm_config.yaml. When
neither can be asked (no key for any of its models), the service returns a
bounded sample of datasets.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional

log = logging.getLogger(__name__)

from routers.schemas import TextToSQLResult  # noqa: E402
from llm_client import llm_generate, profile_available, LLMAllFailedError  # noqa: E402
from services.translate.sql_rewriter import FALLBACK_SQL as _FALLBACK_SQL  # noqa: E402
from services.translate.sql_rewriter import extract_sql as _extract_sql  # noqa: E402
from services.translate.sql_rewriter import finish_corrected_sql as _finish_corrected_sql  # noqa: E402
from services.translate.sql_rewriter import rewrite_generated_sql as _rewrite_generated_sql  # noqa: E402
from services.database_stats.database_stats import database_stats as _database_stats  # noqa: E402
from services.translate import optional_columns as _optional_columns  # noqa: E402
from services.translate.prompt import examples as _examples  # noqa: E402
from services.translate.prompt import instructions as _instructions  # noqa: E402
from services.translate.prompt import schema_ddl as _schema_ddl  # noqa: E402
from services.schema.tables import datasets as _datasets, objects as _objects  # noqa: E402
from services.translate.participant_query_compiler import (  # noqa: E402
    compile_plan as _compile_plan,
)


def _correct_sql(sql: str, db_error: str, question: str) -> str:
    """
    Stage 4 — ask the model to fix a SQL query that failed with a PostgreSQL error.
    Returns the corrected SQL, or the original if correction fails.

    *sql* is the rewritten query that ran (uncertain tier included); the
    answer is not rewritten again (sql_rewriter.finish_corrected_sql).
    """
    schema_ddl = _schema_ddl.build_schema_ddl(_optional_columns.usable_columns(_database_stats.coverage))
    prompt = _instructions.correction_prompt(question, schema_ddl, sql, db_error)

    try:
        raw = llm_generate(prompt, profile="sql", purpose="sql_correction")
        corrected = _finish_corrected_sql(raw, sql)
        return corrected if corrected != _FALLBACK_SQL else sql
    except LLMAllFailedError as exc:
        log.warning("All LLM tiers failed during SQL correction: %s", exc)
        return sql


# ── Stage 3: SQL generation ───────────────────────────────────────────────────

def _generate_sql(augmented_question: str, examples: Optional[List[dict]] = None) -> str:
    """SQL generation by the model (profile sql), with few-shot examples."""
    schema_ddl = _schema_ddl.build_schema_ddl(_optional_columns.usable_columns(_database_stats.coverage))

    # The model never writes a text predicate itself. It copies or writes
    # text_match('...') placeholders, which services/fulltext/text_search.py expands
    # into full-text search.
    long_text_cols = ["d.description_text"]
    if _optional_columns.usable(_database_stats.coverage, "readme_text"):
        long_text_cols.append("d.readme_text")
    long_text_phrase = " or ".join(long_text_cols)

    system_prompt = _instructions.system_prompt(long_text_phrase)
    prompt = _instructions.user_prompt(schema_ddl, augmented_question, examples or [], system_prompt)

    # Cascade + retries + cross-provider fallback handled by llm_client; raises
    # LLMAllFailedError if every tier fails (caught upstream for graceful degrade).
    return _extract_sql(llm_generate(
        prompt, profile="sql", system=system_prompt))


# ── Pipeline (stages 1–2 + SQL generation as stage 3) ─────────────────────────

def _run_local(question: str) -> TextToSQLResult:
    """
    Translation pipeline:
      Stage 1: LLM preprocessing     → QueryPlan
      Stage 2: RAG resolution        → augmented question with DB codes
      Stage 3: LLM SQL generation    → SELECT statement (+ few-shot examples)
    """
    try:
        from preprocess import run_pipeline
        from retriever import NAME_INDEX_PATH, MetadataRetriever
    except ImportError as exc:
        log.warning("Pipeline modules not importable (%s) — using fallback SQL", exc)
        return TextToSQLResult(sql=_FALLBACK_SQL, explanation=str(exc))

    # Stage 2 retriever (author / funding name resolution + optional DB back-check)
    name_index_path = NAME_INDEX_PATH
    retriever: Optional[MetadataRetriever] = None
    if name_index_path.exists():
        try:
            retriever = MetadataRetriever(
                str(name_index_path),
                db_url=os.getenv("DATABASE_URL"),
            )
        except (OSError, ValueError) as exc:  # unreadable, or not JSON
            log.warning("Could not load RAG name index: %s", exc)

    # Stages 1 + 2
    augmented_plan = run_pipeline(
        question, retriever=retriever,
        available_columns=_optional_columns.usable_columns(_database_stats.coverage),
    )
    augmented_question = augmented_plan.augmented_question
    query_family = augmented_plan.plan.query_family.value
    explanation = augmented_plan.plan.natural_language_summary
    # A phrase too few datasets say is left out of the search
    # (preprocess/augment.py): "epilepsy ONLY postictal" listed every epilepsy
    # dataset and said nothing about "postictal" (live, 2026-11-23).
    unmatched_text = list(getattr(augmented_plan, "unmatched_text", None) or [])
    if unmatched_text:
        named = ", ".join(f'"{phrase}"' for phrase in unmatched_text)
        explanation = ((explanation or "").rstrip() + f" Fewer than three datasets mention {named},"
                       f" so the search ran without {'it' if len(unmatched_text) == 1 else 'them'}.").strip()

    # Stage 3: compile cohort plans (diagnosis groups, optionally with scan
    # requirements) deterministically. The LLM has already interpreted the
    # query and RAG resolved each group; asking another model to reconstruct
    # their AND/OR semantics as SQL adds variance without adding information.
    # Unsupported plans still use the established SQL-generation path below.
    resolved_codes = _examples.extract_resolved_codes(augmented_question)
    plan = augmented_plan.plan
    compiled = _compile_plan(plan, getattr(augmented_plan, "resolved", None) or [],
                             augmented_question)
    raw_sql = compiled.sql
    if raw_sql:
        log.info("SQL path: compiled (%d participant groups, no SQL generation)",
                 len(plan.groups))
    else:
        # One line per search: which plans the compiler declines, and why,
        # shows what to support next.
        log.info("SQL path: model (compiler declined: %s)", compiled.reason)
        examples = _examples.pick_examples(_examples.FEW_SHOT_POOL, query_family, n=5, seed=question,
                                  resolved_codes=resolved_codes)
        if examples:
            log.info("Injecting %d few-shot examples for family '%s' (resolved codes: %s)",
                     len(examples), query_family,
                     {k: v for k, v in resolved_codes.items() if v})
        raw_sql = _generate_sql(augmented_question, examples)

    # Keep LLM generation and SQL rewriting separate: the model emits raw SQL,
    # or the supported typed-plan compiler emits it; the rewrite service then
    # handles projection normalization, scan-filter cleanup, and text search.
    rewritten = _rewrite_generated_sql(
        raw_sql,
        augmented_question,
        # Word forms decided for this question (backend/vocabulary/word_forms.py); {} mostly.
        getattr(augmented_plan, "word_forms", None) or None,
    )

    # Relevance ranking is applied later by services.translate.sql_rewriter.build_page_sql
    # so the cached base SELECT can be reused across pages without rebaking the
    # order clause. We still decide *whether* to rank: skip for explicit top-N
    # / user-ordered queries so we don't override the order the user asked for.
    plan = augmented_plan.plan
    apply_relevance = plan.result_limit is None and plan.order_by is None
    # The whole catalogue is no answer to a search: "bonn" and "datasets
    # related to drawings or art" got all 1,721 datasets, in no useful order
    # (live, 2026-11-17/18). Only a ranking or top-N question ("the 10
    # largest datasets") may leave every dataset in.
    if apply_relevance and keeps_every_dataset(rewritten.sql):
        log.warning("Translation of %r has no condition; keyword search instead", question)
        return keyword_fallback(
            question, ((explanation or "").rstrip()
                       + " No search condition could be taken from the question.").strip())
    scored_filters = [
        {"field": f.field, "code": f.code, "score": f.score}
        for f in augmented_plan.scored_filters
    ]

    return TextToSQLResult(
        sql=rewritten.sql,
        explanation=explanation,
        scored_filters=scored_filters,
        apply_relevance=apply_relevance,
        params=rewritten.params,
        # Without this the router ranks by code matches only: the text rank
        # (ts_rank_cd over search_tsv) is silently left out.
        rank_query=rewritten.rank_query,
        filter_sql=rewritten.filter_sql,
        relaxed_sql=rewritten.relaxed_sql,
        value_fixes=rewritten.value_fixes,
        sql_path="compiled" if compiled.sql else f"model: {compiled.reason}",
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def keeps_every_dataset(sql: str) -> bool:
    """True if *sql* returns the whole catalogue.

    That is a single SELECT without WHERE, HAVING or an inner join at its
    outer level. SQL that does not parse counts as filtered; the query guard
    deals with it.
    """
    import sqlglot
    import sqlglot.errors
    from sqlglot import exp

    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.SqlglotError:
        return False
    if not isinstance(tree, exp.Select):
        return False
    if tree.args.get("where") is not None or tree.args.get("having") is not None:
        return False
    return all((join.side or "").upper() in ("LEFT", "FULL")
               for join in tree.args.get("joins") or [])


def keyword_fallback(question: str, explanation: str) -> TextToSQLResult:
    """Plain keyword search when no language model answered.

    Matches any content word of the question in the dataset text and ranks by
    full-text rank, so the best matches come first. Without usable words,
    the bounded sample as before.
    """
    from services.fulltext import tsquery
    from services.fulltext.ranking import RankQuery
    from services.fulltext.text_search import stemmed_match
    from services.translate.sql_rewriter import DEFAULT_DATASET_PROJECTION

    query = tsquery.keyword_query(question)
    if not query:
        return TextToSQLResult(sql=_FALLBACK_SQL, explanation=explanation + " Showing a sample of datasets.")
    sql = (
        f"SELECT {DEFAULT_DATASET_PROJECTION}\n"
        f"FROM {_datasets.name} d\n"
        f"LEFT JOIN {_objects.name} o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
        f"WHERE {stemmed_match('d', query)}\n"
        "GROUP BY d.id"
    )
    return TextToSQLResult(
        sql=sql, explanation=explanation + " Showing a plain keyword search for your question instead.",
        apply_relevance=True, rank_query=RankQuery(stemmed=query), filter_sql=sql, sql_path="keyword",
    )


async def text_to_sql(question: str) -> TextToSQLResult:
    """
    Translate a natural-language question into a PostgreSQL SELECT statement via
    the LLM pipeline. Falls back to a bounded sample of datasets when no model
    of profiles intent and sql can be asked, and to a keyword search when every
    one fails.

    Always runs in a thread pool to avoid blocking the FastAPI event loop.
    """
    if profile_available("intent") and profile_available("sql"):
        log.info("Using the LLM pipeline")
        # Hard wall-clock deadline for the whole LLM pipeline (intent + RAG + SQL
        # generation). Cloudflare drops the request at ~100s; if the LLMs are
        # pathologically slow (a provider's 503 storm + slow fallback) we must return
        # *something* fast rather than let the browser see a NetworkError.
        deadline = float(os.getenv("PIPELINE_DEADLINE_SECONDS", "80"))
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run_local, question),
                timeout=deadline,
            )
        except LLMAllFailedError as exc:
            # Every model of the profile is down.
            # Degrade gracefully instead of 500ing the request.
            log.warning("All LLM providers failed (%s) — falling back to a keyword search", exc)
            return keyword_fallback(
                question, "The language model is unavailable right now; please try again shortly.")
        except asyncio.TimeoutError:
            log.warning("LLM pipeline exceeded %.0fs deadline — falling back to a keyword search", deadline)
            return keyword_fallback(
                question, "Understanding that question took too long right now.")

    log.warning("No LLM configured for profiles intent and sql — returning a sample of datasets")
    return TextToSQLResult(
        sql=_FALLBACK_SQL,
        explanation="No API credentials configured — showing a sample of datasets.",
    )
