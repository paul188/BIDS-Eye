"""
routers/query.py
----------------
Natural-language dataset query endpoint.

Accepts a plain-text question, passes it through the Text-To-SQL layer, executes
the resulting SQL against the BIDS database, and returns matching datasets.

The expensive translation (Gemini intent + RAG + SQL generation) runs once per
question; the resulting base SELECT is cached under an opaque ``query_id`` so
subsequent pages reuse it. Relevance ordering/weighting and LIMIT/OFFSET are
applied per page in postprocessing (see services.sql_rewriter.build_page_sql).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Optional, Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from deps import get_db
from routers.auth import require_auth
from schemas import (
    DatasetSchema,
    LinkedPaperSchema,
    ParticipantSchema,
    ParticipantSummarySchema,
    QueryFilterSchema,
    QueryPageRequest,
    QueryRefineRequest,
    QueryRequest,
    QueryResponse,
    TextToSQLResult,
)
from services import dataset_description, query_filters
from services import uncertain_tier
from services.query_cache import CachedQuery, query_cache
from services.schema_capabilities import has_display_column
from services.rerank import rerank_candidates, settings as rerank_settings, should_rerank
from services.sql_rewriter import FALLBACK_SQL as _FALLBACK_SQL, build_count_sql, build_page_sql
from services.text_to_sql import _correct_sql_with_gemini, text_to_sql

router = APIRouter(prefix="/query", tags=["query"])
log = logging.getLogger(__name__)

_DAILY_LIMIT = int(os.getenv("DAILY_MESSAGE_LIMIT", "20"))
_ADMIN_LOGINS = {
    u.strip() for u in os.getenv("ADMIN_GITHUB_LOGINS", "paul188").split(",") if u.strip()
}


async def _enforce_rate_limit(session: AsyncSession, github_login: str) -> None:
    """Atomically increment today's message count and raise 429 if over the limit."""
    if github_login in _ADMIN_LOGINS:
        return
    result = await session.execute(
        text("""
            INSERT INTO message_usage (github_login, usage_date, count)
            VALUES (:login, :today, 1)
            ON CONFLICT (github_login, usage_date)
            DO UPDATE SET count = message_usage.count + 1
            RETURNING count
        """),
        {"login": github_login, "today": date.today()},
    )
    new_count = result.scalar()
    await session.commit()

    if new_count > _DAILY_LIMIT:
        now = datetime.now(timezone.utc)
        retry_after = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        raise HTTPException(
            status_code=429,
            detail={
                "code": "rate_limit_exceeded",
                "retry_after": retry_after.isoformat(),
                "limit": _DAILY_LIMIT,
            },
        )

# Capped here as well as in the cache so a page can't request an unbounded slice.
_MAX_PAGE_SIZE = 100
_DEFAULT_PAGE_SIZE = 20
_MAX_PARTICIPANTS_PER_DATASET = 50


async def _participants_for(
    session: AsyncSession, dataset_ids: list[UUID]
) -> dict[UUID, list[ParticipantSchema]]:
    """Fetch participants grouped by dataset_id."""
    if not dataset_ids:
        return {}

    from db.models import BIDSParticipant
    from sqlalchemy import select

    rows = await session.execute(
        select(BIDSParticipant).where(BIDSParticipant.dataset_id.in_(dataset_ids))
    )
    result: dict[UUID, list[ParticipantSchema]] = {}
    for p in rows.scalars():
        result.setdefault(p.dataset_id, []).append(ParticipantSchema.model_validate(p))
    return result


def _found_message(total: int) -> str:
    return f"Found {total} dataset{'' if total == 1 else 's'} matching your query."


def summarize_participants(participants: Sequence[ParticipantSchema]) -> Optional[ParticipantSummarySchema]:
    """Group, sex and age counts over every participant, so a card can show
    exact numbers for datasets larger than the participant cap."""
    if not participants:
        return None
    diagnoses: dict[str, int] = {}
    female = male = 0
    ages: list[float] = []
    for p in participants:
        label = (p.diagnosis or "").strip()
        if label:
            diagnoses[label] = diagnoses.get(label, 0) + 1
        sex = (p.sex or "").strip().lower()
        if sex in ("f", "female"):
            female += 1
        elif sex in ("m", "male"):
            male += 1
        if p.age is not None and p.age > 0:
            ages.append(p.age)
    return ParticipantSummarySchema(
        total=len(participants), diagnoses=diagnoses, female=female, male=male,
        age_min=min(ages) if ages else None, age_max=max(ages) if ages else None,
    )


async def _deprecations_for(
    session: AsyncSession, dataset_ids: list[UUID]
) -> dict[UUID, str]:
    """Return {dataset_id: reason} for datasets marked deprecated.

    Read with its own query instead of through the generated SQL, so the
    projection the LLM pipeline relies on stays unchanged, and databases that
    do not have the column yet keep working.
    """
    if not dataset_ids or not has_display_column("deprecated_reason"):
        return {}
    rows = await session.execute(
        text(
            "SELECT id, deprecated_reason FROM bids_datasets "
            "WHERE id = ANY(:ids) AND deprecated_reason IS NOT NULL"
        ),
        {"ids": dataset_ids},
    )
    return {row[0]: row[1] for row in rows}


async def _count_with_base_correction(
    session: AsyncSession, query_id: str, cached: CachedQuery, question: str
) -> int:
    """Count total matching datasets, self-correcting the cached *base* SQL once
    via Gemini if it raises a DB error.

    Running the COUNT validates the base SQL before any page is fetched. A fix is
    persisted back to the cache so later page turns reuse the corrected base.
    """
    try:
        result = await session.execute(
            text(build_count_sql(cached.base_sql)), cached.params
        )
        return int(result.scalar() or 0)
    except Exception as exc:
        # Roll back the aborted transaction before any retry — without this,
        # every subsequent execute on the same session fails with
        # InFailedSQLTransactionError.
        await session.rollback()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        if not api_key:
            raise
        corrected = await asyncio.to_thread(
            _correct_sql_with_gemini, cached.base_sql, str(exc), question, api_key
        )
        if corrected == cached.base_sql:
            cached.base_sql = _FALLBACK_SQL
            cached.params = {}
            cached.self_corrected = True
            query_cache.update(query_id, cached)
            result = await session.execute(text(build_count_sql(_FALLBACK_SQL)))
            return int(result.scalar() or 0)
        cached.base_sql = corrected
        cached.self_corrected = True
        query_cache.update(query_id, cached)
        try:
            result = await session.execute(
                text(build_count_sql(cached.base_sql)), cached.params
            )
            return int(result.scalar() or 0)
        except Exception:
            await session.rollback()
            cached.base_sql = _FALLBACK_SQL
            cached.params = {}
            query_cache.update(query_id, cached)
            result = await session.execute(text(build_count_sql(_FALLBACK_SQL)))
            return int(result.scalar() or 0)


async def _rerank_once(
    session: AsyncSession, query_id: str, cached: CachedQuery, question: str, total: int
) -> None:
    """Let an LLM judge the first candidates and store the new order (once per query)."""
    cached.rerank_done = True
    if not question or not should_rerank(
        cached.apply_relevance, total, cached.base_sql == _FALLBACK_SQL
    ):
        query_cache.update(query_id, cached)
        return
    try:
        candidate_sql, params = build_page_sql(
            cached.base_sql, cached.scored_filters, True,
            rerank_settings()["candidates"], 0, rank_tsquery=cached.rank_tsquery,
        )
        rows = await session.execute(text(candidate_sql), {**cached.params, **params})
        candidate_ids = [row["id"] for row in rows.mappings().all()]
    except Exception:
        await session.rollback()
        query_cache.update(query_id, cached)
        return
    result = await rerank_candidates(session, candidate_ids, question, cached.explanation)
    if result is not None:
        cached.rerank_ordered = result.ordered_ids
        cached.rerank_demoted = result.demoted_ids
    query_cache.update(query_id, cached)


async def _rows_to_datasets(session: AsyncSession, raw: Sequence) -> list[DatasetSchema]:
    """Build DatasetSchema objects (with capped participants) from result rows."""
    dataset_ids = [UUID(str(r["id"])) for r in raw]
    participants_map = await _participants_for(session, dataset_ids)
    deprecations = await _deprecations_for(session, dataset_ids)
    return [
        DatasetSchema(
            id=r["id"],
            name=r["name"],
            accession_id=r.get("accession_id"),
            bids_version=r.get("bids_version"),
            dataset_type=r["dataset_type"],
            source_type=r["source_type"],
            remote_url=r.get("remote_url"),
            validation_status=r.get("validation_status"),
            authors=r.get("authors"),
            institutions=r.get("institutions"),
            description_text=r.get("description_text"),
            deprecated_reason=deprecations.get(UUID(str(r["id"]))),
            uncertain_level=r.get("uncertain_level"),
            uncertain_reasons=r.get("uncertain_reasons"),
            subject_count=r.get("subject_count"),
            participants=participants_map.get(UUID(str(r["id"])), [])[
                :_MAX_PARTICIPANTS_PER_DATASET
            ],
            participant_summary=summarize_participants(participants_map.get(UUID(str(r["id"])), [])),
        )
        for r in raw
    ]


# README excerpts (_attach_snippets). Matched words are wrapped in these
# marks, which READMEs do not use, so the page can bold them without HTML.
SNIPPET_START, SNIPPET_STOP = "⟦", "⟧"
_SNIPPET_README_CHARS = 20000   # bounds ts_headline's work on huge READMEs
_SNIPPET_PLAIN_CHARS = 320      # excerpt length without search words
_HEADLINE_OPTIONS = (
    f"StartSel={SNIPPET_START}, StopSel={SNIPPET_STOP}, MaxFragments=2, "
    "MaxWords=28, MinWords=12, ShortWord=2, FragmentDelimiter=\" … \""
)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+")
_MD_SYNTAX_RE = re.compile(r"[#*`>|]+|(?<!\w)_{1,2}|_{1,2}(?!\w)|^-{3,}|={3,}", re.M)


def clean_snippet(text_: str) -> str:
    """README Markdown reduced to plain text: no images, link targets, URLs,
    heading marks, emphasis or table pipes; whitespace collapsed."""
    text_ = _MD_IMAGE_RE.sub(" ", text_)
    text_ = _MD_LINK_RE.sub(r"\1", text_)
    text_ = _URL_RE.sub(" ", text_)
    text_ = _MD_SYNTAX_RE.sub(" ", text_)
    return re.sub(r"\s+", " ", text_).strip()


def _cut_at_word(text_: str, limit: int) -> str:
    if len(text_) <= limit:
        return text_
    cut = text_[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + " …"


async def _attach_snippets(
    session: AsyncSession, datasets: list[DatasetSchema], highlight: Optional[str]
) -> None:
    """Give each dataset a README excerpt: the passages matching *highlight*
    (an OR tsquery, services/query_filters.highlight_tsquery), or the start.

    Read with its own query, like _deprecations_for, so the generated SQL stays
    unchanged and databases without readme_text keep working. A failure leaves
    the page without excerpts rather than failing it.
    """
    if not datasets or not has_display_column("readme_text"):
        return
    ids = [d.id for d in datasets]
    if highlight:
        sql = (
            "SELECT id, ts_headline('english', left(readme_text, :chars), "
            "to_tsquery('english', :q), :opts) FROM bids_datasets "
            "WHERE id = ANY(:ids) AND readme_text IS NOT NULL AND readme_text <> ''"
        )
        params = {"ids": ids, "q": highlight, "opts": _HEADLINE_OPTIONS,
                  "chars": _SNIPPET_README_CHARS}
    else:
        sql = (
            "SELECT id, left(readme_text, :chars) FROM bids_datasets "
            "WHERE id = ANY(:ids) AND readme_text IS NOT NULL AND readme_text <> ''"
        )
        params = {"ids": ids, "chars": _SNIPPET_PLAIN_CHARS * 4}
    try:
        async with session.begin_nested():
            rows = await session.execute(text(sql), params)
            found = {row[0]: row[1] for row in rows}
    except Exception:
        log.warning("README excerpts unavailable for this page", exc_info=True)
        return
    for dataset in datasets:
        raw = found.get(dataset.id)
        if not raw:
            continue
        snippet = clean_snippet(raw)
        if not highlight or SNIPPET_START not in snippet:
            snippet = _cut_at_word(snippet.replace(SNIPPET_START, "").replace(SNIPPET_STOP, ""),
                                   _SNIPPET_PLAIN_CHARS)
        else:
            # A matched passage usually starts and ends mid-sentence.
            if snippet.lstrip(SNIPPET_START)[:1].islower():
                snippet = "… " + snippet
            if snippet[-1:] not in ".!?…":
                snippet += " …"
        dataset.snippet = snippet or None


def _paper_url(doi, pmid, pmcid, arxiv_id) -> Optional[str]:
    if doi:
        return f"https://doi.org/{doi}"
    if pmid:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if pmcid:
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return None


async def _attach_descriptions(session: AsyncSession, datasets: list[DatasetSchema]) -> None:
    """The generated summary of a thin README, and the linked paper when it was
    checked to report the dataset (services/dataset_description.py). Its own
    query, like _attach_snippets; a failure leaves the page without them."""
    if not datasets or not (has_display_column("paper_relation")
                            and has_display_column("dataset_summary")):
        return
    sql = (
        "SELECT d.id, d.dataset_summary, d.paper_relation->>'relation', "
        "       p.title, p.year, p.doi, p.pmid, p.pmcid, p.arxiv_id "
        "FROM bids_datasets d "
        "LEFT JOIN bids_papers p ON p.paper_key = d.paper_relation->>'paper_key' "
        "     AND (d.paper_relation->>'shown')::boolean "
        "WHERE d.id = ANY(:ids) "
        "  AND (d.dataset_summary IS NOT NULL OR (d.paper_relation->>'shown')::boolean)"
    )
    try:
        async with session.begin_nested():
            rows = (await session.execute(text(sql), {"ids": [d.id for d in datasets]})).all()
    except Exception:
        log.warning("Dataset descriptions unavailable for this page", exc_info=True)
        return
    found = {row[0]: row for row in rows}
    for dataset in datasets:
        row = found.get(dataset.id)
        if row is None:
            continue
        dataset.generated_summary = row[1]
        relation, title = row[2], row[3]
        if title and relation in dataset_description.SHOWN_RELATIONS:
            dataset.linked_paper = LinkedPaperSchema(
                title=title, year=row[4], url=_paper_url(row[5], row[6], row[7], row[8]),
                relation=relation, relation_label=dataset_description.RELATION_LABELS[relation])


def _highlight_for(cached: CachedQuery) -> Optional[str]:
    if cached.self_corrected:
        return None
    return query_filters.highlight_tsquery(cached.filter_sql)


def _filters_for(cached: CachedQuery) -> list[QueryFilterSchema]:
    """The query's conditions as chips; none once the SQL was self-corrected,
    since the corrected query no longer matches the conditions read earlier."""
    if cached.self_corrected or not cached.filter_sql:
        return []
    return [QueryFilterSchema(**f.to_dict()) for f in query_filters.describe(cached.filter_sql)]


def _clamp_page(page: int, page_size: int) -> tuple[int, int]:
    page = max(1, page)
    page_size = max(1, min(page_size or _DEFAULT_PAGE_SIZE, _MAX_PAGE_SIZE))
    return page, page_size


async def _run_page(
    session: AsyncSession,
    query_id: str,
    cached: CachedQuery,
    question: str,
    page: int,
    page_size: int,
) -> tuple[list[DatasetSchema], int]:
    """Fetch one page for a cached query: total count + the page's datasets.

    The COUNT runs first (validating / self-correcting the base SQL); the page
    query then reuses the now-validated base, so it only applies ordering +
    LIMIT/OFFSET.
    """
    if cached.total is None:
        total = await _count_with_base_correction(session, query_id, cached, question)
        cached.total = total
        query_cache.update(query_id, cached)
    else:
        total = cached.total

    if not cached.rerank_done:
        await _rerank_once(session, query_id, cached, question, total)

    offset = (page - 1) * page_size
    page_sql, extra_params = build_page_sql(
        cached.base_sql,
        cached.scored_filters,
        cached.apply_relevance,
        page_size,
        offset,
        rank_tsquery=cached.rank_tsquery,
        rerank_ordered=cached.rerank_ordered,
        rerank_demoted=cached.rerank_demoted,
    )
    result = await session.execute(
        text(page_sql), {**cached.params, **extra_params}
    )
    datasets = await _rows_to_datasets(session, result.mappings().all())
    await _attach_snippets(session, datasets, _highlight_for(cached))
    await _attach_descriptions(session, datasets)
    return datasets, total


@router.post("", response_model=QueryResponse)
async def query_datasets(
    body: QueryRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[dict, Depends(require_auth)],
):
    """
    Submit a natural-language question and receive the first page of matching
    BIDS datasets, plus a ``query_id`` for paging through the rest.
    """
    await _enforce_rate_limit(session, _auth.get("sub", ""))
    page, page_size = _clamp_page(1, _DEFAULT_PAGE_SIZE)
    translation = await text_to_sql(body.question)

    cached = CachedQuery(
        base_sql=translation.sql,
        params=translation.params,
        scored_filters=translation.scored_filters,
        apply_relevance=translation.apply_relevance,
        rank_tsquery=translation.rank_tsquery,
        explanation=translation.explanation,
        self_corrected=translation.self_corrected,
        filter_sql=translation.filter_sql,
    )
    query_id = query_cache.put(cached)

    datasets, total = await _run_page(
        session, query_id, cached, body.question, page, page_size
    )
    # Surface any base-SQL self-correction that happened during the count.
    reranked = len(cached.rerank_ordered) + len(cached.rerank_demoted)
    translation = translation.model_copy(
        update={
            "sql": cached.base_sql,
            "self_corrected": cached.self_corrected,
            "reranked": reranked or None,
        }
    )

    from models import UserQuery
    try:
        async with session.begin_nested():
            session.add(UserQuery(github_login=_auth["sub"], question=body.question))
        # The savepoint only protects the response from a failed insert. Without
        # this commit the row is rolled back when the session closes (get_db
        # never commits), which left user_queries empty from June 2026 on.
        await session.commit()
    except Exception:
        pass  # stale/legacy JWT sub not in users table — don't crash the response

    return QueryResponse(
        message=_found_message(total),
        translation=translation,
        filters=_filters_for(cached),
        datasets=datasets,
        query_id=query_id,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.post("/page", response_model=QueryResponse)
async def query_page(
    body: QueryPageRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
):
    """Fetch another page of an already-translated query, reusing its cached base
    SQL (no new LLM call)."""
    cached = query_cache.get(body.query_id)
    if cached is None:
        raise HTTPException(
            status_code=410,
            detail="This search has expired. Please run the query again.",
        )

    page, page_size = _clamp_page(body.page, body.page_size)
    # The question is only needed for self-correction, which already happened on
    # the first page; an empty string is fine here.
    datasets, total = await _run_page(
        session, body.query_id, cached, "", page, page_size
    )

    return QueryResponse(
        message=_found_message(total),
        translation=None,
        datasets=datasets,
        query_id=body.query_id,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.post("/refine", response_model=QueryResponse)
async def query_refine(
    body: QueryRefineRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[dict, Depends(require_auth)],
):
    """Re-run a translated query without some of its conditions.

    *remove* names chips of the original query (its ``filters``). The
    conditions are dropped from the cached query (services/query_filters.py),
    so no language model runs and the daily limit is not charged. The answer
    carries a new ``query_id`` for paging; the original stays valid, so the
    user can put chips back by refining the original again.
    """
    original = query_cache.get(body.query_id)
    if original is None:
        raise HTTPException(
            status_code=410,
            detail="This search has expired. Please run the query again.",
        )
    if original.self_corrected or not original.filter_sql:
        raise HTTPException(status_code=400, detail="This search has no filters to change.")

    filter_sql = query_filters.remove(original.filter_sql, body.remove)
    if filter_sql is None:
        raise HTTPException(status_code=400, detail="Could not change the filters of this search.")
    base_sql = uncertain_tier.apply(filter_sql) if uncertain_tier.enabled() else filter_sql

    cached = CachedQuery(
        base_sql=base_sql,
        params=original.params,
        scored_filters=original.scored_filters,
        apply_relevance=original.apply_relevance,
        rank_tsquery=original.rank_tsquery,
        explanation=original.explanation,
        # Keep the original re-ranking instead of asking the LLM again.
        rerank_done=True,
        rerank_ordered=list(original.rerank_ordered),
        rerank_demoted=list(original.rerank_demoted),
        filter_sql=filter_sql,
    )
    # Count here rather than in _run_page, whose error path asks Gemini to fix
    # the SQL: dropping AND conditions from a working query should not fail,
    # and if it does, the user should hear so instead of waiting on an LLM.
    try:
        result = await session.execute(text(build_count_sql(base_sql)), cached.params)
        cached.total = int(result.scalar() or 0)
    except Exception:
        await session.rollback()
        raise HTTPException(status_code=400, detail="Could not run the search without these filters.")
    query_id = query_cache.put(cached)

    page, page_size = _clamp_page(1, body.page_size)
    datasets, total = await _run_page(session, query_id, cached, "", page, page_size)
    return QueryResponse(
        message=_found_message(total),
        # The SQL that ran, for the "Show generated SQL" panel.
        translation=TextToSQLResult(sql=base_sql, explanation=original.explanation),
        filters=_filters_for(original),
        datasets=datasets,
        query_id=query_id,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )
