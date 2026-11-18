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
from typing import Annotated, Collection, Optional, Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from deps import get_db
from rate_limits import Category, rate_limit
from routers.auth import require_auth
from schemas import (
    DatasetSchema,
    LinkedPaperSchema,
    ParticipantSchema,
    ParticipantSummarySchema,
    QueryFilterSchema,
    QueryPageRequest,
    QueryRequest,
    QueryResponse,
    TextToSQLResult,
)
from services import catalogue_values, dataset_description, interpretation_store, query_filters
from services import query_guard
from services.query_cache import CachedQuery, query_cache
from services import translation_store
from services.schema_capabilities import has_display_column
from services.rerank import rerank_candidates, settings as rerank_settings, should_rerank
from services.sql_rewriter import FALLBACK_SQL as _FALLBACK_SQL, build_count_sql, build_page_sql
from services.text_to_sql import _correct_sql_with_gemini, text_to_sql

router = APIRouter(prefix="/query", tags=["query"])
log = logging.getLogger(__name__)

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


# Below this many results a free-text-only condition is dropped (see below).
_MIN_RESULTS = 3
# ... and only when what is left without it still narrows the catalogue.
_MAX_RELAXED_RESULTS = 300
async def _guarded_count(session: AsyncSession, base_sql: str, params: dict) -> int:
    """COUNT of a generated query behind the guards of services/query_guard.py:
    per-transaction limits, then the static and plan checks (QueryRefused)."""
    await query_guard.apply_limits(session)
    count_sql = build_count_sql(base_sql)
    if base_sql != _FALLBACK_SQL:
        await query_guard.vet(session, base_sql, count_sql, params)
    result = await session.execute(text(count_sql), params)
    return int(result.scalar() or 0)


async def _count_with_base_correction(
    session: AsyncSession, query_id: str, cached: CachedQuery, question: str
) -> int:
    """Count total matching datasets, self-correcting the cached *base* SQL once
    via Gemini if it raises a DB error.

    Running the COUNT validates the base SQL before any page is fetched. A fix is
    persisted back to the cache so later page turns reuse the corrected base.
    """
    try:
        total = await _guarded_count(session, cached.base_sql, cached.params)
        if total < _MIN_RESULTS and cached.relaxed_sql and not cached.relaxed:
            # An empty or nearly empty page is the worst answer: search once
            # more without the free-text-only conditions and say so. This is
            # the check of a free-text phrase against the datasets that match
            # the other filters: a phrase that leaves fewer than _MIN_RESULTS
            # of them is not used as a filter. The text rank still puts the
            # datasets that do contain the words first.
            # In a savepoint: a refused or failing fallback must not undo the
            # count above or the transaction's limits.
            try:
                async with session.begin_nested():
                    relaxed_total = await _guarded_count(session, cached.relaxed_sql, cached.params)
            except Exception:
                relaxed_total = 0
            # Without the phrase the search must still be a search: 990 or
            # 1,400 datasets in no useful order (0 correct answers among the
            # first 20 in every measured case) are no better than none.
            if total < relaxed_total <= _MAX_RELAXED_RESULTS:
                cached.base_sql = cached.relaxed_sql
                cached.relaxed = True
                cached.filter_sql = None
                cached.explanation = ((cached.explanation or "").rstrip() +
                                      " Fewer than three datasets matched every condition, so the"
                                      " free-text conditions only order the result.").strip()
                query_cache.update(query_id, cached)
                return relaxed_total
        return total
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
            return await _guarded_count(session, _FALLBACK_SQL, {})
        cached.base_sql = corrected
        cached.self_corrected = True
        query_cache.update(query_id, cached)
        try:
            return await _guarded_count(session, cached.base_sql, cached.params)
        except Exception:
            await session.rollback()
            cached.base_sql = _FALLBACK_SQL
            cached.params = {}
            query_cache.update(query_id, cached)
            return await _guarded_count(session, _FALLBACK_SQL, {})


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


_MARK_RE = re.compile(f"({SNIPPET_START}[^{SNIPPET_STOP}]*{SNIPPET_STOP})")
_SNIPPET_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# The Postgres stems of a text, space-separated.
_LEXEMES_SQL = "array_to_string(tsvector_to_array(to_tsvector('english', {})), ' ')"


def keep_justified_marks(headline: str, mark_lexemes: Sequence[str],
                         terms: Sequence[tuple], term_lexemes: dict,
                         exact: Collection[str] = ()) -> str:
    """*headline* with only the marks that *terms* justify.

    ts_headline marks every query word in the passage, also a word of a
    pair whose partner is elsewhere ("structure" for "structural MRI").
    A pair's words keep their marks only next to each other, with at most
    one word between ("Parkinson's disease"). *mark_lexemes* holds the stems
    of each mark in order, *term_lexemes* those of each term word.

    *exact*: term words that only count as written. The search was limited
    to these word forms ("ageing"), so a mark on another word with the same
    stem ("age") goes.
    """
    parts = _MARK_RE.split(headline)
    marks = [i for i, part in enumerate(parts) if part.startswith(SNIPPET_START)]
    if len(marks) != len(mark_lexemes):
        return headline
    stems = [set(lexemes.split()) for lexemes in mark_lexemes]

    def matches(index: int, word: str) -> bool:
        if word.lower() in exact:
            written = re.split(r"['’]", parts[marks[index]][1:-1].lower())[0]
            return written == word.lower()
        if not stems[index] & term_lexemes.get(word.lower(), set()):
            return False
        # A two-letter acronym only in capitals: "MS", not "500 ms".
        return not (len(word) == 2 and word.isupper()) or \
            parts[marks[index]][1:-1].startswith(word)

    keep = set()
    for k in range(len(marks)):
        for term in terms:
            if len(term) == 1:
                if matches(k, term[0]):
                    keep.add(k)
            elif k + 1 < len(marks) and matches(k, term[0]) and matches(k + 1, term[1]):
                gap = "".join(parts[marks[k] + 1:marks[k + 1]])
                if len(_SNIPPET_WORD_RE.findall(gap)) <= 1:
                    keep.update((k, k + 1))
    for k, i in enumerate(marks):
        if k not in keep:
            parts[i] = parts[i][1:-1]
    return "".join(parts)


async def _attach_snippets(
    session: AsyncSession, datasets: list[DatasetSchema], terms: Optional[Sequence[tuple]],
    exact: Collection[str] = (),
) -> None:
    """Give each dataset a README excerpt: the passages matching the highlight
    *terms* (services/query_filters.highlight_terms), or the start.

    *exact*: the terms that only count as written
    (query_filters.highlight_exact_words). Where the README has one of them,
    the excerpt is the passage around it, found without stemming; a passage
    chosen by stem would show "age" for a search on "ageing".

    Read with its own query, like _deprecations_for, so the generated SQL stays
    unchanged and databases without readme_text keep working. A failure leaves
    the page without excerpts rather than failing it.
    """
    if not datasets or not has_display_column("readme_text"):
        return
    ids = [d.id for d in datasets]
    highlight = query_filters.terms_tsquery(terms or ())
    words = list(dict.fromkeys(w.lower() for term in terms or () for w in term))
    exact = frozenset(w.lower() for w in exact if _SNIPPET_WORD_RE.fullmatch(w))
    if highlight:
        # The passage around a word that only counts as written, if there is one.
        exact_headline = (
            ", ts_headline('simple', left(readme_text, :chars), "
            "to_tsquery('simple', :exact_q), :opts) AS hl_exact" if exact else "")
        sql = (
            "SELECT h.id, h.hl, "
            "ARRAY(SELECT " + _LEXEMES_SQL.format("m.t[1]") + " FROM regexp_matches(h.hl, :mark, 'g') "
            "WITH ORDINALITY AS m(t, n) ORDER BY m.n) AS mark_lexemes, "
            "ARRAY(SELECT " + _LEXEMES_SQL.format("w.t") + " FROM unnest(CAST(:words AS text[])) "
            "WITH ORDINALITY AS w(t, n) ORDER BY w.n) AS term_lexemes"
            + (", h.hl_exact " if exact else " ") +
            "FROM (SELECT id, ts_headline('english', left(readme_text, :chars), "
            "to_tsquery('english', :q), :opts) AS hl" + exact_headline + " FROM bids_datasets "
            "WHERE id = ANY(:ids) AND readme_text IS NOT NULL AND readme_text <> '') AS h"
        )
        params = {"ids": ids, "q": highlight, "opts": _HEADLINE_OPTIONS,
                  "chars": _SNIPPET_README_CHARS, "words": words,
                  "mark": f"{SNIPPET_START}([^{SNIPPET_STOP}]*){SNIPPET_STOP}"}
        if exact:
            params["exact_q"] = " | ".join(sorted(exact))
    else:
        sql = (
            "SELECT id, left(readme_text, :chars) FROM bids_datasets "
            "WHERE id = ANY(:ids) AND readme_text IS NOT NULL AND readme_text <> ''"
        )
        params = {"ids": ids, "chars": _SNIPPET_PLAIN_CHARS * 4}
    try:
        async with session.begin_nested():
            rows = await session.execute(text(sql), params)
            found = {}
            for row in rows:
                raw = row[1]
                if highlight and raw:
                    by_form = row[4] if exact and len(row) > 4 else None
                    if by_form and SNIPPET_START in by_form:
                        raw = by_form   # every mark there is a word form asked for
                    else:
                        term_lexemes = {w: set(lexemes.split())
                                        for w, lexemes in zip(words, row[3])}
                        raw = keep_justified_marks(raw, row[2], terms, term_lexemes, exact)
                found[row[0]] = raw
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


def _highlight_for(cached: CachedQuery) -> list[tuple]:
    if cached.self_corrected:
        return []
    return query_filters.highlight_terms(cached.filter_sql)


def _exact_words_for(cached: CachedQuery) -> frozenset:
    """The highlight words that only count as written (exact word forms)."""
    if cached.self_corrected:
        return frozenset()
    return query_filters.highlight_exact_words(cached.filter_sql)


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
    """_run_page_unguarded in one of the few slots for generated SQL."""
    try:
        async with query_guard.slot():
            return await _run_page_unguarded(session, query_id, cached, question, page, page_size)
    except query_guard.SearchBusy:
        raise HTTPException(status_code=503, detail="The search is busy. Please try again in a minute.")


async def _run_page_unguarded(
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

    def page_query(ranked: bool) -> tuple[str, dict]:
        page_sql, extra_params = build_page_sql(
            cached.base_sql,
            cached.scored_filters,
            cached.apply_relevance and ranked,
            page_size,
            offset,
            rank_tsquery=cached.rank_tsquery if ranked else None,
            rerank_ordered=cached.rerank_ordered,
            rerank_demoted=cached.rerank_demoted,
        )
        return page_sql, {**cached.params, **extra_params}

    await query_guard.apply_limits(session)
    rows = None
    if cached.apply_relevance and not cached.ranking_skipped:
        # The relevance ranking checks every matched code per dataset and took
        # 98 s for "MRI datasets" and 174 s for "resting-state datasets"
        # (2026-11-14) where the unranked page takes 4-9 s. A ranked page gets
        # a short timeout of its own; past it the user gets the unranked page.
        page_sql, params = page_query(ranked=True)
        try:
            async with session.begin_nested():
                await session.execute(text(
                    f"SET LOCAL statement_timeout = '{query_guard.ranked_page_timeout()}'"))
                rows = (await session.execute(text(page_sql), params)).mappings().all()
            await query_guard.apply_limits(session)  # the page's timeout outlives its savepoint
        except Exception as exc:
            if not query_guard.is_timeout(exc):
                raise
            log.warning("Ranked page too slow; serving query %s unranked", query_id)
            cached.ranking_skipped = True
            query_cache.update(query_id, cached)
    if rows is None:
        page_sql, params = page_query(ranked=False)
        rows = (await session.execute(text(page_sql), params)).mappings().all()
    datasets = await _rows_to_datasets(session, rows)
    await _attach_snippets(session, datasets, _highlight_for(cached), _exact_words_for(cached))
    await _attach_descriptions(session, datasets)
    return datasets, total


# Searches running now, by (user, normalised question). The frontend sends a
# question only once while it runs; this is the same rule on the server, for
# double submits, retries and second tabs. One uvicorn process serves the API
# (backend/Dockerfile), so a dict in memory sees every request.
_in_flight: dict[tuple[str, str], asyncio.Future] = {}


@router.post("", response_model=QueryResponse, dependencies=[rate_limit(Category.SEARCH)])
async def query_datasets(
    body: QueryRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
    _auth: Annotated[dict, Depends(require_auth)],
):
    """
    Submit a natural-language question and receive the first page of matching
    BIDS datasets, plus a ``query_id`` for paging through the rest.

    The same question from the same user while it is still running does not
    start a second search: it gets the running search's answer (or error),
    without another language-model call. The rate limit (rate_limits.py,
    SEARCH) still counts it, since it ran before this function.
    """
    key = (_auth.get("sub", ""), interpretation_store.normalize(body.question))
    running = _in_flight.get(key)
    if running is not None:
        try:
            return await asyncio.shield(running)
        except asyncio.CancelledError:
            if running.cancelled():
                raise HTTPException(status_code=503, detail="The same search was interrupted. Please try again.")
            raise

    answer: asyncio.Future = asyncio.get_running_loop().create_future()
    # Marks an error as seen when no second request was waiting for it.
    answer.add_done_callback(lambda f: f.cancelled() or f.exception())
    _in_flight[key] = answer
    try:
        response = await _answer_question(body, session, _auth)
    except asyncio.CancelledError:
        answer.cancel()
        raise
    except BaseException as exc:
        answer.set_exception(exc)
        raise
    else:
        answer.set_result(response)
        return response
    finally:
        _in_flight.pop(key, None)


async def _answer_question(body: QueryRequest, session: AsyncSession, _auth: dict) -> QueryResponse:
    """One search: translation, first page, logging."""
    page, page_size = _clamp_page(1, _DEFAULT_PAGE_SIZE)
    # The same question gets the same translation every time (the models that
    # write it are not repeatable): reuse a stored one when there is one.
    await catalogue_values.refresh_if_stale(session)
    stored = await interpretation_store.get(session, body.question)
    if stored is not None:
        translation = TextToSQLResult(
            sql=stored["sql"], explanation=stored.get("explanation") or "",
            scored_filters=stored.get("scored_filters") or [],
            apply_relevance=bool(stored.get("apply_relevance")),
            params=stored.get("params") or {}, rank_tsquery=stored.get("rank_tsquery"),
            filter_sql=stored.get("filter_sql"), relaxed_sql=stored.get("relaxed_sql"),
        )
    else:
        # The exact-match store of services/translation_store.py (off unless
        # TRANSLATION_STORE is set) is asked inside text_to_sql.
        await translation_store.refresh(session)
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
        relaxed_sql=translation.relaxed_sql,
        relaxed=bool(stored and stored.get("relaxed")),
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

    payload = {
        "sql": cached.base_sql, "params": cached.params,
        "scored_filters": cached.scored_filters, "apply_relevance": cached.apply_relevance,
        "rank_tsquery": cached.rank_tsquery, "explanation": cached.explanation,
        "filter_sql": cached.filter_sql,
        "relaxed_sql": None if cached.relaxed else cached.relaxed_sql,
        "relaxed": cached.relaxed,
    }
    # Every search is recorded for good, whatever it found.
    await interpretation_store.log_query(
        session, body.question, payload, total, from_cache=stored is not None,
        relaxed=cached.relaxed, self_corrected=bool(cached.self_corrected),
        fallback=cached.base_sql == _FALLBACK_SQL,
    )
    # Keep a translation that found something and needed no emergency fallback.
    if stored is None and total > 0 and cached.base_sql != _FALLBACK_SQL:
        await interpretation_store.put(session, body.question, payload, total)
    # Keep the translation for the next time this question is asked, but only
    # one that ran as written and found something: a corrected or empty search
    # should get a fresh attempt.
    if (translation.store_entry is not None and not cached.self_corrected
            and cached.base_sql != _FALLBACK_SQL and total > 0):
        translation.store_entry.final_sql = cached.base_sql
        translation.store_entry.result_total = total
        await translation_store.remember(session, translation.store_entry)
    elif translation.from_store:
        await translation_store.count_hit(session, body.question)

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


@router.post("/page", response_model=QueryResponse, dependencies=[rate_limit(Category.PAGE)])
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
