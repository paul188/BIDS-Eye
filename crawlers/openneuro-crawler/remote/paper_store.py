"""
remote/paper_store.py
---------------------
Store the papers datasets name, and the text search reads from them.

    bids_papers            one row per paper (title, abstract, MeSH, …)
    bids_dataset_papers    which dataset names which paper, with its role and
                           the reason for the role
    bids_datasets.paper_text
                           title, abstract and keywords of the dataset's
                           primary papers. A generated column cannot read
                           another table, so the text is copied here; the
                           search vector that includes it is built from this
                           column (crawlers/migrate_add_paper_search.sql)

`enrich()` is the whole step for a set of datasets: read their link fields,
ask OpenNeuro for the three fields the crawler does not store, look the papers
up, decide roles, write. It is used by scripts/backfill_papers.py and can be
called from the crawler cycle for new datasets.

Design and measurements: crawlers/paper_enrichment_plan.md.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import String

from remote import paper_links as links_mod
from remote import paper_lookup
from remote.openneuro_metadata import OpenNeuroAPIError, post_graphql
from remote.paper_links import Link, PaperId, ROLE_PRIMARY, SOURCES

log = logging.getLogger(__name__)

# Datasets per OpenNeuro request for the paper fields (small answers).
PAPER_FIELDS_BATCH = 60

# A lab that lists its whole bibliography would otherwise put twenty
# abstracts into one dataset's searchable text.
MAX_INDEXED_PAPERS = 5

# What of a paper becomes searchable. The benchmark compares subsets.
TEXT_PARTS: Tuple[str, ...] = ("title", "abstract", "mesh", "keywords")

# MeSH check tags that describe every human study and carry no information
# for search.
_MESH_NOISE = frozenset({"Humans", "Male", "Female"})

DDL: Tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS bids_papers (
        id             BIGSERIAL PRIMARY KEY,
        paper_key      TEXT NOT NULL UNIQUE,   -- doi:… / pmid:… / pmcid:… / arxiv:…
        doi            TEXT,                   -- published version
        preprint_doi   TEXT,
        pmid           TEXT,
        pmcid          TEXT,
        arxiv_id       TEXT,
        title          TEXT NOT NULL,
        abstract       TEXT,
        year           INTEGER,
        journal        TEXT,
        is_preprint    BOOLEAN NOT NULL DEFAULT FALSE,  -- no published version known
        authors        TEXT[],
        mesh_terms     TEXT[],
        mesh_major     TEXT[],
        keywords       TEXT[],
        is_open_access BOOLEAN,
        license        TEXT,
        source         TEXT NOT NULL,          -- europepmc, biorxiv, medrxiv, arxiv, openalex
        fetched_at     TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bids_dataset_papers (
        dataset_id  UUID NOT NULL REFERENCES bids_datasets(id) ON DELETE CASCADE,
        paper_id    BIGINT NOT NULL REFERENCES bids_papers(id) ON DELETE CASCADE,
        role        TEXT NOT NULL CHECK (role IN ('primary', 'related', 'tool', 'reuse')),
        role_reason TEXT,
        sources     TEXT[] NOT NULL,           -- associated, references, acknowledge, readme, reuse_field
        raw_ref     TEXT,                      -- the reference string the link was read from
        position    INTEGER NOT NULL DEFAULT 0, -- order within the dataset, most trusted first
        PRIMARY KEY (dataset_id, paper_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_bids_dataset_papers_paper ON bids_dataset_papers (paper_id)",
    "ALTER TABLE bids_datasets ADD COLUMN IF NOT EXISTS paper_text TEXT",
    "ALTER TABLE bids_datasets ADD COLUMN IF NOT EXISTS papers_checked_at TIMESTAMPTZ",
)

# How the primary paper relates to the dataset, and a generated summary for
# thin READMEs (backend/services/dataset_description.py). Kept in sync with
# crawlers/migrate_add_dataset_descriptions.sql (backend/tests/test_dataset_description.py).
DESCRIPTION_DDL: Tuple[str, ...] = (
    "ALTER TABLE bids_datasets ADD COLUMN IF NOT EXISTS paper_relation JSONB",
    "ALTER TABLE bids_datasets ADD COLUMN IF NOT EXISTS dataset_summary TEXT",
    "ALTER TABLE bids_datasets ADD COLUMN IF NOT EXISTS dataset_summary_meta JSONB",
)


# search_tsv plus the paper text at the lowest weight. Kept in sync with
# crawlers/migrate_add_paper_search.sql (backend/tests/test_paper_links.py
# checks this). Built from the README vector's expression, so a change there
# needs the migration rerun.
from remote.openneuro_metadata import SEARCH_VECTOR_EXPRESSION  # noqa: E402

PAPER_SEARCH_COLUMN = "search_tsv_papers"
PAPER_SEARCH_VECTOR_EXPRESSION = (
    f"{SEARCH_VECTOR_EXPRESSION} || "
    "setweight(to_tsvector('english'::regconfig, "
    "translate(coalesce(paper_text, ''), '/', ' ')), 'D')"
)

# The same vector without stemming (crawlers/migrate_add_exact_search.sql;
# backend/tests/test_word_forms.py checks the two agree).
PAPER_EXACT_SEARCH_COLUMN = "search_tsv_exact_papers"
PAPER_EXACT_SEARCH_VECTOR_EXPRESSION = PAPER_SEARCH_VECTOR_EXPRESSION.replace(
    "'english'::regconfig", "'simple'::regconfig")


async def ensure_tables(session: Any) -> None:
    """Create the paper tables and columns, and the description columns, if
    they are missing. Caller commits."""
    for statement in DDL + DESCRIPTION_DDL:
        await session.execute(text(statement))


# ── OpenNeuro fields the crawler does not store ───────────────────────────────

def fetch_paper_fields(accession_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """{accession: {"associated", "reuse_field", "acknowledge"}} from OpenNeuro.

    Datasets OpenNeuro does not answer for are simply absent: their other link
    sources are still used.
    """
    out: Dict[str, Dict[str, Any]] = {}
    ids = [a for a in accession_ids if a]
    for start in range(0, len(ids), PAPER_FIELDS_BATCH):
        chunk = ids[start:start + PAPER_FIELDS_BATCH]
        query = "{" + " ".join(
            f'{acc}: dataset(id: "{acc}") {{ metadata {{ associatedPaperDOI openneuroPaperDOI }} '
            f"latestSnapshot {{ description {{ HowToAcknowledge }} }} }}"
            for acc in chunk
        ) + "}"
        try:
            body = post_graphql(query)
        except OpenNeuroAPIError as exc:
            log.warning("OpenNeuro paper fields failed for %s…: %s", chunk[0], exc)
            continue
        for acc, record in (body.get("data") or {}).items():
            if not record:
                continue
            metadata = record.get("metadata") or {}
            description = (record.get("latestSnapshot") or {}).get("description") or {}
            out[acc] = {
                "associated": metadata.get("associatedPaperDOI"),
                "reuse_field": metadata.get("openneuroPaperDOI"),
                "acknowledge": description.get("HowToAcknowledge"),
            }
    return out


# ── Searchable text ───────────────────────────────────────────────────────────

def paper_text(papers: Sequence[Dict[str, Any]], parts: Sequence[str] = TEXT_PARTS) -> Optional[str]:
    """What full-text search reads for a dataset's primary papers."""
    blocks: List[str] = []
    for paper in papers[:MAX_INDEXED_PAPERS]:
        pieces: List[str] = []
        if "title" in parts and paper.get("title"):
            pieces.append(paper["title"].rstrip(".") + ".")
        if "abstract" in parts and paper.get("abstract"):
            pieces.append(paper["abstract"])
        if "mesh" in parts:
            mesh = [m for m in paper.get("mesh_terms") or [] if m not in _MESH_NOISE]
            if mesh:
                pieces.append("; ".join(mesh) + ".")
        if "keywords" in parts and paper.get("keywords"):
            pieces.append("; ".join(paper["keywords"]) + ".")
        if pieces:
            blocks.append(" ".join(pieces))
    return "\n\n".join(blocks) or None


# ── One dataset: links → roles ────────────────────────────────────────────────

def dataset_entries(
    dataset: Dict[str, Any],
    key_of: Dict[PaperId, str],
    papers: Dict[str, Dict[str, Any]],
    datasets_per_paper: Dict[str, int],
) -> List[Dict[str, Any]]:
    """The papers of one dataset with role, reason, sources and reference string.

    `dataset` has accession, name, authors and `links` (from collect_links).
    Several identifiers of one paper become one entry.
    """
    merged: Dict[str, Link] = {}
    for link in dataset["links"]:
        key = key_of.get(link.paper_id)
        if key is None:
            continue
        if key not in merged:
            merged[key] = Link(paper_id=link.paper_id, sources=list(link.sources), raw=link.raw)
        else:
            for source in link.sources:
                if source not in merged[key].sources:
                    merged[key].sources.append(source)
            merged[key].raw = merged[key].raw or link.raw
    for link in merged.values():
        link.sources.sort(key=SOURCES.index)
    ordered = sorted(merged.items(), key=lambda item: SOURCES.index(item[1].sources[0]))
    roles = links_mod.assign_roles(
        accession=dataset["accession"],
        dataset_name=dataset.get("name") or "",
        dataset_authors=dataset.get("authors") or [],
        links=[(link, papers[key]) for key, link in ordered],
        datasets_per_paper=datasets_per_paper,
    )
    return [
        {"key": key, "role": role, "reason": reason, "sources": link.sources, "raw": link.raw}
        for (key, link), (role, reason) in zip(ordered, roles)
    ]


# ── Database ──────────────────────────────────────────────────────────────────

_ARRAY_FIELDS = ("authors", "mesh_terms", "mesh_major", "keywords")
_PAPER_FIELDS = (
    "doi", "preprint_doi", "pmid", "pmcid", "arxiv_id", "title", "abstract", "year", "journal",
    "is_preprint", "authors", "mesh_terms", "mesh_major", "keywords", "is_open_access",
    "license", "source",
)

_UPSERT_PAPER = text(
    f"""
    INSERT INTO bids_papers (paper_key, {", ".join(_PAPER_FIELDS)})
    VALUES (:paper_key, {", ".join(":" + name for name in _PAPER_FIELDS)})
    ON CONFLICT (paper_key) DO UPDATE SET
        {", ".join(f"{name} = EXCLUDED.{name}" for name in _PAPER_FIELDS)},
        fetched_at = now()
    RETURNING id
    """
).bindparams(*(bindparam(name, type_=ARRAY(String)) for name in _ARRAY_FIELDS))

_INSERT_LINK = text(
    """
    INSERT INTO bids_dataset_papers (dataset_id, paper_id, role, role_reason, sources, raw_ref, position)
    VALUES (:dataset_id, :paper_id, :role, :reason, :sources, :raw, :position)
    ON CONFLICT (dataset_id, paper_id) DO NOTHING
    """
).bindparams(bindparam("sources", type_=ARRAY(String)))


async def upsert_paper(session: Any, paper: Dict[str, Any]) -> int:
    params = {name: paper.get(name) for name in _PAPER_FIELDS}
    for name in _ARRAY_FIELDS:
        params[name] = [str(v) for v in (params[name] or []) if v] or None
    params["is_preprint"] = bool(params["is_preprint"])
    params["paper_key"] = paper["key"]
    result = await session.execute(_UPSERT_PAPER, params)
    return int(result.scalar_one())


async def write_dataset(
    session: Any,
    dataset_id: Any,
    entries: Sequence[Dict[str, Any]],
    papers: Dict[str, Dict[str, Any]],
    paper_ids: Dict[str, int],
    parts: Sequence[str] = TEXT_PARTS,
) -> None:
    """Replace the links of one dataset and rewrite its searchable paper text."""
    await session.execute(
        text("DELETE FROM bids_dataset_papers WHERE dataset_id = :dataset_id"),
        {"dataset_id": dataset_id},
    )
    for position, entry in enumerate(entries):
        await session.execute(_INSERT_LINK, {
            "dataset_id": dataset_id, "paper_id": paper_ids[entry["key"]], "role": entry["role"],
            "reason": entry["reason"], "sources": list(entry["sources"]), "raw": entry["raw"],
            "position": position,
        })
    primary = [papers[e["key"]] for e in entries if e["role"] == ROLE_PRIMARY]
    await session.execute(
        text("UPDATE bids_datasets SET paper_text = :paper_text, papers_checked_at = now() "
             "WHERE id = :dataset_id"),
        {"paper_text": paper_text(primary, parts), "dataset_id": dataset_id},
    )


async def load_datasets(session: Any, accession_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """The link-bearing fields of the indexed OpenNeuro datasets."""
    where = "accession_id IS NOT NULL"
    params: Dict[str, Any] = {}
    if accession_ids:
        where += " AND accession_id = ANY(:accessions)"
        params["accessions"] = list(accession_ids)
    statement = text(
        "SELECT id, accession_id, name, authors, paper_references, readme_text "
        f"FROM bids_datasets WHERE {where} ORDER BY accession_id"
    )
    if accession_ids:
        statement = statement.bindparams(bindparam("accessions", type_=ARRAY(String)))
    rows = await session.execute(statement, params)
    return [
        {"id": row[0], "accession": row[1], "name": row[2], "authors": list(row[3] or []),
         "references": list(row[4] or []), "readme": row[5]}
        for row in rows
    ]


async def _stored_counts(session: Any, excluding: Sequence[Any]) -> Dict[str, int]:
    """Datasets per paper among the datasets that are not being rewritten now."""
    statement = text(
        "SELECT p.paper_key, count(DISTINCT dp.dataset_id) FROM bids_dataset_papers dp "
        "JOIN bids_papers p ON p.id = dp.paper_id "
        "WHERE NOT (dp.dataset_id::text = ANY(:excluding)) GROUP BY 1"
    ).bindparams(bindparam("excluding", type_=ARRAY(String)))
    rows = await session.execute(statement, {"excluding": [str(i) for i in excluding]})
    return {row[0]: int(row[1]) for row in rows}


def plan(
    datasets: Sequence[Dict[str, Any]],
    paper_fields: Dict[str, Dict[str, Any]],
    fetch: Callable[[str], Any] = paper_lookup.http_get,
    *,
    stored_counts: Optional[Dict[str, int]] = None,
    resolved: Optional[Tuple[Dict[PaperId, str], Dict[str, Dict[str, Any]]]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]], Dict[PaperId, str]]:
    """Everything except the writing.

    Returns ({accession: entries}, papers by key, which paper each identifier
    is). `resolved` supplies saved lookup results (the last two) instead of
    calling the services.
    """
    for dataset in datasets:
        extra = paper_fields.get(dataset["accession"]) or {}
        dataset["links"] = links_mod.collect_links(
            associated=extra.get("associated"),
            references=dataset.get("references"),
            acknowledge=extra.get("acknowledge"),
            readme=dataset.get("readme"),
            reuse_field=extra.get("reuse_field"),
        )
    if resolved is None:
        wanted = sorted({link.paper_id for dataset in datasets for link in dataset["links"]})
        resolved = paper_lookup.resolve(wanted, fetch)
    key_of, papers = resolved

    counts: Counter = Counter(stored_counts or {})
    for dataset in datasets:
        for key in {key_of[link.paper_id] for link in dataset["links"] if link.paper_id in key_of}:
            counts[key] += 1
    entries = {
        dataset["accession"]: dataset_entries(dataset, key_of, papers, counts)
        for dataset in datasets
    }
    return entries, papers, key_of


def summarize(entries: Dict[str, List[Dict[str, Any]]], papers: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    stats: Counter = Counter()
    used = set()
    for dataset_entries_ in entries.values():
        stats["datasets"] += 1
        stats["datasets with a paper"] += bool(dataset_entries_)
        primary = [e for e in dataset_entries_ if e["role"] == ROLE_PRIMARY]
        stats["datasets with a primary paper"] += bool(primary)
        stats["… whose primary paper has an abstract"] += any(papers[e["key"]].get("abstract") for e in primary)
        stats["… whose primary paper has MeSH"] += any(papers[e["key"]].get("mesh_terms") for e in primary)
        for entry in dataset_entries_:
            stats[f"links: {entry['role']}"] += 1
            used.add(entry["key"])
    stats["papers"] = len(used)
    return dict(stats)


async def enrich(
    session: Any,
    accession_ids: Optional[Sequence[str]] = None,
    *,
    fetch: Callable[[str], Any] = paper_lookup.http_get,
    paper_fields: Optional[Dict[str, Dict[str, Any]]] = None,
    resolved: Optional[Tuple[Dict[PaperId, str], Dict[str, Dict[str, Any]]]] = None,
    parts: Sequence[str] = TEXT_PARTS,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Find, look up and store the papers of the given datasets (default: all).

    Caller commits. Network calls are synchronous and happen before any write,
    so a failed lookup leaves the database as it was.
    """
    datasets = await load_datasets(session, accession_ids)
    if paper_fields is None:
        paper_fields = fetch_paper_fields([d["accession"] for d in datasets])
    stored: Dict[str, int] = {}
    if await _has_tables(session):
        stored = await _stored_counts(session, [d["id"] for d in datasets])
    entries, papers, _ = plan(datasets, paper_fields, fetch, stored_counts=stored, resolved=resolved)
    if stored or await _has_tables(session):
        apply_overrides(entries, await load_overrides(session, [d["id"] for d in datasets]),
                        {d["accession"]: d["id"] for d in datasets})
    stats = summarize(entries, papers)
    if dry_run:
        return stats

    await ensure_tables(session)
    used = {entry["key"] for dataset_entries_ in entries.values() for entry in dataset_entries_}
    paper_ids = {key: await upsert_paper(session, papers[key]) for key in sorted(used)}
    for dataset in datasets:
        await write_dataset(session, dataset["id"], entries[dataset["accession"]], papers, paper_ids, parts)
    return stats


# A role set by hand or by a later check (scripts/demote_paper_links.py)
# carries this prefix in role_reason and survives every re-run of enrich().
OVERRIDE_PREFIX = "override: "


async def load_overrides(session: Any, dataset_ids: Sequence[Any]) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """{(dataset id, paper key): (role, reason)} for overridden links."""
    statement = text(
        "SELECT dp.dataset_id::text, p.paper_key, dp.role, dp.role_reason FROM bids_dataset_papers dp "
        "JOIN bids_papers p ON p.id = dp.paper_id "
        "WHERE dp.role_reason LIKE :prefix AND dp.dataset_id::text = ANY(:ids)"
    ).bindparams(bindparam("ids", type_=ARRAY(String)))
    rows = await session.execute(statement, {"prefix": OVERRIDE_PREFIX + "%",
                                             "ids": [str(i) for i in dataset_ids]})
    return {(row[0], row[1]): (row[2], row[3]) for row in rows}


def apply_overrides(
    entries: Dict[str, List[Dict[str, Any]]],
    overrides: Dict[Tuple[str, str], Tuple[str, str]],
    dataset_ids: Dict[str, Any],
) -> int:
    """Put overridden roles back into freshly planned entries. Returns how many."""
    applied = 0
    for accession, dataset_entries_ in entries.items():
        dataset_id = str(dataset_ids.get(accession))
        for entry in dataset_entries_:
            override = overrides.get((dataset_id, entry["key"]))
            if override:
                entry["role"], entry["reason"] = override
                applied += 1
    return applied


async def _has_tables(session: Any) -> bool:
    result = await session.execute(text("SELECT to_regclass('bids_dataset_papers') IS NOT NULL"))
    return bool(result.scalar_one())
