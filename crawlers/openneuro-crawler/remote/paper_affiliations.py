"""
remote/paper_affiliations.py
----------------------------
Sites from the dataset's primary paper: the institutions of its first and
last author, as OpenAlex records them (with their ROR id), stored in
bids_dataset_sites with source 'paper_affiliation'. remote/site_places.py then
gives each its city, region and country, so a question that names a place
also finds datasets whose scans and ethics statement name no institution.

Only the first and last author: they usually lead the lab that collected the
data. Middle authors of a consortium paper would place one dataset in a dozen
countries.

A dataset's rows are replaced only when OpenAlex answered for all its primary
papers; on a failed lookup the old rows stay.

    python -m remote.paper_affiliations        # every dataset with a primary paper
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import text

log = logging.getLogger(__name__)

SOURCE = "paper_affiliation"
_OPENALEX_URL = "https://api.openalex.org/works/doi:{doi}?select=id,doi,authorships"
_USER_AGENT = "bids-eye-crawler/1.0 (paper affiliations)"
_PAUSE_SECONDS = 0.15
_POSITIONS = ("first", "last")

_PRIMARY_PAPERS = (
    "SELECT d.id, d.accession_id, p.doi FROM bids_dataset_papers dp "
    "JOIN bids_papers p ON p.id = dp.paper_id JOIN bids_datasets d ON d.id = dp.dataset_id "
    "WHERE dp.role = 'primary' AND p.doi IS NOT NULL AND p.doi <> ''"
)
_DELETE = text("DELETE FROM bids_dataset_sites WHERE dataset_id = :dataset_id AND source = :source")
_INSERT = text(
    "INSERT INTO bids_dataset_sites (dataset_id, name_raw, source, ror_id, evidence) "
    "VALUES (:dataset_id, :name_raw, :source, :ror_id, :evidence) "
    "ON CONFLICT (dataset_id, source, name_raw) DO NOTHING"
)

Affiliation = Tuple[str, str, str]   # (institution name, ROR id, evidence)


def openalex_authorships(doi: str, timeout: float = 30.0, retries: int = 3) -> Optional[List[dict]]:
    """OpenAlex's authorships of the work with *doi*; [] when unknown, None when unreachable."""
    url = _OPENALEX_URL.format(doi=urllib.parse.quote(doi.strip(), safe="/"))
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response).get("authorships") or []
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return []
            log.warning("OpenAlex %s: HTTP %s, attempt %d", doi, exc.code, attempt + 1)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            log.warning("OpenAlex %s failed (%s), attempt %d", doi, exc, attempt + 1)
        time.sleep(5 * (attempt + 1))
    return None


def lead_affiliations(doi: str, authorships: Iterable[dict]) -> List[Affiliation]:
    """The institutions (with a ROR id) of the first and last author."""
    found: Dict[str, Affiliation] = {}
    for authorship in authorships:
        position = authorship.get("author_position")
        if position not in _POSITIONS:
            continue
        for institution in authorship.get("institutions") or []:
            name, ror = (institution.get("display_name") or "").strip(), institution.get("ror") or ""
            if name and ror and name not in found:
                found[name] = (name, ror, f"{position} author of doi:{doi}")
    return list(found.values())


def collect(papers: Sequence[Tuple[Any, str, str]], lookup=openalex_authorships,
            pause: float = _PAUSE_SECONDS) -> Dict[Any, Optional[List[Affiliation]]]:
    """{dataset_id: affiliations} for (dataset_id, accession, doi) rows; None when a lookup failed."""
    by_doi: Dict[str, Optional[List[dict]]] = {}
    result: Dict[Any, Optional[List[Affiliation]]] = {}
    for dataset_id, _accession, doi in papers:
        if doi not in by_doi:
            by_doi[doi] = lookup(doi)
            if pause:
                time.sleep(pause)
        authorships = by_doi[doi]
        if authorships is None or result.get(dataset_id, []) is None:
            result[dataset_id] = None
            continue
        merged = {a[0]: a for a in result.get(dataset_id) or []}
        for affiliation in lead_affiliations(doi, authorships):
            merged.setdefault(affiliation[0], affiliation)
        result[dataset_id] = list(merged.values())
    return result


async def store(conn: Any, affiliations: Dict[Any, Optional[List[Affiliation]]]) -> Dict[str, int]:
    """Replace the paper-affiliation sites of every dataset whose lookups all succeeded."""
    stats = {"datasets": 0, "rows": 0, "kept_after_failure": 0}
    for dataset_id, found in affiliations.items():
        if found is None:
            stats["kept_after_failure"] += 1
            continue
        await conn.execute(_DELETE, {"dataset_id": dataset_id, "source": SOURCE})
        for name, ror, evidence in found:
            await conn.execute(_INSERT, {"dataset_id": dataset_id, "name_raw": name, "source": SOURCE,
                                         "ror_id": ror, "evidence": evidence})
        stats["datasets"] += 1
        stats["rows"] += len(found)
    return stats


async def primary_papers(conn: Any, accessions: Optional[Sequence[str]] = None) -> List[Tuple[Any, str, str]]:
    sql, params = _PRIMARY_PAPERS, {}
    if accessions is not None:
        sql += " AND d.accession_id = ANY(:accessions)"
        params["accessions"] = list(accessions)
    return [tuple(row) for row in await conn.execute(text(sql + " ORDER BY 2, 3"), params)]


async def refresh(engine: Any, accessions: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """Look up and store the lead-author institutions for *accessions* (all when None).

    The lookups run outside any transaction; only the writes hold one.
    """
    async with engine.connect() as conn:
        papers = await primary_papers(conn, accessions)
    affiliations = collect(papers)
    async with engine.begin() as conn:
        return await store(conn, affiliations)


if __name__ == "__main__":  # pragma: no cover
    import asyncio
    import sys

    async def _main() -> int:
        from db.db import engine
        print(await refresh(engine))
        return 0

    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(_main()))
