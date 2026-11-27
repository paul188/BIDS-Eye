"""
remote/site_places.py
---------------------
Where the sites of bids_dataset_sites are (table bids_site_places):
each raw institution or ethics-board name matched to the Research Organization
Registry, with the city, region and country ROR records for it. The search
reads bids_site_places for questions that name a place ("datasets from Bonn").

Only ROR's own confident matches are kept ("chosen" in its affiliation
matching). Its other candidates are often wrong: 'NIH Combined Neuroscience
IRB' scores 1.0 against a Malaysian institute of the same name, 'Lucas Center'
0.94 against a Belgian MS centre. A name without a confident match is stored
with ror_id '' so it is not asked again.

A site that already carries a ROR id (the institutions OpenAlex gives for a
paper's authors, remote/paper_affiliations.py) is looked up by that id
instead of being matched by name.

Names are matched once; post_crawl.py calls match_new() after the sites of a
crawl cycle are written. ROR's API is public and needs no key.

    python -m remote.site_places               # match every name not matched yet
    python -m remote.site_places --from-json F # import matches saved by an earlier run
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

_ROR_URL = "https://api.ror.org/v2/organizations?affiliation="
_ROR_ID_URL = "https://api.ror.org/v2/organizations/"
_USER_AGENT = "bids-eye-crawler/1.0 (site places)"
# ROR allows 2,000 requests per 5 minutes; stay well below.
_PAUSE_SECONDS = 0.2
_MAX_NAME_CHARS = 500

_NEW_NAMES = text(
    "SELECT s.name_raw, max(s.ror_id) FROM bids_dataset_sites s "
    "WHERE NOT EXISTS (SELECT 1 FROM bids_site_places p WHERE p.name_raw = s.name_raw) "
    "GROUP BY s.name_raw ORDER BY 1"
)
_INSERT = text(
    "INSERT INTO bids_site_places "
    "(name_raw, ror_id, institution, city, region, country, country_code, lat, lng) "
    "VALUES (:name_raw, :ror_id, :institution, :city, :region, :country, :country_code, :lat, :lng) "
    "ON CONFLICT (name_raw, ror_id) DO NOTHING"
)


def ror_matches(name: str, timeout: float = 30.0, retries: int = 3) -> Optional[List[Dict[str, Any]]]:
    """ROR's confident matches for *name* (its 'chosen' items); None when ROR cannot be reached."""
    query = name.replace("_", " ").strip()[:_MAX_NAME_CHARS]
    if not query:
        return []
    request = urllib.request.Request(_ROR_URL + urllib.parse.quote(query),
                                     headers={"User-Agent": _USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                items = json.load(response).get("items", [])
            return [item for item in items if item.get("chosen")]
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            log.warning("ROR lookup for %r failed (%s), attempt %d", query[:60], exc, attempt + 1)
            time.sleep(5 * (attempt + 1))
    return None


def ror_by_id(ror_id: str, timeout: float = 30.0, retries: int = 3) -> Optional[List[Dict[str, Any]]]:
    """The organisation with *ror_id*, as a one-item match list; None when ROR cannot be reached."""
    request = urllib.request.Request(_ROR_ID_URL + ror_id.rstrip("/").rsplit("/", 1)[-1],
                                     headers={"User-Agent": _USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return [{"chosen": True, "organization": json.load(response)}]
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return []
            log.warning("ROR id %s: HTTP %s, attempt %d", ror_id, exc.code, attempt + 1)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            log.warning("ROR id %s failed (%s), attempt %d", ror_id, exc, attempt + 1)
        time.sleep(5 * (attempt + 1))
    return None


def place_rows(name: str, matches: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """bids_site_places rows for *name*: one per matched organisation, or one empty row."""
    rows = []
    for item in matches:
        organisation = item.get("organization") or {}
        display = next((n.get("value") for n in organisation.get("names", [])
                        if "ror_display" in (n.get("types") or [])), None)
        location = ((organisation.get("locations") or [{}])[0] or {}).get("geonames_details") or {}
        rows.append({
            "name_raw": name, "ror_id": organisation.get("id") or "", "institution": display,
            "city": location.get("name"), "region": location.get("country_subdivision_name"),
            "country": location.get("country_name"), "country_code": location.get("country_code"),
            "lat": location.get("lat"), "lng": location.get("lng"),
        })
    rows = [r for r in rows if r["ror_id"]]
    if not rows:
        rows = [{"name_raw": name, "ror_id": "", "institution": None, "city": None, "region": None,
                 "country": None, "country_code": None, "lat": None, "lng": None}]
    return rows


async def new_names(conn: Any) -> List[Tuple[str, str]]:
    """(name, ROR id or '') of the site names without a row yet."""
    return [(row[0], row[1] or "") for row in await conn.execute(_NEW_NAMES)]


async def store(conn: Any, name: str, matches: Iterable[Dict[str, Any]]) -> int:
    rows = place_rows(name, matches)
    await conn.execute(_INSERT, rows)
    return len(rows)


async def match_new(conn: Any, limit: Optional[int] = None, lookup=ror_matches,
                    lookup_id=ror_by_id, pause: float = _PAUSE_SECONDS) -> Dict[str, int]:
    """Match the site names that have no row yet. Names ROR could not be asked about stay unmatched."""
    names = await new_names(conn)
    if limit is not None:
        names = names[:limit]
    stats = {"names": len(names), "matched": 0, "no_match": 0, "unreachable": 0}
    for name, ror_id in names:
        matches = lookup_id(ror_id) if ror_id else lookup(name)
        if matches is None:
            stats["unreachable"] += 1
            continue
        await store(conn, name, matches)
        stats["matched" if matches else "no_match"] += 1
        if pause:
            time.sleep(pause)
    return stats


async def import_json(conn: Any, cache: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, int]:
    """Store matches saved as {name: [chosen ROR item, ...]} by an earlier run."""
    stats = {"names": 0, "rows": 0}
    for name, matches in cache.items():
        stats["names"] += 1
        stats["rows"] += await store(conn, name, matches)
    return stats


if __name__ == "__main__":  # pragma: no cover
    import asyncio
    import sys

    async def _main() -> int:
        from db.db import engine
        async with engine.begin() as conn:
            if "--from-json" in sys.argv:
                path = sys.argv[sys.argv.index("--from-json") + 1]
                with open(path, encoding="utf-8") as handle:
                    print(await import_json(conn, json.load(handle)))
            else:
                print(await match_new(conn))
        return 0

    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(_main()))
