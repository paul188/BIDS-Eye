"""
RAG/place_index.py
------------------
Resolve a place named in a question ("Bonn", "Germany", "Bavaria") to the
city, region or country values stored in bids_site_places, where the sites
of bids_dataset_sites (scanner institutions, ethics boards) were matched to
the Research Organization Registry (crawlers/openneuro-crawler/remote/site_places.py).

Places are matched as whole normalised names (accents and German umlaut
spellings folded, as in institution_index), plus a small table of other
names for the same place: ROR writes 'Cologne' and 'Munich', users write
'Köln' and 'München'; 'USA', 'UK' and 'Holland' are countries. A city is
preferred over a region and a region over a country of the same name
('Washington' the city, not the state).

The values are loaded from the database at backend startup (refresh_from_db).
Without the table the index is empty and every place falls back to a text
search (LLM_preprocessor/preprocess/augment.py, _location_lines).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from institution_index import InstitutionIndex, normalize

log = logging.getLogger(__name__)

# bids_site_places columns, most specific first.
COLUMNS = ("city", "region", "country")

# 'Frankfurt am Main', 'Freiburg im Breisgau', 'Newcastle upon Tyne': users
# write the first part, so a stored name is also found by it.
_QUALIFIED_RE = re.compile(r"^(?P<short>.+?) (?:am|im|an der|upon|on the|on) \S.*$", re.IGNORECASE)

# Other names users write for a place, grouped with the name ROR uses.
_OTHER_NAMES: Tuple[Tuple[str, ...], ...] = (
    ("United States", "USA", "US", "U.S.", "U.S.A.", "America", "United States of America"),
    ("United Kingdom", "UK", "U.K.", "Great Britain", "Britain"),
    ("Netherlands", "The Netherlands", "Holland"),
    ("South Korea", "Korea", "Republic of Korea"),
    ("Czechia", "Czech Republic"),
    ("Cologne", "Köln", "Koeln"),
    ("Munich", "München", "Muenchen"),
    ("Nuremberg", "Nürnberg"),
    ("Vienna", "Wien"),
    ("Zurich", "Zürich"),
    ("Geneva", "Genève", "Genf"),
    ("Rome", "Roma"),
    ("Milan", "Milano"),
    ("Florence", "Firenze"),
    ("Turin", "Torino"),
    ("Naples", "Napoli"),
    ("Prague", "Praha"),
    ("Warsaw", "Warszawa"),
    ("Krakow", "Kraków", "Cracow"),
    ("Lisbon", "Lisboa"),
    ("Copenhagen", "København"),
    ("Gothenburg", "Göteborg"),
    ("Brussels", "Bruxelles", "Brussel"),
    ("Leuven", "Louvain"),
    ("The Hague", "Den Haag"),
    ("Beijing", "Peking"),
    ("St Louis", "Saint Louis"),
    ("Montreal", "Montréal"),
    ("Quebec", "Québec"),
    ("Bavaria", "Bayern"),
    ("North Rhine-Westphalia", "NRW", "Nordrhein-Westfalen"),
    ("Lower Saxony", "Niedersachsen"),
    ("Baden-Württemberg", "Baden-Wuerttemberg"),
    ("Saxony", "Sachsen"),
    ("Hesse", "Hessen"),
)


class PlaceIndex:
    """Stored city, region and country values, searchable by a user's place name."""

    def __init__(self, places: Iterable[Tuple[str, str]]):
        """*places*: (column, stored value) pairs, column one of COLUMNS or 'institution'."""
        places = list(places)
        # ROR's names for the matched institutions: 'University of Bonn' is
        # found through its ethics committee even where no scan names it.
        self._institutions = InstitutionIndex(v for c, v in places if c == "institution")
        self._by_name: Dict[str, Dict[str, str]] = {}
        for column, value in places:
            if column in COLUMNS and isinstance(value, str) and value.strip():
                self._by_name.setdefault(normalize(value), {}).setdefault(column, value)
        # Short forms last, so a place stored under the short name itself wins.
        for column, value in places:
            match = _QUALIFIED_RE.match(value or "") if column == "city" else None
            if match:
                self._by_name.setdefault(normalize(match.group("short")), {}).setdefault(column, value)
        self._other: Dict[str, List[str]] = {}
        for group in _OTHER_NAMES:
            keys = [normalize(name) for name in group]
            for key in keys:
                self._other.setdefault(key, []).extend(k for k in keys if k != key)

    def __len__(self) -> int:
        return len(self._by_name)

    def resolve(self, term: str) -> Optional[Tuple[str, str]]:
        """(column, stored value) for *term*, or None when no stored place has that name."""
        key = normalize(term)
        if not key:
            return None
        for candidate in [key] + self._other.get(key, []):
            stored = self._by_name.get(candidate)
            if stored:
                column = next(c for c in COLUMNS if c in stored)
                return column, stored[column]
        return None

    def institutions(self, term: str) -> List[str]:
        """ROR institution names stored in bids_site_places for an institution the user named."""
        return [name for name, _ in self._institutions.resolve(term)]


_index = PlaceIndex(())


def set_places(places: Iterable[Tuple[str, str]]) -> PlaceIndex:
    """Replace the shared index (startup, tests)."""
    global _index
    _index = PlaceIndex(places)
    return _index


def get_index() -> PlaceIndex:
    return _index


async def refresh_from_db(engine: Any) -> int:
    """Load the distinct places from bids_site_places. Returns how many names.

    On any error (table missing, database down) the index stays as it was, so
    place names fall back to a text search.
    """
    import asyncio

    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError

    try:
        async with engine.connect() as connection:
            rows = await connection.execute(text(
                "SELECT DISTINCT 'city', city FROM bids_site_places WHERE city <> '' "
                "UNION SELECT DISTINCT 'region', region FROM bids_site_places WHERE region <> '' "
                "UNION SELECT DISTINCT 'country', country FROM bids_site_places WHERE country <> '' "
                "UNION SELECT DISTINCT 'institution', institution FROM bids_site_places "
                "WHERE institution <> ''"
            ))
            places = [(row[0], row[1]) for row in rows]
    # As in institution_index.refresh_from_db; a missing table is a
    # ProgrammingError (SQLAlchemyError).
    except (SQLAlchemyError, OSError, asyncio.TimeoutError) as exc:
        log.warning("Could not load site places (%s) — place names are searched as text", exc)
        return 0
    index = set_places(places)
    log.info("Loaded %d place names for query matching", len(index))
    return len(index)
