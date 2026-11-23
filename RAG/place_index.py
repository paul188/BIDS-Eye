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
'Köln' and 'München'; 'USA', 'UK' and 'Holland' are countries.

One name can be several places. 'Mexico' is the country and (as 'México') one
of its states; 'Washington' is a city and a state; 'Georgia' a country and a
US state. readings() gives every place of that name, except one that lies
inside another of them: the state of México is in Mexico, so 'Mexico' is the
country (before, the state won and "mexico" found one dataset of sixteen).
'Washington' and 'Georgia' stay both.

A place name is also a word in other place names: 'Mexico' in 'New Mexico',
'York' in 'New York', 'Saxony' in 'Lower Saxony'. foreign_names() lists the
stored ones that lie elsewhere, so the text search for the place can leave
them out ("mexico" matched nine University of New Mexico datasets).

The backend reads the values from the database and hands them to set_places()
at startup and again when they changed
(backend/services/database_stats/database_stats.py). Until then the index is empty and every place falls back to a text
search (LLM_preprocessor/preprocess/augment.py, _location_lines).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Sequence, Tuple

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


# A site's (city, region, country) as stored; '' where ROR records none.
PlaceRow = Tuple[str, str, str]
# One place a name can mean: (column, stored value).
Reading = Tuple[str, str]


class PlaceIndex:
    """Stored city, region and country values, searchable by a user's place name."""

    def __init__(self, places: Iterable[Tuple[str, str]], rows: Iterable[PlaceRow] = ()):
        """*places*: (column, stored value) pairs, column one of COLUMNS or 'institution'.
        *rows*: the stored (city, region, country) combinations. They say which
        place lies in which; without them every reading of a name is kept and
        no other place name is left out of a text search."""
        places = list(places)
        self._rows: List[Dict[str, str]] = [dict(zip(COLUMNS, row)) for row in rows]
        places.extend((column, row[column]) for row in self._rows for column in COLUMNS)
        # ROR's names for the matched institutions: 'University of Bonn' is
        # found through its ethics committee even where no scan names it.
        self._institutions = InstitutionIndex(v for c, v in places if c == "institution")
        # Normalised name → column → the stored spellings ('Zurich', 'Zürich').
        self._by_name: Dict[str, Dict[str, List[str]]] = {}
        for column, value in places:
            if column in COLUMNS and isinstance(value, str) and value.strip():
                values = self._by_name.setdefault(normalize(value), {}).setdefault(column, [])
                if value not in values:
                    values.append(value)
        # Short forms last, so a place stored under the short name itself wins.
        for column, value in places:
            match = _QUALIFIED_RE.match(value or "") if column == "city" else None
            if match:
                self._by_name.setdefault(normalize(match.group("short")), {}).setdefault(column, [value])
        self._other: Dict[str, List[str]] = {}
        for group in _OTHER_NAMES:
            keys = [normalize(name) for name in group]
            for key in keys:
                self._other.setdefault(key, []).extend(k for k in keys if k != key)

    def __len__(self) -> int:
        return len(self._by_name)

    def _inside(self, reading: Reading, containers: Sequence[Reading]) -> bool:
        """Whether every stored site of *reading* is also in one of *containers*.
        False when no stored row has the reading: nothing is known about it."""
        column, value = reading
        rows = [row for row in self._rows if row[column] == value]
        return bool(rows) and all(any(row[c] == v for c, v in containers) for row in rows)

    def readings(self, term: str) -> List[Reading]:
        """Every (column, stored value) *term* names, most specific first; []
        when no stored place has that name. A place inside a larger one of the
        same name is left out: the larger one finds its sites too."""
        key = normalize(term)
        if not key:
            return []
        for candidate in [key] + self._other.get(key, []):
            stored = self._by_name.get(candidate)
            if not stored:
                continue
            found = [(column, value) for column in COLUMNS for value in stored.get(column, [])]
            return [reading for reading in found
                    if not self._inside(reading, [other for other in found
                                                  if COLUMNS.index(other[0]) > COLUMNS.index(reading[0])])]
        return []

    def foreign_names(self, term: str) -> List[str]:
        """Stored place names that contain *term* as a word and lie outside
        every place *term* names: 'New Mexico' for 'Mexico', not 'Mexico City'."""
        key = normalize(term)
        meant = self.readings(term)
        if not key or not meant:
            return []
        word = re.compile(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])")
        names: List[str] = []
        for name, stored in self._by_name.items():
            if name == key or not word.search(name):
                continue
            for column in COLUMNS:
                for value in stored.get(column, []):
                    known = any(row[column] == value for row in self._rows)
                    if known and not self._inside((column, value), meant) and value not in names:
                        names.append(value)
        return sorted(names)

    def institutions(self, term: str) -> List[str]:
        """ROR institution names stored in bids_site_places for an institution the user named."""
        return [name for name, _ in self._institutions.resolve(term)]


_index = PlaceIndex(())


def set_places(places: Iterable[Tuple[str, str]], rows: Iterable[PlaceRow] = ()) -> PlaceIndex:
    """Replace the shared index (the backend's database_stats, tests)."""
    global _index
    _index = PlaceIndex(places, rows)
    return _index


def get_index() -> PlaceIndex:
    return _index

