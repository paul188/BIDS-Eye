"""
services/catalogue/catalogue.py
-------------------------------
What the catalogue database holds, counted.

The structure is fixed: the tables and columns the search reads are declared
in services/schema/tables.py, and services/schema/check.py stops startup when
the database lacks one of them. What changes with every crawl is how full
they are, and Catalogue keeps that in three parts:

  coverage   ColumnCoverage: per optional bids_datasets column
             (constants.OPTIONAL_DATASET_COLUMNS), how many datasets fill it.
             Only the counts: which columns the search then uses is decided
             in services/translate/optional_columns.py.
  summaries  Summaries: whether bids_dataset_facets covers every dataset that
             has files and bids_datasets.n_subjects is filled. The executed
             SQL then reads them instead of bids_objects
             (sql_rewriter._use_summaries). A crawl adds datasets before
             post_crawl.py summarises them, so this can switch off and on.
  values     ValueCounts: per coded field, each value that occurs and how many
             datasets have it. A value is known when one dataset has it.
               datatype, suffix, task         from bids_dataset_facets
               diagnosis, sex, handedness     from bids_participants
             value_check.py checks the datatype, suffix and task values in
             generated SQL; preprocessing drops vocabulary codes no dataset
             has (the DB MISS warning to the SQL model). The counts grow with
             the number of distinct codes, not with the datasets: about 5,500
             values on 2026-11-21 (3,389 suffixes, 1,709 tasks, 199
             diagnoses). Free text (dx_status) and the optional participant
             columns are left out on purpose: they would grow with every
             dataset.

The backend uses one shared instance, catalogue. start() checks the schema
and reads all three at startup; refresh_if_stale() reads them again before a
search once they are REFRESH_SECONDS old. Each part is replaced whole, so a
search never sees half of a reload.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from sqlalchemy import ColumnElement, Text, distinct, exists, func, literal, select, union_all
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession
from sqlalchemy.sql.schema import Column

from constants import OPTIONAL_DATASET_COLUMNS
from services.schema.check import check_schema
from services.schema.tables import dataset_facets, datasets, objects, participants

log = logging.getLogger(__name__)


# ── The three parts ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ColumnCoverage:
    """How many of the datasets fill each optional bids_datasets column."""

    total: int = 0
    filled: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def full(cls, columns: Collection[str]) -> ColumnCoverage:
        """Coverage in which exactly *columns* are filled in every dataset (tests)."""
        unknown = set(columns) - set(OPTIONAL_DATASET_COLUMNS)
        if unknown:
            raise ValueError(f"not optional dataset columns: {sorted(unknown)}")
        return cls(total=1, filled=dict.fromkeys(columns, 1))


@dataclass(frozen=True)
class Summaries:
    """Whether the per-dataset summaries can stand in for bids_objects."""

    # bids_dataset_facets has rows for every dataset that has files.
    facets: bool = False
    # bids_datasets.n_subjects is filled.
    subject_counts: bool = False


@dataclass(frozen=True)
class ValueCounts:
    """Per coded field, each value and the number of datasets that have it.

    A field is missing until its table was read.
    """

    counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def loaded(self, field: str) -> bool:
        """True when the values of *field* were read."""
        return field in self.counts

    def known_values(self, field: str) -> Collection[str]:
        """The values of *field* that occur in some dataset; empty when not read."""
        return self.counts.get(field, {}).keys()

    def dataset_count(self, field: str, value: str) -> int:
        """How many datasets have *value* in *field*; 0 when unknown or not read."""
        return self.counts.get(field, {}).get(value, 0)


# ── The catalogue ─────────────────────────────────────────────────────────────

class Catalogue:
    """Coverage, summaries and coded values of the catalogue database."""

    FACET_COLUMNS = (dataset_facets.c.datatype, dataset_facets.c.suffix, dataset_facets.c.task)
    PARTICIPANT_COLUMNS = (participants.c.diagnosis, participants.c.sex, participants.c.handedness)
    FACET_FIELDS = tuple(column.name for column in FACET_COLUMNS)
    PARTICIPANT_FIELDS = tuple(column.name for column in PARTICIPANT_COLUMNS)
    FIELDS = FACET_FIELDS + PARTICIPANT_FIELDS
    # post_crawl.py refreshes the summaries after each crawl, so a value or
    # dataset added by a crawl is missed for at most 10 minutes.
    REFRESH_SECONDS = 600

    def __init__(self) -> None:
        self.coverage = ColumnCoverage()
        self.summaries = Summaries()
        self.values = ValueCounts()
        # time.monotonic() of the last load(), whether it succeeded or not.
        self._last_attempt: float | None = None
        # True while refresh_if_stale() is reading.
        self._refreshing = False

    def clear(self) -> None:
        """Forget everything read: no optional column, no summaries, no values."""
        self.coverage = ColumnCoverage()
        self.summaries = Summaries()
        self.values = ValueCounts()

    async def start(self, engine: AsyncEngine) -> None:
        """At startup: check_schema(), then the first load()."""
        async with engine.connect() as connection:
            await check_schema(connection)
            await self.load(connection)

    async def refresh_if_stale(self, connection: AsyncConnection | AsyncSession) -> None:
        """load() when the last load is REFRESH_SECONDS old.

        One search reloads at a time. A search that arrives meanwhile does not
        wait and does not read again: it uses what the catalogue holds.
        """
        if self._refreshing:
            return
        if (self._last_attempt is not None
                and time.monotonic() - self._last_attempt < self.REFRESH_SECONDS):
            return
        self._refreshing = True
        try:
            await self.load(connection)
        finally:
            self._refreshing = False

    async def load(self, connection: AsyncConnection | AsyncSession) -> None:
        """Read coverage, summaries and values.

        A part whose read fails keeps what it held: these only decide what the
        search may use, so a failed read must not fail startup or a search.
        The facet values are read only while the facets table is complete.
        """
        coverage = await self._read_coverage(connection)
        if coverage is not None:
            self.coverage = coverage
        summaries = await self._read_summaries(connection)
        if summaries is not None:
            if summaries != self.summaries:
                log.info("Dataset summaries: facets table %s, stored subject count %s",
                         "used" if summaries.facets else "not used",
                         "used" if summaries.subject_counts else "not used")
            self.summaries = summaries
        self.values = await self._read_values(connection)
        # Set only after the reads succeeded or failed with a database error. A
        # bug (any other exception) skips this line, so every search fails until
        # it is fixed instead of the bug going quiet for REFRESH_SECONDS.
        self._last_attempt = time.monotonic()

    # ── Reading ───────────────────────────────────────────────────────────────
    # Each read runs in a savepoint, so a failure leaves the caller's
    # transaction usable.

    async def _read_coverage(self, connection: AsyncConnection | AsyncSession) -> ColumnCoverage | None:
        columns = [datasets.c[name] for name in OPTIONAL_DATASET_COLUMNS]
        statement = select(func.count().label("total"),
                           *(func.count().filter(_filled(column)).label(column.name)
                             for column in columns))
        try:
            async with connection.begin_nested():
                row = (await connection.execute(statement)).mappings().one()
        except SQLAlchemyError as exc:
            log.warning("Counting the filled dataset columns failed (%s); keeping the old counts", exc)
            return None
        return ColumnCoverage(total=int(row["total"] or 0),
                              filled={column.name: int(row[column.name] or 0) for column in columns})

    async def _read_summaries(self, connection: AsyncConnection | AsyncSession) -> Summaries | None:
        subject_counts = select(func.count(datasets.c.n_subjects))
        uncovered = select(func.count()).select_from(datasets).where(
            ~exists().where(dataset_facets.c.dataset_id == datasets.c.id),
            exists().where(objects.c.dataset_id == datasets.c.id))
        try:
            async with connection.begin_nested():
                filled = (await connection.execute(subject_counts)).scalar_one()
                missed = (await connection.execute(uncovered)).scalar_one()
        except SQLAlchemyError as exc:
            log.warning("Checking the dataset summaries failed (%s); keeping the old state", exc)
            return None
        if missed:
            log.warning("%s misses %d datasets that have files: not used until "
                        "remote/dataset_facets.py --refresh-all fills it.",
                        dataset_facets.name, missed)
        return Summaries(facets=not missed, subject_counts=bool(filled))

    async def _read_values(self, connection: AsyncConnection | AsyncSession) -> ValueCounts:
        groups = [self.PARTICIPANT_COLUMNS]
        counts = dict(self.values.counts)
        if self.summaries.facets:
            groups.insert(0, self.FACET_COLUMNS)
        else:
            for name in self.FACET_FIELDS:
                counts.pop(name, None)
        for columns in groups:
            read = await self._read_counts(connection, columns)
            if read is not None:
                counts.update(read)
        log.info("Catalogue values: %s",
                 ", ".join(f"{len(counts[name])} {name}" for name in self.FIELDS if name in counts)
                 or "none")
        return ValueCounts(counts)

    async def _read_counts(self, connection: AsyncConnection | AsyncSession,
                           columns: tuple[Column, ...]) -> dict[str, dict[str, int]] | None:
        """The values of *columns*, all of one table, and how many datasets have each."""
        table = columns[0].table
        statement = union_all(*(
            select(literal(column.name, Text), column, func.count(distinct(table.c.dataset_id)))
            .where(column.is_not(None)).group_by(column)
            for column in columns))
        try:
            async with connection.begin_nested():
                rows = (await connection.execute(statement)).all()
        except SQLAlchemyError as exc:
            note = "keeping the old values" if self.values.loaded(columns[0].name) else "not checked for now"
            log.warning("Reading %s from %s failed (%s); %s",
                        ", ".join(column.name for column in columns), table.name, exc, note)
            return None
        counts: dict[str, dict[str, int]] = {column.name: {} for column in columns}
        for name, value, datasets_with_it in rows:
            counts[name][value] = int(datasets_with_it)
        return counts


def _filled(column: Column) -> ColumnElement[bool]:
    """True for a row whose *column* holds something: not NULL, not blank, not an empty array."""
    if isinstance(column.type, ARRAY):
        return func.cardinality(column) > 0
    if isinstance(column.type, Text):
        return func.nullif(func.btrim(column), "").is_not(None)
    return column.is_not(None)


# The instance the backend uses: main.py starts it, search.py refreshes it.
catalogue = Catalogue()
