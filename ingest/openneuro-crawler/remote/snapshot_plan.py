"""
remote/snapshot_plan.py
-----------------------
Which datasets a crawl cycle indexes again.

Indexing the files of a dataset (S3 listing, local mirror, the BIDS pipeline)
is the expensive part of a crawl, and most datasets do not change between two
cycles: on 2026-11-27, 12 of 1,718 indexed datasets had a new OpenNeuro
snapshot in the last 30 days. So the crawler records which snapshot the files
were indexed from (bids_datasets.snapshot_hexsha, written by
openneuro_metadata.record_snapshot) and compares it with the API's latest
snapshot at the start of each cycle.

A dataset is indexed again when
  - it is not in the database yet                            (NEW)
  - it was indexed before this was recorded                  (UNRECORDED)
  - its latest snapshot is another commit                    (CHANGED)
  - the indexing code changed since (INDEXER_VERSION)        (OUTDATED)
  - its files were indexed more than CRAWL_MAX_AGE_DAYS ago  (STALE)
  - they were indexed less than SNAPSHOT_SETTLE_HOURS after the snapshot was
    made: OpenNeuro copies a new snapshot to S3 some time after creating it,
    so the first indexing may have read the old files              (RECENT)

Everything else is UNCHANGED, or REMOVED (the API says the dataset is gone),
or UNKNOWN (the API gave no answer this time: tried again next cycle, never
indexed blind, so an API outage does not turn into a full crawl).

The API metadata (README, uploader fields, downloads, deprecation) changes
without a new snapshot; the crawler writes it for every dataset from the same
responses, indexed or not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from sqlalchemy import text

from .openneuro_metadata import (
    DEFAULT_BATCH_SIZE,
    OpenNeuroAPIError,
    extract_fields,
    fetch_batch,
    snapshot_of,
)

log = logging.getLogger(__name__)

# Raise by one whenever a change to the indexing (remote/openneuro.py, the
# BIDS-SQL pipeline, what _stamp_dataset writes) should reach datasets whose
# snapshot did not change: the next cycles then index every dataset again.
INDEXER_VERSION = 1

DEFAULT_MAX_AGE = timedelta(days=30)
SNAPSHOT_SETTLE = timedelta(hours=48)

# The API fields that the README profiles and descriptions read: when one
# changes, those post-crawl steps run for the dataset although its files did not.
TEXT_FIELDS: Tuple[str, ...] = ("readme_text", "dx_status", "study_domain", "study_design")


class Action(str, Enum):
    NEW = "new"
    UNRECORDED = "unrecorded"
    CHANGED = "changed"
    OUTDATED = "outdated"
    STALE = "stale"
    RECENT = "recent"
    FORCED = "forced"
    UNCHANGED = "unchanged"
    REMOVED = "removed"
    UNKNOWN = "unknown"


# Indexed in this order, so that updates of known datasets are not queued
# behind a long run of new ones.
INDEX_ORDER: Tuple[Action, ...] = (
    Action.CHANGED, Action.RECENT, Action.FORCED, Action.OUTDATED,
    Action.NEW, Action.UNRECORDED, Action.STALE,
)


@dataclass(frozen=True)
class Stored:
    """What the database holds about one indexed dataset."""

    hexsha: Optional[str]
    indexer_version: Optional[int]
    files_indexed_at: Optional[datetime]
    texts: Mapping[str, Optional[str]] = field(default_factory=dict)


def classify(record: Optional[dict], unavailable: Optional[str], stored: Optional[Stored],
             now: datetime, *, version: int = INDEXER_VERSION,
             max_age: timedelta = DEFAULT_MAX_AGE, settle: timedelta = SNAPSHOT_SETTLE,
             force: bool = False) -> Action:
    """What to do with one dataset this cycle (see the module docstring)."""
    if unavailable:
        return Action.REMOVED
    snapshot = snapshot_of(record)
    if snapshot is None:
        return Action.UNKNOWN
    if stored is None:
        return Action.NEW
    if force:
        return Action.FORCED
    if stored.hexsha is None or stored.files_indexed_at is None:
        return Action.UNRECORDED
    if stored.hexsha != snapshot.hexsha:
        return Action.CHANGED
    if (stored.indexer_version or 0) < version:
        return Action.OUTDATED
    if now - stored.files_indexed_at > max_age:
        return Action.STALE
    if snapshot.created is not None and stored.files_indexed_at - snapshot.created < settle:
        return Action.RECENT
    return Action.UNCHANGED


def texts_changed(stored: Optional[Stored], record: Optional[dict]) -> bool:
    """Whether the API now has another README or uploader field than the database.

    A field the API leaves empty does not count: apply_fields keeps the stored
    value then.
    """
    if stored is None or record is None:
        return False
    fields = extract_fields(record)
    return any(fields.get(name) is not None and fields.get(name) != stored.texts.get(name)
               for name in TEXT_FIELDS)


@dataclass
class Plan:
    """The decisions of one cycle and the API records they were made from."""

    actions: Dict[str, Action]
    records: Dict[str, Optional[dict]]
    unavailable: Dict[str, str]

    def to_index(self, limit: int = 0) -> List[str]:
        """The datasets to index, in INDEX_ORDER; at most *limit* when it is > 0."""
        rank = {action: position for position, action in enumerate(INDEX_ORDER)}
        chosen = sorted((accession for accession, action in self.actions.items() if action in rank),
                        key=lambda accession: (rank[self.actions[accession]], accession))
        return chosen[:limit] if limit > 0 else chosen

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for action in self.actions.values():
            counts[action.value] = counts.get(action.value, 0) + 1
        return dict(sorted(counts.items()))


def make_plan(accessions: Iterable[str], records: Mapping[str, Optional[dict]],
              unavailable: Mapping[str, str], stored: Mapping[str, Stored], now: datetime,
              **options: Any) -> Plan:
    """classify() for every accession; *options* go to classify."""
    actions: Dict[str, Action] = {}
    for accession in dict.fromkeys(accessions):
        actions[accession] = classify(records.get(accession), unavailable.get(accession),
                                      stored.get(accession), now, **options)
    return Plan(actions=actions, records=dict(records), unavailable=dict(unavailable))


def fetch_records(accessions: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE
                  ) -> Tuple[Dict[str, Optional[dict]], Dict[str, str], Set[str]]:
    """The API records of *accessions*, *batch_size* per request (blocking).

    Returns (records, unavailable, failed): *failed* are the accessions whose
    request failed; their record is None and they count as UNKNOWN.
    """
    records: Dict[str, Optional[dict]] = {}
    unavailable: Dict[str, str] = {}
    failed: Set[str] = set()
    for start in range(0, len(accessions), batch_size):
        batch = list(accessions[start:start + batch_size])
        try:
            batch_records, batch_unavailable = fetch_batch(batch)
        except OpenNeuroAPIError as exc:
            log.warning("[plan] OpenNeuro API failed for %s..%s: %s", batch[0], batch[-1], exc)
            failed.update(batch)
            records.update(dict.fromkeys(batch))
            continue
        records.update(batch_records)
        unavailable.update(batch_unavailable)
    return records, unavailable, failed


_STORED_SQL = text(
    "SELECT DISTINCT ON (accession_id) accession_id, snapshot_hexsha, indexer_version, "
    "files_indexed_at, " + ", ".join(TEXT_FIELDS) + " "
    "FROM bids_datasets WHERE accession_id IS NOT NULL "
    "ORDER BY accession_id, files_indexed_at DESC NULLS LAST"
)


async def load_stored(conn: Any) -> Dict[str, Stored]:
    """Stored() of every dataset in the database that has an accession id."""
    stored: Dict[str, Stored] = {}
    for row in (await conn.execute(_STORED_SQL)).mappings():
        stored[row["accession_id"]] = Stored(
            hexsha=row["snapshot_hexsha"], indexer_version=row["indexer_version"],
            files_indexed_at=row["files_indexed_at"],
            texts={name: row[name] for name in TEXT_FIELDS})
    return stored
