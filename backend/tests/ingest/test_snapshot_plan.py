"""The crawler indexes the files of a dataset again only when its OpenNeuro
snapshot changed (ingest/openneuro-crawler/remote/snapshot_plan.py), records
the snapshot only after a complete indexing (remote/openneuro.py), and keeps
the API metadata of the skipped datasets current (crawler_service.py)."""
from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_CRAWLER = ROOT / "ingest" / "openneuro-crawler"
if str(_CRAWLER) not in sys.path:
    sys.path.insert(0, str(_CRAWLER))

pytest.importorskip("boto3")

import crawler_service  # noqa: E402
from remote import openneuro, snapshot_plan  # noqa: E402
from remote import openneuro_metadata as meta  # noqa: E402
from remote.snapshot_plan import Action, Stored, classify  # noqa: E402

NOW = datetime(2026, 11, 27, 12, tzinfo=timezone.utc)
CREATED = NOW - timedelta(days=100)


def _record(hexsha: str = "abc", created: datetime | None = CREATED, **fields) -> dict:
    snapshot = {"tag": "1.0.1", "hexsha": hexsha,
                "created": created.isoformat().replace("+00:00", "Z") if created else None}
    return {"id": "ds000001", "metadata": fields.get("metadata", {}), "latestSnapshot": snapshot}


def _stored(hexsha: str | None = "abc", version: int | None = snapshot_plan.INDEXER_VERSION,
            indexed: datetime | None = NOW - timedelta(days=3), **texts) -> Stored:
    return Stored(hexsha=hexsha, indexer_version=version, files_indexed_at=indexed, texts=texts)


# ── classify ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("record", "unavailable", "stored", "expected"), [
    (_record(), None, _stored(), Action.UNCHANGED),
    (_record(), None, None, Action.NEW),
    (_record(), None, _stored(hexsha=None), Action.UNRECORDED),
    (_record(), None, _stored(indexed=None), Action.UNRECORDED),
    (_record(hexsha="def"), None, _stored(), Action.CHANGED),
    (_record(), None, _stored(version=0), Action.OUTDATED),
    (_record(), None, _stored(version=None), Action.OUTDATED),
    (_record(), None, _stored(indexed=NOW - timedelta(days=31)), Action.STALE),
    # indexed a day after the snapshot was made: S3 may still have had the old files
    (_record(created=NOW - timedelta(days=2)), None, _stored(indexed=NOW - timedelta(days=1)),
     Action.RECENT),
    # indexed three days after it: settled
    (_record(created=NOW - timedelta(days=4)), None, _stored(indexed=NOW - timedelta(days=1)),
     Action.UNCHANGED),
    (_record(created=None), None, _stored(), Action.UNCHANGED),
    (None, "Deleted on OpenNeuro", _stored(), Action.REMOVED),
    (None, None, _stored(), Action.UNKNOWN),
    ({"latestSnapshot": {"tag": "1.0.0"}}, None, None, Action.UNKNOWN),
])
def test_classify(record, unavailable, stored, expected) -> None:
    assert classify(record, unavailable, stored, NOW) is expected


def test_force_indexes_known_datasets_but_not_unknown_or_removed() -> None:
    assert classify(_record(), None, _stored(), NOW, force=True) is Action.FORCED
    assert classify(_record(), None, None, NOW, force=True) is Action.NEW
    assert classify(None, None, _stored(), NOW, force=True) is Action.UNKNOWN
    assert classify(None, "gone", _stored(), NOW, force=True) is Action.REMOVED


def test_max_age_is_configurable() -> None:
    stored = _stored(indexed=NOW - timedelta(days=8))
    assert classify(_record(), None, stored, NOW, max_age=timedelta(days=7)) is Action.STALE
    assert classify(_record(), None, stored, NOW) is Action.UNCHANGED


def test_plan_orders_updates_before_new_datasets_and_limits() -> None:
    records = {"ds1": _record(), "ds2": _record(hexsha="new"), "ds3": _record(),
               "ds4": None, "ds5": _record(hexsha="x")}
    stored = {"ds1": _stored(), "ds2": _stored(), "ds4": _stored(), "ds5": _stored(hexsha=None)}
    plan = snapshot_plan.make_plan(["ds3", "ds2", "ds1", "ds4", "ds5"], records, {}, stored, NOW)

    assert plan.to_index() == ["ds2", "ds3", "ds5"]
    assert plan.to_index(limit=1) == ["ds2"]
    assert plan.counts() == {"changed": 1, "new": 1, "unchanged": 1, "unknown": 1,
                             "unrecorded": 1}


def test_texts_changed_only_for_a_new_value() -> None:
    record = _record(metadata={"dxStatus": "Epilepsy"})
    assert snapshot_plan.texts_changed(_stored(dx_status="Healthy / Control"), record)
    assert not snapshot_plan.texts_changed(_stored(dx_status="Epilepsy"), record)
    # The API leaving a field empty keeps the stored value: no change.
    assert not snapshot_plan.texts_changed(_stored(dx_status="Epilepsy"), _record())
    assert not snapshot_plan.texts_changed(None, record)


# ── the API record ────────────────────────────────────────────────────────────

def test_the_metadata_query_asks_for_the_snapshot_commit() -> None:
    assert "hexsha" in meta.METADATA_SELECTION and "created" in meta.METADATA_SELECTION


def test_snapshot_of() -> None:
    snapshot = meta.snapshot_of(_record(hexsha=" abc "))
    assert snapshot == meta.Snapshot(tag="1.0.1", hexsha="abc", created=CREATED)
    assert meta.snapshot_of({"latestSnapshot": {"hexsha": ""}}) is None
    assert meta.snapshot_of(None) is None


def test_fetch_records_marks_a_failed_batch_unknown(monkeypatch) -> None:
    def fetch_batch(ids):
        if "ds3" in ids:
            raise meta.OpenNeuroAPIError("HTTP 502")
        return {i: _record() for i in ids}, {}

    monkeypatch.setattr(snapshot_plan, "fetch_batch", fetch_batch)
    records, unavailable, failed = snapshot_plan.fetch_records(["ds1", "ds2", "ds3", "ds4"],
                                                               batch_size=2)
    assert failed == {"ds3", "ds4"}
    assert records["ds1"] is not None and records["ds3"] is None
    plan = snapshot_plan.make_plan(records, records, unavailable, {"ds3": _stored()}, NOW)
    assert plan.actions["ds3"] is Action.UNKNOWN
    assert plan.actions["ds4"] is Action.UNKNOWN


# ── crawl cycle ───────────────────────────────────────────────────────────────

class _Session:
    """Records the SQL a step executes."""

    def __init__(self):
        self.statements: list[tuple[str, dict]] = []
        self.committed = False

    async def execute(self, statement, params=None):
        self.statements.append((str(statement), dict(params or {})))
        return types.SimpleNamespace(rowcount=1)

    async def commit(self):
        self.committed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_db(monkeypatch, session: _Session) -> None:
    import db.db

    class _Engine:
        def connect(self):
            return session

    monkeypatch.setattr(db.db, "async_session_maker", lambda: session)
    monkeypatch.setattr(db.db, "engine", _Engine())


def test_a_cycle_indexes_only_what_changed_and_refreshes_the_rest(monkeypatch) -> None:
    session = _Session()
    _patch_db(monkeypatch, session)
    stored = {"ds1": _stored(dx_status="Healthy / Control"), "ds2": _stored(),
              "ds3": _stored(), "ds9": _stored()}
    records = {"ds1": _record(metadata={"dxStatus": "Epilepsy"}), "ds2": _record(hexsha="new"),
               "ds3": _record(), "ds4": _record(), "ds9": None}

    async def load_stored(conn):
        return stored

    monkeypatch.setattr(snapshot_plan, "load_stored", load_stored)
    monkeypatch.setattr(snapshot_plan, "fetch_records",
                        lambda accessions: (records, {"ds9": "Deleted on OpenNeuro"}, set()))
    # ds8 is in the database but not in this cycle (OPENNEURO_ACCESSIONS)
    stored["ds8"] = _stored()
    monkeypatch.setattr(crawler_service, "_load_accessions",
                        lambda known: sorted((set(known) - {"ds8"}) | {"ds4"}))
    monkeypatch.setenv("CRAWL_DELAY_SECONDS", "0")
    crawled: list[tuple[str, dict | None]] = []

    async def crawl_one(accession, delay, record=None):
        crawled.append((accession, record))
        return accession != "ds4"   # ds4 fails

    monkeypatch.setattr(crawler_service, "_crawl_one", crawl_one)
    post: dict = {}

    async def run_post_crawl(accessions, only=None, *, files=None, papers=None):
        post.update(accessions=accessions, files=files, papers=papers)

    import post_crawl
    monkeypatch.setattr(post_crawl, "run_post_crawl", run_post_crawl)

    asyncio.run(crawler_service.run_crawl_cycle())

    assert crawled == [("ds2", records["ds2"]), ("ds4", records["ds4"])]
    assert post == {"accessions": ["ds1", "ds2"], "files": ["ds2"],
                    "papers": ["ds1", "ds2", "ds3", "ds9"]}
    written = {(params.get("accession_id"), params.get("dx_status", params.get("reason")))
               for _, params in session.statements}
    # skipped datasets get the API's metadata; the removed one its mark; the
    # indexed ones are left to the indexing
    assert ("ds1", "Epilepsy") in written
    assert ("ds9", "Deleted on OpenNeuro") in written
    assert not {accession for accession, _ in written} & {"ds2", "ds4"}
    assert session.committed
    assert crawler_service.crawler_state["running"] is False


def test_accessions_from_the_environment_win(monkeypatch) -> None:
    monkeypatch.setenv("OPENNEURO_ACCESSIONS", "ds000002, ds000001")
    assert crawler_service._load_accessions(["ds000009"]) == ["ds000002", "ds000001"]


def test_the_s3_catalogue_includes_the_known_datasets(monkeypatch) -> None:
    monkeypatch.delenv("OPENNEURO_ACCESSIONS", raising=False)
    monkeypatch.delenv("CRAWL_CATALOGUE", raising=False)
    import fetch_accessions
    monkeypatch.setattr(fetch_accessions, "list_all_accessions", lambda: ["ds000002", "ds000003"])
    assert crawler_service._load_accessions(["ds000001", "ds000002"]) == \
        ["ds000001", "ds000002", "ds000003"]


# ── indexing one dataset ──────────────────────────────────────────────────────

def _patch_indexing(monkeypatch, tmp_path, *, failed_downloads: int = 0, n_objects: int = 3):
    monkeypatch.setattr(openneuro, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(openneuro, "_s3_client", lambda: object())
    monkeypatch.setattr(openneuro, "_list_objects", lambda client, accession: [
        {"Key": f"{accession}/f{i}.json", "Size": 1} for i in range(n_objects)])

    def build_mirror(client, accession, objects, mirror_dir):
        (mirror_dir / "dataset_description.json").write_text("{}")
        return failed_downloads

    monkeypatch.setattr(openneuro, "_build_mirror", build_mirror)

    async def run_pipeline(path, skip_validation=False):
        return None

    monkeypatch.setitem(sys.modules, "input_pipeline",
                        types.SimpleNamespace(run_pipeline=run_pipeline))

    async def stamp(root, accession, url):
        return types.SimpleNamespace(id="uuid")

    async def mark_remote(dataset_id):
        return 3

    async def enrich(accession, record=None):
        return record

    recorded: list = []

    async def record_snapshot(accession, record):
        recorded.append((accession, record))

    monkeypatch.setattr(openneuro, "_stamp_dataset", stamp)
    monkeypatch.setattr(openneuro, "_mark_objects_remote", mark_remote)
    monkeypatch.setattr(openneuro, "_enrich_from_openneuro_api", enrich)
    monkeypatch.setattr(openneuro, "_record_snapshot", record_snapshot)
    return recorded


def test_a_complete_indexing_records_the_snapshot_and_removes_the_mirror(monkeypatch, tmp_path):
    recorded = _patch_indexing(monkeypatch, tmp_path)
    record = _record()
    asyncio.run(openneuro.index_openneuro("ds000001", record=record))
    assert recorded == [("ds000001", record)]
    assert not (tmp_path / "ds000001").exists()


def test_missing_metadata_files_leave_the_snapshot_unrecorded(monkeypatch, tmp_path):
    recorded = _patch_indexing(monkeypatch, tmp_path, failed_downloads=1)
    asyncio.run(openneuro.index_openneuro("ds000001", record=_record()))
    assert recorded == []


def test_mirrors_can_be_kept(monkeypatch, tmp_path):
    _patch_indexing(monkeypatch, tmp_path)
    monkeypatch.setenv("CRAWL_KEEP_MIRRORS", "true")
    asyncio.run(openneuro.index_openneuro("ds000001"))
    assert (tmp_path / "ds000001").exists()


def test_a_pipeline_error_records_nothing_and_still_removes_the_mirror(monkeypatch, tmp_path):
    recorded = _patch_indexing(monkeypatch, tmp_path)

    async def broken(path, skip_validation=False):
        raise RuntimeError("pybids")

    monkeypatch.setitem(sys.modules, "input_pipeline", types.SimpleNamespace(run_pipeline=broken))
    with pytest.raises(RuntimeError, match="pybids"):
        asyncio.run(openneuro.index_openneuro("ds000001", record=_record()))
    assert recorded == []
    assert not (tmp_path / "ds000001").exists()


def test_a_dataset_above_the_file_limit_is_deferred(monkeypatch, tmp_path):
    recorded = _patch_indexing(monkeypatch, tmp_path, n_objects=11)
    monkeypatch.setenv("CRAWL_MAX_FILES", "10")
    with pytest.raises(openneuro.DatasetTooLargeError, match="11 files"):
        asyncio.run(openneuro.index_openneuro("ds000001"))
    assert recorded == []
    monkeypatch.setenv("CRAWL_MAX_FILES", "0")
    asyncio.run(openneuro.index_openneuro("ds000001"))
    assert len(recorded) == 1


def test_record_snapshot_writes_the_four_columns() -> None:
    session = _Session()
    snapshot = meta.Snapshot(tag="1.0.1", hexsha="abc", created=CREATED)
    asyncio.run(meta.record_snapshot(session, "ds000001", snapshot, 3, NOW))
    sql, params = session.statements[0]
    assert set(meta.SNAPSHOT_COLUMNS) <= set(sql.replace(",", " ").split())
    assert params == {"tag": "1.0.1", "hexsha": "abc", "at": NOW, "version": 3,
                      "accession_id": "ds000001"}
