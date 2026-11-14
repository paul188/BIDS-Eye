"""Generated dataset descriptions from paper abstracts (services/paper_summary.py),
how the API attaches them, and the crawler's post-crawl step."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

import pytest

from services import paper_summary as ps

ABSTRACT = (
    "Parkinson's disease (PD) affects motor and cognitive function. We acquired "
    "resting-state fMRI and T1-weighted MRI in 42 patients with idiopathic PD and "
    "23 age-matched healthy controls, scanned ON and OFF dopaminergic medication at 3T. "
    "Functional connectivity of the basal ganglia was reduced OFF medication."
)
SOURCE = ps.SummarySource(
    dataset_name="PD rest ON/OFF", accession="ds004392", paper_key="doi:10.1/x",
    title="Dopamine and resting networks in Parkinson's disease", abstract=ABSTRACT)


def test_prompt_names_the_dataset_and_asks_for_data_not_findings():
    prompt = ps.build_prompt(SOURCE)
    assert "ds004392" in prompt and ABSTRACT in prompt
    assert "not the findings" in prompt and "at most 35 words" in prompt


def test_parse_answer_reads_json_and_tidies_the_sentence():
    assert ps.parse_answer('{"summary": "  \\"Resting-state fMRI from 65 adults.\\" "}') == \
        "Resting-state fMRI from 65 adults."
    assert ps.parse_answer('```json\n{"summary": "A b c"}\n```') == "A b c"
    assert ps.parse_answer("not json") is None
    assert ps.parse_answer('{"summary": 3}') is None


def test_check_accepts_a_faithful_paraphrase():
    summary = ("Resting-state fMRI and T1-weighted MRI from 42 people with Parkinson's disease "
               "and 23 healthy controls, each scanned on and off dopaminergic medication at 3T.")
    assert ps.check(summary, SOURCE) is None


def test_check_rejects_invented_numbers():
    summary = "Resting-state fMRI from 60 people with Parkinson's disease and healthy controls at 3T."
    assert "numbers not in the source" in ps.check(summary, SOURCE)


def test_check_rejects_copied_phrases_and_bad_lengths():
    copied = ("Data: resting-state fMRI and T1-weighted MRI in 42 patients with idiopathic PD "
              "and 23 controls.")
    assert "copies" in ps.check(copied, SOURCE)
    assert "length" in ps.check("fMRI.", SOURCE)
    assert ps.check("word " * 60, SOURCE) is not None


def test_paper_url_prefers_doi():
    assert ps.paper_url("10.1/x", "123", None, None) == "https://doi.org/10.1/x"
    assert ps.paper_url(None, "123", None, None) == "https://pubmed.ncbi.nlm.nih.gov/123/"
    assert ps.paper_url(None, None, None, None) is None


def test_up_to_date_needs_same_paper_abstract_and_prompt():
    meta = {"paper_key": SOURCE.paper_key, "abstract_sha": SOURCE.abstract_sha,
            "prompt_version": ps.PROMPT_VERSION}
    assert ps._up_to_date(meta, SOURCE)
    assert ps._up_to_date(json.dumps(meta), SOURCE)
    assert not ps._up_to_date({**meta, "paper_key": "doi:other"}, SOURCE)
    assert not ps._up_to_date({**meta, "prompt_version": "0"}, SOURCE)
    assert not ps._up_to_date(None, SOURCE)


# ── run() over a fake database ────────────────────────────────────────────────

class _Result:
    def __init__(self, rows=(), scalar=None, rowcount=0):
        self._rows, self._scalar, self.rowcount = list(rows), scalar, rowcount

    def scalar(self):
        return self._scalar

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _Conn:
    def __init__(self, engine):
        self.engine = engine

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement, params=None):
        sql = str(statement)
        self.engine.statements.append((sql, params))
        if "information_schema" in sql:
            return _Result(scalar=True)
        if "SELECT DISTINCT ON" in sql:
            return _Result(rows=self.engine.rows)
        if "UPDATE bids_datasets SET paper_summary" in sql:
            self.engine.written[params["id"]] = (params["summary"], json.loads(params["meta"]))
            return _Result(rowcount=1)
        return _Result(rowcount=2)  # the prune

    async def commit(self):
        self.engine.commits += 1


class _Engine:
    def __init__(self, rows):
        self.rows, self.statements, self.written, self.commits = rows, [], {}, 0

    def connect(self):
        return _Conn(self)


def _row(n, accession, meta=None, abstract=ABSTRACT):
    return {"id": UUID(int=n), "accession_id": accession, "name": f"Dataset {n}",
            "paper_summary_meta": meta, "paper_key": f"doi:10.1/{n}",
            "title": "A paper", "abstract": abstract}


def test_run_writes_checked_summaries_and_skips_up_to_date(monkeypatch):
    current = ps.SummarySource("Dataset 3", "ds3", "doi:10.1/3", "A paper", ABSTRACT)
    rows = [
        _row(1, "ds1"),
        _row(2, "ds2"),
        _row(3, "ds3", meta={"paper_key": "doi:10.1/3", "abstract_sha": current.abstract_sha,
                             "prompt_version": ps.PROMPT_VERSION}),
    ]
    answers = {
        "ds1": "Resting-state fMRI from 42 people with Parkinson's disease and 23 controls at 3T.",
        "ds2": "fMRI from 500 adults.",  # invented number: rejected
    }

    def fake_summarise(source, model=ps.DEFAULT_MODEL):
        summary = answers[source.accession]
        problem = ps.check(summary, source)
        return (source, None, f"rejected ({problem})") if problem else (source, summary, "")

    monkeypatch.setattr(ps, "summarise_one", fake_summarise)
    engine = _Engine(rows)
    stats = asyncio.run(ps.run(engine))

    assert stats["done"] == 1 and stats["rejected"] == 1 and stats["skipped"] == 1
    assert stats["pruned"] == 2
    summary, meta = engine.written[UUID(int=1)]
    assert summary.startswith("Resting-state fMRI") and meta["paper_key"] == "doi:10.1/1"
    assert meta["prompt_version"] == ps.PROMPT_VERSION and meta["model"] == ps.DEFAULT_MODEL
    assert UUID(int=2) not in engine.written
    assert engine.commits == 2  # the prune, then the one write


def test_run_for_some_accessions_prunes_only_those_and_dry_run_writes_nothing(monkeypatch):
    monkeypatch.setattr(ps, "summarise_one", lambda s, model=None: (s, "x" * 50, ""))
    engine = _Engine([_row(1, "ds1")])
    stats = asyncio.run(ps.run(engine, ["ds1"], dry_run=True))
    assert stats["done"] == 1 and engine.written == {}
    assert not any("SET paper_summary = NULL" in sql for sql, _ in engine.statements)

    engine = _Engine([_row(1, "ds1")])
    asyncio.run(ps.run(engine, ["ds1"]))
    prune = next((sql, p) for sql, p in engine.statements if "SET paper_summary = NULL" in sql)
    assert "ANY(:acc)" in prune[0] and prune[1] == {"acc": ["ds1"]}


def test_run_refuses_without_the_columns(monkeypatch):
    engine = _Engine([])
    original = _Conn.execute

    async def no_columns(self, statement, params=None):
        if "information_schema" in str(statement):
            return _Result(scalar=False)
        return await original(self, statement, params)

    monkeypatch.setattr(_Conn, "execute", no_columns)
    with pytest.raises(RuntimeError, match="migrate_add_paper_summaries"):
        asyncio.run(ps.run(engine))


# ── API ───────────────────────────────────────────────────────────────────────

class _SummarySession:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def begin_nested(self):
        class _Nested:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *exc):
                return False
        return _Nested()

    async def execute(self, statement, params=None):
        self.calls.append(str(statement))
        return _Result(rows=self.rows)


def test_attach_summaries_adds_the_sentence_and_its_paper():
    import routers.query as query_router
    from schemas import DatasetSchema
    from services import schema_capabilities as caps

    def dataset(n):
        return DatasetSchema(id=UUID(int=n), name=f"d{n}", dataset_type="raw",
                             source_type="openneuro", participants=[])

    session = _SummarySession([
        (UUID(int=1), "fMRI from 65 adults.", "A paper", 2021, "NeuroImage", "10.1/x", None, None, None),
        (UUID(int=2), "Orphaned summary.", None, None, None, None, None, None, None),
    ])
    datasets = [dataset(1), dataset(2), dataset(3)]
    caps.set_display_columns({"paper_summary"})
    try:
        asyncio.run(query_router._attach_summaries(session, datasets))
    finally:
        caps.set_display_columns(())

    assert "paper_summary_meta->>'paper_key'" in session.calls[0]
    assert datasets[0].summary == "fMRI from 65 adults."
    assert datasets[0].summary_paper.url == "https://doi.org/10.1/x"
    assert datasets[0].summary_paper.journal == "NeuroImage"
    # A summary whose paper row is gone is not shown without its source.
    assert datasets[1].summary is None and datasets[2].summary is None


# ── Crawler ───────────────────────────────────────────────────────────────────

def _post_crawl():
    crawler = Path(__file__).resolve().parents[2] / "crawlers" / "openneuro-crawler"
    if str(crawler) not in sys.path:
        sys.path.insert(0, str(crawler))
    import post_crawl
    return post_crawl


def test_post_crawl_runs_both_steps_and_never_raises(monkeypatch):
    post_crawl = _post_crawl()
    import types
    monkeypatch.setitem(sys.modules, "db.db", types.SimpleNamespace(engine=object()))
    calls = []

    async def failing_lookup(engine, accessions):
        calls.append(("papers", list(accessions)))
        raise RuntimeError("Europe PMC down")

    async def summaries(engine, accessions):
        calls.append(("summaries", list(accessions)))
        return {"done": 1}

    monkeypatch.setattr(post_crawl, "lookup_papers", failing_lookup)
    monkeypatch.setattr(post_crawl, "write_summaries", summaries)
    monkeypatch.setattr(post_crawl, "_summaries_available", lambda: True)
    asyncio.run(post_crawl.run_post_crawl(["ds1", "", "ds2"]))
    assert calls == [("papers", ["ds1", "ds2"]), ("summaries", ["ds1", "ds2"])]


def test_post_crawl_switches(monkeypatch):
    post_crawl = _post_crawl()
    import types
    monkeypatch.setitem(sys.modules, "db.db", types.SimpleNamespace(engine=object()))
    calls = []

    async def record(engine, accessions):
        calls.append(len(accessions))
        return {}

    monkeypatch.setattr(post_crawl, "lookup_papers", record)
    monkeypatch.setattr(post_crawl, "write_summaries", record)
    monkeypatch.setattr(post_crawl, "_summaries_available", lambda: True)
    monkeypatch.setenv("CRAWL_PAPERS", "false")
    monkeypatch.setenv("CRAWL_PAPER_SUMMARIES", "false")
    asyncio.run(post_crawl.run_post_crawl(["ds1"]))
    asyncio.run(post_crawl.run_post_crawl([]))
    assert calls == []



def test_post_crawl_skips_summaries_without_a_key(monkeypatch):
    post_crawl = _post_crawl()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert post_crawl._summaries_available() is False


def test_crawler_creates_the_same_summary_columns_as_the_migration():
    import re
    _post_crawl()  # puts the crawler on the path
    from remote import paper_store

    migration = (Path(__file__).resolve().parents[2] / "crawlers"
                 / "migrate_add_paper_summaries.sql").read_text()
    squash = lambda value: re.sub(r"\s+", "", value)  # noqa: E731
    for statement in paper_store.SUMMARY_DDL:
        assert squash(statement) in squash(migration)
