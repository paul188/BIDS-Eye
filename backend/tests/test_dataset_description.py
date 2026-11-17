"""Paper relations and thin-README summaries (services/dataset_description.py),
how the API attaches them, and the crawler's post-crawl step."""

from __future__ import annotations

import asyncio
import json
import re
import sys
import types
from pathlib import Path
from uuid import UUID

import pytest

from services import dataset_description as dd

ROOT = Path(__file__).resolve().parents[2]

README = ("Resting-state fMRI of patients with Parkinson's disease, scanned ON and OFF "
          "medication. See participants.tsv for UPDRS scores.")
ABSTRACT = ("We acquired resting-state fMRI in 42 patients with idiopathic Parkinson's disease "
            "and 23 healthy controls, scanned ON and OFF dopaminergic medication at 3T. "
            "Basal ganglia connectivity was reduced OFF medication.")
ROW = {
    "id": UUID(int=1), "accession_id": "ds004392", "name": "PD rest ON/OFF",
    "readme_text": README, "description_text": None, "study_domain": "Parkinson's",
    "study_design": None, "dx_status": None, "age_min": 48.0, "age_max": 79.0,
    "subject_count": 65, "datatypes": ["functional_mri", "anatomical_mri"],
    "tasks": ["rest"], "diagnoses": {"PD": 42, "control": 23},
    "paper_key": "doi:10.1/x", "title": "Dopamine and resting networks in PD",
    "abstract": ABSTRACT, "year": 2021, "role_reason": "shares an author; title overlaps",
    "paper_relation": None, "dataset_summary_meta": None,
}


def _source(**changes):
    return dd.source_from_row({**ROW, **changes})


# ── Sources ───────────────────────────────────────────────────────────────────

def test_catalogue_lists_recorded_fields_readably():
    text = _source().catalogue_text()
    assert "Name: PD rest ON/OFF" in text
    assert "Subjects: 65" in text
    assert "Modalities: structural MRI, fMRI" in text
    assert "Participant diagnoses (count): PD 42, control 23" in text
    assert "Ages: 48–79" in text and "Study design" not in text


def test_thin_means_readme_and_description_under_500_chars():
    assert _source().thin
    assert not _source(readme_text="x" * 600).thin
    assert dd.worth_summarising(_source())
    bare = _source(readme_text="", datatypes=None, tasks=None, diagnoses=None, subject_count=0,
                   age_min=None, age_max=None, study_domain=None, paper_key=None)
    assert not dd.worth_summarising(bare)


# ── Relations ─────────────────────────────────────────────────────────────────

def test_relation_prompt_labels_every_source():
    prompt = dd.relation_prompt(_source())
    for label in ("[README]", "[CATALOGUE]", "[PAPER]", "[WHY LINKED]"):
        assert label in prompt
    assert "larger_sample" in prompt and "word for word" in prompt


def test_parse_relation_checks_each_quote_against_its_source():
    answer = json.dumps({
        "relation": "reports",
        "mismatches": [{"aspect": "sample_size", "note": "65 subjects vs 42 + 23"}],
        "evidence": [
            {"source": "PAPER", "quote": "scanned ON and OFF dopaminergic medication at 3T"},
            {"source": "README", "quote": "scanned on and off   medication"},  # spacing differs
            {"source": "README", "quote": "dopaminergic medication at 3T"},       # not in README
        ],
    })
    relation = dd.parse_relation(answer, _source())
    assert relation.relation == "reports" and relation.mismatches[0]["aspect"] == "sample_size"
    assert [e["verified"] for e in relation.evidence] == [True, True, False]
    assert relation.shown


def test_relation_is_shown_only_for_reporting_papers_with_a_verified_quote():
    unverified = json.dumps({"relation": "reports", "mismatches": [],
                             "evidence": [{"source": "PAPER", "quote": "an invented sentence here"}]})
    assert not dd.parse_relation(unverified, _source()).shown
    related = json.dumps({"relation": "related_study", "mismatches": [],
                          "evidence": [{"source": "PAPER", "quote": "Basal ganglia connectivity"}]})
    assert not dd.parse_relation(related, _source()).shown
    assert dd.parse_relation('{"relation": "maybe"}', _source()) is None
    assert dd.parse_relation('{"relation": "reports", "evid', _source()) is None


# ── Summaries ─────────────────────────────────────────────────────────────────

def test_summary_prompt_offers_the_paper_only_with_the_framing():
    with_paper = dd.summary_prompt(_source(), use_paper=True)
    assert "[PAPER]" in with_paper and dd.PAPER_PREFIX in with_paper
    without = dd.summary_prompt(_source(), use_paper=False)
    assert "[PAPER]" not in without and "There is no paper" in without


def test_check_summary_requires_the_paper_framing_for_paper_facts():
    ok = ("Resting-state fMRI from 65 participants, people with Parkinson's disease and "
          "controls, scanned on and off medication. According to the linked paper, 42 had "
          "idiopathic Parkinson's disease and scans were at 3T.")
    assert dd.check_summary(ok, _source(), paper_used=True) is None
    # 42 appears in the catalogue counts too; 3T only in the paper.
    unframed = "Resting-state fMRI from 65 participants scanned at 3T on and off medication."
    assert "framing" in dd.check_summary(unframed, _source(), paper_used=True)
    invented = "Resting-state fMRI from 80 participants with Parkinson's disease."
    assert "numbers not in" in dd.check_summary(invented, _source(), paper_used=True)
    assert "not given" in dd.check_summary(ok, _source(), paper_used=False)
    copied = ("Resting-state fMRI data. According to the linked paper, they acquired "
              "resting-state fMRI in 42 patients with idiopathic Parkinson's disease and 23 "
              "healthy controls.")
    assert "copies" in dd.check_summary(copied, _source(), paper_used=True)
    assert "length" in dd.check_summary("fMRI.", _source(), paper_used=False)
    assert "sources" in dd.check_summary(
        "README and CATALOGUE describe resting-state fMRI from 65 participants.", _source(), False)
    assert "sources" in dd.check_summary(
        "Resting-state fMRI in ds004392 from people with Parkinson's disease.", _source(), False)
    assert "missing" in dd.check_summary(
        "Resting-state fMRI from 65 participants; no further details are available.", _source(), False)


# ── Planning reruns ───────────────────────────────────────────────────────────

def test_plan_work_redoes_only_what_changed():
    source = _source()
    assert dd.plan_work(ROW, source, force=False) == (True, True)

    relation = {"input_sha": dd.relation_input_sha(source), "shown": True}
    meta = {"input_sha": dd.summary_input_sha(source, True)}
    done = {**ROW, "paper_relation": relation, "dataset_summary_meta": meta}
    assert dd.plan_work(done, source, force=False) == (False, False)
    assert dd.plan_work(done, source, force=True) == (True, True)

    changed = _source(readme_text=README + " Also T2 scans.")
    assert dd.plan_work(done, changed, force=False) == (True, True)

    thick = _source(readme_text="x" * 600)
    relation_thick = {"input_sha": dd.relation_input_sha(thick), "shown": True}
    assert dd.plan_work({**ROW, "paper_relation": relation_thick}, thick, False) == (False, False)


def test_describe_one_uses_the_paper_only_when_the_relation_is_shown(monkeypatch):
    prompts = []
    relation_answer = json.dumps({"relation": "related_study", "mismatches": [], "evidence": [
        {"source": "PAPER", "quote": "Basal ganglia connectivity was reduced"}]})

    def fake_ask(prompt, system, schema, model):
        prompts.append(prompt)
        if schema is dd.RELATION_SCHEMA:
            return relation_answer
        return json.dumps({"summary": "Resting-state fMRI from 65 participants with Parkinson's "
                                      "disease and controls, on and off medication."})

    monkeypatch.setattr(dd, "_ask", fake_ask)
    outcome = dd.describe_one(_source(), True, None, True)
    assert outcome.relation.relation == "related_study" and not outcome.relation.shown
    assert "[PAPER]" not in prompts[1] and outcome.summary and not outcome.summary_paper_used


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
            return _Result(scalar=self.engine.columns)
        if "LEFT JOIN LATERAL" in sql:
            return _Result(rows=self.engine.rows)
        if "SET paper_relation = CAST" in sql:
            self.engine.relations[params["id"]] = json.loads(params["value"])
        if "SET dataset_summary = :summary" in sql:
            self.engine.summaries[params["id"]] = (params["summary"], json.loads(params["meta"]))
        return _Result(rowcount=1)

    async def commit(self):
        self.engine.commits += 1


class _Engine:
    def __init__(self, rows, columns=True):
        self.rows, self.columns = rows, columns
        self.statements, self.relations, self.summaries, self.commits = [], {}, {}, 0

    def connect(self):
        return _Conn(self)


def test_run_records_relations_and_summaries(monkeypatch):
    def fake_describe(source, need_relation, current, need_summary, model=None):
        outcome = dd.Outcome(source)
        if need_relation:
            outcome.relation = dd.Relation("reports", [], [
                {"source": "PAPER", "quote": "x", "verified": True}])
        if need_summary:
            outcome.summary_asked = True
            outcome.summary_paper_used = True
            outcome.summary = "Resting-state fMRI from 65 participants."
        return outcome

    monkeypatch.setattr(dd, "describe_one", fake_describe)
    engine = _Engine([ROW, {**ROW, "id": UUID(int=2), "accession_id": "ds2", "readme_text": "x" * 600}])
    stats = asyncio.run(dd.run(engine))

    assert stats["relation:reports"] == 2 and stats["relation_shown"] == 2
    assert stats["summary_done"] == 1   # only the thin one
    assert stats["pruned"] == 2         # both prune statements ran
    relation = engine.relations[UUID(int=1)]
    assert relation["paper_key"] == "doi:10.1/x" and relation["shown"] is True
    assert relation["input_sha"] == dd.relation_input_sha(_source())
    summary, meta = engine.summaries[UUID(int=1)]
    assert summary.startswith("Resting-state") and meta["paper_used"] is True
    assert UUID(int=2) not in engine.summaries


def test_run_dry_run_writes_nothing_and_refuses_without_columns(monkeypatch):
    monkeypatch.setattr(dd, "describe_one", lambda s, *a, **k: dd.Outcome(s))
    engine = _Engine([ROW])
    asyncio.run(dd.run(engine, dry_run=True))
    assert not engine.relations and not engine.summaries and engine.commits == 0
    with pytest.raises(RuntimeError, match="migrate_add_dataset_descriptions"):
        asyncio.run(dd.run(_Engine([], columns=False)))


# ── API ───────────────────────────────────────────────────────────────────────

class _Session:
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


def test_attach_descriptions_shows_the_paper_line_and_summary():
    import routers.query as query_router
    from schemas import DatasetSchema
    from services import schema_capabilities as caps

    def dataset(n):
        return DatasetSchema(id=UUID(int=n), name=f"d{n}", dataset_type="raw",
                             source_type="openneuro", participants=[])

    session = _Session([
        (UUID(int=1), None, "reports", "A paper", 2021, "10.1/x", None, None, None),
        (UUID(int=2), "Generated text.", "related_study", None, None, None, None, None, None),
    ])
    datasets = [dataset(1), dataset(2), dataset(3)]
    caps.set_display_columns({"paper_relation", "dataset_summary"})
    try:
        asyncio.run(query_router._attach_descriptions(session, datasets))
    finally:
        caps.set_display_columns(())

    assert datasets[0].linked_paper.relation_label == "reports this dataset"
    assert datasets[0].linked_paper.url == "https://doi.org/10.1/x"
    assert datasets[0].generated_summary is None
    assert datasets[1].generated_summary == "Generated text." and datasets[1].linked_paper is None
    assert datasets[2].linked_paper is None and datasets[2].generated_summary is None


# ── Crawler ───────────────────────────────────────────────────────────────────

def _post_crawl():
    crawler = ROOT / "crawlers" / "openneuro-crawler"
    if str(crawler) not in sys.path:
        sys.path.insert(0, str(crawler))
    import post_crawl
    return post_crawl


def test_post_crawl_runs_both_steps_and_never_raises(monkeypatch):
    post_crawl = _post_crawl()
    monkeypatch.setitem(sys.modules, "db.db", types.SimpleNamespace(engine=object()))
    calls = []

    async def failing_lookup(engine, accessions):
        calls.append(("papers", list(accessions)))
        raise RuntimeError("Europe PMC down")

    async def describe(engine, accessions):
        calls.append(("descriptions", list(accessions)))
        return {}

    monkeypatch.setattr(post_crawl, "lookup_papers", failing_lookup)
    monkeypatch.setattr(post_crawl, "describe", describe)
    monkeypatch.setattr(post_crawl, "_descriptions_available", lambda: True)
    asyncio.run(post_crawl.run_post_crawl(["ds1", "", "ds2"]))
    assert calls == [("papers", ["ds1", "ds2"]), ("descriptions", ["ds1", "ds2"])]


def test_post_crawl_switches(monkeypatch):
    post_crawl = _post_crawl()
    monkeypatch.setitem(sys.modules, "db.db", types.SimpleNamespace(engine=object()))
    calls = []

    async def record(engine, accessions):
        calls.append(len(accessions))
        return {}

    monkeypatch.setattr(post_crawl, "lookup_papers", record)
    monkeypatch.setattr(post_crawl, "describe", record)
    monkeypatch.setattr(post_crawl, "_descriptions_available", lambda: True)
    monkeypatch.setenv("CRAWL_PAPERS", "false")
    monkeypatch.setenv("CRAWL_DESCRIPTIONS", "false")
    asyncio.run(post_crawl.run_post_crawl(["ds1"]))
    asyncio.run(post_crawl.run_post_crawl([]))
    assert calls == []


def test_post_crawl_skips_descriptions_without_a_key(monkeypatch):
    post_crawl = _post_crawl()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert post_crawl._descriptions_available() is False


def test_crawler_creates_the_same_columns_as_the_migration():
    _post_crawl()
    from remote import paper_store

    migration = (ROOT / "crawlers" / "migrate_add_dataset_descriptions.sql").read_text()
    squash = lambda value: re.sub(r"\s+", "", value)  # noqa: E731
    for statement in paper_store.DESCRIPTION_DDL:
        assert squash(statement) in squash(migration)


def test_post_crawl_vacuums_the_file_table_outside_a_transaction(monkeypatch):
    post_crawl = _post_crawl()
    statements, options = [], []

    class Connection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execution_options(self, **kwargs):
            options.append(kwargs)
            return self

        async def execute(self, statement):
            statements.append(str(statement))
            if str(statement).startswith("VACUUM") and fail:
                raise RuntimeError("could not resize shared memory segment")

    engine = types.SimpleNamespace(connect=lambda: Connection())
    monkeypatch.setitem(sys.modules, "db.db", types.SimpleNamespace(engine=engine))
    for name in ("CRAWL_LINKS", "CRAWL_PAPERS", "CRAWL_DESCRIPTIONS", "CRAWL_WORD_FORMS"):
        monkeypatch.setenv(name, "false")

    fail = False
    asyncio.run(post_crawl.run_post_crawl(["ds1"]))
    assert options == [{"isolation_level": "AUTOCOMMIT"}]
    assert statements == ["SET statement_timeout = 0",
                          "VACUUM (PARALLEL 0, ANALYZE) bids_objects",
                          "RESET statement_timeout"]

    # A failing vacuum never fails the crawl, and the timeout is restored.
    fail = True
    statements.clear()
    asyncio.run(post_crawl.run_post_crawl(["ds1"]))
    assert statements[-1] == "RESET statement_timeout"

    monkeypatch.setenv("CRAWL_VACUUM", "false")
    statements.clear()
    asyncio.run(post_crawl.run_post_crawl(["ds1"]))
    assert statements == []
