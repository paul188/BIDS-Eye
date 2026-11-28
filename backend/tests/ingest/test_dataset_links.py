from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_CRAWLER = ROOT / "ingest" / "openneuro-crawler"
if str(_CRAWLER) not in sys.path:
    sys.path.insert(0, str(_CRAWLER))

from remote import dataset_links as links  # noqa: E402
from remote import openneuro_metadata as meta  # noqa: E402


def test_pipeline_names_are_normalised() -> None:
    assert links.normalise_pipeline("fMRIPrep") == "fmriprep"
    assert links.normalise_pipeline("fmriprep-20.2.3") == "fmriprep"
    assert links.normalise_pipeline("fmriprep_v23.1.0") == "fmriprep"
    assert links.normalise_pipeline("freesurfer_subjdir") == "freesurfer"
    assert links.normalise_pipeline("xcp_d") == "xcpd"
    # digits without a separator are part of the name
    assert links.normalise_pipeline("cat12") == "cat12"
    assert links.normalise_pipeline("SPM12") == "spm12"
    assert links.normalise_pipeline("fs_stats") == "fs-stats"
    assert links.normalise_pipeline("") == links.UNSPECIFIED


def test_generated_by_reads_current_and_legacy_descriptions() -> None:
    assert links.generated_by({"GeneratedBy": [{"Name": "fMRIPrep", "Version": "21.0.1"}]}) == ("fMRIPrep", "21.0.1")
    assert links.generated_by({"GeneratedBy": {"Name": "MRIQC"}}) == ("MRIQC", None)
    assert links.generated_by({"PipelineDescription": {"Name": "fmriprep", "Version": "1.2"}}) == ("fmriprep", "1.2")
    assert links.generated_by({"Name": "no pipeline"}) == (None, None)
    assert links.generated_by(None) == (None, None)


def test_site_names_are_cleaned() -> None:
    assert links.clean_site_names(None) == []
    assert links.clean_site_names("  IRB of  Wayne State University ") == ["IRB of Wayne State University"]
    assert links.clean_site_names(["A", "A", "n/a", "", 3, "B"]) == ["A", "B"]
    assert len(links.clean_site_names(["x" * 900])[0]) == 500


def test_metadata_record_gives_new_columns_and_ethics_sites() -> None:
    record = {
        "publishDate": "2024-09-20T21:02:46.777Z",
        "onBrainlife": False,
        "metadata": {"species": "Mouse", "studyLongitudinal": "Longitudinal", "seniorAuthor": " Xin Yu "},
        "latestSnapshot": {
            "description": {"EthicsApprovals": ["MGH IACUC", "MGH IACUC"]},
            "analytics": {"downloads": 28, "views": 2070},
        },
    }
    fields = meta.extract_fields(record)
    assert fields["species"] == "Mouse"
    assert fields["study_longitudinal"] == "Longitudinal"
    assert fields["senior_author"] == "Xin Yu"
    assert fields["publish_date"].isoformat() == "2024-09-20T21:02:46.777000+00:00"
    assert fields["on_brainlife"] is False
    assert (fields["downloads"], fields["views"]) == (28, 2070)
    assert fields["ethics_approvals"] == ["MGH IACUC"]

    # No description in the record: unknown, so stored ethics rows must stay.
    assert meta.extract_fields({"latestSnapshot": {}})["ethics_approvals"] is None
    assert meta.extract_fields({"latestSnapshot": {"description": None}})["ethics_approvals"] is None
    assert meta.extract_fields({"latestSnapshot": {"description": {}}})["ethics_approvals"] == []
    bad = meta.extract_fields({"publishDate": "soon", "onBrainlife": "yes",
                               "latestSnapshot": {"analytics": {"downloads": -1, "views": True}}})
    assert bad["publish_date"] is bad["on_brainlife"] is bad["downloads"] is bad["views"] is None


def test_backfill_sql_replaces_per_source_and_quotes() -> None:
    sys.path.insert(0, str(ROOT / "ingest" / "backfill"))
    import backfill_dataset_links as backfill

    found = {
        links.SOURCE_OPENNEURO_DERIVATIVES: {"ds000001": [links.Derivative(
            pipeline="fmriprep", location="s3://openneuro-derivatives/fmriprep/ds000001-fmriprep/",
            pipeline_name="fMRIPrep", pipeline_version="21.0.1", n_subjects=16)]},
        links.SOURCE_IN_DATASET: {"ds000002": []},
    }
    sql = backfill.sql_for(found, ["ds000001", "ds0'02"])
    assert sql.startswith("BEGIN;") and sql.rstrip().endswith("COMMIT;")
    assert "x.source = 'in_dataset'" in sql            # an empty list still clears old rows
    assert "'fMRIPrep', '21.0.1', 'openneuro_derivatives'" in sql
    assert "'ds0''02'" in sql


def test_an_unexpected_file_tree_is_a_link_source_error(monkeypatch) -> None:
    import pytest

    answer = {"data": {"d0": {"latestSnapshot": {"files": [{"directory": True, "id": "x"}]}}}}
    monkeypatch.setattr(links, "_post_graphql", lambda query, variables: answer)
    with pytest.raises(links.LinkSourceError, match="ds000001"):
        links._list_trees([("ds000001", None)])
    answer["data"]["d0"] = None  # a dataset the API did not return
    assert links._list_trees([("ds000001", None)]) == [None]


def test_a_source_that_cannot_be_read_keeps_the_stored_rows(monkeypatch) -> None:
    import asyncio

    import pytest

    calls = []

    class _Conn:
        async def execute(self, statement, params=None):
            calls.append(str(statement))

    def unreadable(accessions):
        raise links.LinkSourceError("bucket listing refused")

    def bug(accessions):
        raise KeyError("filename")

    monkeypatch.setattr(links, "openneuro_derivatives", unreadable)
    monkeypatch.setattr(links, "in_dataset_derivatives", lambda accessions: {})
    counts = asyncio.run(links.refresh_links(_Conn(), ["ds000001"]))
    assert counts == {"openneuro_derivatives": 0, "in_dataset": 0}
    assert not any("DELETE FROM bids_derivatives" in s for s in calls)

    monkeypatch.setattr(links, "openneuro_derivatives", bug)
    with pytest.raises(KeyError):
        asyncio.run(links.refresh_links(_Conn(), ["ds000001"]))
