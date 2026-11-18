"""The local mirror of an OpenNeuro dataset (crawlers/openneuro-crawler/
remote/openneuro.py _build_mirror): a metadata file whose download fails is
fetched again on the next crawl instead of staying an empty stub."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_CRAWLER = ROOT / "crawlers" / "openneuro-crawler"
if str(_CRAWLER) not in sys.path:
    sys.path.insert(0, str(_CRAWLER))

pytest.importorskip("boto3")
from botocore.exceptions import ClientError, ReadTimeoutError  # noqa: E402

from remote import openneuro  # noqa: E402

PARTICIPANTS = b"participant_id\tage\nsub-01\t23\n"
OBJECTS = [
    {"Key": "ds000001/dataset_description.json", "Size": 30},
    {"Key": "ds000001/participants.tsv", "Size": len(PARTICIPANTS)},
    {"Key": "ds000001/sub-01/anat/sub-01_T1w.nii.gz", "Size": 9_000_000},
]
CONTENT = {"ds000001/dataset_description.json": b'{"Name": "x", "BIDSVersion": "1"}',
           "ds000001/participants.tsv": PARTICIPANTS}


class _Client:
    def __init__(self, failures=None):
        self.failures = dict(failures or {})
        self.fetched = []

    def get_object(self, Bucket, Key):
        self.fetched.append(Key)
        if Key in self.failures:
            raise self.failures.pop(Key)
        return {"Body": io.BytesIO(CONTENT[Key])}


def _timeout():
    return ReadTimeoutError(endpoint_url="https://s3.amazonaws.com/openneuro.org")


def test_a_failed_metadata_download_is_fetched_again_on_the_next_crawl(tmp_path) -> None:
    openneuro._build_mirror(_Client({"ds000001/participants.tsv": _timeout()}),
                            "ds000001", OBJECTS, tmp_path)
    assert (tmp_path / "participants.tsv").read_bytes() == b""  # stub for this crawl

    client = _Client()
    openneuro._build_mirror(client, "ds000001", OBJECTS, tmp_path)
    assert client.fetched == ["ds000001/participants.tsv"]
    assert (tmp_path / "participants.tsv").read_bytes() == PARTICIPANTS


def test_imaging_stubs_are_not_fetched(tmp_path) -> None:
    openneuro._build_mirror(_Client(), "ds000001", OBJECTS, tmp_path)
    client = _Client()
    openneuro._build_mirror(client, "ds000001", OBJECTS, tmp_path)
    assert client.fetched == []
    assert (tmp_path / "sub-01/anat/sub-01_T1w.nii.gz").stat().st_size == 0


def test_access_denied_is_a_restricted_dataset(tmp_path) -> None:
    denied = ClientError({"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "GetObject")
    with pytest.raises(openneuro.DatasetAccessDeniedError) as caught:
        openneuro._build_mirror(_Client({"ds000001/dataset_description.json": denied}),
                                "ds000001", OBJECTS, tmp_path)
    assert caught.value.__cause__ is denied


def test_a_bug_is_raised_not_stubbed(tmp_path) -> None:
    with pytest.raises(TypeError):
        openneuro._build_mirror(_Client({"ds000001/participants.tsv": TypeError("a bug")}),
                                "ds000001", OBJECTS, tmp_path)
