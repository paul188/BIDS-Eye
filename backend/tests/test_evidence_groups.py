"""Which source group a piece of evidence belongs to (services/evidence/SCORING.md)."""

from __future__ import annotations

import dataclasses
import inspect

import pytest

pytest.importorskip("sqlglot")

from services.evidence.base import PieceOfEvidence, Source, SourceGroup, Strength  # noqa: E402
from services.uncertain import classify  # noqa: E402  (loads every kind of evidence)

GROUPS = {
    SourceGroup.RECORDS: {Source.PARTICIPANTS, Source.FIELD, Source.FILES, Source.DERIVATIVES,
                          Source.SITES},
    SourceGroup.OWN_TEXT: {Source.SHORT_TEXT, Source.TEXT, Source.README},
    SourceGroup.PAPER: {Source.PAPERS},
}
WITHOUT_GROUP = {Source.ASSUMED, Source.ASSUMED_ABSENT, Source.NOT_RECORDED}


def _kinds(base: type = PieceOfEvidence) -> set[type]:
    below = set(base.__subclasses__())
    return below.union(*(_kinds(kind) for kind in below))


def _default(kind: type, name: str) -> object:
    return next(f.default for f in dataclasses.fields(kind) if f.name == name)


def test_every_source_is_listed_once() -> None:
    listed = [source for sources in (*GROUPS.values(), WITHOUT_GROUP) for source in sources]
    assert len(listed) == len(set(listed))
    assert set(listed) == set(Source)
    assert set(GROUPS) == set(SourceGroup)


@pytest.mark.parametrize("group, source", [(g, s) for g, sources in GROUPS.items() for s in sorted(sources)])
def test_source_group(group: SourceGroup, source: Source) -> None:
    assert source.group is group


@pytest.mark.parametrize("source", sorted(WITHOUT_GROUP))
def test_source_without_group_raises(source: Source) -> None:
    with pytest.raises(ValueError, match=f"the source {source} has no source group"):
        _ = source.group


def test_weaker_kinds_have_a_group() -> None:
    """Only weaker (probable) pieces are combined by group. ProbablePatientsNotCountedEvidence
    has none: it must stay the only piece of its clause (test_patient_count.py)."""
    weaker = {kind for kind in _kinds() if not inspect.isabstract(kind)
              and _default(kind, "strength") == Strength.PROBABLE}
    grouped = set().union(*GROUPS.values())
    assert {kind for kind in weaker if _default(kind, "source") not in grouped} == {
        classify.ProbablePatientsNotCountedEvidence}
