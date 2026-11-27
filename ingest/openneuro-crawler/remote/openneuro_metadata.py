"""
remote/openneuro_metadata.py
----------------------------
Dataset-level metadata that lives in OpenNeuro itself rather than in the BIDS
files: the README text, the fields the uploader typed into the OpenNeuro
metadata form (diagnosis status, study domain, study design), the reported
participant ages and the total dataset size.

One GraphQL request returns all of it, and several datasets fit in one request,
so filling these columns for the whole catalogue takes minutes and needs no
S3 access and no re-indexing of files.

Used by:
  - remote/openneuro.py   after a dataset is indexed from its S3 mirror
  - remote/graphql.py     the fallback for restricted datasets
  - ingest/backfill/backfill_openneuro_metadata.py   for datasets that are already indexed

Only the standard library and SQLAlchemy are needed, so the module also runs
inside the backend image.

Columns written (all on bids_datasets, created by the migrations in backend/migrations/):
    readme_text   TEXT      sanitised README of the latest snapshot
    dx_status     TEXT      metadata.dxStatus     e.g. "Healthy / Control", "Epilepsy"
    study_domain  TEXT      metadata.studyDomain  e.g. "Basal Ganglia in Parkinson"
    study_design  TEXT      metadata.studyDesign  e.g. "Longitudinal"
    species       TEXT      metadata.species      e.g. "Human", "Mouse", "Macaque"
    study_longitudinal TEXT metadata.studyLongitudinal  "Cross-Sectional", "Longitudinal", …
    senior_author TEXT      metadata.seniorAuthor
    publish_date  TIMESTAMPTZ  publishDate        when the dataset went public
    on_brainlife  BOOLEAN   onBrainlife           processed on brainlife.io
    downloads     INTEGER   latestSnapshot.analytics.downloads   (current count)
    views         INTEGER   latestSnapshot.analytics.views       (current count)

The EthicsApprovals of the latest snapshot go to bids_dataset_sites with
source 'ethics_board' (remote/dataset_links.py), not to a column.
    age_min       FLOAT     youngest reported participant age
    age_max       FLOAT     oldest reported participant age
    size_bytes    BIGINT    total size of the latest snapshot

It also keeps one status column up to date:
    deprecated_reason  TEXT  NULL while the dataset is fine upstream; otherwise
                             why it should be treated as deprecated (removed
                             from OpenNeuro, or flagged deprecated there)
"""

from __future__ import annotations

import http.client
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import String

from .dataset_links import SITE_ETHICS, clean_site_names, replace_sites

log = logging.getLogger(__name__)

GRAPHQL_URL = "https://openneuro.org/crn/graphql"
_USER_AGENT = "bids-metadata-indexer/1.0"

# Datasets per GraphQL request. README bodies make responses large; 12 keeps a
# response comfortably under a few hundred kB.
DEFAULT_BATCH_SIZE = 12

# READMEs are only searched, never sent to an LLM, so keep (almost) all of the
# text. The median README is ~1 100 characters; the cap only guards against the
# rare README with an embedded data dump.
README_MAX_CHARS = 50_000
README_NAMES: Tuple[str, ...] = ("README", "README.md", "README.txt", "README.rst")

# Reported ages outside this range are treated as data-entry errors.
_AGE_RANGE = (0.0, 120.0)

# Columns this module writes, with their PostgreSQL types. `institutions` is not
# OpenNeuro metadata, but the crawler writes it and the ORM model does not
# define it either, so it is created together with the others.
METADATA_COLUMNS: Dict[str, str] = {
    "readme_text":  "TEXT",
    "dx_status":    "TEXT",
    "study_domain": "TEXT",
    "study_design": "TEXT",
    "species":      "TEXT",
    "study_longitudinal": "TEXT",
    "senior_author": "TEXT",
    "publish_date": "TIMESTAMPTZ",
    "on_brainlife": "BOOLEAN",
    "downloads":    "INTEGER",
    "views":        "INTEGER",
    "age_min":      "DOUBLE PRECISION",
    "age_max":      "DOUBLE PRECISION",
    "size_bytes":   "BIGINT",
}
# Upstream status. Unlike the metadata columns it is overwritten on every
# check, because it describes the present, not something to accumulate.
STATUS_COLUMNS: Dict[str, str] = {"deprecated_reason": "TEXT"}
# Which snapshot the files were indexed from (record_snapshot); the crawler
# compares them with the API's latest snapshot (snapshot_plan.py).
SNAPSHOT_COLUMNS: Dict[str, str] = {
    "snapshot_tag": "TEXT",
    "snapshot_hexsha": "TEXT",
    "files_indexed_at": "TIMESTAMPTZ",
    "indexer_version": "INTEGER",
}
REMOVED_REASON = "No longer available on OpenNeuro"

# What bids_datasets.search_tsv is generated from. The column itself comes
# from the migrations (backend/migrations/) and PostgreSQL keeps it current;
# backend/tests/test_migrations_postgres.py checks that the two agree.
# "/" becomes a space because PostgreSQL reads "ADD/ADHD" as one file path.
SEARCH_VECTOR_EXPRESSION = (
    "setweight(to_tsvector('english'::regconfig, "
    "translate(coalesce(name, '')::text, '/', ' ')), 'A') || "
    "setweight(to_tsvector('english'::regconfig, translate("
    "coalesce(dx_status, '') || ' ' || coalesce(study_domain, '') || ' ' || "
    "coalesce(study_design, '') || ' ' || coalesce(description_text, '') || ' ' || "
    "coalesce(llm_conditions, ''), '/', ' ')), 'B') || "
    "setweight(to_tsvector('english'::regconfig, "
    "translate(coalesce(llm_profile, ''), '/', ' ')), 'C') || "
    "setweight(to_tsvector('english'::regconfig, "
    "translate(coalesce(readme_text, ''), '/', ' ')), 'D')"
)

# The same vector without stemming, for searched words that must match in
# the form they were asked for: with the stemmer, "ageing" also finds "age"
# (backend/services/fulltext/text_search.py).
EXACT_SEARCH_COLUMN = "search_tsv_exact"
EXACT_SEARCH_VECTOR_EXPRESSION = SEARCH_VECTOR_EXPRESSION.replace(
    "'english'::regconfig", "'simple'::regconfig")

# Selection shared by every query in this module.
METADATA_SELECTION = """
    publishDate onBrainlife
    metadata { species studyDesign studyDomain studyLongitudinal seniorAuthor dxStatus ages }
    latestSnapshot {
      tag hexsha created readme size summary { size } deprecated { reason }
      description { EthicsApprovals } analytics { downloads views }
    }
"""


class OpenNeuroAPIError(RuntimeError):
    """The GraphQL endpoint could not be reached or returned no usable payload."""


# ── HTTP ──────────────────────────────────────────────────────────────────────

def post_graphql(
    query: str,
    variables: Optional[dict] = None,
    *,
    timeout: float = 90.0,
    retries: int = 3,
) -> dict:
    """POST one GraphQL request and return the decoded JSON body.

    The body may contain both ``data`` and ``errors``: OpenNeuro reports a
    dataset it will not show (deleted, embargoed) as an error entry and still
    answers for the other datasets in the same request.

    Raises OpenNeuroAPIError when every attempt fails or the body has no ``data``.
    """
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            GRAPHQL_URL,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            # 4xx carries a GraphQL validation message in the body; retrying
            # the same query cannot succeed.
            detail = _error_body(exc)
            if 400 <= exc.code < 500 and exc.code != 429:
                raise OpenNeuroAPIError(f"HTTP {exc.code}: {detail}") from exc
            last_error = OpenNeuroAPIError(f"HTTP {exc.code}: {detail}")
        # URLError and timeouts are OSErrors; a body cut short (IncompleteRead)
        # is an HTTPException; JSONDecodeError and UnicodeDecodeError (a body
        # that is not UTF-8) are ValueErrors.
        except (OSError, http.client.HTTPException, ValueError) as exc:
            last_error = exc
        else:
            if isinstance(body, dict) and isinstance(body.get("data"), dict):
                return body
            last_error = OpenNeuroAPIError(f"no data in response: {str(body)[:300]}")
        if attempt < retries:
            time.sleep(3 * attempt)
    raise OpenNeuroAPIError(f"OpenNeuro GraphQL request failed: {last_error}") from last_error


def _error_body(exc: urllib.error.HTTPError) -> str:
    """The start of an error answer's body; empty when it cannot be read."""
    try:
        return exc.read().decode("utf-8", "replace")[:500]
    except (OSError, http.client.HTTPException):
        return ""


_DELETED_RE = re.compile(r"has been deleted\.?(?:\s*Reason:\s*(?P<reason>.*))?$", re.IGNORECASE | re.DOTALL)


def unavailable_reason(message: str) -> Optional[str]:
    """Translate an API error for one dataset into a reason to show users.

    OpenNeuro reports three definite situations (wording as of API 5.13):
        "Dataset ds001019 does not exist."
        "Dataset ds004929 has been deleted. Reason: duplicate dataset."
        "You do not have access to read this dataset."      (made private)
    Any other message returns None: the status is unknown and must not be
    turned into a mark.
    """
    message = (message or "").strip()
    lowered = message.lower()
    if "does not exist" in lowered:
        return REMOVED_REASON
    deleted = _DELETED_RE.search(message)
    if deleted:
        reason = (deleted.group("reason") or "").strip().rstrip(".")
        # The API echoes whatever the uploader typed, including nothing.
        if reason and reason.lower() not in ("undefined", "null", "none"):
            return f"Deleted on OpenNeuro: {reason}"
        return "Deleted on OpenNeuro"
    if "do not have access" in lowered:
        return "No longer public on OpenNeuro"
    return None


def fetch_batch(accession_ids: Sequence[str]) -> Tuple[Dict[str, Optional[dict]], Dict[str, str]]:
    """Fetch the metadata record of several datasets in one request.

    Returns ``(records, unavailable)``:
      records      ``{accession_id: record}``; None when the API returned nothing
      unavailable  ``{accession_id: reason}`` for datasets the API definitely
                   reports as removed, deleted or no longer public

    A None record that is not in ``unavailable`` means "unknown" and must not
    be read as "removed".

    Raises OpenNeuroAPIError when the request itself fails, so a network
    problem is never mistaken for a missing dataset.
    """
    ids = list(accession_ids)
    if not ids:
        return {}, {}
    declarations = ", ".join(f"$id{i}: ID!" for i in range(len(ids)))
    selections = "\n".join(
        f"  d{i}: dataset(id: $id{i}) {{ id {METADATA_SELECTION} }}" for i in range(len(ids))
    )
    query = f"query Metadata({declarations}) {{\n{selections}\n}}"
    variables = {f"id{i}": accession_id for i, accession_id in enumerate(ids)}
    body = post_graphql(query, variables)
    data = body["data"]
    records = {accession_id: data.get(f"d{i}") for i, accession_id in enumerate(ids)}

    alias_to_id = {f"d{i}": accession_id for i, accession_id in enumerate(ids)}
    unavailable: Dict[str, str] = {}
    for error in body.get("errors") or []:
        path = error.get("path") or []
        accession_id = alias_to_id.get(path[0]) if path else None
        if accession_id and records.get(accession_id) is None:
            reason = unavailable_reason(str(error.get("message") or ""))
            if reason:
                unavailable[accession_id] = reason
    return records, unavailable


# ── Cleaning ──────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_WHITESPACE_RE = re.compile(r"\s+")


def decode_readme_bytes(raw: bytes) -> str:
    """Decode a README file from disk; a few OpenNeuro READMEs are UTF-16."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if len(raw) > 20 and raw[1:20:2] == b"\x00" * 10:
        # UTF-16 LE without a byte-order mark: every second byte is zero.
        return raw.decode("utf-16-le", errors="replace")
    return raw.decode("utf-8", errors="replace")


def clean_readme(raw: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """Return ``(searchable_text, urls)`` for a README.

    URLs are collected from the full text before anything is cut, because the
    references usually sit at the bottom.
    """
    if not raw:
        return None, []
    # PostgreSQL rejects NUL bytes in text values.
    cleaned = raw.replace("\x00", "")
    urls = list(dict.fromkeys(
        url.rstrip(").,;>]\"'") for url in _URL_RE.findall(cleaned)
    ))[:20]

    # Markdown images can carry base64 payloads; HTML tags are noise for search.
    cleaned = _MD_IMAGE_RE.sub("", cleaned)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = _BLANK_LINES_RE.sub("\n\n", cleaned).strip()
    if not cleaned:
        return None, urls

    if len(cleaned) > README_MAX_CHARS:
        half = README_MAX_CHARS // 2
        cleaned = cleaned[:half] + "\n…\n" + cleaned[-half:]
    return cleaned, urls


def _clean_line(value: Any) -> Optional[str]:
    """Normalise a short free-text field; empty values become None."""
    if value is None:
        return None
    line = _WHITESPACE_RE.sub(" ", str(value).replace("\x00", "")).strip()
    return line or None


def _age_bounds(ages: Any) -> Tuple[Optional[float], Optional[float]]:
    low, high = _AGE_RANGE
    plausible = [
        float(age) for age in (ages or [])
        if isinstance(age, (int, float)) and not isinstance(age, bool) and low <= age <= high
    ]
    if not plausible:
        return None, None
    return min(plausible), max(plausible)


def _size_bytes(snapshot: dict) -> Optional[int]:
    """Snapshot size in bytes. GraphQL BigInt may arrive as a number or a string."""
    summary = snapshot.get("summary") or {}
    for candidate in (summary.get("size"), snapshot.get("size")):
        try:
            size = int(candidate)
        except (TypeError, ValueError):
            continue
        if size > 0:
            return size
    return None


def _timestamp(value: Any) -> Optional[datetime]:
    """An ISO 8601 time from the API ('2016-10-12T23:40:22.670Z'), or None."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _count(value: Any) -> Optional[int]:
    """A non-negative integer count, or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return int(value)


def extract_fields(record: Optional[dict]) -> Dict[str, Any]:
    """Turn one GraphQL dataset record into column values.

    Every key of METADATA_COLUMNS is present in the result, plus ``readme_urls``
    (links found in the README). Missing information is None.
    """
    record = record or {}
    metadata = record.get("metadata") or {}
    snapshot = record.get("latestSnapshot") or {}

    readme_text, readme_urls = clean_readme(snapshot.get("readme"))
    age_min, age_max = _age_bounds(metadata.get("ages"))
    analytics = snapshot.get("analytics") or {}
    description = snapshot.get("description")  # None (absent or null) = unknown
    return {
        "readme_text":  readme_text,
        "dx_status":    _clean_line(metadata.get("dxStatus")),
        "study_domain": _clean_line(metadata.get("studyDomain")),
        "study_design": _clean_line(metadata.get("studyDesign")),
        "species":      _clean_line(metadata.get("species")),
        "study_longitudinal": _clean_line(metadata.get("studyLongitudinal")),
        "senior_author": _clean_line(metadata.get("seniorAuthor")),
        "publish_date": _timestamp(record.get("publishDate")),
        "on_brainlife": record.get("onBrainlife") if isinstance(record.get("onBrainlife"), bool) else None,
        "downloads":    _count(analytics.get("downloads")),
        "views":        _count(analytics.get("views")),
        "age_min":      age_min,
        "age_max":      age_max,
        "size_bytes":   _size_bytes(snapshot),
        "readme_urls":  readme_urls,
        # None: the record did not include the description (unknown, keep stored rows).
        "ethics_approvals": (clean_site_names(description.get("EthicsApprovals"))
                             if isinstance(description, dict) else None),
    }


def deprecation_reason(record: Optional[dict], unavailable: Optional[str] = None) -> Optional[str]:
    """Why a dataset should be shown as deprecated, or None if it is fine.

    Two upstream situations count:
      - OpenNeuro no longer serves the dataset; *unavailable* is the reason
        from fetch_batch (removed, deleted, or made private)
      - OpenNeuro flags its latest snapshot as deprecated, with a reason
    """
    if unavailable:
        return unavailable
    flag = ((record or {}).get("latestSnapshot") or {}).get("deprecated")
    if not flag:
        return None
    reason = _clean_line(flag.get("reason")) if isinstance(flag, dict) else None
    return f"Deprecated on OpenNeuro: {reason}" if reason else "Deprecated on OpenNeuro"


# ── Database ──────────────────────────────────────────────────────────────────

# A value the API did not provide never overwrites one that is already stored,
# so the update is safe to repeat and safe to run after a partial outage.
# README links fill paper_references only where the dataset declared none.
_UPDATE_SQL = text(
    """
    UPDATE bids_datasets SET
        readme_text  = COALESCE(:readme_text,  readme_text),
        dx_status    = COALESCE(:dx_status,    dx_status),
        study_domain = COALESCE(:study_domain, study_domain),
        study_design = COALESCE(:study_design, study_design),
        species      = COALESCE(:species,      species),
        study_longitudinal = COALESCE(:study_longitudinal, study_longitudinal),
        senior_author = COALESCE(:senior_author, senior_author),
        publish_date = COALESCE(:publish_date, publish_date),
        on_brainlife = COALESCE(:on_brainlife, on_brainlife),
        downloads    = COALESCE(:downloads,    downloads),
        views        = COALESCE(:views,        views),
        age_min      = COALESCE(:age_min,      age_min),
        age_max      = COALESCE(:age_max,      age_max),
        size_bytes   = COALESCE(:size_bytes,   size_bytes),
        paper_references = CASE
            WHEN COALESCE(cardinality(paper_references), 0) = 0
                 AND :readme_urls IS NOT NULL
            THEN :readme_urls
            ELSE paper_references
        END
    WHERE accession_id = :accession_id
    """
).bindparams(bindparam("readme_urls", type_=ARRAY(String)))


async def apply_fields(session: Any, accession_id: str, fields: Dict[str, Any]) -> int:
    """Write extracted fields to the dataset row(s) with this accession id.

    ``session`` is an AsyncSession or AsyncConnection; the caller commits.
    Returns the number of rows updated (0 when the accession is not indexed).
    """
    params: Dict[str, Any] = {name: fields.get(name) for name in METADATA_COLUMNS}
    params["readme_urls"] = list(fields.get("readme_urls") or []) or None
    params["accession_id"] = accession_id
    result = await session.execute(_UPDATE_SQL, params)
    # Ethics boards are site evidence (bids_dataset_sites, source 'ethics_board').
    # Only a record that included the description replaces the stored rows.
    if fields.get("ethics_approvals") is not None:
        await replace_sites(session, accession_id, SITE_ETHICS, fields["ethics_approvals"])
    return int(result.rowcount or 0)


async def set_deprecation(session: Any, accession_id: str, reason: Optional[str]) -> int:
    """Record the upstream status of a dataset. ``None`` clears the mark.

    Call this only when the status is known: the API returned the dataset, or
    said it does not exist. Never call it after a failed request.
    """
    result = await session.execute(
        text("UPDATE bids_datasets SET deprecated_reason = :reason WHERE accession_id = :accession_id"),
        {"reason": reason, "accession_id": accession_id},
    )
    return int(result.rowcount or 0)


# ── Snapshot ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Snapshot:
    """The latest snapshot of a dataset as the API reports it."""

    tag: Optional[str]
    hexsha: str
    created: Optional[datetime]


def snapshot_of(record: Optional[dict]) -> Optional[Snapshot]:
    """The latest snapshot of an API record; None without a commit hash."""
    snapshot = (record or {}).get("latestSnapshot") or {}
    hexsha = snapshot.get("hexsha")
    if not isinstance(hexsha, str) or not hexsha.strip():
        return None
    tag = snapshot.get("tag")
    return Snapshot(tag=tag if isinstance(tag, str) and tag else None,
                    hexsha=hexsha.strip(), created=_timestamp(snapshot.get("created")))


async def record_snapshot(session: Any, accession_id: str, snapshot: Optional[Snapshot],
                          indexer_version: int, indexed_at: datetime) -> int:
    """Note that the files of *accession_id* were indexed from *snapshot*.

    Call it only after the whole indexing succeeded. *snapshot* None (the API
    did not say) stores no hash, so the next cycle indexes the dataset again.
    The caller commits.
    """
    result = await session.execute(
        text("UPDATE bids_datasets SET snapshot_tag = :tag, snapshot_hexsha = :hexsha, "
             "files_indexed_at = :at, indexer_version = :version "
             "WHERE accession_id = :accession_id"),
        {"tag": snapshot.tag if snapshot else None,
         "hexsha": snapshot.hexsha if snapshot else None,
         "at": indexed_at, "version": indexer_version, "accession_id": accession_id},
    )
    return int(result.rowcount or 0)
