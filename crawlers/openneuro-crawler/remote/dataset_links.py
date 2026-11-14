"""
remote/dataset_links.py
-----------------------
Linked tables for things a dataset has several of, each with its own source.

  bids_derivatives     processed versions of a dataset, one row per pipeline:
                         'openneuro_derivatives'  OpenNeuro's own preprocessing,
                             public bucket s3://openneuro-derivatives
                             (fmriprep, mriqc, fitlins, xcpd)
                         'in_dataset'  derivatives/<pipeline>/ folders the
                             uploader shipped inside the dataset
  bids_dataset_sites   where the data was collected, one row per name and source:
                         'dicom'         InstitutionName in the scan sidecars
                                         (copied from bids_datasets.institutions)
                         'ethics_board'  EthicsApprovals in dataset_description.json
                         Later sources (paper affiliations, ROR ids) fit the same rows.

Rows are replaced per (dataset, source) whenever that source was read
successfully, so they describe the present state. A failed read raises and
leaves the old rows alone.

Why a table and not a bids_datasets column: a dataset has several of each,
and each carries its own facts (pipeline version, subject count, where the
evidence came from), which queries filter and count on.

OpenNeuro's GraphQL `derivatives` field is not used: it returned the three
derivatives of ds000001 once and an empty list on the next calls (Nov 2026).
The bucket listing is the stable source.

Only the standard library and SQLAlchemy are needed.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import text

log = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

TABLES_SQL: Tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS bids_derivatives (
        id               BIGSERIAL PRIMARY KEY,
        dataset_id       UUID NOT NULL REFERENCES bids_datasets(id) ON DELETE CASCADE,
        pipeline         TEXT NOT NULL,
        pipeline_name    TEXT,
        pipeline_version TEXT,
        source           TEXT NOT NULL,
        location         TEXT NOT NULL,
        n_subjects       INTEGER,
        fetched_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (dataset_id, source, location)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_bids_derivatives_dataset ON bids_derivatives (dataset_id)",
    "CREATE INDEX IF NOT EXISTS idx_bids_derivatives_pipeline ON bids_derivatives (pipeline)",
    """CREATE TABLE IF NOT EXISTS bids_dataset_sites (
        id          BIGSERIAL PRIMARY KEY,
        dataset_id  UUID NOT NULL REFERENCES bids_datasets(id) ON DELETE CASCADE,
        name_raw    TEXT NOT NULL,
        source      TEXT NOT NULL,
        ror_id      TEXT,
        evidence    TEXT,
        UNIQUE (dataset_id, source, name_raw)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_bids_dataset_sites_dataset ON bids_dataset_sites (dataset_id)",
    "CREATE INDEX IF NOT EXISTS idx_bids_dataset_sites_source ON bids_dataset_sites (source)",
)

SOURCE_OPENNEURO_DERIVATIVES = "openneuro_derivatives"
SOURCE_IN_DATASET = "in_dataset"
SITE_DICOM = "dicom"
SITE_ETHICS = "ethics_board"

# A free-text ethics statement longer than this is cut; it is evidence, not prose.
_MAX_SITE_CHARS = 500


async def ensure_tables(conn: Any) -> None:
    """Create both tables if they are missing. Caller commits."""
    for statement in TABLES_SQL:
        await conn.execute(text(statement))


# ── Pipeline names ────────────────────────────────────────────────────────────

# Spelling variants of the same tool. Keys and values are normalised forms.
_PIPELINE_ALIASES = {
    "xcp-d": "xcpd", "xcp": "xcpd",
    "free-surfer": "freesurfer", "fs": "freesurfer",
    "f-mriprep": "fmriprep", "fmri-prep": "fmriprep",
    "qsi-prep": "qsiprep", "smri-prep": "smriprep",
}
# Only after a separator: 'cat12' and 'spm12' are names, 'fmriprep-20.2.3' has a version.
_VERSION_TAIL_RE = re.compile(r"[-_. ]v?\d+(?:[._]\d+)*(?:[-_.]?(?:rc|a|b|post|dev)\d*)?$", re.I)
UNSPECIFIED = "unspecified"
KNOWN_PIPELINES = frozenset("""
    fmriprep smriprep nibabies qsiprep qsirecon aslprep mriqc xcpd fitlins tedana
    freesurfer fastsurfer cat12 spm fsl afni ants mrtrix mrtrix3 hcp
    eeglab mne brainstorm fieldtrip
""".split())


def normalise_pipeline(name: Optional[str]) -> str:
    """'fMRIPrep' / 'fmriprep-20.2.3' / 'fmriprep_v23' → 'fmriprep'; '' → 'unspecified'."""
    value = (name or "").strip().lower()
    value = re.sub(r"[\s_]+", "-", value)
    previous = None
    while previous != value:
        previous = value
        value = _VERSION_TAIL_RE.sub("", value).strip("-.")
    value = _PIPELINE_ALIASES.get(value, value)
    # 'freesurfer-subjdir', 'fmriprep-ica' → the tool; 'fs-stats' stays (fs is too short to tell).
    head = value.split("-", 1)[0]
    if head in KNOWN_PIPELINES:
        value = head
    return value or UNSPECIFIED


@dataclass(frozen=True)
class Derivative:
    pipeline: str                    # normalised
    location: str                    # s3://… or derivatives/<folder>
    pipeline_name: Optional[str] = None
    pipeline_version: Optional[str] = None
    n_subjects: Optional[int] = None


def generated_by(description: Any) -> Tuple[Optional[str], Optional[str]]:
    """(Name, Version) of the first GeneratedBy entry (or the old PipelineDescription)."""
    if not isinstance(description, dict):
        return None, None
    entries = description.get("GeneratedBy")
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list) or not entries:
        legacy = description.get("PipelineDescription")
        entries = [legacy] if isinstance(legacy, dict) else []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("Name"):
            version = entry.get("Version")
            return str(entry["Name"]).strip() or None, (str(version).strip() or None) if version else None
    return None, None


# ── HTTP ──────────────────────────────────────────────────────────────────────

_USER_AGENT = "bids-metadata-indexer/1.0"


class LinkSourceError(RuntimeError):
    """A source could not be read; the stored rows must stay as they are."""


def _get(url: str, timeout: float = 60.0, retries: int = 3) -> Optional[bytes]:
    """Body of *url*, None on 404. Raises LinkSourceError when every attempt fails."""
    last: Optional[Exception] = None
    for _ in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404):
                return None
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
    raise LinkSourceError(f"GET {url} failed: {last}")


# ── OpenNeuro's own derivatives (public S3 bucket) ────────────────────────────

DERIVATIVES_BUCKET = "openneuro-derivatives"
_BUCKET_URL = f"https://{DERIVATIVES_BUCKET}.s3.amazonaws.com"
BUCKET_PIPELINES: Tuple[str, ...] = ("fmriprep", "mriqc", "fitlins", "xcpd")
_ACCESSION_RE = re.compile(r"^(ds\d{6})-")


def _list_prefixes(prefix: str) -> Tuple[List[str], List[str]]:
    """(sub-prefixes, keys) directly under *prefix*, following continuation tokens."""
    prefixes: List[str] = []
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
        if token:
            query["continuation-token"] = token
        body = _get(f"{_BUCKET_URL}/?{urllib.parse.urlencode(query)}")
        if body is None:
            raise LinkSourceError(f"bucket listing refused for {prefix!r}")
        xml = body.decode("utf-8", "replace")
        prefixes += re.findall(r"<CommonPrefixes><Prefix>([^<]+)</Prefix></CommonPrefixes>", xml)
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        match = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not match:
            return prefixes, keys
        token = match.group(1)


def list_bucket_repositories() -> Dict[str, List[str]]:
    """{accession: [repository prefix, …]} for every repository in the bucket."""
    found: Dict[str, List[str]] = {}
    for pipeline in BUCKET_PIPELINES:
        prefixes, _ = _list_prefixes(f"{pipeline}/")
        for prefix in prefixes:
            repository = prefix[len(pipeline) + 1:].rstrip("/")
            match = _ACCESSION_RE.match(repository)
            if match:
                found.setdefault(match.group(1), []).append(prefix)
    return found


def _bucket_repositories_for(accession_id: str) -> List[str]:
    """Repository prefixes of one dataset (one listing per pipeline)."""
    found = []
    for pipeline in BUCKET_PIPELINES:
        prefix = f"{pipeline}/{accession_id}-{pipeline}/"
        prefixes, keys = _list_prefixes(prefix)
        if prefixes or keys:
            found.append(prefix)
    return found


def describe_bucket_repository(prefix: str) -> Derivative:
    """Pipeline, version and subject count of one repository in the bucket."""
    folder = prefix.split("/")[0]
    subfolders, _ = _list_prefixes(prefix)
    n_subjects = sum(1 for p in subfolders if p[len(prefix):].startswith("sub-"))
    raw = _get(f"{_BUCKET_URL}/{prefix}dataset_description.json")
    description: Any = None
    if raw:
        try:
            description = json.loads(raw.decode("utf-8-sig"))
        except ValueError:
            description = None
    name, version = generated_by(description)
    return Derivative(
        pipeline=normalise_pipeline(name or folder),
        location=f"s3://{DERIVATIVES_BUCKET}/{prefix}",
        pipeline_name=name or folder,
        pipeline_version=version,
        n_subjects=n_subjects or None,
    )


def openneuro_derivatives(accession_ids: Sequence[str], workers: int = 8,
                          repositories: Optional[Dict[str, List[str]]] = None,
                          ) -> Dict[str, List[Derivative]]:
    """{accession: [Derivative, …]} for *accession_ids*; datasets without any get [].

    *repositories* is a prior list_bucket_repositories() result (a full
    backfill lists the bucket once instead of once per dataset).
    """
    if repositories is None:
        repositories = {a: _bucket_repositories_for(a) for a in accession_ids}
    work = [(a, p) for a in accession_ids for p in repositories.get(a, [])]
    result: Dict[str, List[Derivative]] = {a: [] for a in accession_ids}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (accession, _), derivative in zip(work, pool.map(lambda w: describe_bucket_repository(w[1]), work)):
            result[accession].append(derivative)
    return result


# ── Derivatives inside the dataset (OpenNeuro file tree) ──────────────────────

def _post_graphql(query: str, variables: dict) -> dict:
    from remote.openneuro_metadata import post_graphql
    return post_graphql(query, variables)


def _list_trees(requests: Sequence[Tuple[str, Optional[str]]]) -> List[Optional[List[dict]]]:
    """Entries of several folders in one request: [(accession, tree id or None for the root)].

    An entry is {id, filename, directory, urls}. None for a dataset the API did not return.
    """
    if not requests:
        return []
    declarations, selections, variables = [], [], {}
    for i, (accession, tree) in enumerate(requests):
        declarations.append(f"$id{i}: ID!")
        variables[f"id{i}"] = accession
        argument = ""
        if tree:
            declarations.append(f"$t{i}: String")
            variables[f"t{i}"] = tree
            argument = f"(tree: $t{i})"
        selections.append(
            f"d{i}: dataset(id: $id{i}) {{ latestSnapshot {{ files{argument} "
            f"{{ id filename directory urls }} }} }}")
    query = f"query Trees({', '.join(declarations)}) {{\n" + "\n".join(selections) + "\n}"
    data = _post_graphql(query, variables).get("data") or {}
    out: List[Optional[List[dict]]] = []
    for i in range(len(requests)):
        snapshot = ((data.get(f"d{i}") or {}).get("latestSnapshot")) or None
        out.append(None if snapshot is None else list(snapshot.get("files") or []))
    return out


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _is_subject(name: str) -> bool:
    return name.startswith("sub-")


def in_dataset_derivatives(accession_ids: Sequence[str], batch_size: int = 8,
                           ) -> Dict[str, List[Derivative]]:
    """{accession: [Derivative, …]} from each dataset's derivatives/ folder.

    One folder per pipeline is the BIDS layout (derivatives/fmriprep/…).
    Subject folders or files directly under derivatives/ become one
    'unspecified' row. Datasets the API did not return are left out of the
    result, so their stored rows stay.
    """
    result: Dict[str, List[Derivative]] = {}
    derivative_roots: List[Tuple[str, str]] = []
    for batch in _chunks(list(accession_ids), batch_size):
        for accession, entries in zip(batch, _list_trees([(a, None) for a in batch])):
            if entries is None:
                continue
            result[accession] = []
            root = next((e for e in entries if e.get("directory") and e.get("filename") == "derivatives"), None)
            if root:
                derivative_roots.append((accession, root["id"]))

    pipelines: List[Tuple[str, dict]] = []
    for batch in _chunks(derivative_roots, batch_size):
        for (accession, _), entries in zip(batch, _list_trees(batch)):
            if entries is None:
                result.pop(accession, None)
                continue
            folders = [e for e in entries if e.get("directory") and not e["filename"].startswith(".")]
            loose = [e for e in entries if not e.get("directory") and not e["filename"].startswith(".")]
            subjects = [e for e in folders if _is_subject(e["filename"])]
            if subjects or (loose and not folders):
                result[accession].append(Derivative(
                    pipeline=UNSPECIFIED, location="derivatives",
                    n_subjects=len(subjects) or None))
            pipelines += [(accession, e) for e in folders if not _is_subject(e["filename"])]

    for batch in _chunks(pipelines, batch_size):
        listings = _list_trees([(accession, folder["id"]) for accession, folder in batch])
        for (accession, folder), entries in zip(batch, listings):
            if accession not in result:
                continue
            entries = entries or []
            n_subjects = sum(1 for e in entries if e.get("directory") and _is_subject(e["filename"]))
            name = version = None
            description = next((e for e in entries if e.get("filename") == "dataset_description.json"), None)
            for url in (description or {}).get("urls") or []:
                try:
                    raw = _get(url, timeout=30, retries=2)
                except LinkSourceError:
                    raw = None
                if raw:
                    try:
                        name, version = generated_by(json.loads(raw.decode("utf-8-sig")))
                    except ValueError:
                        pass
                    break
            result[accession].append(Derivative(
                pipeline=normalise_pipeline(name or folder["filename"]),
                location=f"derivatives/{folder['filename']}",
                pipeline_name=name or folder["filename"],
                pipeline_version=version,
                n_subjects=n_subjects or None,
            ))
    return result


# ── Writes ────────────────────────────────────────────────────────────────────

_DELETE_DERIVATIVES = text(
    "DELETE FROM bids_derivatives x USING bids_datasets d "
    "WHERE x.dataset_id = d.id AND d.accession_id = :accession AND x.source = :source")
_INSERT_DERIVATIVE = text(
    "INSERT INTO bids_derivatives (dataset_id, pipeline, pipeline_name, pipeline_version, "
    "  source, location, n_subjects) "
    "SELECT d.id, :pipeline, :pipeline_name, :pipeline_version, :source, :location, :n_subjects "
    "FROM bids_datasets d WHERE d.accession_id = :accession "
    "ON CONFLICT (dataset_id, source, location) DO NOTHING")


async def replace_derivatives(conn: Any, accession_id: str, source: str,
                              derivatives: Sequence[Derivative]) -> int:
    """Make *derivatives* the dataset's rows for *source*. Caller commits."""
    await conn.execute(_DELETE_DERIVATIVES, {"accession": accession_id, "source": source})
    for d in derivatives:
        await conn.execute(_INSERT_DERIVATIVE, {
            "accession": accession_id, "source": source, "pipeline": d.pipeline,
            "pipeline_name": d.pipeline_name, "pipeline_version": d.pipeline_version,
            "location": d.location, "n_subjects": d.n_subjects,
        })
    return len(derivatives)


def clean_site_names(values: Any) -> List[str]:
    """Distinct, trimmed names from a string or a list of strings."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    names: List[str] = []
    for value in values if isinstance(values, (list, tuple)) else []:
        if not isinstance(value, str):
            continue
        name = re.sub(r"\s+", " ", value.replace("\x00", "")).strip()[:_MAX_SITE_CHARS]
        if name and name.lower() not in ("n/a", "na", "none", "-") and name not in names:
            names.append(name)
    return names


_DELETE_SITES = text(
    "DELETE FROM bids_dataset_sites s USING bids_datasets d "
    "WHERE s.dataset_id = d.id AND d.accession_id = :accession AND s.source = :source")
_INSERT_SITE = text(
    "INSERT INTO bids_dataset_sites (dataset_id, name_raw, source) "
    "SELECT d.id, :name, :source FROM bids_datasets d WHERE d.accession_id = :accession "
    "ON CONFLICT (dataset_id, source, name_raw) DO NOTHING")


async def replace_sites(conn: Any, accession_id: str, source: str, names: Sequence[str]) -> int:
    """Make *names* the dataset's site rows for *source*. Caller commits."""
    await conn.execute(_DELETE_SITES, {"accession": accession_id, "source": source})
    for name in names:
        await conn.execute(_INSERT_SITE, {"accession": accession_id, "source": source, "name": name})
    return len(names)


_REFRESH_DICOM_SITES = (
    text("DELETE FROM bids_dataset_sites s USING bids_datasets d "
         "WHERE s.dataset_id = d.id AND d.accession_id = ANY(:accessions) AND s.source = 'dicom'"),
    text("INSERT INTO bids_dataset_sites (dataset_id, name_raw, source) "
         "SELECT DISTINCT d.id, btrim(n), 'dicom' FROM bids_datasets d, unnest(d.institutions) n "
         "WHERE d.accession_id = ANY(:accessions) AND btrim(n) <> '' "
         "ON CONFLICT (dataset_id, source, name_raw) DO NOTHING"),
)


async def refresh_dicom_sites(conn: Any, accession_ids: Sequence[str]) -> None:
    """Copy bids_datasets.institutions into site rows with source 'dicom'. Caller commits."""
    for statement in _REFRESH_DICOM_SITES:
        await conn.execute(statement, {"accessions": list(accession_ids)})


async def refresh_links(conn: Any, accession_ids: Sequence[str]) -> Dict[str, int]:
    """Derivatives (both sources) and dicom sites for *accession_ids*. Caller commits.

    Blocking HTTP runs in a thread. A source that cannot be read is logged and
    skipped; its stored rows stay.
    """
    import asyncio

    accession_ids = [a for a in accession_ids if a]
    counts = {"openneuro_derivatives": 0, "in_dataset": 0}
    await ensure_tables(conn)
    for source, fetch in ((SOURCE_OPENNEURO_DERIVATIVES, openneuro_derivatives),
                          (SOURCE_IN_DATASET, in_dataset_derivatives)):
        try:
            found = await asyncio.to_thread(fetch, accession_ids)
        except Exception as exc:  # network, API shape — never fail the crawl
            log.warning("[links] %s not read: %s", source, exc)
            continue
        for accession, derivatives in found.items():
            counts[source] += await replace_derivatives(conn, accession, source, derivatives)
    await refresh_dicom_sites(conn, accession_ids)
    return counts
