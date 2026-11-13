"""
remote/paper_lookup.py
----------------------
Title, abstract and keywords of the papers datasets name.

Order of services (all free, none needs a key or an e-mail address; see
crawlers/paper_enrichment_plan.md for what was measured):

    Europe PMC   abstract, MeSH, author keywords, licence, open-access flag
                 and the published version of a preprint, in one call
    bioRxiv      preprints Europe PMC does not have, and their published DOI
    arXiv        arXiv preprints
    OpenAlex     DOIs outside biomedicine (conference papers, book chapters)

A preprint with a published version becomes ONE paper: it takes abstract and
MeSH from the published version and keeps the preprint DOI as an alias. The
same paper named by DOI, PubMed id and PMC id is also one paper.

Only the standard library is used, so the module runs in the crawler and in
the backend image.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from remote.paper_links import (
    KIND_ARXIV, KIND_DOI, KIND_PMCID, KIND_PMID, PREPRINT_SERVER_PREFIXES, PaperId, doi_fallbacks,
)

log = logging.getLogger(__name__)

# A generic client name. No contact address is sent to any service.
USER_AGENT = "bids-metadata-indexer/1.0"

EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
BIORXIV_DETAILS = "https://api.biorxiv.org/details/{server}/{doi}/na/json"
ARXIV_QUERY = "https://export.arxiv.org/api/query"
OPENALEX_WORK = "https://api.openalex.org/works/doi:{doi}"

# Identifiers per Europe PMC request (OR-joined; keeps the URL short).
EUROPEPMC_BATCH = 15
# Seconds between requests to one host. Europe PMC allows about 10 per second,
# arXiv asks for one request every three seconds.
_MIN_INTERVAL = {"www.ebi.ac.uk": 0.25, "api.biorxiv.org": 0.5, "export.arxiv.org": 3.0,
                 "api.openalex.org": 0.3}

ABSTRACT_MAX_CHARS = 8_000

Fetch = Callable[[str], Any]

_last_call: Dict[str, float] = {}


def http_get(url: str, *, timeout: float = 40.0, retries: int = 3) -> Any:
    """GET a URL and decode JSON; other content is returned as text.

    Returns None when the service answers 404 or keeps failing: a paper that
    cannot be looked up is simply not enriched.
    """
    host = urllib.parse.urlsplit(url).netloc
    for attempt in range(1, retries + 1):
        wait = _MIN_INTERVAL.get(host, 0.3) - (time.monotonic() - _last_call.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                kind = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            log.warning("%s → HTTP %s (attempt %d)", host, exc.code, attempt)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.warning("%s → %s (attempt %d)", host, exc, attempt)
        else:
            text = body.decode("utf-8", "replace")
            if "json" in kind or text[:1] in "{[":
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return None
            return text
        if attempt < retries:
            time.sleep(2 * attempt)
    return None


# ── Text clean-up ─────────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]{1,80}>")
_SPACE_RE = re.compile(r"\s+")


def clean_text(value: Any, limit: Optional[int] = None) -> Optional[str]:
    """Plain text from a title or abstract that may carry markup and entities."""
    if not value:
        return None
    text = str(value)
    # Some records are escaped twice ("&amp;lt;sup&amp;gt;").
    for _ in range(2):
        text = html.unescape(text)
    text = _TAG_RE.sub(" ", text).replace("\x00", "")
    text = _SPACE_RE.sub(" ", text).strip()
    if not text:
        return None
    return text[:limit] if limit else text


def _year(value: Any) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


def _new_paper(source: str) -> Dict[str, Any]:
    return {
        "key": None, "doi": None, "preprint_doi": None, "pmid": None, "pmcid": None,
        "arxiv_id": None, "title": None, "abstract": None, "year": None, "journal": None,
        "is_preprint": False, "authors": [], "mesh_terms": [], "mesh_major": [],
        "keywords": [], "is_open_access": None, "license": None, "source": source,
    }


def paper_key(paper: Dict[str, Any]) -> str:
    """Identity of a paper across datasets."""
    for kind, name in ((KIND_DOI, "doi"), (KIND_PMID, "pmid"), (KIND_PMCID, "pmcid"),
                       (KIND_ARXIV, "arxiv_id"), (KIND_DOI, "preprint_doi")):
        if paper.get(name):
            return f"{kind}:{paper[name]}"
    raise ValueError("paper without any identifier")


# ── Europe PMC ────────────────────────────────────────────────────────────────

def _europepmc_term(paper_id: PaperId) -> str:
    if paper_id.kind == KIND_DOI:
        return f'DOI:"{paper_id.value}"'
    if paper_id.kind == KIND_PMID:
        # SRC:PPR as well: PubMed indexes some preprints under a PubMed id.
        return f"(EXT_ID:{paper_id.value} AND (SRC:MED OR SRC:PPR))"
    if paper_id.kind == KIND_PMCID:
        return f"PMCID:{paper_id.value}"
    return f'DOI:"10.48550/arxiv.{paper_id.value}"'


def _europepmc_url(paper_ids: Sequence[PaperId]) -> str:
    query = " OR ".join(_europepmc_term(p) for p in paper_ids)
    return EUROPEPMC_SEARCH + "?" + urllib.parse.urlencode(
        {"query": query, "resultType": "core", "format": "json", "pageSize": 100}
    )


def parse_europepmc(record: Dict[str, Any]) -> Dict[str, Any]:
    """One Europe PMC "core" result as a paper."""
    paper = _new_paper("europepmc")
    is_preprint = record.get("source") == "PPR"
    doi = (record.get("doi") or "").lower() or None
    paper["is_preprint"] = is_preprint
    paper["preprint_doi" if is_preprint else "doi"] = doi
    if doi and doi.startswith("10.48550/arxiv."):
        paper["arxiv_id"] = doi[len("10.48550/arxiv."):]
    paper["pmid"] = record.get("pmid") or None
    paper["pmcid"] = record.get("pmcid") or None
    paper["title"] = clean_text(record.get("title"))
    paper["abstract"] = clean_text(record.get("abstractText"), ABSTRACT_MAX_CHARS)
    paper["year"] = _year(record.get("pubYear") or record.get("firstPublicationDate"))
    journal = ((record.get("journalInfo") or {}).get("journal") or {}).get("title")
    paper["journal"] = clean_text(journal or (record.get("bookOrReportDetails") or {}).get("publisher"))
    authors = (record.get("authorList") or {}).get("author") or []
    paper["authors"] = [a.get("fullName") or a.get("lastName") or a.get("collectiveName")
                        for a in authors if isinstance(a, dict)
                        and (a.get("fullName") or a.get("lastName") or a.get("collectiveName"))]
    for heading in (record.get("meshHeadingList") or {}).get("meshHeading") or []:
        name = heading.get("descriptorName")
        if name:
            paper["mesh_terms"].append(name)
            if heading.get("majorTopic_YN") == "Y":
                paper["mesh_major"].append(name)
    paper["keywords"] = [clean_text(k) for k in (record.get("keywordList") or {}).get("keyword") or []
                         if clean_text(k)]
    paper["is_open_access"] = record.get("isOpenAccess") == "Y" if record.get("isOpenAccess") else None
    paper["license"] = record.get("license") or None
    return paper


def published_version(record: Dict[str, Any]) -> Optional[PaperId]:
    """The PubMed record a Europe PMC preprint was published as, if any."""
    corrections = (record.get("commentCorrectionList") or {}).get("commentCorrection") or []
    for item in corrections:
        if item.get("type") == "Preprint of" and item.get("source") == "MED" and item.get("id"):
            return PaperId(KIND_PMID, str(item["id"]))
    return None


def _matches(paper_id: PaperId, record: Dict[str, Any]) -> bool:
    doi = (record.get("doi") or "").lower()
    if paper_id.kind == KIND_DOI:
        return doi == paper_id.value
    if paper_id.kind == KIND_PMID:
        return paper_id.value in (str(record.get("pmid") or ""), str(record.get("id") or "")) \
            and record.get("source") in ("MED", "PPR")
    if paper_id.kind == KIND_PMCID:
        return (record.get("pmcid") or "").upper() == paper_id.value
    return doi == f"10.48550/arxiv.{paper_id.value}"


def _best_record(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # The PubMed record carries MeSH; prefer it over other sources of the same paper.
    return max(records, key=lambda r: (r.get("source") == "MED", bool(r.get("abstractText"))))


def europepmc_lookup(paper_ids: Sequence[PaperId], fetch: Fetch) -> Dict[PaperId, Dict[str, Any]]:
    """Raw Europe PMC records for the identifiers it knows."""
    found: Dict[PaperId, Dict[str, Any]] = {}
    ids = list(dict.fromkeys(paper_ids))
    for start in range(0, len(ids), EUROPEPMC_BATCH):
        chunk = ids[start:start + EUROPEPMC_BATCH]
        body = fetch(_europepmc_url(chunk))
        results = ((body or {}).get("resultList") or {}).get("result") or [] if isinstance(body, dict) else []
        for paper_id in chunk:
            matching = [r for r in results if _matches(paper_id, r)]
            if matching:
                found[paper_id] = _best_record(matching)
    return found


# ── bioRxiv / medRxiv ─────────────────────────────────────────────────────────

def biorxiv_lookup(doi: str, fetch: Fetch) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(preprint as a paper, DOI of the published version) for a preprint-server DOI."""
    for server in ("biorxiv", "medrxiv"):
        body = fetch(BIORXIV_DETAILS.format(server=server, doi=doi))
        versions = (body or {}).get("collection") if isinstance(body, dict) else None
        if not versions:
            continue
        latest = versions[-1]
        paper = _new_paper(server)
        paper.update(
            preprint_doi=doi, is_preprint=True,
            title=clean_text(latest.get("title")),
            abstract=clean_text(latest.get("abstract"), ABSTRACT_MAX_CHARS),
            year=_year(latest.get("date")), journal=server,
            authors=[a.strip() for a in str(latest.get("authors") or "").split(";") if a.strip()],
            is_open_access=True, license=latest.get("license") or None,
        )
        published = str(latest.get("published") or "").strip().lower()
        return paper, (published if published.startswith("10.") else None)
    return None, None


# ── arXiv ─────────────────────────────────────────────────────────────────────

_ATOM = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


def arxiv_lookup(arxiv_ids: Sequence[str], fetch: Fetch) -> Dict[str, Tuple[Dict[str, Any], Optional[str]]]:
    """{arXiv id: (paper, DOI of the published version)}."""
    out: Dict[str, Tuple[Dict[str, Any], Optional[str]]] = {}
    ids = list(dict.fromkeys(arxiv_ids))
    for start in range(0, len(ids), 50):
        chunk = ids[start:start + 50]
        body = fetch(ARXIV_QUERY + "?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)}))
        if not isinstance(body, str):
            continue
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            continue
        for entry in root.findall("a:entry", _ATOM):
            match = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", entry.findtext("a:id", "", _ATOM))
            if not match:
                continue
            paper = _new_paper("arxiv")
            paper.update(
                arxiv_id=match.group(1), preprint_doi=f"10.48550/arxiv.{match.group(1)}",
                is_preprint=True, title=clean_text(entry.findtext("a:title", "", _ATOM)),
                abstract=clean_text(entry.findtext("a:summary", "", _ATOM), ABSTRACT_MAX_CHARS),
                year=_year(entry.findtext("a:published", "", _ATOM)), journal="arXiv",
                authors=[a.findtext("a:name", "", _ATOM) for a in entry.findall("a:author", _ATOM)],
                is_open_access=True,
            )
            published = (entry.findtext("x:doi", "", _ATOM) or "").strip().lower()
            out[match.group(1)] = (paper, published or None)
    return out


# ── OpenAlex ──────────────────────────────────────────────────────────────────

def abstract_from_inverted_index(index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """OpenAlex ships abstracts as {word: [positions]}; put the words back in order."""
    if not index:
        return None
    positions: List[Tuple[int, str]] = []
    for word, places in index.items():
        positions.extend((place, word) for place in places)
    return clean_text(" ".join(word for _, word in sorted(positions)), ABSTRACT_MAX_CHARS)


def openalex_lookup(doi: str, fetch: Fetch) -> Optional[Dict[str, Any]]:
    body = fetch(OPENALEX_WORK.format(doi=urllib.parse.quote(doi, safe="/")))
    if not isinstance(body, dict) or not body.get("id"):
        return None
    if body.get("type") in ("dataset", "reference-entry", "paratext"):
        return None
    paper = _new_paper("openalex")
    canonical = (body.get("doi") or "").replace("https://doi.org/", "").lower() or doi
    is_preprint = body.get("type") == "preprint"
    paper["preprint_doi" if is_preprint else "doi"] = canonical
    # OpenAlex merges a preprint into its published version: the DOI asked for
    # may differ from the one returned.
    if canonical != doi and not is_preprint:
        paper["preprint_doi"] = doi
    ids = body.get("ids") or {}
    paper.update(
        is_preprint=is_preprint,
        pmid=(ids.get("pmid") or "").rsplit("/", 1)[-1] or None,
        pmcid=(ids.get("pmcid") or "").rsplit("/", 1)[-1] or None,
        title=clean_text(body.get("title")),
        abstract=abstract_from_inverted_index(body.get("abstract_inverted_index")),
        year=body.get("publication_year"),
        journal=clean_text(((body.get("primary_location") or {}).get("source") or {}).get("display_name")),
        authors=[(a.get("author") or {}).get("display_name") for a in body.get("authorships") or []
                 if (a.get("author") or {}).get("display_name")],
        mesh_terms=[m["descriptor_name"] for m in body.get("mesh") or [] if m.get("descriptor_name")],
        mesh_major=[m["descriptor_name"] for m in body.get("mesh") or []
                    if m.get("descriptor_name") and m.get("is_major_topic")],
        keywords=[k["display_name"] for k in body.get("keywords") or [] if k.get("display_name")],
        is_open_access=(body.get("open_access") or {}).get("is_oa"),
        license=(body.get("primary_location") or {}).get("license"),
    )
    return paper


# ── Resolve ───────────────────────────────────────────────────────────────────

def _merge_preprint(published: Dict[str, Any], preprint: Dict[str, Any]) -> Dict[str, Any]:
    """The published paper, remembering its preprint and borrowing what it lacks."""
    merged = dict(published)
    merged["preprint_doi"] = merged.get("preprint_doi") or preprint.get("preprint_doi")
    merged["arxiv_id"] = merged.get("arxiv_id") or preprint.get("arxiv_id")
    for name in ("abstract", "title", "year", "journal"):
        merged[name] = merged.get(name) or preprint.get(name)
    if not merged.get("authors"):
        merged["authors"] = preprint.get("authors") or []
    return merged


def resolve(
    paper_ids: Iterable[PaperId],
    fetch: Fetch = http_get,
    *,
    use_openalex: bool = True,
) -> Tuple[Dict[PaperId, str], Dict[str, Dict[str, Any]]]:
    """Look every identifier up.

    Returns (which paper each identifier is, papers by key). Identifiers no
    service knows are missing from the first mapping.
    """
    wanted = list(dict.fromkeys(paper_ids))
    assigned: Dict[PaperId, Dict[str, Any]] = {}

    # 1. Europe PMC, with one retry for DOIs that carry a landing-page path.
    records = europepmc_lookup(wanted, fetch)
    retry: Dict[PaperId, PaperId] = {}
    for paper_id in wanted:
        if paper_id not in records and paper_id.kind == KIND_DOI:
            for shorter in doi_fallbacks(paper_id.value):
                retry.setdefault(PaperId(KIND_DOI, shorter), paper_id)
    if retry:
        for shorter_id, record in europepmc_lookup(list(retry), fetch).items():
            records.setdefault(retry[shorter_id], record)

    # 2. Preprints that were published: take the published record.
    published_ids = {pid: published_version(rec) for pid, rec in records.items()
                     if rec.get("source") == "PPR"}
    published_records = europepmc_lookup([p for p in published_ids.values() if p], fetch)
    for paper_id, record in records.items():
        paper = parse_europepmc(record)
        target = published_ids.get(paper_id)
        if target and target in published_records:
            paper = _merge_preprint(parse_europepmc(published_records[target]), paper)
        assigned[paper_id] = paper

    # 3. What Europe PMC does not have.
    missing = [p for p in wanted if p not in assigned]
    follow_up: Dict[PaperId, Tuple[Dict[str, Any], PaperId]] = {}

    arxiv_ids = [p.value for p in missing if p.kind == KIND_ARXIV]
    if arxiv_ids:
        for arxiv_id, (paper, published_doi) in arxiv_lookup(arxiv_ids, fetch).items():
            paper_id = PaperId(KIND_ARXIV, arxiv_id)
            assigned[paper_id] = paper
            if published_doi:
                follow_up[paper_id] = (paper, PaperId(KIND_DOI, published_doi))

    for paper_id in missing:
        if paper_id.kind == KIND_DOI and paper_id.value.startswith(PREPRINT_SERVER_PREFIXES):
            paper, published_doi = biorxiv_lookup(paper_id.value, fetch)
            if paper:
                assigned[paper_id] = paper
                if published_doi:
                    follow_up[paper_id] = (paper, PaperId(KIND_DOI, published_doi))

    if follow_up:
        targets = europepmc_lookup([target for _, target in follow_up.values()], fetch)
        for paper_id, (preprint, target) in follow_up.items():
            if target in targets:
                assigned[paper_id] = _merge_preprint(parse_europepmc(targets[target]), preprint)

    if use_openalex:
        for paper_id in missing:
            if paper_id in assigned or paper_id.kind != KIND_DOI:
                continue
            for doi in (paper_id.value, *doi_fallbacks(paper_id.value)):
                paper = openalex_lookup(doi, fetch)
                if paper:
                    assigned[paper_id] = paper
                    break

    # 4. One paper per key; identifiers that mean the same paper share it.
    papers: Dict[str, Dict[str, Any]] = {}
    by_alias: Dict[str, str] = {}
    key_of: Dict[PaperId, str] = {}
    for paper_id, paper in assigned.items():
        aliases = [f"{kind}:{paper[name]}" for kind, name in (
            (KIND_DOI, "doi"), (KIND_PMID, "pmid"), (KIND_PMCID, "pmcid"),
            (KIND_ARXIV, "arxiv_id"), (KIND_DOI, "preprint_doi")) if paper.get(name)]
        if not aliases or not paper.get("title"):
            continue
        key = next((by_alias[a] for a in aliases if a in by_alias), None)
        if key is None:
            key = paper_key(paper)
            papers[key] = paper
        else:
            papers[key] = _richer(papers[key], paper)
        papers[key]["key"] = key
        for alias in aliases:
            by_alias.setdefault(alias, key)
        key_of[paper_id] = key
    return key_of, papers


def _richer(first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any]:
    """Two records of one paper: keep the published one, fill its gaps."""
    base, other = (first, second) if not first.get("is_preprint") or second.get("is_preprint") \
        else (second, first)
    merged = dict(base)
    for name, value in other.items():
        if merged.get(name) in (None, "", []):
            merged[name] = value
    return merged
