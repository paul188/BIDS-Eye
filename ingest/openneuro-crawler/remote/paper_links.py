"""
remote/paper_links.py
---------------------
Which papers a dataset names, and what each of them is to the dataset.

A dataset's own metadata often does not say who the participants were; the
paper it links usually does (see docs/plans/Paper enrichment plan.md). This
module reads paper identifiers out of the places uploaders put them and decides
the role of each linked paper. It does no network or database work:

    remote/paper_lookup.py   fetches title, abstract and keywords per paper
    remote/paper_store.py    writes papers, links and the searchable text

Where links come from, most trustworthy first:

    associated    OpenNeuro metadata.associatedPaperDOI ("papers published from
                  the lab that collected this dataset")
    references    BIDS ReferencesAndLinks (bids_datasets.paper_references)
    acknowledge   BIDS HowToAcknowledge
    readme        the README
    reuse_field   OpenNeuro metadata.openneuroPaperDOI ("papers published from
                  downloading this dataset"); uploaders often repeat their own
                  paper here

Roles:

    tool      a format, pipeline or toolbox paper (MNE-BIDS, fMRIPrep, …)
    primary   the paper that describes this dataset; the only role whose text
              is searched
    reuse     named only in the reuse field, by other authors
    related   anything else the dataset cites
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ── Identifiers ───────────────────────────────────────────────────────────────

KIND_DOI, KIND_PMID, KIND_PMCID, KIND_ARXIV = "doi", "pmid", "pmcid", "arxiv"

# Link sources in order of trust. A link found in several places keeps them all.
SOURCE_ASSOCIATED = "associated"
SOURCE_REFERENCES = "references"
SOURCE_ACKNOWLEDGE = "acknowledge"
SOURCE_README = "readme"
SOURCE_REUSE_FIELD = "reuse_field"
SOURCES: Tuple[str, ...] = (
    SOURCE_ASSOCIATED, SOURCE_REFERENCES, SOURCE_ACKNOWLEDGE, SOURCE_README, SOURCE_REUSE_FIELD,
)

ROLE_PRIMARY, ROLE_RELATED, ROLE_TOOL, ROLE_REUSE = "primary", "related", "tool", "reuse"

# DOIs of data, not of papers: OpenNeuro itself and the common data
# repositories. A dataset citing its own or a sibling deposit says nothing
# about participants.
DATA_DOI_PREFIXES: Tuple[str, ...] = (
    "10.18112/",   # OpenNeuro
    "10.5281/",    # Zenodo
    "10.17605/",   # OSF
    "10.6084/",    # figshare
    "10.5061/",    # Dryad
    "10.7910/",    # Harvard Dataverse
    "10.12751/",   # G-Node
    "10.15154/",   # NIMH Data Archive
    "10.48324/",   # DANDI
    "10.15387/",   # FCP-INDI
    "10.25452/",   # figshare (institutional)
    "10.60817/",   # EBRAINS
    "10.17632/",   # Mendeley Data
    "10.5524/",    # GigaDB
    "10.34770/",   # Princeton DataSpace
    "10.24352/",   # Magdeburg university data repository
    "10.58071/",   # institutional data repository
    "10.25493/",   # EBRAINS / Human Brain Project
)

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>|,]+)", re.IGNORECASE)
_PMID_RE = re.compile(
    r"(?:pubmed(?:\.ncbi\.nlm\.nih\.gov)?/|ncbi\.nlm\.nih\.gov/pubmed/|\bPMID\s*:?\s*)(\d{6,9})\b",
    re.IGNORECASE,
)
_PMCID_RE = re.compile(r"\b(PMC\d{5,9})\b", re.IGNORECASE)
_ARXIV_RE = re.compile(r"arxiv(?:\.org/(?:abs|pdf)/|\s*:\s*|\.)(\d{4}\.\d{4,5})", re.IGNORECASE)
_NATURE_RE = re.compile(r"nature\.com/articles/([a-z0-9][a-z0-9\-.]+[a-z0-9])", re.IGNORECASE)

# Things uploaders leave glued to a DOI: a second URL, page anchors, landing
# page suffixes.
_DOI_CUT_RE = re.compile(r"(?:https?:|file:|[#?&;]|%20|\\|\]\(|\]\[).*$", re.IGNORECASE)
_DOI_SUFFIX_RE = re.compile(
    r"(?:/(?:meta|full|abstract|pdf|epdf|fulltext|tables/\d+|figures/\d+)"
    r"|\.(?:full|abstract|short|long)(?:\.pdf)?|\.pdf|\.full-text)$",
    re.IGNORECASE,
)
# bioRxiv and medRxiv DOIs (10.1101, and 10.64898 since late 2025) never end
# in a version; "v2" comes from the page URL.
PREPRINT_SERVER_PREFIXES: Tuple[str, ...] = ("10.1101/", "10.64898/")
_BIORXIV_VERSION_RE = re.compile(r"^(10\.(?:1101|64898)/[0-9.]+?)v\d+$")


@dataclass(frozen=True, order=True)
class PaperId:
    kind: str
    value: str

    def __str__(self) -> str:
        return f"{self.kind}:{self.value}"


def clean_doi(raw: str) -> Optional[str]:
    """Normalise a DOI as found in free text. None when it is not a paper DOI."""
    doi = _DOI_CUT_RE.sub("", raw.strip()).lower()
    previous = None
    while doi != previous:
        previous = doi
        doi = doi.rstrip(".,:*_'\"`>([{")
        # A closing bracket belongs to the DOI only if it was opened inside it
        # (old Elsevier DOIs: 10.1016/s1053-8119(03)00123-4).
        for opener, closer in ("()", "[]", "{}"):
            while doi.endswith(closer) and doi.count(closer) > doi.count(opener):
                doi = doi[:-1]
        doi = _DOI_SUFFIX_RE.sub("", doi)
    match = _BIORXIV_VERSION_RE.match(doi)
    if match:
        doi = match.group(1)
    if doi.startswith("10.48550/arxiv."):
        return doi
    if not re.fullmatch(r"10\.\d{4,9}/\S{2,}", doi):
        return None
    if doi.startswith(DATA_DOI_PREFIXES):
        return None
    return doi


def _nature_doi(article_id: str) -> str:
    """DOI of a nature.com/articles/<id> link.

    Newer ids are the DOI suffix (s41597-019-0104-8). Older ones lost their
    dots in the URL: sdata20143 is 10.1038/sdata.2014.3.
    """
    article_id = article_id.lower()
    old_style = re.fullmatch(r"([a-z]+)((?:19|20)\d{2})(\d{1,4})", article_id)
    if old_style:
        return "10.1038/" + ".".join(old_style.groups())
    return "10.1038/" + article_id


def doi_fallbacks(doi: str) -> List[str]:
    """Shorter DOIs to try when `doi` does not resolve.

    Landing-page paths are sometimes copied along
    (10.1093/cercor/bhx202/4080827/how-we-transmit-memories). Each candidate
    drops one trailing path segment; at most two are returned, and never the
    bare prefix.
    """
    out: List[str] = []
    # eLife gives every part of an article its own DOI (….36493.001 is the abstract).
    part = re.fullmatch(r"(10\.7554/elife\.\d+)\.\d{3}", doi)
    if part:
        return [part.group(1)]
    current = doi
    for _ in range(2):
        head, _, _tail = current.rpartition("/")
        if head.count("/") < 1:
            break
        out.append(head)
        current = head
    return out


def extract_ids(text: Optional[str]) -> List[PaperId]:
    """Every paper identifier in a piece of text, in order of appearance."""
    if not text:
        return []
    found: List[Tuple[int, PaperId]] = []
    for match in _DOI_RE.finditer(text):
        doi = clean_doi(match.group(1))
        if doi is None:
            continue
        if doi.startswith("10.48550/arxiv."):
            found.append((match.start(), PaperId(KIND_ARXIV, doi[len("10.48550/arxiv."):])))
        else:
            found.append((match.start(), PaperId(KIND_DOI, doi)))
    for match in _NATURE_RE.finditer(text):
        found.append((match.start(), PaperId(KIND_DOI, _nature_doi(match.group(1)))))
    for match in _PMID_RE.finditer(text):
        found.append((match.start(), PaperId(KIND_PMID, match.group(1))))
    for match in _PMCID_RE.finditer(text):
        found.append((match.start(), PaperId(KIND_PMCID, match.group(1).upper())))
    for match in _ARXIV_RE.finditer(text):
        found.append((match.start(), PaperId(KIND_ARXIV, match.group(1))))
    seen: Set[PaperId] = set()
    ordered: List[PaperId] = []
    for _, paper_id in sorted(found, key=lambda item: item[0]):
        if paper_id not in seen:
            seen.add(paper_id)
            ordered.append(paper_id)
    return ordered


# ── Links of one dataset ──────────────────────────────────────────────────────

@dataclass
class Link:
    """One identifier a dataset names, with every place it was found."""
    paper_id: PaperId
    sources: List[str] = field(default_factory=list)
    raw: Optional[str] = None        # the reference string it was read from


def _as_strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def collect_links(
    *,
    associated: Any = None,
    references: Any = None,
    acknowledge: Any = None,
    readme: Any = None,
    reuse_field: Any = None,
) -> List[Link]:
    """All paper identifiers of one dataset, most trustworthy source first."""
    links: Dict[PaperId, Link] = {}
    for source, value in (
        (SOURCE_ASSOCIATED, associated),
        (SOURCE_REFERENCES, references),
        (SOURCE_ACKNOWLEDGE, acknowledge),
        (SOURCE_README, readme),
        (SOURCE_REUSE_FIELD, reuse_field),
    ):
        for chunk in _as_strings(value):
            for paper_id in extract_ids(chunk):
                link = links.get(paper_id)
                if link is None:
                    # README text is long; the reference string is only kept
                    # where it is a reference.
                    raw = chunk.strip()[:500] if source != SOURCE_README else None
                    link = links[paper_id] = Link(paper_id=paper_id, raw=raw)
                if source not in link.sources:
                    link.sources.append(source)
    return list(links.values())


# ── Roles ─────────────────────────────────────────────────────────────────────

# Papers about a format, pipeline or toolbox. Hand-kept: "cited by many
# datasets" is not a safe test, because the releases of one large study (the
# Healthy Brain Network, for instance) legitimately share one paper.
# Every DOI here was looked up and its title checked on 2026-11-13 (four DOIs
# written from memory turned out to be other papers). Do the same when adding one.
TOOL_PAPERS: Dict[str, str] = {
    "10.21105/joss.01896": "MNE-BIDS",
    "10.1038/sdata.2016.44": "BIDS",
    "10.1038/s41597-019-0104-8": "EEG-BIDS",
    "10.1038/sdata.2018.110": "MEG-BIDS",
    "10.1038/s41597-019-0105-7": "iEEG-BIDS",
    "10.1038/s41597-022-01164-1": "PET-BIDS",
    "10.1038/s41597-022-01615-9": "ASL-BIDS",
    "10.1038/s41597-022-01571-4": "qMRI-BIDS",
    "10.1038/s41597-024-04136-9": "NIRS-BIDS",
    "10.1038/s41592-018-0235-4": "fMRIPrep",
    "10.1371/journal.pone.0184661": "MRIQC",
    "10.3389/fnins.2013.00267": "MNE",
    "10.1016/j.neuroimage.2013.10.027": "MNE",
    "10.1016/j.jneumeth.2003.10.009": "EEGLAB",
    "10.1155/2011/156869": "FieldTrip",
    "10.1016/j.neuroimage.2012.01.021": "FreeSurfer",
    "10.1016/j.neuroimage.2011.09.015": "FSL",
    "10.3389/fninf.2011.00013": "Nipype",
    "10.1016/j.jneumeth.2016.03.001": "dcm2niix",
    "10.1371/journal.pcbi.1005209": "BIDS Apps",
    "10.7554/elife.71774": "OpenNeuro",
    "10.3389/fninf.2013.00012": "OpenfMRI",
}

# A title that reads like software or a standard, for shared papers not on
# the list.
_TOOL_TITLE_RE = re.compile(
    r"\b(toolbox|toolkit|software|pipeline|preprocessing|open[- ]source (?:tool|package|library|platform)"
    r"|brain imaging data structure|\bBIDS\b|python package|matlab (?:toolbox|package)|standard for)\b",
    re.IGNORECASE,
)
# A shared paper needs this many datasets before its title alone marks a tool.
TOOL_TITLE_MIN_DATASETS = 5

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and the of in on for to with from by during as at is are its their using use via "
    "study studies data dataset datasets task tasks human humans brain neural effect effects "
    "fmri mri eeg meg ieeg pet bold imaging functional structural resting state".split()
)


def _fold(value: str) -> str:
    """Lower-case ASCII skeleton of a name, for comparing surnames."""
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


_NAME_NOISE_RE = re.compile(
    r"\([^)]*\)|\[[^\]]*\]|\S+@\S+|\b(?:ph\.?d|m\.?d|dr|prof)\b\.?", re.IGNORECASE)


def _is_initials(token: str) -> bool:
    """"R.A.", "A L", "JMG": single letters, or up to four capitals run together."""
    words = re.split(r"[ .\-]+", token.strip(" .-"))
    words = [w for w in words if w]
    if not words:
        return True
    if all(len(w) == 1 for w in words):
        return True
    return len(words) == 1 and len(words[0]) <= 4 and words[0].isupper()


def surnames(names: Iterable[str]) -> Set[str]:
    """Surnames from author strings in the shapes uploaders and indexes use.

    "Poldrack, R.A." · "Russell A. Poldrack" · "Poldrack RA" · a whole author
    list in one string ("Chen, J., Leong, Y.C." or "Klaus Gramann, Thomas
    Töllner") · with roles or addresses attached ("Jonathan Touryan (data)").

    Each comma-separated piece that is not just initials is one person; the
    surname is its last word, or its first when initials trail it. Particles
    are not kept apart ("van der Meer" gives "meer"), which is fine for
    comparing two lists. Names under three letters are dropped: they match too
    easily.
    """
    out: Set[str] = set()
    for name in names:
        text = _NAME_NOISE_RE.sub(" ", unicodedata.normalize("NFKD", str(name)))
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        for piece in re.split(r"[,;&/]|\band\b", text):
            piece = re.sub(r"[^A-Za-z .\-']", " ", piece).strip(" .-'")
            if not piece or _is_initials(piece):
                continue
            words = piece.replace(".", " ").split()
            if not words:
                continue
            # "Poldrack RA", "van der Meer JN": trailing initials follow the surname.
            if len(words) > 1 and _is_initials(words[-1]) and len(words[-1]) <= 3:
                surname = words[-2].lower()
            else:
                surname = words[-1].lower()
            if len(surname) >= 3:
                out.add(surname)
    return out


def _content_words(text: str) -> Set[str]:
    return {w for w in _WORD_RE.findall(_fold(text)) if len(w) > 2 and w not in _STOPWORDS}


def title_overlap(dataset_name: str, paper_title: str) -> float:
    """Share of the dataset name's content words that the paper title has."""
    name_words = _content_words(dataset_name)
    if len(name_words) < 2:
        return 0.0
    return len(name_words & _content_words(paper_title)) / len(name_words)


def is_tool_paper(paper: Dict[str, Any], n_datasets: int = 1) -> Optional[str]:
    """Why this paper counts as a tool citation, or None."""
    for key in ("doi", "preprint_doi"):
        doi = (paper.get(key) or "").lower()
        if doi in TOOL_PAPERS:
            return f"tool list: {TOOL_PAPERS[doi]}"
    journal = (paper.get("journal") or "").lower()
    if "journal of open source software" in journal:
        return "published in the Journal of Open Source Software"
    title = paper.get("title") or ""
    if n_datasets >= TOOL_TITLE_MIN_DATASETS and _TOOL_TITLE_RE.search(title):
        return f"cited by {n_datasets} datasets and the title reads like a tool"
    return None


# Name overlap that counts as "the title describes this dataset".
TITLE_OVERLAP_MIN = 0.5
# Share of the shorter author list that must be on both, for a paper to count
# as the dataset's own on authorship alone. One shared name is not enough: a
# lab head is on every paper of the lab.
AUTHOR_SHARE_MIN = 0.5


def author_share(dataset_surnames: Set[str], paper_surnames: Set[str]) -> float:
    """Shared surnames as a share of the shorter of the two author lists."""
    if not dataset_surnames or not paper_surnames:
        return 0.0
    shared = len(dataset_surnames & paper_surnames)
    return shared / min(len(dataset_surnames), len(paper_surnames))


def assign_roles(
    *,
    accession: str,
    dataset_name: str,
    dataset_authors: Sequence[str],
    links: Sequence[Tuple[Link, Dict[str, Any]]],
    datasets_per_paper: Optional[Dict[str, int]] = None,
) -> List[Tuple[str, str]]:
    """Role and reason for each (link, paper) of one dataset, in input order.

    `paper` is the resolved record (see remote/paper_lookup.py); `key` in it
    identifies the paper across datasets. `datasets_per_paper` maps that key
    to the number of datasets citing the paper.

    The rules were checked against 60 hand-labelled links
    (docs/plans/Paper enrichment plan.md). What they cannot tell apart is two
    experiments of one lab with the same authors.
    """
    counts = datasets_per_paper or {}
    dataset_surnames = surnames(dataset_authors)
    roles: List[Optional[Tuple[str, str]]] = [None] * len(links)

    for index, (link, paper) in enumerate(links):
        reason = is_tool_paper(paper, counts.get(paper.get("key", ""), 1))
        if reason:
            roles[index] = (ROLE_TOOL, reason)

    candidates = [i for i, role in enumerate(roles) if role is None]
    for index in candidates:
        link, paper = links[index]
        paper_surnames = surnames(paper.get("authors") or [])
        shared = sorted(dataset_surnames & paper_surnames)
        share = author_share(dataset_surnames, paper_surnames)
        overlap = title_overlap(dataset_name, paper.get("title") or "")
        text = " ".join(str(paper.get(k) or "") for k in ("title", "abstract")).lower()
        mentions = accession.lower() in text

        if SOURCE_ASSOCIATED in link.sources:
            roles[index] = (ROLE_PRIMARY, "named in OpenNeuro's associated-paper field")
        elif link.sources == [SOURCE_REUSE_FIELD] and not shared:
            roles[index] = (ROLE_REUSE, "named only in OpenNeuro's reuse field, no shared author")
        elif shared and (overlap >= TITLE_OVERLAP_MIN or mentions):
            why = "mentions the accession" if mentions else f"title overlap {overlap:.2f}"
            roles[index] = (ROLE_PRIMARY, f"shares author {shared[0]}; {why}")
        elif shared and SOURCE_ACKNOWLEDGE in link.sources:
            roles[index] = (ROLE_PRIMARY, f"shares author {shared[0]}; the dataset asks to cite it")
        elif len(candidates) == 1 and (shared or not dataset_surnames):
            why = f"shares author {shared[0]}" if shared else "the dataset lists no authors"
            roles[index] = (ROLE_PRIMARY, f"the only non-tool paper the dataset names; {why}")
        elif share >= AUTHOR_SHARE_MIN:
            roles[index] = (ROLE_PRIMARY, f"author lists agree ({len(shared)} shared, share {share:.2f})")
        elif shared:
            roles[index] = (ROLE_RELATED, f"only {len(shared)} shared author(s) among several papers")
        elif len(candidates) == 1:
            roles[index] = (ROLE_RELATED, "the only paper, but no shared author")
        else:
            roles[index] = (ROLE_RELATED, "no shared author, not the only paper")

    return [role for role in roles if role is not None]
