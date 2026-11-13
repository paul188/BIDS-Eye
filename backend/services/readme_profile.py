"""
services/readme_profile.py
--------------------------
Read a dataset's README once, at index time, and store what it says.

An LLM reads the name, the uploader's study fields and the README and returns
  - a short profile in standard terminology (what was studied, in whom, how),
  - the conditions it mentions, each with its role in the study
    (patients, animal model, controls, excluded, measured, mentioned),
  - species, age groups, research topics and paradigms.

Every condition and paradigm comes with a verbatim quote from the input. A
fact whose quote cannot be found in the input is kept for review but marked
unverified and never used for search.

What search uses (crawlers/migrate_add_readme_profiles.sql):
  bids_datasets.llm_conditions   studied conditions only (role patients or
                                 model), indexed with the uploader's
                                 diagnosis field, so a diagnosis filter finds
                                 "patients with bipolar disorder" but not
                                 "bipolar electrodes" or "excluded: bipolar"
  bids_datasets.llm_profile      the profile, indexed below the short fields
                                 and above the README; also what the re-ranker
                                 reads (services/rerank.py)
  bids_dataset_facts             every fact with role, evidence and, for
                                 conditions, the canonical diagnosis codes

This module has no database or network code of its own; the runner is
scripts/extract_readme_profiles.py.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Bump when the prompt or schema changes: every dataset is then re-extracted.
PROFILE_VERSION = 1

# README text sent to the model. The median README is ~1,400 characters; the
# cap keeps the rare data dump from costing more than the rest together.
MAX_README_CHARS = 16_000
MAX_QUOTE_CHARS = 240

ROLES = ("patients", "model", "controls", "excluded", "measured", "mentioned")
# Roles that make a condition something the dataset is about.
STUDIED_ROLES = ("patients", "model")
AGE_GROUPS = ("infants", "children", "adolescents", "adults", "older_adults")

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "role": {"type": "string", "enum": list(ROLES)},
                    "evidence": {"type": "string"},
                },
                "required": ["term", "role", "evidence"],
            },
        },
        "species": {"type": "array", "items": {"type": "string"}},
        "age_groups": {"type": "array", "items": {"type": "string", "enum": list(AGE_GROUPS)}},
        "topics": {"type": "array", "items": {"type": "string"}},
        "paradigms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["term", "evidence"],
            },
        },
    },
    "required": ["summary", "conditions", "species", "age_groups", "topics", "paradigms"],
}

SYSTEM_PROMPT = """\
You catalogue neuroimaging datasets for a search engine. You read what the
uploader wrote about one dataset and record what the dataset contains, using
standard scientific terminology so that searchers who phrase things
differently still find it. Record only what the text states. Never guess."""

_INSTRUCTIONS = """\
Read the dataset below and answer with JSON that follows the given schema.

summary
  At most 90 words: the research question or topic, who was studied (species,
  age group, clinical groups), the recording modalities and the tasks. Use
  standard terms ("sentence comprehension", "major depressive disorder",
  "resting-state fMRI") even where the text uses other words. Do not mention
  conditions that were only exclusion criteria or questionnaires, and do not
  describe the file layout or BIDS conversion.

conditions
  Every medical, psychiatric or neurological condition the text names, with
  its role in THIS dataset:
    patients   people with the condition were recruited or recorded
    model      an animal or cellular model of the condition was studied
    controls   a comparison group, e.g. healthy controls
    excluded   an exclusion criterion ("no history of stroke")
    measured   a symptom scale or trait measured in a general sample
               (e.g. depressive symptoms on the BDI in healthy adults)
    mentioned  background, motivation or a citation only
  Use the condition's common name as term. Words that are not conditions are
  not conditions: "bipolar" electrodes or montages, the "depression" between
  the eyes, "stroke" of a pen. evidence is one short verbatim quote (under 200
  characters) from the text below that shows the role.

species       e.g. "human", "rat", "macaque"; empty if not stated
age_groups    only groups the text states or makes certain
topics        up to 6 research topics or cognitive constructs, standard terms
paradigms     up to 6 experimental tasks or stimulation paradigms, each with a
              verbatim evidence quote; resting state counts as a paradigm

If the text says almost nothing, return a one-sentence summary from the name
and empty lists.

Dataset
=======
"""


@dataclass
class ProfileSource:
    """What the model reads about one dataset."""
    accession_id: str
    name: str
    readme: str = ""
    dx_status: str = ""
    study_domain: str = ""
    study_design: str = ""
    description: str = ""

    def text(self) -> str:
        parts = [f"Name: {self.name}"]
        for label, value in (
            ("Uploader's diagnosis field", self.dx_status),
            ("Uploader's study domain", self.study_domain),
            ("Uploader's study design", self.study_design),
            ("Description", self.description),
        ):
            if value and value.strip():
                parts.append(f"{label}: {value.strip()}")
        readme = (self.readme or "").strip()
        if len(readme) > MAX_README_CHARS:
            readme = readme[:MAX_README_CHARS] + "\n[README truncated]"
        parts.append("README:\n" + (readme or "(none)"))
        return "\n".join(parts)

    def source_hash(self) -> str:
        payload = f"v{PROFILE_VERSION}\n{self.text()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_prompt(source: ProfileSource) -> str:
    return _INSTRUCTIONS + source.text()


# ── Parsing and evidence check ────────────────────────────────────────────────

@dataclass
class Fact:
    kind: str                      # condition | paradigm
    term: str
    role: Optional[str]
    evidence: str
    verified: bool
    codes: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class Profile:
    summary: str
    species: List[str]
    age_groups: List[str]
    topics: List[str]
    facts: List[Fact]
    raw: Dict[str, Any]

    @property
    def studied_conditions(self) -> List[str]:
        """Verified conditions the dataset is about, in order, without repeats."""
        seen: Dict[str, None] = {}
        for fact in self.facts:
            if fact.kind == "condition" and fact.verified and fact.role in STUDIED_ROLES:
                seen.setdefault(fact.term, None)
        return list(seen)

    def profile_text(self) -> str:
        """The searchable profile: summary plus the structured lists."""
        lines = [self.summary.strip()]
        if self.topics:
            lines.append("Topics: " + "; ".join(self.topics))
        paradigms = [f.term for f in self.facts if f.kind == "paradigm" and f.verified]
        if paradigms:
            lines.append("Paradigms: " + "; ".join(dict.fromkeys(paradigms)))
        population = list(self.species) + [g.replace("_", " ") for g in self.age_groups]
        if population:
            lines.append("Population: " + "; ".join(population))
        studied = self.studied_conditions
        if studied:
            lines.append("Conditions studied: " + "; ".join(studied))
        return "\n".join(line for line in lines if line)


_MARKUP_RE = re.compile(r"[*_`#>|\[\]]+")
_SPACE_RE = re.compile(r"\s+")
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _normalise(text: str) -> str:
    text = (text or "").replace("’", "'").replace("“", '"').replace("”", '"')
    text = _MARKUP_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip().lower()


def quote_in_source(quote: str, source_text: str) -> bool:
    """True if *quote* occurs in the source, ignoring case, spacing and markup.

    A model sometimes shortens a quote with "..." or fixes a typo; the parts
    of an elided quote must all occur, and a near-exact match (rapidfuzz
    partial ratio ≥ 92) is accepted for quotes of 20 characters or more.
    """
    quote_n = _normalise(quote)
    if len(quote_n) < 4:
        return False
    source_n = _normalise(source_text)
    if quote_n in source_n:
        return True
    pieces = [p.strip() for p in re.split(r"\.\.\.|…", quote_n) if len(p.strip()) >= 4]
    if len(pieces) > 1 and all(p in source_n for p in pieces):
        return True
    if len(quote_n) >= 20:
        try:
            from rapidfuzz import fuzz
        except ImportError:  # pragma: no cover - rapidfuzz is a backend dependency
            return False
        return fuzz.partial_ratio(quote_n, source_n) >= 92
    return False


def _clean_list(values: Any, limit: int = 8) -> List[str]:
    out: Dict[str, None] = {}
    for value in values or []:
        if isinstance(value, str) and value.strip():
            out.setdefault(_SPACE_RE.sub(" ", value.strip())[:80], None)
        if len(out) >= limit:
            break
    return list(out)


def parse_response(raw_text: str, source: ProfileSource) -> Profile:
    """Turn the model's JSON into a Profile and check every quote.

    Raises ValueError when the answer is not usable JSON.
    """
    text = _JSON_FENCE_RE.sub("", (raw_text or "").strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model answer is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("model answer is not a JSON object")

    source_text = source.text()
    facts: List[Fact] = []
    for item in data.get("conditions") or []:
        if not isinstance(item, dict):
            continue
        term = _SPACE_RE.sub(" ", str(item.get("term") or "")).strip()[:120]
        role = str(item.get("role") or "").strip().lower()
        evidence = str(item.get("evidence") or "").strip()[:MAX_QUOTE_CHARS]
        if not term or role not in ROLES:
            continue
        facts.append(Fact("condition", term, role, evidence, quote_in_source(evidence, source_text)))
    for item in (data.get("paradigms") or [])[:8]:
        if not isinstance(item, dict):
            continue
        term = _SPACE_RE.sub(" ", str(item.get("term") or "")).strip()[:120]
        evidence = str(item.get("evidence") or "").strip()[:MAX_QUOTE_CHARS]
        if term:
            facts.append(Fact("paradigm", term, None, evidence, quote_in_source(evidence, source_text)))

    summary = _SPACE_RE.sub(" ", str(data.get("summary") or "")).strip()
    words = summary.split()
    if len(words) > 120:
        summary = " ".join(words[:120])
    age_groups = [g for g in _clean_list(data.get("age_groups")) if g in AGE_GROUPS]
    return Profile(
        summary=summary,
        species=[s.lower() for s in _clean_list(data.get("species"), 4)],
        age_groups=age_groups,
        topics=_clean_list(data.get("topics"), 6),
        facts=facts,
        raw=data,
    )


CodeLookup = Callable[[str], List[Tuple[str, float]]]


def attach_codes(profile: Profile, lookup: Optional[CodeLookup]) -> None:
    """Map each verified studied condition to canonical diagnosis codes."""
    if lookup is None:
        return
    for fact in profile.facts:
        if fact.kind == "condition" and fact.verified and fact.role in STUDIED_ROLES:
            try:
                fact.codes = list(lookup(fact.term))[:8]
            except Exception:  # a failed lookup leaves the fact without codes
                fact.codes = []


def meta_record(profile: Profile, source: ProfileSource, model: str, extracted_at: str) -> Dict[str, Any]:
    """What is stored in bids_datasets.llm_profile_meta."""
    return {
        "version": PROFILE_VERSION,
        "model": model,
        "source_hash": source.source_hash(),
        "extracted_at": extracted_at,
        "unverified_facts": sum(not f.verified for f in profile.facts),
        "raw": profile.raw,
    }


def fact_rows(profile: Profile) -> List[Dict[str, Any]]:
    """Rows for bids_dataset_facts; one per (fact, code), or one without a code."""
    rows: List[Dict[str, Any]] = []
    for fact in profile.facts:
        base = {"kind": fact.kind, "term": fact.term, "role": fact.role,
                "evidence": fact.evidence, "verified": fact.verified}
        if fact.codes:
            rows.extend({**base, "code": code, "score": float(score)} for code, score in fact.codes)
        else:
            rows.append({**base, "code": None, "score": None})
    return rows


def studied_conditions_text(profile: Profile) -> Optional[str]:
    studied = profile.studied_conditions
    return "; ".join(studied) if studied else None


def sources_needing_work(
    rows: Sequence[Dict[str, Any]], force: bool = False
) -> List[ProfileSource]:
    """Datasets whose stored profile is missing or was made from other input."""
    todo: List[ProfileSource] = []
    for row in rows:
        source = ProfileSource(
            accession_id=row.get("accession_id") or str(row.get("id")),
            name=row.get("name") or "",
            readme=row.get("readme_text") or "",
            dx_status=row.get("dx_status") or "",
            study_domain=row.get("study_domain") or "",
            study_design=row.get("study_design") or "",
            description=row.get("description_text") or "",
        )
        meta = row.get("llm_profile_meta") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        if force or meta.get("source_hash") != source.source_hash():
            todo.append(source)
    return todo
