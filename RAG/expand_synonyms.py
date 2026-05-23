"""
RAG/expand_synonyms.py
----------------------
Mine synonym candidates from real dataset descriptions and paper abstracts,
then call an LLM to extract new terms — grounded in actual neuroimaging usage.

Data sources (in priority order):
  1. bids_datasets.description_text  — already in PostgreSQL, zero cost
  2. CrossRef REST API               — free, returns title + abstract as JSON
  3. OpenAlex REST API               — free, neuroscience filter, 250M+ works

LLM backend (first key found wins):
  ANTHROPIC_API_KEY → Claude Haiku   (preferred: cheaper, faster)
  GEMINI_API_KEY    → Gemini Flash   (fallback)

Modes:
  --mode augment (default): append new terms to nodes with few synonyms
  --mode replace           : regenerate the full synonym set for every node

Usage (augment):
    python RAG/expand_synonyms.py \\
        --db-url postgresql://bids:changeme@localhost/bids_sql \\
        --min-synonyms 3 \\
        --max-nodes 50 \\
        --out RAG/expand_synonyms_proposals.yaml

Usage (replace, in batches):
    python RAG/expand_synonyms.py \\
        --db-url ... \\
        --mode replace \\
        --max-nodes 200 --offset 0 \\
        --out RAG/proposals_batch1.yaml

    # repeat with --offset 200, --offset 400, …

Review the proposal file, then run merge_proposals.py to apply it atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ── Paths & constants ──────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
_DEFAULT_YAML = _HERE / "value_mappings.yaml"

_METADATA_KEYS = {
    "label", "standard_code", "description",
    "synonyms", "codes", "extra_codes", "dataset_codes",
}

_CATEGORIES = [
    "diagnosis", "task", "suffix", "handedness", "sex",
    "datatype", "sidecar_fields", "participant_extra_fields",
]

# Which DB table/column holds the standard_code for each YAML category
_CATEGORY_DB_COL: Dict[str, Tuple[str, str]] = {
    "task":       ("bids_objects",      "task"),
    "suffix":     ("bids_objects",      "suffix"),
    "datatype":   ("bids_objects",      "datatype"),
    "diagnosis":  ("bids_participants", "diagnosis"),
    "sex":        ("bids_participants", "sex"),
    "handedness": ("bids_participants", "handedness"),
}

_CROSSREF_UA    = "BIDS-Eye/1.0 (mailto:research@example.com)"
_CROSSREF_DELAY = 0.2   # seconds — stays in CrossRef's polite pool
_OPENALEX_DELAY = 0.2   # OpenAlex: 1 req/s is polite, 0.2s is well within limit

# OpenAlex neuroscience concept ID
_OPENALEX_NEURO_ID = "C121332964"


# ── Text helpers ───────────────────────────────────────────────────────────────

def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"


def _extract_doi(ref: str) -> Optional[str]:
    """Return a bare DOI from a URL or plain-string reference, or None."""
    m = re.search(r"\b(10\.\d{4,}/\S+)", ref)
    if m:
        return m.group(1).rstrip(".,;)")
    return None


# ── Node discovery ─────────────────────────────────────────────────────────────

def _collect_node(
    key: str,
    value: dict,
    current_path: List[str],
) -> dict:
    """Build the node dict that represents one leaf in value_mappings.yaml."""
    accession_ids: List[str] = []
    for dc in value.get("dataset_codes") or []:
        if isinstance(dc, dict):
            accession_ids.extend(dc.get("datasets") or [])

    # Collect all alternate raw codes from the YAML 'codes' list
    codes: List[str] = [str(c) for c in (value.get("codes") or []) if c]

    return {
        "path":              " > ".join(current_path),
        "category":          current_path[0],
        "key":               key,
        "standard_code":     value["standard_code"],
        "label":             value.get("label") or key.replace("_", " "),
        "description":       value.get("description") or "",
        "existing_synonyms": [
            s["term"] if isinstance(s, dict) else str(s)
            for s in (value.get("synonyms") or [])
        ],
        "accession_ids":     accession_ids,
        "codes":             codes,
    }


def _walk_sparse(data: Any, category: str, min_synonyms: int, out: List[dict]) -> None:
    """Collect leaf concepts (flat dict) whose synonym list is shorter than min_synonyms."""
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if not isinstance(value, dict) or not value.get("standard_code"):
            continue
        synonyms = value.get("synonyms") or []
        if len(synonyms) < min_synonyms:
            out.append(_collect_node(key, value, [category, key]))


def _walk_all(data: Any, category: str, out: List[dict]) -> None:
    """Collect ALL leaf concepts (flat dict) regardless of synonym count."""
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if not isinstance(value, dict) or not value.get("standard_code"):
            continue
        out.append(_collect_node(key, value, [category, key]))


def find_sparse_nodes(yaml_path: Path, min_synonyms: int, category: Optional[str] = None) -> List[dict]:
    with open(yaml_path) as fh:
        schema = yaml.safe_load(fh)

    results: List[dict] = []
    for cat in _CATEGORIES:
        if category and cat != category:
            continue
        if cat in schema:
            _walk_sparse(schema[cat], cat, min_synonyms, results)

    # Nodes with more dataset_codes linkage first — more grounding available
    results.sort(key=lambda n: (-len(n["accession_ids"]), n["path"]))
    return results


def find_all_nodes(yaml_path: Path, category: Optional[str] = None) -> List[dict]:
    """Return all leaf nodes sorted by context richness (description + accession_ids)."""
    with open(yaml_path) as fh:
        schema = yaml.safe_load(fh)

    results: List[dict] = []
    for cat in _CATEGORIES:
        if category and cat != category:
            continue
        if cat in schema:
            _walk_all(schema[cat], cat, results)

    # Most context-rich first: description length counts less than DB-linked datasets
    results.sort(key=lambda n: -(10 * len(n["accession_ids"]) + len(n["description"])))
    return results


# ── Database context ───────────────────────────────────────────────────────────

def _fetch_dataset_context(
    db_url: str,
    node: dict,
    max_datasets: int = 3,
) -> Tuple[List[str], List[str]]:
    """
    Return (description_texts, dois) for datasets linked to this node.

    Two strategies (results combined, deduplicated):
      1. Direct accession_id lookup via dataset_codes in the YAML node.
      2. JOIN on the standard_code column + alternate codes from node['codes'].
    """
    import psycopg2

    category      = node["category"]
    standard_code = node["standard_code"]
    accession_ids = node["accession_ids"]
    alt_codes     = node.get("codes", [])

    seen_texts: List[str] = []
    all_refs:   List[Any] = []

    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:

            # Strategy 1 — datasets referenced in dataset_codes
            if accession_ids:
                cur.execute(
                    "SELECT description_text, paper_references "
                    "FROM bids_datasets "
                    "WHERE accession_id = ANY(%s) "
                    "  AND description_text IS NOT NULL "
                    "  AND description_text <> '' "
                    "LIMIT %s",
                    (accession_ids, max_datasets),
                )
                for desc, refs in cur.fetchall():
                    seen_texts.append(desc)
                    if refs:
                        all_refs.extend(refs)

            # Strategy 2 — datasets that contain this code (or any alt code) in the DB
            remaining = max_datasets - len(seen_texts)
            if remaining > 0 and category in _CATEGORY_DB_COL:
                table, col = _CATEGORY_DB_COL[category]

                # All candidate values: standard_code + raw alt codes from YAML
                lookup_values = list(dict.fromkeys([standard_code] + alt_codes))

                if table == "bids_objects":
                    join_clause   = "JOIN bids_objects o ON o.dataset_id = d.id"
                    filter_clause = f"o.{col} = ANY(%s)"
                else:
                    join_clause   = "JOIN bids_participants p ON p.dataset_id = d.id"
                    filter_clause = f"p.{col} = ANY(%s)"

                cur.execute(
                    f"SELECT DISTINCT d.description_text, d.paper_references "
                    f"FROM bids_datasets d "
                    f"{join_clause} "
                    f"WHERE {filter_clause} "
                    f"  AND d.description_text IS NOT NULL "
                    f"  AND d.description_text <> '' "
                    f"LIMIT %s",
                    (lookup_values, remaining),
                )
                for desc, refs in cur.fetchall():
                    if desc not in seen_texts:
                        seen_texts.append(desc)
                    if refs:
                        all_refs.extend(refs)

    # Extract unique DOIs from paper_references
    dois: List[str] = []
    seen_dois: set  = set()
    for ref in all_refs:
        doi = _extract_doi(str(ref))
        if doi and doi not in seen_dois:
            dois.append(doi)
            seen_dois.add(doi)

    return seen_texts[:max_datasets], dois


# ── CrossRef ───────────────────────────────────────────────────────────────────

_crossref_cache: Dict[str, Optional[str]] = {}


def _fetch_crossref_abstract(doi: str) -> Optional[str]:
    """Return '<title>. <abstract>' truncated to 200 words, or None on failure."""
    if doi in _crossref_cache:
        return _crossref_cache[doi]

    url = f"https://api.crossref.org/works/{doi}"
    req = urllib.request.Request(url, headers={"User-Agent": _CROSSREF_UA})

    try:
        time.sleep(_CROSSREF_DELAY)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        message  = data.get("message", {})
        title    = (message.get("title") or [""])[0]
        abstract = message.get("abstract") or ""
        abstract = re.sub(r"<[^>]+>", "", abstract)   # strip JATS XML tags

        text   = f"{title}. {abstract}".strip(". ") or None
        result = _truncate_words(text, 200) if text else None

    except Exception as exc:
        print(f"    [crossref] {doi}: {exc}", file=sys.stderr)
        result = None

    _crossref_cache[doi] = result
    return result


# ── OpenAlex ───────────────────────────────────────────────────────────────────

def _fetch_openalex_abstracts(term: str, max_papers: int = 3) -> List[str]:
    """Return up to max_papers '<title>. <abstract>' snippets from OpenAlex.

    Filters to neuroscience works (C121332964) so results stay on-domain.
    Abstract is reconstructed from OpenAlex's inverted index format.
    """
    params = urllib.parse.urlencode({
        "search": term,
        "filter": f"concepts.id:{_OPENALEX_NEURO_ID}",
        "per-page": max_papers,
        "select": "title,abstract_inverted_index",
        "mailto": "research@example.com",
    })
    url = f"https://api.openalex.org/works?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "BIDS-Eye/1.0"})

    try:
        time.sleep(_OPENALEX_DELAY)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        print(f"    [openalex] {term}: {exc}", file=sys.stderr)
        return []

    results: List[str] = []
    for work in (data.get("results") or [])[:max_papers]:
        title = work.get("title") or ""
        inv   = work.get("abstract_inverted_index") or {}

        if inv:
            # Reconstruct sentence from inverted index: {word: [positions]}
            max_pos = max((p for positions in inv.values() for p in positions), default=0)
            words   = [""] * (max_pos + 1)
            for word, positions in inv.items():
                for pos in positions:
                    if pos <= max_pos:
                        words[pos] = word
            abstract = " ".join(w for w in words if w)
        else:
            abstract = ""

        text = f"{title}. {abstract}".strip(". ")
        if text:
            results.append(_truncate_words(text, 200))

    return results


# ── Weight calculation ─────────────────────────────────────────────────────────

def _compute_weight(term: str, source_texts: List[str]) -> float:
    """Score a synonym by how consistently it appears across source texts.

    Components (both measured against lowercased text):
      occurrence_rate  — fraction of texts that contain the term at all   (60 %)
      frequency_score  — log-scaled total mention count across all texts  (40 %)

    Log scaling saturates around 10 mentions so a single over-represented
    paper can't dominate the score.  Result is rounded to 2 d.p. and floored
    at 0.1 so that a term the LLM identified but found only once still carries
    a small positive weight rather than being silently dropped.
    """
    import math

    term_lower = term.lower()
    n = len(source_texts)
    if n == 0:
        return 0.5

    texts_lower = [t.lower() for t in source_texts]

    n_containing = sum(1 for t in texts_lower if term_lower in t)
    occurrence_rate = n_containing / n

    total_count = sum(t.count(term_lower) for t in texts_lower)
    freq_score  = min(1.0, math.log2(1 + total_count) / math.log2(11))

    weight = 0.6 * occurrence_rate + 0.4 * freq_score
    return round(max(0.1, weight), 2)


# ── LLM extraction ─────────────────────────────────────────────────────────────

_SYSTEM_AUGMENT = """\
You are a specialist in neuroimaging, cognitive neuroscience, and clinical terminology.

Your task: given source texts about a specific concept, extract every alternative name,
abbreviation, and informal label that researchers actually use for that concept.

Rules — apply all of them:
1. ONLY include terms that appear verbatim (or as a recognised abbreviation/variant) in
   the provided source texts. Do NOT invent terms from general knowledge.
2. Include ALL of the following that you find in the texts:
   - Abbreviations and acronyms (e.g., "CPT", "rsfMRI", "EEG")
   - Informal / colloquial names used in lab settings
   - Hyphenated, spaced, and unhyphenated variants of the same term
   - Plural forms if they appear (e.g., "auditory oddball tasks")
   - Task-specific labels or paradigm names researchers use as synonyms
   - International spellings / British English variants if present
3. Do NOT include:
   - The concept's own canonical label (already in the system)
   - Generic methodological terms that apply to many unrelated concepts
   - Terms from the "Currently in system" list
4. Be exhaustive: extract every distinct candidate you can find, up to 25 terms.
   More is better than less — the caller will filter by corpus frequency.
5. Respond with a JSON array of strings only. No explanation, no markdown. Return [] if none found.\
"""

_SYSTEM_REPLACE = """\
You are a specialist in neuroimaging, cognitive neuroscience, and clinical terminology.

Your task: generate a comprehensive synonym set for a specific neuroimaging/clinical concept.
Source texts are provided as context to guide vocabulary — also draw on your deep domain
knowledge to produce all terms researchers actually use for this concept.

Rules — apply all of them:
1. Include ALL of the following:
   - Abbreviations and acronyms (e.g., "CPT", "rsfMRI", "fNIRS")
   - Informal and colloquial names used in lab or clinical settings
   - Hyphenated, spaced, and unhyphenated variants (e.g., "n-back", "nback", "n back")
   - Task variant names and paradigm-specific labels used interchangeably
   - International spellings and British English variants where they differ
   - Common misspellings or alternate spellings used in the literature
   - Related terms researchers would type when searching for this concept
2. Do NOT include:
   - The concept's own canonical label (already in the system as the label)
   - Terms from the "Currently in system" list (they are already present)
   - Overly generic terms that map to dozens of unrelated concepts
   - Full sentences or phrases longer than 5 words
3. Aim for 8–25 high-quality terms. Favour precision over volume — if you can only find
   5 truly accurate synonyms, return 5. Do not pad with irrelevant terms.
4. Source texts are hints — use them, but do not restrict yourself to only what appears there.
5. Respond with a JSON array of strings only. No explanation, no markdown, no code fences.\
"""

# Default for backward compatibility
_SYSTEM = _SYSTEM_AUGMENT


def _build_prompt(node: dict, texts: List[str], mode: str = "augment") -> str:
    existing_str = json.dumps(node["existing_synonyms"]) if node["existing_synonyms"] else "[]"
    # Use more words per source in replace mode — richer context → better extraction
    words_per_source = 200 if mode == "replace" else 150
    context_blocks = "\n\n".join(
        f"[Source {i}]\n{_truncate_words(t, words_per_source)}"
        for i, t in enumerate(texts, 1)
    ) or "(no source texts available)"

    if mode == "replace":
        task_instruction = (
            f"Generate the COMPLETE, authoritative synonym set for \"{node['label']}\" "
            f"(standard_code: {node['standard_code']}).\n"
            f"Use the source texts as context, but primarily draw on your neuroimaging domain "
            f"knowledge to include EVERY abbreviation, informal name, task variant, and "
            f"alternative label that researchers actually use.\n"
            f"Include the best terms regardless of what is already listed above — you are "
            f"building a fresh, high-quality list that will replace the old one.\n"
            f"Do NOT include the concept's own canonical label: \"{node['label']}\".\n"
            f"Return a JSON array of strings. Aim for 8-20 high-quality terms. "
            f"Return [] only if this concept truly has no synonyms."
        )
    else:
        task_instruction = (
            f"Extract up to 25 alternative names or abbreviations for \"{node['label']}\" "
            f"from the sources above that are NOT already in the 'Currently in system' list.\n"
            f"Return a JSON array of strings. Return [] if none found."
        )

    desc_line = f"Description: {node['description']}\n" if node.get("description") else ""
    existing_line = (
        f"Current synonyms (for context only — you are replacing these):\n{existing_str}\n"
        if mode == "replace" else
        f"Currently in system (DO NOT repeat these): {existing_str}\n"
    )

    return (
        f"Concept: \"{node['label']}\"  (standard_code: {node['standard_code']})\n"
        f"Category: {node['category']}\n"
        f"{desc_line}"
        f"{existing_line}"
        f"\n"
        f"Source texts:\n"
        f"---\n"
        f"{context_blocks}\n"
        f"---\n"
        f"\n"
        f"{task_instruction}"
    )


def _parse_llm_response(raw: Optional[str]) -> List[str]:
    """Extract a JSON string-array from the LLM response."""
    if not raw:
        return []
    raw = raw.strip()
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            return [s for s in parsed if isinstance(s, str)]
        except json.JSONDecodeError:
            pass
    return []


def _call_anthropic(prompt: str, api_key: str, system: str = _SYSTEM_AUGMENT) -> List[str]:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_llm_response(resp.content[0].text)


def _call_gemini(
    prompt: str,
    api_key: str,
    model: str = "gemini-2.5-pro",
    system: str = _SYSTEM_AUGMENT,
) -> List[str]:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)

    # Retry with exponential backoff on 503 / rate-limit errors.
    delays = [15, 30, 60, 120]
    last_exc: Exception = RuntimeError("no attempt made")
    for attempt, wait in enumerate([0] + delays):
        if wait:
            time.sleep(wait)
        try:
            resp = client.models.generate_content(
                model=model,
                contents=system + "\n\n" + prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    # Gemini 2.5 Pro always runs in thinking mode; thinking tokens count
                    # toward this limit. Set generously so output is never truncated.
                    max_output_tokens=8192,
                ),
            )
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if "503" in msg or "UNAVAILABLE" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if attempt < len(delays):
                    continue   # retry after backoff
            raise
        # Gemini 2.5 Pro (thinking) may return text via candidates when resp.text is None.
        # content.parts can also be None when the model returns only a thinking block.
        text = resp.text
        if text is None:
            candidate = (resp.candidates or [None])[0]
            content   = getattr(candidate, "content", None) if candidate else None
            parts     = getattr(content, "parts", None) or []
            text      = "".join(p.text for p in parts if p and getattr(p, "text", None))
        return _parse_llm_response(text)

    raise last_exc


def _call_llm(
    prompt: str,
    anthropic_key: Optional[str],
    gemini_key: Optional[str],
    gemini_model: str = "gemini-2.5-pro",
    system: str = _SYSTEM_AUGMENT,
) -> List[str]:
    if anthropic_key:
        return _call_anthropic(prompt, anthropic_key, system=system)
    if gemini_key:
        return _call_gemini(prompt, gemini_key, model=gemini_model, system=system)
    raise RuntimeError("No API key available.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine synonyms from DB descriptions, CrossRef, and OpenAlex abstracts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--yaml",         default=str(_DEFAULT_YAML),
                        help="Path to value_mappings.yaml")
    parser.add_argument("--db-url",       required=True,
                        help="PostgreSQL connection URL")
    parser.add_argument("--out",          default="RAG/expand_synonyms_proposals.yaml",
                        help="Output YAML file for proposals")
    parser.add_argument("--mode",         default="augment", choices=["augment", "replace"],
                        help="augment: add to sparse nodes; replace: regenerate all synonyms")
    parser.add_argument("--min-synonyms", type=int, default=3,
                        help="(augment mode only) expand nodes with fewer synonyms than this")
    parser.add_argument("--max-nodes",    type=int, default=50,
                        help="Safety cap: max nodes to process in one run")
    parser.add_argument("--offset",       type=int, default=0,
                        help="Skip the first N nodes (for batching in replace mode)")
    parser.add_argument("--category",     default=None,
                        help="Restrict to one YAML category (e.g. task, diagnosis)")
    parser.add_argument("--no-openalex",   action="store_true",
                        help="Disable OpenAlex queries (use DB + CrossRef only)")
    parser.add_argument("--gemini-model",  default="gemini-2.5-pro",
                        help="Gemini model ID to use (default: gemini-2.5-pro for quality)")
    args = parser.parse_args()

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    gemini_key    = os.environ.get("GEMINI_API_KEY")
    if not anthropic_key and not gemini_key:
        sys.exit("Error: set ANTHROPIC_API_KEY or GEMINI_API_KEY before running.")

    provider = f"Claude Sonnet" if anthropic_key else f"Gemini ({args.gemini_model})"
    print(f"LLM backend : {provider}")
    print(f"Mode        : {args.mode}")

    # 1. Discover nodes to process
    print(f"Scanning    : {args.yaml}")
    if args.mode == "replace":
        nodes = find_all_nodes(Path(args.yaml), args.category)
    else:
        nodes = find_sparse_nodes(Path(args.yaml), args.min_synonyms, args.category)
        print(f"Sparse nodes: {len(nodes)} (min_synonyms={args.min_synonyms})")

    # Apply offset + cap for batching
    nodes = nodes[args.offset: args.offset + args.max_nodes]
    print(f"Processing  : {len(nodes)} nodes (offset={args.offset}, cap={args.max_nodes})\n")

    proposals_key = "proposed_synonyms" if args.mode == "replace" else "proposed_additions"
    proposals: List[dict] = []

    # Open output file immediately and write header, so partial results survive a kill.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# Auto-generated by RAG/expand_synonyms.py (mode={args.mode})\n"
        f"# REVIEW CAREFULLY before merging into value_mappings.yaml.\n"
        f"# {proposals_key} were extracted from DB descriptions, CrossRef, and OpenAlex.\n\n"
    )
    out_fh = open(out_path, "w", encoding="utf-8")
    out_fh.write(header)
    out_fh.flush()

    for i, node in enumerate(nodes, 1):
        label = f"[{i}/{len(nodes)}] {node['path']}"
        print(f"{label}", end="  ", flush=True)

        # 2. DB context
        try:
            desc_texts, dois = _fetch_dataset_context(args.db_url, node)
        except Exception as exc:
            print(f"[db error: {exc}]")
            continue

        # 3. CrossRef abstracts (up to 2 papers)
        abstracts: List[str] = []
        for doi in dois[:2]:
            ab = _fetch_crossref_abstract(doi)
            if ab:
                abstracts.append(ab)

        # 4. OpenAlex abstracts (up to 3 papers from neuroscience literature)
        openalex_texts: List[str] = []
        if not args.no_openalex:
            openalex_texts = _fetch_openalex_abstracts(node["label"], max_papers=3)

        all_texts = desc_texts + abstracts + openalex_texts
        print(
            f"{len(desc_texts)} desc / {len(abstracts)} crossref / {len(openalex_texts)} openalex",
            end="  ",
            flush=True,
        )

        if not all_texts and args.mode != "replace":
            print("[skipped — no text context]")
            continue

        # 5. LLM candidate generation
        sys_prompt = _SYSTEM_REPLACE if args.mode == "replace" else _SYSTEM_AUGMENT
        try:
            prompt     = _build_prompt(node, all_texts, mode=args.mode)
            candidates = _call_llm(prompt, anthropic_key, gemini_key, args.gemini_model,
                                   system=sys_prompt)
        except Exception as exc:
            print(f"[llm error: {exc}]")
            continue

        # 6. Filter and score
        # In replace mode: accept all LLM candidates (domain knowledge + texts);
        #   weight by corpus occurrence but do NOT require verbatim presence.
        # In augment mode: require verbatim grounding and skip existing terms.
        existing_lower = {s.lower() for s in node["existing_synonyms"]}

        weighted: List[dict] = []
        seen_lower: set = set()
        for term in candidates:
            if not isinstance(term, str) or not term.strip():
                continue
            tl = term.strip().lower()
            if tl in seen_lower:
                continue
            seen_lower.add(tl)
            if args.mode == "augment":
                if tl in existing_lower:
                    continue
                # Augment mode: require verbatim grounding in source texts
                if not any(tl in t.lower() for t in all_texts):
                    continue
            w = _compute_weight(term.strip(), all_texts) if all_texts else 1.0
            weighted.append({"term": term.strip(), "weight": w})

        # Sort descending by weight
        weighted.sort(key=lambda x: x["weight"], reverse=True)

        print(f"→ {len(weighted)} proposals")

        if not weighted:
            continue

        entry = {
            "path":              node["path"],
            "standard_code":     node["standard_code"],
            "existing_synonyms": node["existing_synonyms"],
            proposals_key:       weighted,
            "source_datasets":   node["accession_ids"][:5],
            "source_papers":     [
                doi for doi in dois[:2]
                if _crossref_cache.get(doi)
            ],
        }
        proposals.append(entry)
        # Write each proposal immediately so a kill doesn't lose completed work
        yaml.dump(
            [entry], out_fh,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        out_fh.flush()

    out_fh.close()
    print(f"\nWrote {len(proposals)} proposals → {out_path}")
    if proposals:
        print("Next step: review the file, then run RAG/merge_proposals.py to apply atomically.")


if __name__ == "__main__":
    main()
