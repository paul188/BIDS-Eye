"""
RAG/retriever.py
----------------
Unified metadata retriever for the LLM_preprocessor pipeline.

Two resolution strategies matched to field type:

  Semantic fields  (diagnosis / task / suffix / datatype / sex / handedness)
      → yaml_to_llamaindex knowledge base + RapidFuzz WRatio
        Handles synonyms, group expansion, and hierarchy pruning.
        "mood disorders" expands to all child diagnosis codes.

  Name fields  (author / funding)
      → Flat DB-sourced list (name_index.json) + RapidFuzz token_set_ratio
        Handles partial names, abbreviations, word reordering.
        "Friston" → "Karl J. Friston"  |  "NIH" → "National Institutes of..."

Key method for scan terms:
    retrieve_scan_terms(terms) → {field: {term: [codes]}}

    Searches datatype, suffix, and task simultaneously for every term.
    Returns all matches above the similarity threshold, grouped by field.
    This is preferable to per-field lookups because a single concept like
    "fMRI" maps to both datatype ("func") and suffix ("bold") and possibly
    task names containing "fMRI".

Build the name index with:
    python RAG/build_name_index.py --db-url <url> --out RAG/name_index.json

Requires:
    pip install rapidfuzz

Interface:
    retriever = MetadataRetriever("RAG/name_index.json")
    # Scan terms — multi-field lookup (preferred for modality/task queries):
    result = retriever.retrieve_scan_terms(["fMRI", "n-back"])
    # → {"datatype": {"fMRI": ["func"]}, "suffix": {"fMRI": ["bold"]},
    #    "task": {"n-back": ["n_back"]}}

    # Single-field lookup (for diagnosis, author, funding):
    result = retriever.retrieve_for_field_multi("author", ["Friston"])
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

from rapidfuzz import fuzz as _fuzz
from rapidfuzz import process as _process

# Ensure RAG/ is on sys.path so yaml_to_llamaindex is importable regardless
# of which directory this module is imported from.
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from yaml_to_llamaindex import (
    leaf_db,
    display_db,
    group_db,
    group_display_db,
    hierarchy_db,
)

# ── Constants ──────────────────────────────────────────────────────────────────

# Fields backed by value_mappings.yaml (semantic / hierarchical matching)
_YAML_FIELDS = {"diagnosis", "task", "suffix", "datatype", "sex", "handedness"}

# Fields backed by the flat name index (token similarity matching)
_NAME_FIELDS = {"author", "funding"}

# Fields searched together for scan/modality/task terms
_SCAN_FIELDS = ("datatype", "suffix", "task")

# RAGField.value → name_index.json key
_NAME_FIELD_KEY = {
    "author":  "authors",
    "funding": "funding_sources",
}

# WRatio threshold for YAML fields — keeps false positives low on short tokens
_SEMANTIC_THRESHOLD = 80

# token_set_ratio threshold for name fields — looser to handle abbreviations
_NAME_THRESHOLD = 60


# ── YAML-field resolution helpers ──────────────────────────────────────────────
# These mirror the logic in RAG_postprocess.py but accept plain strings instead
# of UserQueryEntities, and return a list of codes for a single term.

def _plural_variants(term: str) -> List[str]:
    """Simple singular/plural variants to improve exact-match recall."""
    variants = [term]
    if term.endswith("s") and not term.endswith("ss"):
        variants.append(term[:-1])          # "disorders" → "disorder"
    else:
        variants.append(term + "s")         # "disorder"  → "disorders"
    if term.endswith("ies"):
        variants.append(term[:-3] + "y")    # "disabilities" → "disability"
    elif term.endswith("y"):
        variants.append(term[:-1] + "ies")
    return variants


def _prune_redundant_parents(codes: List[str]) -> List[str]:
    """Remove a code if a more-specific child of it is already in the list."""
    if len(codes) < 2:
        return codes
    code_paths: Dict[str, set] = {}
    for code in codes:
        info = hierarchy_db.get(code)
        code_paths[code] = set(info["path"]) if (info and "path" in info) else {code}
    to_remove = {
        code
        for code, path in code_paths.items()
        for other, other_path in code_paths.items()
        if other != code and code in other_path
    }
    return [c for c in codes if c not in to_remove]


def _resolve_yaml_term(field: str, term: str) -> List[str]:
    """Resolve one term against the YAML knowledge base for *field*."""
    cat_leaves        = leaf_db.get(field, {})
    cat_display       = display_db.get(field, {})
    cat_groups        = group_db.get(field, {})
    cat_group_display = group_display_db.get(field, {})

    term_lower = term.lower().strip()
    candidates = _plural_variants(term_lower)

    # 1. Exact match (leaf first, then group)
    for t in candidates:
        if t in cat_leaves:
            return [cat_leaves[t]]
        if t in cat_groups:
            return cat_groups[t]

    # 2. Pooled fuzzy match over display labels + group synonyms
    all_targets: Dict[str, List[str]] = {}
    for k, v in cat_display.items():
        all_targets[k] = [v]
    for k, v in cat_group_display.items():
        if k not in all_targets:            # leaf wins if both have the same label
            all_targets[k] = v

    if not all_targets:
        return []

    results = _process.extract(
        term_lower, all_targets.keys(), scorer=_fuzz.WRatio, limit=3
    )
    if not results or results[0][1] < _SEMANTIC_THRESHOLD:
        return []

    # Prefer the more specific (longer) label when top scores are tied
    best_label, best_score = results[0][0], results[0][1]
    for label, score, _ in results[1:]:
        if score >= best_score and len(label) > len(best_label):
            best_label, best_score = label, score

    return all_targets[best_label]


# ── Main retriever class ───────────────────────────────────────────────────────

class MetadataRetriever:
    """
    Unified retriever for LLM_preprocessor/preprocess.py.

    Parameters
    ----------
    name_index_path : path to the JSON file produced by build_name_index.py
                      (contains {"authors": [...], "funding_sources": [...]})
    """

    def __init__(self, name_index_path: str | Path):
        path = Path(name_index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Name index not found: {path}\n"
                "Build it with:\n"
                "  python RAG/build_name_index.py --db-url <url> --out RAG/name_index.json"
            )
        with open(path, encoding="utf-8") as fh:
            self._names: Dict[str, List[str]] = json.load(fh)

    # ------------------------------------------------------------------
    def retrieve_scan_terms(
        self,
        terms: List[str],
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Search datatype, suffix, and task simultaneously for every term.

        Returns {field: {term: [codes]}} for every field where at least one
        term matched above the similarity threshold.  A single user term like
        "fMRI" will typically produce entries under both "datatype" (func) and
        "suffix" (bold), giving the SQL generator a complete picture.

        Parameters
        ----------
        terms : raw user terms, e.g. ["fMRI", "n-back", "resting state"]

        Returns
        -------
        e.g. {
            "datatype": {"fMRI": ["func"]},
            "suffix":   {"fMRI": ["bold"]},
            "task":     {"n-back": ["n_back"], "resting state": ["rest"]},
        }
        """
        result: Dict[str, Dict[str, List[str]]] = {}
        for field in _SCAN_FIELDS:
            term_map: Dict[str, List[str]] = {}
            for term in terms:
                codes = _resolve_yaml_term(field, term)
                codes = _prune_redundant_parents(codes)
                if codes:
                    term_map[term] = codes
            if term_map:
                result[field] = term_map
        return result

    # ------------------------------------------------------------------
    def retrieve_for_field(
        self,
        field: str,
        term: str,
        n_results: int = 20,
    ) -> List[str]:
        """
        Resolve *term* for *field* and return up to *n_results* canonical values.

        Semantic fields (diagnosis / task / suffix / datatype / sex / handedness):
            returns standard_code strings from value_mappings.yaml

        Name fields (author / funding):
            returns exact DB strings from name_index.json
        """
        if field in _YAML_FIELDS:
            codes = _resolve_yaml_term(field, term)
            codes = _prune_redundant_parents(codes)
            return codes[:n_results]

        if field in _NAME_FIELDS:
            candidates = self._names.get(_NAME_FIELD_KEY[field], [])
            if not candidates:
                return []
            results = _process.extract(
                term,
                candidates,
                scorer=_fuzz.token_set_ratio,
                limit=n_results,
                score_cutoff=_NAME_THRESHOLD,
            )
            return [r[0] for r in results]

        return []

    # ------------------------------------------------------------------
    def retrieve_for_field_multi(
        self,
        field: str,
        terms: List[str],
        n_results_per_term: int = 20,
    ) -> Dict[str, List[str]]:
        """
        Resolve multiple *terms* for *field*.
        Returns {term: [codes]} — terms with no match are omitted.

        n_results_per_term is a safety cap; threshold-based filtering in
        _resolve_yaml_term already limits results to high-confidence matches.
        """
        result: Dict[str, List[str]] = {}
        for term in terms:
            codes = self.retrieve_for_field(field, term, n_results_per_term)
            if codes:
                result[term] = codes
        return result
