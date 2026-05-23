"""
RAG/retriever.py
----------------
Unified metadata retriever for the LLM_preprocessor pipeline.

Three resolution strategies in priority order:

  1. Exact match  (all YAML fields)
       leaf_db / group_db exact string lookup.
       Catches programmatic values, accession codes, standard_codes.

  2. Fuzzy match  (YAML fields — primary path)
       RapidFuzz WRatio ≥ 80 over display labels + synonyms.
       Weight-adjusted threshold for low-confidence synonyms.

  3. Embedding fallback  (YAML fields — miss path)
       Two-tier biomedical embedding search when fuzzy match scores < 80.
       Tier 1: BioLORD-2023-C (cosine ≥ 0.75) — SNOMED-CT / MedDRA SOTA
       Tier 2: SapBERT         (cosine ≥ 0.65) — UMLS 4M+ entity linking
       Models are downloaded on first use and cached by sentence-transformers.
       Embeddings of the YAML KB are pre-computed and cached as .npz files.
       Falls back gracefully if sentence-transformers / numpy are not installed.

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

Optional (for embedding fallback):
    pip install sentence-transformers numpy

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
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz as _fuzz
from rapidfuzz import process as _process

log = logging.getLogger(__name__)

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
    weight_db,
    build_embedding_index,
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

# Maximum extra points added to the fuzzy threshold for a weight-0.0 synonym.
# A synonym with weight w requires a fuzzy score of at least:
#   _SEMANTIC_THRESHOLD + (1.0 - w) * _WEIGHT_BOOST
# Examples (SEMANTIC_THRESHOLD=80, WEIGHT_BOOST=15):
#   w=1.0 → threshold 80   (no penalty — fully trusted synonym)
#   w=0.7 → threshold 84.5 (mild penalty)
#   w=0.5 → threshold 87.5 (moderate penalty)
#   w=0.2 → threshold 92   (steep penalty — near-exact match required)
_WEIGHT_BOOST = 8

# Tier-1 embedding model: biomedical ontology SOTA (SNOMED-CT / MedDRA)
_BIOLORD_MODEL = "FremyCompany/BioLORD-2023-C"
_BIOLORD_THRESHOLD = 0.75

# Tier-2 embedding model: UMLS 4M+ concept entity linking
_SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
_SAPBERT_THRESHOLD = 0.65

# Path to value_mappings.yaml — used as cache key for embedding indices
_YAML_PATH = str(_THIS_DIR / "value_mappings.yaml")


# ── Embedding fallback — lazy initialisation ────────────────────────────────────
# Loaded on first miss so startup is not penalised when fuzzy matching suffices.

_emb_indices: Dict[str, Optional[Any]] = {}  # model_name → EmbeddingIndex or None
_emb_models: Dict[str, Any] = {}             # model_name → SentenceTransformer or None


def _load_embedding_tier(model_name: str) -> Optional[Any]:
    """Lazy-load the embedding index and the SentenceTransformer for *model_name*.

    Returns the EmbeddingIndex on success, None if sentence-transformers /
    numpy are not installed or the index cannot be built.  Result is cached
    so each model is loaded at most once per process.
    """
    if model_name in _emb_indices:
        return _emb_indices[model_name]

    try:
        import numpy  # noqa: F401 — probe before heavy work
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _emb_indices[model_name] = None
        return None

    try:
        index = build_embedding_index(model_name, _YAML_PATH)
        if index is not None:
            _emb_models[model_name] = SentenceTransformer(model_name)
        _emb_indices[model_name] = index
        return index
    except Exception as exc:
        log.warning("Embedding index load failed for %s: %s", model_name, exc)
        _emb_indices[model_name] = None
        return None


def _embedding_fallback(field: str, term: str) -> List[str]:
    """Two-tier biomedical embedding search for *term* in *field*.

    Called when RapidFuzz WRatio misses (score < threshold).
    Tier 1: BioLORD-2023-C, cosine ≥ 0.75 — returns on first hit.
    Tier 2: SapBERT,         cosine ≥ 0.65 — reached only if tier 1 misses.
    Returns [] if both tiers miss or if sentence-transformers is not installed.
    """
    try:
        import numpy as np
    except ImportError:
        return []

    term_lower = term.lower().strip()

    for model_name, threshold in (
        (_BIOLORD_MODEL, _BIOLORD_THRESHOLD),
        (_SAPBERT_MODEL, _SAPBERT_THRESHOLD),
    ):
        index = _load_embedding_tier(model_name)
        if index is None:
            continue

        cat_data = index.get(field)
        if not cat_data:
            continue

        emb_model = _emb_models.get(model_name)
        if emb_model is None:
            continue

        terms_list, embs, code_lists = cat_data
        query_emb = emb_model.encode(
            [term_lower], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = np.dot(embs, query_emb)
        best_idx = int(np.argmax(scores))

        if float(scores[best_idx]) >= threshold:
            log.debug(
                "Embedding fallback (%s) resolved '%s' → %s (score %.3f)",
                model_name.split("/")[-1], term, code_lists[best_idx], scores[best_idx],
            )
            return code_lists[best_idx]

    return []


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
        return _embedding_fallback(field, term)

    # Prefer the more specific (longer) label when top scores are tied
    best_label, best_score = results[0][0], results[0][1]
    for label, score, _ in results[1:]:
        if score >= best_score and len(label) > len(best_label):
            best_label, best_score = label, score

    # Weight-adjusted threshold: low-confidence synonyms need a higher fuzzy
    # score to be accepted. Labels and plain-string synonyms default to 1.0.
    weight = weight_db.get(field, {}).get(best_label, 1.0)
    effective_threshold = _SEMANTIC_THRESHOLD + (1.0 - weight) * _WEIGHT_BOOST
    if best_score < effective_threshold:
        return _embedding_fallback(field, term)

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
