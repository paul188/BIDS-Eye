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
from join_registry import field_to_db_col as _field_to_db_col

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

# Terms that are pure modality abbreviations or generic imaging phrases.
# For these, the task field is skipped in retrieve_scan_terms to prevent
# spurious task hits (e.g. "fMRI" → bold_acquisition_general, "DTI" → dual_task).
# Generic imaging terms also skip the suffix field because suffix fuzzy-matches
# produce impossibly specific filters (e.g. "brain imaging" → t1w, pet).
_PURE_MODALITY_TERMS = {
    # Abbreviations
    "fmri", "mri", "meg", "eeg", "dti", "dwi", "ieeg", "ecog",
    "fnirs", "nirs", "pet", "asl", "mrs", "bold", "mr", "ct",
    "bld", "t1", "t2", "t1w", "t2w",
    # Short phrases with modality abbreviation
    "fmri scan", "fmri data", "eeg data", "meg data",
    "eeg recording", "meg recording", "mri scan", "mri data",
    "bold fmri", "bold signal", "bold response", "bold mri",
    # Full names
    "functional mri", "functional magnetic resonance imaging",
    "diffusion tensor imaging", "diffusion weighted imaging",
    "diffusion mri", "diffusion weighted mri",
    "electroencephalography", "magnetoencephalography",
    "intracranial eeg", "intracranial electroencephalography",
    "functional near-infrared spectroscopy",
    "positron emission tomography",
    "arterial spin labeling", "arterial spin labelling",
    "structural mri", "anatomical mri", "structural magnetic resonance imaging",
    "t1 weighted", "t1-weighted", "t2 weighted", "t2-weighted",
}
_GENERIC_IMAGING_TERMS = {
    "brain imaging", "brain scan", "brain scans", "neuroimaging",
    "neuroimaging data", "brain study", "brain studies", "imaging study",
    "imaging data", "brain recording", "brain recordings", "brain imaging data",
    "brain images", "brain image",
}

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

    # 1. Exact match — prefer group (all descendants) over single leaf code
    # so "working memory" returns every working-memory task code, not just
    # working_memory_general alone.
    for t in candidates:
        if t in cat_groups:
            return cat_groups[t]
        if t in cat_leaves:
            return [cat_leaves[t]]

    # 2. Pooled fuzzy match over display labels + group synonyms
    # Exclude keys shorter than 3 chars from the fuzzy pool — short abbreviations
    # like "ng" or "dre" would spuriously match unrelated user queries via
    # partial substring scoring (WRatio uses partial_ratio internally).
    # They are still found by the exact-match step above.
    _MIN_FUZZY_KEY_LEN = 3
    all_targets: Dict[str, List[str]] = {}
    for k, v in cat_display.items():
        if len(k) >= _MIN_FUZZY_KEY_LEN:
            all_targets[k] = [v]
    for k, v in cat_group_display.items():
        if len(k) >= _MIN_FUZZY_KEY_LEN:
            # Group wins for dual nodes — returning all descendant codes means
            # "working memory" finds every dataset tagged with any working-memory
            # task, not just those using the working_memory_general code itself.
            all_targets[k] = v

    if not all_targets:
        return []

    # Increase limit so low-weight best matches can be supplemented by siblings
    results = _process.extract(
        term_lower, all_targets.keys(), scorer=_fuzz.WRatio, limit=20
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

    # When the best match is a low-confidence synonym (weight < 0.5), the
    # matched key is generic (e.g. "motor task" → somatomotor_task, weight 0.1).
    # Strip descriptor adjectives from the user term and re-run fuzzy matching
    # on the core concept — "complex motor task" → "motor task" scores 95+
    # against all "X motor task" labels, not just the synonym key itself.
    if weight < 0.5:
        _DESCRIPTORS = {
            "complex", "simple", "basic", "advanced", "various", "different",
            "specific", "typical", "standard", "common", "regular",
            "experiment", "experiments", "paradigm", "paradigms",
            "study", "studies", "data", "trial", "trials",
        }
        words = term_lower.split()
        core_words = [w for w in words if w not in _DESCRIPTORS]
        core_term = " ".join(core_words) if core_words else term_lower
        if core_term != term_lower:
            core_results = _process.extract(
                core_term, all_targets.keys(), scorer=_fuzz.WRatio, limit=20
            )
        else:
            core_results = results  # nothing stripped; use original results

        collected: list[str] = []
        seen: set[str] = set()
        _MIN_COLLECT_LEN = max(len(core_term) // 2, 5)
        for label, score, _ in core_results:
            if score < _SEMANTIC_THRESHOLD:
                break
            if len(label) < _MIN_COLLECT_LEN:
                continue
            w = weight_db.get(field, {}).get(label, 1.0)
            eff = _SEMANTIC_THRESHOLD + (1.0 - w) * _WEIGHT_BOOST
            if score >= eff:
                for code in all_targets[label]:
                    if code not in seen:
                        seen.add(code)
                        collected.append(code)
        if collected:
            return collected

    return all_targets[best_label]


# ── Main retriever class ───────────────────────────────────────────────────────

class MetadataRetriever:
    """
    Unified retriever for LLM_preprocessor/preprocess.py.

    Parameters
    ----------
    name_index_path : path to the JSON file produced by build_name_index.py
                      (contains {"authors": [...], "funding_sources": [...]})
    db_url : optional PostgreSQL connection URL.  When set, db_verify_codes()
             cross-checks resolved codes against the live database and flags
             any that map to zero rows (DB_MISS).
    """

    def __init__(self, name_index_path: str | Path, db_url: Optional[str] = None):
        path = Path(name_index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Name index not found: {path}\n"
                "Build it with:\n"
                "  python RAG/build_name_index.py --db-url <url> --out RAG/name_index.json"
            )
        with open(path, encoding="utf-8") as fh:
            self._names: Dict[str, List[str]] = json.load(fh)
        self._db_url: Optional[str] = db_url

    # ------------------------------------------------------------------
    def db_verify_codes(
        self,
        field: str,
        codes: List[str],
    ) -> tuple[List[str], List[str]]:
        """Return (live_codes, db_miss_codes) for *codes* in *field*.

        live_codes     : codes that have ≥ 1 row in the DB column.
        db_miss_codes  : codes that resolved in the YAML vocabulary but have
                         zero rows in the live database.  These should NOT be
                         used in EXISTS filters — they will always return empty.

        Returns (codes, []) unchanged when db_url was not provided at init or
        when psycopg2 is not installed.
        """
        if not self._db_url or not codes:
            return codes, []
        if _field_to_db_col(field) is None:
            return codes, []

        table, col = _field_to_db_col(field)
        try:
            import psycopg2  # type: ignore
        except ImportError:
            log.debug("psycopg2 not installed — skipping DB back-check")
            return codes, []

        try:
            with psycopg2.connect(self._db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT DISTINCT {col} FROM {table} WHERE {col} = ANY(%s)",
                        (codes,),
                    )
                    live_set = {row[0] for row in cur.fetchall()}
        except Exception as exc:
            log.warning("DB back-check failed for field=%s: %s", field, exc)
            return codes, []

        live = [c for c in codes if c in live_set]
        miss = [c for c in codes if c not in live_set]
        if miss:
            log.debug("DB_MISS for field=%s: %s", field, miss)
        return live, miss

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
                term_lower = term.lower().strip()
                # Skip task field for pure modality abbreviations and generic
                # imaging terms — fuzzy matching produces spurious task hits
                # (e.g. "fMRI" → bold_acquisition_general, "DTI" → dual_task).
                if field == "task" and term_lower in (
                    _PURE_MODALITY_TERMS | _GENERIC_IMAGING_TERMS
                ):
                    continue
                # Skip suffix field for generic imaging terms — suffix fuzzy
                # matching produces over-specific filters (e.g. "brain imaging"
                # → t1_weighted_mri) that make queries unnecessarily restrictive.
                if field == "suffix" and term_lower in _GENERIC_IMAGING_TERMS:
                    continue
                codes = _resolve_yaml_term(field, term)
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
