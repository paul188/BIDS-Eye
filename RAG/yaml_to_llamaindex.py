import hashlib
import json
import re as _re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    import numpy as _np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False


# ---------------------------------------------------------------------------
# Internal helpers (unchanged from tree version)
# ---------------------------------------------------------------------------

def _extract_synonyms(raw: Any) -> List[Tuple[str, float]]:
    """Parse a synonyms entry into (term, weight) pairs.

    Accepts both the legacy plain-string format and the weighted dict format:
      "rsfMRI"                       → ("rsfMRI", 1.0)
      {"term": "CPT", "weight": 0.8} → ("CPT", 0.8)
    """
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    result: List[Tuple[str, float]] = []
    for s in raw:
        if isinstance(s, str):
            term = s.strip()
            if term:
                result.append((term, 1.0))
        elif isinstance(s, dict) and "term" in s:
            term = str(s["term"]).strip()
            weight = float(s.get("weight", 1.0))
            if term:
                result.append((term, max(0.0, min(1.0, weight))))
    return result


# ---------------------------------------------------------------------------
# Flat SKOS parser — two-pass algorithm
# ---------------------------------------------------------------------------

# Fields that are concept metadata, not child concepts
_CONCEPT_META = frozenset({
    "label", "standard_code", "is_group", "broader", "part_of", "description",
    "definition_text", "ca_id", "synonyms", "codes", "extra_codes",
    "dataset_codes", "count",
})


def _parse_flat(
    category: str,
    concepts: Dict[str, dict],
    leaf_db: Dict[str, Dict[str, str]],
    display_db: Dict[str, Dict[str, str]],
    group_db: Dict[str, Dict[str, List[str]]],
    group_display_db: Dict[str, Dict[str, List[str]]],
    hierarchy_db: Dict[str, Dict],
    weight_db: Dict[str, Dict[str, float]],
) -> None:
    """Populate all six lookup dicts from a flat SKOS category dict.

    Pass 1 — Register leaf concepts (standard_code present):
        Builds leaf_db, display_db, weight_db, hierarchy_db.

    Pass 2 — Build group lookups:
        Derives a children map from 'broader' back-links, then DFS-aggregates
        all descendant standard_codes for each group/dual node.
        Builds group_db and group_display_db.
    """

    # ── Pass 1: leaf concepts ─────────────────────────────────────────────────
    for key, value in concepts.items():
        std_code = value.get("standard_code")
        if not std_code:
            continue  # pure group — handled in pass 2

        label     = value.get("label", "")
        syn_pairs = _extract_synonyms(value.get("synonyms"))
        syn_terms = [t for t, _ in syn_pairs]

        # leaf_db: all surface forms → standard_code (for exact match)
        all_terms: set = set()
        if label:
            all_terms.add(label.lower())
        all_terms.add(std_code.lower())
        for syn in syn_terms:
            all_terms.add(syn.lower())
        for code in value.get("codes") or []:
            all_terms.add(str(code).lower())
        for dc in value.get("dataset_codes") or []:
            if isinstance(dc, dict) and "raw" in dc:
                all_terms.add(str(dc["raw"]).lower())
        for term in all_terms:
            if term:
                leaf_db[category][term] = std_code

        # display_db + weight_db: label (w=1.0) + each synonym with its weight
        if label:
            t = label.lower()
            display_db[category][t] = std_code
            weight_db[category][t] = 1.0
        for syn_term, syn_weight in syn_pairs:
            t = syn_term.lower()
            if t:
                display_db[category][t] = std_code
                weight_db[category][t] = syn_weight

        # hierarchy_db: ancestor chain (std_codes only — group nodes are transparent)
        hierarchy_db[std_code] = {
            "path": _ancestor_codes(key, concepts),
            "label": label or key,
            "description": value.get("description", ""),
        }

    # ── Pass 2: group lookups ─────────────────────────────────────────────────
    # Build children map: concept_key → [direct child keys]
    # Follows both KINDOF (broader) and PARTOF (part_of) edges so that
    # querying "attention" expands to all its parts and kinds alike.
    children: Dict[str, List[str]] = defaultdict(list)
    for key, value in concepts.items():
        for parent_key in (value.get("broader") or []) + (value.get("part_of") or []):
            children[parent_key].append(key)

    # Memoised DFS: collect all descendant standard_codes under a concept.
    # Uses a sentinel to detect back-edges (cycles introduced by CA PARTOF edges)
    # and returns [] for any node that is currently being computed, breaking the
    # cycle without raising RecursionError.
    _COMPUTING = object()
    _cache: Dict[str, object] = {}

    def _all_codes(key: str) -> List[str]:
        cached = _cache.get(key)
        if cached is _COMPUTING:
            return []          # cycle detected — skip this back-edge
        if cached is not None:
            return cached      # type: ignore[return-value]
        _cache[key] = _COMPUTING
        v = concepts.get(key, {})
        codes: List[str] = []
        if "standard_code" in v:
            codes.append(v["standard_code"])
        for child_key in children.get(key, []):
            codes.extend(_all_codes(child_key))
        result = list(dict.fromkeys(codes))
        _cache[key] = result
        return result

    for key, value in concepts.items():
        # Only register group entries for concepts that ARE parents (have children)
        if not children.get(key):
            continue  # pure leaf with no children — no group entry needed

        leaf_codes = _all_codes(key)
        if not leaf_codes:
            continue

        label     = value.get("label", "")
        syn_pairs = _extract_synonyms(value.get("synonyms"))
        syn_terms = [t for t, _ in syn_pairs]

        # group_db: key + label + synonyms → for exact group lookup
        for term in [key] + ([label] if label else []) + syn_terms:
            t = str(term).lower()
            if t:
                existing = group_db[category].get(t, [])
                group_db[category][t] = list(dict.fromkeys(existing + leaf_codes))

        # group_display_db: label + synonyms only → for fuzzy group lookup
        for term in ([label] if label else []) + syn_terms:
            t = str(term).lower()
            if t:
                existing = group_display_db[category].get(t, [])
                group_display_db[category][t] = list(dict.fromkeys(existing + leaf_codes))


def _ancestor_codes(key: str, concepts: Dict[str, dict]) -> List[str]:
    """Return the list of ancestor concept keys that have standard_codes.

    Group nodes (no standard_code) are transparent — traversed but not listed.
    The node's own key is always first (self-referential path for pruning logic).
    Used by retriever._prune_redundant_parents.
    """
    path: List[str] = [key]
    visited: set = {key}
    node0 = concepts.get(key, {})
    queue = list((node0.get("broader") or []) + (node0.get("part_of") or []))
    while queue:
        parent_key = queue.pop(0)
        if parent_key in visited:
            continue
        visited.add(parent_key)
        parent = concepts.get(parent_key, {})
        if "standard_code" in parent:
            path.append(parent_key)
        queue.extend((parent.get("broader") or []) + (parent.get("part_of") or []))
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CATEGORIES = [
    "diagnosis", "task", "suffix", "handedness", "sex",
    "datatype", "sidecar_fields", "participant_extra_fields",
    "cognitive_construct",
]


def build_knowledge_base(yaml_path: str, ca_constructs_path: Optional[str] = None) -> Tuple[
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict],
    Dict[str, Dict[str, float]],
]:
    """Parse the flat SKOS value_mappings.yaml into six lookup structures.

    Returns
    -------
    leaf_db : {category: {surface_form_lower: standard_code}}
        Exact-match lookup for leaf concepts.
        Covers label, standard_code, synonyms, codes, and dataset_codes.

    display_db : {category: {human_readable_lower: standard_code}}
        Fuzzy-match lookup for leaf concepts (label + synonyms only).

    group_db : {category: {term_lower: [standard_code, ...]}}
        Exact-match lookup for group/dual concepts (all surface forms).

    group_display_db : {category: {synonym_lower: [standard_code, ...]}}
        Fuzzy-match lookup for group/dual concepts (label + synonyms only).

    hierarchy_db : {standard_code: {path, label, description}}
        Ancestor chain and metadata for every leaf concept.
        'path' contains the concept's own key plus any ancestor keys that
        have standard_codes — used by retriever._prune_redundant_parents.

    weight_db : {category: {term_lower: float}}
        Match-confidence weight for every term in display_db.
        Labels → 1.0. Plain-string synonyms → 1.0.
        Weighted-dict synonyms → their declared weight.
    """
    with open(yaml_path, "r") as fh:
        schema = yaml.safe_load(fh)

    # Merge cognitive_atlas_constructs.yaml if available
    if ca_constructs_path and Path(ca_constructs_path).exists():
        with open(ca_constructs_path, "r") as fh:
            ca_schema = yaml.safe_load(fh) or {}
        for cat, data in ca_schema.items():
            if isinstance(data, dict):
                schema.setdefault(cat, {}).update(data)

    leaf_db:         Dict[str, Dict[str, str]]        = {cat: {} for cat in _CATEGORIES}
    display_db:      Dict[str, Dict[str, str]]        = {cat: {} for cat in _CATEGORIES}
    group_db:        Dict[str, Dict[str, List[str]]]  = {cat: {} for cat in _CATEGORIES}
    group_display_db:Dict[str, Dict[str, List[str]]]  = {cat: {} for cat in _CATEGORIES}
    hierarchy_db:    Dict[str, Dict]                  = {}
    weight_db:       Dict[str, Dict[str, float]]      = {cat: {} for cat in _CATEGORIES}

    for cat in _CATEGORIES:
        if cat not in schema:
            continue
        cat_data = schema[cat]
        if not isinstance(cat_data, dict):
            continue
        _parse_flat(
            cat, cat_data,
            leaf_db, display_db, group_db, group_display_db, hierarchy_db, weight_db,
        )

    return leaf_db, display_db, group_db, group_display_db, hierarchy_db, weight_db


def build_measures_db(bridge_path: Optional[str] = None) -> Dict[str, List[str]]:
    """Build reverse index: cognitive construct key → [BIDS task standard_codes].

    Populated from bids_to_ca_map.json (the curated human-reviewed bridge
    generated by scripts/generate_task_bridge.py).  Returns {} gracefully when
    the file does not exist so the rest of the system keeps working without CA.
    """
    if not bridge_path or not Path(bridge_path).exists():
        return {}
    import json as _json
    with open(bridge_path, "r", encoding="utf-8") as fh:
        bridge: Dict[str, dict] = _json.load(fh)
    result: Dict[str, List[str]] = {}
    for task_code, entry in bridge.items():
        for construct_key in entry.get("measures") or []:
            result.setdefault(construct_key, [])
            if task_code not in result[construct_key]:
                result[construct_key].append(task_code)
    return result


def get_group_summary(group_db: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """Return a flat map of {category: [group_synonyms]} for use in LLM prompts."""
    summary: Dict[str, List[str]] = {}
    for cat, groups in group_db.items():
        if groups:
            summary[cat] = sorted(groups.keys())
    return summary


# EmbeddingIndex: {category: (terms, float32_embeddings, code_lists)}
EmbeddingIndex = Dict[str, Tuple[List[str], Any, List[List[str]]]]


def build_embedding_index(
    model_name: str,
    yaml_path: str,
    cache_dir: Optional[str] = None,
) -> Optional[EmbeddingIndex]:
    """Build or load from cache a per-category L2-normalised embedding index.

    Covers all human-readable terms from display_db and group_display_db.
    Cache is keyed on YAML content hash + model name — auto-invalidates when
    value_mappings.yaml changes.

    Returns None if sentence_transformers or numpy are unavailable.
    """
    if not _NUMPY_OK:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    import numpy as np

    yaml_hash  = hashlib.md5(Path(yaml_path).read_bytes()).hexdigest()[:12]
    model_slug = _re.sub(r"[^a-zA-Z0-9]", "_", model_name)
    cache_base = Path(cache_dir) if cache_dir else Path(yaml_path).parent
    meta_path  = cache_base / f"embed_{model_slug}_{yaml_hash}.json"
    embs_path  = cache_base / f"embed_{model_slug}_{yaml_hash}.npz"

    # ── Load from cache if valid ───────────────────────────────────────────────
    if meta_path.exists() and embs_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            npz = np.load(str(embs_path))
            index: EmbeddingIndex = {}
            for cat, cat_meta in meta.items():
                if cat in npz:
                    index[cat] = (cat_meta["terms"], npz[cat], cat_meta["codes"])
            if index:
                return index
        except Exception:
            pass

    # ── Build per-category (term, code_list) pairs ────────────────────────────
    # Re-parse so we have access to display_db and group_display_db
    _, display_db_, _, group_display_db_, _, _ = build_knowledge_base(yaml_path)

    cat_entries: Dict[str, List[Tuple[str, List[str]]]] = {}
    for cat in _CATEGORIES:
        seen: set = set()
        entries: List[Tuple[str, List[str]]] = []
        for term, code in display_db_.get(cat, {}).items():
            if term not in seen:
                entries.append((term, [code]))
                seen.add(term)
        for term, codes in group_display_db_.get(cat, {}).items():
            if term not in seen:
                entries.append((term, list(codes)))
                seen.add(term)
        if entries:
            cat_entries[cat] = entries

    if not cat_entries:
        return None

    # ── Encode ────────────────────────────────────────────────────────────────
    model = SentenceTransformer(model_name)
    index = {}
    meta_out: Dict[str, dict] = {}
    embs_out: Dict[str, Any]  = {}

    for cat, entries in cat_entries.items():
        terms      = [t for t, _ in entries]
        code_lists = [c for _, c in entries]
        embs       = model.encode(terms, normalize_embeddings=True, show_progress_bar=False)
        embs32     = np.array(embs, dtype=np.float32)
        index[cat]    = (terms, embs32, code_lists)
        meta_out[cat] = {"terms": terms, "codes": code_lists}
        embs_out[cat] = embs32

    # ── Persist cache ─────────────────────────────────────────────────────────
    try:
        cache_base.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta_out, fh)
        np.savez(str(embs_path), **embs_out)
    except Exception:
        pass

    return index


# ---------------------------------------------------------------------------
# Module-level initialisation (used by downstream modules)
# ---------------------------------------------------------------------------

_YAML_PATH     = str(Path(__file__).parent / "value_mappings.yaml")
_CA_PATH       = str(Path(__file__).parent / "cognitive_atlas_constructs.yaml")
_BRIDGE_PATH   = str(Path(__file__).parent / "bids_to_ca_map.json")

leaf_db, display_db, group_db, group_display_db, hierarchy_db, weight_db = build_knowledge_base(
    _YAML_PATH,
    _CA_PATH if Path(_CA_PATH).exists() else None,
)
measures_db: Dict[str, List[str]] = build_measures_db(
    _BRIDGE_PATH if Path(_BRIDGE_PATH).exists() else None
)
