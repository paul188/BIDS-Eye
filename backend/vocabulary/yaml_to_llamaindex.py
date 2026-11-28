import hashlib
import json
import logging
import re as _re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

log = logging.getLogger(__name__)

try:
    import numpy as _np  # noqa: F401  (availability probe)
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
    "dataset_codes", "count", "text_words", "measures",
})


# Every code that lists a term, in file order. leaf_db and display_db keep one
# code per term (the last concept in the file); these keep all of them.
#   leaf_codes_db    {category: {surface_form_lower: [standard_code, ...]}}
#   display_codes_db {category: {label_or_synonym_lower: [standard_code, ...]}}
leaf_codes_db: Dict[str, Dict[str, List[str]]] = {}
display_codes_db: Dict[str, Dict[str, List[str]]] = {}

# What a dataset's own text says, as opposed to what a request or a column
# value says (read through backend/vocabulary/text_evidence.py):
#   text_words_db    {category: {standard_code: [word, ...]}}  an entry's text_words
#   text_evidence_db the vocabulary's text_evidence section as written
text_words_db: Dict[str, Dict[str, List[str]]] = {}
text_evidence_db: Dict[str, Any] = {}

# What each task measures, reversed: a cognitive construct → the task codes
# whose value_mappings.yaml entry lists it under `measures` (see
# tools/apply_task_measures.py). Only tasks with a standard_code count: a
# group has no datasets of its own.
#   measures_db {construct_code: [task standard_code, ...]}
measures_db: Dict[str, List[str]] = {}


# Shorthands the vocabulary lists, lower-cased: a label or synonym written as
# one short token with at least two capitals ("APT", "TLE", "fMRI", "T1w").
# A term typed as one of them, in any case, is a shorthand (retriever.is_shorthand).
shorthand_db: set = set()
# The labels of the entries that list a shorthand, groups included:
#   shorthand_label_db {category: {shorthand_lower: [label, ...]}}
shorthand_label_db: Dict[str, Dict[str, List[str]]] = {}
# The label of every entry by its key, groups included (hierarchy_db has leaves only):
#   entry_label_db {category: {key_lower: label}}
entry_label_db: Dict[str, Dict[str, str]] = {}
_SHORTHAND_RE = _re.compile(r"[A-Za-z][A-Za-z0-9]{1,5}")


def looks_like_shorthand(term: str) -> bool:
    """One token of two to six letters and digits with at least two capitals."""
    term = (term or "").strip()
    return bool(_SHORTHAND_RE.fullmatch(term)) and sum(c.isupper() for c in term) >= 2


def _collect_shorthands(category: str, concepts: Dict[str, dict]) -> None:
    for key, value in concepts.items():
        if not isinstance(value, dict):
            continue
        label = value.get("label") or str(key).replace("_", " ")
        entry_label_db.setdefault(category, {})[str(key).lower()] = label
        names = [value.get("label") or ""] + [t for t, _ in _extract_synonyms(value.get("synonyms"))]
        for name in names:
            if looks_like_shorthand(name):
                shorthand_db.add(name.strip().lower())
                labels = shorthand_label_db.setdefault(category, {}).setdefault(name.strip().lower(), [])
                if label not in labels:
                    labels.append(label)


def _add_code(index: Dict[str, Dict[str, List[str]]], category: str, term: str, code: str) -> None:
    codes = index.setdefault(category, {}).setdefault(term, [])
    if code not in codes:
        codes.append(code)


def _parse_flat(
    category: str,
    concepts: Dict[str, dict],
    leaf_db: Dict[str, Dict[str, str]],
    display_db: Dict[str, Dict[str, str]],
    group_db: Dict[str, Dict[str, List[str]]],
    group_display_db: Dict[str, Dict[str, List[str]]],
    hierarchy_db: Dict[str, Dict],
    weight_db: Dict[str, Dict[str, float]],
    parts: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Populate all six lookup dicts from a flat SKOS category dict.

    Pass 1 — Register leaf concepts (standard_code present):
        Builds leaf_db, display_db, weight_db, hierarchy_db.

    Pass 2 — Build group lookups:
        Derives a children map from 'broader' back-links, then DFS-aggregates
        all descendant standard_codes for each group/dual node.
        Builds group_db and group_display_db.

    *parts* ({whole: [part, ...]}, construct_parts.yaml) adds one PARTOF step:
    a group also covers the KINDOF descendants of the listed parts of itself
    and of each of its KINDOF descendants, but never the parts of a part.
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
                _add_code(leaf_codes_db, category, term, std_code)

        # display_db + weight_db: label (w=1.0) + each synonym with its weight
        if label:
            t = label.lower()
            display_db[category][t] = std_code
            _add_code(display_codes_db, category, t, std_code)
            weight_db[category][t] = 1.0
        for syn_term, syn_weight in syn_pairs:
            t = syn_term.lower()
            if t:
                display_db[category][t] = std_code
                _add_code(display_codes_db, category, t, std_code)
                weight_db[category][t] = syn_weight

        # hierarchy_db: ancestor chain (std_codes only — group nodes are transparent)
        hierarchy_db[std_code] = {
            "path": _ancestor_codes(key, concepts),
            "label": label or key,
            "description": value.get("description", ""),
        }

        text_words = [str(word).lower() for word in value.get("text_words") or []]
        if text_words:
            text_words_db.setdefault(category, {})[std_code] = text_words

    # ── Pass 2: group lookups ─────────────────────────────────────────────────
    # Build children map: concept_key → [direct child keys]
    # Follows KINDOF (broader) edges only. The Cognitive Atlas PARTOF edges
    # (part_of, which only cognitive_atlas_constructs.yaml has) do not say that
    # a study of the part is a study of the whole: working memory is PARTOF
    # decision making, decision making PARTOF set shifting, so "set shifting"
    # reached 153 tasks, most of them working-memory and reasoning tasks.
    children: Dict[str, List[str]] = defaultdict(list)
    for key, value in concepts.items():
        for parent_key in value.get("broader") or []:
            children[parent_key].append(key)

    # Memoised DFS: collect all descendant standard_codes under a concept.
    # Uses a sentinel to detect back-edges (a cycle in the imported edges)
    # and returns [] for any node that is currently being computed, breaking the
    # cycle without raising RecursionError.
    computing = object()
    _cache: Dict[str, object] = {}

    def _all_codes(key: str) -> List[str]:
        cached = _cache.get(key)
        if cached is computing:
            return []          # cycle detected — skip this back-edge
        if cached is not None:
            return cached      # type: ignore[return-value]
        _cache[key] = computing
        v = concepts.get(key, {})
        codes: List[str] = []
        if "standard_code" in v:
            codes.append(v["standard_code"])
        for child_key in children.get(key, []):
            codes.extend(_all_codes(child_key))
        result = list(dict.fromkeys(codes))
        _cache[key] = result
        return result

    parts = parts or {}

    def _kind_keys(key: str) -> List[str]:
        """*key* and its KINDOF descendants."""
        seen = [key]
        stack = [key]
        while stack:
            for child_key in children.get(stack.pop(), []):
                if child_key not in seen:
                    seen.append(child_key)
                    stack.append(child_key)
        return seen

    def _group_codes(key: str) -> List[str]:
        codes = list(_all_codes(key))
        if parts:
            for kind_key in _kind_keys(key):
                for part_key in parts.get(kind_key, []):
                    codes.extend(_all_codes(part_key))
        return list(dict.fromkeys(codes))

    for key, value in concepts.items():
        # Only register group entries for concepts that ARE parents (have
        # children or listed parts)
        if not children.get(key) and not parts.get(key):
            continue  # pure leaf with no children — no group entry needed

        leaf_codes = _group_codes(key)
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
    The node's own key is always first. KINDOF (broader) edges only, like the
    group lookups (see _parse_flat).
    """
    path: List[str] = [key]
    visited: set = {key}
    node0 = concepts.get(key, {})
    queue = list(node0.get("broader") or [])
    while queue:
        parent_key = queue.pop(0)
        if parent_key in visited:
            continue
        visited.add(parent_key)
        parent = concepts.get(parent_key, {})
        if "standard_code" in parent:
            path.append(parent_key)
        queue.extend(parent.get("broader") or [])
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
        have standard_codes.

    weight_db : {category: {term_lower: float}}
        The "weight" of every term in display_db: how often the synonym
        occurred in the paper abstracts it was proposed from
        (backend/vocabulary/tools/expand_synonyms.py). Labels → 1.0. Plain-string synonyms → 1.0.
        It is not a match confidence and the resolver does not read it. Its
        one use is in backend/vocabulary/text_terms.py, which searches a diagnosis by its
        most common synonyms first.
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

    text_evidence_db.clear()
    text_evidence_db.update(schema.get("text_evidence") or {})
    measures_db.clear()
    measures_db.update(build_measures_db(schema.get("task") or {}))

    leaf_db:         Dict[str, Dict[str, str]]        = {cat: {} for cat in _CATEGORIES}
    display_db:      Dict[str, Dict[str, str]]        = {cat: {} for cat in _CATEGORIES}
    group_db:        Dict[str, Dict[str, List[str]]]  = {cat: {} for cat in _CATEGORIES}
    group_display_db:Dict[str, Dict[str, List[str]]]  = {cat: {} for cat in _CATEGORIES}
    hierarchy_db:    Dict[str, Dict]                  = {}
    weight_db:       Dict[str, Dict[str, float]]      = {cat: {} for cat in _CATEGORIES}

    construct_parts = load_construct_parts()
    for cat in _CATEGORIES:
        if cat not in schema:
            continue
        cat_data = schema[cat]
        if not isinstance(cat_data, dict):
            continue
        _parse_flat(
            cat, cat_data,
            leaf_db, display_db, group_db, group_display_db, hierarchy_db, weight_db,
            construct_parts if cat == "cognitive_construct" else None,
        )
        _collect_shorthands(cat, cat_data)

    return leaf_db, display_db, group_db, group_display_db, hierarchy_db, weight_db


CONSTRUCT_PARTS_PATH = Path(__file__).parent / "construct_parts.yaml"


def load_construct_parts(path: Path = CONSTRUCT_PARTS_PATH) -> Dict[str, List[str]]:
    """The Cognitive Atlas PARTOF links construct expansion follows: {whole: [part, ...]}.

    See construct_parts.yaml for why only these. Without the file no PARTOF
    link is followed.
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {str(whole): [str(p) for p in listed or []] for whole, listed in data.items()}


def build_measures_db(tasks: Dict[str, Any]) -> Dict[str, List[str]]:
    """Reverse the `measures` lists of the task entries: construct code → [task codes].

    *tasks* is the task section of value_mappings.yaml. A group entry's
    measures are ignored: its key is no task code.
    """
    result: Dict[str, List[str]] = {}
    for entry in tasks.values():
        if not isinstance(entry, dict) or not entry.get("standard_code"):
            continue
        for construct_key in entry.get("measures") or []:
            codes = result.setdefault(construct_key, [])
            if entry["standard_code"] not in codes:
                codes.append(entry["standard_code"])
    return result


# EmbeddingIndex: {category: (terms, float32_embeddings, code_lists)}
EmbeddingIndex = Dict[str, Tuple[List[str], Any, List[List[str]]]]


class EmbeddingIndexError(Exception):
    """The embedding index could not be built: the model could not be loaded
    (download, files), value_mappings.yaml not read, or encoding failed."""


def build_embedding_index(
    model_name: str,
    yaml_path: str,
    cache_dir: Optional[str] = None,
) -> Optional[EmbeddingIndex]:
    """Build or load from cache a per-category L2-normalised embedding index.

    Covers all human-readable terms from display_db and group_display_db.
    Cache is keyed on YAML content hash + model name — auto-invalidates when
    value_mappings.yaml changes.

    Returns None if sentence_transformers or numpy are unavailable. Raises
    EmbeddingIndexError when the index cannot be built.
    """
    try:
        return _build_embedding_index(model_name, yaml_path, cache_dir)
    # OSError: the model download (requests and huggingface_hub errors are
    # OSErrors) or a file; RuntimeError: torch while encoding.
    except (OSError, yaml.YAMLError, RuntimeError) as exc:
        raise EmbeddingIndexError(f"embedding index for {model_name}: {exc}") from exc


def _build_embedding_index(
    model_name: str,
    yaml_path: str,
    cache_dir: Optional[str],
) -> Optional[EmbeddingIndex]:
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
        # A cache written by another version or cut short: rebuild it.
        # np.load raises EOFError, ValueError or BadZipFile on a broken file;
        # KeyError is a meta file of another shape.
        except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            log.info("Embedding cache %s not usable (%s); rebuilding", embs_path.name, exc)

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
    except OSError as exc:  # the next start encodes again
        log.warning("Embedding cache not written to %s (%s)", cache_base, exc)

    return index


# ---------------------------------------------------------------------------
# Module-level initialisation (used by downstream modules)
# ---------------------------------------------------------------------------

_YAML_PATH     = str(Path(__file__).parent / "value_mappings.yaml")
_CA_PATH       = str(Path(__file__).parent / "cognitive_atlas_constructs.yaml")

leaf_db, display_db, group_db, group_display_db, hierarchy_db, weight_db = build_knowledge_base(
    _YAML_PATH,
    _CA_PATH if Path(_CA_PATH).exists() else None,
)
