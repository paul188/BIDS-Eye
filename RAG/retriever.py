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
import re
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
    measures_db,
    build_embedding_index,
)
from join_registry import field_to_db_col as _field_to_db_col
import institution_index as _institution_index

# ── Constants ──────────────────────────────────────────────────────────────────

# Fields backed by value_mappings.yaml (semantic / hierarchical matching)
_YAML_FIELDS = {"diagnosis", "task", "suffix", "datatype", "sex", "handedness"}

# Fields backed by the flat name index (token similarity matching)
_NAME_FIELDS = {"author", "funding"}

# Fields searched together for scan/modality/task terms
_SCAN_FIELDS = ("datatype", "suffix", "task")

# Institutions: exact phrase/token matching against the stored names
# (RAG/institution_index.py), not token-set similarity — 'Beijing Normal
# University' must not pull in 'Jiangsu Normal University'.
_INSTITUTION_FIELD = "institution"

# RAGField.value → name_index.json key
_NAME_FIELD_KEY = {
    "author":  "authors",
    "funding": "funding_sources",
}

# WRatio threshold for YAML fields — keeps false positives low on short tokens
_SEMANTIC_THRESHOLD = 80

# Higher WRatio threshold applied specifically to the task field.
# The construct path (_resolve_via_constructs) fires before fuzzy for task
# queries and is the preferred route for construct-level terms.  Raising the
# fuzzy bar here prevents weak word-overlap hits (e.g. "social cognition" →
# ultimatum_game_responder) from contaminating task resolution when the
# construct path legitimately returns nothing.
_TASK_SEMANTIC_THRESHOLD = 88

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

# No fuzzy match below this WRatio is used, whatever the synonym weight.
# At or above it, one string contains the other or the two differ by a typo.
# Below it, a label can qualify on a single shared word: WRatio gives 85.5 to
# any two strings of unequal length with one word in common ("structural mri" /
# "bold mri", "down syndrome" / "west syndrome", "conduct disorder" /
# "major depressive disorder"). Same value as _TASK_SEMANTIC_THRESHOLD, which
# was raised for the same reason.
_MIN_FUZZY_SCORE = 88

# Codes that never narrow a search. entities_and_processing_keys is a bucket
# for BIDS bookkeeping files (participants, channels, coordsystem, masks, ...)
# that almost every dataset has, so a filter on it returns everything: "reward
# processing" matched its label through "processing" and the question
# "Datasets with reward processing tasks" returned all 1,719 datasets.
# Such codes are never returned; the term falls back to text search instead.
_NON_FILTERING_CODES = frozenset({"entities_and_processing_keys"})

# Diagnosis concepts that describe animals, not human participants. A fuzzy
# match reaches them only when the term itself is about animals or models;
# otherwise "Parkinson's" picks up "Parkinson's model of ATP10B knockdown".
# Exact synonyms ("mouse", "ATP10B KD mouse model") are not affected.
_ANIMAL_MODEL_GROUP = "animal_preclinical_models"
_ANIMAL_WORDS = re.compile(
    r"\b(?:animals?|mouse|mice|rats?|rodents?|murine|primates?|macaques?|monkeys?|"
    r"nhp|models?|preclinical|pre-clinical|knock-?outs?|knock-?downs?|ko|kd|"
    r"wild[- ]?type|transgenic|in vivo)\b"
)

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


def _embedding_fallback_scored(field: str, term: str) -> List[tuple[str, float]]:
    """Scored variant of _embedding_fallback.

    Two-tier biomedical embedding search for *term* in *field*.
    Called when RapidFuzz WRatio misses (score < threshold).
    Tier 1: BioLORD-2023-C, cosine ≥ 0.75 — returns on first hit.
    Tier 2: SapBERT,         cosine ≥ 0.65 — reached only if tier 1 misses.
    Returns [(code, cosine_score), ...] for the best hit, or [] if both tiers
    miss or sentence-transformers is not installed.
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
            score = float(scores[best_idx])
            log.debug(
                "Embedding fallback (%s) resolved '%s' → %s (score %.3f)",
                model_name.split("/")[-1], term, code_lists[best_idx], score,
            )
            return [(code, score) for code in code_lists[best_idx]]

    return []


def _embedding_fallback(field: str, term: str) -> List[str]:
    """Two-tier biomedical embedding search (codes only — thin wrapper)."""
    return [code for code, _ in _embedding_fallback_scored(field, term)]


# ── YAML-field resolution helpers ──────────────────────────────────────────────
# These mirror the logic in RAG_postprocess.py but accept plain strings instead
# of UserQueryEntities, and return a list of codes for a single term.

def _aligned_to_words(term: str, label: str) -> bool:
    """False if one string sits inside the other only part-way through a word.

    WRatio scores containment at 90 even inside a word: "als" in "cerebral
    palsy", "chi" in "td children", "phobia" in "agoraphobia".
    - A label inside the term must cover whole words of the term.
    - The term inside a label must start at a word, but may end mid-word, so
      "t1" still finds "t1w" and "trauma" still finds "post-traumatic".
    Strings where neither contains the other (typos, spelling variants) are
    not affected. Both strings are expected in _fuzzy_form.
    """
    short, long = sorted((term, label), key=len)
    if short not in long:
        return True
    must_end_at_word = short == label
    for m in re.finditer(re.escape(short), long):
        starts = m.start() == 0 or not long[m.start() - 1].isalnum()
        ends = m.end() == len(long) or not long[m.end()].isalnum()
        if starts and (ends or not must_end_at_word):
            return True
    return False


def _fuzzy_form(text: str) -> str:
    """Lower-case and treat - _ / as spaces, so "near infrared" equals "near-infrared".

    Other punctuation is kept on purpose: "*" separates T2* from T2, and the
    apostrophe keeps "parkinson's" one word.
    """
    return " ".join(re.sub(r"[-_/]+", " ", text.lower()).split())


# Words that say what kind of record or person the user wants, not which
# concept. "iEEG recording" means the iEEG concept; fuzzy matching on the full
# phrase prefers "MEG recording" (one letter apart) over it.
_FILLER_WORDS = frozenset({
    "data", "dataset", "datasets", "recording", "recordings", "file", "files",
    "participants", "patients", "subjects", "individuals", "cohort",
})


def _without_filler(term: str) -> str:
    return " ".join(w for w in term.split() if w not in _FILLER_WORDS)


# Words that say a term is a task without naming which one. In the task field
# "decision making", "decision making task" and "decision making tasks" are one
# request and must resolve identically; see _match_task_term.
_TASK_SCAFFOLD_WORDS = ("task", "paradigm", "assessment")
_TASK_SCAFFOLD_FORMS = frozenset(
    form for word in _TASK_SCAFFOLD_WORDS for form in (word, word + "s")
)


def _task_core(term: str) -> str:
    """*term* without task scaffold words; "" if it names no task at all ("tasks")."""
    return " ".join(w for w in term.split() if w not in _TASK_SCAFFOLD_FORMS)


def _match_task_term(core: str) -> Optional[List[tuple[str, float]]]:
    """Exact and construct matches for a task term, the same for every phrasing.

    The vocabulary files the same request in different places. "decision
    making" is a cognitive construct linked to 45 tasks, while "decision making
    task" is a synonym of the single code decision_task_general. "language" is
    a raw task label one dataset used (mapped to language_control), while
    "language tasks" reached the language construct and its 22 tasks. Taking
    the first hit made the result depend on whether the user typed "tasks".

    Every phrasing is first reduced to its core (_task_core); the result is
    the union of
      1. the core itself, if it is a task label, synonym or group name
         ("theory of mind", "reward processing"),
      2. single tasks named by the core plus a scaffold word ("lexical decision
         task", "decision making task").
    A core that is not such a name also gets
      3. its raw dataset label, if it is one ("language", "reward", "pain":
         what one dataset called its task, not what the word means),
      4. groups named by the core plus a scaffold word ("reward task"),
      5. the construct of that exact name ("decision making", "language").
    A named core does not get 4 and 5: "theory of mind task" is also listed
    under the whole social-cognition group, which is not what was asked.
    Only when none of 1 to 5 exists does a fuzzy construct match count.
    Returns None when nothing matched, so the caller goes on to fuzzy matching.
    """
    cat_leaves  = leaf_db.get("task", {})
    cat_groups  = group_db.get("task", {})
    cat_display = display_db.get("task", {})

    # "decision-making" and "decision making" are one spelling here.
    spellings = list(dict.fromkeys([core, _fuzzy_form(core)]))

    def forms(suffix: str = "") -> List[str]:
        return [t for spelling in spellings for t in _plural_variants(spelling + suffix)]

    scaffold_leaves: List[str] = []
    scaffold_groups: List[List[str]] = []
    for word in _TASK_SCAFFOLD_WORDS:
        for t in forms(f" {word}"):
            if t in cat_groups:
                scaffold_groups.append(cat_groups[t])
                break
            if t in cat_leaves:
                scaffold_leaves.append(cat_leaves[t])
                break

    def with_scaffold_leaves(named: List[str]) -> List[tuple[str, float]]:
        return [(code, 1.0) for code in dict.fromkeys(named + scaffold_leaves)]

    for t in forms():
        if t in cat_groups:
            return with_scaffold_leaves(cat_groups[t])
        if t in cat_leaves and t in cat_display:
            return with_scaffold_leaves([cat_leaves[t]])

    codes: Dict[str, float] = {}
    for t in forms():
        if t in cat_leaves:
            codes[cat_leaves[t]] = 1.0
            break
    for code in scaffold_leaves:
        codes.setdefault(code, 1.0)
    for group in scaffold_groups:
        for code in group:
            codes.setdefault(code, 1.0)
    for spelling in spellings:
        hits = _resolve_via_constructs(spelling, fuzzy=False)
        if hits:
            for code, score in hits:
                codes.setdefault(code, score)
            break
    if not codes:
        codes.update(_resolve_via_constructs(core))
    return list(codes.items()) or None


def _normalize_term(term: str) -> str:
    """Lower-case, trim, and use the ASCII apostrophe the vocabulary is written with."""
    return term.lower().strip().replace("\u2019", "'")


def _plural_variants(term: str) -> List[str]:
    """Simple singular/plural/possessive variants to improve exact-match recall."""
    variants = [term]
    if term.endswith("'s"):
        variants.append(term[:-2])          # "parkinson's" → "parkinson"
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


def _resolve_via_constructs(term: str, fuzzy: bool = True) -> List[tuple[str, float]]:
    """Try term against cognitive_construct, then expand to task codes via measures_db.

    With fuzzy=False only an exact construct label, synonym or group name counts.

    Returns (task_standard_code, score) pairs — or [] when:
      - the term doesn't match any cognitive construct, or
      - none of the matched constructs have task links in measures_db yet
        (i.e. bids_to_ca_map.json hasn't been generated / reviewed yet).
    """
    if not measures_db:
        return []

    cat_leaves        = leaf_db.get("cognitive_construct", {})
    cat_groups        = group_db.get("cognitive_construct", {})
    cat_display       = display_db.get("cognitive_construct", {})
    cat_group_display = group_display_db.get("cognitive_construct", {})

    term_lower = _normalize_term(term)
    construct_codes: List[str] = []
    score = 0.0

    # Exact match — prefer group (all construct descendants) over single leaf
    for t in _plural_variants(term_lower):
        if t in cat_groups:
            construct_codes = cat_groups[t]
            score = 1.0
            break
        if t in cat_leaves:
            construct_codes = [cat_leaves[t]]
            score = 1.0
            break

    if not construct_codes and fuzzy:
        all_targets: Dict[str, List[str]] = {
            k: [v] for k, v in cat_display.items() if len(k) >= 3
        }
        all_targets.update(
            {k: v for k, v in cat_group_display.items() if len(k) >= 3}
        )
        if all_targets:
            # Same rules as the fuzzy step of _match_yaml_term: no match on a
            # single shared word ("stroop task" / "task set reconfiguration"
            # scored 85.5 and returned 51 unrelated tasks), and containment
            # must line up with words.
            processed = _fuzzy_form(term_lower)
            results = [
                hit for hit in _process.extract(
                    term_lower, all_targets.keys(), scorer=_fuzz.WRatio,
                    processor=_fuzzy_form, score_cutoff=_MIN_FUZZY_SCORE, limit=None,
                )
                if _aligned_to_words(processed, _fuzzy_form(hit[0]))
            ]
            if results:
                best_label = results[0][0]
                score = results[0][1] / 100.0
                construct_codes = all_targets[best_label]

    if not construct_codes:
        return []

    # Expand construct codes → BIDS task codes via measures_db
    seen: set[str] = set()
    task_codes: List[str] = []
    for cc in construct_codes:
        for tc in measures_db.get(cc, []):
            if tc not in seen:
                seen.add(tc)
                task_codes.append(tc)

    if not task_codes:
        return []

    log.debug(
        "Construct expansion: '%s' → %d construct(s) → %d task code(s)",
        term, len(construct_codes), len(task_codes),
    )
    return [(tc, score) for tc in task_codes]


def _resolve_yaml_term_scored(field: str, term: str) -> List[tuple[str, float]]:
    """Resolve one term against the YAML KB, returning (code, score) pairs.

    Codes in _NON_FILTERING_CODES are dropped from every resolution path.
    """
    return [
        (code, score)
        for code, score in _match_yaml_term(field, term)
        if code not in _NON_FILTERING_CODES
    ]


def _match_yaml_term(field: str, term: str) -> List[tuple[str, float]]:
    """All (code, score) pairs for *term* in *field*; see _resolve_yaml_term_scored.

    Score is normalised to 0.0–1.0 so it can be used directly for relevance
    ranking: exact match → 1.0, fuzzy WRatio → score/100, embedding fallback →
    cosine similarity. Codes that share one matched label inherit that label's
    score. This is the single source of truth; ``_resolve_yaml_term`` is a thin
    codes-only wrapper over it.
    """
    cat_leaves        = leaf_db.get(field, {})
    cat_display       = display_db.get(field, {})
    cat_groups        = group_db.get(field, {})
    cat_group_display = group_display_db.get(field, {})

    # Task field uses a higher threshold — construct path handles the heavy
    # lifting for construct-level terms, so fuzzy only fires on strong hits.
    _sem = _TASK_SEMANTIC_THRESHOLD if field == "task" else _SEMANTIC_THRESHOLD

    term_lower = _normalize_term(term)

    # 0. Task field: scaffold words are dropped for good, and exact and
    # construct matches are combined (see _match_task_term). What follows only
    # runs for a task term that matched neither.
    if field == "task":
        term_lower = _task_core(term_lower)
        if not term_lower:
            return []
        task_hits = _match_task_term(term_lower)
        if task_hits:
            return task_hits

    candidates = _plural_variants(term_lower)

    # 1. Exact match — prefer group (all descendants) over single leaf code
    # so "working memory" returns every working-memory task code, not just
    # working_memory_general alone.
    for t in candidates:
        if t in cat_groups:
            return [(c, 1.0) for c in cat_groups[t]]
        if t in cat_leaves:
            return [(cat_leaves[t], 1.0)]

    # 1a. Exact match once filler words are dropped ("iEEG recording" → "ieeg").
    core = _without_filler(term_lower)
    if core and core != term_lower:
        for t in _plural_variants(core):
            if t in cat_groups:
                return [(c, 1.0) for c in cat_groups[t]]
            if t in cat_leaves:
                return [(cat_leaves[t], 1.0)]

    # (Task field: cognitive construct expansion already ran in step 0, before
    # fuzzy matching. Construct terms such as "pain processing" or "social
    # cognition" are not task names, and fuzzy matching on task synonyms gives
    # them word-overlap hits.)

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

    # Drop codes this term must not reach before ranking, so they cannot win a
    # tie against a usable label.
    excluded = set(_NON_FILTERING_CODES)
    if field == "diagnosis" and not _ANIMAL_WORDS.search(term_lower):
        excluded.update(cat_groups.get(_ANIMAL_MODEL_GROUP, []))
    for k in list(all_targets):
        codes = [c for c in all_targets[k] if c not in excluded]
        if codes:
            all_targets[k] = codes
        else:
            del all_targets[k]

    if not all_targets:
        return []

    def fuzzy_hits(query: str) -> list:
        """Top 20 usable labels for *query*, best first.

        Strings are compared in _fuzzy_form.
        """
        processed = _fuzzy_form(query)
        hits = _process.extract(
            query, all_targets.keys(), scorer=_fuzz.WRatio,
            processor=_fuzzy_form,
            score_cutoff=max(_sem, _MIN_FUZZY_SCORE), limit=None,
        )
        return [
            hit for hit in hits
            if _aligned_to_words(processed, _fuzzy_form(hit[0]))
        ][:20]

    # Increase limit so low-weight best matches can be supplemented by siblings
    results = fuzzy_hits(term_lower)
    if not results:
        return _embedding_fallback_scored(field, term)

    # 100 means the strings are identical once separators are normalised
    # ("resting-state eyes open" / "resting state eyes-open"). Treat that as an
    # exact match: weights and the low-weight union below do not apply.
    if results[0][1] >= 100:
        codes = dict.fromkeys(
            code for label, score, _ in results if score >= 100 for code in all_targets[label]
        )
        return [(code, 1.0) for code in codes]

    # Several labels often tie on WRatio: every label containing the term, and
    # every label the term contains, scores the same. Prefer a label made only
    # of words the user typed ("resting state eeg" → "resting-state", not
    # "resting-state tms-eeg"), then the label closest to the term overall.
    best_score = results[0][1]
    processed_term = _fuzzy_form(term_lower)
    term_words = set(processed_term.split())

    def tie_rank(label: str) -> tuple:
        processed_label = _fuzzy_form(label)
        return (
            set(processed_label.split()) <= term_words,
            _fuzz.ratio(processed_term, processed_label),
        )

    best_label = max(
        (label for label, score, _ in results if score >= best_score), key=tie_rank
    )

    # Weight-adjusted threshold: low-confidence synonyms need a higher fuzzy
    # score to be accepted. Labels and plain-string synonyms default to 1.0.
    weight = weight_db.get(field, {}).get(best_label, 1.0)
    effective_threshold = _sem + (1.0 - weight) * _WEIGHT_BOOST
    if best_score < effective_threshold:
        return _embedding_fallback_scored(field, term)

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
            core_results = fuzzy_hits(core_term)
        else:
            core_results = results  # nothing stripped; use original results

        collected: list[tuple[str, float]] = []
        seen: set[str] = set()
        _MIN_COLLECT_LEN = max(len(core_term) // 2, 5)
        for label, score, _ in core_results:
            if len(label) < _MIN_COLLECT_LEN:
                continue
            w = weight_db.get(field, {}).get(label, 1.0)
            eff = _sem + (1.0 - w) * _WEIGHT_BOOST
            if score >= eff:
                for code in all_targets[label]:
                    if code not in seen:
                        seen.add(code)
                        collected.append((code, score / 100.0))
        if collected:
            return collected

    return [(c, best_score / 100.0) for c in all_targets[best_label]]


def _resolve_yaml_term(field: str, term: str) -> List[str]:
    """Resolve one term against the YAML knowledge base for *field* (codes only)."""
    return [code for code, _ in _resolve_yaml_term_scored(field, term)]


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
    def _resolve_institution(self, term: str) -> List[tuple[str, float]]:
        """Exact bids_datasets.institutions strings for *term*, with scores.

        Uses the names loaded from the database at startup; without them,
        the 'institutions' list in name_index.json.
        """
        index = _institution_index.get_index(self._names.get("institutions", []))
        return index.resolve(term)

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

        if field == _INSTITUTION_FIELD:
            return [name for name, _ in self._resolve_institution(term)][:n_results]

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

    # ------------------------------------------------------------------
    # Scored variants — surface the RAG match score for relevance ranking.
    # Mirror the methods above but return (code, score) pairs (score 0.0–1.0).
    # ------------------------------------------------------------------
    def retrieve_scan_terms_scored(
        self,
        terms: List[str],
    ) -> Dict[str, Dict[str, List[tuple[str, float]]]]:
        """Scored variant of retrieve_scan_terms.

        Returns {field: {term: [(code, score), ...]}}.
        """
        result: Dict[str, Dict[str, List[tuple[str, float]]]] = {}
        for field in _SCAN_FIELDS:
            term_map: Dict[str, List[tuple[str, float]]] = {}
            for term in terms:
                term_lower = term.lower().strip()
                if field == "task" and term_lower in (
                    _PURE_MODALITY_TERMS | _GENERIC_IMAGING_TERMS
                ):
                    continue
                if field == "suffix" and term_lower in _GENERIC_IMAGING_TERMS:
                    continue
                scored = _resolve_yaml_term_scored(field, term)
                if scored:
                    term_map[term] = scored
            if term_map:
                result[field] = term_map
        return result

    def retrieve_for_field_multi_scored(
        self,
        field: str,
        terms: List[str],
        n_results_per_term: int = 20,
    ) -> Dict[str, List[tuple[str, float]]]:
        """Scored variant of retrieve_for_field_multi.

        Returns {term: [(code, score), ...]} — terms with no match are omitted.
        """
        result: Dict[str, List[tuple[str, float]]] = {}
        for term in terms:
            if field in _YAML_FIELDS:
                scored = _resolve_yaml_term_scored(field, term)[:n_results_per_term]
            elif field == _INSTITUTION_FIELD:
                scored = self._resolve_institution(term)
            elif field in _NAME_FIELDS:
                candidates = self._names.get(_NAME_FIELD_KEY[field], [])
                if candidates:
                    matches = _process.extract(
                        term,
                        candidates,
                        scorer=_fuzz.token_set_ratio,
                        limit=n_results_per_term,
                        score_cutoff=_NAME_THRESHOLD,
                    )
                    scored = [(r[0], r[1] / 100.0) for r in matches]
                else:
                    scored = []
            else:
                scored = []
            if scored:
                result[term] = scored
        return result
