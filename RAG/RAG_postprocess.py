from __future__ import annotations

import sys
import time
from typing import Dict, List

from rapidfuzz import fuzz as _fuzz
from rapidfuzz import process as _process

from yaml_to_llamaindex import leaf_db, display_db, group_db, group_display_db, hierarchy_db
from llamaindex_extraction import UserQueryEntities, prompt_template_str


# ---------------------------------------------------------------------------
# Gemini model cascade  (mirrors training_data_generation/migrate_tasks_to_hierarchy.py)
# Each entry: (model_id, role_label, max_attempts, [wait_seconds per retry])
# ---------------------------------------------------------------------------
MODEL_CASCADE = [
    # Das aktuelle Flaggschiff für Geschwindigkeit & Intelligenz (GA-Status)
    ("gemini-3.1-pro",         "Primary",    2, [30, 60]),
    
    # Der hocheffiziente Allrounder (stabil und kostengünstig)
    ("gemini-3.1-flash",       "Fallback 1", 2, [30, 60]),
    
    # Die Ultra-Leichtgewicht-Variante für einfache Aufgaben/Massendaten
    ("gemini-3.1-flash-lite",  "Fallback 2", 3, [60, 120, 240]),
    
    # Optional: Spezialmodell für komplexe Logik (Reasoning-Fokus)
    ("gemini-3-deep-think",    "Fallback 3", 5, [120, 240, 480]),
]


def _make_extractor(model_id: str):
    """Instantiate a fresh LlamaIndex extractor for the given Gemini model."""
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.core.program import LLMTextCompletionProgram

    llm = GoogleGenAI(model=model_id, temperature=0.0)
    return LLMTextCompletionProgram.from_defaults(
        output_cls=UserQueryEntities,
        llm=llm,
        prompt_template_str=prompt_template_str,
    )


def extract_entities(query: str) -> UserQueryEntities:
    """Extract entities from *query* using the Gemini model cascade.

    Tries each model in MODEL_CASCADE in order.  Within each model the call
    is retried up to *max_attempts* times with exponential back-off before
    falling through to the next model.  Raises RuntimeError only if every
    model in the cascade is exhausted.
    """
    last_exc: Exception | None = None

    for model_id, role, max_attempts, waits in MODEL_CASCADE:
        print(f"  [{role}] {model_id}", file=sys.stderr)
        extractor = _make_extractor(model_id)

        for attempt in range(1, max_attempts + 1):
            try:
                return extractor(query_str=query)
            except Exception as exc:
                err = str(exc)
                # Hard failures — wrong model name, bad API key, etc.
                is_fatal = any(tag in err for tag in ("404", "NOT_FOUND", "INVALID_ARGUMENT", "401"))
                if is_fatal:
                    print(f"    Fatal error on {model_id}: {err[:120]}", file=sys.stderr)
                    last_exc = exc
                    break  # move to next model immediately

                print(f"    attempt {attempt}/{max_attempts} failed: {err[:120]}", file=sys.stderr)
                last_exc = exc
                if attempt < max_attempts:
                    wait = waits[attempt - 1]
                    print(f"    Retrying in {wait}s…", file=sys.stderr)
                    time.sleep(wait)

    raise RuntimeError(
        f"All models in cascade exhausted. Last error: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Resolution thresholds
# ---------------------------------------------------------------------------
_LEAF_FUZZY_THRESHOLD = 85   # strict – leaf keys can be short, avoid false hits
_GROUP_FUZZY_THRESHOLD = 80  # slightly looser – group synonyms tend to be longer phrases


def _plural_variants(term: str) -> List[str]:
    """Return the term plus simple singular/plural variants to try as extra exact-match candidates.

    Handles the common case where the LLM or user writes "psychiatric disorder"
    but the YAML synonym is "psychiatric disorders" (or vice-versa).
    Only covers the most common English suffixes to stay lightweight.
    """
    variants = [term]
    if term.endswith("s") and not term.endswith("ss"):
        variants.append(term[:-1])        # "disorders" → "disorder"
    else:
        variants.append(term + "s")       # "disorder"  → "disorders"
    if term.endswith("ies"):
        variants.append(term[:-3] + "y")  # "disabilities" → "disability"
    elif term.endswith("y"):
        variants.append(term[:-1] + "ies")
    return variants


# ---------------------------------------------------------------------------
# Core resolution logic
# ---------------------------------------------------------------------------
def _resolve_term(
    term: str,
    cat_leaves: Dict[str, str],
    cat_display: Dict[str, str],
    cat_groups: Dict[str, List[str]],
    cat_group_display: Dict[str, List[str]],
    category: str,
    debug: bool = False
) -> List[str]:
    """
    Finds the best match by pooling leaves and groups together.
    Prioritizes exact matches, then high-confidence fuzzy matches.
    """
    term_lower = term.lower().strip()
    candidates = _plural_variants(term_lower)

    # 1. UNIVERSAL EXACT MATCH
    # Check every possible exact synonym/label across both leaves and groups
    for t in candidates:
        if t in cat_leaves:
            return [cat_leaves[t]]
        if t in cat_groups:
            return cat_groups[t]

    # 2. POOLED FUZZY MATCH
    # Instead of checking leaves then groups, we check them both and compare
    all_targets = {}
    if cat_display:
        for k, v in cat_display.items():
            all_targets[k] = [v] # Wrap in list to unify format
    if cat_group_display:
        for k, v in cat_group_display.items():
            # If a label exists in both, the specific leaf usually wins
            if k not in all_targets:
                all_targets[k] = v

    if not all_targets:
        return []

    if debug and category == "diagnosis":
        print(f"      [DEBUG-POOL] Total targets: {len(all_targets)}", file=sys.stderr)
        print(f"      [DEBUG-POOL] Is 'focal epilepsy' in pool?: {'focal epilepsy' in all_targets}", file=sys.stderr)

    # Get the top matches
    results = _process.extract(
        term_lower, 
        all_targets.keys(), 
        scorer=_fuzz.WRatio, 
        limit=3
    )

    if debug:
        print(f"Top 3 results: '{results}'", file=sys.stderr)

    if not results:
        return []

    best_label, best_score, _ = results[0]

    # Threshold check
    if best_score < _GROUP_FUZZY_THRESHOLD:
        return []

    # SPECIFICITY TIE-BREAKER:
    # If the top match is "epilepsy" (90) and the second is "focal epilepsy" (90),
    # the longer one is more specific and should win.
    final_match = best_label
    for next_label, next_score, _ in results[1:]:
        if next_score >= best_score and len(next_label) > len(final_match):
            final_match = next_label
            best_score = next_score

    if debug:
        print(f"      [DEBUG-SCORE] Winner: '{final_match}' ({best_score})", file=sys.stderr)

    return all_targets[final_match]

def _prune_redundant_parents(codes: List[str]) -> List[str]:
    """
    Removes broad parent codes if a more specific child code is present.
    Example: if ['mri', 'functional_mri'] are present, 'mri' is removed.
    """
    if not codes:
        return []

    # Get the hierarchy paths for all resolved codes
    code_paths = {}
    for code in codes:
        info = hierarchy_db.get(code)
        if info and "path" in info:
            code_paths[code] = set(info["path"])
        else:
            code_paths[code] = {code}

    to_remove = set()
    for code, path in code_paths.items():
        # Check if this 'code' is a parent of any OTHER code in our list
        for other_code, other_path in code_paths.items():
            if code == other_code:
                continue
            # If our 'code' exists in the 'other_code's' lineage, our code is a parent
            if code in other_path:
                to_remove.add(code)
    
    return [c for c in codes if c not in to_remove]


def resolve_to_standard_codes(
    extracted_entities,
    annotate_source: bool = False,
    debug: bool = False,
) -> Dict[str, List]:
    """Convert LLM-extracted terms to standard_codes using the knowledge base.

    Parameters
    ----------
    extracted_entities : UserQueryEntities (Pydantic model)
        Output of the LlamaIndex extraction step.
    annotate_source : bool
        When True, each entry in the result lists is a dict
        ``{"code": <standard_code>, "source": "leaf"|"group", "matched": <term>}``
        instead of a plain string.  Useful for debugging.

    Returns
    -------
    Dict[str, List[str]]  (or List[dict] when annotate_source=True)
        Mapping of category → deduplicated list of standard_codes.
    """
    if debug:
        print("\n[DEBUG] Raw LLM extraction:", file=sys.stderr)
        for cat, terms in extracted_entities.model_dump().items():
            if terms:
                print(f"  {cat}: {terms}", file=sys.stderr)

    resolved: Dict[str, List] = {}

    for category, extracted_terms in extracted_entities.model_dump().items():
        cat_leaves = leaf_db.get(category, {})
        cat_display = display_db.get(category, {})
        cat_groups = group_db.get(category, {})
        cat_group_display = group_display_db.get(category, {})
        
        all_resolved_codes: List = []

        for term in extracted_terms:
            term_lower = term.lower().strip()
            matched_codes = _resolve_term(
                term_lower, cat_leaves, cat_display, cat_groups, cat_group_display, category, debug=debug
            )
            
            if debug and matched_codes:
                print(f"  [DEBUG] {category} '{term}' → {matched_codes}", file=sys.stderr)
            
            all_resolved_codes.extend(matched_codes)

        # Deduplicate
        unique_codes = list(dict.fromkeys(all_resolved_codes))
        
        # Apply Pruning: Remove parents if children exist
        # This fixes the "fMRI" + "MRI" explosion problem
        final_codes = _prune_redundant_parents(unique_codes)

        if annotate_source:
            # Re-map to the annotation format if requested
            resolved[category] = [{"code": c, "matched": "resolved"} for c in final_codes]
        else:
            resolved[category] = final_codes

    return resolved


def get_hierarchy_context(standard_codes: List[str]) -> List[Dict]:
    """Return human-readable hierarchy info for a list of standard_codes.

    Useful for injecting context back into an LLM prompt or for logging.
    Example output item::

        {
            "code":        "major_depressive_disorder",
            "label":       "major depressive disorder",
            "path":        ["psychiatric", "mood_disorders", "depression_spectrum"],
            "description": "A mental health disorder..."
        }
    """
    results = []
    for code in standard_codes:
        info = hierarchy_db.get(code)
        if info:
            results.append({"code": code, **info})
    return results


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    queries = [
        "Give me all datasets that contain at least 15 male participants with focal epilepsy.",
        "Show datasets with psychiatric disorder patients doing a working memory task.",
        "Find datasets with schizophrenia patients and healthy controls.",
        "I need resting-state fMRI data from participants with mood disorders.",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        extracted = extract_entities(query)
        codes = resolve_to_standard_codes(extracted, debug=True)
        print(f"Resolved codes: {codes}")
        # Show hierarchy context for diagnosis codes
        diag_codes = codes.get("diagnosis", [])
        if diag_codes:
            context = get_hierarchy_context(diag_codes)
            for c in context:
                print(f"  [{' > '.join(c['path'])}] {c['label']}")
