"""
The few-shot examples of the SQL prompt: the pool, and which of them a
question is shown.

The pool is few_shot_examples.json in this folder. pick_examples() chooses
from it without a model call; format_examples() writes the chosen ones as the
prompt shows them.
"""

from __future__ import annotations

from functools import lru_cache
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

from services.database_stats.database_stats import database_stats as _database_stats
from services.schema.coded_fields import CONCEPT_FIELDS
from services.schema.tables import objects as _objects, participants as _participants
from services.translate import optional_columns as _optional_columns

log = logging.getLogger(__name__)

# ── Few-shot pool (loaded once at import time) ────────────────────────────────
FEW_SHOT_POOL: Dict[str, List[dict]] = {}
_FEW_SHOT_PATH = Path(__file__).resolve().parent / "few_shot_examples.json"
if _FEW_SHOT_PATH.exists():
    # The file is part of the repository: a broken one stops the backend at
    # startup instead of letting every query be written without examples.
    with open(_FEW_SHOT_PATH) as _f:
        FEW_SHOT_POOL = json.load(_f)
    log.info("Loaded %d few-shot families from %s",
             len(FEW_SHOT_POOL), _FEW_SHOT_PATH)

_FAMILY_ALIASES: Dict[str, str] = {
    "subject_multimodal": "subject_multimodal_query",
    "metadata_query":     "json_query",
    "json_numeric_query": "json_numeric_query",
}
_FALLBACK_FAMILIES = ["combined_filter", "concept_query", "scan_filter"]


_STOPWORDS = {
    "a","an","the","and","or","in","of","to","for","with","that","this",
    "is","are","do","does","me","my","i","all","any","no","not","from",
    "by","be","at","on","as","it","its","have","has","had","give","show",
    "find","list","which","what","how","get","tell","want","please",
    "can","could","would","datasets","dataset","data",
}

_EXAMPLE_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TOKEN_ALIASES = {
    "aged": "age", "ages": "age", "younger": "young", "youngest": "young",
    "older": "old", "oldest": "old", "children": "child", "participants": "participant",
    "subjects": "subject", "datasets": "dataset", "recently": "recent",
    "highest": "high", "lowest": "low", "studies": "study",
    # Common irregular plurals that the conservative suffix rules below cannot
    # reduce without conflating unrelated words (e.g. disease/diseases).
    "analyses": "analysis", "axes": "axis", "bases": "basis",
    "crises": "crisis", "diagnoses": "diagnosis", "hypotheses": "hypothesis",
    "metastases": "metastasis", "psychoses": "psychosis", "syntheses": "synthesis",
    "theses": "thesis",
}


def _example_tokens(text: str) -> List[str]:
    """Normalise words for deterministic few-shot retrieval (no model call)."""
    tokens = []
    for token in _EXAMPLE_TOKEN_RE.findall(text.lower().replace("_", " ")):
        if token in _STOPWORDS or len(token) <= 2:
            continue
        token = _TOKEN_ALIASES.get(token, token)
        # Small, conservative normalisation catches plural/tense variants
        # without adding a stemmer dependency or collapsing technical terms.
        if token.endswith("ies") and len(token) > 5:
            token = token[:-3] + "y"
        elif token.endswith("ing") and len(token) > 6:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 4:
            token = token[:-1]
        tokens.append(token)
    return tokens


def _intent_features(text: str, family: str = "", meta: Optional[dict] = None) -> set:
    """Typed, rule-based intent cues shared by the question and example."""
    # Only populated metadata contributes evidence. Stringifying an all-empty
    # `requested` dict used to inject the words "diagnosis", "task", etc. and
    # falsely require those categories for otherwise unrelated questions.
    meta_text = " ".join(
        f"{key} {value}" for key, value in (meta or {}).items() if value
    ) if isinstance(meta, dict) else str(meta or "")
    value = f"{text} {family} {meta_text}".lower().replace("_", " ")
    words = set(_EXAMPLE_TOKEN_RE.findall(value))
    intents = set()
    if words & {"age", "aged", "young", "younger", "old", "older", "child", "children",
                "adolescent", "adult", "infant", "pediatric", "paediatric"}:
        intents.add("age")
    if words & {"participant", "participants", "subject", "subjects", "sample", "cohort"}:
        intents.add("participants")
    if words & {"diagnosis", "diagnosed", "patient", "patients", "schizophrenia", "autism",
                "epilepsy", "parkinson", "depression", "bipolar", "healthy", "control", "controls"}:
        intents.add("clinical")
    if words & {"both", "versus", "vs", "compare", "comparison", "control", "controls"}:
        intents.add("case_control")
    if words & {"without", "exclude", "excluding", "absent", "lack", "lacking", "neither", "no", "not"}:
        intents.add("absence")
    if words & {"functional", "fmri", "eeg", "meg", "dwi", "diffusion", "anatomical", "mri",
                "pet", "fnirs", "asl", "bold", "modality", "datatype"}:
        intents.add("modality")
    if words & {"task", "paradigm", "rest", "resting", "nback", "stimulus", "stimuli"}:
        intents.add("task")
    if words & {"tr", "repetition", "echo", "tesla", "strength", "voxel", "scanner", "sequence",
                "sampling", "bandwidth", "manufacturer", "acquisition"}:
        intents.add("acquisition")
    if words & {"recent", "recently", "newest", "date", "year", "last", "past", "months", "days"}:
        intents.add("recency")
    if (words & {"recent", "recently", "newest", "date", "year", "last", "past", "months", "days"}
            and words & {"dataset", "datasets", "openneuro"}
            and not words & {"paper", "papers", "article", "journal", "pubmed"}):
        intents.add("dataset_publication_date")
    if (words & {"date", "year", "last", "past", "months", "days", "recent", "recently"}
            and words & {"paper", "papers", "article", "journal", "pubmed", "publication"}):
        intents.add("paper_publication_date")
    if words & {"top", "popular", "highest", "lowest", "largest", "smallest", "most", "least", "downloads"}:
        intents.add("ranking")
    if words & {"tr", "repetition"}:
        intents.add("repetition_time")
    if words & {"te", "echo"}:
        intents.add("echo_time")
    if words & {"tesla"} or {"field", "strength"}.issubset(words):
        intents.add("field_strength")
    if words & {"manufacturer", "vendor", "siemens", "philips", "ge"}:
        intents.add("scanner_vendor")
    if words & {"paper", "papers", "publication", "doi", "pubmed", "reference", "references", "article"}:
        intents.add("paper_reference")
    if words & {"author", "authors", "lab", "laboratory", "institution", "institutions", "site", "sites"}:
        intents.add("provenance")
    if words & {"license", "licence", "redistributable", "permission", "share"}:
        intents.add("license")
    if words & {"session", "sessions", "longitudinal", "visit", "visits"}:
        intents.add("sessions")
    if meta:
        if meta.get("datatype"):
            intents.add("modality")
        if meta.get("task"):
            intents.add("task")
        if meta.get("diagnosis"):
            intents.add("clinical")
    return intents


def _query_structure(question: str) -> set:
    """Infer SQL-shape cues from wording; this only helps choose examples."""
    raw = question.lower()
    words = set(_EXAMPLE_TOKEN_RE.findall(raw))
    structures = set()
    if words & {"without", "no", "not", "neither", "exclude", "excluding", "absent", "lack", "lacking"}:
        structures.add("absence")
    elif re.search(r"\b(any|with|contain|include|have|featur)\w*\b", raw):
        structures.add("presence")
    if words & {"both", "versus", "vs", "compare", "comparison"}:
        structures.add("multi_condition")
    if (words & {"count", "average", "mean", "minimum", "maximum", "fewer", "least", "more", "greater"}
            or re.search(r"\b(at least|at most|more than|fewer than|less than|greater than)\b", raw)):
        structures.add("aggregate")
        structures.add("threshold")
    if re.search(r"\b(top|popular|recent|recently|highest|lowest|largest|smallest|newest|most|least)\b", raw):
        structures.add("ranking")
    if (words & {"age", "aged", "young", "younger", "old", "older", "child", "children", "adolescent"}
            or re.search(r"\b\d+\s*(years?\s*old|or younger|or older)\b", raw)):
        structures.add("age_filter")
    if words & {"echo", "repetition", "spacing", "field", "strength", "scanner", "sequence", "sampling", "tesla"}:
        structures.add("metadata_filter")
    if words & {"title", "named", "name", "called"}:
        structures.add("name_filter")
    if words & {"task", "paradigm", "rest", "resting"}:
        structures.add("task_filter")
    return structures


@lru_cache(maxsize=512)
def _sql_shape(sql: str) -> frozenset:
    """Extract logical SQL features from an example, without executing it."""
    text = sql.lower().replace("{{cols}}", "d.id")
    features = set()
    if re.search(r"\bnot\s+exists\s*\(", text):
        features.add("absence")
    if re.search(r"(?<!not )\bexists\s*\(", text):
        features.add("presence")
    if re.search(r"\bhaving\b|\b(count|avg|average|min|max)\s*\(", text):
        features.add("aggregate")
    if re.search(r"\border\s+by\b", text):
        features.add("ranking")
    if re.search(r"\bbetween\b|>=|<=|(?<!\w)>(?!\w)|(?<!\w)<(?!\w)", text):
        features.add("threshold")
    if re.search(r"\b(age|age_min|age_max)\b", text):
        features.add("age_filter")
    if re.search(r"\bdiagnosis\b", text):
        features.add("clinical_filter")
    if re.search(r"\bsex\b", text):
        features.add("sex_filter")
    if re.search(r"other_entities|->>|jsonb", text):
        features.add("metadata_filter")
    if "text_match(" in text or "readme_text" in text or "description_text" in text:
        features.add("text_search")
    if "paper_references" in text or "pubmed" in text or "doi" in text:
        features.add("paper_reference")
    if _participants.name in text:
        features.add("participants")
    if _objects.name in text:
        features.add("scan_data")
    if re.search(r"\bjoin\b", text):
        features.add("join")
    if len(re.findall(r"\bexists\s*\(", text)) > 1:
        features.add("multi_condition")
    return frozenset(features)


def _bm25_scores(query_tokens: List[str], documents: List[List[str]]) -> List[float]:
    """BM25 over the small, fixed example pool; stable and dependency-free."""
    if not query_tokens or not documents:
        return [0.0] * len(documents)
    n_docs = len(documents)
    lengths = [len(doc) for doc in documents]
    avg_len = sum(lengths) / max(n_docs, 1)
    document_frequency: Dict[str, int] = {}
    term_frequencies: List[Dict[str, int]] = []
    for document in documents:
        counts: Dict[str, int] = {}
        for token in document:
            counts[token] = counts.get(token, 0) + 1
        term_frequencies.append(counts)
        for token in counts:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    query_frequency: Dict[str, int] = {}
    for token in query_tokens:
        query_frequency[token] = query_frequency.get(token, 0) + 1
    scores = []
    for length, counts in zip(lengths, term_frequencies):
        score = 0.0
        for token, qf in query_frequency.items():
            tf = counts.get(token, 0)
            if not tf:
                continue
            df = document_frequency[token]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + 1.2 * (1.0 - 0.75 + 0.75 * length / max(avg_len, 1.0))
            score += idf * (tf * 2.2 / denom) * min(qf, 2)
        scores.append(score)
    return scores


@lru_cache(maxsize=512)
def _static_guard_reason(sql: str) -> Optional[str]:
    """Cached static guard check for an example's SQL template."""
    from services.query_guard import check_static
    return check_static(sql.replace("{{COLS}}", "d.id"))


def _family_key(query_family: str, pool: Dict[str, List[dict]]) -> str:
    """Resolve a legacy family alias only when the exact pool key is absent."""
    if query_family in pool:
        return query_family
    alias = _FAMILY_ALIASES.get(query_family)
    return alias if alias in pool else query_family


def extract_resolved_codes(augmented_question: str) -> Dict[str, List[str]]:
    """Parse resolved ontology codes from the augmented question string.

    Extracts canonical DB codes from lines like:
      o2.datatype = 'functional_mri'
      o2.task IN ('resting_state', 'n_back')
      p.diagnosis IN ('autism_spectrum_disorder')
    Returns a dict with keys 'datatype', 'task', 'diagnosis', 'suffix'.
    """
    codes: Dict[str, List[str]] = {"datatype": [], "task": [], "diagnosis": [], "suffix": []}
    for field in codes:
        for m in re.finditer(rf"\.{field}\s*=\s*'([^']+)'", augmented_question):
            v = m.group(1)
            if v not in codes[field]:
                codes[field].append(v)
        for m in re.finditer(rf"\.{field}\s+IN\s*\(([^)]+)\)", augmented_question, re.IGNORECASE):
            for v in re.findall(r"'([^']+)'", m.group(1)):
                if v not in codes[field]:
                    codes[field].append(v)
    return codes


def pick_examples(pool: Dict[str, List[dict]], query_family: str,
                   n: int = 5, seed: str = "",
                   resolved_codes: Optional[Dict[str, List[str]]] = None) -> List[dict]:
    """Select safe, relevant and varied examples with deterministic retrieval.

    Relevance combines typed ontology-code coverage, BM25 over question and
    example metadata, query-family match, and SQL-shape cues. Greedy MMR then
    reduces near-duplicates so a prompt teaches several useful patterns. It
    never calls an LLM. The weights and diversity penalties below are an
    **ASSISTANT PROPOSAL — REVIEW THESE VALUES**; keep the selection method
    deterministic so they can be tuned against held-out search results.
    """
    if not pool or not seed:
        return []

    family_key = _family_key(query_family, pool)

    all_examples: List[dict] = []
    for fam, examples in pool.items():
        for ex in examples:
            # Don't teach the generator an unavailable-column query or SQL the
            # static safety guard would reject. Runtime EXPLAIN is not repeated
            # here: it depends on the live database and would add latency.
            if _optional_columns.uses_unusable_column(_database_stats.coverage, ex["sql"]):
                continue
            guard_reason = _static_guard_reason(ex["sql"])
            if guard_reason:
                log.debug("Skipping unsafe few-shot example %r: %s", ex["question"], guard_reason)
                continue
            all_examples.append({"family": fam, "example": ex, "order": len(all_examples)})
    if not all_examples:
        return []

    fields = CONCEPT_FIELDS
    requested = {
        field: set((resolved_codes or {}).get(field, []))
        for field in fields
    }
    has_codes = any(requested.values())
    query_tokens = _example_tokens(seed)
    query_shape = _query_structure(seed)
    query_categories = _intent_features(seed, family_key, requested)
    specific_categories = query_categories & {
        "age", "clinical", "acquisition", "recency", "paper_reference",
        "provenance", "license", "sessions", "dataset_publication_date",
        "paper_publication_date",
    }
    technical_topics = specific_categories & {
        "repetition_time", "echo_time", "field_strength", "scanner_vendor",
    }

    documents: List[List[str]] = []
    for item in all_examples:
        ex = item["example"]
        meta = ex.get("meta", {})
        codes = {field: set(meta.get(field, [])) for field in fields}
        item["codes"] = codes
        item["question_tokens"] = set(_example_tokens(ex.get("question", "")))
        item["structure"] = (set(ex.get("structure", [])) |
                              set(_sql_shape(ex.get("sql", ""))) |
                              _query_structure(ex.get("question", "")))
        item["categories"] = _intent_features(ex.get("question", ""), item["family"], meta)
        item["pattern"] = ex.get("pattern", "")
        # BM25 is deliberately based on the natural-language example only;
        # exact values and SQL categories are scored in separate dimensions.
        item["tokens"] = _example_tokens(ex.get("question", ""))
        documents.append(item["tokens"])

    bm25 = _bm25_scores(query_tokens, documents)
    max_bm25 = max(bm25, default=0.0)
    for item, lexical_score in zip(all_examples, bm25):
        # A query may ask for several independent specific dimensions (e.g.
        # institution AND linked paper). A single example need only teach one
        # requested dimension; requiring every category in every example can
        # eliminate the entire pool when no combined example exists.
        if ((specific_categories and not specific_categories.intersection(item["categories"])) or
                (technical_topics and not technical_topics.intersection(item["categories"]))):
            item["relevance"] = 0.0
            continue
        matched = sum(len(requested[field] & item["codes"][field]) for field in fields)
        requested_n = sum(len(requested[field]) for field in fields)
        example_n = sum(len(item["codes"][field]) for field in fields)
        if requested_n:
            recall = matched / requested_n
            precision = matched / max(example_n, 1)
            code_score = 0.8 * recall + 0.2 * precision
        else:
            code_score = 0.0
        lexical = lexical_score / max_bm25 if max_bm25 else 0.0
        family = (1.0 if item["family"] == family_key else
                  0.4 if item["family"] in _FALLBACK_FAMILIES else 0.0)
        shape = (0.8 * len(query_shape & item["structure"]) / len(query_shape) +
                 0.2 * len(query_shape & item["structure"]) / max(len(item["structure"]), 1)
                 if query_shape else 0.0)
        category = (0.8 * len(query_categories & item["categories"]) / len(query_categories) +
                    0.2 * len(query_categories & item["categories"]) / max(len(item["categories"]), 1)
                    if query_categories else 0.0)
        # REVIEW (assistant-proposed): values get only one quarter of the
        # relevance weight. Topic, logic, category and family signals together
        # carry three quarters, to avoid choosing examples just because they
        # share a diagnosis/task code while teaching the wrong SQL structure.
        if has_codes:
            relevance = (0.25 * code_score + 0.27 * lexical + 0.26 * shape +
                         0.14 * category + 0.08 * family)
        else:
            relevance = 0.34 * lexical + 0.30 * shape + 0.22 * category + 0.14 * family
        item["relevance"] = relevance
        item["features"] = {
            "question": item["question_tokens"],
            "codes": {f"{field}:{code}" for field in fields for code in item["codes"][field]},
            "categories": item["categories"],
            "structure": item["structure"],
            "pattern": {item["pattern"]} if item["pattern"] else set(),
            "family": {item["family"]},
        }

    def overlap(left: dict, right: dict) -> float:
        """Weighted feature Jaccard used as the MMR redundancy penalty."""
        weights = (("question", 0.30), ("codes", 0.15), ("categories", 0.20),
                   ("structure", 0.25), ("pattern", 0.06), ("family", 0.04))
        total = 0.0
        for key, weight in weights:
            a, b = left["features"][key], right["features"][key]
            if a or b:
                total += weight * len(a & b) / len(a | b)
        return total

    # Keep diversity soft: topic/logical matches remain more important than
    # forcing an unrelated family into the prompt.
    family_cap = max(2, math.ceil(n * 0.8))
    pattern_cap = max(2, math.ceil(n * 0.6))
    min_relevance = max(0.28, 0.55 * max(item["relevance"] for item in all_examples))
    # REVIEW (assistant proposal): the absolute and relative cutoffs avoid
    # padding a prompt with weak matches; validate the values on a disjoint set.
    picked_items: List[dict] = []
    family_counts: Dict[str, int] = {}
    pattern_counts: Dict[str, int] = {}
    remaining = list(all_examples)
    while remaining and len(picked_items) < n:
        best = None
        best_rank = (-math.inf, -math.inf, -math.inf)
        for item in remaining:
            if item["relevance"] < min_relevance:
                continue
            redundancy = max((overlap(item, prior) for prior in picked_items), default=0.0)
            adjusted = item["relevance"] - 0.12 * redundancy
            if family_counts.get(item["family"], 0) >= family_cap:
                adjusted -= 0.12
            if item["pattern"] and pattern_counts.get(item["pattern"], 0) >= pattern_cap:
                adjusted -= 0.08
            # Deterministic ties: relevance first, then original pool order.
            rank = (adjusted, item["relevance"], -item["order"])
            if rank > best_rank:
                best, best_rank = item, rank
        if best is None:
            break
        remaining.remove(best)
        question_key = " ".join(_example_tokens(best["example"].get("question", "")))
        if any(question_key == " ".join(_example_tokens(p["example"].get("question", "")))
               for p in picked_items):
            continue
        picked_items.append(best)
        family_counts[best["family"]] = family_counts.get(best["family"], 0) + 1
        if best["pattern"]:
            pattern_counts[best["pattern"]] = pattern_counts.get(best["pattern"], 0) + 1
    return [item["example"] for item in picked_items]


def format_examples(examples: List[dict], max_chars: Optional[int] = None) -> str:
    if not examples:
        return ""
    lines = ["### Examples", "Similar query and correct SQL:\n"]
    if max_chars is not None and len("\n".join(lines)) > max_chars:
        return ""
    for ex in examples:
        item = (f"[QUESTION]{ex['question']}[/QUESTION]\n"
                f"[SQL]\n{ex['sql'].strip()}\n[/SQL]")
        candidate = "\n".join([*lines, item])
        if max_chars is not None and len(candidate) > max_chars:
            break  # preserve rank order and keep only complete examples
        lines.append(item)
    return "\n".join(lines)
