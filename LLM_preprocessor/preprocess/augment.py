"""The plan with RAG results: filter lines and the augmented question (augment_with_rag)."""
from __future__ import annotations

import logging
import re
from typing import Collection, Dict, List, Optional, Sequence, Tuple
from join_registry import (
    build_join_context_block as _build_join_context_block,
    check_contradictions as _check_contradictions,
)
from text_terms import (
    MODE_ALL as _MODE_ALL,
    MODE_ALSO as _MODE_ALSO,
    MODE_LOOSE as _MODE_LOOSE,
    MODE_NEAR as _MODE_NEAR,
    MODE_SHORT as _MODE_SHORT,
    diagnosis_alternatives as _diagnosis_alternatives,
    diagnosis_qualifier as _diagnosis_qualifier,
    format_text_match as _format_text_match,
    format_task_text_match as _format_task_text_match,
)
from institution_index import text_alternatives as _institution_text_alternatives
import place_index as _place_index
from cohort_count import placeholder as _patients_placeholder
import text_formula as _text_formula
from criteria import species_condition as _species_condition
from criteria import without_diagnosis as _without_diagnosis
from criteria import normalise_age_groups as _normalise_age_groups, age_condition as _age_condition, \
    age_group_label as _age_group_label

from . import catalogue as _catalogue
from .plan import AugmentedQueryPlan, QueryPlan, RAGField, ResolvedTerms, ResolvedTopic, ScoredFilter
from .rag import _HEALTHY_WORDS, _build_scored_filters, _is_healthy_term, _participant_conditions

log = logging.getLogger("preprocess")


def augment_with_rag(
    plan: QueryPlan,
    rag_results: Dict[str, Dict[str, List[str]]],
    unresolved_terms: Optional[Dict[str, List[str]]] = None,
    db_miss_terms: Optional[Dict[str, List[str]]] = None,
    scores: Optional[Dict[str, Dict[str, float]]] = None,
    available_columns: Collection[str] = (),
    topics: Optional[List[ResolvedTopic]] = None,
) -> AugmentedQueryPlan:
    """
    Combine a QueryPlan with RAG results to produce an AugmentedQueryPlan.

    Parameters
    ----------
    plan             : output of preprocess_query()
    rag_results      : {field: {term: [canonical_code, ...]}}
                       e.g. {"diagnosis": {"schizophrenia": ["schizophrenia_spectrum"]},
                              "datatype":  {"fMRI": ["func"]},
                              "suffix":    {"fMRI": ["bold"]}}
    unresolved_terms : {field: [term, ...]} for terms that RAG could not map.
                       These are surfaced as an anti-ILIKE warning in the augmented
                       question so the SQL model does not fall back to broken text
                       searches on canonical-code columns.
    db_miss_terms    : {field: [code, ...]} for codes that resolved in the YAML
                       vocabulary but have zero rows in the live database.  These
                       are emitted as a DB_MISS warning so the SQL model omits
                       those filters rather than generating always-false EXISTS.

    Returns
    -------
    AugmentedQueryPlan  ready to pass to the SQL generation LLM
    """
    resolved: List[ResolvedTerms] = []
    for field_str, term_map in rag_results.items():
        try:
            field = RAGField(field_str)
        except ValueError:
            continue
        if field == RAGField.SCAN:
            continue  # SCAN is internal; resolved results use specific field names
        resolved.append(ResolvedTerms(field=field, term_to_codes=term_map))

    topics = list(topics or [])
    scored_filters = _build_scored_filters(rag_results, scores or {})
    seen = {(f.field, f.code) for f in scored_filters}
    for topic in topics:
        for code in topic.codes:
            if ("task", code) not in seen:
                seen.add(("task", code))
                scored_filters.append(
                    ScoredFilter(field="task", code=code, score=topic.scores.get(code, 1.0)))

    unmatched_text: List[str] = []
    aug = AugmentedQueryPlan(
        plan=plan,
        resolved=resolved,
        scored_filters=scored_filters,
        topics=topics,
        augmented_question=_build_augmented_question(
            plan, resolved, unresolved_terms or {}, db_miss_terms or {},
            available_columns, topics, unmatched_text,
        ),
        unmatched_text=unmatched_text,
    )
    return aug


def _institution_lines(
    plan: QueryPlan,
    resolved: List[ResolvedTerms],
    unresolved: List[str],
    available_columns: Collection[str],
) -> List[str]:
    """One filter line per institution the user named.

    Resolved names (exact bids_datasets.institutions strings from
    RAG/institution_index.py) become an array overlap, which the GIN index on
    the column serves:
        (d.institutions && ARRAY['Stanford University', ...]::text[]
         OR text_match('Stanford', 'short'))
    The text part catches datasets that name the site in their title,
    description or study fields but whose scans carry no InstitutionName.
    README text only ranks here: README mentions of an institution are often
    citations or tools ('Harvard-Oxford atlas', 'Stanford Sleepiness Scale').

    An institution the stored names do not know (or a database without the
    column) falls back to a full-text search that includes the README, where
    labs usually describe where they scanned.

    An excluded institution removes datasets with any of its stored names:
        NOT COALESCE(d.institutions && ARRAY[...]::text[], false)
    (COALESCE keeps datasets without any recorded institution). Without
    stored names it removes datasets that name it in their short text fields;
    a README mention alone does not exclude.
    """
    resolved_names: Dict[str, List[str]] = {}
    for r in resolved:
        if r.field == RAGField.INSTITUTION:
            for term, names in r.term_to_codes.items():
                resolved_names.setdefault(term, []).extend(names)

    excluded = list(dict.fromkeys(t for t in plan.excluded_institution_terms if t and t.strip()))
    terms = list(dict.fromkeys(
        [t for t in plan.institution_terms if t and t.strip()] + list(resolved_names) + list(unresolved)
    ))
    terms = [t for t in terms if t not in excluded]
    use_column = "institutions" in available_columns

    def _array(names: List[str]) -> str:
        return "ARRAY[" + ", ".join("'" + n.replace("'", "''") + "'" for n in names) + "]::text[]"

    lines: List[str] = []
    for term in excluded:
        names = list(dict.fromkeys(resolved_names.get(term, []))) if use_column else []
        if names:
            condition = f"NOT COALESCE(d.institutions && {_array(names)}, false)"
        else:
            text_part = _format_text_match(_institution_text_alternatives(term) or [term], _MODE_SHORT)
            if not text_part:
                continue
            condition = f"NOT {text_part}"
        label = term.replace('"', "'")
        lines.append(f'  not institution "{label}": {condition}')

    for term in terms:
        names = list(dict.fromkeys(resolved_names.get(term, []))) if use_column else []
        # The same institution as ROR names it, from the sites' ethics boards
        # and scanner names (bids_site_places, RAG/place_index.py).
        ror_names = _place_index.get_index().institutions(term)
        alternatives = _institution_text_alternatives(term)
        if names or ror_names:
            parts = []
            if names:
                parts.append(f"d.institutions && {_array(names)}")
            if ror_names:
                tables = _catalogue.tables()
                parts.append(f"EXISTS (SELECT 1 FROM {tables.dataset_sites.name} AS s "
                             f"JOIN {tables.site_places.name} AS sp "
                             "ON sp.name_raw = s.name_raw WHERE s.dataset_id = d.id "
                             f"AND sp.institution = ANY({_array(ror_names)}))")
            text_part = _format_text_match(alternatives, _MODE_SHORT)
            if text_part:
                parts.append(text_part)
            condition = parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"
        else:
            text_condition = _format_text_match(alternatives or [term])
            if not text_condition:
                continue
            condition = text_condition
        label = term.replace('"', "'")
        lines.append(f'  institution "{label}": {condition}')
    return lines


def _location_lines(plan: QueryPlan) -> List[str]:
    """One filter line per city, region or country the user named.

    A place stored in bids_site_places (the sites of bids_dataset_sites,
    matched to ROR; RAG/place_index.py) becomes a join on the sites, or the
    place named in the dataset's short text fields:
        (EXISTS (SELECT 1 FROM bids_dataset_sites AS s JOIN bids_site_places AS sp
                 ON sp.name_raw = s.name_raw
                 WHERE s.dataset_id = d.id AND sp.city = 'Bonn')
         OR text_match('Bonn', 'short'))
    A name that is several places ("Washington" the city and the state) joins
    the sites of each. A name that is also a word of another place's name
    ("Mexico" in "New Mexico") gets a second condition: a dataset whose text
    names that other place counts only through its sites.
        ((EXISTS (... AND sp.country = 'Mexico') OR text_match('mexico', 'short'))
         AND (EXISTS (... AND sp.country = 'Mexico') OR NOT text_match('new mexico', 'near')))
    Its own top-level condition, not a NOT beside the text search: the
    uncertain tier keeps a negated condition as a hard filter, so the README
    of a University of New Mexico dataset is no "probable" match either, and
    the chips show it as a location chip of its own (query_filters.py).
    A place no site is in is searched in the whole dataset text, README
    included, where labs often say where they scanned.
    """
    lines: List[str] = []
    index = _place_index.get_index()
    for term in dict.fromkeys(t.strip() for t in plan.location_terms if t and t.strip()):
        readings = index.readings(term)
        if readings:
            tables = _catalogue.tables()
            places = " OR ".join(f"sp.{column} = '{value.replace(chr(39), chr(39) * 2)}'"
                                 for column, value in readings)
            sites = (f"EXISTS (SELECT 1 FROM {tables.dataset_sites.name} AS s "
                     f"JOIN {tables.site_places.name} AS sp "
                     "ON sp.name_raw = s.name_raw WHERE s.dataset_id = d.id "
                     f"AND {places if len(readings) == 1 else '(' + places + ')'})")
            text_part = _format_text_match(
                list(dict.fromkeys([term] + [value for _, value in readings])), _MODE_SHORT)
            condition = f"({sites} OR {text_part})" if text_part else sites
            elsewhere = _format_text_match(index.foreign_names(term), _MODE_NEAR)
            if text_part and elsewhere:
                condition = f"({condition} AND ({sites} OR NOT {elsewhere}))"
        else:
            text_condition = _format_text_match([term])
            if not text_condition:
                continue
            condition = text_condition
        lines.append(f'  location "{term.replace(chr(34), chr(39))}": {condition}')
    return lines


_BYTES_PER_GB = 1024 ** 3


def _size_conditions(plan: QueryPlan) -> List[str]:
    """SQL conditions on d.size_bytes for the size limits in the plan."""
    conds: List[str] = []
    if plan.max_size_gb is not None and plan.max_size_gb > 0:
        conds.append(f"d.size_bytes <= {int(plan.max_size_gb * _BYTES_PER_GB)}")
    if plan.min_size_gb is not None and plan.min_size_gb > 0:
        conds.append(f"d.size_bytes >= {int(plan.min_size_gb * _BYTES_PER_GB)}")
    return conds


# Words that describe a request, not a dataset: a text search for them empties
# the result ("standard & manner" matched 17 of 1,721 datasets and removed 975
# of 986 correct answers of one real request).
_REQUEST_WORDS = frozenset("""
    standard manner simple very trivial non-trivial nontrivial complex easy good great nice
    suitable useful appropriate sufficient enough complete completely comprehensive
    available included including provided labeled labelled annotated documented public
    benchmark benchmarking
    publication paper reports reported reporting results that which with have has containing
    contains format formatted dataset datasets data collection collections looking need want
    similar like ideally preferably possible
    analyse analyze analyses analyzes analysing analyzing analysed analyzed analysis specifically
    bids bids-valid bids-compliant valid compliant raw
""".split())


# Request words that are also part of what datasets study: "deep brain
# stimulation", "reward learning", "cognitive training", "sleep quality",
# "eyes open", "free recall", "basic emotions". Dropped only when no
# neighbouring word is content ("for training", "open dataset").
_AMBIGUOUS_REQUEST_WORDS = frozenset("""
    machine learning deep training train classification classify quality open free basic
""".split())


# Whole wish phrases, cut out before single words are looked at.
_REQUEST_PHRASES_RE = re.compile(
    r"\b(?:machine[ -]learning|deep[ -]learning|high[ -]quality|good[ -]quality|open[ -]access|"
    r"open[ -]source|freely available|free to use|publicly available|raw (?:or|and) processed)\b",
    re.IGNORECASE,
)


_FUNCTION_WORDS = frozenset("a an and as at by for from in of on or the to".split())


# Words that name a kind of recording, and filler that may accompany them.
_MODALITY_WORDS = frozenset("""
    fmri mri smri dmri eeg meg ieeg seeg ecog pet ct bold dwi dti asl fnirs nirs t1 t1w t2 t2w
    flair swi mra tof anatomical structural functional diffusion perfusion
    electroencephalography magnetoencephalography neuroimaging imaging
""".split())


_SCAN_FILLER_WORDS = frozenset("task based data scan scans images image recordings recording dataset datasets weighted".split())


def _without_phrases(text: str, phrases: Sequence[str]) -> str:
    """*text* with every phrase of *phrases* cut out (case-insensitive)."""
    for phrase in sorted((p.strip() for p in phrases if p and p.strip()), key=len, reverse=True):
        text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


_MIN_TEXT_MATCHES = 3


def _matches_too_few(text: str) -> bool:
    """True when fewer than _MIN_TEXT_MATCHES datasets contain every word of *text*.

    False when the catalogue cannot be asked (no database): the condition is
    then kept, as before.
    """
    try:
        tsquery = _catalogue.tsquery()
    except ImportError:
        return False
    counts = tsquery.count_matches([text], _MODE_ALL)
    return bool(counts) and counts[0] is not None and counts[0] < _MIN_TEXT_MATCHES


def _formula_matches_too_few(part: "_text_formula.Node") -> bool:
    """True when fewer than _MIN_TEXT_MATCHES datasets match *part* of a description formula.

    An OR of phrases counts each phrase and adds the counts (an upper bound of
    the datasets matching any). A part that is no plain text search (one with
    a NOT inside) is never too few.
    """
    alternatives = _text_formula.alternatives(part)
    if not alternatives:
        return False
    if len(alternatives) == 1:
        return _matches_too_few(alternatives[0])
    try:
        tsquery = _catalogue.tsquery()
    except ImportError:
        return False
    counts = tsquery.count_matches(alternatives, _MODE_ALL)
    if not counts or any(c is None for c in counts):
        return False
    return sum(counts) < _MIN_TEXT_MATCHES


def _content_words(text: str) -> str:
    """*text* without the words that describe the request (see _REQUEST_WORDS).

    A word of _AMBIGUOUS_REQUEST_WORDS stays when a neighbour is content:
    "deep brain stimulation" keeps "deep", "open data, classification" is "".
    """
    words = _REQUEST_PHRASES_RE.sub(" ", text).split()
    bare = [w.lower().strip(".,;:()'\"") for w in words]

    def content(i: int) -> bool:
        return (0 <= i < len(bare) and bare[i] not in _REQUEST_WORDS
                and bare[i] not in _AMBIGUOUS_REQUEST_WORDS and bare[i] not in _FUNCTION_WORDS)

    kept = [
        word for i, word in enumerate(words)
        if bare[i] not in _REQUEST_WORDS
        and (bare[i] not in _AMBIGUOUS_REQUEST_WORDS or content(i - 1) or content(i + 1))
    ]
    return " ".join(kept)


# Words around an age group in "datasets with children", "data from kids".
_AGE_FILLER_WORDS = frozenset((
    "datasets", "dataset", "data", "recordings", "recording", "studies", "study", "research",
    "participants", "participant", "subjects", "subject", "people", "population", "populations",
    "samples", "sample", "cohort", "cohorts", "find", "show", "me", "that", "which", "with",
    "from", "of", "in", "on", "all", "any", "the", "and", "or", "aged", "group", "groups",
))


def _age_groups_only(description_search: Optional[str]) -> Optional[List[str]]:
    """The age groups, when *description_search* names nothing else ("pediatric", "kids")."""
    text = (description_search or "").lower().replace("_", " ")
    words = [w for w in re.findall(r"[a-z]+(?:-[a-z]+)*", text) if w not in _AGE_FILLER_WORDS]
    if not words:
        return None
    # The whole phrase ("older adults"), or every word on its own ("kids, teenagers").
    whole = _normalise_age_groups([" ".join(words)])
    if whole:
        return whole
    if all(_normalise_age_groups([w]) for w in words):
        return _normalise_age_groups(words)
    return None


_SPECIES_ONLY_RE = re.compile(
    r"\s*(?:(?:non[- ]?human|humans?|homo sapiens|animals?|rodents?|rats?|mouse|mice|murine|monkeys?|macaques?|macaca|rhesus|"
    r"baboons?|marmosets?|capuchins?|cynomolgus|lemurs?|primates?|nhps?|pigs?|swine|sheep|dogs?|canine|cats?|"
    r"ferrets?|bats?|birds?|pigeons?|songbirds?|zebra ?finch(?:es)?|zebrafish|species|"
    # words around the species in "datasets that contain monkeys", "data from rats"
    r"datasets?|data|recordings?|studies|study|research|experiments?|models?|subjects?|"
    r"find|show|me|that|which|contain(?:s|ing)?|with|from|of|in|on|all|any|the|and|or|,)\s*)+",
    re.IGNORECASE,
)


def _study_structure_condition(plan: QueryPlan, available_columns: Collection[str]) -> Optional[str]:
    """The study-design filter: the uploader's choice on OpenNeuro first.

    d.study_longitudinal holds what the uploader picked ('Longitudinal',
    'Cross-Sectional', 'Other' …; NULL for about half). Without it the search
    looked for the word in the README only and missed 89 of the 126 datasets
    labelled 'Longitudinal'. A dataset without a label, or with a label that
    does not contradict the design, still counts when its text names the
    design; one labelled with the other design does not ("part of a
    longitudinal cohort" in a cross-sectional dataset).

    The label guard is its own AND part: the uncertain tier widens a text
    condition with paper evidence by OR-ing it to the whole condition, and a
    guard inside the OR let a dataset labelled 'Cross-Sectional' in through
    its paper (ds006670). Labels naming both designs ('Longitudinal with
    Cross-sectional groups') pass either question.
    """
    if not plan.study_structure:
        return None
    if plan.study_structure == "longitudinal":
        label, other, words = "longitudinal", "cross", "longitudinal"
    else:
        label, other, words = "cross", "longitudinal", "cross-sectional"
    text = _format_text_match([words])
    if "study_longitudinal" not in available_columns:
        return text
    return (f"(NOT (COALESCE(d.study_longitudinal, '') ~* '{other}' "
            f"AND COALESCE(d.study_longitudinal, '') !~* '{label}') "
            f"AND (d.study_longitudinal ~* '{label}' OR {text}))")


def _diagnosis_lines(plan: QueryPlan,
                     term_to_codes: Dict[str, List[str]]) -> List[Tuple[str, Dict[str, List[str]]]]:
    """The diagnosis filter lines of the augmented question: (label, terms).

    One line for all diagnoses, except when the plan asks for several groups
    that must all be present ("schizophrenia patients and healthy controls"):
    then one line per group. A single IN list over both groups would also
    accept a dataset with only one of them. The lines of the block are ANDed
    by the SQL stage, so separate lines keep the AND between the groups.
    """
    everything = [("diagnosis", term_to_codes)]
    if plan.group_operator != "AND":
        return everything
    by_key = {term.strip().casefold(): term for term in term_to_codes}
    per_group: List[Dict[str, List[str]]] = []
    for group in plan.groups:
        terms = {by_key[t.strip().casefold()]: term_to_codes[by_key[t.strip().casefold()]]
                 for t in group.diagnosis_terms if t.strip().casefold() in by_key}
        if any(terms.values()):
            per_group.append(terms)
    grouped = {term for terms in per_group for term in terms}
    if len(per_group) < 2 or grouped != set(term_to_codes):
        return everything   # one group, or a term no group owns: as before
    return [(f"diagnosis (participant group {i} of {len(per_group)}, every group is required)", terms)
            for i, terms in enumerate(per_group, 1)]


def _term_key(term: str) -> str:
    return " ".join((term or "").lower().split())


def _diagnosis_condition(term_to_codes: Dict[str, List[str]], qualifier: List[str]) -> Optional[str]:
    """The diagnosis filter for some terms: coded rows OR the words in the short fields.

    With *qualifier* (text_terms.diagnosis_qualifier) the dataset text must
    also hold those words: "Lewy body dementia" keeps the dementia codes, but
    only for datasets that say "Lewy body".
    """
    codes = list(dict.fromkeys(c for cs in term_to_codes.values() for c in cs))
    if not codes:
        return None
    vals = ", ".join(f"'{c}'" for c in codes)
    condition = (
        f"EXISTS (SELECT 1 FROM {_catalogue.tables().participants.name} p2 "
        f"WHERE p2.dataset_id = d.id AND p2.diagnosis IN ({vals}))"
    )
    # Structured diagnosis rows are sparse: back them up with the user's words
    # and the vocabulary synonyms in name, description and study fields.
    text_fallback = _format_text_match(_diagnosis_alternatives(term_to_codes), _MODE_SHORT)
    if text_fallback:
        condition = f"({condition} OR {text_fallback})"
    required = _format_text_match([" ".join(qualifier)], _MODE_ALL) if qualifier else None
    return f"({condition} AND {required})" if required else condition


def _excluded_diagnosis_keys(plan: QueryPlan) -> set:
    """The excluded diagnosis terms, as _term_key. A term a group also asks
    for stays asked: the planner contradicted itself, and a search that
    excludes what it looks for finds nothing."""
    asked = {_term_key(t) for g in plan.groups for t in g.diagnosis_terms}
    return {_term_key(t) for t in plan.excluded_diagnosis_terms
            if t and t.strip() and not _is_healthy_term(t)} - asked


def _excluded_diagnosis_lines(plan: QueryPlan, resolved: List[ResolvedTerms],
                              unresolved: List[str]) -> List[str]:
    """One line per diagnosis the dataset must not have.

    The condition is the negation of the one a request for the diagnosis
    uses (_diagnosis_condition): a participant coded with it, or its words in
    the name, description, study fields or dx_status. A dataset that records
    no diagnosis passes: what is not stated is not there (RAG/criteria.py
    Criterion.default). A README mention alone does not exclude. A term the
    vocabulary does not know is excluded by its words in the short fields.
    """
    excluded = _excluded_diagnosis_keys(plan)
    if not excluded:
        return []
    conditions: Dict[str, str] = {}
    for r in resolved:
        if r.field != RAGField.DIAGNOSIS:
            continue
        for term, codes in r.term_to_codes.items():
            if _term_key(term) not in excluded or not codes:
                continue
            stated = _diagnosis_condition({term: codes}, _diagnosis_qualifier(term, codes))
            if stated:
                conditions.setdefault(term, stated)
    for term in unresolved:
        text = _format_text_match([term], _MODE_SHORT)
        if _term_key(term) in excluded and text:
            conditions.setdefault(term, text)
    return [f'  not diagnosis "{term.replace(chr(34), chr(39))}" (complete, datasets that record '
            f'no diagnosis stay): {_without_diagnosis(stated)}'
            for term, stated in conditions.items()]


def _build_augmented_question(
    plan: QueryPlan,
    resolved: List[ResolvedTerms],
    unresolved_terms: Optional[Dict[str, List[str]]] = None,
    db_miss_terms: Optional[Dict[str, List[str]]] = None,
    available_columns: Collection[str] = (),
    topics: Sequence[ResolvedTopic] = (),
    unmatched_text: Optional[List[str]] = None,
) -> str:
    """
    Inject resolved canonical codes into the natural-language summary.

    *unmatched_text*, when given, receives the description phrases left out
    because too few datasets say them (AugmentedQueryPlan.unmatched_text).

    *available_columns* names the optional bids_datasets columns the database
    has populated (backend/services/database_stats/database_stats.py). A filter on a
    column that is not listed is left out rather than emitted and failing.

    Scan-related fields (datatype / suffix / task) are grouped by the source
    user term and emitted as pre-built EXISTS subqueries so the SQL model can
    copy them verbatim.  All three fields that matched the same user term
    (e.g. "fMRI" → datatype=functional_mri AND suffix=fmri_bold) are
    combined with AND inside a single EXISTS.

    A diagnosis becomes an EXISTS on the structured rows OR a text_match()
    placeholder on the short text fields, because most datasets about a
    disease have no diagnosis rows.  Free-text terms (description_search,
    unresolved vocabulary) become text_match() placeholders as well; the
    backend expands them into full-text predicates (RAG/text_terms.py).

    Author and funding are listed as exact DB strings.
    """
    size_conds = _size_conditions(plan) if "size_bytes" in available_columns else []
    participant_lines = _participant_conditions(plan)
    design_cond = _study_structure_condition(plan, available_columns)
    species_cond = _species_condition(plan.species, "species" in available_columns)
    # An age group the planner left in description_search ("children") is the
    # age filter too, not the word "children" in the text.
    description_age_groups = _age_groups_only(plan.description_search)
    age_codes = _normalise_age_groups(list(plan.age_groups) + (description_age_groups or []))
    age_cond = _age_condition(age_codes, "species" in available_columns)
    if age_cond:
        # The age-group line is the complete age condition (RAG/criteria.py).
        participant_lines = [line for line in participant_lines
                             if not line.lstrip().startswith("participant age:")]
    # Unresolved terms count too: their text filter is the only thing that
    # keeps a question like "eye tracking datasets" from returning everything.
    if (not resolved and not plan.description_search and not plan.name_search
            and not size_conds and not participant_lines
            and not design_cond and not species_cond and not age_cond
            and not plan.institution_terms
            and not plan.excluded_institution_terms
            and not plan.excluded_diagnosis_terms
            and not plan.location_terms
            and not unresolved_terms and not db_miss_terms
            and not any(t.condition for t in topics)):
        return plan.natural_language_summary

    scan_fields = {RAGField.DATATYPE, RAGField.SUFFIX, RAGField.TASK}
    field_order = (RAGField.DATATYPE, RAGField.SUFFIX, RAGField.TASK)

    # Group scan codes by source term: term → {RAGField: [codes]}
    scan_by_term: Dict[str, Dict[RAGField, List[str]]] = {}
    non_scan_lines: List[str] = []
    context_lines: List[str] = []

    excluded_diagnoses = _excluded_diagnosis_keys(plan)

    label_map = {
        RAGField.DIAGNOSIS: "diagnosis codes",
        RAGField.AUTHOR:    "author names (exact DB strings)",
        RAGField.FUNDING:   "funding sources (exact DB strings)",
    }

    for r in resolved:
        if r.field in scan_fields:
            for term, codes in r.term_to_codes.items():
                if codes:
                    scan_by_term.setdefault(term, {})[r.field] = codes
        elif r.field == RAGField.DIAGNOSIS:
            # Excluded diagnoses have lines of their own (_excluded_diagnosis_lines).
            asked = {t: cs for t, cs in r.term_to_codes.items() if _term_key(t) not in excluded_diagnoses}
            for label, term_to_codes in _diagnosis_lines(plan, asked):
                # A term whose codes miss one of its disease words ("Lewy body
                # dementia" → the dementia group) needs those words in the
                # dataset text; such terms get a condition of their own.
                qualifiers = {t: _diagnosis_qualifier(t, cs) for t, cs in term_to_codes.items() if cs}
                if any(qualifiers.values()):
                    parts = [p for t, q in qualifiers.items()
                             if (p := _diagnosis_condition({t: term_to_codes[t]}, q))]
                    condition = (parts[0] if len(parts) == 1
                                 else "(" + " OR ".join(parts) + ")") if parts else None
                else:
                    condition = _diagnosis_condition(term_to_codes, [])
                if condition:
                    context_lines.append(f'  {label}: {condition}')
        elif r.field == RAGField.INSTITUTION:
            continue  # see _institution_lines()
        else:
            if r.all_codes:
                label = label_map.get(r.field, r.field.value)
                non_scan_lines.append(f"  {label}: {', '.join(r.all_codes)}")

    # Other words for a scan term (scan_term_alternatives, checked against the
    # catalogue by check_scan_term_alternatives): an ERP study that says only
    # "oddball" or "P300". Searched next to the term's own text search; a
    # dataset that has only them ranks after those with the user's words.
    term_alternatives = {_term_key(e.term): e.alternatives
                         for e in getattr(plan, "scan_term_alternatives", []) or [] if e.alternatives}

    def with_alternatives(term: str, text: Optional[str]) -> Optional[str]:
        alternatives = term_alternatives.get(_term_key(term))
        other = _format_text_match(alternatives, _MODE_ALSO) if alternatives else None
        if not other:
            return text
        return f"({text} OR {other})" if text else other

    # Contradiction detection: build flat {field_name: [codes]} from all resolved data.
    _flat: Dict[str, List[str]] = {}
    for _term_field_codes in scan_by_term.values():
        for _rag_field, _codes in _term_field_codes.items():
            _flat.setdefault(_rag_field.value, []).extend(_codes)
    for _r in resolved:
        if _r.field == RAGField.DIAGNOSIS:
            _asked_codes = [c for t, cs in _r.term_to_codes.items()
                            if _term_key(t) not in excluded_diagnoses for c in cs]
            if _asked_codes:
                _flat.setdefault(_r.field.value, []).extend(_asked_codes)

    _contradictions = _check_contradictions(_flat)
    if _contradictions:
        _hard = [w.message for w in _contradictions if w.severity == "hard"]
        _soft = [w.message for w in _contradictions if w.severity == "soft"]
        _parts: List[str] = []
        if _hard:
            _parts.append("HARD CONTRADICTION — " + " | ".join(_hard))
        if _soft:
            _parts.append("WARNING — " + " | ".join(_soft))
        context_lines.append("[Query note: " + " ".join(_parts) + "]")

    # Alternatives ("X, Y or Z", ScanRequirement.any_of): the conditions of
    # the listed terms become ONE line joined with OR, so the SQL model copies
    # the "or" instead of deciding between AND and OR itself. A negated list
    # needs every term excluded, which separate lines already say.
    any_of_lists: List[List[str]] = [
        list(dict.fromkeys(t for t in req.scan_terms if t and t.strip()))
        for req in plan.scan_requirements
        if req.any_of and req.required and not req.same_subject
    ]
    # "EEG during sleep or anesthesia" sometimes comes back as one list of
    # EEG, sleep and anesthesia (1 of 3 runs, 2026-11-15). Read as written,
    # that returns every EEG dataset. One modality next to two or more tasks
    # is the modality the tasks belong to: it leaves the list and is required.
    # (Two modalities, "EEG or MEG", and one of each, "EEG or sleep", stay.)
    _task_terms = {_term_key(t) for t, codes in scan_by_term.items() if RAGField.TASK in codes}
    _resolved_terms = {_term_key(t) for t in scan_by_term}
    for terms in any_of_lists:
        modalities = [t for t in terms
                      if _term_key(t) in _resolved_terms and _term_key(t) not in _task_terms]
        if len(modalities) == 1 and len(terms) - 1 >= 2:
            terms.remove(modalities[0])
    any_of_lists = [terms for terms in any_of_lists if len(terms) > 1]
    any_of_index = {_term_key(term): i for i, terms in enumerate(any_of_lists) for term in terms}
    any_of_conditions: Dict[int, Dict[str, str]] = {}

    def add_scan_line(term: str, condition: str) -> None:
        index = any_of_index.get(_term_key(term))
        if index is None:
            context_lines.append(f'  scan "{term}": {condition}')
            return
        held = any_of_conditions.setdefault(index, {})
        key = _term_key(term)
        # A term with a modality and a task condition matches on either.
        held[key] = f"({held[key]} OR {condition})" if key in held else condition

    # Build EXISTS hints per source term.
    # Modality (datatype + suffix) uses OR — a file qualifies if it matches
    # the modality by EITHER its folder (datatype) OR its file suffix.
    # Task gets a separate EXISTS so compound terms like "resting state fMRI"
    # correctly emit both a modality filter and a task filter.
    objects = _catalogue.tables().objects.name
    for term, field_codes in scan_by_term.items():
        modality_conds: List[str] = []   # datatype + suffix codes
        task_conds: List[str] = []       # task codes

        for field in field_order:
            if field not in field_codes:
                continue
            codes = field_codes[field]
            col   = field.value
            if len(codes) == 1:
                cond = f"o2.{col} = '{codes[0]}'"
            else:
                vals = ", ".join(f"'{c}'" for c in codes)
                cond = f"o2.{col} IN ({vals})"
            if field == RAGField.TASK:
                task_conds.append(cond)
            else:
                modality_conds.append(cond)

        # If a term resolves to BOTH modality codes AND task codes, combine
        # everything into one OR EXISTS — matching any of the codes increases
        # confidence that the term is present, so OR is correct within a term.
        # Example: "attention" → (suffix='att' OR task IN (attention_tasks))
        #
        # If the term resolved to ONLY modality codes (e.g. "fMRI", "EEG"),
        # emit a single modality EXISTS with OR between datatype and suffix.
        # If the term resolved to ONLY task codes (e.g. "resting state",
        # "working memory"), emit a separate task EXISTS.
        # A term that names a task ("resting state") must not be satisfied by
        # a data type it merely resembles: "resting state" fuzzy-matches the
        # datatype synonyms "resting state fMRI" / "resting-state EEG", and
        # (datatype IN (fMRI, EEG) OR task IN (rest ...)) is true for every
        # fMRI and EEG dataset. With task codes present, datatype codes are
        # dropped; suffix codes stay only when no datatype matched (the
        # "attention" -> suffix 'att' OR attention tasks case).
        # A term made only of modality words ("fMRI", "task fMRI", "EEG data")
        # is the opposite case: it names a data type, and the generic task
        # code it also matches ('bold_acquisition_general') must not replace it.
        if task_conds and RAGField.DATATYPE in field_codes:
            words = [w for w in re.findall(r"[a-z0-9*]+", term.lower()) if w not in _SCAN_FILLER_WORDS]
            if words and all(w in _MODALITY_WORDS for w in words):
                task_conds = []
            else:
                modality_conds = []
        all_conds = modality_conds + task_conds
        if modality_conds and task_conds:
            # Single OR EXISTS across all fields for this term
            or_part = " OR ".join(all_conds)
            if len(all_conds) > 1:
                or_part = f"({or_part})"
            condition = f"EXISTS (SELECT 1 FROM {objects} o2 WHERE o2.dataset_id = d.id AND {or_part})"
            text_fallback = with_alternatives(term, _format_task_text_match([term]))
            if text_fallback:
                condition = f"({condition} OR {text_fallback})"
            add_scan_line(term, condition)
        else:
            # Modality EXISTS: OR between datatype and suffix so partially-annotated
            # files still match (e.g. file has suffix=fmri_bold but no explicit datatype).
            if modality_conds:
                or_part = " OR ".join(modality_conds)
                if len(modality_conds) > 1:
                    or_part = f"({or_part})"
                add_scan_line(
                    term,
                    f"EXISTS (SELECT 1 FROM {objects} o2 WHERE o2.dataset_id = d.id AND {or_part})",
                )

            # Task EXISTS: separate from modality so both constraints apply independently.
            # e.g. "resting state fMRI" → fMRI modality EXISTS AND resting_state task EXISTS
            if task_conds:
                where_part = " OR ".join(task_conds)
                if len(task_conds) > 1:
                    where_part = f"({where_part})"
                condition = f"EXISTS (SELECT 1 FROM {objects} o2 WHERE o2.dataset_id = d.id AND {where_part})"
                text_fallback = with_alternatives(term, _format_task_text_match([term]))
                if text_fallback:
                    condition = f"({condition} OR {text_fallback})"
                add_scan_line(term, condition)

    # One line per list of alternatives. A listed term the vocabulary does not
    # know joins the OR as a text search and leaves the VOCABULARY MISS note,
    # whose text filters are required one by one.
    unresolved_terms = {field: list(terms) for field, terms in (unresolved_terms or {}).items()}
    for index, terms in enumerate(any_of_lists):
        held = any_of_conditions.get(index, {})
        for term in terms:
            if _term_key(term) in held:
                continue
            for field_name, missed in unresolved_terms.items():
                if field_name != RAGField.INSTITUTION.value and any(_term_key(m) == _term_key(term) for m in missed):
                    text_only = with_alternatives(term, _format_text_match([term], _MODE_LOOSE))
                    if text_only:
                        held[_term_key(term)] = text_only
                        unresolved_terms[field_name] = [m for m in missed if _term_key(m) != _term_key(term)]
                    break
        listed = [term for term in terms if _term_key(term) in held]
        if len(listed) > 1:
            names = ", ".join(f'"{term}"' for term in listed)
            joined = " OR ".join(held[_term_key(term)] for term in listed)
            context_lines.append(f"  scan any of {names} (alternatives, one is enough): ({joined})")
        elif listed:
            context_lines.append(f'  scan "{listed[0]}": {held[_term_key(listed[0])]}')
    unresolved_terms = {field: terms for field, terms in unresolved_terms.items() if terms}

    # Topics: the qualified subject as one condition (see _topic_condition).
    for topic in topics:
        if topic.condition:
            label = (topic.topic.phrase or topic.topic.typed).replace('"', "").replace("'", "")
            context_lines.append(f'  topic "{label}": {topic.condition}')

    context_lines.extend(non_scan_lines)

    # Institutions: an exact filter on d.institutions, or a text search when
    # the stored names do not know the institution. Unresolved institution
    # terms are handled there, not in the VOCABULARY MISS block below.
    unresolved_terms = dict(unresolved_terms or {})
    context_lines.extend(_institution_lines(
        plan, resolved,
        unresolved_terms.pop(RAGField.INSTITUTION.value, []),
        available_columns,
    ))
    context_lines.extend(_location_lines(plan))
    # Excluded diagnoses the vocabulary does not know are excluded by their
    # words, not searched for by the VOCABULARY MISS block below.
    _unknown_diagnoses = unresolved_terms.pop(RAGField.DIAGNOSIS.value, [])
    context_lines.extend(_excluded_diagnosis_lines(
        plan, resolved, [t for t in _unknown_diagnoses if _term_key(t) in excluded_diagnoses]))
    _unknown_diagnoses = [t for t in _unknown_diagnoses if _term_key(t) not in excluded_diagnoses]
    if _unknown_diagnoses:
        unresolved_terms[RAGField.DIAGNOSIS.value] = _unknown_diagnoses

    # Inject free-text hints for qualitative/adjectival terms that the
    # preprocessing LLM routed to description_search / name_search because they
    # can't be resolved to a DB code (e.g. "complex", "pediatric", "demanding").
    # description_search covers name, description, study fields and README.
    description_search = plan.description_search or ""
    if design_cond and description_search.strip().lower().replace("-", " ") in (
            "longitudinal", "cross sectional", "longitudinal study", "cross sectional study"):
        description_search = ""  # already the study-design filter, which also reads the label
    if _SPECIES_ONLY_RE.fullmatch(description_search):
        # The SQL stage filters d.species (with a title fallback); a README
        # match on top dropped mouse datasets whose README says "C57BL/6".
        description_search = ""
    if description_age_groups:
        description_search = ""  # the age-group line below (RAG/criteria.py)
    # description_search is a small formula ("drawings or art", "memory but
    # not sleep"; RAG/text_formula.py): its phrases are cleaned one by one and
    # the operators survive, instead of "or" and "not" being dropped as words.
    formula = _text_formula.parse_formula(_without_phrases(description_search, plan.request_wishes))
    not_searched = [w for w in plan.request_wishes if w.strip()]
    # Who the participants are is decided by the participant lines, never by
    # a text search for "healthy", "adult" or "male".
    said_healthy = any(_is_healthy_term(t) for t in plan.excluded_diagnosis_terms) or any(
        _is_healthy_term(t) for g in plan.groups for t in g.diagnosis_terms)

    def searched_words(phrase: str) -> str:
        nonlocal said_healthy
        kept_words = []
        for word in _content_words(phrase).split():
            bare = word.lower().strip(".,;:()'\"")
            if bare in _HEALTHY_WORDS and bare not in ("typical", "developing", "control"):
                said_healthy = True
            elif not (participant_lines and bare in ("adult", "adults", "male", "female", "males", "females", "men", "women")):
                kept_words.append(word)
        return " ".join(kept_words)

    formula = _text_formula.map_phrases(formula, searched_words)
    if said_healthy:
        not_searched.append("healthy (most datasets include healthy participants and few record it)")
    context_lines.extend(participant_lines)
    # The words dataset authors use for the same thing ("post-mortem" for
    # "dead people"), already checked against the catalogue
    # (check_description_alternatives). With them the user's own words stay a
    # filter even when no dataset text says them. They stand for the whole
    # search, so they are used only when it is one text search: a phrase or
    # phrases joined by OR, not "memory AND NOT sleep".
    single_search = formula is not None and _text_formula.placeholder(formula) is not None
    other_words = (_format_text_match(plan.description_alternatives, _MODE_ALSO)
                   if single_search else None)
    # Other conditions of the question; "[...]" lines are notes, not filters.
    filters_besides_text = bool(
        any(line.startswith("  ") for line in context_lines)
        or plan.name_search or size_conds or design_cond or species_cond or age_cond
        or unresolved_terms
        or any(g.min_subjects is not None for g in plan.groups)
    )
    if formula is not None and not other_words and filters_besides_text:
        # A part no dataset text says would only empty the result as a filter.
        # A NOT part never empties it and stays. Not when the description is
        # the whole question ("bonn", "datasets with astronauts", "drawings or
        # art"): without it there is no condition left and the answer is the
        # entire catalogue; searched, it gives the few datasets that say it, or
        # an honest empty page.
        kept_parts = []
        for part in _text_formula.conjuncts(formula):
            if not isinstance(part, _text_formula.Not) and _formula_matches_too_few(part):
                not_searched.append(_text_formula.render(part))
                if unmatched_text is not None:
                    unmatched_text.append(_text_formula.render(part))
                continue
            kept_parts.append(part)
        formula = _text_formula.And(tuple(kept_parts)) if len(kept_parts) > 1 else (
            kept_parts[0] if kept_parts else None)
    description_match = _text_formula.compile_formula(formula)
    if description_match:
        if other_words:
            description_match = f"({description_match} OR {other_words})"
        context_lines.append(f"  dataset text: {description_match}")
    if not_searched:
        context_lines.append(
            "  not searchable (add NO condition for these): " + "; ".join(dict.fromkeys(not_searched))
        )
    if plan.name_search:
        context_lines.append(
            f"  dataset name: d.name ILIKE '%{plan.name_search}%'"
        )

    # Dataset size limit — a plain condition on bids_datasets, no EXISTS needed.
    if size_conds:
        context_lines.append("  dataset size: " + " AND ".join(size_conds))

    if design_cond:
        context_lines.append(f"  study design: {design_cond}")

    if species_cond:
        context_lines.append(f"  species (complete, no other species condition): {species_cond}")

    if age_cond:
        context_lines.append(f"  age group {_age_group_label(age_codes)} "
                             f"(complete, no other age condition): {age_cond}")

    # "At least N patients with X": count the participants who have the
    # diagnosis (RAG/cohort_count.py), one line per group. The backend expands
    # the placeholder; a dataset that does not record who has the diagnosis
    # goes to the uncertain tier instead of counting all its subjects.
    diagnosis_codes = {
        _term_key(term): codes
        for r in resolved if r.field == RAGField.DIAGNOSIS
        for term, codes in r.term_to_codes.items()
    }
    counted_groups = set()
    for index, group in enumerate(plan.groups):
        if group.min_subjects is None:
            continue
        codes = [code for term in group.diagnosis_terms
                 if not _is_healthy_term(term)
                 for code in diagnosis_codes.get(_term_key(term), [])]
        count = _patients_placeholder(codes, group.min_subjects)
        if count:
            counted_groups.add(index)
            context_lines.append(
                f"  patients (at least {group.min_subjects} participants with this diagnosis; "
                f"complete, add no other count of participants or subjects): {count}"
            )

    # A minimum without a diagnosis is the size of the dataset: a HAVING hint
    # on the total subject count ("datasets with at least 50 participants").
    min_subjects_values = [
        g.min_subjects for i, g in enumerate(plan.groups)
        if g.min_subjects is not None and i not in counted_groups
    ]
    if min_subjects_values:
        n = max(min_subjects_values)
        context_lines.append(
            f"  subject count: HAVING COUNT(DISTINCT o.subject) >= {n}"
            " [apply this HAVING clause on the outer query, not in a subquery]"
        )

    # Warn about terms that RAG could not map — prevent SQL model from generating
    # broken text filters on canonical-code columns (diagnosis, task, datatype, suffix).
    if unresolved_terms:
        missing_parts = []
        for field_name, terms in unresolved_terms.items():
            missing_parts.append(f"{field_name}: {', '.join(repr(t) for t in terms)}")
        # One 'loose' text_match() per term: the dataset must contain the
        # term's most distinctive word anywhere in its text (README included);
        # the other words only move it up in the ranking.
        text_suggestions = []
        for terms in unresolved_terms.values():
            for term in terms:
                suggestion = with_alternatives(term, _format_text_match([term], _MODE_LOOSE))
                if suggestion and suggestion not in text_suggestions:
                    text_suggestions.append(suggestion)
        ilike_hint = ""
        if text_suggestions:
            ilike_hint = (
                " For free-text search use: "
                + " AND ".join(text_suggestions)
            )
        context_lines.append(
            "[VOCABULARY MISS — these terms are not in the known concept vocabulary: "
            + "; ".join(missing_parts)
            + ". Do NOT filter bids_participants.diagnosis or bids_objects.task/datatype/suffix "
            "for these terms — those columns store canonical codes, not free text."
            + ilike_hint + "]"
        )

    # Warn about codes that resolved in the vocabulary but have zero DB rows.
    # The SQL model must omit these from EXISTS filters (they always return empty).
    if db_miss_terms:
        miss_parts = []
        for field_name, codes in db_miss_terms.items():
            unique_codes = list(dict.fromkeys(codes))  # deduplicate
            miss_parts.append(
                f"{field_name}: {', '.join(repr(c) for c in unique_codes)}"
            )
        context_lines.append(
            "[DB MISS — resolved codes with zero rows in the database: "
            + "; ".join(miss_parts)
            + ". These terms resolve in the vocabulary but have no patient/scan data yet. "
            "Omit these codes from EXISTS filters entirely — do not include them in "
            "IN (...) lists or = comparisons. If a broader parent code is available "
            "(e.g. 'anxiety_disorder' when children like 'panic_disorder' are missing), "
            "use the parent code instead.]"
        )

    if not context_lines:
        return plan.natural_language_summary

    header = (
        "[Resolved DB filters — copy each filter expression verbatim into the SQL WHERE clause]"
    )
    join_block = _build_join_context_block()
    context = join_block + "\n\n" + header + "\n" + "\n".join(context_lines)
    return f"{context}\n\n{plan.natural_language_summary}"
