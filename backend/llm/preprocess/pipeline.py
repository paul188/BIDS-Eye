"""The full pipeline: question → QueryPlan → RAG → AugmentedQueryPlan (run_pipeline)."""
from __future__ import annotations

import functools
import logging
import re
from typing import Any, Collection, Dict, List, Tuple
import reference_vocabulary as _reference_vocabulary
from services.schema.fields import FILE_FIELDS

from . import catalogue as _catalogue
from .plan import AugmentedQueryPlan, ParticipantGroup, QueryPlan, RAGField, ScanRequirement
from .intent import create_query_plan
from .prompt_context import count_stem_forms
from .rag import build_rag_requests
from .topics import _count_topic_matches, _detach_topic_terms, _keep_ambiguity_topics, check_description_alternatives, check_scan_term_alternatives, resolve_topics
from .augment import augment_with_rag

log = logging.getLogger("preprocess")


def _decide_allowed_forms(stem_counts: Dict[str, Any],
                          same_meaning: List[Tuple[str, List[str]]]) -> Dict[str, List[str]]:
    """The forms each colliding word of the question must match in ({} for none)."""
    try:
        import stem_collisions
    except ImportError as exc:
        log.warning("Allowed forms not decided (%s) — searching by stem", exc)
        return {}
    decided = {word: list(forms)
               for word, forms in stem_collisions.decide(stem_counts, same_meaning).items()}
    if decided:
        log.info("Allowed forms: %s", decided)
    return decided


_TEXT_MATCH_WORDS_RE = re.compile(r"text_match\(\s*'([^']*)'", re.IGNORECASE)


def _log_undecided_words(augmented_question: str, decided: Dict[str, List[str]]) -> None:
    """Log searched words with colliding forms nobody judged against them.

    These come from the vocabulary or from the model's own wording, not from
    the question, so the model was not asked about them. They are searched by
    stem. The log is the material for backend/vocabulary/stem_collisions.yaml.
    """
    try:
        import stem_collisions
        word_form_counts = _catalogue.word_form_counts()
    except ImportError:  # outside the backend
        return
    searched = list(dict.fromkeys(
        word for terms in _TEXT_MATCH_WORDS_RE.findall(augmented_question)
        for word in re.findall(r"[a-z]+", terms.lower())))
    open_words = [w for w in searched if w not in decided]
    # {} when the catalogue cannot answer (lookup catches that).
    undecided = [e for e in word_form_counts.lookup(open_words).values()
                 if stem_collisions.open_forms(e)]
    for item in undecided:
        log.info("Searched by stem although it collides: %s (%d datasets; own forms %d; "
                 "nobody judged %s)",
                 item.word, item.stem_docs, stem_collisions.family_docs(item),
                 ", ".join(f"{form} {item.count(form)}" for form in stem_collisions.open_forms(item)[:5]))


# Words that say a name is a person: "datasets by Huntington", "the Wilson lab".
_AUTHOR_CUE_RE = re.compile(
    r"\b(?:by|authors?|authored|lab|labs|laborator(?:y|ies)|team|et\s+al|published|"
    r"uploaded|contributed|investigators?|researchers?|scientists?|pi)\b",
    re.IGNORECASE,
)


def _names_that_are_conditions(plan: QueryPlan, question: str, retriever: Any) -> None:
    """Move a name the model took for an author to the disease or task it names.

    "rasmussen" alone was read as an author, and the name lookup then picked
    "Rasmus Sinn" (2026-11-18). A name that is exactly a known disease or
    task (curated vocabulary, or MeSH / Cognitive Atlas through
    backend/vocabulary/reference_vocabulary.py) is searched as that disease or task, unless
    the question says it is a person ("by", "lab", "authors" ...) and the
    catalogue has an author of that name.
    """
    if retriever is None or not plan.author_names or not hasattr(retriever, "known_concept"):
        return
    says_person = bool(_AUTHOR_CUE_RE.search(question or ""))
    kept: List[str] = []
    moved: List[Tuple[str, str, str]] = []          # (name, field, label)
    for name in plan.author_names:
        concept = retriever.known_concept(name)
        if concept is None and not retriever.retrieve_for_field("author", name):
            # No such author: a misspelt disease surname ("rasmussehn").
            surname = _reference_vocabulary.near_surname(name)
            if surname:
                concept = retriever.known_concept(surname)
                name = surname if concept else name
        if concept is None or (says_person and retriever.retrieve_for_field("author", name)):
            kept.append(name)
            continue
        field, label = concept
        if field == "diagnosis":
            if not plan.groups:
                plan.groups.append(ParticipantGroup(group_name="patient_group"))
            if name not in plan.groups[0].diagnosis_terms:
                plan.groups[0].diagnosis_terms.append(name)
        else:
            plan.scan_requirements.append(ScanRequirement(scan_terms=[name]))
        moved.append((name, field, label))
        log.info("Name %r is the %s %r, not an author", name, field, label)
    if not moved:
        return
    plan.author_names = kept
    readings = "; ".join(
        f"'{name}' is the {'diagnosis' if field == 'diagnosis' else 'task'} {label}, not an author"
        for name, field, label in moved
    )
    summary = plan.natural_language_summary or ""
    if not kept and re.search(r"\bauthor", summary, re.IGNORECASE):
        parts = [f"participants with {label}" if field == "diagnosis"
                 else f"the {label}" + ("" if re.search(r"\b(task|test|paradigm)\b", label, re.I)
                                       else " task")
                 for _name, field, label in moved]
        summary = "Find datasets with " + " and ".join(parts) + "."
    plan.natural_language_summary = f"{summary} ({readings}.)".strip()


def run_pipeline(
    query: str,
    retriever: Any = None,
    available_columns: Collection[str] = (),
) -> AugmentedQueryPlan:
    """
    End-to-end: parse query → RAG lookup → return AugmentedQueryPlan.

    `retriever` should be a MetadataRetriever (backend/vocabulary/retriever.py).
    If None, the RAG step is skipped.

    Scan terms are resolved across datatype + suffix + task simultaneously
    using retrieve_scan_terms() when available, falling back to per-field
    retrieve_for_field_multi() calls otherwise.
    """
    # Words whose stem also matches unrelated words ("ageing" and "age"): the
    # model sees the catalogue's forms and picks the ones meant (backend/vocabulary/stem_collisions.py).
    stem_counts = count_stem_forms(query)
    if stem_counts:
        same_meaning: List[Tuple[str, List[str]]] = []
        plan = create_query_plan(query, stem_counts=stem_counts, same_meaning=same_meaning)
        decided = _decide_allowed_forms(stem_counts, same_meaning)
    else:
        plan = create_query_plan(query)
        decided = {}
    _names_that_are_conditions(plan, query, retriever)
    _keep_ambiguity_topics(plan)
    _detach_topic_terms(plan)
    if decided:
        count_matches = functools.partial(_count_topic_matches, allowed_forms=decided)
        topics = resolve_topics(plan, retriever, count_matches)
        check_description_alternatives(plan, count_matches)
        check_scan_term_alternatives(plan, count_matches)
    else:
        topics = resolve_topics(plan, retriever)
        check_description_alternatives(plan)
        check_scan_term_alternatives(plan)

    def finished(augmented: AugmentedQueryPlan) -> AugmentedQueryPlan:
        augmented.allowed_forms = decided
        if stem_counts:
            _log_undecided_words(augmented.augmented_question, decided)
        return augmented

    if retriever is None:
        return finished(augment_with_rag(
            plan, {}, available_columns=available_columns, topics=topics))

    rag_requests = build_rag_requests(plan)
    rag_results: Dict[str, Dict[str, List[str]]] = {}
    # {field: {code: best_score}} — captured when the retriever exposes the
    # *_scored methods; drives relevance ranking downstream.
    rag_scores: Dict[str, Dict[str, float]] = {}

    def _record_scores(field: str, pairs: List[tuple]) -> None:
        cur = rag_scores.setdefault(field, {})
        for code, score in pairs:
            if score > cur.get(code, 0.0):
                cur[code] = score

    for req in rag_requests:
        if req.field == RAGField.SCAN:
            # Multi-field lookup: datatype + suffix + task in one pass
            if hasattr(retriever, "retrieve_scan_terms_scored"):
                scan_scored = retriever.retrieve_scan_terms_scored(req.terms)
                for field, term_map in scan_scored.items():
                    for term, pairs in term_map.items():
                        rag_results.setdefault(field, {})[term] = [c for c, _ in pairs]
                        _record_scores(field, pairs)
                continue
            scan_resolved: Dict[str, Dict[str, List[str]]]
            if hasattr(retriever, "retrieve_scan_terms"):
                scan_resolved = retriever.retrieve_scan_terms(req.terms)
            else:
                # Fallback for retrievers that don't yet have retrieve_scan_terms
                scan_resolved = {}
                for field in FILE_FIELDS:
                    if hasattr(retriever, "retrieve_for_field_multi"):
                        tm = retriever.retrieve_for_field_multi(field, req.terms)
                    else:
                        cat_key = field + "s"
                        tm = {}
                        for term in req.terms:
                            hints = retriever.retrieve(f"{field}: {term}")
                            codes = hints.get(cat_key, [])[:10]
                            if codes:
                                tm[term] = codes
                    if tm:
                        scan_resolved[field] = tm
            # Spread per-field results into rag_results
            for field, term_map in scan_resolved.items():
                if term_map:
                    rag_results.setdefault(field, {}).update(term_map)

        else:
            # Single-field lookup
            if hasattr(retriever, "retrieve_for_field_multi_scored"):
                scored_map = retriever.retrieve_for_field_multi_scored(
                    req.field.value, req.terms
                )
                term_map = {}
                for term, pairs in scored_map.items():
                    term_map[term] = [c for c, _ in pairs]
                    _record_scores(req.field.value, pairs)
                if term_map:
                    rag_results[req.field.value] = term_map
                continue
            if hasattr(retriever, "retrieve_for_field_multi"):
                term_map = retriever.retrieve_for_field_multi(
                    req.field.value, req.terms
                )
            else:
                cat_key = req.field.value + "s"
                term_map = {}
                for term in req.terms:
                    hints = retriever.retrieve(f"{req.field.value}: {term}")
                    codes = hints.get(cat_key, [])[:10]
                    if codes:
                        term_map[term] = codes
            if term_map:
                rag_results[req.field.value] = term_map

    # DB back-check: remove codes that no dataset has and surface them as
    # DB_MISS warnings so the SQL model omits those filters. In the API the
    # backend's cache of coded values answers (catalogue.verify_codes).
    db_miss_terms: Dict[str, List[str]] = {}
    for field in list(rag_results.keys()):
        term_map = rag_results[field]
        new_term_map: Dict[str, List[str]] = {}
        for term, codes in term_map.items():
            if field == RAGField.DIAGNOSIS.value:
                # A diagnosis keeps the codes no participant row carries yet:
                # they cost nothing in the IN list, and the first code names
                # the diagnosis, so the diagnosis line searches its words in
                # the dataset titles and labels. With them dropped, "TLE"
                # (tle and mtle have no rows) was read as its narrower
                # hippocampal sclerosis only (2026-11-23). As a DB MISS the
                # SQL model was told to use the parent code: "frontotemporal
                # dementia" became every dementia (2026-11-17).
                new_term_map[term] = codes
                continue
            live, miss = _catalogue.verify_codes(field, codes, retriever)
            if miss:
                db_miss_terms.setdefault(field, []).extend(miss)
            if live:
                new_term_map[term] = live
            elif not live and codes:
                # All codes for this term are DB misses — term is unresolvable
                db_miss_terms.setdefault(field, []).extend(codes)
        rag_results[field] = new_term_map

    # Collect terms that were requested but returned no codes — surface as warnings
    # so the SQL model does not fall back to ILIKE on canonical-code columns.
    unresolved: Dict[str, List[str]] = {}
    for req in rag_requests:
        if req.field == RAGField.SCAN:
            resolved_scan: set[str] = set()
            for sf in FILE_FIELDS:
                resolved_scan.update(rag_results.get(sf, {}).keys())
            missed = [t for t in req.terms if t not in resolved_scan]
            if missed:
                unresolved["scan"] = missed
        else:
            resolved_terms = set(rag_results.get(req.field.value, {}).keys())
            missed = [t for t in req.terms if t not in resolved_terms]
            if missed:
                unresolved[req.field.value] = missed

    return finished(augment_with_rag(
        plan, rag_results, unresolved_terms=unresolved,
        db_miss_terms=db_miss_terms, scores=rag_scores,
        available_columns=available_columns, topics=topics,
    ))
