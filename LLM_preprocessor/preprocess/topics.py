"""Topics: a qualified subject, searched as one unit; description alternatives."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from text_terms import (
    MODE_NEAR as _MODE_NEAR,
    MODE_SHORT as _MODE_SHORT,
    format_text_match as _format_text_match,
    split_words as _split_words,
)
import senses as _senses

from . import catalogue as _catalogue
from .plan import QueryPlan, ResolvedTopic, ScanRequirement, Topic

log = logging.getLogger("preprocess")


# Alternatives taken from the language model, and searched in total.
_TOPIC_MAX_LLM_ALTERNATIVES = 8


_TOPIC_MAX_ALTERNATIVES = 16


_TOPIC_MAX_CODES = 20


def _topic_max_matches() -> int:
    """An alternative from the language model that more datasets match is too broad.

    90 is about 5% of the 1,721 datasets of 2026-11. Alternatives from
    RAG/senses.yaml are curated and exempt.
    """
    try:
        return int(os.getenv("TOPIC_MAX_ALTERNATIVE_MATCHES", "90"))
    except ValueError:
        return 90


def _stem(part: str) -> str:
    if len(part) > 4 and part.endswith("ies"):
        return part[:-3] + "y"
    if len(part) > 3 and part.endswith("s") and not part.endswith("ss"):
        return part[:-1]
    return part


def _word_set(text: str) -> frozenset:
    """Content words of *text*, plural folded: "social conflicts" → {social, conflict}."""
    return frozenset(_stem(part) for word in _split_words(text or "") for part in word)


def _count_topic_matches(alternatives: List[str],
                         exact_forms: Optional[Dict[str, List[str]]] = None) -> Optional[List[int]]:
    """Datasets matching each alternative, from the backend's full-text search; None if unknown.

    *exact_forms*: word forms decided for this question; such a word is
    counted in those forms, as it will be searched.
    """
    try:
        tsquery = _catalogue.tsquery()  # backend/services; absent outside the API
    except ImportError:
        return None
    # None when the catalogue cannot answer (count_matches catches that).
    if exact_forms:
        return tsquery.count_matches(alternatives, _MODE_NEAR, exact_forms)
    return tsquery.count_matches(alternatives, _MODE_NEAR)


def _is_ambiguity_topic(topic: Topic) -> bool:
    """A topic the search treats as one: its head has senses, or the user abbreviated it.

    On the 167 development questions of 2026-11-14 the language model also
    made topics of plain subjects ("sign language processing", "gambling
    choices") and listed unwanted meanings for nearly all of them; searching
    those as topics lost recall (47% → 38% on the 24 such questions). The
    gains on the ambiguity questions came from heads in RAG/senses.yaml and
    from abbreviations ("emo").
    """
    phrase = topic.phrase or topic.typed
    if _senses.senses_of(topic.head or phrase):
        return True
    return bool(topic.typed) and bool(_word_set(topic.typed)) \
        and _word_set(topic.typed) != _word_set(phrase)


def _keep_ambiguity_topics(plan: QueryPlan) -> None:
    """Hand every other topic back to the scan-term path it had before topics existed."""
    kept: List[Topic] = []
    for topic in plan.topics:
        if _is_ambiguity_topic(topic):
            kept.append(topic)
            continue
        term = topic.phrase or topic.typed
        if term and not any(term in r.scan_terms for r in plan.scan_requirements):
            plan.scan_requirements.append(ScanRequirement(scan_terms=[term]))
    plan.topics = kept


def _detach_topic_terms(plan: QueryPlan) -> None:
    """Remove scan terms and description words that only repeat a topic.

    The topic searches them as one unit; a scan term "conflict" next to the
    topic "social conflict" would bring the head word back on its own.
    """
    names = [_word_set(name) for t in plan.topics for name in (t.phrase, t.typed) if name]
    names = [n for n in names if n]
    if not names:
        return

    def covered(term: str) -> bool:
        words = _word_set(term)
        return bool(words) and any(words <= name for name in names)

    for requirement in plan.scan_requirements:
        requirement.scan_terms = [t for t in requirement.scan_terms if not covered(t)]
    plan.scan_requirements = [r for r in plan.scan_requirements if r.scan_terms]
    if plan.description_search and covered(plan.description_search):
        plan.description_search = None


_DESCRIPTION_MAX_ALTERNATIVES = 8


def _checked_alternatives(own_text: str, alternatives: List[str],
                          count_matches: Any) -> "tuple[List[str], Dict[str, str]]":
    """The alternatives worth searching for *own_text*, and the dropped ones with the reason.

    Dropped: repeats of the user's own words, alternatives no dataset uses,
    and those too many datasets use to say anything (_topic_max_matches).
    Without counts (outside the backend) they are kept unchecked.
    """
    dropped: Dict[str, str] = {}
    own = _word_set(own_text or "")
    if not own:
        return [], dropped
    kept: List[str] = []
    seen: set = set()
    for alternative in alternatives[:_DESCRIPTION_MAX_ALTERNATIVES]:
        words = _word_set(alternative)
        if not words or words in seen or words == own:
            continue
        seen.add(words)
        kept.append(alternative)
    counts = count_matches(kept) if (kept and count_matches) else None
    if counts is not None and len(counts) == len(kept):
        limit = _topic_max_matches()
        checked = []
        for alternative, count in zip(kept, counts):
            if count == 0:
                dropped[alternative] = "no dataset uses it"
            elif count > limit:
                dropped[alternative] = f"too common ({count} datasets)"
            else:
                checked.append(alternative)
        kept = checked
    return kept, dropped


def check_description_alternatives(plan: QueryPlan,
                                   count_matches: Any = _count_topic_matches) -> Dict[str, str]:
    """Keep the other words for description_search that are worth searching.

    "dead people" found nothing: the datasets say "post-mortem" and "ex vivo"
    (2026-11-15). The language model lists such words in
    description_alternatives; like a topic's alternatives they are dropped
    when they only repeat the user's words, match no dataset, or match too
    many to say anything (_topic_max_matches). Without counts (outside the
    backend) they are kept unchecked.

    Changes plan.description_alternatives in place; returns what was dropped
    and why.
    """
    kept, dropped = _checked_alternatives(plan.description_search or "", plan.description_alternatives,
                                          count_matches)
    if kept or dropped:
        log.info("Description search %r: alternatives=%s dropped=%s",
                 plan.description_search, kept, dropped)
    plan.description_alternatives = kept
    return dropped


def check_scan_term_alternatives(plan: QueryPlan,
                                 count_matches: Any = _count_topic_matches) -> Dict[str, str]:
    """Keep the other words for each scan term that are worth searching.

    Datasets name a task family by its instances: an ERP study says "oddball"
    or "P300", a natural-image study "ImageNet" or "photographs". The model
    lists such words per scan term (scan_term_alternatives); they are checked
    like description_alternatives. Entries for a term that is no longer a
    scan term are dropped.

    Changes plan.scan_term_alternatives in place; returns what was dropped.
    """
    dropped: Dict[str, str] = {}
    scan_terms = {t.strip().lower() for r in plan.scan_requirements for t in r.scan_terms}
    kept_entries = []
    for entry in plan.scan_term_alternatives:
        if entry.term.strip().lower() not in scan_terms:
            continue
        kept, gone = _checked_alternatives(entry.term, entry.alternatives, count_matches)
        dropped.update({f"{entry.term}: {k}": v for k, v in gone.items()})
        if kept or gone:
            log.info("Scan term %r: alternatives=%s dropped=%s", entry.term, kept, gone)
        if kept:
            entry.alternatives = kept
            kept_entries.append(entry)
    plan.scan_term_alternatives = kept_entries
    return dropped


def _topic_codes(
    topic: Topic, pick: Optional["_senses.SensePick"], not_meant: List[str], retriever: Any,
) -> Dict[str, float]:
    """Task codes for the whole phrase, never for the head word alone.

    The vocabulary also matches a phrase through one of its words: "cortical
    inhibition" reached the construct inhibition and with it stop-signal and
    go/no-go tasks. With qualifiers: when the phrase gives everything the
    head word alone gives, those codes came through the head and are
    dropped; what the qualifier added stays. A phrase that gives fewer or
    other codes than the head ("spatial memory" vs "memory") was narrowed by
    its qualifier and keeps them. The picked sense's own codes come after.
    """
    codes: Dict[str, float] = {}
    phrase = topic.phrase or topic.typed

    def lookup(term: str) -> Dict[str, float]:
        if retriever is None or not hasattr(retriever, "retrieve_for_field_multi_scored"):
            return {}
        # In memory (the vocabulary and, when loaded, the embedding models):
        # nothing here fails while the code is right.
        found = retriever.retrieve_for_field_multi_scored("task", [term])
        return {code: float(score) for pairs in found.values() for code, score in pairs}

    codes = lookup(phrase)
    head = (topic.head or "").strip()
    if codes and topic.qualifiers and head and _word_set(head) != _word_set(phrase):
        head_codes = lookup(head)
        if head_codes and set(head_codes) <= set(codes):
            codes = {code: score for code, score in codes.items() if code not in head_codes}
    if pick is not None:
        for code in pick.tasks:
            codes.setdefault(code, 1.0)
    excluded = set(pick.excluded_tasks) if pick is not None else set()
    # "simon" among the unwanted meanings removes simon_task.
    unwanted_words = {next(iter(w)) for w in map(_word_set, not_meant) if len(w) == 1}
    codes = {
        code: score for code, score in codes.items()
        if code not in excluded and not (set(code.lower().split("_")) & unwanted_words)
    }
    codes = dict(list(codes.items())[:_TOPIC_MAX_CODES])
    if codes:
        live, _miss = _catalogue.verify_codes("task", list(codes), retriever)
        codes = {code: codes[code] for code in live}
    return codes


def _topic_phrase_terms(topic: Topic) -> List[str]:
    """The user's own subject: the phrase, and the typed form when it differs ("emo")."""
    terms = [topic.phrase or topic.typed]
    if topic.typed and _word_set(topic.typed) and _word_set(topic.typed) != _word_set(terms[0]):
        terms.append(topic.typed)
    return [t for t in terms if t]


def _topic_condition(resolved: ResolvedTopic) -> Optional[str]:
    """One filter for a topic:

        EXISTS (task codes)
        OR text_match('<phrase>', 'near')
        OR ((text_match('<alternatives of several words>', 'near')
             OR text_match('<one-word alternatives>', 'short'))
            AND NOT text_match('<not meant>', 'near'))

    The phrase comes first: the uncertain tier relaxes only the first text
    search of a topic (services/fulltext/uncertain_tier.py, relax_text_condition).

    The phrase is the user's own words and needs no guard. An alternative can
    also occur in a text about another meaning ("conflict resolution task" in
    a Stroop study); such a dataset fails this condition and the uncertain
    tier keeps it as probable (services/fulltext/uncertain_tier.py).
    """
    parts: List[str] = []
    if resolved.codes:
        values = ", ".join(f"'{c}'" for c in resolved.codes)
        task = f"o2.task = {values}" if len(resolved.codes) == 1 else f"o2.task IN ({values})"
        objects = _catalogue.tables().objects.name
        parts.append(f"EXISTS (SELECT 1 FROM {objects} o2 WHERE o2.dataset_id = d.id AND {task})")
    phrase_match = _format_text_match(_topic_phrase_terms(resolved.topic), _MODE_NEAR)
    if phrase_match:
        parts.append(phrase_match)
    # A one-word alternative is searched in the short fields only: anywhere in
    # a README, "aggression" also matched "the defacing was quite aggressive"
    # and an aggression questionnaire. Several words must stand together.
    phrases = [a for a in resolved.alternatives if len(_split_words(a)) > 1]
    words = [a for a in resolved.alternatives if len(_split_words(a)) == 1]
    alternative_parts = [m for m in (_format_text_match(phrases, _MODE_NEAR),
                                     _format_text_match(words, _MODE_SHORT)) if m]
    alternative_match = None
    if len(alternative_parts) == 1:
        alternative_match = alternative_parts[0]
    elif alternative_parts:
        alternative_match = "(" + " OR ".join(alternative_parts) + ")"
    if alternative_match:
        guard = _format_text_match(resolved.not_meant, _MODE_NEAR)
        parts.append(f"({alternative_match} AND NOT {guard})" if guard else alternative_match)
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")"


def resolve_topic(
    topic: Topic,
    retriever: Any = None,
    count_matches: Any = _count_topic_matches,
) -> ResolvedTopic:
    """Check a topic against the sense inventory, the vocabulary and the catalogue.

    - RAG/senses.yaml: a qualifier that picks a sense of the head adds that
      sense's phrases and task codes and excludes the other senses.
    - Alternatives are dropped when they are only the head word or a
      qualifier, name an excluded meaning, match no dataset, or (when the
      language model proposed them) match too many to say anything.
    - Task codes come from the whole phrase and the picked sense.
    """
    phrase = (topic.phrase or topic.typed).strip()
    qualifiers = list(topic.qualifiers)
    pick = _senses.pick_sense(topic.head or phrase, qualifiers, phrase=phrase)
    not_meant = list(dict.fromkeys(
        [n for n in topic.not_meant if n.strip()] + (pick.not_meant if pick else [])))
    not_meant_sets = [w for w in map(_word_set, not_meant) if w]

    base = _word_set(topic.head)
    for qualifier in qualifiers:
        base |= _word_set(qualifier)
    own = {_word_set(t) for t in _topic_phrase_terms(topic)}

    dropped: Dict[str, str] = {}
    kept: List[tuple] = []          # (alternative, curated)
    seen: set = set()
    candidates = [(a, True) for a in (pick.alternatives if pick else [])]
    candidates += [(a, False) for a in topic.alternatives[:_TOPIC_MAX_LLM_ALTERNATIVES]]
    for alternative, curated in candidates:
        words = _word_set(alternative)
        if not words or words in seen or words in own:
            continue
        seen.add(words)
        # Never the head word alone; a qualifier alone ("social") only when
        # senses.yaml lists it as a phrase of the sense ("GABA").
        if words <= base and (not curated or words <= _word_set(topic.head)):
            dropped[alternative] = "only the head word or a qualifier"
        elif any(n <= words for n in not_meant_sets):
            dropped[alternative] = "names a meaning the search excludes"
        else:
            kept.append((alternative, curated))

    counts = count_matches([a for a, _ in kept]) if (kept and count_matches) else None
    if counts is not None and len(counts) == len(kept):
        limit = _topic_max_matches()
        checked = []
        for (alternative, curated), count in zip(kept, counts):
            if count == 0:
                dropped[alternative] = "no dataset uses it"
            elif count > limit and not curated:
                dropped[alternative] = f"too common ({count} datasets)"
            else:
                checked.append((alternative, curated))
        kept = checked

    codes = _topic_codes(topic, pick, not_meant, retriever)
    resolved = ResolvedTopic(
        topic=topic,
        sense=pick.sense.name if (pick and pick.sense) else None,
        codes=list(codes),
        scores=codes,
        alternatives=[a for a, _ in kept][:_TOPIC_MAX_ALTERNATIVES],
        dropped=dropped,
        not_meant=not_meant,
    )
    resolved.condition = _topic_condition(resolved)
    log.info(
        "Topic %r: sense=%s codes=%s alternatives=%s dropped=%s not_meant=%s",
        phrase, resolved.sense, resolved.codes, resolved.alternatives, dropped, not_meant,
    )
    return resolved


def resolve_topics(plan: QueryPlan, retriever: Any = None,
                   count_matches: Any = _count_topic_matches) -> List[ResolvedTopic]:
    return [resolve_topic(t, retriever, count_matches)
            for t in plan.topics if (t.phrase or t.typed).strip()]
