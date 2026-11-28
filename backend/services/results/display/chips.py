"""
services/results/display/chips.py
---------------------------------
Describe the conditions of a generated query as filter chips, for display
only: nothing here changes what a search finds.

The chips are read from the SQL that actually runs, not from the language
model's plan: each top-level AND condition of WHERE, and of HAVING (the
subject count), becomes one chip. A typical condition is "a coded value OR
words in the dataset text":

    (EXISTS(... p2.diagnosis IN ('parkinsons_disease')) OR
     d.search_tsv @@ to_tsquery('english', 'parkinson:AB | parkinsons:AB'))

which becomes {field: "Diagnosis", label: "Parkinson's disease",
text_words: ["parkinson", "parkinsons"]}.

Chip ids are positions ("w0", "w1", "h0") in the query they were read from;
they are only meaningful together with that query.

A topic condition (LLM_preprocessor/preprocess.py, _topic_condition) gives
two chips: the subject ("Topic · social conflict") and, when the condition
excludes other meanings of the word, those meanings ("Not meant · stroop,
flanker +5") with the id of the topic plus "n" ("w0n").
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from sqlglot import exp

from routers.schemas import ChipKind, QueryFilterSchema as Chip
from services import asked, sql_tree
from services.fulltext.text_search import Clause, read_clauses
from services.results.display.values import (
    Age, Codes, Counted, Listed, Ordering, Patients, Phrases, Plain, Quantity, Size, Value,
    PHRASES_SHOWN, capitalised, words_of,
)
from services.schema import tables as _tables
from services.schema.tables import About
from services.translate.sql_rewriter import FALLBACK_SQL


log = logging.getLogger(__name__)


_COUNTED = {"subject": "Subjects", "participant_id": "Subjects", "session": "Sessions", "run": "Runs"}


# ── One condition -> one chip ─────────────────────────────────────────────────

def _venue_words(node) -> List[str]:
    """README words for a condition's recorded fields: the asked sex ("women"),
    a scanner make or model ("siemens"), a field strength ("7T", also when
    the chip is labelled by the modality it shares a condition with: the
    page and highlight_field_strengths read it as a strength). Age words
    come with the age chip."""
    from services.uncertain import scanner as tier_scanner
    from services.uncertain.fields import sex as tier_sex
    words: List[str] = []
    sex = asked.sex(node)
    if sex:
        words.extend(tier_sex.text_words(sex))
    words.extend(f"{strength:g}T" for strength in asked.field_strengths(node))
    for key in ("Manufacturer", "ManufacturerModelName"):
        values = sql_tree.compared_values(node, json_key=key)
        if values:
            words.extend(tier_scanner.sidecar_words(key, values))
    return words


def describe_condition(clause: Clause) -> Chip:
    chip_id = clause.id
    node, negated = asked.unwrapped(clause.node)
    clause = clause.part(node)
    phrases = asked.searched_phrases(clause)

    # Word forms a search is limited to ("ageing", "aging": not every "age").
    # They are words of the text too, so the page can mark them.
    forms = clause.guard_forms()
    searched = phrases + [form for form in forms if form not in phrases]
    # Words of the text venues the uncertain tier adds for fields that can be
    # missing (services/uncertain/tier.py): the page marks them.
    for word in _venue_words(node):
        if word not in searched:
            searched.append(word)
    forms_note = f"Only these word forms count: {', '.join(forms[:8])}." if forms else ""

    # `detail` explains the condition. The lists the chip carries (values,
    # text_words) are not repeated in it: the page adds them to the tooltip.
    def chip(kind: ChipKind, value: Value, detail: str = "", *, field_name: str = "") -> Chip:
        return Chip(id=chip_id, kind=kind, field=field_name, **value.format().chip_fields(),
                    negated=negated, text_words=searched,
                    detail=" ".join(filter(None, (detail, forms_note))))

    def coded_tasks(names: List[str]) -> str:
        """The task codes behind a label of words, which the chip lists nowhere else."""
        if not names:
            return ""
        listed = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
        return f"Coded task: {listed}." if len(names) == 1 else f"Any of {len(names)} coded tasks: {listed}."

    # The search words are lower case; these are names ("New Mexico").
    elsewhere = [name.title() for name in asked.excluded_places(clause)]
    if elsewhere:
        # Its own wording: the generic chip would read "Location: Mexico" a
        # second time, with nothing about what the condition leaves out.
        places = ", ".join(sql_tree.compared_values(node, asked.PLACE_COLUMNS)[:3])
        return Chip(
            id=chip_id, kind=ChipKind.COLUMN, field="Location", **Listed(elsewhere).format().chip_fields(),
            negated=True,
            detail=(f"Text that names these places does not count as {places}; a dataset "
                    "with a site there still does."))

    patients = asked.patients(node)
    if patients:
        # Not "subjects": the result cards would emphasise the dataset's subject count.
        # Counting all subjects, the chip said "Subjects: ≥ 25" for datasets with 88.
        minimum, codes = patients
        return chip(ChipKind.COUNT, Patients(minimum, codes),
                    "Participants recorded with the diagnosis in the dataset's participant list.",
                    field_name="Patients")

    count = asked.count(node)
    if count:
        counted = _COUNTED[count.column]
        if counted == ChipKind.SUBJECTS.named:
            return chip(ChipKind.SUBJECTS, Counted(count))
        return chip(ChipKind.COUNT, Counted(count), field_name=counted)

    diagnoses = sql_tree.compared_values(node, ["diagnosis", "dx_status"])
    if diagnoses:
        return chip(ChipKind.DIAGNOSIS, Codes("diagnosis", diagnoses, shown=1, capitals=True),
                    "A diagnosis coded in the participant list.")

    if asked.is_topic(clause):
        names = Codes("task", sql_tree.compared_values(node, ["task"]), capitals=True).names()
        return chip(ChipKind.TOPIC, Plain(phrases[0] if phrases else (names[0] if names else "topic")),
                    " ".join(filter(None, ("One subject, its words close together.",
                                           coded_tasks(names)))))

    tasks = sql_tree.compared_values(node, ["task"])
    if tasks:
        # Task concepts expand to dozens of codes; the words read better.
        coded = Codes("task", tasks, shown=1, capitals=True)
        if phrases:
            return chip(ChipKind.TASK, Plain(phrases[0]), coded_tasks(coded.names()))
        return chip(ChipKind.TASK, coded, "A task coded in the dataset's files.")

    modalities = [name for column in ("datatype", "suffix")
                  for name in Codes(column, sql_tree.compared_values(node, [column])).names()]
    if modalities:
        return chip(ChipKind.MODALITY, Listed(modalities))

    age = asked.age_bounds(node)
    if age:
        # A README naming the age group is evidence too (the uncertain
        # tier's probable form, services/uncertain/tier.py): its
        # words are what the page marks for a match on recorded ages.
        from services.uncertain.fields.age import age_group_words
        group_words = age_group_words(*asked.age_range(node))
        searched.extend(w for w in group_words if w not in searched)
        return chip(ChipKind.AGE, Age(age))

    species = asked.species(node)
    if species is not None:
        return chip(ChipKind.COLUMN, Plain(" or ".join(species) or "any animal"), field_name="Species")

    sexes = sql_tree.compared_values(node, ["sex"])
    if sexes:
        from participant_codes import sex_code   # RAG/, the vocabulary's spellings
        return chip(ChipKind.SEX, Codes("sex", [sex_code(s) or s for s in sexes]))

    hands = sql_tree.compared_values(node, ["handedness"])
    if hands:
        # Without this the chip showed the raw SQL ("Condition · EXISTS(SELECT 1 …").
        return chip(ChipKind.COLUMN, Codes("handedness", [h.lower() for h in hands]),
                    field_name="Handedness")

    size = asked.size_bounds(node)
    if size:
        return chip(ChipKind.COLUMN, Size(size), field_name="Size")

    json_key = asked.json_key(node)
    if json_key:
        # 'MagneticFieldStrength' -> 'Magnetic field strength'
        key, bounds = json_key
        field_name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", key).capitalize()
        return chip(ChipKind.COLUMN, Quantity(bounds) if bounds else Plain("recorded"),
                    field_name=field_name)

    for column, field_name in _tables.COLUMN_SHOWN_AS.items():
        values = sql_tree.compared_values(node, [column])
        if values:
            section = _tables.COLUMN_VOCABULARY.get(column)
            value = Codes(section, [v.lower() for v in values]) if section else Listed(values)
            return chip(ChipKind.COLUMN, value, field_name=field_name)

    if phrases:
        # The phrases of one text search are its alternatives ('drawings | art');
        # only phrases from searches that must all match are joined by "+".
        one_search = len(clause.queries(negated=False)) == 1
        joiner = " or " if (isinstance(node, exp.Or) or one_search) else " + "
        return chip(ChipKind.TEXT, Phrases(phrases, joiner, PHRASES_SHOWN))

    return plain_condition(node, chip_id, negated)


# ── Any other condition -> words, never SQL ───────────────────────────────────
# A chip showed "Condition · EXISTS(SELECT 1 FROM bids_participants AS p2 WHE…"
# (2026-11-20). The exact condition stays under "Show generated SQL".

_ABOUT_WORDS = {
    About.DATASET: "Dataset", About.PARTICIPANTS: "Participants", About.FILES: "Files",
    About.DERIVATIVES: "Processed data", About.PAPERS: "Papers", About.SITES: "Sites",
    About.WORD_FORMS: "Word forms",
}
_TABLE_WORDS = {table: _ABOUT_WORDS[about] for table, about in _tables.TABLE_ABOUT.items()}
_PLAIN_VALUE_RE = re.compile(r"^[\w][\w .,'()/+-]{0,40}$")


def plain_condition(node, chip_id: str, negated: bool = False) -> Chip:
    """The chip in words for a condition no other chip describes.

    Field: the table the condition looks at ("Participants", "Processed data"),
    "Dataset" for the dataset's own columns. Label: the plain values it compares
    with ("fmriprep"), else "not recorded" for an IS NULL test, else the columns
    it reads ("handedness"). No SQL text in any of the three.
    """
    found = asked.parts(node)
    fields = [_TABLE_WORDS.get(t.lower(), words_of(t).capitalize()) for t in found.tables]
    field_name = next((f for f in fields if f != "Dataset"), fields[0] if fields else "Dataset")

    columns = [words_of(name) for name in found.columns + found.json_keys]
    columns = [c for c in dict.fromkeys(columns) if c]

    values = [v for v in found.literals
              if _PLAIN_VALUE_RE.match(v) and not re.search(r"\b(select|from|where)\b", v, re.I)]
    values = [capitalised(v.replace("_", " ")) if "_" in v else v for v in values]

    def named(column: str) -> str:
        return column.upper() if column in ("doi", "dicom", "url") else column[:1].upper() + column[1:]

    value: Value
    if values:
        value = Listed(values)
    elif found.tests_null:
        value = Plain(f"{named(columns[0])} not recorded" if columns else "Not recorded")
    elif columns:
        value = Listed([named(c) for c in columns], ", ")
    else:
        value = Plain("Other condition")
    about = f": {', '.join(columns[:6])}" if columns else ""
    detail = (f"A condition on the {field_name.lower()}{about}. "
              "It has no summary of its own; “Show generated SQL” has it in full.")
    return Chip(id=chip_id, kind=ChipKind.OTHER, field=field_name, **value.format().chip_fields(),
                negated=negated, detail=detail)


# ── Sort order and limit -> one chip ──────────────────────────────────────────

SORT_CHIP_ID = "o0"

def describe_order(tree) -> Optional[Chip]:
    """('Sorted', 'most participants first · top 1') for the query's ORDER BY and LIMIT.

    "find the largest dataset in openneuro" shows only one dataset; without
    this chip nothing said it was the top one of a ranking.
    """
    ordering = Ordering(sql_tree.order_keys(tree), sql_tree.row_limit(tree))
    orders = ordering.orders()
    if not orders and ordering.top is None:
        return None
    return Chip(id=SORT_CHIP_ID, kind=ChipKind.SORT, field="Sorted" if orders else "Limit",
                **ordering.format().chip_fields())


def describe_exclusion(clause: Clause) -> Optional[Chip]:
    """The meanings a topic condition excludes, as a chip of their own; None if none."""
    chip_id, node = clause.id, asked.without_parentheses(clause.node)
    clause = clause.part(node)
    if isinstance(node, exp.Not) or not asked.is_topic(clause):
        return None
    excluded = asked.excluded_phrases(clause)
    if not excluded:
        return None
    return Chip(
        id=chip_id + "n", kind=ChipKind.SENSE, **Listed(excluded, ", ", PHRASES_SHOWN).format().chip_fields(),
        detail=("Other meanings of the word, ruled out by the question. "
                "Datasets that match the topic only alongside these come after the others."),
    )


# ── Whole query ───────────────────────────────────────────────────────────────

def where_conditions(
    sql: Optional[str],
) -> Tuple[Optional[str], List[Tuple[Chip, Clause]]]:
    """The dataset alias and each top-level WHERE condition with its chip.

    For services/results/display/match_reasons.py, which evaluates a condition's
    branches per dataset. HAVING conditions (counts) are left out: the card
    shows the counts themselves. (None, []) when the SQL cannot be read.
    """
    tree = sql_tree.parse_select(sql or "")
    if tree is None:
        return None, []
    alias = asked.dataset_alias(tree)
    found = []
    for clause in read_clauses(tree):
        if not clause.in_where:
            continue
        try:
            found.append((describe_condition(clause), clause))
        except ValueError as exc:  # a number that is not one; the other conditions still count
            log.warning("chips: could not describe %s (%s)", clause.node, exc)
    return alias, found


def describe(sql: Optional[str]) -> List[Chip]:
    """One chip per top-level condition; [] when the SQL cannot be read, and
    for the fallback sample, which sorts and limits but asks for nothing."""
    tree = sql_tree.parse_select(sql or "")
    if tree is None or sql == FALLBACK_SQL:
        return []
    chips = []
    for clause in read_clauses(tree):
        try:
            chips.append(describe_condition(clause))
            exclusion = describe_exclusion(clause)
            if exclusion is not None:
                chips.append(exclusion)
        except ValueError as exc:  # a number that is not one; the other chips still show
            log.warning("chips: could not describe %s (%s)", clause.node, exc)
            chips.append(plain_condition(clause.node, clause.id))
    sort = describe_order(tree)
    if sort is not None:
        chips.append(sort)
    return chips


def for_sql(sql: Optional[str]) -> List[Chip]:
    """The chips of *sql* as the API sends them."""
    return describe(sql)
