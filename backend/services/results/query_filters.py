"""
services/results/query_filters.py
---------------------------------
Describe the conditions of a generated query as filter chips, and drop the
ones the user removes.

The chips are read from the SQL that actually runs, not from the language
model's plan: each top-level AND condition of WHERE, and of HAVING (the
subject count), becomes one chip. A typical condition is "a coded value OR
words in the dataset text":

    (EXISTS(... p2.diagnosis IN ('parkinsons_disease')) OR
     d.search_tsv @@ to_tsquery('english', 'parkinson:AB | parkinsons:AB'))

which becomes {field: "Diagnosis", label: "Parkinson's disease",
text_words: ["parkinson", "parkinsons"]}.

Removing a chip deletes its condition from the query before the uncertain
tier is applied (services/fulltext/uncertain_tier.py), so a refined search runs
without another language-model call (routers/query.py, /query/refine).

Chip ids are positions ("w0", "w1", "h0") in the query they were read from;
they are only meaningful together with that query.

A topic condition (LLM_preprocessor/preprocess.py, _topic_condition) gives
two chips: the subject ("Topic · social conflict") and, when the condition
excludes other meanings of the word, those meanings ("Not meant · stroop
+5") with the id of the topic plus "n" ("w0n"). Removing the second chip
removes only the exclusion.
"""

from __future__ import annotations

import logging
import re
import dataclasses
from dataclasses import asdict, dataclass
from typing import FrozenSet, Iterable, List, Optional, Tuple, Union

try:  # sqlglot is a backend dependency; keep the module importable without it
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import SqlglotError
except ImportError:  # pragma: no cover
    sqlglot = None  # type: ignore[assignment]
    exp = None  # type: ignore[assignment]
    SqlglotError = ()  # type: ignore[assignment,misc]  # nothing is parsed, so nothing to catch

log = logging.getLogger(__name__)

# How many coded values a label lists before "+N".
_MAX_VALUES_SHOWN = 2

_MODALITY_LABELS = {
    "functional_mri": "fMRI",
    "fmri_bold": "fMRI",
    "bold_map": "fMRI",
    "anatomical_mri": "structural MRI",
    "t1_weighted_mri": "T1w MRI",
    "t2_weighted_mri": "T2w MRI",
    "electroencephalography": "EEG",
    "eeg": "EEG",
    "magnetoencephalography": "MEG",
    "meg": "MEG",
    "intracranial_eeg": "iEEG",
    "diffusion_mri": "diffusion MRI",
    "diffusion_mri_dwi": "diffusion MRI",
    "positron_emission_tomography": "PET",
    "pet": "PET",
    "fnirs": "fNIRS",
    "perfusion_asl": "ASL perfusion",
    "arterial_spin_labelling": "ASL perfusion",
    "magnetic_resonance_spectroscopy": "MR spectroscopy",
    "motion_capture": "motion capture",
    "motion_capture_data": "motion capture",
    "field_maps": "field maps",
    "behavioural_data": "behavioural data",
    "ecg": "ECG",
}

_ACRONYMS = {"adhd", "asd", "ptsd", "ocd", "mci", "als", "tbi", "mdd", "bpd", "hiv", "ms", "rbd", "pd"}
_POSSESSIVE_RE = re.compile(r"\b(alzheimer|parkinson|huntington|tourette|crohn|asperger)s\b", re.I)
_TS_LABEL_RE = re.compile(r":[A-Za-z*]+")
# Exact word distances written by 'near' text searches (text_search._near_query).
_TS_DISTANCE_RE = re.compile(r"<\d+>")
# Suffix of the chip id that stands for a topic's excluded meanings.
EXCLUSION_SUFFIX = "n"
_TS_WORD_RE = re.compile(r"[A-Za-z0-9][\w'-]*")

_FIELD_COLUMNS = {
    "authors": "Author",
    "institutions": "Institution",
    "license": "License",
    "funding": "Funding",
    "name": "Name",
    "doi": "DOI",
    "study_design": "Study design",
    "source_type": "Source",
    # bids_site_places (RAG/place_index.py): the sites' institutions as ROR
    # names them, and where they are.
    "institution": "Institution",
    "city": "Location",
    "region": "Location",
    "country": "Location",
}
# Stored values shown under a friendlier name.
_FIELD_VALUE_LABELS = {"openneuro": "OpenNeuro", "local": "Local upload"}


@dataclass
class QueryFilter:
    id: str                     # "w0" … for WHERE, "h0" … for HAVING
    kind: str                   # diagnosis, modality, task, age, sex, subjects, text, field, other
    field: str                  # what the chip is about, e.g. "Diagnosis"
    label: str                  # the asked value, e.g. "Parkinson's disease"
    negated: bool = False       # the dataset must NOT match
    # Words also searched in the dataset text (README, description). A chip
    # that is only text has kind "text" and these as its label.
    # dataclasses.field: the name 'field' is this class's own attribute.
    text_words: List[str] = dataclasses.field(default_factory=list)
    detail: str = ""            # longer description for a tooltip
    # Every value behind a label shortened to "+N", so the page can open the
    # full list; empty when the label already shows them all.
    values: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Reading values out of a condition ─────────────────────────────────────────

def _humanize_code(code: str) -> str:
    words = code.replace("_", " ").strip()
    if words.lower() in _ACRONYMS:
        return words.upper()
    words = _POSSESSIVE_RE.sub(lambda m: m.group(1) + "'s", words)
    return words[:1].upper() + words[1:]


def _short_list(values: List[str], joiner: str = ", ", shown: int = _MAX_VALUES_SHOWN) -> str:
    values = list(dict.fromkeys(v for v in values if v))
    if len(values) <= shown:
        return joiner.join(values)
    return f"{joiner.join(values[:shown])} +{len(values) - shown}"


def _hidden(values: List[str], shown: int = _MAX_VALUES_SHOWN) -> List[str]:
    """All of *values* when a label shows only *shown* of them; [] when none is hidden."""
    values = list(dict.fromkeys(v for v in values if v))
    return values if len(values) > shown else []


def _literals_for(node, column_names: Iterable[str]) -> List[str]:
    """String literals compared (=, IN, ILIKE, ANY) with any of these columns."""
    names = {n.lower() for n in column_names}
    values: List[str] = []
    for comparison in node.find_all(exp.EQ, exp.In, exp.ILike, exp.Like, exp.NEQ):
        columns = [c for c in comparison.find_all(exp.Column) if c.name.lower() in names]
        if not columns:
            continue
        for literal in comparison.find_all(exp.Literal):
            if literal.is_string:
                values.append(literal.this.strip("%"))
    return list(dict.fromkeys(values))


def _under_not(node, root) -> bool:
    """True if *node* sits inside a NOT below *root*."""
    current = node.parent
    while current is not None and current is not root:
        if isinstance(current, exp.Not):
            return True
        current = current.parent
    return False


# Text search configuration of a guard: the word forms a search requires
# (services/fulltext/text_search.py, exact word forms). Everything else is a search.
_GUARD_CONFIG = "simple"


def _tsqueries(node, negated: Optional[bool] = None, guards: bool = False) -> List[str]:
    """The query strings of every to_tsquery(...) in the condition.

    *negated*: None for all, False for those outside a NOT inside the
    condition, True for those inside one. *guards*: the guards instead of
    the searches.
    """
    queries = []
    for call in node.find_all(exp.Anonymous, exp.Func):
        name = (call.name if isinstance(call, exp.Anonymous) else call.sql_name()).lower()
        if name != "to_tsquery":
            continue
        if negated is not None and _under_not(call, node) != negated:
            continue
        literals = [e for e in call.expressions if isinstance(e, exp.Literal) and e.is_string]
        if not literals:
            continue
        is_guard = len(literals) > 1 and literals[0].this.lower() == _GUARD_CONFIG
        if is_guard == guards:
            queries.append(literals[-1].this)
    return queries


def guard_forms(node, negated: Optional[bool] = False) -> List[str]:
    """The word forms the condition's searches require, e.g. ['ageing', 'aging']."""
    forms = []
    for query in _tsqueries(node, negated=negated, guards=True):
        forms.extend(_TS_WORD_RE.findall(_TS_LABEL_RE.sub("", query)))
    return list(dict.fromkeys(forms))


def _outer_parentheses_removed(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        for index, char in enumerate(text):
            depth += char == "("
            depth -= char == ")"
            if depth == 0 and index < len(text) - 1:
                return text
        text = text[1:-1].strip()
    return text


def _split_top_level(query: str, sep: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for ch in query:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def tsquery_phrases(query: str) -> List[str]:
    """'(alzheimer:AB & disease:AB) | ad:AB' -> ['alzheimer disease', 'ad']."""
    query = _TS_LABEL_RE.sub("", query)
    phrases = []
    for alternative in _split_top_level(query, "|"):
        if _TS_DISTANCE_RE.search(alternative):
            # A 'near' term spells out every word order and gap; the first is the term.
            alternative = _split_top_level(_outer_parentheses_removed(alternative), "|")[0]
            alternative = _TS_DISTANCE_RE.sub(" ", alternative)
        words = _TS_WORD_RE.findall(alternative)
        if words:
            phrases.append(" ".join(words))
    return list(dict.fromkeys(phrases))


def _age_label(node) -> Optional[str]:
    """'18+', 'under 18', 'up to 12', '10–19' from the participant age comparisons."""
    low: Optional[Tuple[float, bool]] = None    # (value, inclusive)
    high: Optional[Tuple[float, bool]] = None

    def is_age(column) -> bool:
        return isinstance(column, exp.Column) and column.name.lower() == "age"

    for between in node.find_all(exp.Between):
        if is_age(between.this):
            try:
                low = (float(between.args["low"].this), True)
                high = (float(between.args["high"].this), True)
            except (KeyError, ValueError, AttributeError):
                pass
    ops = {exp.GT: (">", False), exp.GTE: (">", True), exp.LT: ("<", False), exp.LTE: ("<", True)}
    for cls, (direction, inclusive) in ops.items():
        for comparison in node.find_all(cls):
            left, right = comparison.this, comparison.expression
            if is_age(right) and isinstance(left, exp.Literal):  # 18 <= p2.age
                left, right = right, left
                direction = "<" if direction == ">" else ">"
            if not (is_age(left) and isinstance(right, exp.Literal) and not right.is_string):
                continue
            value = float(right.this)
            if direction == ">":
                low = (value, inclusive)
            else:
                high = (value, inclusive)
    if low is not None and low[0] <= 0:
        low = None  # "age > 0" only leaves out placeholder ages (RAG/criteria.py)
    if low is None and high is None:
        # Only the dataset-level bounds (d.age_min / d.age_max).
        columns = {c.name.lower() for c in node.find_all(exp.Column)}
        if not columns & {"age_min", "age_max"}:
            return None
        from services.fulltext.uncertain_tier import asked_age_range
        lo, hi = asked_age_range(node.sql(dialect="postgres"))
        low = (lo, True) if lo > 0 else None
        high = (hi, True) if hi < 120 else None
        if low is None and high is None:
            return None

    def fmt(value: float) -> str:
        if 0 < value < 1:
            return f"{round(value * 12)} months"
        return f"{value:g}"

    if low and high:
        return f"{fmt(low[0])}–{fmt(high[0])}"
    if low:
        return f"{fmt(low[0])}+" if low[1] else f"over {fmt(low[0])}"
    assert high is not None  # both None returned above
    return f"up to {fmt(high[0])}" if high[1] else f"under {fmt(high[0])}"


_SYMBOLS = {exp.GT: ">", exp.GTE: "≥", exp.LT: "<", exp.LTE: "≤", exp.EQ: "="} if exp else {}
_COUNTED = {"subject": "Subjects", "participant_id": "Subjects", "session": "Sessions", "run": "Runs"}


def _bound(node, matches) -> Optional[str]:
    """'≥ 40', '6.5–7.5' for comparisons whose left side satisfies *matches*."""
    for between in node.find_all(exp.Between):
        if matches(between.this):
            low, high = between.args.get("low"), between.args.get("high")
            if isinstance(low, exp.Literal) and isinstance(high, exp.Literal):
                return f"{low.this}–{high.this}"
    found: dict[str, str] = {}
    for cls, symbol in _SYMBOLS.items():
        for comparison in ([node] if isinstance(node, cls) else []) + list(node.find_all(cls)):
            if matches(comparison.this) and isinstance(comparison.expression, exp.Literal) \
                    and not comparison.expression.is_string:
                found.setdefault(symbol, comparison.expression.this)
    # A lower and an upper bound together are a range ("> 6.5 AND < 7.5"):
    # the first alone read as "> 6.5", and 9.4T and 14T were marked as asked.
    low = next((found[s] for s in (">", "≥") if s in found), None)
    high = next((found[s] for s in ("<", "≤") if s in found), None)
    if low is not None and high is not None:
        return f"{low}–{high}"
    for symbol in _SYMBOLS.values():
        if symbol in found:
            return f"{symbol} {found[symbol]}"
    return None


def _species_label(node) -> Optional[str]:
    """The species a d.species condition asks for: 'human', 'mouse', 'any animal'.

    Patterns as written by RAG/criteria.py; the label that says nothing
    ('other', 'n/a', 'unknown') is no species.
    """
    names: List[str] = []
    seen = False
    for match in node.find_all(exp.RegexpILike):
        column = match.this
        if not (isinstance(column, exp.Column) and column.name.lower() == "species"):
            continue
        pattern = str(getattr(match.expression, "this", "") or "")
        if "unknown" in pattern:
            continue
        seen = True
        if _under_not(match, node):
            continue
        first = re.search(r"\(([^|)]+)", pattern)
        if first:
            name = re.sub(r"\\[a-zA-Z]|.\?|[\^$*\\]", "", first.group(1)).strip()
            if name and name not in names:
                names.append(name)
    if names:
        return " or ".join(names)
    return "any animal" if seen else None
def _patients_label(node) -> Optional[str]:
    """'≥ 25 with Frontotemporal dementia' for "at least N patients with X" (RAG/cohort_count.py).

    That count is of the participants who have the diagnosis; the subject
    count chip called it "Subjects: ≥ 25" although the datasets had 88.
    """
    sql = node.sql(dialect="postgres")
    if "cohort_n" not in sql:
        return None
    try:
        import cohort_count
    except ImportError:  # the RAG package is not on the path
        return None
    parsed = cohort_count.parse_expanded(sql)
    if parsed is None:
        return None
    codes, minimum = parsed
    names = [_humanize_code(code) for code in codes]
    return f"≥ {minimum} with {names[0]}" + (f" +{len(names) - 1}" if len(names) > 1 else "")


def _count_chip(node) -> Optional[Tuple[str, str]]:
    """('Subjects', '≥ 40') for COUNT(DISTINCT o.subject) >= 40; also sessions, runs.

    The count may sit inside the compared expression: "at least 20 adolescents"
    comes back as (SELECT COUNT(DISTINCT p.participant_id) FROM bids_participants
    p WHERE ... p.age >= 10 AND p.age <= 19) >= 20, and "at least 100
    participants" as GREATEST(COUNT(DISTINCT o.subject), (SELECT COUNT ...)) >= 100.
    A count limited to an age range says so: ('Subjects', '≥ 20 aged 10–19').
    """
    for count in node.find_all(exp.Count):
        columns = [c.name.lower() for c in count.find_all(exp.Column)]
        field_name = next((_COUNTED[c] for c in columns if c in _COUNTED), None)
        if not field_name:
            continue
        label = _bound(node, lambda left, count=count: left is count
                       or any(inner is count for inner in left.find_all(exp.Count)))
        if not label:
            continue
        scope = count.find_ancestor(exp.Select)
        if scope is not None and scope is not node and scope.find_ancestor(exp.Subquery) is not None:
            age = _age_label(scope.args.get("where") or scope)
            if age:
                label = f"{label} aged {age}"
        return field_name, label
    return None


def _size_label(node) -> Optional[str]:
    """'≤ 5 GB' for a d.size_bytes comparison."""
    def is_size(left) -> bool:
        return isinstance(left, exp.Column) and left.name.lower() == "size_bytes"

    label = _bound(node, is_size)
    if not label:
        return None
    symbol, _, value = label.partition(" ")
    try:
        gigabytes = float(value) / 1024 ** 3
    except ValueError:
        return label
    return f"{symbol} {gigabytes:.3g} GB"


def _json_key_chip(node) -> Optional[Tuple[str, str]]:
    """('Magnetic field strength', '6.5–7.5') for other_entities ->> 'MagneticFieldStrength'."""
    for extract in node.find_all(exp.JSONExtractScalar, exp.JSONExtract):
        key = extract.expression
        name = key.this if isinstance(key, exp.Literal) else key.sql().strip("'$.")
        if not name:
            continue
        field_name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).capitalize()
        label = _bound(node, lambda left: True) or "recorded"
        return field_name, label
    return None


# ── One condition -> one chip ─────────────────────────────────────────────────

def _venue_words(node) -> List[str]:
    """README words for a condition's recorded fields: the asked sex ("women"),
    a scanner make or model ("siemens"), a field strength ("7T", also when
    the chip is labelled by the modality it shares a condition with: the
    page and highlight_field_strengths read it as a strength). Age words
    come with the age chip."""
    from services.fulltext import uncertain_tier as tier
    words: List[str] = []
    sex = tier.asked_sex(node)
    if sex:
        words.extend(tier.SEX_WORDS[sex])
    words.extend(f"{strength:g}T" for strength in tier.asked_field_strengths(node))
    for key in ("Manufacturer", "ManufacturerModelName"):
        values = tier._key_values(node, key)
        if values:
            words.extend(tier.sidecar_words(key, values))
    return words


def describe_condition(node, chip_id: str) -> QueryFilter:
    negated = False
    while isinstance(node, (exp.Paren, exp.Not)):
        if isinstance(node, exp.Not):
            negated = not negated
        node = node.this

    phrases: List[str] = []
    # Words inside a NOT of the condition are what it excludes, not what it
    # searches for (a topic's unwanted meanings; see describe_exclusion).
    for query in _tsqueries(node, negated=False):
        phrases.extend(tsquery_phrases(query))
    # Older plans search the description with ILIKE instead of full text.
    phrases.extend(_literals_for(node, ["description_text", "readme_text", "readme"]))
    phrases = list(dict.fromkeys(phrases))

    # Word forms a search is limited to ("ageing", "aging": not every "age").
    # They are words of the text too, so the page can mark them.
    forms = guard_forms(node)
    searched = phrases + [form for form in forms if form not in phrases]
    # Words of the text venues the uncertain tier adds for fields that can be
    # missing (services/fulltext/uncertain_tier.py): the page marks them.
    for word in _venue_words(node):
        if word not in searched:
            searched.append(word)
    forms_note = f"Only these word forms count: {', '.join(forms[:8])}." if forms else ""

    def chip(kind: str, field_name: str, label: Union[str, List[str]], detail: str = "",
             shown: int = _MAX_VALUES_SHOWN) -> QueryFilter:
        # A list of values becomes "a or b +N", with the full list kept in `values`.
        values = _hidden(label, shown) if isinstance(label, list) else []
        if isinstance(label, list):
            label = _short_list(label, " or ", shown)
        if phrases:
            words = ", ".join(phrases[:6])
            detail = (f"{detail} Or the dataset text mentions: {words}." if detail
                      else f"Also matches datasets whose text mentions: {words}.")
        if forms_note:
            detail = f"{detail} {forms_note}".strip()
        return QueryFilter(id=chip_id, kind=kind, field=field_name, label=label,
                           negated=negated, text_words=searched, detail=detail, values=values)

    def codes_detail(noun: str, names: List[str]) -> str:
        if len(names) == 1:
            return f"Coded {noun}: {names[0]}."
        shown = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
        return f"Any of {len(names)} coded {noun}s: {shown}."

    patients = _patients_label(node)
    if patients:
        # Not "subjects": the result cards would emphasise the dataset's subject count.
        return chip("count", "Patients", patients,
                    "Participants recorded with the diagnosis in the dataset's participant list.")

    count = _count_chip(node)
    if count:
        return chip("subjects" if count[0] == "Subjects" else "count", count[0], count[1])

    diagnoses = _literals_for(node, ["diagnosis", "dx_status"])
    if diagnoses:
        names = [_humanize_code(d) for d in diagnoses]
        return chip("diagnosis", "Diagnosis", names, codes_detail("diagnosis", names), shown=1)

    if _is_topic(node):
        names = [_humanize_code(t) for t in _literals_for(node, ["task"])]
        # Not `searched`: that is the word list the page marks (text_words).
        searched_as = ", ".join(phrases[:10]) + ("…" if len(phrases) > 10 else "")
        detail = f"One subject, its words close together. Searched as: {searched_as}."
        if names:
            detail += " " + codes_detail("task", names)
        if forms_note:
            detail += " " + forms_note
        return QueryFilter(id=chip_id, kind="topic", field="Topic",
                           label=phrases[0] if phrases else (names[0] if names else "topic"),
                           negated=negated, text_words=searched, detail=detail)

    tasks = _literals_for(node, ["task"])
    if tasks:
        # Task concepts expand to dozens of codes; the words read better.
        names = [_humanize_code(t) for t in tasks]
        return chip("task", "Task", phrases[0] if phrases else names,
                    codes_detail("task", names), shown=1)

    modality_codes = _literals_for(node, ["datatype", "suffix", "modality"])
    if modality_codes:
        labels = [_MODALITY_LABELS.get(c.lower(), _humanize_code(c)) for c in modality_codes]
        return chip("modality", "Modality", labels)

    age = _age_label(node)
    if age:
        # A README naming the age group is evidence too (the uncertain
        # tier's probable form, services/fulltext/uncertain_tier.py): its
        # words are what the page marks for a match on recorded ages.
        from services.fulltext.uncertain_tier import age_group_words, asked_age_range
        group_words = age_group_words(*asked_age_range(node.sql(dialect="postgres")))
        searched.extend(w for w in group_words if w not in searched)
        return chip("age", "Age", age)

    species = _species_label(node)
    if species:
        return chip("field", "Species", species)

    sexes = _literals_for(node, ["sex"])
    if sexes:
        sex_names = {"f": "female", "female": "female", "m": "male", "male": "male"}
        return chip("sex", "Sex", [sex_names.get(s.lower(), s) for s in sexes])

    size = _size_label(node)
    if size:
        return chip("field", "Size", size)

    json_key = _json_key_chip(node)
    if json_key:
        return chip("field", json_key[0], json_key[1])

    for column, field_name in _FIELD_COLUMNS.items():
        values = _literals_for(node, [column])
        if values:
            values = [_FIELD_VALUE_LABELS.get(v.lower(), v) for v in values]
            return chip("field", field_name, values)

    if phrases:
        # The phrases of one text search are its alternatives ('drawings | art');
        # only phrases from searches that must all match are joined by "+".
        one_search = len(_tsqueries(node, negated=False)) == 1
        joiner = " or " if (isinstance(node, exp.Or) or one_search) else " + "
        label = joiner.join(f"“{p}”" for p in phrases[:2])
        if len(phrases) > 2:
            label += f" +{len(phrases) - 2}"
        return QueryFilter(id=chip_id, kind="text", field="Text", label=label,
                           negated=negated, text_words=searched, values=_hidden(phrases),
                           detail=(f"The dataset text mentions: {', '.join(phrases[:8])}. "
                                   f"{forms_note}").strip())

    sql = node.sql(dialect="postgres")
    return QueryFilter(id=chip_id, kind="other", field="Condition",
                       label=(sql[:48] + "…") if len(sql) > 48 else sql,
                       negated=negated, detail=sql)


# ── Sort order and limit -> one chip ──────────────────────────────────────────

SORT_CHIP_ID = "o0"

# (column, words when descending, words when ascending)
_SORT_KEYS = (
    (("subject_count", "subject", "participant_id"), "most participants first", "fewest participants first"),
    (("size_bytes",), "largest size first", "smallest size first"),
    (("publish_date",), "newest first", "oldest first"),
    (("downloads",), "most downloaded first", "least downloaded first"),
    (("views",), "most viewed first", "least viewed first"),
    (("name",), "by name, Z–A", "by name"),
)


def _sort_key_label(ordered) -> str:
    key = ordered.this
    desc = bool(ordered.args.get("desc"))
    columns = {c.name.lower() for c in key.find_all(exp.Column)}
    for names, if_desc, if_asc in _SORT_KEYS:
        if columns & set(names):
            return if_desc if desc else if_asc
    text = key.sql(dialect="postgres")
    text = (text[:40] + "…") if len(text) > 40 else text
    return f"by {text}" + (", highest first" if desc else "")


def _limit_value(tree) -> Optional[str]:
    limit = tree.args.get("limit")
    if limit is None:
        return None
    value = limit.args.get("expression") or limit.this
    return value.this if isinstance(value, exp.Literal) else None


def describe_order(tree) -> Optional[QueryFilter]:
    """('Sorted', 'most participants first · top 1') for the query's ORDER BY and LIMIT.

    "find the largest dataset in openneuro" shows only one dataset; without
    this chip nothing said it was the top one of a ranking. Removing it shows
    every matching dataset in the usual order.
    """
    order = tree.args.get("order")
    labels = list(dict.fromkeys(_sort_key_label(o) for o in (order.expressions if order else [])))
    top = _limit_value(tree)
    if not labels and top is None:
        return None
    if (tree.args.get("where") is None and tree.args.get("having") is None
            and labels == ["by name"] and top == "200"):
        return None  # the fallback sample (sql_rewriter.FALLBACK_SQL), not a ranking
    if labels:
        field_name, label = "Sorted", ", then ".join(labels[:2])
        if top is not None:
            label += f" · top {top}"
    else:
        field_name, label = "Limit", f"first {top}"
    return QueryFilter(id=SORT_CHIP_ID, kind="sort", field=field_name, label=label,
                       detail="Remove to see every matching dataset, not only the top of this order.")


def _is_topic(node) -> bool:
    """A topic condition: a 'near' search, or a search that excludes other meanings."""
    if any(_TS_DISTANCE_RE.search(q) for q in _tsqueries(node, negated=False)):
        return True
    return bool(_tsqueries(node, negated=True)) and bool(_tsqueries(node, negated=False))


def describe_exclusion(node, chip_id: str) -> Optional[QueryFilter]:
    """The meanings a topic condition excludes, as a chip of their own; None if none."""
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Not) or not _is_topic(node):
        return None
    excluded: List[str] = []
    for query in _tsqueries(node, negated=True):
        excluded.extend(tsquery_phrases(query))
    excluded = list(dict.fromkeys(excluded))
    if not excluded:
        return None
    label = ", ".join(excluded[:2]) + (f" +{len(excluded) - 2}" if len(excluded) > 2 else "")
    shown = ", ".join(excluded[:12]) + ("…" if len(excluded) > 12 else "")
    return QueryFilter(
        id=chip_id + EXCLUSION_SUFFIX, kind="sense", field="Not meant", label=label,
        values=_hidden(excluded),
        detail=(f"Other meanings of the word, ruled out by the question: {shown}. "
                "Datasets that match the topic only alongside these come after the others. "
                "Remove to treat them like the rest."),
    )


def _drop_exclusions(node):
    """*node* without the NOT parts that sit inside it: (A AND NOT B) → A."""
    for not_node in list(node.find_all(exp.Not)):
        if not_node is node:
            continue
        child = not_node
        parent = child.parent
        while isinstance(parent, exp.Paren):
            child, parent = parent, parent.parent
        if isinstance(parent, exp.And):
            other = parent.right if parent.left is child else parent.left
            if parent is node:
                node = other
            else:
                parent.replace(other)
    return node


# ── Whole query ───────────────────────────────────────────────────────────────

def _flatten_and(node) -> list:
    from services.fulltext.text_search import is_guarded_pair
    if isinstance(node, exp.And):
        return _flatten_and(node.left) + _flatten_and(node.right)
    if isinstance(node, exp.Paren) and isinstance(node.this, exp.And):
        if is_guarded_pair(node):
            return [node]   # a text search and its word forms: one chip
        return _flatten_and(node.this)
    return [node]


def _parse(sql: str):
    if sqlglot is None or not sql:
        return None
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except SqlglotError:
        return None
    return tree if isinstance(tree, exp.Select) else None


def _conditions(tree) -> List[Tuple[str, exp.Expression]]:
    found: List[Tuple[str, exp.Expression]] = []
    for prefix, clause in (("w", "where"), ("h", "having")):
        node = tree.args.get(clause)
        if node is not None:
            found.extend((f"{prefix}{i}", c) for i, c in enumerate(_flatten_and(node.this)))
    return found


def where_conditions(
    sql: Optional[str],
) -> Tuple[Optional[str], List[Tuple[QueryFilter, exp.Expression]]]:
    """The dataset alias and each top-level WHERE condition with its chip.

    For services/results/match_reasons.py, which evaluates a condition's
    branches per dataset. HAVING conditions (counts) are left out: the card
    shows the counts themselves. (None, []) when the SQL cannot be read.
    """
    tree = _parse(sql or "")
    if tree is None:
        return None, []
    source = tree.args.get("from") or tree.args.get("from_")
    table = getattr(source, "this", None)
    alias = (table.alias_or_name if isinstance(table, exp.Table) and table.name == "bids_datasets"
             else None)
    found = []
    for chip_id, node in _conditions(tree):
        if not chip_id.startswith("w"):
            continue
        try:
            found.append((describe_condition(node, chip_id), node))
        except ValueError as exc:  # a number that is not one; the other conditions still count
            log.warning("query_filters: could not describe %s (%s)", node, exc)
    return alias, found


def describe(sql: Optional[str]) -> List[QueryFilter]:
    """One chip per top-level condition; [] when the SQL cannot be read."""
    tree = _parse(sql or "")
    if tree is None:
        return []
    chips = []
    for chip_id, node in _conditions(tree):
        try:
            chips.append(describe_condition(node, chip_id))
            exclusion = describe_exclusion(node, chip_id)
            if exclusion is not None:
                chips.append(exclusion)
        except ValueError as exc:  # a number that is not one; the other chips still show
            log.warning("query_filters: could not describe %s (%s)", node, exc)
            text = node.sql(dialect="postgres")
            chips.append(QueryFilter(id=chip_id, kind="other", field="Condition",
                                     label=text[:48], detail=text))
    sort = describe_order(tree)
    if sort is not None:
        chips.append(sort)
    return chips


def remove(sql: str, chip_ids: Iterable[str]) -> Optional[str]:
    """*sql* without the conditions named by *chip_ids*; None if it cannot be parsed.

    Unknown ids are ignored. Removing every condition of a clause drops the
    clause, so the query returns every dataset.
    """
    drop = set(chip_ids)
    if not drop:
        return sql
    tree = _parse(sql)
    if tree is None:
        return None
    if SORT_CHIP_ID in drop:
        for key in ("order", "limit", "offset"):
            tree.set(key, None)
    exclusions = {i[:-len(EXCLUSION_SUFFIX)] for i in drop if i.endswith(EXCLUSION_SUFFIX)}
    for prefix, clause in (("w", "where"), ("h", "having")):
        node = tree.args.get(clause)
        if node is None:
            continue
        parts = _flatten_and(node.this)
        relaxed = [_drop_exclusions(p) if f"{prefix}{i}" in exclusions else p
                   for i, p in enumerate(parts)]
        kept = [p for i, p in enumerate(relaxed) if f"{prefix}{i}" not in drop]
        if len(kept) == len(parts) and all(a is b for a, b in zip(relaxed, parts)):
            continue
        if not kept:
            tree.set(clause, None)
            continue
        condition = kept[0]
        for part in kept[1:]:
            condition = exp.and_(condition, part)
        cls = exp.Where if clause == "where" else exp.Having
        tree.set(clause, cls(this=condition))
    return tree.sql(dialect="postgres")


# ── Highlighting ──────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


# Postgres's English stopwords (tsearch_data/english.stop). Inside a phrase
# query they collapse it: to_tsquery('english', 'scanning <-> of') is 'scan'.
_STOPWORDS = frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs
themselves what which who whom this that these those am is are was were be been
being have has had having do does did doing a an the and but if or because as
until while of at by for with about against between into through during before
after above below to from up down in out on off over under again further then
once here there when where why how all any both each few more most other some
such no nor not only own same so than too very s t can will just don should now
""".split())

# A highlight term: one word, or two words that count only side by side.
HighlightTerm = Tuple[str, ...]


def _is_acronym(word: str) -> bool:
    """'MRI', 'fMRI', 'T1w', 'ADHD'; not 'Healthy' or 'Single'."""
    return any(c.isupper() for c in word[1:]) or (
        any(c.isdigit() for c in word) and any(c.isalpha() for c in word))


def _phrase_terms(phrase: str, acronyms_alone: bool = False) -> List[HighlightTerm]:
    """Highlight terms of one searched phrase.

    One word stands alone. Several words only count as adjacent pairs:
    "control participants" alone would mark every "participants", and
    stemming makes it worse ("subjective" marks "subject", "structural"
    marks the "Data Structure" of every BIDS README). *acronyms_alone*:
    acronyms in a coded label ("structural MRI") also stand alone.

    A digit stays as part of a pair ("7 tesla", "2 back"), never alone.
    """
    tokens = [w for w in _WORD_RE.findall(phrase)
              if (len(w) > 1 or w.isdigit()) and w.lower() not in _STOPWORDS]
    if len(tokens) == 1:
        return [(tokens[0],)] if len(tokens[0]) > 1 else []
    terms: List[HighlightTerm] = [(a.lower(), b.lower()) for a, b in zip(tokens, tokens[1:])]
    if acronyms_alone:
        terms.extend((w,) for w in tokens if _is_acronym(w))
    return terms


def highlight_terms(sql: Optional[str]) -> List[HighlightTerm]:
    """What to mark in a README: the chips' text words, and the labels of
    coded chips ("fMRI", "ADHD"), which READMEs often spell out too.

    Every chip that is not negated, whatever its kind: a dataset can match a
    chip in several ways (recorded ages, coded diagnoses, files, text), and
    the words of each way are marked where the text has them. An age chip
    carries its group's words ("infant", "newborn").

    Single words keep their case: a two-letter acronym ("MS", "PD") only
    counts in capitals, not as "ms" or "pd".
    """
    terms: List[HighlightTerm] = []
    for chip in describe(sql):
        if chip.negated:
            continue
        for phrase in chip.text_words:
            terms.extend(_phrase_terms(phrase))
        if chip.kind in ("diagnosis", "modality", "task"):
            label = re.sub(r"\s\+\d+$", "", chip.label)
            for part in re.split(r"\s+or\s+", label):
                terms.extend(_phrase_terms(part, acronyms_alone=True))
    seen, unique = set(), []
    for term in terms:
        key = tuple(w.lower() for w in term)
        if key not in seen:
            seen.add(key)
            unique.append(term)
    return unique


def highlight_exact_words(sql: Optional[str]) -> FrozenSet[str]:
    """The words of highlight_terms(sql) that only count as written.

    A search limited to word forms ("ageing", "aging") must not mark what
    merely shares their stem ("age") in an excerpt.
    """
    tree = _parse(sql or "")
    if tree is None:
        return frozenset()
    forms: List[str] = []
    for _, node in _conditions(tree):
        inner = node
        negated = False
        while isinstance(inner, (exp.Paren, exp.Not)):
            negated ^= isinstance(inner, exp.Not)
            inner = inner.this
        if not negated:
            forms.extend(guard_forms(inner))
    return frozenset(forms)


# ── Field strength in the text ────────────────────────────────────────────────
#
# A field-strength chip is a number range on the sidecar key
# MagneticFieldStrength ("6.5–7.5"); it has no words to mark. READMEs write
# the strength as "7T", "7 T", "7-Tesla", "3.0T", "1.5 Tesla". These are the
# strengths scanners are sold at; a range marks the ones inside it.

FIELD_STRENGTH_FIELD = "Magnetic field strength"
# The strengths scanners are sold at (defined with the README venue of the
# uncertain tier, which the translation uses; this module only displays).
from services.fulltext.uncertain_tier import NOMINAL_TESLA  # noqa: E402
_NUMBER = r"\d+(?:\.\d+)?"
_RANGE_LABEL_RE = re.compile(rf"^({_NUMBER})–({_NUMBER})$")
_BOUND_LABEL_RE = re.compile(rf"^(=|≥|>|≤|<) ({_NUMBER})$")
# A searched phrase that is itself a strength: "7T", "7 tesla", "7tesla".
_STRENGTH_PHRASE_RE = re.compile(rf"^\s*({_NUMBER})\s*-?\s*(?:t|tesla)\s*$", re.IGNORECASE)


def _strengths_of_label(label: str) -> List[float]:
    found = _RANGE_LABEL_RE.match(label)
    if found:
        low, high = float(found.group(1)), float(found.group(2))
        return [t for t in NOMINAL_TESLA if low <= t <= high]
    found = _BOUND_LABEL_RE.match(label)
    if not found:
        return []
    symbol, value = found.group(1), float(found.group(2))
    if symbol == "=":
        return [value]
    keep = {"≥": value.__le__, ">": value.__lt__, "≤": value.__ge__, "<": value.__gt__}[symbol]
    return [t for t in NOMINAL_TESLA if keep(t)]


def highlight_field_strengths(sql: Optional[str]) -> List[float]:
    """The field strengths (tesla) to mark in a dataset's text: those of a
    field-strength chip, and searched phrases that name one ("7 tesla")."""
    strengths: List[float] = []
    for chip in describe(sql):
        if chip.negated:
            continue
        if chip.field == FIELD_STRENGTH_FIELD:
            strengths.extend(_strengths_of_label(chip.label))
        for phrase in chip.text_words:
            found = _STRENGTH_PHRASE_RE.match(phrase)
            if found:
                strengths.append(float(found.group(1)))
    return list(dict.fromkeys(strengths))


def field_strength_pattern(strengths: Iterable[float]) -> Optional["re.Pattern[str]"]:
    """Where a text names one of *strengths*: "7T", "7 T", "7-Tesla", "7.0T".

    Not inside another number ("17T", "0.7T"), a BIDS label ("ses-7T"), a
    word starting with T ("7 T1w images", "3 The") or a gradient spec
    ("200 T/m/s"). Spaces include the thin ones some READMEs use. Checked on
    the 1,720 OpenNeuro READMEs of 2026-11: no false matches found.
    """
    from services.fulltext.uncertain_tier import field_strength_regex
    pattern = field_strength_regex(strengths)
    return re.compile(pattern) if pattern else None


def field_strength_tsquery(strengths: Iterable[float]) -> Optional[str]:
    """Postgres words for the strengths, to pick the README passage.

    The english parser keeps "7T" and "7Tesla" as one word; "T" is a
    stopword, so "1.5T" and "7 T" become the bare number. A decimal strength
    is therefore searched as its number; field_strength_pattern decides what
    is marked.
    """
    parts = []
    for value in dict.fromkeys(strengths):
        number = f"{value:g}"
        if "." in number:
            parts.append(number)
        else:
            parts.append(f"{number}t | {number}tesla | {number} <-> tesla")
    return " | ".join(parts) or None


def terms_tsquery(terms: Iterable[HighlightTerm]) -> Optional[str]:
    """An OR tsquery of *terms*; a pair matches side by side or with one
    word between ("Parkinson's disease", "scanning of one")."""
    parts = []
    for term in terms:
        words = [w.lower() for w in term]
        if len(words) == 1:
            parts.append(words[0])
        else:
            a, b = words
            parts.append(f"({a} <-> {b} | {a} <2> {b})")
    return " | ".join(parts) or None


def highlight_tsquery(sql: Optional[str]) -> Optional[str]:
    """The tsquery that marks *sql*'s highlight terms in a README."""
    return terms_tsquery(highlight_terms(sql))
