"""
services/query_filters.py
-------------------------
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
tier is applied (services/uncertain_tier.py), so a refined search runs
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
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional, Tuple

try:  # sqlglot is a backend dependency; keep the module importable without it
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover
    sqlglot = None
    exp = None

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
    text_words: List[str] = field(default_factory=list)
    detail: str = ""            # longer description for a tooltip
    # Every value the label stands for when it shortens them to "+N", so the
    # user can open the full list; empty when the label already shows them all.
    values: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Reading values out of a condition ─────────────────────────────────────────

def _humanize_code(code: str) -> str:
    words = code.replace("_", " ").strip()
    if words.lower() in _ACRONYMS:
        return words.upper()
    words = _POSSESSIVE_RE.sub(lambda m: m.group(1) + "'s", words)
    return words[:1].upper() + words[1:]


def _short_list(values: List[str], joiner: str = ", ") -> Tuple[str, List[str]]:
    """The label for a list of values, and the full list when the label is cut to "+N"."""
    values = list(dict.fromkeys(v for v in values if v))
    if len(values) <= _MAX_VALUES_SHOWN:
        return joiner.join(values), []
    return f"{joiner.join(values[:_MAX_VALUES_SHOWN])} +{len(values) - _MAX_VALUES_SHOWN}", values


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


def _tsqueries(node, negated: Optional[bool] = None) -> List[str]:
    """The query strings of every to_tsquery(...) in the condition.

    *negated*: None for all, False for those outside a NOT inside the
    condition, True for those inside one.
    """
    queries = []
    for call in node.find_all(exp.Anonymous, exp.Func):
        name = (call.name if isinstance(call, exp.Anonymous) else call.sql_name()).lower()
        if name != "to_tsquery":
            continue
        if negated is not None and _under_not(call, node) != negated:
            continue
        literals = [e for e in call.expressions if isinstance(e, exp.Literal) and e.is_string]
        if literals:
            queries.append(literals[-1].this)
    return queries


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
    parts, depth, current = [], 0, []
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
        words = [w for w in _TS_WORD_RE.findall(alternative)]
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
    if low is None and high is None:
        # Only the dataset-level bounds (d.age_min / d.age_max).
        columns = {c.name.lower() for c in node.find_all(exp.Column)}
        if not columns & {"age_min", "age_max"}:
            return None
        from services.uncertain_tier import asked_age_range
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
    for cls, symbol in _SYMBOLS.items():
        for comparison in ([node] if isinstance(node, cls) else []) + list(node.find_all(cls)):
            if matches(comparison.this) and isinstance(comparison.expression, exp.Literal) \
                    and not comparison.expression.is_string:
                return f"{symbol} {comparison.expression.this}"
    return None


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
        label = _bound(node, lambda left: left is count
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

    def chip(kind: str, field_name: str, label: str, detail: str = "",
             values: Optional[List[str]] = None) -> QueryFilter:
        if phrases:
            words = ", ".join(phrases[:6])
            detail = (f"{detail} Or the dataset text mentions: {words}." if detail
                      else f"Also matches datasets whose text mentions: {words}.")
        return QueryFilter(id=chip_id, kind=kind, field=field_name, label=label,
                           negated=negated, text_words=phrases, detail=detail,
                           values=values or [])

    def codes_detail(noun: str, names: List[str]) -> str:
        if len(names) == 1:
            return f"Coded {noun}: {names[0]}."
        shown = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
        return f"Any of {len(names)} coded {noun}s: {shown}."

    count = _count_chip(node)
    if count:
        return chip("subjects" if count[0] == "Subjects" else "count", count[0], count[1])

    diagnoses = _literals_for(node, ["diagnosis", "dx_status"])
    if diagnoses:
        names = [_humanize_code(d) for d in diagnoses]
        label = names[0] + (f" +{len(names) - 1}" if len(names) > 1 else "")
        return chip("diagnosis", "Diagnosis", label, codes_detail("diagnosis", names),
                    values=names if len(names) > 1 else None)

    if _is_topic(node):
        names = [_humanize_code(t) for t in _literals_for(node, ["task"])]
        searched = ", ".join(phrases[:10]) + ("…" if len(phrases) > 10 else "")
        detail = f"One subject, its words close together. Searched as: {searched}."
        if names:
            detail += " " + codes_detail("task", names)
        return QueryFilter(id=chip_id, kind="topic", field="Topic",
                           label=phrases[0] if phrases else (names[0] if names else "topic"),
                           negated=negated, text_words=phrases, detail=detail)

    tasks = _literals_for(node, ["task"])
    if tasks:
        # Task concepts expand to dozens of codes; the words read better.
        names = [_humanize_code(t) for t in tasks]
        label = phrases[0] if phrases else names[0] + (f" +{len(names) - 1}" if len(names) > 1 else "")
        return chip("task", "Task", label, codes_detail("task", names),
                    values=names if not phrases and len(names) > 1 else None)

    modality_codes = _literals_for(node, ["datatype", "suffix", "modality"])
    if modality_codes:
        labels = [_MODALITY_LABELS.get(c.lower(), _humanize_code(c)) for c in modality_codes]
        label, values = _short_list(labels, " or ")
        return chip("modality", "Modality", label, values=values)

    age = _age_label(node)
    if age:
        return chip("age", "Age", age)

    sexes = _literals_for(node, ["sex"])
    if sexes:
        names = {"f": "female", "female": "female", "m": "male", "male": "male"}
        label, values = _short_list([names.get(s.lower(), s) for s in sexes], " or ")
        return chip("sex", "Sex", label, values=values)

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
            label, values = _short_list(values, " or ")
            return chip("field", field_name, label, values=values)

    if phrases:
        joiner = " or " if isinstance(node, exp.Or) else " + "
        label = joiner.join(f"“{p}”" for p in phrases[:2])
        if len(phrases) > 2:
            label += f" +{len(phrases) - 2}"
        return QueryFilter(id=chip_id, kind="text", field="Text", label=label,
                           negated=negated, text_words=phrases,
                           values=phrases if len(phrases) > 2 else [],
                           detail=f"The dataset text mentions: {', '.join(phrases[:8])}.")

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
        detail=(f"Other meanings of the word, ruled out by the question: {shown}. "
                "Datasets that match the topic only alongside these come after the others. "
                "Remove to treat them like the rest."),
        values=excluded if len(excluded) > 2 else [],
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
    if isinstance(node, exp.And):
        return _flatten_and(node.left) + _flatten_and(node.right)
    if isinstance(node, exp.Paren) and isinstance(node.this, exp.And):
        return _flatten_and(node.this)
    return [node]


def _parse(sql: str):
    if sqlglot is None or not sql:
        return None
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return None
    return tree if isinstance(tree, exp.Select) else None


def _conditions(tree) -> List[Tuple[str, object]]:
    found = []
    for prefix, clause in (("w", "where"), ("h", "having")):
        node = tree.args.get(clause)
        if node is not None:
            found.extend((f"{prefix}{i}", c) for i, c in enumerate(_flatten_and(node.this)))
    return found


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
        except Exception as exc:  # one odd condition must not lose the others
            log.debug("query_filters: could not describe %s (%s)", node, exc)
            text = node.sql(dialect="postgres")
            chips.append(QueryFilter(id=chip_id, kind="other", field="Condition",
                                     label=text[:48], detail=text))
    try:
        sort = describe_order(tree)
    except Exception as exc:
        log.debug("query_filters: could not describe the order (%s)", exc)
        sort = None
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

_HIGHLIGHT_KINDS = {"diagnosis", "modality", "task", "text", "field", "topic"}
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def highlight_tsquery(sql: Optional[str]) -> Optional[str]:
    """An OR tsquery of the words the chips search for, to mark them in a README.

    Uses each chip's text words, and the labels of coded chips ("fMRI",
    "ADHD"), which READMEs often spell out too. None without such words.
    """
    words: List[str] = []
    for chip in describe(sql):
        if chip.negated or chip.kind not in _HIGHLIGHT_KINDS:
            continue
        phrases = list(chip.text_words)
        if chip.kind in ("diagnosis", "modality", "task"):
            phrases.append(re.sub(r"\s\+\d+$", "", chip.label))
        for phrase in phrases:
            words.extend(w.lower() for w in _WORD_RE.findall(phrase) if len(w) > 1)
    words = list(dict.fromkeys(words))
    return " | ".join(words) or None
