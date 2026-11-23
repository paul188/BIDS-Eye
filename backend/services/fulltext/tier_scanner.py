"""
services/fulltext/tier_scanner.py
---------------------------------
Scanner metadata in the dataset's own text: the field strength, vendor and
model a condition asks of the sidecar JSON (an EXISTS on bids_objects), and
their probable form when the sidecars do not record them ("acquired on a 3T
Siemens Prisma").
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Optional, Sequence, Tuple

try:  # sqlglot is a backend dependency; keep the module importable without it
    from sqlglot import exp
except ImportError:  # pragma: no cover
    exp = None  # type: ignore[assignment]

from services.fulltext.tier_conditions import SoftCondition, possible_enabled, sql_literal
from services.fulltext.tier_sex import own_text
from services.schema.tables import objects as _objects


# Sidecar keys a README states as often as the sidecars do ("acquired on a
# 3T Siemens Prisma"). Of 1,210 datasets with MRI files on 2026-11-18, 299
# record no field strength; 52 of them name one in their text. 26 without a
# recorded manufacturer name it. Echo and repetition times are not here: a
# README states them in too many forms to read reliably.
FIELD_STRENGTH_KEY = "MagneticFieldStrength"
# The strengths scanners are sold at; a range asks for the ones inside it.
NOMINAL_TESLA = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.7, 7.0, 9.4, 10.5, 11.7, 14.0)


def _tesla_number(value: float) -> str:
    """The pattern of one strength's number: 7 also as "7.0", 1.5 also as "1,5"."""
    whole, _, fraction = f"{value:g}".partition(".")
    if not fraction:
        return rf"{whole}(?:\.0+)?"
    return rf"{whole}[.,]{fraction}0*"


def field_strength_regex(strengths: Iterable[float], postgres: bool = False) -> Optional[str]:
    """Where a text names one of *strengths*: "7T", "7 T", "7-Tesla", "7.0T".

    Not inside another number ("17T", "0.7T"), a BIDS label ("ses-7T"), a
    word starting with T ("7 T1w images", "3 The") or a gradient spec
    ("200 T/m/s"). Spaces include the thin ones some READMEs use.

    Python's re by default (query_filters.field_strength_pattern marks it in
    excerpts); *postgres*: for Postgres's ~ (POSIX classes, no backslashes),
    which this tier runs over the dataset's text. Both forms are checked
    against each other on a real database
    (tests/test_scanner_sex_venues_postgres.py).

    The Postgres form has plain groups, never "(?:": the query runs through
    SQLAlchemy's text(), which reads ":7" or ":T" after "?" as a bind
    parameter and fails the search.
    """
    numbers = [_tesla_number(v) for v in sorted(set(strengths), reverse=True)]
    if not numbers:
        return None
    space = "[ \u00a0\u2009\u202f]?"
    word = "[[:alnum:]_" if postgres else r"[\w"
    body = "|".join(numbers)
    group = "(?:"
    if postgres:
        body = body.replace(r"\.", "[.]").replace("(?:", "(")
        group = "("
    return (rf"(?<!{word}.,/-]){group}{body}){space}-?{space}"
            rf"{group}T|[Tt]esla|TESLA)(?!{word}/])")


def asked_field_strengths(node) -> List[float]:
    """The nominal strengths a field-strength condition asks for: "> 6.5 AND
    < 7.5" → [7.0], "= 1.5" → [1.5], "≥ 3" → [3.0, 4.0, …]."""
    low, high = 0.0, float("inf")
    low_open = high_open = False
    exact = None
    found = False
    ops = {exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<=", exp.EQ: "="}
    for comparison in node.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.Between):
        if FIELD_STRENGTH_KEY.lower() not in [k.lower() for k in _json_keys(comparison.this)]:
            continue
        try:
            if isinstance(comparison, exp.Between):
                low = max(low, float(comparison.args["low"].this))
                high = min(high, float(comparison.args["high"].this))
                found = True
                continue
            right = comparison.expression
            if not (isinstance(right, exp.Literal) and not right.is_string):
                continue
            value = float(right.this)
        except (KeyError, ValueError, AttributeError):
            continue
        found = True
        op = ops[type(comparison)]
        if op == "=":
            exact = value
        elif op.startswith(">") and value >= low:
            low, low_open = value, op == ">"
        elif op.startswith("<") and value <= high:
            high, high_open = value, op == "<"
    if not found:
        return []
    if exact is not None:
        return [exact]
    return [t for t in NOMINAL_TESLA
            if (t > low if low_open else t >= low) and (t < high if high_open else t <= high)]


_TEXT_SIDECAR_KEYS = {"magneticfieldstrength": FIELD_STRENGTH_KEY,
                      "manufacturer": "Manufacturer",
                      "manufacturermodelname": "ManufacturerModelName"}
# Vendor names as a README writes them (tsquery, english configuration).
_VENDOR_WORDS = {
    "siemens": "siemens",
    "philips": "philips",
    "ge": "(ge <-> healthcare) | (ge <-> medical) | (general <-> electric) | signa | (ge <-> discovery)",
    "general electric": "(ge <-> healthcare) | (ge <-> medical) | (general <-> electric) | signa",
    "bruker": "bruker",
    "canon": "(canon <-> medical) | toshiba",
    "toshiba": "toshiba | (canon <-> medical)",
    "hitachi": "hitachi",
    "united imaging": "(united <-> imaging) | uih",
}
_SIDECAR_LABELS = {FIELD_STRENGTH_KEY: "field strength", "Manufacturer": "scanner manufacturer",
                   "ManufacturerModelName": "scanner model"}


def _json_keys(node) -> List[str]:
    keys = []
    for extract in node.find_all(exp.JSONExtractScalar, exp.JSONExtract):
        key = extract.expression
        name = key.this if isinstance(key, exp.Literal) else key.sql().strip("'$.")
        keys.append(name)
    return keys


def _key_values(node, key: str) -> List[str]:
    """String values compared with other_entities ->> *key* ('%GE%' → 'GE')."""
    values = []
    for comparison in node.find_all(exp.EQ, exp.In, exp.ILike, exp.Like):
        if key.lower() not in [k.lower() for k in _json_keys(comparison)]:
            continue
        for literal in comparison.find_all(exp.Literal):
            if literal.is_string and literal.this.lower() != key.lower():
                value = literal.this.strip("%").strip()
                if value:
                    values.append(value)
    return list(dict.fromkeys(values))


def sidecar_words(key: str, values: Sequence[str]) -> List[str]:
    """Words of a manufacturer or model a README would name ('GE' → 'ge healthcare' …)."""
    words: List[str] = []
    for value in values:
        query = _VENDOR_WORDS.get(value.lower()) if key == "Manufacturer" else None
        if query:
            words.extend(re.sub(r"[()]", "", part).replace(" <-> ", " ").strip()
                         for part in query.split("|"))
        else:
            words.append(re.sub(r"[_-]+", " ", value).strip())
    return [w for w in dict.fromkeys(words) if w]


def _value_tsquery(key: str, values: Sequence[str]) -> Optional[str]:
    parts = []
    for value in values:
        query = _VENDOR_WORDS.get(value.lower()) if key == "Manufacturer" else None
        if query is None:
            tokens = re.findall(r"[a-z0-9]+", value.lower())
            if not tokens or (len(tokens) == 1 and len(tokens[0]) < 3):
                return None   # "GE"-like fragments of unknown makes are no evidence
            query = " <-> ".join(tokens)
        parts.append(f"({query})")
    return " | ".join(parts) or None


def _exists_branches(node) -> Optional[list]:
    """The EXISTS subqueries an OR of them is made of; None for another shape."""
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Or):
        left, right = _exists_branches(node.left), _exists_branches(node.right)
        return left + right if left is not None and right is not None else None
    if isinstance(node, exp.Exists) and isinstance(node.this, exp.Select):
        return [node.this]
    return None


def _and_parts(node) -> list:
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.And):
        return _and_parts(node.left) + _and_parts(node.right)
    return [node]


def _file_rows(select, alias: str) -> Optional[Tuple[str, List[str], List[str], set]]:
    """(inner alias, sidecar parts, file parts, file columns) of
    EXISTS(SELECT 1 FROM bids_objects x WHERE x.dataset_id = alias.id AND …);
    None for another shape (a join, a second correlation such as
    o_high.subject = o_low.subject)."""
    tables = list(select.find_all(exp.Table))
    if len(tables) != 1 or tables[0].name.lower() != _objects.name:
        return None
    if select.args.get("joins") or select.args.get("group") or select.args.get("having"):
        return None
    inner = tables[0].alias_or_name
    where = select.args.get("where")
    if where is None:
        return None
    sidecar: List[str] = []
    files: List[str] = []
    file_columns: set[str] = set()
    correlated = False
    for part in _and_parts(where.this):
        columns = list(part.find_all(exp.Column))
        if (isinstance(part, exp.EQ) and {c.table for c in columns} == {inner, alias}
                and {c.name.lower() for c in columns} == {"dataset_id", "id"}):
            correlated = True
            continue
        if any(c.table not in (inner, "") for c in columns):
            return None
        if _json_keys(part):
            sidecar.append(part.sql(dialect="postgres"))
        else:
            files.append(part.sql(dialect="postgres"))
            file_columns.update(c.name.lower() for c in columns)
    if not correlated or not sidecar:
        return None
    return inner, sidecar, files, file_columns


def _cte_name(select_sql: str) -> str:
    """A stable name for a WITH query; the same lookup in two conditions is one."""
    return "_qf_sidecar_" + hashlib.sha256(select_sql.encode()).hexdigest()[:10]


def _file_set(inner: str, files: List[str], columns: set) -> Tuple[str, Optional[Tuple[str, str]]]:
    """The datasets with the file part's files: (IN subquery, WITH query or None).

    From bids_dataset_facets when the part reads only datatype, suffix and
    task (20,926 rows instead of the objects' ~1M; the same summary executed
    SQL uses, services/translate/sql_rewriter.py), else once from the objects.
    """
    where = " AND ".join(f"({p})" for p in files)
    try:
        from services.catalogue.catalogue import Catalogue, catalogue
        from services.schema.tables import dataset_facets
        facets = catalogue.summaries.facets
    except ImportError:  # pragma: no cover
        facets = False
    if facets and columns <= set(Catalogue.FACET_FIELDS):
        return (f"SELECT {inner}.dataset_id FROM {dataset_facets.name} AS {inner} WHERE {where}",
                None)
    select = f"SELECT DISTINCT {inner}.dataset_id FROM {_objects.name} AS {inner} WHERE {where}"
    name = _cte_name(select)
    return f"SELECT dataset_id FROM {name}", (name, select)


def _strength_tsquery(strengths: Sequence[float]) -> str:
    """Indexed words that every text field_strength_regex matches has: the
    english parser keeps "7T", "7Tesla", "7.0T" as one word and splits "7 T"
    and "7-Tesla" into the bare number. A prefilter only: the regex decides.
    Checked against the regex on Postgres (tests/test_scanner_sex_venues_postgres.py)."""
    words: List[str] = []
    for value in strengths:
        number = f"{value:g}"
        forms = [number, f"{value:.1f}"]
        if "." in number:
            forms.append(number.replace(".", ","))
        for form in dict.fromkeys(forms):
            words.extend((form, f"{form}t", f"{form}tesla"))
    return " | ".join(dict.fromkeys(words))


def sidecar(node, alias: str) -> Optional[SoftCondition]:
    """A field strength, manufacturer or model condition with its text venue.

    Confirmed: the sidecars hold the value (as asked). Probably relevant: no
    file of the dataset records the key and its own text names the value, with
    the condition's file part still required ("7T fMRI": fMRI files). A
    recorded value that does not fit excludes the dataset, whatever the README
    says.
    """
    keys = {k.lower() for k in _json_keys(node)}
    if len(keys) != 1 or not keys <= set(_TEXT_SIDECAR_KEYS):
        return None
    key = _TEXT_SIDECAR_KEYS[next(iter(keys))]
    selects = _exists_branches(node)
    if not selects:
        return None
    rows = [r for s in selects if (r := _file_rows(s, alias)) is not None]
    if len(rows) < len(selects):
        return None
    if len({(tuple(r[2])) for r in rows}) > 1:
        return None   # branches on different files: no single file part

    # Each branch's matching datasets, computed once (the condition is read
    # in the candidate filter and again for its evidence).
    ctes = []
    confirmed = []
    for inner, sidecar, files, _ in rows:
        where = " AND ".join(f"({p})" for p in files + sidecar)
        select = f"SELECT DISTINCT {inner}.dataset_id FROM {_objects.name} AS {inner} WHERE {where}"
        name = _cte_name(select)
        ctes.append((name, select))
        confirmed.append(f"{alias}.id IN (SELECT dataset_id FROM {name})")
    inner, _, files, columns = rows[0]
    file_part = None
    if files:
        file_select, file_cte = _file_set(inner, files, columns)
        file_part = f"{alias}.id IN ({file_select})"
        if file_cte:
            ctes.append(file_cte)

    if key == FIELD_STRENGTH_KEY:
        strengths = asked_field_strengths(node)
        pattern = field_strength_regex(strengths, postgres=True)
        if not pattern:
            return None
        # The indexed words first: the regex then reads a few hundred texts.
        text = (f"{alias}.search_tsv @@ TO_TSQUERY('english', {sql_literal(_strength_tsquery(strengths))}) "
                f"AND {own_text(alias)} ~ {sql_literal(pattern)}")
        named = " or ".join(f"{s:g} T" for s in strengths)
    else:
        values = _key_values(node, key)
        query = _value_tsquery(key, values) if values else None
        if not query:
            return None
        text = f"{alias}.search_tsv @@ TO_TSQUERY('english', {sql_literal(query)})"
        named = " or ".join(values)
    label = _SIDECAR_LABELS[key]
    probable = f"({text}) AND ({file_part})" if file_part else text
    recorded = (f"EXISTS(SELECT 1 FROM {_objects.name} AS uk WHERE uk.dataset_id = {alias}.id "
                f"AND NULLIF(TRIM(uk.other_entities ->> '{key}'), '') IS NOT NULL)")
    # Whether the key is recorded matters only where the text names the value:
    # checked for those few datasets, not for every candidate (6.6 ms per
    # dataset on the live objects table, 11 s for one question). The same
    # states while there is no 'possible' level; with it, every candidate.
    known = recorded if possible_enabled() else f"(({probable}) AND {recorded})"
    return SoftCondition(
        sql=" OR ".join(f"({c})" for c in confirmed) if len(confirmed) > 1 else confirmed[0],
        kind="sidecar", probable=probable, known=known,
        reason_probable=f"{label} not in the file metadata; the description names {named}",
        reason_possible=f"{label} not in the file metadata", ctes=tuple(ctes),
    )
