"""
services/sql_tree.py
--------------------
Reading the sqlglot tree of a generated query: parsing it, and where a node
stands in it.

parse() makes "x NOT LIKE y", a flag on the LIKE node in sqlglot, a Not node
around it. Written back to SQL, the query means the same. Other negations may
keep a flag (NOT LIKE ANY): test with is_negation, not for exp.Not.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import sqlglot
from sql_text import pattern_words  # RAG/ is on the path wherever the backend runs
from sqlglot import exp
from sqlglot.errors import SqlglotError

# The operators that search a text for a pattern or words.
TEXT_MATCHES = (exp.Like, exp.ILike, exp.SimilarTo, exp.RegexpLike, exp.RegexpILike, exp.MatchAgainst)
# Every way a value can be written, and the keys of a JSON path.
_VALUES = (exp.Literal, exp.RawString, exp.ByteString, exp.National, exp.UnicodeString,
           exp.BitString, exp.HexString, exp.JSONPath)

def parse_select(sql: Optional[str]) -> Optional[exp.Select]:
    """The tree of *sql*, None when it cannot be read or is no plain SELECT."""
    tree = parse(sql)
    return tree if isinstance(tree, exp.Select) else None

def parse(sql: Optional[str]) -> Optional[Any]:
    """The tree of *sql*, None when it cannot be read."""
    if not sql:
        return None
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except SqlglotError:
        return None
    return _negations_as_not(tree)

def _negations_as_not(tree: Any) -> Any:
    # Inner ones first: the outer one is replaced by a copy.
    for node in reversed(list(tree.find_all(exp.Like, exp.ILike))):
        # "x NOT LIKE ANY (…)" holds when one pattern does not match; NOT (x
        # LIKE ANY (…)) when none does. It keeps its flag (is_negation).
        if not node.args.get("negate") or isinstance(node.expression, (exp.Any, exp.All)):
            continue
        node.set("negate", None)
        # NOT goes around the whole operator: "x LIKE y ESCAPE z".
        operator = node.parent if isinstance(node.parent, exp.Escape) else node
        negated: exp.Expression = exp.Not(this=operator.copy())
        if operator is tree:
            return negated
        if isinstance(operator.parent, (exp.Binary, exp.Is)) and not isinstance(operator.parent, exp.Connector):
            # "x NOT LIKE y IS NULL": without them the NOT would cover the IS NULL.
            negated = exp.Paren(this=negated)
        operator.replace(negated)
    return tree


def is_negation(node: Any) -> bool:
    """The node negates what stands in it: NOT, NOT LIKE ANY, "= FALSE",
    "<> TRUE", and "IS NULL" of a comparison (of a column it negates nothing)."""
    if isinstance(node, exp.Not) or node.args.get("negate"):
        return True
    if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
        return isinstance(node.this.unnest(), (exp.Predicate, exp.Connector, exp.Not, *TEXT_MATCHES))
    if isinstance(node, (exp.Is, exp.EQ, exp.NullSafeEQ)):
        denying = False
    elif isinstance(node, (exp.NEQ, exp.NullSafeNEQ)):
        denying = True
    else:
        return False
    return any(isinstance(side, exp.Boolean) and side.this is denying
               for side in (node.this, node.expression))


def ancestors(node: Any, root: Any) -> Iterator[Any]:
    """What *node* stands in, up to and including *root*; nothing for *root* itself."""
    while node is not root:
        node = node.parent
        yield node


def in_subquery(node: Any, root: Any) -> bool:
    """*node* stands in a SELECT within *root*, a condition (of a whole query
    every node does)."""
    return any(isinstance(a, exp.Select) for a in ancestors(node, root))


def under_negation(node: Any, root: Any) -> bool:
    """*node* is or stands in a negation within *root*, *root* included. Two
    negations count as one: what they say together is not worked out."""
    return is_negation(node) or any(is_negation(a) for a in ancestors(node, root))


def negations(node: Any) -> int:
    """How many negations *node* stands in, in the whole query."""
    return sum(is_negation(a) for a in ancestors(node, node.root()))


def replace_by_truth(node: Any, value: bool) -> None:
    """Replace a condition by TRUE or FALSE and take out of the query what that
    decides: "NOT EXISTS (… AND FALSE)" holds always, so it goes."""
    while True:
        parent = node.parent
        if isinstance(parent, exp.Paren):
            node = parent
        elif isinstance(parent, exp.Not):
            node, value = parent, not value
        elif isinstance(parent, (exp.And, exp.Or)):
            if value is isinstance(parent, exp.And):   # AND TRUE, OR FALSE: the other side decides
                parent.replace(parent.right if parent.left is node else parent.left)
                return
            node = parent
        elif isinstance(parent, exp.Where):
            select = parent.parent
            if value and isinstance(select, exp.Select) and select.parent is None:
                parent.pop()
                return
            test = None if value else _test_of_rows(select)
            if test is None:
                break
            node = test
        else:
            break
    node.replace(exp.Boolean(this=value))


def _test_of_rows(select: Any) -> Optional[Any]:
    """The EXISTS or IN that is false when *select* finds no row; None when
    there is none, or the SELECT gives a row anyway (COUNT)."""
    if not isinstance(select, exp.Select) or any(e.find(exp.AggFunc) for e in select.expressions):
        return None
    holder = select.parent
    if holder is None or isinstance(holder, exp.Exists):
        return holder
    if isinstance(holder, exp.Subquery) and isinstance(holder.parent, exp.In):
        return holder.parent
    return None


def or_branches(node: Any) -> list:
    """The OR branches of a condition, through parentheses (not through NOT)."""
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Or):
        return or_branches(node.left) + or_branches(node.right)
    return [node]


@dataclass(frozen=True)
class SqlBranch:
    """One leaf of a condition's OR tree, with generic SQL-tree operations."""
    node: Any

    @cached_property
    def sql(self) -> str:
        return self.node.sql(dialect="postgres")

    @cached_property
    def outer_columns(self) -> list:
        """Columns in this branch, excluding columns inside subqueries."""
        return [c for c in self.node.find_all(exp.Column) if not in_subquery(c, self.node)]

    @cached_property
    def outer_selects(self) -> list:
        """Subqueries directly in this branch, excluding nested subqueries."""
        return [s for s in self.node.find_all(exp.Select) if not in_subquery(s, self.node)]

    def reads_a_join(self, alias: str) -> bool:
        """Whether outer columns reference a table other than *alias*."""
        return any(c.table and c.table != alias for c in self.outer_columns)

    def negated(self, node: Any) -> bool:
        """Whether *node* is negated in this branch."""
        return under_negation(node, self.node)


    def has_negation(self) -> bool:
        """Whether this branch contains any negation."""
        return any(is_negation(n) for n in self.node.walk())


@dataclass(frozen=True)
class Condition:
    """A SQL boolean condition and its top-level disjunctive branches."""
    node: Any

    @property
    def branches(self) -> tuple[SqlBranch, ...]:
        return tuple(SqlBranch(node) for node in or_branches(self.node))


def without_literals(node: Any) -> str:
    """The SQL of *node* with every value replaced by '?': for a log, which
    must not hold what people searched for."""
    return node.copy().transform(
        lambda n: exp.Var(this="?") if isinstance(n, _VALUES) else n).sql(dialect="postgres", comments=False)


# ── What a condition compares ─────────────────────────────────────────────────
# The one reader of the values a condition asks for: the chips, the evidence
# and the uncertain tier take them from here.

def json_keys(node: Any) -> List[str]:
    """The keys *node* reads from JSON columns: other_entities ->> 'Manufacturer' → ['Manufacturer']."""
    keys = []
    for extract in node.find_all(exp.JSONExtractScalar, exp.JSONExtract):
        key = extract.expression
        keys.append(str(key.this if isinstance(key, exp.Literal) else key.sql().strip("'$.")))
    return keys


def _own(side: Any, *kinds: type) -> list:
    """The nodes of *kinds* in *side* that are its own value: none in a
    subquery or an aggregate ("(SELECT COUNT(…) WHERE p.age > 9) >= 20" is a
    count), none with a default for a missing value ("COALESCE(d.age_min, 0)
    <= 0" asks whether ages are recorded)."""
    return [n for n in side.find_all(*kinds)
            if not any(isinstance(a, (exp.Select, exp.AggFunc, exp.Coalesce))
                       for a in ancestors(n, side))]


def compares(side: Any, columns: Sequence[str] = (), json_key: Optional[str] = None) -> bool:
    """*side* is one of *columns* or the JSON key *json_key*
    (other_entities ->> 'Manufacturer'), maybe cast or in a function."""
    if side is None:
        return False
    names = {c.lower() for c in columns}
    if names and any(c.name.lower() in names for c in _own(side, exp.Column)):
        return True
    return json_key is not None and json_key.lower() in (
        k.lower() for extract in _own(side, exp.JSONExtractScalar, exp.JSONExtract)
        for k in json_keys(extract))


def _sides(comparison: Any) -> Tuple[Any, list]:
    others = list(comparison.expressions) if isinstance(comparison, exp.In) else [comparison.expression]
    return comparison.this, others


def compared_values(node: Any, columns: Sequence[str] = (), json_key: Optional[str] = None,
                    not_equal: bool = False) -> List[str]:
    """The strings compared (=, IN, ANY, LIKE, ILIKE; also <> with *not_equal*)
    with one of *columns* or the JSON key *json_key*, in order, each once. A
    LIKE pattern gives what it looks for ('%GE%' → 'GE')."""
    kinds = (exp.EQ, exp.In, exp.ILike, exp.Like) + ((exp.NEQ,) if not_equal else ())
    values: List[str] = []
    for comparison in node.find_all(*kinds):
        left, others = _sides(comparison)
        if not compares(left, columns, json_key):
            if not (len(others) == 1 and compares(others[0], columns, json_key)):
                continue
            left, others = others[0], [left]       # 'x' = column
        is_pattern = isinstance(comparison, (exp.ILike, exp.Like))
        for other in others:
            for literal in other.find_all(exp.Literal):
                if literal.is_string:
                    value = pattern_words(literal.this) if is_pattern else literal.this.strip("%")
                    if value.strip():
                        values.append(value.strip())
    return list(dict.fromkeys(values))


@dataclass(frozen=True)
class Bounds:
    """The numbers a condition allows for one value: "age BETWEEN 10 AND 19",
    "COUNT(…) >= 40", "> 6.5 AND < 7.5". None is no bound on that side."""
    low: Optional[float] = None
    high: Optional[float] = None
    low_inclusive: bool = True
    high_inclusive: bool = True

    @property
    def exact(self) -> Optional[float]:
        """The one value allowed ("= 1.5"), else None."""
        return self.low if self.low is not None and self.low == self.high else None

    def allows(self, value: float) -> bool:
        if self.low is not None and (value < self.low or (value == self.low and not self.low_inclusive)):
            return False
        return not (self.high is not None
                    and (value > self.high or (value == self.high and not self.high_inclusive)))


def _number(node: Any) -> Optional[float]:
    if isinstance(node, exp.Neg):
        inner = _number(node.this)
        return -inner if inner is not None else None
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return float(node.this)
        except ValueError:
            return None
    return None


_FLIPPED = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE, exp.EQ: exp.EQ}


def compared_bounds(node: Any, matches: Callable[[Any], bool]) -> Optional[Bounds]:
    """The tightest bounds numbers put on what *matches* (a side of a
    comparison) in *node*: BETWEEN, <, <=, >, >=, = either way round
    ("18 <= p.age"). A negated comparison sets none ("NOT EXISTS(… age > 0)"
    asks for no recorded age). None when no number is compared with it."""
    low: Optional[Tuple[float, bool]] = None      # (value, inclusive)
    high: Optional[Tuple[float, bool]] = None

    def tighten_low(value: float, inclusive: bool) -> None:
        nonlocal low
        if low is None or value > low[0] or (value == low[0] and not inclusive):
            low = (value, inclusive)

    def tighten_high(value: float, inclusive: bool) -> None:
        nonlocal high
        if high is None or value < high[0] or (value == high[0] and not inclusive):
            high = (value, inclusive)

    comparison: Any
    for comparison in node.find_all(exp.Between, *_FLIPPED):   # node itself too
        if under_negation(comparison, node):
            continue
        if isinstance(comparison, exp.Between):
            first, last = _number(comparison.args.get("low")), _number(comparison.args.get("high"))
            if matches(comparison.this) and first is not None and last is not None:
                tighten_low(first, True)
                tighten_high(last, True)
            continue
        kind: Any = type(comparison)
        side, value = comparison.this, _number(comparison.expression)
        if value is None or not matches(side):
            side, value = comparison.expression, _number(comparison.this)
            if value is None or not matches(side):
                continue
            kind = _FLIPPED[kind]
        if kind in (exp.GT, exp.GTE, exp.EQ):
            tighten_low(value, kind is not exp.GT)
        if kind in (exp.LT, exp.LTE, exp.EQ):
            tighten_high(value, kind is not exp.LT)
    if low is None and high is None:
        return None
    return Bounds(low[0] if low else None, high[0] if high else None,
                  low[1] if low else True, high[1] if high else True)


# ── How a query orders and limits its rows ────────────────────────────────────

@dataclass(frozen=True)
class OrderKey:
    """One ORDER BY key; a select alias is replaced by what it names."""
    expression: Any
    descending: bool

    @property
    def columns(self) -> Tuple[str, ...]:
        """The names of the columns it reads, lower case, in order."""
        return tuple(dict.fromkeys(c.name.lower() for c in self.expression.find_all(exp.Column)))


def order_keys(select: Any) -> List[OrderKey]:
    """The ORDER BY keys of *select*: "ORDER BY subject_count" is the
    COUNT(DISTINCT o.subject) that the select calls subject_count."""
    order = select.args.get("order")
    if order is None:
        return []
    aliased = {e.alias.lower(): e.this for e in select.expressions if isinstance(e, exp.Alias)}
    keys = []
    for ordered in order.expressions:
        key = ordered.this
        if isinstance(key, exp.Column) and not key.table and key.name.lower() in aliased:
            key = aliased[key.name.lower()]
        keys.append(OrderKey(key, bool(ordered.args.get("desc"))))
    return keys


def row_limit(select: Any) -> Optional[int]:
    """The number of *select*'s LIMIT; None without one or when it is no number."""
    limit = select.args.get("limit")
    if limit is None:
        return None
    value = limit.args.get("expression") or limit.this
    return int(value.this) if isinstance(value, exp.Literal) and value.is_int else None
