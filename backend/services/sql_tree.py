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
from typing import Any, Iterator, Optional

import sqlglot
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

    def function_calls(self, name: str) -> list:
        """Outermost calls to a SQL function named *name*."""
        name = name.lower()
        return [call for call in self.node.find_all(exp.Anonymous, exp.Func)
                if (call.name if isinstance(call, exp.Anonymous) else call.sql_name()).lower() == name
                and not in_subquery(call, self.node)]

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
