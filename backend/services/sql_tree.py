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
    """The node denies what stands in it: NOT, NOT LIKE ANY, "= FALSE",
    "<> TRUE", and "IS NULL" of a comparison (of a column it denies nothing)."""
    if isinstance(node, exp.Not) or node.args.get("negate"):
        return True
    if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
        return node.this.find(exp.Predicate, *TEXT_MATCHES) is not None
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


def or_branches(node: Any) -> list:
    """The OR branches of a condition, through parentheses (not through NOT)."""
    while isinstance(node, exp.Paren):
        node = node.this
    if isinstance(node, exp.Or):
        return or_branches(node.left) + or_branches(node.right)
    return [node]


def without_literals(node: Any) -> str:
    """The SQL of *node* with every value replaced by '?': for a log, which
    must not hold what people searched for."""
    return node.copy().transform(
        lambda n: exp.Var(this="?") if isinstance(n, _VALUES) else n).sql(dialect="postgres", comments=False)
