"""
RAG/text_formula.py
-------------------
The free-text part of a search as a small logical formula.

"datasets related to drawings or art" put "drawings or art" into
description_search, and that string became one text_match() term: "or" is a
stopword, and every word of a term must match, so the search required both
"drawings" and "art" (2026-11-17). The same happened to "alcohol consumption
or alcohol dependence questionnaires". A negation ("memory but not sleep")
lost its "not" the same way and searched for sleep.

description_search is therefore read as a formula:

    formula   phrase | formula OR formula | formula AND formula
              | NOT formula | ( formula )
    phrase    words next to each other; every word must match (as before)

Operators are the words or, and, not, but (= and), and/or (= or), except and
excluding (= and not), neither / nor (= not / and not) in any case, and the
signs | & !; "either" and "both" are dropped. NOT binds tightest, then AND,
then OR. "without" is not an operator ("with and without MCI" means both):
the intent prompt writes AND NOT for it. A comma separates
alternatives when OR appears between the same parentheses ("drawings,
paintings or sculptures"); otherwise it separates nothing. Parentheses that
do not nest properly are ignored.

compile_formula() turns a formula into SQL over text_match() placeholders
(RAG/text_terms.py):

  - an OR of phrases (and of ANDs of phrases) is one placeholder with
    alternatives, text_match('drawings | art'), so the rank, the paper text
    and the uncertain tier treat it like any other text search;
  - phrases ANDed together stay one term, text_match('structural functional'),
    which is what a description search without operators always gave;
  - the other conditions of a top-level AND are separate conditions, joined
    by AND without outer parentheses, so a NOT stays a hard condition of its
    own (backend/services/uncertain_tier.py never weakens a NOT);
  - anything else is written out with AND / OR / NOT and parentheses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

from text_terms import format_text_match, split_words

# ── Tree ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Phrase:
    text: str


@dataclass(frozen=True)
class Not:
    child: "Node"


@dataclass(frozen=True)
class And:
    children: tuple


@dataclass(frozen=True)
class Or:
    children: tuple


Node = Union[Phrase, Not, And, Or]

# ── Reading ──────────────────────────────────────────────────────────────────

_OR, _AND, _NOT, _OPEN, _CLOSE, _COMMA = "OR", "AND", "NOT", "(", ")", ","
_KEYWORDS = {"or": [_OR], "and/or": [_OR], "|": [_OR], "||": [_OR],
             "and": [_AND], "but": [_AND], "&": [_AND], "&&": [_AND],
             "not": [_NOT], "!": [_NOT], "neither": [_NOT],
             "nor": [_AND, _NOT], "except": [_AND, _NOT], "excluding": [_AND, _NOT],
             # "either X or Y", "both X and Y": the operator says it all.
             "either": [], "both": []}
_TOKEN_RE = re.compile(r"\(|\)|,|;|&&|\|\||[&|!]|[^\s(),;&|!]+")


def _tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        if raw == ";":
            raw = ","
        if raw in (_OPEN, _CLOSE, _COMMA):
            tokens.append(raw)
        else:
            tokens.extend(_KEYWORDS.get(raw.lower(), [raw]))
    if not _nested(tokens):
        tokens = [t for t in tokens if t not in (_OPEN, _CLOSE)]
    return _commas(tokens)


def _nested(tokens: List[str]) -> bool:
    """True when every ')' closes an earlier '(' and none stays open."""
    depth = 0
    for token in tokens:
        depth += (token == _OPEN) - (token == _CLOSE)
        if depth < 0:
            return False
    return depth == 0


def _commas(tokens: List[str]) -> List[str]:
    """A comma is OR where OR appears between the same parentheses, else nothing."""
    out = list(tokens)
    stack: List[List[int]] = [[]]          # comma positions per open level
    has_or: List[bool] = [False]
    for i, token in enumerate(tokens):
        if token == _OPEN:
            stack.append([])
            has_or.append(False)
        elif token == _CLOSE and len(stack) > 1:
            _settle(out, stack.pop(), has_or.pop())
        elif token == _COMMA:
            stack[-1].append(i)
        elif token == _OR:
            has_or[-1] = True
    while stack:
        _settle(out, stack.pop(), has_or.pop())
    return [t for t in out if t is not None]


def _settle(out: List[Optional[str]], positions: List[int], or_level: bool) -> None:
    for i in positions:
        out[i] = _OR if or_level else None


class _Parser:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> Optional[str]:
        token = self.peek()
        self.i += 1
        return token

    def formula(self) -> Optional[Node]:
        parts = [self.conjunction()]
        while self.peek() == _OR:
            self.take()
            parts.append(self.conjunction())
        return _or(parts)

    def conjunction(self) -> Optional[Node]:
        parts = [self.negation()]
        while self.peek() not in (None, _OR, _CLOSE):
            if self.peek() == _AND:
                self.take()
                continue
            parts.append(self.negation())
        return _and(parts)

    def negation(self) -> Optional[Node]:
        if self.peek() == _NOT:
            self.take()
            child = self.negation()
            if child is None:
                return None
            return child.child if isinstance(child, Not) else Not(child)
        return self.primary()

    def primary(self) -> Optional[Node]:
        token = self.peek()
        if token == _OPEN:
            self.take()
            inner = self.formula()
            if self.peek() == _CLOSE:
                self.take()
            return inner
        words = []
        while self.peek() not in (None, _OR, _AND, _NOT, _OPEN, _CLOSE):
            words.append(self.take())
        if not words:
            # A stray operator or closing parenthesis: skip it.
            if token is not None and token != _CLOSE:
                self.take()
            elif token == _CLOSE:
                self.take()
            return None
        return Phrase(" ".join(words))


def _or(parts: Sequence[Optional[Node]]) -> Optional[Node]:
    flat: List[Node] = []
    for part in parts:
        if part is None:
            continue
        flat.extend(part.children if isinstance(part, Or) else [part])
    flat = list(dict.fromkeys(flat))
    if not flat:
        return None
    return flat[0] if len(flat) == 1 else Or(tuple(flat))


def _and(parts: Sequence[Optional[Node]]) -> Optional[Node]:
    flat: List[Node] = []
    for part in parts:
        if part is None:
            continue
        flat.extend(part.children if isinstance(part, And) else [part])
    flat = list(dict.fromkeys(flat))
    if not flat:
        return None
    return flat[0] if len(flat) == 1 else And(tuple(flat))


def parse_formula(text: Optional[str]) -> Optional[Node]:
    """The formula in *text*; None when nothing is left to search."""
    parser = _Parser(_tokens(text or ""))
    parts = []
    while parser.peek() is not None:
        start = parser.i
        parts.append(parser.formula())
        if parser.i == start:          # a stray ')' at the top
            parser.take()
    return _and(parts)


def has_operators(node: Optional[Node]) -> bool:
    """True when *node* says more than "all these words": an OR or a NOT."""
    if node is None or isinstance(node, Phrase):
        return False
    if isinstance(node, (Or, Not)):
        return True
    return any(has_operators(child) for child in node.children)


# ── Changing the phrases ─────────────────────────────────────────────────────

def map_phrases(node: Optional[Node], change: Callable[[str], str]) -> Optional[Node]:
    """*node* with every phrase passed through *change*; empty phrases drop out."""
    if node is None:
        return None
    if isinstance(node, Phrase):
        text = " ".join((change(node.text) or "").split())
        return Phrase(text) if split_words(text) else None
    if isinstance(node, Not):
        child = map_phrases(node.child, change)
        return Not(child) if child is not None else None
    children = [map_phrases(child, change) for child in node.children]
    return _or(children) if isinstance(node, Or) else _and(children)


def phrases(node: Optional[Node], positive: bool = True) -> List[str]:
    """The phrases of *node* under an even (positive) or odd number of NOTs."""
    found: List[str] = []

    def walk(n: Optional[Node], sign: bool) -> None:
        if n is None:
            return
        if isinstance(n, Phrase):
            if sign == positive:
                found.append(n.text)
        elif isinstance(n, Not):
            walk(n.child, not sign)
        else:
            for child in n.children:
                walk(child, sign)

    walk(node, True)
    return found


def conjuncts(node: Optional[Node]) -> List[Node]:
    """The parts of a top-level AND, phrases merged into one (see compile_formula)."""
    if node is None:
        return []
    parts = list(node.children) if isinstance(node, And) else [node]
    words = [p.text for p in parts if isinstance(p, Phrase)]
    rest = [p for p in parts if not isinstance(p, Phrase)]
    return ([Phrase(" ".join(words))] if words else []) + rest


# ── Writing ──────────────────────────────────────────────────────────────────

def alternatives(node: Node) -> Optional[List[str]]:
    """*node* as text_match alternatives (an OR of ANDs of phrases), else None."""
    if isinstance(node, Phrase):
        return [node.text]
    if isinstance(node, And) and all(isinstance(c, Phrase) for c in node.children):
        return [" ".join(c.text for c in node.children)]
    if isinstance(node, Or):
        found: List[str] = []
        for child in node.children:
            sub = alternatives(child)
            if sub is None:
                return None
            found.extend(sub)
        return found
    return None


def placeholder(node: Node) -> Optional[str]:
    """One text_match() for *node* when it is an OR of ANDs of phrases, else None."""
    found = alternatives(node)
    return format_text_match(found) if found else None


def _sql(node: Node, top: bool = False) -> Optional[str]:
    single = placeholder(node)
    if single or isinstance(node, Phrase):
        return single
    if isinstance(node, Not):
        inner = _sql(node.child)
        return f"NOT {inner}" if inner else None
    if isinstance(node, And):
        parts = [_sql(part) for part in conjuncts(node)]
        joiner = " AND "
    else:
        parts = [_sql(part) for part in node.children]
        joiner = " OR "
    parts = [p for p in parts if p]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    joined = joiner.join(parts)
    return joined if top and joiner == " AND " else f"({joined})"


def compile_formula(node: Optional[Node]) -> Optional[str]:
    """SQL for *node* over text_match() placeholders; None when nothing is searchable.

    A top-level AND is not parenthesised: each part is a condition of its own.
    """
    if node is None:
        return None
    if isinstance(node, And):
        parts = [_sql(part) for part in conjuncts(node)]
        parts = [p for p in parts if p]
        return " AND ".join(parts) if parts else None
    return _sql(node, top=True)


def render(node: Optional[Node]) -> str:
    """*node* written back as a formula (logs, chips)."""
    if node is None:
        return ""
    if isinstance(node, Phrase):
        return node.text
    if isinstance(node, Not):
        inner = render(node.child)
        return f"NOT {inner}" if isinstance(node.child, Phrase) else f"NOT ({inner})"
    joiner = " OR " if isinstance(node, Or) else " AND "
    return joiner.join(render(c) if isinstance(c, (Phrase, Not)) else f"({render(c)})"
                       for c in node.children)
