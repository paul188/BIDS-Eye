"""Values written into generated SQL as string literals."""

from __future__ import annotations

import re
from typing import Optional

# "\" first in the class by meaning only: each character gets one backslash.
# ":" is no LIKE wildcard, but SQLAlchemy's text() reads ":word" as a parameter.
# It drops a backslash before ":", and LIKE reads "\:" as ":", so both agree.
_LIKE_SPECIAL_RE = re.compile(r"[\\%_:]")


def sql_string(value: str) -> str:
    """*value* as a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def contains_pattern(value: Optional[str]) -> Optional[str]:
    """The LIKE / ILIKE pattern literal for "contains *value*", on one line; None for no value.

    A "%" around the value is a wildcard the model wrote itself and is dropped.
    """
    value = " ".join((value or "").replace("\0", " ").split()).strip("% ")
    if not value:
        return None
    escaped = _LIKE_SPECIAL_RE.sub(lambda m: "\\" + m.group(0), value)
    return sql_string(f"%{escaped}%")


def pattern_words(pattern: str) -> str:
    """The text a LIKE pattern looks for, without its outer wildcards and escapes."""
    inner = re.fullmatch(r"%*((?:\\.|[^\\])*?)%*", pattern, re.DOTALL)
    return re.sub(r"\\(.)", r"\1", inner.group(1) if inner else pattern, flags=re.DOTALL)
