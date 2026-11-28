"""Every exception class the backend defines derives from ExpectedError
(exceptions/expected.py): a failure the code plans for has its own type, and
a caller that catches ExpectedError never catches a bug."""
from __future__ import annotations

import ast
import builtins
from pathlib import Path

from exceptions.domain import DomainError
from exceptions.expected import ExpectedError

BACKEND = Path(__file__).resolve().parents[1]
BUILTIN_EXCEPTIONS = {
    name for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, BaseException)
}


def _sources():
    for path in sorted(BACKEND.rglob("*.py")):
        relative = path.relative_to(BACKEND)
        # llm/ and vocabulary/ are installed on their own and do not import
        # the backend's exceptions/.
        if relative.parts[0] in {"tests", ".venv", "venv", "llm", "vocabulary"}:
            continue
        yield relative, path.read_text(encoding="utf-8")


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def test_no_exception_class_derives_from_a_builtin_one_but_expected_error() -> None:
    offenders = []
    for relative, source in _sources():
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef) or node.name == "ExpectedError":
                continue
            builtin = [_base_name(b) for b in node.bases if _base_name(b) in BUILTIN_EXCEPTIONS]
            if builtin:
                offenders.append(f"{relative}:{node.lineno} {node.name}({', '.join(builtin)})")
    assert not offenders, "derive these from ExpectedError instead:\n" + "\n".join(offenders)


def test_a_domain_error_is_an_expected_error() -> None:
    assert issubclass(DomainError, ExpectedError)
