"""The schema as it stood before Alembic (0001_baseline.sql).

Revision: 0001
After:    nothing

The production database already has these tables. It is not upgraded to
this revision but marked as being at it: alembic stamp 0001
(migrations/README.md).
"""

from __future__ import annotations

from pathlib import Path

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

_SQL = Path(__file__).with_suffix(".sql")


def upgrade() -> None:
    lines = [line for line in _SQL.read_text().splitlines() if not line.startswith("--")]
    for statement in "\n".join(lines).split(";\n"):
        if statement.strip():
            op.execute(statement.replace(":", r"\:"))


def downgrade() -> None:
    raise NotImplementedError("Nothing is before the baseline; drop the database instead.")
