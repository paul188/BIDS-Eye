"""Drop two tables no code uses.

Revision: 0002
After:    0001

query_translations held stored translations until query_interpretations
replaced it (57 rows on production, the newest from 2026-11-15).
bids_datasets_refs_backup_20261113 is a copy of bids_datasets.paper_references
taken before a rewrite on 2026-11-13 (1,721 rows). A new database has
neither; there this revision does nothing.
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS query_translations")
    op.execute("DROP TABLE IF EXISTS bids_datasets_refs_backup_20261113")


def downgrade() -> None:
    raise NotImplementedError("The rows are gone; restore the two tables from a backup.")
