"""Record which OpenNeuro snapshot each dataset was indexed from.

Revision: 0003
After:    0002

The crawler compares these with the API's latest snapshot and indexes the
files of a dataset again only when the snapshot changed, the indexing code
changed (indexer_version), or the last indexing is old
(ingest/openneuro-crawler/snapshot_plan.py). Rows indexed before this revision
have NULLs and are indexed once more.

Only adds columns: the old backend keeps running while it is applied. The
lock timeout makes it fail fast instead of queueing searches behind the
ALTER TABLE while a long query holds the table; run it again then.
"""

from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("snapshot_tag", "TEXT"),
    ("snapshot_hexsha", "TEXT"),
    ("files_indexed_at", "TIMESTAMPTZ"),
    ("indexer_version", "INTEGER"),
)


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("ALTER TABLE bids_datasets " + ", ".join(
        f"ADD COLUMN IF NOT EXISTS {name} {type_}" for name, type_ in _COLUMNS))


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("ALTER TABLE bids_datasets " + ", ".join(
        f"DROP COLUMN IF EXISTS {name}" for name, _ in _COLUMNS))
