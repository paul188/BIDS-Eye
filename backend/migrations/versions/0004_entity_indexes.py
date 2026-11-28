"""Indexes for searching files by their BIDS filename entities.

Revision: 0004
After:    0003

A scan term such as "dir-AP" is a condition on the files' entity
(backend/vocabulary/bids_entities.py, entity_condition):

    LOWER(o2.other_entities ->> 'direction') = 'ap'

Without an index that parses the JSON of every file: 38 seconds on the
2.7 million rows of production (2026-11-27). PostgreSQL uses an expression
index only for the very expression it was built on, so these are on the
expression entity_condition writes (tests/vocabulary/test_bids_entities.py).
Only rows that have the entity are indexed. direction and acquisition are
the entities most files carry; the others are searched without an index.

Only adds indexes, built CONCURRENTLY: searches keep running meanwhile.
"""

from __future__ import annotations

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

INDEXED_ENTITIES = ("direction", "acquisition")


def _name(entity: str) -> str:
    return f"ix_bids_objects_entity_{entity}"


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("SET statement_timeout = 0")
        for entity in INDEXED_ENTITIES:
            op.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_name(entity)} ON bids_objects "
                f"(LOWER(other_entities ->> '{entity}'), dataset_id) "
                f"WHERE other_entities ->> '{entity}' IS NOT NULL")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for entity in INDEXED_ENTITIES:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_name(entity)}")
