# Database migrations

Alembic creates and changes every table of the database. Nothing else does:
the backend creates no tables when it starts, it only checks that the
database is at the newest revision and stops when it is not
(`services/schema/revision.py`).

A revision is one file in `versions/`: one change to the schema, with the
revision it follows. The database records the revision it is at in the table
`alembic_version`.

All commands run in `backend/`, against the database of the `METADATA_DB_*`
settings.

## Deploying

Run the migrations as their own step, after building the new image and
before starting it:

    docker compose -f <compose file> run --rm --no-deps backend alembic upgrade head
    docker compose -f <compose file> up -d --no-deps backend

`alembic upgrade head` does nothing when the database is already at the
newest revision. `alembic current` prints the revision of the database,
`alembic history` the list of revisions.

Back up the tables a revision changes first. A revision that only adds
things lets the old backend keep running while it is applied; one that drops
or renames something needs the old backend stopped.

## A database that existed before Alembic

It already has the tables of `0001`, so mark it instead of upgrading it,
once:

    alembic stamp 0001

`0001` is the schema of the production database on 2026-11-26, without two
leftover tables no code uses (`query_translations`,
`bids_datasets_refs_backup_20261113`). `0002` drops those two, so after the
stamp run `alembic upgrade head` like on every deploy. Compare any other old
database with the baseline first (`pg_dump --schema-only` of both).

## A new, empty database

    alembic upgrade head

## Changing the schema

1. Change the model (`models.py`, BIDS-SQL's `db/models.py`) or the
   declaration in `services/schema/tables.py`.
2. Write the revision:

       alembic revision --autogenerate -m "Add query_log.duration_ms"

   Autogenerate compares the models and the extension columns of
   `bids_datasets` with the database, which must be at the newest revision.
   Read what it wrote. It does not write: generated columns (the search
   vectors), expression indexes, and the tables declared only in
   `services/schema/tables.py`. Write those by hand with `op.execute(...)`.
3. `alembic upgrade head` on a scratch database, then
   `python -m pytest tests/test_migrations_postgres.py`.

`CREATE INDEX CONCURRENTLY` cannot run in a transaction; wrap it:

    with op.get_context().autocommit_block():
        op.execute("CREATE INDEX CONCURRENTLY ...")

Two revisions written side by side both follow the same one, and
`alembic upgrade head` then refuses. Join them with `alembic merge heads`.
