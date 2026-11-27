# Database migrations

Alembic creates and changes every table of the database. Nothing else does:
neither the backend nor the crawler creates a table or a column. The backend
checks at startup that the database is at the newest revision and stops when
it is not (`services/schema/revision.py`).

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

## What the baseline holds

`0001` took over what fourteen SQL files in `crawlers/` used to create. What
those files explained:

- **Search vectors** on `bids_datasets`, generated columns that PostgreSQL
  keeps current. `search_tsv` holds the dataset's text with weights: A name,
  B the study fields, `description_text` and `llm_conditions`, C
  `llm_profile`, D `readme_text`. "/" is indexed as a space, because
  PostgreSQL reads "ADD/ADHD" as one file path. `search_tsv_papers` adds
  `paper_text` at weight D. `search_tsv_exact` and `search_tsv_exact_papers`
  are the same without stemming ('simple' instead of 'english'). The
  expressions are written out in
  `ingest/openneuro-crawler/remote/openneuro_metadata.py` and
  `paper_store.py`; `tests/test_migrations_postgres.py` checks that the
  columns are generated from exactly those.
- **Changing a search vector** means dropping and recreating the column in a
  revision: a table rewrite under an exclusive lock, about two seconds for
  1,700 datasets.
- **Copies that go stale**: `bids_dataset_facets` and
  `bids_datasets.n_subjects` summarise `bids_objects`, and `bids_word_forms`
  lists the word forms of the catalogue. The crawler refreshes them after
  each crawl (`post_crawl.py`).
- **Indexes on `bids_objects`** (2.7 million rows).
  `ix_bids_objects_dataset_subject` and `ix_bids_objects_dataset_filetype`
  serve the subject count and "has files of type X" (median search 797 ms to
  53 ms). The subject count is an index-only read and needs the table
  vacuumed after a large crawl: `VACUUM (PARALLEL 0, ANALYZE) bids_objects`
  (`PARALLEL 0`: the postgres container's 64 MB /dev/shm is too small for
  parallel workers).
- **Sidecar indexes** on `bids_objects.other_entities` (field strength,
  repetition time, sampling frequency, manufacturer, scanner model; needs the
  extension `pg_trgm`). Without them such a search parses the JSON of every
  file: 3 to 28 seconds, with them under 1.5. PostgreSQL uses an expression
  index only for the very expression it was built on, so the number indexes
  are on the cast the SQL rewriter writes (`sql_rewriter.py`,
  `_guard_json_number_casts`) and the scanner model index on its
  `_unify_scanner_model_key`. `tests/test_sidecar_indexes.py` fails when
  they drift apart.

On a database under load: searches hold locks on `bids_datasets`, so an
`ALTER TABLE` there waits. Set `lock_timeout` to a few seconds in the
revision and retry. Build an index on `bids_objects` with
`CREATE INDEX CONCURRENTLY` (below), with `statement_timeout = 0`.

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
3. Name the revision in `services/schema/tables.py` for a column or table
   declared there (`_created_by`, `_migration_column`).
4. `alembic upgrade head` on a scratch database, then
   `python -m pytest tests/test_migrations_postgres.py`.

`CREATE INDEX CONCURRENTLY` cannot run in a transaction; wrap it:

    with op.get_context().autocommit_block():
        op.execute("CREATE INDEX CONCURRENTLY ...")

Two revisions written side by side both follow the same one, and
`alembic upgrade head` then refuses. Join them with `alembic merge heads`.
