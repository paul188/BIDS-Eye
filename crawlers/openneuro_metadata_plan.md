# Plan: honest prompt schema (step 1), OpenNeuro metadata columns (step 2), deprecated tag

Status on 2026-11-13: code written and tested locally against a copy of the
production `bids_datasets` and `bids_participants` tables, committed on branch
`openneuro-metadata-deprecated-tag`. Rollout result: see the end of this file.

## Why

Production search falls back to free text whenever structured metadata is
missing, and there is almost no free text to search:

| | Production today |
|---|---|
| `description_text` filled | 37 of 1,722 datasets |
| `readme_text` column | does not exist, but the SQL prompt and 11 few-shot examples use it |
| `institutions` filled | 0 datasets, but the prompt and 10 few-shot examples use it |
| datasets with a clinical diagnosis in `bids_participants` | 146 |

## Step 1 — the prompt only mentions columns that hold data

### Mechanism

`backend/services/schema_capabilities.py` inspects `bids_datasets` once at
backend startup. An optional column counts as available when it exists **and**
at least 5% of datasets have a value (`MIN_COVERAGE`). Prompt building uses
that state in three places:

1. `constants.build_schema_ddl(available)` renders the schema; optional columns
   that are not available are left out. This also applies to the prompt of the
   SQL error-correction call.
2. `_pick_examples` skips few-shot examples whose SQL uses an unavailable
   optional column. The examples stay in `few_shot_examples.json`.
3. The disease rule ("always add a free-text fallback") lists the available
   short text columns.

If the inspection fails, every optional column is hidden.

Optional columns: `institutions`, `readme_text`, `dx_status`, `study_domain`,
`study_design`, `age_min`, `age_max`, `size_bytes`.

### Effect

| | Production today | After the step 2 backfill |
|---|---|---|
| optional columns in the schema prompt | none | all except `institutions` |
| few-shot examples usable | 296 of 317 | 307 of 317 |
| disease fallback | `name`, `description_text` | + `dx_status`, `study_domain` |

Nothing has to be edited by hand when a column gets filled later. Run the
backfill, restart the backend, and the column appears in the prompt.

### Decisions worth knowing

- I did not delete prompt text or few-shot examples. Deleting them would have
  to be undone as soon as step 2 lands.
- `readme_text` is **not** part of the disease fallback, even when available.
  A substring match on README text is noisy: "bipolar" mostly matches "bipolar
  channels", "depression" matches "Beck Depression Inventory". The schema
  comment still lets the model use the README for keyword questions, as the
  existing few-shot examples show. Using the README for ranking is step 4.
- The unused constants `SYSTEM` and `EXAMPLE_PAIRS` in `backend/constants.py`
  are removed. Nothing imported them and both described columns and codes that
  do not exist.

## Step 2 — pull README and study metadata from OpenNeuro

### Source

One GraphQL request to `https://openneuro.org/crn/graphql` returns everything,
and 12 datasets fit in one request. No S3 access and no file indexing.

| Column | Type | API field | Datasets with a value (of 1,722) |
|---|---|---|---|
| `readme_text` | TEXT | `latestSnapshot.readme` | 1,463 |
| `dx_status` | TEXT | `metadata.dxStatus` | 586 (464 are "Healthy / Control") |
| `study_domain` | TEXT | `metadata.studyDomain` | 697 |
| `study_design` | TEXT | `metadata.studyDesign` | 411 |
| `age_min`, `age_max` | DOUBLE PRECISION | min / max of `metadata.ages` | 843 |
| `size_bytes` | BIGINT | `latestSnapshot.summary.size` | 1,655 |

53 indexed datasets get nothing because OpenNeuro no longer serves them. They
are marked deprecated instead (see below).

### Do we need to crawl again?

No. The crawler's expensive part is listing S3, downloading every sidecar and
running pybids. None of the new fields come from there. The backfill script
updates one row per dataset and takes about seven minutes for the whole
catalogue. A re-crawl is only needed for things derived from files (tasks,
suffixes, participants.tsv).

### Code

| File | Role |
|---|---|
| `crawlers/migrate_add_openneuro_metadata.sql` | adds the seven columns plus `deprecated_reason`, idempotent |
| `crawlers/openneuro-crawler/remote/openneuro_metadata.py` | fetch, clean, write; shared by everything below |
| `scripts/backfill_openneuro_metadata.py` | fills the columns for datasets that are already indexed |
| `crawlers/openneuro-crawler/remote/openneuro.py` | S3 crawl path: stores the mirror README, then calls the API for the other fields |
| `crawlers/openneuro-crawler/remote/graphql.py` | fallback for restricted datasets, now using fields that exist in the API |
| `crawlers/openneuro-crawler/crawler_service.py` | creates the columns at the start of a crawl cycle |

### Behaviour and decisions

- **Writes never destroy data.** Each column is set with
  `COALESCE(new, old)`. A dataset the API has nothing for keeps what it had.
  The script can be repeated and can be resumed with `--only-missing`.
- **A failed request is not "no data".** Network or server errors skip the
  batch and are counted as `failed_requests`; the script then exits with
  status 1.
- **README length.** The old code kept 8,000 characters (head and tail). The
  text is only searched, never sent to an LLM, so the cap is now 50,000. This
  affects 70 datasets. NUL bytes, HTML tags and embedded images are removed.
- **README links fill `paper_references`** where the dataset declared none.
  The crawler already did this; the backfill does the same. On the copy this
  raised datasets with references from 1,002 to 1,317.
- **Ages** outside 0–120 are ignored. Units are whatever the uploader typed,
  normally years.
- **The crawler writes these columns with plain SQL.** The BIDS-SQL commit this
  repo pins (`a9379ee`) has neither `readme_text` nor `institutions` on the ORM
  model, so the previous `_stamp_dataset` could not run against it.
- **The GraphQL fallback was broken before this change.** Its query asked for
  fields the API does not have (`datasetDOI`, `sex { … }`, `ages { min max }`),
  so every restricted dataset was skipped, and the stub rows lacked the
  required `filename`. Production has zero rows from this path. It now takes
  authors, license, DOI, funding and references from the snapshot description,
  and participants from `summary.subjectMetadata`.

### Deprecated tag

`bids_datasets.deprecated_reason` is NULL for a dataset that is fine upstream.
Otherwise it holds the reason, which the UI shows:

| Upstream situation | `deprecated_reason` | Datasets today |
|---|---|---|
| API: "Dataset … does not exist" | `No longer available on OpenNeuro` | 25 |
| API: "… has been deleted. Reason: …" | `Deleted on OpenNeuro: <their reason>` | 16 |
| API: "You do not have access to read this dataset" | `No longer public on OpenNeuro` | 12 |
| latest snapshot flagged deprecated | `Deprecated on OpenNeuro: <their reason>` | 4 |

57 of 1,722 datasets get a mark. One of them, ds007630, was deleted upstream
for "subject privacy"; the index still holds its participant rows. Consider
deleting that dataset from the database instead of only tagging it.

- The backfill and both crawler paths keep the column current. Unlike the
  metadata columns it is overwritten on every check, so a dataset that comes
  back upstream loses the mark.
- It is only written after a definite answer. A failed request, or an API
  error other than the three in the table, leaves the stored status untouched.
- The API returns it as `deprecated_reason` on each dataset. The backend reads
  it with a separate small query in `routers/query.py`, not through the
  generated SQL, so the SQL pipeline is unchanged and a database without the
  column keeps working.
- `DatasetCard.vue` shows a red "deprecated" tag next to the source tag and the
  reason in a line below the title.
- The column is not shown to the SQL model, and deprecated datasets are neither
  hidden nor ranked lower. Both are one-line changes if wanted later.

### How the new data is stored and indexed

The eight columns are ordinary columns on `bids_datasets`, one value per
dataset. **There is no database index on any of them, on purpose.** The table
has 1,722 rows and Postgres scans all of them for every search anyway.
Measured on the copy, without any index:

| Filter | Time |
|---|---|
| `readme_text ILIKE '%parkinson%'` (all READMEs) | 87 ms |
| disease fallback over name, description, `dx_status`, `study_domain` | 8 ms |
| `size_bytes <= 5 GB AND age_min < 18` | 2 ms |

A whole request takes seconds because of the two LLM calls, so none of this is
noticeable. Indexes matter on `bids_objects` (2.7 M rows), which this change
does not touch. The README scan is the one filter that would profit from an
index, and step 4 replaces it anyway.

What is missing is a full-text index, and that is about capability, not speed.
`ILIKE '%depress%'` is substring matching: no stemming, no ranking, and it
matches inside unrelated words ("aging" matches "imaging"). Step 4 adds a
`tsvector` column over name, README and the study fields with a GIN index, and
ranks by it. The existing GIN indexes on `name` and `description_text` are of
that kind but unused, because the generated SQL uses `ILIKE`.

### Verified locally

Against a throwaway Postgres 16 holding production's schema, `bids_datasets`
(1,722 rows) and `bids_participants` (75,728 rows); `bids_objects` was not
copied.

- Backfill refuses to run before the migration and names the missing columns.
- Migration runs twice without error.
- Full backfill: 1,666 rows updated, 53 unavailable upstream, 0 failed
  requests, 6 min 45 s (a second full run took 3 min 30 s).
- Deprecated marks: 57 datasets, split as in the table above; replaying the
  saved API answers gives identical marks.
- The API lookup for the tag works against the real table and is skipped when
  the column does not exist.
- The frontend type-checks and builds with the new tag.
- Second run from the saved records leaves every row byte-identical.
- An empty API record does not erase stored values.
- Backend startup detects seven columns as available, hides `institutions`,
  and finds the `deprecated_reason` column.
- Both crawler paths write correctly using the pinned BIDS-SQL model.
- 30 backend tests pass. `test_query_flow` was failing on `main` because it
  called `query_datasets` without the `_auth` argument the router requires; I
  updated the call.

Disease filter as the prompt builds it, datasets returned:

| Term | Before | After | Gained |
|---|---|---|---|
| epilepsy | 15 | 34 | 19 |
| Parkinson's | 21 | 27 | 6 |
| Alzheimer's | 7 | 10 | 3 |
| ADHD | 4 | 7 | 3 |
| dyslexia | 4 | 6 | 2 |
| 13 terms together | 102 | 141 | 39 |

New filters: 509 datasets are 5 GB or smaller; 125 have a participant under 18
and 96 one over 65 at dataset level.

## Production rollout (Hetzner, `/home/BIDS-Eye`)

Order: back up → ship files → migrate → backfill → rebuild → verify. Each step
is safe to stop after. Until the rebuild the running backend keeps its current
prompt; the only difference is that SQL mentioning `readme_text` stops failing
once the column exists.

```bash
H=HetznerServer_BidsEye
C="docker compose -f docker-compose-hetzner.yml"
PSQL='docker exec -i bids_postgres sh -c "psql -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -v ON_ERROR_STOP=1"'

# 0) Disk is 94% full; 17 GB of it is Docker build cache. Free it before building.
ssh $H 'docker builder prune -f && df -h / | tail -1'

# 1) Keep the one existing column the backfill touches, and tag the running image.
ssh $H "$PSQL" <<'SQL'
CREATE TABLE IF NOT EXISTS bids_datasets_refs_backup_20261113 AS
  SELECT id, paper_references FROM bids_datasets;
SQL
ssh $H 'docker tag bids-eye-backend bids-eye-backend:pre-metadata; docker tag bids-eye-frontend bids-eye-frontend:pre-metadata'

# 2) Ship the files (the server tree is updated by copying, as before).
rsync -avR \
  backend/constants.py backend/main.py backend/schemas.py backend/routers/query.py \
  backend/services/text_to_sql.py backend/services/schema_capabilities.py \
  backend/tests/test_schema_capabilities.py backend/tests/test_openneuro_metadata.py \
  backend/tests/test_query_flow.py \
  frontend/src/api/index.ts frontend/src/components/DatasetCard.vue \
  crawlers/migrate_add_openneuro_metadata.sql crawlers/openneuro_metadata_plan.md \
  crawlers/openneuro-crawler/crawler_service.py \
  crawlers/openneuro-crawler/remote/openneuro.py \
  crawlers/openneuro-crawler/remote/graphql.py \
  crawlers/openneuro-crawler/remote/openneuro_metadata.py \
  scripts/backfill_openneuro_metadata.py \
  $H:/home/BIDS-Eye/

# 3) Migration: adds eight empty columns, takes milliseconds.
ssh $H "cd /home/BIDS-Eye && $PSQL < crawlers/migrate_add_openneuro_metadata.sql"

# 4) Backfill in a one-off container of the current backend image
#    (it has SQLAlchemy, asyncpg and the METADATA_DB_* variables).
RUN="$C run --rm --no-deps -v /home/BIDS-Eye:/repo:ro backend python /repo/scripts/backfill_openneuro_metadata.py"
ssh $H "cd /home/BIDS-Eye && $RUN --dry-run --limit 24"     # look first
ssh $H "cd /home/BIDS-Eye && $RUN"                          # about 7 minutes

# 5) Rebuild and restart backend and frontend (a few seconds of downtime).
ssh $H "cd /home/BIDS-Eye && $C build backend frontend && $C up -d backend frontend"
```

### Verify

```bash
# Coverage. Expected: readme 1463, dx 586, domain 697, design 411, ages 843,
# size 1655, deprecated 57
ssh $H "$PSQL" <<'SQL'
SELECT count(readme_text) readme, count(dx_status) dx, count(study_domain) domain,
       count(study_design) design, count(age_min) ages, count(size_bytes) size,
       count(deprecated_reason) deprecated
FROM bids_datasets;
SQL

# Startup log. Expected: seven columns "shown to the SQL model", institutions "hidden".
ssh $H 'docker logs bids_backend 2>&1 | grep "Optional dataset column" | tail -8'

# The container runs the shipped code.
ssh $H 'cd /home/BIDS-Eye && for f in backend/constants.py backend/services/schema_capabilities.py backend/services/text_to_sql.py; do
  [ "$(md5sum < $f)" = "$(docker exec bids_backend md5sum < /app/$f)" ] && echo "MATCH $f" || echo "DIFF  $f"; done'
```

Then in the UI: "datasets with epilepsy patients" should return clearly more
than before (15 → 34 at SQL level), and "fMRI datasets no larger than 5 GB"
should return a filtered list instead of everything. A search that returns
ds001354 ("PD De Novo…", try "Parkinson resting state") should show the red
"deprecated" tag on that card.

### Roll back

```bash
# Code: go back to the images tagged in step 1.
ssh $H "docker tag bids-eye-backend:pre-metadata bids-eye-backend && docker tag bids-eye-frontend:pre-metadata bids-eye-frontend && cd /home/BIDS-Eye && $C up -d backend frontend"

# Data: the new columns are harmless when unused. To undo the one change to an
# existing column:
ssh $H "$PSQL" <<'SQL'
UPDATE bids_datasets d SET paper_references = b.paper_references
FROM bids_datasets_refs_backup_20261113 b WHERE b.id = d.id;
SQL
```

Restore the previous versions of the copied files from git if the server tree
should match the old image again.

## Afterwards

- **Keeping it fresh.** New snapshots and edited metadata appear upstream over
  time. Re-running the backfill monthly costs seven minutes. It adds and
  updates values; it does not clear a value that was removed upstream.
- **The HPC crawl database.** If the next production refresh comes from a dump
  of the HPC database, run the migration and the backfill there before
  dumping. New crawls with this code fill the columns on their own.
- **BIDS-SQL model.** Adding the eight columns to `BIDSDataset` in the BIDS-SQL
  repo would let `create_all` build them on a fresh database. Until then the
  crawler's `ensure_columns()` and the migration file cover it.
- **`scripts/backfill_readme.py`** is superseded. It needs
  `BIDSDataset.readme_text`, which the pinned model does not have.
- **Deprecated datasets are still returned by search**, now with a tag. Decide
  later whether to rank them last or drop them from the index.
- **`institutions`** stays hidden until `scripts/backfill_institutions.py` has
  run on production.
- **`.gitmodules`** points BIDS-SQL at `github.com/paul88/…`; the repository
  lives at `paul188`. A fresh clone cannot initialise the submodule.
- **Not yet using the new text:** the `description_search` and vocabulary-miss
  paths still search `name` and `description_text` only. Pointing them at the
  README belongs with full-text search and ranking (step 4), where noisy
  matches can be ranked low instead of filtered in.
