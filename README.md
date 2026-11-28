# BIDS-Eye

<p align="center">
  <strong>Natural-language search over BIDS neuroimaging datasets.</strong><br>
  Ask for datasets in plain English and inspect the SQL-backed result set.
</p>

## About

BIDS-Eye is a full-stack neuroimaging dataset discovery tool built around the BIDS specification. It turns informal search requests into structured database queries so researchers can find datasets by diagnosis, task, modality, metadata fields, and other BIDS-aware concepts without hand-writing SQL.

The project is designed around public dataset discovery, with OpenNeuro as the main demonstrator. It can index local datasets, remote datasets, and public DataLad-backed sources, then expose them through a conversational search interface.

At a high level, BIDS-Eye does three things:

1. Ingests BIDS metadata into a relational schema.
2. Resolves user language into canonical database concepts.
3. Generates transparent SQL queries and returns the matching datasets.

## Highlights

| | |
|---|---|
| **Natural-language search** | Ask for datasets in plain English and translate that request into SQL-backed search. |
| **BIDS-aware indexing** | Parse dataset structure, entities, participants, and metadata into database tables. |
| **Remote dataset support** | Index public datasets without requiring a full local mirror of the imaging files. |
| **OpenNeuro demonstrator** | Use OpenNeuro as the primary public-data example for discovery and validation. |
| **Query transparency** | Surface the generated SQL so users can inspect how a result was produced. |
| **Conversation UI** | A Vue 3 + Vite frontend provides a chat-style search experience. |
| **Backend API** | FastAPI endpoints handle auth, query translation, and dataset retrieval. |
| **Extensible architecture** | The same approach can be extended to more public BIDS collections or local deployments. |

## Stack

### Backend

| Layer | Choice |
|---|---|
| Runtime | Python 3.12 |
| API | FastAPI |
| ORM / DB | SQLAlchemy with PostgreSQL-compatible models |
| Search logic | Natural-language to SQL translation with RAG / rule-based resolution |
| Remote indexing | DataLad-based ghost indexing for public repositories |

### Frontend

| Layer | Choice |
|---|---|
| Framework | Vue 3 |
| Build | Vite |
| State | Pinia |
| Language | TypeScript |
| UI | Chat-style dataset search interface |

### Infra

- Dockerized backend and frontend for local development and deployment.
- BIDS-SQL is packaged as a reusable internal module for indexing and query logic.
- OpenNeuro crawler support is included for public dataset ingestion workflows.

## How It Works

The current system follows this flow:

1. A user asks a question in the frontend, for example: "Show me resting-state fMRI datasets with Alzheimer participants."
2. The backend resolves key terms to canonical BIDS concepts and database filters.
3. The query layer generates SQL against the indexed metadata store.
4. The database returns matching datasets, counts, and metadata.
5. The frontend renders the results and exposes the generated SQL for inspection.

For remote datasets, BIDS-Eye can index metadata from public repositories while keeping the imaging content remote. That makes it useful for discovery across large collections without requiring a full download first.

## Current Capabilities

- Search datasets using natural language.
- Resolve common neuroimaging terms to canonical database values.
- Search across local or remotely indexed BIDS metadata.
- Work with OpenNeuro-backed demonstrator data.
- Surface dataset cards with accession IDs, dataset type, subject counts, and validation metadata.
- Show the generated SQL for debugging and transparency.

## Future Direction

BIDS-Eye is built to grow beyond OpenNeuro. The current architecture could support:

- indexing more public BIDS repositories,
- searching across multiple remote collections in one place,
- local deployments for lab-internal datasets,
- local model or offline model integrations for query translation,
- faster query paths for complex filtering and cohort selection.

## Getting Started

### Prerequisites

- Python 3.11+ for backend and indexing tasks
- Node.js 20+ and npm for the frontend
- A PostgreSQL database or other compatible backend configured for `BIDS-SQL`

### Run locally

The repository is split into two main application layers:

- [`backend/`](./backend/) contains the FastAPI app and query service.
- [`frontend/`](./frontend/) contains the Vue search interface.
- [`BIDS-SQL/`](./BIDS-SQL/) contains the reusable indexing and database layer.
- [`backend/services/translate/prompt/`](./backend/services/translate/prompt/) holds the schema and the examples shown to the SQL model, and [`backend/services/translate/sql_rewriter.py`](./backend/services/translate/sql_rewriter.py) post-processes the generated SQL.

One straightforward local setup is:

```bash
pip install -e BIDS-SQL
pip install -e backend/vocabulary -e backend/llm
pip install -e backend

cd backend
uvicorn main:app --reload --port 8000

cd ../frontend
npm install
npm run dev
```

If you prefer Docker, use the repository's compose setup instead of running the services separately.

### Sign-in

Users sign in with GitHub. The backend refuses to start until `.env` (see [`.env.example`](./.env.example)) has
`SESSION_SECRET`, `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, and one of `GITHUB_ORG=<org>` (members only) or
`ALLOW_ANY_GITHUB_USER=true` (every GitHub account).

The session is a signed cookie; the server keeps no list of sessions. What follows from that:

- Organization membership is checked at sign-in only. Someone removed from the organization keeps access until
  their cookie expires, at most `SESSION_DAYS` (7, in [`backend/security/sessions.py`](./backend/security/sessions.py)).
- Signing out deletes the cookie in that browser. A copied cookie stays valid until it expires.
- **To sign everyone out at once**, set a new `SESSION_SECRET` and restart the backend.

## Project Layout

```text
backend/                     FastAPI app and everything it runs at request time
  services/                  Search pipeline, grouped by stage (see services/__init__.py)
  routers/                   Query, auth, and crawler endpoints
  llm/                       Intent parsing (preprocess/) and the shared LLM client
  vocabulary/                Curated vocabulary (value_mappings.yaml) and term-to-code retriever
    tools/                   Offline tools that build or audit the vocabulary
frontend/                    Vue 3 dataset search UI
  src/                       Chat interface, dataset cards, and state management
ingest/                      Offline jobs that fill the database
  openneuro-crawler/         OpenNeuro/DataLad crawler (its own Docker service)
  backfill/                  Re-run crawler steps over already-indexed datasets
BIDS-SQL/                    Shared indexing and database layer (git submodule)
  input_pipeline.py          BIDS dataset discovery and ingestion
  db/                        ORM models and database helpers
benchmarks/                  Search quality benchmarks
deploy/                      What runs the site on the server
  nginx/                     Reverse proxy and TLS configuration
  backup/                    Database backups (systemd timers on the server)
docs/                        Research notes and design plans
```

## Checks

Every push to `main` and every pull request runs [`.github/workflows/ci.yml`](./.github/workflows/ci.yml):
a secret scan (gitleaks, rules in [`.gitleaks.toml`](./.gitleaks.toml)), ruff's pyflakes rules,
the backend tests, Semgrep (new findings only) and the frontend build.

A pre-commit hook stops secrets and server access details before they are committed.
Enable it once per clone:

```bash
git submodule update --init      # BIDS-SQL, needed by the backend and its tests
git config core.hooksPath .githooks
```

Run the backend tests locally:

```bash
pip install -e "backend[test]"
cd backend
METADATA_DB_NAME=x METADATA_DB_USER=x METADATA_DB_PASSWORD=x METADATA_DB_HOST=localhost \
  python -m pytest -q tests
```

`backend/tests/` mirrors the source: `services/<stage>/`, `routers/`,
`vocabulary/`, `llm/`, `ingest/` (the crawler) and `migrations/`;
`search/` holds end-to-end search behaviour that spans several stages, and
the root the tests of the top-level modules. `python -m pytest tests/vocabulary`
runs one part.

The tests that need real SQL are skipped unless `BIDSEYE_TEST_PG_DSN` names an
empty scratch PostgreSQL; `backend/tests/postgres.py` says how to start one. CI
runs them, and prints a coverage table (`--cov`).

## API Surface

The backend exposes endpoints for:

- authentication,
- natural-language dataset queries,
- dataset result retrieval.

See `backend/main.py` and the routers in `backend/routers/` for the current API wiring.

## Benchmarks

The repository includes benchmark and evaluation material for:

- term resolution quality,
- query generation correctness,
- OpenNeuro and public-dataset search behavior,
- remote indexing workflows.

See the `README.md` files and scripts under `backend/vocabulary/`, `benchmarks/`, and `ingest/` for the current evaluation tooling.

## Contributing

Good contribution areas include:

- improving term resolution,
- refining SQL generation and post-processing,
- expanding support for remote repositories,
- adding more public datasets,
- improving the search UI and dataset cards,
- benchmarking query quality on curated neuroimaging examples.

## License

License information has not been finalized yet.
