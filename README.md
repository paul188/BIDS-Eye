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
| **Backend API** | FastAPI endpoints handle auth, query translation, dataset retrieval, and crawler status. |
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
- Node.js 20+ and pnpm 10+ for the frontend
- A PostgreSQL database or other compatible backend configured for `BIDS-SQL`

### Run locally

The repository is split into two main application layers:

- [`backend/`](./backend/) contains the FastAPI app and query service.
- [`frontend/`](./frontend/) contains the Vue search interface.
- [`BIDS-SQL/`](./BIDS-SQL/) contains the reusable indexing and database layer.
- [`backend/constants.py`](./backend/constants.py), [`value_mappings.py`](./value_mappings.py), and [`sql_expander.py`](./sql_expander.py) provide the shared SQL prompt and post-processing helpers.

One straightforward local setup is:

```bash
cd BIDS-SQL
pip install -e .

cd ../backend
pip install -e .
uvicorn main:app --reload --port 8000

cd ../frontend
npm install
npm run dev
```

If you prefer Docker, use the repository's compose setup instead of running the services separately.

## Project Layout

```text
backend/                     FastAPI app, auth, query routing, crawler status
  services/text_to_sql.py    Natural-language to SQL translation pipeline
  constants.py               Shared schema + prompt constants
  routers/                   Query, auth, and crawler endpoints
frontend/                    Vue 3 dataset search UI
  src/                       Chat interface, dataset cards, and state management
BIDS-SQL/                    Shared indexing and database layer
  input_pipeline.py          BIDS dataset discovery and ingestion
  db/                        ORM models and database helpers
crawlers/openneuro-crawler/  OpenNeuro/DataLad indexing utilities
value_mappings.py            YAML-backed code/label lookup tables
sql_expander.py              SQL post-processor for concept-key expansion
```

## API Surface

The backend exposes endpoints for:

- authentication,
- natural-language dataset queries,
- dataset result retrieval,
- crawler status inspection.

See `backend/main.py` and the routers in `backend/routers/` for the current API wiring.

## Benchmarks

The repository includes benchmark and evaluation material for:

- term resolution quality,
- query generation correctness,
- OpenNeuro and public-dataset search behavior,
- remote indexing workflows.

See the `README.md` files and scripts under `RAG/`, `benchmarks/`, and `crawlers/` for the current evaluation tooling.

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
