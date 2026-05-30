<div align="center">
  <img src="./assets/clinsight-logo.png" alt="ClinSight" width="320" style="margin-bottom: 1rem" />
  
  <p style="font-size: 1.2em; margin: 0.5rem 0 1.5rem 0; color: #666;">
    <strong>Longitudinal clinical intelligence for multimorbid metabolic-syndrome patients.</strong><br/>
    Answer <em>"What matters now?"</em> for a complex patient in under 60 seconds — before the visit.
  </p>

  <p>
    <img src="https://img.shields.io/badge/status-pre--alpha-orange" alt="Pre-alpha" />
    <img src="https://img.shields.io/badge/scope-fullstack-blue" alt="Full stack" />
    <img src="https://img.shields.io/badge/React-19-149ECA?logo=react&logoColor=white" alt="React 19" />
    <img src="https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white" alt="TypeScript" />
    <img src="https://img.shields.io/badge/Vite-bundler-646CFF?logo=vite&logoColor=white" alt="Vite" />
    <img src="https://img.shields.io/badge/Tailwind-CSS-06B6D4?logo=tailwindcss&logoColor=white" alt="Tailwind" />
    <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12" />
    <img src="https://img.shields.io/badge/Anthropic-Claude-D97757?logo=anthropic&logoColor=white" alt="Anthropic Claude" />
    <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker Compose" />
    <img src="https://img.shields.io/badge/pnpm-10-F69220?logo=pnpm&logoColor=white" alt="pnpm 10" />
    <img src="https://img.shields.io/badge/license-TBD-lightgrey" alt="License TBD" />
  </p>

  <table style="margin: 2rem auto; border: none;">
    <tr style="border: none;">
      <td style="border: none; padding: 0.25rem 0.75rem;"><a href="#about">About</a></td>
      <td style="border: none; padding: 0.25rem 0.75rem;"><a href="#highlights">Highlights</a></td>
      <td style="border: none; padding: 0.25rem 0.75rem;"><a href="#stack">Stack</a></td>
      <td style="border: none; padding: 0.25rem 0.75rem;"><a href="#getting-started">Getting started</a></td>
      <td style="border: none; padding: 0.25rem 0.75rem;"><a href="#project-layout">Project layout</a></td>
    </tr>
    <tr style="border: none;">
      <td style="border: none; padding: 0.25rem 0.75rem;"><a href="#pipeline">Pipeline</a></td>
      <td style="border: none; padding: 0.25rem 0.75rem;"><a href="#benchmarks">Benchmarks</a></td>
      <td style="border: none; padding: 0.25rem 0.75rem;"><a href="#api-surface">API surface</a></td>
      <td style="border: none; padding: 0.25rem 0.75rem;"><a href="#conventions">Conventions</a></td>
      <td style="border: none; padding: 0.25rem 0.75rem;"><a href="#roadmap">Roadmap</a></td>
    </tr>
  </table>
</div>

---

## About

ClinSight is a longitudinal clinical-intelligence platform for
internal-medicine physicians treating patients with **diabetes**,
**hypertension**, **CKD**, **obesity**, **dyslipidemia**, and
**polypharmacy**. It turns fragmented patient records into a structured,
source-grounded view of *what matters now* — flagged labs, current
medications, longitudinal trends, and rule-based medication-safety
findings — so the clinician walks into the room oriented in under a
minute.

This repository is a **full-stack monorepo**: a FastAPI backend that
orchestrates the synthetic-data pipeline and serves patient + safety
data, a React frontend that renders the dashboard, and a medication
safety agent that combines declarative rules with LLM-assisted
explanations. Synthetic patients are generated through a Synthea
pipeline and normalized into clinical trajectories the frontend can
consume.

> **Mandatory evidence trail** — every clinical claim is rendered through
> a sourced reference so the doctor can trace any value back to the
> underlying record. If we can't source it, we don't render it.

## Highlights

| | |
|---|---|
| **Patient list & dashboard** | React 19 SPA with overview, trends, timeline, medications, and signals tabs. |
| **Synthetic data pipeline** | Synthea → clinical trajectory normalization → synthetic letters/PDFs/tables → structured re-extraction → medication-safety findings. |
| **Medication safety agent** | `medication_agent/agent.py` evaluates declarative rules (`rules.json`, `drug_classes.json`, `biomarkers.json`) and uses Anthropic Claude to write clinician-readable rationales constrained by curated knowledge passages. |
| **FastAPI backend** | Patient + safety endpoints, in-memory job registry, and per-step pipeline orchestration via subprocess. |
| **PDF / OCR extraction** | Text-layer first, layout/raw fallback, OCR for image-only scans. Used by `extract_from_reports.py`. |
| **Benchmark harness** | Compares extracted output against ground-truth clinical JSON (`benchmarks/evaluate_extraction/`) plus per-format metrics scaffolding. |
| **Dockerized stack** | `docker compose up --build` brings the backend on `:8000` and the frontend on `:3000` with the repo mounted into the backend container. |
| **CI** | `.github/workflows/ai-code-review.yml` runs Claude Code Review on every PR. |

## Stack

### Backend

| Layer | Choice |
|---|---|
| Runtime | ![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white) |
| API | ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white) ![Uvicorn](https://img.shields.io/badge/Uvicorn-standard-0A9EDC) |
| Validation | ![Pydantic](https://img.shields.io/badge/Pydantic-2.8-0083FF) |
| LLM | ![Anthropic Claude](https://img.shields.io/badge/Anthropic-Claude-D97757?logo=anthropic&logoColor=white) |
| Data | ![pandas](https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white) |
| PDF | ![fpdf2](https://img.shields.io/badge/fpdf2-2.8-4285F4) |
| HTTP | ![httpx](https://img.shields.io/badge/httpx-0.27-00AA00) |
| Tests | ![unittest](https://img.shields.io/badge/unittest-stdlib-2E7D32) |

### Frontend

| Layer | Choice |
|---|---|
| Build | ![Vite](https://img.shields.io/badge/Vite-bundler-646CFF?logo=vite&logoColor=white) ![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white) |
| UI | ![React](https://img.shields.io/badge/React-19-149ECA?logo=react&logoColor=white) ![Tailwind](https://img.shields.io/badge/Tailwind-CSS-06B6D4?logo=tailwindcss&logoColor=white) ![shadcn/ui](https://img.shields.io/badge/shadcn-ui-000000?logo=shadcnui&logoColor=white) ![Radix](https://img.shields.io/badge/Radix-primitives-4F46E5) |
| Data | ![TanStack Query](https://img.shields.io/badge/TanStack-Query-FF4154) ![Zod](https://img.shields.io/badge/Zod-3.22-0d47a1) |
| Charts | ![Recharts](https://img.shields.io/badge/Recharts-2.10-62B5EA) |
| Routing | ![React Router](https://img.shields.io/badge/React-Router-CA4245?logo=reactrouter&logoColor=white) |
| Package manager | ![pnpm](https://img.shields.io/badge/pnpm-10-F69220?logo=pnpm&logoColor=white) |
| Serving | ![nginx](https://img.shields.io/badge/nginx-1.27-009639?logo=nginx&logoColor=white) |

### Infra

- **Docker** + **Compose v2** — backend Python 3.12-slim image, frontend multi-stage Node 22 → nginx image.
- Backend container mounts the repository at `/workspace` so pipeline scripts and patient files are shared with the host.

## Getting started

### Prerequisites

- **Docker** with **Compose v2** (recommended path)
- *Optional, for local dev:*
  - **Node.js** ≥ 20 and **pnpm** ≥ 10 (frontend)
  - **Python** ≥ 3.10 (backend / pipeline scripts)
- **Anthropic API key** (only needed for LLM-assisted PDF extraction and the medication agent's narrative output)

### 1. Clone

```bash
git clone https://github.com/paul188/ClinSight.git
cd ClinSight
```

### 2. Configure environment

The backend container reads the root `.env` via Compose. Copy the
example and fill in any secrets (most importantly `ANTHROPIC_API_KEY`):

```bash
cp backend/.env.example .env
# then edit .env and add ANTHROPIC_API_KEY=...
```

### 3. Start the stack

```bash
docker compose up --build
```

This brings up:

- **Backend** → <http://localhost:8000> (FastAPI / Uvicorn)
- **Frontend** → <http://localhost:3000> (nginx serving the Vite build)

Useful smoke endpoints:

- <http://localhost:8000/health>
- <http://localhost:8000/api/safety-report>
- <http://localhost:8000/api/jobs>

### 4. Run the backend locally (optional)

```bash
cd backend
python -m pip install -r requirements.txt
cd ..
uvicorn backend.app.main:app --reload --port 8000
```

### 5. Run the frontend locally (optional)

```bash
cd frontend/Frontend_ClinSight
pnpm install
pnpm dev
```

The Vite dev server proxies `/api` to `http://localhost:8000`, so the
backend must be running for live data.

### 6. Run the tests

```bash
python -m unittest discover -s tests -t . -v
```

The suite covers pipeline argument wiring (unit) and the FastAPI app
against temporary filesystem fixtures (integration).

## Project layout

```
backend/                         FastAPI app, orchestration, job runner
  app/
    main.py                      FastAPI entrypoint, CORS, route wiring
    config.py                    Settings dataclass driven by CLINSIGHT_* env vars
    pipeline.py                  PipelineRunner — runs steps in-process or via subprocess
    jobs.py                      In-memory JobManager
    generate_reports.py          Synthetic letters / PDFs / tables generator
    extractors/                  extract_clinical_main, extract_reports_main
  Dockerfile                     python:3.12-slim image
  requirements.txt               FastAPI, Pydantic, Anthropic, pandas, fpdf2, httpx
  .env.example                   CLINSIGHT_* path defaults

frontend/                        React 19 + Vite + Tailwind app
  Frontend_ClinSight/            Vite project root (managed as a nested project)
  Dockerfile                     Multi-stage Node 22 → nginx 1.27 build
  nginx.conf                     SPA-routing nginx config

medication_agent/                Rule-based + LLM-assisted safety agent
  agent.py                       Entry point — evaluates rules, calls Claude for rationales
  rules.json                     Declarative safety rules
  drug_classes.json              RxNorm-grouped drug classes
  biomarkers.json                Biomarker reference set
  knowledge_sources.json         Curated guideline passages used as the only reasoning source
  safety_report.json             Most recent run output (also served by the API)

evaluation/                      OCR benchmark suite
  generate_ocr_metrics.py        Generate character recovery metrics from scanned PDFs
  _reporting.py                  Shared visualization utilities
  assets/
    evaluate_ocr_scan_to_table/  Generated benchmark SVGs and data

benchmarks/                      Legacy benchmark infrastructure (deprecated)
  evaluate_extraction/           Compares extracted JSON to ground truth
    evaluator.py
  evaluation_metrics/
    evaluate_input_to_json/      PDF / TXT / table → JSON metrics
    evaluate_ocr_scan_to_table/  OCR-on-scans table-extraction metrics
  assets/                        Benchmark visualization SVGs

synthea_output/                  Generated artifacts (not committed in full)
  json/                          Raw Synthea exports
  clinical/                      Normalized clinical trajectories
  reports/                       Synthetic letters, PDFs, CSVs
  extracted/                     Structured re-extraction results
  metadata/                      Generation metadata

tests/
  unit/                          Pipeline-runner / package / PDF-extraction unit tests
  integration/                   FastAPI integration tests against tmpdir fixtures

.github/workflows/
  ai-code-review.yml             Claude Code Review on PRs

docker-compose.yml               backend (:8000) + frontend (:3000)
.dockerignore                    Excludes synthea/, node_modules, dist, caches
ClinSight_demo_patient_Maria_Keller (1).json   Bundled demo patient (~341 KB)
safety_report.md                 Snapshot safety report (~54 KB) for reference
```

## Pipeline

The end-to-end synthetic-data flow:

1. **`synthea/`** generates raw FHIR-style patient records (external generator).
2. **`extract_clinical.py`** (`backend/app/extractors/`) normalizes Synthea JSON into compact clinical trajectories.
3. **`backend/app/generate_reports.py`** synthesizes physician letters, PDFs, and tabular reports from those trajectories.
4. **`extract_from_reports.py`** re-extracts structured data from the synthetic reports — text-layer first, layout/raw fallback, OCR for image-only scans.
5. **`medication_agent/agent.py`** evaluates declarative safety rules and uses Claude to draft clinician-readable rationales constrained by curated knowledge passages.
6. **`evaluation/generate_ocr_metrics.py`** measures OCR character recovery on image-only PDF scans.

Each step is invokable individually through the API or runnable directly
from the repo root. The default pipeline order is wired in
`PipelineRunner.run_pipeline()` (`backend/app/pipeline.py`).

## Benchmarks

BIDS-Eye is evaluated across two complementary dimensions: **RAG term resolution** (how accurately the vocabulary resolver maps informal user terms to canonical BIDS codes) and **end-to-end pipeline performance** (whether the full query chain returns correct results for curated test queries).

### RAG Resolution Performance

The RAG resolver maps informal user terms (e.g. "ADHD", "resting state", "fMRI") to canonical BIDS database codes via a three-tier strategy: exact string match → weighted fuzzy match (RapidFuzz WRatio) → biomedical embedding fallback (BioLORD-2023-C, SapBERT).

Ground truth: **190 term→code entries** across 5 fields (diagnosis, task, datatype, suffix, name), each concept covered by ≥ 2 surface variants (formal, informal, abbreviation).

| Field | N | P@1 | Recall@3 | Pass% | Exact% | Fuzzy% | Embed% | Miss% |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| diagnosis | — | — | — | — | — | — | — | — |
| task | — | — | — | — | — | — | — | — |
| datatype | — | — | — | — | — | — | — | — |
| suffix | — | — | — | — | — | — | — | — |
| name | — | — | — | — | — | — | — | — |
| **Overall** | **190** | **—** | **—** | **—** | **—** | **—** | **—** | **—** |

**Synonym robustness** (P@1 by variant type):

| Variant | N | P@1 | Pass% |
|---|:---:|:---:|:---:|
| formal | — | — | — |
| informal | — | — | — |
| abbreviation | — | — | — |
| compound | — | — | — |

*Run `python benchmarks/eval_rag.py` to populate these tables.*

**Methodology:** P@1 = primary expected code is in the top-1 returned code list. Recall@3 = any expected code appears in top-3 results. Pass% = any expected code appears anywhere in the returned list (lenient; useful when multiple correct codes exist). Resolution path is inferred from which tier returned a result. The target is P@1 ≥ 0.80 overall.

### Pipeline Performance

End-to-end benchmark against **97 curated test queries** from `RAG/run_query_eval.py`, organised by category. Each query is scored on: whether results are returned when expected (Zero%), whether the generated SQL is structurally valid (SQL✓%), whether expected code fragments appear in the SQL (Miss%), and whether forbidden patterns (e.g. ILIKE on canonical-code columns) appear (Spur%).

| Category | N | Pass% | Zero% | Spur% | Miss% | SQL✓% | Fall% |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Participant count | 5 | — | — | — | — | — | — |
| Diagnosis | — | — | — | — | — | — | — |
| Task | — | — | — | — | — | — | — |
| Modality | — | — | — | — | — | — | — |
| Combined | — | — | — | — | — | — | — |
| Developmental | — | — | — | — | — | — | — |
| Edge case | — | — | — | — | — | — | — |
| **Overall** | **97** | **—** | **—** | **—** | **—** | **—** | **—** |

*Run `python benchmarks/eval_pipeline.py --api-url http://localhost:8000` to populate these tables.*

**Metric definitions:**
- **Pass%** — query produced no issues (correct result count, expected SQL fragments present, no forbidden patterns)
- **Zero%** — query returned 0 results when the DB is known to contain matches
- **Spur%** — SQL contained a forbidden pattern (e.g. `ILIKE` on a canonical-code column)
- **Miss%** — SQL was missing an expected code fragment (wrong mapping or dropped filter)
- **SQL✓%** — SQL was generated and syntactically plausible (no API error)
- **Fall%** — result set was suspiciously large (> 500 datasets), suggesting a filter was not applied

**Run benchmarks:**

```bash
# RAG resolver only (no API needed, completes in < 60s)
python benchmarks/eval_rag.py \
  --yaml RAG/value_mappings.yaml \
  --name-index RAG/name_index.json \
  --ground-truth benchmarks/ground_truth/rag_terms.jsonl \
  --out benchmarks/results/rag_metrics.json

# Full pipeline (requires running backend)
python benchmarks/eval_pipeline.py \
  --api-url http://localhost:8000 \
  --out benchmarks/results/pipeline_metrics.json \
  --rate-limit 8
```

Results are written as JSON to `benchmarks/results/` (gitignored).

## API surface

Defined in `backend/app/main.py`:

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health`                          | Liveness + resolved `patients_dir` / `repo_root`. |
| `GET`  | `/patients/index.json`             | Patient index (raw passthrough). |
| `GET`  | `/patients/{filename}`             | Single patient JSON (path-traversal guarded). |
| `GET`  | `/api/patients`                    | Alias of `/patients/index.json`. |
| `GET`  | `/api/patients/{filename}`         | Alias of `/patients/{filename}`. |
| `GET`  | `/api/safety-report`               | Latest medication-safety report. |
| `POST` | `/api/medication-agent/run`        | Run the medication agent and return the resulting safety report. |
| `POST` | `/api/pipeline/run`                | Run the full pipeline (or a subset of `steps`). |
| `POST` | `/api/pipeline/{step}`             | Run a single step (`generate_reports`, `extract_clinical`, `extract_from_reports`, `medication_agent`, `evaluate_extraction`). |
| `POST` | `/api/jobs`                        | Submit a job (kind = `pipeline` or any single step). |
| `GET`  | `/api/jobs`                        | List submitted jobs. |
| `GET`  | `/api/jobs/{job_id}`               | Inspect a single job record. |

CORS is currently fully open (`allow_origins=["*"]`) for development.

## Environment variables

All paths are configurable so the same code works locally and in Docker.
Defaults resolve relative to `CLINSIGHT_REPO_ROOT`.

| Variable | Default (in container) |
|---|---|
| `CLINSIGHT_REPO_ROOT`               | `/workspace` |
| `CLINSIGHT_PATIENTS_DIR`            | `/workspace/frontend/Frontend_ClinSight/public/patients` |
| `CLINSIGHT_SYNTHETIC_JSON_DIR`      | `/workspace/synthea_output/json` |
| `CLINSIGHT_SYNTHETIC_CLINICAL_DIR`  | `/workspace/synthea_output/clinical` |
| `CLINSIGHT_SYNTHETIC_REPORTS_DIR`   | `/workspace/synthea_output/reports` |
| `CLINSIGHT_SYNTHETIC_EXTRACTED_DIR` | `/workspace/synthea_output/extracted` |
| `CLINSIGHT_SAFETY_REPORT_PATH`      | `/workspace/medication_agent/safety_report.json` |
| `CLINSIGHT_RULES_PATH`              | `medication_agent/rules.json` |
| `CLINSIGHT_DRUG_CLASSES_PATH`       | `medication_agent/drug_classes.json` |
| `CLINSIGHT_BIOMARKERS_PATH`         | `medication_agent/biomarkers.json` |
| `CLINSIGHT_KNOWLEDGE_SOURCES_PATH`  | `medication_agent/knowledge_sources.json` |
| `ANTHROPIC_API_KEY`                 | *(required for LLM-assisted extraction and agent narratives)* |

The root `.env` may contain credentials — treat it as sensitive and keep
it out of commits (it is covered by the existing `.gitignore`).

## Conventions

- **Display, not reasoning, on the frontend.** The React app renders the
  patient state, colors values against static reference ranges, sorts,
  and filters. Trend verbs ("rising", "worsening"), interaction
  detection, and gap inference belong to the backend / agent.
- **Mandatory evidence trail.** Every clinical surface routes through a
  sourced reference. If we can't source it, we don't render it.
- **Code keys, not display strings.** Group, route, and join on
  `system:code` (LOINC, RxNorm, SNOMED-CT, CVX), never on display text.
- **Knowledge-bounded LLM output.** The medication agent's prompt
  forbids introducing facts, mechanisms, or thresholds not present in
  the curated `knowledge_sources.json` passages, and forbids dose /
  initiation / discontinuation recommendations.
- **Additive changes.** Keep frontend and backend changes additive when
  possible; don't break existing payload shapes.
- **Tests cover orchestration too.** Integration tests exercise the
  FastAPI app against temporary filesystem fixtures, not just pure
  utilities.

## Roadmap

- [x] FastAPI backend with patient, safety, pipeline, and job endpoints.
- [x] React 19 frontend dashboard with overview, trends, timeline, medications, signals.
- [x] Medication safety agent with declarative rules + Claude-generated rationales.
- [x] Synthea pipeline: extract → reports → re-extract → safety → evaluate.
- [x] Docker Compose stack (backend + frontend).
- [x] PDF extraction with text-layer → layout → OCR fallback.
- [ ] Persist job execution beyond the in-memory registry.
- [ ] Wire frontend signal cards directly to backend-generated patient state.
- [ ] Expand evaluation metrics for OCR and table extraction.
- [ ] Multi-patient cohort views beyond the medication-agent demo.
- [ ] Authentication & deployment story.

## License

License is to be determined. Until one is published, treat this code as
**all rights reserved** by the project owner.
