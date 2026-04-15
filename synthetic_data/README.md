# Training Data Generation

This folder contains an end-to-end pipeline for generating Text-to-SQL training
data with stronger diversity controls than the older ad hoc workflow.

The design goals are:

- generate structurally diverse SQL, not just many rows
- allow a few paraphrases per SQL, not unlimited rewrites
- keep timeout / DB-data failures out of the gold set
- make duplicate review part of the normal pipeline

## Files

- `sample_diverse_prompts.py`
  Builds prompt files for Gemini from live DB statistics and an explicit
  diversity plan. It asks for a controlled mix of SQL structures, query
  families, and paraphrase counts.

- `collect_with_gemini.py`
  Sends prompt files to Gemini with your API key and writes raw model responses
  to `result_XXX.txt` files.

- `post_process_pipeline.py`
  Validates Gemini outputs with the updated collector, writes execution-clean
  pairs to the gold dataset, routes blocked queries into a repair bucket, runs a
  question-diversity filter, and then runs duplicate/conflict analysis.

- `question_diversity_filter.py`
  Filters weak paraphrase bundles by collapsing near-duplicate wording, limiting
  the number of kept questions per SQL intent, and flagging likely semantic
  drift for review.

- `SQL_AND_DUPLICATION_GUIDELINES.md`
  The generation policy used by the sampler. This is what keeps the pipeline
  from collapsing into a small number of repeated templates.

## Assumptions

- Gemini access is via API key, not browser automation.
- Set either `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your environment.
- The Gemini collection script uses the official `google-genai` client.
- Your BIDS database is reachable from the machine where you run the sampler and
  post-processing scripts.

## Suggested Flow

### One-command wrapper

```bash
DB_URL="postgresql://user:password@localhost:5429/bids_sql" \
GEMINI_API_KEY=... \
training_data_generation/run_training_data_generation.sh
```

Useful flags:

```bash
training_data_generation/run_training_data_generation.sh --skip-sample
training_data_generation/run_training_data_generation.sh --skip-collect
training_data_generation/run_training_data_generation.sh --skip-post
training_data_generation/run_training_data_generation.sh --allow-main-dataset-overwrite
```

Useful environment variables:

- `N_PROMPTS`
- `PAIRS_PER_PROMPT`
- `MODEL`
- `PROMPT_DIR`
- `RESULT_DIR`
- `REPORT_DIR`
- `GOLD_OUT`
- `REPAIR_OUT`
- `OVERWRITE_RESULTS=1`
- `OVERWRITE_GOLD=1`

Safety note:

- The wrapper refuses to write directly to `training.jsonl` unless you pass
  `--allow-main-dataset-overwrite`.
- The recommended default is to generate into a separate file such as
  `training.generated.jsonl` and merge only after deduplication and review.

### SLURM job script

There is also a ready-to-submit job file at
`training_data_generation/generate_training_data_job.sh`.

That job script:

- loads the required Python/PostgreSQL modules
- activates `BIDS-SQL/venv_hpc`
- snapshots the shared PostgreSQL data directory to node-local storage
- starts a local PostgreSQL server on the compute node
- points the generation pipeline at that local DB automatically

Example:

```bash
export GEMINI_API_KEY=...
sbatch training_data_generation/generate_training_data_job.sh
```

Optional overrides at submit time:

```bash
export GEMINI_API_KEY=...
sbatch --export=ALL,N_PROMPTS=120,PAIRS_PER_PROMPT=24,MODEL=gemini-1.5-pro \
  training_data_generation/generate_training_data_job.sh
```

### 1. Generate prompts

```bash
python training_data_generation/sample_diverse_prompts.py \
  --db-url "postgresql://user:password@localhost:5429/bids_sql" \
  --out-dir training_data_generation/prompts \
  --n-prompts 20 \
  --pairs-per-prompt 18
```

### 2. Send prompts to Gemini

```bash
export GEMINI_API_KEY=...
python training_data_generation/collect_with_gemini.py \
  --prompt-dir training_data_generation/prompts \
  --out-dir training_data_generation/results \
  --model gemini-1.5-pro
```

### 3. Validate and merge the generated data

```bash
python training_data_generation/post_process_pipeline.py \
  --result-dir training_data_generation/results \
  --db-url "postgresql://user:password@localhost:5429/bids_sql" \
  --gold-out training.generated.jsonl \
  --repair-out training.needs_repair.jsonl \
  --report-dir training_data_generation/reports
```

## Diversity Policy

The sampler explicitly pushes the teacher model toward:

- a mix of `JOIN`, `EXISTS`, `NOT EXISTS`, `GROUP BY`, `HAVING`, and ranking
- different query families: filters, absence, statistics, comparisons, JSON,
  sessions, multimodal combinations, metadata
- only a few paraphrases per SQL intent
- real long-tail values from the DB, not just the most common ones
- no repeated SQL skeleton dominance inside a prompt

The post-processing step additionally:

- keeps only a few strong paraphrases per SQL intent
- removes near-duplicate wording inside paraphrase bundles
- flags likely meaning drift instead of silently accepting it

## Notes

- `result_XXX.txt` is the raw Gemini output. Keep it for auditability.
- `training.needs_repair.jsonl` is intentionally separate from the gold set.
- The post-processing step reuses the repo-level duplicate/conflict analysis
  scripts so the final dataset can be reviewed before training.
