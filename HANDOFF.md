# BIDS-Eye handoff: state on 2026-11-28

For Paul, and for any Claude session, picking the work up on another machine.
Written from GitHub, the Claude memory notes of the sessions on the old machine, and a
read-only look at the live server on 2026-11-28. It replaces the handoff of 2026-11-18.
No server addresses or commands are in this file (public repository, pre-commit hook).
They are in the private archive described in [section 8](#8-moving-to-another-machine).

## 1. Where things stand

- **`main` (`cedd1c8`)** holds all finished work. Every branch with work in it was merged
  on 2026-11-27/28, except PR #14 (section 4). The backend suite passes on `main`
  (2362 passed, 65 skipped), and ruff and mypy are clean.
- **Live is `754cdcd`** (backend and frontend; database at alembic revision 0003).
  Everything merged after it is **not deployed**. See section 3 for the list and what
  a deploy has to do.
- **No LLM provider has credit** (Gemini and Anthropic both refused for billing on
  2026-11-27). A live search that is not in the interpretation cache fails at the planner
  step, and no benchmark that calls a model can run. Nothing merged since 2026-11-27 has been measured end to end.
- **Server flags:** uncertain tier, paper search and interpretation cache on; re-ranker off
  on purpose (it sends dataset text to an external model and needs Paul's OK). Model
  choice, timeouts and fallbacks live in `backend/llm/llm_config.yaml` (profiles), not in
  `.env`.
- **Backups:** production has no automated database backups yet, only hand-made dumps.
  Borg is installed on the server and waits for a storage box that Paul orders
  (`deploy/backup/`).

## 2. How a search works now

1. **Planner** (`backend/llm/preprocess/`): question → `QueryPlan`. `prompt_context.py`
   adds what the catalogue knows about the question's words (listed shorthands, stem
   collisions); `intent.py` calls the model (profile `intent`).
   - Participant groups are **criteria plus a quantifier** (`ParticipantGroup`:
     `criteria` of attribute diagnosis / sex / age / handedness, and `some`, `all` or `none`).
     "Only female" is quantifier `all`. Exclusions are `none` groups. Plans with the old flat
     fields still load. (#58)
   - Order and number of results are one field, `ordering` (sort key, direction, limit),
     with a closed list of sort keys. (#52)
2. **Vocabulary** (`backend/vocabulary/`): plan terms → codes, plus text fallbacks. Where
   each field is stored is on the field itself (`services/schema/fields.py`,
   `CatalogueField` with `Stored(table, alias, column)`, #56). BIDS filename entities
   (`dir-AP`, `acq-highres`) become conditions on the files (`bids_entities.py`, #53).
3. **Compiler** (`backend/services/translate/participant_query_compiler.py`): builds the
   SQL in code for about 80% of recorded questions. It also writes `ORDER BY` and `LIMIT`
   now. It declines sidecar fields, licences, authors/funding/institutions, same-subject
   scans and contradictions; those go to the **SQL model** (`text_to_sql.py`). Its
   few-shot examples are chosen from the families the plan's fields show
   (`prompt/example_families.py`, #48), not from a label the model picks.
4. **Rewriter** (`sql_rewriter.py`): concept keys, catalogue value check, patient counts
   (`vocabulary/cohort_count.py`, recognised on the SQL tree, #29), summary tables.
5. **Text search** (`services/fulltext/`), **uncertain tier** (`services/uncertain/`),
   **evidence and match reasons** (`services/evidence/`, `services/results/`).
6. **Search** (`services/search.py`, `search_runner.py`): paging, relaxed re-run,
   self-correction, interpretation cache (every search kept in `query_log`).

## 3. Not deployed yet

Merged after the live `754cdcd`, in merge order:

| PR | What | Deploy note |
|---|---|---|
| #55 | Typed name parameter in the metadata UPDATE | already in the crawler image |
| #48 | Few-shot example families derived from the plan; `QueryFamily` enum removed; `derivative_query` examples | |
| #52 | `QueryPlan.ordering`; compiler writes ORDER BY / LIMIT | |
| #54 | Ordering gold requests (`benchmarks/demand/gold/ordering/`) | benchmark only |
| #45 | A pipeline name (fMRIPrep, MRIQC) in an unknown term keeps only the derivatives lookup | qx06 53 → 634 |
| #50 | Typed `same_meaning` answer | |
| #47 | A participant sex without a species means people | qx09 precision 80% → 92% |
| #53 | BIDS filename entities searched in the files | **migration 0004** (two partial indexes, built concurrently): ship it with the backend, which refuses to start below the newest revision |
| #56 | Each field's column on the field itself | |
| #27 | Recall benchmark harnesses import the current backend | benchmark only |
| #22 | Optional columns and card labels in `tables.py`; species conditions in the schema prompt from `criteria.py` | species prompt text changed |
| #29 | Patient count read from the SQL tree; `group_labels.py` | |
| #57 | `prompt_context.py` split out of `intent.py` | prompts byte-identical |
| #58 | Participant groups as criteria and a quantifier | **plan schema changed**: rescore the gold set |
| #59 | `question_words` keeps whole Unicode words; word-form question benchmark | |

Every one of these resets the interpretation cache once (it keys on a hash of the
source). Before deploying, delete on the server every path deleted or renamed between
`754cdcd` and the new commit. Then check that the backend starts at migration 0004.

**What to measure first, once a provider has credit:**

1. The 70 gold requests, against the 2026-11-18 numbers (recall about 67–71%,
   precision 63–68%). #58 changes what the planner fills, and #22 changes the species prompt.
2. The ordering gold requests (`benchmarks/demand/gold/ordering/README.md`). The planner
   prompt has no line about ordering of its own: check that the model fills `ordering`.
3. The word-form questions with the model step (`benchmarks/word_forms/questions/run.py
   --model`), old against new prompt (the prompts were shortened in #40, unmeasured).

## 4. Open work

- **PR #14, remove unused material.** Left open: it predates the folder restructure, and
  most of it happened elsewhere (lottery, dev compose, `pg_hba.conf`, two scripts). What it
  would still delete is a decision for Paul, because later work moved these files and kept them:
  - the May benchmarks `benchmarks/eval_rag.py`, `eval_pipeline.py`,
    `run_query_eval.py`, `benchmarks/ground_truth/`, `benchmarks/results/`;
  - `ingest/openneuro-crawler/crawl_job.sh`, `fetch_accessions.py`, `ingest/scan_crawl_errors.py`;
  - `backend/vocabulary/tools/auto_map_unmapped.py`, `patch_metadata_index.py`.

  Either redo it as a fresh PR on `main` with that list, or close it.
- **Crawler image:** `ingest/openneuro-crawler/Dockerfile` does not copy
  `backend/database/`, so the post-crawl word-forms step fails ("No module named
  'database'").
- **Crawler:** some S3 exports have no `dataset_description.json` (ds004560). They should
  fall back to the GraphQL indexer, as access-denied datasets do. About 285 datasets wait
  for the catch-up crawl.
- **Read-only database role for generated SQL:** the backend connects as a superuser, and
  generated SQL runs as it. The `bids_*` table allowlist in `query_guard.py` is live. A
  role with SELECT on the catalogue only (`SET LOCAL ROLE` in `apply_limits()`) is still
  to do. It needs a server step with Paul's OK.
- **Word-form review:** 804 mined word-form directions wait for Paul's labels (review page
  in the Claude memory note `word-form-question-benchmark`). The Claude judges agreed with
  Paul on only 78 of 120, so their "yes" is not trusted.
- **lottery.bidseye.xyz:** the DNS record still points at the server. Delete it to retire
  the name fully.

## 5. To-do list

### Search quality, in order of gain per effort

1. **Per-request fixes for the biggest gold losses** (2026-11-18 run): qf23 dir-AP/PA
   (addressed by #53: 86% recall in a replay), qx09 only female (#47, #58), qp07 isolated
   words (3%), qf13 natural images (10%), qf07 (5%), qf12 (0). Fix them one request at a
   time, not with new query families.
2. **Plan fields for what has no field yet:** stimulus content, same-session pairing,
   file presence, phenotype measures, "datasets like dataset X".
3. **Keep alternatives; no datatype+suffix OR:**
   - a rare own phrase drops the validated alternatives too (qf03);
   - `(datatype='anatomical_mri' OR suffix='mp2rage')` means "any anatomical scan" and is
     empty when negated (qf27);
   - "CT" resolves to PET through the PET-CT synonyms (qf40).
4. **Uncertain-tier evidence:** keep the `:AB` scope when a diagnosis or task is promoted
   from README or paper text (second-tier results were wrong for 9 of 16 Alzheimer's and
   5 of 5 ADHD).
5. **Measurement protocol:** 3 uncached runs per variant, per-request means, a paired
   sign-flip test, a lockbox of new requests from `query_log`.
6. **Few-shot pool lint:** examples that break the SQL prompt's own rules (24 diagnoses
   without a text fallback, 13 healthy filters, 11 HAVING counts).
7. **Hard/soft rule in the plan:** a preference field that never becomes WHERE.
8. **relevance_score:** one count per term, not a sum over all matched codes.
9. **Concept-to-task lists:** language without gambling/delay-discounting/food-choice
   tasks; BCI with oddball and P300; emotion recognition is not emotion induction.
10. **Vocabulary lookup:** retrieve top k, then choose with a "none" option, instead of
    the WRatio partial match ("task-based" → a release-from-suppression task, qf29).
11. **Compiler coverage:** numeric acquisition parameters (sidecar values) are the next
    candidate. The only frequent repair on the model path is the JSON number cast.
12. **Later:** vote over several plan samples, per-criterion LLM check of soft criteria,
    fitted ordering score, re-ranker on.

### Code quality, from the evidence-package review (not done)

- The "stemmed vector @@ query AND exact vector @@ guard" SQL is written twice by hand
  (`ProbablePaperOnlyEvidence.sql`, `WeakenedSearch.sql`), unescaped. Make it one helper
  in `fulltext/`.
- `DEFAULT_SCORES` could be a property of `Strength`.
- `ConfirmedByDefaultEvidence` is two kinds in one class.
- `TEXTS` means two different things in two classes.
- Proposed, not started: `services/asked.py`, read each condition once for chips, tier and
  evidence (now three separate if-chains).

### Known errors, not changed

- Vocabulary: "type 2 diabetes" → type 1, "pain" → fibromyalgia, "palsy" → Parkinson's.
- "at least 60 female participants" is sometimes a count of women, sometimes the subject count.
- Postgres runs with default memory settings.
- 278 public OpenNeuro datasets are missing from the catalogue.
- The gold labels were judged by LLMs; no person has read them.

## 6. How to measure a change

- **Free replay** (no model calls; for compiler, rewriter, text search and tier changes):
  `benchmarks/demand/replay_run.py --recording <recording.json>`. Recordings come from
  `benchmarks/demand/record_run.py` and are kept privately (section 8).
- **Gold run** (about 140 model calls): `benchmarks/recall/eval_recall.py run`, then
  `score … --truth benchmarks/demand/gold/requests_truth.json --held-out-file <empty file>`.
  It runs on the server against the live database, with the candidate code mounted.
- **Ordering gold:** `eval_recall.py run --queries benchmarks/demand/gold/ordering/requests.yaml
  --split all`, then `benchmarks/demand/gold/ordering/score_ordering.py <run.json>`.
- **Word forms:** `benchmarks/word_forms/questions/run.py --evidence
  benchmarks/word_forms/questions/evidence.json` (offline; `--model` needs a provider).
- **Tests:** CI runs ruff, mypy, the backend suite, the frontend checks, gitleaks and
  Semgrep. Locally:
  ```bash
  git submodule update --init BIDS-SQL
  W=$PWD; export PYTHONPATH="$W/BIDS-SQL:$W/backend/vocabulary:$W/backend/llm:$W/backend"
  export METADATA_DB_HOST=localhost METADATA_DB_USER=ci METADATA_DB_PASSWORD=ci METADATA_DB_NAME=ci \
    SESSION_SECRET=ci-only-not-a-secret-ci-only-not-a-secret GITHUB_CLIENT_ID=ci \
    GITHUB_CLIENT_SECRET=ci ALLOW_ANY_GITHUB_USER=true PYTHONDONTWRITEBYTECODE=1
  (cd backend && python -m pytest -q -p no:cacheprovider tests)
  ruff check backend && mypy
  ```
  Install the dependencies under `constraints.txt` (`PIP_CONSTRAINT`, as CI does). Run
  the suite twice before pushing: a few async tests depend on order.

## 7. Working rules Paul has set

- **Labels first:** an uploader label on OpenNeuro decides; README text is only the fallback.
- **Every venue of matching:** each criterion matches through every venue that fits
  (recorded fields and ranges, metadata, text, semantic), and the card says which one
  matched.
- **Backend twins:** every frontend guard gets a matching backend check.
- **Measure, then deploy promptly:** measure first, then deploy when Paul asks. Deploy
  from a commit (`git archive`), never from a working tree. Tag a rollback image, ship a
  migration together with its backend, and append the deploy to the server's deploy log.
- **Ask first:** database writes, migrations, server changes and anything that sends
  dataset text to an external model need Paul's explicit OK in the conversation.
- **No server details in tracked files:** results, logs, reports, research notes and
  server helper scripts stay untracked (`.git/info/exclude`).
- **Parallel sessions:** several Claude sessions run at once. Each works in its own git
  worktree on its own branch, opens a PR, and never switches the branch of a shared
  checkout or stages another session's files.
- **Gold files:** change and commit gold answers only when Paul asks.

## 8. Moving to another machine

Everything that is code is on GitHub. Clone it, then:

```bash
git clone https://github.com/paul188/BIDS-Eye.git && cd BIDS-Eye
git submodule update --init && git config core.hooksPath .githooks
```

**Only on the old machine, packed in a private archive**
(`~/bids-eye-handoff-2026-11-28.tgz`; its `README.txt` says how to unpack). It contains
server details, so keep it private:

| What | In the archive |
|---|---|
| Claude memory notes of all sessions (server access, deploy history, Paul's rules, what each session left open) | `claude-memory/` |
| Private files: research reports, research notes, benchmark results and recordings, the server helper script | `untracked/` |
| Local-only git branches: snapshots of the old main checkout from 2026-11-27 (they hold private reports, so they are not on GitHub) | `local-branches.bundle` |
| Uncommitted edits in the old main checkout (superseded family-enum edits; the deletion of this file) | `patches/main-checkout-uncommitted.patch` |

Not in the archive: the SSH key and SSH config entry for the server (`~/.ssh/`), and the
local `.env` files. Copy those by hand.

**For Claude on the new machine:** put the memory notes where Claude Code keeps a
project's memory (`~/.claude/projects/<repository path with / replaced by ->/memory/`).
Its `MEMORY.md` is the index. Notes that name worktree paths under `/home/lub/Desktop/`
describe the old machine; the branches they name are on GitHub (merged) unless this file
says otherwise.
