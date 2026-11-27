# BIDS-Eye handoff — state on 2026-11-18

For Paul (and any Claude session) picking the work up on another machine.
Written from the git branches, the shared `WORKLOG.md`, the Claude memory notes and the live server, read on 2026-11-18.
No server addresses or commands are in this file (public repository, pre-commit hook); they are in the untracked files listed in [section 8](#8-moving-to-another-machine).

## 1. What is live

| Part | Commit | Branch | Since |
|---|---|---|---|
| Backend (incl. `LLM_preprocessor`, `RAG`, `crawlers`) | `4f49468` | `live-merge` (pushed, [PR #1](https://github.com/paul188/BIDS-Eye/pull/1) into `main`, open, mergeable) | 2026-11-17 22:18 UTC |
| Frontend | `a0c122d` | `merged-line` (contained in `live-merge`) | 2026-11-17 21:26 UTC |

`live-merge` holds all sessions' backend work: the cache and query log, the gold-set search fixes, the summary tables, the catalogue value check, the patient count, description alternatives, frontotemporal dementia, decoded participant groups, the server-side search guards, and the general SQL compiler. The older per-feature branches (`interpretation-cache`, `search-fixes-gold`, `patient-count`, `description-alternatives`, `ftd-diagnosis-live`, `query-guards-live`, `merged-line`, `codex/deterministic-cohort-sql`) are all contained in it; `interpretation-cache` only in content (its feature arrived under other commits). `main` on GitHub is 71 commits behind `live-merge` (the local `main` has 48 unpushed commits, all contained in `live-merge`); merging PR #1 brings it up to date.

Server flags in `.env`: `UNCERTAIN_TIER=true`, `TEXT_SEARCH_PAPERS=true`, `INTERPRETATION_CACHE=true`, `RERANK_ENABLED=false` (on purpose: the re-ranker sends dataset text to Gemini and needs Paul's OK), model cascade gemini-2.5-flash → 2.5-pro → 2.5-flash-lite.

Rollback images exist on the server for every deploy (`bids-eye-backend:pre-live-merge` is the one before the current backend, `bids-eye-frontend:pre-one-search-per-page` before the current frontend).

## 2. How a search works now

1. **Intent model** (`backend/llm/preprocess/`): question → `QueryPlan` (diagnoses per group, scans, tasks, age groups, sex, counts, topics, description text, request wishes …).
2. **Vocabulary** (`backend/vocabulary/retriever.py`, `backend/vocabulary/value_mappings.yaml`, `backend/vocabulary/text_terms.py`): plan terms → codes, plus text fallbacks.
3. **Compiler** (`backend/services/translate/participant_query_compiler.py`, since f3fd9bb): builds the SQL in code for about 55 of the 70 gold requests. It declines sidecar fields, licences, ordering, result limits, institutions/authors/funding, negated or same-subject scans, contradictions and file names; those go to the **SQL model** (`backend/services/translate/text_to_sql.py`, few-shot examples in `backend/few_shot_examples.json`).
4. **Rewriter** (`backend/services/translate/sql_rewriter.py`): concept keys, catalogue value check (`translate/value_check.py`, against the values `database_stats/database_stats.py` counts), patient counts, participants-join fix, summary tables (`bids_dataset_facets`, `bids_datasets.n_subjects`).
5. **Text search** (`backend/services/fulltext/`): `text_search.py` turns `text_match()` placeholders into Postgres full text (modes all / short / loose / near / also); `tsquery.py` builds the tsquery strings, `ranking.py` the text rank, `word_frequencies.py` counts matching datasets, `vectors.py` names the stored vectors and their switches.
6. **Uncertain tier** (`backend/services/fulltext/uncertain_tier.py`): "probable" results that miss one fact or match only part of a text condition. `uncertain_tier.py` does the rewrite; `tier_classify.py` decides which states a condition can take, with `tier_text.py`, `tier_sex.py` and `tier_scanner.py` for the weaker forms and `tier_conditions.py` for what they share.
7. **Search** (`backend/services/search.py`, `search_runner.py`; the endpoints in `backend/routers/search.py`): relaxed re-run below 3 results, self-correction, paging, interpretation cache (`interpretations.py`, every search kept in `query_log`).

## 3. How good it is

Gold set: 70 real requests with judged answers (`benchmarks/demand/gold/`, README there). Scored with `benchmarks/recall/eval_recall.py score <run> --truth benchmarks/demand/gold/requests_truth.json --held-out-file <empty file>`.

| Run | Recall | R@20 | Precision | Empty (of 65 scored) |
|---|---|---|---|---|
| Live `live-merge` 4f49468, two runs | 67–68% | 66% | 68% | 8 |
| Compiler line f3fd9bb without the live branches, two runs | 69–71% | 66–68% | 63–66% | 5–6 |
| Unchanged search before 2026-11-17 (per WORKLOG) | about 50% | — | about 56% | 21% of pages |

The live line has about 2.6 points less recall than the compiler line alone. The cause is that it keeps a description as a filter when it has alternatives, even when no dataset text matches (qf19, qf31, qf42). Paul left that as it is; `rare-words` (section 4) addresses part of it.

The benchmark resolves only differences of about 5 points. Repeated runs differ because the model's translation varies, and the cache makes many requests identical between runs.

## 4. Work not yet in `live-merge`

| Where | What | State |
|---|---|---|
| `loose-text`, worktree `../Bids-Eye-loose-text`, **uncommitted** | Point 1 of the to-do list: a phrase is no longer shortened to one broad word (tier relaxation needs a rare remainder; "syndrome" never kept alone; "eating disorders" stays whole; "deep brain stimulation" keeps "deep"). | Measured by replay on merged-line a0c122d: precision 63% → 65%, recall 70% unchanged, 30 results removed (22 not relevant, 8 partial, 0 relevant). 786 tests pass. **Needs porting onto `live-merge`, a replay there, then commit.** |
| `rare-words` (6ca9529, on 2e89bd3) | Rare description words are searched when they are all the question asks ("datasets with astronauts" returned the whole catalogue); a left-out 'dataset text' line is no longer ANDed back in. Includes `text-formula`. | 13 affected gold requests: recall@20 0.34 → 0.52, precision 0.41 → 0.47 (one run each). Not merged, not deployed. |
| `rare-words-only` (384bb28, on 2e89bd3) | The same without the description formula. | Not merged, not deployed. |
| `text-formula` (26696a6, on 049e939) | The description search read as a formula ("drawings or art" is an OR); only the NOT parts are put back when the model leaves them out. | Not merged, not deployed. |
| `description-alternatives` worktree `../Bids-Eye-desc-alt`, **uncommitted** | `_has_filter` and the rare-description fallback. | An earlier draft of `rare-words-only`. Probably superseded; check before deleting. |

The superseded branch `ftd-diagnosis` (985ddfe, the FTD change on another base) can be deleted; the change is in `live-merge`.

### Evidence package review (2026-11-26/27), branch `evidence-review`

Paul asked for a review of `backend/services/evidence/` for readability, structure and how it fits the codebase. Three findings are fixed. The branch `evidence-review` (worktree `../Bids-Eye-evidence-review`, one commit on main f7cb73d) holds only these changes. Main's working tree holds the same changes, uncommitted, on top of other sessions' work.

**Done:**

1. **`reason()` per family.** `PieceOfEvidence` no longer has a `reason()` that silently returns "". `UncertainEvidence` (base.py) declares `reason() -> str` for the probable, possible and contradicted kinds. `ConfirmedEvidence.phrase(row, name, total)` became `reason(row, suffix, total)`. `test_every_kind_has_a_family` checks that every kind is in one of the two families.
2. **Card column names in one place.** `CardColumn` in confirmed.py (`HOLDS`, `COUNT`, `YOUNGEST`, `OLDEST`, `FIELD`, `.of(key)`). `match_reasons.build_sql` writes the names with it, and the evidence reads them with it. `_suffix(chip, branch)` names a branch's columns.
3. **One rule for source groups.** New `Source.LABEL` for the uploader labels that `search_tsv` also holds (`dx_status`, `study_domain`, `study_design`), mapped to the text group in `_GROUPS`. `branches._from_dataset_columns` picks `LABEL` or `FIELD`. `LABELS_IN_TEXT` now comes from the column flag `label_in_text` in `schema/tables.py`. `Source.group` returns None for assumed/not-recorded sources instead of raising. `ClauseEvidence.__post_init__` rejects a weaker piece without a group next to other weaker pieces. `witnesses` no longer has its own special cases. `tests/data/recorded_conditions.json`: 28 entries changed from `field dx_status/study_domain` to `label …`, which is intended.

**When merging:** main's tree renamed `SourceGroup.OWN_TEXT` to `TEXT_FROM_DATASET`, and the branch still uses `OWN_TEXT` (HEAD's name). Main's tree's `match_reasons.py` lives in `results/display/`, the branch's in `results/`. If main's tree is committed with these changes, the branch is redundant. Don't apply it a second time.

**Open from the same review, not done:**

- **The same SQL pattern twice, quoted by hand.** "Stemmed vector @@ query AND exact vector @@ guard" appears in `ProbablePaperOnlyEvidence.sql` and `WeakenedSearch.sql`, both putting `'{search.query}'` in unescaped. Proposed: one helper in `fulltext/`. `sql_literal` exists in `uncertain/conditions.py` and four more times in `translate/`.
- **`DEFAULT_SCORES`** could be a property of `Strength` (the enum rule).
- **`ConfirmedByDefaultEvidence`** is two kinds in one class (column default, absent diagnosis). Split it.
- **`TEXTS` means two things:** keyed by source in `_ConfirmedWhereFound`, by column in `ConfirmedByDefaultEvidence`.
- **Readability:**
  - `ConfirmedInDatasetColumnEvidence.reason`: `(low is not None or high is not None) and (low or high)` is just `low or high`.
  - `ProbableWeakerSearchEvidence.reason` is hard to follow.
  - Two docstrings in probable.py run to about 140 characters.
  - The continuation lines in `branches._from_subquery` are misaligned.
  - `age_span` could be private.
- **The function-local import in `ProbablePaperOnlyEvidence.sql`** looks unnecessary; not tested.
- **SCORING.md** mostly describes `uncertain/conditions.py`. The corroboration benchmark table belongs in `reports/`.

**Proposal discussed, not started (`services/asked.py`):** three places decide on their own what kind a condition is: `chips.describe_condition`, `uncertain/classify.soft_kind` and `evidence/branches.classify_branch`. Proposed: `asked.read(clause) -> Asked`, read once, with all facts per clause (not only the first match in the if-chain) and per OR branch where it matches (Source). Chips, the tier and evidence would decide from that. Steps:
1. Commit the `asked.py`/`display/` state first, since it is untracked in main's tree.
2. Make `soft_kind` and `describe_condition` use `asked.read()`, with chips unchanged on the benchmark SQLs (`dump_chips.py`).
3. Then the more informative chips ("Parkinson's · over 60", "coded or named in the title").

Also first, move `asked.species` (regex on the pattern) and `asked.patients`/`classify._patient_count_parts` (SQL text) onto the tree.

## 5. To-do list

The first 14 items come from the 2026-11-17 search-quality review (`reports/Text to SQL and prompt optimisation.md`; this list replaced the private `SEARCH_TODO.md` on 2026-11-27, with the status of that day); the rest come from the other sessions' open items in `WORKLOG.md`. File and line numbers refer to merged-line a0c122d. Build new backend work on `live-merge`.

### Search quality, in order of gain per effort

1. ~~Loose-text sweep~~ — done, on main since 2026-11-18 (`loose-text-live`).
2. **Keep alternatives; no datatype+suffix OR** (~3 h):
   - a rare own phrase drops the validated alternatives too (`preprocess/` near "not searchable"; qf03 returned 354 results);
   - `(datatype='anatomical_mri' OR suffix='mp2rage')` means "any anatomical scan", and is empty when negated (qf27); the OR is still built on purpose in `backend/llm/preprocess/augment.py` (modality EXISTS, "partially-annotated files"), against the SQL prompt's own rule not to combine the two (checked 2026-11-27);
   - "CT" resolves to PET through the PET-CT synonyms (qf40).
3. ~~Cache hygiene~~ — done: the version hash covers the whole `backend/llm/`, `preprocess/` and `backend/vocabulary/` folders, and only a search that ran as written with a real result is stored (`services/search.py`).
4. ~~Self-correction of tiered SQL~~ — done in cf27e58: the repair is not rewritten again (`sql_rewriter.finish_corrected_sql`, `tests/test_sql_correction.py`).
5. **Uncertain-tier evidence** (~½ day + gold run): keep the `:AB` scope when a diagnosis or task is promoted from README or paper text. In earlier checks, second-tier results were wrong for 9 of 16 Alzheimer's and 5 of 5 ADHD results. Log `uncertain_level` per result in `record_run.py`.
6. **Measurement protocol** (~1 day): 3 uncached runs per variant, per-request means, a paired sign-flip test, a stability metric, and a lockbox of new requests from `query_log`.
7. **Few-shot pool** (~½ day): of 319 examples, 24 have a diagnosis without a text fallback, 13 filter on healthy, 11 use HAVING counts, and 8 put diagnosis and sex/age in one row. Add a lint test against the SQL prompt rules.
8. **Hard/soft rule in the plan** (1–2 days): a preference field that never becomes WHERE; recorded facts go to their columns (`direction`/`desc` entities, channel types).
9. **relevance_score**: one count per term, not a sum over all matched codes (`sql_rewriter.py`).
10. **Concept-to-task lists**:
    - language without gambling, delay-discounting and food-choice tasks, with reading;
    - BCI with oddball and P300;
    - emotion recognition is not emotion induction.
11. **Vocabulary lookup**: replace the WRatio partial match by retrieve-top-k-then-choose with a "none" option (`backend/vocabulary/retriever.py`; "task-based" → a release-from-suppression task, qf29).
12. **Relaxed fallback**: may drop `(EXISTS task OR text)` conjuncts, never a protected condition.
13. **Compiler coverage** (1–2 weeks): extend `participant_query_compiler.py` beyond cohort searches to modality, datatype, recorded facts, task and text conditions; log its decline rate; add the "patients and healthy controls" case.
14. **Later**:
    - vote over 3–5 plan samples for uncached requests;
    - per-criterion LLM check of soft criteria;
    - fitted ordering score;
    - re-ranker on;
    - prompt optimisation only after ≥150 requests and a headroom test.
15. **Suspected, to check**:
    - ~~no `seed` passed to Gemini~~ — confirmed and fixed in PR #15 (`GEMINI_SEED`, default 0). Whether it makes repeated questions give the same SQL is not measured yet: run the demand set twice. The Anthropic fallback tier takes no seed;
    - `_strip_outer_scan_filters` (`sql_rewriter.py`) may remove a real top-level datatype filter.

### Decisions waiting for Paul

- Merge PR #1 into `main`.
- Whether `rare-words` (or `rare-words-only`) goes into `live-merge` and live.
- Keep or drop the own-words-first ranking ('also' mode): it is in `live-merge`, and the held-out check showed no gain.
- Description kept as a filter when it has alternatives (costs about 2.6 recall points, see section 3).

### Open items from other sessions (from WORKLOG, not re-checked)

- **Age groups:**
  - the check whether the intent model fills `age_groups` was not run (`benchmarks/easy`, "agegroups", about 42 model calls);
  - datasets entirely inside the group should rank first;
  - the tier's "ages recorded" check still counts age 0.
- **Participant groups** (`ingest/openneuro-crawler/remote/participant_groups.py`): applied once on production (17 datasets). The crawler does not run on the server, so new datasets are not decoded.
- **Production gaps:**
  - the retriever's database back-check never runs (no `DATABASE_URL`/psycopg2 in the container);
  - neither does the embedding fallback (no `sentence_transformers`).
- **Vocabulary errors seen, not changed:**
  - "type 2 diabetes" → type 1;
  - "pain" → fibromyalgia / zoster neuralgia;
  - "palsy" → Parkinson's;
  - "frontotemporal dementia"-style misses for Lewy body and vascular dementia (the catalogue has no such datasets).
- **Counts:**
  - "at least 20 children with autism and resting-state fMRI" returns 0 (not looked into);
  - "at least 60 female participants" is sometimes a count of women and sometimes the subject count.
- **Database:** Postgres runs with default memory settings (`shared_buffers` 128 MB, `work_mem` 4 MB).
- **Summary tables:** after any crawl or change to `bids_objects`, refresh `bids_dataset_facets` (`remote/dataset_facets.py`, post_crawl does it) and restart the backend.
- **Catalogue coverage:** 278 public OpenNeuro datasets are missing from the catalogue (gold README).
- **Gold set:** the judges are LLMs; no person has read the labels.

## 6. How to measure a change

- **Free replay** (no model calls; for compiler, rewriter, text search and tier changes): `benchmarks/demand/replay_run.py --recording <recording.json>` runs the current code on recorded plans and model SQL. Recordings come from `benchmarks/demand/record_run.py`. The latest recording (merged-line, 70 gold requests) is on the server in the merged-line bench folder. Copy it into `benchmarks/demand/results/` if you want it locally.
- **Full gold run** (about 140 model calls, 15–20 min): `benchmarks/recall/eval_recall.py run --queries benchmarks/demand/demand_first_run.yaml …`, run in a container on the server against the live database with the candidate code mounted. The exact commands are in the untracked `benchmarks/easy/run_on_server.sh` and in `WORKLOG.md`.
- **Compare runs**: `benchmarks/demand/compare_runs.py`.
- **Tests**:
  ```bash
  git submodule update --init
  cd backend
  METADATA_DB_NAME=x METADATA_DB_USER=x METADATA_DB_PASSWORD=x METADATA_DB_HOST=localhost \
    PYTHONPATH=../BIDS-SQL python3 -m pytest -q
  ```
  `live-merge` had 833 tests passing at merge time.

## 7. Working rules Paul has set

- **Labels first:** an uploader label on OpenNeuro (study design, species, diagnosis …) decides. README text is only the fallback.
- **Backend twins:** every frontend guard gets a deliberate backend check (or the endpoint goes).
- **Measure, then deploy promptly:** measure on the gold requests first, then deploy without delay when Paul asks. Deploy from a commit (`git archive`), never from a working tree. Tag a rollback image, check the server's file hashes against the base commit right before shipping, and append a WORKLOG entry.
- **Ask before writing to the database:** writes, migrations and anything that sends dataset text to an external model need Paul's explicit OK in the conversation.
- **No server details in tracked files:** results, logs, reports, research notes, `WORKLOG.md` and `run_on_server.sh` stay untracked.
- **Parallel sessions:** several Claude sessions run in parallel. Each works in its own worktree, never switches the branch of a shared one, stages only its own files, and commits only when Paul asks.
- **Same answer, every search kept:** every paid-for search is kept (`query_log`), and the same question should give the same answer (interpretation cache).

## 8. Moving to another machine

In git (pushed): `live-merge`, `patient-count`, `codex/deterministic-cohort-sql` (6 local commits not pushed), `main` (48 local commits not pushed). Everything live is on GitHub through `live-merge`.

**Only on this machine:**

| What | Where |
|---|---|
| Unpushed branches | `merged-line`, `loose-text`, `rare-words`, `rare-words-only`, `text-formula`, `ftd-diagnosis-live`, `query-guards-live`, `search-fixes-gold`, `description-alternatives`, `interpretation-cache`, `ftd-diagnosis` (only `rare-words`, `rare-words-only` and `text-formula` hold work that is not in `live-merge`) |
| Uncommitted changes | `../Bids-Eye-loose-text`, `../Bids-Eye-desc-alt` |
| Shared session log | `WORKLOG.md` (server and deploy details) |
| Research reports | `reports/`, `research_notes/` |
| Benchmark outputs | `benchmarks/*/results/`, `benchmarks/demand/blame/*.json`, `benchmarks/openneuro_facets/openneuro_results.json` |
| Server helper (server commands) | `benchmarks/easy/run_on_server.sh` |
| Claude memory: server access, deploy history, Paul's rules | `~/.claude/projects/-home-paul-johannssen-Desktop-Bids-Eye/memory/` (on another machine the folder name follows that machine's repository path) |
| Pre-rewrite history backup and hash map | `~/bids-eye-history-rewrite-2026-11-14/` (never push from it) |
| SSH key and SSH config entry for the server | `~/.ssh/` |

Everything in this table except the SSH key, the SSH config and the pre-rewrite backup bundle is packed in `~/bids-eye-handoff-2026-11-18.tgz` (119 MB, made 2026-11-18): a git bundle of every local branch, the two uncommitted changes as patches, the untracked files, the Claude memory notes and the rewrite hash map. Its `README.txt` says how to unpack. It contains server details: keep it private.

On the new machine:

```bash
git clone https://github.com/paul188/BIDS-Eye.git && cd BIDS-Eye
git fetch <unpacked>/all-branches.bundle 'refs/heads/*:refs/remotes/handoff/*'
git submodule update --init && git config core.hooksPath .githooks
git worktree add -b loose-text ../Bids-Eye-loose-text handoff/loose-text
git -C ../Bids-Eye-loose-text apply <unpacked>/patches/loose-text_on_a0c122d.patch
```
