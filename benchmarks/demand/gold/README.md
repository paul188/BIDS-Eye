# Gold answers for the 70 real requests

Question and answer pairs for the 70 requests in `../requests.yaml` (44 forum
posts, 12 requests for a search feature, 14 selection rules from papers). For
each request: what makes a dataset a correct answer, and which datasets of the
catalogue are correct answers. Built from the production database, our crawl
files and openneuro.org. Snapshot of both: 2026-11-14.

## Files

| File | What |
|---|---|
| `requests_gold.yaml` | Per request: question, original quote, `relevant_if`, `partially_if`, how it was decided, what could not be checked, counts, and the answer list when it has at most 40 datasets |
| `requests_truth.json` | Answer lists in the format of `../../recall/ground_truth.json` (69 scored requests under `queries`, one under `not_scored`) |
| `judgments.jsonl` | One line per decided pair: label, source (`facts`, `presumed`, `judged`, `reviewed`), reason |
| `population.jsonl` | Who the participants of each dataset are (healthy only, patients and healthy, patients only, special group, not human, unclear) and where that comes from |
| `pipeline/` | Scripts, judge instructions, judge output and review corrections |

## How the answers were decided

| Method | Requests | How |
|---|---|---|
| exact | 17 | A rule on recorded facts: files, sidecar fields, participants.tsv, sessions, licence |
| population | 4 | A rule on recorded facts plus the participant classification |
| mixed | 44 | Recorded facts where they decide; every other candidate read by a judge |
| judged | 5 | Every candidate read by a judge |

Candidates per request (`build_pools.py`): every dataset that the facts allow
and whose text (ours or OpenNeuro's) matches one of the request's patterns,
plus what OpenNeuro's own search returns. LLM judges (Claude) read 3,839
question and dataset pairs and classified the participants of 616 datasets,
each from an evidence card with the recorded facts and the full README
(`JUDGE_INSTRUCTIONS.md`, `POPULATION_INSTRUCTIONS.md`).

Review after judging (`corrections.jsonl`, 101 lines, always wins):

- the 7 cases where facts and judge disagreed (the judges were right in all);
- the points the judges raised, one rule each (below);
- every `relevant` label of the 35 requests with fewer than 30 answers;
- every `unsure` label of the judges;
- 60 `not_relevant` labels, 45 of them datasets the live search had returned: none changed;
- the 37 datasets in the top 20 of the live search that were not candidates:
  1 relevant, 4 partially relevant, 1 unsure, 31 not relevant;
- electrode types for the stereo-EEG request read from `channels.tsv`.

A dataset of the catalogue that was not a candidate counts as not relevant
and is listed as such in `requests_truth.json`, so precision counts everything
the search returns. `n_judged` says how many were actually read.

## Rules set in review

- Movie fMRI whose film is not named or described: `unsure` for the talking-people request (qf19).
- Passive mismatch-negativity and roving oddballs are cognitive ERP experiments, so partial for the BCI request (qp13). Frequency tagging counts as SSVEP.
- Rest - task - rest (qf12): rest and task in one session with a single rest run is no longer partial; almost every fMRI dataset has that.
- A raw dataset whose preprocessed outputs are published as sibling accessions (PAFIN, Penn LEAD) counts like one with an OpenNeuro Derivatives companion (qf11, qf18, qf25).
- An fMRIPrep derivative that holds only masks, reference images, transforms and confounds is partial for "preprocessed fMRI" (ds007088, ds005472, ds005481; checked in OpenNeuro's file listing).
- Animated film is neither natural video nor photographs (qf13).
- Resting ASL without resting BOLD is `unsure` for "resting-state fMRI" (qf15).
- iEEG recorded in labelled non-REM sleep is partial for the consciousness request (qf32).
- Female only (qx09): animals do not count; sex recorded for part of the subjects or as a bare number code is not evidence.
- Uploader labels decide before README text.
- Healthy adults with a T1w image (qp02): where no ages are recorded and the text never mentions children, adults are presumed. This turned 193 `unsure` labels into `relevant`; they carry source `presumed`.

## Limits

- Partially relevant lists are complete only among the candidates. For requests whose partial class is wide (qp13: any event-related EEG experiment) lenient recall is indicative only.
- qf12: the order of runs in a session is known only where a README or paper states it.
- "Healthy" by rule means diagnoses, an uploader label, or no clinical term anywhere in the text. In an audit of 45 datasets classified by that last rule a judge confirmed 38; none had patients; 7 could not be confirmed (4 without any description, 2 special groups, 1 baboon dataset).
- 410 pairs stay `unsure` (text or facts do not say). The scorer counts a returned `unsure` dataset as not a hit.
- 278 public OpenNeuro datasets are missing from the catalogue. The ones that would be correct answers are listed per request as `relevant_not_in_catalogue` (445 pairs) and do not count in recall. For exact requests this is filled only where OpenNeuro's summary or file listing decides (qx07, qx11, qf26, qf34).
- Four requests have no correct answer in the catalogue (qf20, qf24, qf28, qp09). qp12 ("normative databases") is kept but not scored.
- The judges are LLMs. A person has not read the labels.

## Scoring a run

```bash
: > /tmp/none.txt
python3 benchmarks/recall/eval_recall.py score <run.json> \
  --truth benchmarks/demand/gold/requests_truth.json --held-out-file /tmp/none.txt
```

After a change to the search, list the top results that were never read and
judge them before trusting precision (they count as not relevant until then):
`python3 tools/unjudged.py <run.json>` (in the work folder) lists them.

First run, live search at commit 5f91230 (2026-11-14, re-ranker off):

| Group | Requests | Recall | Recall@20 | Precision | Returned nothing |
|---|---|---|---|---|---|
| Forum requests | 44 | 47% | 48% | 58% | 12 |
| Search-feature requests | 12 | 65% | 69% | 78% | 1 |
| Selection rules from papers | 13 | 43% | 50% | 45% | 2 |
| All | 69 | 50% | 53% | 60% | 15 |

Averages are over the requests where the figure exists (recall: 65 requests
with an answer; precision: 54 that returned something). 14 of the 65 requests
with an answer found none of it. Per request: `../results/2026-11-15_gold_score_5f91230.txt`.

## Rebuilding

The scripts run in the work folder `benchmarks/demand/work/gold/` (not in
git), next to the fact dumps they read. `pipeline/` holds a copy of the scripts
and of the inputs that cannot be regenerated.

1. Facts: run `f1.sql` … `f5.sql` read-only on the production database into `f*.jsonl`.
2. OpenNeuro: `fetch_on.py` (catalogue), `fetch_trees.py` (file listings), `walk_preproc.py` (`_desc-preproc_bold` files), `channel_types.py`.
3. `build_pools.py`, then `cards.py` and `population.py` write the judge batches.
4. Judges append to `judged/<batch>.jsonl` through `tools/next_cards.py` and `tools/append.py`.
5. `assemble.py` merges facts, participants, judge labels and `corrections.jsonl`; `export_gold.py` writes this folder.

A new catalogue snapshot changes the facts, so steps 1, 3 and 5 are enough
when no new datasets need judging; new candidates show up in
`tools/status.py --todo`.
