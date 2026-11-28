# Recall benchmark

Measures how many of the relevant datasets a search actually returns.

| File | What it is |
|---|---|
| `queries.yaml` | 227 questions from three rounds: 167 for development, 60 held out. Each has a written relevance criterion and the signals used to find candidates. |
| `ground_truth.json` | Relevant / not relevant / unsure datasets per question. |
| `judgments.jsonl` | Every label with its one-line reason, for review. |
| `extra_judgments.jsonl` | Labels for 1,340 datasets outside the pools that the uncertain tier or paper text returned, judged afterwards by the same judge (the cards also showed the dataset's papers). |
| `corrections.jsonl` | Labels changed after review, with reason and who decided; `merge_judgments.py` applies them last. |
| `paired_uncertain_run.py` | Runs each question once and its SQL twice, with and without the uncertain tier. |
| `paired_papers_run.py` | Runs each question once and its SQL twice, with and without paper text in the text filters. |
| `paired_alternatives_run.py` | Runs each question once and its SQL once per variant: without the other words for a description search ("post-mortem" for "dead people"), with them, and with lower limits on how common such a word may be. |
| `compare_alternatives.py` | Compares those variants: recall@20, nDCG@10 and @20 with a bootstrap interval, plus whole-list recall and precision. |
| `build_pool.py` | Builds the candidate pool of each question from a database snapshot. |
| `merge_judgments.py` | Turns judge output into `ground_truth.json` and `judgments.jsonl`. |
| `eval_recall.py` | `run`: sends every question through the deployed search code. `score`: compares the result with the ground truth. |
| `held_out_questions.txt` | The questions set aside for checking finished changes, with the rules for using them. |
| `paired_rank_run.py` | Runs each question once and orders its results twice, with and without the text rank. |
| `compare_pair.py` | Compares the two result files of a paired run: recall@20, nDCG@20, bootstrap interval. |

## How the ground truth was made

Judging all 1,721 datasets for every question is not feasible, so the
benchmark uses pooling: for each question, every dataset that any generous
signal points at (keywords in name, description, README and the uploader's
diagnosis and topic fields; diagnosis and task codes; ages) is a candidate,
and each candidate is judged against the written criterion. Datasets outside
the pool count as not relevant. Three questions are structural (modality and
size only) and are computed exactly.

The 1,514 candidates of the 2026-11-13 snapshot were judged by an LLM (Claude
Sonnet) reading each dataset's metadata and README excerpts, not by a person.
Spot checks found the labels sound, but treat them as a first version:
correct a label in `judgments.jsonl` when you disagree, and widen a pool when
`score` lists returned datasets nobody judged.

## Second round (2026-11-13, 167 questions)

| Source | Questions |
|---|---|
| Real questions from the bidseye.xyz log | 21 |
| Probes for known weaknesses | 18 |
| Explicit AND / OR / NOT logic (9 computed exactly) | 20 |
| bioCADDIE-style "Find <data> on <topic> in <population>" | 15 |
| Research needs written from paper abstracts (DataFinder method) | 15 |
| Generated from READMEs, 8 question types | 78 |

- Pools: keyword signals plus the 20 nearest datasets by embedding
  (`embed_datasets.py`, BAAI/bge-base-en-v1.5), depth-limited to the 30
  keyword matches closest to the question.
- Graded relevance: `partially_if` marks datasets that miss one constraint;
  `score` reports strict and lenient recall.
- Non-human data counts fully only for animal or pure acquisition-method
  questions (`work/apply_species_rule.py`, species from OpenNeuro).
- Reliability: a blind second judge re-read 383 "not relevant" labels, chosen
  mostly among the datasets most similar to their question. It disagreed on
  14 (3.7%) and called 5 (1.3%) relevant, several of them borderline.
- Review of 150 labels (60 relevant, 60 not relevant, 20 partially relevant,
  10 unsure, over 96 questions), each decided without seeing the judge's
  label:
  - by a person: 11 labels, 10 agree with the judge. Too few to stand alone
    (95% interval 62–98%).
  - by a machine reviewer on the review page (all 150), and by a second
    machine reviewer that also read the full README and the dataset's paper
    (all 150). They give the judge's label in 83% and 85% of cases, and agree
    with the judge on relevant-versus-not in 94%. The two reviewers agree
    with each other in 90%.
  - Of 60 "not relevant" labels, the reviewers call 0 and 1 relevant; of 60
    "relevant" labels, 1 is wrong and 4 are downgraded to partially relevant.
    Most disagreement is at the "partially relevant" boundary.
  - The paper changed the reviewer's label in 2 of the 83 items that have
    one, both with an empty README.
  - 15 labels where both reviewers agree with each other and not with the
    judge were corrected (`corrections.jsonl`); 6 of them were "unsure".
- `score` groups results by source and by logic operator.

## Development and held-out questions

A change that was developed by looking at benchmark questions scores better
on those questions than it will for users. So part of the questions is set
aside: the 60 questions listed in `held_out_questions.txt`, whose ids start
with "h". The other 167 are the development set.

- `run` runs the development questions; `run --split held-out` runs the
  held-out ones (`--split all` runs both).
- `score` shows the development questions only.
- `score --held-out` shows the held-out averages, never single questions.
  Do this once, when a change is finished, and report both numbers.
- Do not read held-out results, SQL or failures question by question, and do
  not add vocabulary, examples or prompt text because of a held-out question.
- Other runners that read `queries.yaml` must skip the listed ids too.

Compare gains, not levels: the two sets differ in make-up (the development
set contains the 35 first-round questions, on which the search scores far
higher), so their averages differ for that reason alone.

### Third round: the held-out questions (2026-11-13)

All 167 earlier questions had been run, and their failures read, while the
search was changed, so none of them could serve as a held-out set. The 60
held-out questions were written afterwards (after commit 9d76ad7) with the
recipe of the second round (`work/recipe/`, scripts in `work/heldout2/`):

| Source | Questions |
|---|---|
| Generated from READMEs, 8 question types (seeds never used before, picked by hash) | 36 |
| Research needs written from paper abstracts | 8 |
| bioCADDIE-style | 7 |
| Explicit AND / OR / NOT (4 computed exactly) | 9 |

- The writers saw dataset cards and the existing questions, but no search
  code and no results.
- 2,023 candidates judged by the same LLM judge with the same instructions;
  the same rule for non-human data. 56 questions have at least one relevant
  dataset; 3 have none in the catalogue and one has only partial matches.
- Not done for these labels: the blind second judge and the label review.
- 10 questions have a single relevant dataset, so their recall is 0 or 100%.

## Metrics

| Metric | Meaning |
|---|---|
| recall | Share of the relevant datasets that the search returned at all. |
| lenient | The same, with partially relevant datasets also counting. |
| R@20 | Share of the relevant datasets on the first page of 20 (out of 20 when more exist). |
| nDCG@10 | Whether the first 10 results are the best ones, in the best order: a relevant dataset gains 1, a partially relevant one 0.5, divided by log2(position + 1) and by the best achievable sum. Never-judged datasets gain 0. |
| prec | Share of the returned, judged datasets that are relevant. |

## Running it

On the server, in a one-off backend container (needs the LLM keys and the
database; writes nothing to the database):

```bash
docker compose -f docker-compose-prod.yml run --rm --no-deps \
  -v /home/BIDS-Eye:/repo:ro -v /tmp/bidseye_recall:/out backend \
  python /repo/benchmarks/recall/eval_recall.py run --out /out/run.json
```

About 30 seconds per question, so 15–20 minutes. Then, anywhere:

```bash
python benchmarks/recall/eval_recall.py score run.json                  # development questions
python benchmarks/recall/eval_recall.py score held_out.json --held-out  # once, for a finished change
```

Compare two versions of the search by running both against the same
`ground_truth.json`. LLM answers vary slightly between runs, so treat
differences of one or two datasets on a single question as noise.

## Rebuilding the pools after the catalogue changes

```bash
psql ... -f benchmarks/recall/snapshot.sql > snapshot.jsonl
python benchmarks/recall/build_pool.py snapshot.jsonl --out pools.json
# judge the new candidates, then
python benchmarks/recall/merge_judgments.py pools.json judged/ --snapshot-date YYYY-MM-DD
```
