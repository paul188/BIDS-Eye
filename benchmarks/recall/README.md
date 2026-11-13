# Recall benchmark

Measures how many of the relevant datasets a search actually returns.

| File | What it is |
|---|---|
| `queries.yaml` | 35 questions: 17 from the bidseye.xyz user log, 18 probes for known weaknesses. Each has a written relevance criterion and the signals used to find candidates. |
| `ground_truth.json` | Relevant / not relevant / unsure datasets per question. |
| `judgments.jsonl` | Every label with its one-line reason, for review. |
| `build_pool.py` | Builds the candidate pool of each question from a database snapshot. |
| `merge_judgments.py` | Turns judge output into `ground_truth.json` and `judgments.jsonl`. |
| `eval_recall.py` | `run`: sends every question through the deployed search code. `score`: compares the result with the ground truth. |

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

## Running it

On the server, in a one-off backend container (needs the LLM keys and the
database; writes nothing to the database):

```bash
docker compose -f docker-compose-hetzner.yml run --rm --no-deps \
  -v /home/BIDS-Eye:/repo:ro -v /tmp/bidseye_recall:/out backend \
  python /repo/benchmarks/recall/eval_recall.py run --out /out/run.json
```

About 30 seconds per question, so 15–20 minutes. Then, anywhere:

```bash
python benchmarks/recall/eval_recall.py score run.json
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
