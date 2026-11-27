# Gold requests for order and number of results

The 70 real requests (`../requests_truth.json`) hardly ever ask for an order
or a top N, and recall over a whole answer set would punish a correct top 10
("10 of 469 found"). These 13 written requests check the order and the limit
on their own, scored on the first N results only.

| Group | Requests | What is checked |
|---|---|---|
| Order over the whole catalogue | qo01–qo04 | the N best by subjects, publication date, downloads, size |
| Order over a structural gold request | qo05–qo11 | the same over that request's answers (M/EEG, FLAIR ≥ 50 subjects, MRIQC, ≥ 3 sessions, TR < 1 s, resting healthy, EEG ≥ 200 Hz) |
| A number without an order | qo12–qo13 | any N correct answers; the rest of the list is not penalised |

The bases are structural requests only: their answers come from recorded
facts, so the expected order is exact. Ties count: any dataset tied with the
N-th may take its place. Datasets without a value come last (NULLS LAST,
as in the search).

## Files

| File | What |
|---|---|
| `requests.yaml` | The questions, their base and the expected sort key, direction and limit; also the `--queries` file for `eval_recall.py run` |
| `facts.sql` | Read-only query on the production database for the values |
| `facts.jsonl` | Its output, 2026-11-27 (1,726 datasets; 3 without an accession id are left out) |
| `build_truth.py` | `requests.yaml` + `facts.jsonl` + `../requests_truth.json` → `truth.json` |
| `truth.json` | Per request: the set being ordered, the expected order, the acceptable first N |
| `score_ordering.py` | Scores a run: `top`, `order`, `filter`, `extra`, `new` (definitions in its docstring) |

## Running

```bash
python3 benchmarks/recall/eval_recall.py run --queries benchmarks/demand/gold/ordering/requests.yaml \
  --split all --out <run.json> --label <commits>
python3 benchmarks/demand/gold/ordering/score_ordering.py <run.json>
```

## Limits

- The base answer lists are from the 2026-11-14 snapshot, the values from
  2026-11-27. A returned dataset the snapshot never judged is listed as `new`
  and left out of the shares.
- Downloads, views and dates change. Rebuild `facts.jsonl` and `truth.json`
  together before comparing runs that are far apart.
- Not real requests: they test that an order the search understood is carried
  out, not how often people ask for one.
