# Synthetic questions

`questions.jsonl` holds 5,109 search questions that an LLM generated in April 2026
for a fine-tuning experiment that was later dropped (see commit 4169c80). It keeps
only the questions. The SQL generated with them was for the pipeline of that time:
raw BIDS codes, `ILIKE` on descriptions, `LIMIT 50`, no `{{COLS}}` or `text_match`.
None of that SQL would run correctly today, so it was not kept.

One line per question:

| field               | meaning                                                         |
|---------------------|-----------------------------------------------------------------|
| `question`          | the question, whitespace normalised; duplicates removed (case-insensitive) |
| `family`            | the query family the generator was asked for (`scan_filter`, `author_query`, …) |
| `pattern`           | the SQL shape the generator aimed at (`exists`, `join_distinct`, …) |
| `paraphrase_bundle` | questions with the same value are paraphrases of one request; `null` if none |

## What it is good for

- **Robustness sweeps.** Run many varied questions through the translator and
  count crashes, timeouts, compiler declines, and SQL the guard rejects.
- **Paraphrase consistency.** Questions in one bundle should produce the same
  filters.

## What it is not good for

- **Accuracy.** There are no correct answers. Use `benchmarks/demand/gold/`, which
  is built from real requests with judged results.
- **Representing real demand.** The questions are short (median 9 words) and
  template-like. Some ask for things nobody searches for, such as sidecar field
  names or single grant numbers, and some are garbled ("an equalrange diagnosis").

Every question that reaches the SQL model costs a model call, so sample from the
file rather than running all of it.
