# Word forms per question

`../label_set.json` tests word-form decisions one pair at a time ("should a search for
*aging* find *age*?"). This set tests the path a real question takes:

1. `word_forms.question_words()` picks the question's words to look up (no model yet).
2. The catalogue lookup (`word_form_counts.lookup`) gives each word's forms and counts.
3. Words with colliding forms that nobody judged go to the planner's prompt; the model says
   which forms to keep (`intent.preprocess_query`).
4. `word_forms.decide()` and `forms_for()` give the forms each searched word must match in.

A wrong word list in step 1 hides a word from step 3 (it is then searched by stem) or adds
noise to the prompt. A wrong decision in step 3 or 4 loses datasets or adds false hits.

## Files

| file | what |
|---|---|
| `questions.yaml` | 277 questions with search words, filler words and truth rows (header explains fields) |
| `run.py` | the scorer |
| `fetch_evidence.py` | freezes the catalogue's forms for every word into `evidence.json` (read-only) |
| `evidence.json` | not made yet: needs database access (below) |
| `results/` | model answers cached by `run.py --model` |

## How the set was made (2026-11-27)

- **Questions**: development questions of `../../recall/queries.yaml`,
  `../../demand/demand_queries.yaml`, `../../easy/easy_searches.yaml` and
  `../../recall/ambiguity_queries.yaml` (1,741 in all; held-out ids are left out). Shuffled with
  seed 20261127, then kept while a form of a labelled pair still had fewer than three questions
  (207, tier `labelled`); plus 30 questions with a word of `word_forms.yaml` or
  `word_form_pairs.json` (`reviewed_list`), 30 with neither (`control`) and 10 written
  tokenization traps (`trap`).
- **search_words / filler_words**: only the words that carry a decision were judged. A search
  word names something the dataset should contain or be about. Filler is request language
  ("check", "compared", "independent sample", "pooled", "recorded while"), a purpose clause, or an
  age range that becomes an age filter ("aged 18 to 35"). Judged by Claude; **Paul has not reviewed
  them yet**.
- **truth**: Paul's blind labels of `../label_set.json` (p000–p189), read per direction. Pairs
  labelled `context` are not scored. Two question senses override a label (marked `overridden`,
  with a note): *structural* in "structural MRI" must not find *structure* (Brain Imaging Data
  Structure; the blind label said both, `word_forms.yaml` says no), and *ageing* as growing old
  must not find *age* (label: context). 382 scored rows in 188 questions.

Pairs of origin `sample` are the blind test of the model step: `word_form_pairs.json` leaves
their stems out on purpose, so only the model decides them. Pairs of origin `yaml_*` are covered
by `word_forms.yaml` and test the lists.

## Running

```
python benchmarks/word_forms/questions/run.py                      # part 1, no database
python benchmarks/word_forms/questions/run.py --evidence evidence.json            # + stem, lists
python benchmarks/word_forms/questions/run.py --evidence evidence.json --model    # + model
```

From `backend/` with the CI environment (`METADATA_DB_*` set to anything, `PYTHONPATH=../BIDS-SQL`),
or with `bidseye-pytest`'s environment. `--model` needs the LLM settings of `backend/llm/llm_config.yaml`
and sends each question (not dataset text) to the model; answers are cached in `results/`.

## First results (part 1, 2026-11-27)

| tokenizer | search words found | stopwords looked up | trap checks |
|---|---|---|---|
| v1 (runs of a-z, stopwords in the cap) | 354/356 | 790 | 36/41 |
| current | 356/356 | 0 | 41/41 |

v1 looked up the fragments *gren* (Sjögren) and *rage* (MP2RAGE), and in a long question the
stopwords filled the 40-word cap before *ageing* and *localizer*.

Part 2 waits for `evidence.json`. A check on hand-made counts showed what it can find: a large
word whose stem also holds a small, different word (*publication* and *public*) is never
asked about, because the collision test only looks for forms that bring in *more* datasets than
the asked word; the stem then merges them. Whether this happens on the real catalogue is what
the fixture will tell.
