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
| `evidence.json` | the catalogue's forms of every word, fetched 2026-11-27 (read-only, via `--print-sql`) |
| `results/` | model answers cached by `run.py --model` |

## How the set was made (2026-11-27)

- **Questions**: development questions of `../../recall/queries.yaml`,
  `../../demand/demand_queries.yaml`, `../../easy/easy_searches.yaml` and
  `../../recall/ambiguity_queries.yaml` (1,741 in all; held-out ids are left out). Shuffled with
  seed 20261127, then kept while a form of a labelled pair still had fewer than three questions
  (207, tier `labelled`); plus 30 questions with a word of `stem_collisions.yaml` or
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
  Structure; the blind label said both, `stem_collisions.yaml` says no), and *ageing* as growing old
  must not find *age* (label: context). 382 scored rows in 188 questions.

Pairs of origin `sample` are the blind test of the model step: `word_form_pairs.json` leaves
their stems out on purpose, so only the model decides them. Pairs of origin `yaml_*` are covered
by `stem_collisions.yaml` and test the lists.

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

## First results (part 2, evidence of 2026-11-27, 1,722 datasets, without the model)

| method | right | wrong merge (false hits) | wrong split (lost recall) |
|---|---|---|---|
| stem | 166 | 216 | 0 |
| lists (`stem_collisions.yaml` + `word_form_pairs.json`) | 284 | 96 | 2 |

**80 of the 98 remaining errors are words the model is never asked about.** The collision test
(`colliding_forms`) only flags a form that brings in at least twice the datasets of the asked
word's own forms. When the asked word is the larger one, the smaller different word is merged
silently, and the lists were mined with the same test, so they lack these directions too:

| search word | merged with | rows | datasets: word / other |
|---|---|---|---|
| functional | functioning | 23 | 436 / 22 |
| weighted | weight(s) | 20 | 275 / 34 |
| clinical | clinic(s) | 10 | 112 / 12 |
| emotion | emotionally | 7 | 55 / 7 |
| age | aging | 6 | 220 / 34 |
| experiments | experience(s) | 3 | 65 / 63 |
| publication | public | 2 | 132 / 96 |
| temporal, format, scales, well, emotions | temporally, formation, scaling, wellness, emotionally | 7 | |

(*weighted* mostly stands inside "T1-weighted", which the search keeps together by adjacency, so
its 20 rows overstate the harm.) The other 18 errors are words the model is asked about; the
`lists` method leaves its answers out, so they need `--model`. The 2 wrong splits are
*dependence* → *dependent*/*depending*: `stem_collisions.yaml` splits them, the blind label says both.

**Prompt noise:** 133 words would go to the model across the 277 questions; 23 of them are
search words. The rest are request language: *compare* (27), *listening*, *covering*, *scanned*,
*public*, *predict*, *validate*, *making*, *recordings*.
