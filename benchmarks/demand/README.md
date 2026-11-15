# Demand benchmark

Questions that come from what researchers really looked for, not from what
the catalogue contains. It complements `../recall/`, whose questions were
mostly written from dataset READMEs.

Two sources (built 2026-11-14):

| Source | Questions | Known answer |
|---|---|---|
| Real requests: 44 forum posts, 12 requests for a search feature, 14 selection rules from papers' methods (`requests.yaml`) | 70 | none yet (needs judging) |
| Needs recovered from 561 papers that reused an OpenNeuro dataset: 585 needs, each as keywords (`short`) and as a sentence (`need`) | 1,170 | the dataset(s) the authors used |

`demand_queries.yaml` holds all 1,240. Ids starting with `h` are held out
(170 needs = 340 questions, `held_out_ids.txt`); the rules of
`../recall/held_out_questions.txt` apply to them.

## Why reuse papers

A paper that reused a dataset is a search that ended well: the authors needed
some kind of data and picked this dataset. The question is written from the
paper alone, so it uses the searcher's words, and the answer does not depend
on an LLM judge. The split is by paper, so the development papers can also be
used as data (examples, vocabulary, text indexed per dataset) and the held-out
papers still test the result.

Limits: the known dataset is a lower bound, since other datasets may fit the
need as well, so this measures "would the search have led them to the data
they used", not precision. ds000030 alone accounts for 101 of the 585 needs;
`score_known_item.py` therefore also reports a figure with every dataset
counted once. The questions were written by an LLM (Claude Sonnet) and not
reviewed by a person. Only open-access papers on Europe PMC are covered.

## How it was made

| Step | Script | Result |
|---|---|---|
| List open-access papers that mention OpenNeuro or OpenfMRI | `harvest_epmc.py search` | 1,836 papers |
| Download full text | `fetch_fulltext.sh` | 1,728 (16 have none) |
| Find accession numbers and the paragraphs around them | `extract_mentions.py` | 1,258 papers name an accession |
| Drop the dataset's own paper (linked in the catalogue, or shared authors) | `classify_reuse.py` | 636 papers with a reuse candidate |
| Read each paper, decide reuse or not, write the need | `writer_prompt.md`, one Sonnet subagent per 40 papers (`make_writer_batches.py`) | 563 reuse, 41 own data, 28 mention only, 4 unclear |
| Assemble, split, drop one need that pooled 785 datasets and one wrong pairing | `build_queries.py` | `demand_queries.yaml`, `demand_first_run.yaml` |

Writers saw the paper's title, abstract and the paragraphs naming the dataset
or the repository, plus the dataset authors' names. They did not see the
dataset's title or README, any search code, or any result. A question may not
contain the accession number, the dataset's name or the repository's name.

Intermediate files are in `work/` (not in git).

## Review of the pairings

A check of the written questions (2026-11-14) found no question that names
its accession or the repository. Three things are handled in the files:

- **Requested modality not in the catalogue.** For 8 needs the keyword form
  asks for a recording type that the catalogue records for none of the known
  datasets (`modality_not_in_catalogue`, set by `build_queries.py` from the
  snapshot). Seven are real pairings and gaps in the catalogue: ds000117 has
  EEG channels inside its MEG files, ds000116 and ds002734 keep their EEG
  outside the BIDS folders, ds004107 is MEG with EEG channels. The scorer
  reports them on a line of their own. A search that filters hard on modality
  cannot find them, while both text baselines do.
- **One wrong pairing dropped.** PMC9467511 cites ds000221 for resting EEG
  that the OpenNeuro copy does not hold (`EXCLUDED_PAPERS`).
- **Low-confidence needs.** A fair share of them are "any dataset to try my
  pipeline on", where the dataset used is arbitrary. The scorer reports high
  and low confidence separately; BM25 puts the known dataset on the first
  page for 75% of the high-confidence and 48% of the low-confidence
  development questions. Quote the high-confidence figure as the main one.

## Running and scoring

The existing runner takes the file unchanged. On the server, as in
`../recall/README.md`:

```bash
python /repo/benchmarks/recall/eval_recall.py run \
  --queries /repo/benchmarks/demand/demand_first_run.yaml --split all \
  --label "<commit>" --out /out/demand_first_run.json
```

`demand_first_run.yaml` is the 70 requests plus one development paper per
reused dataset (146 papers, 362 questions, about 3 hours). The full file takes
about 10 hours.

```bash
python3 benchmarks/demand/score_known_item.py run.json --misses 30   # development
python3 benchmarks/demand/score_known_item.py run.json --held-out    # averages only, once per finished change
python3 benchmarks/demand/analyse_demand.py run.json                 # demand statistics, results by catalogue text quality
python3 benchmarks/demand/coverage.py ../recall/results/<dev run>.json  # requests vs. the recall benchmark
```

`baseline_local.py` runs plain BM25 and plain embedding search (bge-base) over
the catalogue snapshot. It needs no server and gives a reference point:

| Development questions (830) | found in top 100 | top 20 | top 5 |
|---|---|---|---|
| BM25 over name, fields and README | 79% | 68% | 53% |
| Embedding similarity | 68% | 60% | 47% |

Held out (340): BM25 62% and embedding 53% in the top 20. The BIDS-Eye
pipeline has not been run on these questions yet.

## What it shows so far

What reuse papers needed (share of 585 needs): a modality 98%, a patient
group or special population 49%, patients and a comparison group 36%, resting
state 34%, healthy participants 30%, a shared clinical score or questionnaire
25%, a task 24%, an age range 24%, a specific stimulus 17%, several sessions
15%, two kinds of recording in the same people 13%, a minimum size 12%,
acquisition parameters 11%, shared derivatives 5%.

What the recall benchmark does not test (`coverage.py`): preprocessed data
and named pipelines (16% of the 70 requests, no benchmark question), numeric
thresholds (21% vs 4%), resting state as a requirement (14% vs 3%), sample
size, counts within a subgroup, labels for machine learning, licence,
auxiliary files, linked papers. On the criteria it does test, development
recall is lowest for acquisition parameters (25%), healthy-participant
requirements (28%) and age (38%).

The catalogue does not describe some of its most reused datasets. ds000245
(Parkinson's, resting state; reused by 18 papers here) is titled
"ds000245_R1.0.0" and its README never names the disease. For datasets with
no real title and a short README, both baselines find the known dataset on
the first page in 0 of 26 development questions.

## Not done yet

- The BIDS-Eye pipeline has not been run on any of these questions.
- The 70 requests have no ground truth. Several need exact answers from the
  database (TR, field strength, sessions, licence, derivatives), the rest
  need pooled judging as in `../recall/work/recipe/`.
- No person has reviewed the written questions or the reuse verdicts; the
  review above was done by an LLM session and by mechanical checks.
- In `requests.yaml` the id `qx06_fmriprep_version` carries the MRIQC
  question. The id is kept so that finished runs still join; it is only a key.
