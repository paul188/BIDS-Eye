# OpenNeuro filter benchmark

Does BIDS-Eye find at least the correct datasets that OpenNeuro's own search
filters find? Each question in `facets.yaml` pairs a filter selection on
openneuro.org/search (modality, species, diagnosis, study design, sex, age,
number of participants, task, study domain, PET tracer and scanner, date,
author, and combinations of up to three) with the plain-English question a
user would type.

| File | What it is |
|---|---|
| `facets.yaml` | 80 questions: OpenNeuro filter selection, question, and the `check` that decides from a dataset's own data whether it is a correct answer. |
| `fetch_openneuro.py` | Asks OpenNeuro's GraphQL `advancedSearch` (the search page's own query) for every selection; a list of selections is their union. |
| `openneuro_results.json` | OpenNeuro's answers, fetched 2026-11-14. |
| `dump_facts.sql` | One JSON line per dataset of the BIDS-Eye database with the facts the checks read. |
| `compare.py` | Compares a BIDS-Eye run (`benchmarks/recall/eval_recall.py run --queries facets.yaml --split all`) with OpenNeuro, dataset by dataset. |
| `results/` | Runs and comparisons (see below). |

## How disagreements are settled

OpenNeuro's filters are not ground truth. They match fuzzily: "rest" also
returns tasks named "test" and "read", "mid" (monetary incentive delay)
returns "main" and "mag" (46 of 53 wrong), "PiB" returns the tracers "SB"
and "DASB". "Human" filters nothing. So every dataset either side returns is
checked against its own data: file types, uploader labels, participant rows,
task labels, PET sidecars. Only datasets in the BIDS-Eye database count.
OpenNeuro datasets published after our last crawl are reported apart: 277
were missing on 2026-11-14, because nothing was crawled after April 2026.

## Results, 2026-11-14

Commit hashes are those after the history rewrite of 2026-11-14 (old → new:
8a3614f → 7dc8187, 7e423f8 → f558ffb).

| | live (7dc8187) | candidate 4 |
|---|---|---|
| OpenNeuro's correct hits found (summed over 79 questions) | 6,767 of 7,752 (87.3%) | 7,722 (99.6%) |
| questions with every correct OpenNeuro hit found | 54 | 71 |
| questions with fewer correct results than OpenNeuro | 13 | 2 |
| correct results OpenNeuro does not return | 1,315 | 1,384 |
| errors (timeouts, disk full) | 2 | 0 |

Candidate 4 is f558ffb plus the facet fixes, the query guards
(`backend/services/query_guard.py`) and bids-eye-f5's species field. What
the live search got wrong:

- study design: it searched the README for "longitudinal" and never read the
  uploader's label (`study_longitudinal`). It missed 89 of 126 longitudinal
  datasets and 601 of 614 cross-sectional ones.
- sex: 'Female' and 'FEMALE' did not match `p.sex = 'female'` (42 datasets).
- species: "animals other than rats and mice" resolved to a vehicle-control
  diagnosis and returned nothing; species words were also ANDed as README
  text ("mouse" 32 → 18).
- participants ≥ 100 counted only participants.tsv rows (22 missed).
- carbon-11 PET was searched as text instead of `TracerRadionuclide` (0 of 7).
- "≥100 children aged 12 or younger" joined participants × files and filled
  the server's disk.

Still short: go/no-go and stop-signal (the crawl stores `gonogo` and
`stopsignal` as `attention_and_executive_control`). And "≥100 children",
where OpenNeuro's filter means "≥100 participants, any of them ≤12" and
BIDS-Eye counts 100 children.

`results/2026-11-14_cand5_design.json`: the study-design questions after the
label guard became its own AND part (the uncertain tier's paper evidence had
let a dataset labelled Cross-Sectional into a longitudinal question). Every
correct OpenNeuro hit is found.
