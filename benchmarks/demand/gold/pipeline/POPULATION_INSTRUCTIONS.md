# Who are the participants? (gold facts for dataset search)

Several test questions depend on whether a dataset contains healthy human
participants. You decide that per dataset. Accuracy matters more than speed.

## Input

`/home/paul_johannssen/Desktop/Bids-Eye/benchmarks/demand/work/gold/batches/<BATCH>.json` holds `{"batch", "cards": {accession: card}}`.
A card holds `name`, `description`, `readme_head`, `uploader_labels`
(what the uploader chose on OpenNeuro: authoritative when present),
`recorded` (facts from the files, with a summary of participants.tsv:
normalised diagnoses, the raw group column, ages, sex), `primary_papers`
and `passages_around_keywords`. Full text of every dataset is in
`/home/paul_johannssen/Desktop/Bids-Eye/benchmarks/demand/work/gold/readmes/<accession>.txt`: open it whenever the card does not settle the
decision. Do not use the web. Read every card yourself; do not label by script
or keyword rule.

## What to decide per dataset

- `population`:
  - `healthy_only`: all participants are healthy / typical humans (a normal
    cognitive-neuroscience sample; exclusion criteria naming diseases do not
    make it clinical; neither do personality or symptom questionnaires in a
    community sample).
  - `patients_and_healthy`: a patient or clinical group AND healthy controls.
  - `patients_only`: only people with a diagnosis or clinical condition
    (including pre-surgical epilepsy patients with implanted electrodes).
  - `special_non_clinical`: humans without a disease but selected for a
    condition that is not "healthy adult / child" in the usual sense
    (preterm infants, people born blind or deaf, heavy drinkers, pregnant
    women, athletes after concussion without diagnosis). Say which.
  - `not_human`: animals, phantoms, simulations, post-mortem tissue.
  - `unclear`: full text read, still cannot tell.
- `n_healthy`: number of healthy human participants if stated or recorded
  (integer), else null.
- `healthy_adults_5`: "yes" if at least 5 healthy participants are adults
  (18 or older), "no" if not, "unknown" if ages cannot be established.
- `healthy_children_5`: "yes" if at least 5 healthy participants are aged 2
  to 12, "no", or "unknown".
- `includes_men`: "yes" / "no" / "unknown" — whether the healthy
  participants include adult men.

## Output

One JSON object per card, appended with `tools/append.py` (no "query" field):

`{"acc": "...", "population": "...", "n_healthy": 24, "healthy_adults_5": "yes", "healthy_children_5": "no", "includes_men": "yes", "reason": "<at most 25 words, citing the evidence>"}`

## How to work (resumable: follow exactly)

You can be stopped at any moment, and a judgment that is not in the output
file is lost. So work only through the two tools below, in small steps.

1. `cd /home/paul_johannssen/Desktop/Bids-Eye/benchmarks/demand/work/gold && python3 tools/next_cards.py <BATCH> 8` prints the criterion and
   the next 8 cards that have no judgment yet. It always continues where the
   output file stops; if earlier work exists, it is skipped automatically.
2. Read those cards (open `readmes/<accession>.txt` where needed) and decide.
3. Append the judgments at once:

   ```
   cd /home/paul_johannssen/Desktop/Bids-Eye/benchmarks/demand/work/gold && python3 tools/append.py <BATCH> <<'JSONL'
   {"query": "...", "acc": "...", "label": "...", "reason": "..."}
   JSONL
   ```

   The tool validates each line, writes it to disk immediately and reports how
   many are left. It never overwrites an existing judgment.
4. Repeat from 1 until `next_cards.py` prints ALL DONE.

Never hold more than 8 undecided or unwritten judgments. Do not keep labels in
notes, variables or scratch files, do not write the output file by any other
means, and do not edit or delete lines that are already there.

Keep any helper scripts in `/home/paul_johannssen/Desktop/Bids-Eye/benchmarks/demand/work/gold/judged/tmp_<BATCH>/`. Do not modify any other file.

When `next_cards.py` prints ALL DONE, reply only with: the number of judgments
in the file, the count of each `population` value, and up to 5 cases where the criterion
itself was ambiguous or the decision was hard (accession, what was unclear).
