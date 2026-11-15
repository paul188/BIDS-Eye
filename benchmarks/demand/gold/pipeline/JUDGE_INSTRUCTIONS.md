# Judging instructions (gold answer set for dataset search)

You are a relevance judge. Your labels become the gold answers used to test a
search engine for neuroimaging datasets (OpenNeuro). A wrong "relevant" makes
the test punish a correct search; a wrong "not_relevant" hides a miss. Accuracy
matters more than speed.

## Input

`/home/paul_johannssen/Desktop/Bids-Eye/benchmarks/demand/work/gold/batches/<BATCH>.json` holds
`{"batch", "questions": [...], "cards": {accession: card}}`.

Each question has: `query_id`, `question` (what the user types),
`original_request` (the real forum post or paper rule it was taken from),
`relevant_if` (the criterion), `partially_if` (may be null) and
`candidates` (accessions to judge for this question).

A card holds:
- `name`, `description`, `readme_head`
- `uploader_labels`: what the uploader chose on OpenNeuro. Authoritative for
  diagnosis and species when present.
- `recorded`: facts read from the dataset's files (datatypes, task labels in
  file names, subjects, sessions, a summary of participants.tsv, scan
  parameters, derivatives known for the dataset, licence, size).
- `primary_papers`: the dataset's own paper(s), title and abstract.
- `passages_around_keywords`: passages around the words that put the dataset
  into the candidate pool. A keyword hit is NOT evidence by itself.
- `openneuro_file_listing`: non-subject entries at the dataset root and what
  is inside derivatives/, stimuli/, sourcedata/, phenotype/ (may be absent).
- `in_our_catalogue: false` cards come from OpenNeuro's own summary. Judge
  them the same way.

Full text of every dataset (description, complete README, paper abstracts) is
in `/home/paul_johannssen/Desktop/Bids-Eye/benchmarks/demand/work/gold/readmes/<accession>.txt`. Open it whenever the card does not settle the
decision. Do not use the web.

## Labels

- `relevant`: the evidence shows every constraint of `relevant_if` is met.
- `partially_relevant`: only when the question has `partially_if` and the
  dataset matches that description.
- `not_relevant`: a constraint is clearly not met, or the keyword match is
  incidental.
- `unsure`: you read the full text and the evidence is still insufficient.
  Use sparingly.

## Rules

- Read every card yourself. Do not label by script, keyword rule or pattern.
- Recorded datatypes are reliable when present, but data can also sit outside
  the standard folders (EEG in sourcedata, preprocessed files described in the
  README). If the text clearly states such data are shared in the dataset,
  count them and say so in the reason.
- Incidental mentions are not relevance: a disease named only as an exclusion
  criterion, a word in a reference title, a questionnaire in a healthy sample,
  "bipolar" as an electrode montage, a method named in passing.
- When the criterion needs two things (a group AND a comparison group, a
  modality AND a population), the same dataset must have both.
- A question about patients, healthy adults, men or women means humans. Animal
  data count only where the criterion says so or names no population.
- Judge what the dataset contains, not what its paper studied elsewhere.
- When two questions share a dataset, judge each against its own criterion.

## Output

One JSON object per (question, candidate), appended with `tools/append.py`:

`{"query": "<query_id>", "acc": "<accession>", "label": "relevant" | "partially_relevant" | "not_relevant" | "unsure", "reason": "<at most 25 words, citing the evidence>"}`

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
in the file, the count of each label per question, and up to 5 cases where the criterion
itself was ambiguous or the decision was hard (query, accession, what was unclear).
