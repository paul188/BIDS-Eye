# SQL And Duplication Guidelines

These rules are meant for teacher-model generation and for human review.

## SQL Quality

- Every query must be valid PostgreSQL.
- Every query must start from the real BIDS schema in this repo.
- Use only real fields and real values sampled from the database.
- Prefer precise constraints over vague ones when the question wording supports
  them.
- Do not add extra output columns unless the question asks for them.
- For dataset-list questions, return dataset identifiers and names unless the
  spec explicitly requests a different output.
- For "all / only / exclusively" participant constraints, prefer alias-aware
  logic over brittle single-value equality checks.
- Use `EXISTS` / `NOT EXISTS` naturally for presence and absence queries rather
  than forcing everything into one canonical join pattern.
- Use JSON access only where the schema supports it: `description->>`,
  `other_entities->>`, `extra->>`.

## Diversity Targets

- Mix SQL structures across a batch:
  - `JOIN + DISTINCT`
  - `EXISTS`
  - `NOT EXISTS`
  - `GROUP BY + HAVING`
  - ranking with `ORDER BY` and `LIMIT`
  - aggregated scalar queries
- Include a mix of question families:
  - scan-content filters
  - participant filters
  - multimodal presence/absence
  - session / longitudinal queries
  - metadata / JSON queries
  - ranking and statistics
  - comparisons and negation
- Include long-tail values from the DB, not just common values like `rest`,
  `nback`, `eeg`, and `T1w`.

## Paraphrase Policy

- It is good to have multiple questions for the same SQL intent.
- Keep that moderate: usually 2 to 4 paraphrases per SQL intent.
- Do not produce many tiny wording edits for the same SQL.
- Each paraphrase must preserve the exact semantics.
- Good paraphrase styles:
  - concise
  - formal / scientific
  - natural / casual
  - indirect wording
- Avoid paraphrases that change:
  - requested output columns
  - negation / absence logic
  - numerical thresholds
  - comparison direction

## Anti-Duplication Rules

- Do not let one SQL skeleton dominate a batch.
- Do not emit more than 2 examples with the same SQL shape in one prompt unless
  the batch is explicitly marked as a paraphrase bundle.
- Avoid generating more than 4 question variants for the same SQL intent.
- Prefer new SQL structures over extra paraphrases once a concept is already
  covered.
- If two queries differ only in alias ordering or column alias names, treat them
  as duplicates for training purposes.

## Repair Bucket

- Queries blocked by statement timeout should not go into the gold dataset.
- Queries blocked by DB-data encoding issues should not go into the gold
  dataset.
- Keep those rows in a separate repair file and re-validate later.
