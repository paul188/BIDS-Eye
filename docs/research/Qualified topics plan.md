# Qualified topics: binding qualifiers to the words they change

Plan of 2026-11-14. Goal: a question such as "datasets that deal specifically
with social conflicts" searches for conflict *between people*, not for any
dataset that contains the word "conflict", and the engine finds datasets that
describe the same thing in other words (aggression, provocation, negotiation,
intergroup discrimination).

## 1. What goes wrong today

Two live questions from 2026-11-14 show the three failure modes.

**"give me all datasets that deal specifically with social conflicts"**
(production, backend faac8e4):

- The intent call put "social conflicts" into `description_search`. The text
  search became `social & conflicts`: both words anywhere in the dataset's
  text, at any distance. Only ds003103 (couples resolving conflicts during
  joint shopping) passed.
- The second tier added every dataset with `conflicts` alone. 21 of its 22
  results are cognitive-control conflict (Simon, Stroop, flanker, response
  conflict) or unrelated wording ("reduce conflict in published literature").
  The qualifier "social" was dropped exactly where it mattered.
- Missed, because none of them writes "social" next to "conflict":
  ds005375 POLEX (Cyberball plus personal and group discrimination tasks),
  ds004274 and ds003791 (aggression between people), and, depending on how
  strict "specifically" is, ds004103 (collaboration vs. deception between
  players), ds006761 (competitive Rock-Paper-Scissors in pairs), ds000214
  (Cyberball) and the Ultimatum Game datasets (ds004000, ds003745, ds004920,
  ds005123).
- The vocabulary has the same blind spot: the task term "conflict" alone
  resolves to `competition_task`, and the construct `response_conflict` to
  `simon_task` and `flanker_task_ern`.

**"give me all emo studies coauthored by Morgenroth"**

- "emo" matched the start of the construct "emotion" (WRatio 90). The only
  task linked to that construct in `backend/vocabulary/bids_to_ca_map.json` is
  `meditation_general`, which became a hard filter, so the search found
  nothing. Fixed in `backend/vocabulary/retriever.py` (see section 12). "emo" now becomes a
  text search and finds both Emo-FilM datasets.
- The underlying problem remains: "emo" means *emotion*, and Emo-FilM
  (ds004892, ds004872) says so nowhere in its own metadata. It has no
  description and a 50-character README; only its linked paper mentions
  emotion. The question "emotion studies by Morgenroth" would still miss it.

The three failure modes:

| # | Failure | Example |
|---|---|---|
| A | A qualifier is not bound to its head word, so the head's other meanings match | "social conflict" → Stroop and Simon datasets |
| B | The same thing written in other words is not found | aggression, provocation, discrimination, Taylor Aggression Paradigm |
| C | The meaning is only implied: abbreviations, dataset names, linked papers | "emo", "Emo-FilM" |

## 2. Approach in one paragraph

The intent call (one Gemini call that already runs for every question) returns
each topic as a structured **qualified topic**: the user's phrase, the head
word, the qualifiers bound to it, a one-line gloss of the meaning, phrasings an
author would use for it, and the meanings of the head word that are *not*
meant. A small curated **sense inventory** covers the most ambiguous head words
deterministically and checks the LLM's output. The search matches the phrase
and its validated alternatives *close together* in the text, never the head
word alone. Datasets that only match a meaning that is not wanted move down;
they are never removed. Later stages add meaning that no word list reaches:
README profiles, linked-paper text, embeddings of the gloss, and a
per-requirement check of the top results.

Stages 1 to 4 run without new data sharing and without extra latency. Stages
5 and 6 send catalogue metadata to Gemini or an embedding API and need your
explicit approval, as recorded in the deployment notes.

## 3. Stage 0: test set and baseline (½–1 day)

Without a fixed yardstick, every later step is anecdote.

- Write 20 to 30 **ambiguity questions**, each with a head word that has
  several meanings in this catalogue. Two to three questions per head, one
  per meaning:
  - conflict: social / cognitive (Stroop, Simon) / approach-avoidance
  - control: cognitive control / control group / motor control
  - reward: monetary / social / food
  - attention: attention task / ADHD / "attention to detail" in a README
  - load: cognitive load / viral load
  - rest: resting state / rest blocks between runs
  - pain: chronic-pain patients / pain-perception task
  - stress: social stress test (TSST) / stress as a mood rating
  - learning: motor / reward / language
  - social: social anxiety / social cognition / social media
  - emotion: emotion task / emotion regulation / emotional faces, plus the
    abbreviation "emo"
- Keep them in a separate file (`benchmarks/recall/ambiguity_queries.yaml`),
  not in `queries.yaml`, and never as prompt examples. The rule in
  `backend/llm/preprocess.py` above `_SYSTEM_PROMPT` applies: a prompt
  that names the test questions makes their score meaningless.
- Judge the pool with the existing judge (`benchmarks/recall/build_pool.py`,
  `merge_judgments.py`). Seed "social conflict" with the verdicts in section 1.
- Metrics per question: precision of the top 20, recall over the judged pool,
  and the share of results that belong to a meaning that is *not* meant
  ("wrong-sense rate"). Wrong-sense rate is the number this plan is meant to
  lower; recall on the main benchmark must not drop.
- Pin the intent parse for each question (cache the QueryPlan JSON) so that
  runs differ only by the code under test. The deploy log traced earlier
  regressions to Gemini extracting different phrases on different runs.

## 4. Stage 1: parse topics as qualified topics (1–2 days)

### Schema

New field on `QueryPlan` in `backend/llm/preprocess.py`:

```python
class Topic(BaseModel):
    phrase: str              # as the user meant it, normalised: "social conflict"
    typed: str               # as typed: "social conflicts", "emo"
    head: str                # the word the qualifiers modify: "conflict"
    qualifiers: List[str]    # words that select a meaning: ["social"]
    gloss: str               # one line, the meaning: "disagreement, hostility or
                             # aggression between people or groups"
    alternatives: List[str]  # up to 8 phrasings an author would write:
                             # "interpersonal conflict", "intergroup conflict",
                             # "conflict resolution", "aggression", "provocation",
                             # "social exclusion", "negotiation", "disagreement"
    not_meant: List[str]     # meanings of the head the qualifiers exclude:
                             # "cognitive conflict", "response conflict",
                             # "Stroop", "Simon task", "flanker"
    strict: bool             # "specifically", "only", "exclusively" → True
```

`topics` replaces the single-keyword `description_search` for subject matter.
`description_search` stays for property words ("complex", "naturalistic",
"pediatric"). The prompt section "Qualitative / adjectival modifiers" already
draws that line; it gets one more distinction:

- A **property modifier** describes the data or the task without changing what
  the noun means: "complex motor tasks" are still motor tasks. It stays in
  `description_search` as today.
- A **sense qualifier** changes which meaning of the noun is meant: "social
  conflict", "cognitive control", "monetary reward", "chronic pain". It is
  bound into the topic and never searched alone.

Abbreviations and clipped words are expanded in `phrase` and kept in `typed`
and `alternatives`: "emo" → phrase "emotion", alternatives include "emo"
(dataset names such as "Emo-FilM" use the short form). This handles the
"hidden meaning" of the first question at no extra cost.

### Prompt examples

The examples must not be test questions (section 3). Use heads outside the
ambiguity set, for example "visual imagery" vs. "medical imaging", "motor
sequence" vs. "DNA sequence", "spatial memory" vs. "memory consolidation".

### Scan terms

Today a task-like topic goes into `ScanRequirement.scan_terms` and is resolved
to task codes. A topic keeps that path for its `phrase`, with one change: the
head alone is never sent to the resolver when qualifiers exist. "social
conflict" is looked up as a whole; if it misses, it does *not* fall back to
"conflict" → `competition_task`.

## 5. Stage 2: resolve and validate (1–2 days)

### Vocabulary

`backend/vocabulary/retriever.py` gets `resolve_topic(topic)`:

1. Look up `phrase`, then each alternative, with the existing task and
   construct matching (`_match_yaml_term`).
2. Codes found only through the head word are dropped as filters. They may
   stay as ranking hints when the code's label or synonyms contain one of the
   qualifiers or alternatives.
3. Codes whose label matches a `not_meant` entry are removed, even when an
   alternative reached them ("conflict resolution task" in a Simon dataset).
4. The thin-construct rule from the "emo" fix applies: a construct linked to
   fewer than three tasks never stands alone.

### Validate the LLM's alternatives against the catalogue

LLM alternatives help lexical search, but hurt when the model does not know
the term (see the local-only report `Search advances and Bids Eye gaps.md`, direction 3).
Each alternative is checked before use with the document-frequency lookup
that `backend/services/text_search.py` already has (`_lookup_counts`):

- 0 datasets: dropped (useless, possibly invented).
- More than about 5% of the catalogue (~85 datasets): dropped as too broad
  ("interaction", "behaviour"), unless it is the user's own phrase.
- Each alternative must contain a word that is not in `not_meant`.
- At most 8 survive, ordered by the LLM's order.

The surviving list is logged with the question, so the chips and the
benchmark can show exactly what was searched.

## 6. Stage 3: search with phrases, not word bags (1–2 days)

### Near matching

The `text_match` grammar (`backend/vocabulary/text_terms.py`) matches every word of a term
anywhere in the text. A new mode `near` requires the words of a term to be
within 3 positions of each other, in either order. PostgreSQL's tsquery has
only exact-distance operators, so `near` expands to an OR of them:

```
social <1> conflict | social <2> conflict | social <3> conflict
| conflict <1> social | conflict <2> social | conflict <3> social
```

The grammar keeps its allowed character set (`[a-z0-9 |-]`); `near` is a mode
like `short` and `loose`, so the placeholder still survives being copied by
the SQL model. Stemming already makes "conflicts" match "conflict".

### Tiers

A topic becomes one condition, mapped onto the existing second tier
(`backend/services/uncertain_tier.py`):

| Level | Topic condition |
|---|---|
| 0 confirmed | the phrase or a validated alternative matches `near`, in name, description, study fields, profile or README; or a resolved task code from stage 2 |
| 1 probable | qualifier and head both occur, but far apart (today's `social & conflict`); or a near match only in a linked paper |
| not shown | the head alone (today's `conflicts` second tier). For an unambiguous head and `strict = false`, the head alone may stay at level 1 |

Datasets whose only match is a `not_meant` term move to the end of their
level. They are not removed: a dataset can contain a Stroop task *and* a
couples-conflict task.

### Expected effect on the "social conflict" question

| Dataset | Today | After stage 3 |
|---|---|---|
| ds003103 couples conflict | confirmed | confirmed |
| ds005375 POLEX | missed | confirmed (social exclusion, discrimination) |
| ds004274, ds003791 aggression | missed | confirmed (aggression) |
| ds004000 etc. Ultimatum Game | missed | confirmed only if "ultimatum game" survives validation as an alternative |
| ds003509, ds003518, ds004580 Simon | second tier | not shown |
| ds000164 Stroop ("header conflicts") | second tier | not shown |

## 7. Stage 4: sense inventory for frequent ambiguous heads (2 days incl. review)

LLM output varies between runs and model versions. For the 20 to 30 heads
that matter most, a curated file makes the behaviour fixed and testable.

`backend/vocabulary/senses.yaml`:

```yaml
conflict:
  social:
    cues: [social, interpersonal, intergroup, couple, partner, family, group]
    alternatives: [interpersonal conflict, intergroup conflict, conflict resolution,
                   aggression, provocation, social exclusion, discrimination]
    tasks: []
  cognitive:
    cues: [cognitive, response, stimulus-response, stroop, simon, flanker, interference]
    alternatives: [response conflict, conflict monitoring, interference control]
    tasks: [simon_task, flanker_task_ern]
  decision:
    cues: [approach-avoidance, decision, choice]
    alternatives: [approach-avoidance conflict, decision conflict]
```

Use:

- If a topic's head is in the file and a qualifier matches a sense's cues,
  the sense's alternatives and tasks are added and the other senses' cues
  become `not_meant`. This does not depend on the LLM.
- If the head is in the file but no qualifier is given ("conflict studies"),
  all senses count, and the chips show the senses so the user can pick one
  (stage 7).
- The LLM's alternatives are kept next to the inventory's, after validation.

Building the file: a script (`backend/vocabulary/tools/mine_senses.py`) lists, for each
frequent head word in names, descriptions and READMEs, the words that occur
next to it (1–2 positions left and right) and how many datasets use each
pair. "conflict" will show *Simon, response, cognitive, approach-avoidance,
social, couples, family*. You review and group these into senses; the script
only proposes. Heads come from the stage-0 list, plus any head whose top
neighbours fall into clearly separate groups.

## 8. Stage 5: meaning beyond the words (needs approval)

Stages 1 to 4 bind and widen the words. They cannot find a dataset whose text
never says what it is about: Emo-FilM says "emotion" only in its paper, and
ds004274 says "aggression" in its title but not "social". Three existing
pieces close that gap. Each one sends catalogue metadata to an outside model.

1. **README profiles** (`backend/services/readme_profile.py`, runner
   `ingest/backfill/extract_readme_profiles.py`). Already written; 0 of 1,721 datasets
   have a profile on production. A profile lists research topics and
   paradigms with verbatim quotes. Adding the linked paper's abstract to the
   profile input would turn "Taylor Aggression Paradigm" into the topic
   "aggression" and the Emo-FilM paper into "emotion". The profile is indexed
   between the short fields and the README, so stage 3 searches it without
   further changes. Map verified paradigms to constructs the way conditions
   already map to diagnosis codes.
2. **Embedding of the gloss.** Embedding the raw question "social conflicts"
   mixes both meanings. Embedding the *gloss* ("disagreement, hostility or
   aggression between people or groups") is how the qualifier binding carries
   over to vector search. Test offline first with the bge vectors that
   `benchmarks/recall/embed_datasets.py` already produces: on the stage-0
   questions, compare lexical only, gloss embedding only, and both. Use it in
   production only if the wrong-sense rate drops and benchmark recall holds.
   It is then an OR-alternative inside the topic condition and a ranking
   term, never a hard filter.
3. **Per-requirement check of the top results** (`backend/services/rerank.py`,
   off by `RERANK_ENABLED=false`). Give the judge the gloss and `not_meant` as
   the requirement ("Is this dataset about conflict between people? Not:
   cognitive or response conflict"). It answers met, not met or not stated,
   with a quote. This is the strongest single disambiguator. Its cost is one
   extra Gemini call per new question, cached with the query.

Recommended order: profiles first (one-off cost, helps every question),
then the offline embedding test, then the re-ranker.

## 9. Stage 6: show the binding in the interface (1 day)

The filter chips (`/query/refine`) show a topic as one chip, not as separate
words:

- `Topic · social conflict`, with the gloss as tooltip and the alternatives
  that were searched listed underneath, each removable.
- `Not: cognitive conflict (Stroop, Simon)`, removable: removing it brings the
  other meaning back.
- For an ambiguous head without a qualifier, one chip per sense to pick
  from: `conflict: social · cognitive · approach-avoidance`.

A wrong guess is then visible and can be fixed with one click instead of a
rephrased question.

## 10. Order, effort and approvals

| Stage | Work | Effort | Approval |
|---|---|---|---|
| 0 | Ambiguity test set, judged pool, pinned parses | ½–1 day | none |
| 1 | `Topic` schema, prompt, plumbing | 1–2 days | none |
| 2 | `resolve_topic`, alternative validation | 1–2 days | none |
| 3 | `near` mode, tier mapping, `not_meant` demotion | 1–2 days | none |
| 4 | Sense inventory and mining script | 2 days incl. review | none |
| 5a | README profiles with paper abstracts | about 1 week incl. backfill | metadata to Gemini |
| 5b | Gloss embedding, offline test then production | 1 day + 3–5 days | metadata to an embedding API |
| 5c | Re-ranker with gloss and `not_meant` | 2–3 days | query-time metadata to Gemini |
| 6 | Topic chips | 1 day | none |

Stages 1 to 3 are one shippable unit, measured on stage 0 and on the 167-question
recall benchmark before deploy. Stage 4 can follow on its own.

## 11. Risks

- **Alternatives too broad.** "aggression" also matches "aggressive tumour".
  The frequency cap catches only part of this; `near` with the qualifier is
  stricter. Watch the wrong-sense rate per alternative in stage 0.
- **The LLM drops a qualifier or invents one.** The sense inventory makes the
  frequent cases deterministic. Pinned parses keep benchmark runs comparable.
- **Recall loss from dropping the head-alone tier.** Measured on the main
  benchmark. If a question loses datasets, keep the head alone at level 1 for
  non-strict questions.
- **Interaction with existing paths.** Few-shot examples in
  `backend/few_shot_examples.json` and the SQL model's copying of placeholders
  need a new example for the `near` mode. `services/uncertain_tier.py` must
  classify the new condition as text, not as a hard filter.
- **Shared working tree.** `preprocess.py` and `retriever.py` are edited by
  other sessions too; changes go in as exact-string edits, one stage per
  commit.

## 12. Status on 2026-11-14 (working tree, not committed, not deployed)

### Built

| Stage | What | Where |
|---|---|---|
| "emo" fixes | short-prefix rule, thin-construct rule | `backend/vocabulary/retriever.py` |
| 1 | `Topic` in the QueryPlan, prompt section "Topics" (examples are not benchmark questions) | `backend/llm/preprocess.py` |
| 2 | `resolve_topic`: sense pick, alternative checks (head or qualifier alone, excluded meaning, 0 datasets, more than `TOPIC_MAX_ALTERNATIVE_MATCHES`=90), task codes from the phrase minus what the head alone gives | `backend/llm/preprocess.py` |
| 3 | `near` text mode (gap ≤ 2, both orders for two words); one topic line per topic; rewriter re-adds a topic line the SQL model dropped | `backend/vocabulary/text_terms.py`, `backend/services/text_search.py`, `backend/services/sql_rewriter.py` |
| 3 | second tier: only the phrase is loosened (words anywhere); dropping the NOT guard is the alternatives' weaker form | `backend/services/uncertain_tier.py` |
| 4 | inventory of 10 heads, loader, mining script | `backend/vocabulary/senses.yaml`, `backend/vocabulary/senses.py`, `backend/vocabulary/tools/mine_senses.py` |
| 6 | "Topic" chip, removable "Not meant" chip (id + `n`) | `backend/services/query_filters.py`, `frontend/src/components/QueryChips.vue`, `frontend/src/api/index.ts` |
| 0 | 12 ambiguity questions (not judged yet) | `benchmarks/recall/ambiguity_queries.yaml` |

The topic condition handed to the SQL model:

```
(EXISTS (task codes)
 OR text_match('<phrase>', 'near')
 OR ((text_match('<alternatives of several words>', 'near')
      OR text_match('<one-word alternatives>', 'short'))
     AND NOT text_match('<not meant>', 'near')))
```

Tests: `backend/tests/test_topics.py` (20 tests); 255 backend tests pass
locally with sqlglot 30.11.0 and python-jose 3.5.0 installed and a stand-in
for the BIDS-SQL `db` package.

### Changed against the plan, after checking on production data

- Gap 2, not 3: gap 3 matched "social cognition. Simon conflict task".
- One-word alternatives search the short fields only. Anywhere in a README,
  "aggression" matched "the defacing was quite aggressive" and an aggression
  questionnaire, "provocation" an "intrusion provocation task".
- Only the phrase is loosened in the second tier. Loosening every
  alternative ("interpersonal & conflict" anywhere) brought back 20 unrelated
  datasets.
- Codes the head word alone gives are dropped: "cortical inhibition"
  reached the construct inhibition and with it stop-signal tasks.
- "discrimination" is not a social-conflict alternative: 36 datasets, mostly
  perceptual discrimination.
- `strict` was left out of the schema: nothing would use it yet.
- Sense choice chips for an ambiguous head without qualifier are not built.

### Measured so far: search side only, without the language model

Each ambiguity question was given a hand-written topic (phrase, head,
qualifiers, no model alternatives), turned into SQL by the new code, and run
read-only on production (1,721 datasets). Social conflict, judged by hand:

| | Confirmed | Relevant among them | Second tier | Relevant among them |
|---|---|---|---|---|
| Production now | 1 | 1 (ds003103) | 22 | 1 (ds002990, empty duplicate) |
| Topic search | 5 | 4 (ds000214, ds003791, ds004274, ds005375) | 7 | 1 (ds003103) |

The fifth confirmed result, ds004516, shows a family conflict in a film
stimulus. The other questions were looked at, not judged: cognitive conflict
returns Simon, flanker and Stroop datasets; monetary reward gambling and MID
datasets; cortical inhibition the one GABA dataset; psychosocial stress the
one public-speaking dataset (the catalogue has no TSST study); "emo" with
Morgenroth both Emo-FilM datasets.

### Measured with the language model (2026-11-14, production image, read-only)

Disposable backend containers on the production image, baseline = image as
deployed, candidate = the same image with the topic files mounted. Results in
`benchmarks/recall/results/2026-11-14_*`; ambiguity verdicts (blind, Claude
Sonnet) in `benchmarks/recall/ambiguity_truth.json` and
`ambiguity_judgments.jsonl`.

Ambiguity questions, 3 runs per version, mean over 12 questions:

| | Baseline | Topics |
|---|---|---|
| Relevant results on page 1 | 4.0 | 6.6 |
| Wrong-meaning results on page 1 | 2.7 | 1.1 |
| Precision of page 1 | 46% | 61% |
| Recall over the judged pool | 50% | 88% |

167 development questions, first candidate (topics for anything the model
called a topic): recall 59% → 57%, recall@20 53% → 51% (bootstrap interval
includes 0). All of the loss sat in the 24 questions where the model made a
topic (47% → 38%); the other 135 were unchanged (61% → 60%). The model made
topics of plain subjects ("sign language processing", "gambling choices")
and listed unwanted meanings for nearly all of them.

Fix: a topic stays a topic only when its head word is in `backend/vocabulary/senses.yaml`
or the user abbreviated it (`_keep_ambiguity_topics`); every other topic
goes back to the scan-term path as before. Rerun of the 24 questions on the
newer production image (8c8d657): baseline 39%, topics 43% recall. One run
of the ambiguity set with this rule: recall 50% → 75%, wrong-meaning results
2.7 → 1.3 per page (11 first-page results unjudged, counted as not
relevant).

Open problems seen in these runs, not caused by topics: the intent step
sometimes makes "healthy volunteers" a hard diagnosis filter, ANDs "films &
music", or requires a word like "manipulating"; such parses return 0. The
same question gives different parses between runs, so single runs move by
±8 points on small question sets.

### Held-out check (commit bc44114, 2026-11-14)

Run once, averages only, per `benchmarks/recall/held_out_questions.txt`.
Baseline: production image 96c719ec (its search files identical to
bc44114^); topics: the same image with the bc44114 files. 56 scorable
questions, no fallbacks or errors in either run.

| | Baseline | Topics |
|---|---|---|
| Recall | 48% | 52% |
| Recall@20 | 44% | 48% |
| nDCG@10 | 42% | 47% |
| Precision | 67% | 68% |
| Questions with nothing found | 20 | 16 |

Paired: recall@20 +4.3 points (95% bootstrap −2.2 to +10.5), nDCG@20 +5.1
(−1.5 to +11.3); better on 10 questions, worse on 3. The direction matches
the development and ambiguity results; the interval still includes zero.

### Not done yet

- A full 167-question run with the final rule (only the 24 affected
  questions were rerun).
- Stage 5 (profiles, gloss embedding, re-ranker): needs approval to send
  metadata to Gemini or an embedding API.
