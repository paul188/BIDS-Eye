# Clause pilot: 108 drafted items

Does the search separate a request into the right logical clauses? These 108
items are the first gold for that question. Why and how: `reports/Logical
clause separation datasets.md` (not in git).

**Status: draft.** Claude wrote or corrected every item and has read the
planner's prompt. A second Claude session reviewed the first 80
(`reviews/2026-11-25_claude.json`: 59 right, 20 wrong, 1 ambiguous); its points
were applied where they hold (`revised` items). No person has checked them. Do
not report scores on them before that.

| File | What |
|---|---|
| `own.yaml` | 68 requests in our domain. 14 are rewritten from forum requests, selection rules or a live search (`source`), 54 are invented. own-45 to own-68 (blocks `hard_*`) are harder: background stories, mixed AND/OR, long negations, self-corrections, separators, spoken or misspelled |
| `chia_items.yaml` | 40 lines of the Chia corpus (Kury et al. 2020, CC BY 4.0) with the logic corrected by hand. A stress set: the lines are no search requests |
| `build.py` | Checks spans, trees and readings; writes the files below |
| `pilot.jsonl` | All items, with the full Chia line and its original relations |
| `review.html` | Check the draft: right, wrong, ambiguous, unsure |
| `blind.html` | Second labeller: the texts only, without the draft |
| `score_plans.py` | Runs the planner on the own requests and compares its plans with the gold |
| `test_score_plans.py` | The scorer on hand-written right and wrong plans |

## Format

- A clause is one condition: `field`, accepted `values`, `op` between the
  values (`any` / `all`), `words` of the text, `attach` (dataset, `group:n`,
  `same_subject:c`, `same_scan:c`) and `strength` (`must` / `wish`). Chia
  clauses have only `words`: their fields are not ours.
- A reading is a tree over the clauses: `&`, `|`, `!`, `-`, brackets. `a - b` is
  "a except b", some a that is not b: "weight loss drugs other than metformin"
  holds for a person on metformin and another drug, which `a & !b` would miss. Negation is
  only in the tree. Two trees count as equal when their truth tables are.
- Several readings mean the text is ambiguous: such items are scored apart, any
  accepted reading counts.
- `representable` says whether the plan (QueryPlan) can hold the reading. It
  belongs to the reading and was judged from the code of 2026-11-25.
- `also` lists plans of the same reading that are accepted too: in own-19 the
  parent term as a third alternative, in own-36 a minimum of one participant
  (owner, 2026-11-25, after the first run).
- Wishes, examples and optional conditions (`strength`) are clauses; they need not be in a tree.
- Another attachment of a clause (`attach_alternatives`) does not change the tree
  and is scored apart.
- Chia: `header` is the list the line stands in; the tree leaves it out.

## Conventions

Decided by the owner on 2026-11-25: a plain "no X" / "not X" removes a dataset
that has X anywhere. A second reading is added only where the wording describes
the people or the wanted scan (own-14, own-16, own-33; own-04, own-11 and own-13
also keep two).

Confirmed by the owner on 2026-11-25:

1. "unmedicated", "drug-naive", "MRI-negative", "non-smokers" are wanted
   phrases, not exclusions (`convention: wanted_phrase`, own-07 to own-10).
2. Diagnoses in a plain list are alternatives (own-25).
3. "healthy" and "only" are conditions the plan cannot hold (own-12, own-26, own-27).
4. "besides X" in our requests adds X (own-21); in Chia it can be an exception (ch-25).
5. Excluding children or animals is not the same as wanting adults or humans,
   since a dataset can hold both (own-35, own-37).

## Limits

Own items: 47 count for the headline (one reading, and the plan can hold it):
23 of the first 44 and all 24 harder ones. 10 are ambiguous, 11 have no reading
the plan can hold. own-41 to own-44 replaced items that left the headline. The
harder requests were written before the planner saw them and none was changed
afterwards, but by a writer who had seen its earlier plans; nobody has reviewed
them. Intervals stay wide: 18 of 24 right is 55 to 88% (Wilson, 95%).

## Scoring

    GEMINI_API_KEY=... python score_plans.py run --out plans.json --repeats 3
    python score_plans.py score plans.json --errors

`run` makes one planner call per request and repeat and continues a stopped run.
`score` needs no model. It reads the plan as written, without the code that
follows it, and does not score attachment or the Chia lines. A clause counts as
stated in whichever field the plan holds it, so "drug-naive first-episode
psychosis" as one diagnosis term counts (own-08) although the search may not
resolve it.
