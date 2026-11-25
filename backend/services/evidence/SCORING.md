# The evidence score

How the evidence a dataset has for a query becomes one number from 0 to 1.
The uncertain tier computes it in its SQL and returns it as the column
`evidence_score`. The default order of results does not use it yet
(`results/rankers.py`, `EvidenceFirst`); a ranker can sort by it.

The reasoning and the sources are in
`reports/Combining evidence into query scores.md`. This file says what the
code does.

## Three sizes

| Size | Class | Its score |
|---|---|---|
| one way a clause is met | `PieceOfEvidence` | `score()` |
| one AND clause of the query | `ClauseEvidence` | `score_sql()` |
| the whole query | `QueryEvidence` | `score_sql()` |

## A piece

`p` is the share of datasets with this piece of evidence that are relevant,
divided by the share for a clause that holds as generated. So 1 means "as
good as confirmed" and 0 means "never relevant".

| Piece | p | From |
|---|---|---|
| a weaker form of a text search (`WeakerSearch`) | 0.37 | 555 judged rows |
| the text names the age group (`AgeGroupNamed`) | 0.53 | 92 judged rows |
| ages not recorded | 0.33 | 21 judged rows |
| diagnosis not recorded | 0.04 | 95 judged rows |
| any other probable piece | 0.4 | hand-set (`base.DEFAULT_SCORES`) |
| any other "not recorded" | 0.1 | hand-set |

The measured values are from the recall benchmark with an LLM judge
(2026-11-13). The ranges of the first three overlap; only the last is
established as low.

On the scale where independent evidence adds up, a piece weighs

    h = -ln(1 - p)

Each piece belongs to a source group (`Source.group`):

- **records**: the participant list, the dataset's columns, its files
- **own text**: title, description, README
- **paper**: a paper the dataset names

The other sources (an assumed default, "not recorded") have no group; asking
for it is an error. A weaker piece without a group must be the only one of
its clause.

## A clause

    S = 1    the clause holds as generated
    S = 0    the dataset records the field with values that do not fit

Otherwise, over the pieces that hold for the dataset:

    H_g = the largest h among the pieces of group g
    h0  = the weight of the field's "not recorded" score, if the dataset
          records nothing for it; else 0
    H   = max(h0, H_g of every group)
          + λ · Σ over the groups other than the strongest of max(H_g - h0, 0)
    S   = 1 - exp(-H)

In words:

- **One source says one thing once.** Inside a group only the strongest
  piece counts: the README repeating the title is not a second witness.
- **A second source adds a share.** The strongest group counts in full, each
  other group with the factor λ (`SHARED_EVIDENCE_DISCOUNT`, 0.5). With 0 the
  clause would score its strongest piece; with 1 the groups would count as
  fully independent (1 - (1-a)(1-b)).
- **"Not recorded" is counted once.** A measured hint such as 0.53 already
  contains what the missing field alone is worth (0.33), so only what a
  second group has above that floor is added.
- **Returns diminish.** Each addition closes a share of what is left to 1;
  S never reaches 1 and never falls below the strongest piece.

Examples with λ = 0.5:

| Pieces that hold | S |
|---|---|
| weaker text match (0.37) | 0.37 |
| only in a paper (0.4) | 0.40 |
| weaker text match and a paper | 1 - 0.60 · 0.63^0.5 = 0.52 |
| ages not recorded, the text names the age group (0.53) | 0.53 |
| ages not recorded, no hint | 0.33 |

## Several sources for a confirmed clause

A clause that holds as generated scores 1 whichever source confirms it, so
the score cannot order two confirmed datasets. The column `corroboration`
does: over the confirmed clauses, the number of source groups that confirm a
clause beyond the first, summed. A diagnosis coded for the participants and
named in the README has 1, one of the two has 0. Pieces of one group count
once, and an assumed default counts for nothing. The uploader's labels
(`dx_status`, `study_domain`, `study_design`) count with the own text,
because `search_tsv` holds them. The column exists only when
some clause can be confirmed by two groups.

`EvidenceFirst` sorts by it after the strength, the judgment and the user's
own words, and before the score of the text and the codes. Measured on
2026-11-25 against the order without it, same SQL for both: nDCG@10 up 0.009
on 136 development questions (95% interval 0.002 to 0.017) and 0.011 on 56
gold requests (-0.001 to 0.029); recall@20 did not change on development.
Placed after the score it changes nothing. The held-out questions were not used.

## A query

    Q = S_weakest · (product of the other clauses' S)^κ

κ is `CLAUSE_INDEPENDENCE`, 1.0: the plain product of all clauses. With 0 it
would be the minimum. The tier keeps a dataset only if at most one clause is
not confirmed (`UNCERTAIN_MAX_UNSURE`), so today every kept dataset has at
most one S below 1 and Q is that S, whatever κ is and however many clauses
the query has. The score is not normalised by the number of clauses: that
would make it depend on the number.

## What is hand-set

- λ = 0.5 and κ = 1.0
- the two default scores, and the assignment of sources to groups
- a clause that holds as generated scores 1 from whichever source

None of these is measured. The report lists what to label to measure them.

## Where it is in the code

- `evidence/base.py`: `PieceOfEvidence.score`, `DEFAULT_SCORES`,
  `Source.group`
- `uncertain/conditions.py`: `ClauseEvidence.score_sql`,
  `QueryEvidence.score_sql`, the two constants
- `uncertain/tier.py`: one boolean column per piece where a clause has
  several, and `evidence_score`
- `tests/test_evidence_score_postgres.py`: the examples above on a real
  database
