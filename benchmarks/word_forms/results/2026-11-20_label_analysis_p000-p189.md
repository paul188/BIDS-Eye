# Word form labels: analysis of p000–p189 (2026-11-20)

**Inputs**
- Labels: p000–p189, read from Chrome's local storage (last save 2026-11-19 20:31 UTC) and copied to
  `../labels_p000-p189_chrome_2026-11-19.json`. p190–p266 are missing from that storage. This report says nothing
  about those 77 pairs.
- Corpus: the local copy `bidseye_fts_test` (1,943 datasets, 1,713 with a README). Name, description and README,
  lower-cased and split on non-alphanumerics, without stemming.
- Gemini: `llm_gemini-2.5-flash_word-forms-1ed381843dc5.json` from 2026-11-19 (words and counts only).
- Scripts: `../analysis/` (they expect the corpus dump made by `corpus.py`).

Labels: 98 both, 78 neither, 12 context, 2 one-way.

## 1. Scores (`score.py`, per direction)

| method | directions right | wrong split (lost recall) | wrong merge (false hits) |
|---|---|---|---|
| stem everything | 55.6% | 0 of 198 | 158 of 158 |
| split everything | 44.4% | 198 of 198 | 0 of 158 |
| gemini-2.5-flash | **76.4%** | 18 of 188 (9.6%) | 44 of 146 (30.1%), plus 11 pairs it answered "context" |
| rule: inflections merge, derivations split | 68.0% | 73 of 198 (36.9%) | 41 of 158 (25.9%) |
| rule above, and split where a form is phrase-bound | 65.7% | 91 of 198 | 31 of 158 |
| Gemini, and split where a form is phrase-bound | 70.8% | 54 of 190 | 34 of 150 |

- **Random sample only** (138 pairs drawn from colliding stems): the stem is wrong in 79 of 256 decided directions (31%).
- **Agreement with word_forms.yaml.** The `exact` entries agree with the blind labels on 77 of 84 directions. The
  7 disagreements are all `dependence` and `structure`. The `stemmed` entries disagree on 2 directions
  (emotion ↔ emotionally), and you marked cognitive ↔ cognition as context.
- **The simple rules don't help.** No rule beats Gemini overall. "Phrase-bound" means a form's most frequent
  neighbour phrase covers ≥40% of its datasets. Splitting a whole word pair because of that is too blunt.

## 2. What the labels follow, read in the neuroimaging context

**Part of speech matters, but it doesn't decide alone.** In the random sample:

- Inflections (the same word with -s/-ed/-ing): you merged 82% (61 of 74 decided pairs).
- Derivations (a new word, often a different part of speech: -al, -ion, -ive, -ly, -er, -ity): you merged 50% (27 of 54).

**Inflections you split.** In each, the -ed or -ing form became an adjective or noun with its own meaning in the
field, or one form is fixed boilerplate. The numbers give how many of the form's datasets use the phrase.

| form | what it mostly is in the catalogue |
|---|---|
| weighted | "T1 weighted" 271 of 343 datasets, "T2 weighted" 115 |
| handed | "right handed" 81 of 87 |
| recorded / recording / recorder | "recorded using" (procedure); "EEG recording" (noun); "BrainVision Recorder" (software) |
| record | "data record", "medical records" |
| scanning | "scanning session", "fMRI scanning" |
| scanned | "scanned using" |
| stripping | "skull stripping" 6 of 6 |
| strip | "strip electrodes", "grid strip" (ECoG) |
| aged | "aged 18–35" |
| aging | "healthy aging", "cognitive aging" |
| review | "institutional review board" 40 of 75 |
| reviewed | "peer reviewed" |
| imagined | "imagined speech" |
| imagining | the everyday verb |
| fix | "fix their gaze", fixation |
| fixed | "fixed order" |
| added | "comments added" 44 of 109 (OpenNeuro curator note), which is why "AD" ≠ "added" |

**Derivations you merged.** They name one concept in the field:

- behaviour / behavioural
- depression / depressive / depressed
- attention / attentional / attentive
- music / musical
- prediction / predictive
- acceleration / accelerated (multiband)
- independent / independently

**Derivations you split.** The derived word is a term of art:

| split pair | why they differ |
|---|---|
| functional ≠ functioning | fMRI vs "executive functioning" 14 of 23 |
| clinical ≠ clinic | clinic is mostly a name: "Mayo Clinic", "Cleveland Clinic" |
| temporal ≠ temporally | "temporal lobe" vs timing |
| superior ≠ superiorly | anatomy vs direction |
| localizer ≠ localization / localize | "source localization" in EEG |
| avoidance ≠ avoid | "harm avoidance", "approach–avoidance" vs the instruction "avoid movement" |
| random ≠ randomized | "random dot", "random order" vs randomized trials and orders |
| selective ≠ selected | "selective attention" vs "randomly selected" |
| positive / positivity ≠ position | valence, "reward positivity" vs head or electrode position |
| measurement ≠ measure | "measures" are instruments: "clinical measures", "symptom measure" |
| medication ≠ medical | |
| comprehension ≠ comprehensive | |
| personality ≠ person | "contact person" 39 of 73 |
| experience ≠ experiment | |

## 3. Where Gemini goes wrong

Gemini judges the words as general English. It doesn't know how they are used in BIDS READMEs.

**Wrong merges.** Nearly all of them are the cases in section 2:

- functional / functioning
- recording / record
- scanned / scanning
- localizer / localization / localize
- predictive / predicted / predict
- avoid / avoidance
- automated / automation
- measurement / measure
- aging / aged and aging → age
- clinic / clinical
- medical → medication
- random / randomized
- experiment → experience
- imagined / imagining
- reviewed / review
- emotion / emotionally

**Wrong splits.** These are mostly one-way answers it read off the counts where you said both:

- transfer, mix/mixed, verify, pool/pooled, task/tasked, results/resulted
- reviewer/review, experimental/experimenter, periodicity, scaled/scale

It said "context" for weighted and handed, which the catalogue settles clearly.

**What Gemini lacks is the catalogue usage**, for example "weighted: T1 weighted in 271 of 343 datasets". A second
run could give it each form's top neighbour phrases with dataset counts. That is aggregated two-word counts, not
README text, but it goes beyond "words and counts only", so it needs your OK.

## 4. Pairs worth a second look

You can't have been aware of these while labelling blind.

| pairs | your labels | what the catalogue shows | suggestion |
|---|---|---|---|
| dependence ↔ dependent (p115), dependence ↔ depending (p004) | both, both (yaml splits) | "dependent variables" 31, "level dependent" (BOLD) 27, "dependent BOLD" 21 of 87; "depending on" is filler. "dependence" itself: alcohol, nicotine, substance dependence | dependent → dependence yes, dependence → dependent no; depending: neither |
| structure ↔ structural (p185) | both (yaml splits) | "data structure" 189 of 303, "structure BIDS" 72 | structure → structural yes, structural → structure no. Your p137 already says structurally → structure no |
| emotion ↔ emotionally (p168) | neither | you said emotional ↔ emotionally both; yaml `stemmed` merges the family | decide whether the yaml line stays |
| cognitive ↔ cognition (p076) | context | yaml `stemmed` merges them; "cognition" is often a journal or lab name | |
| aging ↔ age (p018), ageing ↔ age (p171) | neither / context | the same word in two spellings | make them the same |
| review ↔ reviewed, review ↔ reviewer, reviewer ↔ reviewed | neither, both, both | | non-transitive |
| scaling ↔ scale, scaled ↔ scale, scaled ↔ scaling | neither, both, both | | non-transitive |
| experimental ↔ experimenter | both | you split experimenter from experimentation and experimentally | |

One-way and non-transitive labels are fine as data. word_forms.yaml can't store them, though: every member of an
`exact` group gets the whole group (`RAG/word_forms.py` `_reviewed`). "recorded finds record, record doesn't find
recorded" needs entries keyed by the searched word.

## 5. Context

All 12 context pairs have a research sense and a README-housekeeping sense:

| word | research sense | housekeeping sense |
|---|---|---|
| reading / read | reading task, pre-reading children | "details can be read in the preprint" |
| writing / write | handwriting task | "at the time of writing", "BIDS writing via MNE-BIDS" |
| compensation | neural compensation | payment of 50 €; CTF gradient compensation |
| cross / crossed | fixation cross | cross-over trial; mice crossed to C57BL/6J; fully-crossed design |
| exclusion | Penn Conditional Exclusion Task | inclusion/exclusion criteria |
| cognition | the field | journal and lab names |
| measuring, appearing, applicability, illustrative, independence | | |

A decision about the word can't separate these senses. Only a decision about each occurrence can. The short
fields (name, description, study fields, LLM conditions) almost never carry the housekeeping sense:

| word | READMEs | short fields |
|---|---|---|
| exclusion | 64 | 0 |
| compensation | 22 | 0 |
| scanning | 184 | 1 |
| recorded | 304 | 11 |
| weighted | 341 | 8 |
| cross | 181 | 38 |
| reading | 54 | 22 |
| cognition | 45 | 31 |

(The local copy has stand-in LLM profiles, so the profile couldn't be checked.)

Possible handling, cheapest first:

1. For words labelled context, search only the short fields and the LLM profile (the existing `short` mode).
2. Remove known boilerplate phrases from the README before indexing ("exclusion criteria", "review board",
   "at the time of writing", "data structure", "public domain", "comments added"). Section 7 shows these can be
   mined.
3. Have the LLM tag word senses per dataset at index time. This is the most expensive option.

## 6. Part of speech

The 'english' configuration is the Snowball stemmer. It strips suffixes and knows neither part of speech nor
lemma.

**The inflection signal is real but not enough on its own.** The inflection-only rule scores 68% against
Gemini's 76%. It splits the derivations you merged.

**Where a tagger would help most:** -ed and -ing forms used as adjectives or nouns (weighted, handed, based,
fixed, recording, scanning, stripping). A POS-aware lemma index would keep "recording/NOUN" apart from
"record/VERB" and "weighted/ADJ" apart from "weight/NOUN".

**Testing it** needs spaCy and an English model, which are not installed. The test would score a POS-aware lemma
method on these labels, as a feature next to the LLM rather than a replacement.

## 7. Multi-word terms

**Today:**

- A hyphenated compound must appear adjacent.
- A `near` term needs its words within 2 positions.
- A topic searches its phrase with `near` (LLM_preprocessor/preprocess/topics.py).
- A plain `text_match` defaults to `all`: the words anywhere in the text.
- The uncertain tier relaxes `near` to words anywhere.

No list says "this is a term, always match it as one". Whether a term is kept together depends on which path
wrote the condition. I haven't traced every path.

**How much it matters.** Datasets with all words anywhere vs. the words as a phrase:

| term | words anywhere | as a phrase |
|---|---|---|
| cognitive control | 73 | 22 |
| visual attention | 61 | 16 |
| single trial | 71 | 10 |
| head motion | 87 | 33 |
| visual cortex | 56 | 27 |
| event related | 100 | 58 |
| block design | 62 | 35 |
| motor learning | 19 | 2 |
| semantic memory | 12 | 2 |
| resting state | 319 | 310 |
| working memory | 69 | 67 |

- **Across vocabularies:** the 517 MeSH and Cognitive Atlas multi-word terms that occur as a phrase anywhere in the
  catalogue have 4,650 bag-of-words matches, but only 1,621 contain the phrase. That is 65% extra.
- **`near` is the right tool:** with gaps up to 3 it finds almost exactly the adjacent matches. The mechanism
  exists; what's missing is the list.
- **Mining works:** 229 bigrams in ≥15 datasets are adjacent in ≥70% of the datasets that contain both words
  (PMI > 1). About half are domain terms: resting state, T1 weighted, white matter, fixation cross, eyes
  open/closed, phase encoding, slice timing, test retest, decision making, movie watching. The other half are
  boilerplate: public domain, informed consent, review board, exclusion criteria, comments added, data structure,
  BIDS format.
- **Spellings vary:** restingstate (8), nback (7), gonogo (5), eyetracking (10), fieldmap (49) vs field map (42).
  A term list needs the joined, hyphenated and spaced forms.

**Relation to word forms.** Phrase handling removes only 6% of the unwanted hits that stemming adds (557 of
8,790 dataset hits across the "no" directions). "weighted" is the one big exception: 249 of its 325 extra
datasets come from T1/T2 weighted. These are two separate fixes.

## 8. Open

1. p190–p266: labels not found in Chrome's storage.
2. The second-look list in section 4.
3. Second Gemini run with aggregated phrase counts (needs OK).
4. A reviewed phrase list: Cognitive Atlas + MeSH multi-word entries + mined domain bigrams, always matched with
   `near`; boilerplate bigrams as an index-side stop-phrase list.
