# General-leaf fold / restructure — plan & runbook

Companion to the audit in [`general_leaves_to_merge.txt`](general_leaves_to_merge.txt) and the
transform script [`fold_general_leaves.py`](fold_general_leaves.py).

## Context / problem
Umbrella queries under-returned because an umbrella term resolved to a **narrow,
near-empty leaf** instead of its category. Example: `"memory tasks"` resolved to
`memory_task_general` (2 datasets) rather than the memory family; memory+fMRI
returned 0 of ~118 true matches.

## How resolution works (why the fix is shaped this way)
From `yaml_to_llamaindex.py` / `retriever.py`:
- A concept is a **group** when something lists it in `broader`; a group resolves to
  **all descendant `standard_code`s** (`_all_codes`, DFS) and is preferred over leaves
  on exact/fuzzy match (`retriever.py:298-319`).
- The DB columns (`bids_objects.task/suffix`, `bids_participants.diagnosis`) store
  `standard_code`s; `leaf_db` maps every `standard_code` (+ its `codes`/`dataset_codes`)
  back for matching.
- A node may be a **dual node** (has `standard_code` AND children) — group + leaf at once.
- A pure group has **no DB code of its own**; it only exists as the union of descendants.

So a leaf cannot simply be deleted (its DB rows would orphan and drop from the group's
expansion). The fix either keeps the `standard_code` value on a surviving node or remaps
the DB rows.

## What was changed (11 audited "general-only" leaves)
Applied by `python RAG/fold_general_leaves.py` to `RAG/value_mappings.yaml`.

### A) 7 single-leaf groups → dual node (no DB change)
The group absorbs the leaf's `standard_code` + `codes` + synonyms (+ the leaf's label as a
synonym); the leaf entry is deleted. The `standard_code` value is preserved on the group, so
DB rows still resolve and the group now expands to include it.

| deleted leaf | group → dual node (gets `standard_code`) |
|---|---|
| recognition_task_general | recognition |
| learning_task_general | associative |
| perception_general | sensory_and_perception |
| social_task_general | social_cognition |
| spatial_memory | spatial_memory_group |
| theory_of_mind | theory_of_mind_group |
| field_map_general | field_maps |

### B) memory (keep retrieval sub-group)
`memory_task_general` is folded into `memory_retrieval_general` (kept — it parents 36
recall/retrieval tasks). The leaf is deleted; its synonyms/codes move onto
`memory_retrieval_general`. **DB merge required** (2 rows) — see runbook.

### C) ALS → `MND > ALS > subtypes`
- `motor_neuron_disease_general` becomes the umbrella **group** (dual node; keeps its 1 DB
  row), `broader` gains `spinal_neuromuscular`.
- `als_general` merges **into** `amyotrophic_lateral_sclerosis` (the former empty phantom):
  that node gets `standard_code=als_general` (preserves the 1 DB row, no remap) + the
  synonyms/codes, `broader=[motor_neuron_disease_general]`, `is_group=true`.
- `als_upper_limb_dominant` (and any other `als_spectrum` child) re-parents under
  `amyotrophic_lateral_sclerosis`.
- the now-empty `als_spectrum` group is deleted.
- **No DB change** (both ALS-area `standard_code`s preserved on dual nodes).

## Verification (done locally, read-only)
- Every umbrella term resolves to its **group expansion**; all dual-node `standard_code`s
  remain in `leaf_db`; deleted keys are gone; **0 orphaned `broader` references**.
- Recall gain vs. the production DB (read-only):
  recognition **2 → 17**, field map **26 → 362**, memory tasks **2 → 52** datasets.

## Production runbook (Hetzner `178.104.108.189`, project `bids-eye`)
IMPORTANT: the vocab is **baked into the backend image** (`Dockerfile: COPY RAG/ /app/RAG/`,
no bind-mount), so `scp` + restart is NOT enough — the image must be **rebuilt**.
The DB merge touches **~1937 `bids_objects` rows across 2 datasets** (per-scan rows, not 2).
Do the DB merge and the rebuild together; until the merge runs the old `memory_task_general`
scan rows are orphaned by the new YAML.

```bash
# 0) (recommended) back up the prod YAML before overwriting
ssh -i ~/Desktop/hetzner_key root@178.104.108.189 \
  "cd /home/BIDS-Eye/RAG && cp -p value_mappings.yaml value_mappings.yaml.bak.$(date +%Y%m%d-%H%M%S)"

# 1) ship the vocab into the build context
scp -i ~/Desktop/hetzner_key RAG/value_mappings.yaml root@178.104.108.189:/home/BIDS-Eye/RAG/

# 2) the one DB merge (~1937 rows / 2 datasets; ALS + the 7 dual-node folds need NO DB change)
ssh -i ~/Desktop/hetzner_key root@178.104.108.189 \
  "docker exec bids_postgres psql -U user -d bids_sql -c \
   \"UPDATE bids_objects SET task='memory_retrieval_general' WHERE task='memory_task_general';\""

# 3) REBUILD the backend image (bakes the new YAML) and recreate
ssh -i ~/Desktop/hetzner_key root@178.104.108.189 \
  "cd /home/BIDS-Eye && docker compose -f docker-compose-hetzner.yml build backend \
   && docker compose -f docker-compose-hetzner.yml up -d backend"

# 4) verify the container actually has the new vocab
ssh -i ~/Desktop/hetzner_key root@178.104.108.189 \
  "diff <(md5sum < /home/BIDS-Eye/RAG/value_mappings.yaml) \
        <(docker exec bids_backend md5sum < /app/RAG/value_mappings.yaml) && echo MATCH"
```

## Out of scope (future)
- The 8 BORDERLINE leaves in the audit.
- The `bold_acquisition_general` side-note (carries `fMRI` synonyms; already mitigated by
  `retriever._PURE_MODALITY_TERMS`).
- Any resolver-code change — this was vocabulary-only.
