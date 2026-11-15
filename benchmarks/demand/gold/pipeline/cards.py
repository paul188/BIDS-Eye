"""Evidence cards and judge batches."""
import json, re, os, collections
import lib, spec
D, ON = lib.load()
TREES = {}
if os.path.exists(f"{lib.G}/on_trees.jsonl"):
    for l in open(f"{lib.G}/on_trees.jsonl"):
        r = json.loads(l); TREES[r["id"]] = r
POOLS = json.load(open(f"{lib.G}/pools.json"))

def clip(s, n):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n] + "…"

def tree_summary(acc):
    t = TREES.get(acc)
    if not t or "root" not in t:
        return None
    root = [f for f in t["root"] if not f.startswith("sub-")]
    n_sub = sum(1 for f in t["root"] if f.startswith("sub-"))
    out = {"root_non_subject_entries": root[:40], "n_subject_folders": n_sub}
    for k, v in (t.get("sub") or {}).items():
        out[k + "/"] = v[:25]
    return out

def card(acc, patterns):
    d, o = D.get(acc), ON.get(acc)
    c = {"acc": acc, "in_our_catalogue": d is not None}
    if d:
        p = d.get("participants") or {}
        text = lib.our_text(d, papers=False)
        c.update({
            "name": d.get("name"),
            "uploader_labels": {k: d.get(k) for k in ("dx_status", "study_domain", "study_design", "species", "study_longitudinal") if d.get(k)},
            "recorded": {
                "datatypes": sorted(x for x in (d.get("all_datatypes") or []) if x),
                "task_labels_in_filenames": (d.get("raw_tasks") or [])[:18],
                "n_subjects": d.get("n_sub"), "sessions_per_subject_min_max": [d.get("ses_min"), d.get("ses_max")],
                "subjects_with_2plus_sessions": d.get("n_sub_ge2ses"),
                "subjects_per_file_type": {k: v for k, v in (d.get("sub_per_suffix") or {}).items() if k not in ("README", "datacite", "CITATION")} or None,
                "participants_tsv": {"rows": p.get("n_part"), "male": p.get("n_male"), "female": p.get("n_female"), "age_min": p.get("age_min_p"), "age_max": p.get("age_max_p"),
                                     "adults": p.get("n_adult"), "normalised_diagnoses": p.get("dx_counts"),
                                     "group_column": {k: v for k, v in (p.get("raw_groups") or {}).items() if k != "(none)"} or None,
                                     "other_columns": (p.get("extra_keys") or [])[:30]} if p else None,
                "dataset_age_range": [d.get("age_min"), d.get("age_max")],
                "field_strength_T": d.get("field"), "scanner_manufacturer": d.get("manufacturer") or d.get("manufacturer_bold"),
                "bold_TR_s": d.get("bold_tr"), "t1w_sequence_names": (d.get("t1_seq") or [])[:6] or None,
                "special_sequences": (d.get("special_seq") or [])[:8] or None, "asl_type": d.get("asl_type"),
                "eeg_sampling_hz": d.get("eeg_sfreq"), "seeg_channels": d.get("seeg_ch"), "ecog_channels": d.get("ecog_ch"),
                "eeg_electrodes_files": d.get("n_eeg_electrodes") or None, "physio_files": d.get("n_physio") or None,
                "events_files": d.get("n_events") or None, "phenotype_files": (d.get("phenotype_files") or [])[:20] or None,
                "derivatives": d.get("derivatives"), "task_descriptions": (d.get("task_desc") or [])[:4] or None,
                "licence": d.get("license"), "size_gb": round((d.get("size_bytes") or 0) / 1024**3, 1) or None,
            },
            "description": clip(d.get("description"), 400),
            "readme_head": clip(d.get("readme"), 600),
        })
        c["recorded"] = {k: v for k, v in c["recorded"].items() if v not in (None, [], {}, [None, None])}
        prim = [x for x in d.get("papers") or [] if x.get("role") == "primary"][:2]
        if prim:
            c["primary_papers"] = [{"title": x.get("title"), "year": x.get("year"), "abstract": clip(x.get("abstract"), 500)} for x in prim]
        paper_text = "\n".join((x.get("title") or "") + " " + (x.get("abstract") or "") for x in prim)
    else:
        s = (o.get("latestSnapshot") or {}); m = o.get("metadata") or {}; sm = s.get("summary") or {}
        ages = [a for a in (m.get("ages") or []) if isinstance(a, (int, float))]
        text = lib.on_text(o); paper_text = ""
        subj = sm.get("subjectMetadata") or []
        c.update({
            "name": (s.get("description") or {}).get("Name"),
            "note": "Not in our catalogue: facts below are OpenNeuro's own summary.",
            "uploader_labels": {k: m.get(k) for k in ("dxStatus", "studyDomain", "studyDesign", "species", "studyLongitudinal") if m.get(k)},
            "recorded": {"modalities": (sm.get("modalities") or []) + (sm.get("secondaryModalities") or []), "tasks": (sm.get("tasks") or [])[:18],
                         "n_subjects": len(sm.get("subjects") or []), "n_sessions": len(sm.get("sessions") or []),
                         "age_min_max": [min(ages), max(ages)] if ages else None,
                         "sex_counts": dict(collections.Counter((x.get("sex") or "?") for x in subj)) if subj else None,
                         "groups": dict(collections.Counter((x.get("group") or "?") for x in subj).most_common(8)) if subj else None,
                         "derivatives": [x["name"] for x in o.get("derivatives") or []] or None,
                         "size_gb": round((s.get("size") or 0) / 1024**3, 1) or None},
            "readme_head": clip(s.get("readme"), 700),
        })
    snips = []
    for pat in patterns:
        for sn in lib.snippets(text, pat, radius=150, limit=2):
            if all(sn[:50] not in x for x in snips):
                snips.append(sn)
        if paper_text:
            for sn in lib.snippets(paper_text, pat, radius=130, limit=1):
                if all(sn[:50] not in x for x in snips):
                    snips.append("[paper] " + sn)
    c["passages_around_keywords"] = snips[:6]
    t = tree_summary(acc)
    if t:
        c["openneuro_file_listing"] = t
    return c

def question_block(q, accs):
    return {"query_id": q["id"], "question": REQ[q["id"]]["question"], "original_request": REQ[q["id"]]["quote"],
            "relevant_if": q["relevant_if"], "partially_if": q.get("partially_if"), "candidates": sorted(accs)}

import yaml
REQ = {r["id"]: r for r in yaml.safe_load(open("/home/paul_johannssen/Desktop/Bids-Eye/benchmarks/demand/requests.yaml"))["requests"]}

def to_judge(q):
    """Candidates that need reading: everything pooled except those the facts already decide."""
    out = []
    for acc, why in POOLS[q["id"]].items():
        decided = any(w in ("facts:relevant",) for w in why) and q["id"] not in ("qf08_tof_swi", "qf40_mri_ct_paired", "qx08_seeg", "qx10_nhp", "qx09_female_only")
        only_facts = all(w.startswith("facts:") for w in why)
        if q["method"] in ("mixed", "judged") and not only_facts and not decided:
            out.append(acc)
        elif q["method"] in ("mixed", "judged") and q["id"] in ("qf08_tof_swi", "qf40_mri_ct_paired", "qx08_seeg", "qx10_nhp", "qx09_female_only") and not (only_facts and why == ["facts:partial"]):
            out.append(acc)
    return sorted(set(out))

def make_batch(name, qids, part=None, parts=1):
    cards, questions = {}, []
    for qid in qids:
        q = spec.BY_ID[qid]
        accs = to_judge(q)
        if parts > 1:
            accs = [a for i, a in enumerate(accs) if i % parts == part]
        pats = list(q.get("text", [])) + ([q["modality_text"]] if q.get("modality_text") else [])
        for a in accs:
            c = card(a, pats)
            if a in cards:      # same dataset in two questions of the batch: merge passages
                c["passages_around_keywords"] = (cards[a]["passages_around_keywords"] + [s for s in c["passages_around_keywords"] if s not in cards[a]["passages_around_keywords"]])[:9]
            cards[a] = c
        questions.append(question_block(q, accs))
    os.makedirs(f"{lib.G}/batches", exist_ok=True)
    path = f"{lib.G}/batches/{name}.json"
    json.dump({"batch": name, "questions": questions, "cards": cards}, open(path, "w"), indent=1, ensure_ascii=False)
    return path, len(cards), sum(len(x["candidates"]) for x in questions), os.path.getsize(path)

PLAN = [
    ("b01_psychosis", ["qf01_bipolar_schizophrenia", "qf20_schizophrenia_mri_png"]),
    ("b02_clinical_eeg", ["qf02_eeg_autism_controls_like_abide", "qf04_eeg_autism", "qf35_eeg_autism_patients", "qf14_eeg_alzheimer_and_normal", "qf33_eeg_alzheimer", "qf38_eeg_depression_labeled", "qp09_eeg_stress", "qf16_covid_brain_signals"]),
    ("b03_clinical_misc", ["qf24_fnirs_alzheimer", "qf28_preprocessed_fmri_alzheimer", "qp08_eating_disorders_mri", "qf03_rest_fmri_alcohol_questionnaires"]),
    ("b04_epilepsy_tumour", ["qf10_epilepsy_rest_fmri", "qf22_diffusion_tumor", "qf36_brain_tumor_classes", "qf37_tumor_segmentation_ground_truth"]),
    ("b05_acquisition", ["qf08_tof_swi", "qf09_pcasl_physio", "qf21_low_field_paired_3t", "qf27_7t_rest_fmri_mprage", "qf40_mri_ct_paired"]),
    ("b06_design_topic", ["qf12_rest_task_rest_one_day", "qp11_memory_consolidation", "qx02_spinal"]),
    ("b07_natural_images_fmri", ["qf13_natural_images_visual_cortex"]),
    ("b08_images_eeg", ["qf42_eeg_image_pairs_scenes"]),
    ("b09_videos", ["qf19_videos_people_talking"], 2),
    ("b10_simple_visual", ["qf29_simple_visual_task_20_subjects"], 2),
    ("b11_emotion_fmri", ["qf30_fmri_emotional_state"], 2),
    ("b12_emotion_eeg", ["qf43_eeg_emotion_like_deap"]),
    ("b13_language_dti", ["qf17_language_fmri_and_dti"]),
    ("b14_networks_nhp", ["qf31_brain_networks_phenotype", "qx10_nhp"]),
    ("b15_consciousness", ["qf32_connectomes_consciousness"], 2),
    ("b16_block_design", ["qf07_decoding_block_design_sessions"], 2),
    ("b17_preprocessed", ["qf11_preprocessed_fmriprep", "qf18_preprocessed_fmri_ml", "qf25_preprocessed_rs_fmri"], 3),
    ("b18_rest_review", ["qf15_rest_fmri"]),
    ("b19_rest_eeg_men", ["qf39_eeg_rest_adult_man_healthy"], 2),
    ("b20_eeg_ieeg_scanner", ["qf41_eeg_digitization", "qx04_non_siemens", "qx08_seeg"]),
    ("b21_female_words", ["qx09_female_only", "qp07_isolated_words"]),
    ("b22_language_descriptor", ["qp06_language_50_healthy_adults_descriptor"]),
    ("b23_bci", ["qp13_bci_paradigms"], 2),
]
if __name__ == "__main__":
    covered = set()
    total_cards = total_pairs = total_bytes = 0
    for item in PLAN:
        name, qids = item[0], item[1]; parts = item[2] if len(item) > 2 else 1
        covered |= set(qids)
        for part in range(parts):
            nm = name if parts == 1 else f"{name}_{part+1}"
            path, nc, npairs, size = make_batch(nm, qids, part, parts)
            total_cards += nc; total_pairs += npairs; total_bytes += size
            print(f"{nm:28} cards={nc:4} judgments={npairs:4} {size/1000:7.0f} kB")
    missing = [q["id"] for q in spec.Q if q["method"] in ("mixed", "judged") and q["id"] not in covered]
    print("not in any batch:", missing)
    print("total", total_cards, total_pairs, f"{total_bytes/1e6:.1f} MB", "| trees loaded:", len(TREES))
