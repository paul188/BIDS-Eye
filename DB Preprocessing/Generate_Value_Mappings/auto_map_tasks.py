#!/usr/bin/env python3
"""
auto_map_tasks.py — Automatically map all DB task values to human-readable labels.

Queries the DB for every distinct task value, tries to map it using:
  1. A large built-in dictionary of known neuroimaging task names
  2. Pattern/substring matching for variants

Writes all confident mappings directly into TASK_LABEL in value_mappings.py.
Prints a report of anything it couldn't map (for manual review).

Usage:
    python auto_map_tasks.py --db-url "postgresql://user:password@localhost:5429/bids_sql"
    python auto_map_tasks.py --db-url "..." --dry-run   # print mappings without writing
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import value_mappings as _vm

MAPPINGS_FILE = Path(__file__).resolve().parent / "value_mappings.py"

# ── Master task dictionary ─────────────────────────────────────────────────────
# Covers common task names across OpenNeuro, HCP, UK Biobank, and lab datasets.
# Keys are lowercased raw DB values.
# Values are (human_label, category)

MASTER: dict[str, tuple[str, str]] = {
    # ── Resting state ─────────────────────────────────────────────────────────
    "rest":                     ("resting-state",                    "resting-state"),
    "resting":                  ("resting-state",                    "resting-state"),
    "restingstate":             ("resting-state",                    "resting-state"),
    "resting-state":            ("resting-state",                    "resting-state"),
    "rsfmri":                   ("resting-state fMRI",               "resting-state"),
    "rs":                       ("resting-state",                    "resting-state"),
    "rest1":                    ("resting-state (run 1)",            "resting-state"),
    "rest2":                    ("resting-state (run 2)",            "resting-state"),
    "restpre":                  ("resting-state (pre)",              "resting-state"),
    "restpost":                 ("resting-state (post)",             "resting-state"),
    "rest_pre":                 ("resting-state (pre)",              "resting-state"),
    "rest_post":                ("resting-state (post)",             "resting-state"),
    "rest_eyes_open":           ("resting-state eyes open",          "resting-state"),
    "rest_eyes_closed":         ("resting-state eyes closed",        "resting-state"),
    "resteyesopen":             ("resting-state eyes open",          "resting-state"),
    "resteyesclosed":           ("resting-state eyes closed",        "resting-state"),
    "ec":                       ("eyes closed resting-state",        "resting-state"),
    "eo":                       ("eyes open resting-state",          "resting-state"),
    "eyesclosed":               ("eyes closed resting-state",        "resting-state"),
    "eyesopen":                 ("eyes open resting-state",          "resting-state"),

    # ── Working memory ────────────────────────────────────────────────────────
    "nback":                    ("n-back working memory",            "working memory"),
    "n-back":                   ("n-back working memory",            "working memory"),
    "nbackwm":                  ("n-back working memory",            "working memory"),
    "wm":                       ("working memory",                   "working memory"),
    "workingmemory":            ("working memory",                   "working memory"),
    "working_memory":           ("working memory",                   "working memory"),
    "0back":                    ("0-back working memory",            "working memory"),
    "1back":                    ("1-back working memory",            "working memory"),
    "2back":                    ("2-back working memory",            "working memory"),
    "3back":                    ("3-back working memory",            "working memory"),
    "wmc":                      ("working memory capacity",          "working memory"),
    "ospan":                    ("operation span working memory",    "working memory"),
    "spatialwm":                ("spatial working memory",           "working memory"),
    "verbalwm":                 ("verbal working memory",            "working memory"),
    "vwm":                      ("visual working memory",            "working memory"),

    # ── Attention / inhibition ────────────────────────────────────────────────
    "flanker":                  ("Flanker inhibition",               "attention"),
    "eflanker":                 ("Eriksen Flanker",                  "attention"),
    "stopsignal":               ("stop-signal",                      "attention"),
    "stop":                     ("stop-signal",                      "attention"),
    "stop-signal":              ("stop-signal",                      "attention"),
    "stopSignal":               ("stop-signal",                      "attention"),
    "gonogo":                   ("go/no-go",                         "attention"),
    "go-no-go":                 ("go/no-go",                         "attention"),
    "gng":                      ("go/no-go",                         "attention"),
    "goNogo":                   ("go/no-go",                         "attention"),
    "stroop":                   ("Stroop",                           "attention"),
    "colorstroop":              ("color Stroop",                     "attention"),
    "ant":                      ("Attention Network Task",           "attention"),
    "attention":                ("attention",                        "attention"),
    "sart":                     ("sustained attention to response",  "attention"),
    "gradcpt":                  ("gradual-onset CPT",                "attention"),
    "cpt":                      ("continuous performance task",      "attention"),
    "pvt":                      ("psychomotor vigilance task",       "attention"),
    "vigilance":                ("vigilance",                        "attention"),
    "tsa":                      ("task-switching attention",         "attention"),
    "taskswitch":               ("task switching",                   "attention"),
    "switching":                ("task switching",                   "attention"),

    # ── Memory ────────────────────────────────────────────────────────────────
    "memory":                   ("memory",                           "memory"),
    "encoding":                 ("memory encoding",                  "memory"),
    "retrieval":                ("memory retrieval",                 "memory"),
    "recall":                   ("memory recall",                    "memory"),
    "recognition":              ("recognition memory",               "memory"),
    "sceneencoding":            ("scene encoding",                   "memory"),
    "scenerecognition":         ("scene recognition",                "memory"),
    "wordencoding":             ("word encoding",                    "memory"),
    "wordrecognition":          ("word recognition",                 "memory"),
    "pairedassociates":         ("paired-associate learning",        "memory"),
    "assocmem":                 ("associative memory",               "memory"),
    "ltpfr":                    ("long-term memory free recall",     "memory"),
    "ltpfr2":                   ("long-term memory free recall (v2)","memory"),
    "sourcememory":             ("source memory",                    "memory"),
    "prospectivememory":        ("prospective memory",               "memory"),
    "spatial":                  ("spatial memory",                   "memory"),
    "spatialmemory":            ("spatial memory",                   "memory"),
    "mst":                      ("mnemonic similarity task",         "memory"),

    # ── Language ──────────────────────────────────────────────────────────────
    "language":                 ("language processing",              "language"),
    "reading":                  ("reading",                          "language"),
    "story":                    ("story comprehension",              "language"),
    "storycomprehension":       ("story comprehension",              "language"),
    "listening":                ("auditory language",                "language"),
    "wordgeneration":           ("word generation",                  "language"),
    "wordproduction":           ("word production",                  "language"),
    "verbal":                   ("verbal processing",                "language"),
    "naming":                   ("object naming",                    "language"),
    "picturenaming":            ("picture naming",                   "language"),
    "semantic":                 ("semantic processing",              "language"),
    "phonological":             ("phonological processing",          "language"),
    "sentenceprocessing":       ("sentence processing",              "language"),
    "verbfluency":              ("verbal fluency",                   "language"),
    "fluency":                  ("verbal fluency",                   "language"),

    # ── Emotion / social ──────────────────────────────────────────────────────
    "faces":                    ("face processing",                  "emotion"),
    "emotionalfaces":           ("emotional face processing",        "emotion"),
    "faceperception":           ("face perception",                  "emotion"),
    "facematching":             ("face matching",                    "emotion"),
    "emotion":                  ("emotion processing",               "emotion"),
    "emotionreg":               ("emotion regulation",               "emotion"),
    "emotionregulation":        ("emotion regulation",               "emotion"),
    "fearlearning":             ("fear learning",                    "emotion"),
    "fearextinction":           ("fear extinction",                  "emotion"),
    "fear":                     ("fear processing",                  "emotion"),
    "social":                   ("social cognition",                 "emotion"),
    "socialcognition":          ("social cognition",                 "emotion"),
    "theoryofmind":             ("theory of mind",                   "emotion"),
    "tom":                      ("theory of mind",                   "emotion"),
    "empathy":                  ("empathy",                          "emotion"),
    "trust":                    ("trust / economic decision-making", "emotion"),
    "anger":                    ("anger processing",                 "emotion"),
    "disgust":                  ("disgust processing",               "emotion"),
    "pain":                     ("pain processing",                  "emotion"),
    "thermalstim":              ("thermal pain stimulation",         "emotion"),

    # ── Reward / decision ─────────────────────────────────────────────────────
    "reward":                   ("reward processing",                "reward"),
    "gambling":                 ("gambling / reward",                "reward"),
    "mid":                      ("monetary incentive delay",         "reward"),
    "moneytask":                ("monetary reward",                  "reward"),
    "incentive":                ("incentive processing",             "reward"),
    "effort":                   ("effort-based decision making",     "reward"),
    "riskytask":                ("risky decision making",            "reward"),
    "decisionmaking":           ("decision making",                  "reward"),
    "decision":                 ("decision making",                  "reward"),
    "ultimatum":                ("ultimatum game",                   "reward"),
    "delay":                    ("delay discounting",                "reward"),
    "delaydiscounting":         ("delay discounting",                "reward"),
    "probabilisticlearning":    ("probabilistic reward learning",    "reward"),
    "reinforcementlearning":    ("reinforcement learning",           "reward"),
    "rl":                       ("reinforcement learning",           "reward"),
    "bandit":                   ("multi-armed bandit",               "reward"),

    # ── Motor ─────────────────────────────────────────────────────────────────
    "motor":                    ("motor task",                       "motor"),
    "fingertapping":            ("finger tapping",                   "motor"),
    "tapping":                  ("finger tapping",                   "motor"),
    "grip":                     ("grip force",                       "motor"),
    "handgrip":                 ("hand grip",                        "motor"),
    "movement":                 ("movement",                         "motor"),
    "sequence":                 ("motor sequence learning",          "motor"),
    "motorsequence":            ("motor sequence learning",          "motor"),
    "srt":                      ("serial reaction time",             "motor"),
    "serialreactiontime":       ("serial reaction time",             "motor"),
    "reaction":                 ("reaction time",                    "motor"),
    "reactiontime":             ("reaction time",                    "motor"),
    "rt":                       ("reaction time",                    "motor"),
    "pointing":                 ("pointing",                         "motor"),
    "reaching":                 ("reaching",                         "motor"),
    "gait":                     ("gait analysis",                    "motor"),
    "walking":                  ("walking",                          "motor"),

    # ── Visual / perception ───────────────────────────────────────────────────
    "visual":                   ("visual processing",                "perception"),
    "checkerboard":             ("checkerboard visual",              "perception"),
    "objects":                  ("object recognition",               "perception"),
    "objectrecognition":        ("object recognition",               "perception"),
    "visualsearch":             ("visual search",                    "perception"),
    "contrastdetection":        ("contrast detection",               "perception"),
    "orientationdetection":     ("orientation detection",            "perception"),
    "motionperception":         ("motion perception",                "perception"),
    "depth":                    ("depth perception",                 "perception"),
    "spatial_perception":       ("spatial perception",               "perception"),
    "spatialperception":        ("spatial perception",               "perception"),
    "sceneperception":          ("scene perception",                 "perception"),
    "multisensory":             ("multisensory integration",         "perception"),
    "crossmodal":               ("crossmodal processing",            "perception"),

    # ── Auditory ──────────────────────────────────────────────────────────────
    "auditory":                 ("auditory processing",              "auditory"),
    "audiospat":                ("auditory spatial processing",      "auditory"),
    "auditorylocal":            ("auditory localisation",            "auditory"),
    "sound":                    ("sound processing",                 "auditory"),
    "speech":                   ("speech perception",                "auditory"),
    "speechperception":         ("speech perception",                "auditory"),
    "music":                    ("music processing",                 "auditory"),
    "tone":                     ("tone discrimination",              "auditory"),
    "tonedetection":            ("tone detection",                   "auditory"),
    "mismatch":                 ("mismatch negativity",              "auditory"),
    "mmn":                      ("mismatch negativity",              "auditory"),
    "oddball":                  ("auditory oddball",                 "auditory"),
    "p300":                     ("P300 oddball",                     "auditory"),
    "erp":                      ("event-related potential",          "auditory"),

    # ── Naturalistic / movie ──────────────────────────────────────────────────
    "movie":                    ("movie viewing",                    "naturalistic"),
    "naturalistic":             ("naturalistic viewing",             "naturalistic"),
    "videoviewing":             ("video viewing",                    "naturalistic"),
    "video":                    ("video viewing",                    "naturalistic"),
    "narratives":               ("narrative listening",              "naturalistic"),
    "narrative":                ("narrative listening",              "naturalistic"),
    "sherlock":                 ("Sherlock movie viewing",           "naturalistic"),
    "budapest":                 ("Budapest movie viewing",           "naturalistic"),
    "forrest":                  ("Forrest Gump movie viewing",       "naturalistic"),

    # ── Learning / cognitive control ──────────────────────────────────────────
    "learning":                 ("learning",                         "learning"),
    "errormonitoring":          ("error monitoring",                 "learning"),
    "feedback":                 ("feedback learning",                "learning"),
    "reversal":                 ("reversal learning",                "learning"),
    "habituation":              ("habituation",                      "learning"),
    "adaptation":               ("adaptation",                       "learning"),
    "cognitive":                ("cognitive task",                   "learning"),
    "cognitivecontrol":         ("cognitive control",                "learning"),
    "inhibition":               ("response inhibition",              "learning"),

    # ── Clinical / EEG ────────────────────────────────────────────────────────
    "seizuremonitoring":        ("seizure monitoring",               "clinical"),
    "szmonitoring":             ("seizure monitoring",               "clinical"),
    "iceeg":                    ("intracranial EEG monitoring",      "clinical"),
    "sleep":                    ("sleep",                            "clinical"),
    "sleeprecording":           ("sleep recording",                  "clinical"),
    "anesthesia":               ("anesthesia monitoring",            "clinical"),

    # ── Miscellaneous ─────────────────────────────────────────────────────────
    "practice":                 ("practice run",                     "misc"),
    "training":                 ("training run",                     "misc"),
    "localizer":                ("functional localizer",             "misc"),
    "loc":                      ("functional localizer",             "misc"),
    "baseline":                 ("baseline",                         "misc"),
    "fixation":                 ("fixation baseline",                "misc"),
    "test":                     ("test run",                         "misc"),
    "pilot":                    ("pilot run",                        "misc"),
}

# Values to silently filter (not useful in prompts)
FILTER_SET = {
    "main", "task", "exp", "experiment", "run", "session",
    "scan", "block", "trial", "stim", "stimulus", "mri", "fmri",
    "bold", "func", "functional", "eeg", "meg",
}


def normalize(s: str) -> str:
    """Lowercase, strip, collapse separators."""
    return re.sub(r"[\s_\-]+", "", s.strip().lower())


def auto_map(raw: str) -> tuple[str, str] | None:
    """
    Try to map a raw task value. Returns (label, category) or None if unknown.
    None means filter out; a tuple means map to label.
    """
    key = normalize(raw)

    if key in FILTER_SET:
        return None

    # Direct lookup
    if key in MASTER:
        return MASTER[key]

    # Substring match — if the key contains a known task name
    for pattern, (label, cat) in MASTER.items():
        if len(pattern) >= 4 and pattern in key:
            return label, cat

    return None  # unknown


def append_to_mappings(new_entries: dict[str, str]) -> None:
    """Append new TASK_LABEL entries to value_mappings.py."""
    source = MAPPINGS_FILE.read_text(encoding="utf-8")

    lines = ["\n# Auto-mapped by auto_map_tasks.py\n"]
    for raw, label in sorted(new_entries.items()):
        lines.append(f"TASK_LABEL[{raw!r}] = {label!r}")

    patch = "\n".join(lines) + "\n"
    MAPPINGS_FILE.write_text(source.rstrip() + "\n" + patch, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print mappings without writing to value_mappings.py")
    args = parser.parse_args()

    engine = create_engine(args.db_url)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT o.task, COUNT(*) AS n,
                   ARRAY_AGG(DISTINCT d.accession_id ORDER BY d.accession_id)
                       FILTER (WHERE d.accession_id IS NOT NULL) AS datasets
            FROM bids_objects o
            JOIN bids_datasets d ON d.id = o.dataset_id
            WHERE o.task IS NOT NULL AND o.task != ''
            GROUP BY o.task
            ORDER BY n DESC
        """)).fetchall()

    already_mapped = 0
    auto_mapped: dict[str, str] = {}
    filtered: list[str] = []
    unknown: list[dict] = []

    # Pre-fetch dataset descriptions for lookup
    with engine.connect() as conn:
        desc_rows = conn.execute(text("""
            SELECT accession_id, name,
                   description->>'Name' AS desc_name,
                   description->>'BIDSVersion' AS bids_version
            FROM bids_datasets
            WHERE accession_id IS NOT NULL
        """)).fetchall()
    dataset_info = {r[0]: {"name": r[1], "desc_name": r[2]} for r in desc_rows}

    for raw, count, datasets in rows:
        key = raw.strip().lower()
        if key in _vm.TASK_LABEL:
            already_mapped += 1
            continue

        result = auto_map(raw)
        if result is None:
            filtered.append(raw)
        else:
            label, category = result
            auto_mapped[raw] = label
            continue

        # Unknown — collect with dataset context for display
        ds_names = [dataset_info.get(ds, {}).get("name", ds) for ds in (datasets or [])[:3]]
        unknown.append({"raw": raw, "count": count, "datasets": datasets or [], "ds_names": ds_names})

    # Group auto-mapped by category for display
    by_category: dict[str, list] = {}
    for raw, label in auto_mapped.items():
        _, cat = MASTER.get(normalize(raw), (label, "misc"))
        by_category.setdefault(cat, []).append((raw, label))

    print(f"\n{'='*60}")
    print(f"  TASK AUTO-MAPPING RESULTS")
    print(f"{'='*60}")
    print(f"  Already in TASK_LABEL:  {already_mapped}")
    print(f"  Auto-mapped:            {len(auto_mapped)}")
    print(f"  Filtered out:           {len(filtered)}")
    print(f"  Unknown (needs review): {len(unknown)}")
    print()

    for cat, entries in sorted(by_category.items()):
        print(f"\n  [{cat.upper()}]")
        for raw, label in sorted(entries):
            print(f"    {raw!r:40s} → {label!r}")

    if unknown:
        print(f"\n  [UNKNOWN — needs manual review via audit_db_values.py]")
        for u in unknown:
            ds_display = ", ".join(u["ds_names"][:3]) or ", ".join(u["datasets"][:3])
            print(f"    {u['raw']!r:40s}  ({u['count']}x)  from: {ds_display}")

    if filtered:
        print(f"\n  [FILTERED — not written to TASK_LABEL]")
        for raw in filtered:
            print(f"    {raw!r}")

    if not args.dry_run:
        append_to_mappings(auto_mapped)
        print(f"\n✓ Wrote {len(auto_mapped)} entries to TASK_LABEL in value_mappings.py")
        print(f"  Re-run with --dry-run to preview future changes.")
    else:
        print(f"\n  Dry run — nothing written.")


if __name__ == "__main__":
    main()
