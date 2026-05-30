"""
One-time script to remove colliding synonyms identified by audit_collisions.py.
Does line-level text surgery to preserve YAML formatting exactly.
"""
from __future__ import annotations
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Removal list: (concept_key, synonym_term_case_insensitive)
# Each entry removes that term from the concept's synonyms list in the
# task: section of value_mappings.yaml.
#
# Classification key used below:
#   [parent→child]  parent concept listing a child concept's label — remove from parent
#   [mutual]        two concepts each claiming the other's label
#   [wrong-owner]   specific concept claiming a canonical term from another concept
#   [near-dup]      near-duplicate dataset variants sharing terms — remove from less canonical
#   [rev]           exception: remove from TARGET not source (noted explicitly)
# ---------------------------------------------------------------------------

# fmt: off
REMOVALS: list[tuple[str, str]] = [
    # ── resting state / naturalistic ──────────────────────────────────────
    ("resting_state_technical",          "resting-state baseline"),          # [wrong-owner] owned by calibration_baseline
    ("video_game_and_video",             "naturalistic paradigm"),            # [wrong-owner] owned by naturalistic
    ("video_clips_viewing",              "naturalistic movie viewing"),       # [wrong-owner] owned by movie_viewing
    ("movie_memory",                     "film recall"),                      # [parent→child] film_recall.label
    ("naturalistic_passive_viewing",     "naturalistic movie viewing"),       # [wrong-owner]
    ("naturalistic_passive_viewing",     "movie-watching fMRI"),              # [wrong-owner]
    ("rest_film_viewing",                "video watching"),                   # [wrong-owner] owned by movie_viewing

    # ── working memory ────────────────────────────────────────────────────
    ("working_memory_general",           "sternberg task"),                   # [parent→child]
    ("working_memory_general",           "visuospatial working memory"),      # [parent→child]
    ("working_memory_general",           "n-back"),                           # [parent→child]
    ("working_memory_general",           "nback"),                            # [parent→child]
    ("working_memory_general",           "n back"),                           # [parent→child]
    ("memory_updating",                  "n-back"),                           # [parent→child]
    ("memory_updating",                  "nback"),                            # [parent→child]
    ("memory_updating",                  "n back"),                           # [parent→child]
    ("memory_updating",                  "n-back task"),                      # [parent→child]
    ("one_second_memory",                "VSTM"),                             # [wrong-owner] VSTM = visual_short_term_memory
    ("perception_and_working_memory",    "delayed match-to-sample task"),     # [parent→child]
    ("spatial_location_immediate_working_memory", "VSTM"),                   # [wrong-owner]
    ("visual_working_memory_short_immediate", "visual short-term memory"),   # [wrong-owner] STM ≠ WM
    ("visual_working_memory_short_immediate", "VSTM"),                       # [wrong-owner]
    ("visual_working_memory_short_immediate", "visuospatial working memory"),# [wrong-owner]
    ("visual_working_memory_general",    "VSTM"),                            # [wrong-owner] VSTM = STM not WM

    # ── implicit / statistical learning ──────────────────────────────────
    ("implicit_learning",                "probabilistic classification learning"),  # [parent→child]
    ("implicit_learning",                "weather prediction task"),          # [parent→child]
    ("implicit_learning",                "artificial grammar learning"),      # [parent→child]
    ("implicit_learning",                "AGL"),                              # [parent→child]
    ("statistical_learning_task",        "artificial grammar learning task"), # [wrong-owner] owned by artificial_grammar_nebula

    # ── memory encoding / retrieval ───────────────────────────────────────
    ("search_superiority_recollection_familiarity_encoding", "episodic memory encoding"),  # [wrong-owner]
    ("stimulus_exposure_memory",         "memory encoding"),                  # [wrong-owner] owned by memory_encoding
    ("study_test_memory_task",           "episodic memory task"),             # [wrong-owner] owned by episodic_memory_task
    ("memory_retrieval_general",         "cued recall"),                      # [parent→child]
    ("delayed_repeated_free_recall_read_only", "DFR"),                        # [near-dup] owned by pyfr_delayed_free_recall_word_lists
    ("delayed_repeated_free_recall_read_only", "DFR task"),                   # [near-dup]
    ("delayed_repeated_free_recall_read_only", "Delayed word list recall"),   # [near-dup]
    ("delayed_repeated_free_recall_read_only", "Delayed verbal free recall"), # [near-dup]
    ("face_memory_retrieval_test",       "face memory task"),                 # [wrong-owner] owned by face_memory_task
    ("cued_recall_memory",               "paired-associate learning"),        # [wrong-owner] PAL ≠ cued recall
    ("word_memory_retrieval",            "lexical retrieval"),                # [wrong-owner] lexical retrieval is language not memory
    ("free_recall_variant_1",            "word-list free recall"),            # [near-dup]
    ("free_recall_variant_1",            "delayed word list recall"),         # [near-dup]
    ("free_recall_variant_1",            "DFR task"),                         # [near-dup]
    ("free_recall_variant_1",            "DFR"),                              # [near-dup]
    ("memory_task_general",              "episodic memory task"),             # [parent→child]
    ("semantic_memory",                  "semantic categorization"),          # [parent→child]
    ("semantic_memory",                  "semantic fluency"),                 # [parent→child]
    ("semantic_memory",                  "semantic processing"),              # [wrong-owner] owned by language_speech
    ("familiarity_assessment",           "Old-new judgment"),                 # [wrong-owner] owned by recognition
    ("object_recognition_general",       "picture naming task"),              # [wrong-owner]
    ("visual_memory_general",            "visual short-term memory"),         # [wrong-owner]
    ("visual_memory_general",            "VSTM"),                             # [wrong-owner]
    ("offline_processing_associative_learning", "memory consolidation"),      # [wrong-owner]
    ("paired_associates_learning_general", "paired-associate learning"),      # [near-dup] canonical is associative
    ("associative_inference_memory",     "Transitive Inference Task"),        # [wrong-owner]
    ("associative_learning_what_on",     "Paired-associate learning"),        # [near-dup]
    ("learning_task_general",            "statistical learning task"),        # [parent→child]
    ("memory_task_phase_1_ds001232_ds001430", "memory encoding"),             # [wrong-owner]

    # ── learning_and_memory parent ────────────────────────────────────────
    ("learning_and_memory",              "memory encoding"),                  # [parent→child]
    ("learning_and_memory",              "memory retrieval"),                 # [parent→child]
    ("learning_and_memory",              "memory consolidation"),             # [parent→child]
    ("learning_and_memory",              "episodic memory task"),             # [parent→child]
    ("learning_and_memory",              "implicit learning"),                # [parent→child]
    ("learning_and_memory",              "paired-associate learning"),        # [parent→child]

    # ── spatial navigation ────────────────────────────────────────────────
    ("spatial_memory_group",             "route learning"),                   # [parent→child]
    ("navigation_learning",              "spatial navigation task"),          # [wrong-owner]
    ("route_learning",                   "navigation learning"),              # [mutual] route_learning ≠ navigation_learning
    ("maze_navigation_task",             "spatial navigation task"),          # [wrong-owner]
    ("spatial_navigation_localizer",     "maze navigation task"),             # [wrong-owner]
    ("spatial_memory",                   "topographical memory"),             # [wrong-owner] owned by spatial_memory_group
    ("spatial_memory",                   "wayfinding"),                       # [wrong-owner] owned by geospatial_navigation

    # ── attention / eye movements ─────────────────────────────────────────
    ("eye_movement_magnification_attention", "anti-saccade task"),            # [wrong-owner]
    ("eye_movement_task",                "anti-saccade task"),                # [wrong-owner]
    ("numerical_processing",             "number comparison task"),           # [parent→child]

    # ── oddball / ERP ─────────────────────────────────────────────────────
    ("oddball_task",                     "visual oddball"),                   # [parent→child]
    ("oddball_task",                     "auditory oddball"),                 # [parent→child]
    ("probe_task",                       "Sternberg task"),                   # [wrong-owner]
    ("single_task_weather_prediction",   "Weather Prediction Task"),          # [wrong-owner]
    ("syllogistic_reasoning_task",       "transitive inference task"),        # [wrong-owner]
    # [rev] p300_visual_stimulation_ctos steals from visual_oddball → remove from p300:
    ("p300_visual_stimulation_ctos",     "visual oddball task"),              # [rev: wrong-owner]
    ("p300_visual_stimulation_ctos",     "visual oddball experiment"),        # [rev: wrong-owner]
    ("alertness_attention_litebook",     "Simple Reaction Time"),             # [wrong-owner]
    ("cnos_visual_stimulation",          "visual oddball task"),              # [near-dup] owned by p300_visual_stimulation_ctos
    ("cnos_visual_stimulation",          "visual P300 paradigm"),             # [near-dup]
    ("cnos_visual_stimulation",          "visual P300 task"),                 # [near-dup]
    ("cnos_visual_stimulation",          "vP300"),                            # [near-dup]
    ("cnos_visual_stimulation",          "vP3"),                              # [near-dup]
    ("deduction_task",                   "transitive inference task"),        # [wrong-owner]
    ("deviant_detection_task",           "oddball task"),                     # [wrong-owner]
    ("expectation_task",                 "violation of expectation"),         # [wrong-owner]
    ("expectation_task",                 "prediction error task"),            # [wrong-owner]
    ("repetition_suppression_expectation", "oddball task"),                   # [wrong-owner]

    # ── neurofeedback / meditation ────────────────────────────────────────
    ("neurofeedback_general",            "fMRI neurofeedback"),               # [parent→child]
    ("meditation_variant2",              "breath awareness task"),            # [near-dup] owned by meditation_breathing_variant1
    ("meditation_variant2",              "focused attention meditation"),     # [near-dup]
    ("meditation_variant2",              "breath focus task"),                # [near-dup]
    ("meditation_general",               "focused attention meditation"),     # [parent→child]
    ("meditation_general",               "FAM"),                              # [parent→child]
    ("meditation_general",               "breath awareness task"),            # [parent→child]

    # ── learning tasks ────────────────────────────────────────────────────
    ("picture_matching_task",            "delayed match-to-sample task"),     # [wrong-owner]
    ("probabilistic_classification_learning", "weather prediction task"),     # [wrong-owner]
    ("quick_reaction_task",              "Lexical Decision Task"),            # [wrong-owner]
    ("quick_reaction_task",              "Go/No-Go Task"),                    # [wrong-owner]
    ("reactive_control_task",            "Go/No-Go Task"),                    # [wrong-owner]
    ("rule_learning",                    "implicit rule learning"),           # [wrong-owner] owned by artificial_grammar_nebula

    # ── motor imagery ─────────────────────────────────────────────────────
    # [rev] motor_imagery_vs_rest steals generic motor imagery terms from motor_imagery_general
    ("motor_imagery_vs_rest",            "imagined movement"),                # [rev: wrong-owner]
    ("motor_imagery_vs_rest",            "kinesthetic imagery"),              # [rev: wrong-owner]
    ("mental_imagery_general",           "kinesthetic imagery"),              # [wrong-owner] — leave in motor_imagery_general

    # ── sensory / motor tasks ─────────────────────────────────────────────
    ("tibial_mixed_nerve_stimulation",   "tibial nerve stimulation"),         # [wrong-owner]
    ("tibial_sensory_nerve_stimulation", "tibial nerve stimulation"),         # [wrong-owner]
    ("tactile_stimulation",              "somatosensory task"),               # [wrong-owner]
    ("tactile_perception_task",          "somatosensory task"),               # [wrong-owner]
    ("hand_action_task_variant_3",       "hand motor task"),                  # [wrong-owner]
    ("motor_task_variant_1",             "finger tapping task"),              # [wrong-owner]
    ("motor_mapping_five_fingers",       "finger tapping task"),              # [wrong-owner]
    ("motor_task_general",               "motor localizer"),                  # [parent→child]
    ("sensorimotor_task_general",        "finger tapping task"),              # [wrong-owner]
    ("body_motor_task",                  "HCP motor task"),                   # [wrong-owner]
    ("basketball_dribble_motor_imagery", "basketball motor imagery"),         # [wrong-owner]
    ("finger_motor_task",                "finger tapping task"),              # [wrong-owner] already partly removed
    ("hand_grasp_motor_task",            "ball squeeze task"),                # [wrong-owner]
    ("hand_motor_task",                  "finger tapping task"),              # [wrong-owner]
    ("somatomotor_task",                 "HCP motor task"),                   # [wrong-owner]
    ("somatomotor_task",                 "finger tapping task"),              # [wrong-owner]

    # ── language / reading ────────────────────────────────────────────────
    ("semantic_processing_general",      "semantic categorization"),          # [parent→child]
    ("visual_word_form_area_task",       "lexical decision task"),            # [wrong-owner]
    ("brocanto_language_task",           "AGL"),                              # [wrong-owner] AGL=artificial_grammar_nebula
    ("brocanto_language_task",           "Artificial Grammar Learning"),      # [wrong-owner]
    ("brocanto_language_task",           "Artificial Grammar Learning task"), # [wrong-owner]
    ("language_production",              "verb generation"),                  # [parent→child]
    ("language_production",              "verbal fluency task"),              # [parent→child]
    ("language_production",              "picture naming task"),              # [parent→child]
    ("grammatical_processing",           "grammaticality judgment task"),     # [wrong-owner]
    ("sentence_processing",              "sentence reading"),                 # [parent→child]
    ("sentence_processing",              "grammaticality judgment task"),     # [wrong-owner]
    ("verb_generation_general",          "covert verb generation"),           # [parent→child]
    ("verb_generation_general",          "overt verb generation"),            # [parent→child]
    ("covert_speech_decoding",           "inner speech task"),                # [wrong-owner]
    ("word_processing_general",          "lexical decision task"),            # [parent→child]
    ("word_processing_general",          "phonological processing"),          # [parent→child]
    ("word_processing_general",          "semantic processing"),              # [wrong-owner]
    ("word_reading_general",             "word naming task"),                 # [parent→child]
    ("auditory_spelling_processing",     "auditory orthographic processing"), # [wrong-owner]
    ("semantic_localizer",               "language localizer"),               # [wrong-owner]
    ("rsvp_language_processing",         "RSVP"),                             # [wrong-owner] RSVP owned by rapid_serial_visual_presentation
    ("semantic_fluency",                 "animal fluency"),                   # [wrong-owner] owned by animal_stimuli_task
    ("semantic_fluency",                 "animal naming task"),               # [wrong-owner]
    ("semantic_listening",               "auditory semantic processing"),     # [wrong-owner]
    ("semantic_listening",               "narrative listening"),              # [wrong-owner]
    ("verbal_fluency_task",              "animal naming task"),               # [wrong-owner]
    ("glass_lexical_task",               "Lexical Decision Task"),            # [wrong-owner]
    ("glass_lexical_task",               "Word Naming Task"),                 # [wrong-owner]
    ("visual_orthographic_processing",   "lexical decision task"),            # [wrong-owner]
    ("word_naming_task",                 "picture naming task"),              # [wrong-owner]
    ("general_reading_task",             "sentence processing"),              # [wrong-owner]
    ("general_reading_task",             "silent reading"),                   # [parent→child]

    # ── perception / localizers ───────────────────────────────────────────
    ("animate_stimuli_perception",       "biological motion task"),           # [wrong-owner]
    ("pheromone_olfaction",              "body odor perception"),             # [wrong-owner]
    ("random_dot_motion_perception",     "random dot kinematogram task"),     # [wrong-owner]
    ("retinotopy_expanding",             "population receptive field mapping"),# [wrong-owner]
    ("auditory_perception_general",      "auditory localizer"),               # [parent→child]
    ("auditory_general",                 "auditory localizer"),               # [parent→child]
    ("auditory_perception_mapping",      "auditory localizer"),               # [wrong-owner]
    ("flavour_pleasantness_task",        "taste rating task"),                # [wrong-owner]
    ("flavor_perception_rating",         "taste rating task"),                # [wrong-owner]
    ("face_perception",                  "face localizer"),                   # [parent→child]
    ("retinotopy",                       "population receptive field mapping"),# [parent→child]
    ("retinotopy_fixed_bar",             "meridian mapping"),                 # [wrong-owner]
    ("auditory_entrainment_40hz",        "auditory gamma entrainment"),       # [wrong-owner]
    ("visual_dots_task",                 "dot-probe task"),                   # [wrong-owner]
    ("dichoptic_stimulation",            "binocular rivalry"),                # [wrong-owner] owned by perceptual_rivalry_task
    ("navigational_affordances_ego_motion", "wayfinding"),                   # [wrong-owner]
    ("visual_motion_perception_general", "random dot kinematogram task"),     # [parent→child]
    ("visual_motion_perception_general", "biological motion task"),           # [parent→child]
    ("ventral_visual_localizer",         "visual category localizer"),        # [wrong-owner]
    ("adaptation_general",               "motor adaptation"),                 # [parent→child]
    ("category_perception",              "semantic categorization"),          # [wrong-owner]
    ("high_level_visual_auditory_cognitive_localizer", "language localizer"), # [wrong-owner]
    ("effloc_visual_conditions",         "visual category localizer"),        # [wrong-owner]
    ("visual_functional_localizer",      "retinotopy"),                       # [wrong-owner]
    ("half_field_checkerboard_stimulation", "visual hemifield stimulation"),  # [wrong-owner]
    ("object_task_general",              "object memory task"),               # [parent→child]
    ("scene_perception_manual_response", "parahippocampal place area localizer"), # [wrong-owner]
    ("parahippocampal_place_area_mapper","parahippocampal place area localizer"), # [wrong-owner]
    ("feature_discrimination_serial_presentation", "delayed match-to-sample task"), # [wrong-owner]

    # ── social / emotion / reward ─────────────────────────────────────────
    ("social_emotional_reward",          "reinforcement learning"),           # [wrong-owner] owned by reinforcement_learning
    ("weather_prediction_reversal_learning", "Probabilistic Reversal Learning Task"), # [wrong-owner]
    ("weather_prediction_reversal_learning", "Probabilistic reversal learning"),      # [wrong-owner]
    ("social_task_general",              "social judgment task"),             # [parent→child]
    ("shared_reward_task",               "Dictator Game"),                    # [parent→child]
    ("shared_reward_task",               "Ultimatum Game"),                   # [parent→child]
    ("shared_reward_task",               "Trust Game"),                       # [parent→child]
    ("emotion_evaluation_task",          "emotion matching task"),            # [wrong-owner]
    ("risky_decision_making",            "Columbia Card Task"),               # [wrong-owner]
    ("risky_decision_making",            "gambling task"),                    # [wrong-owner] owned by hcp_gambling_task
    ("monetary_incentive_delay_localizer","Monetary Incentive Delay Task"),   # [wrong-owner] owned by monetary_incentive_delay_task
    ("reward_processing",                "monetary incentive delay"),         # [wrong-owner]
    ("hariri_emotional_face_task",       "HCP emotion task"),                 # [wrong-owner]
    ("emotion_general",                  "HCP emotion task"),                 # [wrong-owner]
    ("emotion_general",                  "emotion processing"),               # [wrong-owner] owned by emotion_processing_regulation
    ("social_interaction_psts",          "theory of mind localizer"),         # [wrong-owner]
    ("theory_of_mind",                   "social cognition"),                 # [wrong-owner] owned by social_emotional_reward
    ("strategic_interaction_game",       "Ultimatum Game"),                   # [parent→child]
    ("strategic_interaction_game",       "Trust Game"),                       # [parent→child]
    ("strategic_interaction_game",       "Dictator Game"),                    # [parent→child]
    ("prediction_error_task",            "Reversal Learning Task"),           # [wrong-owner]
    ("social_risky_choice_task",         "Trust Game"),                       # [wrong-owner]
    ("social_risky_choice_task",         "Ultimatum Game"),                   # [wrong-owner]
    ("communicative_gaze_co_condition",  "Communicative gaze"),               # [wrong-owner]
    ("debate_task",                      "persuasion task"),                  # [wrong-owner]
    ("food_processing_general",          "food choice task"),                 # [parent→child]
    ("probability_decision_making",      "probabilistic selection task"),     # [wrong-owner]
    ("polex_personal_rating",            "POLEX-PR"),                         # [near-dup]
    ("response_to_snack_stimuli",        "food choice task"),                 # [wrong-owner]
    ("trait_role_perception_task",       "social judgment task"),             # [wrong-owner]
    ("social_judgment_task",             "moral dilemma task"),               # [wrong-owner] owned by moral_dilemma

    # ── breathing / physiology ────────────────────────────────────────────
    ("deep_breathing",                   "paced breathing"),                  # [wrong-owner] owned by paced_breathing
    ("deep_breathing",                   "slow-paced breathing"),             # [wrong-owner]
    ("hypercapnia_challenge",            "breath hold"),                      # [mutual] different techniques
    ("hypercapnia_challenge",            "breath-hold"),                      # [mutual]
    ("breath_hold",                      "hypercapnia challenge"),            # [mutual]

    # ── stimulation / BCI ─────────────────────────────────────────────────
    ("eeg_motor_imagery_neurofeedback",  "sensorimotor rhythm BCI"),          # [wrong-owner]
    ("electrical_stimulation",           "Direct cortical stimulation"),      # [mutual] DCS is a subtype, not synonymous
    ("electrical_stimulation",           "tDCS"),                             # [wrong-owner] tDCS is in physiological_neurostimulation
    # Note: "DES" stays in direct_cortical_stimulation (it IS DCS); electrical_stimulation should not claim it

    # ── sleep ─────────────────────────────────────────────────────────────
    ("sleep_state",                      "SWS"),                              # [wrong-owner] SWS = slow_wave_sleep

    # ── deepmreye near-duplicate ──────────────────────────────────────────
    ("deepmreye_closed_training",        "deepmreye eyes-closed run"),        # [near-dup] keep in resting_eyes_closed

    # ── baseline / control ────────────────────────────────────────────────
    ("functional_localizer_general",     "scout scan"),                       # [wrong-owner] owned by calibration_baseline
    ("blank_control_condition",          "null condition"),                   # [wrong-owner]
    ("baseline_general_task",            "null condition"),                   # [wrong-owner]
    ("control_condition_general",        "null condition"),                   # [wrong-owner]

    # ── misc ──────────────────────────────────────────────────────────────
    ("thinking_task_general",            "self-referential processing"),      # [wrong-owner]
]
# fmt: on


# ---------------------------------------------------------------------------
# Text-level surgery
# ---------------------------------------------------------------------------

def _find_concept_block(lines: list[str], concept_key: str) -> tuple[int, int] | None:
    """Return (start_line, end_line) of the concept block (exclusive end).
    The block starts at the line '  concept_key:' and ends before the next
    sibling key (same or lesser indentation) or EOF.
    """
    # Match "  key:" at exactly 2-space indent (task concepts are indented 2)
    start_pattern = re.compile(r"^  " + re.escape(concept_key) + r"\s*:")
    for i, line in enumerate(lines):
        if start_pattern.match(line):
            # Block ends at next line with indent <= 2 that is a YAML key
            for j in range(i + 1, len(lines)):
                if re.match(r"^  \S", lines[j]) or re.match(r"^\S", lines[j]):
                    return (i, j)
            return (i, len(lines))
    return None


def _remove_synonym(lines: list[str], start: int, end: int, term: str) -> int:
    """Remove the synonym entry for *term* in the slice lines[start:end].
    Returns number of lines removed.
    """
    term_lower = term.lower().strip()
    i = start
    removed = 0
    while i < end - removed:
        line = lines[i]
        # Match "    - term: <value>" (4-space indent, weighted format)
        m = re.match(r'^(\s+)-\s+term:\s+(.+)$', line)
        if m:
            found_term = m.group(2).strip().strip('"').strip("'")
            if found_term.lower() == term_lower:
                # Remove this line and the following weight line (if present)
                del lines[i]
                end -= 1
                removed += 1
                if i < end and re.match(r'^\s+weight:', lines[i]):
                    del lines[i]
                    end -= 1
                    removed += 1
                continue
        # Also match plain-string format "    - term string" (no weight)
        m2 = re.match(r'^(\s+)-\s+(.+)$', line)
        if m2 and not m2.group(2).startswith(('term:', 'weight:', '{')):
            found_term = m2.group(2).strip().strip('"').strip("'")
            if found_term.lower() == term_lower:
                del lines[i]
                end -= 1
                removed += 1
                continue
        i += 1
    return removed


def main() -> None:
    path = Path("RAG/value_mappings.yaml")
    if not path.exists():
        print(f"ERROR: {path} not found. Run from the BIDS-Eye root directory.")
        return

    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()

    total_removed = 0
    not_found_key: list[str] = []
    not_found_syn: list[str] = []

    for concept_key, term in REMOVALS:
        block = _find_concept_block(lines, concept_key)
        if block is None:
            not_found_key.append(concept_key)
            continue
        start, end = block
        n = _remove_synonym(lines, start, end, term)
        if n == 0:
            not_found_syn.append(f"{concept_key} / '{term}'")
        else:
            total_removed += n

    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)

    print(f"Done. Removed {total_removed} synonym entries.")
    if not_found_key:
        print(f"\n{len(not_found_key)} concept keys not found:")
        for k in not_found_key:
            print(f"  KEY MISSING: {k}")
    if not_found_syn:
        print(f"\n{len(not_found_syn)} synonyms not found in their concept:")
        for s in not_found_syn:
            print(f"  SYN MISSING: {s}")


if __name__ == "__main__":
    main()
