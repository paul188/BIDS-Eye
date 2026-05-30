#!/usr/bin/env python3
"""
run_query_eval.py — Batch query evaluator for BIDS-Eye.
Runs a curated set of queries against the local API and reports:
  - PASS: results match intent
  - ZERO: query returned 0 results (should have had results)
  - ILIKE: SQL used ILIKE on a canonical-code column
  - WRONG_SQL: SQL missing expected filter
  - PARTIAL: results exist but first-result spot check suggests wrong mapping
"""

import json, time, sys, re, urllib.request, urllib.error
from datetime import datetime

API = "http://localhost:8000/api/query"

# ── Expected mappings for automatic SQL verification ─────────────────────────
# If a query MUST produce SQL containing a specific fragment, list it here.
# If a query MUST NOT return 0 results given the DB content, mark expected_min.
QUERIES = [
    # ── Participant count fix ─────────────────────────────────────────────────
    {"q": "datasets with at least 50 participants with major depressive disorder",
     "expected_min": 1,
     "must_contain": ["HAVING", "50"],
     "must_not": ["p2.diagnosis.*ILIKE", "ILIKE.*major"]},

    {"q": "give me fMRI studies with more than 100 subjects",
     "expected_min": 1,
     "must_contain": ["HAVING", "100"],
     "must_not": []},

    {"q": "autism datasets with more than 30 participants",
     "expected_min": 1,
     "must_contain": ["HAVING", "30"],
     "must_not": []},

    {"q": "large resting state datasets with over 200 subjects",
     "expected_min": 1,
     "must_contain": ["HAVING", "200"],
     "must_not": []},

    {"q": "schizophrenia studies with at least 30 patients",
     "expected_min": 1,
     "must_contain": ["HAVING", "30"],
     "must_not": []},

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    {"q": "datasets with autism spectrum disorder",
     "expected_min": 1,
     "must_contain": ["autism_spectrum_disorder"],
     "must_not": ["ILIKE"]},

    {"q": "autistic children brain imaging",
     "expected_min": 1,
     "must_contain": ["autism_spectrum_disorder"],
     "must_not": ["ILIKE"]},

    {"q": "ASD fMRI studies",
     "expected_min": 1,
     "must_contain": ["autism_spectrum_disorder"],
     "must_not": ["ILIKE"]},

    {"q": "ADHD brain imaging",
     "expected_min": 1,
     "must_contain": ["adhd"],
     "must_not": ["ILIKE"]},

    {"q": "attention deficit hyperactivity disorder studies",
     "expected_min": 1,
     "must_contain": ["adhd"],
     "must_not": ["ILIKE"]},

    {"q": "schizophrenia patients brain scans",
     "expected_min": 1,
     "must_contain": ["schizophrenia"],
     "must_not": ["ILIKE"]},

    {"q": "schizophrenic patients neuroimaging",
     "expected_min": 1,
     "must_contain": ["schizophrenia"],
     "must_not": ["ILIKE"]},

    {"q": "psychosis brain imaging",
     "expected_min": 1,
     "must_contain": ["schizophrenia"],
     "must_not": ["ILIKE"]},

    {"q": "major depressive disorder fMRI",
     "expected_min": 1,
     "must_contain": ["major_depressive_disorder"],
     "must_not": ["ILIKE"]},

    {"q": "depressed patients resting state",
     "expected_min": 1,
     "must_contain": ["major_depressive_disorder"],
     "must_not": ["ILIKE"]},

    {"q": "MDD and healthy volunteers",
     "expected_min": 1,
     "must_contain": ["major_depressive_disorder"],
     "must_not": ["ILIKE"]},

    {"q": "bipolar disorder brain imaging",
     "expected_min": 1,
     "must_contain": ["bipolar_disorder"],
     "must_not": ["ILIKE"]},

    {"q": "Parkinson's disease brain imaging",
     "expected_min": 1,
     "must_contain": ["parkinsons_disease"],
     "must_not": ["ILIKE"]},

    {"q": "PD patients EEG studies",
     "expected_min": 1,
     "must_contain": ["parkinsons_disease"],
     "must_not": ["ILIKE"]},

    {"q": "epilepsy datasets",
     "expected_min": 1,
     "must_contain": ["epilepsy"],
     "must_not": ["ILIKE"]},

    {"q": "seizure monitoring recordings",
     "expected_min": 1,
     "must_contain": ["epilepsy"],
     "must_not": ["ILIKE"]},

    {"q": "focal cortical dysplasia recordings",
     "expected_min": 1,
     "must_contain": ["focal_cortical_dysplasia"],
     "must_not": ["ILIKE"]},

    {"q": "Alzheimer's disease fMRI",
     "expected_min": 0,   # may not be in DB — just check no ILIKE
     "must_contain": [],
     "must_not": ["ILIKE.*alzheimer", "alzheimer.*ILIKE"]},

    {"q": "mild cognitive impairment datasets",
     "expected_min": 0,
     "must_contain": [],
     "must_not": ["ILIKE.*cogni", "ILIKE.*impair"]},

    {"q": "dyslexia brain studies",
     "expected_min": 1,
     "must_contain": ["dyslexia"],
     "must_not": ["ILIKE"]},

    {"q": "PTSD datasets",
     "expected_min": 0,
     "must_contain": [],
     "must_not": ["ILIKE.*ptsd", "ptsd.*ILIKE"]},

    {"q": "post-traumatic stress disorder brain imaging",
     "expected_min": 0,
     "must_contain": [],
     "must_not": ["ILIKE.*stress", "ILIKE.*trauma"]},

    {"q": "OCD obsessive compulsive disorder fMRI",
     "expected_min": 0,
     "must_contain": [],
     "must_not": ["ILIKE.*obsess", "ILIKE.*compuls"]},

    {"q": "anxiety disorder datasets",
     "expected_min": 1,
     "must_contain": ["anxiety_disorder"],
     "must_not": ["ILIKE"]},

    {"q": "anxious patients brain scans",
     "expected_min": 1,
     "must_contain": ["anxiety_disorder"],
     "must_not": ["ILIKE"]},

    {"q": "fibromyalgia neuroimaging",
     "expected_min": 1,
     "must_contain": ["fibromyalgia"],
     "must_not": ["ILIKE"]},

    {"q": "chronic pain fMRI",
     "expected_min": 1,
     "must_contain": ["fibromyalgia"],
     "must_not": ["ILIKE"]},

    {"q": "traumatic brain injury datasets",
     "expected_min": 1,
     "must_contain": ["traumatic_brain_injury"],
     "must_not": ["ILIKE"]},

    {"q": "healthy controls resting state fMRI",
     "expected_min": 1,
     "must_contain": ["healthy_control"],
     "must_not": ["neurotypical", "ILIKE"]},

    {"q": "typically developing children EEG",
     "expected_min": 1,
     "must_contain": ["typically_developing"],
     "must_not": ["ILIKE"]},

    # ── Tasks ─────────────────────────────────────────────────────────────────
    {"q": "resting state fMRI datasets",
     "expected_min": 10,
     "must_contain": ["resting_state"],
     "must_not": ["ILIKE"]},

    {"q": "eyes closed resting state EEG",
     "expected_min": 1,
     "must_contain": ["resting_state"],
     "must_not": ["ILIKE"]},

    {"q": "n-back working memory fMRI",
     "expected_min": 1,
     "must_contain": ["n_back_working_memory"],
     "must_not": ["ILIKE"]},

    {"q": "nback cognitive control EEG",
     "expected_min": 1,
     "must_contain": ["n_back_working_memory"],
     "must_not": ["ILIKE"]},

    {"q": "sustained attention task neuroimaging",
     "expected_min": 1,
     "must_contain": ["attention_and_executive_control"],
     "must_not": ["ILIKE"]},

    {"q": "face perception fMRI",
     "expected_min": 1,
     "must_contain": ["face_perception"],
     "must_not": ["ILIKE"]},

    {"q": "watching faces EEG",
     "expected_min": 1,
     "must_contain": ["face_perception"],
     "must_not": ["ILIKE"]},

    {"q": "emotion regulation datasets",
     "expected_min": 1,
     "must_contain": ["emotion_processing_regulation"],
     "must_not": ["ILIKE"]},

    {"q": "sad faces emotion processing",
     "expected_min": 1,
     "must_contain": ["emotion_processing_regulation"],
     "must_not": ["ILIKE"]},

    {"q": "movie watching fMRI",
     "expected_min": 1,
     "must_contain": ["movie_viewing"],
     "must_not": ["ILIKE"]},

    {"q": "naturalistic viewing brain imaging",
     "expected_min": 1,
     "must_contain": ["movie_viewing"],
     "must_not": ["ILIKE"]},

    {"q": "motor task hand movement fMRI",
     "expected_min": 1,
     "must_contain": ["motor_task_general"],
     "must_not": ["ILIKE"]},

    {"q": "motor imagery paradigms",
     "expected_min": 1,
     "must_contain": ["motor_imagery"],
     "must_not": ["ILIKE"]},

    {"q": "imagined movement brain imaging",
     "expected_min": 1,
     "must_contain": ["motor_imagery"],
     "must_not": ["ILIKE"]},

    {"q": "sleep EEG recordings",
     "expected_min": 1,
     "must_contain": ["sleep_state"],
     "must_not": ["ILIKE"]},

    {"q": "spatial navigation fMRI",
     "expected_min": 1,
     "must_contain": ["geospatial_navigation"],
     "must_not": ["ILIKE"]},

    {"q": "virtual navigation task",
     "expected_min": 1,
     "must_contain": ["geospatial_navigation"],
     "must_not": ["ILIKE"]},

    {"q": "theory of mind fMRI",
     "expected_min": 1,
     "must_contain": ["theory_of_mind"],
     "must_not": ["ILIKE"]},

    {"q": "social cognition brain imaging",
     "expected_min": 1,
     "must_contain": ["social_task_general"],
     "must_not": ["ILIKE"]},

    {"q": "reward learning monetary incentive task",
     "expected_min": 1,
     "must_contain": ["monetary_incentive_delay_task"],
     "must_not": ["ILIKE"]},

    {"q": "language processing reading fMRI",
     "expected_min": 1,
     "must_contain": ["general_reading_task"],
     "must_not": ["ILIKE"]},

    {"q": "auditory oddball EEG",
     "expected_min": 1,
     "must_contain": ["auditory_oddball"],
     "must_not": ["ILIKE"]},

    {"q": "pain perception fMRI",
     "expected_min": 1,
     "must_contain": ["pain_perception_general"],
     "must_not": ["ILIKE"]},

    # ── Modalities ────────────────────────────────────────────────────────────
    {"q": "functional MRI datasets",
     "expected_min": 10,
     "must_contain": ["functional_mri"],
     "must_not": ["ILIKE"]},

    {"q": "EEG brain recordings",
     "expected_min": 10,
     "must_contain": ["electroencephalography"],
     "must_not": ["ILIKE"]},

    {"q": "MEG magnetoencephalography datasets",
     "expected_min": 1,
     "must_contain": ["magnetoencephalography"],
     "must_not": ["ILIKE"]},

    {"q": "structural brain MRI T1 weighted",
     "expected_min": 10,
     "must_contain": ["anatomical_mri"],
     "must_not": ["ILIKE"]},

    {"q": "diffusion tensor imaging DTI",
     "expected_min": 1,
     "must_contain": ["diffusion_mri"],
     "must_not": ["ILIKE"]},

    {"q": "white matter tractography DWI",
     "expected_min": 1,
     "must_contain": ["diffusion_mri"],
     "must_not": ["ILIKE"]},

    {"q": "intracranial EEG iEEG recordings",
     "expected_min": 1,
     "must_contain": ["intracranial_eeg"],
     "must_not": ["ILIKE"]},

    {"q": "fNIRS near-infrared spectroscopy",
     "expected_min": 1,
     "must_contain": ["fnirs"],
     "must_not": ["ILIKE"]},

    {"q": "PET positron emission tomography brain",
     "expected_min": 1,
     "must_contain": ["positron_emission_tomography"],
     "must_not": ["ILIKE"]},

    {"q": "ASL perfusion MRI cerebral blood flow",
     "expected_min": 1,
     "must_contain": ["perfusion_asl"],
     "must_not": ["ILIKE"]},

    {"q": "simultaneous EEG and fMRI",
     "expected_min": 1,
     "must_contain": ["functional_mri", "electroencephalography"],
     "must_not": []},

    # ── Combined diagnosis + modality ─────────────────────────────────────────
    {"q": "autism EEG datasets",
     "expected_min": 1,
     "must_contain": ["autism_spectrum_disorder", "electroencephalography"],
     "must_not": ["ILIKE"]},

    {"q": "ADHD resting state fMRI",
     "expected_min": 1,
     "must_contain": ["adhd", "functional_mri", "resting_state"],
     "must_not": ["ILIKE"]},

    {"q": "schizophrenia MEG recordings",
     "expected_min": 0,
     "must_contain": ["schizophrenia"],
     "must_not": ["ILIKE"]},

    {"q": "Parkinson disease MEG",
     "expected_min": 1,
     "must_contain": ["parkinsons_disease", "magnetoencephalography"],
     "must_not": ["ILIKE"]},

    {"q": "depression resting state fMRI",
     "expected_min": 1,
     "must_contain": ["major_depressive_disorder", "resting_state"],
     "must_not": ["ILIKE"]},

    {"q": "epilepsy iEEG intracranial recordings",
     "expected_min": 1,
     "must_contain": ["intracranial_eeg"],
     "must_not": ["ILIKE"]},

    {"q": "autism structural MRI T1",
     "expected_min": 1,
     "must_contain": ["autism_spectrum_disorder", "anatomical_mri"],
     "must_not": ["ILIKE"]},

    {"q": "ADHD diffusion MRI white matter",
     "expected_min": 0,
     "must_contain": ["adhd"],
     "must_not": ["ILIKE"]},

    {"q": "TBI traumatic brain injury DTI fMRI",
     "expected_min": 1,
     "must_contain": ["traumatic_brain_injury"],
     "must_not": ["ILIKE"]},

    # ── Developmental/population ──────────────────────────────────────────────
    {"q": "pediatric neuroimaging datasets",
     "expected_min": 1,
     "must_contain": [],
     "must_not": []},

    {"q": "adolescent brain development fMRI",
     "expected_min": 1,
     "must_contain": [],
     "must_not": []},

    {"q": "elderly participants resting state",
     "expected_min": 1,
     "must_contain": ["older_adult"],
     "must_not": ["ILIKE"]},

    {"q": "aging brain structural MRI",
     "expected_min": 1,
     "must_contain": [],
     "must_not": []},

    {"q": "only female participants fMRI",
     "expected_min": 1,
     "must_contain": [],
     "must_not": []},

    # ── Unusual / edge cases ──────────────────────────────────────────────────
    {"q": "pharmacological fMRI drug effects",
     "expected_min": 1,
     "must_contain": [],
     "must_not": []},

    {"q": "ketamine brain imaging",
     "expected_min": 0,
     "must_contain": [],
     "must_not": ["ILIKE.*ketamine", "ketamine.*ILIKE"]},

    {"q": "deep brain stimulation local field potentials",
     "expected_min": 1,
     "must_contain": [],
     "must_not": []},

    {"q": "TMS EEG combined brain stimulation",
     "expected_min": 1,
     "must_contain": [],
     "must_not": []},

    {"q": "brain computer interface BCI EEG",
     "expected_min": 1,
     "must_contain": ["sensorimotor_rhythm_bci"],
     "must_not": ["ILIKE"]},

    {"q": "music listening EEG fMRI",
     "expected_min": 1,
     "must_contain": ["music_listening"],
     "must_not": ["ILIKE"]},

    {"q": "cortico-cortical evoked potentials iEEG",
     "expected_min": 1,
     "must_contain": ["cortico_cortical_evoked_potentials"],
     "must_not": ["ILIKE"]},

    {"q": "spinal cord fMRI",
     "expected_min": 1,
     "must_contain": [],
     "must_not": []},

    {"q": "datasets with both fMRI and EEG",
     "expected_min": 1,
     "must_contain": ["functional_mri", "electroencephalography"],
     "must_not": []},

    {"q": "datasets with fMRI and DTI",
     "expected_min": 1,
     "must_contain": ["functional_mri", "diffusion_mri"],
     "must_not": []},

    {"q": "Healthy Brain Network pediatric EEG",
     "expected_min": 1,
     "must_contain": [],
     "must_not": []},
]


def run_query(question):
    data = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        API,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def evaluate(q_spec, response):
    issues = []
    if "error" in response:
        return ["API_ERROR: " + response["error"]]

    sql = (response.get("translation") or {}).get("sql", "")
    count = len(response.get("datasets", []))
    message = response.get("message", "")

    # Zero result check
    if q_spec["expected_min"] > 0 and count == 0:
        issues.append(f"ZERO_RESULTS (expected ≥{q_spec['expected_min']})")

    # Suspicious fallback: very high result counts often mean no filter applied
    if count > 500:
        issues.append(f"POSSIBLE_FALLBACK ({count} results — filter may not have applied)")

    # must_contain checks (case-insensitive on SQL)
    sql_lower = sql.lower()
    for fragment in q_spec.get("must_contain", []):
        if fragment.lower() not in sql_lower:
            issues.append(f"MISSING_IN_SQL: '{fragment}'")

    # must_not checks — treated as regex on the original SQL
    for pattern in q_spec.get("must_not", []):
        if re.search(pattern, sql, re.IGNORECASE):
            issues.append(f"FORBIDDEN_IN_SQL matched '{pattern}'")

    return issues


def main():
    total = len(QUERIES)
    passed = 0
    results = []

    print(f"Running {total} queries against {API}\n{'─'*70}")

    for i, spec in enumerate(QUERIES, 1):
        q = spec["q"]
        print(f"[{i:3d}/{total}] {q[:70]}", end=" ... ", flush=True)
        resp = run_query(q)
        issues = evaluate(spec, resp)

        sql = (resp.get("translation") or {}).get("sql", "SQL unavailable")
        count = len(resp.get("datasets", []))
        first_names = [d["name"][:50] for d in resp.get("datasets", [])[:3]]
        message = resp.get("message", "")

        if not issues:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"

        print(f"{status}  ({count} results)")
        results.append({
            "n": i, "q": q, "status": status, "issues": issues,
            "count": count, "sql": sql, "first": first_names, "message": message
        })

        # Rate-limit: ~8 req/min to stay under Gemini limits
        if i < total:
            time.sleep(8)

    print(f"\n{'═'*70}")
    print(f"SUMMARY: {passed}/{total} passed ({100*passed//total}%)")
    print(f"{'═'*70}\n")

    failures = [r for r in results if r["status"] == "FAIL"]
    if failures:
        print(f"FAILURES ({len(failures)}):\n{'─'*70}")
        for r in failures:
            print(f"\n[{r['n']:3d}] {r['q']}")
            for issue in r["issues"]:
                print(f"      ⚠  {issue}")
            print(f"      Results: {r['count']}")
            if r["first"]:
                for name in r["first"]:
                    print(f"        • {name}")
            # Print abbreviated SQL for diagnosis
            sql_lines = r["sql"].splitlines()
            for line in sql_lines[:8]:
                print(f"      SQL: {line}")

    # Save full results to JSON
    out_path = "/tmp/query_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
