"""Who the participants are, per dataset: rules first (uploader label, recorded diagnoses), the rest is judged."""
import re, json, random, os
import lib, spec
D, ON = lib.load()
CLIN = re.compile(r"patient|disorder|disease|syndrome|diagnos|clinical|lesion|stroke|epilep|tumou?r|schizo|depress|autis|adhd|parkinson|alzheimer|dementia|injur|impair|deficit|chronic|\bpain\b|addict|dependen|abuse|obes|ptsd|\bocd\b|bipolar|psychosis|aphasi|amnesi|blind|deaf|concussion|\btbi\b|sclerosis|migraine|insomnia|apnea|dyslexi|cancer|infarct|\bhiv\b|covid|preterm|premature|surgery|surgical|implant|medicat", re.I)
def rule(d):
    if not spec.HUMAN(d): return "not_human"
    dx = (d.get("participants") or {}).get("dx_counts") or {}
    codes = {k for k in dx if k != "(null)"}
    hc = {k for k in codes if re.search(r"healthy|control", k)}
    pat = codes - hc
    label = (d.get("dx_status") or "").strip().lower()
    if hc and pat: return "mixed:dx"
    if label == "healthy / control" and not pat: return "healthy_only:label"
    if hc and not pat and not label: return "healthy_only:dx"
    if pat or (label and label != "healthy / control"): return "judge:patient signal"
    text = lib.our_text(d)
    if len(text.strip()) < 200: return "judge:thin text"
    return "judge:clinical words" if CLIN.search(text) else "presumed_healthy"
NEED = sorted(k for k, d in D.items() if spec.SPS(d, "t1_weighted_mri") or spec.REST_BOLD(d) or spec.EEG(d))
RULES = {k: rule(D[k]) for k in D}
if __name__ == "__main__":
    import cards
    random.seed(7)
    judge = [k for k in NEED if RULES[k].startswith("judge")]
    audit = random.sample([k for k in NEED if RULES[k] == "presumed_healthy"], 45) + random.sample([k for k in NEED if RULES[k] == "healthy_only:label"], 25) + [k for k in NEED if RULES[k] == "mixed:dx"][:20]
    allk = sorted(set(judge + audit)); random.shuffle(allk)
    pats = [r"patient|diagnos|disorder|disease", r"healthy|control(s| group| participants)|neurotypical|typically developing", r"\bage[ds]?\b|years? old|children|adults|adolescen|infant", r"exclu|inclusion"]
    nb = 6
    for i in range(nb):
        part = allk[i::nb]
        b = {"batch": f"pop_{i+1}", "cards": {k: cards.card(k, pats) for k in part}}
        json.dump(b, open(f"{lib.G}/batches/pop_{i+1}.json", "w"), ensure_ascii=False)
        print(f"pop_{i+1}", len(part), os.path.getsize(f"{lib.G}/batches/pop_{i+1}.json") // 1000, "kB")
    json.dump({"rules": RULES, "audit": audit}, open(f"{lib.G}/population_rules.json", "w"))
