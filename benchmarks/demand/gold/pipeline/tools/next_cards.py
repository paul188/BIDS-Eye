"""Print the next cards of a batch that have no judgment yet.

    python3 tools/next_cards.py BATCH [N]        (N defaults to 8)

Shows the question (criterion) the cards belong to, then N cards in full.
Run it again after appending your judgments: it always continues where the
output file stops, so the work can be resumed at any time.
"""
import json, sys
from common import load_batch, done_pairs
name = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
b, need = load_batch(name)
done = done_pairs(name)
todo = [p for p in need if p not in done]
print(f"BATCH {name}: {len(need) - len(todo)} of {len(need)} judged, {len(todo)} to go")
if not todo:
    print("ALL DONE. Give your final report."); sys.exit()
qid = todo[0][0]
chunk = [p for p in todo if p[0] == qid][:n]
if "questions" in b:
    q = next(x for x in b["questions"] if x["query_id"] == qid)
    left = sum(1 for p in todo if p[0] == qid)
    print(f"\nQUESTION {qid}  ({left} candidates left for this question)")
    print("  what the user types :", q["question"])
    print("  original request    :", q["original_request"])
    print("  RELEVANT IF         :", q["relevant_if"])
    print("  PARTIALLY IF        :", q["partially_if"])
else:
    print("\nTASK: who are the participants of each dataset (see POPULATION_INSTRUCTIONS.md)")
def compact(c):
    """A participant card cut to what the decision needs (about a third of the full card)."""
    r = c.get("recorded") or {}
    p = r.get("participants_tsv") or {}
    lines = [f"{c['acc']} | {c.get('name')}",
             f"  uploader labels: {c.get('uploader_labels') or '-'}",
             f"  subjects: {r.get('n_subjects')} | datatypes: {', '.join(r.get('datatypes') or [])} | dataset ages: {r.get('dataset_age_range') or '-'}"]
    if p:
        lines.append(f"  participants.tsv: rows={p.get('rows')} male={p.get('male')} female={p.get('female')} age={p.get('age_min')}-{p.get('age_max')} adults={p.get('adults')}"
                     f" diagnoses={p.get('normalised_diagnoses') or '-'} group column={p.get('group_column') or '-'} other columns={(p.get('other_columns') or [])[:14]}")
    if c.get("description"): lines.append("  description: " + c["description"][:260])
    if c.get("readme_head"): lines.append("  readme: " + c["readme_head"][:480])
    for s_ in (c.get("passages_around_keywords") or [])[:3]:
        lines.append("  passage: " + s_[:230])
    for pp in (c.get("primary_papers") or [])[:1]:
        lines.append("  paper: " + str(pp.get("title"))[:160])
    return "\n".join(lines)

for _, acc in chunk:
    print("\n" + "-" * 60)
    if "questions" in b:
        print(json.dumps(b["cards"][acc], ensure_ascii=False, indent=1))
    else:
        print(compact(b["cards"][acc]))
print("\n" + "=" * 100)
print("Judge these", len(chunk), "now and append them with tools/append.py before asking for more:", ", ".join(a for _, a in chunk))
