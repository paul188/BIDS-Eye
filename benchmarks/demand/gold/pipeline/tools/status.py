"""What is judged and what is missing, per batch.   python3 tools/status.py [--todo]"""
import glob, os, sys
from common import G, load_batch, done_pairs
tot_need = tot_done = 0
todo = []
for p in sorted(glob.glob(f"{G}/batches/*.json")):
    name = os.path.basename(p)[:-5]
    b, need = load_batch(name)
    done = done_pairs(name)
    n = sum(1 for x in need if x in done)
    tot_need += len(need); tot_done += n
    if n < len(need): todo.append((name, len(need) - n))
    if "--todo" not in sys.argv:
        print(f"{name:26} {n:4} / {len(need):4} {'complete' if n == len(need) else ''}")
print(f"TOTAL {tot_done} / {tot_need} judged; {len(todo)} batches incomplete")
if "--todo" in sys.argv:
    for name, left in todo: print(f"  {name:26} {left} to go")
