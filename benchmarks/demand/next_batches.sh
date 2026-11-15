#!/bin/sh
# Re-extract from the full texts on disk and cut new writer batches.
# Without "all", an incomplete last batch is dropped so its papers wait for more.
cd "$(dirname "$0")" || exit 1
python3 -I extract_mentions.py && python3 -I classify_reuse.py 2>&1 | sed -n 5p && python3 -I make_writer_batches.py
python3 -I - "$1" <<'PY'
import json, glob, os, sys
todo = []
for f in sorted(glob.glob('work/writer_batches/batch_*.json')):
    written = f.replace('writer_batches', 'written').replace('.json', '.jsonl')
    if os.path.exists(written):
        continue
    n = len(json.load(open(f)))
    if n < 40 and (len(sys.argv) < 2 or sys.argv[1] != 'all'):
        os.remove(f)
        print('dropped incomplete', os.path.basename(f), n)
    else:
        todo.append(os.path.basename(f))
print('to write:', todo)
PY
