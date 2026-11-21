"""Dump dataset text from the local catalogue copy to a token file (read-only)."""
import json, os, re, subprocess, sys

pw = os.environ["PGPASSWORD"]
sql = ("set default_transaction_read_only=on; "
       "copy (select accession_id, coalesce(name,''), coalesce(description_text,''), coalesce(readme_text,'') "
       "from bids_datasets order by 1) to stdout with (format csv)")
out = subprocess.run(["psql", "-h", "127.0.0.1", "-p", "55439", "-U", "user", "-d", "bids_sql", "-Atc", sql],
                     capture_output=True, text=True, check=True).stdout
import csv, io
rows = list(csv.reader(io.StringIO(out)))
tok = re.compile(r"[a-z0-9]+")
docs = {}
for ds, name, desc, readme in rows:
    text = " ".join([name, desc, readme]).lower()
    docs[ds] = tok.findall(text)
json.dump(docs, open(sys.argv[1], "w"))
print(len(docs), sum(len(v) for v in docs.values()))
