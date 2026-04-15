import sys
sys.path.insert(0, '/home/s24pjoha_hpc/Text_To_SQL/training_data_generation')
from sqlalchemy import create_engine, text
engine = create_engine('postgresql://user:password@localhost:5429/bids_sql')
with engine.connect() as conn:
    rows = conn.execute(text(
        "SELECT sex, COUNT(*) FROM bids_participants WHERE sex IS NOT NULL AND sex != '' GROUP BY sex ORDER BY COUNT(*) DESC"
    )).fetchall()
print(f"Total distinct values: {len(rows)}")
for v, c in rows:
    print(f"  {v!r:40s} ({c})")
