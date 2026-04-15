import sys
sys.path.insert(0, '/home/s24pjoha_hpc/Text_To_SQL/training_data_generation')
import value_mappings as vm
from sqlalchemy import create_engine, text
engine = create_engine('postgresql://user:password@localhost:5429/bids_sql')
with engine.connect() as conn:
    rows = conn.execute(text("SELECT diagnosis, COUNT(*) FROM bids_participants WHERE diagnosis IS NOT NULL AND diagnosis != '' GROUP BY diagnosis ORDER BY COUNT(*) DESC")).fetchall()
passthrough = [(r[0], r[1]) for r in rows if vm.clean_diagnosis(r[0]) is not None and vm.clean_diagnosis(r[0]) == r[0]]
print('Pass-throughs (%d total):' % len(passthrough))
for v, c in passthrough:
    print('  %r (%d)' % (v, c))
