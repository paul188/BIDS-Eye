import sys
sys.path.insert(0, '/home/s24pjoha_hpc/Text_To_SQL/training_data_generation')
import value_mappings as vm
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://user:password@localhost:5429/bids_sql')

fields = {
    'task':       ('bids_objects',      'task',       vm.clean_task),
    'suffix':     ('bids_objects',      'suffix',     vm.clean_suffix),
    'datatype':   ('bids_objects',      'datatype',   vm.clean_datatype),
    'sex':        ('bids_participants', 'sex',        vm.clean_sex),
    'handedness': ('bids_participants', 'handedness', vm.clean_handedness),
    'diagnosis':  ('bids_participants', 'diagnosis',  vm.clean_diagnosis),
}

with engine.connect() as conn:
    for field, (table, col, fn) in fields.items():
        sql = "SELECT %s, COUNT(*) FROM %s WHERE %s IS NOT NULL AND %s != '' GROUP BY %s ORDER BY COUNT(*) DESC" % (col, table, col, col, col)
        rows = conn.execute(text(sql)).fetchall()
        total = len(rows)
        filtered = sum(1 for r in rows if fn(r[0]) is None)
        mapped = sum(1 for r in rows if fn(r[0]) is not None and fn(r[0]) != r[0])
        passthrough = [r[0] for r in rows if fn(r[0]) == r[0]]
        print("%s: total=%d  mapped=%d  filtered=%d  pass-through=%d" % (field, total, mapped, filtered, len(passthrough)))
        for p in passthrough[:5]:
            print("  pass: %r" % p)
