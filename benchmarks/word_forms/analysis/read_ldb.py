"""Pull one localStorage key out of a Chrome LevelDB .ldb table (no deps)."""
import struct, sys

def varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7f) << s; s += 7
        if c < 0x80: return r, i

def unsnappy(src):
    n, i = varint(src, 0); out = bytearray()
    while i < len(src):
        t = src[i]; i += 1; k = t & 3
        if k == 0:
            ln = t >> 2
            if ln >= 60:
                nb = ln - 59; ln = int.from_bytes(src[i:i+nb], 'little'); i += nb
            ln += 1; out += src[i:i+ln]; i += ln; continue
        if k == 1:
            ln = ((t >> 2) & 7) + 4; off = ((t >> 5) << 8) | src[i]; i += 1
        elif k == 2:
            ln = (t >> 2) + 1; off = int.from_bytes(src[i:i+2], 'little'); i += 2
        else:
            ln = (t >> 2) + 1; off = int.from_bytes(src[i:i+4], 'little'); i += 4
        for _ in range(ln): out.append(out[-off])
    assert len(out) == n
    return bytes(out)

def block(data, off, size):
    raw = data[off:off+size]; typ = data[off+size]
    return unsnappy(raw) if typ == 1 else raw

def entries(blk):
    nrest = struct.unpack('<I', blk[-4:])[0]
    end = len(blk) - 4 - 4*nrest; i = 0; key = b''
    while i < end:
        sh, i = varint(blk, i); ns, i = varint(blk, i); vl, i = varint(blk, i)
        key = key[:sh] + blk[i:i+ns]; i += ns
        yield key, blk[i:i+vl]; i += vl

data = open(sys.argv[1], 'rb').read()
footer = data[-48:]
_, j = varint(footer, 0); _, j = varint(footer, j)
io, j = varint(footer, j); isz, j = varint(footer, j)
found = []
for k, v in entries(block(data, io, isz)):
    bo, p = varint(v, 0); bs, p = varint(v, p)
    for key, val in entries(block(data, bo, bs)):
        if sys.argv[2].encode() in key[:-8]:
            seq = struct.unpack('<Q', key[-8:])[0]
            found.append((seq >> 8, seq & 0xff, val))
for seq, kind, val in sorted(found):
    # Chrome localStorage value: 1 prefix byte, 0 = UTF-16LE, 1 = Latin-1
    txt = val[1:].decode('utf-16-le') if val[:1] == b'\x00' else val[1:].decode('latin-1')
    print(seq, kind, len(txt), file=sys.stderr)
    last = txt
open(sys.argv[3], 'w').write(last)
