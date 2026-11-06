#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors. Confidential, not for distribution.
#
# tools/probes/sdecode/tg2_tags.py - the Q4_TG2 tag histogram of a Pelias tier file (R3 rebuild design, stage 0.2; 2026-11-06).
# Read-only: maps the GGUF, walks every Q4_TG2 tensor (type id 204; 144-byte blocks of 256 values; the block's 32 groups of 8
# values carry one 2-bit grid tag each in header bytes 8..15, group g at bits 2g), and counts the tags. Grids (quant_cpu.cpp,
# Q4TGGrids v2): tag 0 = (i - 7) / 8 (uniform, exact zero) and tag 3 = 0.6 x tag 0 (uniform) are integer codes x one scale
# -> exact on dp4a; tags 1 and 2 are log grids (1.3^m / 1.3^7, 1.45^m / 1.45^7; tag 1's smallest positive level = 0) ->
# float math only. The share of tags 0 + 3 is how much of the Q4_TG2 decode an exact dp4a path recovers.
# Usage: python3 tg2_tags.py <file.gguf>    (numpy; ~1-3 min for a 22 GB file; prints per tensor kind and in total)
import struct, sys, time
import numpy as np

Q4_TG2 = 204
BLOCK = 144

def read_str(f):
    (n,) = struct.unpack('<Q', f.read(8))
    return f.read(n).decode('utf-8', 'replace')

def skip_value(f, t):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t == 8:
        read_str(f)
    elif t == 9:
        (et,) = struct.unpack('<I', f.read(4))
        (n,) = struct.unpack('<Q', f.read(8))
        if et == 8:
            for _ in range(n):
                read_str(f)
        elif et == 9:
            for _ in range(n):
                (et2,) = struct.unpack('<I', f.read(4))
                skip_value(f, 9) if et2 == 9 else skip_value(f, et2)
        else:
            f.seek(sizes[et] * n, 1)
    else:
        f.seek(sizes[t], 1)

def read_value(f, t):
    if t == 4:
        return struct.unpack('<I', f.read(4))[0]
    if t == 5:
        return struct.unpack('<i', f.read(4))[0]
    if t == 10:
        return struct.unpack('<Q', f.read(8))[0]
    skip_value(f, t)
    return None

def main(path):
    t0 = time.time()
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            print('not a GGUF file'); return 1
        (ver,) = struct.unpack('<I', f.read(4))
        (nt,) = struct.unpack('<Q', f.read(8))
        (nkv,) = struct.unpack('<Q', f.read(8))
        align = 32
        for _ in range(nkv):
            k = read_str(f)
            (t,) = struct.unpack('<I', f.read(4))
            v = read_value(f, t)
            if k == 'general.alignment' and v is not None:
                align = v
        tensors = []
        for _ in range(nt):
            name = read_str(f)
            (nd,) = struct.unpack('<I', f.read(4))
            dims = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
            (ty,) = struct.unpack('<I', f.read(4))
            (off,) = struct.unpack('<Q', f.read(8))
            tensors.append((name, dims, ty, off))
        data0 = (f.tell() + align - 1) // align * align
    print(f'{path}: GGUF v{ver}, {nt} tensors, data at {data0}, alignment {align}')
    mm = np.memmap(path, dtype=np.uint8, mode='r')
    total = np.zeros(4, dtype=np.int64)
    kinds = {}
    ntg2 = 0
    for name, dims, ty, off in tensors:
        if ty != Q4_TG2:
            continue
        ntg2 += 1
        nvals = 1
        for d in dims:
            nvals *= d
        nblk = nvals // 256
        start = data0 + off
        blocks = mm[start:start + nblk * BLOCK].reshape(nblk, BLOCK)
        tags = blocks[:, 8:16].copy().view('<u8').reshape(-1)
        h = np.zeros(4, dtype=np.int64)
        for g in range(32):
            h += np.bincount(((tags >> np.uint64(2 * g)) & np.uint64(3)).astype(np.int64), minlength=4)
        kind = 'ffn_down' if 'ffn_down' in name else 'ffn_up' if 'ffn_up' in name else 'ffn_gate' if 'ffn_gate' in name else 'other'
        kinds.setdefault(kind, np.zeros(4, dtype=np.int64))
        kinds[kind] += h
        total += h
    def line(label, h):
        n = int(h.sum())
        if n == 0:
            return f'{label}: none'
        p = [100.0 * int(x) / n for x in h]
        return (f'{label}: groups {n:,}: tag0 (uniform, exact 0) {p[0]:.1f} %, tag1 (log 1.3, 0 level) {p[1]:.1f} %, '
                f'tag2 (log 1.45) {p[2]:.1f} %, tag3 (0.6 x uniform) {p[3]:.1f} %  ->  exact-dp4a share (tags 0 + 3) {p[0] + p[3]:.1f} %')
    print(f'Q4_TG2 tensors: {ntg2}')
    for k in sorted(kinds):
        print(line(k, kinds[k]))
    print(line('TOTAL', total))
    print(f'done in {time.time() - t0:.0f} s')
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ''))
