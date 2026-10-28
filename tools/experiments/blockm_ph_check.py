# provenance: original (Block M experiment, sandbox only; needs pt.py and the kq helper from the Block M sandbox)
# Copyright (c) 2026 John Lipka and the Pelias contributors
# planes by heat: error of the full value and of the base plane alone (cold read), zip scales (fp16 per 256 + 6-bit per 32)
exec(open('pt.py').read().split("meta = {")[0])
import json
meta = {m['name']: m['shape'] for m in json.load(open(S + '/slices.json'))}
def planes(w, bits, base_bits):
    qmax = 2 ** (bits - 1) - 1                      # symmetric -qmax..qmax, offset to 0..2^bits-1
    b = w.reshape(w.shape[0], -1, 32)
    d32 = np.abs(b).max(-1, keepdims=True) / qmax
    sup = d32.reshape(b.shape[0], -1, 8, 1).max(2, keepdims=True).astype(np.float16).astype(np.float64); sup[sup == 0] = 1e-30
    lv = np.clip(np.ceil(d32.reshape(b.shape[0], -1, 8, 1) / sup * 63), 1, 63)
    d = (lv / 63 * sup).reshape(b.shape[0], -1, 1)
    q = np.clip(np.round(b / d), -qmax, qmax)
    full = q * d
    off = q + 2 ** (bits - 1)                       # 0..2^bits-1 (the -2^(bits-1) code unused)
    sh = bits - base_bits
    top = np.floor(off / 2 ** sh)                   # the base plane: the top base_bits bits
    mid = top * 2 ** sh + (2 ** sh - 1) / 2.0       # reconstruct at the middle of the bucket
    base = (mid - 2 ** (bits - 1)) * d
    return full.reshape(w.shape), base.reshape(w.shape)
for name, label in (('blk.20.ffn_gate_exps.weight', 'gate_exps (Q4_K in urm)'), ('blk.20.ffn_down_exps.weight', 'down_exps (Q5_K in urm)')):
    w = load(name, meta[name][0])
    print('== ' + label)
    for tid, tn in ((10, 'Q2_K 2.625'), (11, 'Q3_K 3.4375'), (12, 'Q4_K 4.5'), (13, 'Q5_K 5.5')):
        print('  %-28s error %.6f' % (tn, rel(w, kquant(w, tid))))
    for bits, bb in ((4, 2), (5, 4)):
        f, b = planes(w, bits, bb)
        print('  %d-bit zip, all planes       error %.6f' % (bits, rel(w, f)))
        print('  %d-bit zip, base plane %d bits error %.6f' % (bits, bb, rel(w, b)))
