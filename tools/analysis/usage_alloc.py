#!/usr/bin/env python3
"""Usage-aware expert precision at a fixed size (DERIVED from MEASURED inputs).
Inputs: per-layer expert routing weight (urm_ZT2 routes, WikiText-2; ranked on chunks 0-49, scored on 50-99); weighted
error per storage option relative to Q4_K (blk.20 experts 0-3, base_planes_v1.txt); Unsloth's expert bytes (gate/up Q4_K
4.5 bpw, down Q5_K/Q6_K: 19.57 GB). Each (layer, expert) gets one option for gate+up and one for down; the usage-weighted
error is minimised under a byte budget (Lagrangian, exact per item). Error of a layer-expert = its share of the layer's
routing weight x the option's error; layers weighted equally (per-layer sensitivity UNMEASURED)."""
import struct, numpy as np
r = open('/mnt/user-data/uploads/pelias-tests/gp/C6/urmzt2_routes.bin', 'rb').read()
_, _, _, L, _ = struct.unpack_from('<5I', r, 8)
RT = np.dtype([('sel', '<i2', 8), ('w', '<f4', 8), ('pl', '<f4'), ('pn', '<f4'), ('pred', '<i2', 8)])
rec = np.frombuffer(r, RT, offset=36).reshape(-1, L); T = rec.shape[0]; h = T // 2; E = 256
U1 = np.zeros((L, E)); U2 = np.zeros((L, E))
for l in range(L):
    U1[l] = np.bincount(rec['sel'][:h, l].ravel().astype(int), weights=rec['w'][:h, l].ravel(), minlength=E)
    U2[l] = np.bincount(rec['sel'][h:, l].ravel().astype(int), weights=rec['w'][h:, l].ravel(), minlength=E)
U1 /= U1.sum(1, keepdims=True); U2 /= U2.sum(1, keepdims=True)
# options: name, bpw, error vs Q4_K (gate measured / down measured)
GATE = {'Q2_K': (2.625, 14.112), 'Q3_K': (3.4375, 4.382), 'IQ4_XS': (4.25, 1.182), 'Q4_K': (4.5, 1.0), 'Q5_K': (5.5, 0.254),
        'Q6_K': (6.5625, 0.064), 'Q8_0': (8.5, 0.007), 'Q4_K+1p': (5.5625, 0.355), 'Q4_K+2p': (6.625, 0.097), 'IQ4_XS+1p': (5.3125, 0.380),
        'Q3_K+1p': (4.5, 1.497), 'Q5_K+1p': (6.5625, 0.086)}
DOWN = {'Q2_K': (2.625, 14.031), 'Q3_K': (3.4375, 4.233), 'IQ4_XS': (4.25, 1.166), 'Q4_K': (4.5, 1.0), 'Q5_K': (5.5, 0.253),
        'Q6_K': (6.5625, 0.062), 'Q8_0': (8.5, 0.006), 'Q4_K+1p': (5.5625, 0.316), 'Q4_K+2p': (6.625, 0.092), 'IQ4_XS+1p': (5.3125, 0.353),
        'Q3_K+1p': (4.5, 1.368), 'Q5_K+1p': (6.5625, 0.079)}
NW = 2048 * 512  # weights per expert matrix
def run(opts_g, opts_d, budget_gb):
    # per item: gate+up (2 matrices) and down (1 matrix); minimise sum u*err*(matrices) + lam*bytes
    def pick(lam):
        tot_b = 0.0; err1 = 0.0; err2 = 0.0; cnt = {}
        for kind, opts, mats in (('gu', opts_g, 2), ('d', opts_d, 1)):
            names = list(opts); bp = np.array([opts[n][0] for n in names]); er = np.array([opts[n][1] for n in names])
            cost = U1[..., None] * er[None, None, :] * mats + lam * (bp[None, None, :] * NW * mats / 8 / 1e9)
            j = cost.argmin(-1)
            tot_b += (bp[j] * NW * mats / 8).sum() / 1e9
            err1 += (U1 * er[j]).sum() * mats; err2 += (U2 * er[j]).sum() * mats
            for k in range(len(names)): cnt[f'{kind}:{names[k]}'] = int((j == k).sum())
        return tot_b, err1 / (3 * L), err2 / (3 * L), cnt
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if pick(mid)[0] > budget_gb: lo = mid
        else: hi = mid
    return pick(hi)
# reference: Unsloth's own allocation, error relative to all-Q4_K: gate/up Q4_K (1.0), down Q5_K (0.253) -> per matrix mean
ref = (2 * 1.0 + 0.253) / 3
print('Unsloth experts: gate/up Q4_K, down Q5_K: 19.57 GB, usage-weighted error %.3f (relative to all-Q4_K = 1)' % ref)
types_only = [n for n in GATE if '+' not in n]
for label, og, od in (('types only', {k: GATE[k] for k in types_only}, {k: DOWN[k] for k in types_only}), ('types + planes', GATE, DOWN)):
    print(f'\n{label}:')
    for b in (17.5, 18.5, 19.57, 20.5, 21.5, 23.0):
        tb, e1, e2, cnt = run(og, od, b)
        used = ', '.join(f'{k} {v}' for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]) if v)
        print(f'  budget {b:5.2f} GB: experts {tb:5.2f} GB; error held-out {e2:.3f} ({e2/ref:.3f} x Unsloth; ranking half {e1/ref:.3f}); {used}')
