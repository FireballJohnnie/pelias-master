#!/usr/bin/env python3
"""Hot-set coverage for usage-weighted planes: per layer, the experts ranked by picks on chunks 0-49 (WikiText-2, urm_ZT2
routes, engine format); share of the picks and of the routing weight on chunks 50-99 that land in the top N."""
import struct, numpy as np
r = open('/mnt/user-data/uploads/pelias-tests/gp/C6/urmzt2_routes.bin', 'rb').read()
assert r[:8] == b'PELROUT1'
_, _, _, L, _ = struct.unpack_from('<5I', r, 8)
RT = np.dtype([('sel', '<i2', 8), ('w', '<f4', 8), ('pl', '<f4'), ('pn', '<f4'), ('pred', '<i2', 8)])
rec = np.frombuffer(r, RT, offset=36).reshape(-1, L)
T = rec.shape[0]; half = T // 2
sel, w = rec['sel'].astype(np.int64), rec['w']
E = 256
print(f'tokens {T}, layers {L}; ranking on the first half, coverage on the second')
Ns = (16, 32, 48, 64, 84, 120, 161)
cov_p = np.zeros((L, len(Ns))); cov_w = np.zeros((L, len(Ns))); self_p = np.zeros((L, len(Ns)))
for l in range(L):
    a = np.bincount(sel[:half, l].ravel(), minlength=E); order = np.argsort(-a)
    bpick = np.bincount(sel[half:, l].ravel(), minlength=E).astype(float)
    bw = np.bincount(sel[half:, l].ravel(), weights=w[half:, l].ravel(), minlength=E)
    for j, n in enumerate(Ns):
        hot = order[:n]
        cov_p[l, j] = bpick[hot].sum() / bpick.sum(); cov_w[l, j] = bw[hot].sum() / bw.sum()
        self_p[l, j] = a[hot].sum() / a.sum()
for j, n in enumerate(Ns):
    print(f'  top {n:3d} of 256 ({n/256*100:4.1f} % of expert bytes): picks covered {cov_p[:, j].mean()*100:5.1f} % (worst layer {cov_p[:, j].min()*100:5.1f} %), routing weight covered {cov_w[:, j].mean()*100:5.1f} %; on the ranking half itself {self_p[:, j].mean()*100:5.1f} %')
