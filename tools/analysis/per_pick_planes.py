#!/usr/bin/env python3
"""DPA-MoE's lesson tested offline (author 07:17): read the planes only for the picks the router weights most, per token.
WikiText-2 routes (urm_ZT2 dump, 25,500 tokens x 40 layers x 8 picks, router weights as used). An expert's output enters the
layer as w x y_e, so its error enters as w^2 x (its error); the planes' benefit captured by reading a subset of picks is
sum over the subset of w^2 / sum over all picks of w^2 (the plane cuts each expert's error by the same factor; ASSUMPTION),
and the plane bytes read are the subset's share of picks. Also with linear weight (the usage metric of tier_project.py)."""
import struct, numpy as np
r = open('/mnt/user-data/uploads/pelias-tests/gp/C6/urmzt2_routes.bin', 'rb').read()
_, _, _, L, _ = struct.unpack_from('<5I', r, 8)
RT = np.dtype([('sel', '<i2', 8), ('w', '<f4', 8), ('pl', '<f4'), ('pn', '<f4'), ('pred', '<i2', 8)])
w = np.frombuffer(r, RT, offset=36).reshape(-1, L)['w'].astype(np.float64)  # (T, L, 8)
ws = -np.sort(-w, -1)
print(f'router weight by rank (mean over tokens and layers): ' + ' '.join(f'{x:.3f}' for x in ws.mean((0, 1))))
print(f'sum of the 8 weights: mean {w.sum(-1).mean():.3f}')
for k in (1, 2, 3, 4, 6, 8):
    sq = (ws[..., :k] ** 2).sum() / (ws ** 2).sum(); lin = ws[..., :k].sum() / ws.sum()
    print(f'  top {k} picks per token (bytes {k / 8 * 100:5.1f} %): benefit captured {sq * 100:5.1f} % (w^2), {lin * 100:5.1f} % (w)')
for tau in (0.05, 0.1, 0.15, 0.2):
    m = w >= tau; sq = (w[m] ** 2).sum() / (w ** 2).sum()
    print(f'  picks with w >= {tau:.2f} (bytes {m.mean() * 100:5.1f} %): benefit captured {sq * 100:5.1f} % (w^2), {w[m].sum() / w.sum() * 100:5.1f} % (w)')
