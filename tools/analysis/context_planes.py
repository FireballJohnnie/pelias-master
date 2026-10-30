#!/usr/bin/env python3
"""Projection: a 22.65 GB tiered base (ranked on the 50/50 WikiText + chat/code mix, Q4_TG2 floor) plus a 1-plane side file on
every expert, read at run time only for the experts hot in the current context. Per WikiText chunk (255 scored tokens,
urm_ZT2 routes) the hot set is causal: the top N experts per layer by routing weight over the tokens already seen in the
chunk (first W tokens: no planes). Error per pick = its type's error x (0.34 when its plane is read: +1 plane measured
0.32-0.36x across Q4_K/IQ4_XS/Q5_K/Q6_K on blk.20). Usage-weighted error relative to Unsloth's experts; KL from the line."""
import runpy, io, contextlib, struct, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    m = runpy.run_path('alloc_cross.py')
G = {'Q4_TG2': (4.5, 0.899), 'Q5_K': (5.5, 0.254), 'Q6_K': (6.5625, 0.064), 'Q8_0': (8.5, 0.007)}
D = {'Q4_TG2': (4.5, 0.886), 'Q5_K': (5.5, 0.253), 'Q6_K': (6.5625, 0.062), 'Q8_0': (8.5, 0.006)}
m['G'].clear(); m['G'].update(G); m['Dn'].clear(); m['Dn'].update(D)
tb, out = m['alloc'](m['U']['mix 50/50'], 19.57)
egu = out['gu'][1][out['gu'][0]]; ed = out['d'][1][out['d'][0]]; EX = (2 * egu + ed) / 3 / m['ref']   # (L, E) error per pick
r = open('/mnt/user-data/uploads/pelias-tests/gp/C6/urmzt2_routes.bin', 'rb').read()
_, _, _, L, _ = struct.unpack_from('<5I', r, 8)
RT = np.dtype([('sel', '<i2', 8), ('w', '<f4', 8), ('pl', '<f4'), ('pn', '<f4'), ('pred', '<i2', 8)])
rec = np.frombuffer(r, RT, offset=36).reshape(-1, L); T = rec.shape[0]; C = 100; P = T // C
sel = rec['sel'].astype(int); w = rec['w'].astype(np.float64)
li = np.arange(L)[None, :, None]
base_err = (w * EX[li, sel]).sum() / w.sum()
print(f'tiered base alone (experts {tb:.2f} GB, file 22.65 GB), on WikiText: error {base_err:.3f} -> KL {0.0051 + 0.0078 * base_err:.4f}')
PL = 0.34; plane_gb = 3 * 40 * 256 * 2048 * 512 * 1.0625 / 8 / 1e9
for N in (16, 32, 64):
    for Wm in (32, 64):
        num = den = hits = picks = 0.0
        for c in range(C):
            s, ww = sel[c * P:(c + 1) * P], w[c * P:(c + 1) * P]
            acc = np.zeros((L, 256))
            for t in range(P):
                hot = np.zeros((L, 256), bool)
                if t >= Wm: np.put_along_axis(hot, np.argsort(-acc, 1)[:, :N], True, 1)
                h = hot[np.arange(L)[:, None], s[t]]
                e = EX[np.arange(L)[:, None], s[t]] * np.where(h, PL, 1.0)
                num += (ww[t] * e).sum(); den += ww[t].sum(); hits += h.sum(); picks += h.size
                np.add.at(acc, (np.repeat(np.arange(L), 8), s[t].ravel()), ww[t].ravel())
        err = num / den; mb = hits / (T) * 3 * 2048 * 512 * 1.0625 / 8 / 1e6
        print(f'  + planes on the context-hot top {N:2d} (after {Wm} tokens): picks with a plane {hits/picks*100:4.1f} %, error {err:.3f} -> KL {0.0051 + 0.0078 * err:.4f}; extra read {mb:.0f} MB per token')
print(f'side file, 1 plane on every expert: {plane_gb:.2f} GB')
