#!/usr/bin/env python3
"""Author's 0-weight idea (2026-10-28 23:5x) applied to Q4_TG: the four tag grids as built for run E have no exact 0
(symmetric levels); about 13 % of 4-bit expert weights sit at level 0. Grids with an exact 0 vs without, full importance
weighting, real blk.20 experts (4 experts, every 8th row), per-block scale before zip rounding (slightly optimistic).
MEASURED 2026-10-28 23:5x (weighted error): gate: no 0 0.004438, uniform with 0 0.004293, uniform + log with 0 0.004131;
down: 0.004461 / 0.004332 / 0.004159; Q4_K + imatrix in the file: gate 0.004623, down 0.004721.
Run from scratchpad/q4tg (imp_*.f32 written from ../urm/merged.gguf)."""
import numpy as np
SL = '/mnt/user-data/uploads/pelias-tests/gp/M/slices/'
def lg(r, n=16): m = r ** np.arange(n // 2); return np.concatenate([-m[::-1], m])
nm = lambda g: g / np.abs(g).max()
uni = nm(np.arange(16) - 7.5); uz = nm(np.arange(16) - 7.0)
G = {'no zero (E)': [uni, nm(lg(1.3)), nm(lg(1.45)), uni * 0.6],
     'uniform with 0': [uz, nm(lg(1.3)), nm(lg(1.45)), uz * 0.6],
     'uniform + log with 0': [uz, nm(np.concatenate([lg(1.3)[:8], [0], lg(1.3)[9:]])), nm(lg(1.45)), uz * 0.6]}
FS = np.linspace(0.35, 1.25, 19)
def q(Xb, W, grids):
    B = len(Xb); amax = np.abs(Xb).max(1) + 1e-30; best = np.full(B, np.inf); bq = np.zeros_like(Xb)
    for f in FS:
        s = amax * f; qg = []; eg = []
        for g in grids:
            qq = g[np.abs(Xb[:, :, None] / s[:, None, None] - g).argmin(2)] * s[:, None]; qg.append(qq)
            eg.append((W * (Xb - qq) ** 2).reshape(B, 4, 8).sum(2))
        eg = np.stack(eg); t = eg.argmin(0); e = np.take_along_axis(eg, t[None], 0)[0].sum(1); m = e < best
        if m.any():
            qq = np.take_along_axis(np.stack(qg).reshape(len(grids), B, 4, 8), t[None, :, :, None].repeat(8, 3), 0)[0].reshape(B, 32)
            best[m] = e[m]; bq[m] = qq[m]
    return bq
for name, ne0, f in (('blk.20.ffn_gate_exps.weight', 2048, 'imp_ffn_gate_exps.f32'), ('blk.20.ffn_down_exps.weight', 512, 'imp_ffn_down_exps.f32')):
    raw = np.fromfile(SL + name + '.bin', dtype=np.uint16).astype(np.uint32) << 16
    rows = raw.view(np.float32).astype(np.float64).reshape(4, -1, ne0)[:, ::8]
    imp = np.fromfile(f, np.float32).reshape(4, ne0).astype(np.float64); out = {}
    for lab, grids in G.items():
        num = den = 0
        for e in range(4):
            X = rows[e].reshape(-1, 32); Wt = np.broadcast_to(imp[e], rows[e].shape).reshape(-1, 32)
            Q = q(X, Wt, grids); num += (Wt * (X - Q) ** 2).sum(); den += (Wt * X ** 2).sum()
        out[lab] = num / den
    print(name.split('.')[2], ' '.join(f"{k}: {v:.6f}" for k, v in out.items()))
