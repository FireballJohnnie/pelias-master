#!/usr/bin/env python3
"""The usage-tiered allocation (usage_alloc.py's options and Lagrangian) ranked on one text and scored on another, picks
as usage. Experts only; error relative to Unsloth's experts (gate/up Q4_K, down Q5_K) = 1; KL from the two-point line."""
import numpy as np
d = np.load('domain_usage_v1.npz'); U = {'WikiText': d['wiki'], 'chat+code': d['chatcode']}
U['mix 50/50'] = (U['WikiText'] + U['chat+code']) / 2
L, E = U['WikiText'].shape; NW = 2048 * 512
G = {'IQ4_XS': (4.25, 1.182), 'Q4_K': (4.5, 1.0), 'Q5_K': (5.5, 0.254), 'Q6_K': (6.5625, 0.064), 'Q8_0': (8.5, 0.007)}
Dn = {'IQ4_XS': (4.25, 1.166), 'Q4_K': (4.5, 1.0), 'Q5_K': (5.5, 0.253), 'Q6_K': (6.5625, 0.062), 'Q8_0': (8.5, 0.006)}
ref = (2 * 1.0 + 0.253) / 3
def alloc(u, budget):
    def pick(lam):
        out = {}; tb = 0.0
        for kind, opts, m in (('gu', G, 2), ('d', Dn, 1)):
            bp = np.array([v[0] for v in opts.values()]); er = np.array([v[1] for v in opts.values()])
            j = (u[..., None] * er * m + lam * bp * NW * m / 8e9).argmin(-1); out[kind] = (j, er, m); tb += (bp[j] * NW * m / 8e9).sum()
        return tb, out
    lo, hi = 0.0, 100.0
    for _ in range(70):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pick(mid)[0] > budget else (lo, mid)
    return pick(hi)
def score(out, u): return sum((u * er[j]).sum() * m for j, er, m in out.values()) / (3 * L) / ref
print('experts 19.57 GB (file 22.65 GB), lowest type IQ4_XS; rows: ranked on, columns: scored on (error x Unsloth -> KL, DERIVED)')
for b, tot in ((19.57, 22.65), (21.0, 24.08)):
    print(f'file {tot} GB:')
    for src in U:
        _, out = alloc(U[src], b)
        cells = '   '.join(f'{dst}: {score(out, U[dst]):.3f} -> {0.0051 + 0.0078 * score(out, U[dst]):.4f}' for dst in ('WikiText', 'chat+code'))
        print(f'  ranked on {src:9s}  {cells}')
