#!/usr/bin/env python3
"""The same 4.28 GB spent in the fixed tiered base instead of a side file: experts 23.85 GB (file 26.93 GB), mix ranking,
Q4_TG2 floor; scored on both texts. For comparison with context_planes.py."""
import runpy, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    m = runpy.run_path('alloc_cross.py')
G = {'Q4_TG2': (4.5, 0.899), 'Q5_K': (5.5, 0.254), 'Q6_K': (6.5625, 0.064), 'Q8_0': (8.5, 0.007)}
D = {'Q4_TG2': (4.5, 0.886), 'Q5_K': (5.5, 0.253), 'Q6_K': (6.5625, 0.062), 'Q8_0': (8.5, 0.006)}
m['G'].clear(); m['G'].update(G); m['Dn'].clear(); m['Dn'].update(D)
for b, f in ((19.57, 22.65), (23.85, 26.93)):
    tb, out = m['alloc'](m['U']['mix 50/50'], b)
    s = {t: m['score'](out, m['U'][t]) for t in ('WikiText', 'chat+code')}
    print(f'file {f} GB: WikiText {s["WikiText"]:.3f} -> KL {0.0051 + 0.0078 * s["WikiText"]:.4f}; chat+code {s["chat+code"]:.3f} -> KL {0.0051 + 0.0078 * s["chat+code"]:.4f}')
import numpy as np
print('expected expert bytes read per token (8 picks x 40 layers, usage-weighted bits per weight):')
for b, f in ((19.57, 22.65), (23.85, 26.93)):
    tb, out = m['alloc'](m['U']['mix 50/50'], b)
    bp = {k: np.array([v[0] for v in (G if k == 'gu' else D).values()])[out[k][0]] for k in ('gu', 'd')}
    for t in ('WikiText', 'chat+code'):
        u = m['U'][t]; bits = (u * (2 * bp['gu'] + bp['d'])).sum(1).mean()  # per pick: gate+up+down bits per weight-triple
        print(f'  file {f} GB, {t}: {bits / 8 * 2048 * 512 * 8 * 40 / 1e6:.0f} MB per token')
print('  Unsloth file: %.0f MB per token' % ((2 * 4.5 + 5.5 + 0.0) / 8 * 2048 * 512 * 8 * 40 / 1e6))
