#!/usr/bin/env python3
"""alloc_cross.py with Q4_TG2 (4.5 bpw; blk.20 error 0.899x Q4_K gate, 0.886x down, floor_norm_v1.txt) as an option and as
the floor instead of IQ4_XS."""
import runpy, io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    m = runpy.run_path('alloc_cross.py')
U, alloc, score = m['U'], m['alloc'], m['score']
base_g, base_d = dict(m['G']), dict(m['Dn'])
for label, drop, add in (('floor IQ4_XS (as before)', [], False), ('Q4_TG2 added, floor IQ4_XS', [], True), ('floor Q4_TG2', ['IQ4_XS', 'Q4_K'], True)):
    G = {k: v for k, v in base_g.items() if k not in drop}; D = {k: v for k, v in base_d.items() if k not in drop}
    if add: G['Q4_TG2'] = (4.5, 0.899); D['Q4_TG2'] = (4.5, 0.886)
    m['G'].clear(); m['G'].update(G); m['Dn'].clear(); m['Dn'].update(D)
    _, out = alloc(U['mix 50/50'], 19.57)
    s = {t: score(out, U[t]) for t in ('WikiText', 'chat+code')}
    print(f'{label:28s} file 22.65 GB, ranked on the 50/50 mix: WikiText {s["WikiText"]:.3f} -> KL {0.0051 + 0.0078 * s["WikiText"]:.4f}; '
          f'chat+code {s["chat+code"]:.3f} -> KL {0.0051 + 0.0078 * s["chat+code"]:.4f}')
