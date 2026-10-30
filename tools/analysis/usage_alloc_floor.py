#!/usr/bin/env python3
"""usage_alloc.py with a floor: no expert below a given type (cold experts on other kinds of text are unmeasured)."""
import runpy, sys, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path('usage_alloc.py')
GATE, DOWN, run, ref = g['GATE'], g['DOWN'], g['run'], g['ref']
order = ['Q2_K', 'Q3_K', 'Q3_K+1p', 'IQ4_XS', 'Q4_K']
for floor in ('Q3_K', 'IQ4_XS'):
    drop = order[:order.index(floor)]
    for label, planes in (('types only', False), ('types + planes', True)):
        og = {k: v for k, v in GATE.items() if k not in drop and (planes or '+' not in k)}
        od = {k: v for k, v in DOWN.items() if k not in drop and (planes or '+' not in k)}
        print(f'\nfloor {floor}, {label}:')
        for b in (19.57, 21.0):
            tb, e1, e2, cnt = run(og, od, b)
            used = ', '.join(f'{k} {v}' for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]) if v)
            kl = 0.0051 + 0.0078 * e2 / ref
            print(f'  budget {b:5.2f} GB: error held-out {e2/ref:.3f} x Unsloth -> KL ~{kl:.4f} (DERIVED); {used}')
