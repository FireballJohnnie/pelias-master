#!/usr/bin/env python3
# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# provenance: original
"""--block-weights for `pelias planes --select _exps` (docs/PELIAS_RESIDUAL_PLANES.md 6.7): the builder scores a block by the
importance-weighted weight error it removes, relative to its matrix (the 'wrel' measure); this file turns that into relative
error at the expert's output through SwiGLU ('orel', tools/analysis/tier_slopes.py), per layer and class:
factor = mean over the sampled experts of orel / wrel at Q4_TG2 (the tiered file's floor), scaled so the mean factor is 1.
  plane_block_weights.py TIER_SLOPES.npz OUT.tsv      (written only when OUT.tsv does not exist)"""
import os, sys
import numpy as np

if len(sys.argv) != 3:
    sys.exit(__doc__)
src, out = sys.argv[1:]
if os.path.exists(out):
    sys.exit(f'{out} exists (nothing is overwritten)')
d = np.load(src)
t = list(d['types']).index('Q4_TG2')
ratio = (d['orel'][:, :, :, t] / d['wrel'][:, :, :, t]).mean(1)  # (layers, 3)
ratio /= ratio.mean()
with open(out, 'x') as f:
    for l in range(ratio.shape[0]):
        for m, c in enumerate(('gate', 'up', 'down')):
            f.write(f'blk.{l}.ffn_{c}_exps.weight\t{ratio[l, m]:.6f}\n')
print(f'{out}: {ratio.size} weights, factor {ratio.min():.3f} to {ratio.max():.3f}')
