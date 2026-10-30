#!/usr/bin/env python3
"""Per-chunk KL of the tiered file vs the Unsloth file (and Unsloth + 2 planes), from the --dump-kld files (36-byte
header, 24-byte records, KL as f64 at offset 0; 100 chunks x 255 tokens). Author 08:16: chunks 79-100 'jump'."""
import numpy as np
U = '/mnt/user-data/uploads/pelias-tests/gp/'
def load(p):
    b = open(U + p, 'rb').read()
    return np.frombuffer(b, dtype=np.dtype([('kl', '<f8'), ('x', 'V16')]), offset=36)['kl'].reshape(100, 255)
t, u, p2 = load('TIER/tier_v1_x8.kld'), load('PL/unsloth_x8_p0_h0.kld'), load('PL/unsloth_x8_p2_h0.kld')
ct, cu, cp = t.mean(1), u.mean(1), p2.mean(1)
for lo, hi in ((0, 50), (50, 78), (78, 100)):
    s = slice(lo, hi)
    print(f'chunks {lo+1}-{hi}: tier {ct[s].mean():.6f}  unsloth {cu[s].mean():.6f}  ({(ct[s].mean()/cu[s].mean()-1)*100:+.1f} %)  unsloth+2 planes {cp[s].mean():.6f}')
print('chunk  tier      unsloth   tier/unsloth  unsloth+2p   (chunks 70-100)')
for c in range(69, 100):
    print(f'{c+1:5d}  {ct[c]:.6f}  {cu[c]:.6f}  {ct[c]/cu[c]:6.2f}      {cp[c]:.6f}')
print(f'chunks where the tier is lower: {(ct < cu).sum()} of 100; ratio tier/unsloth by chunk: median {np.median(ct/cu):.2f}')
top = np.argsort(-ct)[:8] + 1
print('8 highest chunks for the tier:', top.tolist(), ' for unsloth:', (np.argsort(-cu)[:8] + 1).tolist())
