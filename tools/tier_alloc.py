#!/usr/bin/env python3
# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# provenance: original
"""Usage-tiered expert allocation for `pelias quantize --tier` (docs/PELIAS_RESIDUAL_PLANES.md 6.3-6.7).

  tier_alloc.py SLOPES.npz USAGE.npz OUT.tsv [--budget-bytes N] [--floor Q4_TG2]

Every routed expert of every trunk layer gets its own storage type for gate, up and down. The choice minimises the
usage-weighted error at a fixed byte budget (a Lagrangian per expert: error x usage + lambda x bytes; exact per item):
  error  per layer, class and type = the SwiGLU output error relative to the expert's own output ('orel' of
         tools/analysis/tier_slopes.py, the mean of the sampled experts of that layer);
  usage  per layer and expert = the 50/50 mix of the two pick shares in USAGE.npz ('wiki', 'chatcode';
         tools/analysis/domain_stability.py);
  budget the experts' bytes in the Unsloth UD-Q4_K_M file (gate and up Q4_K; down Q5_K, Q6_K on layers 34, 38 and 39),
         so the tiered file is no larger. --budget-bytes overrides it.
Types offered: the floor (default Q4_TG2), Q5_K, Q6_K, Q8_0.
OUT.tsv: one line per tensor, `blk.L.ffn_{gate,up,down}_exps.weight<TAB>TYPE,TYPE,...` (one type per expert, in expert
order). Written only when OUT.tsv does not exist (nothing is overwritten). Standard library + numpy.
"""
import argparse, os, sys
import numpy as np

BPW = {'Q4_K': 4.5, 'Q4_TG2': 4.5, 'Q5_K': 5.5, 'Q6_K': 6.5625, 'Q8_0': 8.5}
CLASSES = ('gate', 'up', 'down')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('slopes')
    ap.add_argument('usage')
    ap.add_argument('out')
    ap.add_argument('--budget-bytes', type=int, default=0)
    ap.add_argument('--floor', default='Q4_TG2')
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f'{a.out} exists (nothing is overwritten)')
    d = np.load(a.slopes)
    u = np.load(a.usage)
    types = [str(t) for t in d['types']]
    S = d['orel'].mean(1)  # (L, 3, types)
    mix = (u['wiki'] + u['chatcode']) / 2
    L, E = mix.shape
    ne0, ne1 = 2048, 512  # weights per expert matrix: 2048 x 512 (gate, up) and 512 x 2048 (down)
    opts = [a.floor, 'Q5_K', 'Q6_K', 'Q8_0']
    for t in opts:
        if t not in types:
            sys.exit(f'{t}: no slopes for this type in {a.slopes}')
    oi = np.array([types.index(t) for t in opts])
    nbytes = np.array([BPW[t] * ne0 * ne1 / 8 for t in opts])
    if a.budget_bytes:
        budget = float(a.budget_bytes)
    else:
        down = ['Q6_K' if l in (34, 38, 39) else 'Q5_K' for l in range(L)]
        budget = sum((2 * BPW['Q4_K'] + BPW[down[l]]) * ne0 * ne1 / 8 * E for l in range(L))
    err = S[:, :, oi]  # (L, 3, opts)

    def pick(lam):
        j = (mix[:, :, None, None] * err[:, None, :, :] + lam * nbytes / 1e9).argmin(-1)  # (L, E, 3)
        return nbytes[j].sum(), j

    lo, hi = 0.0, 1e3
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pick(mid)[0] > budget else (lo, mid)
    total, j = pick(hi)
    with open(a.out, 'x') as f:
        for l in range(L):
            for m, c in enumerate(CLASSES):
                f.write(f'blk.{l}.ffn_{c}_exps.weight\t' + ','.join(opts[k] for k in j[l, :, m]) + '\n')
    counts = {(c, t): int((j[..., m] == k).sum()) for m, c in enumerate(CLASSES) for k, t in enumerate(opts)}
    print(f'experts: {total:,.0f} bytes (budget {budget:,.0f}); ' + ', '.join(f'{c} {t} {n}' for (c, t), n in counts.items() if n))


if __name__ == '__main__':
    main()
