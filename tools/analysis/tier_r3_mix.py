#!/usr/bin/env python3
# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# provenance: original
"""The R3 one-token probe's expert types for a tiered file (r3 stream --distinct --mix FILE).
  tier_r3_mix.py ALLOC.tsv USAGE.npz {wiki,chatcode} OUT.txt [--seed N]
Per layer, 7 distinct experts drawn by their measured pick share in USAGE.npz (the 7 the GPU holds of a token's 8; drawn in
proportion to use, so hot experts come up as often as they are picked), and their gate, up and down types from ALLOC.tsv
(tools/tier_alloc.py). One line per layer: 7 triples gate,up,down. Written only when OUT.txt does not exist."""
import argparse, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('alloc')
    ap.add_argument('usage')
    ap.add_argument('which', choices=('wiki', 'chatcode'))
    ap.add_argument('out')
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f'{a.out} exists (nothing is overwritten)')
    alloc = {}
    for line in open(a.alloc):
        name, _, types = line.rstrip('\n').partition('\t')
        if name:
            alloc[name] = types.split(',')
    u = np.load(a.usage)[a.which]
    rng = np.random.default_rng(a.seed)
    bpw = {'Q4_TG2': 4.5, 'Q5_K': 5.5, 'Q6_K': 6.5625}
    lines, bits = [], []
    for l in range(u.shape[0]):
        ex = rng.choice(u.shape[1], size=7, replace=False, p=u[l] / u[l].sum())
        trip = []
        for e in ex:
            t = [alloc[f'blk.{l}.ffn_{c}_exps.weight'][e] for c in ('gate', 'up', 'down')]
            if any(x not in bpw for x in t):
                sys.exit(f'layer {l} expert {e}: a type the probe has no kernel for ({t})')
            trip.append(','.join(t))
            bits.append(sum(bpw[x] for x in t))
        lines.append(' '.join(trip))
    with open(a.out, 'x') as f:
        f.write('\n'.join(lines) + '\n')
    mb = sum(bits) * 2048 * 512 / 8 / 1e6
    print(f'{a.out}: 280 expert picks, {mb:.1f} MB (baseline Q4_TG2 gate/up + Q5_K down: {280 * 14.5 * 2048 * 512 / 8 / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
