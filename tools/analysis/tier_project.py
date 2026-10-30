#!/usr/bin/env python3
"""Projections from tier_slopes_v2.npz (DERIVED). Units: SwiGLU output error relative to the expert's own output energy
(orel; 'abs' variant: oabs), per layer and class, mean of experts 0-1, applied to all 256 experts of that layer.
Usage = picks share per layer (50/50 WikiText + chat+code mix for ranking). Error total relative to Unsloth's experts
(gate/up Q4_K; down Q5_K, Q6_K on layers 34, 38, 39) = 1, mapped to KL by the two-point line (KL = 0.0051 + 0.0078 x).
1. static: one type per (layer, expert, class), floor Q4_TG2, experts 19.57 GB (file 22.65 GB).
2. runtime: the static base + one residual plane on the 256-blocks with the largest error removed per byte, chosen by
   weight error and layer slope only (no text), read on every use ('always') or only for the context-hot top 32 experts
   (55.2 % of routing weight on WikiText after 32 tokens, context_planes_v1.txt; applied as a flat factor: ASSUMPTION)."""
import sys, numpy as np
d = np.load('tier_slopes_v2.npz'); u = np.load('domain_usage_v1.npz')
U = {'WikiText': u['wiki'], 'chat+code': u['chatcode']}; MIX = (U['WikiText'] + U['chat+code']) / 2
L, E = MIX.shape; T = list(d['types']); BPW = {'Q4_K': 4.5, 'Q4_TG2': 4.5, 'Q5_K': 5.5, 'Q6_K': 6.5625, 'Q8_0': 8.5}
NW = 2048 * 512; GB = lambda bpw: bpw * NW / 8e9; NBLK = NW // 256; PLB = 34  # bytes per plane block
def run(metric):
    S = d[metric].mean(1)  # (L, 3 classes, types)
    unsl = np.zeros((L, 3), int); unsl[:, 0] = unsl[:, 1] = T.index('Q4_K'); unsl[:, 2] = T.index('Q5_K'); unsl[[34, 38, 39], 2] = T.index('Q6_K')
    ref = {k: (U[k][:, :, None] * S[np.arange(L)[:, None], np.arange(3)[None, :], unsl][:, None, :]).sum() for k in U}
    unsl_gb = sum(GB(BPW[T[unsl[l, m]]]) * E for l in range(L) for m in range(3))
    opts = [T.index(t) for t in ('Q4_TG2', 'Q5_K', 'Q6_K', 'Q8_0')]; bp = np.array([GB(BPW[T[i]]) for i in opts])
    def alloc(budget):
        def pick(lam):
            cost = MIX[:, :, None, None] * S[:, None, :, :][..., opts] + lam * bp
            j = cost.argmin(-1); return bp[j].sum(), j
        lo, hi = 0.0, 1e3
        for _ in range(80):
            mid = (lo + hi) / 2; lo, hi = (mid, hi) if pick(mid)[0] > budget else (lo, mid)
        return pick(hi)
    gb, j = alloc(unsl_gb)
    tsel = np.array(opts)[j]  # (L, E, 3) type index
    perr = S[np.arange(L)[:, None, None], np.arange(3)[None, None, :], tsel]  # (L, E, 3) error per pick
    score = {k: (U[k][:, :, None] * perr).sum() / ref[k] for k in U}
    print(f'[{metric}] Unsloth experts {unsl_gb:.2f} GB; static tiered {gb:.2f} GB: ' + '; '.join(f'{k} {v:.3f} -> KL {0.0051 + 0.0078 * v:.4f}' for k, v in score.items()))
    cnt = {(m, T[i]): int((tsel[..., m] == i).sum()) for m in range(3) for i in opts}
    print('   types chosen (of 10,240 per class): ' + ', '.join(f'{["gate", "up", "down"][m]} {t} {n}' for (m, t), n in cnt.items() if n))
    byl = [(BPW_l := np.mean([BPW[T[i]] for i in tsel[l].ravel()])) for l in range(L)]
    print('   mean bits per weight by layer: ' + ' '.join(f'{b:.2f}' for b in byl))
    # block planes: per (layer, class, type) the reduction of each block as a share of that matrix's error, x the slope
    red = {}
    for l in range(L):
        for m in range(3):
            for i in opts:
                arr = []
                for e in range(2):
                    bb, ba = d[f'blk_{l}_{e}_{m}_{i}']; arr.append((bb - ba) / bb.sum())
                red[(l, m, i)] = np.sort(np.concatenate(arr))[::-1] * S[l, m, i]  # per block, both experts pooled (2x blocks)
    for side_gb in (0.5, 1.0, 2.0, 4.28):
        # threshold tau on reduction per block (block bytes equal): count blocks over all experts above tau, by type used
        def size_and_removed(tau):
            nb = 0.0; rem = np.zeros((L, E, 3))
            for (l, m, i), r in red.items():
                k = np.searchsorted(-r, -tau)  # blocks above tau in the pooled sample of 2 experts
                frac = k / len(r); mask = tsel[l, :, m] == i
                nb += frac * NBLK * mask.sum(); rem[l, mask, m] = r[:k].sum() / 2  # per expert (pooled over 2)
            return nb * PLB / 1e9, rem
        lo, hi = 0.0, max(r[0] for r in red.values())
        for _ in range(60):
            mid = (lo + hi) / 2; lo, hi = (mid, hi) if size_and_removed(mid)[0] > side_gb else (lo, mid)
        sz, rem = size_and_removed(hi)
        for mode, f in (('always', 1.0), ('context-hot', 0.552)):
            sc = {k: (U[k][:, :, None] * (perr - f * rem)).sum() / ref[k] for k in U}
            print(f'   + block planes {sz:.2f} GB ({mode}): ' + '; '.join(f'{k} {v:.3f} -> KL {0.0051 + 0.0078 * v:.4f}' for k, v in sc.items()))
for metric in ('orel', 'oabs', 'wrel'):
    run(metric)
