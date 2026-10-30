#!/usr/bin/env python3
"""Companion to tier_project.py (orel metric): the static file with the side file's bytes put in it instead, and the VRAM
for the GPU's hot set (top 64 experts per layer by the 50/50 mix) plus the expert bytes read per token, by option."""
import numpy as np
d = np.load('tier_slopes_v2.npz'); u = np.load('domain_usage_v1.npz')
U = {'WikiText': u['wiki'], 'chat+code': u['chatcode']}; MIX = (U['WikiText'] + U['chat+code']) / 2
L, E = MIX.shape; T = list(d['types']); BPW = {'Q4_K': 4.5, 'Q4_TG2': 4.5, 'Q5_K': 5.5, 'Q6_K': 6.5625, 'Q8_0': 8.5}
NW = 2048 * 512; S = d['orel'].mean(1)
unsl = np.zeros((L, 3), int); unsl[:, :2] = T.index('Q4_K'); unsl[:, 2] = T.index('Q5_K'); unsl[[34, 38, 39], 2] = T.index('Q6_K')
ref = {k: (U[k][:, :, None] * S[np.arange(L)[:, None], np.arange(3)[None, :], unsl][:, None, :]).sum() for k in U}
opts = [T.index(t) for t in ('Q4_TG2', 'Q5_K', 'Q6_K', 'Q8_0')]; bpw = np.array([BPW[T[i]] for i in opts]); bp = bpw * NW / 8e9
def alloc(budget):
    def pick(lam):
        j = (MIX[:, :, None, None] * S[:, None, :, :][..., opts] + lam * bp).argmin(-1); return bp[j].sum(), j
    lo, hi = 0.0, 1e3
    for _ in range(80):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if pick(mid)[0] > budget else (lo, mid)
    return pick(hi)[1]
hot = np.argsort(-MIX, 1)[:, :64]
def report(label, bits):  # bits: (L, E, 3) bits per weight
    vram = sum(bits[l, hot[l]].sum() for l in range(L)) * NW / 8e9
    rd = {k: (U[k][:, :, None] * bits).sum(axis=(1, 2)).mean() * 8 * 40 * NW / 8e6 for k in U}
    return f'{label}: hot-64 VRAM {vram:.2f} GB; expert MB per token WikiText {rd["WikiText"]:.0f}, chat+code {rd["chat+code"]:.0f}'
ub = np.broadcast_to(np.array([[BPW[T[unsl[l, m]]] for m in range(3)] for l in range(L)])[:, None, :], (L, E, 3))
print(report('Unsloth', ub))
for extra in (0.0, 1.0, 2.0, 4.28):
    j = alloc(19.57 + extra); tsel = np.array(opts)[j]
    perr = S[np.arange(L)[:, None, None], np.arange(3)[None, None, :], tsel]
    sc = {k: (U[k][:, :, None] * perr).sum() / ref[k] for k in U}
    print(f'static {22.65 + extra:.2f} GB: WikiText {sc["WikiText"]:.3f} -> KL {0.0051 + 0.0078 * sc["WikiText"]:.4f}; chat+code {sc["chat+code"]:.3f} -> KL {0.0051 + 0.0078 * sc["chat+code"]:.4f}; ' + report('', bpw[j])[2:])
