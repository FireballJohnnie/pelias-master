#!/usr/bin/env python3
"""Lower-bit bases + residual planes vs the stock types, on the real blk.20 expert slices (experts 0-3, bf16), each expert
with its own importance; scale per 256, refit. Bits per weight include the base type and the planes (1.0625 per plane).
Question: at a fixed size, which mix gives the lowest weighted error, and what does a thinner base + planes buy."""
import os, sys, subprocess, itertools, numpy as np
sys.path.insert(0, '/home/claude/pelias-master/tools'); from imatrix_merge import Gguf
HERE = os.path.dirname(os.path.abspath(__file__)); KQ = HERE + '/kqw'
SL = '/mnt/user-data/uploads/pelias-tests/gp/M/slices'; IM = '/tmp/claude-0/-home-claude/c7ff3710-2054-5bef-9957-5760a8e25c56/scratchpad/urm/merged.gguf'
g = Gguf(IM); E = g.entries(); B = 256
f16 = lambda s: s.astype(np.float16).astype(np.float64)
def kq(tid, x, wt):
    a, b, c = HERE + '/in.f32', HERE + '/out.f32', HERE + '/imp.f32'
    x.astype(np.float32).tofile(a); wt.astype(np.float32).tofile(c)
    r = subprocess.run([KQ, str(tid), a, b, str(x.shape[1]), c], capture_output=True, text=True)
    if r.returncode != 0: return None, None
    return np.fromfile(b, dtype=np.float32).reshape(x.shape).astype(np.float64), float(r.stdout.split(',')[1].split()[0])
def planes(r, iw, K, rounds=8):
    S, Sg = [], []; rr = r.copy()
    for _ in range(K):
        sg = np.where(rr >= 0, 1.0, -1.0); s = f16((iw * np.abs(rr)).sum(-1, keepdims=True) / iw.sum(-1, keepdims=True)); S.append(s); Sg.append(sg); rr = rr - s * sg
    if K >= 2:
        pats = np.array(list(itertools.product((-1.0, 1.0), repeat=K)))
        for _ in range(rounds):
            Bm = np.stack(Sg, -1); A = np.einsum('rbi,rbik,rbil->rbkl', iw, Bm, Bm) + 1e-30 * np.eye(K)
            y = np.einsum('rbi,rbik,rbi->rbk', iw, Bm, r); s = f16(np.linalg.solve(A, y[..., None])[..., 0])
            vals = np.einsum('rbk,pk->rbp', s, pats); P = pats[np.abs(r[..., None] - vals[:, :, None, :]).argmin(-1)]
            Sg = [P[..., k] for k in range(K)]; S = [s[..., k:k + 1] for k in range(K)]
    return sum(S[k] * Sg[k] for k in range(K))
TYPES = [(10, 'Q2_K'), (11, 'Q3_K'), (12, 'Q4_K'), (13, 'Q5_K'), (14, 'Q6_K'), (23, 'IQ4_XS'), (22, 'IQ3_S'), (8, 'Q8_0')]
for name, n_in, n_out in (('blk.20.ffn_gate_exps.weight', 2048, 512), ('blk.20.ffn_down_exps.weight', 512, 2048)):
    raw = np.fromfile(f'{SL}/{name}.bin', dtype=np.uint16).astype(np.uint32) << 16
    W = raw.view(np.float32).astype(np.float64).reshape(4, n_out, n_in); e = E[name]
    s2 = np.frombuffer(g.read(e['.in_sum2'][2], e['.in_sum2'][3]), dtype=np.float32).astype(np.float64).reshape(-1, n_in)
    cnt = np.frombuffer(g.read(e['.counts'][2], e['.counts'][3]), dtype=np.float32).astype(np.float64)
    res = {}; den = 0.0
    for x in range(4):
        w = W[x]; imp = s2[x] / max(cnt[x], 1.0)
        if imp.sum() == 0: imp = np.ones(n_in)
        den += ((w ** 2) * imp).sum()
        iw = np.broadcast_to(imp, w.shape).reshape(n_out, -1, B)
        for tid, tn in TYPES:
            q, bpw = kq(tid, w, imp)
            if q is None: continue
            r = (w - q).reshape(n_out, -1, B)
            for K in (0, 1, 2):
                err = (((w - q) - (planes(r, iw, K).reshape(w.shape) if K else 0)) ** 2 * imp).sum()
                key = (tn, K); res.setdefault(key, [0.0, bpw + K * (1 + 16 / B)]); res[key][0] += err
    print(f'\n{name} (experts 0-3), weighted error relative to the weights; sorted by bits per weight')
    ref = res[('Q4_K', 0)][0] / den
    for (tn, K), (er, bpw) in sorted(res.items(), key=lambda kv: kv[1][1]):
        print(f'  {tn:7s} + {K} plane{"s" if K != 1 else " "}  {bpw:5.2f} bpw  error {er/den:.6f}  ({er/den/ref:.3f} x Q4_K)')
