#!/usr/bin/env python3
"""Author 09:24-09:30: (1) 7-bit + a plane against the 8-bit read (the gated 8th bit, Q8_Z, Q8_0), on the rotated dense
weights (blk.20 attn_qkv, 2048 of 8192 rows, rotated per 32 as the file stores it, importance in the rotated basis);
(2) coarse bases + planes on the experts (blk.20 gate + down, experts 0-3); (3) a pure 1-bit base (one sign plane fitted to
the weights themselves) + 1, 2, 3 more planes, i.e. binary-coded quantization grown plane by plane.
Error = sum imp (w - wq)^2 / sum imp w^2. Bits per weight include the planes (1.0625 each: 16-bit scale + 256 signs)."""
import os, sys, subprocess, itertools, numpy as np
sys.path.insert(0, '/home/claude/pelias-master/tools'); from imatrix_merge import Gguf
HERE = os.path.dirname(os.path.abspath(__file__)); B = 256
SL = '/mnt/user-data/uploads/pelias-tests/gp/M/slices'
g = Gguf('/tmp/claude-0/-home-claude/c7ff3710-2054-5bef-9957-5760a8e25c56/scratchpad/urm/merged.gguf'); E = g.entries()
f16 = lambda s: s.astype(np.float16).astype(np.float64)
def kq(tid, w, im):
    r = subprocess.run([HERE + '/kqwp', str(tid), str(w.shape[0]), str(w.shape[1])], input=w.astype(np.float32).tobytes() + im.astype(np.float32).tobytes(), capture_output=True)
    if r.returncode != 0: return None, None
    return np.frombuffer(r.stdout, dtype=np.float32).reshape(w.shape).astype(np.float64), float(r.stderr.decode().strip())
def planes(r, iw, K, rounds=8):  # K planes on residual r (rows x n), importance iw per value; the builder's fit
    rows, n = r.shape; R = r.reshape(rows, n // B, B); I = np.broadcast_to(iw.reshape(1, n // B, B) if iw.ndim == 1 else iw.reshape(rows, n // B, B), R.shape)
    S, Sg = [], []; rr = R.copy()
    for _ in range(K):
        sg = np.where(rr >= 0, 1.0, -1.0); s = f16((I * np.abs(rr)).sum(-1, keepdims=True) / np.maximum(I.sum(-1, keepdims=True), 1e-30)); S.append(s); Sg.append(sg); rr = rr - s * sg
    if K >= 2:
        pats = np.array(list(itertools.product((-1.0, 1.0), repeat=K)))
        for _ in range(rounds):
            Bm = np.stack(Sg, -1); A = np.einsum('rbi,rbik,rbil->rbkl', I, Bm, Bm) + 1e-30 * np.eye(K)
            y = np.einsum('rbi,rbik,rbi->rbk', I, Bm, R); s = f16(np.linalg.solve(A, y[..., None])[..., 0])
            vals = np.einsum('rbk,pk->rbp', s, pats); P = pats[np.abs(R[..., None] - vals[:, :, None, :]).argmin(-1)]
            Sg = [P[..., k] for k in range(K)]; S = [s[..., k:k + 1] for k in range(K)]
    return sum(S[k] * Sg[k] for k in range(K)).reshape(rows, n)
def had32(x):  # orthonormal Walsh-Hadamard per 32 along the rows
    y = x.reshape(x.shape[0], -1, 32).copy(); h = 1
    while h < 32:
        a = y.reshape(y.shape[0], y.shape[1], -1, 2 * h); u, v = a[..., :h].copy(), a[..., h:2 * h].copy()
        a[..., :h], a[..., h:2 * h] = u + v, u - v; y = a.reshape(y.shape); h *= 2
    return (y / np.sqrt(32)).reshape(x.shape)
def err(w, wq, I): return float((I * (w - wq) ** 2).sum() / (I * w * w).sum())
def imp_of(name, n_in, x=None):
    e = E[name]; s2 = np.frombuffer(g.read(e['.in_sum2'][2], e['.in_sum2'][3]), dtype=np.float32).astype(np.float64).reshape(-1, n_in)
    cnt = np.frombuffer(g.read(e['.counts'][2], e['.counts'][3]), dtype=np.float32).astype(np.float64)
    i = 0 if x is None else x
    v = s2[i] / max(cnt[i], 1.0)
    return v if v.sum() > 0 else np.ones(n_in)

print('== (1) dense attn_qkv, rotated (2048 rows): 7-bit + plane vs the 8-bit reads')
raw = (np.fromfile(f'{SL}/blk.20.attn_qkv.weight.bin', dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(8192, 2048)[::4]
W = had32(raw); imp = imp_of('blk.20.attn_qkv.weight', 2048); I = imp[None, :]
rows = []
for tid, name in ((201, 'Q7_Z (7-bit, no 8th bit)'), (202, 'Q7_ZT (7-bit + gated 8th bit)'), (200, 'Q8_Z (8-bit, zip scales)'), (8, 'Q8_0')):
    wq, bpw = kq(tid, W, imp)
    if wq is None: print(f'  {name}: not quantizable here'); continue
    rows.append((bpw, name, err(W, wq, I)))
    if tid == 201:
        for k in (1, 2):
            p = planes(W - wq, imp, k); rows.append((bpw + 1.0625 * k, f'Q7_Z + {k} plane(s)', err(W, wq + p, I)))
    if tid == 202:
        p = planes(W - wq, imp, 1); rows.append((bpw + 1.0625, 'Q7_ZT + 1 plane', err(W, wq + p, I)))
for bpw, name, e in sorted(rows):
    print(f'  {name:34s} {bpw:6.3f} bpw  error {e:.3e}')

print('\n== (2)+(3) experts (gate and down, experts 0-3): coarse bases + planes, and a pure 1-bit base grown plane by plane')
res = {}
for name, n_in, n_out in (('blk.20.ffn_gate_exps.weight', 2048, 512), ('blk.20.ffn_down_exps.weight', 512, 2048)):
    raw = (np.fromfile(f'{SL}/{name}.bin', dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(4, n_out, n_in)
    for x in range(4):
        w = raw[x]; imp = imp_of(name, n_in, x); I = imp[None, :]
        def add(key, bpw, e): res.setdefault(key, [bpw, []])[1].append(e)
        for tid, tn in ((10, 'Q2_K'), (11, 'Q3_K'), (204, 'Q4_TG2'), (12, 'Q4_K'), (13, 'Q5_K')):
            wq, bpw = kq(tid, w, imp)
            add(f'{tn}', bpw, err(w, wq, I))
            for k in (1, 2):
                p = planes(w - wq, imp, k); add(f'{tn} + {k} plane(s)', bpw + 1.0625 * k, err(w, wq + p, I))
        for k in (1, 2, 3, 4):  # pure 1-bit base = 1 sign plane on w itself; then more planes (joint refit)
            p = planes(w, imp, k); add(f'1-bit sign base + {k - 1} plane(s)', 1.0625 * k, err(w, p, I))
for key, (bpw, es) in sorted(res.items(), key=lambda kv: kv[1][0]):
    print(f'  {key:30s} {bpw:6.3f} bpw  error {np.mean(es):.3e}')
