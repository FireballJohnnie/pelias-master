#!/usr/bin/env python3
"""C1, the H gate on KV pages (author's H gate; Fable's form): does a 1-bit code of each key find the keys that carry the
attention mass? Input: a PELQK001 record (pelias kld --dump-qk), every position's q and k heads in every full-attention
layer. For each query position t >= T0 and head, exact attention p = softmax(q.k / sqrt(d)) over keys u <= t; then keep a
fraction f of the keys (1/8, 1/32) chosen by:
  oracle  the f largest p (upper bound)
  1bit    Fable's scaled 1-bit dot: a fixed random sign diagonal and a Walsh-Hadamard per 32 on q and k (q.k unchanged),
          then sum over 32-blocks of mean|q~|_b * mean|k~|_b * (32 - 2 * hamming(sign q~_b, sign k~_b))  (40 B per key)
  quest   page bound (Quest): pages of 16 keys, per-dim min/max per page, bound = sum_i max(q_i min_i, q_i max_i); best pages
  recent  the most recent f of the keys (a sliding window, the cheap baseline)
Recall = attention mass inside the kept keys. Usage: c1_recall_check.py QK.bin [T0]"""
import struct, sys
import numpy as np
path = sys.argv[1]; T0 = int(sys.argv[2]) if len(sys.argv) > 2 else 64
b = open(path, 'rb').read()
assert b[:8] == b'PELQK001'
n_ctx, chunks, na, nh, nkv, hd = struct.unpack_from('<6I', b, 8)
layers = struct.unpack_from('<%dI' % na, b, 40)
off = 40 + 4 * na; row = (nh + nkv) * hd
data = np.frombuffer(b, np.float16, offset=off).reshape(chunks, na, n_ctx, row)
group = nh // nkv; scale = 1 / np.sqrt(hd)
H = np.array([[1.]])
for _ in range(5): H = np.block([[H, H], [H, -H]])
H /= np.sqrt(32)
D = np.where(np.random.default_rng(1234).integers(0, 2, hd) == 1, -1.0, 1.0)
def code(X):  # (n, hd) -> sign matrix scaled per block, so A @ B.T gives the scaled 1-bit dot
    Y = (X * D).reshape(len(X), hd // 32, 32) @ H.T
    s = np.abs(Y).mean(2, keepdims=True)
    return (np.sign(Y) * s).reshape(len(X), hd)
FR = (8, 32); PAGE = 16
res = {(m, f): [] for m in ('oracle', '1bit', 'quest', 'recent') for f in FR}
per_layer = {}
causal = np.tril(np.ones((n_ctx, n_ctx), bool))
for c in range(chunks):
    for a in range(na):
        Q = data[c, a, :, :nh * hd].astype(np.float64).reshape(n_ctx, nh, hd)
        K = data[c, a, :, nh * hd:].astype(np.float64).reshape(n_ctx, nkv, hd)
        for h in range(nh):
            q = Q[:, h]; k = K[:, h // group]
            S = np.where(causal, q @ k.T * scale, -np.inf); P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            A = np.where(causal, code(q) @ code(k).T, -np.inf)
            npg = n_ctx // PAGE; kp = k[:npg * PAGE].reshape(npg, PAGE, hd); kmin, kmax = kp.min(1), kp.max(1)
            QB = np.maximum(q[:, None, :] * kmin[None], q[:, None, :] * kmax[None]).sum(2)  # (n, pages)
            for t in range(T0, n_ctx):
                p = P[t, :t + 1]
                for f in FR:
                    m = max(1, (t + 1) // f)
                    res[('oracle', f)].append(np.sort(p)[-m:].sum())
                    res[('1bit', f)].append(p[np.argpartition(A[t, :t + 1], -m)[-m:]].sum())
                    res[('recent', f)].append(p[-m:].sum())
                    full = (t + 1) // PAGE; mp = max(1, -(-m // PAGE))  # pages wholly visible; the open page is always kept
                    keep = np.zeros(t + 1, bool); keep[full * PAGE:] = True
                    if full: keep[(np.argsort(-QB[t, :full])[:mp][:, None] * PAGE + np.arange(PAGE)).ravel()] = True
                    res[('quest', f)].append(p[keep].sum())
            for f in FR:
                per_layer.setdefault((layers[a], f), []).append(np.mean(res[('1bit', f)][-(n_ctx - T0):]))
print('PELQK001: %d chunks x %d positions, %d full-attention layers %s, %d q heads, %d kv heads, head dim %d; queries t >= %d'
      % (chunks, n_ctx, na, list(layers), nh, nkv, hd, T0))
print('%-8s %12s %12s' % ('method', 'keep 1/8', 'keep 1/32'))
for m in ('oracle', '1bit', 'quest', 'recent'):
    print('%-8s %11.1f%% %11.1f%%' % (m, 100 * np.mean(res[(m, 8)]), 100 * np.mean(res[(m, 32)])))
print('1bit recall by layer (1/8, 1/32): ' + ', '.join('L%d %.1f/%.1f' % (L, 100 * np.mean(per_layer[(L, 8)]), 100 * np.mean(per_layer[(L, 32)])) for L in layers))
print('1bit worst 5 %% of queries at 1/32: %.1f%%' % (100 * np.quantile(res[('1bit', 32)], 0.05)))
