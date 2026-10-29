#!/usr/bin/env python3
"""KV keys at 4 bits (author 01:26-01:28: the DeltaNet-checkpoint trick, rotated int4 (+ rANS), applied to the active KV
cache). Real keys and queries from a PELQK001 record (pelias kld --dump-qk). Each key head is stored per 32 values as:
  q8_0   8-bit, fp16 scale per 32 (8.5 bits; today's KV cache)
  int4   4-bit RTN, absmax scale per 32, -7..7 (4.5 bits)
  int4r  the same after a fixed random sign diagonal + Walsh-Hadamard per 32 (the query is rotated the same way, so q.k is
         unchanged in exact arithmetic) (4.5 bits)
and the attention of every query t >= T0 is recomputed with the stored keys. Reported: key error (relative squared),
attention change (mean total variation 0.5 sum |p - p'| and KL(p || p')), and the order-0 entropy of the 4-bit codes (what
rANS would store). Values (V) are not in the record. Usage: kv_int4_check.py QK.bin [T0]"""
import struct, sys
import numpy as np
path = sys.argv[1]; T0 = int(sys.argv[2]) if len(sys.argv) > 2 else 64
b = open(path, 'rb').read(); assert b[:8] == b'PELQK001'
n_ctx, chunks, na, nh, nkv, hd = struct.unpack_from('<6I', b, 8)
off = 40 + 4 * na; row = (nh + nkv) * hd
data = np.frombuffer(b, np.float16, offset=off).reshape(chunks, na, n_ctx, row)
H = np.array([[1.]])
for _ in range(5): H = np.block([[H, H], [H, -H]])
H /= np.sqrt(32)
D = np.where(np.random.default_rng(7).integers(0, 2, hd) == 1, -1.0, 1.0)
rot = lambda X: ((X * D).reshape(*X.shape[:-1], hd // 32, 32) @ H.T).reshape(X.shape)
def store(K, kind):
    B = K.reshape(*K.shape[:-1], hd // 32, 32); a = np.abs(B).max(-1, keepdims=True); a[a == 0] = 1
    if kind == 'q8_0':
        s = (a / 127).astype(np.float16).astype(np.float64); c = np.clip(np.round(B / s), -127, 127)
    else:
        s = (a / 7).astype(np.float16).astype(np.float64); c = np.clip(np.round(B / s), -7, 7)
    return (c * s).reshape(K.shape), c
scale = 1 / np.sqrt(hd); group = nh // nkv
causal = np.tril(np.ones((n_ctx, n_ctx), bool))
res = {k: {'kerr': [], 'tv': [], 'kl': []} for k in ('q8_0', 'int4', 'int4r')}; codes = {'int4': [], 'int4r': []}
for c in range(chunks):
    for a in range(na):
        Q = data[c, a, :, :nh * hd].astype(np.float64).reshape(n_ctx, nh, hd)
        K = data[c, a, :, nh * hd:].astype(np.float64).reshape(n_ctx, nkv, hd)
        for kind in res:
            if kind == 'int4r':
                Kq, cc = store(rot(K), 'int4'); Qe = rot(Q); Ke = rot(K)
            else:
                Kq, cc = store(K, kind); Qe = Q; Ke = K
            if kind in codes: codes[kind].append(cc.astype(np.int8).ravel())
            res[kind]['kerr'].append(((Ke - Kq) ** 2).sum() / (Ke ** 2).sum())
            for h in range(nh):
                q = Qe[:, h]; k0 = Ke[:, h // group]; k1 = Kq[:, h // group]
                S0 = np.where(causal, q @ k0.T * scale, -np.inf); S1 = np.where(causal, q @ k1.T * scale, -np.inf)
                P0 = np.exp(S0 - S0.max(1, keepdims=True)); P0 /= P0.sum(1, keepdims=True)
                P1 = np.exp(S1 - S1.max(1, keepdims=True)); P1 /= P1.sum(1, keepdims=True)
                P0, P1 = P0[T0:], P1[T0:]
                res[kind]['tv'].append(0.5 * np.abs(P0 - P1).sum(1).mean())
                with np.errstate(divide='ignore', invalid='ignore'):
                    res[kind]['kl'].append(np.nansum(np.where(P0 > 0, P0 * (np.log(P0) - np.log(np.maximum(P1, 1e-300))), 0), 1).mean())
def ent(x):
    p = np.bincount(x + 8, minlength=16).astype(float); p = p[p > 0] / p.sum(); return -(p * np.log2(p)).sum()
print('PELQK001: %d chunks, %d attention layers, %d kv heads x %d; queries t >= %d' % (chunks, na, nkv, hd, T0))
print('%-6s %6s | %12s %16s %14s | %s' % ('store', 'bits', 'key error', 'attention TV', 'attention KL', 'code entropy (rANS bits per value)'))
for kind, bits in (('q8_0', 8.5), ('int4', 4.5), ('int4r', 4.5)):
    r = res[kind]; e = '%.2f of 4 (+0.5 scale)' % ent(np.concatenate(codes[kind])) if kind in codes else '-'
    print('%-6s %6.1f | %12.3e %15.3f%% %14.2e | %s' % (kind, bits, np.mean(r['kerr']), 100 * np.mean(r['tv']), np.mean(r['kl']), e))
