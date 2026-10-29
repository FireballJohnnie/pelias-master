#!/usr/bin/env python3
"""Author's question (2026-10-29 00:20): which tensors were never tested rotated, and would rotation help them?
urm_ZT2's own list (pelias.rot32.tensors): 257 of 259 Q7_ZT tensors rotated; token_embd and blk.40.nextn.eh_proj not.
This reads the bf16 source GGUF (read-only), samples rows, and compares per-32 Walsh-Hadamard rotated vs not:
  crest = mean per-block amax/rms (drives absmax-scale error), kurt = mean per-block E[x^4]/E[x^2]^2 (3 = Gaussian),
  q8 = 8-bit per-32 absmax error, q7 = pure 7-bit, q7zt = 7-bit + light 8th bit on the best 2 of 8 blocks per 256
  (the same proxy as light_heavy_bit_check.py; per-32 exact scales, not Q7_ZT's fp16+6-bit scales). Plain relative
  squared error, no importance weights. Usage: rot_coverage_check.py SOURCE_BF16.gguf"""
import struct, sys
import numpy as np

f = open(sys.argv[1], 'rb')
rd = lambda fmt: struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
def skip(t):
    S = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in S: f.seek(S[t], 1)
    elif t == 8: n, = rd('Q'); f.seek(n, 1)
    elif t == 9:
        at, n = rd('IQ')
        for _ in range(n): skip(at)
assert f.read(4) == b'GGUF'
ver, nt, nkv = rd('IQQ'); align = 32
for _ in range(nkv):
    kl, = rd('Q'); k = f.read(kl).decode(); t, = rd('I')
    if k == 'general.alignment' and t == 4: align, = rd('I')
    else: skip(t)
info = {}
for _ in range(nt):
    nl, = rd('Q'); name = f.read(nl).decode(); nd, = rd('I'); ne = list(rd('Q' * nd)); ty, off = rd('IQ'); info[name] = (ne, ty, off)
data0 = (f.tell() + align - 1) // align * align

def rows(name, want):
    ne, ty, off = info[name]
    assert ty == 30, (name, ty)
    rl = ne[0]; nrows = int(np.prod(ne[1:]))
    idx = np.unique(np.linspace(0, nrows - 1, min(want, nrows)).astype(np.int64))
    out = np.empty((len(idx), rl), np.float32)
    for i, r in enumerate(idx):
        f.seek(data0 + off + int(r) * rl * 2)
        out[i] = (np.frombuffer(f.read(rl * 2), np.uint16).astype(np.uint32) << 16).view(np.float32)
    return out.reshape(-1, 32).astype(np.float64), ne

H = np.array([[1.]])
for _ in range(5): H = np.block([[H, H], [H, -H]])
H /= np.sqrt(32)

def stats(X):
    a = np.abs(X).max(1, keepdims=True); a[a == 0] = 1
    ms = (X ** 2).mean(1); ok = ms > 0
    crest = (a[:, 0][ok] / np.sqrt(ms[ok])).mean()
    kurt = ((X[ok] ** 4).mean(1) / ms[ok] ** 2).mean()
    u = X / (a / 127)
    e = lambda q: ((X - q * (a / 127)) ** 2).sum(1)
    E8 = e(np.clip(np.round(u), -127, 127)); E7 = e(np.clip(2 * np.floor(u / 2) + 1, -127, 127))
    n = len(E8) // 8 * 8; E8, E7 = E8[:n], E7[:n]
    gain = (E7 - E8).reshape(-1, 8); top = -np.sort(-gain, 1)[:, :2].sum()
    ref = (X[:n] ** 2).sum()
    return crest, kurt, E8.sum() / ref, E7.sum() / ref, (E7.sum() - top) / ref

TARGETS = [('token_embd.weight', 16384), ('blk.40.nextn.eh_proj.weight', 2048), ('blk.20.attn_qkv.weight', 1024)]
for L in (5, 20, 35):
    for k in ('gate', 'up', 'down'): TARGETS.append(('blk.%d.ffn_%s_exps.weight' % (L, k), 2048))
print('%-30s %-16s %6s | %-6s %6s %6s %9s %9s %9s | rot vs not (q7zt)' % ('tensor', 'shape', 'rows', 'basis', 'crest', 'kurt', 'q8', 'q7', 'q7zt'))
for name, want in TARGETS:
    if name not in info: print(name, 'not in source'); continue
    X, ne = rows(name, want)
    r = [stats(X), stats(X @ H.T)]
    for lab, s in zip(('plain', 'rot32'), r):
        tail = '%+.1f %%' % (100 * (r[1][4] / r[0][4] - 1)) if lab == 'rot32' else ''
        print('%-30s %-16s %6d | %-6s %6.2f %6.2f %9.3e %9.3e %9.3e | %s' % (name, 'x'.join(map(str, ne)), len(X) * 32 // ne[0], lab, *s, tail))
    sys.stdout.flush()
