#!/usr/bin/env python3
"""Author's question (2026-10-28 23:58): a 7-bit tensor plus a gated 8th bit (2 of 8 32-blocks per 256), the extra bit
LIGHT (refines the step: q = 2b + h, ungated blocks on the odd grid) or HEAVY (extends the range: base at the fine step
+-63, heavy bit adds +-64 steps; ungated blocks on the 7-bit grid at amax/63), rotated (Walsh-Hadamard per 32) or not.
Also each design when its bit is not read (light: the 7-bit base; heavy: the base clipped at +-63 fine steps).
Real blk.20 attn_qkv (bf16 slice), plain relative squared error. MEASURED 2026-10-29 00:0x:
  unrotated light 9.428e-05 (missing 1.410e-04) | heavy 9.579e-05 (missing 3.220e-02)
  rotated   light 7.860e-05 (missing 1.122e-04) | heavy 7.979e-05 (missing 2.876e-02)"""
import numpy as np
SL = '/mnt/user-data/uploads/pelias-tests/gp/M/slices/'
H = np.array([[1.]])
for _ in range(5): H = np.block([[H, H], [H, -H]])
H /= np.sqrt(32)
raw = np.fromfile(SL + 'blk.20.attn_qkv.weight.bin', dtype=np.uint16).astype(np.uint32) << 16
W = raw.view(np.float32).astype(np.float64).reshape(-1, 32)[:400000]
def run(X):
    a = np.abs(X).max(1, keepdims=True); a[a == 0] = 1
    s8 = a / 127; u = X / s8
    q8 = np.clip(np.round(u), -127, 127) * s8
    lo = np.clip(2 * np.floor(u / 2) + 1, -127, 127) * s8
    s7 = a / 63; ho = np.clip(np.round(X / s7), -63, 63) * s7
    hm = np.clip(np.round(u), -63, 63) * s8
    e = lambda q: ((X - q) ** 2).sum(1)
    E8, EL, EH, EHm = e(q8), e(lo), e(ho), e(hm)
    ref = (X ** 2).sum(); out = {}
    for name, Eu, Em in (('light', EL, EL), ('heavy', EH, EHm)):
        gain = (Eu - E8).reshape(-1, 8); idx = np.argsort(-gain, 1)[:, :2]
        tot = Eu.sum() - np.take_along_axis(gain, idx, 1).sum()
        mask = np.zeros_like(gain, bool); np.put_along_axis(mask, idx, True, 1)
        out[name] = (tot / ref, np.where(mask, Em.reshape(-1, 8), Eu.reshape(-1, 8)).sum() / ref)
    return out
for lab, X in (('unrotated', W), ('rotated', W @ H.T)):
    o = run(X)
    print(f"{lab:9s} light {o['light'][0]:.3e} (missing {o['light'][1]:.3e}) | heavy {o['heavy'][0]:.3e} (missing {o['heavy'][1]:.3e})")
