#!/usr/bin/env python3
"""Author 09:33: 1-bit planes lined up as 4-bit / 8-bit integer arrays for the tensor cores, plus a 1-bit plane run against each
aligned block. Sandbox, blk.20 experts 0-3 (gate, down), error as in planes_bits_check.py.
  free-k     : k planes with free per-block scales (binary-coded quantization; needs one 1-bit pass per plane)
  locked-k   : k planes whose scales are locked to s, s/2, s/4, ... (+ one fp16 offset per 256): the k planes are then exactly
               the bits of one k-bit integer, so the int4 / int8 tensor cores read them as one array (k = 4: 4 + 0.125 bpw)
  locked-k + f free: the locked integer core + f free planes on the 1-bit path, each over the same 256-value block (the b1
               tensor-core tile is k = 256: one plane block = one tile)"""
import sys, numpy as np
sys.path.insert(0, '.')
_src = open('planes_bits_check.py').read()
exec(_src[:_src.index("print('== (1)")])  # the definitions only (planes, imp_of, err, SL): the builder-equivalent fit
B = 256
f16 = lambda s: s.astype(np.float16).astype(np.float64)
def locked(w, imp, k):  # value = m + s * q, q in 0 .. 2^k - 1, per 256 block; s, m by weighted least squares over a few rounds
    rows, n = w.shape; W = w.reshape(rows, n // B, B); I = np.broadcast_to(imp.reshape(1, n // B, B), W.shape)
    lo, hi = W.min(-1, keepdims=True), W.max(-1, keepdims=True); L = 2 ** k - 1
    s = (hi - lo) / L; s = np.where(s > 0, s, 1e-12); m = lo
    for _ in range(6):
        q = np.clip(np.rint((W - m) / s), 0, L)
        Sw, Sq, Sqq = I.sum(-1, keepdims=True), (I * q).sum(-1, keepdims=True), (I * q * q).sum(-1, keepdims=True)
        Sx, Sxq = (I * W).sum(-1, keepdims=True), (I * W * q).sum(-1, keepdims=True)
        det = Sw * Sqq - Sq * Sq; ok = np.abs(det) > 1e-30
        s = np.where(ok, (Sw * Sxq - Sq * Sx) / np.where(ok, det, 1), s); m = np.where(ok, (Sx - s * Sq) / Sw, m)
        s, m = f16(s), f16(m); s = np.where(s > 0, s, 1e-12)
    q = np.clip(np.rint((W - m) / s), 0, L)
    return (m + s * q).reshape(rows, n)
if __name__ == '__main__':
    res = {}
    for name, n_in, n_out in (('blk.20.ffn_gate_exps.weight', 2048, 512), ('blk.20.ffn_down_exps.weight', 512, 2048)):
        raw = (np.fromfile(f'{SL}/{name}.bin', dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(4, n_out, n_in)
        for x in range(4):
            w = raw[x]; imp = imp_of(name, n_in, x); I = imp[None, :]
            def add(key, bpw, e): res.setdefault(key, [bpw, []])[1].append(e)
            for k in (2, 3, 4):
                add(f'free-{k} (BCQ)', 1.0625 * k, err(w, planes(w, imp, k), I))
                base = locked(w, imp, k); add(f'locked-{k} (int{k} core)', k + 0.125, err(w, base, I))
                for f in (1, 2):
                    add(f'locked-{k} + {f} free plane(s)', k + 0.125 + 1.0625 * f, err(w, base + planes(w - base, imp, f), I))
    for key, (bpw, es) in sorted(res.items(), key=lambda kv: kv[1][0]):
        print(f'  {key:30s} {bpw:6.3f} bpw  error {np.mean(es):.3e}')
    print('  reference (planes_bits_v1.txt): Q2_K 6.593e-02 @2.625, Q3_K 2.018e-02 @3.438, Q4_K 4.684e-03 @4.5, Q4_TG2 4.182e-03 @4.5, Q5_K 1.187e-03 @5.5')
