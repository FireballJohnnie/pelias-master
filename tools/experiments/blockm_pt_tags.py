# provenance: original (Block M experiment, sandbox only)
# Copyright (c) 2026 John Lipka and the Pelias contributors
# Block M: _PT tag bits. Word-aligned packing leaves spare bits; use them as per-word scale tags (choose the best of
# 2^t scale multipliers for the weights sharing the word) vs leaving them unused. Same real slices, same error measure.
import json, subprocess, numpy as np
S = '/mnt/user-data/uploads/pelias-tests/gp/M/slices'; rng = np.random.default_rng(0)
H = np.array([[1.]])
for _ in range(5): H = np.block([[H, H], [H, -H]])
H /= np.sqrt(32)
def load(name, ne0):
    raw = np.fromfile('%s/%s.bin' % (S, name), dtype=np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32).astype(np.float64).reshape(-1, ne0)
def rel(w, q): return float(((w - q) ** 2).sum() / (w ** 2).sum())
def kquant(w, tid):
    w.astype(np.float32).tofile('in.f32')
    subprocess.run(['./kq', str(tid), 'in.f32', 'out.f32', str(w.shape[1])], capture_output=True)
    return np.fromfile('out.f32', dtype=np.float32).reshape(w.shape).astype(np.float64)
MULT = {0: [1.0], 1: [1.0, 0.55], 2: [1.0, 0.8, 0.6, 0.4]}
def packed(w, bits, per_word, tag):
    # symmetric `bits`-bit values, fp16 scale per 32, `per_word` weights share a 16-bit word with `tag` spare bits
    qmax = 2 ** (bits - 1) - 1
    b = w.reshape(w.shape[0], -1, 32)
    d = (np.abs(b).max(-1, keepdims=True) / qmax).astype(np.float16).astype(np.float64); d[d == 0] = 1e-30
    n = 32 - 32 % per_word  # whole words inside the block; the remainder uses the plain scale
    best = None
    for m in MULT[tag]:
        g = b[..., :n].reshape(b.shape[0], b.shape[1], -1, per_word)
        dd = (d * m)[..., None]
        q = np.clip(np.round(g / dd), -qmax, qmax) * dd
        e = ((g - q) ** 2).sum(-1)
        if best is None: best_q, best = q, e
        else:
            better = e < best
            best = np.where(better, e, best); best_q = np.where(better[..., None], q, best_q)
    out = b.copy()
    out[..., :n] = best_q.reshape(b.shape[0], b.shape[1], n)
    if n < 32: out[..., n:] = np.clip(np.round(b[..., n:] / d), -qmax, qmax) * d
    words = 32 // per_word + (1 if 32 % per_word else 0)
    bpw = (words * 16 + 16) / 32  # words + the fp16 scale, per 32 weights
    return out.reshape(w.shape), bpw
meta = {m['name']: m['shape'] for m in json.load(open(S + '/slices.json'))}
cases = [('attn_qkv rotated', 'blk.20.attn_qkv.weight', True), ('gate_exps x4', 'blk.20.ffn_gate_exps.weight', False), ('down_exps x4', 'blk.20.ffn_down_exps.weight', False)]
for label, name, rot in cases:
    w = load(name, meta[name][0])
    if rot: w = (w.reshape(w.shape[0], -1, 32) @ H.T).reshape(w.shape)
    if w.shape[0] > 4096: w = w[rng.choice(w.shape[0], 4096, replace=False)]
    print('== ' + label)
    for tid, tn, bpw in ((11, 'Q3_K', 3.4375), (12, 'Q4_K', 4.5), (13, 'Q5_K', 5.5), (14, 'Q6_K', 6.5625), (8, 'Q8_0', 8.5)):
        print('  %-30s %.4f bpw  error %.6f' % (tn, bpw, rel(w, kquant(w, tid))))
    for bits, pw, tag, nm in ((3, 5, 0, '3-bit x5 per word, spare unused'), (3, 5, 1, 'Q3_PT: x5 + 1 tag bit'),
                              (5, 3, 0, '5-bit x3 per word, spare unused'), (5, 3, 1, 'Q5_PT: x3 + 1 tag bit'),
                              (7, 2, 0, '7-bit x2 per word, spare unused'), (7, 2, 2, 'Q7_PT: x2 + 2 tag bits')):
        q, bpw = packed(w, bits, pw, tag); print('  %-30s %.4f bpw  error %.6f' % (nm, bpw, rel(w, q)))
