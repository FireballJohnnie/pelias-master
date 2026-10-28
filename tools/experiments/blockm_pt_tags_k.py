# provenance: original (Block M experiment, sandbox only)
# Copyright (c) 2026 John Lipka and the Pelias contributors
# _PT with K-quant-grade scales: fp16 per 256 + 6-bit per 32 ("zip code" levels), a scale search per 32-block,
# then the word tags on top. Equal-bits comparison with Q5_K / Q6_K / Q8_0.
import json, subprocess, numpy as np
exec(open('pt.py').read().split("meta = {")[0])  # reuse load/rel/kquant/H
def packed_k(w, bits, per_word, tag, search=True):
    qmax = 2 ** (bits - 1) - 1
    b = w.reshape(w.shape[0], -1, 32)
    amax = np.abs(b).max(-1, keepdims=True)
    cands = np.linspace(0.70, 1.0, 13) if search else [1.0]
    n = 32 - 32 % per_word
    best_err = None
    for c in cands:
        d32 = amax * c / qmax
        sup = d32.reshape(b.shape[0], -1, 8, 1).max(2, keepdims=True).astype(np.float16).astype(np.float64); sup[sup == 0] = 1e-30
        lv = np.clip(np.round(d32.reshape(b.shape[0], -1, 8, 1) / sup * 63), 1, 63)
        d = (lv / 63 * sup).reshape(b.shape[0], -1, 1)
        g = b[..., :n].reshape(b.shape[0], b.shape[1], -1, per_word)
        bq, be = None, None
        for m in MULT[tag]:
            dd = (d * m)[..., None]
            q = np.clip(np.round(g / dd), -qmax, qmax) * dd
            e = ((g - q) ** 2).sum(-1)
            if bq is None: bq, be = q, e
            else:
                bt = e < be; be = np.where(bt, e, be); bq = np.where(bt[..., None], q, bq)
        out = b.copy(); out[..., :n] = bq.reshape(b.shape[0], b.shape[1], n)
        if n < 32: out[..., n:] = np.clip(np.round(b[..., n:] / d), -qmax, qmax) * d
        err = ((b - out) ** 2).sum(-1)  # per 32-block: keep the best candidate scale per block
        if best_err is None: best_out, best_err = out, err
        else:
            bt = err < best_err; best_err = np.where(bt, err, best_err); best_out = np.where(bt[..., None], out, best_out)
    words = 32 // per_word + (1 if 32 % per_word else 0)
    bpw = (words * 16) / 32 + 16 / 256 + 6 / 32
    return best_out.reshape(w.shape), bpw
meta = {m['name']: m['shape'] for m in json.load(open(S + '/slices.json'))}
for label, name, rot in (('attn_qkv rotated', 'blk.20.attn_qkv.weight', True), ('gate_exps x4', 'blk.20.ffn_gate_exps.weight', False), ('down_exps x4', 'blk.20.ffn_down_exps.weight', False)):
    w = load(name, meta[name][0])
    if rot: w = (w.reshape(w.shape[0], -1, 32) @ H.T).reshape(w.shape)
    if w.shape[0] > 4096: w = w[rng.choice(w.shape[0], 4096, replace=False)]
    print('== ' + label)
    for tid, tn, bpw in ((12, 'Q4_K', 4.5), (13, 'Q5_K', 5.5), (14, 'Q6_K', 6.5625), (8, 'Q8_0', 8.5)):
        print('  %-36s %.4f bpw  error %.6f' % (tn, bpw, rel(w, kquant(w, tid))))
    for bits, pw, tag, nm in ((5, 3, 0, '5-bit x3, zip scales, no tag'), (5, 3, 1, 'Q5_PT, zip scales + 1 tag bit'),
                              (7, 2, 0, '7-bit x2, zip scales, no tag'), (7, 2, 2, 'Q7_PT, zip scales + 2 tag bits')):
        q, bpw = packed_k(w, bits, pw, tag); print('  %-36s %.4f bpw  error %.6f' % (nm, bpw, rel(w, q)))
