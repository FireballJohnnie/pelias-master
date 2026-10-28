# provenance: original (Block M experiment, sandbox only; paths are the sandbox staging paths)
# Copyright (c) 2026 John Lipka and the Pelias contributors
# Block M gate 1: does a learned 3-weight code ("language") hold more accuracy per bit than Q4_K / IQ4_XS on real weights?
import json, subprocess, sys, numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans
S = '/mnt/user-data/uploads/pelias-tests/gp/M/slices'
rng = np.random.default_rng(0)
H = np.array([[1.]])
for _ in range(5): H = np.block([[H, H], [H, -H]])
H /= np.sqrt(32)

def load(name, shape):
    raw = np.fromfile('%s/%s.bin' % (S, name), dtype=np.uint16).astype(np.uint32) << 16
    v = raw.view(np.float32).astype(np.float64)
    return v.reshape(-1, shape[0])  # rows of ne0 (input dim)

def rel(w, q): return float(((w - q) ** 2).sum() / (w ** 2).sum())

def kquant(w, tid):
    w.astype(np.float32).tofile('in.f32')
    out = subprocess.run(['./kq', str(tid), 'in.f32', 'out.f32', str(w.shape[1])], capture_output=True, text=True).stdout.strip()
    return np.fromfile('out.f32', dtype=np.float32).reshape(w.shape).astype(np.float64), out

def to_triples(x):  # rows -> padded triples (one zero per row when ne0 % 3 == 2, etc.)
    pad = (-x.shape[1]) % 3
    xp = np.concatenate([x, np.zeros((x.shape[0], pad))], 1)
    return xp.reshape(x.shape[0], -1, 3), pad

def scales_flat(w):  # fp16 absmax per 32
    b = w.reshape(w.shape[0], -1, 32); s = np.abs(b).max(-1, keepdims=True)
    s = s.astype(np.float16).astype(np.float64); s[s == 0] = 1
    return s, 16 / 32

def scales_tags(w):  # K-quant-like tags: fp16 per 256, 6-bit relative per 32
    b = w.reshape(w.shape[0], -1, 32); s32 = np.abs(b).max(-1, keepdims=True)
    s256 = s32.reshape(w.shape[0], -1, 8, 1).max(2, keepdims=True).astype(np.float16).astype(np.float64); s256[s256 == 0] = 1
    lv = np.clip(np.ceil(s32.reshape(w.shape[0], -1, 8, 1) / s256 * 63), 1, 63)
    s = (lv / 63 * s256).reshape(w.shape[0], -1, 1)
    return s, 16 / 256 + 6 / 32

def vq(w, scal, k=4096):
    s, sbits = scal(w)
    nrm = (w.reshape(w.shape[0], -1, 32) / s).reshape(w.shape)
    tr, pad = to_triples(nrm); X = tr.reshape(-1, 3)
    samp = X[rng.choice(len(X), min(len(X), 300000), replace=False)]
    km = MiniBatchKMeans(n_clusters=k, batch_size=20000, n_init=1, max_iter=60, random_state=0).fit(samp)
    C = km.cluster_centers_
    _, idx = cKDTree(C).query(X, workers=2)
    rec = C[idx].reshape(tr.shape).reshape(w.shape[0], -1)[:, :w.shape[1]]
    q = (rec.reshape(w.shape[0], -1, 32) * s).reshape(w.shape)
    bpw = (tr.shape[1] * np.log2(k)) / w.shape[1] + sbits
    return q, bpw, C, X

def triangle(w, scal, ntri=256, order=4):
    # 8-bit triangle "word" + barycentric grid point "shade" (order 4 -> 15 points, 4 bits): one RT hit returns both
    s, sbits = scal(w)
    nrm = (w.reshape(w.shape[0], -1, 32) / s).reshape(w.shape)
    tr, pad = to_triples(nrm); X = tr.reshape(-1, 3)
    samp = X[rng.choice(len(X), min(len(X), 300000), replace=False)]
    km = MiniBatchKMeans(n_clusters=ntri, batch_size=20000, n_init=1, max_iter=60, random_state=0).fit(samp)
    lab = km.predict(samp)
    pts = []
    grid = [(i / order, j / order, (order - i - j) / order) for i in range(order + 1) for j in range(order + 1 - i)]
    for c in range(ntri):
        P = samp[lab == c]
        mu = km.cluster_centers_[c]
        if len(P) < 4: V = np.stack([mu, mu, mu])
        else:
            u, sv, vt = np.linalg.svd(P - mu, full_matrices=False)
            d1, d2 = vt[0], vt[1]; r = 2.0 * np.sqrt((sv[:2] ** 2).sum() / len(P))
            V = np.stack([mu + r * (np.cos(a) * d1 + np.sin(a) * d2) for a in (np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3)])
        for a, b2, cc in grid: pts.append(a * V[0] + b2 * V[1] + cc * V[2])
    Cg = np.array(pts)
    _, idx = cKDTree(Cg).query(X, workers=2)
    rec = Cg[idx].reshape(tr.shape).reshape(w.shape[0], -1)[:, :w.shape[1]]
    q = (rec.reshape(w.shape[0], -1, 32) * s).reshape(w.shape)
    bpw = (tr.shape[1] * (np.log2(ntri) + 4)) / w.shape[1] + sbits
    return q, bpw

meta = {m['name']: m['shape'] for m in json.load(open(S + '/slices.json'))}
cases = [('attn_qkv rotated (as R2 stores it)', 'blk.20.attn_qkv.weight', True),
         ('attn_qkv plain', 'blk.20.attn_qkv.weight', False),
         ('gate_exps x4 plain', 'blk.20.ffn_gate_exps.weight', False),
         ('down_exps x4 plain', 'blk.20.ffn_down_exps.weight', False)]
for label, name, rot in cases:
    w = load(name, meta[name])
    if rot: w = (w.reshape(w.shape[0], -1, 32) @ H.T).reshape(w.shape)
    if w.shape[0] > 4096: w = w[rng.choice(w.shape[0], 4096, replace=False)]  # 4096 rows keep the fit quick
    print('== %s: %d rows x %d' % (label, w.shape[0], w.shape[1])); sys.stdout.flush()
    for tid, tn in ((12, 'Q4_K'), (23, 'IQ4_XS'), (13, 'Q5_K')):
        q, info = kquant(w, tid); print('  %-34s %s  relative error %.5f' % (tn, info.split(', ')[1], rel(w, q)))
    q, bpw, C, X = vq(w, scales_flat); print('  %-34s %.4f bits per weight  relative error %.5f' % ('3-weight code, 4096 words, fp16/32', bpw, rel(w, q)))
    q, bpw, C, X = vq(w, scales_tags); print('  %-34s %.4f bits per weight  relative error %.5f' % ('3-weight code, 4096 words, tags', bpw, rel(w, q)))
    q, bpw = triangle(w, scales_tags); print('  %-34s %.4f bits per weight  relative error %.5f' % ('triangle word + shade (256 x 15)', bpw, rel(w, q)))
    sys.stdout.flush()
