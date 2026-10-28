# provenance: original (Block M experiment, sandbox only; paths are the sandbox staging paths)
# Copyright (c) 2026 John Lipka and the Pelias contributors
# Can a 3-D map of experts (the kind of data RT cores can search) predict upcoming expert picks?
# Train on chunks 0-4 of the stock routing record, test on chunks 5-9.
import struct, sys, numpy as np
p = sys.argv[1]
b = open(p, 'rb').read(); n_ctx, C, S, L, nu = struct.unpack_from('<5I', b, 8)
dt = np.dtype([('sel', '<i2', 8), ('w', '<f4', 8), ('pl', '<f4'), ('pn', '<f4'), ('pred', '<i2', 8)])
a = np.frombuffer(b, dtype=dt, offset=36).reshape(C, S, L)
E = 256; tr, te = slice(0, C // 2), slice(C // 2, C)
sel, w, pred = a['sel'].astype(int), a['w'].astype(float), a['pred'].astype(int)

def onehot(s, ww=None):  # tokens x experts
    m = np.zeros((s.shape[0], E))
    np.add.at(m, (np.repeat(np.arange(s.shape[0]), s.shape[1]), s.ravel()), 1.0 if ww is None else ww.ravel())
    return m

def embed3(M):  # M: |A| x |B| affinity (source -> target); joint spectral embedding into 3-D
    nA, nB = M.shape
    W = np.zeros((nA + nB, nA + nB)); W[:nA, nA:] = M; W[nA:, :nA] = M.T
    W += 1e-6
    d = W.sum(1); Dm = 1 / np.sqrt(d)
    N = Dm[:, None] * W * Dm[None, :]
    vals, vecs = np.linalg.eigh(N)
    X = vecs[:, -4:-1] * Dm[:, None]  # skip the trivial top vector, keep the next 3
    X /= X.std(0) + 1e-12
    return X[:nA], X[nA:]

def evaluate(src_sel, src_w, tgt_sel, src_tr, srcw_tr, tgt_tr, hot, K):
    A_tr = onehot(src_tr.reshape(-1, nu), srcw_tr.reshape(-1, nu)); B_tr = onehot(tgt_tr.reshape(-1, nu))
    M = A_tr.T @ B_tr  # transition counts (weighted)
    full = M / (M.sum(1, keepdims=True) + 1e-9)
    XA, XB = embed3(M)
    D2 = ((XA[:, None, :] - XB[None, :, :]) ** 2).sum(-1)
    sig = np.median(D2) * 0.05
    near = np.exp(-D2 / sig)
    A = onehot(src_sel.reshape(-1, nu), src_w.reshape(-1, nu)); T = tgt_sel.reshape(-1, nu)
    res = {}
    pop = B_tr.sum(0)
    for name, score in (('full', A @ full), ('3d', A @ near), ('popular', np.tile(pop, (A.shape[0], 1)))):
        top = np.argsort(-score, 1)[:, :K]
        hit = np.array([[t in set(r) for t in row] for r, row in zip(top, T)])
        cold = np.array([[t not in hot for t in row] for row in T])
        res[name] = (hit.mean(), hit[cold].mean() if cold.any() else float('nan'))
    return res

K = 16
hot_n = 120
print("candidates per token-layer K=%d; cold = outside the %d most-picked experts of the target layer (train half)" % (K, hot_n))
for task, dL, dT in (("next token, same layer", 0, 1), ("next layer, same token", 1, 0), ("two layers ahead", 2, 0), ("next token, next layer", 1, 1)):
    acc = {k: [0, 0, 0] for k in ('full', '3d', 'popular')}
    for l in range(L - dL):
        src = sel[:, :S - dT, l]; srcw = w[:, :S - dT, l]; tgt = sel[:, dT:, l + dL]
        cnt = np.bincount(sel[tr, :, l + dL].ravel(), minlength=E); hot = set(np.argsort(-cnt)[:hot_n])
        r = evaluate(src[te], srcw[te], tgt[te], src[tr], srcw[tr], tgt[tr], hot, K)
        for k, (h, c) in r.items(): acc[k][0] += h; acc[k][1] += c; acc[k][2] += 1
    print("%-24s full map %.1f %% (cold %.1f %%) | 3-D map %.1f %% (cold %.1f %%) | most popular %.1f %% (cold %.1f %%)" % (
        task, *(100 * acc['full'][i] / acc['full'][2] for i in (0, 1)), *(100 * acc['3d'][i] / acc['3d'][2] for i in (0, 1)),
        *(100 * acc['popular'][i] / acc['popular'][2] for i in (0, 1))))
# reference: the engine's exact pre-gating prediction (8 candidates) on the test half
hit = tot = 0
for l in range(L - 1):
    pr = pred[te, :, l]; tg = sel[te, :, l + 1]
    for x, y in zip(pr.reshape(-1, nu), tg.reshape(-1, nu)): s = set(x); hit += sum(t in s for t in y); tot += nu
print("reference: exact pre-gating prediction (8 candidates, next layer): %.1f %%" % (100 * hit / tot))
