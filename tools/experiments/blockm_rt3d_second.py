# provenance: original (Block M experiment, sandbox only; paths are the sandbox staging paths)
# Copyright (c) 2026 John Lipka and the Pelias contributors
# RT as a second opinion for prefetch: exact pre-gating (8 picks) plus extra cold candidates from a 3-D map.
# Train on chunks 0-4, test on 5-9 (stock routing record). Next layer, same token.
import struct, sys, numpy as np
b = open(sys.argv[1], 'rb').read(); n_ctx, C, S, L, nu = struct.unpack_from('<5I', b, 8)
dt = np.dtype([('sel', '<i2', 8), ('w', '<f4', 8), ('pl', '<f4'), ('pn', '<f4'), ('pred', '<i2', 8)])
a = np.frombuffer(b, dtype=dt, offset=36).reshape(C, S, L)
E = 256; tr, te = slice(0, C // 2), slice(C // 2, C)
sel, w, pred = a['sel'].astype(int), a['w'].astype(float), a['pred'].astype(int)
def onehot(s, ww=None):
    m = np.zeros((s.shape[0], E)); np.add.at(m, (np.repeat(np.arange(s.shape[0]), s.shape[1]), s.ravel()), 1.0 if ww is None else ww.ravel()); return m
def embed3(M):
    nA, nB = M.shape; W = np.zeros((nA + nB, nA + nB)); W[:nA, nA:] = M; W[nA:, :nA] = M.T; W += 1e-6
    d = W.sum(1); Dm = 1 / np.sqrt(d); vals, vecs = np.linalg.eigh(Dm[:, None] * W * Dm[None, :])
    X = vecs[:, -4:-1] * Dm[:, None]; X /= X.std(0) + 1e-12; return X[:nA], X[nA:]
hot_n = 120
for extra in (2, 4, 8):
    tot_c = hit_pg = hit_un = hit_full = 0; n_extra = 0; tot = hp = hu = 0
    for l in range(L - 1):
        src_tr, w_tr, tgt_tr = sel[tr, :, l].reshape(-1, nu), w[tr, :, l].reshape(-1, nu), sel[tr, :, l + 1].reshape(-1, nu)
        M = onehot(src_tr, w_tr).T @ onehot(tgt_tr)
        XA, XB = embed3(M); D2 = ((XA[:, None] - XB[None]) ** 2).sum(-1); near = np.exp(-D2 / (np.median(D2) * 0.05))
        full = M / (M.sum(1, keepdims=True) + 1e-9)
        cnt = np.bincount(sel[tr, :, l + 1].ravel(), minlength=E); hot = np.zeros(E, bool); hot[np.argsort(-cnt)[:hot_n]] = True
        A = onehot(sel[te, :, l].reshape(-1, nu), w[te, :, l].reshape(-1, nu))
        s3, sf = A @ near, A @ full
        s3[:, hot] = -1e9; sf[:, hot] = -1e9       # second opinion only proposes cold experts
        T = sel[te, :, l + 1].reshape(-1, nu); P = pred[te, :, l].reshape(-1, nu)
        top3 = np.argsort(-s3, 1)[:, :extra]; topf = np.argsort(-sf, 1)[:, :extra]
        for t3, tf, p, tg in zip(top3, topf, P, T):
            sp = set(p); su = sp | set(t3); sfu = sp | set(tf); n_extra += len(su) - len(sp)
            for x in tg:
                tot += 1; hp += x in sp; hu += x in su
                if not hot[x]: tot_c += 1; hit_pg += x in sp; hit_un += x in su; hit_full += x in sfu
    print("extra %d cold candidates: cold picks caught - pre-gating alone %.1f %%, + 3-D map %.1f %%, + full map %.1f %%; all picks %.1f -> %.1f %%; extra fetches per token-layer %.2f"
          % (extra, 100 * hit_pg / tot_c, 100 * hit_un / tot_c, 100 * hit_full / tot_c, 100 * hp / tot, 100 * hu / tot, n_extra / (tot / nu)))
