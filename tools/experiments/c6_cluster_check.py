#!/usr/bin/env python3
"""C6 accuracy-assisted miss map (author 19:15-19:18), check 2: do the stuck places cluster in STATE space?
Inputs (run B, urm_ZT2, engine format): PELCODE1 state codes (per scored token and layer: 256 sign bytes of the layer's
normed input after a fixed sign diagonal + Hadamard per 32, 64 log-energy bytes), PELROUT1 routes, PELKLD01 per-token KL.
1) Stuck = top 1 % KL. For each stuck token in chunks 50-99, its nearest neighbours by Hamming distance among chunks 0-49
   (a miss map logged on the first half, tested on the second): how often is the nearest neighbour itself stuck (base 1 %)?
   Then the flag rule "nearest logged miss within radius r": recall of stuck tokens vs the share of all tokens flagged.
2) Prefetch: for each token and layer L, the experts of layer L+2 predicted by (a) the nearest earlier state's layer L+2
   experts (code kNN), vs (b) today's one-layer-ahead pre-gating record at L+1 (pred). Hit rate on cold-style picks."""
import numpy as np, struct, sys
D = '/mnt/user-data/uploads/pelias-tests/gp/C6/'
b = open(D + 'urmzt2_codes.bin', 'rb').read()
assert b[:8] == b'PELCODE1'
n_ctx, ch, sc, nl, ne, cb = struct.unpack_from('<6I', b, 8)
T = ch * sc
codes = np.frombuffer(b, np.uint8, offset=40).reshape(T, nl, cb)
k = open(D + 'urmzt2_c6.kld', 'rb').read(); n = struct.unpack_from('<Q', k, 28)[0]
kl = np.frombuffer(k, np.dtype([('kld', '<f8'), ('nll', '<f4'), ('nllb', '<f4'), ('pd', '<f4'), ('top', 'u1'), ('pad', 'V3')]), count=n, offset=36)['kld']
assert n == T
stuck = kl >= np.quantile(kl, 0.99)
chunk = np.arange(T) // sc
first, second = chunk < ch // 2, chunk >= ch // 2
print(f"tokens {T} ({ch} chunks x {sc}), layers {nl}, code bytes {cb}; stuck {stuck.sum()} (first half {stuck[first].sum()}, second {stuck[second].sum()})")
def sign_words(L):
    s = np.ascontiguousarray(codes[:, L, :ne // 8])
    return s.view(np.uint64)  # 2048 bits = 32 words
def hamming(Q, Db):  # Q (q, 32), Db (d, 32) -> (q, d)
    out = np.empty((len(Q), len(Db)), np.uint16)
    for i in range(0, len(Q), 64):
        x = np.bitwise_xor(Q[i:i + 64, None, :], Db[None, :, :])
        out[i:i + 64] = np.bitwise_count(x).sum(2)
    return out
rng = np.random.default_rng(0)
print("\n1) Stuck tokens in state space (map = chunks 0-49, test = chunks 50-99)")
db_idx = np.where(first)[0]; db_stuck = stuck[db_idx]
q_stuck = np.where(second & stuck)[0]
q_rand = rng.choice(np.where(second & ~stuck)[0], 1500, replace=False)
for L in (3, 10, 20, 30, 39):
    W = sign_words(L)
    Hs = hamming(W[q_stuck], W[db_idx]); Hr = hamming(W[q_rand], W[db_idx])
    nn_s = Hs.argmin(1); nn_r = Hr.argmin(1)
    p_s = db_stuck[nn_s].mean(); p_r = db_stuck[nn_r].mean()
    # distance to the nearest logged miss
    dm_s = Hs[:, db_stuck].min(1); dm_r = Hr[:, db_stuck].min(1)
    rows = []
    for r in np.percentile(dm_s, [25, 50, 75]):
        rec = (dm_s <= r).mean(); far = (dm_r <= r).mean()
        rows.append(f"r={int(r)}: recall {rec*100:.0f} % flags {far*100:.1f} % of others")
    print(f"  layer {L:2d}: nearest neighbour stuck: stuck queries {p_s*100:.1f} % vs other queries {p_r*100:.1f} % (base 1 %); "
          f"median distance to nearest logged miss {np.median(dm_s):.0f} vs {np.median(dm_r):.0f} bits; " + "; ".join(rows))
print("\n2) Prefetch two layers ahead: code kNN vs today's one-layer-ahead pre-gating")
r = open(D + 'urmzt2_routes.bin', 'rb').read()
assert r[:8] == b'PELROUT1'
_, _, _, rl, nu = struct.unpack_from('<5I', r, 8)
rec = np.frombuffer(r, np.dtype([('sel', '<i2', 8), ('w', '<f4', 8), ('pl', '<f4'), ('pn', '<f4'), ('pred', '<i2', 8)]), offset=36).reshape(T, rl)
sel = rec['sel']; pred = rec['pred']
def hit(a, b):  # fraction of b's experts found in a, per row
    return np.array([len(set(x) & set(y)) / len(y) for x, y in zip(a, b)])
q = rng.choice(np.where(second)[0], 1500, replace=False)
for L in (3, 10, 20, 30):
    W = sign_words(L)
    H = hamming(W[q], W[db_idx]); nn = db_idx[H.argmin(1)]
    knn2 = hit(sel[nn, L + 2], sel[q, L + 2])            # nearest logged state's experts at L+2
    knn1 = hit(sel[nn, L + 1], sel[q, L + 1])
    pg1 = hit(pred[q, L], sel[q, L + 1])                  # pre-gating record: layer L+1's router on layer L's output
    print(f"  layer {L:2d}: one layer ahead: pre-gating {pg1.mean()*100:.0f} %, code kNN {knn1.mean()*100:.0f} %; two layers ahead: code kNN {knn2.mean()*100:.0f} % (random pick of 8 of 256: 3 %)")
