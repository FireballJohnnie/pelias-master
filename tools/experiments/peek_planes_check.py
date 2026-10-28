# Items 2 (C9 1-bit peek) and 3 (C11 planes by need) on the real blk.20 experts (first 4; gate: 512 x 2048, down: 2048 x 512).
# Activations are synthetic: Gaussian per input, scaled by each expert's own importance (merged matrix), 400 per expert.
# Planes: the expert quantized to 5-bit values with zip scales (fp16 per 256 + 6-bit per 32, the Q5_PH layout on paper),
# q = 2b + h: base plane b (4-bit), refinement h (1 bit).
import sys, numpy as np
sys.path.insert(0, '/home/claude/pelias-master/tools'); from imatrix_merge import Gguf
SL = '/mnt/user-data/uploads/pelias-tests/gp/M/slices'; rng = np.random.default_rng(1)
gi = Gguf('../urm/merged.gguf'); ent = gi.entries()
def zip_q(w, bits):  # integer values q and per-32 scales s (Q8_Z-style design with `bits`)
    qmax = 2 ** (bits - 1) - 1; b = w.reshape(w.shape[0], -1, 8, 32)
    top = np.abs(b).max((-1, -2), keepdims=True); d = (top / qmax).astype(np.float16).astype(np.float64); d[d == 0] = 1e-30
    amax = np.abs(b).max(-1, keepdims=True); ideal = amax / qmax / d * 63; best = None
    for c in np.linspace(0.70, 1.0, 13):
        lv = np.clip(np.ceil(ideal * c), 1, 63); s = d * lv / 63; q = np.clip(np.round(b / s), -qmax, qmax)
        er = ((b - q * s) ** 2).sum(-1)
        if best is None: Q, S, best = q, np.broadcast_to(s, q.shape), er
        else: t = er < best; best = np.where(t, er, best); Q = np.where(t[..., None], q, Q); S = np.where(t[..., None], s, S)
    return Q.reshape(w.shape), np.asarray(S).reshape(w.shape)
for name, ne0, rows in (('blk.20.ffn_gate_exps.weight', 2048, 512), ('blk.20.ffn_down_exps.weight', 512, 2048)):
    raw = np.fromfile('%s/%s.bin' % (SL, name), dtype=np.uint16).astype(np.uint32) << 16
    W = raw.view(np.float32).astype(np.float64).reshape(4, rows, ne0)
    e = ent[name]; s2 = np.frombuffer(gi.read(e['.in_sum2'][2], e['.in_sum2'][3]), np.float32).reshape(256, ne0)[:4].astype(np.float64)
    cnt = np.frombuffer(gi.read(e['.counts'][2], e['.counts'][3]), np.float32)[:4].astype(np.float64)
    corr, nerr, tight_b, tight_s = [], [], [], []
    for x in range(4):
        w = W[x]; X = rng.standard_normal((400, ne0)) * np.sqrt(s2[x] / max(cnt[x], 1))
        y = X @ w.T
        # item 2: sign plane (centred per row), one fitted gain per expert
        sg = np.sign(w - w.mean(1, keepdims=True)); ys = X @ sg.T
        a = (ys * y).sum() / (ys * ys).sum(); ys = a * ys
        corr.append(np.corrcoef(y.ravel(), ys.ravel())[0, 1])
        nerr.append(np.abs(np.linalg.norm(ys, axis=1) / np.linalg.norm(y, axis=1) - 1).mean())
        # item 3: planes by need on the 5-bit layout
        Q, S = zip_q(w, 5); b = np.floor_divide(Q, 2); h = Q - 2 * b
        yfull = X @ (Q * S).T; ybase = X @ ((2 * b + 0.5) * S).T          # base plane + the expected refinement 0.5
        actual = np.linalg.norm(yfull - ybase, axis=1)                    # per activation: error of base-only, whole output
        bound = np.linalg.norm(np.abs(X) @ (0.5 * S).T, axis=1)           # exact worst case: |h - 0.5| <= 0.5 per weight
        stat = np.sqrt(((X ** 2) @ ((0.5 * S) ** 2).T).sum(1))           # expected size if h is a fair coin
        tight_b.append((bound / actual).mean()); tight_s.append((stat / actual).mean())
        rel = (actual / np.linalg.norm(yfull, axis=1)).mean()
        print('  %s expert %d: base-only output error %.4f of the output' % (name.split('.')[2], x, rel))
    print('== %s: item 2 sign-plane correlation %.3f (theory 0.80), output-norm estimate error %.3f; item 3 exact bound / actual %.1fx, expected-size estimate / actual %.2fx'
          % (name, np.mean(corr), np.mean(nerr), np.mean(tight_b), np.mean(tight_s)))
