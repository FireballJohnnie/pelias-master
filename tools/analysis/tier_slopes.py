#!/usr/bin/env python3
"""#2: error slopes of every expert class in every layer (experts 0-1 of blk.0-39 from the BF16 original, gp/TIER slices).
Per expert, matrix (gate, up, down) and type (Q4_K, Q4_TG2, Q5_K, Q6_K, Q8_0; each quantized with the expert's own
importance from the imatrix in merged.gguf, Unsloth's chat+code calibration):
  wrel  = sum imp (w-wq)^2 / sum imp w^2                     (the weight-domain error used so far)
  orel  = E||y' - y||^2 / E||y||^2 through SwiGLU, y = Wd (silu(Wg x) * Wu x), one matrix quantized at a time
  oabs  = E||y' - y||^2
x is drawn N(0, c^2 diag(imp_x)), with c set so that E[h^2] matches down's imatrix energy (the proxy's check: the
per-channel correlation of E[h^2] with down's importance is printed). Correlations between input channels are not modelled.
Per 256-block (along the input): the weighted error of the base type and what one residual sign plane removes from it
(scale = importance-weighted mean |r|, fp16; the planes builder's k=1 fit)."""
import os, sys, glob, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/home/claude/pelias-master/tools'); from imatrix_merge import Gguf
HERE = os.path.dirname(os.path.abspath(__file__)); KQ = HERE + '/kqw'; TMP = HERE + '/tier_v2'; os.makedirs(TMP, exist_ok=True)
SL = '/mnt/user-data/uploads/pelias-tests/gp/TIER'
g = Gguf('/tmp/claude-0/-home-claude/c7ff3710-2054-5bef-9957-5760a8e25c56/scratchpad/urm/merged.gguf'); EN = g.entries()
TYPES = [(12, 'Q4_K'), (204, 'Q4_TG2'), (13, 'Q5_K'), (14, 'Q6_K'), (8, 'Q8_0')]
L, NE, B = 40, 2, 256
f16 = lambda a: a.astype(np.float16).astype(np.float64)
def path(name):
    for d in ('slices_b1', 'slices_b2', 'slices'):
        p = f'{SL}/{d}/{name}.bin'
        if os.path.exists(p): return p
def load(l, c):
    raw = np.fromfile(path(f'blk.{l}.ffn_{c}_exps.weight'), dtype=np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32).astype(np.float64).reshape(NE, -1, 2048 if c != 'down' else 512)
def imp(l, c):
    e = EN[f'blk.{l}.ffn_{c}_exps.weight']
    s2 = np.frombuffer(g.read(*e['.in_sum2'][2:4]), dtype=np.float32).astype(np.float64)
    cnt = np.frombuffer(g.read(*e['.counts'][2:4]), dtype=np.float32).astype(np.float64)
    return s2.reshape(len(cnt), -1)[:NE] / np.maximum(cnt[:NE, None], 1.0), cnt[:NE]
def kq(args):  # through pipes (kqwp): v1 filled the sandbox disk with temp files
    tid, w, im, tag = args
    r = subprocess.run([HERE + '/kqwp', str(tid), str(w.shape[0]), str(w.shape[1])],
                       input=w.astype(np.float32).tobytes() + im.astype(np.float32).tobytes(), capture_output=True)
    assert r.returncode == 0, (tag, r.returncode)
    return np.frombuffer(r.stdout, dtype=np.float32).reshape(w.shape).astype(np.float64)
def plane1(r, im):  # one residual plane per 256-block of each row; returns the block errors before and after
    rows, n = r.shape; R = r.reshape(rows, n // B, B); I = np.broadcast_to(im.reshape(1, n // B, B), R.shape)
    sw = I.sum(-1, keepdims=True); s = f16((I * np.abs(R)).sum(-1, keepdims=True) / np.where(sw > 0, sw, 1))
    before = (I * R * R).sum(-1); after = (I * (R - s * np.where(R >= 0, 1.0, -1.0)) ** 2).sum(-1)
    return before.ravel(), after.ravel()
silu = lambda z: z / (1 + np.exp(-z))
rng = np.random.default_rng(7); S = 256
out = {k: np.zeros((L, NE, 3, len(TYPES))) for k in ('wrel', 'orel', 'oabs')}
blocks = {}; checks = []
pool = ThreadPoolExecutor(2)
for l in range(L):
    if os.path.exists(f'{TMP}/layer_{l}.npz'): print(f'layer {l}: saved earlier'); continue
    rng = np.random.default_rng(7 + l)
    W = {c: load(l, c) for c in ('gate', 'up', 'down')}
    ix, cx = imp(l, 'gate'); iu, _ = imp(l, 'up'); ih, ch = imp(l, 'down')
    for e in range(NE):
        Z = rng.standard_normal((2048, S))
        hfun = lambda c_: silu(W['gate'][e] @ (c_ * np.sqrt(ix[e])[:, None] * Z)) * (W['up'][e] @ (c_ * np.sqrt(ix[e])[:, None] * Z))
        lo, hi = 0.01, 100.0  # c so that mean E[h^2] = mean of down's importance (h is monotone in c here)
        for _ in range(50):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if (hfun(mid) ** 2).mean() < ih[e].mean() else (lo, mid)
        c0 = np.sqrt(lo * hi); X = c0 * np.sqrt(ix[e])[:, None] * Z
        G, U = W['gate'][e] @ X, W['up'][e] @ X; H = silu(G) * U; Y = W['down'][e] @ H; ey = (Y * Y).sum(0).mean()
        checks.append((l, e, c0, np.corrcoef((H * H).mean(1), ih[e])[0, 1], cx[e], ch[e]))
        jobs = []
        for m, c, im in ((0, 'gate', ix[e]), (1, 'up', iu[e]), (2, 'down', ih[e])):
            for t, (tid, tn) in enumerate(TYPES): jobs.append((m, c, im, t, (tid, W[c][e], im, f'l{l}e{e}{c}{tid}')))
        res = list(pool.map(kq, [j[-1] for j in jobs]))
        for (m, c, im, t, _), wq in zip(jobs, res):
            w = W[c][e]; I = im[None, :]
            out['wrel'][l, e, m, t] = (I * (w - wq) ** 2).sum() / (I * w * w).sum()
            if c == 'gate': Y2 = W['down'][e] @ (silu(wq @ X) * U)
            elif c == 'up': Y2 = W['down'][e] @ (silu(G) * (wq @ X))
            else: Y2 = wq @ H
            d = ((Y2 - Y) ** 2).sum(0).mean(); out['oabs'][l, e, m, t] = d; out['orel'][l, e, m, t] = d / ey
            bb, ba = plane1(w - wq, im); blocks[(l, e, m, t)] = np.stack([bb, ba]).astype(np.float32)
    np.savez(f'{TMP}/layer_{l}.npz', **{k: v[l] for k, v in out.items()}, checks=np.array(checks[-NE:]),
             **{f'blk_{e}_{m}_{t}': v for (ll, e, m, t), v in blocks.items() if ll == l})
    print(f'layer {l}: orel Q4_K gate/up/down {out["orel"][l, :, 0, 0].mean():.2e} {out["orel"][l, :, 1, 0].mean():.2e} {out["orel"][l, :, 2, 0].mean():.2e}; '
          f'wrel {out["wrel"][l, :, 0, 0].mean():.2e} {out["wrel"][l, :, 1, 0].mean():.2e} {out["wrel"][l, :, 2, 0].mean():.2e}; proxy corr {checks[-1][3]:.2f}', flush=True)
# assemble from the per-layer files (also those of an earlier, stopped run)
A = {k: np.zeros((L, NE, 3, len(TYPES))) for k in ('wrel', 'orel', 'oabs')}; BL = {}; CK = []
for l in range(L):
    z = np.load(f'{TMP}/layer_{l}.npz')
    for k in A: A[k][l] = z[k]
    CK.append(z['checks']); BL.update({f'blk_{l}_' + k[4:]: z[k] for k in z.files if k.startswith('blk_')})
np.savez(HERE + '/tier_slopes_v2.npz', **A, checks=np.concatenate(CK), types=np.array([t[1] for t in TYPES]), **BL)
print('done')
