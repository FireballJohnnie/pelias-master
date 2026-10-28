import os
# Q7_ZT design check: H on exactly k of the 8 sub-blocks of each 256-block (fixed block size), chosen by importance-
# weighted or plain error gain, vs the free global ranking. Ungated sub-blocks: their own best level on the odd grid
# (2b - 127) * s (a native 7-bit grid); gated: 8-bit values (2b + h - 127) * s. Real blk.20 attn_qkv, rotated.
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'q7zh_check.py')).read().split("q7c = kq(201, w)")[0])
nb = w.shape[1] // 32
def odd_grid(wt):  # best per-32 level with values restricted to odd integers in -127..127
    b = w.reshape(w.shape[0], -1, 8, 32); iw = wt.reshape(1, -1, 8, 32)
    top = np.abs(b).max((-1, -2), keepdims=True); d = (top / 127).astype(np.float16).astype(np.float64); d[d == 0] = 1e-30
    amax = np.abs(b).max(-1, keepdims=True); ideal = amax / 127 / d * 63; best = None
    for c in np.linspace(0.70, 1.05, 36):
        lv = np.clip(np.ceil(ideal * c), 1, 63); s = d * lv / 63
        q = np.clip(2 * np.floor(b / s / 2) + 1, -127, 127)  # nearest odd integer
        er = (((b - q * s) ** 2) * iw).sum(-1)
        if best is None: O, best = q * s, er
        else: t = er < best; best = np.where(t, er, best); O = np.where(t[..., None], q * s, O)
    return O.reshape(w.shape)
ones = np.ones(w.shape[1])
for lab, wt in (('weighted ranking', imp), ('plain ranking', ones)):
    Q8, S8 = zip_qs(8, None); full = Q8 * S8; un = odd_grid(ones)
    eu = (((w - un) ** 2) * wt).reshape(w.shape[0], -1, 8, 32).sum(-1); ef = (((w - full) ** 2) * wt).reshape(w.shape[0], -1, 8, 32).sum(-1)
    gain = eu - ef; tot = ((w ** 2) * imp).sum()
    EU = (((w - un) ** 2) * imp).reshape(w.shape[0], -1, 8, 32).sum(-1); EF = (((w - full) ** 2) * imp).reshape(w.shape[0], -1, 8, 32).sum(-1)
    for k in (0, 1, 2, 4, 8):
        rk = np.argsort(-gain, axis=-1).argsort(axis=-1) < k
        e = np.where(rk, EF, EU).sum() / tot
        bpw = (2 + 6 + 1 + 224 + 4 * k) * 8 / 256
        print('%-16s k=%d of 8 (%3d %%): %.4f bpw, weighted error %.6f' % (lab, k, 100 * k // 8, bpw, e))
