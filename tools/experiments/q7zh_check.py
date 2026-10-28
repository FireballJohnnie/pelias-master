import os
# Q7_ZH (author 14:30): Q8_Z's 8-bit values q split as q = 2b + h (7-bit base plane b, 1-bit H plane h). Read both = Q8_Z;
# read base only = 7-bit grid (reconstruct 2b + 0.5, the expected h). Compared with Q7_Z (native 7-bit) and with the level
# search weighted by the importance matrix. Real blk.20 attn_qkv, rotated, all rows; error weighted by the matrix.
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 's1_rungs_check.py')).read().split("rows = []")[0])
def zip_qs(bits, wt=None):  # integer values and per-value scale; level search on plain or importance-weighted error
    qmax = 2 ** (bits - 1) - 1; b = w.reshape(w.shape[0], -1, 8, 32)
    iw = np.ones(32 * 8 * (w.shape[1] // 256)) if wt is None else wt
    iw = iw.reshape(1, -1, 8, 32)
    top = np.abs(b).max((-1, -2), keepdims=True); d = (top / qmax).astype(np.float16).astype(np.float64); d[d == 0] = 1e-30
    amax = np.abs(b).max(-1, keepdims=True); ideal = amax / qmax / d * 63; best = None
    for c in np.linspace(0.70, 1.0, 31):
        lv = np.clip(np.ceil(ideal * c), 1, 63); s = d * lv / 63; q = np.clip(np.round(b / s), -qmax, qmax)
        er = (((b - q * s) ** 2) * iw).sum(-1)
        if best is None: Q, S, best = q, np.broadcast_to(s, q.shape), er
        else: t = er < best; best = np.where(t, er, best); Q = np.where(t[..., None], q, Q); S = np.where(t[..., None], s, S)
    return Q.reshape(w.shape), np.asarray(S).reshape(w.shape)
q7c = kq(201, w)
print('Q7_Z (C++, plain search)                 7.25 bpw  weighted error %.6f' % errs(q7c)[1])
for lab, wt in (('plain', None), ('importance-weighted', imp)):
    Q8, S8 = zip_qs(8, wt); Q7, S7 = zip_qs(7, wt)
    b = np.floor_divide(Q8, 2)
    print('-- level search %s' % lab)
    print('  Q8_Z full (base + H)                   8.25 bpw  weighted error %.6f' % errs(Q8 * S8)[1])
    print('  Q7_ZH base only (2b + 0.5)             7.25 read weighted error %.6f' % errs((2 * b + 0.5) * S8)[1])
    print('  Q7_Z native 7-bit                      7.25 bpw  weighted error %.6f' % errs(Q7 * S7)[1])
