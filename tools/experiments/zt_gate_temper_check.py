import os
# ZT fix candidates (real blk.20 attn_qkv, rotated): level search always plain; gate ranking by block gain weighted
# with importance tempered in different ways. Report plain error (no calibration bias) and weighted error.
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'q7zt_quota_check.py')).read().split("ones = np.ones")[0])
ones = np.ones(w.shape[1]); m = imp.mean()
variants = {'plain': ones, 'full importance': imp, 'sqrt importance': np.sqrt(imp / m),
            'clipped 0.25-4x mean': np.clip(imp / m, 0.25, 4.0), 'importance + mean (1:1)': imp / m + 1.0}
Q8, S8 = zip_qs(8, None); full = Q8 * S8; un = odd_grid(ones)
d_u = (w - un) ** 2; d_f = (w - full) ** 2
tot_w = ((w ** 2) * imp).sum(); tot_p = (w ** 2).sum()
for nm, wt in variants.items():
    gain = ((d_u - d_f) * wt).reshape(w.shape[0], -1, 8, 32).sum(-1)
    rk = (np.argsort(-gain, axis=-1).argsort(axis=-1) < 2)[..., None]
    sel = np.where(rk, d_f.reshape(w.shape[0], -1, 8, 32), d_u.reshape(w.shape[0], -1, 8, 32)).reshape(w.shape)
    print('  gates by %-26s plain error %.6f  weighted error %.6f' % (nm, sel.sum() / tot_p, (sel * imp).sum() / tot_w))
