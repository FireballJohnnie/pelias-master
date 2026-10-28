import os
# Q7_ZH with H as a per-32-block gate (author 14:31): a static bit per block says "read this block's H plane". Blocks
# are ranked by how much reading H lowers the importance-weighted error; read H on the top fraction f. Bytes read per
# weight = 7.25 + f (+ 1/32 for the gate bit). Real blk.20 attn_qkv, rotated.
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'q7zh_check.py')).read().split("q7c = kq(201, w)")[0])
Q8, S8 = zip_qs(8); b = np.floor_divide(Q8, 2)
full = Q8 * S8; base = (2 * b + 0.5) * S8
iw = imp[None, :]
eb = (((w - base) ** 2) * iw).reshape(w.shape[0], -1, 32).sum(-1); ef = (((w - full) ** 2) * iw).reshape(w.shape[0], -1, 32).sum(-1)
gain = (eb - ef).ravel(); order = np.argsort(-gain); tot = ((w ** 2) * iw).sum()
for f in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
    k = int(f * gain.size); e = eb.sum() - gain[order[:k]].sum()
    lin = eb.sum() + f * (ef.sum() - eb.sum())
    print('H read on %3d %% of blocks: %.4f bpw read, weighted error %.6f (straight line would be %.6f)' % (100 * f, 7.25 + f + 1 / 32, e / tot, lin / tot))
