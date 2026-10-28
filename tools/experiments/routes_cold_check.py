# Item 4: shares from the routing record (PELROUT1, src/cli/main.cc): cold picks (outside each layer's 120 most-picked
# experts), their router weights and ranks, how many are unpredicted per token, and which of those a peek could target.
import struct, numpy as np
f = open('/mnt/user-data/uploads/pelias-tests/gp/K/stock_10.rt', 'rb'); assert f.read(8) == b'PELROUT1'
n_ctx, chunks, scored, nl, nu = struct.unpack('<5I', f.read(20)); f.read(8)
rec = np.frombuffer(f.read(), dtype=np.dtype([('sel', '<i2', 8), ('w', '<f4', 8), ('pl', '<f4'), ('pn', '<f4'), ('pred', '<i2', 8)]))
T = chunks * scored; rec = rec.reshape(T, nl); sel = rec['sel']; w = rec['w']; pred = rec['pred']
cnt = np.zeros((nl, 256), int)
for l in range(nl): np.add.at(cnt[l], sel[:, l].ravel(), 1)
hot = np.zeros((nl, 256), bool)
for l in range(nl): hot[l, np.argsort(-cnt[l])[:120]] = True
L = np.broadcast_to(np.arange(nl)[None, :, None], sel.shape)
cold = ~hot[L, sel]
# predicted: in the previous layer's prediction list for this layer (pred of layer l-1 names layer l's picks)
predicted = np.zeros_like(cold)
for l in range(1, nl):
    p = pred[:, l - 1]; predicted[:, l] = (sel[:, l][:, :, None] == p[:, None, :]).any(-1)
rank = np.broadcast_to(np.arange(8)[None, None, :], sel.shape)
print('tokens %d, layers %d; cold picks %.1f %% of picks, %.1f per token' % (T, nl, 100 * cold.mean(), cold.sum() / T))
c1 = cold[:, 1:]; pr = predicted[:, 1:]
print('cold picks predicted one layer ahead (layers 1-39): %.1f %%; unpredicted cold picks per token %.1f' % (100 * pr[c1].mean(), (c1 & ~pr).sum() / T))
for nm, m in (('all picks', np.ones_like(cold)), ('cold picks', cold), ('unpredicted cold picks', cold & ~predicted & (L > 0))):
    ww = w[m]; rk = rank[m]
    print('  %-24s mean weight %.3f; rank 5-8 %.1f %%; weight < 0.08: %.1f %%; < 0.05: %.1f %%; < 0.03: %.1f %%' % (
        nm, ww.mean(), 100 * (rk >= 4).mean(), 100 * (ww < 0.08).mean(), 100 * (ww < 0.05).mean(), 100 * (ww < 0.03).mean()))
