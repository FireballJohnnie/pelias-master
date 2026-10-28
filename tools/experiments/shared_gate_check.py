import os
# Idea 1 (author 16:24 "ye"): the live gate with ONE block ranking per token shared by every row (by the token's
# activation energy per 32-block, optionally times a fixed per-block weight scale), against the per-row ranking
# (s^2 * sum x^2) used in --h-gate live, at 25 % of blocks. Q8_Z values on the real blk.20 attn_qkv (rotated);
# synthetic activations: Gaussian x sqrt(importance) per input, with a heavy-tailed per-token, per-block energy.
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'q7zh_check.py')).read().split("q7c = kq(201, w)")[0])
rng = np.random.default_rng(7)
R = 2048; W = w[:R]
Q8, S8 = zip_qs(8); Q8 = Q8[:R]; S8 = S8[:R]
h = ((Q8 + 127) % 2) - 0.5                          # H minus its 0.5 fill
full = Q8 * S8
nb = W.shape[1] // 32; T = 96
X = rng.standard_normal((T, W.shape[1])) * np.sqrt(imp)[None, :]
X *= np.repeat(np.exp(rng.standard_normal((T, nb)) * 0.8), 32, axis=1)   # heavy-tailed block energy per token
X = (X.reshape(T, -1, 32) @ H.T).reshape(T, -1)                           # rotated as the file expects
y_true = X @ W.T; y_full = X @ full.T
# per-row, per-block contribution of dropping H: d[t, r, j] = s_rj * sum_i h_ri x_ti over block j
s_blk = S8.reshape(R, nb, 32)[:, :, 0]
d = np.einsum('rjk,tjk->trj', (h * S8).reshape(R, nb, 32), X.reshape(T, nb, 32))
Xe = (X.reshape(T, nb, 32) ** 2).sum(-1)                                   # token energy per block
keep = int(0.25 * nb)
def err_with(mask):  # mask[t, r, j] True = H read
    y = y_full[:, :, None] * 0 + 0
    drop = (~mask) * d
    yy = y_full - drop.sum(-1)
    return float(((yy - y_true) ** 2).sum() / (y_true ** 2).sum())
base_err = float(((y_full - y_true) ** 2).sum() / (y_true ** 2).sum())
none_err = err_with(np.zeros((T, R, nb), bool))
score_row = (s_blk[None] ** 2) * Xe[:, None, :]
m_row = score_row >= np.sort(score_row, -1)[..., [nb - keep]]
m_shared = np.broadcast_to((Xe >= np.sort(Xe, -1)[:, [nb - keep]])[:, None, :], (T, R, nb))
s2mean = (s_blk ** 2).mean(0)
sc2 = Xe * s2mean[None]
m_shared2 = np.broadcast_to((sc2 >= np.sort(sc2, -1)[:, [nb - keep]])[:, None, :], (T, R, nb))
rnd = rng.random((T, nb)); m_rand = np.broadcast_to((rnd >= np.sort(rnd, -1)[:, [nb - keep]])[:, None, :], (T, R, nb))
print('output error / output energy (real attn_qkv rows, %d tokens):' % T)
print('  all H read (Q8_Z)                         %.6f' % base_err)
print('  no H (0.5 fill everywhere)                %.6f' % none_err)
for nm, m in (('per-row ranking s^2*sum x^2 (live gate)', m_row), ('shared ranking: token energy only', m_shared),
              ('shared ranking: energy x mean s^2', m_shared2), ('shared, random blocks', m_rand)):
    e = err_with(m); print('  25 %% H, %-36s %.6f   (%.0f %% of the H gap closed)' % (nm, e, 100 * (none_err - e) / (none_err - base_err)))
