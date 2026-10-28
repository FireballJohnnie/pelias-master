# Item 1 (K5): residual E8-lattice codebook (2^16 smallest-norm E8 points = 16 bits per 8 weights = 2 bpw per stage) on the
# real blk.20 experts (first 4), inputs rotated per 32 (shared by all experts of a layer), vs the K-quants on the weights as
# stored (unrotated), Pelias's C++ quantizer. Error plain and weighted by each expert's own importance (merged matrix).
import os, sys, itertools, subprocess, numpy as np
sys.path.insert(0, '/home/claude/pelias-master/tools'); from imatrix_merge import Gguf
SL = '/mnt/user-data/uploads/pelias-tests/gp/M/slices'; rng = np.random.default_rng(0)
H = np.array([[1.]])
for _ in range(5): H = np.block([[H, H], [H, -H]])
H /= np.sqrt(32)
# codebook
P = []
for vals, off in ((np.arange(-3, 4), 0.0), (np.arange(-3, 3) + 0.5, 0.5)):
    g = np.array(list(itertools.product(vals, repeat=8)), dtype=np.float32)
    g = g[(np.round(g.sum(1)) % 2 == 0) & ((g ** 2).sum(1) <= 12.0001)]
    P.append(g)
P = np.concatenate(P); nrm = (P ** 2).sum(1); P = P[np.lexsort((np.arange(len(P)), nrm))][:65536]
C = P.astype(np.float32); C2 = (C ** 2).sum(1) / 2
print('codebook: %d points, norm^2 up to %.1f' % (len(C), (C ** 2).sum(1).max()))
def vq(v):  # nearest codeword per 8-vector
    out = np.empty_like(v)
    for i in range(0, len(v), 512):
        b = v[i:i + 512]; out[i:i + 512] = C[np.argmax(b @ C.T - C2, 1)]
    return out
def stage(x, k):  # x: rows x n ; per-row scale = rms * k
    s = np.sqrt((x ** 2).mean(1, keepdims=True)) * k; s[s == 0] = 1
    v = (x / s).reshape(-1, 8).astype(np.float32)
    return (vq(v).reshape(x.shape) * s)
def best_stage(x):
    sub = x[: max(8, x.shape[0] // 8)]; best = None
    for k in (0.55, 0.65, 0.75, 0.85, 0.95, 1.05):
        e = ((sub - stage(sub, k)) ** 2).sum()
        if best is None or e < best[0]: best = (e, k)
    return stage(x, best[1]), best[1]
def kq(tid, x):
    x.astype(np.float32).tofile('in.f32')
    subprocess.run([os.environ.get('KQ', 'kq2'), str(tid), 'in.f32', 'out.f32', str(x.shape[1])], check=True, capture_output=True)
    return np.fromfile('out.f32', dtype=np.float32).reshape(x.shape).astype(np.float64)
gi = Gguf('../urm/merged.gguf'); ent = gi.entries()
for name, ne0, rows_per in (('blk.20.ffn_gate_exps.weight', 2048, 512), ('blk.20.ffn_down_exps.weight', 512, 2048)):
    raw = np.fromfile('%s/%s.bin' % (SL, name), dtype=np.uint16).astype(np.uint32) << 16
    W = raw.view(np.float32).astype(np.float64).reshape(4, rows_per, ne0)
    e = ent[name]; s2 = np.frombuffer(gi.read(e['.in_sum2'][2], e['.in_sum2'][3]), np.float32).reshape(256, ne0)[:4].astype(np.float64)
    cnt = np.frombuffer(gi.read(e['.counts'][2], e['.counts'][3]), np.float32)[:4].astype(np.float64)
    imp = s2 / np.maximum(cnt, 1)[:, None]
    take = rows_per // 4
    res = {}
    for x in range(4):
        w = W[x][rng.choice(rows_per, take, replace=False)]; im = imp[x]
        wr = (w.reshape(take, -1, 32) @ H.T).reshape(w.shape); imr = imp[x]  # importance per rotated input: use block mean
        imr = np.repeat(imp[x].reshape(-1, 32).mean(1), 32)
        def err(q, ref, wt): d = (ref - q) ** 2; return (d.sum() / (ref ** 2).sum(), (d * wt).sum() / ((ref ** 2) * wt).sum())
        for tid, tn, bpw in ((10, 'Q2_K', 2.625), (11, 'Q3_K', 3.4375), (12, 'Q4_K', 4.5), (13, 'Q5_K', 5.5)):
            res.setdefault((tn, bpw), []).append(err(kq(tid, w), w, im))
        s1, k1 = best_stage(wr); res.setdefault(('E8 stage 1 (rotated)', 2.0 + 16 / 2048), []).append(err(s1, wr, imr))
        r = wr - s1; s2_, k2 = best_stage(r); q2 = s1 + s2_
        res.setdefault(('E8 stages 1+2 (rotated)', 4.0 + 32 / 2048), []).append(err(q2, wr, imr))
        print('  expert %d done (k1 %.2f, k2 %.2f)' % (x, k1, k2), flush=True)
    print('== %s (4 experts, %d rows each sampled)' % (name, take))
    for (tn, bpw), v in res.items():
        v = np.array(v); print('  %-26s %6.3f bpw  plain %.6f  weighted %.6f' % (tn, bpw, v[:, 0].mean(), v[:, 1].mean()))
