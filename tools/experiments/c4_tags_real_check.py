# C4 (author's bit-tag matrix at 3 bits; Fable 19:1x plane form): 3-bit values with a 2-bit grid tag per 8 values (one tag
# byte per 32-block) and zip scales (6-bit level per 32 + fp16 per 256), on the REAL blk.20 experts (first 4, rows
# sampled, stored unrotated as the engine keeps them), against Q3_K / Q4_K from Pelias's C++ quantizer (kq2).
# Run from scratchpad/blk_m (kq2 there, imatrix at ../urm/merged.gguf). Targets (Fable, K-line): 3.5 bpw <= the K-line at
# 3.5, 3.78 bpw (with the H gate) <= the K-line at 3.78. Error = relative squared error, plain and importance-weighted.
import os, sys, subprocess, numpy as np
sys.path.insert(0, '/home/claude/pelias-master/tools'); from imatrix_merge import Gguf
SL = '/mnt/user-data/uploads/pelias-tests/gp/M/slices'; rng = np.random.default_rng(0)
def kq(tid, x):
    x.astype(np.float32).tofile('in.f32')
    subprocess.run([os.environ.get('KQ', './kq2'), str(tid), 'in.f32', 'out.f32', str(x.shape[1])], check=True, capture_output=True)
    return np.fromfile('out.f32', dtype=np.float32).reshape(x.shape).astype(np.float64)
def uni(n): return np.arange(n) - (n - 1) / 2
def lg(r, n=8): m = r ** np.arange(n // 2); return np.concatenate([-m[::-1], m])
def add(w): # additive plane grid (Fable): value = w0 b0 + w1 b1 + w2 b2, centred
    v = np.array([w[0]*(i&1) + w[1]*((i>>1)&1) + w[2]*((i>>2)&1) for i in range(8)], float); return np.sort(v - v.mean())
def nest(g):  # 16-level grid whose even points are g (H refines a word): midpoints (uniform) or geometric means (log)
    out = []
    for a, b in zip(g[:-1], g[1:]):
        out += [a, (np.sign(a) * np.sqrt(abs(a * b)) if a * b > 0 else (a + b) / 2)]
    out.append(g[-1]); out.append(g[-1] + (g[-1] - out[-2]))  # one extra level past the top, as 2b+h does
    return np.array(sorted(out))
# every grid scaled to a top level of 1, so one block scale serves all four (fix 19:4x: the first run scaled every grid by
# the largest top level among the four, which squeezed the uniform grid and made the tags look worse than no tags)
nm = lambda g: g / g.max()
GRIDS = {'lut4': [nm(uni(8)), nm(lg(1.6)), nm(lg(2.2)), nm(uni(8)) * 0.6],
         'add4': [nm(add((1, 2, 4))), nm(add((1, 1.5, 3))), nm(add((1, 2, 3))), nm(add((1, 2, 4))) * 0.6],
         'uni1': [nm(uni(8))],
         # 4-bit with the same tag byte (open-mind follow-up 19:4x: like for like against Q4_K at 4.5 bpw)
         'lut4_16': [nm(uni(16)), nm(lg(1.3, 16)), nm(lg(1.45, 16)), nm(uni(16)) * 0.6],
         'uni16': [nm(uni(16))]}
FS = np.linspace(0.35, 1.25, 19)
def qblocks(x, grids, gate=0):
    """x: (B, 32) blocks. Each 8-group picks a grid (tag); one scale per block, zip-quantized (6-bit level of the
    256-superblock max). gate: per 8 blocks (one superblock) this many get the nested 16-level grid of their tags (H)."""
    B = x.shape[0]; G = len(grids); amax = np.abs(x).max(1, keepdims=True) + 1e-30
    best_e = np.full(B, np.inf); best_q = np.zeros_like(x); best_s = np.zeros(B)
    for f in FS:
        s = amax[:, 0] * f / max(g.max() for g in grids)
        eg = np.empty((G, B, 4)); qg = np.empty((G, B, 32))
        for k, g in enumerate(grids):
            q = g[np.abs(x[:, :, None] / s[:, None, None] - g).argmin(2)] * s[:, None]
            qg[k] = q; eg[k] = ((x - q) ** 2).reshape(B, 4, 8).sum(2)
        tag = eg.argmin(0); e = np.take_along_axis(eg, tag[None], 0)[0].sum(1)
        better = e < best_e
        if better.any():
            q = np.take_along_axis(qg.reshape(G, B, 4, 8), tag[None, :, :, None].repeat(8, 3), 0)[0].reshape(B, 32)
            best_e[better] = e[better]; best_q[better] = q[better]; best_s[better] = s[better]
            best_tag = tag if 'best_tag' not in dir() else np.where(better[:, None], tag, best_tag)
    # zip scales: 6-bit level per block of the superblock max (fp16 superblock scale), then requantize with the stored scale
    sb = best_s.reshape(-1, 8); d = sb.max(1, keepdims=True) / 63; lvl = np.clip(np.round(sb / d), 1, 63); sq = (lvl * d).reshape(-1)
    tags = best_tag
    q = np.empty_like(x)
    for k, g in enumerate(grids):
        qk = g[np.abs(x[:, :, None] / sq[:, None, None] - g).argmin(2)] * sq[:, None]
        m = np.repeat(tags == k, 8, 1); q[m] = qk[m]
    if gate:
        qh = np.empty_like(x)
        for k, g in enumerate(grids):
            gn = nest(g); qk = gn[np.abs(x[:, :, None] / sq[:, None, None] - gn).argmin(2)] * sq[:, None]
            m = np.repeat(tags == k, 8, 1); qh[m] = qk[m]
        gain = (((x - q) ** 2) - ((x - qh) ** 2)).sum(1).reshape(-1, 8)
        pick = np.zeros_like(gain, bool); idx = np.argsort(-gain, 1)[:, :gate]; np.put_along_axis(pick, idx, True, 1)
        q = np.where(pick.reshape(-1)[:, None], qh, q)
    return q
gi = Gguf('../urm/merged.gguf'); ent = gi.entries()
for name, ne0, rows_per in (('blk.20.ffn_gate_exps.weight', 2048, 512), ('blk.20.ffn_down_exps.weight', 512, 2048)):
    raw = np.fromfile('%s/%s.bin' % (SL, name), dtype=np.uint16).astype(np.uint32) << 16
    W = raw.view(np.float32).astype(np.float64).reshape(4, rows_per, ne0)
    e = ent[name]; s2 = np.frombuffer(gi.read(e['.in_sum2'][2], e['.in_sum2'][3]), np.float32).reshape(256, ne0)[:4].astype(np.float64)
    cnt = np.frombuffer(gi.read(e['.counts'][2], e['.counts'][3]), np.float32)[:4].astype(np.float64)
    imp = s2 / np.maximum(cnt, 1)[:, None]; take = rows_per // 4; res = {}
    def err(q, ref, wt): d = (ref - q) ** 2; return (d.sum() / (ref ** 2).sum(), (d * wt).sum() / ((ref ** 2) * wt).sum())
    for xx in range(4):
        w = W[xx][rng.choice(rows_per, take, replace=False)]; im = np.broadcast_to(imp[xx], w.shape)
        for tid, tn, bpw in ((11, 'Q3_K', 3.4375), (12, 'Q4_K', 4.5)):
            res.setdefault((tn, bpw), []).append(err(kq(tid, w), w, im))
        blocks = w.reshape(-1, 32)
        for lab, grids, gate, bpw in (('3-bit uniform, zip scales', GRIDS['uni1'], 0, 3.25 + 1 / 256),
                                      ('C4 LUT grids, tag byte', GRIDS['lut4'], 0, 3.5 + 1 / 256),
                                      ('C4 additive grids, tag byte', GRIDS['add4'], 0, 3.5 + 1 / 256),
                                      ('C4 LUT grids + H gate 2/8', GRIDS['lut4'], 2, 3.75 + 9 / 256),
                                      ('C4 additive + H gate 2/8', GRIDS['add4'], 2, 3.75 + 9 / 256),
                                      ('3-bit uniform + H gate 2/8', GRIDS['uni1'], 2, 3.5 + 9 / 256),
                                      ('4-bit uniform, zip scales', GRIDS['uni16'], 0, 4.25 + 1 / 256),
                                      ('C4-4 LUT grids, tag byte', GRIDS['lut4_16'], 0, 4.5 + 1 / 256)):
            res.setdefault((lab, bpw), []).append(err(qblocks(blocks, grids, gate).reshape(w.shape), w, im))
        print('  expert %d done' % xx, flush=True)
    print('== %s (4 experts, %d rows each)' % (name, take))
    q3 = np.mean([v[0] for v in res[('Q3_K', 3.4375)]]); q4 = np.mean([v[0] for v in res[('Q4_K', 4.5)]])
    kline = lambda b: np.exp(np.log(q3) + (np.log(q4) - np.log(q3)) * (b - 3.4375) / (4.5 - 3.4375))
    for (tn, bpw), v in sorted(res.items(), key=lambda t: t[0][1]):
        v = np.array(v); kl = kline(bpw)
        print('  %-30s %6.3f bpw  plain %.6f  weighted %.6f  | K-line at this bpw %.6f -> %s' % (tn, bpw, v[:, 0].mean(), v[:, 1].mean(), kl,
              'BEATS' if v[:, 0].mean() < kl * 0.999 else ('ties' if v[:, 0].mean() < kl * 1.02 else 'above')))
