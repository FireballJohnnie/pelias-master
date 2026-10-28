# S1 (game plan run list): the 7-bit rungs against Q8_Z on the real blk.20 attn_qkv (bf16, rotated per 32 as in the file),
# importance-weighted with the merged matrix urm used, plus the exactness checks (J1 split, popcount H) and the fp16
# partial-sum emulation (Fable check a). Q8_0 and Q8_Z come from Pelias's own C++ quantizer (format-match rule).
import os, sys, subprocess, numpy as np
sys.path.insert(0, '/home/claude/pelias-master/tools'); from imatrix_merge import Gguf
SL = '/mnt/user-data/uploads/pelias-tests/gp/M/slices'; HERE = os.path.dirname(os.path.abspath(__file__))
KQ = os.environ.get('KQ', HERE + '/kq2'); IM = os.environ.get('IMAT', HERE + '/merged.gguf'); rng = np.random.default_rng(0)
H = np.array([[1.]])
for _ in range(5): H = np.block([[H, H], [H, -H]])
H /= np.sqrt(32)
raw = np.fromfile(SL + '/blk.20.attn_qkv.weight.bin', dtype=np.uint16).astype(np.uint32) << 16
w = raw.view(np.float32).astype(np.float64).reshape(-1, 2048)            # 8192 rows x 2048 inputs
w = (w.reshape(w.shape[0], -1, 32) @ H.T).reshape(w.shape)               # rotated per 32, as stored in urm
g = Gguf(IM); e = g.entries()['blk.20.attn_qkv.weight']
s2 = np.frombuffer(g.read(e['.in_sum2'][2], e['.in_sum2'][3]), dtype=np.float32).astype(np.float64)
cnt = np.frombuffer(g.read(e['.counts'][2], e['.counts'][3]), dtype=np.float32).astype(np.float64)
imp = s2 / max(cnt.max(), 1.0); assert imp.size == 2048
def kq(tid, x):
    a, b = HERE + '/in.f32', HERE + '/out.f32'; x.astype(np.float32).tofile(a)
    r = subprocess.run([KQ, str(tid), a, b, str(x.shape[1])], capture_output=True)
    assert r.returncode == 0, r.stderr
    return np.fromfile(b, dtype=np.float32).reshape(x.shape).astype(np.float64)
def errs(q):
    d2 = (w - q) ** 2
    return float(d2.sum() / (w ** 2).sum()), float((d2 * imp).sum() / ((w ** 2) * imp).sum())
def zip_n(bits):  # Q8_Z's design with `bits`-bit values: fp16 d per 256 (= top / qmax), 6-bit level per 32, search 70-100 %
    qmax = 2 ** (bits - 1) - 1; b = w.reshape(w.shape[0], -1, 8, 32)
    top = np.abs(b).max((-1, -2), keepdims=True); d = (top / qmax).astype(np.float16).astype(np.float64); d[d == 0] = 1e-30
    amax = np.abs(b).max(-1, keepdims=True); ideal = amax / qmax / d * 63
    best = None
    for c in np.linspace(0.70, 1.0, 13):
        lv = np.clip(np.ceil(ideal * c), 1, 63); s = d * lv / 63
        q = np.clip(np.round(b / s), -qmax, qmax) * s; er = ((b - q) ** 2).sum(-1)
        if best is None: out, best = q, er
        else: t = er < best; best = np.where(t, er, best); out = np.where(t[..., None], q, out)
    return out.reshape(w.shape)
rows = []
q80 = kq(8, w); rows.append(('Q8_0 (C++)', 8.5, q80))
q8z = kq(200, w); rows.append(('Q8_Z (C++)', 8.25, q8z))
rows.append(('Q8_Z design, numpy (format check)', 8.25, zip_n(8)))
rows.append(('Q7_P tight: 7-bit, Q8_Z scales', 7.25, zip_n(7)))
rows.append(('Q6 tight: 6-bit, Q8_Z scales', 6.25, zip_n(6)))
print('blk.20.attn_qkv, rotated, all 8192 rows; error = sum (w-q)^2 / sum w^2, weighted = same with the importance per input')
base = None
for n, bpw, q in rows:
    pe, we = errs(q)
    if base is None: base = we
    print('  %-36s %5.2f bpw  plain %.6f  weighted %.6f  (x %.2f of Q8_0)' % (n, bpw, pe, we, we / base))
print('  Q7_PT (7 + 7 + 2 tag bits per word, 8.25 bpw): earlier MEASURED 0.000040 plain, beaten by Q8_Z at the same bits; bit planes do not change values')
# --- exactness: J1 split (base plane = high nibble, refinement = low nibble) and popcount H
q = rng.integers(-127, 128, size=(64, 2048)); x = rng.integers(-127, 128, size=2048)
hi = np.floor_divide(q, 16); lo = q - 16 * hi
whole = q @ x; split = 16 * (hi @ x) + (lo @ x)
print('J1 split dot (int, 64 rows x 2048): max |whole - split| =', int(np.abs(whole - split).max()), '(0 = exact)')
hb = rng.integers(0, 2, size=(64, 2048)); xu = (x + 128).astype(np.int64)
pc = sum((1 << k) * ((hb & ((xu >> k) & 1)[None, :]).sum(1)) for k in range(8)) - 128 * hb.sum(1)
print('popcount H: max |sum H x - (sum_k 2^k popcount(H AND x_k) - 128 popcount(H))| =', int(np.abs(hb @ x - pc).max()), '(0 = exact)')
# --- fp16 partial sums per 256 flushed to fp32 (texture path), on Q8_Z weights, activations int8 per 32 as the engine
xa = rng.standard_normal((16, 2048)) * np.sqrt(imp)[None, :]             # activation scale per input from the matrix
xa = (xa.reshape(16, -1, 32) @ H.T).reshape(16, 2048)                    # rotated input, as the file expects
ref = xa @ q8z.T
p = (q8z.reshape(8192, 8, 256)[None] * xa.reshape(16, 1, 8, 256)).astype(np.float16)
acc = np.zeros(p.shape[:3], np.float16)
for i in range(256): acc = (acc + p[..., i]).astype(np.float16)
y16 = acc.astype(np.float32).sum(-1).astype(np.float64)
qerr = xa @ (q8z - w).T
print('fp16 partial sums per 256: output error rms / output rms = %.5f; Q8_Z weight error on the same outputs = %.5f' % (
    np.sqrt(((y16 - ref) ** 2).mean() / (ref ** 2).mean()), np.sqrt((qerr ** 2).mean() / ((xa @ w.T) ** 2).mean())))
