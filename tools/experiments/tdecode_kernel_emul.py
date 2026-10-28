# CPU emulation of tdecode.cu's integer paths (pack7 -> unpack7 with and without H, dp4a, the 1-bit route, the ZH
# 0.5 fill) against the per-value reference, on random data with every edge (b = 63 so 2b - 127 = -1, q = +-127).
import numpy as np
rng = np.random.default_rng(3)
def pack7(vals):  # 32 values (b in 0..127) -> 28 bytes, value i at bits 7i..7i+6
    o = bytearray(28)
    for i, b in enumerate(vals):
        for t in range(7):
            if (b >> t) & 1: o[(7 * i + t) >> 3] |= 1 << ((7 * i + t) & 7)
    return o
def unpack7(o, H):
    wd = [int.from_bytes(o[4 * w:4 * w + 4], 'little') for w in range(7)] + [0]
    out = []
    for i in range(32):
        bit = 7 * i; w = bit >> 5; sh = bit & 31
        v = wd[w] >> sh
        if sh > 25: v |= (wd[w + 1] << (32 - sh)) & 0xffffffff
        b = v & 127
        out.append(2 * b + ((H >> i) & 1) - 127)
    return out
bad = 0
for trial in range(4000):
    q = rng.integers(-127, 128, 32); q[rng.integers(0, 32)] = -1; q[rng.integers(0, 32)] = 127; q[rng.integers(0, 32)] = -127
    x = rng.integers(-127, 128, 32)
    u = q + 127; b = u >> 1; h = u & 1
    H = int(sum(int(h[i]) << i for i in range(32)))
    o = pack7([int(v) for v in b])
    fold = unpack7(o, H); base = unpack7(o, 0)
    assert all(-128 <= v <= 127 for v in fold)
    pf = int(np.dot(fold, x)); pb = int(np.dot(base, x))
    xu = x + 128; planes = [int(sum(((int(xu[i]) >> k) & 1) << i for i in range(32))) for k in range(8)]
    hc = -128 * bin(H).count('1') + sum(bin(H & planes[k]).count('1') << k for k in range(8))
    ref = int(np.dot(q, x))
    if pf != ref or pb + hc != ref: bad += 1
    # ZH 0.5 fill: 2P = 2 * base + sum(x) equals 2 * sum (2b + 0.5 - 127) x
    if 2 * pb + int(x.sum()) != int(np.dot(2 * (2 * b - 127) + 1, x)): bad += 1
    # odd-grid Q7_ZT ungated values: q odd -> base alone is exact
    qo = 2 * ((q + 127) >> 1) - 127
    if int(np.dot(unpack7(pack7([int(v) for v in (qo + 127) >> 1]), 0), x)) != int(np.dot(qo, x)): bad += 1
print('emulated 4000 blocks: %d mismatches (fold route, 1-bit route, 0.5 fill, odd grid)' % bad)
