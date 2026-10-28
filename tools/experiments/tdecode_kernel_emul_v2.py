# CPU emulation of version 2's integer paths: a lane's 112-byte chunk (4 blocks of 32 seven-bit values, value i at bits
# 7i), funnel-shift extraction, sum (2b - 127) x = 2 dp4a(b, x) - 127 sum x, H as spread nibble bytes into dp4a, H by
# popcount, the ZH 0.5 fill 2P + sum x; against the per-value reference.
import numpy as np
rng = np.random.default_rng(5)
M = 0xffffffff
def funnel_r(lo, hi, n): return ((hi << 32 | lo) >> n) & M
bad = 0
for trial in range(1500):
    q = rng.integers(-127, 128, 128); q[rng.integers(0, 128, 6)] = [-127, 127, -1, 0, 1, -2]
    x = rng.integers(-127, 128, 128)
    u = q + 127; b = u >> 1; h = u & 1
    chunk = bytearray(112)
    for i, bv in enumerate(b):
        for t in range(7):
            if (int(bv) >> t) & 1: chunk[(7 * i + t) >> 3] |= 1 << ((7 * i + t) & 7)
    wd = [int.from_bytes(chunk[4 * w:4 * w + 4], 'little') for w in range(28)] + [0]
    for jj in range(4):
        xs = x[32 * jj:32 * jj + 32]; sx = int(xs.sum())
        bb = [funnel_r(wd[(7 * (32 * jj + i)) >> 5], wd[((7 * (32 * jj + i)) >> 5) + 1], (7 * (32 * jj + i)) & 31) & 127 for i in range(32)]
        pb = int(np.dot(bb, xs))
        H = sum(int(h[32 * jj + i]) << i for i in range(32))
        # H as spread nibble bytes
        hx1 = 0
        for g in range(8):
            nib = (H >> (4 * g)) & 0xf; spread = (nib * 0x00204081) & 0x01010101
            hx1 += sum(((spread >> (8 * t)) & 1) * int(xs[4 * g + t]) for t in range(4))
        xu = xs + 128; planes = [sum(((int(xu[i]) >> k) & 1) << i for i in range(32)) for k in range(8)]
        hx2 = -128 * bin(H).count('1') + sum(bin(H & planes[k]).count('1') << k for k in range(8))
        P = 2 * pb - 127 * sx; ref = int(np.dot(q[32 * jj:32 * jj + 32], xs))
        if P + hx1 != ref or P + hx2 != ref: bad += 1
        if 2 * P + sx != int(np.dot(2 * (2 * b[32 * jj:32 * jj + 32] - 127) + 1, xs)): bad += 1
print('version 2 emulation, 6000 blocks: %d mismatches' % bad)
