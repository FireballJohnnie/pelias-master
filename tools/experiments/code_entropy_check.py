#!/usr/bin/env python3
"""rANS on bus transfers (author 01:04): how many bits do the stored codes of real tensors actually carry?
Reads raw blocks of a few tensors straight from a GGUF (read-only, rows sampled evenly) and measures, per tensor, the
order-0 entropy of the quantized codes (what a static rANS table per tensor would reach) and a per-256-block adaptive
bound (entropy with a fresh table per superblock, an optimistic cap). Scales and headers are counted as stored.
Types: Q4_K (12), Q5_K (13), Q6_K (14), Q7_ZT (202). Usage: code_entropy_check.py MODEL.gguf [rows_per_tensor]"""
import struct, sys
import numpy as np
path = sys.argv[1]; want = int(sys.argv[2]) if len(sys.argv) > 2 else 256
f = open(path, 'rb'); rd = lambda fmt: struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
def skip(t):
    S = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in S: f.seek(S[t], 1)
    elif t == 8: n, = rd('Q'); f.seek(n, 1)
    elif t == 9:
        at, n = rd('IQ')
        for _ in range(n): skip(at)
assert f.read(4) == b'GGUF'
v, nt, nkv = rd('IQQ'); align = 32
for _ in range(nkv):
    kl, = rd('Q'); k = f.read(kl).decode(); t, = rd('I')
    if k == 'general.alignment' and t == 4: align, = rd('I')
    else: skip(t)
info = {}
for _ in range(nt):
    nl, = rd('Q'); nm = f.read(nl).decode(); nd, = rd('I'); ne = list(rd('Q' * nd)); ty, off = rd('IQ'); info[nm] = (ne, ty, off)
data0 = (f.tell() + align - 1) // align * align
BB = {12: 144, 13: 176, 14: 210, 202: 241}
def codes(ty, blk):  # (nblocks, bytes) -> (nblocks, 256) integer codes, code bits
    if ty == 12:
        qs = blk[:, 16:144]; lo = qs & 15; hi = qs >> 4
        c = np.concatenate([np.concatenate([lo[:, 32 * j:32 * j + 32], hi[:, 32 * j:32 * j + 32]], 1) for j in range(4)], 1); return c, 4
    if ty == 13:
        qh = blk[:, 16:48]; qs = blk[:, 48:176]; out = []
        for j in range(4):
            lo = qs[:, 32 * j:32 * j + 32] & 15; hi = qs[:, 32 * j:32 * j + 32] >> 4
            out += [lo | (((qh >> (2 * j)) & 1) << 4), hi | (((qh >> (2 * j + 1)) & 1) << 4)]
        return np.concatenate(out, 1), 5
    if ty == 14:
        ql = blk[:, 0:128]; qh = blk[:, 128:192]; out = []
        for h in range(2):
            l = ql[:, 64 * h:64 * h + 64]; q = qh[:, 32 * h:32 * h + 32]
            out += [(l[:, :32] & 15) | ((q & 3) << 4), (l[:, 32:] & 15) | (((q >> 2) & 3) << 4),
                    (l[:, :32] >> 4) | (((q >> 4) & 3) << 4), (l[:, 32:] >> 4) | (((q >> 6) & 3) << 4)]
        return np.concatenate(out, 1), 6
    if ty == 202:
        w = blk[:, 9:233].reshape(len(blk), 32, 7).astype(np.uint64)  # 32 groups of 8 values in 7 bytes
        word = sum(w[:, :, b] << np.uint64(8 * b) for b in range(7))
        c = np.stack([(word >> np.uint64(7 * k)) & np.uint64(127) for k in range(8)], 2).reshape(len(blk), 256)
        return c.astype(np.int64), 7
def H(c, nsym):
    p = np.bincount(c.ravel(), minlength=nsym).astype(float); p = p[p > 0] / p.sum(); return -(p * np.log2(p)).sum()
names = ['blk.20.ffn_gate_exps.weight', 'blk.20.ffn_up_exps.weight', 'blk.20.ffn_down_exps.weight', 'blk.5.ffn_down_exps.weight',
         'blk.35.ffn_gate_exps.weight', 'blk.20.attn_qkv.weight', 'blk.23.attn_q.weight', 'blk.20.ffn_down_shexp.weight', 'output.weight']
print('%-30s %-6s %5s | %8s %8s %8s | %s' % ('tensor', 'type', 'bits', 'H order0', 'H /256blk', 'bpw now', 'rANS size vs stored (order 0 / adaptive cap)'))
for nm in names:
    if nm not in info: print(nm, 'absent'); continue
    ne, ty, off = info[nm]
    if ty not in BB: print(nm, 'type', ty, 'not handled'); continue
    rowb = ne[0] // 256 * BB[ty]; nrows = int(np.prod(ne[1:]))
    idx = np.unique(np.linspace(0, nrows - 1, min(want, nrows)).astype(np.int64))
    raw = []
    for r in idx: f.seek(data0 + off + int(r) * rowb); raw.append(np.frombuffer(f.read(rowb), np.uint8))
    blk = np.concatenate(raw).reshape(-1, BB[ty]).astype(np.uint16)
    c, bits = codes(ty, blk); h0 = H(c, 1 << bits)
    ha = np.mean([H(c[i], 1 << bits) for i in range(0, len(c), max(1, len(c) // 400))])
    bpw = BB[ty] * 8 / 256; side = bpw - bits
    print('%-30s %-6s %5d | %8.3f %8.3f %8.3f | %.1f %% / %.1f %%' % (nm, {12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 202: 'Q7_ZT'}[ty], bits, h0, ha, bpw,
          100 * (side + h0) / bpw, 100 * (side + ha) / bpw))
