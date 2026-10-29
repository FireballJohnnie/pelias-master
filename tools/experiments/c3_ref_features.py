#!/usr/bin/env python3
"""C3 fragility tags (game plan, web v2): per scored token of the wikitext base, the reference's own signals.
Reads the KL-divergence base file only (read-only) and writes, per scored token (order = PELKLD01 order):
  tok_next (the true next token), ref_top1, ref_p1, ref_margin (log p1 - log p2), ref_entropy (nats), ref_nll (of the
  true token, to check against the .kld files' nll_base), copy_n{1,2,3}: an earlier occurrence in the chunk of the last
  n tokens exists (prompt lookup could draft), copy_ok_n{1,2,3}: the token that followed it equals the true next token.
Counts and numbers only; the text is public wikitext. Usage: c3_ref_features.py BASE.bin OUT.npz"""
import struct, sys, time
import numpy as np

base, out = sys.argv[1:3]
f = open(base, 'rb')
assert f.read(8) == b'_logits_'
n_ctx, n_vocab, n_chunk = struct.unpack('<Iii', f.read(12))
toks = np.fromfile(f, np.int32, n_ctx * n_chunk).reshape(n_chunk, n_ctx)
nv = 2 * ((n_vocab + 1) // 2) + 4
first = n_ctx // 2; nrec = n_ctx - 1 - first
data0 = 20 + 4 * n_ctx * n_chunk
N = n_chunk * nrec
R = {k: np.zeros(N, np.float32) for k in ('ref_p1', 'ref_margin', 'ref_entropy', 'ref_nll')}
R['tok_next'] = np.zeros(N, np.int32); R['ref_top1'] = np.zeros(N, np.int32)
for n in (1, 2, 3):
    R['copy_n%d' % n] = np.zeros(N, bool); R['copy_ok_n%d' % n] = np.zeros(N, bool)
t0 = time.time()
for c in range(n_chunk):
    blk = np.fromfile(base, np.uint16, nrec * nv, offset=data0 + 2 * c * nrec * nv).reshape(nrec, nv)
    sc = blk[:, 0:2].copy().view(np.float32)[:, 0]; mn = blk[:, 2:4].copy().view(np.float32)[:, 0]
    lp = blk[:, 4:4 + n_vocab].astype(np.float32) * sc[:, None] + mn[:, None]
    T = toks[c]
    for j in range(nrec):
        i = c * nrec + j; pos = first + j; nxt = T[pos + 1]
        row = lp[j]
        top2 = np.argpartition(row, -2)[-2:]; a, b = (top2[0], top2[1]) if row[top2[0]] > row[top2[1]] else (top2[1], top2[0])
        live = row > -16.0; p = np.exp(row[live])
        R['ref_p1'][i] = np.exp(row[a]); R['ref_margin'][i] = row[a] - row[b]; R['ref_top1'][i] = a
        R['ref_entropy'][i] = -(p * row[live]).sum(); R['ref_nll'][i] = -row[nxt]; R['tok_next'][i] = nxt
        for n in (1, 2, 3):
            suf = T[pos - n + 1:pos + 1]
            # earlier occurrences of the suffix inside the chunk, ending before pos
            hit = None
            for e in range(pos - 1, n - 2, -1):
                if np.array_equal(T[e - n + 1:e + 1], suf): hit = e; break
            if hit is not None:
                R['copy_n%d' % n][i] = True; R['copy_ok_n%d' % n][i] = T[hit + 1] == nxt
    if c % 10 == 9: print('chunk %d/%d, %.0f s' % (c + 1, n_chunk, time.time() - t0), flush=True)
np.savez_compressed(out, n_ctx=n_ctx, n_vocab=n_vocab, n_chunk=n_chunk, **R)
print('wrote %s: %d tokens, %.0f s' % (out, N, time.time() - t0))
