#!/usr/bin/env python3
"""Author (2026-10-29 00:08): pull the state codes of the worst chunks (run B, urm_ZT2, engine format, wikitext 100 chunks).
Worst = the 10 chunks with the highest mean per-token KL. Writes their codes (scored tokens x 40 layers x 320 bytes:
256 sign bytes + 64 log-energy bytes), per-token KL and a top-1 % flag to OUT.npz, and a summary to OUT.json.
Reads the PELCODE1 / PELKLD01 files only. Usage: worst_chunk_codes.py CODES.bin KLD.kld OUT_STEM"""
import json, struct, sys
import numpy as np
cpath, kpath, out = sys.argv[1:4]
b = np.memmap(cpath, np.uint8, 'r')
assert bytes(b[:8]) == b'PELCODE1'
n_ctx, ch, sc, nl, ne, cb = struct.unpack('<6I', bytes(b[8:32]))
T = ch * sc
codes = b[40:40 + T * nl * cb].reshape(T, nl, cb)
k = open(kpath, 'rb').read(); n = struct.unpack_from('<Q', k, 28)[0]
kl = np.frombuffer(k, np.dtype([('kld', '<f8'), ('nll', '<f4'), ('nllb', '<f4'), ('pd', '<f4'), ('top', 'u1'), ('pad', 'V3')]), count=n, offset=36)['kld']
assert n == T
per = kl.reshape(ch, sc)
mean = per.mean(1); worst = np.argsort(-mean)[:10]
q99 = np.quantile(kl, 0.99)
rows = []
for c in worst:
    t = per[c]
    rows.append({'chunk': int(c), 'mean_kl': float(t.mean()), 'share_of_total_kl': float(t.sum() / kl.sum()),
                 'max_token_kl': float(t.max()), 'max_token_pos': int(t.argmax()), 'top1pct_tokens': int((t >= q99).sum())})
idx = np.concatenate([np.arange(c * sc, (c + 1) * sc) for c in worst])
np.savez_compressed(out + '.npz', chunks=worst.astype(np.int32), codes=np.asarray(codes[idx]), kl=kl[idx].astype(np.float32),
                    top1pct=(kl[idx] >= q99), layout=np.array([sc, nl, ne, cb], np.int32))
summ = {'source': 'run B, urm_ZT2, engine format, wikitext 100 chunks x %d scored tokens' % sc, 'n_ctx': n_ctx,
        'file_mean_kl': float(kl.mean()), 'top1pct_threshold': float(q99), 'median_chunk_mean_kl': float(np.median(mean)),
        'worst10_share_of_total_kl': float(per[worst].sum() / kl.sum()), 'worst_chunks': rows}
json.dump(summ, open(out + '.json', 'w'), indent=1)
print(json.dumps(summ, indent=1))
