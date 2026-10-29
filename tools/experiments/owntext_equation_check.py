#!/usr/bin/env python3
"""Run C follow-up (author 22:38-22:40: the governing equations sit in the records 30-40 % of the way through the dataset
file). On the author's PC only; prints counts only, never text. Re-does owntext_select.py's choice exactly (same seed,
same order, same overlap rule), maps every scored token of the 100-chunk KLD runs back to its source record, and asks:
do the stuck tokens (top 1 % KL of each file) come from the equation band more often than its share of the text?
usage: owntext_equation_check.py DATASET.jsonl CALIB.txt PELIAS MODEL.gguf TARGET OT_DIR"""
import json, os, random, struct, subprocess, sys, tempfile
import numpy as np
ds, calib, pelias, model, target, otd = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
def toks(text):
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(text); p = f.name
    try:
        r = subprocess.run([pelias, 'tokenize', '--model', model, '--header-only', '--file', p, '--no-special'], check=True, capture_output=True, text=True)
    finally:
        os.unlink(p)
    return json.loads(r.stdout)
def shingles(ids, n=64):
    return {hash(tuple(ids[i:i + n])) for i in range(0, max(0, len(ids) - n + 1))}
recs = [json.loads(l) for l in open(ds, encoding='utf-8')]
for i, r in enumerate(recs): r['_pos'] = i / len(recs)
def render(r):
    return "\n\n".join("%s:\n%s" % (m['role'], m['content']) for m in r['messages']
                       if m.get('role') != 'system' and isinstance(m.get('content'), str) and m['content'])
rng = random.Random(20261028)
ev = [r for r in recs if r.get('split') == 'eval']; tr = [r for r in recs if r.get('split') != 'eval']
rng.shuffle(ev); rng.shuffle(tr)
cal = shingles(toks(open(calib, encoding='utf-8').read()))
chosen, n_tok = [], 0
for pool in (ev, tr):
    for r in pool:
        if n_tok >= target: break
        t = render(r)
        if not t: continue
        ids = toks(t)
        if shingles(ids) & cal: continue
        chosen.append((r['_pos'], len(ids), t)); n_tok += len(ids)
text = "\n\n\n".join(c[2] for c in chosen)
if text.encode('utf-8') != open(os.path.join(otd, 'owntext_eval.txt'), 'rb').read():
    sys.exit("STOP: the re-done choice does not reproduce owntext_eval.txt")
whole = len(toks(text)); parts = sum(c[1] for c in chosen); sep = (whole - parts) / max(1, len(chosen) - 1)
# token index -> record band (cumulative counts, separators spread evenly)
band = np.zeros(whole, bool); at = 0.0
for pos, n, _ in chosen:
    a, b = int(round(at)), int(round(at + n)); band[a:min(b, whole)] = 0.30 <= pos < 0.40; at += n + sep
print("text reproduced exactly; records %d (%d in the 30-40 %% band); tokens %d (separators %.2f tokens each)" % (len(chosen), sum(0.30 <= c[0] < 0.40 for c in chosen), whole, sep))
def rd(p):
    b = open(p, 'rb').read(); n = struct.unpack_from('<Q', b, 28)[0]
    return np.frombuffer(b, np.dtype([('kld', '<f8'), ('nll', '<f4'), ('nllb', '<f4'), ('pd', '<f4'), ('top', 'u1'), ('pad', 'V3')]), count=n, offset=36)
idx = np.array([c * 512 + 257 + i for c in range(100) for i in range(255)])  # the token each scored position predicts
idx = np.minimum(idx, whole - 1); inband = band[idx]
print("scored tokens from the band: %.1f %%" % (inband.mean() * 100))
for name in ('zt2_x8', 'z_x8', 'mtp_a8'):
    k = rd(os.path.join(otd, name + '.kld'))['kld']; st = k >= np.quantile(k, 0.99)
    print("%-7s stuck 1 %% from the band: %.1f %% (share of text %.1f %%); mean KL in band %.5f vs outside %.5f; band holds %.1f %% of the KL"
          % (name, inband[st].mean() * 100, inband.mean() * 100, k[inband].mean(), k[~inband].mean(), k[inband].sum() / k.sum() * 100))
