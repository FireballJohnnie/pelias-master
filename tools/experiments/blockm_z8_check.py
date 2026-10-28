# provenance: original (Block M experiment, sandbox only; needs pt.py and kq from the Block M sandbox)
# Copyright (c) 2026 John Lipka and the Pelias contributors
# Self-review of Q7_PT: compare against an 8-bit layout with the same "zip" scales (fp16 per 256 + 6-bit per 32), same 8.25 bpw
exec(open('pt.py').read().split("meta = {")[0])
src = open('/home/claude/pelias-master/tools/experiments/blockm_pt_tags_k.py').read()
exec(src[src.index('def packed_k'):src.rindex('meta = {')])
import json
meta = {m['name']: m['shape'] for m in json.load(open(S + '/slices.json'))}
for label, name, rot in (('attn_qkv rotated', 'blk.20.attn_qkv.weight', True), ('attn_qkv plain', 'blk.20.attn_qkv.weight', False)):
    w = load(name, meta[name][0])
    if rot: w = (w.reshape(w.shape[0], -1, 32) @ H.T).reshape(w.shape)
    if w.shape[0] > 4096: w = w[rng.choice(w.shape[0], 4096, replace=False)]
    print('== ' + label)
    print('  Q8_0 8.5                          error %.6f' % rel(w, kquant(w, 8)))
    for bits, pw, tag, nm in ((7, 2, 0, '7-bit, zip, no tag'), (7, 2, 2, 'Q7_PT, zip + 2 tag bits'), (8, 2, 0, '8-bit, zip, no tag')):
        q, bpw = packed_k(w, bits, pw, tag); print('  %-32s %.4f bpw  error %.6f' % (nm, bpw, rel(w, q)))
