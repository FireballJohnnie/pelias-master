#!/usr/bin/env python3
"""Step 1 (offline part): does the WikiText-2 hot set hold on chat + code? WikiText-2 picks from the urm_ZT2 route dump
(100 chunks, engine format) vs the per-expert pick counts in Unsloth's own imatrix (unsloth_calibration_Qwen3.6-35B-A3B.txt,
chat + code, 76 chunks x 11008 tokens; the expert entries of merged.gguf). Both are picks (not routing weight)."""
import struct, sys, numpy as np
sys.path.insert(0, '/home/claude/pelias-master/tools'); from imatrix_merge import Gguf
r = open('/mnt/user-data/uploads/pelias-tests/gp/C6/urmzt2_routes.bin', 'rb').read()
_, _, _, L, _ = struct.unpack_from('<5I', r, 8)
RT = np.dtype([('sel', '<i2', 8), ('w', '<f4', 8), ('pl', '<f4'), ('pn', '<f4'), ('pred', '<i2', 8)])
rec = np.frombuffer(r, RT, offset=36).reshape(-1, L); E = 256
W = np.stack([np.bincount(rec['sel'][:, l].ravel().astype(int), minlength=E) for l in range(L)]).astype(float)
g = Gguf('/tmp/claude-0/-home-claude/c7ff3710-2054-5bef-9957-5760a8e25c56/scratchpad/urm/merged.gguf'); EN = g.entries()
C = np.stack([np.frombuffer(g.read(*EN[f'blk.{l}.ffn_gate_exps.weight']['.counts'][2:4]), dtype=np.float32) for l in range(L)]).astype(float)
D = np.stack([np.frombuffer(g.read(*EN[f'blk.{l}.ffn_down_exps.weight']['.counts'][2:4]), dtype=np.float32) for l in range(L)]).astype(float)
print(f'tokens: WikiText {rec.shape[0]}, chat+code {C[0].sum() / 8:.0f}; gate and down counts identical: {np.array_equal(C, D)}')
Wn, Cn = W / W.sum(1, keepdims=True), C / C.sum(1, keepdims=True)
def cover(rank_src, eval_dst, n): return np.take_along_axis(eval_dst, np.argsort(-rank_src, 1)[:, :n], 1).sum(1)
print('top N ranked on WikiText: share of chat+code picks covered (mean / worst layer)  | ranked on chat+code itself (best possible)')
for n in (16, 32, 64, 120, 161):
    a, b = cover(Wn, Cn, n), cover(Cn, Cn, n)
    print(f'  {n:3d}: {a.mean()*100:5.1f} % / {a.min()*100:5.1f} % (layer {a.argmin()})   | {b.mean()*100:5.1f} %   WikiText on itself {cover(Wn, Wn, n).mean()*100:5.1f} %')
rk = lambda x: np.argsort(np.argsort(-x, 1), 1)
sp = [np.corrcoef(rk(Wn)[l], rk(Cn)[l])[0, 1] for l in range(L)]
print(f'rank correlation per layer (Spearman): mean {np.mean(sp):.3f}, min {np.min(sp):.3f} (layer {int(np.argmin(sp))}), first 3 layers {np.round(sp[:3], 3)}')
dead = (Cn < 1e-4) & (Wn < 1e-4); hotc = (Cn > 4 / E) & (Wn < 0.5 / E)
print(f'experts under 0.01 % of picks on both texts: {int(dead.sum())} of {L*E}; hot on chat+code (>4x mean) but cold on WikiText (<0.5x mean): {int(hotc.sum())}')
np.savez('domain_usage_v1.npz', wiki=Wn, chatcode=Cn)
