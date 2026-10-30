#!/usr/bin/env python3
"""#3 offline: the cold floor (Q4_TG2 vs IQ4_XS vs Q4_K) on the blk.20 expert slices (experts 0-3, bf16, own importance),
and the output-norm shrink of each type with the one-scalar-per-expert fix. Diagonal proxy: the input is taken as
uncorrelated with the importance as its variance, so E||Wx||^2 = sum imp w^2 and the best scalar a = sum imp w wq / sum imp wq^2."""
import os, sys, subprocess, numpy as np
sys.path.insert(0, '/home/claude/pelias-master/tools'); from imatrix_merge import Gguf
HERE = os.path.dirname(os.path.abspath(__file__)); KQ = HERE + '/kqw'
SL = '/mnt/user-data/uploads/pelias-tests/gp/M/slices'
g = Gguf('/tmp/claude-0/-home-claude/c7ff3710-2054-5bef-9957-5760a8e25c56/scratchpad/urm/merged.gguf'); E = g.entries()
def kq(tid, x, imp):
    a, b, c = HERE + '/fn_in.f32', HERE + '/fn_out.f32', HERE + '/fn_imp.f32'
    x.astype(np.float32).tofile(a); imp.astype(np.float32).tofile(c)
    r = subprocess.run([KQ, str(tid), a, b, str(x.shape[1]), c], capture_output=True, text=True)
    if r.returncode != 0: return None, None
    return np.fromfile(b, dtype=np.float32).reshape(x.shape).astype(np.float64), float(r.stdout.split(',')[1].split()[0])
TYPES = [(23, 'IQ4_XS'), (204, 'Q4_TG2'), (12, 'Q4_K'), (13, 'Q5_K'), (14, 'Q6_K')]
for name, n_in, n_out in (('blk.20.ffn_gate_exps.weight', 2048, 512), ('blk.20.ffn_down_exps.weight', 512, 2048)):
    raw = np.fromfile(f'{SL}/{name}.bin', dtype=np.uint16).astype(np.uint32) << 16
    Wt = raw.view(np.float32).astype(np.float64).reshape(4, n_out, n_in); e = E[name]
    s2 = np.frombuffer(g.read(e['.in_sum2'][2], e['.in_sum2'][3]), dtype=np.float32).astype(np.float64).reshape(-1, n_in)
    cnt = np.frombuffer(g.read(e['.counts'][2], e['.counts'][3]), dtype=np.float32).astype(np.float64)
    print(f'\n{name} (experts 0-3): error = sum imp (w-wq)^2 / sum imp w^2, pooled; norm = sum imp wq^2 / sum imp w^2 per expert')
    q4 = None
    for tid, tn in TYPES:
        num = den = num_a = 0.0; norms, alphas = [], []; bpw = None
        for x in range(4):
            w = Wt[x]; imp = s2[x] / max(cnt[x], 1.0)
            if imp.sum() == 0: imp = np.ones(n_in)
            wq, bpw = kq(tid, w, imp)
            if wq is None: break
            I = imp[None, :]
            den += (I * w * w).sum(); num += (I * (w - wq) ** 2).sum()
            a = (I * w * wq).sum() / (I * wq * wq).sum(); alphas.append(a); norms.append((I * wq * wq).sum() / (I * w * w).sum())
            num_a += (I * (w - a * wq) ** 2).sum()
        if wq is None: print(f'  {tn:7s} not available in kqw'); continue
        err = num / den; q4 = err if tn == 'Q4_K' else q4
        print(f'  {tn:7s} {bpw:.4f} bpw  error {err:.6f}  norm {np.round(norms, 5)}  best scalar {np.round(alphas, 5)}  error with it {num_a/den:.6f} ({(1 - num_a/num)*100:.2f} % less)')
