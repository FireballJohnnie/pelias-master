#!/usr/bin/env python3
"""C3 fragility tags: can the hard tokens be flagged in advance from signals that cost nothing?
Inputs: c3_ref_features.npz (reference margin, entropy, p1, copyable n-grams per scored token), run B's per-token KL
(urm_ZT2, engine format) and the wikitext miss log v1 (35 files). Hard = top 1 % KL of urm_ZT2; consensus hard = top 1 % in
at least half of the 35 files. Thresholds are set on chunks 0-49 and scored on 50-99. Usage: c3_fragility_check.py FEAT.npz KLD"""
import struct, sys
import numpy as np
F = np.load(sys.argv[1]); k = open(sys.argv[2], 'rb').read(); n = struct.unpack_from('<Q', k, 28)[0]
R = np.frombuffer(k, np.dtype([('kld', '<f8'), ('nll', '<f4'), ('nllb', '<f4'), ('pd', '<f4'), ('top', 'u1'), ('pad', 'V3')]), count=n, offset=36)
M = np.load('/home/claude/pelias-master/data/probe_results/c6/miss_log_wikitext_v1.npz')
assert n == len(F['ref_nll']) == M['stuck'].shape[1]
d = np.abs(F['ref_nll'] - R['nllb'])
print('check: reference NLL from the base file vs the .kld nll_base: max |diff| %.2e, mean %.2e' % (d.max(), d.mean()))
kl = R['kld']; hard = kl >= np.quantile(kl, 0.99); cons = M['stuck'].mean(0) >= 0.5
chunk = np.arange(n) // F['n_ctx'].item() * 0 + np.repeat(np.arange(F['n_chunk'].item()), n // F['n_chunk'].item())
tr, te = chunk < 50, chunk >= 50
print('tokens %d; hard %d (consensus %d, overlap %d)' % (n, hard.sum(), cons.sum(), (hard & cons).sum()))
def auc(score, y):  # P(score of a positive > score of a negative)
    r = np.argsort(np.argsort(score)); npos = y.sum(); return (r[y].sum() - npos * (npos - 1) / 2) / (npos * (len(y) - npos))
sig = {'small margin (-margin)': -F['ref_margin'], 'high entropy': F['ref_entropy'], 'low top-1 prob (-p1)': -F['ref_p1'],
       'reference NLL of the true token (oracle-ish)': F['ref_nll']}
print('\n1) Signals (AUROC on chunks 50-99; 0.5 = no signal, 1.0 = perfect)')
for name, s in sig.items():
    print('  %-46s hard %.3f   consensus %.3f' % (name, auc(s[te], hard[te]), auc(s[te], cons[te])))
print('\n2) Flag the riskiest share of tokens by small margin (threshold from chunks 0-49), scored on 50-99:')
for share in (0.02, 0.05, 0.10, 0.20):
    thr = np.quantile(-F['ref_margin'][tr], 1 - share); fl = -F['ref_margin'][te] >= thr
    print('  flag %4.0f %% of tokens -> catches %5.1f %% of hard tokens, %5.1f %% of consensus hard, %5.1f %% of all KL'
          % (100 * fl.mean(), 100 * (fl & hard[te]).sum() / hard[te].sum(), 100 * (fl & cons[te]).sum() / cons[te].sum(), 100 * kl[te][fl].sum() / kl[te].sum()))
print('\n3) Copyable n-grams (an earlier occurrence of the last n tokens exists in the chunk)')
for nn in (1, 2, 3):
    c, ok = F['copy_n%d' % nn], F['copy_ok_n%d' % nn]
    wrong = c & ~ok
    print('  n=%d: copy available %4.1f %% of tokens; copy right %4.1f %%; P(hard | copy right) %.2f %%, P(hard | copy wrong) %.2f %%, P(hard | no copy) %.2f %% (base 1.00 %%)'
          % (nn, 100 * c.mean(), 100 * ok.mean(), 100 * hard[ok].mean(), 100 * hard[wrong].mean(), 100 * hard[~c].mean()))
c3 = F['copy_ok_n3']; ref_agrees = F['ref_top1'] == F['tok_next']
print('  n=3 copy right but the reference top-1 is something else: %d tokens, P(hard) %.1f %%' % ((c3 & ~ref_agrees).sum(), 100 * hard[c3 & ~ref_agrees].mean()))
print('  share of hard tokens with a right n=3 copy: %.1f %%; with a wrong n=3 copy: %.1f %%' % (100 * (hard & c3).sum() / hard.sum(), 100 * (hard & F['copy_n3'] & ~c3).sum() / hard.sum()))
print('\n4) Combined: small margin (flag 10 %) OR a wrong n=2 copy, scored on 50-99')
thr = np.quantile(-F['ref_margin'][tr], 0.90); fl = (-F['ref_margin'] >= thr) | (F['copy_n2'] & ~F['copy_ok_n2'])
print('  flags %.1f %% of tokens -> %.1f %% of hard, %.1f %% of consensus hard, %.1f %% of KL' % (100 * fl[te].mean(), 100 * (fl & hard)[te].sum() / hard[te].sum(),
      100 * (fl & cons)[te].sum() / cons[te].sum(), 100 * kl[te][fl[te]].sum() / kl[te].sum()))
