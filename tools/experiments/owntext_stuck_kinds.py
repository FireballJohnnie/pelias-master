#!/usr/bin/env python3
"""Own-text stuck tokens by kind (author OK 01:46). COUNTS ONLY: no text, no token strings are printed or written.
Inputs: the own-text KL base (token ids in its header), the model GGUF (its vocabulary, to tell what kind each token is),
and the per-token KL dumps of the own-text run C (urm_ZT2, urm_Z, Unsloth MTP). The token stream is walked in order from
the first chunk, so a code fence opened earlier is still open later. Kinds of the token being predicted:
  role    special tokens (<|...|>) and the role word right after <|im_start|>
  code    inside a ``` fenced block
  number  a token with a digit, or a math symbol (= + - * / ^ < > % and the like) outside code
  punct   whitespace or punctuation only
  prose   everything else
Stuck = top 1 % KL of that file. Usage: owntext_stuck_kinds.py BASE.bin MODEL.gguf A.kld [B.kld ...]"""
import struct, sys
import numpy as np
base, model, klds = sys.argv[1], sys.argv[2], sys.argv[3:]
f = open(base, 'rb'); assert f.read(8) == b'_logits_'
n_ctx, n_vocab, n_chunk = struct.unpack('<Iii', f.read(12))
toks = np.fromfile(f, np.int32, n_ctx * n_chunk).reshape(n_chunk, n_ctx); f.close()
g = open(model, 'rb'); rd = lambda fmt: struct.unpack('<' + fmt, g.read(struct.calcsize('<' + fmt)))
def val(t, keep):
    S = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    if t in S: return rd(S[t])[0]
    if t == 8:
        n, = rd('Q'); b = g.read(n); return b.decode('utf-8', 'replace') if keep else None
    if t == 9:
        at, n = rd('IQ'); out = [val(at, keep) for _ in range(n)]; return out if keep else None
assert g.read(4) == b'GGUF'
v, nt, nkv = rd('IQQ'); vocab = None
for _ in range(nkv):
    kl, = rd('Q'); k = g.read(kl).decode(); t, = rd('I')
    x = val(t, k == 'tokenizer.ggml.tokens')
    if k == 'tokenizer.ggml.tokens': vocab = x; break
g.close()
assert vocab is not None and len(vocab) >= n_vocab - 1000
SP, NL = 'Ġ', 'Ċ'  # byte-level BPE: space, newline
def clean(s): return s.replace(SP, ' ').replace(NL, '\n').replace('ĉ', '\t')
MATH = set('=+-*/^<>%×÷∑∫√≈≠≤≥∂πΔ∇±')
kinds = np.zeros(n_chunk * n_ctx, np.int8)  # 0 prose 1 role 2 code 3 number 4 punct
flat = toks.ravel(); in_code = False; prev_start = False
for i, t in enumerate(flat):
    s = vocab[t] if 0 <= t < len(vocab) else ''
    c = clean(s); special = s.startswith('<|') and s.endswith('|>')
    if special or prev_start: kinds[i] = 1
    elif in_code: kinds[i] = 2
    elif any(ch.isdigit() for ch in c) or any(ch in MATH for ch in c): kinds[i] = 3
    elif c.strip() == '' or all(not ch.isalnum() for ch in c): kinds[i] = 4
    else: kinds[i] = 0
    if '```' in c: in_code = not in_code
    if special and 'im_end' in s: in_code = False
    prev_start = special and 'im_start' in s
first = n_ctx // 2; scored = n_ctx - 1 - first
# scored record j of chunk c predicts the token at position first + j + 1
idx = np.concatenate([c * n_ctx + first + 1 + np.arange(scored) for c in range(n_chunk)])
K = kinds[idx]; names = ['prose', 'role', 'code', 'number/math', 'punct/space']
print('own text: %d chunks x %d scored tokens = %d; vocabulary %d (counts only, no text)' % (n_chunk, scored, len(idx), len(vocab)))
print('share of scored tokens: ' + ', '.join('%s %.1f %%' % (names[k], 100 * (K == k).mean()) for k in range(5)))
stuck_all = []
for path in klds:
    b = open(path, 'rb').read(); n = struct.unpack_from('<Q', b, 28)[0]
    kl = np.frombuffer(b, np.dtype([('kld', '<f8'), ('nll', '<f4'), ('nllb', '<f4'), ('pd', '<f4'), ('top', 'u1'), ('pad', 'V3')]), count=n, offset=36)['kld']
    assert n == len(idx), (path, n, len(idx))
    st = kl >= np.quantile(kl, 0.99); stuck_all.append(st)
    print('\n%s: mean KL %.6f; stuck = top 1 %% (%d tokens)' % (path.split('/')[-1], kl.mean(), st.sum()))
    print('  %-12s %8s %9s %9s %10s %12s' % ('kind', 'tokens', 'stuck', 'stuck %', 'mean KL', 'share of KL'))
    for k in range(5):
        m = K == k
        if not m.any(): continue
        print('  %-12s %7.1f%% %9d %8.2f%% %10.5f %11.1f%%' % (names[k], 100 * m.mean(), (st & m).sum(), 100 * st[m].mean(), kl[m].mean(), 100 * kl[m].sum() / kl.sum()))
if len(stuck_all) > 1:
    S = np.array(stuck_all); cons = S.sum(0) >= 2
    print('\nstuck in at least 2 of %d files: %d tokens; by kind: %s' % (len(klds), cons.sum(), ', '.join('%s %d' % (names[k], (cons & (K == k)).sum()) for k in range(5))))
