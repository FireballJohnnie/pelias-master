#!/usr/bin/env python3
"""Run C (own-text check): build an evaluation text from the author's own dataset, on the author's PC only.
Prints counts only, never any text (the dataset is private and stays local).
Takes the held-out 'eval' split first (never trained on), then 'train' records if more tokens are needed, in a fixed
shuffled order (seed 20261028); renders each record as plain role headers + content (no special tokens); drops every
record that shares any 64-token shingle with the imatrix calibration text (game plan rule: the evaluation text is never
the calibration text); stops at the token target. Tokens counted with `pelias tokenize` (the reference's tokenizer).
usage: owntext_select.py DATASET.jsonl CALIB.txt OUT.txt PELIAS MODEL.gguf TARGET_TOKENS"""
import json, os, random, subprocess, sys, tempfile
ds, calib, out, pelias, model, target = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
if os.path.exists(out):
    sys.exit("STOP: %s exists (nothing is overwritten)" % out)
def toks(text):
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(text); p = f.name
    try:
        r = subprocess.run([pelias, 'tokenize', '--model', model, '--header-only', '--file', p, '--no-special'],
                           check=True, capture_output=True, text=True)
    finally:
        os.unlink(p)
    return json.loads(r.stdout)
def shingles(ids, n=64):
    return {hash(tuple(ids[i:i + n])) for i in range(0, max(0, len(ids) - n + 1))}
recs = [json.loads(l) for l in open(ds, encoding='utf-8')]
def render(r):
    # system messages left out: the same system prompt repeats across records and the calibration text, so it would
    # mark every record as overlapping; the conversation itself is what is being evaluated
    return "\n\n".join("%s:\n%s" % (m['role'], m['content']) for m in r['messages']
                       if m.get('role') != 'system' and isinstance(m.get('content'), str) and m['content'])
rng = random.Random(20261028)
ev = [r for r in recs if r.get('split') == 'eval']; tr = [r for r in recs if r.get('split') != 'eval']
rng.shuffle(ev); rng.shuffle(tr)
cal = shingles(toks(open(calib, encoding='utf-8').read()))
print("calibration shingles: %d; records: eval %d, train %d" % (len(cal), len(ev), len(tr)), flush=True)
chosen, n_tok, dropped, used = [], 0, {'eval': 0, 'train': 0}, {'eval': 0, 'train': 0}
for name, pool in (('eval', ev), ('train', tr)):
    for r in pool:
        if n_tok >= target:
            break
        t = render(r)
        if not t:
            continue
        ids = toks(t)
        if shingles(ids) & cal:
            dropped[name] += 1
            continue
        chosen.append(t); n_tok += len(ids); used[name] += 1
text = "\n\n\n".join(chosen)
ids = toks(text)
over = len(shingles(ids) & cal)
print("chosen records: eval %d, train %d; dropped for calibration overlap: eval %d, train %d" % (used['eval'], used['train'], dropped['eval'], dropped['train']))
print("text: %d bytes, %d tokens (target %d); 64-token shingles shared with the calibration text: %d" % (len(text.encode('utf-8')), len(ids), target, over))
if over or len(ids) < target:
    sys.exit("STOP: overlap %d or too few tokens %d" % (over, len(ids)))
with open(out, 'x', encoding='utf-8') as f:
    f.write(text)
print("written (local only): %s" % out)
