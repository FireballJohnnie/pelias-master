#!/usr/bin/env python3
# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# provenance: original
"""Grade the GP3-07 split tokens against the original model (author 2026-10-29 20:01: agreement with llama.cpp says
nothing about who is right; the original model does).

  split_grade.py llama-ids RUN_DIR OUT_DIR
      writes OUT_DIR/llama_NN.ids (generate --ids-out format) from RUN_DIR/llama_NN.json and prompt_NN.json, so
      `pelias teacher` can read llama.cpp's output too (new files only).
  split_grade.py grade RUN_DIR TEACH_DIR [--label NAME] [--out FILE.tsv]
      for every prompt NN: the first position where pelias_NN.ids and llama_NN.json differ; there both sides share the
      prefix, so the original model's own pick (TEACH_DIR/teacher_pelias_NN.tsv) says who is right: pelias, llama, or
      neither, with the original's own top-1 minus top-2 margin (a small one means the original itself is near a tie).
      Over all positions of each side: how often its token is the original's own top-1 (on its own prefix) and its mean
      NLL under the original (TEACH_DIR/teacher_pelias_NN.tsv, teacher_llama_NN.tsv).
"""
import argparse, json, os, sys


def read_ids(p):
    d = {}
    for line in open(p):
        k, _, v = line.partition(':')
        d[k.strip()] = v.split()
    return [int(x) for x in d.get('prompt', [])], [int(x) for x in d.get('output', [])]


def llama_tokens(p):
    r = json.load(open(p))
    toks = list(r.get('tokens') or [])
    if not toks:
        toks = [int(s['id']) for s in r.get('completion_probabilities') or []]
    return [int(t) for t in toks]


def read_teacher(p):
    rows = []
    with open(p) as f:
        next(f)
        for line in f:
            c = line.rstrip('\n').split('\t')
            rows.append(dict(token=int(c[1]), top1=int(c[2]), l1=float(c[3]), top2=int(c[4]), l2=float(c[5]), lp=float(c[7])))
    return rows


def names(run):
    return sorted(f[7:9] for f in os.listdir(run) if f.startswith('pelias_') and f.endswith('.ids'))


def cmd_llama_ids(a):
    os.makedirs(a.out, exist_ok=True)
    for nn in names(a.run):
        out = os.path.join(a.out, f'llama_{nn}.ids')
        if os.path.exists(out):
            sys.exit(f'{out} exists (nothing is overwritten)')
        prompt, _ = read_ids(os.path.join(a.run, f'pelias_{nn}.ids'))
        sp = os.path.join(a.run, f'prompt_{nn}.json')
        if os.path.exists(sp) and json.load(open(sp)) != prompt:
            sys.exit(f'prompt {nn}: the ids sent to llama.cpp differ from pelias_{nn}.ids')
        with open(out, 'w') as f:
            f.write('prompt: ' + ' '.join(map(str, prompt)) + '\noutput: ' + ' '.join(map(str, llama_tokens(os.path.join(a.run, f'llama_{nn}.json')))) + '\n')
    print(f'wrote {len(names(a.run))} llama ids files to {a.out}')


def cmd_grade(a):
    rows, wins = [], {'pelias': 0, 'llama': 0, 'neither': 0, 'equal': 0}
    agree = {'pelias': [0, 0], 'llama': [0, 0]}
    nll = {'pelias': [0.0, 0], 'llama': [0.0, 0]}
    for nn in names(a.run):
        _, pel = read_ids(os.path.join(a.run, f'pelias_{nn}.ids'))
        lla = llama_tokens(os.path.join(a.run, f'llama_{nn}.json'))
        tp = os.path.join(a.teach, f'teacher_pelias_{nn}.tsv')
        if not os.path.exists(tp):
            continue
        tpel = read_teacher(tp)
        tl = os.path.join(a.teach, f'teacher_llama_{nn}.tsv')
        tlla = read_teacher(tl) if os.path.exists(tl) else None
        n = min(len(pel), len(lla), len(tpel))
        d = next((i for i in range(n) if pel[i] != lla[i]), None)
        for side, t in (('pelias', tpel), ('llama', tlla)):
            if t is None:
                continue
            agree[side][0] += sum(r['token'] == r['top1'] for r in t)
            agree[side][1] += len(t)
            nll[side][0] += -sum(r['lp'] for r in t)
            nll[side][1] += len(t)
        if d is None:
            wins['equal'] += 1
            rows.append((nn, '-', '', '', '', '', 'equal'))
            continue
        r = tpel[d]
        who = 'pelias' if r['top1'] == pel[d] else 'llama' if r['top1'] == lla[d] else 'neither'
        wins[who] += 1
        rows.append((nn, d, pel[d], lla[d], r['top1'], f"{r['l1'] - r['l2']:.4f}", who))
    hdr = 'prompt\tfirst_split\tpelias_tok\tllama_tok\toriginal_top1\toriginal_margin\tright'
    print(hdr)
    for r in rows:
        print('\t'.join(map(str, r)))
    ag = {s: (agree[s][0] / agree[s][1] if agree[s][1] else float('nan')) for s in agree}
    nl = {s: (nll[s][0] / nll[s][1] if nll[s][1] else float('nan')) for s in nll}
    summary = (f"GRADE {a.label}: at the first split the original picks pelias {wins['pelias']}, llama {wins['llama']}, neither {wins['neither']} "
               f"({wins['equal']} prompts equal); over all positions, the token is the original's own top-1: pelias {ag['pelias'] * 100:.1f} %, "
               f"llama {ag['llama'] * 100:.1f} %; mean NLL under the original: pelias {nl['pelias']:.4f}, llama {nl['llama']:.4f}")
    print(summary)
    if a.out:
        if os.path.exists(a.out):
            sys.exit(f'{a.out} exists (nothing is overwritten)')
        with open(a.out, 'w') as f:
            f.write(hdr + '\n' + ''.join('\t'.join(map(str, r)) + '\n' for r in rows) + summary + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('llama-ids')
    p.add_argument('run')
    p.add_argument('out')
    p.set_defaults(fn=cmd_llama_ids)
    g = sub.add_parser('grade')
    g.add_argument('run')
    g.add_argument('teach')
    g.add_argument('--label', default='')
    g.add_argument('--out', default='')
    g.set_defaults(fn=cmd_grade)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
