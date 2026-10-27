#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""Gate P-D1 against the Hugging Face reference: the model folder's own tokenizer (transformers) and `pelias tokenize`
must give the same token ids for every file. Run inside the reference venv (transformers installed).

    python3 tools/pd1_compare_hf.py --pelias BUILD/pelias --gguf MODEL.gguf --folder HF_MODEL_FOLDER FILE [FILE ...]

Prints one line per file (tokens, same or the first mismatch with the tokens around it) and exits 1 on any mismatch.
NFC is on in Pelias (as the model's tokenizer.json asks); control tokens are parsed in both (the reference matches
every added token in plain text by default).
"""
import argparse
import subprocess
import sys
import time


def pelias_ids(pelias, gguf, path, header_only):
    cmd = [pelias, 'tokenize', '--model', gguf, '--file', path]
    if header_only:
        cmd.append('--header-only')
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith('[')][-1]
    body = line.strip()[1:-1].strip()
    return [int(x) for x in body.split(',')] if body else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pelias', required=True)
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--folder', required=True)
    ap.add_argument('--header-only', action='store_true')
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()

    from transformers import AutoTokenizer  # imported here so --help works without it
    import transformers
    tok = AutoTokenizer.from_pretrained(a.folder)
    print(f'reference: transformers {transformers.__version__}, {type(tok).__name__}, vocabulary {len(tok)}')

    bad = 0
    total = 0
    t_ref = t_pel = 0.0
    for path in a.files:
        with open(path, encoding='utf-8', newline='') as f:
            text = f.read()
        t0 = time.perf_counter()
        ref = tok(text, add_special_tokens=False)['input_ids']
        t1 = time.perf_counter()
        got = pelias_ids(a.pelias, a.gguf, path, a.header_only)
        t2 = time.perf_counter()
        t_ref += t1 - t0
        t_pel += t2 - t1
        total += len(ref)
        if ref == got:
            print(f'{path}: {len(ref)} tokens, same (reference {t1 - t0:.2f} s, pelias {t2 - t1:.2f} s)')
            continue
        bad += 1
        i = next((k for k, (x, y) in enumerate(zip(ref, got)) if x != y), min(len(ref), len(got)))
        show = lambda ids: [tok.convert_ids_to_tokens(x) for x in ids[max(0, i - 3):i + 5]]
        print(f'{path}: MISMATCH at token {i} (reference {len(ref)}, pelias {len(got)})')
        print(f'  reference: {show(ref)}')
        print(f'  pelias:    {show(got)}')
    print(f'P-D1 (Hugging Face): {total} reference tokens in {len(a.files)} files, {bad} file(s) with a mismatch; '
          f'time reference {t_ref:.2f} s (tokenize only), pelias {t_pel:.2f} s (process, incl. loading the vocabulary)')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
