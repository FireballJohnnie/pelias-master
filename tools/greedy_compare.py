#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""greedy_compare.py - GP3-07 gate: Pelias greedy decode against llama.cpp's, token for token.

Reads, for each prompt NN in a folder:
  pelias_NN.ids   from `pelias generate --ids-out` (lines "prompt: ...", "output: ...", "gap: ...")
  prompt_NN.json  (optional) the prompt ids sent to llama-server, as `pelias tokenize` prints them; checked equal to
                  the prompt line of pelias_NN.ids, so both sides decoded from the same tokens
  llama_NN.json   llama-server's /completion response for the same prompt ids, with "return_tokens": true and
                  "n_probs": 2 (its completion_probabilities carry the top-2 log-probabilities per step)
and prints one row per prompt: tokens compared, equal or the first differing position, and at that position the top-1
minus top-2 gap each side saw (a near-tie explains a split; a large gap does not). Ends with a VERDICT line.
Nothing is written unless --out names a file that does not exist yet.

  python3 tools/greedy_compare.py FOLDER [--n 128] [--out report.tsv]
"""
import argparse
import json
import os
import sys


def read_ids(path):
    rec = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, _, rest = line.partition(":")
            vals = rest.split()
            rec[key.strip()] = [float(v) for v in vals] if key.strip() == "gap" else [int(v) for v in vals]
    return rec


def read_llama(path):
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    toks = list(r.get("tokens") or [])
    gaps = []
    for step in r.get("completion_probabilities") or []:
        top = step.get("top_logprobs") or []
        gaps.append(top[0]["logprob"] - top[1]["logprob"] if len(top) >= 2 else float("nan"))
    return toks, gaps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--n", type=int, default=128, help="tokens per prompt the gate needs (default 128)")
    ap.add_argument("--out", help="write the rows as TSV here (refused if it exists)")
    a = ap.parse_args()
    if a.out and os.path.exists(a.out):
        sys.exit(f"greedy_compare: {a.out} already exists (nothing is overwritten)")
    names = sorted(f[len("pelias_"):-len(".ids")] for f in os.listdir(a.folder) if f.startswith("pelias_") and f.endswith(".ids"))
    if not names:
        sys.exit("greedy_compare: no pelias_NN.ids files in " + a.folder)
    rows, full, compared = [], 0, 0
    print("prompt\tprompt_tokens\tcompared\tresult\tfirst_diff\tpelias_tok\tllama_tok\tpelias_gap\tllama_gap")
    for nm in names:
        p = read_ids(os.path.join(a.folder, f"pelias_{nm}.ids"))
        lp = os.path.join(a.folder, f"llama_{nm}.json")
        sp = os.path.join(a.folder, f"prompt_{nm}.json")
        if os.path.exists(sp):
            with open(sp, encoding="utf-8") as f:
                sent = json.load(f)
            if sent != p.get("prompt"):
                rows.append([nm, str(len(p.get("prompt", []))), "0", "prompt ids differ", "", "", "", "", ""])
                print("\t".join(rows[-1]))
                continue
        if not os.path.exists(lp):
            row = [nm, str(len(p.get("prompt", []))), "0", "no llama.cpp output", "", "", "", "", ""]
        else:
            lt, lg = read_llama(lp)
            pt, pg = p.get("output", []), p.get("gap", [])
            n = min(len(pt), len(lt), a.n)
            compared += 1
            d = next((i for i in range(n) if pt[i] != lt[i]), None)
            if d is None:
                res = "equal" if n >= a.n else f"equal but short ({n} < {a.n})"
                full += n >= a.n
                row = [nm, str(len(p["prompt"])), str(n), res, "", "", "", "", ""]
            else:
                f5 = lambda v, i: f"{v[i]:.5f}" if i < len(v) else ""
                row = [nm, str(len(p["prompt"])), str(n), "differs", str(d), str(pt[d]), str(lt[d]), f5(pg, d), f5(lg, d)]
        rows.append(row)
        print("\t".join(row))
    verdict = "PASS" if full == len(names) else "NOT EQUAL"
    print(f"VERDICT {verdict}: {full} of {len(names)} prompts equal for {a.n} tokens ({compared} compared); "
          "first differing positions are reported above, with each side's top-1 minus top-2 gap there")
    if a.out:
        with open(a.out, "x", encoding="utf-8") as f:
            f.write("prompt\tprompt_tokens\tcompared\tresult\tfirst_diff\tpelias_tok\tllama_tok\tpelias_gap\tllama_gap\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
    return 0 if verdict == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
