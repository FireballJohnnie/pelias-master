#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""greedy_compare.py - GP3-07 gate: Pelias greedy decode against llama.cpp's, token for token.

Reads, for each prompt NN in a folder:
  pelias_NN.ids   from `pelias generate --ids-out` (lines "prompt: ...", "output: ...", "gap: ...")
  prompt_NN.json  (optional) the prompt ids sent to llama-server, as `pelias tokenize` prints them; checked equal to
                  the prompt line of pelias_NN.ids, so both sides decoded from the same tokens
  llamab_NN.json  (optional) llama-server again with its fast paths off (--no-repack, GGML_CPU_TILED_MM=0): only the
                  order of its own float sums changes, so llama vs llamab is the yardstick for how far two correct
                  programs drift apart on a quantized file (MEASURED 2026-10-29, tiny Q4_K_M: 0 of 20 equal, gap
                  difference median 0.030; Pelias vs llama 0.057)
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
    # completion_probabilities can hold fewer entries than tokens (llama-server leaves one out where a character's UTF-8
    # bytes span two tokens; MEASURED in the sandbox dry run: 127 entries for 128 tokens), so the entries are matched to
    # the tokens by id, in order; a token with no entry gets no gap (nan) instead of its neighbour's
    gaps = [float("nan")] * len(toks)
    j = 0
    steps = r.get("completion_probabilities") or []
    for i, tid in enumerate(toks):
        if j < len(steps) and steps[j].get("id") == tid:
            top = steps[j].get("top_logprobs") or []
            if len(top) >= 2:
                gaps[i] = top[0]["logprob"] - top[1]["logprob"]
            j += 1
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
    # gap agreement on the steps before each split, Pelias vs llama and (when present) llama vs llama with its fast paths off
    def gap_stats(pairs):
        d = sorted(pairs)
        if not d:
            return "no steps"
        return "median %.2e, p90 %.2e, max %.2e over %d steps" % (d[len(d) // 2], d[int(0.9 * (len(d) - 1))], d[-1], len(d))
    pl, ll, eq_ll, first_ll = [], [], 0, []
    for nm in names:
        p = read_ids(os.path.join(a.folder, f"pelias_{nm}.ids"))
        lp = os.path.join(a.folder, f"llama_{nm}.json")
        if not os.path.exists(lp):
            continue
        lt, lg = read_llama(lp)
        pt, pg = p.get("output", []), p.get("gap", [])
        for i in range(min(len(pt), len(lt), len(pg))):
            if pt[i] != lt[i]:
                break
            if lg[i] == lg[i]:
                pl.append(abs(pg[i] - lg[i]))
        bp = os.path.join(a.folder, f"llamab_{nm}.json")
        if os.path.exists(bp):
            bt, bg = read_llama(bp)
            n = min(len(lt), len(bt))
            d = next((i for i in range(n) if lt[i] != bt[i]), None)
            eq_ll += d is None and n >= a.n
            first_ll.append(n if d is None else d)
            for i in range(n if d is None else d):
                if lg[i] == lg[i] and bg[i] == bg[i]:
                    ll.append(abs(lg[i] - bg[i]))
    print("GAPS pelias vs llama: " + gap_stats(pl))
    if first_ll:
        print("YARDSTICK llama vs llama (fast paths off): %d of %d prompts equal for %d tokens; first split median %d; gaps %s"
              % (eq_ll, len(first_ll), a.n, sorted(first_ll)[len(first_ll) // 2], gap_stats(ll)))
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
