#!/usr/bin/env python3
# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
"""Routing statistics from `pelias kld --route-dump` files (docs/PELIAS_GAME_PLAN.md, D-T2/D-T3, Block J).

  route_stats.py stats ROUTE.bin [--hot N]
      weight spread by rank; router margin between the last picked and the first unpicked expert; how often a pick
      was also picked by the previous 1, 2, 4, 8 tokens (all picks, and cold picks: outside the N most-picked experts
      of that layer in this file, default 120 per layer); distinct cold experts over the last W tokens (what a warm
      tier would have to hold); the next layer's pre-gating prediction (picks of layer L+1 predicted from layer L).
  route_stats.py compare A.bin B.bin [--kld A.kld]
      routing flips between two files on the same tokens: per layer, the share of token-layers whose picked set
      differs and whose order differs, and the weight mass (in A) on experts B did not pick; per token, the number of
      changed layers; with --kld (A's --dump-kld file), the mean KL by number of changed layers and the share of all KL
      carried by tokens with at least one change.

File format PELROUT1 (written by pelias kld): 8-byte magic, u32 n_ctx, chunks, scored, n_layer, n_used, u64 token hash,
then per scored token (chunk-major) and layer: i16 sel[8], f32 w[8], f32 p_last, f32 p_next, i16 pred[8] (72 bytes).
Exit codes: 0 good, 2 bad input. Standard library only; reads only.
"""
import struct
import sys

REC = struct.Struct("<8h8f2f8h")


def die(msg):
    print("route_stats: " + msg, file=sys.stderr)
    sys.exit(2)


def load(path):
    try:
        b = open(path, "rb").read()
    except OSError as e:
        die("cannot read %s: %s" % (path, e))
    if len(b) < 36 or b[:8] != b"PELROUT1":
        die(path + " is not a PELROUT1 routing record")
    n_ctx, chunks, scored, n_layer, n_used = struct.unpack_from("<5I", b, 8)
    (token_hash,) = struct.unpack_from("<Q", b, 28)
    n = chunks * scored * n_layer
    if n_used < 1 or n_used > 8 or n_layer < 1 or len(b) != 36 + n * REC.size:
        die("%s: size or header does not match (n_used %d, n_layer %d)" % (path, n_used, n_layer))
    sel, w, plast, pnext, pred = [], [], [], [], []
    for i in range(n):
        r = REC.unpack_from(b, 36 + i * REC.size)
        sel.append(r[0:n_used])
        w.append(r[8:8 + n_used])
        plast.append(r[16])
        pnext.append(r[17])
        pred.append(r[18:18 + n_used])
    return dict(n_ctx=n_ctx, chunks=chunks, scored=scored, n_layer=n_layer, n_used=n_used, hash=token_hash,
                sel=sel, w=w, plast=plast, pnext=pnext, pred=pred)


def idx(d, chunk, tok, layer):
    return (chunk * d["scored"] + tok) * d["n_layer"] + layer


def pct(x):
    return "%.1f %%" % (100.0 * x)


def stats(path, hot_n):
    d = load(path)
    L, nu, C, S = d["n_layer"], d["n_used"], d["chunks"], d["scored"]
    n = len(d["sel"])
    print("routing record %s: %d chunks x %d scored tokens x %d layers, %d experts picked per token" % (path, C, S, L, nu))
    # weight spread
    by_rank = [0.0] * nu
    for wv in d["w"]:
        for k in range(nu):
            by_rank[k] += wv[k]
    print("mean weight by rank: " + "  ".join("%d: %.3f" % (k + 1, by_rank[k] / n) for k in range(nu)))
    top2 = sum(wv[0] + wv[1] for wv in d["w"]) / n if nu >= 2 else 1.0
    tail = ("on ranks 5-%d: %s" % (nu, pct(sum(sum(wv[4:]) for wv in d["w"]) / n))) if nu > 4 else "no ranks past 4"
    print("weight on the top 2: %s; %s" % (pct(top2), tail))
    # margin
    m = sorted(a - b for a, b in zip(d["plast"], d["pnext"]))
    q = lambda f: m[min(len(m) - 1, int(f * len(m)))]
    print("router margin (prob. of last picked minus first unpicked): median %.5f, 10 %% %.5f, 1 %% %.5f; share under 0.001: %s"
          % (q(0.5), q(0.1), q(0.01), pct(sum(1 for x in m if x < 0.001) / len(m))))
    # hot sets per layer by pick count in this file
    hot = []
    for layer in range(L):
        cnt = {}
        for c in range(C):
            for t in range(S):
                for e in d["sel"][idx(d, c, t, layer)]:
                    cnt[e] = cnt.get(e, 0) + 1
        ranked = sorted(cnt, key=lambda e: (-cnt[e], e))
        hot.append(set(ranked[:hot_n]))
    cold_picks = sum(1 for c in range(C) for t in range(S) for layer in range(L) for e in d["sel"][idx(d, c, t, layer)] if e not in hot[layer])
    print("cold picks (outside the %d most-picked experts of each layer): %s of all picks" % (hot_n, pct(cold_picks / (n * nu))))
    # reuse over the previous W tokens of the same chunk
    for W in (1, 2, 4, 8):
        seen_all = seen_cold = tot_all = tot_cold = 0
        distinct_cold = 0
        windows = 0
        for c in range(C):
            for t in range(W, S):
                cold_union = set()
                for layer in range(L):
                    prev = set()
                    for back in range(1, W + 1):
                        prev.update(d["sel"][idx(d, c, t - back, layer)])
                    for e in d["sel"][idx(d, c, t, layer)]:
                        tot_all += 1
                        if e in prev:
                            seen_all += 1
                        if e not in hot[layer]:
                            tot_cold += 1
                            if e in prev:
                                seen_cold += 1
                    for back in range(0, W):
                        cold_union.update((layer, e) for e in d["sel"][idx(d, c, t - back, layer)] if e not in hot[layer])
                distinct_cold += len(cold_union)
                windows += 1
        print("previous %d token(s): picks seen before %s; cold picks seen before %s; distinct cold experts in a %d-token window %.1f"
              % (W, pct(seen_all / max(1, tot_all)), pct(seen_cold / max(1, tot_cold)), W, distinct_cold / max(1, windows)))
    # pre-gating prediction
    hit = tot = hit_cold = tot_cold = 0
    for c in range(C):
        for t in range(S):
            for layer in range(L - 1):
                pr = set(d["pred"][idx(d, c, t, layer)])
                if -1 in pr:
                    continue
                for e in d["sel"][idx(d, c, t, layer + 1)]:
                    tot += 1
                    hit += e in pr
                    if e not in hot[layer + 1]:
                        tot_cold += 1
                        hit_cold += e in pr
    print("next-layer prediction from this layer's output: %s of the next layer's picks; %s of its cold picks"
          % (pct(hit / max(1, tot)), pct(hit_cold / max(1, tot_cold))))


def compare(pa, pb, kld_path):
    a, b = load(pa), load(pb)
    for k in ("n_ctx", "chunks", "scored", "n_layer", "n_used", "hash"):
        if a[k] != b[k]:
            die("the two records are not on the same tokens and layers (%s differs)" % k)
    L, C, S = a["n_layer"], a["chunks"], a["scored"]
    set_diff = [0] * L
    order_diff = [0] * L
    mass = [0.0] * L
    changed = []
    for c in range(C):
        for t in range(S):
            ch = 0
            for layer in range(L):
                i = idx(a, c, t, layer)
                sa, sb = a["sel"][i], b["sel"][i]
                if set(sa) != set(sb):
                    set_diff[layer] += 1
                    ch += 1
                    mass[layer] += sum(wv for e, wv in zip(sa, a["w"][i]) if e not in set(sb))
                elif tuple(sa) != tuple(sb):
                    order_diff[layer] += 1
            changed.append(ch)
    T = C * S
    print("routing flips %s vs %s on %d tokens x %d layers" % (pa, pb, T, L))
    print("layer  set differs  order only  weight on changed experts (mean per token-layer)")
    for layer in range(L):
        print("%5d  %10s  %10s  %.4f" % (layer, pct(set_diff[layer] / T), pct(order_diff[layer] / T), mass[layer] / T))
    print("all layers: set differs %s of token-layers" % pct(sum(set_diff) / (T * L)))
    bins = [(0, 0), (1, 1), (2, 3), (4, 10 ** 9)]
    for lo, hi in bins:
        k = sum(1 for x in changed if lo <= x <= hi)
        print("tokens with %s changed layers: %s" % (("%d" % lo) if lo == hi else ("%d-%d" % (lo, hi) if hi < 10 ** 9 else ">= %d" % lo), pct(k / T)))
    if kld_path:
        kb = open(kld_path, "rb").read()
        if kb[:8] != b"PELKLD01":
            die(kld_path + " is not a --dump-kld file")
        n_ctx, chunks, scored = struct.unpack_from("<3I", kb, 8)
        (h,) = struct.unpack_from("<Q", kb, 20)
        (n,) = struct.unpack_from("<Q", kb, 28)
        if (n_ctx, chunks, scored, h) != (a["n_ctx"], C, S, a["hash"]) or n != T:
            die("the KL dump is not on the same tokens as the routing record")
        kl = [struct.unpack_from("<d", kb, 36 + 24 * i)[0] for i in range(n)]
        total = sum(kl)
        for lo, hi in bins:
            sel = [kl[i] for i in range(T) if lo <= changed[i] <= hi]
            if sel:
                print("mean KL of tokens with %s changed layers: %.6f (%d tokens)" % (("%d" % lo) if lo == hi else ("%d-%d" % (lo, hi) if hi < 10 ** 9 else ">= %d" % lo), sum(sel) / len(sel), len(sel)))
        carried = sum(kl[i] for i in range(T) if changed[i] > 0)
        print("share of all KL carried by tokens with at least one changed layer: %s" % pct(carried / total if total else 0.0))


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if len(argv) >= 2 else 2
    args = argv[2:]
    if argv[1] == "stats":
        hot_n = 120
        if "--hot" in args:
            i = args.index("--hot")
            try:
                hot_n = int(args[i + 1])
            except (IndexError, ValueError):
                die("--hot needs a number")
            del args[i:i + 2]
        if len(args) != 1:
            die("stats needs one routing record")
        stats(args[0], hot_n)
    elif argv[1] == "compare":
        kld = None
        if "--kld" in args:
            i = args.index("--kld")
            if i + 1 >= len(args):
                die("--kld needs a file")
            kld = args[i + 1]
            del args[i:i + 2]
        if len(args) != 2:
            die("compare needs two routing records")
        compare(args[0], args[1], kld)
    else:
        die("unknown command; see --help")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
