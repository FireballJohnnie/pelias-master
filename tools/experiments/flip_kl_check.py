#!/usr/bin/env python3
"""Coin flips of the 8-bit activation rounding against drift (game plan: "Activation coin flips", author 2026-10-31).

Reads, read-only:
  --rounded R.kld   pelias kld --dump-kld with --act-int8 all --act-int8-path gpu256x8
  --float F.kld     pelias kld --dump-kld without activation rounding (same model, same base)
  --flips X.bin     pelias kld --dump-flips from the rounded run
  [--check C.kld]   an earlier rounded dump: must equal R token for token (the run is deterministic)
Prints: the rounding's cost (paired, chunk-clustered), and whether a token's coin flips predict its own damage and the
next token's (rank correlation, AUROC for the worst 1 %, and the t -> t+1 pattern the author compares with 11 % vs 1 %).
Writes nothing.
"""
import argparse
import math
import struct
import sys


def read_kld(path):
    with open(path, "rb") as f:
        b = f.read()
    if b[:8] != b"PELKLD01":
        sys.exit(f"{path}: not a PELKLD01 dump")
    n_ctx, chunks, scored, thash, nrec = struct.unpack_from("<IIIQQ", b, 8)
    off = 8 + 12 + 16
    kl, top = [], []
    for i in range(nrec):
        k, _nll, _nb, _pd, st = struct.unpack_from("<dfffB", b, off + 24 * i)
        kl.append(k)
        top.append(st)
    return dict(n_ctx=n_ctx, chunks=chunks, scored=scored, hash=thash, kl=kl, top=top)


def read_flips(path):
    with open(path, "rb") as f:
        b = f.read()
    if b[:8] != b"PELFLIP1":
        sys.exit(f"{path}: not a PELFLIP1 record")
    n_ctx, chunks, scored, thash = struct.unpack_from("<IIIQ", b, 8)
    n = (len(b) - 28) // 12
    v = [struct.unpack_from("<III", b, 28 + 12 * i) for i in range(n)]
    return dict(n_ctx=n_ctx, chunks=chunks, scored=scored, hash=thash, v=v)


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sab / (sa * sb) if sa > 0 and sb > 0 else float("nan")


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def auroc(score, label):
    """P(score of a positive > score of a negative), ties half."""
    r = ranks(score)
    pos = [r[i] for i in range(len(r)) if label[i]]
    npos, nneg = len(pos), len(r) - len(pos)
    if npos == 0 or nneg == 0:
        return float("nan")
    return (sum(x + 1 for x in pos) - npos * (npos + 1) / 2.0) / (npos * nneg)  # ranks from 1 (Mann-Whitney U / n+ n-)


def top_frac(x, frac):
    k = max(1, int(round(len(x) * frac)))
    thr = sorted(x, reverse=True)[k - 1]
    return [v >= thr for v in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounded", required=True)
    ap.add_argument("--float", dest="flt", required=True)
    ap.add_argument("--flips", required=True)
    ap.add_argument("--check")
    a = ap.parse_args()
    R, F, X = read_kld(a.rounded), read_kld(a.flt), read_flips(a.flips)
    for name, d in (("float", F), ("flips", X)):
        if (d["n_ctx"], d["chunks"], d["scored"], d["hash"]) != (R["n_ctx"], R["chunks"], R["scored"], R["hash"]):
            sys.exit(f"{name}: not the same chunks / tokens as the rounded dump")
    n, sc = len(R["kl"]), R["scored"]
    if len(F["kl"]) != n or len(X["v"]) != n:
        sys.exit("record counts differ")
    if a.check:
        C = read_kld(a.check)
        same = sum(1 for x, y in zip(C["kl"], R["kl"]) if x == y)
        print(f"determinism: {same} / {n} tokens equal to {a.check}")

    # the rounding's cost, paired by token, chunk-clustered standard error
    dmg = [r - f for r, f in zip(R["kl"], F["kl"])]
    ch = [sum(dmg[c * sc:(c + 1) * sc]) / sc for c in range(R["chunks"])]
    m = sum(dmg) / n
    se = math.sqrt(sum((x - m) ** 2 for x in ch) / (len(ch) - 1) / len(ch))
    print(f"mean KL: rounded {sum(R['kl']) / n:.6f}, float {sum(F['kl']) / n:.6f}; rounding cost {m:+.6f} "
          f"(SE per chunk {se:.6f}, t {m / se if se > 0 else float('nan'):.2f}); same top: rounded {sum(R['top']) / n * 100:.3f} %, "
          f"float {sum(F['top']) / n * 100:.3f} %")
    print(f"tokens where rounding raised KL: {sum(1 for x in dmg if x > 0) / n * 100:.2f} %")

    bands = ["1e-4", "1e-3", "1e-2"]
    for b in range(3):
        v = [x[b] for x in X["v"]]
        print(f"coin flips within {bands[b]} of halfway, per token: mean {sum(v) / n:.1f}, min {min(v)}, max {max(v)}")

    # within each chunk, token t and its next scored token t+1
    idx_t = [c * sc + i for c in range(R["chunks"]) for i in range(sc - 1)]
    absd = [abs(x) for x in dmg]
    worst_d = top_frac(absd, 0.01)
    worst_kl = top_frac(R["kl"], 0.01)
    print("\nflips at t against damage |KL rounded - KL float| and rounded KL (Spearman rho; AUROC for the worst 1 %):")
    for b in range(3):
        f = [x[b] for x in X["v"]]
        ft = [f[i] for i in idx_t]
        print(f"  band {bands[b]}: same token: rho(damage) {spearman(f, absd):+.3f}, AUROC worst-1% damage {auroc(f, worst_d):.3f}, "
              f"rho(KL) {spearman(f, R['kl']):+.3f}, AUROC worst-1% KL {auroc(f, worst_kl):.3f}")
        print(f"               next token: rho(damage) {spearman(ft, [absd[i + 1] for i in idx_t]):+.3f}, "
              f"AUROC worst-1% damage {auroc(ft, [worst_d[i + 1] for i in idx_t]):.3f}, "
              f"AUROC worst-1% KL {auroc(ft, [worst_kl[i + 1] for i in idx_t]):.3f}")

    # the t -> t+1 pattern (author: like the 11 % vs 1 % of misses)
    def cond(flag_t, flag_next):
        hits = [i for i in idx_t if flag_t[i]]
        p = sum(1 for i in hits if flag_next[i + 1]) / len(hits) if hits else float("nan")
        base = sum(1 for i in idx_t if flag_next[i + 1]) / len(idx_t)
        return p, base, len(hits)

    print("\nnext token in the worst 1 %, given this token is flagged (base rate in brackets):")
    p, base, k = cond(worst_d, worst_d)
    print(f"  worst-1% damage at t -> worst-1% damage at t+1: {p * 100:.2f} % [{base * 100:.2f} %] ({k} tokens flagged)")
    p, base, k = cond(worst_kl, worst_kl)
    print(f"  worst-1% KL at t     -> worst-1% KL at t+1:     {p * 100:.2f} % [{base * 100:.2f} %] ({k} tokens flagged)")
    for b in range(3):
        f = [x[b] for x in X["v"]]
        flag = top_frac(f, 0.10)
        p1, b1, k1 = cond(flag, worst_d)
        p2, b2, k2 = cond(flag, worst_kl)
        print(f"  flips in the top 10 % at t (band {bands[b]}): worst-1% damage at t+1 {p1 * 100:.2f} % [{b1 * 100:.2f} %], "
              f"worst-1% KL at t+1 {p2 * 100:.2f} % [{b2 * 100:.2f} %] ({k1} tokens flagged)")


if __name__ == "__main__":
    main()
