#!/usr/bin/env python3
"""C6 miss map, first check (sandbox, existing per-token dumps only; no hidden states yet).
Question: do the places the model gets stuck (high-KL tokens) repeat across files and cluster in time?
If the same positions are stuck in every file and stuck tokens come in runs, a logged miss map has something to learn.
Reads PELKLD01 dumps: header 36 bytes, then per token kld f64, nll f32, nll_base f32, p_diff f32, same_top u8, 3 pad."""
import struct, sys, numpy as np
D = "data/probe_results/gp_runs/"
FILES = {"urm_Z": D+"U/urmz_x8.kld", "urm_ZT2": D+"U/urmzt2_x8.kld", "urm_ZT": D+"U/urmzt_x8.kld",
         "urm_Q7": D+"U/urmq7_x8.kld", "urm": D+"U/urm_x8.kld", "live25": D+"U/hgate25_x8.kld",
         "unsloth_mtp_g256": D+"N/mtp_g256.kld"}
def read(p):
    b = open(p, "rb").read()
    assert b[:8] == b"PELKLD01", p
    n_ctx, chunks, scored, th, n = struct.unpack_from("<IIIQQ", b, 8)
    rec = np.frombuffer(b, dtype=np.dtype([("kld","<f8"),("nll","<f4"),("nllb","<f4"),("pd","<f4"),("top","u1"),("pad","V3")]), count=n, offset=36)
    return dict(n_ctx=n_ctx, chunks=chunks, scored=scored, hash=th, r=rec)
data = {k: read(v) for k, v in FILES.items()}
hashes = {k: d["hash"] for k, d in data.items()}
base = data["urm_Z"]; S = base["scored"]; C = base["chunks"]; N = len(base["r"])
print(f"tokens {N} = {C} chunks x {S} scored; same text in every file: {len(set(hashes.values()))==1}")
K = {k: d["r"]["kld"] for k, d in data.items() if d["hash"] == base["hash"] and len(d["r"]) == N}
q = 0.01
stuck = {k: v >= np.quantile(v, 1-q) for k, v in K.items()}
print(f"\n1) Same stuck positions across files? stuck = top {q*100:.0f} % KL of each file ({int(q*N)} tokens). Random overlap = {q*100:.0f} %.")
names = list(K)
print("   " + " ".join(f"{n[:9]:>9}" for n in names))
for a in names:
    print(f"{a[:9]:>9} " + " ".join(f"{(stuck[a]&stuck[b]).sum()/stuck[a].sum()*100:8.0f}%" for b in names))
print("\n2) Do stuck tokens come in runs? P(stuck at t | stuck at t-d) vs base rate, within chunks")
for k in ("urm_Z", "urm_ZT2", "unsloth_mtp_g256"):
    if k not in K: continue
    s = stuck[k].reshape(C, S)
    row = []
    for d in (1, 2, 4, 8):
        prev = s[:, :-d]; cur = s[:, d:]
        row.append(f"d={d}: {(prev&cur).sum()/max(prev.sum(),1)*100:4.1f} %")
    print(f"   {k:16s} base {s.mean()*100:.1f} %  " + "  ".join(row))
print("\n3) Is the reference itself unsure at stuck tokens? reference NLL of the true next token (nats)")
for k in ("urm_Z", "urm_ZT2", "unsloth_mtp_g256"):
    if k not in K: continue
    r = data[k]["r"]; m = stuck[k]
    print(f"   {k:16s} stuck: median {np.median(r['nllb'][m]):.2f}, all: median {np.median(r['nllb']):.2f}; top-token agrees at stuck {r['top'][m].mean()*100:.0f} % vs all {r['top'].mean()*100:.0f} %")
print("\n4) How much of each file's KL sits in its stuck 1 %?")
for k, v in K.items():
    print(f"   {k:16s} mean {v.mean():.6f}; stuck 1 % holds {v[stuck[k]].sum()/v.sum()*100:.0f} % of the KL")
