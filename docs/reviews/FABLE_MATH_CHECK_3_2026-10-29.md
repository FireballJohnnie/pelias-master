# Fable read-only math check of #3 (the engine plan), 2026-10-29 02:0x

Read-only (no builds, no code run). Inputs: game plan (idea register, DDR4 cold path, idle units, web v2, runs A-E, tour), 7/T/H plan, build log, spec tensor sizes, urmzt.tsv. All new numbers are DERIVED unless quoted.

## Verdicts

1. **Token-time ledger: holds in structure, three fixes.**
   - (a) The cold step does not sum: 3.0 ms reads + 5.6 ms handoffs = 8.6 of the measured 10.1 ms; 1.5 ms (0.9-2.5 depending on which measured handoff / read rate is used) is unaccounted. Honest rung 1 = 9.7-11.2 ms = 89-103 tok/s. The first engine timer must name the residual.
   - (b) The 5.2 ms assumes 100 % of 504 GB/s; per-token bytes 2.59-2.69 GB. The measured Q8_Z kernel runs at 324 GB/s (64 %): then weight reads are 7.9 ms and the "3.2 ms overhead" is 0.5 ms, so U8 (one graph) would give little and bytes-in-flight a lot. The 5.2 / 3.2 split is the most important unmeasured number. Realistic memory floor at 85-90 % efficiency: 5.6-6.0 ms = 167-177 tok/s (193 is the 100 % number).
   - (c) Q7_ZT per token: of -236.4 MB, 61.6 MB is token_embd (one row read per token) and ~5 MB blk.40; the per-token trunk saving vs Q8_Z is 126.6 MB = -0.25 ms at 504 GB/s (-0.17 to -0.20 ms at the measured kernel ratio): 5.2 -> ~4.95-5.0 ms; floor 193 -> ~200 tok/s. Q8_Z's "-0.12 ms" is -0.09 ms.
   - Double counting: "bytes kept off DDR4 (0.3 ms)" + "hidden behind the GPU (0.6)" exceed the 0.9 ms of reads; call rung 3 8.65-8.8 ms. U2's BAR1 -0.2 ms and U8's -3.2 ms both remove launch overhead: count once. U5's codebook -0.2 ms is exclusive with Q4_TG on the same tensors.
   - Shaky: 2 MB pages alone will not give 13.31 -> 40-45 GB/s; the 3x is memory-level parallelism (8 pinned threads + software prefetch); pages add 10-30 %. bench_cpu: four cells (threads x pages). Spinning SMT siblings steal issue slots: test `mwaitx` against a `pause` spin.
   - Better route: the unpredicted cold read can overlap the same layer's hot-expert GEMVs if the GPU does not wait on the CPU first (hides ~60 % of the remaining 0.30 ms, no prediction needed). Over-prefetch: the DDR4 window is 14x the need; prefetch pre-gate top-12/16 + kNN two-ahead + previous-token repeats; measure the union's coverage offline.
2. **C1: arithmetic holds; C1 is a >= 32K feature.** KV read at 32K is 0.71 ms (7 % of a token), at 128K 23 %. Keeping 72.6 % of the mass = 27 % TV; q8_0 keys move 0.49 %, int4 keys 8.68 %: working bar (interpolated, to be set by a whole-model run) >= 95-97 % mass kept. Test order: (1) long-context q/k dump (4K, 16K, 32K), oracle mass at 1/8 and 1/32 vs length; if the oracle does not clear ~95 % at 1/8 by 32K, go to per-head budgets; (2) then selectors: union with sink + recent; 1-bit pick of 2x + exact rescoring (0.70 ms at 128K, 4x under 2.83); (3) score on attention-output error / whole-model KL, not mass alone.
3. **C3/C6: formula incomplete; reframe as a speed feature.** Missing precision and recoverability R (share of a stuck token's KL the expensive path removes). KL recovered = P(flag) x precision x R x KL_stuck (0.31 per stuck token, WikiText); time = P(flag) x cost (cheap for every flag measured). R is probably small (stuck tokens shared with Unsloth's file, reference unsure there, chunk 24 non-monotonic; the expensive path is only H planes on ~7 cold experts, ~2 % of KL). So the flag licenses cheap paths on unflagged tokens. Test R first: oracle-flag paired run on the top-1 % positions (~25 min); if R < 0.2 stop the flag work. Engine flag: free in-engine signals: the 2-3 tokens after a rejected draft (runs: stuck at t-1 -> 11x) + number/math tokens (2x, known before the forward pass). A C6 kNN scan over 1 M entries is 25 ms per token: the lookup must be an index (hash buckets or the author's RT BVH).
4. **C5: direction right, ratios ~10-15 % optimistic.** Verify of M rows reads more than one main-model pass: 5 rows touch ~28-32 distinct hot experts (~2.3 GB). Rung-1 cycle at M = 5: 147 tok/s (1.42x) at a = 0.7, 178 (1.73x) at 0.8. "5 rows gain nothing today" does not follow from the ledger (95 vs 54 tok/s derived); the tier build's 1.13-1.17x is the only data. n-gram / fuzzy drafts cost no model read.
5. **MTP 7 vs Q3: holds; stay at 7.** The head is 417 of 470 MB per draft: a 1- or 2-bit sketch of the output head (64 / 127 MB) picks 64 candidates, exact Q6_K rescoring of those 64 rows keeps the argmax exact whenever recall holds: draft 470 -> 117-180 MB, a 3-draft cycle -25 % (13x what Q3 buys). Gate: top-1-in-top-64 recall of the sketch from existing logits.
6. **Missing:** the ledger is short-context (at 128K add 2.83 ms of KV; DeltaNet state traffic 126 MB per token belongs in the floor); C6 kNN prediction (43-53 % two ahead) is a union member, not a replacement; DDR4-3600 also needs FCLK 1800; J1 condition stands.

## Most valuable next tests (Fable's order)

1. GPU step split: read-only ceiling kernel + one nsys profile of a full token.
2. Decompose the 10.1 ms cold step (reads, handoffs, residual).
3. bench_cpu four cells {1, 8 threads} x {4 KB, 2 MB pages} with software prefetch; `mwaitx` vs `pause`.
4. Counts-only from the K route files: distinct hot experts for M = 2..8 (C5 verify cost); pre-gate top-8/12/16/24 coverage, union with kNN and repeats.
5. C1 oracle mass vs context length (4K/16K/32K), then union sink + recent and 2x rescoring.
6. C3/C6 recoverability R: oracle-flag paired run (~25 min).
7. MTP head sketch: top-1-in-top-64 recall from existing logits.
8. Draft-rejection flag precision proxy from existing dumps.
