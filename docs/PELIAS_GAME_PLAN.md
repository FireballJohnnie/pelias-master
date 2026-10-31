# PELIAS GAME PLAN: tests first, then fixes, then the build, then the final tests

Written 2026-10-27 at 13:5x, at the author's request. Every step has enough detail to write its script without guessing.

Copyright (c) 2026 John Lipka and the Pelias contributors

**What this file is:**
- The single list of what happens next, in order.
- Status: what is settled (with results), what is retired, what is open.
- Part 1: tests that must run before any further building.
- Part 2: fixes that follow from those tests.
- Part 3: the rest of the build.
- Part 4: the tests that tune and finalise the finished build.
- Theories to get closer to the speed ceiling and to bf16 accuracy, each with its test.
- Appendices: the ID map (old names to new), the decisions waiting on the author, the scripting rules, and the sources.

**Labels** (same as every Pelias document): MEASURED, READ-SRC, PUBLISHED, DERIVED, ESTIMATE, UNVERIFIED, DECISION, AUTHOR-STATED. Every time estimate names its basis.

**Owners:**
- **A** = the author runs it on the PC.
- **C** = Claude writes, builds and checks it in the sandbox.
- **A+C** = Claude writes it; the author runs it.

**Privacy:**
- No dataset text, persona wording or private equations appear here.
- Machine paths are variables. Their values live in `profiles/local_only/GAME_PLAN_PATHS.env`, which git ignores.

---

## Open items, whole build (sweep 2026-10-31 14:3x; merged with the author's consolidation 15:09)

~~Struck~~ = done or superseded by where the build stands now. "(confirm)" = no clear ruling found in the logs. "Corrected:" = a number or name in the author's 15:09 draft checked against the logs and fixed. Every leftover in Appendix E maps to an item below. The Theories section's tests are all GP3-10, GP3-14 or GP4-02.

### Struck since the 14:3x sweep (author 15:09, checked against the logs)
- ~~`r2b_exps.gguf`~~: already gone. It was moved to a to-delete folder on 10-28 at the author's request (BUILD_LOG), and the three to-delete folders are empty now (checked 15:1x). D: has 116 GB free (MEASURED 14:49). Corrected: rotating the routed experts was no different from stock (R2b fp32 0.039544, +0.000530 vs stock, t 1.14, not resolved, MEASURED); it sits +0.003980 above dense-only rotation R2 (0.035564; DERIVED from the two MEASURED runs). The 57 GB free figure is older; nothing needs deleting.
- ~~O13 persistent-tile route~~: closed 10-30 20:08-20:12. v13 ran at 0.84-1.03x of v12 and v13b at v12's speed ("fill/drain not the cost", MEASURED). The locked S3b MoE slice stays: GPU-built work list 0.040-0.130 ms vs host build 0.20-0.45 ms; whole slice 1.30-1.39x at 255 tokens, 1.11-1.15x at 2,040 tokens (MEASURED). **Corrected:** O13 itself stays open. The down projection still runs at 0.27-0.29 of peak; only this route to it is closed.
- ~~GP1-12 per-layer ffn_down sweep~~: confirmed. The 40-layer slope rerun found Q4_K relative error nearly flat over layers, with down 10-35 % above gate/up only in layers 32-39 (MEASURED). That is built into the tiered quantizer.
- ~~GP1-16 CPU speed of the shared format~~: confirmed. AVX2 decoders at 16 threads, float / int8 GB/s: Q8_0 35 / 39, Q6_K 38 / 36, Q5_K 31 / 27, Q4_K 22 / 28, IQ4_XS 24 / 29, against 8.5 for the reference path (MEASURED). Q4_TG2 17 / 17 and IQ2_XXS 10 / 11 are below the 27 GB/s gate (DERIVED gate); they move into Line C.
- ~~R2-256 / R2-row~~: struck. **Corrected reason:** the +0.000999 in the draft is the cost of one *activation scale* per 256 (the gpu256 format, MEASURED), a different thing from the *weight rotation* block size these two runs would test. They were to run "only if Block C lands close to Unsloth". The tiered file is 26.6 % below Unsloth (0.009404 vs 0.012808, MEASURED), so the condition never came up.
- ~~v18 first engine run~~: done 2026-10-31 14:50-15:13 (MEASURED, gp/ENGINE). Build clean in 38 s; 27 / 27 GPU tests; decode 12.1-13.0 tok/s; sanitizers 0 errors from all four tools (the only test failure under them is verify/keep at 1.07e-4 over 1e-4, the drift below). Its follow-ups are in the GP3-14 section below.

### Post-v22 order (author 17:39, from the GPU / CPU mismatch; checked against the logs; this order governs, the 15:09 lines below stay as the item detail)

**Framing (MEASURED v19-v21):**
- The engine's step was the GPU and the CPU taking turns: per token, the CPU waited 28.0 ms for the GPU and the GPU waited 15.0 ms for the CPU.
- Each side paid per job rather than per byte:
  - CPU: 5.9 GB/s, threads 36-39 % busy.
  - GPU: two slow operations. The router + top-k took 13.0 ms (a one-thread serial loop over global memory). The bank experts took 21.6 ms (grids launched for all 256 experts, almost all empty).
- v22 fixes the GPU pair and runs the bank self-test (results pending at the time of writing).

**Standing rule (author):**
- Engine gates and speed numbers use real text, never made-up token IDs. v18 → v19: picks on the GPU 40 % → 64 %, and the top-1 gate went from FAIL to PASS on the same engine (MEASURED).
- Made-up IDs stay fine for the small-model bit-for-bit tests.

**Strikes (confirmed; corrections to the 17:39 draft):**
- ~~r2b_exps.gguf~~: already gone. It was moved 10-28 and the to-delete folders were empty at 15:1x. D: has 116 GB free (not 57). Corrected reason: R2b (experts rotated) showed no difference from stock (+0.000530, not resolved); the +0.003980 is its gap to dense-only R2 (DERIVED).
- ~~O13 persistent-tile route~~: closed. v13 ran at 0.84-1.03x of v12, v13b at 1.00x. Corrected:
  - The bank's 21.6 ms tie-in to empty blocks is v22's hypothesis until v22 measures it.
  - O13 itself (the down projection at 0.27-0.29 of peak) stays open, behind the steps below.
- ~~GP1-12~~ and ~~GP1-16~~: confirmed.
- ~~R2-256 / R2-row~~: struck. Corrected reason: 0.000999 is the cost of one activation scale per 256, not the rotation size. The runs were conditional on Block C landing close to Unsloth, which never happened (tiered is 26.6 % below).

**Step 0 (new): the split fault.** v22 (MEASURED): no bank expert is wrong on its own: 5,496 / 5,302 checks + 2,560 mixes, worst gap 2.1e-03. The fault repeats bit for bit and is input-dependent. v23 probes the residual stream and the router picks at that position, layer by layer, in both paths against the CPU reference, to find the first layer that departs. Then fix; the split check must then match plain decode.
**v22 standing (MEASURED):** decode 21.0 tok/s (from 16.2) with the shared-memory router and the expert grids sized for the picks, logits unchanged. Bank experts are still 15.1 ms per token, router + top-k 2.3. MTP stays below 1x, because the verify's CPU cold work (68-70 ms) now dominates. Step 1 (the CPU) and the drafts' cold experts on the GPU are the MTP levers.

**Step 1: the CPU cold-expert gap.**
- **Target:** cold work is 30 ms per token (v20) at 5.9-7.1 GB/s.
- **Corrected ceiling:** most cold bytes are Q4_TG2, the floor type for little-used experts. Its CPU decoder measured 17 GB/s; 28-39 GB/s is Q4_K / Q6_K / Q8_0 int8. Getting past 17 needs the Q4_TG2 decoder raised (game plan C6, byte-plane grid). RAM reads at 40-42 GB/s (MEASURED).
- 218 MB per token at 17 GB/s = 12.8 ms; at 25-35 GB/s = 6.2-8.7 ms (DERIVED).
- **Action:** one dispatch per layer. Workers wake once at the ring, then run gate/up + silu and down over all cold picks with one atomic barrier between them and no re-dispatch. Plus the thread count and the spin-vs-sleep policy (corrected: these are the engine's own counterparts of the tier build's `-t 7` / `--poll 100`).
- **Corrected:** Fable's CPU items are for the CPU reference / kld path, not the engine's cold path: AVX2 in the quantizer, the 3.2 GB of zeroes, the 128 KB copy tile, rounding shared inputs once. The persistent pool is already built (GP3-14). The engine's cold path already uses kern::dot_row_q8 (AVX2). Fable's items move to Step 5, where kld speed matters.

**Step 2: the CPU's wait for the GPU (U8 + O15 + int32 fold).**
- **Corrected:** the doorbell already rings right after the router (v21 stamps: picks to host + ring sit between the router and the bank experts / shared expert).
- **Corrected:** v19's "dense 22.6 ms" included the router's 13.0 ms (v21). After v22 the pre-ring GPU work is about 12-13 ms per token: products 7.0, core 2.2, norms / small ops / residuals ~3, router under 1 (DERIVED). The non-product share is ~5 ms, not 15.6.
- **Action:** U8 (one R3 plan per token + CUDA Graph; replay 0.716 µs vs 4.70 µs eager, MEASURED), O15's two untested doorbell orderings, int32 fold.

**Step 3: MTP.**
- **Target:** drafts 3.1 ms each; verify of 2 / 3 / 4 tokens 117 / 132 / 157 ms; GPU busy 45 → 66 / 79 / 89 ms (MEASURED v20).
- v22's grid change already applies to verify batches (no tile launch at 8 tokens or fewer).
- **Actions:**
  - R3 reading each weight once for batches of 2-4.
  - Drafts' cold experts read by the GPU from host memory (dual bus; kept from the 16:34 order, step 4). Without it, the verify's CPU work still grows with its tokens.
  - MTP bank uploads only the swapped experts. Corrected: an expert is 1.8-3.3 MB by tier, not 1.9 (DERIVED); copy and compute together measured 1.97-1.98x of an ideal 1.99x.

**Step 4: second text + context (Q4 / GP1-10, B4, GP2-11, the locked file on a second text, hot-set re-profile, GP1-09, B6 / GP2-06, DeltaNet path).**
- **Corrected:** 64 % (v19-v22) and 77.5 % are both WikiText:
  - 64 % is the engine's bank of 7.93 GB on held-out WikiText (tokens 102,400 on).
  - 0.775 is the tier build's live hit rate with a 7.0 GiB bank.
  - Chat+code is a separate measurement: WikiText's top 64 cover 35.9 % of its picks.
- **Corrected:** B4 needs a held-out chat+code text. The chat+code text behind 35.9 % is Unsloth's calibration text, which tier_v1's importance matrix was built on.
- **Corrected (DeltaNet):** lock the registers kernel + checkpoints for prefill, decode and verify. 2.01 ms per 1,000-token prompt vs C16's 3.12 ms; decode 11.97 µs vs 27.0 µs (MEASURED). C16 stays only if GP1-09 shows it ahead at 4K+.
- B6 / GP2-06 as ruled 15:09 (KV q8_0 + int8-row checkpoints every 256 tokens), validated on GP1-09.

**Step 5: parked accuracy, after Step 1.**
- Fable's CPU items, then `PELIAS_PROF=1 kld --chunks 2` and the 8/12/16-thread kld.
- Tie band B = 0.1 vs the top 10 % / 20 % of block-scaled error energy, against +0.001009 KL.
- Planes in the engine, top 4 picks (0.008800, 92 % of the every-pick gain, 0 VRAM).
- **Corrected (name):** B9 / GP2-12 is the Pelias-made *scoring* base (bf16 reference logits). The master file (rotated Q7_ZT dense at 7.53 bpw and −236.4 MB + tiered experts + v2 dense and head planes) is a separate item. "5.7-6.9 % faster" is the Q7_ZT fold route vs Q8_Z.

**Step 6: native Windows (GP4-04 + BAR1 + NVML) for the GP4-01 / GP4-02 benchmarks.**
- Nsight gave an empty trace under WSL (v20).
- **Corrected (P2):** v18 sat in P2 for 98 % of samples at 11,251 vs 11,501 MHz, about 2.2 % of bandwidth (MEASURED / DERIVED). Windows also showed P2 at idle (15:11). Whether Windows holds P0 is UNMEASURED, so the gain is at most ~2 %.
- BAR1 flag 0.8-1.4 µs vs 2.40 µs; uploads 12.5-12.7 → 22.5-22.7 GB/s (MEASURED).
- The engine needs its Windows build first; block_wingpu_v2c built the earlier kernels at 30c78ab.

### Execution lines (the author's grouping, 15:09)
**Run order:** superseded by the post-v22 order above (author 17:39).

**v18 friction → line (MEASURED 14:52-14:59):**
- Host waits of 24-26 ms per token → Line A.
- CPU cold work of 50-57 ms per token, within a 77 ms median step → Line C.
- Only 39-42 % of picks on the GPU (the bank holds 2,748-2,926 of 10,240 experts) → Line B.

**Line A: U8 (one R3 plan per token + CUDA Graph) + doorbell counter inside R3 + int32 fold + O15**
- Gap to close: R3's real token measures 5.685 ms against 4.75 ms of pure streaming (about 478 GB/s). The read ceiling is 494 GB/s, about 89.5 % of the card's 552 GB/s as run (MEASURED).
- Graph replay costs 0.716 µs against 4.70 µs per eager kernel (MEASURED, P-H9).
- The deterministic int32 folded combine runs at 1.08x, against 1.13x for the f32 fold, at 2,040 tokens (MEASURED).
- Doorbell: the release store is 2.0 µs and the round trip 2.40 µs (MEASURED).
- Plan: one R3 launch per token. The cold-expert completion flag becomes a host-mapped counter inside R3's counter array, so the combine waits inline. The launch is wrapped in a CUDA Graph. The int32 fold and O15's two untested ordering forms are fixed in the same pass.
- **Corrected:**
  - "5.685 ms vs 6.06 ms grid sync" mixes two measurements. Fix 6 measured counter barriers at 6.16 ms against 6.06 for the grid sync (5.91 with one-task prefetch). The 5.685 came later, with the Q4_TG2 grid in shared memory and per-expert counters.
  - The 0.52 µs is the grid sync itself.
  - "19.2 ms vs 24.0 ms": 19.2 ms is graph replay of path (a) on both WSL and Windows. 24 is the µs drain/ramp per stage boundary, not a ms figure.
- Test: real-token time against 5.685 ms; v18's host waits against 24-26 ms per token.
- **v19 timeline (real text, decode, MEASURED, ms per token):** GPU dense 22.64, bank + shared experts 27.12, idle waiting for the CPU 15.02; CPU waiting for the ring 28.03, cold 37.11 (gate/up 17.62, silu + rounding 7.30, down 11.62); cold 218 MB per token at 5.9 GB/s; thread use 39 % (gate/up) / 36 % (down) of 16; first worker starts 21.9 us after a job opens. The GPU is busy ~50 ms of a 65 ms step, so the GPU's own path (dense on separate launches, the bank's experts through the batched prompt kernels at n = 1) is as large a cost as the CPU's; R3 streams a whole token in 5.685 ms (MEASURED). Line A's one-launch R3 + the bank's experts on R3 come first; Line C's CPU fixes target the 37 ms (fused per-expert jobs: the silu stage alone costs 7.3 ms of barrier time).

**Line B: second text (Q4, GP1-10, B4, GP2-11, the locked file on a second text) + hot-set re-profile + MTP bank delta swaps**
- WikiText's top 64 experts per layer cover 35.9 % of chat+code picks, against 72.2 % of its own. The rank correlation averages 0.41 per layer, lowest 0.22 (MEASURED).
- Token-time formula t = D/(e·B) + h·X/(e·B) + (1−h)·X/C + o, with X = 0.532 GB and C = 42.5 GB/s (DERIVED).
- The live hit rate was 0.775-0.779 with a 7.0 GiB bank on WikiText (MEASURED, tier build). DDR4 saturates at 47.8 GB/s at 2 threads (MEASURED), about 93 % of 51.2 (DERIVED).
- **Corrected:**
  - h = 0.36 on chat+code is not measured. The 35.9 % is a top-64 coverage, not a hit rate, and the live h on chat+code is UNMEASURED; this line measures it (engine pick share on the second text).
  - At h = 0.36 the extra cold read would be (0.775 − 0.36) × 0.532 GB ≈ 221 MB per token (DERIVED), not 212.
- **Headache found:** the chat+code text behind the 35.9 % is Unsloth's calibration text (842,201 tokens), and tier_v1 uses Unsloth's importance matrix, which was calibrated on it. The GP1-10 rule says never evaluate on a calibration text. B4 must therefore pick a held-out chat+code text.
- Plan:
  1. bf16 base on that text: about 57 min at 100 × 512 (MEASURED for WikiText).
  2. tier_v1 scored on it. Projected about 0.0110 base and 0.0092 with 2 GB planes (DERIVED).
  3. The same run's routing trace re-profiles the hot set.
  4. MTP bank promotions upload only the swapped experts, over the copy engines. Copy and compute together reached 1.97-1.98x of an ideal 1.99x (MEASURED).
- **Corrected:** one expert is 3 × 2048 × 512 weights, about 1.8 MB at Q4_TG2 and 3.3 MB at Q8_0 (DERIVED), so a swap moves 1.8-3.3 MB by tier, not a flat 1.9.

**Line C: Fable's CPU fixes → 8/12/16-thread kld + tie band vs top-block error energy + planes in the engine**
- Profile of one 512-token CPU chunk: 15.98 s = experts 8.00, DeltaNet 4.50, attention 1.16, head 0.74, shared 0.64, router 0.32 (MEASURED).
- v17b: Tctl 91.0-91.6 °C, 102-104 W, 14.1-14.35 s per chunk (MEASURED).
- Fable's five fixes: persistent pool, AVX2 in the quantizer, drop the 3.2 GB of zeroes, drop the 128 KB copy tile, round shared inputs once. Estimated gain 7-12 s per chunk (ESTIMATE, not measured).
1. Land the five fixes. Raise the Q4_TG2 (17 GB/s) and IQ2_XXS (10-11 GB/s) CPU decoders toward the 27 GB/s gate (game plan C6: byte-plane grid). Verify with `PELIAS_PROF=1 kld --chunks 2`, then the 8/12/16-thread kld sweep.
2. Tie band B = 0.1 (touches about every 32-wide weight block, DERIVED) side by side with the top 10 % and 20 % of 32-blocks by scaled error energy.
   - Measured against the rounding cost +0.001009 KL (t 8.27, 10.7 % of 0.009404), whose worst 1 % persists 7x at t+1 (7.09 % vs 1 %; KL 9x), MEASURED.
   - **Corrected:** this needs new kld runs, one per variant, about 24 min each before the fixes. Tonight's dumps hold per-token KL, not the activations a new rule needs.
3. Planes in the engine on the same CPU pool, read for the top 4 picks.
   - 0.008800 KL (−6.4 % vs tiered, t −2.88, MEASURED), keeping 92 % of the every-pick 2 GB gain (0.008747, DERIVED) at half the plane reads, 0 VRAM.
   - **Corrected:** −31.7 % vs Unsloth's 0.012808 is the every-pick figure (0.008747). Top 4 is −31.3 % (MEASURED).
   - These numbers are from the CPU reference; the engine must match them.

**Line D: DeltaNet path + KV default (B6, GP2-06) + GP1-09 (4096 base) + GP3-14c (infinite context)**
- Registers kernel: decode 11.9-11.97 µs per token, 2.01-2.18 ms per 1,000-token prompt, 2.33x plain recurrent, lowest energy (MEASURED, v5, v7 and v18).
- The checkpoint variant equals it bit for bit, with free rollback for rejected drafts (MEASURED v18).
- C16 chunked rows: within 3.1e-7 of fp64 (MEASURED).
- **Corrected (the prefill choice):** the registers kernel also wins the prompt. 2.01 ms (v18) / 2.18 ms (v5) per 1,000 tokens against C16's 3.12-3.13 ms, and its decode (11.97 µs) beats C16's 27.0 µs (MEASURED). Lock registers + checkpoint for prefill, decode and verify. C16 is kept only if chunking wins at 4K+ prompts (UNMEASURED; GP1-09 measures it).
- **B6 / GP2-06 (author 15:09):** KV q8_0 + int8-row DeltaNet checkpoints every 256 tokens.
  - Costs: KV q8 +0.000350, t 2.00, growing with context: +0.00066 at positions 384-510. State int8row +0.000356, t 2.07, one restore. Rejected: KV int4r +0.004156; state int4r +0.032824 (MEASURED).
  - A 4K repeat skips about 1.8 s (DERIVED).
  - Because the KV cost grows with context, the 4096 base (GP1-09) re-measures both.
- C1 1-bit page gate: keeps 72.6 % of attention mass at 1/8 of keys, vs Quest 62.0 % (MEASURED).
- GP3-14c: 1M tokens ≈ 10.9 GB of q8 KV; a 64-page pull takes under 2 ms from RAM (DERIVED).

**Line E: combined file + Pelias-made base (B9, GP2-12) + v2 dense and head planes**
- **Corrected (name):** B9 / GP2-12's "Pelias-made base" is the *scoring reference*: Pelias's own bf16 logits, to replace llama.cpp's bf16 base for rankings. It is not a model file.
  - The floor 0.002801 is the gap between the two references: ~23 % of urm_TG2's 0.012056 and ~32 % of 0.008747 (DERIVED).
  - The new file is its own item. Both are done in this line, and the file is scored against both bases.
- Combined file, from BF16:
  - Dense: rotated Q7_ZT (7.53 bpw), with blk.40 eh_proj rotated (−49.5 % error, MEASURED).
  - Experts: usage-tiered on the Q4_TG2 floor.
  - **Corrected:** urm_TG2 0.012056 at −236.4 MB; tier_v1 0.009404 (−26.6 % vs 0.012808) (MEASURED). The Q7_ZT speed figure is the fold route: 5.7-6.9 % faster than Q8_Z (probe v2, MEASURED), not "large matrices" in general.
  - The ~236 MB saved pays for the v2 dense and head planes at the same total size. Head routes: a 3rd head plane 0.067 GB or frequent-token rows 0.13 GB, against 1.0 GB for the exact head re-score. That re-score gained 0.005818 → 0.005741 (−1.3 %, t −16.8) on the Unsloth file with 2 planes (MEASURED; routes DERIVED).
- **Corrected:** Q6 / GP1-08 (re-quantize the same bytes) stays in the trainer phase. It decides the trainer's Mode Q; building this file from BF16 does not answer it.

**Line F: Windows-native vs WSL (GP4-04) + BAR1 + NVML and ceiling reads**
- WSL2 has no BAR1 mapping. A VRAM flag answers in 0.8-1.4 µs on Windows vs 2.40 µs for the system-RAM flag (MEASURED, graphics probe). Uploads 12.5-12.7 → 22.5-22.7 GB/s, 1.8x (MEASURED).
- Windows GPU build, gp/block_wingpu_v2c, 30c78ab: clean, 16 / 16, digits identical to WSL, host 86 / 0, memcheck 0 (MEASURED).
- **Corrected:** the "logits within 2.6e-6, top-1 14/14" in the draft belong to other runs. 2.6e-6 is the Windows CPU (MSVC) check at 14:28; 14 / 14 is the BitNet top-1 against transformers.
- **Corrected (P2):** v18's sampler shows the card in P2 for 1,647 of 1,684 samples, memory 11,251 MHz vs 11,501 MHz in P0 (MEASURED). That is about 2.2 % less bandwidth (552 → ~540 GB/s, DERIVED), not a large throttle; the 494 GB/s ceiling was measured on both WSL and Windows. Windows also showed P2 at idle (15:11). Whether a Windows run holds P0 is UNMEASURED; the run logs pstate and memory clock.

### Still open outside the lines
- **GP3-14 engine, v18 follow-ups:**
  - The batched verify path drifts from the one-token R3 path: verify/keep KL 6.3e-05 vs 1.0e-05; under memcheck 1.07e-04 over the 1e-4 test limit. MTP plain vs drafted split after token 16 (draft 1, 2) and after token 1 (draft 3) (MEASURED v18, made-up IDs).
  - **v21 (MEASURED):** the fault is a bank expert giving a wrong result (the path that is off flips with the bank size; both wrong answers pick token 16 where the reference says 24 by 10.35 logits; absent with the bank off); it hits plain decode too. Next: a bank self-test (every bank expert, 1 and 2 rows, against the CPU path) to name the slot. GPU per operation (ms per token): router + top-k 13.0 (F32 product + top-k, 0.325 per layer), bank experts 21.6 (0.54 per layer), wait for the CPU 11.2, R3 products 7.0, shared expert 4.5; the GPU's "flat cost" is two slow operations, not launches in general.
  - **v20 (MEASURED):** grouped CPU cold path + silu fold kept (cold 37.1 -> 30.0 ms per token, prefill 27 -> 35 tok/s, logits bit for bit). MTP: drafts cost 3.1 ms each; verify steps of 2 / 3 / 4 tokens cost 117 / 132 / 157 ms against 68 for one (GPU busy 66 / 79 / 89 vs 45; CPU cold 69 / 73 / 85 vs 43), so the batch path on the GPU grows with its tokens too (correction to "flat per call": that holds across layers at 1 token, not across batch sizes). The fault: needs the bank (absent with the bank off), not the doorbell, not verify / checkpoints, not the batch size (2-8 identical); one position of the greedy text, with 2 of 4 bank sizes. Next check: both rows at that position against the CPU reference, to see which path is wrong. Nsight Systems gave an empty trace under WSL.
  - **v19 split check (real text, MEASURED):** verify batches of 2 and 3 vs one token: KL 2.3e-03 / 2.2e-03, top-1 126 / 127 and 127 / 127, i.e. the rounding floor; the draft-1 split sits at a one-token margin of 0.108 (a near-tie), draft 2 equals plain for all 128. **Batches of 4 are a fault:** KL mean 1.43e-02, max 1.17; the draft-3 split at token 114 has a one-token margin of 10.93 logits, which no rounding explains. Find it before any MTP tuning.
  - ~~Float gate top-1 555 vs 566 on made-up IDs~~: resolved by v19 on real text (WikiText, held out). Gate PASS: engine 623 / 640 vs CPU 619 / 640 against float; KL 2.906e-03 vs 3.780e-03; McNemar 12 vs 8, p 0.503; all 17 engine misses sit where float's top-2 margin is under 1.0 logit (15 under 0.1) (MEASURED). The v18 fail was the made-up prompt.
  - ~~The 3 skipped unit tests~~: the usual data-file skips (the Unsloth header file is kept on the author's drive).
  - O13 (down projection at 0.27-0.29 of peak); gates E4-E9, G3, K3.
- **GP1-14:** spec checks recorded as DONE in the docs.
- **Appendix B:**
  - Struck: ~~B1~~ (locked tiered path), ~~B3~~ (v2 calibration text), ~~B5~~ (per-32 format and the floor gate), ~~B7~~ (tier types), ~~B10~~ (embedding on the host), ~~B11~~ (budget-derived), ~~B13~~, ~~B14~~.
  - ~~B6~~ ruled by the author at 15:09 (Line D).
  - Open: B2 (PEL bytes for AWQ/GPTQ; Q4_TG2 is already a precedent), B4 (held-out chat+code text, Line B), B8 (parameter conflicts), B9 (Line E), B12 (LoRA scaling).
- **GP2 docs:**
  - Struck: ~~GP2-01~~, ~~GP2-05~~ (the correction itself is Line C), ~~GP2-09~~, ~~GP2-06~~ (Line D).
  - Open: GP2-02 / 02b (status corrections, stale records), GP2-07 (parameter conflicts), GP2-08 (data port), GP2-11 (Line B), GP2-12 (Line E), GP2-13 (namespace), GP2-14 (version string). GP2-03 is in the trainer phase.
- **GP3:**
  - Struck: ~~GP3-01, 02, 07, 08, 09, 10, 11~~; ~~GP3-06~~ (flips don't carry the tail); ~~router fp32 check~~ (the router is F32 and never rounded).
  - Then: small-card layout, GP3-03 AWQ, GP3-04 GPTQ, GP3-05 data pipeline, GP3-12 backward/LoRA, GP3-12b, GP3-13 repair training, GP3-13b idle-work scheduler, GP3-15/15b server and web, GP3-16 steps 13-17.
- **Trainer phase:** Q6 / GP1-08, GP2-03, B12.
- **Parked accuracy:**
  - Split-token grade.
  - Real text and long prompts on the GPU (the second text in Line B and 4096 in Line D cover part of this).
- **Build/platform:** native Linux (GP4-04).
- **GP4 release:**
  - GP4-01 benchmark
  - GP4-02 sweeps (bank size, Lane D, `-t 7` / `--poll`, huge pages, live hit rate)
  - GP4-03 acceptance; restore the persona criterion
  - GP4-04 hardware
  - GP4-05 gates; trademark check
  - GP4-05b future tests
  - GP4-06 freeze
- **Housekeeping:**
  - Display-sleep set to never (author 14:44).
  - A WSL window stays open during runs: WSL powers itself off when idle (14:35 and 14:41 today).
  - Use `stdbuf -oL` in run scripts.
  - The case fans don't ramp under all-core load.

---

## QUEUED, MUST NOT BE LOST (author 2026-10-30 22:19)

**PDF update:** how the speedup works rev4 -> rev5; Section 21 calibrated gains rev4 -> rev5; 7-bit and 1-bit decoder plain language rev3 -> rev4; pathways whole project v6b -> v7. Skip pathways v5 (the older one, overlapping v6b). Sources and generators in the backup folder (2026-10-28). Content: the 10-30 evening work (fused norm, per-32 prompt format, the GPU work list, the locked MoE slice, gpu_prompt in src, the fp32 min term, the Q4_TG2 probe results). Slot: after the Q4_TG2 probe code ships, before the GP3-12 rungs.

**DONE 2026-10-31 03:4x (author 03:37 "Update the PDFs"):** speedup rev5, Section 21 calibrated gains rev5, 7-bit / 1-bit decoder rev4, pathways v7 (pathways v5 skipped as asked); copies in docs/pdf/ and the backup folder. Content added: the GPU forward pass and the rounding floor, the new real-model gate, the activation-rounding cost and its 7x persistence, the tie rule and the block-energy plan, the one-copy weight layout, GP3-14 coded, the CPU heat and the code review.

## Integration status (2026-10-30 22:4x, MEASURED; BUILD_LOG 22:13-22:33)

- v10b (ef9fb0b): the Q4_K prompt min term in fp32 FMAs: 1.4-2.4e-07 vs fp64, four sanitizers clean; TG2 probe: 1.00-1.02x the bf16 mma's time. Proposed LOCK (author's 22:09 condition met).
- Q4_TG2 prompt path (TG2 probe, e0117db): int8 x 3 (24-bit fixed point, three int8 mmas) is the fastest variant under the 1e-6 gate (1.1-1.7e-07; 1.34-1.57x the Q4_K tile time); warp-vector rows with the exact fp32 table (1.1e-07; 1.7-1.9x the Q4_K vector). Proposed LOCK; awaiting the author.
- Open before GP3-12: the probe's dense 2048 x 2048 Q4_K case at 0.507 relative L2 (both min variants); first check: a 2048 x 2048 dense case in src's GPU test.
- Order (author 23:22 "Go ahead", 23:43 "Yes"): lock (done) -> the dense check (done: probe upload race; src 2.38e-07) -> Q4_TG2 into kern/gpu_prompt (v12 running) -> the queued PDF update -> **exact paths get the GPU work list, the int32 fold and the fused norm's float output; Q5_K and Q6_K int8 prompt tiles (exact: Q5_K 0-31 with Q4_K's min term, Q6_K -32..31 with int8 sub-block scales; DERIVED, to be measured)** -> GP3-12 rungs -> GP3-14.

**Note (author 2026-10-31 00:15): the power / temperature watchdog (gp/watchdog_v1.sh) is a test-harness tool, not part of the program.** Its limits (200 W, the CPU's PPT) are this PC's. If the product ever guards power or temperature, it reads each machine's own limits from a probe (NVML power.limit and temperatures; the CPU's PPT / TDP from the platform) and never assumes these numbers; other GPUs and CPUs have other limits.

## CPU path speed items (Fable read-only review 2026-10-31 01:4x-01:55; author 02:36 "add"; measure first, results unchanged)

Context (MEASURED): kld on the tiered file, 16 threads, 14.1-14.35 s per 512-token chunk; CPU pinned at Tctl 91 C (its 90 C
limit), 102-104 W, CPU fan at its ceiling; the chip regulates temperature, so these items shorten runs (energy), not peak heat.
They also sit in the CPU expert path GP3-14's cold experts use. Every item keeps every reported number bit for bit (same
summation order). Savings are Fable's ESTIMATES against the PELIAS_PROF split of 2026-10-30 (experts 8.0 s, DeltaNet 4.5 s per
chunk); first step: `PELIAS_PROF=1 pelias kld ... --chunks 2` on the tiered file (about 1 min) to measure the split.
1. **Persistent thread pool** (author priority): parallel_items spawns and joins a thread team per product, about 7,200
   clone/join per chunk (DERIVED). ESTIMATE 0.2-0.5 s per chunk.
2. **AVX2 for the dequantizers** (author priority): pelias_quant is compiled without -mavx2 (CMakeLists.txt), so Q4_TG2 / Q5_K /
   Q6_K row decodes run scalar; keep -ffp-contract=off / no fast-math. ESTIMATE 1-1.5 s per chunk.
3. **Drop the 3.2 GB of zeroed buffers per chunk** (author priority): per-layer std::vector buffers value-initialised then fully
   overwritten (DeltaNet about 86 MB, attention about 64 MB per layer; ys 32 MB mmap'd and faulted each layer); one workspace
   per call sized to the largest layer. ESTIMATE 0.5-0.8 s per chunk.
4. **Drop the 128 KB fp32 copy buffer in the expert tile** (author priority): load_tile decodes each row to fp32 and scatters
   it into a 2048 x 16 x 4 B tile (4x L1); transpose per 256-value block in L1 and carry the accumulators. ESTIMATE 2-4 s per chunk.
5. **Round a shared input once** (author 02:36): mm() rounds the same normed input 3x for wq / wk / wv, 2x for the DeltaNet
   wqkv / wgate and 2x for the shared expert gate / up; round once, reuse (identical values). Small time; also makes the flip
   counter per value instead of per product.
6. Next by size: dense tile_dot at 20-30 % of peak (check the disassembly for accumulator spills, 2-3 s if so); DeltaNet and
   attention dot products as latency chains (transposed state / K, same i-order, 0.6-1.2 s); base-file read serial (prefetch).
Parked (author 02:36): the 8 / 12 / 16-thread CPU sweep, until another CPU sweep is needed.

## STANDING RULE: GPU first, exact CPU late (author 2026-10-31 01:16; the dual-act formula of 2026-10-29 17:44)

Apply to every area the 1-bit accuracy + plane system does not cover:
1. The GPU answers first, fast (rounded / approximate), inside step t.
2. The CPU answers late with the more exact value (fp32, unrounded, or the higher-precision path), only where flagged.
3. When the two disagree, the CPU's answer wins (it is the more exact one); weight the CPU more.
4. A CPU answer too late for step t still counts: it makes t+1 careful ("careful+1"); hard deadline, never wait.
5. Each use is measured first (does the flag at t predict a miss at t+1, as the 11 % vs 1 % miss pattern did), then wired.
First use: the activation coin flips (next section).

## Activation coin flips: GPU first, exact CPU late (author 2026-10-31 01:07-01:14; dual-act logic of 2026-10-29 17:44)

**Why (MEASURED, v16, 40 layers, 64 made-up tokens):** GPU and CPU round the same activations to int8 (per 32); values a hair
from halfway land either way (fp32 sum order), and the layers carry the flips (CPU fast vs plain kernels KL 5.3e-3 = GPU vs
CPU 5.5e-3). The rounding itself costs KL about 7e-3 against the unrounded model (CPU 6.95e-3, GPU 6.82e-3): a cost of
8-bit activations, apart from the weights. The weight planes do not touch it.

**The author's rule:** the GPU answers first (rounded, fast, in step t); the CPU answers late with the **more exact** value
(only the flagged coin-flip values, recomputed in fp32, unrounded). When they disagree, side with the CPU. A late CPU answer
still counts: it makes t+1 careful ("careful+1": planes read, flagged values kept unrounded, head re-score), as the late CPU
hit did in the dual act. Spotting the coin flips is nearly free: the rounding kernel already has |frac - 0.5|; ballot + popc
counts them per token.

**First measurement (MEASURED 2026-10-31 v17b, tiered file, WikiText 100 x 512):** rounding cost +0.001009 KL (t 8.27; 10.7 %
of the file's 0.009404). The COUNT of coin flips per token predicts nothing (rho about 0, AUROC 0.51-0.55 at t and t+1). The
damage itself persists: worst-1 % rounding damage at t -> t+1 7.09 % vs 1 % (7x), worst-1 % KL 9.06 % (9x). So careful+1 holds,
but its trigger must come from where the flips land (their effect), not how many there are. Next forms (UNMEASURED): per-value
count; flips weighted by the following weight row's magnitude; the late exact CPU pass priced as the 0.001 it can win back.
Author 02:36: the counter is unweighted (a value near halfway in a block of scale 1e-5 counts as one in a block of scale 10),
while the damage is the rounding error itself, (r - round(r)) x scale, on every value; the next form measures that error
energy per token (and per projection), not the count near 0.5. Router: NOT rounded (MEASURED 2026-10-31 02:4x, pelias inspect:
ffn_gate_inp F32 [2048,256], ssm_alpha / ssm_beta F32 [2048,32]; act_format never rounds float weights, CPU and GPU alike), so
the +0.001009 comes from the quantized-weight inputs (attention, DeltaNet projections, shared and routed experts). Which of them
carries it: kld with --act-int8 per group (UNMEASURED; a CPU run, parked with the sweep).
Author 02:40-02:41: the tie goes to the CPU, literally: a value within B of halfway keeps its exact fp32 value (ties carry
the largest rounding error, about half a step; DERIVED, with an even spread the values within 0.1 of halfway are 20 % of values
and 49 % of the error energy). Test: reference option --act-tie-band B, kld at B = 0.1 and 0.05 (40 chunks each), paired against
the v17b dumps; prediction about 0.0089 at B = 0.1 if KL follows error energy. Engine form: GPU int8 pass + CPU exact
correction W (x - round(x)) on the tied values (careful+1). The +0.001009 and the 9x t+1 persistence sit in the quantized
projections writing the recurrent state and residual stream (DeltaNet main projections, attention Q/K/V/O, experts), which
then move the unrounded router input on the next layer and token. Order (author): after the CPU speed items, with the
scale-weighted error energy (r - round(r)) x scale per projection; PC idle until then.
**Author 02:49 (DECISIONS_LOG):** (a) kld runs parked until after the GP3-14 build; NO further long CPU kld (80-100 chunks)
until Fable's CPU fixes are wired in (persistent thread pool, AVX2 dequant, the 3.2 GB of zeroed buffers dropped, shared
inputs rounded once): no reason to hold the CPU at 90 C for 20+ minutes when those fixes cut the run to a fraction.
(b) Engine cost of the scalar tie band: 20 % of single values scattered over x hit about every 32-wide block of W (about 6.4
of 32 per block), so W (x - round(x)) on the CPU decodes and multiplies nearly whole matrices: it would sink the speed.
(c) The 20 % / 49 % table holds within one scale only; block scales vary 10x-1000x across a row, so a half-step tie in a
quiet block (scale 0.001) carries 1000x less error than one in an outlier block (scale 1.0).
(d) Rank 32-wide activation sub-blocks by their real scaled error energy, sum((x - dequant(x))^2) per 32-block, and correct the
top 10-20 %: more of the true error energy, and the correction stays on contiguous 32-wide blocks (W columns 32 at a time)
that the GPU or AVX2 handle cleanly. Test order: CPU fixes first, then scalar tie band B = 0.1 side by side with the top
block-energy correction (10 % and 20 %), both paired against the v17b dumps, to see which wins back more of +0.001009.



**First measurement (UNMEASURED; offline, existing tools):** per token, the count of coin-flip roundings (|frac - 0.5| < eps)
against that token's KL and the next token's KL on WikiText; and the activation-rounding cost itself on WikiText (kld with
and without act rounding). It says whether a token's coin flips predict a miss at t+1 (the author assumes it follows the
11 % vs 1 % pattern measured for misses) and whether the late exact CPU pass is worth wiring.

## PRIORITY (author 2026-10-29 18:44): near-tie accuracy first; every other step waits until it is solved

**Problem.** The GP3-07 gate (v8, Pelias's own format, no wormhole) split from llama.cpp on 11 of 20 prompts: 8 at near-ties, 3 at end-of-text setup (MEASURED); 4 of the 11 splits are in the first 5 output tokens. The KL drift sits in few tokens: the top 1 % hold 23-52 % of it (MEASURED). Better or worse is decided by KL against the original model (scoreboard); agreement is the gate.

**What the offline checks settled (2026-10-29, MEASURED; D:\pelias-tests\gp\c3c6_pretoken):**
- A pre-token flag (C6 map keyed on the state codes of step t-1) works but weakly: 10 % flagged catches 30 % of stuck tokens (layer 30, 16-nearest vote).
- fp32 activations on stuck tokens: no gain (dropped from the careful path).
- Expert picks: the router is already F32; stuck tokens carry no extra flips vs bf16; pick sensitivity is spread over all tokens. Not the lever.
- **Weight error is the lever, mostly in the experts** (their weight error 20-75x the dense part's).

**Solution: residual sign planes on the 1-bit path** (author 18:26-18:52; `pelias planes`, quant/planes.h, io/planes_build.h). Per 256 values: K fp16 scales and K sign bits per value, fitted against the original weights (plane by plane, then refitted together); added to the stored row as it is read. Any model file (ours, Unsloth's, stock) given its original weights once. Each plane cuts the weight error about 3x (MEASURED, blk.20 attn_qkv / gate_exps / down_exps); two planes beat storing one whole type up; rotating the residual first is worse (dropped). In the engine: planes on every token, on the prompt only, or on the prompt and the first N tokens (`--planes-mode`), and the exact re-score of the top output candidates from the original head rows (`--head-rescore`). Later (GP3-14): the planes on the GPU 1-bit path and the CPU bus, read per token by the flag / dual predict as a graded dial (0, 1 or 2 planes; author 19:03: proofreading in proportion to the uncertainty).

**Standing procedure (author 20:50), every run until the engine (GP3-14) is built:** prediction written at the start; results listed at the end; logged (BUILD_LOG, RUNBOOK, game plan) and everything unpushed pushed; then the next run on the list, the same way. On an error: stop, alert the author, fix, rerun.

**Grading (author 20:01-20:46):** the splits are graded by the original model itself (BF16 through the memory map, as block K): `pelias teacher` gives the original's own top-1 at every position of each side's text, `tools/split_grade.py` says at each first split who matches the original (Pelias, llama.cpp or neither), and over all positions each side's share of the original's own picks and its NLL. Checked: the original on its own greedy output agrees 16 of 16 (tiny model, sandbox).

**Standing rule (author 20:38):** every split-token decode test keeps the model and planes loaded in RAM for all prompts. Before the next one: `pelias generate --prompt-list` (one load, every prompt, one ids file each). The 2026-10-29 runs reloaded per prompt (43 s each, MEASURED).

**Results 2026-10-29 (MEASURED; BUILD_LOG):** Unsloth MTP UD-Q4_K_M in Pelias's own format, 100 chunks: KLD 0.012808 without planes, **0.005818 with 2 planes (-54.6 %, t -12.25)**, 0.005741 with the exact head re-score of 64 (-1.3 % more, t -16.8). Splits graded by the BF16 original: v8 (no planes) Pelias 7 / llama.cpp 4; **with planes Pelias 12 / llama.cpp 3**; the original's own top-1 share of the held tokens: llama.cpp 98.0 %, Pelias 98.7 %, Pelias with planes 99.1 %. Open, folded into GP3-14: planes per token from the flag / dual predict (graded 0-2 planes), the fast plane add on the GPU 1-bit path and the CPU bus.

**Next (author 2026-10-30 05:08): GP3-08 paused; the cost-down runs first** (docs/PELIAS_RESIDUAL_PLANES.md section 4.4): traditional Q5_K_M / Q6_K through Pelias; 1- and 3-plane side files; block-selective (A), hot-experts-only (B) and A + B planes; the flagged-only reading.

**Analysis (author 2026-10-30 05:32): a file the size of Unsloth's with an accuracy win** (docs/PELIAS_RESIDUAL_PLANES.md section 6; DERIVED, UNMEASURED). A usage-tiered expert file: each expert of each layer gets its own type by its routing weight (hot Q5_K/Q6_K/Q8_0, cold IQ4_XS), everything else as in the Unsloth file; at the same 22.65 GB the projected KL is about 0.0079 (-38 % vs 0.012808), about 0.0066 at 24.1 GB. At a fixed size planes add almost nothing over per-expert types; they stay the add-on to existing files and the per-token reserve. Needs engine work (experts stored per type plus a map) and a routing check outside WikiText. Proposed runs, waiting for the author's go: routing capture on code/chat; the tiered quantize + kld; split-token decode + grade on it; tiered + planes on the top 32 experts; a Q5_K_M reference.

**Locked path (author 2026-10-30 06:36-06:41):** the 22.65 GB tiered base (Q4_TG2 floor, 50/50 usage mix, 40-layer slopes with the layers 32-39 down weighting) first; then 1 GB and 2 GB worst-error block plane side files on it, plus a dense plane and a 3rd output-head plane, graded on the split tokens. Layer-sensitivity runs skipped for now. Built: `pelias quantize --tier` (b3fd4b5). Runs: gp/block_tier_v1.sh build, tier, kld. **MEASURED 08:16: KL 0.009404 vs 0.012808 (-26.6 %, t -12.41, settled), same size (22,663,406,784 bytes), same top 95.80 vs 95.02 %.** R3 (08:44, MEASURED): +0.3-1.3 % GPU token time. Plane builder options built (d898233). GUI (author 09:09): the tiered file, planes (count, picks, mode) and the head re-score become user toggles in the web interface (masterwork 11.5), each showing its measured cost and gain. Three ways (author 09:17, masterwork 11.5): run a premade file as is; improve it with a plane side file (original weights needed once; shareable per exact file); or make a Pelias tiered quant from the original. Queued (author 09:23, do NOT start before the final draft build is finished): update the PDFs (how the speedup works + Section 21 calibrated gains -> rev4; 7-bit and 1-bit decoder plain language -> rev3; pathways -> v5) and the project markdown (masterwork 10 / 20 / 21, PELIAS_NEW_METHODS, PELIAS_VS_EXISTING_FRAMEWORKS, PELIAS_7_T_H_PLAN, README, PELIAS_SYSTEM_MASTER) with the planes, tiered and per-pick work; the physics briefing only when the author asks. **Done 2026-10-30 14:4x-15:1x (author 14:38, 14:56):** PDFs speedup + Section 21 rev4, 7-bit rev3, pathways v5 (backup folder), docs/Pelias_project_overview.pdf; markdown: masterwork (header, 10.8, 20, 21.6, 21.8.1), PELIAS_NEW_METHODS (methods 19-26), PELIAS_VS_EXISTING_FRAMEWORKS (scoreboard 2), PELIAS_7_T_H_PLAN (section 10), README, PELIAS_SYSTEM_MASTER (status notes). Next: block_tierplanes_v1.sh chain (planes A1 1 GB, A2 2 GB; kld A1 every pick, A2 every pick, A2 top 4), then the best + dense plane + 3rd head plane, then the split-token grade. **Chain done (10:36, MEASURED, doc 6.12): A2 top 4 is the winner (0.008800).** **Build first (author 10:55-10:57):** nothing else is tested before the first engine build; the GPU goes online and accuracy testing resumes on the engine, where a kld takes minutes instead of 25. Parked until then: v2 (A2 top 4 + dense plane + 3 head planes, gp/block_tierplanes_v2.sh ready) and the split-token grade on the winner. Order now: GP3-10 CPU kernels (AVX2 dot products for Q4_TG2 / Q5_K / Q6_K / Q8_0 / Q4_K / IQ4_XS with int8 activations, vectorized plane add with per-pick, thread pinning) -> GP3-10 GPU kernels (the measured R3 probe kernels moved into src) -> GP3-12 inference rungs only (verify harness, one layer, full forward on the GPU) -> GP3-14 engine without MTP drafting -> first test build on WSL and Windows -> parked tests. Moved after the first build: GP3-08 MTP, GP3-12's backward / LoRA / training, the small-card layout. **Changed (author 11:13-11:24):** GP3-08 MTP is back before the first build (built into the engine); every readable type gets a fast CPU decoder (all 30, mass appeal); fused gate/up and fused q/k/v tensors are read; and before the first build also: GP3-09 dense models (Llama / Qwen3 / Mistral-style families with their tokenizers and chat templates), GP3-11 FP8 and NVFP4 files, and BitNet (b1.58 ternary models: TQ1_0 / TQ2_0 storage and the BitNet architecture). All five of the author's mass-appeal items are on this pre-build list (11:24): (1) fused tensors, (2) dense model families, (3) FP8 / NVFP4, (4) the Windows (MSVC) build of the new kernels proven on the PC, (5) the int8-activation CPU path (the engine's per-32 int8 format, integer dot products) for speed. **Pre-build list status (2026-10-30 13:49, MEASURED):** done: GP3-08 MTP (real file 9.1e-6), fused tensors, all-31-type AVX2 decoders, int8-activation CPU path, batch kernels; GP3-09 dense families qwen3 / qwen2 / llama (tiny models vs llama.cpp <= 7.5e-7; Qwen3-1.7B Q8_0 KLD 0.003617 with llama.cpp's rounding vs llama.cpp 0.00367, 0.002263 with exact inputs; chat answer sensible); BitNet b1.58 2B4T I2_S (top-1 14/14 vs transformers, KL 0.0016 at the int8-rounding floor; chat answer sensible). GP3-11 (14:12): FP8 files (through llama.cpp's converter) 0.00862 vs llama.cpp 0.00863; NVFP4 GGUF (type 40 + per-weight .scale) 0.1334 vs 0.1345 (open: a Pelias-native FP8 / compressed-tensors safetensors reader). Windows MSVC (14:28): Visual Studio 2026 /W4 /WX, 0 warnings, 84 tests pass on Windows, Windows vs WSL logits 2.6e-6 / 6.9e-7. All five mass-appeal items done. Next: the GPU kernels into src (GP3-10 GPU), GP3-12 inference rungs, GP3-14 first engine test build. Open for the engine: CPU decode 4.0-4.2 tok/s on 1.7-2.4 B models (thread start-up per product suspected); dense projections on the CPU batch path. **After the first build (author 2026-10-30 13:00):** the parked accuracy checks come first (v2 dense + head planes, the split-token grade, the rest of the accuracy runs), before any of the new ideas (small-card layout, GP3-14c infinite context, D-1B dense 1-bit and the other post-build items).

**Small-card layout (author 2026-10-30 09:40-09:56; added to the run list at 09:56; moved to after GP3-14, the first engine test build, at 10:53):** a locked int4 expert core (planes at scales s, s/2, s/4, s/8, one int4 per value) plus free 1-bit planes per aligned 256-block. The top bit of every expert and the non-expert parts sit in VRAM (about 7 GB, DERIVED); the lower core bits and free planes sit in a RAM side file, added on the CPU in parallel with the GPU (y = Bx + sum of P x). Spare VRAM holds the full bits of the hottest experts. Sandbox weight error (MEASURED, plane_stack_check): core 0.01051 (4.125 bpw), + 1 free plane 0.00281 (5.19), + 2 free 0.00071 (6.25); Q4_K 0.00468, Q4_TG2 0.00418. Sizing with 32 GB RAM (about 24 GB usable, ESTIMATE): 8 GB card, core + 2 free (about 25.2 GB of experts, about 20.9 GB in RAM), between Q5_K and Q6_K; 12 GB card, core + 3 free (about 29.4 GB, about 21.1 GB in RAM), about 0.00018 (DERIVED from the trend, UNMEASURED), between Q6_K and Q8_0. Speed on a 5800X3D with dual-channel DDR4-3200 (DERIVED; 51.2 GB/s peak, about 40 GB/s ESTIMATE): 8 GB card 61-78 tok/s, 12 GB card 130-167 tok/s; R3 today 5.7 ms, about 175 tok/s (MEASURED). Against CPU expert offload of a Q4_K_M at about the same bytes read, the accuracy is about 0.15x Q4_K's error (DERIVED from the sandbox). Becomes a GUI toggle (masterwork 11.5). Offered by the machine probe (author 09:57, corrected 10:09: at least 8 GB, not only 8 GB): when it finds a GPU of 8 GB or more running a supported MoE model, it reads the VRAM, RAM and CPU and suggests the plane count that fits, up to the ceiling those allow. The user may pick more planes while RAM allows (author 10:12; 64, 128, 192 GB machines and up), capped at core + 5 free planes (about 9.4 bpw, about 1.1e-5 weight error DERIVED, past Q8_0); planes that don't fit RAM are greyed out, and each shows its measured KL gain and speed cost. The cap is checked by a PC kld of + 4 vs + 5: if + 5 isn't better beyond the error bar, the cap drops to + 4. **Planes on any file (author 10:13-10:18):** every base file and type (1-bit, tiered, Unsloth, any n-bit quant) may take more planes if the user wants, limited by RAM (plane size shown). Once the chosen count reaches 4, the 4 planes are stored as one int4 residual tensor (locked scales, one fp16 scale per 256; MEASURED on raw weights: 0.01051 at 4.125 bpw vs 4 free planes 0.01299 at 4.25), with free 1-bit planes on top; an int8 residual tensor is the option at 8. The ladder (author 10:20), bits added on top of the base (DERIVED): 1, 2, 3 free 1-bit planes (1.06 / 2.13 / 3.19); then optional int4 residual (4.06), + 1 or 2 free planes (5.13 / 6.19); then optional int8 residual (8.06). Same RAM (slightly less), about the same CPU time (memory-bound), lighter compute; the residual stays readable bit by bit from the top, so the plane slider still works. Residual estimates on the 1-bit base (DERIVED, UNMEASURED): + int4 about 0.0038 at 5.19 bpw, + int8 about 1e-5 at 9.19 bpw. **Literature to keep at hand when the 1-bit base misbehaves (read 2026-10-30 17:1x at the author's request; angles to try, not instructions; little is published for a 1-bit MoE base with RAM planes, so measured behaviour here leads):** (a) the base plane cannot be fixed later by the planes above it (AnyBCQ, ICLR 2026: codes frozen during progressive expansion propagate early errors), so fit the base with care; (b) greedy plane stacking drifts from the full-precision distribution; alternating refits of codes and scales fix it (ARB-LLM, ICLR 2025; BPDQ, ICML 2026: bit-planes + coefficient refits on a variable grid). Our MEASURED locked int4 core (0.01051) already beats 4 greedy free planes (0.01299): an alternating joint refit is the next angle; (c) protect salient weights in the base (BiLLM 2024, PB-LLM 2023); (d) routing shift: at 1.5 bits over 40 % of tokens go to other experts (GEMQ, ICML 2026, Qwen3-30B-A3B), fixed by cross-layer bit allocation and router fine-tuning: watch the flip rate per layer and the pre-router hidden-state cosine at base + 1..5; (e) planes per pick by live gate score: DPA-MoE (EMNLP Findings 2026, Qwen3-30B-A3B) minimises sum g_e^2 x residual_e(k) under a chunk budget (knapsack); gate scores carry most of the gain, as our MEASURED top-4 picks (0.008800, 92 % of the gain) did; DPA never goes below 2 chunks and keeps every chunk in VRAM: no 1-bit base, no CPU planes; (f) closest prior art to RAM planes: QAQ (NeurIPS 2025 workshop) loads router-chosen bit-planes CPU -> GPU; ours compute on the CPU in parallel instead; (g) Gated DeltaNet: a whole-block 4-bit study (arXiv 2609.04098) finds state error flat over 32K tokens and the delta rule erasing noise within hundreds of steps; the decay alpha is the sensitive gate, beta is not; 1-bit is coarser than 4-bit, so check long-context KL; the recurrent-state papers (arXiv 2610.15709, LeapQuant) concern storing the state in low bits, which Pelias does not do (FP32 state) unless GP3-14c compresses it. Tests these add: alternating base refit vs greedy; gate^2-weighted plane budget vs top-4; cross-layer plane allocation; salient-weight protection in the base; long-context KL.
Runs, in order: (1) sandbox: top bit of the locked core as the VRAM part, + 3 free planes, finer per-32 scales, joint fit, imatrix-weighted fit; (2) PC: build the split file and kld vs the tiered file and Unsloth; (3) PC: a memory-bandwidth read on the 5800X3D, then a timed token run once the CPU plane kernels exist (GP3-10).

**Run order:** (1) build the side file from the original BF16 model against the Unsloth MTP file (and urm_ZT2); (2) `pelias kld` with 0 / 1 / 2 planes and with the head re-score; (3) the 20-prompt decode again with planes; (4) score against the original model's own greedy output.

**Parked until after Pelias-master 1.0 (author):** the wormhole DAX tail test (don't serve ~22 GB models through a DAX daemon; plain virtiofs loaded the model in 25 s).

---

## Status on 2026-10-27 (15:0x): what is settled, what is open

### Where we are

- **Build:**
  - Steps 1–3 are done: measuring the machine, foundations, quantizers and conversion.
  - Step 4 is half done: the tokenizer is finished; the chat template and data pipeline remain.
  - Milestone 1 (the CPU reference that scores drift) is done.
  - Tasks 1 and 2 are done: the importance matrix and the weighted quantizer.
  - Tasks 3–7 haven't started.
- **Latest commit:** cc22071, with the Fable review folded in. Unit tests pass on gcc, clang and asan.
- **Goal (O20):** a 4-bit PEL mix with less drift than Unsloth. Best so far is T8: 0.014834 against Unsloth's 0.013605, measured by llama.cpp. That is close, but not proven better or worse.

### Settled: no need to test again (all MEASURED)

**Correctness: Pelias matches the reference**

| What | Result |
|---|---|
| Our quantizer vs llama.cpp's (P-E6) | 753 of 753 tensors identical; whole file byte-identical |
| Converting the original weights (P-X6) | 744 of 753 identical. The other 9 differ in 13 values by 1 rounding step, and ours are the correctly rounded ones (accepted by the author) |
| Tokenizer vs llama.cpp | 0 mismatches over 2,455,727 tokens |
| Tokenizer vs Hugging Face | 8 of 8 files identical, 2,764,292 tokens; Unicode probe 0 differences |
| Reading every GGUF type, 1 to 8 bit | 0 values differ from llama.cpp's own decoder |
| Weighted (importance-matrix) quantizer | Byte-identical to llama.cpp, 0 differences |
| Our importance matrix vs llama.cpp's | Same effect on drift (5.4326 vs 5.4352 on the tiny model) |
| CPU reference on the real model | Stock file: 0.039013 (llama.cpp reads 0.041124 on the same file) |
| Why those two differ | llama.cpp rounds activations to 8 bits. With the corrected option (cc22071), the tiny model closes 96.3 % of the gap on the Q6_K file and 93.5 % on the Q8_0 file. The rest is most likely llama.cpp's 16-bit attention cache |
| Determinism | llama.cpp's drift test repeats to every digit |

**Drift results (llama.cpp's scale, WikiText-2, 100 chunks)**

| File | Drift | Size without the MTP block |
|---|---|---|
| Stock Q4_K_M | 0.040965 | 21.17 GB |
| Stock + Unsloth's matrix (a) | 0.027317 | — |
| PEL-6 (`ssm_out` 8-bit) | 0.031616 | 21.29 GB |
| T6 (`ssm_out` 6-bit) | 0.032472 | 21.23 GB |
| T1 / PEL-6+ | 0.025958 | 21.34 GB |
| T2 / PEL-4 | 0.016450 | 21.64 GB |
| T3 / PEL-5 | 0.016493 | 21.80 GB |
| T4 | 0.015176 | — |
| T7 | 0.015454 | 21.72 GB |
| **T8** | **0.014834** | **21.97 GB** |
| (b) Unsloth's layout, no matrix | 0.014174 | 22.13 GB |
| **Unsloth UD-Q4_K_M** | **0.013605** | **22.13 GB** |
| T5 / PEL3-a (3-bit) | 0.045287 | 17.48 GB |

**What those results proved**
- Which bits go where does almost all the work. The importance matrix adds only about 0.0006 on top of a good layout.
- `ssm_out` is the most drift-sensitive group (22.8 % of stock's drift on its own).
- Gains add up across groups, and slightly more than the single runs predicted.
- 6-bit is nearly as good as 8-bit: `attn_qkv`/`attn_q` gain nothing from 8 over 6, and `ssm_out` at 6-bit keeps 91 % of its 8-bit gain.
- Raising whole expert groups to 8-bit costs 4–5.5 GB each, which is poor value per byte.
- Unsloth's expert layout is worth 0.0013 for only 41 MB.
- The 0.53 GB size difference to Unsloth is exactly the MTP block, to the byte.
- llama-quantize quietly changes other tensors when `attn_v` or `attn_qkv` is overridden (they share a counter). The recipe tool makes that impossible.

**Hardware (Wave 0, Batch C)**
- **Card:** 46 SMs; free VRAM 10.83 GiB, giving the 10.7 GiB ceiling; 26.3 GB/s pinned transfers.
- **Host:** 32 GiB of pinned memory held.
- **Kernels:** our bf16 kernel reaches ≥ 0.85× cuBLASLt on 17 of 20 shapes. The int8 path is 1.39–1.84× faster (its accuracy half is still open). Dequantization is memory-bound. fp8 e4m3 passes; e5m2 fails.
- **Handoff:** the CPU–GPU doorbell answers in 2.4 µs.
- **CPU-only speed:** stock llama.cpp runs at 16.28 tok/s.
- **Drift base:** the bf16 base file used for all drift tests took 57 min to make and doesn't need remaking.

### Retired (don't use again)

- **The "±0.0006 is noise" rule:** never proven. Only paired per-token comparisons decide (0.5).
- **The 0.017353 float spread as a pass mark:** it's larger than T8's whole drift, so it can't tell mixes apart (GP2-01).
- **The 57.0 tok/s Windows figure.**
- **The patched llama.cpp tier build as a product:** it's now only a record.
- **The 98 % figure for the first activation-rounding option (3a272b3):** it used one format for every weight type and skipped the output layer.

### Open (Part 1 answers these)

| # | Question | Test |
|---|---|---|
| 1 | Do the llama.cpp rankings hold under Pelias's exact scoring? (T8 vs Unsloth not yet paired token by token) | GP1-04 |
| 2 | Were the old llama.cpp runs secretly using the GPU's rounding path? | GP1-02, GP1-03 |
| 3 | Does T8 + an importance matrix, or T9 with IQ4_XS experts, beat Unsloth? | GP1-05 |
| 4 | Do the results hold at long context (4096) and on a second kind of text? | GP1-09, GP1-10 |
| 5 | How much error does the bf16 base carry itself? Do routing flips cause the worst tokens? | GP1-07, GP1-06 |
| 6 | Does re-quantizing an unchanged weight give the same bytes back? | GP1-08 |

Until question 1 is settled, all the llama.cpp drift numbers count as rankings, not exact values.

---

## 0. Conventions every script in this plan follows

### 0.1 Variables (values in `profiles/local_only/GAME_PLAN_PATHS.env`)

| Variable | What it points to | Checked by preflight |
|---|---|---|
| `PELIAS` | The `pelias` binary in the clone's `build/m1` folder (the Release, Ninja, AVX2, CPU-only build that produced Milestone 1; MEASURED 10-27), rebuilt at the commit named in the step | `$PELIAS version` runs; the build's `CMakeCache.txt` has `PELIAS_CPU_AVX2:BOOL=ON`; `git -C $PELIAS_REPO rev-parse --short HEAD` equals the expected commit |
| `PELIAS_REPO` | The author's WSL clone of pelias-master (its real name is in the paths file; MEASURED 10-27). Its `origin` is a backup bundle, not GitHub, so `git pull` cannot reach new commits: updates come from the newest bundle with `git fetch <bundle> main` then `git merge --ff-only FETCH_HEAD` | `.git` exists; `git status --porcelain` is empty |
| `LLAMA_BIN` | The unchanged llama.cpp build at commit 2ca15f540 (test oracle only); the one that wrote every sweep log | `$LLAMA_BIN/llama-perplexity --version` shows the short hash `2ca15f5` |
| `T` | The test folder on D: (`kld-base-bf16-100chunks.bin`, `qwen36-bf16.llamacpp.gguf`, `qwen36-Q4_K_M.pelias.gguf`, `sweep/`) | each file exists with the byte size in 0.3 |
| `M` | The model folder on C: where new candidate files are written | free space |
| `U` | The Unsloth folder (`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, `imatrix_unsloth.gguf_file`) | exists, sizes |
| `R` | The results folder for this plan (one subfolder per test ID) | writable |
| `WIKI` | `wiki.test.raw` (WikiText-2 test), needed only to make new bases | exists; the 512 base already holds its own tokens |

### 0.2 Script rules (all scripts, no exceptions)

1. **Header.** `bash` with `set -euo pipefail`. The first lines print the test ID, the start time, the commit, the prediction (written before the run) and the time estimate.
2. **Logs.** Each run writes its own log, `$R/<ID>/<name>_<date>_<time>.log` (a new name every run, so a retry never overwrites), and the same lines go into one fixed file, `$R/live.log`, which the author can follow live from any terminal (`tail -f` in WSL, or `Get-Content -Wait` in PowerShell). Only a few summary lines (prediction, result, verdict, log path) go to the console. Never `| tail` and never `> /dev/null` on a long run's log, so progress and the ETA stay visible in the log.
3. **Never overwrite.** Each output path is checked first and the script stops if it exists. `pelias kld --dump-kld` and `pelias imatrix` also refuse by themselves.
4. **Free space.** Before writing a large file (a GGUF or a base), print `df -h` for C: and D:. Stop if either drive would drop below 50 GB after the write (author's rule). Expected size = the DERIVED size in the step.
5. **Nothing else heavy runs.** No llama-server, no other `pelias kld`, no training. CPU runs are timed, and a second job slows both.
6. **Type check.** After every quantize, `$PELIAS inspect --model OUT --json` is compared against the recipe tensor by tensor (0.4). Any difference stops the script before the drift test. This is the lesson of the shared `attn_v`/`attn_qkv` counter (PEL_QUANT_DECISION_DATA 3.3).
7. **Size check.** The output's byte size must equal the expected byte size where one is MEASURED or DERIVED (to ±32 bytes for DERIVED sizes, exact for MEASURED rebuilds). Expected size = `recipe.py data-bytes RECIPE SOURCE` + 10,991,136. That offset is MEASURED identical for PEL-6, T5 and T8 (data start plus trailing padding, same metadata). It stays constant only while the metadata and the last tensor's padded size are unchanged; the type check, not the size, is the proof.
8. **Summary line.** One line is appended to `$R/summary.tsv` with these columns: date-time, test ID, file (and route), the tool's `Mean    KLD` line, its `Same top p:` line, wall seconds, log path. The file's byte size goes in the quantize step's own log.
9. **Results first.** Claude reads the summary lines (and the log where needed), labels each number MEASURED and checks the prediction before anything else is decided.
10. **How a block runs (from 10-27).** The blocks are script files in `$R` (`common.sh`, `block_a.sh`, `block_b.sh`, later blocks alike; a copy of each sits in the git-ignored `profiles/local_only/gp/` and in the backup folder). Claude runs them on the author's machine through the host command tool as `wsl bash $R/<block>.sh [step]`, one call per run, and only after the author's OK. Before asking, Claude states what the run is for, its log path, and its ETA. No single call may exceed the tool's 7,200 s window, so Block B and later blocks take a step argument and run one test per call. Every script was dry-run end to end in the sandbox on the tiny model (paths overridden, `GP_TEST=1` skipping only the real-file size checks) before it was copied over; its SHA-256 on the author's drive is checked against the tested copy. **While a run is active, no other host command is started**: a second command stops the first (MEASURED 10-27: step c killed at chunk 35). Progress is read from `live.log` with the read-only file tool; scheduled check-ins at a third and two thirds of the ETA and at the ETA.
11. **No deletions.** No script deletes anything. If a file must go, Claude prints the command for the author to review and run by hand.

### 0.3 Fixed inputs (MEASURED sizes; a script stops if a size differs)

| File | Bytes | What it is |
|---|---|---|
| `$T/kld-base-bf16-100chunks.bin` | 12,664,728,820 | llama-perplexity KL base: bf16 model, WikiText-2 test, 100 chunks × 512, made with the GPU visible and `-ngl 0 -t 8` (56 min 54 s) |
| `$T/qwen36-bf16.llamacpp.gguf` | 71,066,995,072 | llama.cpp's bf16 conversion, the source of every candidate |
| `$T/qwen36-Q4_K_M.pelias.gguf` | 21,713,463,328 (on disk) | Pelias stock Q4_K_M. llama.cpp read 0.041124 and Pelias read 0.039013 |
| `$U/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 22,134,528,992 | Unsloth: the bar, 0.013605 with llama.cpp |
| `$U/imatrix_unsloth.gguf_file` | 192,223,904 | Unsloth's importance matrix |
| `$M/qwen36-Q4_K_M-typesU.gguf` | 22,660,245,536 | Experiment (b): Unsloth's types, no imatrix (0.014174) |
| `$M/qwen36-Q4_K_M-imatU.gguf` | 21,713,463,616 | Experiment (a): stock types plus Unsloth's imatrix (0.027317) |
| `$T/sweep/q-*.log` | — | llama-quantize logs of every sweep and T run. They hold the exact type of every tensor (0.4) |

Note: the stock file's size on disk (21,713,463,328) is 32 bytes more than the figure in older notes (21,713,463,296). Scripts check the on-disk value, and Claude confirms it from `pelias inspect` before the first run.

### 0.4 Recipes: one exact type per tensor

A recipe is a TSV file with 753 rows: `tensor_name<TAB>type`, where type is a ggml type name in upper case (`Q4_K`, `Q8_0`, `F32`, ...). Recipes live in `$R/recipes/` until Part 2 moves the final ones into the repository.

- **From a file that exists:** `$PELIAS inspect --model FILE --json`, then read `tensors[].name` and `tensors[].type`.
- **From a llama-quantize log:** read every line of the form
  ```
  [  N/ 753] NAME - [ d0, d1, d2, d3], type =   SRC, converting to DST .. size = ...
  [  N/ 753] NAME - [ d0, d1, d2, d3], type =   f32, size = ...
  ```
  A tensor's record starts at `^\[\s*(\d+)/\s*(\d+)\]\s+(\S+)\s+-\s+\[[^\]]*\],\s+type =\s+(\S+),` and runs to the next record. Imatrix notices can land inside a record when stdout and stderr share the log, so the result is searched for in the whole span. The type is the `converting to DST ..` value if present, else SRC (the record then says `size =`). Upper-case it. The tool requires as many records as the log announces, numbered in order (753 for this model's files; Unsloth's file has 733, so a recipe from it cannot be applied to the 753-tensor bf16 source: `llama-file` refuses the count mismatch).
- **Feeding a recipe to llama-quantize** (until `pelias quantize --recipe` exists, Part 3):
  - `python3 tools/recipe.py llama-file RECIPE.tsv $T/qwen36-bf16.llamacpp.gguf TYPES.txt --pelias $PELIAS` writes a `--tensor-type-file`.
  - One line `^<escaped name>$=<type>` for every tensor, including those whose type equals the source's. Without a line, llama-quantize would apply its mix rule to a tensor. A tensor llama-quantize never quantizes (norms, the routers `ffn_gate_inp`, `ssm_conv1d`; READ-SRC `tensor_allows_quantization`) keeps its type whatever its line says.
  - llama-quantize reads the file as whitespace-separated `pattern=type` tokens. It lower-cases the pattern, matches with `std::regex_search`, uses the first matching pattern, and reads type names case-insensitively (READ-SRC: tools/quantize/quantize.cpp `parse_tensor_type_file`, src/llama-quant.cpp `llama_tensor_get_type`, at 2ca15f5).
  - A tensor that matches a pattern skips llama-quantize's own mix rules and its per-layer counters. So with every quantized tensor named, nothing else can move.
  - Command: `$LLAMA_BIN/llama-quantize --tensor-type-file TYPES.txt $T/qwen36-bf16.llamacpp.gguf OUT.gguf Q4_K_M 16`.
  - The type check (0.2 rule 6, `tools/recipe.py check`) is what proves it on every file. The tool's own test on the tiny model is in its unit run.
- **With an importance matrix:** add `--imatrix FILE`. Type-check exactly the same way.

### 0.5 How two results are compared (applies to every drift comparison from now on)

- Every Pelias drift run writes `--dump-kld $R/<ID>/<name>.kld` (25,500 records per 100-chunk run, 612,036 bytes; DERIVED from the format).
- A comparison is always `pelias kld-compare --a A.kld --b B.kld --name-a … --name-b …`. It is paired token by token. Its standard error is clustered by chunk, and its interval uses Student's t with chunks − 1 degrees of freedom (99 for 100 chunks: t = 1.984).
- **Verdict:**
  - Interval entirely below 0: A has less drift than B.
  - Interval entirely above 0: B has less drift.
  - Interval contains 0: a tie on this text.
- Unpaired "±" values from a single run are never used to rank files again. The old "±0.0006" rule is retired.
- Scripts parse the line `difference (A - B) <d>   SE per token <s1>   SE per chunk (used) <s2>   95% interval [<lo>, <hi>] (t <t>, <df> df)   t-stat <z>` and the line starting `VERDICT:`.
- kld-compare refuses two files scored on different tokens, chunk counts or contexts with **exit 1**; a missing or damaged file is exit 2. A script treats any non-zero exit as a stop, never as a tie.

### 0.6 Predictions

Each test below carries a prediction written now (ESTIMATE). Claude checks it against the result and logs right or wrong in the RUNBOOK and TALLY.

Where a llama.cpp number exists, the Pelias-scale prediction is the range between two models of llama.cpp's activation-rounding effect, both taken from the stock file (MEASURED: 0.041124 llama.cpp against 0.039013 Pelias):
- **multiplicative:** Pelias = llama × 0.94867;
- **additive:** Pelias = llama − 0.002111.

Which model holds is itself a finding (Part 1, GP1-04).

---

## Part 1. Tests before any further building

Ordered so each block feeds the next. Blocks A to C need no new code; they run on the commit that carries the Fable review fixes (named in the RUNBOOK and the handoff) or later.

The recipe tool `tools/recipe.py` (0.4) is written before Block C, which needs it. Blocks D to F need further **measurement tools only**: scripts or flags that produce numbers and add no product feature. Each tool is specified below so it can be written without guessing. Claude writes them, and they pass the unit/clang/asan/oracle routine before hand-over.

### Block A: preflight and determinism (A; about 25 min)

#### GP1-00 Preflight (A, 2 min)

**Checks:**
- Every variable in 0.1 resolves.
- Every size in 0.3 matches.
- `$PELIAS` was built at the commit this plan names for the block (Block A: the fixes after the Fable review) with AVX2.
- `git status --porcelain` is empty in `$PELIAS_REPO` before the update from the bundle (local edits would block or mix into the build).
- `df -h` shows C: and D: free space.
- `nproc` ≥ 16.
- Nothing named `llama-server` or `pelias` is running: `pgrep` inside WSL and `tasklist.exe` for the Windows side (the interim Windows server is invisible to `pgrep`).

**Rebuild if needed:**
```
cd $PELIAS_REPO && git fetch <newest backup bundle> main && git merge --ff-only FETCH_HEAD
# the clone's origin is a bundle, so git pull cannot reach new commits (0.1)
# build/m1 must already have PELIAS_CPU_AVX2:BOOL=ON and PELIAS_CUDA:BOOL=OFF (the script stops otherwise)
cmake -S . -B build/m1
cmake --build build/m1 -j16
```
- Configure again every time. The version string's commit is captured at configure time and goes stale otherwise (MEASURED in the sandbox: a build at 3a272b3 printed 686926d). Fix item GP2-14.

**Pass:** all checks print OK.

#### GP1-01 Stock Q4_K_M through Pelias again, with the per-token file (A, 22 min)

**Question:** is Pelias's drift number deterministic on the real model, and does the new per-token file match the printed mean?

**Command:**
```
$PELIAS kld --model $T/qwen36-Q4_K_M.pelias.gguf --base $T/kld-base-bf16-100chunks.bin --threads 16 --dump-kld $R/GP1-01/stock.kld
```

**Time:** 1309.3 s for this exact command without the dump (MEASURED, Milestone 1). The dump adds under 1 s (DERIVED: 612 KB).

**Prediction:**
- Mean KLD 0.039013, PPL 6.6898, Same top p 91.573 %, identical in every printed digit.
- The mean of the 25,500 `kld` values in the file equals the printed mean to 6 digits.

**Pass / decision:**
- Identical digits: the reference is deterministic at 16 threads.
- Any difference: stop. Find the cause (thread count, build flags) before any other Pelias number is trusted.

Results file: `stock.kld` is the stock anchor for every later comparison.

### Block B: closing the llama.cpp measurement questions (A; about 2 h 10 min)

Run one test per call (0.2 rule 10): `block_b.sh nooffload`, then `a`, `b`, `c`, `d`. Run `d` compares c with d when c is done. Each call refuses a run that already has its result.

#### GP1-02 llama.cpp with the GPU path switched off (old name F3) (A, ESTIMATE 15–45 min)

**Question:** did the earlier llama.cpp drift runs (GPU visible, `-ngl 0`, batch 2048) multiply on the GPU with q8_1 activations? Fable's audit says they may have. The private master records that turning op-offload off changed prompt speed (MEASURED, its line 5665).

**Command** (same as every earlier llama.cpp drift run, plus the one switch):
```
$LLAMA_BIN/llama-perplexity -m $T/qwen36-Q4_K_M.pelias.gguf --kl-divergence --kl-divergence-base $T/kld-base-bf16-100chunks.bin -c 512 --chunks 100 -ngl 0 -t 8 --no-op-offload 2>&1 | tee $R/GP1-02/stock_nooffload.log
```
- `-f` is not needed: in compare mode the tokens come from the base file (READ-SRC, perplexity.cpp `kl_divergence`).
- If `--no-op-offload` is not accepted by this build, run the same command with `CUDA_VISIBLE_DEVICES=""` in front and log which route was used.

**Time:** the GPU-visible run took about 4 min (MEASURED). CPU-only prompt processing is slower; its rate on this file is unmeasured, hence the 15–45 min ESTIMATE. The tool prints progress per chunk.

**Prediction:** 0.0405–0.0425. The direction is unknown: on the tiny model, CPU-style Q8_K rounding added more drift than GPU-style q8_1, 1.344 against 1.285 (MEASURED, sandbox).

**Decision:**
- If it differs from 0.041124 by more than 0.0005: every earlier llama.cpp number is labelled "GPU q8_1 path".
- From then on, llama.cpp is used only to screen candidates (fast); Pelias confirms them (exact).

#### GP1-03 Activation rounding in Pelias on the real model (old name F6) (A, 4 runs × 22 min = 88 min)

**Question:** how much drift does 8-bit activation rounding add, and where? This decides how the engine and the trainer's Mode F may round (GP2-05).

**Commands** (each writes its own dump):

| Run | Extra flags | Prediction |
|---|---|---|
| a | `--act-int8 all --act-int8-path cpu` (llama.cpp's CPU format for each weight type: Q8_K for the K-quants, Q8_0 for Q8_0 weights) | within ±0.0005 of GP1-02 |
| b | `--act-int8 all --act-int8-path gpu` (q8_1 for every type) | within ±0.0005 of 0.041124 |
| c | `--act-int8 _exps --act-int8-path gpu` (experts only) | between 0.0390 and run b |
| d | `--act-int8 attn,ssm_,shexp,output --act-int8-path gpu` (dense only: attention, every DeltaNet projection, shared experts, `lm_head`) | between 0.0390 and run b |

**Checked on the tiny model before hand-over (MEASURED, sandbox, after the Fable review fixes):**
- Off: output identical to before.
- Q6_K file, `all` cpu: 1.341948 against llama.cpp's CPU-only 1.346882 (fp32 1.214946): 96.3 % of the gap.
- Q8_0 file, `all` cpu: 0.352762 against llama.cpp's 0.352261 (fp32 0.359950): 93.5 %. The rest is most likely llama.cpp's f16 KV cache and summation order (the reference keeps the KV cache in fp32); not modelled.
- The output layer is rounded too (it was missed before the review).

**Base command:**
```
$PELIAS kld --model $T/qwen36-Q4_K_M.pelias.gguf --base $T/kld-base-bf16-100chunks.bin --threads 16 --dump-kld $R/GP1-03/<run>.kld <flags>
```
- `--act-int8-path` is new after the review (it replaced `--act-int8-block 256|32`, which used one format for every weight type, unlike llama.cpp's CPU path). A bare `--act-int8` with no value is refused.
- Group strings match tensor names by substring. `_exps` matches the three routed-expert tensors in every layer. `attn` matches `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_qkv` and `attn_gate`. `ssm_` matches `ssm_out`, `ssm_alpha` and `ssm_beta` (Q4_K in the stock file). `output` matches `output.weight` (the `lm_head`) and also `attn_output`, which `attn` already covers. Runs c and d together cover every quantized product.
- F32/F16/BF16 weights are never rounded: the routers, norms and `ssm_alpha`/`ssm_beta` when F32.

**Comparisons** (seconds each):
- `kld-compare` of each run against `GP1-01/stock.kld`.
- Run c against run d.

**Pass:**
- Runs a and b reproduce llama.cpp's numbers within ±0.0005. That closes the 5.1 % gap and proves the reference matches llama.cpp's arithmetic when asked to.
- If they do not, the remaining difference is llama.cpp's f16 KV cache or its order of operations. Log it, and decide whether a `--kv-f16` option is worth adding (measurement tool, Block D).

#### GP1-04 The bench set through Pelias, with the llama.cpp rankings (A, 3 quantizes + 3 Pelias runs, about 80 min)

**Question:** do the llama.cpp sweep rankings hold under the exact reference? Is llama.cpp's rounding effect additive or multiplicative?

**Rebuild the three files** from the bf16 with llama-quantize; they were overwritten after the sweep:

| File | Recipe from | Expected bytes (MEASURED) | llama.cpp KLD |
|---|---|---|---|
| `$M/pel6.gguf` (PEL-6: `ssm_out` Q8_0) | `$T/sweep/q-4.log`. Checked 2026-10-27 in the sandbox: against the stock recipe it changes exactly the 30 `ssm_out` tensors to Q8_0 (`recipe.py diff`). Note `q-0.log` is the `token_embd` run, not stock; the stock recipe comes from `recipe.py from-gguf` of the stock file | 21,839,292,448 | 0.031616 |
| `$M/t8.gguf` | `$T/sweep/q-t8_pel4_udexp_embd8.log` | 22,495,815,712 | 0.014834 |
| `$M/pel3a.gguf` (T5) | `$T/sweep/q-t5_pel3a.log` | 17,887,414,304 | 0.045287 |

**Steps per file:**
1. Recipe from the log (0.4).
2. ```
   $LLAMA_BIN/llama-quantize <--tensor-type list> $T/qwen36-bf16.llamacpp.gguf $M/<name>.gguf Q4_K_M 16 2>&1 | tee …
   ```
3. Type check against the recipe.
4. Size check (exact).
5. `$PELIAS kld … --dump-kld $R/GP1-04/<name>.kld`

**Time per file:** 253.5 s to quantize (MEASURED, T8) + 1309 s Pelias (MEASURED, stock) ≈ 26 min.

**Predictions** (Pelias scale, 0.6):

| File | Prediction |
|---|---|
| PEL-6 | 0.0295–0.0300 |
| T8 | 0.0127–0.0141 |
| PEL3-a | 0.0430–0.0432 (both models agree within 0.0002) |

Ranking: PEL3-a > stock > PEL-6 > T8, with every pair resolved by kld-compare.

**Decision:**
- The ranking holds: every llama.cpp screening result stands as a ranking.
- It breaks: the sweep conclusions are reopened before Part 2.
- Which prediction model fits: additive means llama.cpp adds a fixed noise floor (it matters most for the best files). Multiplicative means it scales.

**Disk:** +62.2 GB on C: (DERIVED from the three sizes). Check rule 4.

### Block C: the Unsloth question on the exact reference (A; about 92 min)

#### GP1-05 Unsloth, imatrix and IQ4_XS through Pelias (old name Block 2)

| Run | File | How it is made | Expected bytes | Prediction (Pelias) |
|---|---|---|---|---|
| a | Unsloth UD-Q4_K_M | exists | 22,134,528,992 | 0.0115–0.0130 |
| b | T8 + Unsloth imatrix | T8 recipe + `--imatrix $U/imatrix_unsloth.gguf_file` | 22,495,815,712 (types unchanged) | 0.0121–0.0136 |
| c | T9 = T8 + imatrix, gate/up experts IQ4_XS | T8 recipe with `ffn_gate_exps` and `ffn_up_exps` set to IQ4_XS in all 41 blocks, + imatrix | 21,807,949,856 (DERIVED: T8 minus 82 tensors × 8,388,608 bytes; IQ4_XS is 136 against Q4_K's 144 bytes per 256 weights, READ-SRC ggml-common.h). The script also computes it with `recipe.py data-bytes` + 10,991,136 | 0.0125–0.0145 |

**Comparisons** (the decisive ones of this plan):
- a vs b (does T8 + imatrix tie or beat Unsloth?)
- a vs c
- b vs c
- b vs GP1-04 T8 (the imatrix's own worth, paired)

**Time:** 2 × 4.2 min quantize + 3 × 22 min Pelias ≈ 75 min, plus the type checks. About 92 min with checks (ESTIMATE).

**Decision:**
- Any candidate whose interval against Unsloth is entirely below zero beats Unsloth on this text.
- A tie means the next levers (Blocks D to F, Part 3) are needed.
- Size is judged at the same time: without blk.40, T8 is 21.97 GB against Unsloth's 22.13 GB.

### Block D: measurement tools, then the tests that need them (C builds; A runs)

The tools come first. Each is small, has unit tests, and adds no product feature.

| Tool | What it does | Spec |
|---|---|---|
| D-T1 `tools/recipe.py` (written now, before Block C) | `from-log`, `from-gguf`, `edit --set REGEX=TYPE`, `llama-file`, `check`, `data-bytes`, `diff` | Python 3 standard library only. Rules of 0.4. Block sizes READ-SRC from ggml-common.h. Exits 1 on any mismatch, 2 on bad input; never overwrites. Later addition: `overlap` (64-token shingles through `pelias tokenize`) for GP1-10 |
| D-T2 `pelias kld --route-dump OUT.bin` | For every scored token and each of the 40 trunk layers: the top-8 expert ids in rank order, their routing weights (fp32), the 8th and 9th router logits | Header `PELROUT1`, n_ctx, chunks, scored, the same token hash as `--dump-kld`. Then per token per layer: 8 × u8 ids, 8 × f32 weights, f32 logit8, f32 logit9 (52 bytes). 25,500 × 40 × 52 = 53.0 MB per run (DERIVED) |
| D-T3 `pelias route-compare --a A.bin --b B.bin [--kld K.kld]` | Per layer: share of tokens whose top-8 set differs; share whose order differs; mean weight mass on changed experts. Per token: number of changed layers. With `--kld`: mean KL of tokens with 0, 1, 2–3, ≥4 changed layers, and the share of total KL carried by tokens with any change. Margin histogram (logit8 − logit9) in 0.01-wide bins | Refuses different token hashes |
| D-T4 `pelias kld --layer-subset` | Not needed; dropped. Per-layer tests use recipes (Block E) | — |
| D-T5 `pelias requant-check --model FILE [--imatrix F] [--tensors REGEX]` | For each quantized tensor: dequantize to fp32, quantize again with Pelias's quantizer at the same type (with or without the imatrix), then count blocks that come back byte-identical. Report per tensor and per type | Uses `quantize_rows`, the oracle-checked quantizer. Read-only on the input |
| D-T6 `tools/adapter_stats.py` | From the v0.2 LoRA adapter (safetensors; path given by the author): per module, ‖scale·B·A‖ RMS, ‖W‖ RMS of the matching served weight (dequantized), their ratio, and the share of entries of scale·B·A smaller than half a quantization step of the served type at that position | Python with numpy (test-only). Reads the served GGUF through `pelias inspect --json` offsets |
| D-T7 `pelias imatrix --gram TENSORS` | Next to the diagonal sums, accumulate the full Gram matrix XᵀX (fp64) for the named dense input spaces | Writes `W.gram` (cols × cols, f32 after the chunk average) into a sidecar GGUF; dense tensors only (2048 or 4096 columns: 16 or 64 MB each, DERIVED) |

Routine for each tool: unit tests, gcc/clang/asan, oracle where one exists, privacy scan, dependency check, commit, push on the author's word, bundle, backup.

#### GP1-06 Routing flips (old name L3) (A, ESTIMATE 2 runs, see time)

**Question:** is the KL tail caused by quantization changing which experts are chosen?

**Runs:**
1. ```
   $PELIAS kld --model $T/qwen36-bf16.llamacpp.gguf --base $T/kld-base-bf16-100chunks.bin --threads 16 --chunks 10 --dump-kld $R/GP1-06/bf16_10.kld --route-dump $R/GP1-06/bf16_10.rt
   ```
   This is the bf16 file through the reference. It also gives **the floor of the base itself** (GP1-07).
2. The same 10 chunks for the stock file and for T8, with `--chunks 10 --route-dump`.

**Time:**
- bf16: unmeasured. The 71 GB file does not fit the 48 GB WSL memory cap, so each chunk reads most of it from D: through the Windows mount. The engine measured that mount at about 0.5 GB/s for large files, so ESTIMATE 2–3 min per chunk, 20–30 min for 10 chunks.
- The tool prints the ETA after chunk 1. Stop rule: if the ETA exceeds 2 h, stop and copy the file to the WSL disk first (it was measured at 2.99–3.06 GB/s).
- Stock and T8: 10/100 of 22 min ≈ 2.5 min each.

**Compare:** `pelias route-compare` of bf16 against stock, and of bf16 against T8, each with that file's KL dump.

**Prediction (ESTIMATE):**
- In 2–8 % of token-layers the top-8 set differs.
- Tokens with ≥ 1 changed layer carry over half of the mean KL. The H6 tail note says "likely flips, not tested".

**Decision:**
- The tail is carried by flips: routing-preserving weighting (Part 3, GP3-06) goes ahead.
- It is not: drop it.

**Also measured:** the margin histogram, i.e. how close the 8th and 9th experts are (P-E3).

#### GP1-07 The floor of the base (new) (A, comes from GP1-06 run 1)

**Question:** the base was written by llama.cpp from bf16 weights with bf16 or f16 arithmetic on the GPU path. How far is that from Pelias's fp32 reading of the same bf16 weights? Every candidate's drift sits on top of this number.

**Measure:** the mean KLD printed by GP1-06 run 1 (10 chunks, 2,550 tokens).

**Prediction:** 0.0005–0.003 (ESTIMATE).

**Decision:**
- If it is more than a quarter of the gap between T8 and Unsloth, a Pelias-made base (fp32 reference logits from the bf16 file) replaces the llama.cpp base for final rankings. That is a measurement tool (`pelias kld-base`) written in Part 2.

#### GP1-08 Grid idempotence (P-E7) and the fine-tune size (ΔW ≈ 3 % σ) (A+C)

**Moved to the trainer phase (2026-10-27):** it decides the trainer's Mode Q only; the engine never re-quantizes. Run it when the trainer build starts.

**Run 1: re-quantization on the served weights.**
```
$PELIAS requant-check --model $T/qwen36-Q4_K_M.pelias.gguf
```
Then the same with `--imatrix $U/imatrix_unsloth.gguf_file`, and on `$M/t8.gguf`.
- Time ESTIMATE 5–15 min each (one dequantize and one quantize pass over 21 GB, 16 threads).
- **Question:** if Mode Q re-quantizes an unchanged weight, does it get the same bytes back?
- **Prediction (ESTIMATE):** Q8_0 ≥ 99 % identical blocks. Q4_K/Q6_K without imatrix below 90 % (the scale search starts from the data, not from the grid). With imatrix lower still.
- **Decision:** the Mode Q design (GP2-04). Below about 99 %, Mode Q must skip unchanged rows or quantize only the delta-touched rows, or the rounding change itself becomes drift. That drift is measured with `pelias kld` on the re-quantized file.

**Run 2: adapter statistics.**
```
python3 tools/adapter_stats.py --adapter <v0.2 adapter> --model $T/qwen36-Q4_K_M.pelias.gguf
```
- The author gives the adapter path (local only).
- **Question:** the scaffold presumes ΔW ≈ 3 % of σ. What is it really, and what share of the update is smaller than half a Q4_K step (and would be erased by a merge)?
- **Prediction:** unknown (UNVERIFIED by design).
- **Decision:** the core type and Mode Q requirement (scaffold 4.3, 9.4).

### Block E: the second and third evaluation texts and long context (A; base generation is the long part)

#### GP1-09 A 4096-token base (old name F4) (A, ESTIMATE 2 h base + 45–60 min per Pelias file)

**Make the base:**
```
$LLAMA_BIN/llama-perplexity -m $T/qwen36-bf16.llamacpp.gguf -f $WIKI -c 4096 --chunks 25 -ngl 0 -t 8 --kl-divergence-base $T/kld-base-bf16-25x4096.bin 2>&1 | tee …
```
- Same mode as the 512 base (GPU visible), so the two bases share arithmetic.
- 25 chunks × 4096 = 102,400 tokens, 2 × the 512 base. WikiText-2 test has enough: UNVERIFIED, so the tool's chunk count is checked in the log.
- Scored: positions 2048–4094, so 2,047 per chunk and 51,175 in total.
- Size DERIVED from the base format (496,648 bytes per scored row; the 512 base's 12,664,728,820 bytes reproduce exactly from it at 255 rows per chunk): 2,047 rows per chunk, about 1.017 GB per chunk and **25.4 GB** in total. On D: (140 GB free).
- Time ESTIMATE about 2 h: the 512 base took 57 min for half the tokens, and attention cost grows with context.

**Runs:** `pelias kld --base $T/kld-base-bf16-25x4096.bin --dump-kld …` on stock, T8 and Unsloth. Time ESTIMATE 45–60 min each: twice the tokens; the 10 full-attention layers' cost grows with length.

**Question:** do the rankings and the T8–Unsloth verdict hold at 8 × the context?

**Prediction:**
- Same ranking.
- Absolute KLD within ±30 % of the 512 values. Later positions carry more context, so drift could go either way.

**Decision:** the rankings hold, or long context becomes its own selection criterion.

#### GP1-10 A domain base (old name F5) (A; the author chooses the text)

**Candidate texts:**
- `code_cpp.txt` (1,628,968 bytes, C++ source already in the P-D1 corpus; public).
- A tool-call / agent text.
- The local-only calibration text from the v2 dataset build (2026-10-19).

**Rule:** the evaluation text must never be the calibration text of any imatrix being judged. Overlap is checked with a 64-token shingle comparison before use (tool D-T1 `--overlap`).

**Base:** as GP1-09 at `-c 512 --chunks 100` (57 min, MEASURED for WikiText).

**Runs:** stock, T8, Unsloth, and the winner of GP1-05.

**Question:** does the choice change with the text? Unsloth calibrates on chat and code; their imatrix lowered WikiText KLD but raised its PPL (MEASURED, experiment a).

**Decision:** the final PEL recipe is chosen on the paired verdicts of both texts. A candidate must not lose on either.

#### GP1-11 KV cache q8_0 against f16 at long context (A, llama.cpp only, ESTIMATE 2 × 15 min)

**Question:** the engine's default KV cache is q8_0. Its quality was never measured (masterwork 18).

**Commands:**
```
$LLAMA_BIN/llama-perplexity -m $T/qwen36-Q4_K_M.pelias.gguf --kl-divergence --kl-divergence-base $T/kld-base-bf16-25x4096.bin -c 4096 --chunks 25 -ngl 99 --n-cpu-moe 40 -fa on 2>&1 | tee …
```
- Run it once as is (f16 cache), then once with `-ctk q8_0 -ctv q8_0` added.
- The GPU path is used on purpose: this is the serving arithmetic.

**Measure:** the difference between the two mean KLDs. Pairing is not available from llama.cpp output.

**Decision:** q8_0 stays the default if the difference is below the smallest gap between finalists. Otherwise f16 or bf16 KV becomes the default and its VRAM cost is planned (masterwork 18.2).

### Block F: per-layer allocation (Unsloth's "dynamic" lever) and the own imatrix (A; screening with llama.cpp, confirmation with Pelias)

#### GP1-12 Per-layer `ffn_down_exps` sweep (A, ESTIMATE 8 × 8.5 min screening + 2 × 26 min confirmation)

**Question:** in which layers does Q6_K (or Q5_K) for the down experts pay?

**Screening:**
- Start from the T8 recipe. Raise `ffn_down_exps` to Q6_K in one group of 5 layers at a time: 0–4, 5–9, …, 35–39. That is 8 runs.
- Each run: quantize (4.2 min), then llama.cpp KLD (4 min, the GPU-visible path, consistent within the screen).
- Size Δ per group (DERIVED by D-T1). Rank by KLD drop per GB.

**Refinement:** inside the two best groups, single layers (10 runs).

**Confirmation:** the best two allocations through Pelias with dumps, paired against T8.

**Prediction:** the first and last layers pay most. Stock already favours 0–4 and 34–39, and Unsloth uses Q6_K at 34, 38 and 39.

**Decision:** a per-layer down layout inside the 22.13 GB budget (without blk.40).

#### GP1-13 Own importance matrix (A, ESTIMATE 45 min + 26 min)

**Command:**
```
$PELIAS imatrix --model $T/qwen36-Q4_K_M.pelias.gguf --file <calibration text> --out $M/imat_own.gguf --ctx 512 --chunks 200 --threads 16
```
- Time ESTIMATE about 12 s per chunk: the forward pass without the output layer is slightly under the 12.7 s per chunk of `kld`. So about 40 min for 200 chunks.
- Calibration text: the author's choice, never the evaluation text (GP1-10 rule).
- Computed on the stock Q4_K_M file. Collecting on bf16 would be more exact but slower (GP1-06 timing). DECISION, proposed: stock file first, bf16 later if GP1-07 shows the floor matters.

**Then:** T8 + own imatrix (llama-quantize `--imatrix $M/imat_own.gguf`, type check), then Pelias with dump, paired against GP1-05 run b (T8 + Unsloth's imatrix) on both evaluation texts.

**Decision:** the calibration source for PEL.

### Block G: reading what the build must respect (C; no machine time)

#### GP1-14 Spec checks Claude does from files

| Check | How | Feeds |
|---|---|---|
| Attention scale | The GGUF key `qwen35moe.attention.scale` is read by `src/model/spec.cpp`. Confirm whether the served file and Unsloth's file carry it, and what the trainer scaffold hard-codes | GP2-03 |
| Norm +1 convention (P-S2) | Already exercised by the reference forward pass (MEASURED equal to llama.cpp on the tiny model). Record as DONE for the reference; the GPU kernels inherit the test | GP2-02 |
| IMRoPE, V-head order, conv tap order, top-k ties (P-S3, P-S4, P-S1, P-S8) | Same: the reference matches llama.cpp end to end on the tiny model (MEASURED), which covers these semantics for the CPU. Record each as "DONE by the reference, CPU", with the GPU repeat at step 5 | GP2-02 |
| MTP block semantics (P-S5) | Not covered: the reference does not run blk.40. Stays open for the decode loop (GP3-08) | GP3-08 |

### Block H: one activation format for CPU and GPU (author's idea, 10-27; A then C)

**Background (MEASURED 10-27, GP1-03 a and b, paired against the fp32 stock run):**
- llama.cpp's CPU activation format for K-quant weights (Q8_K, one scale per 256 values) costs +0.002383 drift, 95 % interval [+0.001640, +0.003126].
- The GPU format (q8_1, one scale per 32 values) costs +0.000610, interval [+0.000115, +0.001104]. About a quarter.
- Every llama.cpp drift number on record (Unsloth's 0.013605 included) carries the CPU cost: the GPU-visible runs used the CPU path (0.041124 against 0.041468 with the GPU off and 0.041397 for Pelias's CPU-format model).

**Idea:** round each activation once, in groups of 32 with a scale each, laid out so eight groups form the 256-value block the CPU K-quant loop walks. The GPU reads the groups as q8_1 does; the CPU reads eight in a row as one block. One rounding, identical inputs on both sides, about a quarter of fp32's bytes across PCIe.

**Why the CPU should not lose speed (DERIVED, to be measured):** the author's CPU (AMD Ryzen 7 5800X3D, AVX2, no AVX-512; READ from /proc/cpuinfo) holds 32 int8 values per 256-bit register, so a group of 32 is one register. The K-quant dot product already works per sub-block (32 for Q4_K, 16 for Q6_K) to apply the weight's sub-scale; an activation sub-scale adds one multiply per group. Bytes per 256 values: Q8_K 292, eight q8_1 groups 288.

#### GP1-15 Accuracy of the shared format (A, about 25 min)
- **Code (measurement only):** an activation format `q8_k32` in `act_round_row` (block 256 laid out as eight groups of 32, each with its own fp16 scale, the same precision as q8_1), selected with `--act-int8-path cpu32`. Unit test: identical rounded values to q8_1 on the same row.
- **Run:** `block_h.sh 15`: `pelias kld` on the stock file with `--act-int8 all --act-int8-path cpu32 --dump-kld`, then `kld-compare` against `GP1-01/stock.kld` and `GP1-03/b_allgpu.kld`.
- **Prediction:** equal to run b (0.039623) in every digit if the scale is fp16 per 32, because the rounded values are the same; a variant with 8-bit sub-scales under one fp32 scale is predicted between run b and run a.
- **Decision:** feeds GP2-05 (the engine's activation policy) and GP3's CPU expert kernels.

#### GP1-16 CPU speed of the shared format (A, ESTIMATE under 10 min)
- **Code (measurement only):** an AVX2 Q4_K × `q8_k32` dot product next to the existing Q4_K × Q8_K one, same loop shape, checked against the scalar reference first.
- **Run:** both kernels on the real expert shapes (row lengths read from the file with `pelias inspect`) over many rows, 5 repeats, median.
- **Prediction:** within 5 % of the Q8_K kernel (DERIVED from one extra multiply per 32-value group; the 5 % line is proposed, the author sets it).
- **Pass:** within 5 %, or the accuracy gain from GP1-15 judged worth the cost by the author.

---

### Block J: rotation and kernel efficiency (author, 10-27 21:1x; A)

**Background (MEASURED 10-27):** GP1-03 d puts the GPU-style rounding cost on the dense layers: +0.000975 [+0.000280, +0.001669]; the experts cost nothing measurable (c). The int8 kernel is issue-bound: 13.4 instructions per mma, 37.7 % tensor busy (Nsight Compute r8-r10); expert down (K = 512) runs at 0.27-0.29 of int8 peak (O13).

| Test | What | File changed | Prediction | Time |
|---|---|---|---|---|
| R1 | `--act-int8 attn,ssm_,shexp,output --act-int8-path gpu --act-rot h32`: each 32-value block of the dense inputs rotated (orthonormal Walsh-Hadamard), rounded, rotated back; weights untouched (exact when the weights are rotated the same way) | No | Between fp32 0.039013 and d 0.039988 if the real activations carry large single values per block; the tiny random model got worse with it (1.302053 against 1.286214), where activations have no outliers, so the direction is open | 25 min |
| R2 | Dense weights rotated per 32 from bf16 and quantized to Q8_0; Pelias fp32 | Yes (B2) | Equal to the unrotated dense drift or better | About 1 h |
| R3 | `ncu` on the int8 kernel before and after a `_P` plane layout with 1-2 instruction unpack, scales converted per tile on chip, next-tile overlap; shapes attn_qkv M=4096 and expert down K=512 | No | Instructions per mma below 13.4, tensor busy above 37.7 %, O13 above 0.29 | Kernel work, scoped before it starts |
| R4 | VRAM bandwidth with and without locked memory clocks (admin) | No | At most +2 % (CUDA lowered the memory clock 11,501 to 11,251 MHz in the 10-27 power test) | 5 min |

Run R1: `bash $R/block_j.sh r1` (updates the clone from the newest bundle, rebuilds build/m1, runs R1, compares against GP1-01 and GP1-03 d).

**R1 RESULT (MEASURED 2026-10-27 21:58, 25,500 tokens, 1517 s):** 0.038802. Against d (same rounding, no rotation): -0.001186 [-0.001913, -0.000458], t -3.23, rotation wins. Against fp32: -0.000211 [-0.000586, +0.000163], no difference. The rotation removes the whole dense-layer 8-bit rounding cost on the real model (the tiny random model said the opposite: its activations have no large values). Prediction (0.039013-0.039988) landed 0.0002 below, within noise: best case.

**What R1 changes (DECISION proposed, 10-27 22:0x):**
- **R2 moves first and becomes required** for deployment: R1 assumes exactly rotated weights; a real file holds rotated weights quantized again. R2: rotate the dense weights per 32-wide input block from the bf16 original, quantize to Q8_0 (Pelias quantizer), score with Pelias with the same activation rotation, against GP1-01 and R1.
- **R2b (new): rotated expert weights.** Rotation is known to help low-bit weight quantization too (QuaRot, PUBLISHED). Rotate `ffn_gate_exps`/`ffn_up_exps` (input 2048) and `ffn_down_exps` (input 512) per 32 before Q4_K/Q5_K, with the importance matrix collected in the rotated basis (`pelias imatrix` with the rotation). If it lowers the file's own drift, it is a direct O20 lever, so it runs before Block C, whose candidates would then be built rotated. Files are Pelias-only (decision B2).
- **Dropped for the dense layers:** selective rotation, adaptive quasi and the outlier map (R1 already reaches fp32 with every block rotated). Two-limb quasi becomes the fallback, run only if R2 loses much to re-quantization.
- **Unchanged:** routing statistics (router weight spread, cold-expert reuse between tokens, next-layer prediction, routing flips), recorded in R2's run; the cold-step breakdown; T6 as a Pelias Q5_P; R3 (simpler: one int8 pass, the rotation is 5 shuffles in the fused RMSNorm); Blocks D-G; GP1-08 moves to the trainer phase (it decides Mode Q only).

**R2 and R2b as built (2026-10-27 22:1x-22:2x; the author authorized writing and running both):**
- Tool: `pelias quantize --model <bf16> --type Q4_K_M --rotate32 GROUPS --out OUT`. The selected product weights are rotated row by row (32-value Walsh-Hadamard) before quantization at the stock Q4_K_M types (no importance matrix, like the stock file) and listed in `pelias.rot32.tensors`; the reference rotates their inputs. Everything else follows the stock path exactly, so `pelias verify` against the stock file must show only the rotated tensors differing ("CONTROL OK"). If not, `block_j.sh ctrl` builds an unrotated control the same way and both are compared against it.
- Files: `D:\pelias-tests\rot\` (author's choice): `r2_dense.gguf` (dense rotated), `r2b_exps.gguf` (experts rotated), `q4km_ctrl.gguf` only if needed. About 21.7 GB each; the script stops if a write would leave under 50 GB free.
- Runs: `block_j.sh r2` (build, verify, Pelias fp32, Pelias dense q8_1; ESTIMATE 65-80 min), `block_j.sh r2b` (build, verify, Pelias fp32; ESTIMATE 40-55 min). Build time ESTIMATE 10-25 min: the bf16 source (66 GB) is read through the Windows mount (MEASURED 0.17-0.52 GB/s for large files) and Pelias's whole-file quantize time on this model is unmeasured.
- **Predictions (written before the runs):** R2 fp32 below the stock 0.039013 with the paired interval below zero (rotation helps 4-bit weights, PUBLISHED QuaRot; tiny model -1.1 %), the gain at most about 0.0216 (DERIVED: the dense groups' drops at Q8_0, summed). R2 dense q8_1: no difference from R2 fp32 (R1). R2b fp32 below 0.039013, the gain at most about 0.0059 (DERIVED: 0.002384 + 0.002127 + 0.001414); tiny model -0.1 %.
- **Decision:** a resolved gain in R2b makes rotation part of the PEL recipes, so Block C's candidates are built rotated, which needs importance-matrix collection in the rotated basis (accumulate (Hx)^2 instead of x^2 for rotated tensors; not built yet).

**R2, ctrl and R2b results (2026-10-27 22:23 to 2026-10-28 00:14, MEASURED, 25,500 scored tokens, paired, 99 df; every number re-computed from the per-token dumps):**

| Run | Mean KLD vs bf16 | Paired against | Difference [95 % interval] | t | Verdict |
|---|---|---|---|---|---|
| R2 dense rotated, fp32 | 0.035564 | stock 0.039013 | -0.003449 (from the means) | | less drift |
| R2 dense rotated, dense q8_1 | 0.035297 | R2 fp32 | -0.000267 [-0.000793, +0.000258] | -1.01 | no difference |
| R2 dense rotated, dense q8_1 | 0.035297 | stock fp32 | -0.003716 [-0.004934, -0.002498] | -6.05 | less drift |
| ctrl (unrotated twin, same build path) | 0.039018 | stock | +0.000005 [-0.000018, +0.000028] | | no difference |
| R2 fp32 | 0.035564 | ctrl | -0.003454 [-0.004585, -0.002323] | -6.06 | less drift (-8.9 %) |
| R2b experts rotated, fp32 | 0.039544 | stock | +0.000530 [-0.000396, +0.001457] | 1.14 | no difference |
| R2b | 0.039544 | ctrl | +0.000526 [-0.000402, +0.001454] | 1.12 | no difference |
| R2b | 0.039544 | R2 fp32 | +0.003979 [+0.002492, +0.005467] | 5.31 | R2 less drift |

- **Predictions:** R2 met (below stock, interval below zero, gain within the 0.0216 ceiling); R2 q8_1 met (no difference from fp32); **R2b missed** (predicted below stock; no resolvable change).
- **Reading:** rotating the dense weights (Q8_0/Q6_K/Q4_K dense groups in the stock mix) lowers the file's own drift by 8.9 % and removes the activation-rounding cost. Rotating the K-quant experts does nothing measurable. Likely cause (THEORY, not tested): K-quants already give each 32-value sub-block its own scale, so the experts' outliers were absorbed before rotation; the tiny model showed the same (-0.1 %).
- **Decision:** PEL recipes rotate the dense product weights only (`--rotate32` on the R2 groups); routed and shared expert weights stay unrotated, so the expert reader path needs no rotation. Block C's candidates are built dense-rotated; importance-matrix collection in the rotated basis is needed for the dense tensors only. `r2b_exps.gguf` has no further use (the author decides on removal).
- **Waiting for the author's OK:** R2-256 and R2-row (larger rotations, dense only), `block_k.sh routes` (GP1-06/07 and routing statistics).

**Test-list changes after R2/R2b (the author agreed 2026-10-28 00:21):**
- **Ruled out:** (1) all expert-rotation follow-ups (no 256/row rotation of experts, no expert-rotated files in Block C); (2) GP1-15's machine run: the `q8_k32` format rounds to the same values as q8_1 by construction, the experts lose nothing at per-32 rounding (GP1-03 c) and the dense rounding cost is gone with rotation (R1, R2 q8_1), so the unit test "identical rounded values to q8_1" replaces the run; GP1-16 (CPU speed) stays; (3) the "effective bits" measurement and the two-limb quasi fallback (insurance no longer needed).
- **Narrowed:** importance-matrix collection in the rotated basis is needed for dense tensors only; R3's rotation step applies to the dense inputs only, the expert kernels are unchanged.
- **Lower priority:** R2-256 and R2-row. R2 q8_1 already equals R2 exact, so a larger rotation can only lower the dense weights' own drift; R2b suggests per-32 blocks already capture most of it. Run them only if Block C lands close to Unsloth.
- **Unchanged:** Block C (GP1-05), GP1-11, GP1-12, GP1-13, `block_k.sh routes`, R4.

**block_k routes result (MEASURED 2026-10-28 00:28-00:50, 10 chunks, 2,550 scored tokens x 40 layers; re-computed from the record files):**
- **GP1-07 floor:** the bf16 file through Pelias against llama.cpp's bf16 base: 0.002801 +/- 0.000281 (stock on the same 10 chunks 0.041817, so the floor is about 6.7 % of stock's drift). Prediction 0.0005-0.003: met, at the top edge.
- **Routing flips (stock vs bf16):** the picked set differs in 56.4 % of token-layers (prediction 2-8 %: MISSED). 81.6 % of those swap exactly one expert; the expert bf16 picked and stock dropped is rank 7 or 8 in 80.3 % (rank 8 alone 54.6 %); the weight lost is 0.097 per flipped token-layer. Router margin between the 8th and 9th expert: median 0.00055, 69 % of token-layers under 0.001: the tail picks are near-ties.
- **Flips do not drive the drift:** every token has 4 or more changed layers, so the planned "share of KL on changed tokens" cannot separate anything; per token, the weight lost to swaps has correlation -0.004 with KL, and KL by quartile of swapped weight is flat (0.037-0.046). Prediction "changed tokens carry over half the KL": NOT SUPPORTED. The drift comes from the weights' own error, not from routing.
- **Weight spread:** rank 1-8 mean weights 0.246 / 0.169 / 0.135 / 0.113 / 0.098 / 0.087 / 0.079 / 0.073; top two 41.5 %, ranks 5-8 33.8 % (dropping to fewer experts would lose a third of the mass).
- **Hot sets (stock):** picks outside the per-layer top 84 / 120 / 161: 14.2 / 6.5 / 2.3 %. Picks already made by the previous token: 41.5 % (2 tokens 50.0, 4: 59.3, 8: 68.3); cold picks only 13.6 % at top 120. Distinct cold experts in an 8-token window at top 120: 134 over 40 layers.
- **Next-layer prediction** (this layer's output through the next layer's norm and router): 83.6 % of the next layer's picks, 67.1 % of its cold picks at top 120: usable as a prefetch hint one layer ahead.
- **What it changes:** no protection of router ties is needed for accuracy (drift is in the weights; Block C and F stay the levers); the hot bank sizes in GP3 use the measured coverage (120 per layer = 93.5 % of picks); cold-expert prefetch can start one layer early from the prediction (67 % hit on cold picks). GP1-06's remaining question (does the floor matter) is answered: 0.0028 is small against the 0.0136 target, but not zero.

**Block L: rotated candidates against Unsloth (authorized by the author 2026-10-28 00:59; script `$R/block_l.sh`, dry-run end to end on the tiny model):**
- `check` (about 12 min): R2 rebuilt from bf16 through `--recipe` (recipe read from `r2_dense.gguf`); prediction: byte-identical to `r2_dense.gguf`. Writes 21.7 GB to D: (a duplicate the author can remove after).
- `imatrix` (about 45 min): `pelias imatrix` on `r2_dense.gguf` (dense collected rotated, experts unrotated), from the v2 calibration text (local only, never published; read through a link named `calib.txt` so only that neutral name is recorded), up to 200 chunks of 512.
- `c` (about 85 min): Unsloth through Pelias; t8r = T8 recipe (from the sweep log) + dense rotated + the matrix; t9r = t8r with gate/up experts IQ4_XS; all paired. Candidates on C: ($M).
- **Predictions (written before the runs, Pelias scale, DERIVED from the llama.cpp sweep with the stock offset 0.039013 vs 0.040965):** Unsloth 0.0117-0.0130; t8r 0.0113-0.0135 (T8 0.0129-0.0141, minus about 0.0006 for the matrix, minus 0-0.001 for rotating mostly-Q8_0 dense weights); t8r vs Unsloth a tie or a t8r win; t9r 0.0003-0.0010 above t8r at 21.81 GB.
- **Decision:** a resolved t8r win against Unsloth meets O20 on WikiText (Block E's second text still to confirm); a tie sends the work to Block F (per-layer allocation) with rotation kept.

**Block L c results so far and the size rule (2026-10-28 03:0x):**
- Unsloth through Pelias 0.012095 (prediction 0.0117-0.0130 met). t8r 0.012941; t8r vs Unsloth +0.000846 [+0.000283, +0.001408], t 2.98: Unsloth has less drift (7.0 %), prediction "tie or t8r win" MISSED; t8r carries Q6_K on attn_qkv/attn_q (40 tensors) where Unsloth has Q8_0, and the gap widened on hard chunks 79-80.
- **t9r (MEASURED 03:35, verified):** 0.013479; vs Unsloth +0.001384 [+0.000847, +0.001922] t 5.11; vs t8r +0.000539 [-0.000018, +0.001095] t 1.92 (not resolved). IQ4_XS on every gate/up costs about +0.0005 for 688 MB; prediction 0.0003-0.0010 met, and the per-tensor error ratio (IQ4_XS 1.153x Q4_K on the real slices) predicted +0.00066, within 20 %: the error-ratio method for estimating mix trades is usable.
- **Size rule (author, 03:04, refined 03:11): a candidate counts only if it is no larger than Unsloth's 22,134,528,992 bytes, measured WITHOUT blk.40, because Unsloth's file has the MTP block removed (523.8 MB at T8 types); our file keeps the MTP block as an extra.** (03:04 first reading, kept for reference: whole file including blk.40.) Under this rule t8r's file (22,495,826,144) is 361 MB over, and Unsloth's exact types plus the MTP block would be 524 MB over. A winner must beat Unsloth while also carrying the MTP block inside the same bytes.
- **Two fitting recipes ready (sandbox, exact byte counts from Unsloth's header and T8's types; MTP block not scored by the reference, so its types change speed only):**
  - **Mix A** (22,131,470,560 bytes, 3.1 MB under): Unsloth's types (all dense Q8_0), token_embd Q6_K, ssm_alpha/beta Q8_0 (were F32), MTP experts Q3_K, gate/up IQ4_XS in layers 13-27.
  - **Mix B** (22,119,936,224 bytes, 14.6 MB under): as A but attn_qkv/attn_q at Q6_K (t8r style), so gate/up IQ4_XS only in layers 17-22.
  - Both with dense rotated and the rotated imatrix. Under the refined rule they are not needed; they stay on file for a whole-file comparison.
  - **Under the refined rule the like-for-like candidate is ur:** Unsloth's exact types on the 733 shared tensors (22.13 GB without blk.40, the same bytes), dense rotated, MTP block kept. **Predictions (DERIVED 03:09):** t8r's gap splits into about 0.0002 from attn_qkv/attn_q at Q6_K (Q6_K error 6.2 % of Q4_K's on rotated attn_qkv x the sweep's 0.00339 for those groups) and about 0.00065 of method, mostly the importance data on the experts; ur with our imatrix about 0.0127 (loses by about 5 %); **ur with a merged imatrix (Unsloth's entries for the unrotated experts, ours for the rotated dense) about 0.0121-0.0124, a near tie.** The merge tool is not built yet. Middle layers take the IQ4_XS on the assumption (not measured) that first and last layers are the most sensitive; GP1-12-style per-layer measurement would replace it. t9r (IQ4_XS gate/up in every layer) measures the cost that decides between A and B.

**Block N result (MEASURED 2026-10-28 03:49-04:33, verified from the dumps):** Unsloth's MTP UD-Q4_K_M, 22,663,387,424 bytes, SHA-256 equal to Hugging Face's. Against the non-MTP file: identical types, 613 tensors identical, all 120 routed-expert tensors re-rounded (same types, other values), 20 MTP-only tensors.
- fp32 **0.012231** (vs non-MTP +0.000136, not resolved; vs t8r -0.000710 [-0.001265, -0.000154], Unsloth MTP better at fp32 while t8r is 167 MB smaller).
- 8-bit activations on every weight (unrotated, q8_1): **0.012864**, +0.000633 [+0.000220, +0.001046] over its own fp32 (prediction +0.0006 to +0.001 met): Unsloth's price for the fast path.
- **Unsloth's fast-path number equals t8r's fp32** (-0.000077, not resolved). Block M c measures t8r with 8-bit activations (rotated dense) and pairs it with Unsloth's 8-bit run directly.

**The proper test against Unsloth, and urm (author and Claude, 2026-10-28 03:2x):**
- **Bar:** Unsloth's MTP UD-Q4_K_M (listed at 22.7 GB; to be downloaded, SHA-256 checked, and compared with the scored non-MTP file by `pelias verify`).
- **urm = "Unsloth's file, rotated":** every tensor at the type Unsloth's MTP file uses (read from that file), dense weights rotated, merged importance data (Unsloth's own entries for the unrotated experts, ours for the rotated dense). Bytes equal to theirs (our ~11 KB of rotation labels offset by one ssm_alpha tensor F32 -> Q8_0, about 200 KB); the build stops if the file is even 1 byte larger.
- **Expected:** with Unsloth's importance data and Pelias rounding byte-identically to llama-quantize, the expert tensors come out bit-identical to Unsloth's (if Unsloth used standard llama-quantize; `pelias verify` shows it); the dense layers are Q8_0 in both (relative error 0.000028), so in fp32 urm is essentially the same model: **fp32 prediction a tie with 0.012095.** With 8-bit activations (the engine's int8 path) Unsloth's unrotated dense layers pay the rounding cost (GP1-03 d: +0.000975 on stock; expected about +0.0006 to +0.001 here) while rotation makes it free for urm (R1; R2 8-bit = exact): **8-bit prediction: urm ahead by about 0.0005, interval below zero.**
- **Protocol:** (1) size gate, whole files with MTP; (2) paired KLD vs bf16, 100 chunks, fp32 (must not lose) and 8-bit activations (must win); (3) a second text (domain base, GP1-10; must not lose on either; calibration never an evaluation text); (4) optional 4096-token context. Not testable yet: MTP draft acceptance (needs GP3-08) and speed (needs the engine). Note: a rotated file runs only in Pelias; an unrotated variant scored the same way would answer the llama.cpp-compatibility side.
- **Beyond a tie in fp32:** bits by usage per expert, GPTQ-style rounding (GP3-04), better importance data.

**Building Block C's rotated candidates (2026-10-28 00:4x):**
- Done: `pelias imatrix` collects rotated dense weights in the rotated basis (BUILD_LOG 00:3x).
- Gap found: Block C's recipes (per-tensor types, `--imatrix`, IQ4_XS) are built with llama-quantize, which cannot rotate; `pelias quantize --rotate32` rotates but has no recipe or imatrix input.
- Proposed route (needs the author's OK): `pelias quantize --type BF16 --rotate32 <R2 groups>` writes a rotated bf16 source (the key `pelias.rot32.tensors` is carried; llama-quantize copies all metadata, READ-SRC llama-quant.cpp), then llama-quantize builds each recipe from it with the rotated-basis imatrix. Check first: the same route at plain Q4_K_M against `r2_dense.gguf` (`pelias verify` and one KLD run); the only expected difference is the bf16 rounding of the rotated dense weights (DERIVED: relative error about 2^-9, below Q8_0's rounding). The rotated source is about 66 GB: D: has about 78 GB free, so it goes to C: or follows the removal of `r2b_exps.gguf`. Alternative: a `--rot32-f32` option keeping the rotated tensors in F32 (exact, a few GB larger).

**Fable review of the author's 5-mechanism proposal (read-only, 2026-10-27 21:3x), adopted:** Q5_K's `qh` is the most significant bit, so dropping it is not a 4-bit approximation (the working form is a Pelias Q5_P with a top-4-bit base plane; hot-bank gain 0.52 GiB, not 1.24); the 256->32 transcoder makes data 1.89x larger and gains nothing; RT cores cannot rank 2048-wide routers; the memory clock lock is worth at most 2.2 %; Q7_P on attn_qkv/attn_q saves 168 MB; the trainer's optimizer state is 323 MiB and does not fit in L3. The E0 cold step (10.1 ms) is about 4.3 ms of kernel time and about 5.8 ms of per-step overhead (DERIVED from llama.cpp's CPU-only 16.28 tok/s), so the step structure must change, not only the kernel. Non-temporal loads act only on write-combining memory: on cached RAM the tool is `prefetchnta`.

## Part 2. What gets fixed after the tests

Each fix names the test that triggers it. Fixes with no trigger are corrections known now.

### GP2-01 Accuracy gates rewritten around paired comparisons (known now)

- **Replace** "KL ≤ 1.0 × σ_float (0.017353)" everywhere: trainer scaffold 10.2, 9.6, K1–K3, rungs 3 and 7, P-X7, P-Q3, P-Q5, P-Q6; engine blueprint E3/E7. σ_float is larger than T8's whole drift, so it cannot separate mixes.
- **New wording:**

| Gate | New definition |
|---|---|
| K1 (the trainer's forward pass vs the reference on the same GGUF) | Mean KLD of the trainer forward against Pelias's CPU reference ≤ the GP1-07 floor, with the paired interval reported |
| K2 (exported file vs the trainer's own reference) | Paired kld-compare of exported against trainer-reference logits shows no regression: interval upper bound ≤ 0 + the tolerance set from GP1-03 |
| K3 (engine vs the reference) | The engine's KLD against the bf16 base is within the activation-rounding cost measured in GP1-03 for the rounding mode the engine uses |
| P-X7 | Gets a pass line: the Pelias file is no worse than the stock reference file on a paired comparison |

- **Pick one KL method:**
  - Full-vocabulary KL for anything computed on the CPU reference.
  - "Top-256 + lumped tail" only where storage forces it (trainer teacher logits). Its bias against full KL is measured once on 10 chunks (DERIVED from the same logits).

### GP2-02 Status corrections in the trainer scaffold, notes, engine docs and master (known now)

- **P-E10:** partly done (13.31 GB/s, Q4_K rows only).
- **P-X7:** partly done (stock 0.039013 against bf16, CPU reference).
- **P-S6/P-Q6:** partly done by GP1-03.
- **P-S1–S4, S7, S8:** done for the CPU reference (GP1-14).
- **P-D1:** done.
- **P-E6, P-X6:** done.
- **G15 (pinned memory):** refuted above 20.4 GiB (32 GiB held).
- **Probe count:** settle at 102.
- **P-E13:** partly done.
- **ID collisions:** renamed by Appendix A of this file.

### GP2-02b Stale records found by the reader of the masterwork and the attached specifications (known now)

| Where | What is stale or in conflict | Fix |
|---|---|---|
| Masterwork header | Still says "No product code exists" | Name what exists (reader, quantizers, tokenizer, Stage A, CPU reference, `kld`, `imatrix`, `kld-compare`) |
| Masterwork 18.2 | Step numbers predate the revised 15.2 order (Vulkan is step 16; Mode L step 16; intake/coach step 13; installers step 15; dense step 14). O12 and O13 each mean two things | Renumber; split the O12/O13 meanings |
| Masterwork O20 | Says the 0.53 GB size difference to Unsloth is "not yet explained" | It is explained, exact to the byte: blk.40 under (b)'s layout is 525,716,544 bytes and Unsloth's file has no MTP head (PEL_QUANT_DECISION_DATA 2.3) |
| Masterwork 17.2, P-M4 | The P-M4 row reads a "22.13 GB Q4_K_M" file. That size is Unsloth's file, not the served Q4_K_M (21.71 GB) | Name the file that was read |
| Masterwork step 11 gate | Dropped the persona criterion (≤ 12 of 111 not-pass) that the ground-up specification carries | Restore it (GP3-15) |
| Masterwork 3.3 | Says the reference is fp64 (`tests/ref/`) | Record the fp32 C++ reference behind `pelias kld` and the 0.039013 reading |
| Masterwork | P-G0b is both "repeats pending" (intro) and "done" (21.7) | Settle as done |
| Universal build master, ground-up specification | Port 8012, the 10.9 GiB ceiling, "native Linux held", P-M4 "awaits", Batch C "next" | Frozen input copies. The registry marks each line as superseded (A8, D26/D28, A2, 17.2, 17.5) |
| LoRA scaling | rsLoRA α/√r in the research update and the parameter dictionary option, against α/r in the Mode Q formula and in v0.2. A 4 × difference in the update at r = 16, α = 32 | Author ruling (Appendix B, B12). Affects training code |
| Rung 5 stall gate | "N ≥ 3,072" (universal build master) against "≥ 16 k tokens per step" (masterwork) | The masterwork governs (3,072 is the default minimum step, not the stall gate) |
| Rung 1 tolerances | Two tables: against the unrounded op (universal) and against the same rounded inputs (trainer 6.11) | The trainer's governs (C14). The universal values are recorded for information |
| Stream time | 1.49 s (80 fetches) against 1.47 s (79 fetches) | 79 fetches; 1.47 s (C23) |
| v0.2 numbers | 3.01 M against 3.72 M processed tokens; median 247.85 against "about 300 tok/s averaged" | Different definitions (median against mean; processed against training tokens). Label them |
| "Spine" size | 1,409.0 M core parameters against about 1.94 B / 3.62 GiB (The Accidental Find) | Different scope (the latter includes routers, `lm_head`, norms). Label them |
| Hardware research plan (2026-10-23) | Uses the retired 57.0 tok/s as data in 9 places, including the E0 gate (within ±3 % of 57.0) and the E1 gate, which cannot pass as written. Stock baseline "32–38" against the measured 41.1–42.0. Reference numbers in its body run one behind its source list | Mark the file "audited hypothesis list, not a measurement source" (engine blueprint already does). E0/E1 gates come from the engine blueprint, not from this file |
| Parameter dictionary row `api.special_in_user_content` | The row is cut off mid-sentence | Restore from the masterwork (O10) |
| Ground-up master specification | Ends mid-sentence in 10.2 | The repo copy under `docs/inputs/` is checked for the same cut; the masterwork governs |

### GP2-03 Trainer spec numbers that were presumed (after Block D and G)

- **Attention scale:** read from the GGUF key when present, else 1/√head_dim. Never hard-coded.
- **ΔW:** the measured value from GP1-08 run 2 replaces "≈ 3 % σ".
- **Mode Q:** designed from the GP1-08 run 1 idempotence numbers. If below about 99 %, quantize only rows whose delta is non-zero, and record the rounding drift of untouched rows.
- **MTP:** an MTP oracle is defined. The reference forward pass gains blk.40 (GP3-08), and its draft logits are compared with llama.cpp's MTP path (oracle only).
- **v0.2 mismatch:** experts were trained as affine group-64 4-bit but served as Q4_K. Fixed by design in the new trainer (frozen experts are the served bytes). Proven by P-X1 (hash) at export.

### GP2-04 Author rulings needed before the build continues (Appendix B lists them all)

The tests supply the facts. The author rules. The main ones:
- D9/D33 (stock map, no imatrix) against PEL mixes with imatrix.
- The bit-exact rule against L1/L2.
- Whether PEL recipes become the trainer's base.

### GP2-05 Activation rounding policy for the engine and Mode F (after GP1-03)

- Per tensor group: fp16/bf16 activations or int8 (q8_1 or Q8_K).
- **Rule:** a group may round to int8 only if its paired cost (GP1-03 c or d) is below a tolerance the author sets. Proposed: a quarter of the gap between the chosen PEL mix and the next candidate.
- Recorded as a DECISION in the masterwork. Engine kernels and Mode F follow it.

### GP2-06 KV cache default (after GP1-11)

q8_0 or f16/bf16, with the VRAM cost from the planner.

### GP2-07 Parameter conflicts (known now; author rules each)

| Parameter | Ruling | Status |
|---|---|---|
| Server port | Pelias **8062** (A8); llama.cpp comparison server stays **8012** (author, 2026-10-27 13:50) | Ruled |
| Decode threads | 6 (dictionary) against 7 (CSV) | Open |
| `--prio` | CSV default 2 against the master's "never use --prio 2" | Open |
| Thinking default | true against false | Open |
| `max_tokens` default | −1 against 256 | Open |
| `poll` | 50 against 100 | Open |
| NTA prefetch | off against on | Open |
| Defaults that cite a measurement | The q8_0 "12/12 needles" is a capability test, not a drift test (GP1-11) | Open |
| Tier bypass default | 32 in both the dictionary and the CSV, but 32 is the tier build's crash path for 13–31-token chunks; production used 9 | Open. The Pelias engine gets its own crash sweep (GP4-02) |
| Pinned pointers | "Identical host and device pointers guaranteed on WSL2" (dictionary) against "dev_base − host_base offset, Path A" (CSV) | Open. Pointer rule: use dev_base, never assume equality, and test both |
| Sampling temperature | 1.0 "vendor thinking default" (no citation) against the vendor's 0.6 for coding in thinking mode (VRAM guide) | Open. Cite the model card |
| One flag, two meanings | `PELIAS_TIER_MAX_MIB` (ceiling and bank, GiB vs MiB); `--load-mode` (alloc backend vs load mode); stream flush (once per batch vs every enqueue, both citing D27); `--prio` applies `cpu_dma_latency` on Linux while the dictionary keeps it off | Open. The one dictionary (masterwork 7.1) decides each |
| Ranges and enum names that differ | Bank, pinned pool, ctx, batch, ubatch, top_p, top_k, seed, timeouts, KV types, backend, pin modes, hot-set objective names, L2 persist maximum, doorbell backoff (`__nanosleep` 64 against 200), zero-copy share (0.44 against "0.60 Pass 1"), MTP load naming, `max_connections` (8, "16 max", range 1–64) | Open. One dictionary row each, with a source |

### GP2-08 Training data port (known now)

The dataset was built with `--server-port 8012` (masterwork C20). It is rebuilt with 8062 before the next training run. Check: a grep of the rebuilt data for the old port returns nothing (count only, no text printed).

### GP2-09 PEL recipes locked (after Blocks C, E, F)

- **Files:** the winning recipes go into `profiles/pel/` as TSV plus a small JSON: name, version, the source bf16 hash, the imatrix file hash and its calibration text's SHA-256 (the hash only, never the text), measured KLD on each evaluation text, size with and without blk.40, and the date.
- **Naming:** PEL4 vN, PEL3 vN (author DECISION 01:59).
- **GGUF metadata keys:** `pelias.pel.recipe`, `pelias.pel.version`, `pelias.pel.calibration_sha256`.

### GP2-10 Docs updated with every result (after each block)

`PEL_QUANT_DECISION_DATA.md`, the masterwork O20 section, BUILD_LOG, RUNBOOK, TALLY (one row per result, with the "faster (time saved; %)" column).

### GP2-11 The 4096 and domain bases get IDs in the docs (after Block E)

All new base files are recorded with size, hash and how they were made.

### GP2-12 A Pelias-made base, if GP1-07 says the floor matters

`pelias kld-base --model BF16 --file TEXT --ctx N --chunks K --out BASE.bin` writes the same base layout as llama-perplexity from the fp32 reference. Then every ranking is repeated on it, with paired comparisons only.

### GP2-13 Engine and trainer docs carry the new namespace (Appendix A)

### GP2-14 Version string refresh (known now)

The build records the commit at configure time only (MEASURED: a stale hash after a rebuild). Fix: regenerate `version.h` at build time from `git rev-parse` (CMake custom target that always runs, writes only on change). Test: build, commit, build again; `pelias version` prints the new hash.

---

## Part 3. Finishing the build

Order: each step's gate closes it. A failed gate stops that branch and is logged. Steps follow masterwork 15.2, with the work items found today inserted where they belong.

The steady routine applies to every step:
- Code in the sandbox, unit tests, gcc/clang/asan, oracle, privacy scan, dependency check.
- Commit; push only on the author's word; bundle and backup.
- The author runs anything that touches the GPU or the real model.

### GP3-01 `pelias quantize --recipe FILE [--imatrix F]` (task 86, needed by everything after)

**Build:** per-tensor types from a recipe TSV, with the weighted quantizer (oracle-checked) when an imatrix is given. GGUF metadata keys from GP2-09.

**Gate:**
1. T8 built by Pelias equals T8 built by llama-quantize from the same recipe, byte for byte (`pelias verify`, exit 0), with and without Unsloth's imatrix. The first run is by the author: 2 × ~4 min.
2. The imatrix name map handles the expert slices (READ-SRC: imatrix + (row / rows-per-expert) × n_per_row).

**Status (2026-10-28 00:5x, d48b88d): BUILT.** `--recipe` and `--imatrix` are in; byte-identical to llama-quantize at 2ca15f5 on the tiny model and the 753-tensor synthetic model with a mixed recipe and an imatrix (unit test pins the hash); the expert slices use imatrix + (row / ne1) × ne0; rotated tensors require a matrix collected rotated. Gate 1 on the real model: the R2 rebuild through `--recipe` was byte-identical (Block L check, 01:16); the T8 comparison against llama-quantize remains for the author's next machine slot.

### GP3-02 Per-expert types, "cold experts" (task 3)

**Inputs:** `--expert-stats` from stock and T8 on both evaluation texts. Each Pelias run already writes this CSV when asked. Add `--expert-stats` to the GP1-04/05 runs at no time cost.

**Format:**
- GGUF holds one type per tensor. Per-expert types need the expert tensors split into a hot and a cold tensor plus an index map. This is a Pelias-only GGUF extension; llama.cpp cannot read it (masterwork PEL-3 note).
- Recorded as DECISION (proposed) before code. The author rules on the format.

**Gate:**
- The reference reads the split file and gives identical logits to the unsplit file when both halves keep the same type (KLD 0.000000).
- Then a cold-at-lower-bits variant is measured paired against its parent.

### GP3-03 AWQ-style scale folding (old name L1)

**Exactness first (C, sandbox):**
- On the tiny model, fold per-channel scales s into `attn_norm` / `post_attention_norm`. Divide the matching input columns of every consumer by s: the q/k/v/qkv/gate projections; the F32 router `ffn_gate_inp` (exact, F32); `ffn_gate_inp_shexp`; the shared and routed experts' gate/up.
- **Gate:** logits identical within fp32 rounding (KLD ≤ 1e-6 on the tiny base) before any quantization.
- Gated `ssm_norm` and the +1 convention are handled explicitly or excluded.

**Search (A+C):**
- Per layer and input space, s = mean|x|^α with α on a grid {0, 0.1, …, 1}, using `in_sum` / `in_sum2` from the imatrix. No new calibration pass is needed.
- Objective: the output error of the quantized weight, weighted by activations (the Gram from D-T7 if available, else the diagonal).

**Gate:**
- The folded T8 (quantized with Pelias, imatrix-weighted) beats T8 + imatrix on a paired comparison on both evaluation texts.
- Requires the author's ruling on D9/D33 (calibration on the base).

### GP3-04 GPTQ-style error feedback inside the K-quant blocks (old name L2)

**Data:** the Gram matrices from D-T7 for the dense groups first (`ssm_out`, `attn_output`, `attn_gate`, `shexp`; the groups the sweep ranks highest).

**Method:**
- Quantize columns in order. Spread each column's rounding error onto the not-yet-quantized columns through the inverse-Hessian update, keeping the K-quant block structure: per-block scales are found first, then values are re-rounded with feedback.
- The result is a valid Q4_K/Q6_K/Q8_0 block; any reader decodes it.

**Gate:**
1. The file is readable by llama.cpp and by the reference.
2. A paired win over the same recipe without feedback, on both texts.

**Ruling needed:** it breaks "byte-identical to ggml's quantizer" for these tensors by design. Proposed: byte identity stays the rule for stock types under stock names; PEL types are defined by Pelias's own quantizer, with their own golden hashes.

### GP3-05 Remaining step 4: chat-template renderer (P-D2) and data pipeline

**Renderer:** the Qwen template subset (158 lines, 8,057 bytes, MEASURED census), rendered natively.

**Gate:**
- P-D2: every assistant turn of the author's dataset and of a public set renders byte-identical to the Hugging Face Jinja render. Zero mismatches, including `tojson` formatting and the think-tag trap (82 of 3,000 turns, scaffold).
- Data audits A-LEN, A-TOK, A-TPL, A-MASK, A-DUP, A-SPECIAL, A-FAMILY with their rules (scaffold).
- The dataset port rebuild (GP2-08) happens here.

### GP3-06 Routing-preserving weighting (only if GP1-06 says flips carry the tail)

The imatrix or the GPTQ objective adds weight to the router's input space and to the experts whose margins are small.

**Gate:** a paired win plus a measured drop in the flip share (route-compare).

**Status (2026-10-28): NOT TRIGGERED.** block_k showed flips do not carry the drift (swapped weight vs KL correlation -0.004); this step stays parked.

### GP3-07 Decode loop and sampling on the CPU reference (task 4)

**Build:**
- KV cache and DeltaNet state carried across steps (the reference today runs one sequence from position 0).
- Greedy, temperature, top-k, top-p, min-p, and the penalties of the dictionary.
- A seeded RNG (counter-based Philox; dictionary seed default 20261019).

**Gate:**
- Greedy decode on 20 fixed prompts equals llama.cpp's greedy decode token for token for 128 tokens. Run llama.cpp CPU-only with `--no-op-offload`, and the reference with `--act-int8 all --act-int8-path cpu`, so the arithmetic matches.
- Without matched rounding, the first differing position is reported, not failed.
- Incremental decode equals the full re-run of the same sequence: logits within 1e-5 relative.

**Status (2026-10-29):** built. Incremental = full re-run: passed in the sandbox, bit-identical (max |difference| 0 of 55.4, MEASURED; `decode_incremental_equals_the_fresh_pass`). Sampler and Philox known answers: passed. Greedy vs llama.cpp: prompts and comparison tool in `tools/gates/` and `tools/greedy_compare.py`, llama-server with f32 K/V and flash attention off so the attention arithmetic is fp32 on both sides; run pending.

### GP3-08 MTP head in the reference (O18; needed by the engine)

**Build:** blk.40 run after the trunk as the model's graph does (READ-SRC llama.cpp at 2ca15f5).

**Gate:**
- The draft logits equal llama.cpp's MTP path on the tiny model within the float tolerance.
- Greedy drafted decode equals plain greedy decode (masterwork 15.2 step 10).

### GP3-09 Shared blocks, `arch_dense` and a small test model (task 5)

**Build:** the trunk split into reusable blocks (attention, DeltaNet, MoE, dense FFN). A dense architecture module. A tiny random dense model in the sandbox, plus one real small dense model the author supplies (never downloaded by the program).

**Gate:** both match llama.cpp end to end on F32 and on a quantized file (as Milestone 1).

### GP3-10 Fast CPU kernels, then GPU kernels (task 6 = masterwork step 5)

**CPU:**
- AVX2 dot products for Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS against fp32 and int8 activations, following the GP2-05 policy.
- Threadpool with 1-thread-per-core pinning (E-A1 A/B).
- **Gate:** outputs within the rung-1 tolerances against the reference (16.2). Throughput at or above llama.cpp's CPU expert path (about 27 GB/s effective, DERIVED, E4).
- **Status (2026-10-30, MEASURED on the PC, chain-3, c79229d):** decode (1 token), 16 threads, float / int8 GB/s: Q8_0 35 / 39, Q6_K 38 / 36 (41.7 int8 at 8 threads), Q5_K 31 / 27, Q4_K 22 / 28, IQ4_XS 24 / 29, Q4_TG2 17 / 17, IQ2_XXS 10 / 11; RAM 40-42; reference path Q4_K 8.5. Gate met by Q8_0, Q6_K, Q5_K, Q4_K (int8), IQ4_XS (int8); open for Q4_TG2 and IQ2_XXS (decode-bound: a faster grid lookup is the route to try). MTP block on the real file, reference vs fast: 9.1e-6, top-1 13/13. kld (255-token batches) was not faster (116 vs 117 s): rows were decoded once per four tokens. Fix built (cdfdb72): `kern::dot_row_n` / `dot_row_q8_n` decode a row once per batch, same bits as groups of four; `PELIAS_PROF=1` phase timer. Next: gp/block_cpukern_v3.sh (waiting for the okay) settles where the batch time goes; if the products (not the decode) bound a batch, a blocked int8 GEMM for CPU prefill is the follow-up (prefill normally runs on the GPU).

**GPU (sm_89):**
- Dequant-in-register GEMV/GEMM for the same types, RMSNorm, IMRoPE, attention, DeltaNet chunked and recurrent, router, top-k.
- **Gate:** rung 1 (fp64 reference, zero compute-sanitizer errors) and kernel parity. The author runs the GPU tests.

**Status (2026-10-30 15:3x, sandbox; the PC run is next):** first cut built. `kern/gpu_decode.h`: one decoder per 32-value sub-block for F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS, Q8_Z, Q7_ZT, Q4_TG2, compiled for host and device; MEASURED (sandbox): every sub-block equal to quant::dequantize_row bit for bit (11 types x 3 rows; every half value). `kern/gpu_ops.h` + `kern/cuda/gpu_ops.cu`: products straight from GGUF storage (one warp per row, up to 8 tokens per decoded sub-block), grouped experts in one launch (CSR slots, unequal counts), row gather, the gpu256x8 activation rounding (plain or rotated; the reference's expressions, unfused), segmented RMS / L2 norm, IMRoPE / NEOX / adjacent rope (the reference's angle recurrence), silu * mul, row adds, router softmax + top-k (ties by lower index), attention (scores, softmax, gated output; position-major cache), DeltaNet conv + tail, gates, the recurrence (state in shared memory, the reference's order) and the gated norm. Host builds get stubs (`gpu_ops_none.cpp`). tests/gpu/test_gpu_rung1.cpp: every kernel vs fp64 with the edge shapes below. Checked in the sandbox (no CUDA toolkit there: PyPI and apt are blocked): clang type-check of the .cu, host and device sides, 0 warnings (tools/cuda_check, test-only stand-in headers); the test compiles with the strict host flags; 86 unit tests pass on gcc, clang, gcc-asan/ubsan. PC script gp/block_gpukern_v1.sh (build / test / sanitize). **Rung 1 on the RTX 4070, WSL (2026-10-30 15:32-15:33, MEASURED, gp/block_gpukern_v1b.sh chain, logs GPUK/*_20261030_1532*-1533*):** nvcc 13.0 build clean with warnings as errors (34 s); pelias_gpu_test 11 tests, 0 failures: products 11 types x n_tok 1/3/9 worst rel L2 1.347e-07 (gate 1e-6); grouped experts (Q4_TG2 / Q5_K / Q6_K, edge counts and all on one expert) 1.155e-07 to 1.175e-07; activation rounding plain and rotated bit for bit; gather exact; norms 2.5e-08 to 7.4e-08; rope max 6.9e-08 to 8.7e-08 (gate 2e-5, incl. position 200000); router picks identical, weights 7.1e-08; attention 6.8e-08 / 1.43e-07 (gate 5e-4); DeltaNet 40 tokens 2.713e-07 (gate 1e-4), 25 + 15 = 40 bit for bit; host tests of the GPU build 86 / 0 failures; compute-sanitizer memcheck, racecheck, initcheck, synccheck: 0 errors each. All predictions met. Not yet: the Windows (MSVC + nvcc) build of the kernels. **All 32 readable types on the GPU (author 15:36 "Before [R3]. Build gpu decoders for each version that supports it"; sandbox 15:4x):** decoders added for Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4, NVFP4, Q2_K, Q3_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M, TQ1_0, TQ2_0, I2_S (BitNet), Q7_Z, Q4_TG (v1), with the format tables (Q4_TG grids, IQ grids, signs; 34,432 bytes) uploaded once (kern::gpu::Tables). MEASURED (sandbox): every sub-block of all 32 types equal to quant::dequantize_row bit for bit (3 rows of 2048 each; read-only types from random bytes with finite scales: tests/unit/gpu_test_rows.h); 86 unit tests pass on gcc, clang, gcc-asan/ubsan. FP8: no GGUF FP8 type exists; FP8 files reach the engine through the converter as BF16 / F16 / Q8_0 (covered); a native FP8 safetensors reader stays open (GP3-11). Rung 1 now runs the product on all 32 types and the grouped-expert / gather checks on the new block sizes (64, 128 values). PC run: gp/block_gpukern_v2.sh. **MEASURED on the RTX 4070, WSL (2026-10-30 15:56-15:57, v2 chain, logs GPUK/*_20261030_1557*):** nvcc build clean (34 s); products on all 32 types within 1.153e-07 (TQ1_0) to 1.793e-07 (Q5_1), gate 1e-6; grouped experts on 7 types 1.007e-07 to 1.179e-07 (edge counts and all on one expert); gather on 8 types bit for bit; the rest as v1b; host tests of the GPU build 86 / 0 failures; compute-sanitizer x4 0 errors. All predictions met. **Windows GPU build PROVEN (2026-10-30 16:02-16:13, MEASURED, gp/block_wingpu_v1b.sh build + v1d rebuild / test / sanitize, logs WIN/gpu*_20261030_16*):** Visual Studio 18 2026 + CUDA 13.4 (nvcc 13.4.92, MSVC 19.51), sm_89: MSBuild 0 warnings, 0 errors with /WX and nvcc --Werror=all-warnings; pelias_gpu_test.exe 11 / 11 with every per-type error equal to the WSL run's printed digits (32 types 1.153e-07-1.793e-07); host tests on the Windows GPU build 86 / 0 failures / 3 skipped; compute-sanitizer memcheck on Windows 0 errors. Script lessons: the toolkit path has to be passed from WSL (-T cuda=<path>, CUDA_PATH via WSLENV); a Windows build launched in the background from WSL loses its console when the launching call returns (run Windows steps in the foreground). Next: R3 streaming kernel into src. **R3 moved into src (2026-10-30 16:2x-16:4x, sandbox; author 16:21 okay):** kern/gpu_stream.h + kern/cuda/gpu_stream.cu: the probe's measured kernel math unchanged (grid path 2, route-2 lookup, item-7 per-expert counters, counter barriers between stages), behind a host API (StreamWeights: upload repacks Q7_ZT / Q6_K once; StreamPlan: a token's table built once, run(), run_unstaged()); the plan refuses tables that could deadlock (a wait on a group signalled later or never), misaligned activations and uncovered stages. tests/gpu/test_gpu_stream.cpp: one table with every format and lane count (k 256-4096), 3 stages, two experts' gate/up signalling and down waiting, vs fp64 of the same rounded activations (gate 1e-6); epoch-2 run and unstaged run bit for bit equal. Sandbox: clang CUDA type-check 0 warnings (shim gained cuda_fp16.h and the cooperative-launch declarations, test-only); host tests 86 / 0 on gcc, clang, asan; module check passed. **MEASURED on the RTX 4070, WSL (2026-10-30 17:21, gp/block_gpukern_v3.sh chain, logs GPUK/*_20261030_1721*):** build clean (36 s); the R3 stream test: 10 jobs, 3 stages, 2 expert groups, every job 5.883e-08 to 8.969e-08 relative L2 vs fp64 (gate 1e-6; Q7_ZT 8.271e-08 / 5.883e-08, Q6_K 7.866e-08 / 7.653e-08, Q5_K 6.357e-08 at 2 lanes, Q4_TG2 8.969e-08 at 32 lanes, the waiting down jobs 7.663e-08 / 7.884e-08); epoch-2 and unstaged runs bit for bit equal; unsafe tables refused; the rest as v2 (32 types worst 1.793e-07); pelias_gpu_test 13 / 13; host tests 86 / 0; compute-sanitizer memcheck / racecheck / initcheck / synccheck 0 errors, no timeouts. All predictions met. **Chunked Gated DeltaNet (2026-10-30 17:35-17:44, MEASURED, gp/block_gpukern_v4b.sh chain, logs GPUK/*_20261030_1735*):** accuracy vs fp64: 40 tokens chunked 2.618e-07 (25 + 15: 2.602e-07), recurrent 2.713e-07; 1,000 tokens chunked 2.690e-07, recurrent 2.759e-07, recurrent 300 then chunked 700 2.703e-07 (gate 1e-4; predicted 3e-7 to 1e-6: better); chunked vs recurrent 3.2e-07; 15 / 15; host 86 / 0; compute-sanitizer 0 errors (racecheck 494 s). **Speed missed:** 1,000 tokens x 32 heads, recurrence alone, warm: recurrent 5.094 ms, chunked 5.390 ms (0.94x; predicted faster). Reading (not yet measured): both kernels run one block per value head = 32 blocks on 46 SMs; the chunked kernel also calls expf per state entry per chunk. Next (author 17:39: look at the kernel outside the box, no textbook tunnel vision; 17:42: think a slowdown out before assuming the fix): variants measured side by side in one build: (1) the state in registers, 4 threads per row (decode); (2) each head split over 4 blocks (128 blocks), both kernels; (3) chunked with the decays hoisted and 32-token chunks; timed at 1 token and 1,000 tokens; the fastest kept per case. Later angle: the decode DeltaNet step as a job type inside R3's streaming kernel. Then the rest of GP3-10 GPU ( Q8_0 / Q4_K / IQ4_XS on the int8 path, O13 down projection, O14 real routing counts), then GP3-12 inference rungs. Still to come in GP3-10 GPU: the R3 streaming kernel moved in (its int8 path for Q7_ZT / Q4_TG2 / Q5_K / Q6_K), the chunked DeltaNet for long prompts, Q8_0 / Q4_K / IQ4_XS on the int8 path.

**Rung 1 edge shapes (universal build master):**
- Expert token counts M_e ∈ {0, 1, 7, 8, 63, 64, 65}.
- All tokens routed to one expert; zero tokens to the others. This is the negative-ID / hole crash class seen in the tier build.
- Element-wise kernels: 0 NaN/Inf on zero, denormal and outlier inputs.
- compute-sanitizer memcheck, racecheck, initcheck and synccheck all at 0.

**Routes toward the ceilings on today's misses (author 17:46-17:47 "yes"; each gets a prediction before its run; measure the slowdown before assuming the fix):**
- C1 R3's remaining 0.93 ms (5.685 ms real token vs 4.75 ms streaming; drain / ramp at 120 stage barriers): put the small steps (norms, router, the decode DeltaNet step) inside R3's job table as job types, so a token is one persistent launch with counter waits; would replace the one-CUDA-graph-per-step plan (U8, up to 3.2 ms, ESTIMATE). Same on WSL and Windows. Test: real-token time vs today's 5.685 ms.
- C2 chunked DeltaNet 0.94x of recurrent (MEASURED 17:36): variants side by side (state in registers, head split over 4 blocks, hoisted decays, 32-token chunks), 1 token and 1,000 tokens. **MEASURED 18:44 (gp/block_gpukern_v5.sh, logs GPUK/*_20261030_1844*):** 1,000-token prompt, warm / decode per token: recurrent 5.085 ms / 51.36 us; chunked 5.402 ms / 37.32 us; **recurrent with the state in registers 2.181 ms (2.33x) / 31.21 us**; **chunked rows, 16-token chunks 3.125 ms / 29.30 us (best decode)**; chunked rows, 32-token chunks 5.573 ms / 44.64 us (a 66 KB block fits once per SM vs twice at 16). Accuracy vs fp64 2.06e-07 to 3.07e-07 for all five, the five in turn on one state 2.674e-07; 16 / 16; sanitizers 0 errors (racecheck 640 s). Predictions: accuracy met; registers 1.5-3 ms met; chunked rows 1.5-3 ms (3.125: just over); 32 faster than 16 MISSED; registers best decode MISSED (chunked rows C16 29.3 vs 31.2 us). Reading (not yet measured): the decode figures of kernels 0, 1, 3, 4 include host work per call (device query and shared-memory opt-in every call); do it once and re-time. Author 18:47-18:48: stacked rows / stacked planes keep winning (CPU row blocks, int4 core vs free planes, R3, grouped experts, now DeltaNet); size a stack to the on-chip memory. Next: cache the per-call setup, re-time decode; the engine picks registers for long prompts (until re-timed) and keeps all five callable. **Re-timed 19:01 (gp/block_gpukern_v6.sh, 30c78ab, setup once per kernel; MEASURED):** decode per token recurrent 51.01, chunked 37.26, chunked rows C16 29.55, C32 44.57 us (unchanged: the per-call setup was not their cost; prediction missed); recurrent, registers 13.39 us (was 31.21) and prompt 2.014 ms (was 2.181), with no change to that kernel: unexplained (run-to-run variance or clock state suspected); next run times each kernel several times in one run to measure the spread. Accuracy digits identical to v5; 16 / 16; sanitizers 0 errors (racecheck 635 s). **Windows at 30c78ab (19:15-19:20, gp/block_wingpu_v2c.sh, MEASURED):** clean build, 16 / 16, digits identical to WSL, host 86 / 0, memcheck 0 errors; registers 2.141 ms / 11.97 us, C16 4.058 ms / 28.18 us (WSL 3.12 ms: open), C32 5.223 / 39.78, recurrent 5.058 / 50.63, chunked 5.370 / 37.46. Power logging started (HWiNFO64 CSV, ~2 s samples in practice; GPU idles at 36 W with the clock held at 2,490 MHz); kernel power needs the 250 ms nvidia-smi sampler and multi-second loops. **v7 spread + power (19:32-19:35, MEASURED):** 5 interleaved rounds: registers 2.040 ms prompt / 11.89 us decode median (decode max / min 1.46x; v5's 31.2 us not reproduced: the outlier), C16 3.116 ms / 30.66 us, chunked 4.866 / 33.92, recurrent 4.626 / 46.97, C32 5.586 / 44.59; GPU board power in 5 s loops 64.7-119.1 W (registers prompt highest, yet lowest energy: 0.239 J per 1,000 tokens, 1.26 mJ per decoded token, DERIVED); the card ran CUDA in P2 (memory 11,251 vs 11,501 MHz at idle P0) despite Force P2 off. Choice for the engine (until a later run changes it): registers for decode and prompt; C16 kept. Open: registers decode in a long loop 15.63 vs 11.89 us; P2 under WSL vs Windows. **R3 int8 path for the stock types done (v8b, 2026-10-30 19:48-20:00, c1784ed, MEASURED):** Q8_0, Q4_K, IQ4_XS stream on the card, 5.65e-08 to 8.82e-08 vs fp64 at 2-32 lanes per row; repeats bit for bit; 17 / 17; all four sanitizers 0 errors (racecheck 634 s). Registers DeltaNet decode moves 12-18 us between runs (single rounds to 41 us); prompt steady at 2.0 ms. Next in GP3-10 GPU: O13 (expert down K = 512, persistent next-tile, probe first), O14 (grouped launch with real routing counts).
- C3 the bus handoff flag: the graphics probe measured a VRAM flag at 0.8-1.4 us on Windows vs 2.40 us for CUDA's system-RAM flag; WSL2 cannot map that VRAM window, so the bus hand-offs may be fastest in the native Windows engine. Test: one A/B, same build, WSL vs Windows, once the engine runs.
- C4 clock state: checked 17:44 (nvidia-smi): CUDA runs in P2 with the memory overclock holding (11,251 MHz), so P2 does not throttle memory here. Recheck if a run's GB/s drops.
- C5 CPU decode about 4 tok/s on small models (suspected: thread start-up per product, not proven): a pool of spinning threads pinned one per core (the 5800X3D's 8 cores share its 96 MB L3; the author's L3 rule). Test: PELIAS_PROF before / after.
- C6 Q4_TG2 and IQ2_XXS CPU decode below 27 GB/s (grid lookup): keep each tag's 16 grid floats as 4 byte-planes and build the exact floats with 4 byte shuffles (pshufb) per 32 values; bit exact (no int8 rounding of the grid). Test: sandbox bit-for-bit vs dequantize_row, then bench-dot GB/s on the PC.

**Open note, check later (author 2026-10-30 18:50-18:54; after the first engine build):** a V100 32 GB test machine. CUDA 13 dropped Volta (sm_70) and driver R580 is the last branch supporting it (a maintained legacy branch); so: R580 + a CUDA 12.9 toolkit, in a second box or in dedicated sessions (R580 in this PC would also drive the 4070 and change the conditions behind every measurement so far). Pelias side: a CUDA 12 build configuration for sm_70 (the kernels use dp4a, cooperative launch, acquire / release, nanosleep: all Volta). What it could show (DERIVED, UNMEASURED): the 22.65 GB tiered file entirely in VRAM; HBM2 about 900 GB/s vs 504 GB/s, up to about 1.8x on R3's read-bound part; no int8 tensor cores (Turing+). Not an answer to the chunk-size question (about 96 KB shared memory per SM, like the 4070); that needs an H100-class card (chunks 16 / 32 / 64). Hardware note: many cheap 32 GB V100s are SXM2 modules (adapter board and cooling needed). Own-driver idea: weighed and parked (NVIDIA's open modules need GSP, Turing+; nouveau cannot reclock Volta without signed firmware and has no CUDA); R580 already covers it.

**Open kernel items to close here:**
- O13: the expert down projection (K = 512) runs at 0.27–0.29 of the int8 peak. Try a persistent next-tile kernel, probe first. **Probe 2026-10-30 20:08 (MEASURED, gp/block_o13_v1.sh):** the persistent v13 equals v12 bit for bit but runs 0.84-1.03x of v12 (best expert down 0.903x / 0.884x dense, 0.926x grouped): the persistent walk as written is slower. Next: registers / spills and one Nsight Compute launch of v12 vs v13 (expert down M 2048) before any change. **v13b (20:12, MEASURED):** index divisions out of the loop: back to v12 (expert down 1.001x / 0.973x dense, 0.927x grouped; registers 255 vs 249-254, 16 spill bytes), still bit for bit. Fill / drain is not the cost (hiding it gains about 0): the persistent route is closed for O13; the limit stays instruction issue per mma (10-26). Next on the list: O14.
- O14: the grouped launch with real, unequal routing counts. **D slice 2026-10-30 21:05 (MEASURED, gp/block_moe_v1.sh, real routing):** a work list over the experts that have rows (each padded to 64) is 1.15-5.9x faster than the padded grouped launch at 255 tokens and 4.0-12.5x at 2,040 (one expert can take 1,347 of 16,320 picks), bit for bit equal; a whole MoE slice (fused norm + per-32 rounding, gather, gate|up, silu + rounding, down, combine) 3.75-3.89 ms at 2,040 tokens per layer, per-32 costs 2-8 % of the slice (dense GEMMs alone 15-26 %, grouped experts 1.5-3 %: B). Adopted for the engine (proposed): per-32 activation format everywhere (accuracy 1.29x lower error vs the bf16 path; expert cost near zero), work-list expert GEMMs, fused norm. Next: fold the down output into the combine (Fable D3), device-built work list, 16-row padding. **D v2 21:28 (MEASURED):** 16-row tails slower at 2,040 tokens (0.88-0.98x, each tail tile re-reads its weight tile); warp-vector dp4a path for experts with 1-8 rows 1.27-1.37x of the whole slice at 255 tokens; down folded into the combine: fp32 reductions 1.13-1.14x, fixed-point int32 1.08-1.09x and bit-for-bit deterministic, int64 slower (0.86-0.90x), at 2,040 tokens. Next: S3 (64-row tiles for 9+ rows, warp-vector below, its row limit swept 8-32) + int32 fold. **GPU-built work list (author 21:27, for the GPU scheduler):** in the router epilogue, __ballot_sync + __popc for active-expert compaction and empty-expert culling in registers, a masked popcount for each lane's offset, shifts to split each expert's count into 64-row bodies and the small-expert vector units; one warp builds the whole mixed work list on the device (no host round trip, 40 layers per prompt). **S3 / S3b 21:34-21:40 (MEASURED, all checks passed):** the popcount list built on the GPU matches the host's, 0.035-0.13 ms in-stream vs 0.11-0.45 ms host build + copy + sync; no padding; 64-row tiles above 8 rows with the column tile in blockIdx.x; warp-vector 1-8 rows; int32 folded combine (deterministic): whole slice 1.30-1.39x of D v1 at 255 tokens, 1.11-1.15x at 2,040 (3.99-4.11 -> 3.50-3.71 ms). **LOCKED (author 21:44, DECISIONS_LOG):** the five steps are the engine's MoE slice; wired into the forward pass in GP3-12, built in GP3-14.
- O15: the two untested correct doorbell ordering forms.

### GP3-11 FP8 safetensors, then NVFP4 (task 7)

**Build:** readers and dequantizers for FP8 (e4m3/e5m2) safetensors and NVFP4 (scale layout READ-SRC from the published configs).

**Gate:** a small published file in each format decodes equal to its reference library's dequantization (oracle only).

### GP3-12 Masterwork steps 6–9: verify harness, one layer to full forward on the GPU, backward + LoRA + Mode Q, training loop + export

The gates are the ladder rungs 2–7 (masterwork 16.1), with the K-gates of GP2-01:
- 79 fetches per step;
- zero allocations in the step loop;
- the P-X1 hash proof;
- probes P-K4 to P-K13, P-Q2, P-Q5, P-Q7–P-Q9, P-Q11, P-S7, P-T1–P-T13, P-D2–P-D6, P-X1–P-X5, P-M1, P-M2, P-M5, at the stages the scaffold names (T3–T12).

### GP3-12b The trainer's central claims, tested (The Accidental Find 14.1 and 15.2; 03 blueprint H1, H2, H8)

| Test | What | Pass / falsified if |
|---|---|---|
| Parity on a small MoE (bp-H1, 15.2) | LoRA against a frozen quantized base and bf16 LoRA, on a small MoE that fits both ways; 3 seeds each | Loss curves within seed noise, and a downstream score within noise. Falsified by a gap larger than seed noise |
| Train/serve gap (bp-H2, 14.1-c) | KL of the exported file against the bf16 merge of the same adapter, on held-out text | Reported with a paired interval. In the new design, Mode Q exports the served bytes directly. This test is the check that the design closed the gap |
| Quantized-trained against bf16-trained adapter on the served file (bp-H8) | Both adapters judged on the served quantized model | The quantized-trained adapter is no worse (paired) |
| General ability, long context, identity, tool calls (14.1-d to g) | Base against tuned: an ability battery; retrieval at 16 k, 32 k, 64 k, 128 k; identity with thinking on and off, no system prompt and other clients' prompts; tool-call format through the serving stack | Reported; no regression beyond the paired interval |
| Peak logging | The true maximum over every micro-batch of a step, not the last micro-batch (the v0.2 lesson: a run peaked 0.39 GiB above its smoke test) | The governor logs the step maximum |

### GP3-13 Repair training against a bf16 teacher (old name L4; right after GP2-03's spec corrections)

**Teacher:**
- Pelias CPU reference logits of the bf16 file on a calibration text: top-256 plus lumped tail per position, stored once.
- Time unmeasured; GP1-06 gives the per-chunk rate of the bf16 reference.

**Student and objective:**
- The PEL file in Mode Q: core LoRA through the exact quantizer, plus optional norms and router. Experts frozen at their served bytes.
- Objective: KL(teacher ‖ student) on positions of the calibration text.

**Gate:**
- A paired win against the untrained PEL file on both evaluation texts (never the calibration text).
- No regression on the domain text.
- P-X1 hash proof on every frozen tensor.

### GP3-13b Idle units and the idle-work scheduler (author's idea, 2026-10-28 01:1x-01:25)

**Background (MEASURED):** during the dual-bus power test the GPU drew 45-47 W while reading RAM and at most 107 W in the soak (limit 200 W); the CPU stayed at or under about 58 W (PPT 142 W). The int8 kernel is issue-bound (13.4 instructions per mma, tensor busy 37.7 %). The power and the units are there; they wait on data.

**Idle units and the measured bottleneck each could relieve (each is tested alone, tokens/s with and without, paired):**

| Unit | Job | Bottleneck it attacks | How measured |
|---|---|---|---|
| Copy engines (DMA) | Prefetch next-layer cold experts from the pre-gating prediction (83.6 % of picks, 67.1 % of cold picks, block_k) | PCIe wait in the cold step | Nsight Systems: copy/compute overlap; cold-step time |
| Regular CUDA cores (warp specialization) | Unpacking warps feed the tensor-core warps in a producer/consumer pipeline (FlashAttention-3's pattern, PUBLISHED) | 13.4 instructions per mma | ncu: instructions per mma, tensor busy % (R3) |
| Texture units | Hardware format conversion of 8-bit data on the read path | Same unpack cost (alternative) | ncu, same metrics |
| L2 (36 MB on the 4070) persistence | Pin router, norms and the current activations (cudaAccessPolicyWindow) | Repeated VRAM reads | ncu: L2 hit rate, DRAM bytes |
| CPU cores (headroom about 84 W) | Cold experts computed on the CPU at the same time as the GPU (dual bus) | Cold-step serial wait | Cold-step time, tokens/s |
| CPU L3 (96 MB) | One expert is about 1.8 MB at ~4.5 bpw (3 x 2048 x 512, DERIVED), so L3 holds about 50: read a cold expert from DDR4 once and use it for every token of a speculative / MTP batch | DDR4 bandwidth per token | Timed CPU kernel: GB/s effective vs batch size |
| Idle RAM / VRAM / L3 | Hold preloaded text: the KV cache of a known prefix (system prompt, documents in use), computed in idle time | Prefill time on repeated context | Time to first token with and without |
| GPU slack between steps | Draft tokens with the MTP block (blk.40, kept) for speculative decoding | One token per weight read | Accepted tokens per step, tokens/s |
| RT cores, optical flow, video units | No fitting job found (RT cores are 3-D only; Fable 10-27) | n/a | Kept as an idea: a naturally low-dimensional job (scheduling, a coarse prefilter); vector quantization with RT-core nearest-codeword search at build time only |

**RT-core prefetch map, tested 2026-10-28 01:3x (MEASURED offline on block_k's stock routing record, train on chunks 0-4, test on 5-9; 16 candidates per token-layer; script in the sandbox, `rt3d.py`):** experts placed in 3-D by a joint spectral embedding of their pick transitions (the data an RT-core nearest-neighbour search could serve) against the same transitions kept in full and against "the most popular 16".

| Task | Full transition map | 3-D map | Most popular |
|---|---|---|---|
| Next token, same layer | 53.1 % of picks (cold 1.9 %) | 32.6 % (cold 9.0 %) | 36.8 % (cold 0 %) |
| Next layer, same token | 63.7 % (cold 3.2 %) | 31.9 % (cold 14.7 %) | 37.2 % |
| Two layers ahead | 62.2 % (cold 2.5 %) | 29.3 % (cold 13.3 %) | 37.6 % |
| Next token, next layer | 50.3 % (cold 0.8 %) | 24.9 % (cold 7.6 %) | 37.2 % |

The exact pre-gating prediction gets 83.9 % of the next layer's picks with only 8 candidates (67.1 % of cold picks, block_k). Squeezing the routing into 3-D loses about half of what the full map knows and does worse than "most popular"; it reaches more cold picks than the full map but far fewer than pre-gating. **Verdict: the RT-core prefetch map is not worth building**; pre-gating on the tensor cores (one 2048x256 product per layer) stays the prefetch hint. Caveat: 1,275 training tokens and one embedding method; a larger record would not close a gap this size.

**The idle-work scheduler:**
1. Inventory at startup (the plan module already measures the machine): the units above and their sizes.
2. Live load per token: GPU timers, copy-engine queues, CPU core load.
3. A queue of optional jobs from the table, each with its measured payoff on this machine.
4. Hard rule: an optional job runs only on an idle unit and is always cancellable; it never delays the token being made.
5. Each job is kept only if its own paired test shows a gain on that machine (universal engine: other hardware fills its own idle units).

**Also noted (MEASURED):** int8 259.5 TOPS against bf16 65.2 TFLOPS is a 4x gap, where 2x would be expected; GeForce Ada runs 16-bit tensor math with 32-bit accumulation at half rate (NVIDIA Ada whitepaper, to be cited exactly before use). The int8 path, made free in accuracy by rotation (R1, R2), avoids that limit. Test: the same 16-bit GEMM with 16- and 32-bit accumulation, about 5 min.

**Research sweep on idle units (2026-10-28 01:4x; PUBLISHED unless marked; numbers are the authors', not ours):**

| Idea | Source | What they measured | Fit for this machine |
|---|---|---|---|
| Draft-driven expert prefetch: an INT4 draft of the model runs ahead and its router picks drive prefetch | MoE-SpeQ (arXiv 2511.14102) | Draft matches the full model's top-4 experts 90.9 % (vs 84.7 % for one-layer-ahead prediction); 2.5-3.3x vs Mixtral-Offloading, A100 + PCIe 4 | HIGH: our weak spot is the next token's cold experts (13.6 % repeat); MTP draft = our version |
| Speculative decoding to give each transferred expert more tokens, transfers overlapped with compute | SpecMoEOff (arXiv 2508.21706); SP-MoE (2510.10302) | 1.45-1.53x from speculation alone; 2.1x avg vs MoE-Lightning; tested on a 4090D | HIGH: the batch-per-expert idea, GPU side and the L3 side |
| Expert deferral: some experts' outputs merged two layers later (residual tolerates it) so CPU and GPU never wait on each other | KTransformers (SOSP 2025) | CPU use 74 -> 100 %, up to +45 % throughput, -0.5 % LiveBench | HIGH and testable now: a reference option, paired KLD tells the accuracy cost |
| Whole decode step in one CUDA Graph, CPU hooks via cudaLaunchHostFunc | KTransformers | up to 1.23x decode | HIGH: attacks the 5.8 ms per-step overhead of the E0 cold step |
| cp.async pipelining, `evict_first` L2 hint for single-use weights, loads several stages ahead | MARLIN (arXiv 2408.11743) | about 3.9x over FP16 at batch <= 16-32 (A10) | HIGH for R3: the in-SM copy units overlapping tensor-core math; the L2 hint is the "L2 for the right things" idea |
| Table lookup instead of multiply on the CPU (vpshufb, 16-entry tables) | T-MAC (arXiv 2407.00088) | 4-bit GEMV up to 3.1x vs llama.cpp single-thread; multi-thread limited by bandwidth | HIGH: our CPU kernel is at 13.31 of 51.2 GB/s, the regime T-MAC targets |
| Video codecs (NVENC/NVDEC) as tensor codecs | LLM.265 (MICRO 2025); torch-nvenc-compress; arXiv 2602.09725 | weights ~3 bits near FP16 accuracy, KV 2.9 bits; NVDEC about 1.3 GB/s; loses on PCIe Gen4, wins on slow links (Ethernet, Thunderbolt); training: activations 3.5 bits, gradients 10.1 bits in pipeline training | LOW here: the codec is slower than our PCIe bus; note for multi-machine training later |
| RT cores for MoE routing (experts as boxes, tokens as rays) | Spectral-AI (independent, OLMoE, RTX 5070 Ti) | router 48-89x faster but routing is 2.8 % of the step; HellaSwag 53.1 -> 52.0; needs a distilled router | LOW: our own 3-D test (above) agrees; RT review (arXiv 2603.28771): memory x9, FP32 only, switching overhead |
| ROPs / raster pipeline for sums; optical flow unit | patents, old GPGPU | no modern ML use found | NONE |
| GeForce 16-bit tensor math with 32-bit accumulate | NVIDIA forum (users, RTX 4090) | FP16/FP32-acc about 170 TFLOPS vs FP16/FP16-acc about 270; int8 about 560 TOPS; no NVIDIA statement | Consistent with our 4x int8/bf16 gap; our int8 path avoids it (correction: user-measured, not a whitepaper line) |

**Order proposed for the engine phase (each with its own paired or timed test):** ~~expert deferral~~ (ruled out by Block M a, +0.004213 for one expert) -> one CUDA Graph per step -> MARLIN-style pipelining in the int8 kernel (R3) -> T-MAC-style CPU kernel -> MTP drafting with draft-driven prefetch and batch-per-expert on GPU and L3.

### Block M: side track, the author's "language" (codes written at build time) and speed ideas measured for accuracy (2026-10-28)

**Rule:** sandbox-first, one cheap test per step, a gate decides build or park; never takes PC time from the main path.

**Reference options built (sandbox, not yet run on the real model; tiny model: all off = bit-identical to before):** `kld --act-bits 4` (W4A4 with rotation), `--neuron-skip F` (oracle neuron skipping, the upper bound of TEAL/PowerInfer-style byte savings), `--defer-experts D` (KTransformers' expert deferral), `--swap-near-tie M --hot-set FILE` (cache-aware near-tie swap; hot set from the engine's 7.5 GiB file, 328 prompts). Tiny-model drift (random weights, direction only): base 5.43, skip 25 % 5.57, skip 50 % 5.69, defer 1 6.98, 8-bit rotated 5.54, 4-bit rotated 7.96, 4-bit plain 8.83.

**Price list on the real model (block_m.sh, t8r = dense rotated, 100 chunks, each paired against t8r's fp32 run 0.012941; MEASURED, every result re-computed from the dumps in Python):**

| Knob | KLD | vs t8r (95 % interval) | Prediction | Verdict |
|---|---|---|---|---|
| Expert deferral 1 (`--defer-experts 1`) | 0.017154 | +0.004213 [+0.003624, +0.004802], t 14.2 | +0.0002 to +0.002: missed (about 2x the top) | ruled out as a default: one expert adds a third of t8r's drift |
| Expert deferral 2 | 0.023002 | +0.010061 [+0.009218, +0.010904], t 23.7 | 2-3x defer 1: met (2.39x) | ruled out |
| Near-tie swap, margin 0.001, 7.5 GiB hot set | 0.014606 | +0.001665 [+0.001102, +0.002228], t 5.9 | below +0.0005: missed | cheapest, still resolved; 302,248 swaps of 1,274,700 token-layers whose last pick was cold (23.7 %); a smaller margin is the only version worth a second look |
| Neuron skip 25 % (oracle: smallest hidden values of each routed expert zeroed) | 0.013526 | +0.000585 [+0.000196, +0.000974], t 3.0 | +0.0003 to +0.002: met | the only cheap knob; an upper bound (a real predictor picks worse) |
| Neuron skip 40 % | 0.017611 | +0.004670 [+0.004021, +0.005319], t 14.3 | +0.001 to +0.004: missed (just above) | as costly as one deferred expert |
| Neuron skip 50 % | 0.024159 | +0.011218 [+0.009803, +0.012634], t 15.7 | +0.002 to +0.008: missed | ruled out |
| 8-bit activations on every weight (gpu path) | 0.013106 | +0.000165 [-0.000148, +0.000478], t 1.1 | within about +0.0003: met | free; **vs Unsloth MTP 8-bit (0.012864): +0.000242 [-0.000198, +0.000682], a tie** (prediction "below Unsloth's 8-bit" missed) |
| **8-bit, every weight, in the int8 kernel's format** (`gpu256`: one scale per 256, Block M2 d) | 0.013939 | +0.000999 [+0.000705, +0.001293], t 6.7; vs per-32 +0.000834 [+0.000422, +0.001245] | +0.0003 to +0.0024: met | **not free at per-256**; Unsloth's MTP file at the same format 0.015665 (+0.003434 over its fp32): **t8r ahead by -0.001726 [-0.002359, -0.001092], t -5.4**, 167.6 MB smaller. Against Unsloth on llama.cpp's own GPU path (q8_1, 0.012864) t8r_g256 trails by 0.001075; per-32 scales in the kernel (Block H q8_k32) would bring t8r to its per-32 0.013106 (a tie with that) |
| One scale per 256, dense only / experts only (Block M3 e) | 0.013367 / 0.013320 | +0.000426 [+0.000084, +0.000768] / +0.000379 [+0.000060, +0.000699] | dense +0.0001..+0.0005 met; experts +0.0005..+0.0010 missed (lower) | **cost split about evenly** (sum +0.000805 vs all +0.000999): the fix covers every weight |
| 4-bit activations on the rotated dense weights | 0.045766 | +0.032825 [+0.031543, +0.034107], t 50.8 | +0.002 to +0.01: missed (3x the top) | ruled out: no int4-activation kernel path |

- **Format mismatch found 2026-10-28 08:4x (author's question):** every 8-bit run so far (GP1-03 b-d, R1, R2, Block N, Block M c) used q8_1, one fp16 scale per 32; the adopted int8 kernel (v10, and v7-v12) quantizes with one fp32 scale per 256 (`q8k_warp`). Stock at llama.cpp's Q8_K, also one scale per 256, cost +0.002383 (GP1-03 a). Reference option `--act-int8-path gpu256` added (value for value the kernel's format, unit test); Block M2 d (`block_m2.sh d`) measures t8r and Unsloth's MTP file at it. Not affected: deferral, swap, neuron skip (no activation rounding), the 4-bit test, every fp32 comparison. Rule added (docs/PELIAS_NEW_METHODS.md): a test that stands in for our code reads that code's exact format first.

- **Reading:** the routed experts are not interchangeable at the margin. The 8th pick carries about 7 % of the routed weight (block_k) but moving or swapping it costs 13-33 % of the whole file's drift. KTransformers' -0.5 % on LiveBench is a task score; per-token KLD is far more sensitive, and the speed knobs that touch routing must be judged on it.

**Gate 1, the 3-weight code (MEASURED 02:1x, sandbox, real bf16 weights of blk.20 extracted read-only from the author's file: attn_qkv 2048x8192 sampled to 4096 rows, 4 gate and 4 down experts; relative squared error, no importance weights):**

| Format | Bits per weight | attn_qkv rotated | attn_qkv plain | gate_exps | down_exps |
|---|---|---|---|---|---|
| Q4_K | 4.50 | 0.00502 | 0.00586 | 0.00515 | 0.00512 |
| IQ4_XS | 4.25 | 0.00582 | 0.00668 | 0.00594 | 0.00591 |
| 3-weight code, 4,096 words, fp16 scale per 32 | 4.50 | 0.00597 | 0.00632 | 0.00600 | 0.00588 |
| 3-weight code, 4,096 words, "tags" (fp16 per 256 + 6-bit per 32) | 4.25 | 0.00598 | 0.00636 | 0.00599 | 0.00586 |
| Triangle word + shade (256 triangles x 15 points) | 4.25 | 0.00902 | 0.00974 | 0.00920 | 0.00930 |
| Q5_K (reference) | 5.50 | 0.00129 | 0.00150 | 0.00132 | 0.00131 |

- **Verdict: gate 1 not passed.** A learned 3-weight code ties IQ4_XS at 4.25 bits (within -1 % to +3 %) and loses to Q4_K at 4.5; the triangle code loses by about 55 %. Three dimensions are too few for a codebook to pay (published gains come from 8 dimensions, QuIP#'s E8 lattice, and from importance-weighted fitting).
- **Kept:** the author's "tags" (a label per 256 plus a smaller label per 32, the block's "zip code") give the same error as a full fp16 scale per 32 for 0.25 fewer bits per weight. K-quants already use exactly this two-level addressing, which is why they are hard to beat.
- **Possible next step (parked unless the author asks):** an 8-weight code (E8-style) at 4.25 bits on the same slices, sandbox only, about 15 min; RT cores cannot serve 8 dimensions, so it would be a build-time tool on the CUDA cores.

**Split planes, and the definition of _P (author's principle, 2026-10-28 02:1x-02:20):**
- **_P = any Pelias format stored as a base plane (the top bits) plus refinement planes (the low bits), each plane in its own array.** With every plane present the values are bit-identical to the source format (same bits, different arrangement), so accuracy changes only when a plane is deliberately left out.
- **Three jobs, one layout:** (1) R3: the base plane laid out for a 1-2 instruction unpack into the tensor cores (today 13.4 instructions per mma); (2) anytime precision on the bus: a cold expert's base plane crosses PCIe first (copy engine) and the tensor cores start on it; the refinement follows and, the product being linear, its share (W_ref x) is computed and added when it arrives, or skipped for that token if it is late; (3) VRAM: a hot expert may keep only its base plane resident (Fable: Q5_P hot-bank gain about 0.52 GiB).
- **Odd widths fit best (author, 02:21):** an odd width splits into byte-aligned power-of-two planes with the top bits as the base: 3 = 2 + 1, 5 = 4 + 1 (base = a plain nibble), 7 = 4 + 2 + 1 (fallbacks at 4 and 6 bits); 1 bit is already a base. ggml's K-quants also split (Q3_K 2 + 1, Q5_K 4 + 1, Q6_K 4 + 2) but keep the separate bit on top (Fable: Q5_K's qh is the most significant bit), so their low part is not usable alone; _P orders the planes top-down. Odd widths then unpack like even ones and carry a built-in coarser-if-late fallback (Q5_P, Q7_P on attn_qkv: Fable's 168 MB).
- **Word-aligned packing with tag bits, naming `_PT` (author, 02:24-02:31):** weights packed so they never straddle a 16-bit word, the leftover bits carrying "zip code" tags: 7-bit 7 + 7 + 2 tag bits; 5-bit 5 + 5 + 5 + 1; 3-bit 5 x 3 + 1; 4-bit 4 x 4 (no tag). Unpack is one shift and one mask per weight; tags (a sub-scale nudge, a sign trick, a "refinement present" flag) ride free inside the word. Combines with planes: the word holds the base plane of several weights, refinement planes follow in their own words. **Names:** `Q<bits>_P` = planes; `Q<bits>_PT` = planes + tag bits; `PEL<n>_PT` = a PEL recipe built from them (e.g. Q5_P, Q5_PT, Q7_PT, PEL7_PT). Reference: Q8_0 is 8.5 bits per weight; a 7-bit format with an fp16 scale per 32 is 7.5, with K-quant-style tags about 7.25. **Test (sandbox, the blk.20 slices):** do the tag bits buy accuracy at equal bits per weight against Q6_K / Q8_0.
- **Planes by heat, naming `_PH` / `_PTH` (author, 2026-10-28 11:00):** the engine reads every plane of a hot expert (GPU bank) and only the base plane of a cold one (from RAM); when a cold expert turns hot its refinement plane is added as it moves into the bank. The file keeps every plane of every expert, so H is a read rule of the engine, not a file layout: a `_PT` file runs as `_PTH`. (Yes, pH: the format has a base plane.) Gate: the _P cold-expert run, then GP3-02.
- **Q7_PT reviewed before building (2026-10-28 12:1x; self-review + Fable, read-only):** (1) at the same 8.25 bpw an 8-bit value in the same zip layout beats 7 bits + 2 tag bits (MEASURED, blk.20 attn_qkv: rotated 0.000031 vs 0.000040, unrotated 0.000040 vs 0.000044; Q8_0 0.000028 / 0.000035; tools/experiments/blockm_z8_check.py): a value bit quarters the error, the tags cut it about 3x; this supersedes "Q7_PT about even per bit". (2) Q7_PT cannot feed the int8 kernel: the tag changes every 2 of the 32 k-values one mma reduces, and folding {5,4,3,2}/5 into the weights reaches 315 > 127 (only {1, 0.5} is int8-exact); the fp16 route gives up the int8 path's 1.39x-1.84x prefill gain. (3) The 9 x 7 + 1 word (7.36 bpw) shares one tag over 9 weights and lands at the no-tag error. **Passes: Q8_Z** (the author's zip scales with 8-bit values: fp16 d per 256, 6-bit level per 32, int8 values; 8.25 bpw, -59.7 MB vs Q8_0, int8-exact, no nibble unpack, no min term; planes 4 + 4 with a plain-nibble base for _PH; the unused level 0 is a free per-32 flag). Row lengths: all 251 Q8_0 tensors in the header are 2048 / 512 / 4096, multiples of 256. R3 caveat (Fable): with 8-bit weights and per-32 integer activation levels the int32 bound is 16.4 G > 2^31, so the activation level stays fp32 per mma (one FFMA). Tight Q7_P (7.25 bpw, int8-friendly, error 0.000118) stays as the bolder option.
- **Idle units for the slow parts (Fable, read-only, 2026-10-28 12:2x; author: "I like what I see on the texture cores"):** copy engines carry the cold reads and the _PH base-then-refinement two-copy queue; RT cores, L2, NVENC/NVDEC, SFU, int4 / FP8 mma and the CPU do not help the dense path (arithmetic in the review). The tag is a representation job: the build-time quantizer absorbs it (round(2mq) fits int8 = Q8_Z). **Texture units:** Q8_Z stored as RGBA8 snorm (4 weights per texel, scales in a side buffer, still 8.25 bpw) decodes at 2.04 TB/s (4x VRAM), so at decode the fetch is the memory read and the int8-to-float conversion is free; tokens/s stays read-bound (3.9 ms for 1.97 GB) but SM instructions drop about 10x, freeing the SMs for the MTP draft or attention. Prefill hinges on the fp16-accumulate mma rate: fp32 accumulate 0.64x v10; fp16 accumulate at full rate about 1.28x v10 (0.2 % partial-sum noise per 256, below resolution by the urm data: fp32 activations 0.012467 vs int8 0.012499). Checks: (a) sandbox: emulate fp16 partial sums per 256 in the reference's dense GEMM, paired run on urm; (b) bench_gpu, 5 min: fp16 vs fp32 accumulate mma rate; (c) bench_gpu: texture-stream GB/s vs a plain read. Both pass -> dense tensors as Q8_Z texels on an fp16-accumulate path; else Q8_Z on int8 v10. Also open: 7-bit weights with 4-bit activation scale levels (int32-exact in v10's IMAD), sandbox accuracy check.
- **File names (agreed 2026-10-28 11:23):** files are named by what they store, not by heat (H is the engine's read rule): **P7_PT_M** = the mixed recipe with Q7_PT dense and planed experts (runs as **P7_PTH_M**); **P7_P_M** = the tight version with Q7_P dense. Full-model scenario on urm's tensor sizes (DERIVED; assumes each format holds its accuracy, which Q4_PH does not on paper): P7_PT_M -60 MB file, about 0.6-1.1 ms per token saved; P7_P_M -299 MB, about 1.1-1.6 ms (dense read -0.12 / -0.6 ms; about 21 cold expert reads per token, each 1.90 -> 1.25 MB with base planes, -13.6 MB per token, 0.5-1.0 ms at 27 / 13.3 GB/s).
- **MTP block at Q3 (author, 11:25; parked until GP3-08):** blk.40 only drafts tokens and the main model checks every draft, so its bits cannot change the output, only the draft acceptance rate. Experts at Q3_K save about 140 MB (DERIVED, regular-layer shapes assumed). Test after GP3-08: draft acceptance at Q3 (or Q3_PH) against Unsloth's types, then tok/s in the engine. **Q3_PH for MTP, read by draft depth (author 13:52, DERIVED):** expected accepted tokens per step = a1 + a1·a2 + a1·a2·a3 + ..., so the first draft's acceptance multiplies every later one: draft 1 reads base + H, deeper drafts read the 3-bit base only. The batched verify of those drafts (M = 5-8) is where Fable placed the 1-bit mma. Test after GP3-08: acceptance per draft depth at Q3 base, Q3 + H and Unsloth's types. **7 vs Q3 for PEL7_P_(T)_H's MTP (2026-10-29, DERIVED from urm_ZT2's sizes, time ~ bytes):** per draft step 7-bit rotated reads ~470 MB (blk.40 dense 37.6 + 8 active experts 15.2 + output head ~417), Q3 ~445 MB; verify >= one main-model read ~2,456 MB. Q3 saves only 0.9 % of a cycle (1 draft) to 1.9 % (3 drafts). **Rule: stay at 7 unless Q3 keeps acceptance within ~1.5 points (1 draft, e.g. 75 % -> 73.5 %) or ~2 % of expected accepted tokens (3 drafts).** Q3_PH (H on draft 1) is the variant most likely to pass. Also test 7-bit rotated (eh_proj rotated, -49.5 % error). Bigger draft-speed lever (idea, untested): the ~417 MB output head per draft; a draft that scores only a short list of likely tokens would read far less.
- **The 7pH ladder (author, 2026-10-28 11:01):** Q7_P -> Q7_PT -> Q7_PH (and Q7_PTH, "the pH 7 test": neutral, the balance of accuracy and cost). **Goal:** Q7_PH becomes the Pelias standard for the tensors Q8_0 holds today, after deliberation, only if all three gates pass: (1) accuracy, a paired run against Q8_0 on the same tensors with no resolved difference; (2) cost, fewer bytes (MEASURED layout: Q7_PT as tested is 8.25 bits per weight, 7+7+2 tag bits per 16-bit word plus zip scales, about 3 % fewer than Q8_0's 8.5; a 64-bit word of 9 x 7 bits + 1 tag would be about 7.36, DERIVED, untested; the earlier 7.25 was a tight 7-bit layout without tag bits; sandbox error Q7_PT 0.000040 vs Q8_0 0.000028); (3) speed, faster in the engine (timed, GP3-14 / GP4). Until then Q8_0 stays the default.
- **_PT tag test (MEASURED 02:3x, sandbox, same blk.20 slices, relative squared error; scripts tools/experiments, pt.py / pt2.py in the sandbox):** with K-quant-style "zip" scales (fp16 per 256 + 6-bit per 32) and a scale search, using the spare word bits as per-word scale tags cuts error against leaving them unused by about 40 % (5-bit: 0.00189-0.00195 -> 0.00115-0.00117 at 5.75 bpw) and 67 % (7-bit: 0.000118-0.000122 -> 0.000040 at 8.25 bpw). Against existing formats per bit: Q5_PT (5.75 bpw, 0.00115) vs Q5_K (5.5 bpw, 0.00129) is about 11 % less error for 0.25 more bits, which per bit is still about 25 % behind K-quant packing (DERIVED from the Q5_K -> Q6_K slope, about 4x per bit); Q7_PT (8.25 bpw, 0.000040) vs Q8_0 (8.5 bpw, 0.000028) is about even per bit and 3 % smaller. **Reading:** the tags are a strong use of bits that word packing leaves over; word packing itself spends more bits than K-quant planes, so _PT's case is speed (one shift and mask per weight) and the fallback, with Q7_PT as a near-equal, slightly smaller stand-in for Q8_0. Accuracy alone does not favour _PT over K-quants.
- **Readers:** copy engines, tensor cores and regular CUDA cores (one shift-and-add per weight to merge). Not the RT cores: merging planes is arithmetic, not a search, and their tree costs about 9x the memory per number.
- **Gate (one paired run on the reference):** worst case, every cold expert (outside the engine's 7.5 GiB hot set) at its base plane only (Q4_K -> 3-bit base), against the full file. A small paired cost makes split planes a speed lever with no VRAM cost; the reference option is not built yet.

### Accuracy path agreed with the author (2026-10-28 10:48-10:58)

Likeliest candidates first, then stop and decide (keep the best file or keep exploring):
1. **urm** (running 10:54, block_urm2.sh): Unsloth's MTP types, dense rotated, merged matrix; fp32 and the engine format (gpu256x8).
2. **urm_PT** (urm-plus, kept by the author): attention at Q7_PT, the split planes plus tag bits; needs the Q7_PT quantizer (sandbox, written while urm runs).
3. **Hot/cold bits (author's idea, the "over/under" ledger):** hot experts full planes in VRAM, cold experts base plane only from RAM; a cold expert that turns hot gets its refinement plane added when it moves into the bank. Measured by the _P cold gate (reference option, sandbox) then GP3-02 per-expert types. Selection rule: accuracy gained per byte spent, per expert, filled to the VRAM and time budget.
- **Gating (author, 11:15):** the heat tests (Q5_PH on the down experts, Q4_PH on gate/up) run only if the Q7 family (urm_PT: Q7_PT, and Q7_P as the bolder option) holds Q8_0's accuracy at lower cost. On paper (sandbox, blk.20 slices; drift from the t8r / t9r anchors, linear): Q7_PT about +0.00002 at -3 % bytes; Q7_P about +0.0002 at -15 %; Q5_PH about +0.0002 to +0.0003 for -18 % cold reads (base plane error 0.0051-0.0086); Q4_PH about +0.004 to +0.007 for -44 % cold reads (2-bit base error 0.089-0.159, coarser than Q2_K): fails on paper, comes back only with a better base split. Script: scratchpad blk_m/ph_check.py.
- Weighed on paper and set aside: rotation on/off by hot/cold (rotation costs under 1 % of a cold read and gains nothing at per-32; it helps coarse formats instead); shedding 2 bits from active experts into L2 (about 256 MB per token against a 36 MB L2, or about 10 ms per token over PCIe, and the cost lands on 93.5 % of picks); RT cores for tracking (a table does it, Block M RT test).
- **GPU shortlist (author, 11:44):** the CPU reference picks the accuracy winners; the GPU engine tests (K3, speed) run only the best two or three files by their engine-format result (per-32, gpu256x8) against Unsloth on llama.cpp's GPU path (0.012864), at or under Unsloth's MTP size. fp32 results are background only (author, 11:41). Standings: **urm_ZT2 0.012536 (-0.000328 vs the bar, not resolved; -236.4 MB; MEASURED 15:47; gate 1 passed vs urm_Z)**; urm_ZT 0.012870 (-0.000006 vs the bar, exact tie, -236.4 MB; MEASURED 15:13; fix rerun urm_ZT2 running)**; urm_Q7 0.013524 (+0.000660, resolved worse, -305 MB); **urm_Z 0.012402 (-0.000462 [-0.001019, +0.000095], t -1.65, lower but not resolved; -61.0 MB under Unsloth MTP; MEASURED 2026-10-28 13:27)**; urm 0.012499 (-0.000365 [-0.000986, +0.000256], lower but not resolved; -6 KB); t8r 0.013106 (+0.000242 not resolved; -168 MB); splice and urm_PT not built. urm's engine format costs nothing over its fp32 (+0.000032); its experts are our rounding of Unsloth's types (fp32 +0.000236 vs Unsloth MTP, not resolved), so the splice (Unsloth's expert bytes, our rotated dense) is next: predicted about 0.01226, about -0.0006 vs the bar.
- Build-readiness track unchanged: step 2 benches, R3 kernel with per-32 activation scales, MTP head, engine.

### Next steps (author, 2026-10-28 12:35)

1. **DONE 2026-10-28 12:4x: Q8_Z built** (type id 200, src/quant/quant_cpu.cpp; level search 70-100 % per 32-block; unit test). MEASURED on the real blk.20 attn_qkv (every row, C++ quantizer, tools/experiments/q8z_real_check.cpp): rotated 0.000029 vs Q8_0 0.000028 (ratio 1.017), plain 0.000036 vs 0.000035 (1.019), at 8.25 against 8.5 bits per weight. Was: quantizer (zip scales with a scale search, int8 values), reference dequantizer, recipe name, Pelias-only type id, unit tests against the measured error (0.000031 rotated attn_qkv).
2. **Fable's check (a) (sandbox):** emulate fp16 partial sums per 256 flushed to fp32 in the reference's dense GEMM.
3. **bench_gpu additions:** (b) fp16 vs fp32 accumulate mma rate (5 min); (c) texture-stream GB/s vs a plain read.
4. **DONE 2026-10-28 13:27: urm_Z (MEASURED, verified from the dumps with pair.py).** urm with all 259 Q8_0 tensors as Q8_Z: 22,602,383,584 bytes, **-61,003,840 bytes (-61.0 MB) vs Unsloth MTP** (predicted about -60 MB). Engine format (gpu256x8) 0.012402: vs urm -0.000098 [-0.000453, +0.000258], t -0.54, no difference (predicted +0.00002, no difference: met; Q8_Z costs nothing measurable); vs Unsloth on llama.cpp's GPU path (0.012864) -0.000462 [-0.001019, +0.000095], t -1.65, lower but not resolved (predicted about -0.00035). **7pH ladder gate 1 for Q8_Z passed on the whole model: accuracy equal to Q8_0, 61 MB fewer bytes** (speed: engine). urm_Z is now first on the GPU shortlist.
5. **The engine handles P and H (author: "I want to see the engine handle P and H"):** in GP3-14 the engine reads _P files (a base plane plus refinement planes in their own arrays) and applies the H read rule (heat): hot experts in the GPU bank hold every plane; cold experts are read base plane first by the copy engines, the refinement plane queued second and applied if it arrives in time, skipped for that token if not; a cold expert promoted to the bank gets its refinement plane added as it moves. Dense tensors have no heat (read every token): Q8_Z through the texture path if checks (b) and (c) pass, else int8 v10. Engine gates for P/H: (i) with every plane read, logits identical to the unsplit file (KLD 0); (ii) base-only cold reads at the measured cold share, paired KLD against (i) (the _P cold gate, run on the reference first); (iii) tok/s and cold-path bytes per token, timed.

### S1 sandbox results (2026-10-28 13:5x-14:0x, MEASURED, real blk.20 attn_qkv, rotated, all 8192 rows; tools/experiments/s1_rungs_check.py)

| Format | bpw | Importance-weighted error | x Q8_0 |
|---|---|---|---|
| Q8_0 (C++) | 8.50 | 0.000028 | 1.00 |
| Q8_Z (C++) | 8.25 | 0.000029 | 1.02 |
| **Q7_Z (C++, new type id 201, commit 4a34c8d)** | **7.25** | **0.000113** | **4.0** |
| 6-bit, same scales (numpy) | 6.25 | 0.000457 | 16.3 |
| Q7_PT (7 + 7 + 2 tag bits, earlier) | 8.25 | 0.000040 plain | beaten by Q8_Z at the same bits; bit planes do not change values |

- Exactness: J1 plane split (base nibble x 16 + low nibble) = whole dot, max difference 0; author's popcount H = plain dot, max difference 0.
- fp16 partial sums per 256 (texture path, Fable check a): 0.22 % output error against Q8_Z's own 0.54 % on the same outputs (synthetic activations scaled by the importance matrix, rotated).
- Whole-model test started 14:04 (author OK 13:59): block_urmq7.sh, urm_Z with all 259 Q8_Z tensors as Q7_Z; prediction about -305 MB vs Unsloth MTP and +0.0002 vs urm_Z.

### Sandbox items 1-6 (2026-10-28 14:09-14:2x, author OK; MEASURED in the sandbox unless marked)

1. **K5 codebook (E8 lattice, 2^16 smallest-norm points = 2 bpw per stage, residual), real blk.20 experts, inputs rotated per 32** (tools/experiments/k5_codebook_check.py). gate_exps: stage 1 (2.0 bpw) 0.0915 vs Q2_K (2.625) 0.0886; stages 1+2 (4.0 bpw) 0.0105 vs Q3_K (3.44) 0.0230 and Q4_K (4.5) 0.0051; down_exps the same pattern (0.0909 / 0.0102 vs 0.0880 / 0.0228 / 0.0051). On the K-quant line between Q3_K and Q4_K, 4.0 bpw interpolates to about 0.0104: **a tie, not the -17 % on paper.** A plain two-stage codebook with row scales does not beat K-quants; QuIP#'s gain needs its Hessian-aware rounding (LDLQ) and a full random Hadamard, not built. **Parked**: comes back only as an LDLQ test.
2. **C9 1-bit peek** (tools/experiments/peek_planes_check.py): sign-plane output correlation 0.784 (gate) / 0.788 (down) against theory 0.80: confirmed. But its estimate of an expert's output size is off by about 21 % on average, so it can only separate contributions far apart; decision rule must lean on the router weight.
3. **C11 planes by need** (same script, 5-bit zip layout split 4 + 1): reading the base plane only leaves about 8 % output error, nearly the same for every expert (0.079-0.084). The exact worst-case bound is 17-33x looser than the real error, so a rule on it would read every plane; the expected-size estimate (refinement bit as a fair coin; needs only the scales and the sum of x^2 per block, no plane read) matches the real error at 1.00x. **Rule: use the expected-size estimate x router weight.**
4. **Route shares** (tools/experiments/routes_cold_check.py, K/stock_10.rt): cold picks 6.5 %, 20.7 per token; 67.1 % predicted one layer ahead; **6.3 unpredicted cold picks per token**. Router weights: all picks mean 0.125, 20.7 % under 0.08; cold picks mean 0.097, 39.9 % under 0.08, 73.9 % at rank 5-8; **unpredicted cold picks mean 0.083, 52.4 % under 0.08, 9.2 % under 0.05, 87 % at rank 5-8.** The peek's 40 % bar is met only at the 0.08 cut (whose accuracy cost is unmeasured), not at 0.05.
5. **C12 empty activation planes** (synthetic Gaussian and Laplace, per-32 int8 sign-magnitude): 0.000 empty planes per block at every bit. With one scale per 32 every plane carries bits. **Rejected.**
6. **POPCNT on Zen 3** (uops.info, PUBLISHED/MEASURED there): r64 latency 1, throughput 0.25 measured (4 per cycle), 0.31 unrolled. Fable's assumption holds.

### Q7_ZH and the binary gate (author 14:30-14:31: "H is like a gate", "the binary level", "what the Pelias variant is about"; sandbox MEASURED, real blk.20 attn_qkv, rotated, importance-weighted)

- **Split:** Q8_Z's values as q = 2b + h (7-bit base plane, 1-bit H plane). Base only (reconstruct 2b + 0.5) 0.000116 vs native Q7_Z 0.000113: the split costs about 3 %, so one 8.25-bit file serves both 8-bit (H read, 0.000029) and 7-bit reads.
- **Importance-weighted level search:** no change for Q8_Z or Q7_Z on this rotated tensor (0.000029 / 0.000113 either way); the rotation leaves importance nearly flat inside a 32-block.
- **H as a per-block gate (one bit per 32 says "this block carries H"; blocks ranked by importance-weighted error removed):**

| H on | Bits per weight | Weighted error | Straight line |
|---|---|---|---|
| 0 % | 7.28 | 0.000116 | 0.000116 |
| 10 % | 7.38 | 0.000078 | 0.000108 |
| 25 % | 7.53 | 0.000062 | 0.000094 |
| 50 % | 7.78 | 0.000045 | 0.000072 |
| 100 % | 8.28 | 0.000029 | 0.000029 |

  The curve is strongly convex: the first 10 % of H bits remove a third of the gap. Stored as a format, this is a T (tag) bit used as a gate: **Q7_ZT** (7.25 + f + 1/32 bpw, H stored only where the gate is set); kept as a read rule on a full Q7_ZH file, the same ranking decides which H blocks to read. Next: the urm_Q7 result (all base) calibrates KLD per unit of error, then a C++ Q7_ZT at f = 10-25 % and one PC run.
- **urm_Q7 (MEASURED 2026-10-28 14:36, verified with pair.py):** all 259 dense tensors as Q7_Z, 22,358,327,520 bytes, **-305.1 MB vs Unsloth MTP** (predicted -305). Engine format 0.013524: vs urm_Z **+0.001122 [+0.000508, +0.001736], t 3.62, resolved worse**; vs Unsloth on llama.cpp's GPU path +0.000660 [+0.000081, +0.001239], t 2.26, resolved worse. **Prediction missed** (+0.0002 predicted; 5.6x). **Gate 1 for Q7_P (pure 7-bit) fails.** Calibration: +0.001122 KLD for +0.000084 importance-weighted error on attn_qkv, about 13.4 KLD per unit (one tensor type as the proxy).
- **Q7_ZT design (sandbox, MEASURED):** a fixed quota of 2 gated 32-blocks per 256 (fixed block size) loses nothing to the free ranking (0.000061 vs 0.000062); ungated blocks on the odd grid (2b - 127) beat the midpoint base (0.000105 vs 0.000116); the gate ranking needs the importance matrix (0.000061 weighted vs 0.000074 plain). C++ Q7_ZT (id 202, 241 bytes per 256 = 7.53 bpw, importance path): 0.000060 weighted on the real slice, matching the sandbox. **T-decode (author 14:39-14:40, name and direction; "that 1-bit decoder would be clutch"):** the decode that reads the 7-bit base and adds the 8th bit where the T gate says it exists. Two routes, both exact: (a) fold h into the value (q = 2b + h - 127 stays in -127..127) and feed the existing int8 v10 kernel; (b) the 1-bit decoder: the gated H bits are a 1-bit plane, their contribution is AND + popcount against the bit-planed activation (exactness proven in S1), on CUDA cores at decode, on the 1-bit mma for batched MTP verification, on the CPU integer side (author's popcount H) for cold or spilled tensors. Route (b) is the one that lets the base and the H bits live in different places: base in VRAM, H bits (3 % of Q7_ZT's bytes) streamed from RAM; J1's CPU/GPU split; dense models spilling onto RAM. - **urm_ZT (MEASURED 2026-10-28 15:13, verified with pair.py):** -236,419,136 bytes vs Unsloth MTP (predicted -236 MB). Engine format **0.012870**: vs urm_Z +0.000468 [-0.000266, +0.001201], t 1.27, no difference; vs urm_Q7 -0.000654, t -1.27, no difference; **vs Unsloth on llama.cpp's GPU path -0.000006 [-0.000976, +0.000988], t -0.01: an exact tie at 236 MB smaller** (predicted +0.00041 vs urm_Z, a tie with the bar: met). Defect found: chunks 23 / 24 at 0.0424 / 0.0397 against urm_Q7's 0.0146 / 0.0205 although Q7_ZT holds strictly more bits: the importance-weighted level search over-fit the calibration text (ssm_out importance spans 121x the mean). Fixed (commit 24b3d80: levels by plain error, gates by sqrt(importance)); rerun urm_ZT2 started 15:14 (file on D:). urmzt.gguf moved to C:\AI\to-delete for the author.
- **urm_ZT2 (MEASURED 2026-10-28 15:47, verified with pair.py; quantizer fix 24b3d80):** -236.4 MB vs Unsloth MTP. Engine format **0.012536**: vs urm_Z +0.000134 [-0.000174, +0.000443], t 0.86, no difference; vs urm_ZT -0.000334, no difference; vs urm_Q7 -0.000988 [-0.001542, -0.000433], t -3.54, **resolved better**; **vs Unsloth on llama.cpp's GPU path -0.000328 [-0.000849, +0.000194], t -1.25: lower, not resolved, at 236 MB smaller.** Chunks 23 / 24: 0.0138 / 0.0291 (urm_Z 0.0150 / 0.0275): the over-fit is gone. Prediction +0.0004 vs urm_Z: measured +0.000134 (better). **Gate 1 of the 7pH ladder passes for Q7_ZT: no resolved difference from urm_Z (Q8_Z) at 175 MB fewer bytes.** Gates 2 (bytes: passed) and 3 (speed, engine) remain.
- **Shared gate ranking (author 16:24; sandbox MEASURED, real attn_qkv rows, synthetic activations with heavy-tailed block energy, tools/experiments/shared_gate_check.py):** ranking the blocks by the token's activation energy alone (one ranking per token for every row) closes 87 % of the H gap at 25 % of blocks, against 88 % for the per-row ranking (s^2 x sum x^2) and 23 % for random blocks. So every row can read the 8th bit of the **same** input blocks: H stored grouped by input block becomes one contiguous, coalesced read per token on the GPU, and the selection is computed once per token, not per row. Caveat: the gain depends on how unevenly real activations spread their energy over blocks (assumed here, not measured); the whole-model live-gate run and an activation-energy dump settle it.
- **Live H gate, 25 %, per-row ranking on urm_Z (MEASURED 2026-10-28 17:13, verified with pair.py; --h-gate live:0.25, 5029 s):** 0.012919: vs urm_Z +0.000517 [-0.000006, +0.001040], t 1.96 (borderline, not resolved); vs urm_ZT2 +0.000382, t 1.40, no difference; vs Unsloth +0.000055, tie. **Prediction (+0.0001 vs urm_Z) missed on the plain mean.** Without each chunk's top 3 tokens: live 0.009941, urm_Z 0.009827 (+0.000114, where the prediction sat), urm_ZT2 0.009869, Unsloth 0.010326 (indicative, not a paired test): the live gate's extra cost is in the fragile tokens, and on everyday tokens every rotated file sits about 0.0004-0.0005 below Unsloth. The sandbox's 88 % (live) vs 60 % (fixed gate) did not carry to the whole model: real activations' block energies are evidently flatter than the assumed heavy tail, and the fixed gate (odd grid + importance) is at least as good. Reads: live 7.5 bits vs ZT2's 7.53 stored, so the fixed gate (smaller file, no run-time selection) stays the choice; the live gate stays a read-rule option for the engine.
- **T-decode GPU probe, first run (MEASURED 2026-10-28 17:17, RTX 4070, tools/probes/tdecode, log data/probe_results/tdecode/):** every route matches the double-precision reference (max error 1-3e-7 of max |y|); the warp-ballot activation planes equal the host's. **1-bit tensor path: 937.4 T bit-MAC/s vs 117.2 T int8-MAC/s, the same 28,609 mma per us: 8.00x the bit-products** (prediction 8x: met). **The 8th bit by AND + popcount costs about 2 % over folding it into int8** (8192x2048: 54.88 vs 53.88 us). Every 7-bit route ran 1.4-2.8x slower than Q8_Z (8192x2048: Q8_Z 19.3 us, Q7_ZT 53.9 us, the base with no H 52.8 us); prediction (8-9 % faster) missed. Cause (Fable read-only review agreed): the kernel gave one thread per output row, about 1.4 warps per scheduler, a latency-bound chain (the arithmetic is about 7 us of issue); and all timings were L2-warm (Q8_Z at 895 GB/s > 504). Cold, Q8_Z is about 34 us and Q7_ZT's bytes about 31 us.
- **T-decode GPU probe v2 (MEASURED 2026-10-28 19:05, RTX 4070, 46 SMs; cold L2, median of 50; log data/probe_results/tdecode/tdecode_20261028_190518.log):** every route exact (max error <= 1.33e-7 of max |y|). **The 7-bit format is now faster than 8-bit on the large shapes:** 8192x2048 Q8_Z 53.34 us, Q7_ZT fold 50.18 (-5.9 %), Q7_ZT 1-bit decode 51.20 (-4.0 %), ZH base only 47.10 (-11.7 %), ZH 25 % 52.22, ZH 100 % 62.46; 2048x4096 Q8_Z 29.70, fold 27.65 (-6.9 %), 1-bit 29.70 (tie), ZH 0 % 26.62; 4096x2048 Q8_Z 29.31, fold 27.65 (-5.7 %), 1-bit 29.54. Predictions: fold 8-9 % faster (byte ratio -8.7 %): **partly met** (-5.7 to -6.9 %); 1-bit within a few % of fold: **met on the largest shape (+2.0 %), missed on the 2048-row shapes (+6.8-7.4 %)**; ZH 25 % between ZT and Q8_Z: **missed** (at or above Q8_Z: the separate H array costs more than its bytes); every route ok: met. Small shapes (1 MB) are launch/latency-bound (Q8_Z 8.19 us, 132 GB/s) and every 7-bit route is slower there. Reads: Q8_Z reaches 324 GB/s cold, 64 % of the 504 GB/s VRAM rate, so both formats still have headroom (more bytes in flight per lane, several rows per warp, a persistent kernel). Decision for decode (M = 1): fold is the fastest exact gated route; the 1-bit route is the one for split storage (base and H in different places: J1, spill-over) and batched verify; the live ZH gate stays a read-rule option only.
- **Fable on the author's array idea (read-only, 17:3x):** it holds, and it is the format's structural win: bytes are only optional on the bus if they sit in their own sectors, so arrays by role make "read H on f of the blocks" cost exactly f x 1 bpw. Order: the mapping first (lanes split each row; 54 -> 31-35 us predicted, -9 % vs Q8_Z cold), then the arrays as built with H kept block-major (hp[block][row]: a 32-byte sector holds one block for 8 rows, so a gate transfers exactly the selected sectors; row-major H would fetch 68 % of H sectors at f = 25 %) and the base tiled for the warp (each LDG.128 reads 512 contiguous bytes), then the base itself as 4 + 2 + 1 planes (a per-token knob 6.25-8.25 bpw in one file; also the CPU's layout), H for dense in VRAM (246 MB, 2 % of VRAM; the bus and the 0.5 fill for cold experts, J1), MTP batches on the int8 tensor route over the planes, the b1 path a niche. Author 17:32: the 1-bit decode array is what gives the units their 8 bits; make it work first.
- **CORRECTION (2026-10-28 16:1x, from the per-token dumps):** the chunk 23-24 spikes are one or two tokens each, not an over-fit spread over the text. Chunk 23: urm_ZT's excess is one token at KL 5.68; chunk 24: one fragile token that every rotated file flips (urm_Z 4.80, urm_ZT2 5.22, urm_ZT 8.08, urm_Q7 2.52; Unsloth 0.56). Without each chunk's top 3 tokens all files sit together (urm_ZT 0.0111 / 0.0068). urm_ZT2 vs urm_ZT is -0.000334, t -0.86, **not resolved**: the over-fit explanation above is a hypothesis, not a result. urm_ZT2's own numbers stand; the plain level search stays (safer), but why urm_ZT did worse is open. New lead: the chunk 24 token where every rotated file diverges and Unsloth does not.
Test: the b1 mma rate on sm_89 (S2), then a T-decode kernel timed against route (a). **Filling a missing H bit (author 14:42; sandbox MEASURED, real attn_qkv, tools/experiments/h_fill_check.py):** when a gated block's H bit is skipped or late, any integer fill costs the same: h = 0 0.000202, h = 1 0.000206, alternating 0.000204, toward zero 0.000203 (7x Q8_Z). The expected value 0.5 costs 0.000116 (**43 % less**) and stays exact on the int8 kernel: compute sum (2b - 127) x in int8, then add 0.5 x s x sum(x) per 32-block (one block sum of the activation per 32, one multiply-add per block). Rule for T-decode and the H read rule: a missing bit is filled with 0.5 through the block-sum term, never with an integer. (Q7_ZT's ungated blocks need no fill: they are quantized on the odd grid, 0.000105.) Predicted whole model: +0.00041 vs urm_Z (0.01282, a tie with the 0.012864 bar), about -236 MB vs Unsloth MTP.

### Idea register: the unified bus and the idle units (author + Claude + Fable, 2026-10-28 13:0x-13:5x)

Every idea here held up on paper (math checked; Fable read-only where marked). All DERIVED, none MEASURED; each has its test. Details are in the sections below this one. Rejected ideas are listed at the end so they are not re-proposed.

**Clusters on the unified bus (author 13:29: test in groups of 2-3 that depend on one another; each measured as the bus, into one per-token ledger so no gain is counted twice):**

| Cluster | Bus leg | Members | Gain (paper) | Test | Status |
|---|---|---|---|---|---|
| U1 | Both DRAM legs | 2 MB pages + 8 pinned streams (CPU), copy engines / GPU zero-copy, PMU counters | Cold reads 3.0 -> 0.9 ms | Dual-bus bench rerun with large pages: combined GB/s and the new split (0.44 today) | Fable checked; first |
| U2 | Handoffs | Spinning SMT doorbells, BAR1 write-combining into VRAM (Windows build only), idle-time promotion | Handoffs 5.6 -> 0.4 ms; BAR1 -0.2 ms | Handoff round trip vs 13.34 us measured | Fable checked; U1 + U2 = 9.7 ms, 103 tok/s |
| U3 | What the bus carries | Author's rule (predicted cold experts get all planes into L3 during the GPU step), rest one layer late (A2), planes by need | Exposed cold 0.19 ms; 8.65 ms, 116 tok/s | Prediction hit rate from the K route files, paired KLD for A2, cold-step timer + L3 hit rate | Fable checked (planes by need: not yet) |
| U4 | Compute at the ends | GPU: texture units, fp16 accumulate, 1-bit path. CPU: author's popcount H, both schedulers | Decode about 0; prefill CPU +20 %, GPU 0.64 -> 1.28x v10 | Compute per layer against U1's read time per layer | Fable checked popcount H (exact) |
| U5 | One format for both ends | Codebook (E8 residual) for all experts vs PT on bit planes; Q8_Z | Codebook -0.2 ms (0.57 -> 0.47 GB); Q8_Z -0.12 ms; about 8.33 ms, 120 tok/s | 8-weight sandbox test; Q7_PT vs Q8_Z at matched bytes | Q8_Z MEASURED (urm_Z); rest open |
| U6 | The draft leg | MTP draft on GPU slack, draft-driven prefetch, one read per expert for the verified batch | Bytes per token fall with each accepted draft (headless: 17 -> about 32 tok/s) | After GP3-08: acceptance rate | Waits on GP3-08 |
| U7 | The skip leg | 1-bit peek, empty activation planes skipped, neuron skip 25 % on cold experts only | Cold-only skip -9.5 % cold bytes for about +0.00004; peek pays only above 40 % skippable | Offline from the block_k record, then paired KLD | Fable checked (conditional) |
| U8 | The spine | One CUDA Graph per step with the doorbell waits inside, L2 persistence (router, norms, activations), R3 kernel | Up to -3.2 ms: about 120 -> toward the 193 tok/s floor | Engine timers (the 3.2 ms split is unmeasured) | The largest remaining gain |
| U9 | Idle memory as space | VRAM for context and MTP instead of more experts (one more expert saves about 0.6 us once cold is hidden), MTP block at Q3, preloaded prefix KV | MTP Q3 -140 MB; time to first token on repeats | With U6 and U11 | Fable checked |
| U10 | Build-time units | RT cores for codebook search, codebook building, hot-bank re-profiling | Faster builds only | n/a | Build time only |
| U11 | Prefix store | CRC32 page keys (one 32-bit key + exact token compare), DeltaNet checkpoint grid every 256 tokens stored as int8 or rotated int4 + rANS on idle CPU cores, copy engine restore | Full hit: 4K-token prefix skips about 1.83 s; partial hit: about -400 ms | Sandbox: quantize one DeltaNet checkpoint, restore, continue 512 tokens, paired KLD | Fable checked; NVENC loses to rANS |

**J1, split-plane cold expert (CPU base plane over DDR4 + GPU H plane by zero-copy, merged at the doorbell):** exact (Fable). Full-file accuracy on the unpredicted tail for about +0.1 ms; a time gain only if the CPU leg stays under about 35 GB/s after U1. Tests: split dot = whole dot bit-exact (unit test), U1's CPU-leg rate.

**Formats and planes:**
- Q8_Z: whole-model gate 1 passed (MEASURED: equal accuracy, -61 MB); the bar every 7-bit rung now has to beat.
- Q7_P / Q7_PT / Q7_PH: open. PT's 315 > 127 overflow is gone on bit planes (popcount sums exact in int32, scale after), but 8-bit values beat 7 + tags at the same bits, so Q7_PT must beat Q8_Z at matched bytes (sandbox).
- 1-bit tensor path (mma .b1 AND + popcount, sm_89): for batched MTP verification (M = 5-8); at M = 1 CUDA cores do it in 0.7 us. Rate on the 4070: 5-minute probe.
- MTP block as Q3_PH, H read by draft depth: draft 1 reads base + H (its acceptance multiplies every later draft), deeper drafts read base only. After GP3-08.
- Planes by need: after k planes the remaining error is bounded exactly (2^(planes left) x sum |x|); read more planes only where router weight x bound matters. Not yet checked by Fable.
- Both CPU schedulers on different experts at once; activation planes as sign + magnitude so empty planes are skipped. Not yet checked by Fable.

**Carries over to dense models spilling onto CPU/RAM:** U1, U2, J1, planes by need, U5, U6 (strongest), U8, U11. Not: hot bank, expert prediction, expert skip. Prefetch is 100 % predictable there.

**Rejected on the math (do not re-propose):** MTP draft on the CPU (critical path, 1.4 ms from DRAM); AES / SHA / CLMUL; vector gather; 1 GB pages (2 us); L3 as a static bank (0.13 % of picks); NVENC/NVDEC for the live KV cache (1.3 s per token) or the prefix store (rANS on CPU cores is 10x faster at equal size); GPU L2 to keep bytes off DDR4 (it caches VRAM); RT cores at run time; optical flow, display engine, GSP, FP64, x87, NVMe for decode.

### The DDR4 cold path, outside the box (Fable, read-only, 2026-10-28 13:0x; author: "the CPU is blindingly fast, DDR4 is the wall")

Arithmetic is not the problem (66 M MAC per token, 0.03-0.11 ms); the cold step is reads and handoffs. Per token: 21 cold experts x 1.90 MB = 40 MB through DDR4 (13.31 GB/s today = 3.0 ms; ceiling 51.2, about 45 practical on one Zen 3 CCD), plus 80 OS wake-ups x about 70 us = 5.6 ms of handoffs. DERIVED ledger, each item its own test:
- **Doorbells (5.4 ms):** a pinned SMT sibling spinning on a flag the GPU kernel writes into host-mapped memory (no host callback, outside the CUDA Graph host path): 5 us per handoff; with the CPU self-starting on predictions, 40 x 5 us = 0.2 ms. About 10 W while spinning.
- **Bandwidth reached (2.1 ms):** 2 MB pages (the L2 streamer restarts at every 4 KB page: about 90 ns per 102 ns of transfer), one expert per 2 MB page, 8 pinned streams split by output rows, prefetchnta for one-shot experts (keeps L3 for reuse): 13.31 -> 40-45 GB/s. bench_cpu: 8 vs 16 threads, large pages (already listed). DDR4-3600 would add 12.5 % (a hardware setting, the author's call).
- **Bytes kept off DDR4 (0.3 ms):** L3 cross-token reuse (13.6 % of cold picks repeat: 5.5 MB per token); Q5_PH base planes on the down matrices (40 -> 33 MB; planes stored as two contiguous arrays per expert). Entropy coding: no (decoder slower than DRAM).
- **Reads avoided (0.1-0.2 ms, accuracy-gated):** neuron skip 25 % on cold experts only (Block M's +0.000585 on every routed expert scales to about +0.00004 at the 6.5 % cold share, for -9.5 % of cold bytes); MTP batches (8 tokens pick 134 distinct cold experts for 168 picks, -20 %); more zero-miss layers via the bank objective.
- **Hidden behind the GPU (0.6 ms):** one-layer-ahead prediction (67 % of cold picks) and MTP-draft-driven prefetch; copy engines take the predicted experts into the bank with no CPU wake, the CPU takes the unpredicted tail (about 7 experts, 0.3 ms). Copy engines and CPU share the same DRAM (they split the bytes, not multiply them).
- **Combined (Fable's corrected figure):** today 18.5 ms per token (54 tok/s, MEASURED) = 10.1 ms cold step + 8.4 ms rest (5.2 ms weight reads, 2.60 GB at 504 GB/s, + about 3.2 ms launches, attention, norms, sampling that the CPU fixes do not touch). Rung 1, bus speed + doorbells, no hiding: 1.3 ms serial, 9.7 ms, **103 tok/s**. Rung 2, + one-layer-ahead prediction: 0.5 ms exposed, 8.9 ms, **112 tok/s**. Rung 3, + L3 repeats, Q5_PH, draft prefetch: about 0.25 ms, 8.65 ms, **116 tok/s** (Q8_Z dense: 118). At 27 GB/s instead of 45: rung 1 = 10.3 ms, 97 tok/s. Ceilings: 119 tok/s with the 3.2 ms overhead untouched; 193 tok/s memory floor, reachable only by GPU-side work (one CUDA Graph per step, R3). All DERIVED; engine timers decide. (An earlier line here said 16 -> 6.5 ms; that left out the 3.2 ms overhead.)
- **H by prediction (author 13:06, DERIVED):** caches keep few bytes off DDR4 (GPU L2 36 MB caches VRAM, not DDR4: zero; CPU L2 4 MB + L3 96 MB against a cold pool of about 5,400 experts, 2.1 GB even at 1 bit each: resident H covers about 255 experts, about 5 %; reuse 13.6 %). Their value is prefetch: during the 8.4 ms GPU step DDR4 can move about 378 MB at 45 GB/s, against 40 MB of cold need, so the predicted 67 % of cold experts get **all planes** pulled into L3 off the clock (1-bit H 0.39 MB per expert, 2-bit 0.79 MB; 21 picks = 8.2 / 16.5 MB per token), and only the unpredicted third reads the base plane on the critical path. Effect: the cold side's base-plane cost (Q5_PH about +0.0002-0.0003) shrinks to about a third, at about zero exposed time. Rule for the engine: hot = all planes (VRAM); cold predicted = all planes (prefetched to L3); cold unpredicted = base plane. Test: sandbox from the K route files (prediction hit rate per layer) + bench_cpu (prefetch to L3 during a timed GPU-length window).
- **VRAM's marginal value (Fable):** once the cold tier runs at bus speed and hidden, one more expert in the bank saves about 0.6 us per token; spend VRAM on what has no slow tier (context, batches, the MTP block) rather than hoarding experts.
- Small or no: vpshufb LUT, vpmaddubsw/vpsignb (use them; compute is not the limit), pdep (Q4_K scale fields, 0.07 ms), F16C, L3 as a static bank (0.13 % of picks), attention/KV on the CPU (no).

### Idle units, outside the box (Fable read-only pass 2, 2026-10-28 13:1x; author: power is not a limit, turn everything on)

Case A order (cold experts on the CPU), all DERIVED, each with its test:
1. 2 MB pages + 8 pinned streams (13.31 -> 40-45 GB/s).
2. Doorbells: SMT siblings spin all the time (power is free); results written straight into VRAM through BAR1 write-combining, the GPU spins on a flag there (-0.2-0.4 ms of copy launches; needs ReBAR). Test: handoff timer. **Tabled for the Windows-native build (author 13:22):** WSL2 does not expose BAR1 mapping, so A3 is tested only on the Windows build on the reference machine.
3. Author's rule: predicted cold experts get all planes into L3 during the GPU step (0.67 MB per layer vs 0.21 ms of GPU time: fits 14x; L3 is a victim cache, so the SMT sibling pulls them through L2 with dummy loads); they then run from L3 in about 0.02 ms. Test: cold-step timer + L3 hit rate from the PMU.
4. Unpredicted cold experts (about 7): base plane now, refinement plane one layer late (Fable A2: about +0.00005 on paper, against deferral 1's +0.0042 for a whole expert). Exposed DDR4: 8.75 MB = 0.19 ms. Test: `--defer-refinement` reference option, paired KLD.
5. Idle-time promotion: a cold expert turning hot is merged and DMA'd to the bank behind the GPU step.
- After 1-5 the CPU side is about 0.2 ms exposed; the rest of the gap to the 193 tok/s floor is GPU-side (3.2 ms overhead: one CUDA Graph per step, R3).

GPU side and formats:
- **Codebook (residual E8-lattice) for all experts, not cold only:** 4.0 vs 4.83 bpw (-17 %); hot experts 0.57 -> 0.47 GB in VRAM = -0.2 ms per token; 2 KB table in shared memory / L1. Stage 1 alone is about Q3-class (coarser than Q5_PH's base): measure. Gate: the parked 8-weight sandbox test (line 1065).
- **_PT back on paper (author 13:19, "the PT is back"):** bit-plane products on the GPU's 1-bit tensor path (PTX mma .b1 with AND + popcount, sm_80 and up; the 4070 is sm_89). Each plane's popcount is exact in int32 and the tag scale is applied after, so Q7_PT's blocker (tag x level 315 > 127 cannot feed the int8 mma) disappears. Cost: 8 activation planes x k weight planes of 1-bit mma; decode is memory-bound, so it may hide. Still open: b1 mma throughput on sm_89 (unmeasured: bench_gpu), and the second reason Q7_PT lost (8-bit zip is better at the same bits) is an accuracy result this does not change; Q7_PT must beat Q8_Z at matched bytes in the sandbox.
- **Author's popcount H on the CPU integer side (Fable: exact):** sum H x = sum_b 2^b popcount(H AND x_b) - 128 popcount(H); about +20 % compute throughput; matters for CPU prefill (2.7 -> 2.2 s per 2048 tokens headless), not decode. Verify POPCNT 4 per cycle on Zen 3 (Agner Fog) first.
- **1-bit peek (Fable):** correlation sqrt(2/pi) right; pays only if more than 40 % of peeked experts can be skipped KLD-neutral, and the top bit must be its own plane. Test offline from the block_k record.
- Case B (whole model on the CPU): MTP verify is the lever (about 32 vs 17 tok/s, hinging on GP3-08's acceptance rate); dense is 77 % of the bytes.
- Rejected on the math: MTP draft on the CPU, AES/SHA/CLMUL, gather, 1 GB pages, L3 as a static bank.
- Sent to Fable next: planes read by need (exact error bound per plane), both CPU schedulers on different experts, skipping empty activation planes (sign + magnitude).

### J1 and U11, Fable read-only check (2026-10-28 13:5x; DERIVED)

- **J1 (split-plane cold expert: CPU base plane over DDR4, GPU H plane by zero-copy, merge at the doorbell):** exact (integer sums per sub-block, merge differs only in fp32 addition order, 2^-24; int32 bound 4.1 M); the GPU half also needs the sub-block scales, 0.2 MB per expert (0.85 MB per leg). Both legs read one DDR4, so the 47.62 GB/s dual bus is the ceiling, not two buses: J1 gives full-file accuracy on the unpredicted tail for about +0.1 ms over A2, and pays in time only if the CPU leg stays near 26.56 GB/s after the page and stream fixes. Build it only if bench_cpu's CPU-leg rate stays under about 35 GB/s. At decode (M = 1) the H plane runs fine on CUDA cores (0.7 us); the 1-bit mma (m16n8k128 .and.popc, listed for sm_89, deprecated from sm_90) belongs to batched verification (M = 5-8). Tests: split dot = whole dot bit-exact (unit test); b1 mma rate on sm_89 (5-minute probe).
- **U11 (prefix store):** CRC32 pays (128K tokens hashed in about 15 us; one 32-bit key with the exact token compare is enough; two CRC seeds do not make 64 bits, CRC is affine in the seed). The DeltaNet checkpoint grid pays. NVENC/NVDEC lose to cheaper stores: int8 per row (4x, 0.6 ms restore) or rotated int4 + rANS on idle CPU cores (8-10x, about 1 ms decode); rotation helps quantization but removes the spatial smoothness the codec's prediction needs. Per 128K context: int8 every 256 tokens 8 GB / 57 ms TTFT penalty; rANS-int4 every 256 3 GB / 60 ms; NVENC every 256 2.9 GB / 114 ms. Best pairing: CRC32 (integer side) + rotation, RTN, rANS (idle CPU cores) + copy engine for the restore; NVDEC stays idle. The 4070 has one NVENC (8th gen) and one NVDEC (5th gen); rates unmeasured. Lossless HEVC: about 1.2-1.3x, a fallback only. Capacity in about 45 GB of RAM: about 25 contexts at 32K, about 8 at 128K. **Gate (sandbox, minutes):** quantize one DeltaNet checkpoint (int8, then int4 rotated), restore, continue 512 tokens, paired KLD against the unquantized continuation; a lossy recurrent state's error compounds, so accuracy decides.

### The web, version 2: idle units with documented AI uses, paper math, Fable's once-over (author + Claude + Fable, 2026-10-28 17:5x-19:1x)

Doc "Idle Units List (Unified Bus)" (claude.ai artifact) holds the full tables and sources. All DERIVED unless marked; each line has its test.
- **Research pass (NVIDIA/AMD docs first):** Zen 3 POPCNT reciprocal throughput 0.25 (uops.info; closes the "verify first" note on popcount H). AES-NI and CLMUL run on FP pipes 0-1, shared with vector multiply (AMD Zen 3 optimization guide): not free, stay rejected. CRC32, PDEP, PEXT share ALU1. Zen 3 ymm gather about 8-9.8 cycles per 8 elements (no better than scalar). GSP not user-programmable; FP64 1/64; OFA's only AI use is DLSS 3. The 4070 has one 8th-gen NVENC with lossless HEVC.
- **C1 one decoder (H gate on KV pages, author's H gate):** 128K context, KV reads 2.83 -> 0.26-0.52 ms per token at top 1/32-1/8; codes 84 MB. Fable: codes need their own warp ballot on q and k; score with the scaled 1-bit dot (per-32 scales, 40 B per key); fold a random sign diagonal into W_q and W_k; page bound (Quest) first; CUDA cores suffice. CPU scoring of all heads = 5.2 ms per token (too slow). Test: one-chunk q, k dump, recall of attention mass at 1/8 and 1/32.
- **C2 tiered context:** raw tiers VRAM -> pinned RAM -> NVMe (CRC32 keys). Fable: KVFetcher's 11.9x is after integer quantization; lossless on q8_0 KV is at most about 1.11x from entropy, so NVENC/NVDEC stay idle; capacity, not speed (45 GB RAM about 126 contexts at 32K raw). Test: engine restore timer.
- **C3 fragility tags + preload (author):** chunk 24's token is non-monotonic in dense bits (urm_Q7 2.52 < urm_Z 4.80 < ZT2 5.22 < ZT 8.08; Unsloth 0.56, MEASURED): noise-driven. Next: one-chunk reference runs to find which side flips it (experts or dense). Signal (author's ROP idea, Fable's form): attention near-tie flag by red.global.or inside the token, flag by attention layer 3, cold-expert refinement planes pulled in 0.2-0.4 ms. Tags calibrated on chunks 0-49, scored on 50-99. Test from existing dumps: margin/entropy, KL autocorrelation, copyable n-grams of the top-3 tokens.
- **C4 Q3 with grid tags (author's bit-tag matrix at 3 bits):** synthetic 2 tag bits per 10 values = -19/-34/-42 % error (Gaussian/Laplace/t4) for +0.2 bpw. Fable: real Q3_K on blk.20 gate_exps is 0.0230 at 3.44 bpw (MEASURED), so synthetic overstates; plane form per 32-block = 3 planes + 1 tag byte + zip scales = 3.5 bpw; additive grids keep the b1 path exact; tags + H gate stack at 3.75 bpw. Targets: 3.5 bpw <= 0.0211, 3.75 bpw <= 0.0148. Uses: the whole MTP block (error costs acceptance only), expert gate/up; expert down >= 4 bits; rotated dense stays ZT. The 7+7+7+7+4 word pays nowhere at 7 bits.
- **C5 three drafters, one verify (author's matrix array):** 1.6x at acceptance 0.7, 1.9x at 0.8, but only after U1/U2 (at today's cold step 5 rows gain nothing; at rung 1, 169 vs 103 tok/s). One chain with priority exact n-gram > MTP > fuzzy (DeltaNet state is per branch). Test: one-chunk hidden-state dump, fuzzy and prompt-lookup acceptance.
- **Combinations (Fable):** a rejected exact-n-gram draft marks a copy-head flip, so that position gets refinement planes; one bit-plane machinery for T-decode, KV codes and verify; one CRC32 key space; the C3 flag shares the U2 doorbell mechanism.
- **Nulls:** SFU (softmax already uses it), NVENC/NVDEC (no job survives), ROPs only publish the flag.
- **C6 accuracy-assisted miss map (author 19:15-19:18; "it will activate and boom the tensor understands better ... whenever a model is having an issue"; "the more we look into areas that it got stuck on, the better we understand when to tell it to look"; "we won't know what to look for unless we keep logging it, understanding it, then plan for it"):** log every place the system gets stuck (fragile tokens, cold-expert prefetch misses, rejected drafts) with its hidden-state code (C1's sign plane); each token, check the current state against the map; a hit takes the expensive path (refinement planes, deeper prefetch, re-verify). Popcount does the lookup; RT cores take over once the map reaches millions of entries (JUNO range 1M-100M). Also a multi-layer-ahead prefetch map (today one layer ahead catches 67 % of cold picks, MEASURED). Gain per token = recall x loss on a miss - false-alarm rate x cost of the expensive path (DERIVED). **First check (MEASURED 2026-10-28 19:2x, tools/experiments/miss_cluster_check.py, existing dumps, 25,500 tokens):** the stuck places repeat across files: of each file's top 1 % KL tokens, 55-66 % are also top 1 % in another rotated file and 47-52 % in Unsloth's MTP file (random: 1 %), so they belong to the text and state, not to one file's noise; they come in runs (stuck at t-1 makes t stuck 11 % of the time, base 1 %); the reference itself is less sure there (median NLL of the true token 1.50-1.67 vs 0.89 overall; top-token agreement 55-65 % vs 94-95 %); the stuck 1 % holds 23-26 % of every file's KL. Next: log hidden-state codes per token and layer in the reference (built 0803210: `pelias kld --dump-codes`, 320 bytes per layer, 12.8 KB per token, about 326 MB per 100-chunk run) to test clustering in state space, and the K route records for prefetch misses. **Author 19:21-19:24:** RT cores are the fit here because of their speed: "a shot of adrenaline for the LLM to continue", only as needed; the RT lookup is separate hardware, so it runs at the same time as the SMs work on the rest of the token (the model keeps working while the problem area is looked up). Render work (Blender, Unreal, math visuals) as a later use; note the RT cores are shared with the renderer when both run on the one GPU.

### Runs A-E and the checks around them (author OK 19:32 "go for a-e"; 2026-10-28 19:3x to 2026-10-29 00:1x)

All MEASURED unless marked. Engine format (gpu256x8), 100 x 512 WikiText-2 base unless marked. Logs under D:\pelias-tests\gp (C6, TD, OT, U); dumps and scripts in the repo (tools/experiments, data/probe_results).
- **B, C6 state codes (urm_ZT2):** logging changes nothing (0.012536, difference 0.000000); codes 326,400,040 bytes as predicted; 23 min. **State-space check (tools/experiments/c6_cluster_check.py; map = chunks 0-49, test = 50-99):** a stuck token's nearest logged neighbour is stuck 3.0-5.2 % at layers 10/30/39 vs 1 % base (none at layer 3); "near a logged miss" catches 25 % of stuck tokens while flagging about 10 % of the rest; the nearest logged state predicts 43-53 % of the experts two layers ahead (random 3 %; pre-gating one layer ahead 74-88 %). Plain unlearned map; next: several layers, energy weighting, the previous-token signal, a learned projection.
- **A, kernel v3 (each lane 2 or 4 rows, loads first):** every route exact; prediction (430-450 GB/s) MISSED: v3x2 Q8_Z 331 GB/s vs v2 325 on 8192 x 2048, 7-20 % slower on the other shapes; x4 slower still (fewer warps in flight). Open question: is ~325 GB/s the ceiling of one cold 17 MB launch (fixed start-up 10-15 us against 34 us of data at 504 GB/s)? Test: a plain read-only kernel over the same 17 MB cold; if it also lands near 330, the gain is U8 (one graph per step), not the kernel. v2 stays.
- **C, own-text check:** text from the author's dataset, local only (256 records: 23 held-out eval + 233 train; 271 records dropped for sharing a 64-token run with the imatrix calibration text, including 45 of the 68 eval records; 53,334 tokens, 0 shared shingles); bf16 base 72 min (12.66 GB, D:). KL: urm_Z 0.020977, urm_ZT2 0.021198, Unsloth MTP (llama.cpp GPU format) 0.022368; ZT2 vs Unsloth -0.001170 (t -0.75), Z vs Unsloth -0.001391 (t -0.98), ZT2 vs Z +0.000222: order as predicted, nothing resolves (chunk-to-chunk spread about 4x WikiText's; about 4x more chunks would halve the error). C6 on own text: 59-63 % of each file's top-1 % tokens shared with every other file, Unsloth included; the top 1 % holds 52 % of the KL (WikiText 23-26 %); runs 11.4 %; reference NLL at stuck tokens 1.31 vs 0.18; without each chunk's top 3 tokens: urm_Z 0.011772, urm_ZT2 0.012240, Unsloth 0.012583. Author: the text is unusual (governing equations for LLMs written in text, CTT and coding languages, records about 30-40 % through the file); equation-band check below.
- **D, T-decode whole-model proof:** 0.012923 vs urm_ZT2's 0.012536, +0.000387 (t 1.47, not resolved); prediction (identical to 6 decimals) MISSED. Diagnosis (tools/experiments/tdecode_forward_check.cpp, synthetic Q7_ZT model, whole forward): **the 1-bit route equals the integer fold bit for bit (max logit difference 0)**; integer path vs the float path (dequantized weights): tokens 0-8 agree to 5e-7, later tokens differ by up to 9e-3. Summation-order noise flips int8 activation roundings downstream (a discontinuity, not an error). Consequence: two numerically equivalent paths can differ by about +-0.0004 KL on 100 chunks; a noise floor for every paired test. Confirms the author's point: filling the 7 bits to 8 recovers the accuracy (pure 7: +0.001122 vs urm_Z; 7 + gated 8th: +0.000134), and the 1-bit decoder delivers that 8th bit exactly.
- **E, urm_TG (82 expert gate/up tensors Q4_K -> Q4_TG, the author's bit tags at 4 bits):** size identical to urm_ZT2 (22,426,968,288 bytes, -236.4 MB vs Unsloth; met); quantize 34 min; KL 0.012876: vs urm_ZT2 +0.000339 (t 0.77), vs urm_Z +0.000474, vs Unsloth +0.000012 (tie); prediction (-0.0002 to -0.0004) MISSED in direction, inside the noise floor. **Diagnosis (my setup error):** the sandbox compared the tags against Q4_K without the importance matrix (plain error); the file's Q4_K uses it. Importance-weighted on real blk.20: Q4_K+imatrix 0.004623 / 0.004721 (gate / down) vs Q4_TG as built (sqrt importance) 0.004601 / 0.004575 (-0.5 / -3.1 %), full importance 0.004468 / 0.004495 (-3.4 / -4.8 %). **Author's 0-weight idea:** E's four grids have no exact 0, while about 13 % of 4-bit expert weights sit at level 0; grids with a 0 (uniform -7..8 and a log grid with 0): 0.004131 / 0.004159, about -11 % vs imatrix Q4_K (sandbox, before scale rounding). Candidate **Q4_TG v2**: full importance + grids with 0; for the author's review.
- **Author's rotation questions (sandbox, real blk.20):** experts rotated or not have the same shape (per-block kurtosis 2.85 vs 2.81; 13.4 vs 13.2 % of 4-bit values at 0; 0.73 % at 8 bits): they are near-Gaussian, not heavy-tailed (my earlier claim corrected). Dense attn_qkv, 7-bit + gated bit on 2 of 8: **light bit (refines the step) vs heavy bit (extends the range)**: unrotated light 9.43e-5, heavy 9.58e-5; rotated light 7.86e-5, heavy 7.98e-5: rotated wins both by about 17 %, the light bit by 1.5 %; the gate removes a slightly larger share of the gap unrotated (44 vs 40 % at 2 of 8). **When the bit is missing:** light falls back to the 7-bit base (1.41e-4 / 1.12e-4), heavy collapses (3.2e-2 / 2.9e-2, about 300x; values past half the range clipped). Choice confirmed: rotated + light bit (urm_ZT2). tools/experiments/light_heavy_bit_check.py. **Rotation coverage (2026-10-29):** urm_ZT2 rotates 257 of 259 Q7_ZT tensors; token_embd (rotation -0.9 %, near Gaussian, not worth it) and blk.40.nextn.eh_proj (**rotation -49.5 %**) are not. **Decision (author, 2026-10-29 00:31): rotate the MTP piece only; token_embd and the routed experts stay as they are.** Done in code: nextn.eh_proj.weight is a product weight for --rotate32; future builds add "eh_proj" to the dense groups (attn,ssm_,shexp,output,eh_proj). It pays in MTP acceptance (speed) once the engine runs the MTP head; main-model KL unchanged by design. Rule (author): always keep the light 1-bit rotated.
- **Equation-band check (own text; tools/experiments/owntext_equation_check.py, run on the author's PC, counts only):** the choice re-done exactly (text reproduced byte for byte); 24 of 256 records come from the 30-40 % band, 13.4 % of the scored tokens; the stuck 1 % come from the band 9.8-11.4 % of the time (urm_Z / urm_ZT2 / Unsloth), slightly under its share; the band holds 12.7-14.4 % of the KL. The equations are not where the knife edges sit; next (counts only): place the stuck tokens by kind (role headers, code, prose, numbers).
- **Logged ideas (author):** heavy developer section (owner-only, password known only to the author, every warning firing; for when the system is ready; password as a strong one-way hash only); X list: a Pelias VBIOS explored on the 3070 Ti (blown Holtek MCU, headless) before the 4070; the 3070 Ti on a riser at x4 with the 850 W PSU; on the 3070 Ti (author 01:34), try putting the inactive units to work as a decoder for entropy-coded (rANS) weights and see how fast one, or several at once, can go: the 3-8 % storage saving only becomes free if decoding keeps up with the card's VRAM read rate; RT cores as the as-needed miss-map lookup beside the SMs.

### The idle-units tour (author 00:47-00:50: "do those before the 2 we have on hold"; 2026-10-29 00:5x-01:2x)

Author: "any of these steps may make something read that much better." Each step: goal, type, prediction, result. Full status board of every idle-unit / unified-bus item (achieved, ruled out, on the docket): the Idle Units List doc, section "Status board".
- **Rotation coverage (00:2x, MEASURED):** urm_ZT2 rotates 257 of 259 Q7_ZT tensors; token_embd (-0.9 %, near Gaussian) stays unrotated; **blk.40.nextn.eh_proj -49.5 % if rotated: rotated from now on (author 00:31)**, `rot32_selected` + groups `attn,ssm_,shexp,output,eh_proj` (e7b5bc0). Routed experts -0.8 to -3.4 % per tensor, stay as stored (R2b: +0.000530 vs stock, no difference). Rule (author): always keep the light 1-bit rotated (light vs heavy table: 7/T/H plan 9.1).
- **MTP 7 vs Q3 (DERIVED):** see the U6 / GP3-08 entry: stay at 7-bit rotated unless Q3 is within ~1.5 points of acceptance (1 draft) / ~2 % (3 drafts).
- **C3 fragility tags (MEASURED, prediction missed):** reference margin / entropy / p1 AUROC 0.61-0.65 for the top-1 % KL tokens; a wrong n=3 copy 2.0 % risk vs 1 % base, a right one 0.38 %. Free signals are weak; C3 tags come from the C6 miss map (3-5x). Open: the in-model attention near-tie flag (ROP / red.global.or). tools/experiments/c3_ref_features.py, c3_fragility_check.py.
- **C4 tags + H at 3.785 bpw (MEASURED, sandbox):** 0.01491 / 0.01485, at Fable's target but about 6 % above the K-line at that size; H on top of tags does not pay; tags alone -9 % at 3.5 bpw.
- **C1 H gate on KV pages (MEASURED, real q, k):** `pelias kld --dump-qk` (PELQK001, adf1b92; KL unchanged, identical to run B). urm_ZT2, 4 chunks x 10 attention layers x 16 heads, queries t >= 64: attention mass kept at 1/8 of the keys: best possible 85.7 %, **1-bit scaled code 72.6 %**, Quest pages 62.0 %, recent window 42.6 %; at 1/32: 64.6 / **47.1** / 45.2 / 28.1 %. Prediction (>= 85 / >= 70 %) missed in absolute terms: 512-token chunks spread the attention (the best possible 1/8 holds only 86 %). Beats both baselines. Next: long-context dump (attention sparser there); candidates: union with the sink + recent tokens; 1-bit pick of 2x then exact rescoring. tools/experiments/c1_recall_check.py; D:/pelias-tests/gp/C1.
- **rANS on bus transfers (MEASURED, urm_ZT2 read-only):** order-0 entropy of the stored codes: Q4_K 3.87 of 4 bits, Q5_K 4.87 of 5, Q7_ZT base 6.75 of 7, Q6_K 5.80 of 6 (per-256 adaptive cap 3.82 / 4.77 / 6.36 / 5.61). Lossless coding saves 3-4 % on expert transfers: ruled out for active weights (prediction 5-10 % missed). tools/experiments/code_entropy_check.py.
- **Worst-chunk codes (C6):** the 10 worst wikitext chunks hold 22.7 % of all KL (chunk 78 alone 4.8 %; chunks 39 and 23 are single spike tokens); their state codes saved (D:/pelias-tests/gp/C6/worst_chunks_wikitext_v1.npz; summary in data/probe_results/c6).
- **Process lessons:** a background job started with `wsl -e` ends with its WSL session (run foreground: the tool times out, the job keeps running); a bundle needs the `main` ref (`git bundle create F main ^BASE`).
- **What the rANS test missed (author 01:26):** the 8 GB -> 3 GB was mostly the lossy step (rotated int4) on the DeltaNet *state*; rANS adds about 1.3x. The active tensors not yet squeezed are the KV cache (q8_0, 10,880 B per token: about 1.4 GB at 128K, 2.85 GB at 262K) and the DeltaNet state (30 layers x 32 heads x 128 x 128 fp32, about 63 MB per conversation). rANS on weights: storage -3 to -4 % (up to about -8 % on the 7-bit dense base), speed only if a GPU decoder runs at VRAM speed (unmeasured on the 4070; probe: a GPU ANS decoder's rate).
- **KV keys at 4 bits (MEASURED, sandbox):** attention moved 0.49 % (q8_0), 8.68 % (int4), 6.21 % (rotated int4). Whole-model gate: `pelias kld --kv-store` / `--state-store` (built in the sandbox, f53610d). **On hold (author 01:45): not a big overall item, looked at down the road** (4 runs; about 95 min). **Done 2026-10-29 10:29 (MEASURED, urm_TG2, vs exact 0.012056): KV q8 +0.000350 (t 2.00, grows with context), KV int4r +0.004156 (t 16.8), state int8row +0.000356 (t 2.07, one restore), state int4r +0.032824 (t 32.3); a same-build exact rerun matches the reference bit for bit (2,550 of 2,550 tokens). Decision input: the state stays at 8 bits or more; KV q8 is the cheap option; KV int4 needs the H gate or a better format before it is used.** The two state runs test one checkpoint restore per chunk (the prefix-store case); live per-token state storage (the ~0.19 ms per token read saving, DERIVED) needs a per-token mode, added with the next build.
- **Own-text stuck tokens by kind (MEASURED 01:47, counts only):** number/math tokens are stuck at twice the base rate (2.0-2.2 %) in every file and carry 26-28 % of the KL from 18.5 % of the tokens; prose sits at the base rate (60 % of the KL); punctuation and code are easy. For the miss map: number/math tokens are a cheap first tag (the token kind is known before the forward pass).
- **Done (was on hold):** Q4_TG v2 became Q4_TG2 (id 204) and urm_TG2 (run A, KL 0.012056, the best file); Fable's math check #3 done; R3 v1 measured.

### Fable's math check of #3 (2026-10-29 02:0x, read-only; full report docs/reviews/FABLE_MATH_CHECK_3_2026-10-29.md)

- **Ledger:** holds in structure. Fixes: the measured 10.1 ms cold step has a 0.9-2.5 ms residual the rungs assume away (rung 1 = 9.7-11.2 ms, 89-103 tok/s); the 5.2 ms of weight reads assumes 100 % of 504 GB/s, while the measured kernel runs at 64 %, so the 5.2 / 3.2 split decides U8 vs bytes-in-flight (most important unmeasured number); realistic floor 167-177 tok/s; Q7_ZT saves 126.6 MB per token on the trunk (-0.17 to -0.25 ms), not 236 MB (61.6 MB of it is token_embd, read one row per token); rung 3 8.65-8.8 ms (double counting removed). 2 MB pages alone will not triple DDR4: the 3x is threads + software prefetch (test the four cells); `mwaitx` vs `pause` for the spinning doorbells. New: overlap the unpredicted cold read with the same layer's hot GEMVs; over-prefetch (the window has 14x slack).
- **C1:** a >= 32K feature; bar >= 95-97 % of the mass kept; test the oracle vs context length first, then union sink + recent and 2x + exact rescoring; score on output error / KL.
- **C3 / C6:** add precision and recoverability R; likely R is small, so the flag is a speed feature (license cheap paths on unflagged tokens). Test R first (oracle-flag paired run, ~25 min). Engine flag: tokens after a rejected draft + number/math tokens. The map lookup must be an index, not a scan.
- **C5:** 1.42-1.73x at rung 1 (verify of M rows reads more hot experts); measure distinct hot experts for M = 2..8 from the route files.
- **MTP:** stay at 7 (holds). Bigger lever: a 1-2-bit sketch of the output head + exact rescoring of the top 64: draft 470 -> 117-180 MB, a 3-draft cycle -25 %; gate = top-1-in-top-64 recall from existing logits.
- **R3 item 7 (11:43, MEASURED):** per-expert counters (120 barriers): 5.685 ms per real token, -0.8 % (predicted 5.4-5.6, missed). Best: streaming 4.75 ms, real token 5.69 ms. The remaining ~0.93 ms gap is explained (drain/ramp at real barriers, format serialization, polling) and documented for others; next: R3 into the engine (GP3-08).
- **R3 routes 1+2 and HAGS A/B (11:28 / 11:31, MEASURED):** route 2 (incremental job lookup) is a keeper: counted (real-token) 5.74 ms, streaming 4.76; route 1 (rolling prefetch) gives nothing. HAGS on vs off: every device-side row within ~1 %; one launch per matrix median unchanged (22.7 vs 22.6 ms), wider spread. Next: per-expert counters (item 7, 40 of 160 barriers), per-SM release lines.
- **R3 v2 fix 6 measured (10:59, WSL):** counter barrier 6.16 ms, + one-task prefetch 5.91 ms, vs grid sync 6.06 and streaming 4.82: small gain (-2.6 %). The barrier is 0.5 us; the ~7 us per boundary is the stage tail (a partial last round of ~5.4 us). Whole-share prefetch (11:03) made it worse, 6.99 ms: the head stage alone is 417 MB and ~169 lookups per warp; next: a byte-capped prefetch (few tasks per warp, none into the head), then tail-splitting and fewer barriers (item 7).
- **R3 v2 fix 2 done (10:43, MEASURED, WSL / Windows):** Q4_TG2 grid in shared memory: one token 8.6 -> 4.55 ms (about 500 GB/s, 1.9x); with 161 staged syncs 12.5 -> 5.8-5.9 ms; Q4_TG2 alone 40 -> 347 GB/s; all paths exact. Confirmed with 40 distinct layers (10:46): 4.80 ms per token, 473 GB/s (the 4-layer cycle's 4.55 ms had ~0.25 ms of L2 reuse); the GPU reached its 200 W software power cap in the soak. Next fixes: 6 (load-before-wait for (c)), 3 (dp4a for Q5_K / Q6_K), 4 (job lookup).
- **R3 v2 step 1 done (10:32, MEASURED, WSL / Windows):** Q4_TG2 alone streams at 40-45 GB/s while Q7_ZT 467-470, Q5_K 407-418, Q6_K 426-461; the grid sync itself is 0.52 us, so the 24 us per boundary is drain/ramp; graph replay of (a) 19.2 ms on both; (b) 8.6-8.7 ms; SM clock 2.774 GHz. Next: fix 2 (Q4_TG2 shared-memory LUT), measured alone.
- **R3 v2 plan (Fable review 08:40, docs/reviews/FABLE_R3_V1_REVIEW_2026-10-29.md):** 1 per-format (b) rows; 2 Q4_TG2 shared-memory LUT; 3 Q5_K/Q6_K dp4a; 4 one-load job lookup; 5 bytes in flight; 6 (c) load-before-wait + counter barrier (+ empty-stage sync row); 7 real-token sync cuts; 8 drop the 161st sync. Keep v1 modes as switches; build the same v2 code on WSL and natively on Windows and run back to back (author 08:35). Runs now under the Ultimate Performance plan.
- **Correction (08:32, author's memory overclock):** the card runs its memory at 11,501 MHz (stock max 10,501), so its peak is about 552 GB/s, not 504: the measured 494 GB/s ceiling is ~89.5 % of the card as run (the '98 %' below is against the stock 504). All '% of 504' figures for this machine read against 552; R3's headroom is larger than assumed.
- **Test 1 part 1 done (02:11, MEASURED):** read-only ceiling per cold launch: 1.08 MB 151 GB/s, 8.65 MB 334, 17.3 MB 352, 64 MiB 445, 256 MiB 494 (98 %); empty launch 3.07 us. The GEMV v2 (325-331 at 17.3 MB) is at 92-94 % of the single-launch ceiling: the lever for R3 is a persistent kernel streaming across matrices (toward 494 GB/s), not the inner loop. Part 2 (author chose a derivation instead of profiling the tier build): one token's 2.41 GB of weight reads at the measured per-launch ceilings = 7.8-8.1 ms (296-308 GB/s, 431-471 launches) vs 4.88 ms as one persistent stream at 494 GB/s. **R3 target: a persistent kernel that streams every matrix of the token back to back, worth about 3 ms per token** (the ledger's 3.2 ms 'overhead' is mostly per-launch read ramps).
- **Fable's next tests:** GPU step split (ceiling kernel + nsys); decompose the cold step; bench_cpu four cells; counts-only route checks (C5 verify, over-prefetch union); C1 oracle vs context length; C3/C6 R; MTP head sketch recall; draft-rejection flag proxy.

### Decision: R3 goes with Q4_TG2 option A (author 02:24; 2026-10-29)

- **PEL7 candidate (urm_TG2):** dense Q7_ZT rotated (light bit), expert gate/up Q4_TG2 (id 204), expert down at Unsloth's types (Q5_K / Q6_K), MTP eh_proj rotated. Run A (block_urmtg2.sh, ~60-70 min) confirms it whole-model; waits for D: to be cleared (author emptying D:/AI/to-delete).
- **Struck / parked:** Q4_TG2 option B (down at 4 bits: down error 3.5x Q5_K's, MEASURED 0.00419 vs 0.00119); Q4_TG v1 (retired); U5 E8 codebook for experts (exclusive with Q4_TG on the same tensors, Fable); a 5-bit tag type for down (optional, later); inner-loop GEMV tuning (at 92-94 % of the single-launch ceiling).
- **Next, in order:** (1) run A; (2) R3 in the sandbox during run A: one persistent kernel streaming every matrix of a token (dense Q7_ZT fold first, then Q4_TG2 gate/up with a grid table and Q5_K / Q6_K down), target ~4.9 ms of weight reads per token (MEASURED ceiling 494 GB/s); (3) R3 on the PC: exactness, then throughput vs the ceiling; (4) Fable's remaining tests (cold-step decomposition, bench_cpu four cells, route counts for C5 / over-prefetch, MTP head sketch, C1 at long context, C3 / C6 R); (5) GP3-08, engine, server.
- **Fallback:** if run A does not improve on urm_ZT2, gate/up stay Q4_K; the streaming kernel reads both.
- **Run A result (MEASURED 03:37):** urm_TG2 **0.012056** (same top 95.47 %) at -236.4 MB: vs Unsloth -0.000808 [-0.001618, +0.000002] (t -1.98, a hair short of resolved), vs urm_ZT2 -0.000480, vs run E -0.000820, vs urm_Z (8.25-bit dense) -0.000346. The best file so far, smaller than Unsloth's; **Q4_TG2 option A is confirmed as the expert gate/up format for R3** (PEL7 candidate). Next: resolve it (more chunks or the own-text run), and R3 step 3.
- **R3 step 3 (MEASURED 07:36, probe v1):** exact on every format. One token, 2.27 GB: per-matrix launches 20.8 ms (109 GB/s); one streaming launch **8.8 ms (259 GB/s)**; staged with 160 grid syncs 12.6 ms (181 GB/s, 23.6 us per boundary). Speed predictions missed (440-480 GB/s, a few us per sync); streaming beats per-matrix launches 2.37x. v1 suspects: 2 blocks per SM (one generic kernel for four formats), a job-table search before every warp task, divergent constant-memory grid reads (Q4_TG2), pipeline drain at each stage boundary (fix: prefetch the next stage's weights into L2 before the sync, since only the activations wait on it). **Paused here for the author's assessment (author 02:37: steps 1-3, then pause).**
- **WSL2 CPU check (author 07:45; MEASURED in WSL + the author's Windows system log):** WSL2 sees all 16 threads (8 cores x 2, presented as pairs 0+1, 2+3, ...), the full 96 MB L3, 46 GiB RAM, no CPU cap (cpu.max unlimited). It is a Hyper-V guest: its CPUs are virtual and scheduled by Windows, so thread pinning in WSL pins to virtual CPUs (the pairs need not be the physical SMT siblings) and Windows work competes; `monitorx/mwaitx` is not exposed (Fable's mwaitx doorbell is Windows-native only, if at all); AVX2, POPCNT, BMI2 present. Consequence: R3 (GPU) unaffected; the CPU-side tests (U1 pinned streams, U2 spinning doorbells, bench_cpu four cells, the cold-step decomposition) run on both WSL and the Windows-native build, each labelled.

### The machine probe's memory and threads check, and the bus view (author 07:52-08:00, 2026-10-29)

- **`pelias membw` (built, sandbox; part of the probe):** RAM read speed by thread count (1, 2, 4, 6, 8, 12, 16), 4 KB vs 2 MB pages, software prefetch off / on, readers pinned one per physical core first; and the user's side: a foreground job that wakes every 1 ms meanwhile, its wake-up lateness (p50 / p99) and job time vs the idle machine (author: "by slow everything down, i meant as a user utilization of doing something on the system vs what pelias is doing"). Answers Fable's test 3 cells and the 6-core / 12-thread question; the engine's settings get a "reserve cores for me" option sized from it. Windows-native run with the Windows build. **Result (MEASURED 08:05, WSL):** 2 threads read 45.9-47.8 GB/s (93 % of the DDR4 peak); 1 thread 35.8-37.5; more threads slightly lower (40-45); prefetch and 2 MB pages make little difference (THP already [always]). User side: no measurable lag up to 4 threads; p99 stutter 5 ms and work x1.38 at 16. **Engine default: 4 cold-reader threads** (reserve the rest for the user); **U1 is revised:** the DDR4 leg reaches the peak with 2-4 streaming threads, so the old 13.31 GB/s was the CPU compute kernel; the work is a CPU dequant-dot kernel that streams like the probe.
- **The bus view (author's proposal):** let the user see the unified bus working: live speed charts in the web interface and a compact one in the terminal. Per token and over time: tok/s; GPU weight-read GB/s (vs the measured 494 ceiling); each CPU reader thread's GB/s and the DDR4 total (vs 47.6 GB/s); hot-bank hit rate and cold experts per token; doorbell hand-off latency; the GPU step vs the cold step on one timeline (overlap visible); prefetch hits into L3; VRAM / RAM in use; the foreground-lag figure from the probe. Source: the engine's own counters (CUDA events, per-thread timers, PMU where available), sampled every token; no extra work on the hot path beyond writing the counters. Lands with the engine (GP3-14) and the server's web interface (GP3-15). **Layout (author 08:03):** CPU on the left (reader threads, DDR4 GB/s, L3 prefetch hits, doorbell thread), GPU on the right (weight-read GB/s vs the ceiling, hot-bank hit rate, VRAM), and below them a green **"Unified bus confirmed"** line for the user's comfort, shown only while every check holds over the last N tokens (proposed, each measured by the engine itself): (1) both legs moved bytes in the same tokens (CPU reads from DDR4 and GPU reads from VRAM, not one idle); (2) the cold step overlaps the GPU step (the share hidden is above a floor set from the probe); (3) CPU-GPU hand-offs answer in microseconds (median under ~20 us, i.e. doorbells, not OS wake-ups); (4) DDR4 total within reach of the probe's measured rate; (5) the periodic exactness check passes (a sampled token recomputed on the plain path matches). Otherwise amber with the failing check named (e.g. "hand-offs slow: doorbell thread not running").

### Carry-over to dense (non-MoE) models that spill onto CPU/RAM (author, 2026-10-28 13:48)

Every cluster is written for the MoE engine but is checked for what it gives a dense model too big for VRAM (layers spilled to system RAM, read every token). DERIVED differences: no hot/cold choice (every spilled byte is read every token), and the order of reads is fully known, so prefetch is 100 % predictable (no router guess). Carries over: U1 (both DRAM legs at once on the spilled layers, rows split CPU/GPU by the measured share), U2 (doorbells per spilled layer), J1 (CPU base plane + GPU 1-bit H plane of each spilled layer, both legs at once), U3/C11 (planes by need), U5 (Q8_Z, codebook, PT), U6 (MTP / speculative: the strongest lever when every token is bandwidth-bound), U8 (one graph per step), U11 (CRC32 prefix store; NVENC checkpoint grid for hybrid/recurrent models). Not applicable: the hot bank, expert prediction, expert skip. Test target when the engine runs: one dense model larger than 12 GB, spilled, against llama.cpp's -ngl split at the same file size.

### Run list before the engine build (updated 2026-10-28 09:4x, after Block M and M2 d)

Each PC run first matches the reference to our code's exact format (docs/PELIAS_NEW_METHODS.md rule).

**Standing rule (author 2026-10-29 00:08): every `pelias kld` run from now on adds `--dump-codes` (and `--dump-kld`, `--route-dump`)** so each run feeds the C6 miss map's prediction data (logging changes nothing: MEASURED, run B). Cost about 326 MB per 100-chunk run on D:; outputs next to the run's KL dump, named after the run; the miss log (data/probe_results/c6) is extended after each run.

1. **R3 activation format (DECISION proposed 10-28 10:3x, from Blocks M2 d / M3 e and the author's paper questions):** per-32 scales everywhere (Block H, eight scales per 256), on the GPU and the CPU, rounded once. Evidence: the per-256 cost is split dense +0.000426 / experts +0.000379, so a cold-expert-only hybrid (per-256 on the CPU) would add only the cold share of +0.000379, below what a 100-chunk test resolves; decode is memory-bound on both sides, so one extra multiply per 32 values is predicted near 0 % of tok/s (prefill GEMM about -37 % on paper, measured in R3). Reference formats ready: gpu256x8, gpu256e, the 6-bit tag (sandbox), --act-cold-path. Previous text: the int8 kernel moves from one scale per 256 to eight per 256 (Block H `q8_k32`, the precision of per-32 in the kernel's layout). Reason: per-256 costs t8r +0.000834 over per-32 (Block M2 d). Measure: instructions per mma and tensor busy against v10 (`ncu`, as Batch C r10), and the reference format matched value for value before any KLD run.
2. **urm merge tool and the _P cold-expert reference option (sandbox).**
3. **Step 2 benches (PC):** bench_cpu (CPU cold kernel GB/s, 8 vs 16 threads, large pages, file location), bench_gpu (PCIe copies during compute, FP16 half-rate check, R4 clock lock).
4. **urm runs (PC):** fp32 and the kernel's activation format, paired against Unsloth's MTP file at the same format; then the _P gate run.
5. **Q7_PT quantizer, then urm-plus** (attention at Q7_PT, below Unsloth's size).
6. **R3 kernel**, then **GP3-08 MTP head in the reference**, then the engine (GP3-14).

### GP3-14 Masterwork step 10: the engine

**Scope:**
- CPU cold kernels, GPU bank, doorbell, decode loop, prefill.
- MTP drafting.
- The hot set re-profiled on the clean base.
- P-E4: NVML used before start and at peak.
- The ceiling read again before and during loading.

**Gates:** E4–E9, G3 (≥ 55.0 tok/s at 12 k), K3 (GP2-01).

**Status 2026-10-31 03:3x (sandbox, not yet run on the GPU):** stages 1-3 coded: GpuEngine (resident dense, hot bank from
a routing record, CPU cold experts on a persistent pool, cache and state carried), generate --engine, engine-check, MTP
drafting with verify / keep (checkpointing recurrence, author 03:1x). Not yet: the doorbell (a device sync per layer for
now), the MTP block's own bank, the chunked DeltaNet kernel for long prompts, the int32 fold with a calibrated scale,
residual planes in the engine, NVML at start and peak. **Order (author 03:18):** (1) the doorbell as designed (pinned host-mapped
flags, the GPU writes the router's picks, a spinning CPU thread runs the cold experts and rings back, the GPU waits on
the flag in-stream; no cudaMemcpy syncs), (2) the MTP block's bank by live pick counts with idle-time promotion (U2;
option 1 of 03:14), (3) DeltaNet: see the per-token kernel in action in the first engine runs, then decide the prefill path. **Done in the sandbox 03:2x-04:0x:** doorbell (4c736fc), MTP live bank (4c736fc), R3 for one-token steps with one device
copy per dense weight (62b84aa; author 03:29: R3-only + batched reads of R3's layout). First PC run (when the author allows): build, pelias_gpu_test
(rung1 checkpoint, rung4 engine + verify / keep), sanitizers, then engine-check on the tiered file with --routes
TIER/tier_v1_x8_routes.bin (prefill 512, decode 128, --mtp-draft 1-3, --float-ref).

### GP3-14c Infinite context loop (author 2026-10-30 12:01-12:15; after the first engine build)

**Idea (author):** a document or an agent's history of any length is read once and kept; the model is never stuck at its window. MEASURED from the served file: context_length 262,144; RoPE on 64 of each attention head's 256 key dims (rope.dimension_count 64, sections 11/11/10/0, freq_base 1e7); 30 of the 41 blocks are DeltaNet (fixed-size state, no positions); KV grows only in the 10 full-attention trunk layers (2 KV heads x 256): about 10.9 KB per token at q8 (DERIVED).
- **Tiers (author 12:15: RAM first, NVMe as overflow):** VRAM holds the 1-bit key codes' page summaries and a bank of hot pages; RAM holds every KV page (q8) and the DeltaNet checkpoints until the RAM budget the probe sets is full; only then do the oldest / coldest pages overflow to NVMe (U11 prefix store, CRC32 keys). The DeltaNet state never overflows (fixed size); its checkpoints are what make a restore possible.
- **Each step:** score the page summaries (VRAM), then the 1-bit codes of the candidate pages, pull the top pages' q8 KV RAM -> VRAM (or NVMe -> RAM -> VRAM) on a copy engine, attend exactly.
- **Positions (way 1, the default):** keys are stored before RoPE; a pulled page is placed at fresh positions inside the trained window, so the stream is unbounded and the model stays in its trained range. Way 2 (a toggle): RoPE scaling at load (YaRN / NTK) for a longer raw window. Way 3 (trainer roadmap): fine-tune at longer lengths.
- **Sizes (DERIVED):** 1M tokens = about 10.9 GB of q8 KV (RAM, or NVMe beyond the budget), codes about 0.8 GB, page summaries about 12.5 MB; a cold pull of 64 pages of 64 tokens about 45 MB, about 15 ms from NVMe at 3 GB/s (MEASURED disk rate), under 2 ms from RAM.
- **Tests, in order:** C1 page recall (sandbox, existing q/k dumps: share of attention mass in the top 1/8 and 1/32 of pages); re-placed pages vs full attention, paired KL on text that fits the window; YaRN x2 kld at short context; needle tests past 262,144 (PG7 needles extended).

### GP3-15 Masterwork step 11: the server on 8062 and the web interface

**Gates:**
- A drop-in replacement for the interim launcher.
- Client replay: captured requests compared by structure with the llama.cpp server on 8012.
- A Visual Studio session works.
- PG14 (Host/Origin checks, session token, `web.fetch` refuses private addresses).

### GP3-15b Server details carried from the specifications

- Persona criterion restored to the step 11 gate: ≤ 12 of 111 not-pass, baseline 9 (ground-up specification).
- One GPU owner at a time: the server unloads before a job, and the governor checks VRAM and pinned RAM across processes (masterwork 11.6, N9).
- O10: policy for special-token text inside user content (pass and count, escape, or refuse). Author ruling.

### GP3-16 Masterwork steps 13–17

Checks before step 15:
- The Windows 10 driver check: does the last Windows 10 driver meet the CUDA 13.4 minimum? Else build with the 13.0 toolkit.
- The CUDA static runtime redistribution licence is read (UNVERIFIED).

Start-up and intake checks (masterwork 4.1, 6.3–6.5, 7.1):
- Quick-pass running time measured.
- APU path (unified memory, hot bank 0) offered only when H1 is measured on one.
- Installer requirement checks (driver, OS, disk, `.wslconfig` cap) report with fix steps.
- Import report (lines read, keys matched, unknown keys, out-of-range values).
- Coach checks (duplicates, empty descriptions, answers parse, every parameter covered).
- USER-FORCED trial (the forced value is allocated and touched for 2 s).
- Dictionary check (every key used by a module, every module key in the dictionary).
- P-E15: ReBAR routes A (Vulkan host-visible device-local) and B (D3D12 write-combined) against zero-copy; `auto_probe` turns one on only on a measured gain.

Universality items:
- O21: the CPU-only gate (measured tok/s against the ceiling recomputed per mix; the 40 tok/s target needs ≤ 1.15 GB per token or about 1.7 accepted MTP tokens per pass, DERIVED in masterwork O21).
- E13: a second MoE through the plug-in, and a dense model passing the K-gate.

- Start-up probes, plan card, intake, coach (U1–U11).
- Universality part a: AVX-512/VNNI, CPU-only mode, dense plug-in.
- Web tools and installers (PG11, PG12, BC7).
- Universality part b: Vulkan, gated extras (zero-copy third branch, ReBAR path, Mode L, LRU).
- Release gates PG1–PG14.

---

## Part 4. Tests on the finished build: tune, then finalise

### GP4-01 The first benchmark (masterwork 16.4, repeated on the release build)

| Item | Detail |
|---|---|
| Files | F0 bf16 (P-Q11 layer-major or the CPU reference); F1 Pelias base; F2 stock Q4_K_M; F3 official 4-bit upload if the author supplies one; F4 Unsloth; F5 Pelias trained (Mode Q); F6 the final PEL recipe |
| Quality | Paired KLD against F0 on WikiText 512, WikiText 4096 and the domain text; same-top-p; PPL as a sanity check only; the author's gauntlet and client matrix; published task benchmarks only where the settings match the model card, otherwise "not comparable" |
| Speed | Decode and prefill at short, 4 k and 12 k context, 3 launches per side with 2 warm-ups, spread reported (launch-to-launch spread was 5–21 %, MEASURED) |
| Memory | Peak VRAM (NVML whole card and process) against the 10.7 GiB ceiling |
| Rule | No invented pass mark: the table is the result |

### GP4-02 Engine tuning sweeps (each an A/B with 3 launches per side; a change is kept only if its gain exceeds the spread)

| Knob | Range | Measure |
|---|---|---|
| Bank size | 6.5, 7.0, 7.5 GiB, and the largest that fits the measured ceiling (P-E4) | tok/s at 12 k, live hit rate, peak VRAM |
| Bank objective | byte-weighted hit against zero-miss share (P-E2 regression of c0, c1) | tok/s |
| Decode threads | 5–8; 1 thread per physical core against SMT (E-A1) | tok/s, spread |
| Poll level, handoff order | acquire/release against seq_cst (E-A2) | handoff p99, tok/s |
| NTA prefetch (E-A3), L2 persisting window (E-A4), 2 MiB pages (P-E11) | on/off | tok/s, cold read GB/s |
| Bypass N, `-ub` | 9, 16, 32; 2048 or 4096 | prefill tok/s, peak VRAM, crash sweep 1–64 tokens |
| Zero-copy third branch | off/on with the restated gate (masterwork 10.4) | combined GB/s, tok/s |
| MTP draft count | 0–3 | acceptance rate, tok/s; greedy equality |
| Activation rounding per group | per GP2-05 | K3 paired KLD and tok/s together |
| KV type | per GP2-06 | needles, KLD, VRAM |

### GP4-03 Capability and acceptance (masterwork 16.3, engine E10, 03 blueprint 8.6)

- Persona probe: not-pass ≤ 12 of 111 (baseline 9).
- Needles 12/12 to 120 k.
- Tool-call test.
- Loop test: replies that hit the output cap, counted.
- Thinking on and off.
- Client replay (Visual Studio Copilot, odysseus-ai, web UI, Ollama alias).
- A real Visual Studio session with the server log open.
- The four client unknowns, read from the server log: does the client send a thinking-off switch; does it send its own system prompt every turn; how does it declare tools; does it truncate context on its side.
- An Ollama-style client: its template renders byte-identical to the server's rendering of the same conversation.
- SSE heartbeat: a long prefill does not time the client out (the 15 s default is untested).

### GP4-04 Hardware leftovers (only what the engine's choices depend on)

- Stock-clock repeats of P-H3, P-H4, P-H10.
- The ReBAR on/off pair (optional, O17).
- Power plan.
- The VRAM disturbance margin (sets the ceiling margin).
- P-M5 (zero allocations in the hot loop).
- P-E4 per bank and context.
- The Windows timer fix with P-M3 again.
- Three launches per side for any Windows-against-WSL speed claim.
- The PCIe link width read under load (4.0 × 16 is assumed from throughput; never read from the GPU tool under load).
- Native Linux floors (E-A6, E1), now that native Linux is no longer held: the floor tests 02, 03, 04, 07 and the base cell.

### GP4-05 Release gates

- PG1–PG14 and BC1–BC7 (masterwork 16.3).
- PG10: runs with the oracles deleted.
- A fresh clone on a clean machine installs and passes the CPU tests from the README alone.

### GP4-05b After the release: the future-hypothesis tests (masterwork 21; nothing starts without the author's word)

| Test | What |
|---|---|
| P-W1 | Energy per token: a plug-in wall meter during the speed runs, J/token at short, 4 k and 12 k; plus the idle cost of a spinning doorbell |
| P-G1 | Games: a small Vulkan scene with and without the bus; average fps, 1 % low, frame-time variance |
| H-U1 | The bus outside LLMs: port one workload to `libpelias_substrate`, compare with its best conventional version |
| H-U2 / H-U3 | A tiny CPU model keeps the bus tuned, against a coded rule. The model's CPU time per decision must be below the miss time it saves; compare three residency policies |
| The 14 per-field probes (masterwork 21.3) | Each against a stock baseline with unchanged output, measuring its own f and k |
| D-1B | Dense models at 1-bit + planes (author 10:24-10:31; explore after the build): 1-bit base fully in VRAM, planes from RAM (every weight read each token, so selective planes by default); fill spare VRAM with full layers; the 1-bit base as a self-drafter checked by the full stack in batches. DERIVED on the 4070, no planes: 27B ~56-62 tok/s, 70B ~24-27 tok/s (from measured 1-2-bit GPU runs at 48-55 % of spec bandwidth); plain 1-bit weight error 0.366 (MEASURED sandbox), so quality needs the planes or the drafter role. Measure: draft acceptance on a small dense model, then kld + speed |

### GP4-06 Freeze

1. The final PEL recipe and its numbers are written into the GGUF metadata and the docs.
2. The benchmark table is regenerated from the release build.
3. Every result carries its log path.

---

## Theories to get closer to the ceilings

Every item is a hypothesis: it names the measurement behind it and the test that settles it.

### Speed

The modelled ceiling is 88.2–92.0 tok/s (DERIVED). The best recorded run is about 54 tok/s (MEASURED, patched llama.cpp tier build).

1. **Replace the per-layer handoff.**
   - Measured gap: the recorded runs spend 4.6–5.0 ms per token in handoff, against 0.25 ms in the model.
   - The tier build waits on a stream sync of about 94 µs per layer; 40 layers × 94 µs = 3.76 ms (DERIVED). The doorbell answers in 2.4 µs (MEASURED, P-E16), so 40 × 2.4 µs = 0.10 ms (DERIVED).
   - **Theory:** about 3.7 ms per token comes back from this alone.
   - Test: engine timers (GP3-14).
2. **Make the CPU and GPU actually overlap.**
   - Measured gap: zero overlap. The CPU's cold-expert work took 10.1 ms against the 2.86 ms modelled.
   - **Theory:** our own AVX2 cold-expert kernels close most of that gap: one thread per physical core (E-A1), non-temporal prefetch to protect the 96 MB L3 (E-A3), int8 activations where GP2-05 allows. Running them at the same time as the GPU removes most of the rest.
   - Test: GP3-10 (≥ 27 GB/s effective) and GP4-02.
3. **Wide loads for GPU reads of cold experts from system RAM.**
   - Measured gap: the compute kernel reads at 9.59 GB/s; the first 16-byte-load version reached 13.31; the copy ceiling is 26.2.
   - Test: P-E10 (GP3-10).
4. **More zero-miss layers.**
   - 36.1 % of layer-steps already need no cold expert (MEASURED).
   - **Theory:** filling the bank for zero-miss layers, rather than for raw hit rate, saves more time if a layer's first miss costs much more than later ones.
   - Test: P-E2 measures that cost; the bank objective is swept in GP4-02.
5. **Spend bytes where they buy speed.**
   - Denser dense types take bank room: about −1.3 tok/s for T8 (DERIVED, PEL_QUANT_DECISION_DATA 9.3).
   - **Theories:**
     - Keep `token_embd` in host RAM (one row per token), so its 8-bit cost is file size only (B10).
     - Use 6-bit where 6 ≈ 8 is proven.
     - Lower-bit cold experts mean fewer bytes read on a miss (GP3-02).
     - The 3-bit mix fits 24.7 % more experts per GiB of bank (DERIVED) for 0.0043 more drift (MEASURED).
6. **MTP drafting.**
   - It gave 1.13–1.17× in the tier build (MEASURED).
   - It is required in the engine (step 10), with greedy-equality and KL gates.
7. **CUDA graphs and the flush rule.**
   - A graph launch costs 0.716 µs against 4.70 µs per eager kernel (MEASURED, P-H9).
   - Flushing after each submission lifts copy/compute overlap from 0.47 to 0.92 (MEASURED, P-H12).
   - **Theory:** both are already in the design and remove most launch overhead per token.
   - Test: GP3-14.
8. **Platform.**
   - Windows-native was 5.1–5.2 ms per token faster than WSL in the reference cell. That is two launches against one, so it is not a claim yet; three launches per side settle it (GP4-04).
   - Native Linux floors are untested (GP4-04).

### Accuracy

The bf16 model is zero drift. The bar to beat is Unsloth UD-Q4_K_M.

1. **Per-layer bit allocation**, which is Unsloth's "dynamic" method. Unsloth's expert layout alone was worth 0.0013 for 41 MB (MEASURED). Test: GP1-12.
2. **An importance matrix built from our own calibration text.** It is worth about 0.0006 on a good layout (MEASURED with Unsloth's matrix), and our text may fit the intended use better. Test: GP1-13.
3. **IQ4_XS gate/up experts with the matrix (T9).** Frees 687,865,856 bytes (DERIVED) to spend on sensitive groups. Test: GP1-05.
4. **AWQ-style scale folding into the norm weights.** Exact before quantization, and needs no new data (it uses the importance-matrix sums). Test: GP3-03.
5. **GPTQ-style error feedback inside the Q4_K/Q6_K blocks**, dense groups first. The files stay readable by any loader. Test: GP3-04.
6. **Protecting routing.** If the worst tokens come from changed expert picks, weight the router's inputs and low-margin experts during quantization. Test: GP1-06, then GP3-06.
7. **Repair training against a bf16 teacher.** Train the core through the exact quantizer, with experts frozen. Test: GP3-13.
8. **An activation-rounding policy per tensor group.** Keep 16-bit activations on the groups where int8 costs drift; serving then gives up nothing the file won. Test: GP1-03, which sets the policy in GP2-05.
9. **One activation format for CPU and GPU (author, 10-27).** Groups of 32 with their own scale, laid out as 256-value blocks for the CPU: GPU-level rounding cost (+0.0006 instead of +0.0024) on the CPU expert path too, one rounding for both sides. Tests: GP1-15, GP1-16 (Block H).

---

## Appendix A. ID map: old names to the names in this file (fixes the collisions found by the readers)

| Old name (where) | Meaning | New ID |
|---|---|---|
| F1 (today's plan) | Per-token dump + paired compare | done in 3a272b3; first real use GP1-01 |
| F2 (today's plan) | Doc fix: 25,500 scored tokens | done |
| F3 (today's plan) | llama.cpp `--no-op-offload` | GP1-02 |
| F4 (today's plan) | 4096-context base | GP1-09 |
| F5 (today's plan) | Domain base | GP1-10 |
| F6 (today's plan) | Activation-rounding sweep | GP1-03 (code done in 3a272b3) |
| L1 / L2 / L3 / L4 (today's plan) | AWQ folding / GPTQ feedback / routing flips / repair training | GP3-03 / GP3-04 / GP1-06 + GP3-06 / GP3-13 |
| Block 1 / Block 2 (today's messages) | Bench set through Pelias / Unsloth + imatrix + IQ4_XS | GP1-04 / GP1-05 |
| F1–F9 (trainer scaffold, v0.2 findings) | Findings of the old trainer | keep as "v0.2-F1…F9" |
| L1 (trainer scaffold) | Chunked cross-entropy kernel | keep as "kern L1" |
| K1–K3 (trainer) and P-K1–K3 (probes) | KL gates and kernel probes | KL gates renamed "KG1–KG3" in GP2-01 |
| G1–G18 (claims), G1–G18 (server gaps), G1–G7 (trainer gates), G0–G3 (prototype) | Four lists | "claim-G", "gap-G", "gate-G", "proto-G" |
| E0–E13 (engine phases), E0–E7 (prototype track 2), E-A1–A7 (audited hypotheses), E1–E5 (elementwise kernels), E-A…E-F (export options) | Five lists | "phase-E", "proto-E", "E-A" kept, "kern-E", "export-E" |
| R1–R16 (risks), R1–R22 (requirements), R1–R3 (router kernels), R1–R4 (checkpoint registries), [sync R1–R4] (rulings) | Five lists | "risk-R", "req-R", "kern-R", "ckpt-R", "sync-R" |
| T1–T8 (O20 tests) and T0–T14 (trainer stages) | Two lists | "O20-T1…T8" and "stage-T0…T14" |
| H1–H8 (03 blueprint), H6 (engine accuracy gate), H9/H11/H12 (engine notes) | Different things | "bp-H1…H8", "gate-H6", "eng-H9/H11/H12" |
| M0–M10 (03 blueprint milestones) | General-tool milestones | "bp-M0…M10" (they follow Part 3) |
| P-M5 | Zero allocations in the hot loop (scaffold, universal) against RAM busy % per memory level (masterwork O16) | "P-M5" keeps the scaffold meaning; the masterwork one becomes "P-M5b" |
| P-Q11 | bf16 reference logits by layer streaming (masterwork) against a GEMM gate probe (ground-up specification) | masterwork meaning "P-Q11"; the other "P-Q11g" |
| PEL-6 / PEL-4 / PEL-5 / PEL-1… (experiment shapes) and PEL4 / PEL3 (family names = bit width) | PEL-6 is `ssm_out` at Q8_0, not a 6-bit mix | experiment shapes renamed "shape-6, shape-6+, shape-4, …"; family names stay PEL4, PEL3 |
| H1–H5 (masterwork hardware requirements) = R1–R5 (universal build master) | The same five requirements | "bus-H1…H5" |
| PG1–PG14 (universal numbering) against the engine runtime specification's PG list (its PG3 is the tokenizer) | Two gate lists | universal numbering governs; the engine list becomes "rt-PG" |
| Step numbers: masterwork 15.2 (R0, 1–17) against the ground-up specification 9.2 (1–15) | Different content per number | masterwork 15.2 governs |

## Appendix B. Decisions waiting on the author (the tests above give each its facts)

| # | Decision | Facts from |
|---|---|---|
| B1 | D9/D33 (stock map, no imatrix) against PEL mixes with an importance matrix as the base | GP1-05, GP1-10, GP1-13 |
| B2 | Whether PEL types may differ from ggml's quantizer bytes (L2, AWQ folding) while stock types stay byte-identical | GP3-03, GP3-04 |
| B3 | The calibration text (and that it is never an evaluation text) | GP1-13 |
| B4 | The domain evaluation text | GP1-10 |
| B5 | Activation-rounding tolerance per group | GP1-03 |
| B6 | KV cache default | GP1-11 |
| B7 | Per-expert type container (a Pelias-only extension) | GP3-02 |
| B8 | Parameter conflicts (threads, prio, thinking, max_tokens, poll, prefetch) | GP2-07 |
| B9 | Whether a Pelias-made base replaces the llama.cpp base for final rankings | GP1-07 |
| B10 | `token_embd` placement (host RAM, so its Q8_0 costs file size only) | PEL_QUANT_DECISION_DATA 9.2 |
| B11 | The 7.5 GiB bank as a ruling or a derived value (O1) | GP4-02 |
| B12 | LoRA scaling: α/r or rank-stabilised α/√r. They differ 4 × at r = 16, α = 32. The Mode Q formula and v0.2 use α/r | GP2-02b; ablation T-A3 |
| B13 | The size bar with MTP required (O18): T8 is 21.97 GB without blk.40 and 22.50 GB with it. Unsloth's 22.13 GB file has no MTP head. Proposed: compare without blk.40 (like for like), and report the blk.40 cost separately | PEL_QUANT_DECISION_DATA 1, 2.3 |
| B14 | Which reading is the O20 bar: llama.cpp's (8-bit activations) or Pelias's fp32 reference. Proposed: Pelias's, with Unsloth re-measured the same way (GP1-05 run a) | GP1-03, GP1-05 |

## Appendix C. Time and disk summary for Part 1 (ESTIMATE unless marked)

| Block | Author's machine time | New disk |
|---|---|---|
| A | about 25 min | under 1 MB |
| B | about 2 h 10 min (GP1-02 15–45 min; GP1-03 88 min) | under 3 MB |
| C bench (GP1-04) | about 80 min (MEASURED parts: 253.5 s quantize, 1309 s Pelias) | +62.2 GB on C: |
| C Unsloth (GP1-05) | about 92 min | +about 44 GB on C: (T8+imat, T9) |
| D | tools first (Claude), then GP1-06 about 30–60 min, GP1-08 about 30–45 min | +about 0.2 GB |
| E | GP1-09 about 2 h base + 3 × 45–60 min; GP1-10 57 min base + 4 × 26 min; GP1-11 about 30 min | +25.4 GB (4096 base) + 12.7 GB (domain base) on D: |
| F | GP1-12 about 2 h 30 min; GP1-13 about 70 min | +22–45 GB on C: (each screening file gets a new name; nothing is overwritten; the author decides deletions between steps) |

Free space after Part 1, if nothing is deleted: C: about 191 − 62 − 44 − 45 ≈ 40 GB, under the 50 GB warning line. **Before Block F, the author decides which candidate files to delete** (only on the author's word).

## Appendix D. Sources used for this plan

- **Read in full this session:**
  - the private master (2026-10-25 version, 8,542 lines);
  - the trainer CUDA master scaffold (2,515 lines);
  - the trainer notes (267 lines);
  - the engine master scaffold (482 lines);
  - the engine blueprint, system master, the 10-24 server handoff and runbook, the 10-21 handoff;
  - the 02 VRAM guide and 03 blueprint;
  - the hardware research plan;
  - the parameter dictionary and server parameter CSV.
- **Read in full by reader agents** (findings folded in; full reports in the author's backup folder, 2026-10-27/bk_1413):
  - the masterwork (2026-10-27);
  - the universal build master;
  - the ground-up master specification;
  - the research update;
  - the Accidental Find;
  - a second reader of the six newly attached files.
- **Fable's read-only review** of the plan, the code and Blocks A and B (2026-10-27): every finding checked and fixed in cc22071.
- **Repository:** `docs/PEL_QUANT_DECISION_DATA.md` v5, `docs/PELIAS_SYSTEM_MASTERWORK.md` 15–16, `logs/BUILD_LOG.md`, the RUNBOOK and TALLY, the sweep logs `sweep/q-*.log` and `summary*.txt`.

## Appendix E. Open items from earlier eras, and where each one went (so nothing is lost)

| Item (source) | Disposition |
|---|---|
| T0 log backup of `~/p1` (10-21 handoff) | Carried: GP1-00 preflight asks whether `~/p1` is in the backup folder; a copy command is given if not |
| T1 "no key 12", T2 bypass-9 sweep, T4 crash fix, sanitizer results (10-21 handoff) | Tier build only, which D37 retires from the product. Carried as a design rule for the Pelias engine: unique slot ids, no duplicate zero-slot id, and the crash sweep of GP4-02 |
| T3 H6 yardsticks (10-21 handoff) | Done 2026-10-22 (stock16 0.017353, stock1024 0.018286, tier8_nobank 0.0017). Superseded as a gate by GP2-01 |
| H6 definition (10-21 handoff) | Ruled 2026-10-22 (float spread governs). Replaced by the paired gates of GP2-01 |
| Bank-size sweep, B1 rerun, drift repeat before trusting `-t 7` / `--poll 100` (10-21) | GP4-02 |
| Match the 11,294 MiB reading to its log line (10-21) | Superseded by D28 (the ceiling is measured at start). Kept as a curiosity only |
| Fold the pin edit into `pelias_tier_v1.patch` (10-21) | Record only (D37). No action |
| Ledger claims: "MMVQ ignores −1"; RESULTS §3 stream order and 9.59 GB/s (10-21) | MMVQ: tier only, retired. 9.59 GB/s: carried to P-E10 (GP3-10 GPU kernels) |
| Lane D promotion (70/30 copy-aware LRU) (10-21) | GP4-02 bank policy |
| Remove the PELIAS-DEBUG fprintf (10-21) | Tier only, retired |
| Trademark web check for "Pelias" (10-21) | Carried to release (GP4-05) |
| Defender exclusion result (10-21) | Carried to the installers (GP3-16) |
| Q6_K shared-memory kernel (10-21) | GP3-10 (P-E10) |
| The 12 parked trainer problem areas (10-21) | In the trainer scaffold; GP3-12 |
| Finish-5 (overlap under real CPU load), Finish-6 (KL and accuracy) (10-21) | Finish-5: GP3-14 engine gates (E6, P-E2). Finish-6: Part 1 |
| The research queue R1 (10-21) | Tier crash analysis; retired with the tier build |
| Research plan E0.1–E0.6 (10-23) | E0.2 timeline: engine timers at GP3-14. E0.3 live hit rate: GP4-02. E0.4 overlap and 1T/2T: GP3-10, GP4-02. E0.5 wide load: GP3-10. E0.6 huge pages and large pages: GP4-02. Its gates are rewritten without the retired 57.0 |
| Research plan E1.1–E1.3 (native Linux) | GP4-04 |
| 03 blueprint H3 (long rows compute-bound, short rows copy-bound) | GP3-12 (rung 5 stall measurements at several step sizes) |
| 03 blueprint H4 (faster PCIe), H5 (more resident layers on a 24 GB card) | Need other hardware. Parked with the Threadripper notes |
| 03 blueprint H6 (activation memory linear to 16 k and 32 k) | GP3-12: P-M2 at 8 k, 16 k and 32 k step windows |
| 03 blueprint H7 (speed-up against a conventional recipe on the same card) | Before any "x times faster" claim: GP4-01 adds a QLoRA-or-offload baseline run if the author wants that claim |
| 03 blueprint M0–M10 | M1 = GP3-12b parity; M2, M3 = GP3-09 plug-in and E13; M4 = the planner (step 13); M5 = levers (GP3-12); M6 = 16 k / 32 k (GP3-12); M7 = third family (step 14); M8 = disk tier (parked); M9 = export and serve for two models (step 14); M10 = packaging (GP4-05) |
| 02 VRAM guide: smoke stage, guard rails, data audits, loop checks, persona-test design | Guard rails and smoke: the trainer's governor (GP3-12). Data audits: GP3-05. Loop checks and persona design: GP4-03 |
| 02 VRAM guide: confirm the 8.02 GiB main-run peak against the full log | v0.2 Python only. Kept as a note; the new trainer logs the step maximum (GP3-12b) |
| The Accidental Find: launcher banner said "thinking switched off" while it was on | v0.2 launcher only. Noted; no action in the C++ product |
| The Accidental Find 14.1-a to h, 15.2 parity gate, 15.3 maths review | GP3-12b |
