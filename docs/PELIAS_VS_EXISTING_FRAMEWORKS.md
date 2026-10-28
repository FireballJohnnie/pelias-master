# Pelias: what we found by thinking outside the box, and how it compares

*Draft of 2026-10-28 16:1x. One result is still running (the live H gate, step 1 of the 1-bit decoder); its line is marked. A second update follows when the 1-bit decoder runs as a real kernel.*

## Plain-language summary

A language model is billions of numbers. To fit it on a home computer, every tool stores those numbers with fewer bits, and every bit removed costs a little accuracy. The best-known small file for this model, Unsloth's, is the bar we set out to beat at the same size or smaller.

Three things made the difference:

1. **We rotate the numbers before storing them.** A simple mathematical shuffle (Walsh-Hadamard, in groups of 32) spreads out the few extreme values that make rounding hard. It cut the file's error by about 9 %, and it makes the 8-bit maths the graphics card is fastest at cost nothing in accuracy.
2. **We store 7 bits and add the 8th only where it matters.** Computers like 8 bits; 7 is "odd" and normally awkward. We keep a 7-bit base everywhere and save the 8th bit only on the quarter of each block where it removes the most error. When the computer calculates, every number becomes an exact 8-bit number again, so the fast hardware sees what it expects.
3. **We put idle parts of the chip to work.** The extra bit can be added with a simple "count the ones" instruction that the processor's integer units can do while the main maths units are busy, on either the CPU or the graphics card, and the base and the extra bit can even live in different memory.

**Where it stands:** our 7.53-bit file is **236 MB smaller than Unsloth's and slightly more accurate** (0.012536 against 0.012864; not yet a statistically settled win), and it is **as accurate as our own 8-bit file**. The speed is the part still to be measured in the engine.

## Scoreboard (whole model, measured)

Paired runs over the same 25,500 tokens, 8-bit activations as the engine uses them, every number checked again from the saved per-token results. Lower KL = closer to the original model.

| File | Dense weights | KL | vs Unsloth (0.012864) | Size vs Unsloth |
|---|---|---|---|---|
| Unsloth MTP UD-Q4_K_M (llama.cpp's GPU path) | Q8_0, 8.5 bits | 0.012864 | the bar | 0 |
| urm (Unsloth's types, our rotation) | Q8_0 | 0.012499 | -0.000365, not resolved | -6 KB |
| urm_Z | **Q8_Z** (zip scales), 8.25 bits | 0.012402 | -0.000462, not resolved | -61.0 MB |
| urm_Q7 | Q7_Z, pure 7-bit, 7.25 bits | 0.013524 | +0.000660, **resolved worse** | -305.1 MB |
| urm_ZT (first version) | Q7_ZT, 7.53 bits | 0.012870 | -0.000006, exact tie | -236.4 MB |
| **urm_ZT2 (fixed)** | **Q7_ZT, 7.53 bits** | **0.012536** | **-0.000328, not resolved** | **-236.4 MB** |
| urm_Z with the live H gate (25 %) | Q8_Z stored, ~7.5 bits read | *running* | *at chunk 20: 0.01213 vs urm_Z 0.01189 (partial)* | -61.0 MB stored |

## What makes the system possible (the chain)

Each link depends on the one before it:

1. **Rotation** (dense tensors only). Makes rounding cheap, both for the stored weights and for the 8-bit activations. Without it, today's 7-bit result would land about 0.0008 worse (paper: 24 % more weight error measured on attn_qkv, plus the activation cost the unrotated Unsloth file pays).
2. **The format** (Z, T). Cheaper scales (Z), then the 8th bit as a gate per 32-block (T). Every value still rebuilds to an exact int8.
3. **The read rule** (H). Decides, per expert or per token, which bits are read. The expected-size estimate tells exactly what a dropped bit costs; a missing bit is filled with 0.5, never a whole number.
4. **The decoder** (T-decode). Adds the 8th bit by AND + popcount, exact, on units that are otherwise idle, from memory that can be separate from the base.
5. **The bus** (the unified CPU + GPU system). The CPU and GPU read the same DDR4 at once; the CPU computes the cold experts, the GPU the hot ones; predicted experts are fetched into L3 while the GPU works.
6. **The engine** (GP3-14). Puts 1-5 into one token loop. Not built yet; it is where speed is measured.

## Outside-the-box wins

Labels: **MEASURED (whole model)** = paired run on the PC; **MEASURED (sandbox)** = real weights or unit tests; **PAPER** = derived or checked read-only by Fable; **PLAN** = designed, not built or not run. Numbers in brackets are entries in `docs/PELIAS_NEW_METHODS.md`.

### A. Executed and measured on the whole model

| Win | Result | Label |
|---|---|---|
| Rotation per 32 on the dense inputs [1] | Removes the dense 8-bit activation cost: -0.001186 vs the same rounding unrotated (t -3.23); equal to no rounding | MEASURED |
| Dense weights rotated from bf16, then quantized [2] | -8.9 % drift vs the unrotated twin (-0.003454, t -6.06) | MEASURED |
| Rotated file + 8-bit activations [3] | Free at one scale per 32; Unsloth's unrotated file pays +0.000633 for the same switch | MEASURED |
| Rotation on the int8 kernel's format (one scale per 256) [7] | Our file pays +0.000999, Unsloth's +0.003434; ours ahead by 0.001726 (t -5.40) | MEASURED |
| Rotate the dense tensors, not the experts | Rotating the experts made the file worse (+0.003979, R2b), so they stay as stored | MEASURED |
| Q8_Z, the author's zip scales with 8-bit values [8] | Same accuracy as Q8_0 at -61 MB | MEASURED |
| Q7_ZT, the 8th bit as a gate [9] | -236 MB, -0.000328 vs Unsloth, no difference from the 8-bit file; pure 7-bit, by contrast, loses (resolved) | MEASURED |
| Catching our own over-fit | The first Q7_ZT weighted its level search by the calibration text and spiked on two chunks (0.0424 / 0.0397); the fix (plain levels, square-root importance for the gate only) removed the spikes and moved the result from a tie to ahead | MEASURED |
| Fused int8 kernel (v10) [4] | 1.39-1.84x the bf16 path, all 20 shapes, quantization fused into RMSNorm | MEASURED (kernel bench) |
| Grouped int8 launch for all experts (v12) [5] | 2.5-3.0x the bf16 kernel grouped the same way | MEASURED (kernel bench) |
| Dual bus | CPU and GPU reading RAM at once: 26.56 + 21.06 = 47.62 GB/s, 93 % of the 51.2 GB/s peak | MEASURED (probe) |

### B. Working in the sandbox

| Win | Result | Label |
|---|---|---|
| 1-bit decoder, T-decode [11] | Same integers as the int8 route on every block (gcc, clang, ASan); split base / H storage round-trips exactly; AVX2 activation planes equal the portable ones | MEASURED (unit tests) |
| The author's popcount H on the integer units | Exact: sum h x = sum_k 2^k popcount(H AND x_k) - 128 popcount(H); POPCNT runs 4 per cycle on Zen 3 (published) | MEASURED (unit test) + published rate |
| 0.5 fill through the block sum [10] | A missing 8th bit costs 43 % less than any whole-number fill, exact on int8 | MEASURED (sandbox) |
| Odd grid for blocks without the 8th bit | 0.000105 vs 0.000116 | MEASURED (sandbox) |
| Gate curve | 10 % of the 8th bits remove a third of the 7-bit gap; 25 % about 60 %; a fixed 2-of-8 quota equals a free ranking | MEASURED (sandbox) |
| Planes by need, expected-size estimate [12] | Exact at 1.00x where the worst-case bound is 17-33x off | MEASURED (sandbox) |
| J1 split (CPU base + GPU H plane) | Integer-exact | MEASURED (unit test) |
| fp16 partial sums (texture path) | 0.22 % output error against the format's own 0.54 % | MEASURED (sandbox) |
| _PT tag bits [6] | Spare word bits used as scale tags: 28-67 % less error than leaving them empty | MEASURED (sandbox) |
| Live H gate (`--h-gate live:F`) | Built and unit-tested; whole-model run in progress | MEASURED (unit test) / running |
| Cold-expert routing shares | 6.3 unpredicted cold picks per token, 52 % under router weight 0.08; 67 % of cold picks predicted one layer ahead | MEASURED (routing record) |

### C. Planned, with the math checked

| Idea | Paper result | Label |
|---|---|---|
| The unified bus: 2 MB pages + 8 pinned streams, always-spinning doorbells, predicted experts prefetched with all planes into L3 during the GPU step, the rest one layer late | Token time 18.5 ms -> about 8.65 ms (54 -> about 116 tok/s); ceiling about 119 with the GPU's own overhead, 193 as the memory floor | PAPER (Fable) |
| J1 split-plane cold expert | Full-file accuracy on the unpredicted cold experts for about +0.1 ms; a speed gain only if the CPU leg stays under about 35 GB/s | PAPER (Fable) |
| GPU 1-bit tensor path (and-popc mma) | Right unit for batched MTP verification (5-8 tokens); at single-token decode the CUDA cores do it in about 0.7 us | PAPER (Fable) |
| MTP block as Q3 base + H, H read on the first draft only | About -140 MB; draft 1's acceptance multiplies every later draft | PLAN (needs GP3-08) |
| CRC32 prefix store with a DeltaNet checkpoint grid (int8 or rotated int4 + rANS on idle CPU cores) | A repeated 4K-token prompt skips about 1.8 s; a partial hit about -400 ms | PAPER (Fable) |
| Texture units decoding Q8_Z as RGBA8 | About 10x fewer instructions at decode; prefill depends on the fp16-accumulate rate | PAPER (Fable) |
| Carry-over to dense models that spill onto CPU/RAM | Bus, J1, planes by need, Q8_Z / Q7_ZT, MTP all apply; prefetch is 100 % predictable there | PAPER |

### D. The author's ideas from the int8 v10 / v12 runs (2026-10-26 to 10-28), and where each went

| Idea | What came of it | Status |
|---|---|---|
| One activation format for CPU and GPU (Block H): round once, in groups of 32 laid out as the 256-block the CPU walks | Became the gpu256x8 reference format and the decision for the R3 kernel: one scale per 32 everywhere (the per-256 format costs +0.000999 measured) | MEASURED (format); kernel PLAN |
| Rotation (Block J): spread the outliers before rounding | R1, R2, R2b: methods 1-3 and 7, the base every later win stands on | MEASURED |
| The dual-bus power test | GPU at 45-47 W while reading RAM, CPU at about 58 W: the power and the units are there, waiting on data; led to the idle-units work | MEASURED (probe) |
| Idle units and an idle-work scheduler (GP3-13b) | Copy engines, warp specialization, texture units, L2 persistence, CPU cores, L3: the source of the unified-bus clusters and of T-decode | PAPER / PLAN |
| The "language": codes written at build time (Block M) | Price list of speed knobs: deferral and near-tie swap ruled out; 25 % neuron skip the only cheap one | MEASURED |
| Hot / cold bits ("the over/under ledger") | Hot experts read every plane, cold ones the base; became the H read rule and, with prediction, "all planes into L3 for predicted cold experts" | PAPER + sandbox pieces |
| Zip scales | Q8_Z | MEASURED |
| Tags in the spare bits (_PT) | 28-67 % less error; then superseded at 7 bits by the gated 8th bit | MEASURED |
| "Look at it on a binary level"; "H is like a gate" | Q7_ZT and the 1-bit decoder | MEASURED |
| "Fill the empty bit so the int8 kernel throws less error" | The 0.5 fill (43 % less error) | MEASURED (sandbox) |
| "Flood L2 and L3", the unused CPU parts | Measured limits (GPU L2 caches VRAM, not DDR4) and the prefetch rule that does work | PAPER |
| CRC32 + the video units for the KV cache | CRC32 and the checkpoint grid pay; Fable showed rANS on idle CPU cores beats NVENC here | PAPER |
| "Keep the 1-bit decode as a path the int8 tensor can still see as 8" | The fold route rebuilds exact int8; the 1-bit route gives the same integers | MEASURED (unit test) |

## Proven not to work (kept so nobody retries them)

| Idea | Result |
|---|---|
| Pure 7-bit everywhere | +0.001122 vs the 8-bit file, resolved worse |
| Importance-weighted level search | Over-fits the calibration text (two chunks at 2-3x) |
| Plain two-stage E8 codebook | Ties the K-quant line at 4 bits, no gain without Hessian-aware rounding |
| Empty activation bit planes with one scale per 32 | None occur |
| Expert deferral; near-tie expert swap | +0.0042 and +0.0017, resolved worse |
| Rotating the routed experts | +0.003979 |
| Video units (NVENC/NVDEC) for the live KV cache | 1.3 s per token |
| AES / SHA / CLMUL, vector gather, 1 GB pages, L3 as a fixed expert bank | No gain on the math |

## Compared with existing frameworks

Limited to what their formats do; we have not tested the other projects themselves.

| Area | Existing (format facts) | Pelias (labelled) |
|---|---|---|
| Between 6 and 8 bits | llama.cpp goes from Q6_K (6.56 bpw) to Q8_0 (8.5 bpw, 0.5 bpw of it scales); Unsloth keeps sensitive dense tensors at Q8_0, a whole tensor at a time | Q8_Z 8.25 bpw (same accuracy, MEASURED); Q7_ZT 7.53 bpw with the 8th bit per 32-block (no difference from Q8_Z, MEASURED) |
| Importance matrix | Weighted K-quant search; none for Q8_0 | Used only to rank the gates, square-rooted; a weighted level search over-fit in our own test (MEASURED); not shown for other tools |
| Bits and memory | One tensor, one memory pool, one unit | Base and 8th bit splittable across memories and units, exact (MEASURED); speed PLAN |
| Rotation | QuaRot, SpinQuant, QuIP# rotate linear layers of the (mostly dense) models they report | Dense tensors only on this MoE model; experts left as stored (MEASURED) |
| Bit-serial decoding | Exists (T-MAC, Any-Precision LLM) | New here: the per-32-block gate, the 0.5 fill, the split across memories (MEASURED pieces) |
| Training | QLoRA re-quantizes after merging | Training on the served bytes and int8 backward passes: PLAN, untested |

## Next

1. The live H gate result (running; final about 21:15 UTC).
2. The 1-bit decoder as a real kernel: the 1-bit mma rate on the 4070, the CPU popcount kernel timed, the GPU T-decode kernel timed against the plain int8 route. **This document gets its next update when that works.**
3. The PC bench session (the bus clusters), then the engine.
