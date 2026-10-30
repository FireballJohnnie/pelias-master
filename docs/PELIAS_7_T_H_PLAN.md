# Making 7 bits viable: what was done, and the plan for T, H and the 1-bit decoder

Author's request, 2026-10-28 14:48; updated 2026-10-29 01:2x (section 9: light vs heavy, rotation coverage, MTP rule, the idle-units tour) and 2026-10-30 15:0x (section 10: planes on the 7-bit and 1-bit bases, the int4 core, Q4_TG2 as the tiered floor, BitNet). Every number is labelled: **MEASURED** (sandbox on real weights, or a whole-model run on the PC) or **DERIVED** (worked out on paper, to be measured). Engine format = 8-bit activations, eight scales per 256 (gpu256x8), paired against the same 25,500 tokens.

## 1. The goal and the bar

- Replace the 8-bit dense tensors (259 tensors: attention, SSM, shared expert, output; all rotated per 32) with something smaller that keeps the accuracy.
- **Bar:** Unsloth's MTP UD-Q4_K_M on llama.cpp's GPU path, KLD **0.012864**.
- **Current best:** urm_Z (dense as Q8_Z, 8.25 bits per weight), **0.012402**, **-61.0 MB** vs Unsloth (MEASURED).

## 2. What has been done for 7 bits (in order)

| Step | What | Result | Why it was done |
|---|---|---|---|
| 1 | **Q8_Z** (zip scales: fp16 per 256 + 6-bit level per 32, int8 values; 8.25 bpw) | Whole model: no difference vs Q8_0, **-61 MB** (MEASURED) | The new 8-bit floor every 7-bit rung has to beat |
| 2 | **Q7_PT** (7 + 7 + 2 tag bits per word, 8.25 bpw) | Loses to Q8_Z at the same bits: 0.000040 vs 0.000029 (MEASURED) | Tags help, but a full 8th value bit helps more for the same cost |
| 3 | **Q7_Z** (Q8_Z's scales, plain 7-bit values, 7.25 bpw); type id 201 | Sandbox: 4.0x Q8_0's error. Whole model: **+0.001122 vs urm_Z, resolved worse; +0.000660 vs Unsloth, resolved worse**; -305 MB (MEASURED) | Tests "pure 7". Pure 7 fails gate 1. My paper prediction (+0.0002) was 5.6x too low |
| 4 | **Calibration from step 3** | About 13.4 KLD per unit of importance-weighted error on the dense tensors (DERIVED from MEASURED) | Lets every later sandbox number be turned into a whole-model prediction |
| 5 | **Importance weighting of the level search** | No change for Q8_Z or Q7_Z (MEASURED) | The rotation leaves importance nearly flat inside each 32-block |
| 6 | **Q7_ZH**: Q8_Z's 8-bit value split as 7-bit base + 1-bit H (q = 2b + h) | Base only 0.000116 vs native Q7_Z 0.000113: the split costs about 3 % (MEASURED) | One file can serve both 8-bit and 7-bit reads |
| 7 | **H as a gate** (one bit per 32-block says "this block carries H") | Strongly convex: H on 10 % of blocks removes a third of the 7-bit gap; 25 % removes about 60 % (MEASURED) | Blocks differ in how much the 8th bit matters; spend it only where it pays |
| 8 | **Fixed quota per 256** (exactly 2 of 8 gated, so the block size stays fixed) | 0.000061 vs 0.000062 for a free global ranking (MEASURED) | The file format needs a fixed size per block |
| 9 | **Odd grid for ungated blocks** ((2b - 127) x s) | 0.000105 vs 0.000116 for the midpoint base (MEASURED) | Ungated blocks are then a true 7-bit grid with nothing missing |
| 10 | **Gate ranking by the importance matrix** | 0.000061 weighted vs 0.000074 plain (MEASURED) | The ranking is where the matrix matters, not the level search |
| 11 | **Q7_ZT** (type id 202, 241 bytes per 256 = 7.53 bpw, importance path) | C++ 0.000060, matches the sandbox (MEASURED); unit test | The buildable form of steps 7-10 |
| 12 | **urm_ZT whole-model run** (running, started 14:40) | Predicted: **-236 MB** vs Unsloth, **+0.00041 vs urm_Z**, about **0.01282** = a tie with the bar (DERIVED) | The real test of "7 + a gated 8th bit" |
| 13 | **Missing-H fill** | Any integer fill 0.000202-0.000206; **0.5 fill 0.000116 (43 % less)**, exact on int8 via a block-sum term (MEASURED) | Whenever H is skipped or late, the fill decides the damage |
| 14 | **T-decode reference + 1-bit route** (commit 6d539ee) | 1-bit route = int8 fold route, integer-exact on every 32-block; split storage round-trips; AVX2 activation planes = portable; 54 tests pass on gcc, clang, ASan (MEASURED) | Proves the 1-bit decoder gives exactly the same answer as the int8 kernel |

## 3. The letters, and what each one is for

| Letter | Meaning | What it does | Why |
|---|---|---|---|
| **Z** | Zip scales | fp16 per 256 + 6-bit level per 32 | Cheaper scales than Q8_0 (-0.25 bpw) at the same accuracy |
| **P** | Planes | Bits stored as separate planes (base, then H) | Lets a reader take only the planes it needs |
| **T** | Tag / gate | One bit per 32-block (or a fixed quota per 256) says "this block has its 8th bit" | Spends the 8th bit only where it removes the most error |
| **H** | Heat / read rule | Decides at run time whether a block's H bits are read | Trades accuracy for bytes per token, per expert or per tensor |

## 4. Proposed plan for T

1. **Wait for urm_ZT** (step 12). If it ties or beats the bar at -236 MB, Q7_ZT is a real alternative to Q8_Z.
2. **Map the curve on the whole model**: Q7_ZT with 1 and 4 gates per 256 (7.41 and 7.78 bpw). Predicted: k = 1 about +0.0006, k = 4 about +0.0002 vs urm_Z (DERIVED). Needs C: disk space (each file about 22.4 GB; C: is near the 50 GB warning after two more).
3. **Per-tensor quota**: sensitive tensors get more gates, tolerant ones fewer, at the same total bytes. Sandbox ranking first, then one PC run.
4. **Why:** T is the only lever measured so far that moves 7 bits toward 8-bit accuracy without paying the full 8th bit.

## 5. Proposed plan for H

1. **Dense tensors**: read every token, so H is not a heat rule there. It is a quality switch per tensor or per token, using the 0.5 fill when skipped.
2. **Experts**: the heat rule. Hot experts in VRAM read every plane; predicted cold experts get every plane prefetched into L3 during the GPU step; unpredicted cold experts read the base plane, with H filled by 0.5 if it is late.
3. **Planes by need** (sandbox, MEASURED): decide with the expected-size estimate (exact to 1.00x), never the worst-case bound (17-33x too loose); multiply by the router weight. 52 % of unpredicted cold picks have router weight under 0.08.
4. **MTP block**: Q3 base + H; the first draft reads H, deeper drafts read the base only (draft 1's acceptance multiplies every later draft). **Rule for PEL7_P_(T)_H (2026-10-29, DERIVED):** Q3 saves only 0.9-1.9 % of a draft + verify cycle, so the MTP block stays at 7-bit rotated unless Q3 keeps acceptance within about 1.5 points (1 draft) or 2 % of expected tokens (3 drafts); measured with the acceptance test once the MTP path runs.
5. **Why:** H turns stored bits into a per-read choice, so one file serves the fast path and the accurate path.

## 6. Proposed plan for the 1-bit decoder (T-decode)

| # | Step | Where | Status |
|---|---|---|---|
| 1 | Reference T-decode (fold, 1-bit, missing) | Sandbox | **Done**: integer-exact, tested |
| 2 | Activation bit planes (CPU pmovmskb; GPU warp ballot) | Sandbox (CPU) / PC (GPU) | **CPU done** |
| 3 | CPU kernel (popcount on the integer units, beside the vector base) | Sandbox correctness / PC timing | Correctness done (reference); speed kernel next |
| 4 | GPU kernels: CUDA-core AND + popc for decode (M = 1); 1-bit mma for batched MTP verify (M = 5-8) | PC | **Probe v2 done** (19:05): exact; fold 5.7-6.9 % faster than Q8_Z cold; 1-bit route tie to +7 % vs fold |
| 5 | 1-bit mma rate probe on sm_89 | PC, 5 min | **Done**: 937.4 T bit-MAC/s, 8.00x int8 (MEASURED) |
| 6 | `--t-decode` option in `pelias kld` (whole-model proof on urm_ZT2) | Sandbox build, PC run | **Done (run D)**: 1-bit route = integer fold bit for bit over a whole forward pass; vs the float path +0.000387 (t 1.47), from int8 rounding flips (noise floor about +-0.0004) |
| 7 | Split storage (base 233 bytes + H 8 bytes per 256, separate arrays) | Sandbox | **Done** (split / join, tested) |
| 8 | Timing: T-decode vs the plain int8 route | PC | **Probe done**; kernel v3 (bytes in flight, Q8_Z at 64 % of 504 GB/s) next |

**Why the 1-bit decoder:**
- **It switches on idle units.** The H bits are added by AND + popcount: on the CPU's scalar integer units (POPCNT, 4 per cycle on Zen 3, which sit idle while the vector units do the base), on the GPU's CUDA cores (popc) or its 1-bit tensor path (idle during decode), and the activation planes come from pmovmskb on the CPU and the warp ballot on the GPU. The 0.5 fill is plain math (derived), not a unit.
- **It lets the base and the H bits live in different places** (H is 3 % of Q7_ZT's bytes): base in VRAM, H streamed from RAM; CPU base + GPU H (J1); dense models spilling onto RAM.
- **The int8 kernel still "sees 8".** The fold route rebuilds exact int8 values (-127..127) for the existing v10 kernel, so nothing chokes; the 1-bit route gives the same integers.

## 7. Order from here (updated 2026-10-29 00:1x, after runs A-E)

Runs A-E done (game plan, "Runs A-E and the checks around them"). Stopped before R3 for the author's review (author 19:36).
1. Author's review; Fable's math check of #3 (engine: U1/U2, C3, C1, C5, C6).
2. R3 (the per-32 activation kernel; the T-decode decode path folds into it), with the read-only ceiling kernel first.
3. Idle-units tour done (2026-10-29, section 9). On hold (author): own-text stuck tokens by kind; Q4_TG v2 (full importance + 0 levels). Then C6 sharpening (several layers, energy weighting, previous-token signal); more own-text chunks; C1 at long context.
4. GP3-08 MTP head (3-bit tag type), engine, server.

## 8. Where everything lands (author + Claude + Fable, 2026-10-28 19:1x)

The 7-bit / T / H / 1-bit decoder work is the spine; each newer idea plugs into one layer. Full tables and sources: the "Idle Units List (Unified Bus)" doc and the game plan section "The web, version 2".

### 8.1 Quantizer and file format (build time)

| Idea | Lands as | Status | Next |
|---|---|---|---|
| T gate, dense (Q7_ZT) | Dense tensors at 7.53 bpw | Gates 1 and 2 passed (MEASURED) | Gate 3, speed, via kernel v3 |
| C4 grid tags + H (author's bit-tag matrix) | 3-bit: tag byte per 32-block (3.5 bpw), for the MTP block; 4-bit: Q4_TG (id 203, 4.5 bpw, the size of Q4_K) for expert gate/up | 3-bit sandbox passed (0.0190 vs K-line 0.0210); **tags + H at 3.785 bpw do not pay** (0.0149, about 6 % above the K-line; 2026-10-29). Q4_TG whole model (run E): 0.012876, tie with Unsloth, +0.000339 vs ZT2 (not resolved); built with sqrt importance and no 0 level | **Q4_TG v2** (full importance + grids with an exact 0, the author's 0-weight idea): sandbox about -11 % vs imatrix Q4_K; author's review |
| C3 fragility tags | Per-tensor sensitivity from calibration (fit on chunks 0-49, scored on 50-99); tagged experts get refinement planes | MEASURED 2026-10-29: free signals weak (reference margin AUROC 0.63; wrong n-gram copy 2x risk); tags come from the C6 miss map instead; in-model attention near-tie flag untested | Knife-edge view (author 19:12): no separate chunk-24 diagnosis; the near-tie flag treats fragile tokens as a class. Revisit only if the own-text check shows rotated files losing fragile tokens Unsloth keeps |

### 8.2 Kernels

| Idea | Lands as | Status |
|---|---|---|
| Fold route | Default decode (M = 1) for Q7_ZT | MEASURED: 5.7-6.9 % faster than Q8_Z |
| 1-bit decoder (T-decode) | The route when base and H live apart (J1, spill-over) and for batched verify | Exact on GPU; tie to +7 % vs fold |
| Kernel v3 | 2 or 4 rows per lane, loads first | MEASURED (run A): no gain (331 vs 325 GB/s on the largest shape, slower elsewhere); v2 stays. Next: a read-only ceiling kernel over the same 17 MB cold |
| C1 KV codes | Same bit-plane family: scaled 1-bit dot on q and k codes (own warp ballot, random sign diagonal folded into W_q and W_k, page bound first) | MEASURED 2026-10-29 (real q, k, 4 chunks, `--dump-qk`): attention mass kept at 1/8 of keys 72.6 % (best possible 85.7 %, Quest 62.0 %, recent 42.6 %); at 1/32 47.1 % (64.6 / 45.2 / 28.1). Next: long context |

### 8.3 Engine runtime (decode loop)

| Idea | Lands as |
|---|---|
| H read rule | Dense: fixed gate (live gate optional). Experts: planes by need. MTP: H by draft depth |
| C3 flag (author's ROP idea) | Attention near-tie -> `red.global.or` flag, shared with the U2 doorbell -> cold-expert refinement planes pulled in 0.2-0.4 ms |
| C1 KV page selector | 10 attention layers; at 128K, KV reads 2.83 -> 0.26-0.52 ms per token |
| C5 three drafters + batched verify | One chain (exact n-gram > MTP > fuzzy); after GP3-08 and U1/U2; 1.64x at rung 1 |
| U1 / U2 / U8 bus fixes | Before C5 pays anything |
| C6 accuracy-assisted miss map (author) | Log every stuck place (fragile token, prefetch miss, rejected draft) with its state code; a hit takes the expensive path. MEASURED: stuck places repeat across files (WikiText 47-66 %, own text 59-63 %, random 1 %), come in runs (11x), hold 23-26 % of the KL (own text 52 %); in state space the nearest logged neighbour is stuck 3-5x the base rate; code kNN predicts 43-53 % of experts two layers ahead. `pelias kld --dump-codes` built |

### 8.4 Server and context

| Idea | Lands as |
|---|---|
| C2 tiered context | Raw tiers VRAM -> pinned RAM -> NVMe with CRC32 keys (U11), restored on a low-priority stream |
| DeltaNet checkpoint grid | U11: int8 or rotated int4 + rANS |

## 9. Since the review stop (2026-10-29 00:1x-01:2x)

### 9.1 The 8th bit: light vs heavy, rotated vs not (author's question; MEASURED, sandbox, real blk.20 attn_qkv)

Light = the extra bit is one fine step (q = 2b + h, ours, the "0-weighted" bit); heavy = 64 steps (range). Relative squared error, lower is better.

| | Light | Light, not read | Heavy | Heavy, not read |
|---|---|---|---|---|
| Unrotated | 9.43e-5 | 1.41e-4 | 9.58e-5 | 3.22e-2 |
| **Rotated** | **7.86e-5** | 1.12e-4 | 7.98e-5 | 2.88e-2 |

Rotated wins every column (about 17 %), light wins every row (1.5 %); a missing light bit costs 1.4x, a missing heavy bit about 360x. **Rule (author): always keep the light 1-bit rotated.**

### 9.2 Rotation coverage (MEASURED: urm_ZT2's own list; sandbox error on the bf16 source)

- urm_ZT2 rotates 257 of its 259 Q7_ZT tensors (the original 7-bit proof was rotated).
- Not rotated: token_embd (rotation -0.9 %, near Gaussian, kurtosis 2.86: stays) and blk.40.nextn.eh_proj (rotation **-49.5 %**: rotated from now on, author's decision; `--rotate32 ...,eh_proj`, commit e7b5bc0).
- Routed experts -0.8 to -3.4 % per tensor, no type stands out; R2b showed no whole-model gain: they stay as stored. Routers and norms are F32 / BF16.

### 9.3 The idle-units tour (author: "any of these steps may make something read that much better")

| Step | Result | Verdict |
|---|---|---|
| C3 fragility tags from free signals | Reference margin / entropy / top-1 prob AUROC 0.61-0.65; flag 10 % by margin -> 16 % of hard tokens; wrong n=3 copy 2.0 % risk vs 1 %, right copy 0.38 % | Weak; C3 tags come from the C6 miss map |
| C4 tags + H at 3.785 bpw | 0.01491 / 0.01485 vs the K-line 0.0141 / 0.0140 | H on top of tags does not pay; tags alone (-9 %) do |
| C1 1-bit KV codes on real attention | 1/8 of keys: 72.6 % of attention mass (best 85.7, Quest 62.0, recent 42.6); 1/32: 47.1 % (64.6 / 45.2 / 28.1) | Beats both baselines; long context next |
| rANS on bus transfers (entropy of stored codes) | Q4_K 3.87 of 4 bits, Q5_K 4.87 of 5, Q7_ZT base 6.75 of 7, Q6_K 5.80 of 6: lossless coding -3 to -4 % | Ruled out for active weights |
| Worst-chunk codes (C6) | The 10 worst chunks hold 22.7 % of all KL; codes saved for the miss map | Data |

### 9.4 Run A: urm_TG2 (MEASURED 2026-10-29 03:37)

Expert gate/up as Q4_TG2, MTP eh_proj rotated, same size as urm_ZT2 (-236.4 MB vs Unsloth): **KL 0.012056**, same top 95.47 %. Paired: vs Unsloth -0.000808 [-0.001618, +0.000002] (t -1.98); vs urm_ZT2 -0.000480; vs run E -0.000820; vs urm_Z -0.000346. The best file so far; Q4_TG2 option A is the expert gate/up format for R3.

## 10. Since 2026-10-29: planes on the bit system, and where the line went (updated 2026-10-30 15:0x)

### 10.1 Residual planes on the 7-bit and 1-bit bases (MEASURED, sandbox, real blk.20; tools/analysis/planes_bits_check.py)

| Base | bpw | Error | Reading |
|---|---|---|---|
| Q7_Z (rotated attn_qkv) | 7.25 | 1.133e-4 | |
| Q7_Z + 1 plane | 8.31 | 3.262e-5 (-71 %) | a plane removes about what a true 8th bit removes (even-noise theory -75 %), not twice |
| Q8_Z / Q8_0 | 8.25 / 8.5 | 2.854e-5 / 2.806e-5 | |
| Q7_ZT | 7.53 | 6.089e-5 | the gated 8th bit stays the cheapest bit |
| Q7_ZT + 1 plane | 8.59 | 1.980e-5 | -29 % vs Q8_0 at about the same size |
| 1-bit sign base (experts gate + down 0-3) | 1.06 | 0.366 | |
| + 1 / + 2 / + 3 planes | 2.13 / 3.19 / 4.25 | 0.116 / 0.0372 / 0.0130 | no blow-up, each plane about 1/3; 1.8-2.8x behind Q2_K 0.066, Q3_K 0.020, Q4_K 0.0047 at similar bits |
| Q4_TG2 / Q4_K | 4.5 / 4.5 | 0.00418 / 0.00468 | -11 %; + 1 plane both about 0.0016 |

### 10.2 The planes as one integer (MEASURED, sandbox; tools/analysis/plane_stack_check.py; author 09:33)

Four planes with scales locked to s, s/2, s/4, s/8 (+ one fp16 offset per 256) are the bits of one int4 number: 0.01051 at 4.125 bpw vs 4 free planes 0.01299 at 4.25 (-19 %); locked-3 0.0384 vs free-3 0.0372; locked-2 0.1170 vs free-2 0.1163. Free 1-bit planes on top: + 1 0.00281 (5.19), + 2 0.00071 (6.25). This is the T/H idea one level up: a fixed integer base the int4 / int8 hardware reads, plus 1-bit refinements read by need. The plane ladder and the small-card layout built on it are in the game plan (PLAN, after the first engine build).

### 10.3 Where Q4_TG2 went (MEASURED, whole model)

Q4_TG2 became the floor of the usage-tiered expert file (`quantize --tier`): every expert as Q4_TG2, Q5_K, Q6_K or Q8_0 by use and error slope at the Unsloth file's exact expert bytes. KL 0.009404 vs Unsloth's 0.012808 read by Pelias (-26.6 %, t -12.41); + 2 GB worst-block planes 0.008747; R3 streams the tiered types at +0.3 to +1.3 % per token (docs/PELIAS_RESIDUAL_PLANES.md 6.9-6.12).

### 10.4 BitNet b1.58: a model built on 1-bit-class weights (MEASURED, 2026-10-30)

BitNet b1.58 2B4T (I2_S: ternary weights in 2 bits, one scale per tensor, inputs rounded to 8 bits per token) runs in Pelias: top-1 14 of 14 against transformers on the same weights, KL 0.0016 (the floor of the 8-bit input rounding). Its products are sums of +x, -x and 0: the AND + popcount arithmetic of the 1-bit decoder fits them (PLAN: a T-decode kernel for ternary weights, after the first engine build).

### 10.5 Order from here

1. GP3-10 GPU kernels into src (the R3 kernels with Q7_ZT fold, Q4_TG2 and the K-quants), GP3-12 rungs, GP3-14 first engine test build.
2. Then, before new ideas: the parked accuracy checks (dense plane + 3 head planes on the tiered file; the split-token grade).
3. Then the T/H items in the engine: the H read rule (planes by need), the 1-bit route for split storage, the small-card layout.
