# Making 7 bits viable: what was done, and the plan for T, H and the 1-bit decoder

Author's request, 2026-10-28 14:48. Every number is labelled: **MEASURED** (sandbox on real weights, or a whole-model run on the PC) or **DERIVED** (worked out on paper, to be measured). Engine format = 8-bit activations, eight scales per 256 (gpu256x8), paired against the same 25,500 tokens.

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
4. **MTP block**: Q3 base + H; the first draft reads H, deeper drafts read the base only (draft 1's acceptance multiplies every later draft).
5. **Why:** H turns stored bits into a per-read choice, so one file serves the fast path and the accurate path.

## 6. Proposed plan for the 1-bit decoder (T-decode)

| # | Step | Where | Status |
|---|---|---|---|
| 1 | Reference T-decode (fold, 1-bit, missing) | Sandbox | **Done**: integer-exact, tested |
| 2 | Activation bit planes (CPU pmovmskb; GPU warp ballot) | Sandbox (CPU) / PC (GPU) | **CPU done** |
| 3 | CPU kernel (popcount on the integer units, beside the vector base) | Sandbox correctness / PC timing | Correctness done (reference); speed kernel next |
| 4 | GPU kernels: CUDA-core AND + popc for decode (M = 1); 1-bit mma for batched MTP verify (M = 5-8) | PC | To build |
| 5 | 1-bit mma rate probe on sm_89 | PC, 5 min | To run (S2) |
| 6 | `--t-decode` option in `pelias kld` (whole-model proof on urm_ZT) | Sandbox build, PC run | To build |
| 7 | Split storage (base 233 bytes + H 8 bytes per 256, separate arrays) | Sandbox | **Done** (split / join, tested) |
| 8 | Timing: T-decode vs the plain int8 route | PC | To run |

**Why the 1-bit decoder:**
- **It switches on idle units.** The H bits are added by AND + popcount: on the CPU's scalar integer units (POPCNT, 4 per cycle on Zen 3, which sit idle while the vector units do the base), on the GPU's CUDA cores (popc) or its 1-bit tensor path (idle during decode), and the activation planes come from pmovmskb on the CPU and the warp ballot on the GPU. The 0.5 fill is plain math (derived), not a unit.
- **It lets the base and the H bits live in different places** (H is 3 % of Q7_ZT's bytes): base in VRAM, H streamed from RAM; CPU base + GPU H (J1); dense models spilling onto RAM.
- **The int8 kernel still "sees 8".** The fold route rebuilds exact int8 values (-127..127) for the existing v10 kernel, so nothing chokes; the 1-bit route gives the same integers.

## 7. Order from here

1. urm_ZT result (check-ins 18:59Z, 19:08Z, 19:17Z).
2. If it holds: the T curve (k = 1, 4) and the `--t-decode` whole-model proof.
3. PC bench session (S2): 1-bit mma rate, CPU popcount kernel timing, then the GPU T-decode kernel.
