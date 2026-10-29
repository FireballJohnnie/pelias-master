# Pelias: what we found by thinking outside the box, and how it compares

*Updated 2026-10-29 01:2x: light vs heavy 8th bit and rotation coverage (every tensor checked), the MTP piece rotated, and the idle-units tour (C1 KV codes on real attention, C3 fragility signals, C4 tags + H, rANS entropy). Previous versions: 2026-10-29 00:1x (1-bit decoder exact on GPU and whole model), 2026-10-28 16:1x.*

## Plain-language summary

A language model is billions of numbers. To fit it on a home computer, every tool stores those numbers with fewer bits, and every bit removed costs a little accuracy. The best-known small file for this model, Unsloth's, is the bar we set out to beat at the same size or smaller.

Three things made the difference:

1. **We rotate the numbers before storing them.** A simple mathematical shuffle (Walsh-Hadamard, in groups of 32) spreads out the few extreme values that make rounding hard. It cut the file's error by about 9 %, and it makes the 8-bit maths the graphics card is fastest at cost nothing in accuracy.
2. **We store 7 bits and add the 8th only where it matters.** Computers like 8 bits; 7 is "odd" and normally awkward. We keep a 7-bit base everywhere and save the 8th bit only on the quarter of each block where it removes the most error. When the computer calculates, every number becomes an exact 8-bit number again, so the fast hardware sees what it expects.
3. **We put idle parts of the chip to work.** The extra bit can be added with a simple "count the ones" instruction that the processor's integer units can do while the main maths units are busy, on either the CPU or the graphics card, and the base and the extra bit can even live in different memory.
4. **The 1-bit decoder works.** On the graphics card, the 7-bit file now runs 5.7-6.9 % faster than the 8-bit one on the large matrices, with every result exact. Across a whole model, adding the 8th bit by counting ones gives exactly the same numbers as rebuilding full 8-bit values. Filling the 7 bits back to 8 is what recovers the accuracy: pure 7-bit costs about 8x more error than 7 bits plus the gated 8th.

5. **The light bit, rotated, is the rule.** Of the two ways to use the 8th bit, the light one (one fine step, "0-weighted") rotated is the most accurate in every case, and it is the only safe one: if it is not read, the number falls back to its 7-bit value, while a heavy (range) bit that goes missing wrecks the number (about 360x more error).

**Where it stands:** our 7.53-bit file is **236 MB smaller than Unsloth's and slightly more accurate** (0.012536 against 0.012864; not yet a statistically settled win), and it is **as accurate as our own 8-bit file**. On the author's own chat text the order holds (0.021198 against Unsloth's 0.022368, not yet settled). The speed of the whole engine is still to be measured.

## New for consumer PCs (what we have not seen done elsewhere)

| What | Why it matters on a home PC | Status |
|---|---|---|
| A 7-bit file that the 8-bit hardware reads as exact 8-bit, 236 MB smaller | More of the model fits on a 12 GB card at the same accuracy | MEASURED |
| The 8th bit stored separately and added back by counting ones (1-bit decoder) | Base and extra bit can sit in different memory (card, RAM) and be added by idle units; exact | MEASURED (GPU kernel, whole model) |
| A 7-bit GPU kernel faster than the 8-bit one | Fewer bytes per token, the limit on a consumer card | MEASURED (5.7-6.9 % on large matrices) |
| The miss map (the author's accuracy-assisted miss map) | The hard tokens repeat across every file tested (47-66 % shared, 1 % by chance) and hold a quarter to a half of the error; logging them tells the system where to spend extra bits | MEASURED (pattern); map PLAN |
| Bit tags per 8 values (the author's bit-tag matrix) with a 0 level | Same bytes as Q4_K, about 11 % less weighted error on real experts in the sandbox | MEASURED (sandbox); whole model next |
| CPU and GPU reading RAM at once | 93 % of the memory's rated speed on a consumer desktop | MEASURED (probe) |
| A 1-bit code per attention key (the author's H gate on KV pages) | Picks the keys that hold the attention better than page bounds (Quest) or a recent window, so long contexts read a fraction of the KV cache | MEASURED (real q, k; 512-token chunks) |
| The light 8th bit, rotated | Most accurate and safe to skip: a late or missing bit costs 1.4x, not 360x | MEASURED (sandbox, real weights) |

## Scoreboard (whole model, measured)

Paired runs over the same 25,500 tokens, 8-bit activations as the engine uses them, every number checked again from the saved per-token results. Lower KL = closer to the original model.

| File | Dense weights | KL | vs Unsloth (0.012864) | Size vs Unsloth |
|---|---|---|---|---|
| Unsloth MTP UD-Q4_K_M (llama.cpp's GPU path) | Q8_0, 8.5 bits | 0.012864 | the bar | 0 |
| urm (Unsloth's types, our rotation) | Q8_0 | 0.012499 | -0.000365, not resolved | -6 KB |
| urm_Z | **Q8_Z** (zip scales), 8.25 bits | 0.012402 | -0.000462, not resolved | -61.0 MB |
| urm_Q7 | Q7_Z, pure 7-bit, 7.25 bits | 0.013524 | +0.000660, **resolved worse** | -305.1 MB |
| urm_ZT (first version, weighted levels) | Q7_ZT, 7.53 bits | 0.012870 | -0.000006, exact tie | -236.4 MB |
| **urm_ZT2 (fixed)** | **Q7_ZT, 7.53 bits** | **0.012536** | **-0.000328, not resolved** | **-236.4 MB** |
| urm_Z with the live H gate (25 %) | Q8_Z stored, ~7.5 bits read | 0.012919 | +0.000055, tie | -61.0 MB stored |
| urm_ZT2 through the 1-bit decoder (run D) | Q7_ZT, 8th bit by counting ones | 0.012923 | +0.000059, tie (vs the float path: +0.000387, not resolved; int8 rounding flips, see A) | -236.4 MB |
| urm_TG (expert gate/up as Q4_TG, bit tags) | Q7_ZT dense | 0.012876 | +0.000012, tie | -236.4 MB |
| **urm_TG2 (expert gate/up as Q4_TG2: bit tags with an exact 0, full importance; MTP eh_proj rotated)** | **Q7_ZT dense** | **0.012056** | **-0.000808, t -1.98 (a hair short of resolved)** | **-236.4 MB** |

**On the author's own chat text** (run C; 53,334 tokens, never published): urm_Z 0.020977, urm_ZT2 0.021198, Unsloth 0.022368. Ours lower by 0.0012-0.0014, not resolved (chat text swings about 4x more from chunk to chunk).

## What makes the system possible (the chain)

Each link depends on the one before it:

1. **Rotation** (dense tensors only). Makes rounding cheap, both for the stored weights and for the 8-bit activations. Without it, today's 7-bit result would land about 0.0008 worse (paper: 24 % more weight error measured on attn_qkv, plus the activation cost the unrotated Unsloth file pays).
2. **The format** (Z, T). Cheaper scales (Z), then the 8th bit as a gate per 32-block (T). Every value still rebuilds to an exact int8.
3. **The read rule** (H). Decides, per expert or per token, which bits are read. The expected-size estimate tells exactly what a dropped bit costs; a missing bit is filled with 0.5, never a whole number.
4. **The decoder** (T-decode). Adds the 8th bit by AND + popcount, exact, on units that are otherwise idle, from memory that can be separate from the base.
5. **The bus** (the unified CPU + GPU system). The CPU and GPU read the same DDR4 at once; the CPU computes the cold experts, the GPU the hot ones; predicted experts are fetched into L3 while the GPU works.
6. **The engine** (GP3-14). Puts 1-5 into one token loop. Not built yet; it is where speed is measured.

## The 8th bit: light vs heavy, rotated vs not (author's question, 2026-10-29)

Same 7-bit base plus one gated extra bit on 2 of every 8 blocks (the Q7_ZT layout), on real blk.20 attn_qkv weights. **Light** = the extra bit is worth one fine step (refines; q = 2b + h, our design, the "0-weighted" bit). **Heavy** = the extra bit is worth 64 steps (extends the range). "If not read" = the extra bit is skipped (a fast path, or H not on the card yet). Relative squared error, **lower is better** (e-5 is about 0.0001, e-2 about 0.03). MEASURED (sandbox), `tools/experiments/light_heavy_bit_check.py`.

| | Light bit | Light bit, if not read | Heavy bit | Heavy bit, if not read |
|---|---|---|---|---|
| Unrotated | 9.43e-5 | 1.41e-4 | 9.58e-5 | 3.22e-2 |
| **Rotated** | **7.86e-5** | 1.12e-4 | 7.98e-5 | 2.88e-2 |

- Rotated wins every column (about 17 % less error); the light bit wins every row (1.5 %).
- A missing light bit costs 1.4x (the 7-bit value still stands); a missing heavy bit costs about 360x (large values are clipped).
- **Rule (author): always keep the light 1-bit rotated.** Our best file (urm_ZT2) already does: 257 of its 259 Q7_ZT tensors are rotated.

**Rotation coverage, every tensor checked (2026-10-29, sandbox on the bf16 source):** the two Q7_ZT tensors left unrotated were the token table (rotation -0.9 % error: its values are already near a bell curve, kurtosis 2.86; left as is) and the MTP input projection blk.40 eh_proj (rotation **-49.5 %**: now rotated in every new build, author's decision). Routed experts -0.8 to -3.4 % per tensor, whole model no gain (R2b), left as stored. Routers and norms are stored in full precision, nothing to round.

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
| Plain level search for Q7_ZT | The first Q7_ZT (importance-weighted levels) tied Unsloth; the second (plain levels, square-root importance for the gate only) came out ahead. The gap between them does not resolve (t -0.86), and the first version's two bad chunks were one or two flipped tokens each, so the cause is open | MEASURED (result) / open (cause) |
| Fused int8 kernel (v10) [4] | 1.39-1.84x the bf16 path, all 20 shapes, quantization fused into RMSNorm | MEASURED (kernel bench) |
| Grouped int8 launch for all experts (v12) [5] | 2.5-3.0x the bf16 kernel grouped the same way | MEASURED (kernel bench) |
| Dual bus | CPU and GPU reading RAM at once: 26.56 + 21.06 = 47.62 GB/s, 93 % of the 51.2 GB/s peak | MEASURED (probe) |
| **1-bit decoder on the GPU (T-decode kernel v2)** | Every route exact; Q7_ZT fold 5.7-6.9 % faster than Q8_Z cold on the large shapes; the 1-bit route tie to +7 % vs fold; the 1-bit tensor path 937.4 T bit-MAC/s, 8.00x int8 | MEASURED (GPU probe) |
| **1-bit decoder through a whole model** | Bit-identical to the integer fold over a full forward pass (max logit difference 0); against the float path, int8 rounding flips give a noise floor of about +-0.0004 KL on 100 chunks | MEASURED |
| Live H gate (25 %) | 0.012919: ties Unsloth; the fixed gate (Q7_ZT) stays the choice | MEASURED |
| Logging q and k for the KV-page check (C1) | KL identical to run B on the same 4 chunks (max difference 0) | MEASURED |
| Hard tokens repeat across files (C6) | Top-1 % KL tokens shared 47-66 % between files (own text 59-63 %), 1 % by chance; they hold 23-26 % of the KL (own text 52 %); logging them changes nothing (exact) | MEASURED |

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
| Bit tags at 3 bits (C4) | 0.0190 at 3.5 bpw vs the K-quant line's 0.0210 on real experts | MEASURED (sandbox) |
| Bit tags at 4 bits with a 0 level (Q4_TG v2) | About -11 % weighted error vs imatrix Q4_K at the same bytes | MEASURED (sandbox) |
| Light vs heavy 8th bit, rotated vs not (author's question) | Rotated + light bit best (by 17 % and 1.5 %); a missing heavy bit collapses (about 300x), a missing light bit falls back to the 7-bit base | MEASURED (sandbox) |
| Miss map in state space | Nearest logged neighbour of a hard token is hard 3-5x the base rate; the nearest logged state predicts 43-53 % of the experts two layers ahead | MEASURED (sandbox, run B codes) |
| The author's H gate on KV pages (C1), on real attention | Share of attention mass kept, 1/8 of the keys: best possible 85.7 %, **1-bit code 72.6 %**, Quest pages 62.0 %, recent window 42.6 %; at 1/32: 64.6 / **47.1** / 45.2 / 28.1 %. Beats both baselines; absolute numbers are a floor (512-token chunks spread the attention) | MEASURED (real q, k from urm_ZT2, 4 chunks x 10 layers) |
| MTP input projection (eh_proj) rotated | -49.5 % error; built into `--rotate32` | MEASURED (sandbox) |
| C4 tags + H gate at 3.785 bpw | 0.01491 / 0.01485: at Fable's target, but about 6 % above the K-quant line at the same size; tags alone stay -9 % | MEASURED (sandbox) |
| C3 free fragility signals (reference margin, entropy, copyable n-grams) | Weak: AUROC 0.61-0.65; a wrong n-gram copy doubles the risk (2.0 % vs 1 %), a right one halves it. The miss map's state codes are far stronger (3-5x) | MEASURED (wikitext base + run B) |
| Cold-expert routing shares | 6.3 unpredicted cold picks per token, 52 % under router weight 0.08; 67 % of cold picks predicted one layer ahead | MEASURED (routing record) |

### C. Planned, with the math checked

| Idea | Paper result | Label |
|---|---|---|
| The unified bus: 2 MB pages + 8 pinned streams, always-spinning doorbells, predicted experts prefetched with all planes into L3 during the GPU step, the rest one layer late | Token time 18.5 ms -> about 8.65 ms (54 -> about 116 tok/s); ceiling about 119 with the GPU's own overhead, 193 as the memory floor | PAPER (Fable) |
| J1 split-plane cold expert | Full-file accuracy on the unpredicted cold experts for about +0.1 ms; a speed gain only if the CPU leg stays under about 35 GB/s | PAPER (Fable) |
| GPU 1-bit tensor path (and-popc mma) | Right unit for batched MTP verification (5-8 tokens); at single-token decode the CUDA cores do it in about 0.7 us | PAPER (Fable) |
| MTP block as Q3 base + H, H read on the first draft only | About -140 MB; draft 1's acceptance multiplies every later draft. Break-even (derived from the file's sizes): Q3 saves only 0.9-1.9 % of a draft + verify cycle, so the MTP block stays at 7-bit rotated unless Q3 keeps acceptance within about 1.5 points (1 draft) / 2 % (3 drafts). Most of a draft's read is the 417 MB output head | PLAN (needs GP3-08) |
| CRC32 prefix store with a DeltaNet checkpoint grid (int8 or rotated int4 + rANS on idle CPU cores) | A repeated 4K-token prompt skips about 1.8 s; a partial hit about -400 ms | PAPER (Fable) |
| Texture units decoding Q8_Z as RGBA8 | About 10x fewer instructions at decode; prefill depends on the fp16-accumulate rate | PAPER (Fable) |
| Carry-over to dense models that spill onto CPU/RAM | Bus, J1, planes by need, Q8_Z / Q7_ZT, MTP all apply; prefetch is 100 % predictable there | PAPER |
| The author's H gate on KV pages (C1) | Sign-plane codes pick the KV pages a token needs: at 128K, KV reads 2.83 -> 0.26-0.52 ms per token. First real-attention check passed against the baselines (section B); long-context recall still to measure | PAPER (speed) / MEASURED (selection) |
| Three drafters, one verify (C5, the author's matrix array) | 1.6-1.9x at acceptance 0.7-0.8, after the bus fixes | PAPER (Fable checked) |
| RT cores as the miss-map lookup (author) | Hardware nearest-neighbour search beside the SMs, only on a hit; pays once the map reaches millions of entries | PLAN |

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
| "Keep the 1-bit decode as a path the int8 tensor can still see as 8" | The fold route rebuilds exact int8; the 1-bit route gives the same integers, now across a whole model | MEASURED |
| "The H gate" for KV pages and drafts (2026-10-28 evening) | C1 and C5 | PAPER |
| The accuracy-assisted miss map, "the more we look at where it got stuck, the better we know when to look" | C6: logging built, the pattern measured, miss log v1 in the repo | MEASURED (pattern) |
| The bit-tag matrix (7+7+7+7+4) | Pays at 3 and 4 bits as a tag byte per 32-block (C4, Q4_TG) | MEASURED (sandbox), whole model next |
| "Does a 0 have more weight?" | A 0 level in the 4-bit tag grids: about -7 % more error | MEASURED (sandbox) |
| Light vs heavy 8th bit, rotated vs not | Rotated + light bit; the heavy bit collapses when missing; rule: always keep the light 1-bit rotated | MEASURED (sandbox) |
| "Did we check which tensors to rotate?" | Every Q7_ZT tensor checked: the MTP eh_proj gains 49.5 % and is now rotated; the token table gains 0.9 % and stays | MEASURED (sandbox) |
| MTP at 7 vs 3 bits (PEL7_P_(T)_H) | Stay at 7 unless Q3 is within about 1.5 points of acceptance | PAPER |
| The rANS int4 checkpoint vs NVENC, applied to active tensors | Stored codes already carry 3.87 of 4 bits (experts) and 6.75 of 7 (dense): lossless coding saves 3-4 %, not worth a decoder | MEASURED |
| ROPs to flag fragile tokens | Became the near-tie flag (`red.global.or`, no wait) feeding the doorbell | PAPER (Fable) |

## Proven not to work (kept so nobody retries them)

| Idea | Result |
|---|---|
| Pure 7-bit everywhere | +0.001122 vs the 8-bit file, resolved worse |
| Plain two-stage E8 codebook | Ties the K-quant line at 4 bits, no gain without Hessian-aware rounding |
| Empty activation bit planes with one scale per 32 | None occur |
| Expert deferral; near-tie expert swap | +0.0042 and +0.0017, resolved worse |
| Rotating the routed experts | No gain: +0.000530 vs stock (not resolved), +0.003979 vs rotating the dense tensors only (R2b) |
| Video units (NVENC/NVDEC) for the live KV cache | 1.3 s per token |
| AES / SHA / CLMUL, vector gather, 1 GB pages, L3 as a fixed expert bank | No gain on the math (AES and CLMUL share the vector pipes on Zen 3) |
| More rows per GPU lane (kernel v3) | No gain over v2 (331 vs 325 GB/s), slower on smaller shapes |
| A heavy (range) 8th bit | Ties the light bit when read; collapses when missing |
| NVENC / NVDEC for the stored-context tier; the SFU as a decoder | No job survives the math (Fable) |
| Rotating the token table | -0.9 % error only (near Gaussian) |
| The H gate on top of 3-bit tags (3.785 bpw) | About 6 % above the K-quant line at the same size |
| Lossless entropy coding (rANS) of weights on bus transfers | Codes already near full entropy: 3-4 % smaller |
| The reference's own uncertainty as the fragility flag | AUROC 0.61-0.65, too weak; use the miss map |

## Compared with existing frameworks

Limited to what their formats do; we have not tested the other projects themselves.

| Area | Existing (format facts) | Pelias (labelled) |
|---|---|---|
| Between 6 and 8 bits | llama.cpp goes from Q6_K (6.56 bpw) to Q8_0 (8.5 bpw, 0.5 bpw of it scales); Unsloth keeps sensitive dense tensors at Q8_0, a whole tensor at a time | Q8_Z 8.25 bpw (same accuracy, MEASURED); Q7_ZT 7.53 bpw with the 8th bit per 32-block (no difference from Q8_Z, MEASURED) |
| Importance matrix | Weighted K-quant search; none for Q8_0 | Used only to rank the gates, square-rooted (ranking by importance: 0.000061 vs 0.000074 plain, MEASURED sandbox); whether a weighted level search hurts is open (our first Q7_ZT's bad chunks were single flipped tokens) |
| Fragile tokens | — | The hard tokens are a property of the text, shared by every file including Unsloth's; on chat text 1 % of tokens hold 52 % of the error; logged as the miss map's data (C6) |
| Bits and memory | One tensor, one memory pool, one unit | Base and 8th bit splittable across memories and units, exact (MEASURED); speed PLAN |
| Rotation | QuaRot, SpinQuant, QuIP# rotate linear layers of the (mostly dense) models they report | Dense tensors only on this MoE model; experts left as stored (MEASURED) |
| Bit-serial decoding | Exists (T-MAC, Any-Precision LLM) | New here: the per-32-block gate, the 0.5 fill, the split across memories (MEASURED pieces) |
| Training | QLoRA re-quantizes after merging | Training on the served bytes and int8 backward passes: PLAN, untested |

## Next

1. The author's review; Fable's math check of the engine plan (#3). The idle-units tour is done (C1, C3, C4 stack, rANS entropy); the status board of every idle-unit item is in the Idle Units List doc.
2. R3: the per-32 int8 kernel, which the T-decode decode path folds into; first a read-only ceiling kernel (is ~325 GB/s the limit of one cold launch?).
3. On hold (author): stuck tokens by kind, Q4_TG v2 (full importance + the 0 level) on the whole model. Then the miss map sharpened; more own-text chunks; C1 at long context.
4. The engine, where the speed of all of this is measured end to end.
