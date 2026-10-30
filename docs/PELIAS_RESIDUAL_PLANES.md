# Residual sign planes: what was measured, the GPU plan, and how to keep the cost down

Pelias, 2026-10-29/30. Model: Qwen3.6-35B-A3B. Every number is labelled: **MEASURED** (on the author's one machine: RTX 4070
12 GB, Ryzen 7 5800X3D, 64 GB DDR4, WSL2 48 GB; or in the sandbox where it says so), **DERIVED** (worked out from measured
numbers, the working shown), **ESTIMATE** or **UNMEASURED** (with the run that would measure it). Scoring: KL divergence
against the BF16 original on WikiText-2, 100 chunks x 512 tokens, positions 256-510 (25,500 scored tokens), unless noted.

## 1. The idea in one paragraph

A quantized file stores each weight with an error against the original. A **residual sign plane** stores, per 256 weights,
one fp16 scale and one sign bit per weight: "the original is a little above (+) or below (-) what the file stores, by
about this much." Adding the plane to the stored row brings it back toward the original; a second plane corrects what the
first left. The planes sit in a side file next to any model file (ours, Unsloth's, stock), are fitted once from the
original weights, and can be read on every token, on some tokens, or not at all (a plane not read costs nothing and adds
nothing). In the engine they are added by the 1-bit path (AND + popcount), which is idle during decode today. In the
author's terms: the base file is the fast first pass; the planes are proofreading, applied in proportion to the
uncertainty.

## 2. What was measured

### 2.1 Weight error on real tensors (sandbox, MEASURED; blk.20, importance-weighted error relative to the weights)

| Tensor (stored type) | Stored | + 1 plane | + 2 planes | + 3 planes | One type up (old way) |
|---|---|---|---|---|---|
| attn_qkv (Q7_ZT, dense) | 0.000061 | 0.000020 (0.33x) | 0.000006 (0.10x) | 0.000002 (0.03x) | Q8_Z 0.000029 (0.47x) |
| ffn_gate_exps (Q4_K) | 0.004629 | 0.001643 (0.36x) | 0.000447 (0.10x) | 0.000129 (0.03x) | Q5_K 0.001177 (0.25x) |
| ffn_down_exps (Q5_K) | 0.001195 | 0.000372 (0.31x) | 0.000106 (0.09x) | 0.000031 (0.03x) | Q6_K 0.000291 (0.24x) |

- Each plane cuts the error about 3x. Planes fitted together (refit) beat plane-by-plane by 5-10 % at 2 planes, 21-37 % at 3.
- Rotating the residual before taking signs is worse (dropped).
- The experts' error is 20-75x the dense part's: the experts are where the drift lives.
- The C++ fit reproduces the sandbox numbers exactly (0.325 / 0.110 / 0.102 / 0.030 x Q7_ZT on attn_qkv).

### 2.2 The side file for Unsloth's file (PC, MEASURED)

`pelias planes`, 2 planes per row, 8 refit rounds, importance where it was collected in the file's basis, exact output rows.

| Part | Bytes |
|---|---|
| Expert planes | 8.556 GB |
| Dense planes (attention, DeltaNet, shared experts) | 0.373 GB |
| Embedding + output-layer planes | 0.270 GB |
| Exact output rows (BF16, for the head re-score) | 1.017 GB |
| **Total side file** | **10.217 GB** (built in 262 s) |

Weight error left after 2 planes over the whole model: 0.098 of before (predicted 0.09-0.11: met).

### 2.3 KL against the original (PC, MEASURED, paired on 25,500 tokens)

| Unsloth MTP UD-Q4_K_M read by | KL | Same top | vs no planes |
|---|---|---|---|
| llama.cpp GPU path (traditional) | 0.012864 | 95.04 % | — |
| Pelias, no planes | 0.012808 | 95.02 % | baseline |
| **Pelias, 2 planes** | **0.005818** | **96.77 %** | **-54.6 %**, 95 % [-0.00812, -0.00586], t -12.25 |
| **Pelias, 2 planes + exact head re-score (64)** | **0.005741** | **96.86 %** | **-55.2 %**, t -12.40 |

- With planes the KL is lower on 81.7 % of tokens; the top token is right only with planes 830 times, only without 383.
- The head re-score adds -1.3 % on top of the planes (t -16.8).
- For scale: the original BF16 through Pelias (the floor of the scoring itself) 0.0028 (10 chunks); our best own 4-bit
  file urm_TG2 0.012056; stock Q4_K_M on llama.cpp 0.0411.

### 2.4 The split tokens, graded by the original model (PC, MEASURED)

20 fixed prompts x 128 greedy tokens; `pelias teacher` runs the BF16 original over each side's text and gives its own pick
at every position; at the first split both sides share the prefix, so the original's pick says who is right.

| | v8 (no planes) | With 2 planes |
|---|---|---|
| Prompts equal to llama.cpp | 9 of 20 | 5 of 20 |
| At the first split the original sides with Pelias | 7 | **12** |
| ... with llama.cpp | 4 | **3** |
| ... with neither | 0 | 0 |
| Held token = the original's own top-1 (all positions) | 98.7 % (llama.cpp 98.0 %) | **99.1 %** |
| Mean NLL under the original | 0.1895 (llama.cpp 0.2078) | **0.1887** |

With planes, Pelias disagrees with llama.cpp more often because it is right more often: where they split, the original
sides with Pelias 4 times in 5.

### 2.5 Cost in today's CPU reference (PC, MEASURED)

| | Decode | Model + planes in RAM |
|---|---|---|
| No planes (v8) | 1.46 tok/s | 22.66 GB |
| 2 planes, every token | 1.08 tok/s (after the table add fix; 0.37 before) | 32.88 GB |

The CPU reference is the correctness path, not the engine; the engine's numbers are section 3.

## 3. The GPU side: plan and predictions

### 3.1 What it stands on (MEASURED)

- R3 streaming kernels: a whole token, about 2.27 GB of weights, in 4.80 ms (473 GB/s; 40 distinct layers, no reuse).
- 1-bit tensor path: 937.4 T bit-MAC/s (8.00x the int8 path's products).
- An extra bit plane by AND + popcount against the activation's 8 bit planes is exact (bit-identical to the int8 route on a
  whole synthetic model) and costs about 2 % over folding it into int8 on the large shapes.
- CPU bus: 47.8 GB/s at 2 threads (DDR4 saturated), with no lag felt by the user up to 4 threads.

### 3.2 Plan

1. Dense, embedding/output planes (0.643 GB) in VRAM, added on the GPU's 1-bit path in the same pass as the base weights.
2. Expert planes in RAM; for the experts a token routes to, added on the CPU over its own bus while the GPU runs the step
   (the dual-bus split), or held in VRAM for the hot experts if the bank has room.
3. The flag (pre-token C6 map) and the dual predict decide per token how many planes to read: 0, 1 or 2 (graded).
4. The exact head re-score: the top 64 candidates' original rows (64 x 2048 x 2 B = 256 KB per token).
5. Gate: the GPU engine must equal the CPU reference with the same planes (rung 1 tolerance), so its accuracy is section 2.3.

### 3.3 Predictions (DERIVED from the MEASURED numbers above)

Bytes per token for the planes:
- Dense planes: all dense rows are read every token: 0.373 GB; output-layer planes 0.135 GB; embedding planes one row
  (0.5 KB). About **0.51 GB per token**.
- Expert planes: 8 of 256 experts per layer are active, so 8.556 GB x 8 / 256 = **0.267 GB per token**.

| Mode | Extra time per token | Token time | tok/s | Accuracy (KL) |
|---|---|---|---|---|
| No planes | 0 | 4.80 ms | about 208 | 0.0128 (MEASURED) |
| Planes on every token, dense on GPU (0.51 GB at 494 GB/s = 1.03 ms), experts on the CPU bus in parallel (0.267 GB at 47.8 GB/s = 5.6 ms) | the CPU leg sets the pace | about 5.6-5.8 ms | about 172-178 | 0.0057 (MEASURED in the reference; the engine must match) |
| Every token, all planes from VRAM (0.78 GB more at 494 GB/s = 1.58 ms) | +1.58 ms | about 6.4 ms | about 156 | 0.0057 |
| Flagged tokens only, 20 % flagged | +0.2 x the above | about 5.1-5.2 ms | about 192-196 | between 0.0057 and 0.0128, UNMEASURED |

- The every-token rows need the plane add itself to stay under the bandwidth cost: at 937 T bit-MAC/s the adds for 0.78 GB
  of planes take microseconds (DERIVED), so bandwidth decides.
- The flagged-only accuracy needs a `kld` option that reads planes only on the flagged positions (the pre-token flag
  catches 30 % of the stuck tokens at 10 % flagged, MEASURED offline); to measure.
- **UNMEASURED: how much more the GPU-era options add** (3 planes on flagged tokens, planes on every token for free on the
  dense part). We will know when they run.

## 4. Keeping the cost down while keeping an accuracy win

### 4.1 The honest starting point

Bit for bit, a plane is not better than storing more bits up front (MEASURED, gate_exps: 1 plane +1.06 bits per weight
leaves 0.355x; Q5_K +1.0 bits leaves 0.254x). So the win has to come from **putting planes only where the drift lives**,
and from reading them only when needed. The experts' planes are 8.556 GB of the 10.217 GB: that is the lever.

### 4.2 Options (sizes DERIVED from 2.2; KL DERIVED from the line below)

KL model (DERIVED, two-point fit through the measured runs): **KL = 0.0051 + 0.0078 x (weight error left)**, with weight
error 1.0 for the file alone (0.012808) and 0.098 with 2 planes (0.005818). The 0.0051 is what the planes do not touch
(the scoring floor, the 8-bit activation format, the cache). It is a straight line through two points: every projection
below is a prediction until its run.

| Option | Side file | Total | % of BF16 (71.07 GB) | Weight error left | Projected KL |
|---|---|---|---|---|---|
| BF16 original | — | 71.07 GB | 100 % | 0 | 0.0028 (floor, MEASURED) |
| 2 planes everywhere + exact head (MEASURED) | 10.22 GB | 32.88 GB | 46 % | 0.098 | 0.0057 (MEASURED) |
| 2 planes everywhere, no exact head | 9.20 GB | 31.86 GB | 45 % | 0.098 | 0.0058 (MEASURED) |
| **B: 2 planes on the hot experts only** (120 of 256 per layer, 93.5 % of picks, MEASURED) + dense | 4.01 + 0.64 = 4.65 GB | 27.31 GB | 38 % | 0.935 x 0.1 + 0.065 x 1 = 0.16 | **about 0.0063** |
| A: 2 planes on the worst half of the expert blocks (blk.20: half the blocks keep 0.23x) + dense | 4.28 + 0.64 = 4.92 GB | 27.58 GB | 39 % | about 0.23 | about 0.0069 |
| 1 plane everywhere | 4.60 GB | 27.26 GB | 38 % | about 0.33 | about 0.0077 |
| **A + B: worst half of the hot experts' blocks** + dense | 2.00 + 0.64 = 2.64 GB | 25.30 GB | 36 % | 0.935 x 0.23 + 0.065 = 0.28 | **about 0.0073** |
| The 4-bit file alone | 0 | 22.66 GB | 32 % | 1.0 | 0.0128 (MEASURED) |

### 4.3 Against the traditional way at the same size (DERIVED, UNMEASURED on this model)

Assumption: KL follows the weight error the same way however the error was cut.
- One type up (a Q5-class file, about 27-28 GB): weight error about 0.25x, KL about **0.0071**.
- Two types up (a Q6-class file, about 32-33 GB): about 0.06x, KL about **0.0056**.

Reading:
- Uniform planes roughly **tie** with storing more bits at the same size.
- **Selective planes win at the same size:** B at 27.3 GB about 0.0063 vs a Q5-class file about 0.0071; A + B at 25.3 GB
  about 0.0073 vs about 0.009 for a file half a type up.
- **Planes win on speed by design:** a Q6-class file reads about 1.45x the bytes on every token; planes are read only where
  and when they matter (every token, flagged tokens, hot experts only). Speed is set by the bytes read per token, so the
  engine runs near the 22 GB file's speed with accuracy near the 33 GB one.
- **Planes need no new model file:** they work on any existing quantized file, given the original weights once.

**Correction (6.3):** offered per-expert types as well, selective planes lose at a fixed size; a usage-tiered expert file
reaches about 0.0079 at 22.65 GB with no planes. The rows above compare planes only with uniform files.

### 4.4 The runs that turn this into MEASURED (ESTIMATE about 4-5 h of PC time)

1. A traditional Q5_K_M and Q6_K of this model through Pelias (quantize from BF16, about 15 min each; `kld`, about 25 min
   each): settles 4.3.
2. 1-plane and 3-plane side files + `kld`: checks the KL line.
3. Builder options for A (block-selective, with a 1-bit block map) and B (hot experts only, from the measured hot set), one
   `kld` each, and A + B.
4. The flagged-only reading (`kld` with planes on the flagged positions only): the accuracy of the cheapest engine mode.

## 5. Removing the exact head rows: does it still work?

Yes, and it is already measured.

- What would be removed: `output.weight.exact`, the original output rows in BF16 (1.017 GB), used only by
  `--head-rescore` to recompute the top 64 logits exactly.
- What stays: the output layer's own **planes** (part of the 0.270 GB), which correct the head like every other weight.
- MEASURED cost of removing it: KL 0.005741 -> 0.005818 (+1.3 %; the head re-score's whole gain), same top 96.86 % ->
  96.77 %. That is the `kld 2 0` run: the planes without the exact rows. It still works; it saves 1.0 GB (3 % of the
  total) and gives back about 1.3 % of the accuracy.
- Middle routes (DERIVED, UNMEASURED):
  - **Exact rows for the frequent tokens only.** Most of what a model writes comes from a small part of the vocabulary;
    rows for, say, the 32k most frequent of 248,320 tokens are 32k x 2048 x 2 B = 0.13 GB, and the re-score keeps its gain
    whenever the top 64 candidates fall inside that set. To measure: the share of top-64 candidates inside the set, from
    the existing `--dump-kld` runs, then one `kld`.
  - **A third plane on the output layer only** instead of exact rows: 0.067 GB (one plane of the output's 0.135 GB) for
    about 3x less head error (2.1); predicted to keep most of the re-score's gain at 7 % of its size.
  - **Exact rows stored in 8 bits** (0.51 GB): half the size, a small error left on the very rows that decide near-ties;
    the first two routes are cheaper.

## 6. A file the size of Unsloth's with an accuracy win (analysis 2026-10-30)

The author asked (05:32) for a method that stays near the Unsloth file's size (22.65 GB) and still wins on accuracy.

### 6.1 What the research says

- **The traditional step up is size-efficient.** Unsloth's published table for Qwen3.5-35B-A3B (an earlier model, other
  eval text; MEASURED by Unsloth, UNVERIFIED here) shows UD-Q4_K_XL 19.17 GB, KLD 0.0137 and UD-Q5_K_XL 23.22 GB, KLD
  0.0069. About +4 GB halves KLD; uniform planes needed +10 GB for -55 %.
- **The same ±1-plane representation is known** as binary-coded quantization: AnyBCQ (greedy plus alternating refit,
  several precisions from one store), Any-Precision LLM (one file read at several bit widths), BitStack (sign plus
  residual blocks, any size). Pelias's planes are this family; the fit here was built ground up.
- **MoE work points at per-expert precision.**
  - DynaExq keeps hot experts at high precision and cold ones low, chosen at run time by usage (on Qwen3-30B-A3B).
  - MoPEQ chooses bits per expert by Hessian or by frequency.
  - MoBiQuant reads residual slices per token through a small router.
- **Stock GGUF cannot express it.** All experts of one layer share one tensor and one type, so a GGUF file gives every
  expert the same precision, the hot ones and the ones almost never picked.

### 6.2 Measured inputs

- **The Unsloth file by tensor class:**
  - experts 19.57 GB of 22.65 (86 %);
  - gate and up Q4_K, 6.04 GB each;
  - down Q5_K/Q6_K, 7.49 GB;
  - dense part 1.60 GB; embedding 0.54 GB; head 0.42 GB; MTP 0.53 GB.
- **Expert usage is very uneven.** Ranked on chunks 0-49 and scored on chunks 50-99 of WikiText-2 (held out):

  | Top experts of 256 per layer | Share of routing weight |
  |---|---|
  | 32 | 53 % |
  | 64 | 74 % |
  | 120 | 91 % |

  The worst layer at 120 covers 70 % of picks. The 93.5 % in 4.2 was measured on the same text it was ranked on; 89.4 %
  held out.
- **Error per storage option** on blk.20 experts, relative to Q4_K:

  | Option | Error vs Q4_K |
  |---|---|
  | Q2_K | 14.1x |
  | Q3_K | 4.4x |
  | IQ4_XS | 1.18x |
  | Q5_K | 0.25x |
  | Q6_K | 0.064x |
  | Q8_0 | 0.007x |
  | Q4_K + 1 plane | 0.36x |
  | Q4_K + 2 planes | 0.097x |
  | IQ4_XS + 1 plane | 0.38x |

  Down shows the same pattern. Scripts and results: `base_planes_check.py`, `hotset_coverage.py`.

### 6.3 The method: a usage-tiered expert file (Pelias-native)

**The idea.** Each expert of each layer gets its own storage type, chosen by how much routing weight it carries. Hot
experts get Q5_K/Q6_K/Q8_0, cold ones IQ4_XS (or Q3_K). Everything outside the experts stays exactly as in the Unsloth file.

**The allocation is optimal for the measured inputs:** it minimises usage-weighted error under a byte budget, by a
Lagrangian per expert. Script: `usage_alloc.py`, `usage_alloc_floor.py`.

**Results (all DERIVED).**
- The error is the held-out, usage-weighted expert error relative to the Unsloth file's experts (= 1).
- KL uses the line in 4.2.
- 22.65 GB is the Unsloth file's own size; 24.1 GB is +1.4 GB.

| Experts | Lowest type allowed | Total file | Error vs Unsloth | Projected KL |
|---|---|---|---|---|
| Unsloth (gate/up Q4_K, down Q5_K) | — | 22.65 GB | 1.00 | 0.0128 (MEASURED) |
| Tiered | IQ4_XS | **22.65 GB** | 0.35 | **about 0.0079 (-38 %)** |
| Tiered | Q3_K | 22.65 GB | 0.29 | about 0.0074 (-42 %) |
| Tiered | none (Q2_K allowed) | 22.65 GB | 0.28 | about 0.0072 (-44 %) |
| Tiered | IQ4_XS | 24.1 GB | 0.20 | about 0.0066 (-48 %) |
| Unsloth planes, 2 everywhere (MEASURED) | — | 31.9 GB | 0.10 | 0.0058 (MEASURED) |

**With the IQ4_XS floor at 22.65 GB, the optimiser gives:**

| Precision | gate/up | down |
|---|---|---|
| IQ4_XS (cold) | 6,188 | 6,227 |
| Q5_K | 2,962 | 2,914 |
| Q6_K | 1,069 | 1,081 |
| Q8_0 | 21 | 18 |

These are counts of layer-experts out of 10,240.

**Planes at a fixed size: an honest finding.** Offered planes as well as types, the optimiser hardly uses them: the error
at 22.65 GB is the same to three digits. At a fixed file size, a higher base type beats a plane bit for bit. So the
size-neutral win comes from **per-expert precision**, which GGUF cannot express. It does not come from planes.

**Where planes still earn their place:**
- On top of a file you cannot or will not requantize, the Unsloth file included. This is what was measured: −55 % KL and
  the 12-3 split result.
- Read only when needed: on flagged tokens, on hot experts only, through the dual bus. They cost disk but few bytes per
  token, which the fixed-size table does not count.
- As the accuracy reserve above the tiered file: for example, 1 plane on the top 32 experts per layer, about 0.5 GB, read
  on flagged tokens only.

### 6.4 Risks and what is UNMEASURED

- **Usage was measured on WikiText-2 only.** Code, math or chat may pick other experts. The IQ4_XS floor caps the damage:
  an expert the ranking got wrong is at most 1.18x worse than Q4_K, never Q2_K's 14x. A second routing capture on code and
  chat prompts settles the ranking's stability. Until then the floor stays at IQ4_XS.
- **Layers are weighted equally.** Their sensitivity to error is UNMEASURED; a per-layer KL sweep would refine the budget.
- **The error ratios come from blk.20 experts 0-3 only;** the rest is assumed alike.
- **The KL line has two points;** a tiered file is not uniform, so the line may not hold.
- **The file is Pelias-only.** llama.cpp cannot load mixed expert types in one tensor. A stock GGUF can still be written
  per layer (one type for all experts of a layer, the hottest layers higher), which is weaker. Not computed yet.
- **Engine work needed:**
  - the experts of a layer stored as one tensor per type plus a 256-entry map from expert to (type, row);
  - the loader, dequantizer and matvec choose per expert;
  - `quantize` gains a usage-tiered option fed by a routes file;
  - the gate, up and down kernels for IQ4_XS/Q5_K/Q6_K/Q8_0 already exist.

### 6.5 Proposed run list (needs the author's go and order)

1. **A routing capture on code and chat prompts.** `generate --prompt-list` with `--route-dump`, about 20 min. This checks
   that the hot set holds outside WikiText.
2. **The tiered quantize from BF16** (IQ4_XS floor, 22.65 GB), then `kld` with `--dump-codes`, `--dump-kld` and
   `--route-dump`. Build about 2-3 h, run about 1 h. Prediction: KL about 0.0079, between 0.0070 and 0.0095.
3. **The split-token decode and grade on the tiered file** (`--prompt-list`, the model held in RAM; `teacher`;
   `split_grade.py`). Prediction: at least as good as v8 (Pelias 7, llama 4); below the planes run (12-3).
4. **Tiered plus planes on the top 32 experts.** `kld` and decode, the planes on flagged tokens and on all tokens.
5. Then the runs of 4.4, cut to what is still open: a traditional Q5_K_M as the reference at about 27 GB.

## 7. Status

- Built and pushed: `pelias planes`; `--planes / --plane-k / --head-rescore` on `kld` and `generate`; `generate
  --planes-mode all|prefill|start:N` and `--prompt-list` (model loaded once); `pelias teacher` and
  `tools/split_grade.py` (grading against the original). Commits fdb63e1, 52c3eda, 76b1166 and the build-log entries after.
- GP3-08 (MTP) is paused (author 2026-10-30 05:08): the cost-down runs of 4.4 come first.

## Sources

- [Unsloth: Qwen3.5 GGUF benchmarks (KLD by file)](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) (English path UNVERIFIED; the [German copy](https://unsloth.ai/docs/de/modelle/qwen3.5/gguf-benchmarks) was found)
- [AnyBCQ: hardware-efficient flexible binary-coded quantization for multi-precision LLMs](https://arxiv.org/abs/2510.10467)
- [Any-Precision LLM (Park et al., 2024)](https://arxiv.org/abs/2402.10517)
- [BitStack: any-size compression of LLMs](https://arxiv.org/abs/2410.23918)
- [DynaExq: dynamic expert quantization for MoE inference](https://arxiv.org/abs/2511.15015)
- [MoPEQ: mixture of mixed-precision quantized experts](https://arxiv.org/abs/2509.02512)
- [MoBiQuant: mixture-of-bits quantization for token-adaptive elastic LLMs](https://arxiv.org/abs/2602.20191)
