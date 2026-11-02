# Residual sign planes: what was measured, the GPU plan, and how to keep the cost down

Confidential, not for distribution. Updated 2026-11-02 (section 8).

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

### 6.6 Refinement checks before the build (author 2026-10-30 05:58; offline)

**The hot set does not hold across texts (MEASURED).**
- Compared: WikiText-2 picks (urm_ZT2 route dump, 25,500 tokens) against the per-expert pick counts in Unsloth's own
  imatrix (their chat + code calibration text, 842,201 tokens).
- Script: `domain_stability.py`.

| Top N per layer, ranked on WikiText | Share of chat+code picks covered (mean / worst layer) | Ranked on chat+code itself | WikiText on itself |
|---|---|---|---|
| 32 | 19.1 % / 14.5 % | 30.9 % | 51.1 % |
| 64 | 35.9 % / 30.0 % | 49.4 % | 72.2 % |
| 120 | 59.9 % / 52.3 % | 72.2 % | 90.2 % |

- The rank correlation per layer is 0.41 on average (lowest 0.22, layer 14).
- Chat+code usage is also flatter: its own top 64 carry 49 % of its picks.

**What that does to 6.3 (DERIVED).** Files are 22.65 GB with an IQ4_XS floor; the cells are error vs Unsloth and the
projected KL. Script: `alloc_cross.py`.

| Usage ranked on | Scored on WikiText | Scored on chat+code |
|---|---|---|
| WikiText | 0.381, KL 0.0081 | 0.864, KL 0.0118 |
| chat+code | 0.735, KL 0.0108 | 0.653, KL 0.0102 |
| 50/50 mix | 0.430, KL 0.0085 | 0.723, KL 0.0107 |

- 6.3's -38 % holds only on the text the ranking came from. Ranked on a mix, the file is about -34 % on WikiText and about
  -16 % on chat+code.
- The KL line was fitted on WikiText. Chat+code's own KL is UNMEASURED.

**Cold floor, Q4_TG2 against IQ4_XS (MEASURED on blk.20, experts 0-3).** Script: `floor_norm_check.py`.

| Type | Bits per weight | gate error vs Q4_K | down error vs Q4_K |
|---|---|---|---|
| IQ4_XS | 4.25 | 1.18x | 1.17x |
| Q4_TG2 | 4.50 | 0.90x | 0.89x |

- IQ4_XS saves 5.6 % of the cold bytes (4.25 / 4.5), not 11 %.
- At a fixed 22.65 GB the two floors are about equal (DERIVED, `alloc_cross_tg2.py`, mix ranking):

  | Floor | WikiText KL | chat+code KL |
  |---|---|---|
  | Q4_TG2 | 0.0088 | 0.0107 |
  | IQ4_XS | 0.0085 | 0.0107 |

- Q4_TG2 caps a misranked expert at 0.90x instead of 1.18x and already has the shared-memory kernel (347 GB/s alone).
  With rankings this unstable, that cap matters.

**Output-norm scalar per expert: not worth it (MEASURED, diagonal input proxy).**
- Every type shrinks the activation-weighted output energy slightly: IQ4_XS 0.997, Q4_TG2 0.995, Q4_K 0.999, Q6_K 0.999.
- The best scalar per expert is 0.998-1.0003 and removes 0.00-0.06 % of the error.
- The shrink is noise, not a gain error, so a scalar cannot undo it.

**Projections before any PC run (author 06:06; DERIVED, Q4_TG2 floor, mix ranking).** Scripts: `context_planes.py`,
`alloc_bigger.py`.
- Context-hot planes: one plane per expert on disk (4.28 GB), read only for the top 32 experts per layer by routing weight
  so far in the context, from token 32 on.
- On WikiText 55 % of picks get their plane.
- The chat+code figure for that row assumes the same relative cut as WikiText; it is UNMEASURED (no per-token routes).

| Option | On disk | Expert MB read per token | KL WikiText | KL chat+code |
|---|---|---|---|---|
| Unsloth Q4_K_M | 22.65 GB | about 610 | 0.0128 (MEASURED) | UNMEASURED (about 0.0129 by the line) |
| Tiered | 22.65 GB | 641-689 | 0.0088 (-31 %) | 0.0107 (-17 %) |
| Tiered + context-hot planes | 26.93 GB | about 760 | 0.0075 (-41 %) | about 0.0091 (ASSUMED) |
| Tiered, the 4.28 GB put in the file | 26.93 GB | 786-838 | 0.0060 (-53 %) | 0.0065 (-50 %) |
| Unsloth + 2 planes | 31.9 GB | about 1,000 | 0.0058 (MEASURED) | — |

**Reading:**
- At the same disk size, putting bytes in the file beats context planes on accuracy; context planes read about 10 % fewer
  bytes per token.
- The +4.3 GB tiered file matches Unsloth's own finding for a traditional step up (+4 GB halves KLD), and also holds on
  chat+code.

**Still open: per-layer and gate/up-vs-down weighting.**
- These need gate, up and down from every layer; only blk.20 gate and down are in the sandbox.
- Next input: experts 0-1 of all 40 layers from the BF16 file, read only, with the existing `gp/M/extract_slice.py`
  (480 MB).
- Then two checks:
  - the activation-weighted error per layer and per type;
  - the error propagated through SwiGLU (silu(gate)·up into down). Inputs are drawn with the imatrix's variances, and the
    proxy is checked against down's measured input energy.

### 6.7 The 40-layer slopes and the two options (author 06:12; extraction MEASURED, projections DERIVED)

**Inputs (MEASURED).**
- Experts 0-1 of every layer, gate, up and down, from the BF16 original: `gp/TIER/slices*`, 120 matrices.
- Each quantized to Q4_K, Q4_TG2, Q5_K, Q6_K and Q8_0 with its own importance.
- Error measured three ways:
  - weight error (wrel);
  - error through SwiGLU relative to the expert's own output (orel);
  - the same in absolute terms (oabs).
- Scripts: `tier_slopes.py`, `tier_project.py`, `tier_vram.py`; `kqwp.cc` is the pipe helper.

**Findings.**
- **Relative errors are nearly flat across layers.**
  - Q4_K orel is 3.6-5.3e-3, layers 1-39 alike.
  - Layer 0 is lowest (2.2-3.5e-3), so in these measures the early layers do not stand out as needing protection.
- **Gate/up vs down.**
  - Down's orel matches gate/up in layers 0-31.
  - In layers 32-39 it is 10-35 % higher, so the allocator gives down slightly more: 3,500 Q5_K vs 3,124 for gate.
- **The SwiGLU proxy is weak.**
  - Its per-channel down-input energy correlates -0.18 to 0.63 with the imatrix.
  - So orel's gate/up-vs-down split carries that caveat. wrel and orel give the same answer within 0.0002 KL.
- **Absolute output error (oabs) would move bytes to layers 27-39** (5-6 bits per weight there) and project KL 0.0072 /
  0.0082.
  - It ignores that the residual stream also grows with depth.
  - How much a unit of error costs per layer is UNMEASURED. To measure it: `kld` with one group of 10 layers changed at a
    time (4 runs, about 25 min each).

**Option 1: static Q4_TG2-floor file, 50/50 mix ranking (orel).**
- VRAM is for the hot set, the top 64 experts per layer.
- "+ x GB in the file" puts the side file's bytes into the file instead, for comparison.

| File | KL WikiText | KL chat+code | VRAM, hot 64 | Expert MB per token |
|---|---|---|---|---|
| Unsloth, 22.65 GB | 0.0128 (MEASURED) | about 0.0129 | 4.89 GB | 612 |
| **Tiered, 22.65 GB** | **0.0089 (-30 %)** | **0.0108 (-16 %)** | 5.60 GB | 641-688 |
| + 1 GB in the file (23.65 GB) | 0.0077 | 0.0093 | 5.85 GB | 681-729 |
| + 2 GB in the file (24.65 GB) | 0.0070 | 0.0081 | 6.21 GB | 716-762 |

**Option 2: the 22.65 GB tiered base + block-selective planes.**
- The planes are chosen by weight error and layer slope only (Option A), so the choice does not depend on the text.
- If the CPU side computes the plane term, the planes need no VRAM. The CPU plane kernel is not built yet.
- MB per token is the extra read on the system-RAM bus, ESTIMATE from block share x picks.
- "Hot only" reads the planes only for the context-hot top 32 experts, which carry 55 % of routing weight on WikiText; it
  is applied as a flat factor (ASSUMPTION).

| Side file | Read | KL WikiText | KL chat+code | VRAM added | Extra MB per token |
|---|---|---|---|---|---|
| 0.5 GB | always | 0.0086 | 0.0101 | 0 | about 16 |
| 1.0 GB | always | 0.0083 | 0.0096 | 0 | about 31 |
| 1.0 GB | hot only | 0.0086 | 0.0101 | 0 | about 17 |
| 2.0 GB | always | 0.0079 | 0.0087 | 0 | about 63 |
| 4.23 GB | always | 0.0065 | 0.0072 | 0 | about 134 |

**Reading.**
- At the same total bytes, the file is more accurate. +2 GB gives 0.0070 / 0.0081 in the file, against 0.0079 / 0.0087 as
  planes.
- The planes add no VRAM and only CPU-bus reads; the file adds 0.25-0.6 GB of VRAM to the hot set.
- Reading block planes always beats reading them for context-hot experts only.
- The tiered file costs 0.71 GB more VRAM than Unsloth at the same size (5.60 vs 4.89 GB for the hot 64). That fits a
  12 GB card either way.

### 6.8 Planes for the most-weighted picks only (DPA-MoE's lesson; author 07:17; offline)

DPA-MoE (arXiv:2610.21378) found the router's own per-token scores carry most of its gain. Tested on the WikiText route dump
(`per_pick_planes.py`): an expert's output enters the layer as w x y_e, so its error counts as w^2; reading planes for a subset
of a token's 8 picks captures the subset's share of w^2 (ASSUMPTION: the plane cuts each expert's error by the same factor).

| Planes read for | Plane bytes read | Benefit kept (w^2) | Benefit kept (w) |
|---|---|---|---|
| top 1 pick per token | 12.5 % | 42.1 % | 24.5 % |
| top 2 | 25 % | 60.7 % | 41.4 % |
| top 4 | 50 % | **80.8 %** | 66.1 % |
| all 8 | 100 % | 100 % | 100 % |

Router weights by rank: 0.245, 0.168, 0.134, 0.113, 0.098, 0.088, 0.080, 0.074 (MEASURED).
DERIVED: the 2 GB worst-error side file read for the top 4 picks costs about 32 MB per token on the CPU bus (as the 1 GB file
read always) and projects KL about 0.0081 / 0.0091 (WikiText / chat+code), against 0.0083 / 0.0096 for the 1 GB file read
always. This replaces "context-hot" reading (6.7), which tested weaker.

### 6.9 The tiered file, MEASURED (2026-10-30 07:18-08:16)

**Build** (`block_tier_v1.sh tier`, 2,031 s):
- Expert matrices: Q4_TG2 20,311, Q5_K 9,801, Q6_K 608 (as predicted).
- Importance used on all 120 tiered weights.
- File 22,663,406,784 bytes, against Unsloth's 22,663,387,424. The extra 19,360 bytes are the type lists.

**`kld`**: 100 chunks, engine format, no planes, the same base and setup as the Unsloth run.

| File | Mean KLD | Same top | Label |
|---|---|---|---|
| Unsloth MTP UD-Q4_K_M | 0.012808 ± 0.000368 | 95.02 % | MEASURED |
| **Tiered, same size** | **0.009404 ± 0.000370** | **95.80 %** | MEASURED |
| Unsloth + 2 planes (31.9 GB) | 0.005818 ± 0.000169 | 96.77 % | MEASURED |

- Tiered vs Unsloth: **-0.003405 (-26.6 %)**, 95 % interval [-0.003949, -0.002860], t -12.41, settled.
- The tiered file has the lower KL on 66.3 % of tokens. The top token is right only with the tiered file 702 times, only with
  Unsloth 502 (z 5.76).
- Other statistics of the tiered run: ln(PPL ratio) 0.003129; 99.9 % KLD 0.400.
- **Prediction:** about 0.0089, range 0.0080-0.0100. Within range, 0.0005 above the point; the two-point line was slightly
  optimistic.

**Chunks 79-100** (author 08:16: "seemed to make it jump"; `tier_chunks.py`). The jump is the text, not the tiered file:

| Chunks | Tiered | Unsloth | Change | Unsloth + 2 planes |
|---|---|---|---|---|
| 1-50 | 0.009407 | 0.012464 | -24.5 % | 0.005884 |
| 51-78 | 0.008657 | 0.011810 | -26.7 % | 0.005192 |
| 79-100 | 0.010346 | 0.014862 | **-30.4 %** | 0.006463 |

- Chunk 79 is the hardest chunk for every file: Unsloth 0.0579, tiered 0.0374, planes 0.0122. Chunk 80 is fifth for the tiered
  file and fourth for Unsloth.
- The tiered file is lower on 94 of 100 chunks, median ratio 0.69. Where the text is hardest it cuts the most.

**Projections re-based on three measured points** (`kl_recal.py`; DERIVED). The error left maps to KL piecewise-linearly through
1.0 -> 0.012808, 0.484 -> 0.009404 and 0.098 -> 0.005818. Chat+code still has no measured KL. The previous two-point figure is in
brackets.

| Option | KL WikiText | KL chat+code |
|---|---|---|
| Tiered, 22.65 GB | **0.009404 (MEASURED)** | about 0.0110 (0.0108) |
| + 1 GB worst-error block planes, read always | about 0.0088 (0.0083) | about 0.0100 (0.0096) |
| + 2 GB, read always | about 0.0083 (0.0079) | about 0.0092 (0.0087) |
| + 2 GB, read for the top 4 picks | about 0.0085 (0.0081) | about 0.0096 (0.0091) |
| Tiered static, 23.65 GB | about 0.0080 (0.0077) | about 0.0097 (0.0093) |
| Tiered static, 24.65 GB | about 0.0071 (0.0070) | about 0.0085 (0.0081) |

### 6.10 GPU speed of the tiered experts, MEASURED (R3 probe, 2026-10-30 08:43)

**Setup.**
- One token: 40 distinct layers, cold, median of 21.
- Each layer's 7 GPU experts take the tiered types, drawn by measured use (`tier_r3_mix.py`).
- The baseline is Q4_TG2 gate/up + Q5_K down, the Unsloth sizes.
- Exactness check: all 10 formats and shapes match the host reference.

| Token | Routed experts | Pure streaming (b) | Real token path (c, per-expert counters) |
|---|---|---|---|
| Baseline | 532.2 MB | 4.766 ms | 5.693 ms |
| Tiered, WikiText use | 596.8 MB | 4.891 ms (+2.6 %) | 5.767 ms (**+1.3 %**) |
| Tiered, chat+code use | 560.2 MB | 4.816 ms (+1.0 %) | 5.710 ms (**+0.3 %**) |

- Predicted +0.13 / +0.06 ms on the stream: met.
- The new types stream at the same rate as the old ones (about 477 GB/s).
- The GPU side of the tiered file costs 0.3-1.3 % per token. The CPU leg (cold experts, planes) is not in R3; the engine
  measures it.

### 6.11 Three ways a user gets the accuracy (author 2026-10-30 09:17)

| Way | What the user needs | What changes | Accuracy (this model, WikiText) | Runs in |
|---|---|---|---|---|
| 1. A premade file as is | The file (Unsloth's, anyone's) | Nothing | Unsloth UD-Q4_K_M: KL 0.012808 (MEASURED) | Any engine that reads it |
| 2. Improve that file | A plane side file for that exact file: downloaded ready-made, or built once from the original weights | Nothing in the file; the side file is read next to it, and refused on any other file | + 2 planes everywhere (10.2 GB): 0.005818 (MEASURED); 1-2 GB worst-error planes on a stock file: UNMEASURED (measured so far on the tiered file only) | The file in any engine; the planes in Pelias |
| 3. A Pelias tiered quant | The original weights once, about an hour of PC time at 35B | The experts rebuilt by use, at the same size; planes optional on top | 0.009404 at Unsloth's size (MEASURED); + planes: chain running | Pelias only |

- **Why way 2 needs the original only once.** The planes store the gap between a file and the original weights. Someone who
  has the original builds the side file once, and anyone with the same file can use it: a 1-2 GB download instead of 71 GB.
- **Why ways 2 and 3 differ.** Way 2 keeps the file portable. Way 3 is the most accurate per byte, because the bytes go where
  the experts are used.

### 6.12 Block-selected planes on the tiered file, MEASURED (2026-10-30 09:09-10:09)

Chain gp/block_tierplanes_v1.sh on the 22.65 GB tiered file (100 WikiText chunks, judged against BF16).

| Run | Plane file | Blocks kept | Mean KL | Same top | vs tiered alone (0.009404) | vs Unsloth (0.012808) | Prediction |
|---|---|---|---|---|---|---|---|
| A1, every pick | 1,157,315,040 bytes | 29,411,764 (23.4 %) | 0.008963 ± 0.000330 | 95.847 % | -0.000441 (-4.7 %), 95 % [-0.000828, -0.000054], t -2.26 | -0.003846 (-30.0 %), t -12.27 | about 0.0088 (0.0082-0.0093): in range, above the point |
| A2, every pick | 2,157,315,360 bytes | 58,823,529 (46.7 %) | 0.008747 ± 0.000385 | 96.106 % | -0.000656 (-7.0 %), 95 % [-0.000975, -0.000338], t -4.09 | -0.004061 (-31.7 %), t -12.81 | about 0.0083 (0.0077-0.0089): in range, above the point |
| A2, top 4 of 8 picks (`--plane-picks 4`) | same as A2 | same as A2 | 0.008800 ± 0.000355 | 96.027 % | -0.000604 (-6.4 %), 95 % [-0.001019, -0.000188], t -2.88 | -0.004008 (-31.3 %), t -11.18 | about 0.0085 (0.0079-0.0091): in range, above the point |

A2: mean PPL ratio 1.003082; tokens where A2 has lower KL than the tiered file 53.51 %; top token right only in A2 416, only in the tiered file 338 (z 2.84). A1: PPL ratio 1.003742; lower KL than the tiered file on 52.01 % of tokens; top token right only in A1 384, only in the tiered file 372 (z 0.44, no top-token gain). Per GB (DERIVED): the first GB buys -0.000441 (-4.7 %), the second -0.000216 (-2.3 %), about half. A2 top-4: half the plane reads keep 92 % of A2's gain (0.000604 of 0.000656, DERIVED; 80.8 % was projected from the router weights); top token z 2.15 vs the tiered file. Default for the GUI: planes on the top 4 picks. The planes add 2.16 GB of RAM (no VRAM) for a further -7.0 % KL on top of the tiered file's -26.6 %.

## 7. Status

- Built and pushed: `pelias planes`; `--planes / --plane-k / --head-rescore` on `kld` and `generate`; `generate
  --planes-mode all|prefill|start:N` and `--prompt-list` (model loaded once); `pelias teacher` and
  `tools/split_grade.py` (grading against the original). Commits fdb63e1, 52c3eda, 76b1166 and the build-log entries after.
- GP3-08 (MTP) is paused (author 2026-10-30 05:08): the cost-down runs of 4.4 come first.

## 8. In the engine (2026-11-01/02, update of 2026-11-02)

Sections 1-7 are kept as written on 2026-10-29/30. Since then the engine runs, and two different kinds of plane are in play;
they must not be confused:

- **Bit planes of the stored codes** (built, MEASURED): the engine re-lays each Q4_TG2 / Q5_K expert matrix outside the
  bank into planes of its own code bits (top bit first) at open, so a reader can take 2-5 bits of every code from the same
  file (one slider, `--plane-bits`). These are not corrections; fewer bits are a rougher copy of the stored weight.
- **Residual sign planes** (this document, sections 1-7): one sign bit per weight plus a scale per 256 that move the stored
  weight *toward the original*. Measured on the CPU reference (sections 2, 6.9, 6.12); **not yet in the GPU engine.**

### 8.1 What the engine measured (labels as in the build log)

- The 1-bit path this document planned for the planes (section 1: "added by the 1-bit path (AND + popcount), which is idle
  during decode") is in the engine's expert stage since v55f: every plane of a Q5_K matrix is read as
  sum_k 2^k popcount(w AND u_k) - 128 popcount(w) against the inputs' bit planes. **Exact on the real model at 5, 3 and 2
  bits** (accuracy gates bit for bit equal to the earlier dp4a route, MEASURED) and **cheaper**: a Q5_K plane task 29,883 ->
  23,966 GPU cycles (-20 %, MEASURED). Q4_TG2's grid codes still go through the grid table.
- Plane A (the top 2 bits) is stored as two 1-bit planes (Q4_TG2 bits 3 / 2, Q5_K bits 4 / 3); same bytes.
- Accuracy by bits read (W1 gate, MEASURED): 5 bits (as stored) KL 3.767e-03, top-1 624 / 640, PASS; 4 bits 1.120e-02 FAIL;
  3 bits 4.415e-02, 580 / 640 FAIL; 2 bits FAIL. Rank matters more than count: with every pick at 2 bits the 4 top-ranked
  picks carry 83 % of the added error (v54, MEASURED). The author parked the sub-5-bit accuracy until the speed work is done
  (2026-11-02 00:31).
- Speed (decode median, MEASURED, v58): 13.66-13.73 ms per token at 5 bits (72.8 tok/s, gate PASS); 11.57-11.65 ms at
  3 bits (86.4 tok/s, gate FAIL). At 3 bits the bank holds 3,110-3,311 experts in the same VRAM as 2,062 at 5 bits.

### 8.2 Where the residual planes come back in

- The 3-bit read is the fast mode that fails the gate; a residual sign plane fitted to *the 3-bit copy* (not the stored
  file) is the correction this document describes, and the engine already has the 1-bit path to add it. Its accuracy and
  cost in the engine are **UNMEASURED**; the run is after the speed work (game plan order: speed, then P1, then P4 with the
  7+1 gated bit and residual sign planes).
- Section 6.8's lesson (planes for the most-weighted picks only, 92 % of the gain at half the reads) fits the measured rank
  effect above: the top 4 picks are where a correction pays (DERIVED from the two).

### 8.3 Status (replaces section 7's last line)

- GP3-08 (MTP) resumed and measured on the engine: output equal to plain decoding, 89.6 % of single drafts kept, but
  0.93-0.94x of plain speed (MEASURED, v57-v58); the fused draft path is built and its first run is pending.

## Sources

- [Unsloth: Qwen3.5 GGUF benchmarks (KLD by file)](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) (English path UNVERIFIED; the [German copy](https://unsloth.ai/docs/de/modelle/qwen3.5/gguf-benchmarks) was found)
- [AnyBCQ: hardware-efficient flexible binary-coded quantization for multi-precision LLMs](https://arxiv.org/abs/2510.10467)
- [Any-Precision LLM (Park et al., 2024)](https://arxiv.org/abs/2402.10517)
- [BitStack: any-size compression of LLMs](https://arxiv.org/abs/2410.23918)
- [DynaExq: dynamic expert quantization for MoE inference](https://arxiv.org/abs/2511.15015)
- [MoPEQ: mixture of mixed-precision quantized experts](https://arxiv.org/abs/2509.02512)
- [MoBiQuant: mixture-of-bits quantization for token-adaptive elastic LLMs](https://arxiv.org/abs/2602.20191)
