# PEL quantization: the decision data (2026-10-27, v5, 11:00 local: T1–T8 results, noise calibration, bench set)

Everything measured or derived so far that decides which tensor groups get which type in PEL. It covers the drift tests, the engine (hot bank, VRAM, CPU reads, kernels) and the trainer (Mode Q, memory, streaming).

**Labels:**
- **MEASURED:** run on your machine, with a log.
- **DERIVED:** arithmetic from labelled inputs, shown or traceable.
- **READ-SRC:** read from a file or from source code.
- **PUBLISHED:** someone else's documentation.
- **ESTIMATE:** a stated guess, with the way to replace it.
- **UNVERIFIED:** not checked yet.
- **DECISION:** a ruling already made.

Sources are the RUNBOOK, TALLY.csv, `docs/PELIAS_SYSTEM_MASTER.md` (sections 4.1 to 4.9) and `docs/PELIAS_SYSTEM_MASTERWORK.md` (sections 6.2, 17.5).

---

## 0. The cost model: why PEL is chosen by where each byte lives

The published recipes were tuned for other hardware, so their choices don't transfer:

- **Apple (MLX, unified memory):** every byte costs the same bandwidth, so the right answer is simply the smallest file at a given drift.
- **Blackwell (NVFP4):** the card does 4-bit math natively and has VRAM to spare, so keeping every dense weight at 16-bit is cheap there.
- **Pelias on this card:** a byte costs a different amount depending on where it lives:

| Where a byte lives | What it costs | Rate (DERIVED, section) |
|---|---|---|
| Dense spine (VRAM, read every token) | Hot-bank room once the 0.16 GiB headroom is used up | about 4.5 tok/s per GiB of bank lost (5.3) |
| Routed experts (bank in VRAM, the rest in RAM) | Bank capacity (experts per GiB) and cold reads; only 8 of 256 read per token | +1 % experts per GiB ≈ +0.003 hit rate at 7.5 GiB (5.4) |
| Trained core (Mode Q) | Trainer headroom, twice (base + `W_eff` copy) | 1.7–2.0 GiB available (6.3) |
| CPU-only path (exploratory) | Every dense byte read every token | 45.9 GB/s (5.5) |

So PEL ranks each choice by **drift removed per byte, in the place that byte goes**, using the sweep (section 3) for the drift side and sections 5–6 for the cost side.

---

## 1. Goal and bars

| Item | Value | Label |
|---|---|---|
| Goal | A 4-bit PEL file that drifts less from bf16 than stock Q4_K_M and Unsloth's UD-Q4_K_M, judged together with size and speed | DECISION (author) |
| Bar to beat | Unsloth UD-Q4_K_M: mean KLD **0.013605**, same top token 94.851 %, file 22,134,528,992 bytes | MEASURED |
| Size target | **≤ 22.13 GB without blk.40, or ≤ 22.66 GB with blk.40** (see 2.3) | DERIVED |
| CPU-only speed | Exploratory sub-goal (author target about 40 tok/s); it is recorded but cannot veto a choice | DECISION (author) |
| Drift test | llama-perplexity `--kl-divergence` against the bf16 reference, WikiText-2 test, 100 chunks of 512 tokens (51,200 tokens), `-ngl 0 -t 8`; deterministic (a repeat gave identical digits) | MEASURED |

## 2. Whole-file drift results (same test, same bf16 base)

| File | Mean KLD | Same top p | PPL (bf16 6.6153) | Bytes |
|---|---|---|---|---|
| Stock Q4_K_M, made by llama.cpp | 0.040965 ± 0.000733 | 91.329 % | 6.6897 | 21,713,463,296 |
| Stock Q4_K_M, made by Pelias | 0.041124 | 91.208 % | 6.6983 | 21,713,463,296 |
| (a) stock types + Unsloth's imatrix | 0.027317 ± 0.000498 | 92.796 % | 6.7136 | — |
| (b) Unsloth's type layout, no imatrix | 0.014174 ± 0.000370 | 94.792 % | 6.6495 | 22,660,245,536 |
| Unsloth UD-Q4_K_M (types + imatrix) | 0.013605 ± 0.000385 | 94.851 % | 6.6365 | 22,134,528,992 |

All MEASURED.

### 2.1 What this says (DERIVED)

- **The type layout does nearly all the work.** It takes stock 0.040965 down to 0.014174, a drop of 0.026791.
- **The imatrix adds little on top of a good layout.** On top of Unsloth's layout it is worth 0.000569. On stock types it is worth 0.013648, because there it is rescuing tensors left at low bits.
- **Pelias vs llama.cpp stock:** the 0.00016 gap between the two stock files comes from the 13 ssm_a values. The bf16 reference carries llama.cpp's rounding of those values. Accepted (DECISION).
- **PPL is not reliable here.** (a) has lower KLD than stock but higher PPL. KLD is the judge; PPL is only a sanity check.

### 2.2 Unsloth's layout (READ-SRC, from its file)

- **Q8_0:** every dense weight (`attn_gate`, `attn_k`, `attn_output`, `attn_q`, `attn_qkv`, `attn_v`, all `shexp`, `ssm_out`, `token_embd`).
- **F32:** `ssm_alpha`, `ssm_beta`.
- **Q5_K:** `ffn_down_exps`, except **Q6_K** in blocks 34, 38 and 39.
- **Q4_K:** `ffn_gate_exps`, `ffn_up_exps`.
- **Q6_K:** `output`.
- **Dense bytes:** 2.56 GB, against stock's 1.65 GB.

### 2.3 Why our files are 0.53 GB bigger than Unsloth's: blk.40 (DERIVED, exact to the byte)

- blk.40 (the MTP block) under (b)'s layout comes to exactly 525,716,544 bytes. That counts `nextn.eh_proj`, which matched no regex and so stayed at Q4_K.
- (b) minus blk.40 equals 22,134,528,992 bytes, which is Unsloth's file exactly.
- Unsloth's docs say their MTP builds are separate repositories (PUBLISHED). So their standard file has no MTP, and the layouts are otherwise identical.
- We keep blk.40 because the engine will run MTP.

---

## 3. The 8-bit sensitivity sweep: one group at Q8_0, everything else stock (2026-10-27, 116 min)

The base is stock Q4_K_M made by llama.cpp (0.040965). Each run raises one group to Q8_0 and runs the same KLD test.

- "Reads per token" is the extra bytes read on every decoded token: the full delta for dense groups, 8/256 of the delta for experts, and one row for `token_embd`.
- "KLD drop per GB" is the drop divided by the file growth.

| Group at Q8_0 | Mean KLD | Drop vs stock | Drop % | Same top p | PPL | File bytes | File Δ | KLD drop per GB of file | Reads per token Δ (MB) | KLD drop per GB read |
|---|---|---|---|---|---|---|---|---|---|---|
| `ssm_out` | 0.031616 ± 0.000576 | **0.009349** | 22.8 % | 92.373 % | 6.7182 | 21,839,292,448 | +125,829,152 | **0.0743** | +125.8 | 0.074 |
| `attn_gate` | 0.038228 ± 0.000649 | 0.002737 | 6.7 % | 91.576 % | 6.6818 | 21,839,292,448 | +125,829,152 | 0.0218 | +125.8 | 0.022 |
| `attn_output` | 0.038266 ± 0.000775 | 0.002699 | 6.6 % | 91.510 % | 6.6666 | 21,759,600,672 | +46,137,376 | **0.0585** | +46.1 | 0.058 |
| `ffn_up_exps` | 0.038581 ± 0.000691 | 0.002384 | 5.8 % | 91.545 % | 6.6829 | 27,216,390,176 | +5,502,926,880 | 0.0004 | +172.0 | 0.014 |
| `ffn_gate_exps` | 0.038838 ± 0.000679 | 0.002127 | 5.2 % | 91.506 % | 6.6853 | 27,216,390,176 | +5,502,926,880 | 0.0004 | +172.0 | 0.012 |
| `shexp` (gate, up, down) | 0.038866 ± 0.000737 | 0.002099 | 5.1 % | 91.604 % | 6.6990 | 21,772,273,696 | +58,810,400 | 0.0357 | +58.8 | 0.036 |
| `attn_qkv` (clean rerun) | 0.038958 ± 0.000815 | 0.002007 | 4.9 % | 91.635 % | 6.6945 | 21,904,566,304 | +191,103,008 | 0.0105 | +191.1 | 0.011 |
| `ffn_down_exps` | 0.039551 ± 0.000683 | 0.001414 | 3.5 % | 91.525 % | 6.6809 | 25,763,063,840 | +4,049,600,544 | 0.0003 | +126.6 | 0.011 |
| `attn_q` | 0.039623 ± 0.000745 | 0.001342 | 3.3 % | 91.447 % | 6.7051 | 21,805,738,016 | +92,274,720 | 0.0145 | +92.3 | 0.015 |
| `attn_v` (clean rerun) | 0.040377 ± 0.000835 | 0.000588 | 1.4 % | 91.435 % | 6.6904 | 21,717,338,144 | +3,874,848 | 0.1517 | +3.9 | 0.152 |
| `token_embd` | 0.040481 ± 0.000721 | 0.000484 | 1.2 % | 91.247 % | 6.6972 | 21,967,743,008 | +254,279,712 | 0.0019 | ~0 (one row) | n/a |
| `output` | 0.040585 ± 0.000733 | 0.000380 | 0.9 % | 91.373 % | 6.6879 | 21,836,630,048 | +123,166,752 | 0.0031 | +123.2 | 0.003 |
| `attn_k` | 0.040698 ± 0.000731 | 0.000267 | 0.7 % | 91.251 % | 6.6909 | 21,719,230,496 | +5,767,200 | 0.0463 | +5.8 | 0.046 |
| `ssm_alpha` + `ssm_beta` | 0.041044 ± 0.000725 | −0.000079 | −0.2 % | 91.231 % | 6.6926 | 21,715,429,408 | +1,966,112 | none | +2.0 | none |

MEASURED: KLD, PPL, top p, bytes. DERIVED: the deltas and the ratios.

### 3.1 Reading the sweep

- **`ssm_out` is the single most drift-causing group.** At Q8_0 it removes 22.8 % of stock's drift for 0.126 GB, about 3.4 times the next group. Unsloth's own Qwen3.5 tests (PUBLISHED) and bartowski's sensitivity maps (PUBLISHED) single it out too. It is the answer to your proposal 2, "all 4-bit plus the one most drift-causing group at 8-bit".
- **Second tier, each about 0.002 to 0.0027:** `attn_gate`, `attn_output`, `shexp`, `attn_qkv`. Then `attn_q` at 0.0013.
- **Cheap but small:** `attn_v` (0.0006 for 3.9 MB) and `attn_k` (0.0003 for 5.8 MB). Their ratio per GB is the highest of all, but each gain is near the noise threshold. Together they cost about 10 MB, so they're worth including if a combined run shows they help.
- **Experts at Q8_0 help, but they cost 4 to 5.5 GB each.** That's 30 to 50 times worse per GB than the dense groups. `ffn_down_exps` gains least of the three, because stock already gives half its layers Q6_K (READ-SRC: Q6_K-down layers 0–4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34–39).
- **No measurable gain:** `token_embd`, `output`, `ssm_alpha|beta`.
- **The singles don't simply add up.** They sum to about 0.028; Unsloth's layout gets 0.0268 with experts at Q4_K/Q5_K. Combined runs measure combinations (section 9).
- **Noise:** each ± is the standard error over tokens (MEASURED). The runs share the same tokens, so the differences between runs are tighter than ±0.0007, but how much tighter isn't measured. Treat drops under about 0.0005 as not separated (ESTIMATE). A second evaluation text would settle it (section 10).

### 3.2 Prediction check (written before the sweep)

- **Wrong:** I said `ffn_down_exps` would give the largest drop, to 0.030–0.035. It is 8th.
- **Right:** dense groups alone at 0.036–0.040. `ssm_out` was the exception, landing better at 0.0316.
- **Right:** `token_embd`/`output` change less than 0.001.
- **Right:** the research pick of `ssm_out` as the 8-bit group.

### 3.3 The anomaly, solved: a shared counter in llama-quantize (MEASURED from the quantize logs)

**What happened:**
- The first `attn_v` run made the file 450,528 bytes *smaller*.
- The quantize logs show that llama-quantize counts `attn_v` and `attn_qkv` on one shared counter, so overriding one shifted the other's per-layer types:
  - the `attn_qkv` run moved 4 `attn_v` layers (blocks 11 and 15 up to Q6_K, 35 and 39 down to Q4_K);
  - the `attn_v` run moved 13 `attn_qkv` layers, several from Q6_K down to Q4_K.
- The other 12 runs changed only their own group, so their results stand.

**The fix:** both groups were rerun with the other tensor's stock per-layer types pinned from the clean run 0. The check "non-q8 type changes vs run 0" printed nothing for either rerun, so they are clean.

| Group | Confounded run | Clean rerun | Prediction (written before) |
|---|---|---|---|
| `attn_v` | 0.041196 (worse than stock) | **0.040377** (drop 0.000588) | 0.0408–0.0412: wrong, slightly better |
| `attn_qkv` | 0.038921 | **0.038958** (drop 0.002007) | 0.0386–0.0392: right |

**Lessons:**
- **Overrides in llama-quantize can have side effects.** Every candidate built with it needs the same type-list check.
- **Pelias's own quantizer avoids this.** It has no shared counters, which is a reason to build PEL candidates with `pelias quantize` once per-group overrides exist.

---

## 4. Tensor byte map (stock file; READ-SRC from its header unless marked)

| Group | Parameters (40 trunk layers) | Stock size (MiB) | Q8_0 size (MB) | Used in decode | Trainable (LoRA) |
|---|---|---|---|---|---|
| `ffn_gate_exps` | 10,737.4 M | 5,904.0 | 11,409 | 8 of 256 per layer | No (frozen, spliced) |
| `ffn_up_exps` | 10,737.4 M | 5,904.0 | 11,409 | 8 of 256 | No |
| `ffn_down_exps` | 10,737.4 M | 4,410.0 (Q6_K) + 2,880.0 (Q4_K) | 11,409 | 8 of 256 | No |
| `token_embd` | 508.6 M | 272.8 (Q4_K) | 540 | 1 row | No |
| `output` | 508.6 M | 397.9 (Q6_K) | 540 | All | No (an option) |
| `attn_qkv` (DeltaNet) | 503.3 M | 183.8 + 144.0 | 535 | All | Yes |
| `attn_gate` (DeltaNet z) | 251.7 M | 135.0 | 267 | All | Yes |
| `ssm_out` (DeltaNet out) | 251.7 M | 135.0 | 267 | All | Yes |
| `attn_q` (full attention) | 167.8 M | 99.0 | 178 | All | Yes |
| `attn_output` | 83.9 M | 49.5 | 89 | All | Yes |
| `shexp` (3 tensors) | 125.8 M | 23.1 + 23.1 + 17.2 + 11.2 | 134 | All | Yes |
| `ffn_gate_inp` (routers, F32) | 21.0 M | 82.0 | (kept F32) | All | Mode L only |
| `attn_k` / `attn_v` | 10.5 M each | 6.2 / 5.7 + 2.2 | 11 each | All | Yes |
| `ssm_alpha` / `ssm_beta` | 2.0 M each | 1.1 each | 2 each | All | Yes |
| `ssm_conv1d` | 1.0 M | 3.8 | n/a | All | No |
| blk.40 (MTP block) | 844.6 M | about 500 | about 530 under (b) | When drafting | Co-trained (λ 0.1–0.3) |

- **Sources:** the parameter and Q8_0 columns are DERIVED from the served header's shapes. Stock MiB is READ-SRC (master 4.3).
- **Block sizes (READ-SRC):** Q4_K 144 B, Q5_K 176 B, Q6_K 210 B per 256 weights; Q8_0 34 B per 32 weights.

---

## 5. How the engine uses each tensor (decode)

### 5.1 Placement

- **Dense weights (the "spine": attention, DeltaNet, shared expert, routers, `output`):** live in VRAM and are read on every token.
- **Routed experts:** a hot bank in VRAM holds the most-used experts. The rest are read from RAM by the CPU (cold path) or by the GPU through zero-copy.
- **The trade-off:** every byte added to the dense spine comes out of the bank at a fixed VRAM ceiling.

### 5.2 Hot bank and hit rate

| Item | Value | Label |
|---|---|---|
| Simulated hit rate by bank size (stock bytes per expert) | 0.43 / 0.54 / 0.63 / 0.70 / 0.74 / 0.76 / 0.85 at 2 / 3 / 4 / 5 / 5.6 / 6 / 8 GiB; 0.8273 at 7.5 GiB | DERIVED (simulated, TM 7.12; EB) |
| Speed rule | About +1 tok/s per +0.01 of hit rate | DERIVED (TM 7.12) |
| Live hit rate, 7.0 GiB bank | 0.7753 to 0.7793 at 1 token; 0.733 at 2 to 31 tokens; simulated 0.806 | MEASURED |
| By layer | Layers 0–2 lowest (0.61–0.65), layer 20 highest (0.90–0.92) | MEASURED |
| By domain, 6.1 GiB bank | Agent 0.790, code 0.753, persona 0.799, reasoning 0.674 | MEASURED |
| Peak VRAM by bank (whole card) | 7.0 GiB bank `-ub 4096`: 10.7; 7.5 GiB: 10.40–10.44 (default micro-batch), 10.67–10.69 (`-ub 2048`); 8.0: 11.0–11.19; 8.5: 11.4 | MEASURED |
| **Headroom at the 7.5 GiB bank** | 0.26–0.30 GiB (default micro-batch); **0.16 GiB after the 0.1 GiB margin**; none at `-ub 2048` | DERIVED |
| Free VRAM | Total 11.99 GiB; CUDA free after the context 10.83 / 10.85; Windows holds 914 / 1,031 MiB before start (WSL / Windows) | MEASURED |
| Ceiling | 10.7 GiB, your machine's typical free VRAM (could be lower); the plan will read the real free VRAM at each start | AUTHOR-STATED / DECISION |

### 5.3 What dense bytes cost the bank (DERIVED, rough)

- **The rate:** between the 6.0 and 7.5 GiB banks the hit rate rises (0.8273 − 0.76) / 1.5 GiB ≈ **0.045 per GiB**. Using the speed rule, that's about **4.5 tok/s per GiB of bank**, so **about 0.45 tok/s per 0.1 GiB** of dense growth that comes out of the bank.
- **`ssm_out` at Q8_0:** +0.117 GiB fits inside the 0.16 GiB headroom with no bank change.
- **`ssm_out` + `attn_output` at Q8_0:** +0.160 GiB uses the headroom almost exactly.
- **All dense at Q8_0 (Unsloth-style):** about +0.85 GiB. The bank drops from 7.5 to about 6.65 GiB, the hit rate falls about 0.04, which is about **−4 tok/s** (DERIVED from the linear rule; real cost UNMEASURED until the engine runs).
- **blk.40 not included:** none of the tier's VRAM measurements included blk.40, because llama.cpp ignores it. MTP's VRAM in Pelias is **UNMEASURED**.

### 5.4 Expert bytes and the bank (DERIVED)

The bank holds experts, not GiB. Bytes per expert set how many experts fit.

| Expert layout | Bytes per expert (gate + up + down) | Experts per GiB vs stock |
|---|---|---|
| Stock (down Q6_K in 20 layers, Q4_K in 20) | 2,039,808 / 1,769,472; average 1,904,640 | 1.000 |
| Unsloth (down Q5_K; Q6_K in 34, 38, 39) | 1,900,544 (2,039,808 in 3 layers) | about 1.0 (neutral) |
| Gate/up IQ4_XS, down Q5_K (untested) | 1,835,008 | about +3.8 % more experts than stock, about +0.01 hit rate at 7.5 GiB |
| Gate/up Q8_0 (stock down) | about 2.95 M | −36 %: the bank loses about a third of its experts |

### 5.5 CPU reads (O21, exploratory)

| Item | Value | Label |
|---|---|---|
| Stock decode bytes per token | 1,975,676,544 (dense 1,366,190,592 + 8/256 of experts 609,484,800 + one embedding row) | DERIVED |
| Memory read | 45.9 GB/s measured; 51.2 GB/s peak | MEASURED / PUBLISHED |
| Ceiling, stock | 23.2 tok/s (45.9) / 25.9 (51.2) | DERIVED |
| Measured, CPU only | 16.28 ± 0.11 tok/s (llama-bench tg128, `-ngl 0 -t 8`) = 70.2 % of ceiling | MEASURED |
| Unsloth-style dense Q8 | Ceiling about 17.5 tok/s; at the same 70 % about 12 tok/s | DERIVED, rough |
| Dense in VRAM, CPU reads experts only | 45.9 / 0.609 GB = about 75 tok/s ceiling (the Pelias split, not CPU-only) | DERIVED |
| 40 tok/s CPU-only | Needs ≤ about 1.15 GB read per token, or MTP averaging about 1.72 accepted tokens per pass | DERIVED |

**Rule that follows:** on the CPU, bits on dense weights cost speed every token, while bits on experts cost only 8/256 as much. On the GPU split the dense weights sit in VRAM, so the CPU cost of dense bits disappears, but the VRAM cost (5.3) appears.

### 5.6 Recorded engine speeds (patched llama.cpp tier, stock file)

| Item | Value | Label |
|---|---|---|
| Best WSL2 cell (7.5 GiB bank, bypass 32, `-ub 2048`) | 53.5 short / 53.0 at 4k / 49.0 at 12k tok/s | MEASURED |
| Windows-native cell A | 47.3 / 46.8 short; 53.7 / 54.1 at 12k | MEASURED |
| Ceiling model | 92.0 tok/s (26.2 GB/s kernel), 88.2 (9.59 GB/s kernel) | DERIVED |
| MTP draft 2/3 at 7.0 GiB bank | 12k decode 51.5–52.9 against 45.4 base | MEASURED |
| Where WSL2 time goes | CPU cold experts 10.1 ms vs 2.86 modelled; GPU 6.6 ms; handoff and copies 4.6–5.0 ms; 63.9 % of layer-steps have at least one cold expert | MEASURED / DERIVED |

### 5.7 Kernels: what each type costs to run (Batch C, Nsight Compute, microbenchmarks)

| Item | Value | Label |
|---|---|---|
| Streamed dequantization (P-K3) | Q4_K 487 of 501–506 GB/s; Q6_K 496–501 of 510 GB/s, so 96–98 % of a plain read. Dequantization is free in bandwidth | MEASURED |
| Int8 path for Q4_K (P-K2, kernel v10) | 1.39×–1.84× the bf16 path per layer incl. norm and conversions, on all 20 shapes; cosine to the bf16 path **0.999974** | MEASURED |
| Why int8 stops near a third of peak (Nsight Compute, r8–r10) | Q4_K's 6-bit sub-block scales cost extra instructions per mma (13.4 per mma at v10; 37.7 % tensor busy); the scales don't fit inside an int8 operand | MEASURED |
| Grouped expert launch (P-K2b) | 256 experts × M=64 in one launch: 2.5×–3.0× the bf16 kernel | MEASURED |
| Zero-copy expert reads | 26.2 GB/s checksum kernel; **9.59 GB/s compute kernel on Q6_K's 420-byte rows**; the gap is open (gate E5) | MEASURED |
| Q8_0 on the int8 path | Q8_0 weights are already int8 with one scale per 32 values, so they should map onto int8 mma more directly than Q4_K | ESTIMATE; no kernel measured |
| Q5_K, Q3_K, IQ4_XS | No Pelias quantizer and no Pelias kernel yet (O19). Choosing them means building and proving both (P-E6-style byte identity) | READ-SRC (repo state) |

---

## 6. How the trainer uses each tensor

### 6.1 What trains and what stays frozen

| Item | Value | Label |
|---|---|---|
| Frozen | All routed experts (123 tensors incl. blk.40, 18.65 GiB): pinned in RAM, never updated, byte-spliced into the export with a SHA-256 proof | DECISION / READ-SRC |
| Trained (LoRA) | **310 modules: 10 × 4 full attention (q, k, v, o), 30 × 5 DeltaNet (qkv, z = `attn_gate`, a, b, out = `ssm_out`), 40 × 3 shared expert.** Rank 16 = 21,166,080 parameters | READ-SRC / DERIVED |
| Matches the sweep | **The trained modules are exactly the dense groups the sweep ranked.** Their type affects inference drift and how much of the fine-tune survives | DERIVED |
| Mode Q (default) | Every step builds `W_eff = Q(deq(W_base) + (α/r)·B·A)` at the **served type**, with Pelias's bit-exact quantizer; the forward pass uses `W_eff`; the export is the final `W_eff` (+0 bytes) | DECISION (D23, D31) |
| Merge erase (F4) | A LoRA update merged into Q4_K is erased if smaller than half a step: Q4_K step about 0.320σ, rounding noise 9.2 % σ | DERIVED (spec 6.2) |
| Finer grids keep more of the fine-tune | Q8_0 has 255 levels per block against 16 for 4-bit, so its step is roughly 16× finer | DERIVED from the formats; the real retention is measured by P-E8 / K2 |
| Byte-budget solver | Raising the core classes: all Q6_K +282 MiB, all Q8_0 +607 MiB. Off by default; Expert mode | DECISION (D22, D32) |

**Core sizes by class, MiB (DERIVED, master 4.3):**

| Class | Parameters | Served mix | All Q8_0 | All Q6_K | bf16 |
|---|---|---|---|---|---|
| Shared experts | 125.8 M | 72.7 | 127.5 | 98.4 | 240.0 |
| DeltaNet in (qkv + z) | 755.0 M | 462.8 | 765.0 | 590.6 | 1,440.0 |
| DeltaNet alpha, beta | 3.9 M | 2.1 | 4.0 | 3.1 | 7.5 |
| DeltaNet out (`ssm_out`) | 251.7 M | 135.0 | 255.0 | 196.9 | 480.0 |
| Full attention | 272.6 M | 147.8 | 276.2 | 213.3 | 520.0 |
| **Total** | **1,409.0 M** | **820.4** | **1,427.7 (+607)** | **1,102.3 (+282)** | **2,687.5** |

### 6.2 Trainer VRAM (window 8,192, step up to 32k tokens; ceiling 10.7 GiB)

| Item | GiB | Label |
|---|---|---|
| CUDA context | 0.60 | ESTIMATE |
| Frozen core base (served bytes) | 0.80 | READ-SRC |
| `lm_head` (Q6_K) | 0.39 | READ-SRC |
| Token embedding (Q4_K) | 0.27 | READ-SRC |
| Routers and small F32 | 0.09 | READ-SRC |
| Expert slots, 2 × 498 MiB | 0.97 | READ-SRC |
| LoRA masters, gradients, Adam (r 16) | 0.32 | DERIVED |
| LoRA bf16 copies | 0.04 | DERIVED |
| Layer boundaries (fp32, 32k tokens) | 0.75 | DERIVED |
| Per-layer working set | 2.50 | ESTIMATE (P-M2 measures) |
| Cross-entropy chunk | 0.65 | MEASURED (v0.2) |
| Routing tape | 0.02 | DERIVED |
| Safety margin | 0.50 | DECISION |
| **Total** | **7.90** | |
| **Headroom** | **2.8**; with the Mode Q `W_eff` copy (+0.8 to 1.1) **1.7 to 2.0** | DERIVED |

Measured history: the v0.2 peaks were 8.02 GiB (run 1) and 9.36 GiB (second phase).

### 6.3 What PEL core choices cost the trainer (DERIVED)

The `W_eff` copy is the size of the core at its served types: 820 MiB, which matches its +0.8 GiB. So raising a core class costs its bytes twice, once for the base and once for the copy. This is an inference until P-M2 measures it (UNVERIFIED).

| Choice | Core Δ | With `W_eff` copy | Headroom left (from 1.7–2.0) |
|---|---|---|---|
| `ssm_out` Q8_0 | +120 MiB | +0.23 GiB | 1.5–1.8 |
| `ssm_out` + `attn_output` + `shexp` Q8_0 | +120 + 35 + 53 MiB ≈ 208 MiB | +0.41 GiB | 1.3–1.6 |
| All core Q8_0 (Unsloth-style) | +607 MiB | +1.19 GiB | 0.5–0.8 |
| All core Q6_K | +282 MiB | +0.55 GiB | 1.2–1.5 |
| `token_embd` Q8_0 | +0.24 GiB (not doubled; not trained) | +0.24 | |
| `output` Q8_0 | +0.12 GiB | +0.12 | |

The headroom competes with a third expert slot (+0.49 GiB), LoRA rank 32 (+0.32) or rank 64 (+0.96), and resident expert layers (+0.42 to 0.49 each).

### 6.4 Experts in training: streaming, host memory, traffic

| Item | Value | Label |
|---|---|---|
| Streamer | 79 fetches, about 35.8 GiB per step regardless of tokens; 2 × 18.16 GiB per step at 26.1 GB/s = 1.49 s | DERIVED |
| v0.2 for comparison | 72 fetches, 30.38 GiB per micro-batch, about 121.5 GiB per optimizer step; stall 26.8–29.5 % | MEASURED / DERIVED |
| Host | Pinned expert store 18.65 GiB; boundary store 5.0 GiB (16k) or 10.0 GiB (32k); total about 28–33 GiB; WSL2 cap 48 GiB | DERIVED |
| Pinned transfers | 26.3 GB/s each way; 32 GiB pinned held on both platforms; 24 GiB pinned held 30 min with no swap | MEASURED |
| Slot size | Largest expert layer: 498 MiB (Q6_K down) | READ-SRC |
| Unsloth expert layout | 464 MiB per layer (Q5_K down), 498 in blocks 34, 38, 39: same slot size, traffic about +0.3 % | DERIVED |
| Gate/up Q8_0 | About 750 MiB per layer: slots +0.5 GiB, pinned store about +10 GiB (to about 29 GiB), traffic about +55 % | DERIVED |

### 6.5 Trainer speed context

| Item | Value | Label |
|---|---|---|
| Targets | 600 tok/s minimum, 1,200 target, 2,000 stretch (v0.2: 247.85 median) | DECISION / MEASURED |
| Int8 share of FLOPs | 52 % at window 8,192 | DERIVED |
| Measured tensor rates | bf16 65.2 TFLOPS, int8 259.5 TOPS, fp8 129.8 | MEASURED (P-H3) |

---

## 7. Published evidence from other quantizers (same model or family)

| Source | What they found or did | Label |
|---|---|---|
| Unsloth, Qwen3.5 GGUF benchmarks | `ssm_out` low "dramatically increases KLD"; any `attn_*` is "especially sensitive for hybrid architectures"; `ffn_down_exps` "slightly more sensitive"; `ffn_up_exps`/`ffn_gate_exps` "generally ok to quantize to 3bit" | PUBLISHED |
| bartowski, per-tensor maps | `attn_v`, `attn_output`, `ffn_up`, `ssm_out` "extreme standouts"; `shexp` "pretty sensitive despite being incredibly small"; `ffn_gate` "basically never worth any extra bits"; early and late layers most sensitive (U shape) | PUBLISHED |
| ubergarm, Qwen3.5-35B-A3B | All attention, `ssm_out`, `shexp`, `token_embd`, `output` at Q8_0; `ssm_alpha`/`beta` F32; `ffn_down_exps` IQ5_KS, gate/up IQ4_KS | PUBLISHED |
| ik_llama, Qwen3 tests | Upgrading `ffn_down_exps` gave the largest PPL drop per GiB in their ladder | PUBLISHED |
| NVFP4 recipes (ModelOpt, RedHat, Infatoshi, kyaky) | Keep linear attention (DeltaNet) at BF16 or FP8 almost always; routers BF16; only routed experts at 4-bit | PUBLISHED |
| mlx-lm default | Only router and shared-expert gate at 8-bit | READ-SRC |

**How our sweep relates:** it agrees on `ssm_out` and on the dense groups generally, and on gate/up experts being poor value per GB. It disagrees on `attn_v`, which shows no gain here. Our `ffn_down_exps` result is weaker than ik_llama's, but stock already gives half its layers Q6_K.

---

## 8. What is ruled out, and why

| Ruled out | Why | Label |
|---|---|---|
| NVFP4 as a format | FP4 tensor cores are Blackwell-only; the 4070 is Ada (FP8, not FP4); CPUs have neither | PUBLISHED / DECISION |
| MLX formats | Apple Silicon only | PUBLISHED |
| All dense at 16-bit | Costs bank VRAM (GPU) and per-token reads (CPU) | DERIVED / DECISION |
| Experts at Q8_0 (any of the three) | +4 to 5.5 GB each, about 0.0004 KLD per GB; shrinks the bank by about a third (gate/up); about +10 GiB pinned host memory and +55 % traffic in training | MEASURED / DERIVED |
| `token_embd`, `output`, `ssm_alpha|beta` at Q8_0 for drift | Gains under the noise threshold (`attn_k`, `attn_v`: small, but about 10 MB together, so not ruled out) | MEASURED |
| Removing blk.40 | You keep MTP; size comparisons use the without-blk.40 figure instead | DECISION (author) |

**Still open:** 3-bit (your proposal 1). It's exploratory, needs Q3_K in Pelias (O19), and gate/up experts are the published candidates for it.

---

## 9. The six PEL variant shapes (second view, corrected), with costs DERIVED exactly

The six shapes come from the author's second-view spec (written with Gemini, 2026-10-27). They were reviewed here and corrected:

- **File sizes** are computed from the real tensor shapes and the stock type rules. The stock file rebuilt this way matches the real file to the byte (data start 10,991,104), and PEL-6 matches its measured sweep file within 32 bytes.
- **Engine and trainer costs** use the rates in section 0 (DERIVED, rough). blk.40 is excluded from VRAM because it was never in the measured runs.

| Variant | What it is | File with / without blk.40 (GB) | Dense VRAM Δ (trunk) | Bank Δ after 0.16 GiB headroom | Experts per GiB | Speed effect (rough) | Trainer Δ (core × 2) | Drift | Ready? |
|---|---|---|---|---|---|---|---|---|---|
| Stock Q4_K_M | Reference | 21.713 / 21.167 | 0 | 0 | 1.000 | 0 | 0 | 0.040965 (MEASURED) | Yes |
| **PEL-6** | `ssm_out` Q8_0, rest stock | 21.839 / 21.293 | +0.117 GiB | 0 | 1.000 | 0 | +0.23 GiB | **0.031616 (MEASURED)** | Yes |
| **PEL-6+** | `ssm_out`, `attn_output`, `attn_v`, `attn_k` Q8_0 | 21.895 / 21.343 | +0.165 GiB | −0.005 GiB | 1.000 | ~0 | +0.33 GiB | **0.025958 (MEASURED, T1)** | Yes |
| **PEL-4** | Q8_0: `ssm_out`, `attn_gate`, `attn_output`, `shexp`, `attn_v`, `attn_k`; Q6_K: `attn_qkv`, `attn_q`, `eh_proj`; experts stock | 22.199 / 21.639 | +0.440 GiB | −0.280 GiB | 1.000 | about −1.3 tok/s | +0.88 GiB | **0.016450 (MEASURED, T2)** | Yes |
| **PEL-5** | All trained core at Q8_0 (`ssm_alpha`/`beta` stock); `eh_proj` Q6_K; experts stock | 22.365 / 21.802 | +0.591 GiB | −0.431 GiB | 1.000 | about −1.9 tok/s | +1.18 GiB | **0.016493 (MEASURED, T3)** | Yes |
| **PEL-1** | PEL-4's dense; gate/up IQ4_XS; down Q6_K in blocks 0–5, 34–39, Q5_K elsewhere; blk.40 experts Q2_K | 21.655 / 21.354 | +0.440 GiB | −0.280 GiB | 1.015 | about −0.8 tok/s | +0.88 GiB | UNMEASURED | Needs O19 (IQ4_XS, Q5_K, Q2_K) |
| **PEL-2** | Q8_0: `ssm_out`, `attn_output`, `attn_v`, `attn_k`; Q6_K: `shexp`; gate Q3_K in blocks 8–31; down Q6_K/Q5_K; blk.40 experts Q3_K | 21.501 / 21.125 | +0.133 GiB | 0 | 1.010 | about +0.3 tok/s | +0.27 GiB | UNMEASURED | Needs O19 (Q3_K, Q5_K) |
| **PEL-3** | Hot/cold split inside each expert tensor | not computed | — | — | — | — | — | — | Needs O19 and a Pelias-only container; llama.cpp couldn't read it |
| (b) Unsloth's layout, for comparison | All dense Q8_0 incl. `token_embd`; down Q5_K/Q6_K | 22.660 / 22.135 | +0.841 GiB | −0.681 GiB | 0.997 | about −3.2 tok/s | +1.44 GiB | 0.014174 (MEASURED) | Yes |

**Corrections to the second-view spec** (logged in the RUNBOOK):
- **PEL-1's size** is 21.66 / 21.35 GB, not 21.84 / 21.50.
- **PEL-2's expert gain** is +1.0 % experts per GiB, not +5.2 %.
- **`attn_qkv` in stock** is 14 layers Q6_K and 16 Q4_K (30 layers), not 20 + 20.
- **`ssm_alpha`/`ssm_beta` in stock** are Q4_K, not F32.
- **"31.8 tok/s"** is experts-on-CPU, not all-CPU (all-CPU is 16.28).
- **PEL-6+** uses +0.165 GiB against the 0.16 headroom: right at the edge.
- **The `eh_proj` regex** must match `blk.40.nextn.eh_proj.weight`.
- **Tensor counts include blk.40:** `attn_v`/`attn_k` 22, `shexp` 123.
- **Trainer costs double the Q6_K part too.**
- **blk.40 at Q2_K** has zero output drift by construction, not "verified"; draft acceptance is unmeasured.
- **PEL-3's "64 hot experts" and "−22 % cold reads"** aren't derived. The 7.5 GiB bank holds about 104 experts per layer, and the byte maths gives about −15 %.

**Reading the table:**
- **PEL-6 is the measured anchor.** It gives −22.8 % drift with no speed cost.
- **PEL-4 and PEL-5 are the two "today" builds that could approach (b).** They cost less speed than (b) because `token_embd` stays Q4_K and the Q6_K choices are cheaper.
- **PEL-1 is the most likely to beat Unsloth on size and drift together,** but it waits on O19.
- **The importance matrix from your own text** adds at most about 0.0006 on top of a good layout (MEASURED on Unsloth's matrix).

### 9.1 Tests that would firm up the choice (runnable today with llama-quantize; each about 8.5 min)

Each test prints every type change against the clean run 0 before measuring, so a shared-counter side effect would show. Predictions are written before the runs (ESTIMATE):

| Test | Build | Exact size expected (DERIVED) | Prediction | What it decides |
|---|---|---|---|---|
| T1 | PEL-6+ | 21,895,071,744 (±32) | 0.028–0.031 | Whether the cheap groups add on top of `ssm_out` (are the singles additive?) |
| T2 | PEL-4 dense layout | 22,198,659,072 (±32) | 0.020–0.026 | How close a "today" build gets with no expert changes and no `token_embd` |
| T3 | PEL-5 (all core Q8_0) | 22,365,251,584 (±32) | 0.016–0.022 | The full-core cost and gain; against (b) it isolates the expert layout (stock down vs Unsloth's Q5_K down) |
| T4 | (b) with `token_embd` back to Q4_K | 22,405,965,824 | 0.0142–0.0148 | First reverse-sweep step: is `token_embd` Q8_0 worth 0.254 GB inside a good layout? |

### 9.2 Results of T1–T4 (2026-10-27, MEASURED with llama.cpp; every type check clean)

| Test | Mean KLD | Same top p | File (bytes) | Prediction | Finding |
|---|---|---|---|---|---|
| T1 PEL-6+ | **0.025958** ± 0.000529 | 92.945 % | 21,895,071,776 | 0.028–0.031: wrong, better | The four groups' single gains summed to 0.0129; together they give 0.0150. **Gains compound** across the residual stream |
| T2 PEL-4 dense | **0.016450** ± 0.000443 | 94.384 % | 22,198,659,104 | 0.020–0.026: wrong, better | Within 0.0023 of (b) for 0.46 GB less file and about half the bank cost |
| T3 PEL-5 | **0.016493** ± 0.000572 | 94.443 % | 22,365,251,616 | 0.016–0.022: right | Same as PEL-4 within noise: **`attn_qkv`/`attn_q` at Q8_0 buy nothing over Q6_K** ("6 ≈ 8" holds there). PEL-5 is dominated by PEL-4 |
| T4 (b) without `token_embd` Q8_0 | **0.015176** ± 0.000429 | 94.557 % | 22,405,965,856 | 0.0142–0.0148: wrong, worse | `token_embd` at Q8_0 is worth **0.0010** inside a good layout (twice its single-run value) |

**DERIVED from T3 against T4:**
- The two differ only in the expert layout (Unsloth's `ffn_down_exps`: Q5_K, with Q6_K in blocks 34, 38, 39, against stock's Q6_K/Q4_K split) and `ssm_alpha`/`beta` at F32.
- That difference is worth **0.0013 for only +40.7 MB**: spreading the down bits evenly beats stock's layer bumps at almost the same size.
- Caveat: Q5_K needs O19 in Pelias's own quantizer; llama.cpp can build it today.

**`token_embd` placement (DERIVED, a DECISION to make):** the engine reads one embedding row per token. So `token_embd` can live in host RAM, and then its Q8_0 costs file size only, not bank VRAM.

**Next tests, built from these results:**

| Test | Build | Exact size (DERIVED) | Without blk.40 | Prediction (ESTIMATE) |
|---|---|---|---|---|
| T7 | PEL-4 dense + Unsloth expert down layout + `ssm_alpha`/`beta` F32 | 22,241,536,000 | 21.718 GB | 0.0148–0.0156 |
| T8 | T7 + `token_embd` Q8_0 | 22,495,815,680 | 21.972 GB | 0.0138–0.0146 |

If T8 lands near 0.0142, adding an importance matrix (worth about 0.0006 on Unsloth's layout) would put it at about Unsloth's 0.0136 with a file 0.16 GB smaller (without blk.40). That's the first concrete route to "beat Unsloth".

### 9.3 Results of T5–T8 (2026-10-27, MEASURED with llama.cpp; every type check clean)

| Test | Mean KLD | Same top p | File (bytes) | Without blk.40 | Prediction | Finding |
|---|---|---|---|---|---|---|
| T5 PEL3-a (3-bit gate/up experts, protected dense) | **0.045287** ± 0.000832 | 90.753 % | 17,887,414,304 | 17.48 GB | 0.045–0.065: right | Only 0.0043 worse than stock 4-bit at **82 % of its size**, with +24.7 % experts per GiB of bank |
| T6 `ssm_out` alone at Q6_K | **0.032472** ± 0.000633 | 91.984 % | 21,778,343,968 | 21.23 GB | 0.0325–0.0345: right | **90.8 % of the Q8_0 gain for 52 % of the bytes** (0.131 KLD per GB against 0.074 at Q8_0) |
| T7 PEL-4 dense + Unsloth expert layout + `ssm_alpha`/`beta` F32 | **0.015454** ± 0.000390 | 94.604 % | 22,241,536,032 | 21.72 GB | 0.0148–0.0156: right | |
| T8 = T7 + `token_embd` Q8_0 | **0.014834** ± 0.000456 | 94.635 % | 22,495,815,712 | 21.97 GB | 0.0138–0.0146: wrong, slightly worse | `token_embd` Q8_0 worth 0.0006 here |

**Noise calibration (DERIVED; this changes how close calls are read):**
- (b) and T8 differ only in `attn_qkv`/`attn_q` (Q8_0 against Q6_K) and `eh_proj`, which the test never uses. Yet they differ by 0.00066.
- T2 and T3 differ by the same swap and only 0.00004 apart.
- So **comparisons between runs on this one text are good to about ±0.0006.** Contenders closer than that are a tie until a second evaluation text.

**Standing against the bars (MEASURED with llama.cpp; size without blk.40; engine and trainer costs DERIVED):**

| File | Mean KLD | Size | Dense VRAM | Bank effect | Trainer |
|---|---|---|---|---|---|
| Unsloth UD-Q4_K_M (with imatrix) | 0.013605 | 22.13 GB | +0.84 GiB | about −3.2 tok/s | +1.44 GiB |
| (b) Unsloth's types, no imatrix | 0.014174 | 22.13 GB | +0.84 GiB | about −3.2 tok/s | +1.44 GiB |
| **T8** | **0.014834** | **21.97 GB** | +0.45 GiB (`token_embd` in RAM) | **about −1.3 tok/s** | +0.91 GiB |
| T7 | 0.015454 | 21.72 GB | +0.45 GiB | about −1.3 tok/s | +0.91 GiB |
| PEL-6+ (T1) | 0.025958 | 21.34 GB | +0.165 GiB | ~0 | +0.33 GiB |
| PEL-6 | 0.031616 | 21.29 GB | +0.117 GiB | 0 | +0.23 GiB |
| PEL3-a (T5) | 0.045287 | 17.48 GB | +0.165 GiB | **bank +24.7 % experts** | +0.33 GiB |
| Stock Q4_K_M | 0.040965 | 21.17 GB | 0 | 0 | 0 |

**Reading:**
- T8 sits 0.0007 behind (b), which is right at the noise limit. It's 0.16 GB smaller and costs the bank about 2 tok/s less (DERIVED).
- With an importance matrix (+about 0.0006, MEASURED on Unsloth's) it would land at about (b)'s drift.
- Beating Unsloth outright needs a new lever, not more of the same palette:
  - a better expert format (IQ4_XS or Q5_K with imatrix);
  - Q6_K in place of Q8_0 where T6 shows it keeps 90 % of the gain;
  - Mode Q training.

**Bench set (proposal):**
1. **PEL-6** as the measured anchor.
2. **T8** (or T7 if `token_embd` stays at Q4_K) as the best 4-bit mix.
3. **PEL3-a** as the compact 3-bit.

Later, not today:
- **A second evaluation text:** a new bf16 base on a different domain (code or tool calls). The first base took 57 min.
- **The per-layer `ffn_down_exps` sweep:** where Q6_K down pays.
- **O19 builds:** PEL-1 and PEL-2.

---

## 10. Not yet measured, and how each gets measured

| Unknown | Path |
|---|---|
| ~~Whether the sweep overrides changed other tensors' types~~ | Done (3.3): only `attn_v`/`attn_qkv` were affected; both rerun clean |
| Drift differences under about 0.0005 | A second evaluation text (another 51,200 tokens, different domain) with the same base procedure |
| Combined-group drift | Tests T1–T4 (9.1) |
| Trainer working set (2.50 GiB) and the `W_eff` copy size | P-M2 (trainer step 8) |
| Engine VRAM with blk.40 and with PEL dense types | The Pelias engine's own run (steps 5 → 7 → 11) |
| Real bank/speed cost of dense bytes | Engine decode at two bank sizes with a PEL file |
| How much fine-tune survives per core type | P-E8 / K2 (Mode Q against the merge erase) |
| Q8_0 on the int8 tensor path | A Batch-C-style probe once the kernel exists |
| MTP acceptance with low-bit blk.40 experts | Engine MTP run |
| Hugging Face match of the tokenizer (step 4, P-D1) | `tools/pd1_compare_hf.py` and `tools/hf_unicode_probe.py`, after the sweep |

---

## 11. Sources

- Your machine's runs: the RUNBOOK, TALLY.csv, and the sweep log of 2026-10-27 (02:16–04:12).
- Pelias docs: `PELIAS_SYSTEM_MASTER.md` sections 4.1–4.9; `PELIAS_SYSTEM_MASTERWORK.md` sections 6.2, 17.5; `pelias_ground_up_master_specification.md` section 6.2.
- Published:
  - [Unsloth Qwen3.5 GGUF benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
  - [Unsloth Qwen3.6 docs](https://unsloth.ai/docs/models/qwen3.6)
  - [bartowski per-tensor layout maps](https://huggingface.co/blog/bartowski/per-tensor-layout-maps-for-gguf-quantization)
  - [ubergarm Qwen3.5-35B-A3B](https://huggingface.co/ubergarm/Qwen3.5-35B-A3B-GGUF)
  - [ik_llama Qwen3 experiments](https://github.com/ikawrakow/ik_llama.cpp/discussions/359)
  - [mmangkad NVFP4 config](https://huggingface.co/mmangkad/Qwen3.6-35B-A3B-NVFP4/blob/main/hf_quant_config.json)
  - [RedHatAI NVFP4](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4)
  - [Infatoshi NVFP4-FP8](https://huggingface.co/Infatoshi/Qwen3.6-35B-A3B-NVFP4-FP8)
