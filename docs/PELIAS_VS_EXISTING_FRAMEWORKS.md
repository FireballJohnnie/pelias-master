# Pelias's bit-level line compared with existing frameworks (corrected)

Author's request, 2026-10-28 15:48: a corrected version of an outside summary, every claim labelled.

- **MEASURED (whole model):** a paired 100-chunk KL run on the PC (25,500 scored tokens), engine format (8-bit activations, eight scales per 256), verified independently from the per-token dumps.
- **MEASURED (sandbox):** real weights from the model (blk.20 slices), or unit tests; one tensor type unless named.
- **PAPER:** derived on paper or by a read-only review; not yet measured.
- **PLAN:** designed, not built or not run.

The bar: Unsloth's MTP UD-Q4_K_M on llama.cpp's GPU path, KL 0.012864. Statements about other projects are limited to what their formats do; we have not tested the other projects themselves.

## 1. Between 6 and 8 bits

| | Fact | Label |
|---|---|---|
| llama.cpp | K-quants go up to Q6_K (6.5625 bpw); the next step is Q8_0 (8.5 bpw), whose fp16 scale per 32 costs 0.5 bpw. Q8_0 has no importance-weighted path. | Format facts |
| Unsloth | Its dynamic files keep sensitive dense tensors at Q8_0, a whole tensor at a time. | Format facts (from their file's types) |
| **Q8_Z** (Pelias) | The author's zip scales (fp16 per 256 + 6-bit level per 32) with int8 values: 8.25 bpw, int8-exact. Saves 0.25 of Q8_0's 0.5 scale bits. Whole model (urm_Z): no difference from Q8_0, **-61.0 MB**, KL 0.012402. | MEASURED (whole model) |
| **Q7_Z** (pure 7-bit) | 7.25 bpw. Whole model (urm_Q7): **+0.001122 vs urm_Z, resolved worse**; +0.000660 vs Unsloth, resolved worse. Pure 7-bit fails. | MEASURED (whole model) |
| **Q7_ZT** (7 + gated 8th bit) | 7-bit base everywhere; the 8th bit kept on 2 of every 8 32-blocks per 256, the 2 where it removes the most error. 241 bytes per 256 = 7.53 bpw. The base is packed odd-width (8 values per 7 bytes); every value rebuilds to an exact int8, so the int8 kernel computes on plain 8-bit values. | MEASURED (format, unit tests) |
| | Whole model (urm_ZT2): **KL 0.012536; vs Unsloth -0.000328 (lower, not resolved, t -1.25) at -236.4 MB; vs urm_Z (Q8_Z) +0.000134, no difference; vs pure 7-bit -0.000988, resolved better.** Passes gate 1 of the 7pH ladder (accuracy equal to the 8-bit format at fewer bytes); speed not yet measured. | MEASURED (whole model) |
| Gate curve | H on 10 % of blocks removes about a third of the 7-bit gap; 25 % about 60 %; a fixed 2-of-8 quota equals a free ranking. At 25 %, the block error is still about 2x Q8_Z's (0.000060 vs 0.000029); the whole model shows no difference. | MEASURED (sandbox, attn_qkv) |

## 2. The importance matrix

| | Fact | Label |
|---|---|---|
| Inside a rotated 32-block | Importance is nearly flat: weighting the level search changes nothing for Q8_Z or Q7_Z on attn_qkv. | MEASURED (sandbox, attn_qkv) |
| Our own over-fit | Our first Q7_ZT weighted its level search by the importance matrix. On tensors with lopsided importance (ssm_out: up to 121x the mean) it over-fit the calibration text: chunks 23-24 at 0.0424 / 0.0397 against pure 7-bit's 0.0146 / 0.0205 (urm_ZT, KL 0.012870). | MEASURED (whole model) |
| The fix | Levels by plain error; the matrix only ranks which 2 of 8 blocks get the 8th bit, tempered to its square root. Chunks 23-24 back to 0.0138 / 0.0291 (urm_ZT2). | MEASURED (whole model) |
| Ranking by importance | Matters: 0.000061 weighted vs 0.000074 plain (25 % gated). | MEASURED (sandbox, attn_qkv) |
| Not shown | That llama.cpp's or Unsloth's importance weighting over-fits. Their K-quant search mixes importance with other terms; we have not tested it. | — |

## 3. Split bit planes and idle units

| | Fact | Label |
|---|---|---|
| Split storage | Q7_ZT's 241-byte block splits into a 233-byte base and 8 bytes of H bits (3 % of the bytes) and joins back exactly, so the H bits can live elsewhere (RAM vs VRAM, CPU vs GPU). | MEASURED (unit test) |
| 1-bit decoder (T-decode) | The 8th bit added by AND + popcount against the activation's 8 bit planes gives the same integers as the int8 route on every 32-block. | MEASURED (unit tests, gcc / clang / ASan) |
| Activation bit planes | AVX2 (pmovmskb, one instruction per plane) equals the portable version. GPU warp-ballot planes: not built. | MEASURED (CPU) / PLAN (GPU) |
| Idle integer units | Zen 3 runs POPCNT at 4 per cycle (published table). That the integer units are idle while the vector units work, and that the decoder runs there at no cost, is not yet measured: no speed kernel exists. | Published rate / PAPER |
| GPU route | CUDA-core AND + popc at decode (about 0.7 us, Fable's estimate); the 1-bit mma for batched MTP verification (rate on the 4070 unmeasured). | PAPER |
| 0.5 fill | When an 8th bit is not read, filling it with 0.5 through the activation's block sum costs 43 % less than any integer fill (0.000116 vs 0.000202-0.000206), and stays exact on int8. (Engines that never drop the bit at run time do not face this choice.) | MEASURED (sandbox, attn_qkv) |
| Odd grid | Blocks stored without the 8th bit use the odd grid (2b - 127) x s: 0.000105 vs 0.000116. | MEASURED (sandbox, attn_qkv) |
| Prior work | Bit-serial and bit-plane decoding exist elsewhere (T-MAC, Any-Precision LLM); the new parts here are the gate per 32-block, the 0.5 fill and the split across memory spaces. | Literature |

## 4. Rotation on a MoE model

| | Fact | Label |
|---|---|---|
| What is rotated | 318 tensors (Walsh-Hadamard per 32): every dense multiply (attention, SSM, shared expert, output, the MTP block's dense tensors) plus ssm_alpha / ssm_beta (F32, they read the same rotated input). Not rotated: the 123 routed expert tensors, token_embd (a lookup), the routers, the norms. | MEASURED (file) |
| Dense activations | Rotating the dense inputs removes the 8-bit activation rounding cost: -0.001186 vs the same rounding unrotated (t -3.23); vs no rounding -0.000211, no difference. | MEASURED (whole model, R1) |
| Dense weights | Rotated from bf16 then quantized: -0.003454 vs the unrotated twin, -8.9 % drift (t -6.06). | MEASURED (whole model, R2) |
| Experts | Rotating the routed experts made the file worse: +0.003979 vs dense-only rotation (R2b). Why is not measured. | MEASURED (whole model) |
| Expert activation cost | Not zero: at one scale per 256 the experts cost +0.000379. | MEASURED (whole model, Block M3 e) |
| Other projects | QuaRot, SpinQuant and QuIP# rotate the linear layers of the (mostly dense) models they report on; we have not tested them on this MoE model. | Literature |

## 5. Reading bits by need

| | Fact | Label |
|---|---|---|
| Error estimate | The expected-size estimate (s^2 x sum x^2 per block) matches the real error of a dropped plane at 1.00x; the exact worst-case bound is 17-33x too loose. | MEASURED (sandbox, real experts) |
| Cold experts | 6.3 unpredicted cold picks per token; 52 % of them have router weight under 0.08, 87 % rank 5-8. | MEASURED (routing record) |
| Live gate on dense tensors | `--h-gate live:F`: each token reads the 8th bit only on the fraction F of blocks with the largest expected loss, the rest filled by 0.5. Built and unit-tested; the first whole-model run (F = 0.25 on urm_Z) started 15:50. | MEASURED (unit test) / running |
| MTP block | Q3 base + H, the first draft reads H, deeper drafts the base only (draft 1's acceptance multiplies every later one). | PLAN (needs GP3-08) |

## 6. Training

| | Fact | Label |
|---|---|---|
| Inference int8 path | The fused int8 GEMM (v10) is 1.39-1.84x faster than the bf16 path including norm and conversions, all 20 shapes; with rotation, 8-bit activations cost nothing measurable at one scale per 32. | MEASURED (forward pass, inference) |
| Training on it | Training matmuls (forward and backward) on the int8 path, training on the served GGUF bytes, and the Mode Q export: trainer design. Backward passes in int8 are untested. | PLAN |

## Headline, as measured today

Q7_ZT (7.53 bpw dense, the 8th bit gated per 32-block, rotated dense tensors) is **236 MB smaller than Unsloth's MTP file and lower in KL by 0.000328** (not resolved at this test's size), with **no measurable difference from the 8.25-bit Q8_Z file**. Speed is the open gate.
