// provenance: re-expressed (not copied) from ggml ggml/src/ggml-quants.c and ggml/src/ggml-impl.h at llama.cpp commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a (MIT; THIRD_PARTY_NOTICES.md), read as a specification so the bytes match
//   (masterwork 3.2 rule 6, 5.3, gate P-E6)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// quant_cpu.h - the reference quantizers and dequantizers on the CPU: the same arithmetic, in the same order, as the
// reference library's "ref" routines, so a tensor quantized here is byte-identical to one quantized there (P-E6).
// This module is compiled with floating-point contraction off (no fused multiply-add) for the same reason: the
// reference compiles these routines in its base library with plain float arithmetic. Speed comes later from separate
// kernels, checked against these.
//
// Types in this step (5.3 build order): F32, F16, BF16, Q8_0, Q4_K, Q6_K for weights; Q8_1 and Q8_K for activations.
// Read only (dequantize, no quantizer yet; Milestone 1 / O19): Q2_K, Q3_K, Q5_K, IQ4_XS; Q4_0, Q4_1, Q5_0, Q5_1,
// IQ4_NL, MXFP4, TQ1_0, TQ2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M (quant/dequant_more.cpp).
#pragma once

#include <cstddef>
#include <cstdint>

#include "pelias/status.h"
#include "quant/block_types.h"

namespace pelias::quant {

// ---- scalar conversions (IEEE binary16 and bfloat16, round to nearest even; NaN stays NaN, quiet)
float fp16_to_fp32(std::uint16_t h);
std::uint16_t fp32_to_fp16(float f);
float bf16_to_fp32(std::uint16_t h);
std::uint16_t fp32_to_bf16(float f);

// ---- the reference rounding used by the K-quants: nearest integer, ties to even (valid for |x| <= 4194303)
int nearest_int(float x);

// ---- whole rows. `n` is the number of elements; it must be a whole number of blocks of the type (block_types.h).
// `out` receives row_bytes(type, n) bytes; `in` holds that many. No alignment is required of either pointer.
bool can_quantize(Type t);    // this step's quantizers
bool can_dequantize(Type t);  // this step's dequantizers (the activation types Q8_1 and Q8_K are not dequantized)
Status quantize_row(Type t, const float* x, std::uint8_t* out, std::int64_t n);
// The orthonormal Walsh-Hadamard transform of every 32-value block of x[0..n) in place (n a multiple of 32; a tail
// shorter than 32 is left as it is). Symmetric and its own inverse. A weight row rotated with it and an input rotated
// with it give the same product: (W H)(H x) = W x.
void hadamard32(float* x, std::int64_t n);
Status dequantize_row(Type t, const std::uint8_t* in, float* y, std::int64_t n);

// ---- with an importance matrix (quant_weighted.cpp): nrow rows of n_per_row values, as the reference's
// ggml_quantize_chunk does. `imatrix` holds n_per_row weights (one per column; for a stack of experts the caller passes
// that expert's slice) or is null. The types that use it: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS; every other
// type quantizes as without it.
bool uses_importance(Type t);
Status quantize_rows(Type t, const float* x, std::uint8_t* out, std::int64_t nrow, std::int64_t n_per_row, const float* imatrix);

// ---- per type, for tests and for callers that already checked the type (n a multiple of the block)
void quantize_q8_0(const float* x, std::uint8_t* out, std::int64_t n);
// Q8_Z (Pelias-only, 256 per block, 264 bytes = 8.25 bits per weight): fp16 d, then eight 6-bit levels lv (1..63) packed
// little-endian (level j in bits 6j..6j+5 of the 6 bytes), then 256 int8 values q (-127..127). Value = s_j * q with
// s_j = (d * lv_j) / 63 in fp32, exactly as written (the kernels must compute the same). The quantizer searches each
// 32-block's level for the least squared error; level 0 is never written (free for a flag).
void quantize_q8_z(const float* x, std::uint8_t* out, std::int64_t n);
void dequantize_q8_z(const std::uint8_t* in, float* y, std::int64_t n);
float q8_z_scale(float d, int lv);
// Q7_Z (Pelias-only, 256 per block, 232 bytes = 7.25 bits per weight): Q8_Z's header (fp16 d = top / 63, eight 6-bit
// levels), then 32 groups of 7 bytes, each holding eight 7-bit values u = q + 64 (q in -63..63) little-endian, value k in
// bits 7k..7k+6. Value = s_j * q with s_j = (d * lv_j) / 63 in fp32 (j = 32-block). Same level search as Q8_Z.
void quantize_q7_z(const float* x, std::uint8_t* out, std::int64_t n);
void dequantize_q7_z(const std::uint8_t* in, float* y, std::int64_t n);
float q7_z_scale(float d, int lv);
// Q7_ZT (Pelias-only, 256 per block, 241 bytes = 7.53 bits per weight): fp16 d = top / 127, eight 6-bit levels (Q8_Z's
// scales, s_j = (d * lv_j) / 63), one byte gate mask with exactly two bits set, 224 bytes of 7-bit base values b (eight per
// 7 bytes, as Q7_Z), then 2 x 4 bytes of H bits for the gated 32-blocks in block order (bit k = value k). Gated 32-block:
// q = 2b + h - 127; others: q = 2b - 127 (odd grid). Value = s_j * q. The gates go to the two 32-blocks where the 8th bit
// removes the most squared error, weighted by the square root of the importance when one is given (levels by plain error).
void q7_zt_block(const float* x, std::uint8_t* out, const float* qw);
void quantize_q7_zt(const float* x, std::uint8_t* out, std::int64_t n);
void dequantize_q7_zt(const std::uint8_t* in, float* y, std::int64_t n);
void quantize_q8_1(const float* x, std::uint8_t* out, std::int64_t n);
void quantize_q4_k(const float* x, std::uint8_t* out, std::int64_t n);
void quantize_q6_k(const float* x, std::uint8_t* out, std::int64_t n);
void quantize_q8_k(const float* x, std::uint8_t* out, std::int64_t n);
void quantize_q2_k(const float* x, std::uint8_t* out, std::int64_t n);  // quant_weighted.cpp (the reference's "ref" routines)
void quantize_q3_k(const float* x, std::uint8_t* out, std::int64_t n);
void quantize_q5_k(const float* x, std::uint8_t* out, std::int64_t n);
void quantize_iq4_nl(const float* x, std::uint8_t* out, std::int64_t n);
void quantize_iq4_xs(const float* x, std::uint8_t* out, std::int64_t n);
void dequantize_q8_0(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_q4_k(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_q6_k(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_q2_k(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_q3_k(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_q5_k(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_iq4_xs(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_q4_0(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_q4_1(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_q5_0(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_q5_1(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_iq4_nl(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_mxfp4(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_tq1_0(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_tq2_0(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_iq2_xxs(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_iq2_xs(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_iq2_s(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_iq3_xxs(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_iq3_s(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_iq1_s(const std::uint8_t* in, float* y, std::int64_t n);
void dequantize_iq1_m(const std::uint8_t* in, float* y, std::int64_t n);

}  // namespace pelias::quant
