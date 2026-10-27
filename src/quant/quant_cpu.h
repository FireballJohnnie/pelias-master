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
Status dequantize_row(Type t, const std::uint8_t* in, float* y, std::int64_t n);

// ---- per type, for tests and for callers that already checked the type (n a multiple of the block)
void quantize_q8_0(const float* x, std::uint8_t* out, std::int64_t n);
void quantize_q8_1(const float* x, std::uint8_t* out, std::int64_t n);
void quantize_q4_k(const float* x, std::uint8_t* out, std::int64_t n);
void quantize_q6_k(const float* x, std::uint8_t* out, std::int64_t n);
void quantize_q8_k(const float* x, std::uint8_t* out, std::int64_t n);
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
