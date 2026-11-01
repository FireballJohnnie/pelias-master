// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
#pragma once
// Fast CPU row dot products straight from quantized storage (game plan GP3-10, CPU; author 2026-10-30 10:57 "build first").
//
// out[t] = sum_i w_i x_t[i] for one weight row and up to four activation rows x_t (fp32, already rounded as the forward
// pass rounds them). w is the row exactly as quant::dequantize_row gives it, plus its residual sign planes when given
// (dense or block-selective, as quant::plane_add_row / plane_add_row_sparse add them). Each 256-value segment is dequantized
// into a small buffer with the reference's own float expressions (so every w_i is bit-identical to the reference's), the
// planes are added plane by plane in the reference's order, and the products are summed with fused multiply-adds into 32
// lanes (lane = i mod 32), reduced by a fixed tree. The products match the reference; only the order of the sum differs,
// so results agree with the reference forward within the rung-1 tolerance, not bit for bit. The AVX2 path and the
// portable path use the same lanes, the same fused multiply-adds and the same tree, so they agree bit for bit.
//
// AVX2 dequantizers for all 30 readable types (dot_has_fast_path). Rows whose length is not a multiple of 256 are
// dequantized whole.
#include <cstdint>
#include <vector>

#include "quant/block_types.h"

namespace pelias::kern {

// A row's residual sign planes (quant/planes.h layout). data null: none. mask null: dense (34 k_stored bytes per 256
// values); else block-selective: bit b of mask[b / 8] set = block b has planes, stored back to back from data on.
struct RowPlanes {
  const std::uint8_t* data = nullptr;
  const std::uint8_t* mask = nullptr;
  int k_stored = 0;
  int k_use = 0;
};

enum class Isa { portable, avx2 };

Isa best_isa();                 // what this CPU supports (AVX2 and FMA, or portable)
Isa active_isa();               // what dot_row uses now (best_isa unless forced)
void force_isa(Isa isa);        // tests and benches; forcing avx2 on a CPU without it falls back to portable

bool dot_supported(quant::Type t);       // every type quant::dequantize_row reads
bool dot_has_fast_path(quant::Type t);   // an AVX2 dequantizer of its own

// k values per row; nt activation rows (1..4); planes may be null.
void dot_row(quant::Type t, const std::uint8_t* row, std::int64_t k, const RowPlanes* planes, const float* const* xs, int nt,
             float* out);

// n tokens at once (batches: prefill, the MTP verify step, kld): out[t * ldo] is exactly what dot_row gives token t (bit for
// bit), the row decoded once instead of once per four tokens.
void dot_row_n(quant::Type t, const std::uint8_t* row, std::int64_t k, const RowPlanes* planes, const float* const* xs, std::int64_t n,
               float* out, std::int64_t ldo);

// A block of nr rows of one type (row r at rows[r], its planes at planes[r]; planes itself may be null) against n tokens:
// out[t * ldo + r] exactly what dot_row gives (row r, token t). Each row is decoded once and each group of four tokens is
// run across the whole block, so a batch reuses both the weights and the inputs (prefill, kld).
void dot_rows_n(quant::Type t, const std::uint8_t* const* rows, const RowPlanes* const* planes, int nr, std::int64_t k, const float* const* xs,
                std::int64_t n, float* out, std::int64_t ldo);

// ---- the int8-activation path (GP3-10 item 5; the engine's activation format, ForwardOptions gpu256x8)
//
// Activations rounded per 32 values to an fp32 scale s_b and integers q in -127..127 (round half to even, as
// model::act_round_q8_256x8). For every weight type whose values are integers times per-16 scales (minus per-16 mins)
// the products inside a 32-block are integers, summed exactly (AVX2: maddubs + madd, 8 lanes of 4 values each), then
// scaled: y = sum_b s_b (sum_j sw_j isum_j - sum_h mw_h A_h) with A_h the activation sums per 16. Planes add
// s_p s_b (2 sum_{sign set} q - A_b) per plane and block, also in integers. Same quantities as the float path's
// products; another order of the sum (the rung-1 tolerance against fp64 of the same rounded inputs). The AVX2 and portable
// paths agree bit for bit. Types without an integer form (Q4_TG, Q4_TG2, IQ1_M, F16, BF16, F32) take the float path on the
// rounded values.
struct ActQ8x32 {
  std::int64_t k = 0;
  std::vector<std::int8_t> q;      // k
  std::vector<float> s;            // k / 32
  std::vector<std::int32_t> sum16; // k / 16
  std::vector<float> x;            // k: s_b * q, the rounded values (the float path's input)
};
// From unrounded values, with act_round_q8_256x8's rule (scale amax / 127, 1 when amax is 0; q = clamp(nearbyint(x / s))).
void act_q8x32_quantize(const float* x, std::int64_t k, ActQ8x32& a);
// v45: act_q8x32_quantize takes an AVX2 form when the CPU has it (bit for bit with this one); the scalar form, for checks.
void act_q8x32_quantize_ref(const float* x, std::int64_t k, ActQ8x32& a);
// From values already rounded that way (the forward pass rounds before the product): the scale is recovered as max |x| /
// 127 and q = nearbyint(x / s) (exactly the integers; the scale to within one rounding).
void act_q8x32_from_rounded(const float* xr, std::int64_t k, ActQ8x32& a);
bool dot_q8_has_int_path(quant::Type t);
// v40: two rows against one float vector (decode's down projection, one token): out[0], out[1] exactly what dot_row gives
// each row (nt = 1, no planes). Q4_TG / Q4_TG2 on AVX2 load the grids once and share each x load; other types call dot_row twice.
void dot_rows2_f32(quant::Type t, const std::uint8_t* r0, const std::uint8_t* r1, std::int64_t k, const float* x, float* out);
// k a multiple of 32; nt rows (1..4) of acts, each acts[i]->k == k.
void dot_row_q8(quant::Type t, const std::uint8_t* row, std::int64_t k, const RowPlanes* planes, const ActQ8x32* const* acts, int nt,
                float* out);

// n tokens at once (acts[0..n)): out[t * ldo] exactly what dot_row_q8 gives token t, the row decoded once.
void dot_row_q8_n(quant::Type t, const std::uint8_t* row, std::int64_t k, const RowPlanes* planes, const ActQ8x32* acts, std::int64_t n,
                  float* out, std::int64_t ldo);

// The int8 form of dot_rows_n: out[t * ldo + r] exactly what dot_row_q8 gives (row r, token t).
void dot_rows_q8_n(quant::Type t, const std::uint8_t* const* rows, const RowPlanes* const* planes, int nr, std::int64_t k, const ActQ8x32* acts,
                   std::int64_t n, float* out, std::int64_t ldo);

}  // namespace pelias::kern
