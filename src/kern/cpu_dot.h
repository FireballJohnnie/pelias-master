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
// AVX2 dequantizers: Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS, Q4_TG, Q4_TG2. Any other readable type goes through
// quant::dequantize_row for its segment and the same summation. Rows whose length is not a multiple of 256 are
// dequantized whole.
#include <cstdint>

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

}  // namespace pelias::kern
