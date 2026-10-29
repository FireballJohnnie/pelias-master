// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
#pragma once
// Residual sign planes (author 2026-10-29 18:26-18:52, game plan "near-tie accuracy: residual planes on the 1-bit path").
// A weight row as some file stores it (any type: ours, Unsloth's, stock) leaves an error r = w - q against the original
// weights w. K sign planes approximate it: per 256 values, plane k holds one fp16 scale s_k and 256 sign bits b_k (+1/-1),
// and the row read with planes is q + sum_k s_k b_k. Each plane cuts the importance-weighted error about 3x on real
// tensors (MEASURED 2026-10-29, blk.20 attn_qkv / ffn_gate_exps / ffn_down_exps, tools data in D:\pelias-tests\gp\c3c6_pretoken).
// Fit: plane by plane (scale = importance-weighted mean |r|, which is the optimum for one plane), then for K >= 2 the
// planes refitted together by alternating (a) the K scales by importance-weighted least squares given the signs and
// (b) each value's best of the 2^K sign patterns given the scales (2 planes -5 to -10 %, 3 planes -21 to -37 % error,
// MEASURED). A plane not read contributes nothing (its expected value is 0), so reading fewer planes than stored is valid.
// Layout per 256-value block: K fp16 scales (2K bytes), then K x 32 bytes of signs (plane-major; value i of the block in
// byte i / 8, bit i % 8; set = +1). Row bytes = (n / 256) x 34 K.
#include <cstdint>

namespace pelias::quant {

constexpr std::int64_t kPlaneBlock = 256;
constexpr int kPlaneMaxK = 4;
constexpr std::int64_t plane_row_bytes(std::int64_t n, int k) { return (n / kPlaneBlock) * 34 * k; }

// Fits k planes (1..kPlaneMaxK) to the residual r[0..n) (n a multiple of 256) with per-value importance imp (null: all 1)
// and writes plane_row_bytes(n, k) bytes. refit_rounds: alternating rounds for k >= 2 (a round is kept only when it lowers
// the weighted error). Returns the weighted squared error left, sum imp (r - planes)^2.
double plane_fit_row(const float* r, const float* imp, std::int64_t n, int k, int refit_rounds, std::uint8_t* out);

// y[0..n) += the first k_use of the k_stored planes in `in` (one row as written by plane_fit_row).
void plane_add_row(const std::uint8_t* in, std::int64_t n, int k_stored, int k_use, float* y);

}  // namespace pelias::quant
