// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// The GPU decoders (kern/gpu_decode.h, GP3-10 GPU) compiled for the host: every supported type, every sub-block of a
// quantized row, gives exactly the floats of quant::dequantize_row (bit for bit, NaN-free). The GPU build compiles the same
// functions with mul / add mapped to __fmul_rn / __fadd_rn, which the device never fuses, so this test pins the device
// values too (the rung-1 GPU test checks them on the card).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "kern/gpu_decode.h"
#include "pelias_test.h"
#include "quant/block_types.h"
#include "quant/quant_cpu.h"

namespace q = pelias::quant;
namespace g = pelias::kern::gpu;

namespace {
std::vector<float> wave(std::int64_t n, std::uint32_t seed, float scale) {
  std::vector<float> x(static_cast<std::size_t>(n));
  for (float& v : x) {
    seed = seed * 1664525u + 1013904223u;
    const float u = (static_cast<float>(seed >> 8) / 16777216.0f) - 0.5f;
    v = scale * u / (0.25f + std::fabs(u));
  }
  return x;
}

std::vector<std::uint8_t> quantized_row(q::Type t, const std::vector<float>& x) {
  const std::int64_t n = static_cast<std::int64_t>(x.size());
  std::vector<std::uint8_t> b(q::row_bytes(static_cast<std::uint32_t>(t), static_cast<std::uint64_t>(n)));
  switch (t) {
    case q::Type::f32: std::memcpy(b.data(), x.data(), b.size()); break;
    case q::Type::f16:
      for (std::int64_t i = 0; i < n; ++i) {
        const std::uint16_t h = q::fp32_to_fp16(x[static_cast<std::size_t>(i)]);
        std::memcpy(b.data() + 2 * i, &h, 2);
      }
      break;
    case q::Type::bf16:
      for (std::int64_t i = 0; i < n; ++i) {
        const std::uint16_t h = q::fp32_to_bf16(x[static_cast<std::size_t>(i)]);
        std::memcpy(b.data() + 2 * i, &h, 2);
      }
      break;
    case q::Type::q8_0: q::quantize_q8_0(x.data(), b.data(), n); break;
    case q::Type::q4_k: q::quantize_q4_k(x.data(), b.data(), n); break;
    case q::Type::q5_k: q::quantize_q5_k(x.data(), b.data(), n); break;
    case q::Type::q6_k: q::quantize_q6_k(x.data(), b.data(), n); break;
    case q::Type::iq4_xs: q::quantize_iq4_xs(x.data(), b.data(), n); break;
    case q::Type::q8_z: q::quantize_q8_z(x.data(), b.data(), n); break;
    case q::Type::q7_zt: q::quantize_q7_zt(x.data(), b.data(), n); break;
    case q::Type::q4_tg2: q::quantize_q4_tg(x.data(), b.data(), n, 2); break;
    default: break;
  }
  return b;
}
}  // namespace

TEST(gpu_decoders_equal_dequantize_row_bit_for_bit) {
  const q::Type types[] = {q::Type::f32, q::Type::f16, q::Type::bf16, q::Type::q8_0, q::Type::q4_k, q::Type::q5_k, q::Type::q6_k,
                           q::Type::iq4_xs, q::Type::q8_z, q::Type::q7_zt, q::Type::q4_tg2};
  float grid[64];
  for (int t = 0; t < 4; ++t)
    for (int i = 0; i < 16; ++i) grid[16 * t + i] = q::q4_tg_grid(t, 2)[i];
  const std::int64_t n = 2048;
  int checked = 0;
  for (const q::Type t : types) {
    const auto id = static_cast<std::uint32_t>(t);
    CHECK(g::type_supported(id));
    CHECK(g::block_bytes(id) == q::type_info(id).block_bytes);
    CHECK(g::block_values(id) == q::type_info(id).block_elements);
    for (std::uint32_t seed : {1u, 77u, 9001u}) {
      const std::vector<float> x = wave(n, seed + id, seed == 77u ? 3.0f : 0.05f);
      const std::vector<std::uint8_t> row = quantized_row(t, x);
      std::vector<float> ref(static_cast<std::size_t>(n)), dev(static_cast<std::size_t>(n));
      CHECK(q::dequantize_row(t, row.data(), ref.data(), n).ok());
      for (std::int64_t j = 0; j < n / 32; ++j) g::decode32(id, row.data(), j, grid, dev.data() + 32 * j);
      bool same = true, finite = true;
      for (std::int64_t i = 0; i < n; ++i) {
        std::uint32_t a, b;
        std::memcpy(&a, &ref[static_cast<std::size_t>(i)], 4);
        std::memcpy(&b, &dev[static_cast<std::size_t>(i)], 4);
        same = same && a == b;
        finite = finite && std::isfinite(dev[static_cast<std::size_t>(i)]);
      }
      if (!same) std::printf("    %s seed %u: decode32 differs from dequantize_row\n", q::type_info(id).name, seed);
      CHECK(same);
      CHECK(finite);
      ++checked;
    }
  }
  CHECK(checked == 33);
  CHECK(!g::type_supported(static_cast<std::uint32_t>(q::Type::q2_k)));
  std::printf("    11 types x 3 rows of 2048: every sub-block equal to dequantize_row\n");
}

TEST(gpu_decoder_half_float_is_exact_for_every_half) {
  bool same = true;
  for (std::uint32_t h = 0; h < 65536; ++h) {
    const float a = q::fp16_to_fp32(static_cast<std::uint16_t>(h)), b = g::half_float(static_cast<std::uint16_t>(h));
    std::uint32_t ua, ub;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    if (std::isnan(a)) same = same && std::isnan(b);
    else same = same && ua == ub;
  }
  CHECK(same);
}
