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

#include "gpu_test_rows.h"
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

}  // namespace

TEST(gpu_decoders_equal_dequantize_row_bit_for_bit) {
  namespace tg = pelias::test_gpu;
  const std::vector<q::Type> types = tg::readable_types();
  const g::Tables tb = tg::host_tables();
  const std::int64_t n = 2048;
  int checked = 0;
  for (const q::Type t : types) {
    const auto id = static_cast<std::uint32_t>(t);
    CHECK(g::type_supported(id));
    CHECK(g::block_bytes(id) == q::type_info(id).block_bytes);
    CHECK(g::block_values(id) == q::type_info(id).block_elements);
    for (std::uint32_t seed : {1u, 77u, 9001u}) {
      const std::vector<float> x = wave(n, seed + id, seed == 77u ? 3.0f : 0.05f);
      const std::vector<std::uint8_t> row = tg::make_row(t, x, seed * 31u + id);
      std::vector<float> ref(static_cast<std::size_t>(n)), dev(static_cast<std::size_t>(n));
      CHECK(q::dequantize_row(t, row.data(), ref.data(), n).ok());
      for (std::int64_t j = 0; j < n / 32; ++j) g::decode32(id, row.data(), j, tb, dev.data() + 32 * j);
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
  CHECK(types.size() == 32);  // every type dequantize_row reads
  CHECK(checked == 3 * static_cast<int>(types.size()));
  CHECK(!g::type_supported(static_cast<std::uint32_t>(q::Type::q8_1)));  // an activation type: never a weight
  CHECK(!g::type_supported(static_cast<std::uint32_t>(q::Type::q8_k)));
  std::printf("    %zu types x 3 rows of 2048: every sub-block equal to dequantize_row\n", types.size());
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
