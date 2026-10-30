// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// gpu_test_rows.h - shared by test_gpu_decode (host) and test_gpu_rung1 (card): every type quant::dequantize_row reads, a
// row of each, and the host copies of the decoders' format tables. Types with a quantizer get a quantized row of the
// given floats; the read-only types (no quantizer: Q4_0 / Q4_1 / Q5_0 / Q5_1, MXFP4, NVFP4, TQ1_0 / TQ2_0, I2_S, the IQ1 /
// IQ2 / IQ3 grids) get random bytes, so every code, grid index and sign pattern is reached, with each block's scale field
// set to a finite small half (MXFP4's exponent byte kept in 100-139) so the products stay finite.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

#include "kern/gpu_decode.h"
#include "quant/block_types.h"
#include "quant/iq_tables.h"
#include "quant/quant_cpu.h"

namespace pelias::test_gpu {

namespace q = pelias::quant;
namespace g = pelias::kern::gpu;

// every type the CPU dequantizes, in id order (31)
inline std::vector<q::Type> readable_types() {
  std::vector<q::Type> v;
  for (std::uint32_t id = 0; id < 256; ++id)
    if (q::type_info(id).known && q::can_dequantize(static_cast<q::Type>(id))) v.push_back(static_cast<q::Type>(id));
  return v;
}

inline g::Tables host_tables() {
  static float g1[64], g2[64];
  for (int t = 0; t < 4; ++t)
    for (int i = 0; i < 16; ++i) {
      g1[16 * t + i] = q::q4_tg_grid(t, 1)[i];
      g2[16 * t + i] = q::q4_tg_grid(t, 2)[i];
    }
  g::Tables tb;
  tb.tg1 = g1;
  tb.tg2 = g2;
  tb.iq2xxs = q::iq::kIq2xxsGrid;
  tb.iq2xs = q::iq::kIq2xsGrid;
  tb.iq2s = q::iq::kIq2sGrid;
  tb.iq3xxs = q::iq::kIq3xxsGrid;
  tb.iq3s = q::iq::kIq3sGrid;
  tb.iq1s = q::iq::kIq1sGrid;
  tb.signs = q::iq::kSigns;
  return tb;
}

inline void put_half(std::uint8_t* p, std::uint16_t h) {
  p[0] = static_cast<std::uint8_t>(h & 0xFF);
  p[1] = static_cast<std::uint8_t>(h >> 8);
}

// a row of type t: quantized from x when the CPU has a quantizer, else random bytes with finite scales
inline std::vector<std::uint8_t> make_row(q::Type t, const std::vector<float>& x, std::uint32_t seed) {
  const auto id = static_cast<std::uint32_t>(t);
  const auto n = static_cast<std::int64_t>(x.size());
  std::vector<std::uint8_t> b(q::row_bytes(id, static_cast<std::uint64_t>(n)));
  if (q::can_quantize(t)) {
    (void)q::quantize_row(t, x.data(), b.data(), n);
    return b;
  }
  for (std::uint8_t& v : b) {
    seed = seed * 1664525u + 1013904223u;
    v = static_cast<std::uint8_t>(seed >> 24);
  }
  const q::TypeInfo ti = q::type_info(id);
  const std::size_t nb = b.size() / ti.block_bytes;
  for (std::size_t i = 0; i < nb; ++i) {
    std::uint8_t* blk = b.data() + i * ti.block_bytes;
    seed = seed * 1664525u + 1013904223u;
    const float u = static_cast<float>(seed >> 8) / 16777216.0f;
    const std::uint16_t hf = q::fp32_to_fp16((seed & 1u ? -1.0f : 1.0f) * (0.002f + 0.03f * u));
    switch (t) {
      case q::Type::q4_0: case q::Type::q5_0: case q::Type::iq2_xxs: case q::Type::iq2_xs: case q::Type::iq2_s:
      case q::Type::iq3_xxs: case q::Type::iq3_s: case q::Type::iq1_s:
        put_half(blk, hf);
        break;
      case q::Type::q4_1: case q::Type::q5_1:
        put_half(blk, hf);
        put_half(blk + 2, q::fp32_to_fp16(0.25f * u));
        break;
      case q::Type::tq1_0: put_half(blk + 52, hf); break;
      case q::Type::tq2_0: put_half(blk + 64, hf); break;
      case q::Type::mxfp4: blk[0] = static_cast<std::uint8_t>(100 + (seed >> 3) % 40); break;
      case q::Type::iq1_m:  // the half sits in the top nibbles of the four scale words (bytes 49, 51, 53, 55)
        for (int k = 0; k < 4; ++k) blk[49 + 2 * k] = static_cast<std::uint8_t>((blk[49 + 2 * k] & 0x0F) | (((hf >> (4 * k)) & 0xF) << 4));
        break;
      default: break;  // NVFP4 (E4M3 scales are finite for every byte), I2_S (no scale)
    }
  }
  return b;
}

}  // namespace pelias::test_gpu
