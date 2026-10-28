// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "io/sha256.h"
#include "pelias_test.h"
#include "quant/quant_cpu.h"
#include "unit/quant_golden.h"

using namespace pelias;
namespace q = pelias::quant;

TEST(quant_matches_the_reference_hashes) {
  // The bytes the reference library produces on a fixed input (P-E6 on synthetic data, every platform).
  const std::vector<float> x = ptest::golden_input();
  const auto n = static_cast<std::int64_t>(x.size());
  struct T {
    const char* name;
    q::Type type;
  };
  const T types[] = {{"Q8_0", q::Type::q8_0}, {"Q4_K", q::Type::q4_k}, {"Q6_K", q::Type::q6_k}, {"F16", q::Type::f16},
                     {"BF16", q::Type::bf16}, {"Q8_1", q::Type::q8_1}, {"Q8_K", q::Type::q8_k}};
  for (const T& t : types) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(q::row_bytes(static_cast<std::uint32_t>(t.type), static_cast<std::uint64_t>(n))), 0);
    CHECK(q::quantize_row(t.type, x.data(), out.data(), n).ok());
    std::string want;
    for (const auto& g : ptest::kGoldenHashes)
      if (std::string(g.type) == t.name) want = g.sha256;
    const std::string got = io::Sha256::hex_of(out.data(), out.size());
    CHECK_EQ(got, want);
    if (got != want) std::printf("      %s: got %s\n", t.name, got.c_str());
  }
}

TEST(quant_dequantizers_match_the_reference_hashes) {
  // Q2_K, Q3_K, Q5_K, IQ4_XS: the floats the reference produced from fixed blocks (every platform, bit for bit)
  for (const auto& g : ptest::kDequantGolden) {
    const std::vector<std::uint8_t> in = ptest::dequant_golden_input(g);
    const auto n = static_cast<std::int64_t>(64 * q::type_info(g.type_id).block_elements);
    std::vector<float> y(static_cast<std::size_t>(n));
    CHECK(q::dequantize_row(static_cast<q::Type>(g.type_id), in.data(), y.data(), n).ok());
    std::vector<std::uint8_t> bytes(y.size() * 4);
    for (std::size_t i = 0; i < y.size(); ++i) {  // little-endian, whatever the host
      std::uint32_t u;
      std::memcpy(&u, &y[i], 4);
      for (int b = 0; b < 4; ++b) bytes[i * 4 + static_cast<std::size_t>(b)] = static_cast<std::uint8_t>(u >> (8 * b));
    }
    const std::string got = io::Sha256::hex_of(bytes.data(), bytes.size());
    CHECK_EQ(got, std::string(g.sha256));
    if (got != g.sha256) std::printf("      %s: got %s\n", g.type, got.c_str());
  }
}

TEST(quant_rounding_and_halves) {
  CHECK_EQ(q::nearest_int(2.5f), 2);  // ties to even
  CHECK_EQ(q::nearest_int(3.5f), 4);
  CHECK_EQ(q::nearest_int(-2.5f), -2);
  CHECK_EQ(q::nearest_int(-0.4f), 0);
  CHECK_EQ(q::fp32_to_fp16(1.0f), 0x3C00);
  CHECK_EQ(q::fp32_to_fp16(-2.0f), 0xC000);
  CHECK_EQ(q::fp32_to_fp16(65504.0f), 0x7BFF);       // the largest half
  CHECK_EQ(q::fp32_to_fp16(65520.0f), 0x7C00);       // rounds to infinity
  CHECK_EQ(q::fp32_to_fp16(5.9604645e-8f), 0x0001);  // the smallest subnormal half
  CHECK_EQ(q::fp32_to_fp16(std::nanf("")) & 0x7FFF, 0x7E00);
  CHECK(q::fp16_to_fp32(0x3C00) == 1.0f);
  CHECK(q::fp16_to_fp32(0x0001) == 5.9604645e-8f);
  CHECK(std::isinf(q::fp16_to_fp32(0x7C00)));
  CHECK_EQ(q::fp32_to_bf16(1.0f), 0x3F80);
  CHECK(q::bf16_to_fp32(0x3F80) == 1.0f);
  CHECK_EQ(q::fp32_to_bf16(1.00390625f), 0x3F80);  // 1 + 2^-8: a tie, rounds to even (down)
  CHECK_EQ(q::fp32_to_bf16(1.01171875f), 0x3F82);  // 1 + 3*2^-8: a tie, rounds to even (up)
}

TEST(quant_round_trip_error_is_bounded) {
  // dequantize(quantize(x)) stays within the format's step on weight-like data
  const std::vector<float> x = ptest::golden_input();
  std::vector<float> w(x.begin(), x.begin() + 256);  // block 0: weights of size about 0.03
  struct T {
    q::Type type;
    float tol;  // a bound well above the format's worst step for |x| <= 0.03
  };
  const T types[] = {{q::Type::q8_0, 0.0002f}, {q::Type::q4_k, 0.004f}, {q::Type::q6_k, 0.001f}, {q::Type::f16, 0.00002f}, {q::Type::bf16, 0.0002f}};
  for (const T& t : types) {
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(q::row_bytes(static_cast<std::uint32_t>(t.type), 256)));
    std::vector<float> back(256);
    CHECK(q::quantize_row(t.type, w.data(), buf.data(), 256).ok());
    CHECK(q::dequantize_row(t.type, buf.data(), back.data(), 256).ok());
    float worst = 0;
    for (int i = 0; i < 256; ++i) worst = std::max(worst, std::fabs(back[static_cast<std::size_t>(i)] - w[static_cast<std::size_t>(i)]));
    CHECK(worst <= t.tol);
    if (worst > t.tol) std::printf("      %s worst %g\n", q::type_info(t.type).name, static_cast<double>(worst));
  }
}

TEST(quant_refuses_what_it_cannot_do) {
  std::vector<float> x(256, 1.0f);
  std::vector<std::uint8_t> buf(1024);
  CHECK(q::quantize_row(q::Type::iq2_xxs, x.data(), buf.data(), 256).code == Code::unsupported);  // not built yet (grid search)
  CHECK(q::quantize_row(q::Type::q4_k, x.data(), buf.data(), 100).code == Code::invalid_argument);  // not whole blocks
  CHECK(q::dequantize_row(q::Type::q8_k, buf.data(), x.data(), 256).code == Code::unsupported);    // activation type
  CHECK(q::can_quantize(q::Type::q6_k));
  CHECK(q::can_quantize(q::Type::iq4_xs));
  CHECK(!q::can_quantize(q::Type::iq2_xxs));
}

TEST(q8_z_layout_and_round_trip) {
  // Pelias-only id 200, 256 per block, 264 bytes (8.25 bits per weight)
  const q::TypeInfo ti = q::type_info(q::Type::q8_z);
  CHECK(ti.known && std::string(ti.name) == "Q8_Z" && ti.block_elements == 256 && ti.block_bytes == 264);
  CHECK(q::bits_per_weight(200) == 8.25);
  // heavy-tailed rows with a few outliers (dense-weight-like); 16 blocks
  std::vector<float> x(4096);
  std::uint32_t s = 12345;
  for (float& v : x) {
    s = s * 1664525u + 1013904223u;
    const float u = (static_cast<float>(s >> 8) / 16777216.0f) - 0.5f;
    v = 0.02f * u / (0.05f + std::fabs(u));
  }
  x[17] = 0.9f;
  x[1030] = -0.7f;
  std::vector<std::uint8_t> z(static_cast<std::size_t>(q::row_bytes(200, 4096))), z0(static_cast<std::size_t>(q::row_bytes(8, 4096)));
  std::vector<float> bz(4096), b0(4096);
  CHECK(q::quantize_row(q::Type::q8_z, x.data(), z.data(), 4096).ok());
  CHECK(q::dequantize_row(q::Type::q8_z, z.data(), bz.data(), 4096).ok());
  CHECK(q::quantize_row(q::Type::q8_0, x.data(), z0.data(), 4096).ok());
  CHECK(q::dequantize_row(q::Type::q8_0, z0.data(), b0.data(), 4096).ok());
  double ez = 0, e0 = 0, ex = 0;
  for (int i = 0; i < 4096; ++i) {
    ez += std::pow(bz[size_t(i)] - x[size_t(i)], 2);
    e0 += std::pow(b0[size_t(i)] - x[size_t(i)], 2);
    ex += std::pow(x[size_t(i)], 2);
  }
  std::printf("      Q8_Z relative error %.3g, Q8_0 %.3g\n", ez / ex, e0 / ex);
  CHECK(ez / ex < 2.0 * e0 / ex);  // same class as Q8_0 (the sandbox measured 0.000031 vs 0.000028 on real weights)
  // every stored value sits on its block's grid: level 1..63, q within -127..127, value = scale(d, lv) * q exactly
  for (int b = 0; b < 16; ++b) {
    const std::uint8_t* p = z.data() + 264 * b;
    std::uint16_t dh;
    std::memcpy(&dh, p, 2);
    const float d = q::fp16_to_fp32(dh);
    std::uint64_t packed = 0;
    for (int k = 0; k < 6; ++k) packed |= static_cast<std::uint64_t>(p[2 + k]) << (8 * k);
    for (int j = 0; j < 8; ++j) {
      const int lv = static_cast<int>((packed >> (6 * j)) & 63u);
      CHECK(lv >= 1 && lv <= 63);
      for (int k = 0; k < 32; ++k) {
        const auto qv = static_cast<std::int8_t>(p[8 + 32 * j + k]);
        CHECK(qv >= -127);
        CHECK(bz[size_t(256 * b + 32 * j + k)] == static_cast<float>(qv) * q::q8_z_scale(d, lv));
      }
    }
  }
  // an all-zero block stays zero and writes level 1, never level 0
  std::vector<float> zero(256, 0.0f), back(256, 1.0f);
  std::vector<std::uint8_t> zb(264);
  CHECK(q::quantize_row(q::Type::q8_z, zero.data(), zb.data(), 256).ok());
  CHECK(q::dequantize_row(q::Type::q8_z, zb.data(), back.data(), 256).ok());
  for (float v : back) CHECK(v == 0.0f);
  CHECK(zb[2] != 0);
}

TEST(q7_zt_layout_gates_and_round_trip) {
  // Pelias-only id 202, 256 per block, 241 bytes (7.53 bits per weight): 7-bit base, the 8th bit on two 32-blocks
  const q::TypeInfo ti = q::type_info(q::Type::q7_zt);
  CHECK(ti.known && std::string(ti.name) == "Q7_ZT" && ti.block_elements == 256 && ti.block_bytes == 241);
  std::vector<float> x(4096), imp(4096, 1.0f);
  std::uint32_t s = 4242;
  for (float& v : x) {
    s = s * 1664525u + 1013904223u;
    const float u = (static_cast<float>(s >> 8) / 16777216.0f) - 0.5f;
    v = 0.02f * u / (0.05f + std::fabs(u));
  }
  std::vector<std::uint8_t> zt(static_cast<std::size_t>(q::row_bytes(202, 4096))), z7(static_cast<std::size_t>(q::row_bytes(201, 4096))),
      z8(static_cast<std::size_t>(q::row_bytes(200, 4096)));
  std::vector<float> bt(4096), b7(4096), b8(4096);
  CHECK(q::quantize_row(q::Type::q7_zt, x.data(), zt.data(), 4096).ok());
  CHECK(q::dequantize_row(q::Type::q7_zt, zt.data(), bt.data(), 4096).ok());
  CHECK(q::quantize_row(q::Type::q7_z, x.data(), z7.data(), 4096).ok());
  CHECK(q::dequantize_row(q::Type::q7_z, z7.data(), b7.data(), 4096).ok());
  CHECK(q::quantize_row(q::Type::q8_z, x.data(), z8.data(), 4096).ok());
  CHECK(q::dequantize_row(q::Type::q8_z, z8.data(), b8.data(), 4096).ok());
  double et = 0, e7 = 0, e8 = 0;
  for (int i = 0; i < 4096; ++i) {
    et += std::pow(bt[size_t(i)] - x[size_t(i)], 2);
    e7 += std::pow(b7[size_t(i)] - x[size_t(i)], 2);
    e8 += std::pow(b8[size_t(i)] - x[size_t(i)], 2);
  }
  std::printf("      Q7_ZT %.3g, Q7_Z %.3g, Q8_Z %.3g\n", et, e7, e8);
  CHECK(et < e7 && et > e8);  // between the 7-bit and the 8-bit format
  for (int b = 0; b < 16; ++b) {
    const std::uint8_t mask = zt[size_t(241 * b + 8)];
    int bits = 0;
    for (int j = 0; j < 8; ++j) bits += (mask >> j) & 1;
    CHECK(bits == 2);
  }
  // the weighted path: heavy importance on one 32-block of each 256 pulls a gate onto it
  for (int b = 0; b < 16; ++b) for (int k = 0; k < 32; ++k) imp[size_t(256 * b + 32 * 5 + k)] = 1000.0f;
  CHECK(q::quantize_rows(q::Type::q7_zt, x.data(), zt.data(), 1, 4096, imp.data()).ok());
  for (int b = 0; b < 16; ++b) CHECK((zt[size_t(241 * b + 8)] >> 5) & 1u);
  std::vector<float> zero(256, 0.0f), back(256, 1.0f);
  std::vector<std::uint8_t> zb(241);
  CHECK(q::quantize_row(q::Type::q7_zt, zero.data(), zb.data(), 256).ok());
  CHECK(q::dequantize_row(q::Type::q7_zt, zb.data(), back.data(), 256).ok());
  for (float v : back) CHECK(std::fabs(v) == 0.0f);
}

TEST(q7_z_layout_and_round_trip) {
  // Pelias-only id 201, 256 per block, 232 bytes (7.25 bits per weight): Q8_Z's header with tight 7-bit values
  const q::TypeInfo ti = q::type_info(q::Type::q7_z);
  CHECK(ti.known && std::string(ti.name) == "Q7_Z" && ti.block_elements == 256 && ti.block_bytes == 232);
  CHECK(q::bits_per_weight(201) == 7.25);
  std::vector<float> x(4096);
  std::uint32_t s = 777;
  for (float& v : x) {
    s = s * 1664525u + 1013904223u;
    const float u = (static_cast<float>(s >> 8) / 16777216.0f) - 0.5f;
    v = 0.02f * u / (0.05f + std::fabs(u));
  }
  x[40] = 0.8f;
  std::vector<std::uint8_t> z7(static_cast<std::size_t>(q::row_bytes(201, 4096))), z8(static_cast<std::size_t>(q::row_bytes(200, 4096)));
  std::vector<float> b7(4096), b8(4096);
  CHECK(q::quantize_row(q::Type::q7_z, x.data(), z7.data(), 4096).ok());
  CHECK(q::dequantize_row(q::Type::q7_z, z7.data(), b7.data(), 4096).ok());
  CHECK(q::quantize_row(q::Type::q8_z, x.data(), z8.data(), 4096).ok());
  CHECK(q::dequantize_row(q::Type::q8_z, z8.data(), b8.data(), 4096).ok());
  double e7 = 0, e8 = 0, ex = 0;
  for (int i = 0; i < 4096; ++i) {
    e7 += std::pow(b7[size_t(i)] - x[size_t(i)], 2);
    e8 += std::pow(b8[size_t(i)] - x[size_t(i)], 2);
    ex += std::pow(x[size_t(i)], 2);
  }
  std::printf("      Q7_Z relative error %.3g, Q8_Z %.3g\n", e7 / ex, e8 / ex);
  CHECK(e7 > e8 && e7 / ex < 8.0 * e8 / ex);  // one value bit fewer: about 4x (the sandbox measured 4.0x on real weights)
  // every value decodes from its 7-bit field on the block's grid
  for (int b = 0; b < 16; ++b) {
    const std::uint8_t* p = z7.data() + 232 * b;
    std::uint16_t dh;
    std::memcpy(&dh, p, 2);
    const float d = q::fp16_to_fp32(dh);
    std::uint64_t packed = 0;
    for (int k = 0; k < 6; ++k) packed |= static_cast<std::uint64_t>(p[2 + k]) << (8 * k);
    for (int g = 0; g < 32; ++g) {
      std::uint64_t w = 0;
      for (int k = 0; k < 7; ++k) w |= static_cast<std::uint64_t>(p[8 + 7 * g + k]) << (8 * k);
      const int lv = static_cast<int>((packed >> (6 * (g / 4))) & 63u);
      CHECK(lv >= 1 && lv <= 63);
      for (int k = 0; k < 8; ++k) {
        const int u = static_cast<int>((w >> (7 * k)) & 127u);
        CHECK(u >= 1 && u <= 127);
        CHECK(b7[size_t(256 * b + 8 * g + k)] == static_cast<float>(u - 64) * q::q7_z_scale(d, lv));
      }
    }
  }
  std::vector<float> zero(256, 0.0f), back(256, 1.0f);
  std::vector<std::uint8_t> zb(232);
  CHECK(q::quantize_row(q::Type::q7_z, zero.data(), zb.data(), 256).ok());
  CHECK(q::dequantize_row(q::Type::q7_z, zb.data(), back.data(), 256).ok());
  for (float v : back) CHECK(v == 0.0f);
}

TEST(q4_tg_layout_tags_and_beats_q4_k_on_heavy_tails) {
  // Pelias-only id 203, 256 per block, 144 bytes (4.5 bits per weight, the size of Q4_K): the author's bit-tag matrix at 4
  // bits (C4): a 2-bit grid tag per 8 values. On heavy-tailed values (unrotated experts) it must beat Q4_K at equal bytes.
  const q::TypeInfo ti = q::type_info(q::Type::q4_tg);
  CHECK(ti.known && std::string(ti.name) == "Q4_TG" && ti.block_elements == 256 && ti.block_bytes == 144);
  const int n = 8192;
  std::vector<float> x(static_cast<std::size_t>(n));
  std::uint32_t s = 777;
  for (float& v : x) {  // Laplace: sign x -log(u), heavy-tailed like the unrotated experts
    s = s * 1664525u + 1013904223u;
    const float u = (static_cast<float>(s >> 8) + 0.5f) / 16777216.0f;
    s = s * 1664525u + 1013904223u;
    v = 0.01f * ((s >> 31) ? 1.0f : -1.0f) * -std::log(u);
  }
  std::vector<std::uint8_t> tg(static_cast<std::size_t>(q::row_bytes(203, n))), k4(static_cast<std::size_t>(q::row_bytes(12, n)));
  CHECK(tg.size() == k4.size());
  std::vector<float> bt(static_cast<std::size_t>(n)), bk(static_cast<std::size_t>(n));
  CHECK(q::quantize_row(q::Type::q4_tg, x.data(), tg.data(), n).ok());
  CHECK(q::dequantize_row(q::Type::q4_tg, tg.data(), bt.data(), n).ok());
  CHECK(q::quantize_row(q::Type::q4_k, x.data(), k4.data(), n).ok());
  CHECK(q::dequantize_row(q::Type::q4_k, k4.data(), bk.data(), n).ok());
  double et = 0, ek = 0, ref = 0;
  for (int i = 0; i < n; ++i) {
    et += std::pow(bt[size_t(i)] - x[size_t(i)], 2);
    ek += std::pow(bk[size_t(i)] - x[size_t(i)], 2);
    ref += std::pow(x[size_t(i)], 2);
  }
  int tags[4] = {0, 0, 0, 0};
  for (int b = 0; b < n / 256; ++b)
    for (int j = 0; j < 8; ++j)
      for (int g = 0; g < 4; ++g) ++tags[(tg[size_t(144 * b + 8 + j)] >> (2 * g)) & 3];
  std::printf("      Q4_TG %.4g vs Q4_K %.4g relative error (%.1f %%); tags used %d / %d / %d / %d\n", et / ref, ek / ref, 100.0 * (et / ek - 1.0),
              tags[0], tags[1], tags[2], tags[3]);
  CHECK(et < ek);
  CHECK(tags[0] + tags[1] + tags[2] + tags[3] == n / 8);
  // the grids: 16 ascending levels with top 1 (grid 3: 0.6)
  for (int t = 0; t < 4; ++t) {
    const float* g = q::q4_tg_grid(t);
    for (int i = 1; i < 16; ++i) CHECK(g[i] > g[i - 1]);
    CHECK(std::fabs(g[15] - (t == 3 ? 0.6f : 1.0f)) < 1e-6f);
  }
}
