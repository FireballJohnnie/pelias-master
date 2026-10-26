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
  CHECK(q::quantize_row(q::Type::q5_k, x.data(), buf.data(), 256).code == Code::unsupported);  // not built yet
  CHECK(q::quantize_row(q::Type::q4_k, x.data(), buf.data(), 100).code == Code::invalid_argument);  // not whole blocks
  CHECK(q::dequantize_row(q::Type::q8_k, buf.data(), x.data(), 256).code == Code::unsupported);    // activation type
  CHECK(q::can_quantize(q::Type::q6_k));
  CHECK(!q::can_quantize(q::Type::iq4_xs));
}
