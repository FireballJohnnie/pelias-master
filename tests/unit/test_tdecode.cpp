// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// T-decode on Q7_ZT: the 1-bit route equals the int8 fold route exactly; the missing-H route is the 0.5 fill; the split
// storage round-trips; the activation bit planes reproduce the values (portable and AVX2).
#include <cmath>
#include <cstring>
#include <vector>

#include "pelias_test.h"
#include "quant/quant_cpu.h"
#include "quant/tdecode.h"

namespace q = pelias::quant;

namespace {
std::vector<float> rows(int n, std::uint32_t seed, float amp) {
  std::vector<float> x(static_cast<std::size_t>(n));
  for (float& v : x) {
    seed = seed * 1664525u + 1013904223u;
    const float u = (static_cast<float>(seed >> 8) / 16777216.0f) - 0.5f;
    v = amp * u / (0.05f + std::fabs(u));
  }
  return x;
}
}  // namespace

TEST(tdecode_activation_planes_reproduce_values) {
  const std::vector<float> x = rows(4096, 99, 1.0f);
  std::vector<q::ActBlock32> a(128);
  q::act_quantize_32(x.data(), 4096, a.data());
  for (const q::ActBlock32& b : a) {
    std::uint32_t ref[8];
    q::act_planes_32(b.q, ref);
    for (int k = 0; k < 8; ++k) CHECK(ref[k] == b.plane[k]);  // the fast path (when present) equals the portable one
    for (int i = 0; i < 32; ++i) {
      int u = 0;
      for (int k = 0; k < 8; ++k) u |= static_cast<int>((b.plane[k] >> i) & 1u) << k;
      CHECK(u - 128 == b.q[i]);
    }
  }
}

TEST(tdecode_one_bit_route_equals_int8_fold_exactly) {
  const int n = 4096;
  const std::vector<float> w = rows(n, 7, 0.02f), x = rows(n, 11, 1.0f);
  std::vector<std::uint8_t> z(static_cast<std::size_t>(q::row_bytes(202, n)));
  CHECK(q::quantize_row(q::Type::q7_zt, w.data(), z.data(), n).ok());
  std::vector<std::uint8_t> base(static_cast<std::size_t>(q::kQ7ZTBaseBytes * n / 256)), hb(static_cast<std::size_t>(q::kQ7ZTHBytes * n / 256)),
      back(z.size());
  q::q7_zt_split(z.data(), n, base.data(), hb.data());
  q::q7_zt_join(base.data(), hb.data(), n, back.data());
  CHECK(std::memcmp(back.data(), z.data(), z.size()) == 0);
  std::vector<q::ActBlock32> a(static_cast<std::size_t>(n / 32));
  q::act_quantize_32(x.data(), n, a.data());
  std::vector<std::int64_t> pf(static_cast<std::size_t>(n / 32)), p1(pf.size()), pm(pf.size());
  const float yf = q::q7_zt_dot(base.data(), hb.data(), a.data(), n, q::HRoute::fold, pf.data());
  const float y1 = q::q7_zt_dot(base.data(), hb.data(), a.data(), n, q::HRoute::one_bit, p1.data());
  const float ym = q::q7_zt_dot(base.data(), nullptr, a.data(), n, q::HRoute::missing, pm.data());
  for (std::size_t j = 0; j < pf.size(); ++j) CHECK(pf[j] == p1[j]);  // integer-exact, every 32-block
  CHECK(yf == y1);
  // against a double-precision dot of the dequantized weights and activations
  std::vector<float> wd(static_cast<std::size_t>(n));
  CHECK(q::dequantize_row(q::Type::q7_zt, z.data(), wd.data(), n).ok());
  double ref = 0, refm = 0, mag = 0;
  for (int i = 0; i < n; ++i) {
    const q::ActBlock32& b = a[static_cast<std::size_t>(i / 32)];
    const double xv = static_cast<double>(b.q[i % 32]) * b.scale;
    ref += static_cast<double>(wd[static_cast<std::size_t>(i)]) * xv;
    mag += std::fabs(static_cast<double>(wd[static_cast<std::size_t>(i)]) * xv);
  }
  CHECK(std::fabs(static_cast<double>(y1) - ref) < 1e-5 * mag);
  // the missing route: gated blocks decoded with h = 0.5
  for (int i = 0; i < n / 256; ++i) {
    const std::uint8_t mask = z[static_cast<std::size_t>(241 * i + 8)];
    for (int j = 0; j < 8; ++j) {
      const q::ActBlock32& b = a[static_cast<std::size_t>(8 * i + j)];
      const std::int64_t full2 = pf[static_cast<std::size_t>(8 * i + j)], miss2 = pm[static_cast<std::size_t>(8 * i + j)];
      if (!((mask >> j) & 1u)) { CHECK(full2 == miss2); continue; }
      // 2 * sum (2b - 127 + 0.5) x = 2 * (full - sum h x) + sum x
      std::int64_t hx = 0;
      std::uint32_t h = 0;
      int slot = 0;
      for (int jj = 0; jj < j; ++jj) slot += static_cast<int>((mask >> jj) & 1u);
      std::memcpy(&h, z.data() + 241 * i + 233 + 4 * slot, 4);
      for (int k = 0; k < 32; ++k) hx += static_cast<std::int64_t>((h >> k) & 1u) * b.q[k];
      CHECK(miss2 == full2 - 2 * hx + b.sum);
    }
  }
  refm = static_cast<double>(ym);
  std::printf("      T-decode: fold %.6f, 1-bit %.6f, H missing (0.5 fill) %.6f, double reference %.6f\n", static_cast<double>(yf),
              static_cast<double>(y1), refm, ref);
}

// C6 miss map: the state code keeps signs and log-energy; scaling the vector by 2 moves every energy byte by 16 and
// keeps every sign; negating it flips every sign bit; an all-zero block codes energy 0.
TEST(tdecode_state_code_signs_and_energy) {
  const int n = 2048;
  std::vector<float> x = rows(n, 7u, 1.0f);
  for (int k = 0; k < 32; ++k) x[static_cast<std::size_t>(64 + k)] = 0.0f;  // block 2 all zero
  const std::int64_t cb = q::state_code_bytes(n);
  CHECK(cb == n / 8 + n / 32);
  std::vector<std::uint8_t> a(static_cast<std::size_t>(cb)), b(static_cast<std::size_t>(cb)), c(static_cast<std::size_t>(cb));
  q::state_code(x.data(), n, a.data());
  std::vector<float> x2(x), xm(x);
  for (float& v : x2) v *= 2.0f;
  for (float& v : xm) v = -v;
  q::state_code(x2.data(), n, b.data());
  q::state_code(xm.data(), n, c.data());
  int flips = 0, zero_bits = 0;
  for (int i = 0; i < n / 8; ++i) {
    CHECK(a[static_cast<std::size_t>(i)] == b[static_cast<std::size_t>(i)]);
    if (i / 4 == 2) { zero_bits += __builtin_popcount(a[static_cast<std::size_t>(i)]) + __builtin_popcount(c[static_cast<std::size_t>(i)]); continue; }
    flips += __builtin_popcount(static_cast<unsigned>(a[static_cast<std::size_t>(i)] ^ c[static_cast<std::size_t>(i)]));
  }
  CHECK(zero_bits == 0);                // an all-zero block has no positive value either way
  CHECK(flips == n - 32);               // every other value flips sign (no exact zeros after the rotation here)
  for (int j = 0; j < n / 32; ++j) {
    const std::uint8_t ea = a[static_cast<std::size_t>(n / 8 + j)], eb = b[static_cast<std::size_t>(n / 8 + j)];
    if (j == 2) { CHECK(ea == 0 && eb == 0); continue; }
    CHECK(ea >= 1 && eb == ea + 16);
  }
  // the sign bits equal the signs of the rotated block (fixed diagonal, then Hadamard per 32)
  std::printf("      state code: %lld bytes per vector of %d values (%d sign bytes, %d energy bytes)\n", static_cast<long long>(cb), n, n / 8, n / 32);
}
