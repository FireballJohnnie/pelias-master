// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// Residual sign planes (quant/planes.h): the encoding reads back what the fit chose, each plane lowers the error, the joint
// refit is never worse than plane by plane, reading fewer planes than stored is valid, and a block without importance is
// fitted with equal weights.
#include <cmath>
#include <vector>

#include "pelias_test.h"
#include "quant/planes.h"
#include "quant/quant_cpu.h"

namespace q = pelias::quant;

namespace {
std::vector<float> residual(int n, std::uint32_t seed) {  // heavy-ish tails, as a rounding residual mixed over rows
  std::vector<float> x(static_cast<std::size_t>(n));
  for (float& v : x) {
    seed = seed * 1664525u + 1013904223u;
    const float u = (static_cast<float>(seed >> 8) / 16777216.0f) - 0.5f;
    v = 0.01f * u / (0.2f + std::fabs(u));
  }
  return x;
}
double err(const std::vector<float>& r, const std::vector<float>& y) {
  double e = 0.0;
  for (std::size_t i = 0; i < r.size(); ++i) e += static_cast<double>(r[i] - y[i]) * (r[i] - y[i]);
  return e;
}
}  // namespace

TEST(planes_encode_reads_back_and_each_plane_lowers_the_error) {
  const int n = 2048;
  const std::vector<float> r = residual(n, 7);
  const double e0 = err(r, std::vector<float>(n, 0.0f));
  double prev = e0;
  for (int k = 1; k <= 3; ++k) {
    std::vector<std::uint8_t> p(static_cast<std::size_t>(q::plane_row_bytes(n, k)));
    const double fit = q::plane_fit_row(r.data(), nullptr, n, k, 8, p.data());
    std::vector<float> y(n, 0.0f);
    q::plane_add_row(p.data(), n, k, k, y.data());
    const double e = err(r, y);
    std::printf("    %d plane(s): error %.3g of %.3g (%.3f)\n", k, e, e0, e / e0);
    CHECK(std::fabs(e - fit) <= 1e-6 * e0);  // the encoding gives back exactly what the fit measured
    CHECK(e < 0.5 * prev);
    prev = e;
    if (k >= 2) {  // the first plane alone of a stored pair is still a valid, better-than-nothing read
      std::vector<float> y1(n, 0.0f);
      q::plane_add_row(p.data(), n, k, 1, y1.data());
      CHECK(err(r, y1) < e0);
    }
  }
}

TEST(planes_refit_never_worse_and_importance_weighted) {
  const int n = 1024;
  const std::vector<float> r = residual(n, 11);
  std::vector<float> imp(n);
  for (int i = 0; i < n; ++i) imp[static_cast<std::size_t>(i)] = (i % 7 == 0) ? 50.0f : 1.0f;
  for (int k = 2; k <= 3; ++k) {
    std::vector<std::uint8_t> a(static_cast<std::size_t>(q::plane_row_bytes(n, k))), b(a.size());
    const double greedy = q::plane_fit_row(r.data(), imp.data(), n, k, 0, a.data());
    const double refit = q::plane_fit_row(r.data(), imp.data(), n, k, 8, b.data());
    std::printf("    %d planes, weighted: plane by plane %.4g, refit %.4g\n", k, greedy, refit);
    CHECK(refit <= greedy);
  }
  // a block whose importance is all zero is fitted as if every value weighed the same
  std::vector<float> zero(n, 0.0f);
  std::vector<std::uint8_t> p(static_cast<std::size_t>(q::plane_row_bytes(n, 1))), p1(p.size());
  (void)q::plane_fit_row(r.data(), zero.data(), n, 1, 0, p.data());
  (void)q::plane_fit_row(r.data(), nullptr, n, 1, 0, p1.data());
  CHECK(p == p1);
}

TEST(planes_add_matches_value_by_value) {
  // the table-driven add gives exactly the sums of the plain definition: y += s_p x (bit ? +1 : -1), plane by plane
  const int n = 1024;
  const std::vector<float> r = residual(n, 23);
  for (int k = 1; k <= 3; ++k) {
    std::vector<std::uint8_t> p(static_cast<std::size_t>(q::plane_row_bytes(n, k)));
    (void)q::plane_fit_row(r.data(), nullptr, n, k, 8, p.data());
    for (int use = 1; use <= k; ++use) {
      std::vector<float> fast(r), plain(r);
      q::plane_add_row(p.data(), n, k, use, fast.data());
      for (int b0 = 0; b0 < n; b0 += 256) {
        const std::uint8_t* o = p.data() + (b0 / 256) * 34 * k;
        for (int pl = 0; pl < use; ++pl) {
          const float s = q::fp16_to_fp32(static_cast<std::uint16_t>(o[2 * pl] | (o[2 * pl + 1] << 8)));
          for (int i = 0; i < 256; ++i) plain[static_cast<std::size_t>(b0 + i)] += ((o[2 * k + 32 * pl + i / 8] >> (i % 8)) & 1) ? s : -s;
        }
      }
      CHECK(fast == plain);
    }
  }
}
