// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// Fast CPU row dot products (kern/cpu_dot.h, GP3-10): for every type with an AVX2 dequantizer and a few that take the
// generic path, the AVX2 and portable paths agree bit for bit, and both match the reference (the row as
// quant::dequantize_row gives it, planes added as quant::plane_add_row adds them, summed in double) within the rung-1
// tolerance; dense and block-selective planes, fewer planes read than stored, and rows that are not whole segments.
#include <cmath>
#include <cstdio>
#include <vector>

#include "kern/cpu_dot.h"
#include "pelias_test.h"
#include "quant/planes.h"
#include "quant/quant_cpu.h"

namespace q = pelias::quant;
namespace kn = pelias::kern;

namespace {
std::vector<float> noise(std::int64_t n, std::uint32_t seed, float scale) {
  std::vector<float> x(static_cast<std::size_t>(n));
  for (float& v : x) {
    seed = seed * 1664525u + 1013904223u;
    const float u = (static_cast<float>(seed >> 8) / 16777216.0f) - 0.5f;
    v = scale * u / (0.25f + std::fabs(u));  // heavy-ish tails
  }
  return x;
}

struct Case {
  q::Type t;
  const char* name;
};

const Case kCases[] = {{q::Type::q8_0, "Q8_0"},     {q::Type::q4_k, "Q4_K"},  {q::Type::q5_k, "Q5_K"},   {q::Type::q6_k, "Q6_K"},
                       {q::Type::iq4_xs, "IQ4_XS"}, {q::Type::q4_tg, "Q4_TG"}, {q::Type::q4_tg2, "Q4_TG2"}, {q::Type::q8_z, "Q8_Z"},
                       {q::Type::q3_k, "Q3_K"},     {q::Type::f16, "F16"},     {q::Type::bf16, "BF16"},   {q::Type::f32, "F32"}};

// the reference: dequantize_row, planes as the forward pass adds them, then the dot in double; also sum |w x|
void reference(q::Type t, const std::vector<std::uint8_t>& row, std::int64_t k, const kn::RowPlanes* pl, const float* x, double* dot,
               double* mag) {
  std::vector<float> w(static_cast<std::size_t>(k));
  (void)q::dequantize_row(t, row.data(), w.data(), k);
  if (pl && pl->data) {
    const int ku = std::min(pl->k_use, pl->k_stored);
    if (pl->mask) q::plane_add_row_sparse(pl->data, pl->mask, k, pl->k_stored, ku, w.data());
    else q::plane_add_row(pl->data, k, pl->k_stored, ku, w.data());
  }
  *dot = 0.0;
  *mag = 0.0;
  for (std::int64_t i = 0; i < k; ++i) {
    const double p = static_cast<double>(w[static_cast<std::size_t>(i)]) * static_cast<double>(x[i]);
    *dot += p;
    *mag += std::fabs(p);
  }
}

// runs one configuration on every active path, checks the paths agree and match the reference; returns false on a failure
bool run(q::Type t, const char* name, std::int64_t k, const kn::RowPlanes* pl, std::uint32_t seed) {
  const std::vector<float> src = noise(k, seed, 0.05f);
  std::vector<std::uint8_t> row(static_cast<std::size_t>(q::row_bytes(static_cast<std::uint32_t>(t), static_cast<std::uint64_t>(k))));
  if (row.empty() || !q::quantize_row(t, src.data(), row.data(), k)) {
    std::printf("  cpu_dot: %s k=%lld could not be quantized\n", name, static_cast<long long>(k));
    return false;
  }
  std::vector<std::vector<float>> xs;
  for (int tt = 0; tt < 4; ++tt) xs.push_back(noise(k, seed * 7u + static_cast<std::uint32_t>(tt) + 1u, 1.0f));
  const float* xp[4] = {xs[0].data(), xs[1].data(), xs[2].data(), xs[3].data()};
  bool ok = true;
  for (int nt = 1; nt <= 4; ++nt) {
    float port[4] = {}, fast[4] = {};
    kn::force_isa(kn::Isa::portable);
    kn::dot_row(t, row.data(), k, pl, xp, nt, port);
    kn::force_isa(kn::Isa::avx2);
    kn::dot_row(t, row.data(), k, pl, xp, nt, fast);
    for (int tt = 0; tt < nt; ++tt) {
      if (kn::best_isa() == kn::Isa::avx2 && !(port[tt] == fast[tt])) {
        std::printf("  cpu_dot: %s k=%lld nt=%d t=%d: avx2 %.9g vs portable %.9g\n", name, static_cast<long long>(k), nt, tt,
                    static_cast<double>(fast[tt]), static_cast<double>(port[tt]));
        ok = false;
      }
      double ref, mag;
      reference(t, row, k, pl, xp[tt], &ref, &mag);
      const double err = std::fabs(static_cast<double>(port[tt]) - ref);
      if (err > 2e-6 * mag + 1e-30) {
        std::printf("  cpu_dot: %s k=%lld nt=%d t=%d: %.9g vs reference %.9g (|err| %.3g, sum|wx| %.3g)\n", name,
                    static_cast<long long>(k), nt, tt, static_cast<double>(port[tt]), ref, err, mag);
        ok = false;
      }
    }
  }
  return ok;
}
}  // namespace

TEST(cpu_dot_matches_the_reference_on_every_type) {
  for (const Case& c : kCases)
    for (std::int64_t k : {256, 512, 2048}) CHECK(run(c.t, c.name, k, nullptr, static_cast<std::uint32_t>(k) + 17u));
}

TEST(cpu_dot_with_dense_and_block_selective_planes) {
  const std::int64_t k = 2048;
  for (const Case& c : kCases) {
    const std::vector<float> r = noise(k, 99, 0.004f);
    for (int ks : {1, 2, 3}) {
      std::vector<std::uint8_t> pl(static_cast<std::size_t>(q::plane_row_bytes(k, ks)));
      (void)q::plane_fit_row(r.data(), nullptr, k, ks, 4, pl.data());
      for (int ku = 0; ku <= ks + 1; ++ku) {  // ku > ks: read as ks
        kn::RowPlanes dense{pl.data(), nullptr, ks, ku};
        CHECK(run(c.t, c.name, k, &dense, 5u + static_cast<std::uint32_t>(ks * 10 + ku)));
      }
      // block-selective: blocks 0, 2, 3, 7 of 8 carry planes, stored back to back
      const std::uint8_t mask[1] = {0x8D};
      std::vector<std::uint8_t> sp;
      for (int b = 0; b < 8; ++b)
        if ((mask[0] >> b) & 1) sp.insert(sp.end(), pl.begin() + b * 34 * ks, pl.begin() + (b + 1) * 34 * ks);
      kn::RowPlanes sparse{sp.data(), mask, ks, ks};
      CHECK(run(c.t, c.name, k, &sparse, 41u + static_cast<std::uint32_t>(ks)));
    }
  }
}

TEST(cpu_dot_rows_that_are_not_whole_segments) {
  CHECK(run(q::Type::q8_0, "Q8_0", 96, nullptr, 3));
  CHECK(run(q::Type::f32, "F32", 100, nullptr, 4));
  CHECK(run(q::Type::f16, "F16", 300, nullptr, 5));
}

TEST(cpu_dot_reports_its_paths) {
  CHECK(kn::dot_has_fast_path(q::Type::q4_tg2));
  CHECK(kn::dot_has_fast_path(q::Type::q6_k));
  CHECK(!kn::dot_has_fast_path(q::Type::q8_z));
  CHECK(kn::dot_supported(q::Type::q8_z));
  kn::force_isa(kn::Isa::portable);
  CHECK(kn::active_isa() == kn::Isa::portable);
  kn::force_isa(kn::Isa::avx2);
  CHECK(kn::active_isa() == kn::best_isa());
}
