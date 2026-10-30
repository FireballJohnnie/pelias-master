// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// Fast CPU row dot products (kern/cpu_dot.h, GP3-10): for every type with an AVX2 dequantizer and a few that take the
// generic path, the AVX2 and portable paths agree bit for bit, and both match the reference (the row as
// quant::dequantize_row gives it, planes added as quant::plane_add_row adds them, summed in double) within the rung-1
// tolerance; dense and block-selective planes, fewer planes read than stored, and rows that are not whole segments.
#include <cmath>
#include <cstdio>
#include <cstring>
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
      const std::size_t blk = static_cast<std::size_t>(34 * ks);
      std::vector<std::uint8_t> sp(4 * blk);  // blocks 0, 2, 3, 7 (a plain copy: gcc 15 misreads vector::insert here, -Wfree-nonheap-object)
      std::size_t at = 0;
      for (int b = 0; b < 8; ++b)
        if ((mask[0] >> b) & 1) {
          std::memcpy(sp.data() + at, pl.data() + static_cast<std::size_t>(b) * blk, blk);
          at += blk;
        }
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
  for (std::uint32_t id = 0; id < 256; ++id)  // every readable type has its own AVX2 dequantizer (author 11:20: all 30)
    if (q::can_dequantize(static_cast<q::Type>(id))) CHECK(kn::dot_has_fast_path(static_cast<q::Type>(id)));
  CHECK(!kn::dot_supported(static_cast<q::Type>(250)));
  kn::force_isa(kn::Isa::portable);
  CHECK(kn::active_isa() == kn::Isa::portable);
  kn::force_isa(kn::Isa::avx2);
  CHECK(kn::active_isa() == kn::best_isa());
}

// Every type the reference reads, on synthetic bytes (random, with bit 6 of every byte cleared so no fp16 / E8M0 scale is
// Inf or NaN): AVX2 = portable bit for bit, both = the reference within the tolerance, no value left out (a row whose
// reference dot is non-zero never comes back as 0). Covers the types without a quantizer here (Q4_0, Q4_1, Q5_0, Q5_1,
// MXFP4, the IQ1 / IQ2 / IQ3 grids, TQ1_0 / TQ2_0, ...).
TEST(cpu_dot_reads_every_readable_type) {
  int types = 0, fast = 0;
  for (std::uint32_t id = 0; id < 256; ++id) {
    const q::Type t = static_cast<q::Type>(id);
    if (!q::can_dequantize(t)) continue;
    for (std::int64_t k : {256, 1024}) {
      const std::uint64_t rb = q::row_bytes(id, static_cast<std::uint64_t>(k));
      if (rb == 0) continue;
      std::vector<std::uint8_t> row(rb);
      std::uint32_t seed = id * 2654435761u + static_cast<std::uint32_t>(k);
      for (std::uint8_t& b : row) {
        seed = seed * 1664525u + 1013904223u;
        b = static_cast<std::uint8_t>((seed >> 24) & 0xBF);
      }
      const std::vector<float> x = noise(k, seed, 1.0f);
      const float* xp[1] = {x.data()};
      float port = 0.0f, avx = 0.0f;
      kn::force_isa(kn::Isa::portable);
      kn::dot_row(t, row.data(), k, nullptr, xp, 1, &port);
      kn::force_isa(kn::Isa::avx2);
      kn::dot_row(t, row.data(), k, nullptr, xp, 1, &avx);
      double ref, mag;
      reference(t, row, k, nullptr, x.data(), &ref, &mag);
      const bool same = kn::best_isa() != kn::Isa::avx2 || port == avx;
      const bool close = std::isfinite(ref) && std::fabs(static_cast<double>(port) - ref) <= 2e-6 * mag + 1e-30;
      if (!same || !close)
        std::printf("  cpu_dot: type %u (%s) k=%lld: avx2 %.9g portable %.9g reference %.9g (sum|wx| %.3g)\n", id, q::type_info(id).name,
                    static_cast<long long>(k), static_cast<double>(avx), static_cast<double>(port), ref, mag);
      CHECK(same);
      CHECK(close);
      CHECK(ref == 0.0 || port != 0.0f);
    }
    ++types;
    if (kn::dot_has_fast_path(t)) ++fast;
  }
  std::printf("    cpu_dot: %d readable types checked, %d with an AVX2 dequantizer of their own\n", types, fast);
  CHECK(types >= 30);
  CHECK(fast == types);
}

// The int8-activation path (GP3-10 item 5): activations rounded per 32 values (the engine's gpu256x8 format); every type
// with an integer form sums integer products per block. AVX2 = portable bit for bit; both within the rung-1 tolerance of
// fp64 over the same rounded inputs (the row as dequantize_row gives it, planes added, times s_b q); recovering the
// integers from rounded values gives the same integers.
namespace {
bool run_q8(std::uint32_t id, std::int64_t k, const kn::RowPlanes* pl, std::uint32_t seed, int* checked) {
  const q::Type t = static_cast<q::Type>(id);
  std::vector<std::uint8_t> row(q::row_bytes(id, static_cast<std::uint64_t>(k)));
  if (row.empty()) return true;
  if (q::can_quantize(t)) {
    const std::vector<float> src = noise(k, seed, 0.05f);
    if (!q::quantize_row(t, src.data(), row.data(), k)) return false;
  } else {
    std::uint32_t r = seed;
    for (std::uint8_t& b : row) {
      r = r * 1664525u + 1013904223u;
      b = static_cast<std::uint8_t>((r >> 24) & 0xBF);
    }
  }
  std::vector<kn::ActQ8x32> acts(4);
  for (int tt = 0; tt < 4; ++tt) {
    const std::vector<float> x = noise(k, seed * 3u + static_cast<std::uint32_t>(tt), 2.0f);
    kn::act_q8x32_quantize(x.data(), k, acts[static_cast<std::size_t>(tt)]);
    kn::ActQ8x32 back;  // the same integers recovered from the rounded values
    kn::act_q8x32_from_rounded(acts[static_cast<std::size_t>(tt)].x.data(), k, back);
    if (back.q != acts[static_cast<std::size_t>(tt)].q) return false;
  }
  const kn::ActQ8x32* ap[4] = {&acts[0], &acts[1], &acts[2], &acts[3]};
  bool ok = true;
  for (int nt = 1; nt <= 4; ++nt) {
    float port[4] = {}, fast[4] = {};
    kn::force_isa(kn::Isa::portable);
    kn::dot_row_q8(t, row.data(), k, pl, ap, nt, port);
    kn::force_isa(kn::Isa::avx2);
    kn::dot_row_q8(t, row.data(), k, pl, ap, nt, fast);
    for (int tt = 0; tt < nt; ++tt) {
      double ref, mag;
      reference(t, row, k, pl, acts[static_cast<std::size_t>(tt)].x.data(), &ref, &mag);
      const bool same = kn::best_isa() != kn::Isa::avx2 || port[tt] == fast[tt];
      const bool close = std::isfinite(ref) && std::fabs(static_cast<double>(port[tt]) - ref) <= 2e-6 * mag + 1e-30;
      if (!same || !close) {
        std::printf("  dot_q8: type %u (%s) k=%lld nt=%d: avx2 %.9g portable %.9g reference %.9g (sum|wx| %.3g)\n", id, q::type_info(id).name,
                    static_cast<long long>(k), nt, static_cast<double>(fast[tt]), static_cast<double>(port[tt]), ref, mag);
        ok = false;
      }
    }
  }
  ++*checked;
  return ok;
}
}  // namespace

TEST(cpu_dot_int8_activation_path_on_every_type) {
  int checked = 0, ints = 0;
  for (std::uint32_t id = 0; id < 256; ++id) {
    if (!q::can_dequantize(static_cast<q::Type>(id))) continue;
    if (kn::dot_q8_has_int_path(static_cast<q::Type>(id))) ++ints;
    for (std::int64_t k : {256, 2048}) CHECK(run_q8(id, k, nullptr, id * 31u + static_cast<std::uint32_t>(k), &checked));
  }
  std::printf("    dot_q8: %d type/length cases, %d types with an integer form\n", checked, ints);
  CHECK(ints >= 10);
  // planes: dense k 1..3 (fewer read than stored) and block-selective, on the main types
  const std::int64_t k = 2048;
  const std::vector<float> r = noise(k, 77, 0.004f);
  for (q::Type t : {q::Type::q4_k, q::Type::q6_k, q::Type::q8_0, q::Type::iq4_xs, q::Type::q5_k, q::Type::q4_tg2})
    for (int ks : {1, 3}) {
      std::vector<std::uint8_t> pl(static_cast<std::size_t>(q::plane_row_bytes(k, ks)));
      (void)q::plane_fit_row(r.data(), nullptr, k, ks, 4, pl.data());
      for (int ku : {1, ks}) {
        kn::RowPlanes dense{pl.data(), nullptr, ks, ku};
        CHECK(run_q8(static_cast<std::uint32_t>(t), k, &dense, 9u + static_cast<std::uint32_t>(ku), &checked));
      }
      const std::uint8_t mask[1] = {0x5A};
      const std::size_t blk = static_cast<std::size_t>(34 * ks);
      std::vector<std::uint8_t> sp(4 * blk);  // blocks 1, 3, 4, 6
      std::size_t at = 0;
      for (int b = 0; b < 8; ++b)
        if ((mask[0] >> b) & 1) {
          std::memcpy(sp.data() + at, pl.data() + static_cast<std::size_t>(b) * blk, blk);
          at += blk;
        }
      kn::RowPlanes sparse{sp.data(), mask, ks, ks};
      CHECK(run_q8(static_cast<std::uint32_t>(t), k, &sparse, 13u, &checked));
    }
}
