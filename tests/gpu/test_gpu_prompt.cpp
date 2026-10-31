// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// Rung 1 of the int8 prompt path (kern/gpu_prompt.h, the locked MoE slice): the dense GEMM and the whole MoE slice
// against fp64 references of the same rounded inputs (masterwork 16.2: products against fp64 of the same rounded
// inputs, gate 1e-6 relative L2). The slice: real per-32 activations (act_q8x32 + act_digits), a skewed routing (one
// expert with more than 64 rows: body + partial tile; experts with 1-8 rows: the warp-vector path; experts with none),
// the work list built on the GPU, gate / up checked row by row against fp64 through the device's row map, the down
// projection + combine against fp64 of the device's own rounded silu x up, for the int32 (deterministic, repeat bit for
// bit, no overflow), int64 and float combines. Each for Q4_K, Q5_K, Q6_K and Q4_TG2 weights (2026-10-30 23:3x; Q4_TG2: int8 x 3 tiles,
// exact fp32 table on the warp-vector path); plus a dense N = K = 2048 case (2048 rows; 130 under PELIAS_GPU_QUICK).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../unit/gpu_test_rows.h"
#include "kern/gpu_ops.h"
#include "kern/gpu_prompt.h"
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

template <class T>
struct Dev {
  T* p = nullptr;
  std::size_t n = 0;
  explicit Dev(std::size_t count) : p(static_cast<T*>(g::dalloc(count * sizeof(T)))), n(count) {
    if (p) g::dzero(p, count * sizeof(T));
  }
  explicit Dev(const std::vector<T>& v) : Dev(v.size()) {
    if (p && !v.empty()) g::to_device(p, v.data(), v.size() * sizeof(T));
  }
  ~Dev() { g::dfree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  std::vector<T> get() const {
    std::vector<T> v(n);
    if (p && n) g::to_host(v.data(), p, n * sizeof(T));
    return v;
  }
};

template <class T>
std::vector<T> pull(const T* p, std::size_t n) {
  std::vector<T> v(n);
  if (n) g::to_host(v.data(), p, n * sizeof(T));
  return v;
}

// weight rows of type ty (Q4_K or Q4_TG2): the GGUF bytes and their exact floats
struct QW {
  std::int64_t rows, k;
  std::vector<std::uint8_t> bytes;
  std::vector<float> deq;
  QW(q::Type ty, std::int64_t r, std::int64_t kk, std::uint32_t seed) : rows(r), k(kk) {
    const auto id = static_cast<std::uint32_t>(ty);
    const std::uint64_t rb = q::row_bytes(id, static_cast<std::uint64_t>(k));
    bytes.resize(static_cast<std::size_t>(rb * static_cast<std::uint64_t>(rows)));
    deq.resize(static_cast<std::size_t>(rows * k));
    for (std::int64_t i = 0; i < rows; ++i) {
      const std::uint32_t rs = seed + static_cast<std::uint32_t>(i) * 7919u;
      const std::vector<std::uint8_t> row = pelias::test_gpu::make_row(ty, wave(k, rs, 0.05f), rs);
      std::memcpy(bytes.data() + static_cast<std::size_t>(i) * rb, row.data(), static_cast<std::size_t>(rb));
      (void)q::dequantize_row(ty, row.data(), deq.data() + i * k, k);
    }
  }
};

// per-32 rounded activations of `rows` rows of k on the device (readable to rows rounded up to 64), and their host floats
struct Act {
  std::int64_t rows, k, pad;
  Dev<std::int8_t> qv;
  Dev<float> sc;
  Dev<int> sum;
  Dev<float> dig;
  std::vector<double> eff;
  Act(std::int64_t r, std::int64_t kk, std::uint32_t seed)
      : rows(r), k(kk), pad((r + 63) / 64 * 64), qv(static_cast<std::size_t>(pad * kk)), sc(static_cast<std::size_t>(pad * kk / 32)),
        sum(static_cast<std::size_t>(pad * kk / 32)), dig(static_cast<std::size_t>(pad * kk / 32)) {
    Dev<float> x(wave(rows * k, seed, 2.0f)), xr(static_cast<std::size_t>(rows * k));
    CHECK(g::act_q8x32(x.p, k, rows, k, false, xr.p, k, qv.p, sc.p, sum.p));
    CHECK(g::act_digits(sc.p, sum.p, rows, k, dig.p));
    CHECK(g::sync());
    const std::vector<std::int8_t> hq = qv.get();
    const std::vector<float> hs = sc.get();
    eff.resize(static_cast<std::size_t>(rows * k));
    for (std::int64_t i = 0; i < rows * k; ++i)
      eff[static_cast<std::size_t>(i)] = static_cast<double>(hq[static_cast<std::size_t>(i)]) * static_cast<double>(hs[static_cast<std::size_t>(i / 32)]);
  }
};

double dot_row(const std::vector<float>& w, std::int64_t wrow, std::int64_t k, const double* x) {
  double s = 0.0;
  for (std::int64_t i = 0; i < k; ++i) s += static_cast<double>(w[static_cast<std::size_t>(wrow * k + i)]) * x[i];
  return s;
}

constexpr double kTol = 1e-6;

}  // namespace

static void dense_case(q::Type ty, const char* nm) {
  struct Shape {
    std::int64_t n, k;
  };
  for (const Shape sh : {Shape{256, 2048}, Shape{512, 512}}) {
    const QW w(ty, sh.n, sh.k, 4100u + static_cast<std::uint32_t>(sh.k));
    Dev<std::uint8_t> dw(w.bytes);
    g::PromptWeights pw;
    CHECK(g::prompt_prepare(static_cast<std::uint32_t>(ty), dw.p, sh.n, sh.k, &pw));
    for (const std::int64_t rows : {std::int64_t{1}, std::int64_t{37}, std::int64_t{130}}) {
      const Act a(rows, sh.k, 4200u + static_cast<std::uint32_t>(rows));
      Dev<float> y(static_cast<std::size_t>(rows * sh.n));
      CHECK(g::prompt_gemm(pw, sh.n, a.qv.p, a.sc.p, a.dig.p, rows, y.p));
      CHECK(g::sync());
      const std::vector<float> hy = y.get();
      double num = 0.0, den = 0.0;
      bool finite = true;
      for (std::int64_t r = 0; r < rows; ++r)
        for (std::int64_t c = 0; c < sh.n; ++c) {
          const double ref = dot_row(w.deq, c, sh.k, a.eff.data() + r * sh.k), v = hy[static_cast<std::size_t>(r * sh.n + c)];
          finite = finite && std::isfinite(v);
          num += (v - ref) * (v - ref);
          den += ref * ref;
        }
      const double e = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
      CHECK(finite);
      CHECK(e <= kTol);
      std::printf("    %s dense %4lld x %4lld, %3lld rows: relative L2 vs fp64 %.3e (tolerance 1e-6)\n", nm, static_cast<long long>(sh.n), static_cast<long long>(sh.k),
                  static_cast<long long>(rows), e);
    }
    g::prompt_release(&pw);
  }
  // N = K = 2048 (the TG2 probe's open dense case, 2026-10-30 22:33: 0.507 there): 2048 rows (130 under
  // PELIAS_GPU_QUICK); fp64 on the first 16 and last 16 rows, every column
  {
    const bool quick = std::getenv("PELIAS_GPU_QUICK") != nullptr;
    const std::int64_t n = 2048, k = 2048, rows = quick ? 130 : 2048;
    const QW w(ty, n, k, 4300u);
    Dev<std::uint8_t> dw(w.bytes);
    g::PromptWeights pw;
    CHECK(g::prompt_prepare(static_cast<std::uint32_t>(ty), dw.p, n, k, &pw));
    const Act a(rows, k, 4400u);
    Dev<float> y(static_cast<std::size_t>(rows * n));
    CHECK(g::prompt_gemm(pw, n, a.qv.p, a.sc.p, a.dig.p, rows, y.p));
    CHECK(g::sync());
    const std::vector<float> hy = y.get();
    double num = 0.0, den = 0.0, worst = 0.0;
    std::int64_t worst_r = -1, worst_c = -1;
    bool finite = true;
    for (std::int64_t r = 0; r < rows; ++r) {
      if (r >= 16 && r < rows - 16) continue;
      for (std::int64_t c = 0; c < n; ++c) {
        const double ref = dot_row(w.deq, c, k, a.eff.data() + r * k), v = hy[static_cast<std::size_t>(r * n + c)];
        finite = finite && std::isfinite(v);
        num += (v - ref) * (v - ref);
        den += ref * ref;
        if (std::fabs(v - ref) > worst) {
          worst = std::fabs(v - ref);
          worst_r = r;
          worst_c = c;
        }
      }
    }
    const double e = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    CHECK(finite);
    CHECK(e <= kTol);
    std::printf("    %s dense %4lld x %4lld, %4lld rows (32 checked): relative L2 vs fp64 %.3e (tolerance 1e-6); largest |diff| %.3e at row %lld col %lld\n",
                nm, static_cast<long long>(n), static_cast<long long>(k), static_cast<long long>(rows), e, worst, static_cast<long long>(worst_r),
                static_cast<long long>(worst_c));
    g::prompt_release(&pw);
  }
}

TEST(rung1_prompt_gemm_against_fp64) {
  CHECK(g::compiled());
  if (!g::available()) {
    std::printf("    no GPU: %s\n", g::last_error());
    CHECK(g::available());
    return;
  }
  dense_case(q::Type::q4_k, "Q4_K  ");
  dense_case(q::Type::q5_k, "Q5_K  ");
  dense_case(q::Type::q6_k, "Q6_K  ");
  dense_case(q::Type::q4_tg2, "Q4_TG2");
  // refusals: not Q4_K / Q4_TG2, k not a multiple of 256, rows not a multiple of 128
  g::PromptWeights bad;
  Dev<std::uint8_t> junk(4096);
  CHECK(!g::prompt_prepare(static_cast<std::uint32_t>(q::Type::iq4_xs), junk.p, 128, 256, &bad));
  CHECK(!g::prompt_prepare(static_cast<std::uint32_t>(q::Type::q4_k), junk.p, 128, 320, &bad));
  CHECK(!g::prompt_prepare(static_cast<std::uint32_t>(q::Type::q4_k), junk.p, 100, 256, &bad));
  std::printf("    other types, widths and row counts refused\n");
}

static void moe_case(q::Type ty, const char* nm) {
  // a small expert layer with the model's structure: 32 experts, hidden 512, ffn 256, 8 picks per token
  const int E = 32, NU = 8;
  const std::int64_t H = 512, F = 256;
  const bool quick = std::getenv("PELIAS_GPU_QUICK") != nullptr;
  const std::int64_t T = quick ? 40 : 150;
  // routing (distinct picks per token): heavy tokens pick experts 0-7 (100 rows each: a full tile and a partial one);
  // mid tokens pick 8-15 (30 rows: one partial tile); light tokens pick 4 heavy experts and 4 of the light pool 16-27
  // (6-7 rows each: the warp-vector path); experts 28-31 get none
  const std::int64_t heavy_end = quick ? 20 : 100, mid_end = quick ? 30 : 130;
  std::vector<std::int32_t> sel(static_cast<std::size_t>(T * NU));
  std::vector<float> wsel(static_cast<std::size_t>(T * NU));
  std::vector<int> cnt(E, 0);
  for (std::int64_t t = 0; t < T; ++t) {
    float tot = 0.0f;
    for (int k = 0; k < NU; ++k) {
      int e;
      if (t < heavy_end) e = k;
      else if (t < mid_end) e = 8 + k;
      else e = k < 4 ? k : 16 + static_cast<int>(((t - mid_end) * 4 + (k - 4)) % 12);
      sel[static_cast<std::size_t>(t * NU + k)] = e;
      wsel[static_cast<std::size_t>(t * NU + k)] = 0.3f / static_cast<float>(k + 1);
      tot += wsel[static_cast<std::size_t>(t * NU + k)];
      ++cnt[static_cast<std::size_t>(e)];
    }
    for (int k = 0; k < NU; ++k) wsel[static_cast<std::size_t>(t * NU + k)] /= tot;
  }
  int n_tile = 0, n_vec = 0, n_none = 0;
  for (int e = 0; e < E; ++e) {
    if (cnt[static_cast<std::size_t>(e)] > 8) ++n_tile;
    else if (cnt[static_cast<std::size_t>(e)]) ++n_vec;
    else ++n_none;
  }
  const QW wg(ty, E * F, H, 5100u), wu(ty, E * F, H, 5200u), wd(ty, E * H, F, 5300u);
  Dev<std::uint8_t> dg(wg.bytes), du(wu.bytes), dd(wd.bytes);
  g::PromptWeights pg, pu, pd;
  CHECK(g::prompt_prepare(static_cast<std::uint32_t>(ty), dg.p, E * F, H, &pg));
  CHECK(g::prompt_prepare(static_cast<std::uint32_t>(ty), du.p, E * F, H, &pu));
  CHECK(g::prompt_prepare(static_cast<std::uint32_t>(ty), dd.p, E * H, F, &pd));
  const Act a(T, H, 5400u);
  Dev<std::int32_t> dsel(sel);
  Dev<float> dw(wsel), y(static_cast<std::size_t>(T * H));
  g::MoeScratch m;
  CHECK(g::moe_scratch_alloc(T, NU, E, H, F, &m));
  // the float combine first (no scale needed): the device's row map, gate / up rows and rounded silu x up come from it
  CHECK(g::moe_prompt(pg, pu, pd, a.qv.p, a.sc.p, a.dig.p, T, dsel.p, dw.p, m, g::MoeCombine::kFloat, 0, y.p));
  CHECK(g::sync());
  const std::size_t np = static_cast<std::size_t>(T * NU);
  const std::vector<int> prow = pull(m.prow, np), meta = pull(m.meta, std::size_t{4});
  const std::vector<float> hg = pull(m.hg, np * static_cast<std::size_t>(F)), hu = pull(m.hu, np * static_cast<std::size_t>(F));
  const std::vector<std::int8_t> q2 = pull(m.q2, np * static_cast<std::size_t>(F));
  const std::vector<float> sc2 = pull(m.sc2, np * static_cast<std::size_t>(F / 32));
  int exp_t = 0, exp_v = 0, active = 0;
  for (int e = 0; e < E; ++e) {
    if (cnt[static_cast<std::size_t>(e)] > 8) exp_t += (cnt[static_cast<std::size_t>(e)] + 63) / 64;
    else if (cnt[static_cast<std::size_t>(e)]) ++exp_v;
    if (cnt[static_cast<std::size_t>(e)]) ++active;
  }
  CHECK(meta[0] == exp_t && meta[1] == exp_v && meta[2] == active && meta[3] == static_cast<int>(np));
  std::vector<int> seen(np, 0);
  bool perm = true;
  for (std::size_t i = 0; i < np; ++i) perm = perm && prow[i] >= 0 && static_cast<std::size_t>(prow[i]) < np && seen[static_cast<std::size_t>(prow[i])]++ == 0;
  CHECK(perm);
  // gate / up vs fp64; down per pick vs fp64 of the device's own rounded silu x up; the combined y reference
  double ng = 0.0, dgd = 0.0, maxc = 0.0;
  std::vector<double> yref(static_cast<std::size_t>(T * H), 0.0);
  for (std::int64_t t = 0; t < T && perm; ++t)
    for (int k = 0; k < NU; ++k) {
      const std::size_t i = static_cast<std::size_t>(t * NU + k);
      const int e = sel[i];
      const std::size_t p = static_cast<std::size_t>(prow[i]);
      for (std::int64_t c = 0; c < F; ++c) {
        const double rg = dot_row(wg.deq, e * F + c, H, a.eff.data() + t * H), ru = dot_row(wu.deq, e * F + c, H, a.eff.data() + t * H);
        const double vg = hg[p * static_cast<std::size_t>(F) + static_cast<std::size_t>(c)], vu = hu[p * static_cast<std::size_t>(F) + static_cast<std::size_t>(c)];
        ng += (vg - rg) * (vg - rg) + (vu - ru) * (vu - ru);
        dgd += rg * rg + ru * ru;
      }
      std::vector<double> x2(static_cast<std::size_t>(F));
      for (std::int64_t j = 0; j < F; ++j)
        x2[static_cast<std::size_t>(j)] = static_cast<double>(q2[p * static_cast<std::size_t>(F) + static_cast<std::size_t>(j)]) *
                                          static_cast<double>(sc2[p * static_cast<std::size_t>(F / 32) + static_cast<std::size_t>(j / 32)]);
      for (std::int64_t c = 0; c < H; ++c) {
        const double o = dot_row(wd.deq, e * H + c, F, x2.data()) * static_cast<double>(wsel[i]);
        maxc = std::max(maxc, std::fabs(o));
        yref[static_cast<std::size_t>(t * H + c)] += o;
      }
    }
  const double e_gu = dgd > 0 ? std::sqrt(ng / dgd) : 0.0;
  CHECK(e_gu <= kTol);
  auto rel_y = [&](const std::vector<float>& v) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
      const double d = static_cast<double>(v[i]) - yref[i];
      num += d * d;
      den += yref[i] * yref[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  };
  const double e_f = rel_y(y.get());
  CHECK(e_f <= kTol);
  // the int32 combine: scale 2^k with 8 addends below 2^31, from the largest single contribution
  const int k32 = maxc > 0 ? static_cast<int>(std::floor(std::log2(268435455.0 / maxc))) : 20;
  CHECK(g::moe_prompt(pg, pu, pd, a.qv.p, a.sc.p, a.dig.p, T, dsel.p, dw.p, m, g::MoeCombine::kFixed32, k32, y.p));
  CHECK(g::sync());
  const std::vector<float> y32 = y.get();
  unsigned ovf = 0;
  CHECK(g::to_host(&ovf, m.ovf, sizeof ovf));
  CHECK(g::moe_prompt(pg, pu, pd, a.qv.p, a.sc.p, a.dig.p, T, dsel.p, dw.p, m, g::MoeCombine::kFixed32, k32, y.p));
  CHECK(g::sync());
  const std::vector<float> y32b = y.get();
  const bool same = std::memcmp(y32.data(), y32b.data(), y32.size() * sizeof(float)) == 0;
  const double e_32 = rel_y(y32);
  CHECK(same);
  CHECK(ovf == 0);
  CHECK(e_32 <= kTol);
  const int k64 = maxc > 0 ? std::min(60, static_cast<int>(std::floor(std::log2(5.7e17 / maxc)))) : 40;
  CHECK(g::moe_prompt(pg, pu, pd, a.qv.p, a.sc.p, a.dig.p, T, dsel.p, dw.p, m, g::MoeCombine::kFixed64, k64, y.p));
  CHECK(g::sync());
  const double e_64 = rel_y(y.get());
  CHECK(e_64 <= kTol);
  std::printf("    %s %lld tokens x %d picks over %d experts (%d on tiles, %d on the warp-vector path, %d empty); GPU list: %d tile + %d vector units, rows permuted %s\n",
              nm, static_cast<long long>(T), NU, E, n_tile, n_vec, n_none, meta[0], meta[1], perm ? "once each" : "WRONG");
  std::printf("    %s gate / up vs fp64 %.3e; y vs fp64: float combine %.3e, int32 (2^%d) %.3e repeat %s overflows %u, int64 (2^%d) %.3e (tolerance 1e-6)\n", nm, e_gu, e_f,
              k32, e_32, same ? "bit for bit" : "DIFFERENT", ovf, k64, e_64);
  g::moe_scratch_free(&m);
  g::prompt_release(&pg);
  g::prompt_release(&pu);
  g::prompt_release(&pd);
}

TEST(rung1_moe_prompt_against_fp64) {
  if (!g::available()) {
    CHECK(g::available());
    return;
  }
  moe_case(q::Type::q4_k, "Q4_K  ");
  moe_case(q::Type::q5_k, "Q5_K  ");
  moe_case(q::Type::q6_k, "Q6_K  ");
  moe_case(q::Type::q4_tg2, "Q4_TG2");
}

TEST(rung1_moe_prompt_refuses_bad_shapes) {
  if (!g::available()) return;
  g::MoeScratch m;
  CHECK(!g::moe_scratch_alloc(16, 8, 300, 512, 256, &m));  // more than 256 experts
  CHECK(!g::moe_scratch_alloc(16, 8, 32, 500, 256, &m));   // hidden not a multiple of 256
  CHECK(g::moe_scratch_alloc(16, 8, 32, 512, 256, &m));
  g::PromptWeights none;
  Dev<std::int8_t> qv(64 * 512);
  Dev<float> sc(64 * 16), y(16 * 512), wv(16 * 8);
  Dev<float> dg(64 * 16);
  Dev<std::int32_t> sel(16 * 8);
  CHECK(!g::moe_prompt(none, none, none, qv.p, sc.p, dg.p, 16, sel.p, wv.p, m, g::MoeCombine::kFixed32, 10, y.p));  // stacks do not match
  CHECK(!g::moe_prompt(none, none, none, qv.p, sc.p, dg.p, 17, sel.p, wv.p, m, g::MoeCombine::kFixed32, 10, y.p));  // more tokens than the scratch
  g::moe_scratch_free(&m);
  std::printf("    bad shapes, mismatched stacks and too many tokens refused\n");
}

// Speed by weight type (2026-10-30 23:5x, author 23:45 "I would rather see the speed right now"): prompt_gemm at the
// model's shapes, 2048 rows, mean of 20 calls after 3 warm-ups (host clock around a device sync). Skipped under
// PELIAS_GPU_QUICK (the sanitizers).
TEST(rung1_prompt_speed_by_type) {
  if (!g::available() || std::getenv("PELIAS_GPU_QUICK") != nullptr) return;
  struct Shape {
    std::int64_t n, k;
    const char* name;
  };
  const std::int64_t rows = 2048;
  for (const Shape sh : {Shape{512, 2048, "expert gate/up shape 2048 -> 512"}, Shape{2048, 512, "expert down shape 512 -> 2048"},
                         Shape{2048, 2048, "dense 2048 -> 2048"}}) {
    const Act a(rows, sh.k, 4500u);
    Dev<float> y(static_cast<std::size_t>(rows * sh.n));
    double base = 0.0;
    for (const q::Type ty : {q::Type::q4_k, q::Type::q5_k, q::Type::q6_k, q::Type::q4_tg2}) {
      const QW w(ty, sh.n, sh.k, 4600u);
      Dev<std::uint8_t> dw(w.bytes);
      g::PromptWeights pw;
      CHECK(g::prompt_prepare(static_cast<std::uint32_t>(ty), dw.p, sh.n, sh.k, &pw));
      for (int i = 0; i < 3; ++i) CHECK(g::prompt_gemm(pw, sh.n, a.qv.p, a.sc.p, a.dig.p, rows, y.p));
      CHECK(g::sync());
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 20; ++i) CHECK(g::prompt_gemm(pw, sh.n, a.qv.p, a.sc.p, a.dig.p, rows, y.p));
      CHECK(g::sync());
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20.0;
      if (ty == q::Type::q4_k) base = ms;
      const double tops = 2.0 * static_cast<double>(rows * sh.n * sh.k) / (ms * 1e-3) / 1e12;
      std::printf("    %-34s %-6s %8.4f ms %6.1f TOPS  %.3fx the Q4_K time\n", sh.name, q::type_info(static_cast<std::uint32_t>(ty)).name, ms, tops, base > 0 ? ms / base : 1.0);
      g::prompt_release(&pw);
    }
  }
}
