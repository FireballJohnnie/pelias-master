// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// Rung 1 of the chunked Gated DeltaNet (kern/gpu_ops.h deltanet_chunked): the layer window (conv, gates, L2 norms, the
// recurrence, gated norm) on the model's shape (16 key heads, 32 value heads, d_state 128) against an fp64 reference, for
// a 40-token window and a 1,000-token prompt (62 full chunks of 16 and a partial one); the per-token kernel
// (deltanet_recurrent) checked the same way; the two mixed across calls on one state (recurrent for the first 300 tokens,
// chunked for the rest) must agree with the whole run. Both kernels are timed on the 1,000-token prompt (information, not a
// gate). Gate 1e-4 relative L2 per layer window (masterwork 16.2).
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "kern/gpu_ops.h"
#include "pelias_test.h"

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
  explicit Dev(std::size_t count) : p(static_cast<T*>(g::dalloc(count * sizeof(T)))), n(count) {}
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

double rel_l2(const std::vector<float>& a, const std::vector<double>& b) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

double rel_l2f(const std::vector<float>& a, const std::vector<float>& b) {
  std::vector<double> bd(b.begin(), b.end());
  return rel_l2(a, bd);
}

bool all_finite(const std::vector<float>& v) {
  for (const float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

constexpr std::int64_t kNk = 16, kNv = 32, kDs = 128, kTaps = 4, kCh = (2 * kNk + kNv) * kDs, kDin = kNv * kDs;
constexpr float kEps = 1e-6f;

struct Window {
  std::int64_t n;
  std::vector<float> mixed, cw, z, nw, beta, alpha, dt, a;
  Window(std::int64_t nn, std::uint32_t seed)
      : n(nn), mixed(wave(nn * kCh, seed + 1, 1.0f)), cw(wave(kCh * kTaps, seed + 2, 0.5f)), z(wave(nn * kDin, seed + 3, 1.0f)),
        nw(wave(kDs, seed + 4, 1.0f)), beta(wave(nn * kNv, seed + 5, 2.0f)), alpha(wave(nn * kNv, seed + 6, 2.0f)), dt(wave(kNv, seed + 7, 1.0f)),
        a(wave(kNv, seed + 8, 1.0f)) {
    for (float& v : a) v = -std::fabs(v) - 0.1f;  // A < 0: decay below 1
  }
};

// the GPU layer window in parts; mode[p] = 0 recurrent, 1 chunked for part p; seconds = time of the recurrence calls only
std::vector<float> run_gpu(const Window& W, const std::vector<std::int64_t>& parts, const std::vector<int>& mode, double* seconds = nullptr) {
  const std::int64_t n = W.n;
  Dev<float> dm(W.mixed), dcw(W.cw), dz(W.z), dnw(W.nw), db(W.beta), da(W.alpha), ddt(W.dt), dA(W.a);
  Dev<float> conv(static_cast<std::size_t>(n * kCh)), core(static_cast<std::size_t>(n * kDin)), state(static_cast<std::size_t>(kNv * kDs * kDs)),
      tail1(static_cast<std::size_t>((kTaps - 1) * kCh)), tail2(static_cast<std::size_t>((kTaps - 1) * kCh));
  CHECK(g::dzero(state.p, state.n * sizeof(float)));
  CHECK(g::deltanet_gates(db.p, da.p, ddt.p, dA.p, n, kNv));
  std::int64_t t0 = 0;
  double secs = 0.0;
  for (std::size_t pi = 0; pi < parts.size(); ++pi) {
    const std::int64_t m = parts[pi];
    float* tin = pi == 0 ? nullptr : (pi % 2 ? tail1.p : tail2.p);
    float* tout = pi % 2 ? tail2.p : tail1.p;
    CHECK(g::deltanet_conv(dm.p + t0 * kCh, tin, dcw.p, m, kCh, kTaps, conv.p + t0 * kCh, tout));
    CHECK(g::l2_norm(conv.p + t0 * kCh, kDs, 2 * kNk, kDs, kCh, m, kEps));
    CHECK(g::sync());
    const auto c0 = std::chrono::steady_clock::now();
    if (mode[pi] == 0) CHECK(g::deltanet_recurrent(conv.p + t0 * kCh, kCh, db.p + t0 * kNv, da.p + t0 * kNv, state.p, m, kNk, kNv, kDs, core.p + t0 * kDin));
    else CHECK(g::deltanet_chunked(conv.p + t0 * kCh, kCh, db.p + t0 * kNv, da.p + t0 * kNv, state.p, m, kNk, kNv, kDs, core.p + t0 * kDin));
    CHECK(g::sync());
    secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
    t0 += m;
  }
  CHECK(g::gated_norm(core.p, dz.p, dnw.p, n, kNv, kDs, kEps));
  CHECK(g::sync());
  if (seconds) *seconds = secs;
  return core.get();
}

// fp64 reference of the layer window (model::delta_net from the products on), as test_gpu_rung1's
std::vector<double> run_fp64(const Window& W) {
  const std::int64_t n = W.n;
  std::vector<double> conv(static_cast<std::size_t>(n * kCh));
  for (std::int64_t t = 0; t < n; ++t)
    for (std::int64_t c = 0; c < kCh; ++c) {
      double s = 0.0;
      for (std::int64_t kk = 0; kk < kTaps; ++kk) {
        const std::int64_t src = t + kk - (kTaps - 1);
        if (src >= 0) s += static_cast<double>(W.mixed[static_cast<std::size_t>(src * kCh + c)]) * W.cw[static_cast<std::size_t>(c * kTaps + kk)];
      }
      conv[static_cast<std::size_t>(t * kCh + c)] = s / (1.0 + std::exp(-s));
    }
  for (std::int64_t t = 0; t < n; ++t)
    for (std::int64_t h = 0; h < 2 * kNk; ++h) {
      double ss = 0.0;
      for (std::int64_t i = 0; i < kDs; ++i) ss += conv[static_cast<std::size_t>(t * kCh + h * kDs + i)] * conv[static_cast<std::size_t>(t * kCh + h * kDs + i)];
      const double sc = 1.0 / std::sqrt(ss + static_cast<double>(kEps));
      for (std::int64_t i = 0; i < kDs; ++i) conv[static_cast<std::size_t>(t * kCh + h * kDs + i)] *= sc;
    }
  std::vector<double> core(static_cast<std::size_t>(n * kDin));
  std::vector<double> M(static_cast<std::size_t>(kDs * kDs)), delta(static_cast<std::size_t>(kDs));
  for (std::int64_t hv = 0; hv < kNv; ++hv) {
    std::fill(M.begin(), M.end(), 0.0);
    const std::int64_t hk = hv % kNk;
    for (std::int64_t t = 0; t < n; ++t) {
      const double* qv = conv.data() + t * kCh + hk * kDs;
      const double* kv = conv.data() + t * kCh + kNk * kDs + hk * kDs;
      const double* vv = conv.data() + t * kCh + 2 * kNk * kDs + hv * kDs;
      const double b = 1.0 / (1.0 + std::exp(-static_cast<double>(W.beta[static_cast<std::size_t>(t * kNv + hv)])));
      const double x = static_cast<double>(W.alpha[static_cast<std::size_t>(t * kNv + hv)]) + W.dt[static_cast<std::size_t>(hv)];
      const double sp = x > 20.0 ? x : std::log(1.0 + std::exp(x));
      const double decay = std::exp(sp * W.a[static_cast<std::size_t>(hv)]);
      for (double& m : M) m *= decay;
      for (std::int64_t j = 0; j < kDs; ++j) {
        double s = 0.0;
        for (std::int64_t i = 0; i < kDs; ++i) s += M[static_cast<std::size_t>(j * kDs + i)] * kv[i];
        delta[static_cast<std::size_t>(j)] = (vv[j] - s) * b;
      }
      for (std::int64_t j = 0; j < kDs; ++j)
        for (std::int64_t i = 0; i < kDs; ++i) M[static_cast<std::size_t>(j * kDs + i)] += kv[i] * delta[static_cast<std::size_t>(j)];
      for (std::int64_t j = 0; j < kDs; ++j) {
        double s = 0.0;
        for (std::int64_t i = 0; i < kDs; ++i) s += M[static_cast<std::size_t>(j * kDs + i)] * qv[i];
        core[static_cast<std::size_t>(t * kDin + hv * kDs + j)] = s / std::sqrt(static_cast<double>(kDs));
      }
    }
  }
  for (std::int64_t t = 0; t < n; ++t)
    for (std::int64_t h = 0; h < kNv; ++h) {
      double ss = 0.0;
      for (std::int64_t i = 0; i < kDs; ++i) ss += core[static_cast<std::size_t>(t * kDin + h * kDs + i)] * core[static_cast<std::size_t>(t * kDin + h * kDs + i)];
      const double sc = 1.0 / std::sqrt(ss / static_cast<double>(kDs) + static_cast<double>(kEps));
      for (std::int64_t i = 0; i < kDs; ++i) {
        const double zz = W.z[static_cast<std::size_t>(t * kDin + h * kDs + i)];
        double& o = core[static_cast<std::size_t>(t * kDin + h * kDs + i)];
        o = o * sc * W.nw[static_cast<std::size_t>(i)] * (zz / (1.0 + std::exp(-zz)));
      }
    }
  return core;
}

constexpr double kTol = 1e-4;

}  // namespace

TEST(rung1_deltanet_chunked_against_fp64) {
  CHECK(g::compiled());
  if (!g::available()) {
    std::printf("    no GPU: %s\n", g::last_error());
    CHECK(g::available());
    return;
  }
  {  // the 40-token window of the recurrent test: two full chunks and a partial one; 25 + 15 with the state carried
    const Window W(40, 80u);
    const std::vector<double> ref = run_fp64(W);
    const std::vector<float> rec = run_gpu(W, {40}, {0}), chk = run_gpu(W, {40}, {1}), chk2 = run_gpu(W, {25, 15}, {1, 1});
    const double er = rel_l2(rec, ref), ec = rel_l2(chk, ref), e2 = rel_l2(chk2, ref), ecr = rel_l2f(chk, rec);
    CHECK(all_finite(chk));
    CHECK(all_finite(chk2));
    CHECK(er <= kTol);
    CHECK(ec <= kTol);
    CHECK(e2 <= kTol);
    std::printf("    40 tokens: chunked %.3e, chunked 25 + 15 %.3e, recurrent %.3e vs fp64 (tolerance 1e-4); chunked vs recurrent %.3e\n", ec, e2, er, ecr);
  }
  {  // a 1,000-token prompt: 62 full chunks + 8; the two kernels mixed on one state; both timed
    const Window W(1000, 90u);
    const std::vector<double> ref = run_fp64(W);
    double ts_rec = 0.0, ts_chk = 0.0;
    const std::vector<float> rec = run_gpu(W, {1000}, {0}, &ts_rec), chk = run_gpu(W, {1000}, {1}, &ts_chk);
    const std::vector<float> mix = run_gpu(W, {300, 700}, {0, 1});
    const double er = rel_l2(rec, ref), ec = rel_l2(chk, ref), em = rel_l2(mix, ref), ecr = rel_l2f(chk, rec);
    CHECK(all_finite(chk));
    CHECK(all_finite(mix));
    CHECK(er <= kTol);
    CHECK(ec <= kTol);
    CHECK(em <= kTol);
    std::printf("    1,000 tokens: chunked %.3e, recurrent %.3e, recurrent 300 then chunked 700 %.3e vs fp64 (tolerance 1e-4); chunked vs recurrent %.3e\n", ec, er,
                em, ecr);
    // timed again warm (the first calls above include the shared-memory opt-in and first-launch costs)
    double tr2 = 0.0, tc2 = 0.0;
    (void)run_gpu(W, {1000}, {0}, &tr2);
    (void)run_gpu(W, {1000}, {1}, &tc2);
    std::printf("    1,000 tokens x 32 value heads, the recurrence alone: recurrent %.3f ms (warm %.3f), chunked %.3f ms (warm %.3f); warm ratio %.2fx\n",
                ts_rec * 1e3, tr2 * 1e3, ts_chk * 1e3, tc2 * 1e3, tc2 > 0.0 ? tr2 / tc2 : 0.0);
  }
}

TEST(rung1_deltanet_chunked_refuses_bad_shapes) {
  if (!g::available()) return;
  float dummy = 0.0f;
  CHECK(!g::deltanet_chunked(&dummy, 64, &dummy, &dummy, &dummy, 1, 1, 1, 48, &dummy));    // d_state not a multiple of 32
  CHECK(!g::deltanet_chunked(&dummy, 2048, &dummy, &dummy, &dummy, 1, 1, 1, 512, &dummy));  // d_state above 256
  CHECK(!g::deltanet_chunked(&dummy, 64, &dummy, &dummy, &dummy, 1, 2, 3, 32, &dummy));     // value heads not a multiple of key heads
  CHECK(g::deltanet_chunked(&dummy, 96, &dummy, &dummy, &dummy, 0, 1, 1, 32, &dummy));      // zero tokens: nothing to do
  std::printf("    bad shapes refused; zero tokens accepted as no work\n");
}
