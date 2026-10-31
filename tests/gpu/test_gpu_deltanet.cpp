// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// Rung 1 of the chunked Gated DeltaNet (kern/gpu_ops.h deltanet_chunked): the layer window (conv, gates, L2 norms, the
// recurrence, gated norm) on the model's shape (16 key heads, 32 value heads, d_state 128) against an fp64 reference, for
// a 40-token window and a 1,000-token prompt (62 full chunks of 16 and a partial one); the per-token kernel
// (deltanet_recurrent) checked the same way; the two mixed across calls on one state (recurrent for the first 300 tokens,
// chunked for the rest) must agree with the whole run. Both kernels are timed on the 1,000-token prompt (information, not a
// gate). Gate 1e-4 relative L2 per layer window (masterwork 16.2).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// the recurrence kernels: 0 per token (shared memory), 1 chunked, 2 per token with the state in registers, 3 chunked with
// the head split over row blocks (16-token chunks), 4 the same with 32-token chunks
const char* const kModeName[5] = {"recurrent", "chunked", "recurrent, registers", "chunked rows, C16", "chunked rows, C32"};
bool recur(int mode, const float* conv, const float* beta, const float* alpha, float* state, std::int64_t m, float* core) {
  switch (mode) {
    case 0: return g::deltanet_recurrent(conv, kCh, beta, alpha, state, m, kNk, kNv, kDs, core);
    case 1: return g::deltanet_chunked(conv, kCh, beta, alpha, state, m, kNk, kNv, kDs, core);
    case 2: return g::deltanet_recurrent_reg(conv, kCh, beta, alpha, state, m, kNk, kNv, kDs, core);
    case 3: return g::deltanet_chunked_rows(conv, kCh, beta, alpha, state, m, kNk, kNv, kDs, 16, core);
    default: return g::deltanet_chunked_rows(conv, kCh, beta, alpha, state, m, kNk, kNv, kDs, 32, core);
  }
}

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
    CHECK(recur(mode[pi], conv.p + t0 * kCh, db.p + t0 * kNv, da.p + t0 * kNv, state.p, m, core.p + t0 * kDin));
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

// decode timing: n_calls single-token calls back to back on one state (conv prepared once), synchronised at the end;
// returns microseconds per call (what one DeltaNet layer's recurrence costs per decoded token)
double decode_us(const Window& W, int mode, std::int64_t n_calls) {
  Dev<float> dm(W.mixed), dcw(W.cw), db(W.beta), da(W.alpha), ddt(W.dt), dA(W.a);
  Dev<float> conv(static_cast<std::size_t>(W.n * kCh)), core(static_cast<std::size_t>(W.n * kDin)), state(static_cast<std::size_t>(kNv * kDs * kDs));
  CHECK(g::dzero(state.p, state.n * sizeof(float)));
  CHECK(g::deltanet_gates(db.p, da.p, ddt.p, dA.p, W.n, kNv));
  CHECK(g::deltanet_conv(dm.p, nullptr, dcw.p, W.n, kCh, kTaps, conv.p, nullptr));
  CHECK(g::l2_norm(conv.p, kDs, 2 * kNk, kDs, kCh, W.n, kEps));
  CHECK(recur(mode, conv.p, db.p, da.p, state.p, 1, core.p));  // first call: opt-in and first-launch costs outside the timing
  CHECK(g::sync());
  const auto c0 = std::chrono::steady_clock::now();
  for (std::int64_t t = 1; t <= n_calls; ++t) CHECK(recur(mode, conv.p + t * kCh, db.p + t * kNv, da.p + t * kNv, state.p, 1, core.p + t * kDin));
  CHECK(g::sync());
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count() * 1e6 / static_cast<double>(n_calls);
}

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

TEST(rung1_deltanet_variants_side_by_side) {
  if (!g::available()) {
    CHECK(g::available());
    return;
  }
  // accuracy: every variant on the 40-token window (incl. 25 + 15) and the long prompt, against fp64. Under the sanitizers
  // (PELIAS_GPU_QUICK set by the PC script) the long prompt is 120 tokens (7 chunks of 16 + 8) and decode 16 calls, so the
  // tools finish; the timings that count come from the plain run.
  const bool quick = std::getenv("PELIAS_GPU_QUICK") != nullptr;
  const std::int64_t nlong = quick ? 120 : 1000, ndec = quick ? 16 : 512;
  const Window W40(40, 80u), W1k(nlong, 90u);
  const std::vector<double> r40 = run_fp64(W40), r1k = run_fp64(W1k);
  double worst = 0.0;
  for (int mode = 0; mode < 5; ++mode) {
    const std::vector<float> a = run_gpu(W40, {40}, {mode}), b = run_gpu(W40, {25, 15}, {mode, mode});
    double t1 = 0.0;
    const std::vector<float> c = run_gpu(W1k, {nlong}, {mode}, &t1);
    double t2 = 0.0;
    (void)run_gpu(W1k, {nlong}, {mode}, &t2);  // warm
    const double ea = rel_l2(a, r40), eb = rel_l2(b, r40), ec = rel_l2(c, r1k);
    worst = std::max(worst, std::max(ea, std::max(eb, ec)));
    CHECK(all_finite(c));
    CHECK(ea <= kTol);
    CHECK(eb <= kTol);
    CHECK(ec <= kTol);
    const double dus = decode_us(W1k, mode, ndec);
    std::printf("    %-22s vs fp64: 40 tok %.3e, 25 + 15 %.3e, %lld tok %.3e | %lld-token prompt %.3f ms (warm) | decode %.2f us / token\n",
                kModeName[mode], ea, eb, static_cast<long long>(nlong), ec, static_cast<long long>(nlong), t2 * 1e3, dus);
  }
  // all five on one state, in turn (200 tokens each)
  const std::int64_t q5 = nlong / 5;
  const std::vector<float> mix = run_gpu(W1k, {q5, q5, q5, q5, nlong - 4 * q5}, {0, 1, 2, 3, 4});
  const double em = rel_l2(mix, r1k);
  CHECK(em <= kTol);
  std::printf("    the five kernels in turn on one state (%lld tokens each): %.3e vs fp64; worst of all %.3e (tolerance 1e-4)\n", static_cast<long long>(q5), em, std::max(worst, em));
}

// Timing spread (C2 follow-up, 2026-10-30 v7): the five kernels timed in 5 rounds, interleaved (round r runs mode 0..4 in
// turn) so a clock or power state change shows up across all of them rather than in one. Information, not a gate; skipped
// under PELIAS_GPU_QUICK (the sanitizers). With PELIAS_GPU_POWER_SECS=s (s > 0) each kernel then runs back to back for about
// s seconds as a prompt loop and as a decode loop; the loop's wall-clock start / end (Unix ms) are printed so a power log
// (nvidia-smi / HWiNFO) can be sliced by them, and the loop's own average per call is a long-run timing sample.
namespace {

long long unix_ms() {
  return static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

// m tokens per call (m = 1: decode, walking the window's tokens in a cycle); calls until secs have passed (checked every
// batch, synchronised); returns calls made and the time per call
double busy_loop(const Window& W, int mode, std::int64_t m, double secs, long long* calls, long long* t0_ms, long long* t1_ms) {
  Dev<float> dm(W.mixed), dcw(W.cw), db(W.beta), da(W.alpha), ddt(W.dt), dA(W.a);
  Dev<float> conv(static_cast<std::size_t>(W.n * kCh)), core(static_cast<std::size_t>(W.n * kDin)), state(static_cast<std::size_t>(kNv * kDs * kDs));
  CHECK(g::dzero(state.p, state.n * sizeof(float)));
  CHECK(g::deltanet_gates(db.p, da.p, ddt.p, dA.p, W.n, kNv));
  CHECK(g::deltanet_conv(dm.p, nullptr, dcw.p, W.n, kCh, kTaps, conv.p, nullptr));
  CHECK(g::l2_norm(conv.p, kDs, 2 * kNk, kDs, kCh, W.n, kEps));
  CHECK(recur(mode, conv.p, db.p, da.p, state.p, m, core.p));
  CHECK(g::sync());
  const std::int64_t batch = m == 1 ? 256 : 8;
  long long n = 0;
  std::int64_t t = 0;
  *t0_ms = unix_ms();
  const auto c0 = std::chrono::steady_clock::now();
  double el = 0.0;
  while (el < secs) {
    for (std::int64_t i = 0; i < batch; ++i) {
      if (m == 1) {
        t = (t + 1) % W.n;
        CHECK(recur(mode, conv.p + t * kCh, db.p + t * kNv, da.p + t * kNv, state.p, 1, core.p + t * kDin));
      } else {
        CHECK(recur(mode, conv.p, db.p, da.p, state.p, m, core.p));
      }
    }
    CHECK(g::sync());
    n += batch;
    el = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
  }
  *t1_ms = unix_ms();
  *calls = n;
  return el / static_cast<double>(n);
}

}  // namespace

TEST(rung1_deltanet_timing_spread) {
  if (!g::available()) {
    CHECK(g::available());
    return;
  }
  if (std::getenv("PELIAS_GPU_QUICK") != nullptr) {
    std::printf("    skipped under PELIAS_GPU_QUICK (timing only)\n");
    return;
  }
  constexpr int kRounds = 5;
  const std::int64_t nlong = 1000, ndec = 512;
  const Window W1k(nlong, 90u);
  double pr[5][kRounds] = {}, de[5][kRounds] = {};
  for (int r = 0; r < kRounds; ++r) {
    for (int mode = 0; mode < 5; ++mode) {
      double t1 = 0.0;
      (void)run_gpu(W1k, {nlong}, {mode}, &t1);  // cold call of this round
      double t2 = 0.0;
      (void)run_gpu(W1k, {nlong}, {mode}, &t2);  // warm
      pr[mode][r] = t2 * 1e3;
      de[mode][r] = decode_us(W1k, mode, ndec);
    }
    std::printf("    round %d at %lld ms:", r + 1, unix_ms());
    for (int mode = 0; mode < 5; ++mode) std::printf(" [%d] %.3f ms / %.2f us", mode, pr[mode][r], de[mode][r]);
    std::printf("\n");
  }
  for (int mode = 0; mode < 5; ++mode) {
    double a[kRounds], b[kRounds];
    for (int r = 0; r < kRounds; ++r) {
      a[r] = pr[mode][r];
      b[r] = de[mode][r];
    }
    std::sort(a, a + kRounds);
    std::sort(b, b + kRounds);
    std::printf("    %-22s prompt (1,000 tok, warm) min %.3f median %.3f max %.3f ms | decode min %.2f median %.2f max %.2f us (max / min %.2fx)\n",
                kModeName[mode], a[0], a[kRounds / 2], a[kRounds - 1], b[0], b[kRounds / 2], b[kRounds - 1], b[0] > 0.0 ? b[kRounds - 1] / b[0] : 0.0);
  }
  const char* ps = std::getenv("PELIAS_GPU_POWER_SECS");
  const double secs = ps ? std::atof(ps) : 0.0;
  if (secs <= 0.0) return;
  for (int mode = 0; mode < 5; ++mode) {
    for (int dec = 0; dec < 2; ++dec) {
      long long calls = 0, s0 = 0, s1 = 0;
      const double per = busy_loop(W1k, mode, dec ? 1 : nlong, secs, &calls, &s0, &s1);
      std::printf("    power loop %-22s %-6s start %lld end %lld ms | %lld calls, %.3f %s per call\n", kModeName[mode], dec ? "decode" : "prompt", s0, s1, calls,
                  dec ? per * 1e6 : per * 1e3, dec ? "us" : "ms");
    }
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

// GP3-14 (MTP rollback, author 2026-10-31 03:1x): deltanet_recurrent_reg_ckpt gives core and the final state bit for bit
// equal to deltanet_recurrent_reg, and its checkpoint t equals the state after the batch's first t + 1 tokens run by
// deltanet_recurrent_reg alone (bit for bit: the same expressions in the same order).
TEST(rung1_deltanet_checkpoints_equal_a_shorter_batch) {
  if (!g::available()) {
    CHECK(g::available());
    return;
  }
  const std::int64_t n = 5, ss = kNv * kDs * kDs;
  const Window W(n + 7, 33u);
  Dev<float> dm(W.mixed), dcw(W.cw), db(W.beta), da(W.alpha), ddt(W.dt), dA(W.a);
  Dev<float> conv(static_cast<std::size_t>(W.n * kCh));
  CHECK(g::deltanet_gates(db.p, da.p, ddt.p, dA.p, W.n, kNv));
  CHECK(g::deltanet_conv(dm.p, nullptr, dcw.p, W.n, kCh, kTaps, conv.p, nullptr));
  CHECK(g::l2_norm(conv.p, kDs, 2 * kNk, kDs, kCh, W.n, kEps));
  // a non-zero start state: the first 7 tokens through the plain kernel
  Dev<float> s0(static_cast<std::size_t>(ss)), core0(static_cast<std::size_t>(7 * kDin));
  CHECK(g::dzero(s0.p, s0.n * sizeof(float)));
  CHECK(g::deltanet_recurrent_reg(conv.p, kCh, db.p, da.p, s0.p, 7, kNk, kNv, kDs, core0.p));
  const std::vector<float> start = s0.get();
  const float* c7 = conv.p + 7 * kCh;
  const float* b7 = db.p + 7 * kNv;
  const float* a7 = da.p + 7 * kNv;
  // the batch of n with checkpoints
  Dev<float> sc(start), corec(static_cast<std::size_t>(n * kDin)), ck(static_cast<std::size_t>((n - 1) * ss));
  CHECK(g::deltanet_recurrent_reg_ckpt(c7, kCh, b7, a7, sc.p, n, kNk, kNv, kDs, corec.p, ck.p));
  // the plain kernel on the same batch, and on each prefix
  Dev<float> sp(start), corep(static_cast<std::size_t>(n * kDin));
  CHECK(g::deltanet_recurrent_reg(c7, kCh, b7, a7, sp.p, n, kNk, kNv, kDs, corep.p));
  CHECK(g::sync());
  const std::vector<float> fc = sc.get(), fp = sp.get(), cc = corec.get(), cp = corep.get(), cks = ck.get();
  CHECK(std::memcmp(fc.data(), fp.data(), fc.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(cc.data(), cp.data(), cc.size() * sizeof(float)) == 0);
  int prefixes_equal = 0;
  for (std::int64_t k = 1; k < n; ++k) {
    Dev<float> sk(start), corek(static_cast<std::size_t>(k * kDin));
    CHECK(g::deltanet_recurrent_reg(c7, kCh, b7, a7, sk.p, k, kNk, kNv, kDs, corek.p));
    CHECK(g::sync());
    const std::vector<float> fk = sk.get();
    const bool eq = std::memcmp(fk.data(), cks.data() + (k - 1) * ss, static_cast<std::size_t>(ss) * sizeof(float)) == 0;
    CHECK(eq);
    prefixes_equal += eq ? 1 : 0;
  }
  // n == 1 needs no checkpoint buffer
  Dev<float> s1(start), core1(static_cast<std::size_t>(kDin));
  CHECK(g::deltanet_recurrent_reg_ckpt(c7, kCh, b7, a7, s1.p, 1, kNk, kNv, kDs, core1.p, nullptr));
  CHECK(!g::deltanet_recurrent_reg_ckpt(c7, kCh, b7, a7, s1.p, 2, kNk, kNv, kDs, core1.p, nullptr));  // n > 1 needs one
  CHECK(g::sync());
  std::printf("    checkpoint kernel: core and final state bit for bit equal to the plain kernel; %d / %lld prefix states bit for bit\n", prefixes_equal,
              static_cast<long long>(n - 1));
}
