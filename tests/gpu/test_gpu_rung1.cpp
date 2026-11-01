// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// Rung 1 of the GPU kernels (masterwork 16.1 / 16.2, game plan GP3-10 GPU): every kernel of kern/gpu_ops.h on the card against
// an fp64 host reference of the same inputs, with the rung-1 edge shapes (expert token counts 0, 1, 7, 8, 63, 64, 65; every
// token on one expert; zero, denormal and outlier inputs: no NaN / Inf). The values that the CPU reference computes in a
// fixed order (the decoders, the 8-bit activation rounding, the router's picks) must match it bit for bit; the rest within
// the tolerances of 16.2 (trainer master 6.11 governs: products relative L2 <= 1e-6 against fp64 of the same rounded inputs;
// attention <= 5e-4; DeltaNet <= 1e-4 per layer window). Run on the PC under compute-sanitizer (memcheck, racecheck,
// initcheck, synccheck: the script runs each). Every measured error is printed, pass or fail.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../unit/gpu_test_rows.h"
#include "kern/gpu_decode.h"
#include "kern/gpu_ops.h"
#include "model/arch_qwen35moe.h"
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

// a matrix of n rows of k values in type t (each row quantized on its own), and its exact float rows
struct Mat {
  std::vector<std::uint8_t> bytes;
  std::vector<float> deq;  // n * k, the values the decoders must give
  std::uint64_t rb = 0;
};
Mat make_mat(q::Type t, std::int64_t n, std::int64_t k, std::uint32_t seed) {
  Mat m;
  m.rb = q::row_bytes(static_cast<std::uint32_t>(t), static_cast<std::uint64_t>(k));
  m.bytes.resize(static_cast<std::size_t>(m.rb) * static_cast<std::size_t>(n));
  m.deq.resize(static_cast<std::size_t>(n * k));
  for (std::int64_t r = 0; r < n; ++r) {
    const std::uint32_t rs = seed + static_cast<std::uint32_t>(r) * 7919u;
    const std::vector<std::uint8_t> row = pelias::test_gpu::make_row(t, wave(k, rs, 0.05f), rs * 2654435761u);
    std::memcpy(m.bytes.data() + static_cast<std::size_t>(r) * m.rb, row.data(), m.rb);
    (void)q::dequantize_row(t, row.data(), m.deq.data() + r * k, k);
  }
  return m;
}

// relative L2 of a against the fp64 reference b
double rel_l2(const std::vector<float>& a, const std::vector<double>& b) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < b.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

bool all_finite(const std::vector<float>& v) {
  for (const float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

bool bits_equal(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool gpu_or_skip() {
  if (g::available()) return true;
  std::printf("    no GPU: %s\n", g::last_error());
  CHECK(g::available());  // this binary exists to run on a card; no card is a failure, not a skip
  return false;
}

constexpr double kProductTol = 1e-6;  // trainer master 6.11 (masterwork 16.2)

}  // namespace

TEST(rung1_device_answers) {
  CHECK(g::compiled());
  if (!gpu_or_skip()) return;
  std::printf("    CUDA device found\n");
}

TEST(rung1_matmul_every_type_against_fp64) {
  if (!gpu_or_skip()) return;
  const std::vector<q::Type> types = pelias::test_gpu::readable_types();  // all 32 types the CPU reads
  CHECK(types.size() == 32);
  const std::int64_t n_out = 97, k = 2048;  // 97: the last warp of the last block has one row
  double worst = 0.0;
  for (const q::Type t : types) {
    const Mat m = make_mat(t, n_out, k, 11u + static_cast<std::uint32_t>(t));
    Dev<std::uint8_t> dw(m.bytes);
    double tw = 0.0;
    for (const std::int64_t n_tok : {std::int64_t{1}, std::int64_t{3}, std::int64_t{9}}) {  // 9: across the 8-token pass
      const std::vector<float> x = wave(n_tok * k, 5u + static_cast<std::uint32_t>(n_tok), 1.0f);
      Dev<float> dx(x), dy(static_cast<std::size_t>(n_tok * n_out));
      CHECK(g::matmul(static_cast<std::uint32_t>(t), dw.p, m.rb, n_out, k, dx.p, n_tok, k, dy.p, n_out));
      CHECK(g::sync());
      const std::vector<float> y = dy.get();
      std::vector<double> ref(y.size());
      for (std::int64_t tt = 0; tt < n_tok; ++tt)
        for (std::int64_t r = 0; r < n_out; ++r) {
          double s = 0.0;
          for (std::int64_t i = 0; i < k; ++i) s += static_cast<double>(m.deq[static_cast<std::size_t>(r * k + i)]) * x[static_cast<std::size_t>(tt * k + i)];
          ref[static_cast<std::size_t>(tt * n_out + r)] = s;
        }
      const double e = rel_l2(y, ref);
      worst = std::max(worst, e);
      tw = std::max(tw, e);
      CHECK(all_finite(y));
      CHECK(e <= kProductTol);
      if (e > kProductTol) std::printf("    %s n_tok %lld: relative L2 %.3e > %.0e\n", q::type_info(static_cast<std::uint32_t>(t)).name,
                                       static_cast<long long>(n_tok), e, kProductTol);
    }
    std::printf("    %-8s worst relative L2 %.3e\n", q::type_info(static_cast<std::uint32_t>(t)).name, tw);
  }
  std::printf("    %zu types x n_tok 1 / 3 / 9, 97 x 2048: worst relative L2 vs fp64 %.3e (tolerance %.0e)\n", types.size(), worst, kProductTol);
}

TEST(rung1_matmul_refuses_what_it_cannot_read) {
  if (!gpu_or_skip()) return;
  float dummy = 0.0f;
  CHECK(!g::matmul(static_cast<std::uint32_t>(q::Type::q8_1), &dummy, 36, 1, 256, &dummy, 1, 256, &dummy, 1));  // activation type
  CHECK(!g::matmul(static_cast<std::uint32_t>(q::Type::i2_s), &dummy, 32, 1, 64, &dummy, 1, 64, &dummy, 1));    // half an I2_S block
  CHECK(!g::matmul(static_cast<std::uint32_t>(q::Type::q4_k), &dummy, 144, 1, 200, &dummy, 1, 200, &dummy, 1));
  CHECK(!g::matmul(static_cast<std::uint32_t>(q::Type::q4_k), &dummy, 143, 1, 256, &dummy, 1, 256, &dummy, 1));
}

TEST(rung1_moe_matmul_edge_counts) {
  if (!gpu_or_skip()) return;
  const std::int64_t n_out = 64, k = 512;
  const int n_expert = 9;
  for (const q::Type t : {q::Type::q4_tg2, q::Type::q5_k, q::Type::q6_k, q::Type::iq1_m, q::Type::tq1_0, q::Type::i2_s, q::Type::nvfp4}) {
    std::vector<Mat> mats;
    std::vector<std::uint8_t> all;
    for (int e = 0; e < n_expert; ++e) {
      mats.push_back(make_mat(t, n_out, k, 100u * static_cast<std::uint32_t>(e) + 3u));
      all.insert(all.end(), mats.back().bytes.begin(), mats.back().bytes.end());
    }
    const std::uint64_t rb = mats[0].rb, eb = rb * static_cast<std::uint64_t>(n_out);
    for (int pattern = 0; pattern < 2; ++pattern) {
      // pattern 0: the rung-1 counts 0, 1, 7, 8, 63, 64, 65 (+ 0, 9); pattern 1: every slot on expert 4, none elsewhere
      const int counts0[n_expert] = {0, 1, 7, 8, 63, 64, 65, 0, 9};
      std::vector<int> off(static_cast<std::size_t>(n_expert) + 1, 0), src, dst;
      int total = 0, maxc = 0;
      for (int e = 0; e < n_expert; ++e) {
        const int c = pattern == 0 ? counts0[e] : (e == 4 ? 217 : 0);
        for (int i = 0; i < c; ++i) {
          src.push_back((total * 37 + 11) % 300);  // tokens in any order, repeats allowed (a token's slots read one row)
          dst.push_back(total);
          ++total;
        }
        maxc = std::max(maxc, c);
        off[static_cast<std::size_t>(e) + 1] = total;
      }
      const std::vector<float> x = wave(300 * k, 77u, 1.0f);
      Dev<std::uint8_t> dw(all);
      Dev<int> doff(off), dsrc(src), ddst(dst);
      Dev<float> dx(x), dy(static_cast<std::size_t>(total) * static_cast<std::size_t>(n_out));
      CHECK(g::moe_matmul(static_cast<std::uint32_t>(t), dw.p, eb, rb, n_out, k, n_expert, doff.p, dsrc.p, ddst.p, dx.p, k, dy.p, n_out, maxc));
      CHECK(g::sync());
      const std::vector<float> y = dy.get();
      std::vector<double> ref(y.size());
      for (int e = 0; e < n_expert; ++e)
        for (int sl = off[static_cast<std::size_t>(e)]; sl < off[static_cast<std::size_t>(e) + 1]; ++sl)
          for (std::int64_t r = 0; r < n_out; ++r) {
            double s = 0.0;
            for (std::int64_t i = 0; i < k; ++i)
              s += static_cast<double>(mats[static_cast<std::size_t>(e)].deq[static_cast<std::size_t>(r * k + i)]) *
                   x[static_cast<std::size_t>(src[static_cast<std::size_t>(sl)] * k + i)];
            ref[static_cast<std::size_t>(dst[static_cast<std::size_t>(sl)] * n_out + r)] = s;
          }
      const double e = rel_l2(y, ref);
      CHECK(all_finite(y));
      CHECK(e <= kProductTol);
      std::printf("    %s, %s: %d slots, relative L2 %.3e\n", q::type_info(static_cast<std::uint32_t>(t)).name,
                  pattern == 0 ? "counts 0,1,7,8,63,64,65,0,9" : "all on one expert", total, e);
    }
  }
}

TEST(rung1_gather_rows_exact) {
  if (!gpu_or_skip()) return;
  const std::int64_t n = 50, k = 1024;
  for (const q::Type t : {q::Type::q8_0, q::Type::q6_k, q::Type::q4_tg2, q::Type::bf16, q::Type::mxfp4, q::Type::iq2_xxs, q::Type::i2_s,
                          q::Type::nvfp4}) {
    const Mat m = make_mat(t, n, k, 9u);
    const std::vector<std::int32_t> ids = {7, 0, 49, 7, 23};
    Dev<std::uint8_t> dw(m.bytes);
    Dev<std::int32_t> di(ids);
    Dev<float> dout(ids.size() * static_cast<std::size_t>(k));
    CHECK(g::gather_rows(static_cast<std::uint32_t>(t), dw.p, m.rb, k, di.p, static_cast<std::int64_t>(ids.size()), dout.p));
    CHECK(g::sync());
    const std::vector<float> out = dout.get();
    std::vector<float> ref;
    for (const std::int32_t id : ids) ref.insert(ref.end(), m.deq.begin() + id * k, m.deq.begin() + (id + 1) * k);
    CHECK(bits_equal(out, ref));
  }
}

TEST(rung1_act_q8x32_equals_the_reference_rounding) {
  if (!gpu_or_skip()) return;
  const std::int64_t n = 2048, rows = 5;
  std::vector<float> x = wave(n * rows, 31u, 4.0f);
  x[0] = 0.0f;
  for (int i = 32; i < 64; ++i) x[static_cast<std::size_t>(i)] = 0.0f;  // an all-zero block: scale 1
  x[100] = 1e-40f;                                                       // a denormal
  x[300] = 3.0e30f;                                                      // an outlier
  for (const bool rot : {false, true}) {
    Dev<float> dx(x), dr(x.size()), ds(static_cast<std::size_t>(rows * n / 32));
    Dev<std::int8_t> dq(x.size());
    Dev<int> dsum(static_cast<std::size_t>(rows * n / 32));
    CHECK(g::act_q8x32(dx.p, n, rows, n, rot, dr.p, n, dq.p, ds.p, dsum.p));
    CHECK(g::sync());
    std::vector<float> ref = x;
    for (std::int64_t t = 0; t < rows; ++t) {
      if (rot) q::hadamard32(ref.data() + t * n, n);
      pelias::model::act_round_q8_256x8(ref.data() + t * n, n);
    }
    const std::vector<float> got = dr.get();
    CHECK(bits_equal(got, ref));
    CHECK(all_finite(got));
    const std::vector<std::int8_t> qv = dq.get();
    const std::vector<float> sc = ds.get();
    const std::vector<int> sm = dsum.get();
    bool consistent = true;
    for (std::int64_t b = 0; b < rows * n / 32; ++b) {
      int s = 0;
      for (int i = 0; i < 32; ++i) {
        s += qv[static_cast<std::size_t>(32 * b + i)];
        consistent = consistent && sc[static_cast<std::size_t>(b)] * static_cast<float>(qv[static_cast<std::size_t>(32 * b + i)]) == got[static_cast<std::size_t>(32 * b + i)];
      }
      consistent = consistent && s == sm[static_cast<std::size_t>(b)];
    }
    CHECK(consistent);
    std::printf("    %s: rounded values bit for bit equal to model::act_round_q8_256x8; int8, scales and sums consistent\n",
                rot ? "rotated" : "plain");
  }
}

// the fused norm + rounding (rms_norm_q8x32) equals rms_norm then act_q8x32 bit for bit: normalized floats, rounded
// floats, int8 values, scales, sums; with and without the weight, plain and rotated, rows of 512 / 2048 / 4096 (a row
// longer than 256 blocks of 32 is not needed by the model; 4096 = 128 blocks). Both paths timed (information only).
TEST(rung1_rms_norm_q8x32_fused_equals_two_kernels) {
  if (!gpu_or_skip()) return;
  for (const std::int64_t n : {512, 2048, 4096}) {
    const std::int64_t rows = 7;
    std::vector<float> x = wave(n * rows, 51u + static_cast<std::uint32_t>(n), 3.0f), w = wave(n, 52u, 1.0f);
    for (float& v : w) v += 1.0f;
    for (std::int64_t i = 0; i < n; ++i) x[static_cast<std::size_t>(3 * n + i)] = 0.0f;  // an all-zero row
    x[static_cast<std::size_t>(n + 5)] = 2.0e20f;                                      // an outlier row
    for (const bool rot : {false, true})
      for (const bool hasw : {false, true}) {
        const std::size_t nn = static_cast<std::size_t>(n * rows), nb = static_cast<std::size_t>(rows * n / 32);
        Dev<float> dx(x), dw(w), y1(nn), r1(nn), s1(nb), y2(nn), r2(nn), s2(nb);
        Dev<std::int8_t> q1(nn), q2(nn);
        Dev<int> m1(nb), m2(nb);
        const float* wp = hasw ? dw.p : nullptr;
        CHECK(g::rms_norm(dx.p, wp, y1.p, n, 1, n, n, rows, 1e-6f));
        CHECK(g::act_q8x32(y1.p, n, rows, n, rot, r1.p, n, q1.p, s1.p, m1.p));
        CHECK(g::rms_norm_q8x32(dx.p, wp, n, n, rows, 1e-6f, rot, y2.p, n, r2.p, n, q2.p, s2.p, m2.p));
        CHECK(g::sync());
        const bool same = bits_equal(y1.get(), y2.get()) && bits_equal(r1.get(), r2.get()) && q1.get() == q2.get() && bits_equal(s1.get(), s2.get()) &&
                          m1.get() == m2.get();
        CHECK(same);
        CHECK(all_finite(r2.get()));
        if (!same || !hasw || std::getenv("PELIAS_GPU_QUICK") != nullptr) continue;  // timing: not under the sanitizers
        // timing (warm, information only): the two kernels against the fused one, int8 outputs only (xr / y not written),
        // at 1 row (decode) and 2,048 rows (a prompt)
        const auto time_us = [&](int reps, const auto& fn) {
          fn();
          CHECK(g::sync());
          const auto c0 = std::chrono::steady_clock::now();
          for (int i = 0; i < reps; ++i) fn();
          CHECK(g::sync());
          return std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count() * 1e6 / reps;
        };
        const std::int64_t big = 2048;
        Dev<float> bx(wave(n * big, 53u, 3.0f)), by(static_cast<std::size_t>(n * big)), bs(static_cast<std::size_t>(big * n / 32));
        Dev<std::int8_t> bq(static_cast<std::size_t>(n * big));
        Dev<int> bm(static_cast<std::size_t>(big * n / 32));
        for (const std::int64_t tr : {std::int64_t{1}, big}) {
          const double t2 = time_us(200, [&] {
            CHECK(g::rms_norm(bx.p, wp, by.p, n, 1, n, n, tr, 1e-6f));
            CHECK(g::act_q8x32(by.p, n, tr, n, rot, by.p, n, bq.p, bs.p, bm.p));
          });
          const double t1 = time_us(200, [&] { CHECK(g::rms_norm_q8x32(bx.p, wp, n, n, tr, 1e-6f, rot, nullptr, n, nullptr, n, bq.p, bs.p, bm.p)); });
          std::printf("    n %4lld, %s, %4lld row(s): two kernels %8.2f us, fused %8.2f us (%.2fx)\n", static_cast<long long>(n), rot ? "rotated" : "plain  ",
                      static_cast<long long>(tr), t2, t1, t1 > 0.0 ? t2 / t1 : 0.0);
        }
      }
  }
  std::printf("    fused = rms_norm then act_q8x32, bit for bit: normalized, rounded, int8, scales, sums (n 512 / 2048 / 4096, with and without w, plain and rotated)\n");
}

TEST(rung1_norms_and_elementwise) {
  if (!gpu_or_skip()) return;
  const std::int64_t n = 2048, rows = 3;
  std::vector<float> x = wave(n * rows, 41u, 2.0f), w = wave(n, 42u, 1.0f);
  for (float& v : w) v += 1.0f;
  for (std::int64_t i = 0; i < n; ++i) x[static_cast<std::size_t>(2 * n + i)] = 0.0f;  // row 2 all zero: eps keeps it finite
  // rms norm, whole rows
  {
    Dev<float> dx(x), dw(w), dy(x.size());
    CHECK(g::rms_norm(dx.p, dw.p, dy.p, n, 1, n, n, rows, 1e-6f));
    CHECK(g::sync());
    const std::vector<float> y = dy.get();
    std::vector<double> ref(y.size());
    for (std::int64_t t = 0; t < rows; ++t) {
      double ss = 0.0;
      for (std::int64_t i = 0; i < n; ++i) ss += static_cast<double>(x[static_cast<std::size_t>(t * n + i)]) * x[static_cast<std::size_t>(t * n + i)];
      const double sc = 1.0 / std::sqrt(ss / static_cast<double>(n) + 1e-6);
      for (std::int64_t i = 0; i < n; ++i) ref[static_cast<std::size_t>(t * n + i)] = x[static_cast<std::size_t>(t * n + i)] * sc * w[static_cast<std::size_t>(i)];
    }
    const double e = rel_l2(y, ref);
    CHECK(all_finite(y));
    CHECK(e <= 1e-6);
    std::printf("    rms_norm: relative L2 %.3e\n", e);
  }
  // per-head segments (q / k norm: 16 heads of 256) and l2 norm (DeltaNet q / k: 32 heads of 128)
  {
    Dev<float> dx(x), dw(std::vector<float>(w.begin(), w.begin() + 256)), dy(x.size());
    CHECK(g::rms_norm(dx.p, dw.p, dy.p, 256, 8, 256, n, rows, 1e-6f));
    Dev<float> dl(x);
    CHECK(g::l2_norm(dl.p, 128, 16, 128, n, rows, 1e-6f));
    CHECK(g::sync());
    const std::vector<float> y = dy.get(), l = dl.get();
    std::vector<double> r1(y.size()), r2(l.size());
    for (std::int64_t t = 0; t < rows; ++t) {
      for (std::int64_t h = 0; h < 8; ++h) {
        double ss = 0.0;
        for (std::int64_t i = 0; i < 256; ++i) {
          const double v = x[static_cast<std::size_t>(t * n + h * 256 + i)];
          ss += v * v;
        }
        const double sc = 1.0 / std::sqrt(ss / 256.0 + 1e-6);
        for (std::int64_t i = 0; i < 256; ++i)
          r1[static_cast<std::size_t>(t * n + h * 256 + i)] = x[static_cast<std::size_t>(t * n + h * 256 + i)] * sc * w[static_cast<std::size_t>(i)];
      }
      for (std::int64_t h = 0; h < 16; ++h) {
        double ss = 0.0;
        for (std::int64_t i = 0; i < 128; ++i) {
          const double v = x[static_cast<std::size_t>(t * n + h * 128 + i)];
          ss += v * v;
        }
        const double sc = 1.0 / std::sqrt(ss + 1e-6);
        for (std::int64_t i = 0; i < 128; ++i) r2[static_cast<std::size_t>(t * n + h * 128 + i)] = x[static_cast<std::size_t>(t * n + h * 128 + i)] * sc;
      }
      for (std::int64_t i = 16 * 128; i < n; ++i) r2[static_cast<std::size_t>(t * n + i)] = x[static_cast<std::size_t>(t * n + i)];  // untouched
    }
    // the segments' own outputs only (the rms test wrote heads 0..7 of each row; the rest of y is not written)
    std::vector<float> y8;
    std::vector<double> r8;
    for (std::int64_t t = 0; t < rows; ++t)
      for (std::int64_t i = 0; i < 8 * 256; ++i) {
        y8.push_back(y[static_cast<std::size_t>(t * n + i)]);
        r8.push_back(r1[static_cast<std::size_t>(t * n + i)]);
      }
    const double e1 = rel_l2(y8, r8), e2 = rel_l2(l, r2);
    CHECK(e1 <= 1e-6);
    CHECK(e2 <= 1e-6);
    CHECK(all_finite(l));
    std::printf("    rms_norm per head %.3e, l2_norm per head %.3e\n", e1, e2);
  }
  // silu * mul and weighted row adds, with outliers (+-80, +-1e30) and denormals: finite, and close to fp64
  {
    std::vector<float> a = wave(4096, 51u, 6.0f), b = wave(4096, 52u, 1.0f);
    a[0] = 80.0f; a[1] = -80.0f; a[2] = 1e-40f; a[3] = 0.0f;
    Dev<float> da(a), db(b), dout(a.size());
    CHECK(g::silu_mul(da.p, db.p, dout.p, 4096));
    std::vector<float> wgt = {0.5f, -2.0f};
    Dev<float> dyy(b), dwt(wgt);
    CHECK(g::add_rows(dyy.p, 2048, da.p, 2048, dwt.p, 2048, 2));
    CHECK(g::sync());
    const std::vector<float> o = dout.get(), yy = dyy.get();
    std::vector<double> ro(o.size()), ry(yy.size());
    for (std::size_t i = 0; i < o.size(); ++i) ro[i] = (a[i] / (1.0 + std::exp(-static_cast<double>(a[i])))) * b[i];
    for (std::size_t i = 0; i < yy.size(); ++i) ry[i] = static_cast<double>(b[i]) + static_cast<double>(a[i]) * wgt[i / 2048];
    const double e1 = rel_l2(o, ro), e2 = rel_l2(yy, ry);
    CHECK(all_finite(o));
    CHECK(all_finite(yy));
    CHECK(e1 <= 1e-6);
    CHECK(e2 <= 1e-6);
    std::printf("    silu_mul %.3e, add_rows %.3e\n", e1, e2);
  }
}

TEST(rung1_rope_three_ways) {
  if (!gpu_or_skip()) return;
  const std::int64_t hd = 256, nh = 4, rows = 3;
  const int n_rot = 64;
  const float base = 1.0e7f;
  struct Way {
    const char* name;
    int sec[4];
    bool neox;
    std::int64_t pos0;
  } ways[] = {{"IMRoPE (11/11/10/0, NEOX)", {11, 11, 10, 0}, true, 0}, {"IMRoPE at position 200000", {11, 11, 10, 0}, true, 200000},
              {"plain NEOX", {0, 0, 0, 0}, true, 5}, {"adjacent pairs", {0, 0, 0, 0}, false, 5}};
  for (const Way& wy : ways) {
    const std::vector<float> v = wave(rows * nh * hd, 61u, 1.0f);
    Dev<float> dv(v);
    CHECK(g::rope(dv.p, nh, hd, nh * hd, rows, wy.pos0, n_rot, base, wy.sec, wy.neox, nullptr));
    CHECK(g::sync());
    const std::vector<float> got = dv.get();
    // fp64 reference of the same angles (the angles themselves follow the reference's float recurrence)
    std::vector<double> ref(v.begin(), v.end());
    const float ts = std::pow(base, -2.0f / static_cast<float>(n_rot));
    const int sect = wy.sec[0] + wy.sec[1] + wy.sec[2] + wy.sec[3];
    for (std::int64_t t = 0; t < rows; ++t) {
      float th = static_cast<float>(wy.pos0 + t), the = 0.0f;
      for (int p = 0; p < n_rot / 2; ++p) {
        bool use_e = false;
        if (sect > 0) {
          const int sector = p % sect;
          if (sector % 3 == 1 && sector < 3 * wy.sec[1]) use_e = false;
          else if (sector % 3 == 2 && sector < 3 * wy.sec[2]) use_e = false;
          else if (sector % 3 == 0 && sector < 3 * wy.sec[0]) use_e = false;
          else use_e = true;
        }
        const double theta = use_e ? the : th;
        const double c = std::cos(theta), s = std::sin(theta);
        for (std::int64_t h = 0; h < nh; ++h) {
          const std::size_t o = static_cast<std::size_t>(t * nh * hd + h * hd);
          const std::size_t a = o + static_cast<std::size_t>(wy.neox ? p : 2 * p), b = o + static_cast<std::size_t>(wy.neox ? p + n_rot / 2 : 2 * p + 1);
          const double x0 = v[a], x1 = v[b];
          ref[a] = x0 * c - x1 * s;
          ref[b] = x0 * s + x1 * c;
        }
        th *= ts;
        the *= ts;
      }
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - ref[i]));
    CHECK(all_finite(got));
    CHECK(worst <= 2e-5);  // sinf / cosf of angles up to 2e5 rad: a few ulp of the argument reduction
    std::printf("    %s: max |GPU - fp64| %.3e\n", wy.name, worst);
  }
}

TEST(rung1_router_picks_equal_the_reference) {
  if (!gpu_or_skip()) return;
  const std::int64_t ne = 256, rows = 64;
  const int nu = 8;
  std::vector<float> lg(static_cast<std::size_t>(ne * rows));
  for (std::int64_t t = 0; t < rows; ++t)
    for (std::int64_t i = 0; i < ne; ++i)  // distinct logits per row (a permutation), so the picks are unambiguous
      lg[static_cast<std::size_t>(t * ne + i)] = 0.01f * static_cast<float>((i * 97 + t * 13) % ne) - 1.0f;
  for (std::int64_t i = 0; i < ne; ++i) lg[static_cast<std::size_t>(5 * ne + i)] = 0.25f;  // row 5: every expert tied
  Dev<float> dl(lg), dw(static_cast<std::size_t>(rows * nu));
  Dev<std::int32_t> ds(static_cast<std::size_t>(rows * nu));
  CHECK(g::router_topk(dl.p, ne, rows, nu, 1.0f, ds.p, dw.p));
  CHECK(g::sync());
  const std::vector<std::int32_t> sel = ds.get();
  const std::vector<float> wsel = dw.get();
  bool same = true;
  double worst = 0.0;
  for (std::int64_t t = 0; t < rows; ++t) {
    std::vector<double> p(static_cast<std::size_t>(ne));
    double mx = -1e300, sum = 0.0;
    for (std::int64_t i = 0; i < ne; ++i) mx = std::max(mx, static_cast<double>(lg[static_cast<std::size_t>(t * ne + i)]));
    for (std::int64_t i = 0; i < ne; ++i) sum += p[static_cast<std::size_t>(i)] = std::exp(lg[static_cast<std::size_t>(t * ne + i)] - mx);
    std::vector<std::int32_t> order(static_cast<std::size_t>(ne));
    for (std::int64_t i = 0; i < ne; ++i) order[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
    std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) { return p[static_cast<std::size_t>(a)] > p[static_cast<std::size_t>(b)]; });
    double tot = 0.0;
    for (int kk = 0; kk < nu; ++kk) tot += p[static_cast<std::size_t>(order[static_cast<std::size_t>(kk)])] / sum;
    for (int kk = 0; kk < nu; ++kk) {
      same = same && sel[static_cast<std::size_t>(t * nu + kk)] == order[static_cast<std::size_t>(kk)];
      const double w = p[static_cast<std::size_t>(order[static_cast<std::size_t>(kk)])] / sum / tot;
      worst = std::max(worst, std::fabs(wsel[static_cast<std::size_t>(t * nu + kk)] - w) / w);
    }
  }
  CHECK(same);
  CHECK(worst <= 1e-6);
  CHECK(sel[static_cast<std::size_t>(5 * nu)] == 0 && sel[static_cast<std::size_t>(5 * nu + 7)] == 7);  // ties: lower index first
  std::printf("    64 rows x 256 experts, top 8: picks identical (ties by lower index), weights within %.3e relative\n", worst);
}

// v22: the shared-memory router kernel against the earlier serial one, bit for bit (picks and weights), on random rows,
// rows with ties and a row with every expert tied, 256 and 160 experts, 8 and 4 picks, a weight scale
TEST(rung1_router_fast_equals_serial_bit_for_bit) {
  if (!gpu_or_skip()) return;
  bool all = true;
  int rows_checked = 0;
  for (const std::int64_t ne : {std::int64_t{256}, std::int64_t{160}})
    for (const int nu : {8, 4})
      for (const float wscale : {1.0f, 2.5f}) {
        const std::int64_t rows = 41;
        std::vector<float> lg(static_cast<std::size_t>(ne * rows));
        std::uint32_t r = static_cast<std::uint32_t>(ne * 7 + nu);
        for (float& v : lg) {
          r = r * 1664525u + 1013904223u;
          v = static_cast<float>(static_cast<int>(r >> 9) % 4001 - 2000) * 0.003f;
        }
        for (std::int64_t i = 0; i < ne; ++i) lg[static_cast<std::size_t>(3 * ne + i)] = 0.5f;  // all tied
        for (std::int64_t i = 0; i < ne; i += 3) lg[static_cast<std::size_t>(7 * ne + i)] = 1.25f;  // many ties at the top
        Dev<float> dl(lg), dw1(static_cast<std::size_t>(rows * nu)), dw2(static_cast<std::size_t>(rows * nu));
        Dev<std::int32_t> ds1(static_cast<std::size_t>(rows * nu)), ds2(static_cast<std::size_t>(rows * nu));
        g::router_topk_serial_for_tests(true);
        CHECK(g::router_topk(dl.p, ne, rows, nu, wscale, ds1.p, dw1.p));
        g::router_topk_serial_for_tests(false);
        CHECK(g::router_topk(dl.p, ne, rows, nu, wscale, ds2.p, dw2.p));
        CHECK(g::sync());
        const std::vector<std::int32_t> s1 = ds1.get(), s2 = ds2.get();
        const std::vector<float> w1 = dw1.get(), w2 = dw2.get();
        all = all && s1 == s2 && std::memcmp(w1.data(), w2.data(), w1.size() * sizeof(float)) == 0;
        rows_checked += static_cast<int>(rows);
      }
  CHECK(all);
  std::printf("    %d rows (256 / 160 experts, 8 / 4 picks, weight scale 1 / 2.5, ties): shared-memory router = serial router bit for bit%s\n", rows_checked,
              all ? "" : " (NOT)");
}

// v24 tie rule: with a band, the picks are the host's order by floor((logit - max) / band), descending, lower index first
// within a band; the weights are the picked probabilities renormalized (positive, summing to the weight scale)
TEST(rung1_router_tie_band_picks_equal_host) {
  if (!gpu_or_skip()) return;
  bool all = true;
  int rows_checked = 0;
  double worst_sum = 0.0;
  for (const std::int64_t ne : {std::int64_t{256}, std::int64_t{160}})
    for (const int nu : {8, 4})
      for (const float band : {0.01f, 0.05f}) {
        const std::int64_t rows = 41;
        std::vector<float> lg(static_cast<std::size_t>(ne * rows));
        std::uint32_t r = static_cast<std::uint32_t>(ne * 11 + nu);
        for (float& v : lg) {
          r = r * 1664525u + 1013904223u;
          v = static_cast<float>(static_cast<int>(r >> 9) % 4001 - 2000) * 0.003f;
        }
        for (std::int64_t i = 0; i < ne; ++i) lg[static_cast<std::size_t>(3 * ne + i)] = 0.5f;
        for (std::int64_t i = 0; i < ne; i += 3) lg[static_cast<std::size_t>(7 * ne + i)] = 1.25f - 1e-5f * static_cast<float>(i % 5);  // near-ties
        Dev<float> dl(lg), dw(static_cast<std::size_t>(rows * nu));
        Dev<std::int32_t> ds(static_cast<std::size_t>(rows * nu));
        CHECK(g::router_topk(dl.p, ne, rows, nu, 1.0f, ds.p, dw.p, nullptr, band));
        CHECK(g::sync());
        const std::vector<std::int32_t> sg = ds.get();
        const std::vector<float> wg = dw.get();
        for (std::int64_t t = 0; t < rows; ++t) {
          const float* l = lg.data() + t * ne;
          float mx = l[0];
          for (std::int64_t i = 1; i < ne; ++i) mx = std::max(mx, l[i]);
          std::vector<float> key(static_cast<std::size_t>(ne));
          for (std::int64_t i = 0; i < ne; ++i) key[static_cast<std::size_t>(i)] = std::floor((l[i] - mx) / band);
          std::vector<std::int32_t> ord(static_cast<std::size_t>(ne));
          for (std::int64_t i = 0; i < ne; ++i) ord[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
          std::partial_sort(ord.begin(), ord.begin() + nu, ord.end(), [&](std::int32_t a, std::int32_t b) {
            return key[static_cast<std::size_t>(a)] > key[static_cast<std::size_t>(b)] || (key[static_cast<std::size_t>(a)] == key[static_cast<std::size_t>(b)] && a < b);
          });
          double ws = 0.0;
          for (int j = 0; j < nu; ++j) {
            all = all && sg[static_cast<std::size_t>(t * nu + j)] == ord[static_cast<std::size_t>(j)] && wg[static_cast<std::size_t>(t * nu + j)] > 0.0f;
            ws += wg[static_cast<std::size_t>(t * nu + j)];
          }
          worst_sum = std::max(worst_sum, std::fabs(ws - 1.0));
        }
        rows_checked += static_cast<int>(rows);
      }
  CHECK(all);
  CHECK(worst_sum < 1e-5);
  std::printf("    %d rows (256 / 160 experts, 8 / 4 picks, bands 0.01 / 0.05, ties and near-ties): band picks = host%s; weights sum to 1 within %.2e\n",
              rows_checked, all ? "" : " (NOT)", worst_sum);
}

// v24 picked-expert reader: 6 experts of mixed types, 3 in device memory and 3 in registered host memory (as the file
// mapping), 1-4 tokens of 3 picks, against fp64 products with the per-32 rounding of silu x up; and repeatable bit for bit
TEST(rung1_moe_reader_against_fp64) {
  if (!gpu_or_skip()) return;
  const std::int64_t H = 512, F = 256, ne = 6;  // k a multiple of 256 (the K-quant blocks)
  const int nu = 3;
  const q::Type types[3] = {q::Type::q8_0, q::Type::q6_k, q::Type::q4_tg2};
  std::vector<Mat> M;  // expert e, projection pj at 3 * e + pj
  for (std::int64_t e = 0; e < ne; ++e)
    for (int pj = 0; pj < 3; ++pj) {
      const q::Type t = types[(e + pj) % 3];
      M.push_back(pj < 2 ? make_mat(t, F, H, static_cast<std::uint32_t>(100 + 10 * e + pj)) : make_mat(t, H, F, static_cast<std::uint32_t>(100 + 10 * e + pj)));
    }
  // host memory for experts 3..5, page aligned and registered; device memory for 0..2
  std::size_t host_b = 0;
  for (std::int64_t e = 3; e < ne; ++e)
    for (int pj = 0; pj < 3; ++pj) host_b += (M[static_cast<std::size_t>(3 * e + pj)].bytes.size() + 4095) & ~std::size_t{4095};
  void* hm = std::aligned_alloc(4096, host_b);
  CHECK(hm != nullptr);
  if (!hm) return;
  void* hdev = nullptr;
  std::string why;
  const bool reg = g::host_register(hm, host_b, &hdev, &why);
  if (!reg) std::printf("    host_register refused (%s): the host experts are copied to the device instead\n", why.c_str());
  std::vector<g::ExpertPtr> tab(static_cast<std::size_t>(ne));
  std::vector<void*> dmem;
  std::size_t hoff = 0;
  for (std::int64_t e = 0; e < ne; ++e)
    for (int pj = 0; pj < 3; ++pj) {
      const Mat& m = M[static_cast<std::size_t>(3 * e + pj)];
      g::ExpertPtr& X = tab[static_cast<std::size_t>(e)];
      X.type[pj] = static_cast<std::uint32_t>(types[(e + pj) % 3]);
      X.rb[pj] = m.rb;
      if (e >= 3 && reg) {
        std::memcpy(static_cast<std::uint8_t*>(hm) + hoff, m.bytes.data(), m.bytes.size());
        X.w[pj] = static_cast<const std::uint8_t*>(hdev) + hoff;
        hoff += (m.bytes.size() + 4095) & ~std::size_t{4095};
      } else {
        void* d = g::dalloc(m.bytes.size());
        CHECK(d && g::to_device(d, m.bytes.data(), m.bytes.size()));
        dmem.push_back(d);
        X.w[pj] = static_cast<const std::uint8_t*>(d);
      }
    }
  Dev<g::ExpertPtr> dtab(tab);
  std::uint64_t row_max = 0;
  for (const g::ExpertPtr& X : tab)
    for (int pj = 0; pj < 3; ++pj) row_max = std::max<std::uint64_t>(row_max, X.rb[pj]);
  bool ok_all = true, same = true, v25_eq = true, alone_eq = true;
  double worst = 0.0;
  for (const std::int64_t T : {std::int64_t{1}, std::int64_t{2}, std::int64_t{4}}) {
    const std::vector<float> x = wave(T * H, static_cast<std::uint32_t>(7 + T), 1.0f);
    std::vector<std::int32_t> sel(static_cast<std::size_t>(T * nu));
    std::vector<float> wsel(static_cast<std::size_t>(T * nu));
    for (std::int64_t t = 0; t < T; ++t)
      for (int j = 0; j < nu; ++j) {
        sel[static_cast<std::size_t>(t * nu + j)] = static_cast<std::int32_t>((t + 2 * j + (j == 2 ? 1 : 0)) % ne);
        wsel[static_cast<std::size_t>(t * nu + j)] = 0.2f + 0.15f * static_cast<float>(j);
      }
    Dev<float> dx(x), dw(wsel), dgu(static_cast<std::size_t>(T * nu * 2 * F)), dh(static_cast<std::size_t>(T * nu * F)), dy(static_cast<std::size_t>(T * H)),
        dy2(static_cast<std::size_t>(T * H)), dy1(static_cast<std::size_t>(T * H)), dyd(static_cast<std::size_t>(T * nu * H));
    Dev<std::int32_t> ds(sel);
    Dev<int> dgrp(static_cast<std::size_t>(1 + 6 * T * nu));
    Dev<unsigned long long> dts(6);
    // v25 grouped reader (the engine's), twice (repeatable), with stamps the second time; the v24 reader for comparison
    CHECK(g::moe_reader(dtab.p, ds.p, dw.p, nu, dx.p, T, H, F, row_max, dgu.p, dh.p, dyd.p, dgrp.p, dy.p));
    CHECK(g::moe_reader(dtab.p, ds.p, dw.p, nu, dx.p, T, H, F, row_max, dgu.p, dh.p, dyd.p, dgrp.p, dy2.p, dts.p));
    CHECK(g::moe_reader_v24(dtab.p, ds.p, dw.p, nu, dx.p, T, H, F, dgu.p, dh.p, dy1.p));
    CHECK(g::sync());
    const std::vector<float> y = dy.get(), y2 = dy2.get(), y1 = dy1.get();
    same = same && std::memcmp(y.data(), y2.data(), y.size() * sizeof(float)) == 0;
    v25_eq = v25_eq && std::memcmp(y.data(), y1.data(), y.size() * sizeof(float)) == 0;
    const std::vector<unsigned long long> tsv = dts.get();
    for (int k = 1; k < 6; ++k) CHECK(tsv[static_cast<std::size_t>(k)] >= tsv[static_cast<std::size_t>(k - 1)]);
    // batch invariance: each token alone gives its row of the batch bit for bit
    for (std::int64_t t = 0; t < T; ++t) {
      Dev<float> dxa(std::vector<float>(x.begin() + t * H, x.begin() + (t + 1) * H)),
          dwa(std::vector<float>(wsel.begin() + t * nu, wsel.begin() + (t + 1) * nu)), dya(static_cast<std::size_t>(H));
      Dev<std::int32_t> dsa(std::vector<std::int32_t>(sel.begin() + t * nu, sel.begin() + (t + 1) * nu));
      CHECK(g::moe_reader(dtab.p, dsa.p, dwa.p, nu, dxa.p, 1, H, F, row_max, dgu.p, dh.p, dyd.p, dgrp.p, dya.p));
      CHECK(g::sync());
      const std::vector<float> ya = dya.get();
      alone_eq = alone_eq && std::memcmp(ya.data(), y.data() + t * H, static_cast<std::size_t>(H) * sizeof(float)) == 0;
    }
    std::vector<double> ref(static_cast<std::size_t>(T * H), 0.0);
    for (std::int64_t t = 0; t < T; ++t)
      for (int j = 0; j < nu; ++j) {
        const std::int64_t e = sel[static_cast<std::size_t>(t * nu + j)];
        const Mat &mg = M[static_cast<std::size_t>(3 * e)], &mu = M[static_cast<std::size_t>(3 * e + 1)], &md = M[static_cast<std::size_t>(3 * e + 2)];
        std::vector<float> h(static_cast<std::size_t>(F));
        for (std::int64_t b = 0; b < F / 32; ++b) {
          float v[32], a = 0.0f;
          for (int i = 0; i < 32; ++i) {
            const std::int64_t r = 32 * b + i;
            double gg = 0.0, uu = 0.0;
            for (std::int64_t k = 0; k < H; ++k) {
              gg += static_cast<double>(mg.deq[static_cast<std::size_t>(r * H + k)]) * x[static_cast<std::size_t>(t * H + k)];
              uu += static_cast<double>(mu.deq[static_cast<std::size_t>(r * H + k)]) * x[static_cast<std::size_t>(t * H + k)];
            }
            const float gf = static_cast<float>(gg);
            v[i] = (gf / (1.0f + std::exp(-gf))) * static_cast<float>(uu);
            a = std::max(a, std::fabs(v[i]));
          }
          const float sc = a > 0.0f ? a / 127.0f : 1.0f, inv = 1.0f / sc;
          for (int i = 0; i < 32; ++i) h[static_cast<std::size_t>(32 * b + i)] = sc * std::min(127.0f, std::max(-127.0f, std::rint(v[i] * inv)));
        }
        for (std::int64_t r = 0; r < H; ++r) {
          double acc = 0.0;
          for (std::int64_t k = 0; k < F; ++k) acc += static_cast<double>(md.deq[static_cast<std::size_t>(r * F + k)]) * h[static_cast<std::size_t>(k)];
          ref[static_cast<std::size_t>(t * H + r)] += wsel[static_cast<std::size_t>(t * nu + j)] * acc;
        }
      }
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      num += (y[i] - ref[i]) * (y[i] - ref[i]);
      den += ref[i] * ref[i];
    }
    const double rel = std::sqrt(num / std::max(den, 1e-300));
    worst = std::max(worst, rel);
    ok_all = ok_all && rel <= 2e-3 && std::isfinite(rel);
  }
  CHECK(ok_all);
  CHECK(same);
  CHECK(v25_eq);
  CHECK(alone_eq);
  std::printf("    1 / 2 / 4 tokens x 3 picks, q8_0 / q6_k / q4_tg2, %s: worst rel L2 %.3e against fp64 (the per-32 rounding may flip a step); repeat %s; "
              "grouped = v24 %s; each token alone = its batch row %s\n",
              reg ? "3 experts from registered host memory" : "all on the device", worst, same ? "bit for bit" : "DIFFERS", v25_eq ? "bit for bit" : "DIFFERS",
              alone_eq ? "bit for bit" : "DIFFERS");
  for (void* d : dmem) g::dfree(d);
  if (reg) g::host_unregister(hm);
  std::free(hm);
}

// v26 reader v3: equal to the v24 / v25 reader bit for bit (Q4_TG2 from the per-lane grid, two rows per warp, half-warp
// down rows at F = 256 and a full warp at F = 1024), with the cache (a first call fills the slots, a second reads them:
// both bit for bit, and the second serves every file expert from VRAM), and with the shared expert folded in (against fp64)
TEST(rung1_moe_reader3_equals_v25_cache_and_shared) {
  if (!gpu_or_skip()) return;
  const q::Type types[3] = {q::Type::q8_0, q::Type::q6_k, q::Type::q4_tg2};
  bool eq_all = true, cache_eq = true, cache_hits_ok = true, shared_ok = true, plane_ok = true;
  double worst_sh = 0.0, worst_plane = 0.0;
  for (const std::int64_t F : {std::int64_t{256}, std::int64_t{1024}}) {
    const std::int64_t H = 512, ne = 6;
    const int nu = 3;
    std::vector<Mat> M;
    for (std::int64_t e = 0; e < ne + 1; ++e)  // expert 6: the shared expert
      for (int pj = 0; pj < 3; ++pj) {
        const q::Type ty = types[(e + pj) % 3];
        M.push_back(pj < 2 ? make_mat(ty, F, H, static_cast<std::uint32_t>(300 + 10 * e + pj + F)) : make_mat(ty, H, F, static_cast<std::uint32_t>(300 + 10 * e + pj + F)));
      }
    std::vector<g::ExpertPtr> tab(static_cast<std::size_t>(ne));
    g::ExpertPtr shx;
    std::vector<void*> dmem;
    std::uint64_t row_max = 0, ebytes_max = 0;
    for (std::int64_t e = 0; e < ne + 1; ++e) {
      g::ExpertPtr& X = e < ne ? tab[static_cast<std::size_t>(e)] : shx;
      std::uint64_t eb = 0;
      for (int pj = 0; pj < 3; ++pj) {
        const Mat& m = M[static_cast<std::size_t>(3 * e + pj)];
        void* d = g::dalloc(m.bytes.size());
        CHECK(d && g::to_device(d, m.bytes.data(), m.bytes.size()));
        dmem.push_back(d);
        X.w[pj] = static_cast<const std::uint8_t*>(d);
        X.type[pj] = static_cast<std::uint32_t>(types[(e + pj) % 3]);
        X.rb[pj] = m.rb;
        row_max = std::max<std::uint64_t>(row_max, m.rb);
        eb += m.bytes.size();
      }
      X.vram = e < 2 || e == ne ? 1u : 0u;  // experts 0, 1 and the shared expert "in the bank"; 2-5 "from the file"
      ebytes_max = std::max(ebytes_max, eb);
    }
    Dev<g::ExpertPtr> dtab(tab), dshx(std::vector<g::ExpertPtr>{shx});
    const int K = 4;
    const std::uint64_t slot = (ebytes_max + 255) & ~std::uint64_t{255};
    Dev<std::uint8_t> cmem(static_cast<std::size_t>(K) * slot);
    Dev<int> ctag(std::vector<int>(K, -1));
    Dev<unsigned> cuse(std::vector<unsigned>(K, 0u)), cst(std::vector<unsigned>(1, 0u));
    g::ReaderCache rc;
    rc.K = K;
    rc.tag = ctag.p;
    rc.use = cuse.p;
    rc.stamp = cst.p;
    rc.base = cmem.p;
    rc.slot_bytes = slot;
    for (const std::int64_t T : {std::int64_t{1}, std::int64_t{2}, std::int64_t{4}}) {
      const std::int64_t P = T * (nu + 1);
      const std::vector<float> x = wave(T * H, static_cast<std::uint32_t>(17 + T + F), 1.0f);
      std::vector<std::int32_t> sel(static_cast<std::size_t>(T * nu));
      std::vector<float> wsel(static_cast<std::size_t>(T * nu)), gshv(static_cast<std::size_t>(T));
      for (std::int64_t t = 0; t < T; ++t) {
        gshv[static_cast<std::size_t>(t)] = 0.3f + 0.1f * static_cast<float>(t);
        for (int j = 0; j < nu; ++j) {
          sel[static_cast<std::size_t>(t * nu + j)] = static_cast<std::int32_t>((t + 2 * j + (j == 2 ? 1 : 0)) % ne);
          wsel[static_cast<std::size_t>(t * nu + j)] = 0.2f + 0.15f * static_cast<float>(j);
        }
      }
      Dev<float> dx(x), dw(wsel), dg(gshv), dgu(static_cast<std::size_t>(P * 2 * F)), dh(static_cast<std::size_t>(P * F)), dyd(static_cast<std::size_t>(P * H)),
          y24(static_cast<std::size_t>(T * H)), y3(static_cast<std::size_t>(T * H)), yc1(static_cast<std::size_t>(T * H)), yc2(static_cast<std::size_t>(T * H)),
          ysh(static_cast<std::size_t>(T * H));
      Dev<std::int32_t> ds(sel);
      Dev<int> dgrp(static_cast<std::size_t>(1 + 6 * P));
      Dev<g::ExpertPtr> gsrc(static_cast<std::size_t>(P)), gfill(static_cast<std::size_t>(P));
      CHECK(g::moe_reader_v24(dtab.p, ds.p, dw.p, nu, dx.p, T, H, F, dgu.p, dh.p, y24.p));
      CHECK(g::moe_reader3(dtab.p, nullptr, ds.p, dw.p, nullptr, nu, dx.p, T, H, F, row_max, nullptr, dgu.p, dh.p, dyd.p, dgrp.p, gsrc.p, gfill.p, y3.p));
      CHECK(g::moe_reader3(dtab.p, nullptr, ds.p, dw.p, nullptr, nu, dx.p, T, H, F, row_max, &rc, dgu.p, dh.p, dyd.p, dgrp.p, gsrc.p, gfill.p, yc1.p));
      CHECK(g::moe_reader3(dtab.p, nullptr, ds.p, dw.p, nullptr, nu, dx.p, T, H, F, row_max, &rc, dgu.p, dh.p, dyd.p, dgrp.p, gsrc.p, gfill.p, yc2.p));
      CHECK(g::sync());
      const std::vector<g::ExpertPtr> src2 = gsrc.get();
      const std::vector<int> grp2 = dgrp.get();
      // the second cached call: every group from VRAM if the step's file experts fit in the K slots
      int file_groups = 0, from_vram = 0;
      for (int u = 0; u < grp2[0]; ++u) {
        const int ex = grp2[static_cast<std::size_t>(1 + 6 * u)];
        if (ex >= 2) ++file_groups;
        if (ex >= 2 && src2[static_cast<std::size_t>(u)].vram) ++from_vram;
      }
      if (file_groups <= K) cache_hits_ok = cache_hits_ok && from_vram == file_groups;
      CHECK(g::moe_reader3(dtab.p, dshx.p, ds.p, dw.p, dg.p, nu, dx.p, T, H, F, row_max, nullptr, dgu.p, dh.p, dyd.p, dgrp.p, gsrc.p, gfill.p, ysh.p));
      // v27 plane test: tail 0 is the plain reader bit for bit; tail 1 changes the output (the last pick's Q4_TG2 rows on the
      // 2-bit base plane), finite and close
      Dev<float> yt0(static_cast<std::size_t>(T * H)), yt1(static_cast<std::size_t>(T * H));
      CHECK(g::moe_reader3(dtab.p, nullptr, ds.p, dw.p, nullptr, nu, dx.p, T, H, F, row_max, nullptr, dgu.p, dh.p, dyd.p, dgrp.p, gsrc.p, gfill.p, yt0.p, nullptr,
                           nullptr, 0));
      CHECK(g::moe_reader3(dtab.p, nullptr, ds.p, dw.p, nullptr, nu, dx.p, T, H, F, row_max, nullptr, dgu.p, dh.p, dyd.p, dgrp.p, gsrc.p, gfill.p, yt1.p, nullptr,
                           nullptr, 1));
      CHECK(g::sync());
      {
        const std::vector<float> a0 = yt0.get(), a1 = yt1.get(), ref3 = y3.get();
        CHECK(std::memcmp(a0.data(), ref3.data(), a0.size() * sizeof(float)) == 0);
        double dn = 0.0, dd = 0.0;
        bool fin = true;
        for (std::size_t i = 0; i < a0.size(); ++i) {
          fin = fin && std::isfinite(a1[i]);
          dn += (static_cast<double>(a1[i]) - a0[i]) * (static_cast<double>(a1[i]) - a0[i]);
          dd += static_cast<double>(a0[i]) * a0[i];
        }
        const double rel = std::sqrt(dn / std::max(dd, 1e-300));
        plane_ok = plane_ok && fin && rel > 0.0 && rel < 0.5;
        worst_plane = std::max(worst_plane, rel);
      }
      const std::vector<float> a24 = y24.get(), a3 = y3.get(), c1 = yc1.get(), c2 = yc2.get(), sh = ysh.get();
      const std::size_t nb = a24.size() * sizeof(float);
      eq_all = eq_all && std::memcmp(a24.data(), a3.data(), nb) == 0;
      cache_eq = cache_eq && std::memcmp(a24.data(), c1.data(), nb) == 0 && std::memcmp(a24.data(), c2.data(), nb) == 0;
      // the shared term: sh - routed = gsh x the shared expert (fp64, the per-32 rounding of silu x up as the reader)
      double num = 0.0, den = 0.0;
      for (std::int64_t t = 0; t < T; ++t) {
        const Mat &mg = M[static_cast<std::size_t>(3 * ne)], &mu = M[static_cast<std::size_t>(3 * ne + 1)], &md = M[static_cast<std::size_t>(3 * ne + 2)];
        std::vector<float> hh(static_cast<std::size_t>(F));
        for (std::int64_t b = 0; b < F / 32; ++b) {
          float v[32], am = 0.0f;
          for (int i = 0; i < 32; ++i) {
            const std::int64_t r = 32 * b + i;
            double gg = 0.0, uu = 0.0;
            for (std::int64_t k = 0; k < H; ++k) {
              gg += static_cast<double>(mg.deq[static_cast<std::size_t>(r * H + k)]) * x[static_cast<std::size_t>(t * H + k)];
              uu += static_cast<double>(mu.deq[static_cast<std::size_t>(r * H + k)]) * x[static_cast<std::size_t>(t * H + k)];
            }
            const float gf = static_cast<float>(gg);
            v[i] = (gf / (1.0f + std::exp(-gf))) * static_cast<float>(uu);
            am = std::max(am, std::fabs(v[i]));
          }
          const float sc = am > 0.0f ? am / 127.0f : 1.0f, inv = 1.0f / sc;
          for (int i = 0; i < 32; ++i) hh[static_cast<std::size_t>(32 * b + i)] = sc * std::min(127.0f, std::max(-127.0f, std::rint(v[i] * inv)));
        }
        for (std::int64_t r = 0; r < H; ++r) {
          double acc = 0.0;
          for (std::int64_t k = 0; k < F; ++k) acc += static_cast<double>(md.deq[static_cast<std::size_t>(r * F + k)]) * hh[static_cast<std::size_t>(k)];
          const double want = gshv[static_cast<std::size_t>(t)] * acc;
          const double got = static_cast<double>(sh[static_cast<std::size_t>(t * H + r)]) - a3[static_cast<std::size_t>(t * H + r)];
          num += (got - want) * (got - want);
          den += want * want;
        }
      }
      const double rel = std::sqrt(num / std::max(den, 1e-300));
      worst_sh = std::max(worst_sh, rel);
      shared_ok = shared_ok && std::isfinite(rel) && rel <= 2e-3;
    }
    for (void* d : dmem) g::dfree(d);
  }
  CHECK(eq_all);
  CHECK(cache_eq);
  CHECK(cache_hits_ok);
  CHECK(shared_ok);
  CHECK(plane_ok);
  std::printf("    plane test (last pick's Q4_TG2 rows on the 2-bit base plane): tail 0 bit for bit; tail 1 changes the output by up to %.3e (rel L2)\n", worst_plane);
  std::printf("    F 256 (half-warp down) / 1024 (warp down), 1 / 2 / 4 tokens, q8_0 / q6_k / q4_tg2: v3 = v24 %s; cached (fill, then read) %s, "
              "second call from VRAM %s; shared expert folded in: worst rel L2 %.3e against fp64\n",
              eq_all ? "bit for bit" : "DIFFERS", cache_eq ? "bit for bit" : "DIFFERS", cache_hits_ok ? "yes" : "NO", worst_sh);
}

TEST(rung1_attention_against_fp64) {
  if (!gpu_or_skip()) return;
  const std::int64_t nh = 16, nkv = 2, hd = 256, qh = 2 * hd, qs = nh * qh, kv_w = nkv * hd;
  for (const std::int64_t p0 : {std::int64_t{0}, std::int64_t{37}}) {  // a prompt from position 0, and 5 tokens after 37 cached
    const std::int64_t n_tok = 5, L = p0 + n_tok;
    const std::vector<float> q = wave(n_tok * qs, 71u, 1.0f), K = wave(L * kv_w, 72u, 1.0f), V = wave(L * kv_w, 73u, 1.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    Dev<float> dq(q), dk(K), dvv(V), dout(static_cast<std::size_t>(n_tok * nh * hd)), dsc(static_cast<std::size_t>(n_tok * nh * L));
    CHECK(g::attention(dq.p, qs, qh, dk.p, dvv.p, kv_w, nh, nkv, hd, n_tok, p0, scale, true, dout.p, dsc.p));
    CHECK(g::sync());
    const std::vector<float> out = dout.get();
    std::vector<double> ref(out.size());
    for (std::int64_t t = 0; t < n_tok; ++t)
      for (std::int64_t h = 0; h < nh; ++h) {
        const std::int64_t kvh = h / (nh / nkv), last = p0 + t;
        std::vector<double> s(static_cast<std::size_t>(last + 1));
        double mx = -1e300;
        for (std::int64_t u = 0; u <= last; ++u) {
          double d = 0.0;
          for (std::int64_t i = 0; i < hd; ++i)
            d += static_cast<double>(q[static_cast<std::size_t>(t * qs + h * qh + i)]) * K[static_cast<std::size_t>(u * kv_w + kvh * hd + i)];
          s[static_cast<std::size_t>(u)] = d * scale;
          mx = std::max(mx, s[static_cast<std::size_t>(u)]);
        }
        double sum = 0.0;
        for (double& e : s) sum += e = std::exp(e - mx);
        for (std::int64_t i = 0; i < hd; ++i) {
          double o = 0.0;
          for (std::int64_t u = 0; u <= last; ++u) o += s[static_cast<std::size_t>(u)] / sum * V[static_cast<std::size_t>(u * kv_w + kvh * hd + i)];
          const double gte = q[static_cast<std::size_t>(t * qs + h * qh + hd + i)];
          ref[static_cast<std::size_t>(t * nh * hd + h * hd + i)] = o / (1.0 + std::exp(-gte));
        }
      }
    const double e = rel_l2(out, ref);
    CHECK(all_finite(out));
    CHECK(e <= 5e-4);
    std::printf("    16 heads / 2 KV heads x 256, gated, %lld tokens after %lld cached: relative L2 %.3e (tolerance 5e-4)\n",
                static_cast<long long>(n_tok), static_cast<long long>(p0), e);
  }
}

TEST(rung1_deltanet_window_against_fp64) {
  if (!gpu_or_skip()) return;
  // the model's DeltaNet shape: 16 key heads, 32 value heads, d_state 128, conv taps 4
  const std::int64_t nk = 16, nv = 32, ds = 128, taps = 4, ch = (2 * nk + nv) * ds, d_inner = nv * ds, n = 40;
  const std::vector<float> mixed = wave(n * ch, 81u, 1.0f), cw = wave(ch * taps, 82u, 0.5f), z = wave(n * d_inner, 83u, 1.0f),
                           nw = wave(ds, 84u, 1.0f);
  std::vector<float> beta = wave(n * nv, 85u, 2.0f), alpha = wave(n * nv, 86u, 2.0f), dt = wave(nv, 87u, 1.0f), a = wave(nv, 88u, 1.0f);
  for (float& v : a) v = -std::fabs(v) - 0.1f;  // A < 0: decay below 1
  const float eps = 1e-6f;
  // GPU: the whole window in one call, and the same window as 25 + 15 tokens with the state and conv tail carried
  auto run = [&](std::int64_t split, std::vector<float>& core_out) {
    Dev<float> dm(mixed), dcw(cw), dz(z), dnw(nw), db(beta), da(alpha), ddt(dt), dA(a);
    Dev<float> conv(static_cast<std::size_t>(n * ch)), core(static_cast<std::size_t>(n * d_inner)), state(static_cast<std::size_t>(nv * ds * ds)),
        tail1(static_cast<std::size_t>((taps - 1) * ch));
    CHECK(g::dzero(state.p, state.n * sizeof(float)));
    CHECK(g::deltanet_gates(db.p, da.p, ddt.p, dA.p, n, nv));
    const std::int64_t parts[2] = {split, n - split};
    std::int64_t t0 = 0;
    for (int pi = 0; pi < 2; ++pi) {
      const std::int64_t m = parts[pi];
      if (m == 0) continue;
      CHECK(g::deltanet_conv(dm.p + t0 * ch, pi == 0 ? nullptr : tail1.p, dcw.p, m, ch, taps, conv.p + t0 * ch, pi == 0 ? tail1.p : nullptr));
      CHECK(g::l2_norm(conv.p + t0 * ch, ds, 2 * nk, ds, ch, m, eps));
      CHECK(g::deltanet_recurrent(conv.p + t0 * ch, ch, db.p + t0 * nv, da.p + t0 * nv, state.p, m, nk, nv, ds, core.p + t0 * d_inner));
      t0 += m;
    }
    CHECK(g::gated_norm(core.p, dz.p, dnw.p, n, nv, ds, eps));
    CHECK(g::sync());
    core_out = core.get();
  };
  std::vector<float> whole, split;
  run(n, whole);
  run(25, split);
  CHECK(all_finite(whole));
  CHECK(bits_equal(whole, split));  // carrying the state and the conv tail changes nothing, bit for bit
  // fp64 reference of the layer window (model::delta_net from the products on)
  std::vector<double> conv(static_cast<std::size_t>(n * ch));
  for (std::int64_t t = 0; t < n; ++t)
    for (std::int64_t c = 0; c < ch; ++c) {
      double s = 0.0;
      for (std::int64_t kk = 0; kk < taps; ++kk) {
        const std::int64_t src = t + kk - (taps - 1);
        if (src >= 0) s += static_cast<double>(mixed[static_cast<std::size_t>(src * ch + c)]) * cw[static_cast<std::size_t>(c * taps + kk)];
      }
      conv[static_cast<std::size_t>(t * ch + c)] = s / (1.0 + std::exp(-s));
    }
  for (std::int64_t t = 0; t < n; ++t)
    for (std::int64_t h = 0; h < 2 * nk; ++h) {
      double ss = 0.0;
      for (std::int64_t i = 0; i < ds; ++i) ss += conv[static_cast<std::size_t>(t * ch + h * ds + i)] * conv[static_cast<std::size_t>(t * ch + h * ds + i)];
      const double sc = 1.0 / std::sqrt(ss + static_cast<double>(eps));
      for (std::int64_t i = 0; i < ds; ++i) conv[static_cast<std::size_t>(t * ch + h * ds + i)] *= sc;
    }
  std::vector<double> core(static_cast<std::size_t>(n * d_inner));
  for (std::int64_t hv = 0; hv < nv; ++hv) {
    std::vector<double> M(static_cast<std::size_t>(ds * ds), 0.0), delta(static_cast<std::size_t>(ds));
    const std::int64_t hk = hv % nk;
    for (std::int64_t t = 0; t < n; ++t) {
      const double* qv = conv.data() + t * ch + hk * ds;
      const double* kv = conv.data() + t * ch + nk * ds + hk * ds;
      const double* vv = conv.data() + t * ch + 2 * nk * ds + hv * ds;
      const double b = 1.0 / (1.0 + std::exp(-static_cast<double>(beta[static_cast<std::size_t>(t * nv + hv)])));
      const double x = static_cast<double>(alpha[static_cast<std::size_t>(t * nv + hv)]) + dt[static_cast<std::size_t>(hv)];
      const double sp = x > 20.0 ? x : std::log(1.0 + std::exp(x));
      const double decay = std::exp(sp * a[static_cast<std::size_t>(hv)]);
      for (double& m : M) m *= decay;
      for (std::int64_t j = 0; j < ds; ++j) {
        double s = 0.0;
        for (std::int64_t i = 0; i < ds; ++i) s += M[static_cast<std::size_t>(j * ds + i)] * kv[i];
        delta[static_cast<std::size_t>(j)] = (vv[j] - s) * b;
      }
      for (std::int64_t j = 0; j < ds; ++j)
        for (std::int64_t i = 0; i < ds; ++i) M[static_cast<std::size_t>(j * ds + i)] += kv[i] * delta[static_cast<std::size_t>(j)];
      for (std::int64_t j = 0; j < ds; ++j) {
        double s = 0.0;
        for (std::int64_t i = 0; i < ds; ++i) s += M[static_cast<std::size_t>(j * ds + i)] * qv[i];
        core[static_cast<std::size_t>(t * d_inner + hv * ds + j)] = s / std::sqrt(static_cast<double>(ds));
      }
    }
  }
  for (std::int64_t t = 0; t < n; ++t)
    for (std::int64_t h = 0; h < nv; ++h) {
      double ss = 0.0;
      for (std::int64_t i = 0; i < ds; ++i) ss += core[static_cast<std::size_t>(t * d_inner + h * ds + i)] * core[static_cast<std::size_t>(t * d_inner + h * ds + i)];
      const double sc = 1.0 / std::sqrt(ss / static_cast<double>(ds) + static_cast<double>(eps));
      for (std::int64_t i = 0; i < ds; ++i) {
        const double zz = z[static_cast<std::size_t>(t * d_inner + h * ds + i)];
        double& o = core[static_cast<std::size_t>(t * d_inner + h * ds + i)];
        o = o * sc * nw[static_cast<std::size_t>(i)] * (zz / (1.0 + std::exp(-zz)));
      }
    }
  const double e = rel_l2(whole, core);
  CHECK(e <= 1e-4);
  std::printf("    40 tokens, 32 value heads x 128: relative L2 vs fp64 %.3e (tolerance 1e-4); 25 + 15 with carried state = one call, bit for bit\n", e);
}
