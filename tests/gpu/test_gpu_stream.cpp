// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// Rung 1 of the R3 streaming kernel moved into src (kern/gpu_stream.h; masterwork 16.1 / 16.2): one token's table with
// every format the kernel reads (Q7_ZT, Q4_TG2, Q5_K, Q6_K; Q8_0, Q4_K, IQ4_XS since 2026-10-30), every lane count (k 256 to 4096: 2 to 32 lanes per row), three
// stages, and per-expert waits inside a stage (two gate/up jobs signal their groups, two down jobs wait for them), against
// an fp64 host reference of the same rounded activations (int8 x scale per 32, from act_q8x32). The same table run a second
// time (the counters are never reset: epoch 2) and run unstaged must give the same bits. The plan refuses tables it cannot
// run safely. Run on the card under compute-sanitizer by the PC script.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "../unit/gpu_test_rows.h"
#include "kern/gpu_ops.h"
#include "kern/gpu_stream.h"
#include "kern/tg2_planes.h"
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

// the rounded activation of one input width k: device int8 / scale / sum and the host copy of q x scale
struct Act {
  std::int64_t k;
  Dev<float> x, xr, s;
  Dev<std::int8_t> qv;
  Dev<int> sum;
  std::vector<double> eff;  // q[i] * scale[i / 32], the values the kernel multiplies by
  Act(std::int64_t kk, std::uint32_t seed)
      : k(kk), x(wave(kk, seed, 2.0f)), xr(static_cast<std::size_t>(kk)), s(static_cast<std::size_t>(kk / 32)),
        qv(static_cast<std::size_t>(kk)), sum(static_cast<std::size_t>(kk / 32)) {
    CHECK(g::act_q8x32(x.p, k, 1, k, false, xr.p, k, qv.p, s.p, sum.p));
    CHECK(g::sync());
    const std::vector<std::int8_t> hq = qv.get();
    const std::vector<float> hs = s.get();
    eff.resize(static_cast<std::size_t>(k));
    for (std::int64_t i = 0; i < k; ++i)
      eff[static_cast<std::size_t>(i)] = static_cast<double>(hq[static_cast<std::size_t>(i)]) * static_cast<double>(hs[static_cast<std::size_t>(i / 32)]);
  }
};

// a matrix: GGUF rows, their exact floats, the device copy in the streaming layout
struct Mat {
  q::Type t;
  std::int64_t n, k;
  std::vector<float> deq;
  std::vector<std::uint8_t> bytes;  // v60 E: the GGUF rows (the head sketch test rebuilds the planes from them on the host)
  g::StreamWeights w;
  Mat(q::Type tt, std::int64_t nn, std::int64_t kk, std::uint32_t seed) : t(tt), n(nn), k(kk) {
    const auto id = static_cast<std::uint32_t>(t);
    const std::uint64_t rb = q::row_bytes(id, static_cast<std::uint64_t>(k));
    bytes.resize(static_cast<std::size_t>(rb * static_cast<std::uint64_t>(n)));
    deq.resize(static_cast<std::size_t>(n * k));
    for (std::int64_t r = 0; r < n; ++r) {
      const std::uint32_t rs = seed + static_cast<std::uint32_t>(r) * 7919u;
      const std::vector<std::uint8_t> row = pelias::test_gpu::make_row(t, wave(k, rs, 0.05f), rs);
      std::memcpy(bytes.data() + static_cast<std::size_t>(r) * rb, row.data(), static_cast<std::size_t>(rb));
      (void)q::dequantize_row(t, row.data(), deq.data() + r * k, k);
    }
    CHECK(g::stream_upload(id, bytes.data(), n, k, &w));
  }
  ~Mat() { g::stream_release(&w); }
  Mat(const Mat&) = delete;
  Mat& operator=(const Mat&) = delete;
};

double rel_l2_vs(const std::vector<float>& y, const Mat& m, const Act& a) {
  double num = 0.0, den = 0.0;
  for (std::int64_t r = 0; r < m.n; ++r) {
    double ref = 0.0;
    for (std::int64_t i = 0; i < m.k; ++i) ref += static_cast<double>(m.deq[static_cast<std::size_t>(r * m.k + i)]) * a.eff[static_cast<std::size_t>(i)];
    const double d = static_cast<double>(y[static_cast<std::size_t>(r)]) - ref;
    num += d * d;
    den += ref * ref;
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

bool finite(const std::vector<float>& v) {
  for (const float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

constexpr double kTol = 1e-6;  // products against fp64 of the same rounded inputs (masterwork 16.2)

}  // namespace

TEST(rung1_stream_r3_against_fp64) {
  CHECK(g::compiled());
  if (!g::available()) {
    std::printf("    no GPU: %s\n", g::last_error());
    CHECK(g::available());
    return;
  }
  Act a256(256, 3u), a512(512, 5u), a1024(1024, 13u), a2048(2048, 7u), a4096(4096, 11u);
  // stage 0: one of each format and lane count; stage 1: two experts' gate/up (signal) and down (wait); stage 2: a head
  Mat m0(q::Type::q7_zt, 300, 2048, 101u), m1(q::Type::q6_k, 97, 2048, 202u), m2(q::Type::q5_k, 65, 256, 303u), m3(q::Type::q4_tg2, 33, 4096, 404u),
      m4(q::Type::q4_tg2, 512, 2048, 505u), m5(q::Type::q4_tg2, 512, 2048, 606u), m6(q::Type::q5_k, 2048, 512, 707u),
      m7(q::Type::q5_k, 2048, 512, 808u), m8(q::Type::q6_k, 1000, 2048, 909u), m9(q::Type::q7_zt, 31, 1024, 111u);
  // stage 2 also carries the int8-path formats added 2026-10-30 (Q8_0, Q4_K, IQ4_XS), each at two lane counts
  Mat m10(q::Type::q8_0, 129, 1024, 1010u), m11(q::Type::q4_k, 77, 4096, 1111u), m12(q::Type::iq4_xs, 200, 512, 1212u),
      m13(q::Type::q8_0, 64, 256, 1313u), m14(q::Type::q4_k, 300, 2048, 1414u), m15(q::Type::iq4_xs, 45, 4096, 1515u);
  const Mat* mats[] = {&m0, &m1, &m2, &m3, &m9, &m4, &m5, &m6, &m7, &m8, &m10, &m11, &m12, &m13, &m14, &m15};
  const Act* acts[] = {&a2048, &a2048, &a256, &a4096, &a1024, &a2048, &a2048, &a512, &a512, &a2048, &a1024, &a4096, &a512, &a256, &a2048, &a4096};
  constexpr int nj = 16;
  std::vector<Dev<float>*> ys;
  std::vector<g::StreamJob> jobs(nj);
  for (int j = 0; j < nj; ++j) {
    ys.push_back(new Dev<float>(static_cast<std::size_t>(mats[j]->n)));
    g::StreamJob& J = jobs[static_cast<std::size_t>(j)];
    J.w = &mats[j]->w;
    J.xq = acts[j]->qv.p;
    J.xs = acts[j]->s.p;
    J.xsum = acts[j]->sum.p;
    J.y = ys.back()->p;
  }
  jobs[5].signal = 0;  // expert A gate/up
  jobs[6].signal = 1;  // expert B gate/up
  jobs[7].wait = 0;    // expert A down waits for A's gate/up only
  jobs[8].wait = 1;
  const int stage_end[] = {5, 9, 16};
  g::StreamPlan plan;
  CHECK(plan.build(jobs.data(), nj, stage_end, 3, 2));
  if (plan.jobs() == 0) {
    std::printf("    build failed: %s\n", g::stream_last_error());
    for (auto* y : ys) delete y;
    return;
  }
  CHECK(plan.run());
  CHECK(g::sync());
  std::vector<std::vector<float>> first(nj);
  double worst = 0.0;
  for (int j = 0; j < nj; ++j) {
    first[static_cast<std::size_t>(j)] = ys[static_cast<std::size_t>(j)]->get();
    const double e = rel_l2_vs(first[static_cast<std::size_t>(j)], *mats[j], *acts[j]);
    worst = std::max(worst, e);
    CHECK(finite(first[static_cast<std::size_t>(j)]));
    CHECK(e <= kTol);
    std::printf("    job %d %-6s %5lld x %4lld (%2lld lanes per row)%s: relative L2 vs fp64 %.3e\n", j, q::type_info(static_cast<std::uint32_t>(mats[j]->t)).name,
                static_cast<long long>(mats[j]->n), static_cast<long long>(mats[j]->k), static_cast<long long>(mats[j]->k / 128),
                jobs[static_cast<std::size_t>(j)].signal >= 0 ? " signals" : (jobs[static_cast<std::size_t>(j)].wait >= 0 ? " waits  " : "        "), e);
  }
  // run 2 (epoch 2) and the unstaged run: outputs cleared first, then the same bits
  for (int pass = 0; pass < 2; ++pass) {
    for (auto* y : ys) CHECK(g::dzero(y->p, y->n * sizeof(float)));
    CHECK(pass == 0 ? plan.run() : plan.run_unstaged());
    CHECK(g::sync());
    bool same = true;
    for (int j = 0; j < nj; ++j) {
      const std::vector<float> v = ys[static_cast<std::size_t>(j)]->get();
      same = same && std::memcmp(v.data(), first[static_cast<std::size_t>(j)].data(), v.size() * sizeof(float)) == 0;
    }
    CHECK(same);
    std::printf("    %s: %s\n", pass == 0 ? "second staged run (epoch 2)" : "unstaged run", same ? "bit for bit equal to the first" : "DIFFERENT");
  }
  std::printf("    %d jobs, %lld warp tasks, %llu weight bytes, 3 stages, 2 expert groups: worst relative L2 %.3e (tolerance %.0e)\n", plan.jobs(),
              plan.tasks(), static_cast<unsigned long long>(plan.bytes()), worst, kTol);
  for (auto* y : ys) delete y;
}

TEST(rung1_stream_refuses_unsafe_tables) {
  if (!g::available()) return;
  Act a(2048, 17u);
  Mat m(q::Type::q4_tg2, 64, 2048, 21u);
  Dev<float> y(64);
  g::StreamJob j0, j1;
  j0.w = j1.w = &m.w;
  j0.xq = j1.xq = a.qv.p;
  j0.xs = j1.xs = a.s.p;
  j0.xsum = j1.xsum = a.sum.p;
  j0.y = j1.y = y.p;
  {  // a wait on a group signalled later in the table (would deadlock)
    g::StreamJob js[2] = {j0, j1};
    js[0].wait = 0;
    js[1].signal = 0;
    const int se[] = {2};
    g::StreamPlan p;
    CHECK(!p.build(js, 2, se, 1, 1));
  }
  {  // a wait on a group nobody signals
    g::StreamJob js[1] = {j0};
    js[0].wait = 0;
    const int se[] = {1};
    g::StreamPlan p;
    CHECK(!p.build(js, 1, se, 1, 1));
  }
  {  // misaligned activation
    g::StreamJob js[1] = {j0};
    js[0].xq = a.qv.p + 1;
    const int se[] = {1};
    g::StreamPlan p;
    CHECK(!p.build(js, 1, se, 1, 0));
  }
  {  // stage ends that do not cover the table
    g::StreamJob js[2] = {j0, j1};
    const int se[] = {1};
    g::StreamPlan p;
    CHECK(!p.build(js, 2, se, 1, 0));
  }
  g::StreamWeights w;
  std::vector<std::uint8_t> junk(1024);
  CHECK(!g::stream_upload(static_cast<std::uint32_t>(q::Type::q4_0), junk.data(), 1, 256, &w));   // not a streaming type (Q4_K streams since 23bd80e)
  CHECK(!g::stream_upload(static_cast<std::uint32_t>(q::Type::q5_k), junk.data(), 1, 768, &w));   // k not a power of two
  std::printf("    deadlocking, unsignalled, misaligned and uncovered tables refused; unreadable types and widths refused\n");
}

// v60 E: the head sketch (kern/tg2_planes.h g_q6_ksk; gpu_ops.h head_sketch_*): (1) the planes built on the device equal the
// host builder's byte for byte; (2) the sketch product equals the host's run of the same integer formula (float sum order
// aside); (3) topk_select returns the exact set a host sort gives; (4) the gathered top rows, run through a fixed 64-row R3
// plan, equal the full plan's values for those rows bit for bit (the re-score is the head's own kernel on the same bytes);
// (5) the picked id is the re-scored argmax. Recall on random weights is printed, not gated: the real head's recall is the
// gate's number (the kept rate).
TEST(rung1_head_sketch_two_planes) {
  if (!g::available()) return;
  constexpr std::int64_t k = 2048, n = 1000;
  constexpr int top = 64;
  Mat m(q::Type::q6_k, n, k, 909u);
  const std::int64_t nbr = k / 256;
  const std::uint64_t srb = g::q6k_sketch_row_bytes(k);
  CHECK(srb == 640);
  // (1) planes
  Dev<std::uint8_t> sk(static_cast<std::size_t>(n) * srb);
  CHECK(g::head_sketch_build(m.w.w, n, k, sk.p));
  CHECK(g::sync());
  const std::vector<std::uint8_t> dsk = sk.get();
  std::vector<std::uint8_t> hsk(dsk.size());
  for (std::int64_t r = 0; r < n; ++r)
    for (std::int64_t b = 0; b < nbr; ++b) {
      std::uint32_t pl[16];
      std::uint8_t sc[16];
      // R3's 208-byte block = the GGUF block's first 208 bytes (ql, qh, scales); d apart
      g::q6k_sketch_block(m.bytes.data() + static_cast<std::size_t>((r * nbr + b) * 210), pl, sc);
      std::memcpy(hsk.data() + static_cast<std::size_t>(r) * srb + static_cast<std::size_t>(64 * b), pl, 64);
      std::memcpy(hsk.data() + static_cast<std::size_t>(r) * srb + static_cast<std::size_t>(64 * nbr + 16 * b), sc, 16);
    }
  const bool planes_equal = std::memcmp(dsk.data(), hsk.data(), dsk.size()) == 0;
  CHECK(planes_equal);
  // the full product (the reference the sketch approximates, and the bits the re-score must reproduce)
  Act a(k, 7u);
  Dev<float> yfull(static_cast<std::size_t>(n));
  {
    g::StreamJob J;
    J.w = &m.w;
    J.xq = a.qv.p;
    J.xs = a.s.p;
    J.xsum = a.sum.p;
    J.y = yfull.p;
    const int se[] = {1};
    g::StreamPlan plan;
    CHECK(plan.build(&J, 1, se, 1, 0));
    CHECK(plan.run_unstaged());
    CHECK(g::sync());
  }
  const std::vector<float> hfull = yfull.get();
  // (2) the sketch product vs the host's run of the same formula
  Dev<float> ysk(static_cast<std::size_t>(n));
  CHECK(g::head_sketch_dot(sk.p, static_cast<const std::uint16_t*>(m.w.d6), n, k, a.qv.p, a.s.p, ysk.p));
  CHECK(g::sync());
  const std::vector<float> hsk_y = ysk.get();
  const std::vector<std::int8_t> hq = a.qv.get();
  const std::vector<float> hs = a.s.get();
  std::vector<std::uint16_t> hd6(static_cast<std::size_t>(n * nbr));
  CHECK(g::to_host(hd6.data(), m.w.d6, hd6.size() * 2));
  const int nsub = static_cast<int>(k / 32);
  std::vector<std::uint32_t> u8(static_cast<std::size_t>(nsub) * 8);
  for (int sb = 0; sb < nsub; ++sb) g::act_bits32(hq.data() + 32 * sb, u8.data() + 8 * sb);
  double worst_rel = 0.0;
  for (std::int64_t r = 0; r < n; ++r) {
    const auto* pw = reinterpret_cast<const std::uint32_t*>(hsk.data() + static_cast<std::size_t>(r) * srb);
    const std::uint8_t* sc = hsk.data() + static_cast<std::size_t>(r) * srb + static_cast<std::size_t>(8 * nsub);
    double acc = 0.0;
    for (int sb = 0; sb < nsub; ++sb) {
      const std::uint32_t* u = u8.data() + 8 * sb;
      const int x0 = g::plane_dot32(0x0000ffffu, u), x1 = g::plane_dot32(0xffff0000u, u);
      const int t0 = g::q6k_sketch_half2(pw[2 * sb], pw[2 * sb + 1], u, 0, x0), t1 = g::q6k_sketch_half2(pw[2 * sb], pw[2 * sb + 1], u, 1, x1);
      const std::uint16_t dh = hd6[static_cast<std::size_t>(r * nbr + sb / 8)];
      const float d = q::fp16_to_fp32(dh);
      acc += static_cast<double>(g::q6k_sketch_sub(d, hs[static_cast<std::size_t>(sb)], static_cast<std::int8_t>(sc[2 * sb]), static_cast<std::int8_t>(sc[2 * sb + 1]), t0, t1));
    }
    const double dev = std::fabs(static_cast<double>(hsk_y[static_cast<std::size_t>(r)]) - acc), mag = std::max(1e-6, std::fabs(acc));
    worst_rel = std::max(worst_rel, dev / mag);
  }
  CHECK(worst_rel <= 1e-4);
  // (3) top-64 of the sketch logits: the exact set
  Dev<std::int32_t> cand(static_cast<std::size_t>(top));
  CHECK(g::topk_select(ysk.p, n, top, cand.p));
  CHECK(g::sync());
  std::vector<std::int32_t> hc = cand.get(), order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int i, int j) { return hsk_y[static_cast<std::size_t>(i)] > hsk_y[static_cast<std::size_t>(j)]; });
  std::vector<std::int32_t> want(order.begin(), order.begin() + top);
  std::sort(want.begin(), want.end());
  std::vector<std::int32_t> got = hc;
  std::sort(got.begin(), got.end());
  const bool set_equal = got == want;  // ties exactly at the 64th place would make this order-dependent; random floats have none
  CHECK(set_equal);
  // (4) gather + the fixed 64-row plan = the full plan's rows, bit for bit
  Dev<std::uint8_t> tw(static_cast<std::size_t>(top * nbr * 208));
  Dev<std::uint16_t> td(static_cast<std::size_t>(top * nbr));
  CHECK(g::gather_q6k_rows(m.w.w, static_cast<const std::uint16_t*>(m.w.d6), k, cand.p, top, tw.p, td.p));
  g::StreamWeights wt;
  wt.type = m.w.type;
  wt.n = top;
  wt.k = k;
  wt.w = tw.p;
  wt.d6 = td.p;
  wt.bytes = static_cast<std::uint64_t>(top * nbr) * 210;
  Dev<float> ytop(static_cast<std::size_t>(top));
  {
    g::StreamJob J;
    J.w = &wt;
    J.xq = a.qv.p;
    J.xs = a.s.p;
    J.xsum = a.sum.p;
    J.y = ytop.p;
    const int se[] = {1};
    g::StreamPlan plan;
    CHECK(plan.build(&J, 1, se, 1, 0));
    CHECK(plan.run_unstaged());
    CHECK(g::sync());
  }
  const std::vector<float> htop = ytop.get();
  bool rows_equal = true;
  for (int i = 0; i < top; ++i) rows_equal = rows_equal && std::memcmp(&htop[static_cast<std::size_t>(i)], &hfull[static_cast<std::size_t>(hc[static_cast<std::size_t>(i)])], 4) == 0;
  CHECK(rows_equal);
  // (5) the pick: argmax over the 64 re-scored values, mapped back
  Dev<std::int32_t> idx(1), pick(1);
  CHECK(g::argmax_rows(ytop.p, top, 1, top, idx.p));
  CHECK(g::pick_id(cand.p, idx.p, pick.p));
  CHECK(g::sync());
  const std::int32_t picked = pick.get()[0];
  int best = 0;
  for (int i = 1; i < top; ++i)
    if (htop[static_cast<std::size_t>(i)] > htop[static_cast<std::size_t>(best)]) best = i;
  CHECK(picked == hc[static_cast<std::size_t>(best)]);
  // recall on these random rows (information): is the full product's argmax among the sketch's top 64?
  int full_best = 0;
  for (std::int64_t i = 1; i < n; ++i)
    if (hfull[static_cast<std::size_t>(i)] > hfull[static_cast<std::size_t>(full_best)]) full_best = static_cast<int>(i);
  const bool recall = std::find(hc.begin(), hc.end(), full_best) != hc.end();
  Dev<unsigned> cnt(2);
  CHECK(g::dzero(cnt.p, 8));
  Dev<std::int32_t> truth(std::vector<std::int32_t>{full_best});
  CHECK(g::recall_count(cand.p, top, truth.p, cnt.p, cnt.p + 1));
  CHECK(g::sync());
  const std::vector<unsigned> hcnt = cnt.get();
  CHECK(hcnt[1] == 1 && (hcnt[0] == 1) == recall);
  std::printf("    Q6_K %lld x %lld: sketch planes %s the host builder (%llu B per row); sketch product vs the host formula worst rel %.2e; top-%d set %s; "
              "gathered rows through the 64-row plan %s the full plan; pick = row %d (%s); full argmax %d %s the sketch's top %d (random rows: information only)\n",
              static_cast<long long>(n), static_cast<long long>(k), planes_equal ? "equal" : "DIFFER from", static_cast<unsigned long long>(srb), worst_rel, top,
              set_equal ? "exact" : "WRONG", rows_equal ? "bit for bit equal to" : "DIFFER from", picked, picked == hc[static_cast<std::size_t>(best)] ? "ok" : "WRONG",
              full_best, recall ? "inside" : "OUTSIDE", top);
}
