// provenance: original (the layer order and every step are forward_impl's in model/arch_qwen35moe.cpp, as GpuForward runs
// them, with the state carried between calls as forward_decode carries it)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// gpu_engine.cpp - see gpu_engine.h.
#include "model/gpu_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "hal/host_alloc_pages.h"
#include "kern/cpu_dot.h"
#include "kern/gpu_ops.h"
#include "kern/gpu_decode.h"
#include "kern/gpu_decode.h"
#include "kern/gpu_prompt.h"
#include "kern/gpu_stream.h"
#include "quant/block_types.h"
#include "quant/quant_cpu.h"

namespace pelias::model {

namespace g = pelias::kern::gpu;
using std::int64_t;

namespace {

using Clock = std::chrono::steady_clock;
inline double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }
inline std::size_t z(int64_t v) { return static_cast<std::size_t>(v); }
inline std::size_t al(std::size_t x) { return (x + 255) / 256 * 256; }
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }  // the reference's expression

bool is_float_type(std::uint32_t t) {
  return t == static_cast<std::uint32_t>(quant::Type::f32) || t == static_cast<std::uint32_t>(quant::Type::f16) ||
         t == static_cast<std::uint32_t>(quant::Type::bf16);
}
Status gfail(const std::string& what) { return Status::fail(Code::internal, what + ": " + g::last_error()); }

// A persistent pool (game plan CPU item 1): the workers live as long as the engine. A job hands out items 0 .. n-1 by an
// atomic counter and the calling thread works too. Between jobs a worker spins briefly (kSpin checks, about tens of
// microseconds: the cold path runs several jobs back to back per layer) and then sleeps on a condition variable, so the
// CPU is not held busy while the GPU runs a layer (the author's heat concern, 2026-10-31).
class Pool {
 public:
  explicit Pool(int threads) {
    const int workers = std::max(0, threads - 1);
    for (int i = 0; i < workers; ++i) th_.emplace_back([this] { loop(); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
    for (std::thread& t : th_) t.join();
  }
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;
  [[nodiscard]] int threads() const { return static_cast<int>(th_.size()) + 1; }
  // v19 timeline: summed item time over every thread, and the first worker's start after the job opened (-1: none)
  struct RunStats {
    std::atomic<std::int64_t> busy_ns{0};
    std::atomic<std::int64_t> wake_ns{-1};
  };
  // v28b (author 23:14): while hot, idle workers spin (no sleep, no OS wake-up) so a split step's per-layer jobs start at once
  void set_hot(bool h) {
    hot_.store(h, std::memory_order_release);
    if (h) {  // v29 fix: workers that went to sleep before the step must be woken to start spinning
      std::lock_guard<std::mutex> lk(mu_);
      cv_.notify_all();
    }
  }
  // v28b: two phases in one dispatch: items 0 .. n1-1 run fn1, then (each worker waits until every phase-1 item is done)
  // items 0 .. n2-1 run fn2. One generation bump per call: one wake-up per layer instead of one per stage.
  void run2(int64_t n1, const std::function<void(int64_t)>& fn1, int64_t n2, const std::function<void(int64_t)>& fn2) {
    if (n1 + n2 <= 0) return;
    rs_ = nullptr;
    t0_ = Clock::now();
    if (th_.empty()) {
      for (int64_t i = 0; i < n1; ++i) fn1(i);
      for (int64_t i = 0; i < n2; ++i) fn2(i);
      return;
    }
    fn_ = &fn1;
    fn2_ = &fn2;
    n_ = n1;
    n2_ = n2;
    done1_.store(0, std::memory_order_relaxed);
    next_.store(0, std::memory_order_relaxed);
    active_.store(static_cast<int>(th_.size()), std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(mu_);
      gen_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_all();  // v29 fix: always (v29 skipped it when hot; a worker asleep from before the step then never woke: S4 hung)
    work2(true);
    for (int spin = 0; active_.load(std::memory_order_acquire) != 0; ++spin)
      if (spin > kSpin && !hot_.load(std::memory_order_relaxed)) std::this_thread::yield();
    fn2_ = nullptr;
    n2_ = 0;
  }
  void run(int64_t n, const std::function<void(int64_t)>& fn, RunStats* rs = nullptr) {
    if (n <= 0) return;
    rs_ = rs;
    t0_ = Clock::now();
    if (th_.empty() || n == 1) {
      work(fn, n, false);
      return;
    }
    fn_ = &fn;
    fn2_ = nullptr;
    n_ = n;
    next_.store(0, std::memory_order_relaxed);
    active_.store(static_cast<int>(th_.size()), std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(mu_);  // a sleeping worker sees the new generation under the lock
      gen_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_all();
    work(fn, n, false);
    for (int spin = 0; active_.load(std::memory_order_acquire) != 0; ++spin)
      if (spin > kSpin) std::this_thread::yield();
  }

 private:
  static constexpr int kSpin = 4000;
  static void relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
  }
  // v28b: the phased job's items (phase 2 waits for every phase-1 item)
  void work2(bool) {
    const int64_t n1 = n_, n2 = n2_;
    for (int64_t i = next_.fetch_add(1, std::memory_order_relaxed); i < n1 + n2; i = next_.fetch_add(1, std::memory_order_relaxed)) {
      if (i < n1) {
        (*fn_)(i);
        done1_.fetch_add(1, std::memory_order_acq_rel);
      } else {
        while (done1_.load(std::memory_order_acquire) < n1) relax();
        (*fn2_)(i - n1);
      }
    }
  }
  static std::int64_t ns(Clock::duration d) { return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count(); }
  void work(const std::function<void(int64_t)>& fn, int64_t n, bool worker) {
    RunStats* rs = rs_;
    if (!worker && (th_.empty() || n == 1)) {  // inline job: the caller alone, no counter
      for (int64_t i = 0; i < n; ++i) {
        const Clock::time_point t = Clock::now();
        fn(i);
        if (rs) rs->busy_ns.fetch_add(ns(Clock::now() - t), std::memory_order_relaxed);
      }
      return;
    }
    for (int64_t i = next_.fetch_add(1, std::memory_order_relaxed); i < n; i = next_.fetch_add(1, std::memory_order_relaxed)) {
      if (!rs) {
        fn(i);
        continue;
      }
      const Clock::time_point t = Clock::now();
      if (worker) {
        std::int64_t none = -1;
        rs->wake_ns.compare_exchange_strong(none, ns(t - t0_), std::memory_order_relaxed);
      }
      fn(i);
      rs->busy_ns.fetch_add(ns(Clock::now() - t), std::memory_order_relaxed);
    }
  }
  void loop() {
    std::uint64_t seen = 0;
    for (;;) {
      bool got = false;
      for (int spin = 0; !got; ++spin) {
        got = gen_.load(std::memory_order_acquire) != seen || stop_.load(std::memory_order_acquire);
        if (got) break;
        if (spin >= kSpin && !hot_.load(std::memory_order_acquire)) break;  // v28b: a hot pool never sleeps
        if (spin >= kSpin) relax();
      }
      if (!got) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_.load(std::memory_order_acquire) || gen_.load(std::memory_order_acquire) != seen; });
      }
      if (stop_.load(std::memory_order_acquire)) return;
      seen = gen_.load(std::memory_order_acquire);
      if (fn2_) work2(true);
      else work(*fn_, n_, true);
      active_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }
  std::vector<std::thread> th_;
  std::mutex mu_;
  std::condition_variable cv_;
  const std::function<void(int64_t)>* fn_ = nullptr;
  const std::function<void(int64_t)>* fn2_ = nullptr;  // v28b: phase 2 of a run2 job (nullptr: a plain job)
  int64_t n_ = 0, n2_ = 0;
  std::atomic<int64_t> done1_{0};
  std::atomic<bool> hot_{false};
  RunStats* rs_ = nullptr;  // written before the job's generation is released; read by the workers after they see it
  Clock::time_point t0_{};
  std::atomic<int64_t> next_{0};
  std::atomic<int> active_{0};
  std::atomic<std::uint64_t> gen_{0};
  std::atomic<bool> stop_{false};
};

// a bump allocator over one device block
struct Arena {
  std::uint8_t* base = nullptr;
  std::size_t cap = 0, used = 0;
  void* take(std::size_t bytes) {
    const std::size_t at = al(used);
    if (at + bytes > cap) return nullptr;
    used = at + bytes;
    return base + at;
  }
};

struct DevW {
  const WeightView* v = nullptr;
  const void* d = nullptr;
  int64_t n_out = 0, k = 0;
  bool int8 = false;
  g::PromptWeights pw;
  std::uint32_t dtype = 0;   // the type the batched kernels read d as (g_q8_0r for Q8_0 held in R3's layout)
  std::uint64_t drb = 0;     // its row bytes
  bool r3 = false;           // sw holds this weight for R3 (one-token steps)
  g::StreamWeights sw;
};

struct Layer {
  DevW wq, wk, wv, wo, wqkv, wgate, wbeta, walpha, ssm_out, gate_inp, gate_sh, up_sh, down_sh;
  const float *attn_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *conv1d = nullptr, *ssm_dt = nullptr,
              *ssm_a = nullptr, *ssm_norm = nullptr, *gate_inp_sh = nullptr;
  // the bank: typed stacks per projection (each with its expert map), and which experts it holds
  std::vector<g::PromptWeights> bg, bu, bd;
  std::vector<std::uint8_t> hot;  // n_expert flags
  int n_hot = 0;
  std::vector<std::uint8_t> light;  // v28 split: n_expert flags (every matrix Q4_TG2, not in the bank): the CPU computes them
  int n_light = 0;
  void* bank_mem = nullptr;
  // state
  float *K = nullptr, *V = nullptr;          // full attention: max_ctx x kv_w each
  float* state = nullptr;                    // DeltaNet: nv x ds x ds
  float* tail[2] = {nullptr, nullptr};       // DeltaNet: (taps - 1) x ch, ping-pong
  int cur = 0;
  float* ckpt = nullptr;        // DeltaNet, verify: the state after each of the first max_verify - 1 tokens
  float* mixed_save = nullptr;  // DeltaNet, verify: the batch's conv inputs (max_verify x ch), to rebuild the tail of a prefix
  std::size_t bank_cap = 0;     // > 0: bank_mem is a fixed arena of this many bytes (rebuilt in place by promotion)
  int bank_slots = 0;           // experts the fixed arena may hold
  std::vector<std::uint64_t> picks;  // picks per expert (the MTP block counts them for promotion)
  std::vector<const std::uint8_t*> bank_at[3];  // v24: per projection (gate, up, down), each hot expert's matrix in the bank
};

// one routed expert's rows of one projection
struct ExpRows {
  quant::Type t = quant::Type::f32;
  const std::uint8_t* base = nullptr;
  std::uint64_t rb = 0;
};
ExpRows exp_rows(const WeightView& v, int e) {
  if (!v.slots.empty()) return {static_cast<quant::Type>(v.slots[z(e)].type), v.slots[z(e)].data, v.slots[z(e)].row_bytes};
  return {static_cast<quant::Type>(v.type), v.data + static_cast<std::uint64_t>(e) * v.ne[1] * v.row_bytes, v.row_bytes};
}
std::uint32_t exp_type(const WeightView& v, int e) { return v.slots.empty() ? v.type : v.slots[z(e)].type; }

struct ColdPick {
  int t = 0, e = 0;
  float w = 0.0f;
};

}  // namespace

struct GpuEngine::Impl {
  Qwen35MoeSpec s;
  const Qwen35MoeWeights* w = nullptr;
  EngineOptions o;
  int npad = 0;
  int64_t kmax = 0, kv_w = 0;
  std::int64_t pos = 0;
  std::uint64_t dev_bytes = 0, dense_bytes_total = 0, bank_bytes_total = 0, cache_bytes = 0;
  int bank_count = 0;
  std::vector<Layer> L;
  // device memory
  void* act_mem = nullptr;
  void* dense_mem = nullptr;
  void* state_mem = nullptr;
  Arena arena;  // over dense_mem (resident dense weights, norms, the output layer)
  DevW out;
  const float* out_norm = nullptr;
  const float* rope_ff = nullptr;
  g::MoeScratch moe;
  float *x = nullptr, *xn = nullptr, *y = nullptr, *xr = nullptr, *scale = nullptr, *dig = nullptr;
  std::int8_t* q = nullptr;
  int* sum = nullptr;
  float *qg = nullptr, *att = nullptr, *ascr = nullptr;
  float *mixed = nullptr, *zg = nullptr, *beta = nullptr, *alpha = nullptr, *conv = nullptr, *core = nullptr;
  float *logits_r = nullptr, *wsel = nullptr, *gs = nullptr, *us = nullptr, *sh = nullptr, *gsh = nullptr, *galpha = nullptr, *zero2 = nullptr;
  float* yc = nullptr;  // the cold sum, copied up
  std::int32_t* sel = nullptr;
  float* lg = nullptr;
  // v24 picked-expert reader
  static constexpr int kReaderMax = 4;   // steps of up to this many tokens use it
  bool reader = false;
  std::string reader_note = "off";
  std::vector<void*> reg_bases;          // v27: the registered ranges of the model file's mapping (host)
  g::ExpertPtr* rtab = nullptr;          // n_layer x n_expert entries (device)
  float *rgu = nullptr, *rh = nullptr;   // scratch: kReaderMax x nu x 2F, kReaderMax x nu x F
  float* ryd = nullptr;                  // v25 scratch: kReaderMax x nu x H (each pick's down row)
  int* rgrp = nullptr;                   // v25 scratch: the picks grouped by expert
  std::uint64_t rrow_max = 0;            // the longest expert row (bytes)
  std::vector<g::ExpertPtr> rtab_h;      // host copy of the table (the trace's byte counts)
  // v26 reader v3: the shared expert per layer (device), the groups' sources / fills, the cache
  g::ExpertPtr *rshx = nullptr, *rgsrc = nullptr, *rgfill = nullptr;
  bool fold_ok = false;
  // v28 the author's split: the light experts' mask (n_layer x n_expert, device), the doorbell generation on the device
  std::uint8_t* lmask_d = nullptr;
  unsigned* gen_d = nullptr;
  unsigned* gen_hp = nullptr;
  void* gen_hv = nullptr;
  bool split_ok = false;
  std::string split_note = "off";
  // v30 pre-load of the light experts (background, in place)
  std::thread preload_th;
  std::atomic<bool> preload_stop{false};
  double preload_s = 0.0;
  std::uint64_t preload_b = 0;
  bool preload_on = false;
  std::uint64_t cache_reserved = 0, cslot = 0;
  void* cache_mem = nullptr;
  void* cache_meta = nullptr;
  std::vector<g::ReaderCache> rcache;
  g::ExpertPtr *trc_src_h = nullptr, *trc_src_d = nullptr, *trc_fill_h = nullptr, *trc_fill_d = nullptr;
  // v25 trace: pinned mapped, per layer 6 reader stamps, the picks (kReaderMax x nu) and the group count
  void* trc_hv = nullptr;
  unsigned long long *trc_ts_h = nullptr, *trc_ts_d = nullptr;
  std::int32_t *trc_sel_h = nullptr, *trc_sel_d = nullptr;
  int *trc_u_h = nullptr, *trc_u_d = nullptr;
  std::string trace_text;
  std::uint64_t trace_seq = 0;
  static constexpr int kOutRows = 64;
  // host side of the cold path (sized once)
  std::unique_ptr<Pool> pool;
  std::vector<float> xn_h, ycold_h, wsel_h, xe_h;
  std::vector<std::int32_t> sel_h;
  std::vector<kern::ActQ8x32> actx, acth;
  std::vector<float> gbuf, ubuf, dbuf;
  std::vector<ColdPick> cold;
  std::vector<std::uint8_t> need;
  // v20: cold picks grouped by expert (up to 4 tokens each) and each group's gate / up blocks still to finish
  struct PickGroup {
    int e = 0, m = 0;
    int p[4] = {};
  };
  std::vector<PickGroup> grp;
  std::vector<int> ord;
  std::unique_ptr<std::atomic<int>[]> grp_left;
  // verify / keep (stage 3)
  bool ckpt_mode = false;
  int last_n = 0;            // tokens of the last verify batch (0: none pending)
  std::int64_t verify_pos0 = 0;
  // the MTP block (stage 3)
  bool has_mtp = false;
  Layer M;
  DevW eh, head;
  const float *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
  std::int64_t mtp_pos = 0;
  float* cat = nullptr;  // max_batch x 2 n_embd
  std::vector<float> cat_h;
  // the doorbell
  bool bell = false;
  void* bell_h = nullptr;  // one pinned mapped block: two flags, then the picks, the normed rows and the cold sum
  unsigned *fa_h = nullptr, *fa_d = nullptr, *fb_h = nullptr, *fb_d = nullptr;  // GPU -> CPU (picks ready), CPU -> GPU (cold sum ready)
  std::int32_t* sel_m = nullptr;
  float *wsel_m = nullptr, *xn_m = nullptr, *yc_m = nullptr, *yc_md = nullptr;  // host addresses; yc_md: the cold sum's device address
  unsigned gen = 0;
  std::thread bell_th;
  std::mutex bmu;
  std::condition_variable bcv;
  bool bstop = false, bjob = false, bbusy = false;
  bool bsplit = false;  // v28: this step's CPU side takes the light picks (reader step) rather than every non-bank pick
  int bn = 0;
  unsigned bg0 = 0;
  std::atomic<bool> bcancel{false};
  EngineStats bst;
  std::string berr;
  // v19 timeline
  bool tl_on = false;
  EngineTimeline tline;
  void* ts_hv = nullptr;  // pinned mapped: per layer [ring, wait enter, wait release, spare] GPU ns
  unsigned long long *ts_h = nullptr, *ts_d = nullptr;
  // v23 probe
  std::int64_t probe_pos = -1;
  bool probe_have = false;
  float* probe_xd = nullptr;
  std::int32_t* probe_seld = nullptr;
  float* probe_wd = nullptr;
  float *probe_bd = nullptr, *probe_ld = nullptr, *probe_md = nullptr;  // v25 per-operation rows
  std::vector<float> probe_bh, probe_lh, probe_mh;
  std::vector<float> probe_xh, probe_wh;
  std::vector<std::int32_t> probe_selh;
  // v21 per-operation stamps: pinned mapped, per layer kOps GPU ns; cur_ops = this layer's row while it is enqueued
  void* ops_hv = nullptr;
  unsigned long long *ops_h = nullptr, *ops_d = nullptr, *cur_ops = nullptr;
  bool opstamp(int k) { return !cur_ops || g::stamp(cur_ops + k); }

  // ---- weights
  static std::size_t view_bytes(const WeightView& v) {
    if (!v.slots.empty()) {
      std::size_t b = 0;
      for (const ExpertSlot& sl : v.slots) b += al(static_cast<std::size_t>(v.ne[1] * sl.row_bytes));
      return b;
    }
    return static_cast<std::size_t>(v.ne[1] * v.ne[2] * v.row_bytes);
  }
  static bool int8_ok(std::uint32_t type, int64_t n_out, int64_t k) { return g::prompt_type_ok(type, k) && n_out % 128 == 0; }
  bool use_r3 = false;  // EngineOptions::r3
  // R3 holds the matrix and the batched kernels read that same copy
  static bool r3_shared(std::uint32_t t) {
    return t == static_cast<std::uint32_t>(quant::Type::q8_0) || t == static_cast<std::uint32_t>(quant::Type::q4_k) ||
           t == static_cast<std::uint32_t>(quant::Type::q5_k) || t == static_cast<std::uint32_t>(quant::Type::q4_tg2);
  }
  bool r3_takes(const WeightView& v) const {
    return use_r3 && v.present() && !v.rot32 && v.slots.empty() && v.ne[2] == 1 && g::stream_type_ok(v.type, static_cast<int64_t>(v.ne[0]));
  }
  std::size_t dense_bytes(const WeightView& v) const {
    if (!v.present()) return 0;
    const int64_t n_out = static_cast<int64_t>(v.ne[1]), k = static_cast<int64_t>(v.ne[0]);
    std::size_t b = r3_takes(v) && r3_shared(v.type) ? 256 : al(view_bytes(v)) + 256;
    if (int8_ok(v.type, n_out, k)) b += al(g::prompt_table_bytes(v.type, n_out, k)) + 256;
    return b;
  }
  static std::size_t vec_bytes(const std::vector<float>& f) { return f.empty() ? 0 : al(f.size() * sizeof(float)) + 256; }
  std::size_t layer_dense_bytes(const Qwen35MoeLayer& l) const {
    std::size_t b = 0;
    for (const WeightView* v : {&l.wq, &l.wk, &l.wv, &l.wo, &l.wqkv, &l.wgate, &l.wbeta, &l.walpha, &l.ssm_out, &l.gate_inp, &l.gate_shexp, &l.up_shexp,
                                &l.down_shexp})
      b += dense_bytes(*v);
    for (const std::vector<float>* f : {&l.attn_norm, &l.post_norm, &l.q_norm, &l.k_norm, &l.conv1d, &l.ssm_dt, &l.ssm_a, &l.ssm_norm, &l.gate_inp_shexp})
      b += vec_bytes(*f);
    return b;
  }
  Status put_dense(const WeightView& v, DevW& dw) {
    dw = DevW{};
    if (!v.present()) return Status::good();
    dw.v = &v;
    dw.n_out = static_cast<int64_t>(v.ne[1]);
    dw.k = static_cast<int64_t>(v.ne[0]);
    dw.dtype = v.type;
    dw.drb = v.row_bytes;
    const void* d = nullptr;
    if (r3_takes(v)) {
      if (!g::stream_upload(v.type, v.data, dw.n_out, dw.k, &dw.sw)) return Status::fail(Code::internal, std::string("engine: R3 upload ") + v.name + ": " + g::stream_last_error());
      dw.r3 = true;
      dev_bytes += dw.sw.bytes;
      dense_bytes_total += dw.sw.bytes;
      if (r3_shared(v.type)) {  // the batched kernels read R3's copy
        d = dw.sw.w;
        if (v.type == static_cast<std::uint32_t>(quant::Type::q8_0)) {
          dw.dtype = g::g_q8_0r;
          dw.drb = static_cast<std::uint64_t>(dw.k / 256) * 272;
        }
      }
    }
    if (!d) {
      const std::size_t bytes = view_bytes(v);
      void* dd = arena.take(bytes);
      if (!dd || !g::to_device(dd, v.data, bytes)) return Status::fail(Code::internal, "engine: weight upload " + v.name);
      d = dd;
    }
    dw.d = d;
    if (int8_ok(v.type, dw.n_out, dw.k)) {
      void* tb = arena.take(g::prompt_table_bytes(v.type, dw.n_out, dw.k));
      if (!tb || !g::prompt_prepare_at(v.type, d, dw.n_out, dw.k, tb, &dw.pw)) return gfail("engine: side tables " + v.name);
      dw.int8 = true;
    }
    return Status::good();
  }
  Status put_vec(const std::vector<float>& f, const float*& out_ptr) {
    out_ptr = nullptr;
    if (f.empty()) return Status::good();
    void* d = arena.take(f.size() * sizeof(float));
    if (!d || !g::to_device(d, f.data(), f.size() * sizeof(float))) return Status::fail(Code::internal, "engine: vector upload");
    out_ptr = static_cast<const float*>(d);
    return Status::good();
  }
  Status put_layer(const Qwen35MoeLayer& l, Layer& D) {
    Status st;
    for (auto [v, d] : {std::pair{&l.wq, &D.wq}, {&l.wk, &D.wk}, {&l.wv, &D.wv}, {&l.wo, &D.wo}, {&l.wqkv, &D.wqkv}, {&l.wgate, &D.wgate},
                        {&l.wbeta, &D.wbeta}, {&l.walpha, &D.walpha}, {&l.ssm_out, &D.ssm_out}, {&l.gate_inp, &D.gate_inp},
                        {&l.gate_shexp, &D.gate_sh}, {&l.up_shexp, &D.up_sh}, {&l.down_shexp, &D.down_sh}})
      if (!(st = put_dense(*v, *d))) return st;
    for (auto [f, d] : {std::pair{&l.attn_norm, &D.attn_norm}, {&l.post_norm, &D.post_norm}, {&l.q_norm, &D.q_norm}, {&l.k_norm, &D.k_norm},
                        {&l.conv1d, &D.conv1d}, {&l.ssm_dt, &D.ssm_dt}, {&l.ssm_a, &D.ssm_a}, {&l.ssm_norm, &D.ssm_norm},
                        {&l.gate_inp_shexp, &D.gate_inp_sh}})
      if (!(st = put_vec(*f, *d))) return st;
    return Status::good();
  }

  // ---- the bank
  // device bytes of expert e's three projections in the bank (matrices and their share of the side tables)
  std::size_t expert_bytes(const Qwen35MoeLayer& l, int e) const {
    std::size_t b = 0;
    for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
      const std::uint32_t t = exp_type(*v, e);
      const int64_t n_out = static_cast<int64_t>(v->ne[1]), k = static_cast<int64_t>(v->ne[0]);
      const quant::TypeInfo ti = quant::type_info(t);
      b += static_cast<std::size_t>(n_out) * static_cast<std::size_t>(k / static_cast<int64_t>(ti.block_elements)) * ti.block_bytes;
      b += g::prompt_table_bytes(t, n_out, k);
    }
    return b;
  }
  // an expert can sit in the bank when every projection's type has an int8 step (the bank's stacks carry an expert map)
  bool bankable(const Qwen35MoeLayer& l, int e) const {
    for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps})
      if (!g::prompt_type_ok(exp_type(*v, e), static_cast<int64_t>(v->ne[0])) || v->rot32) return false;
    return true;
  }
  // the bank of one layer from its hot flags: per projection one stack per type, with an expert map
  Status build_bank(const Qwen35MoeLayer& l, Layer& D) {
    const int ne = static_cast<int>(s.n_expert);
    // bytes: matrices + tables + one map per stack (with slack for alignment)
    std::size_t need_b = 0;
    for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
      std::map<std::uint32_t, int> cnt;
      for (int e = 0; e < ne; ++e)
        if (D.hot[z(e)]) ++cnt[exp_type(*v, e)];
      const int64_t n_out = static_cast<int64_t>(v->ne[1]), k = static_cast<int64_t>(v->ne[0]);
      for (const auto& [t, c] : cnt) {
        const quant::TypeInfo ti = quant::type_info(t);
        need_b += al(static_cast<std::size_t>(c) * static_cast<std::size_t>(n_out) * static_cast<std::size_t>(k / static_cast<int64_t>(ti.block_elements)) * ti.block_bytes) +
                  al(g::prompt_table_bytes(t, static_cast<int64_t>(c) * n_out, k)) + al(z(ne) * sizeof(int)) + 768;
      }
    }
    D.bg.clear();
    D.bu.clear();
    D.bd.clear();
    if (need_b == 0) return Status::good();
    if (D.bank_cap > 0) {
      if (need_b > D.bank_cap) return Status::fail(Code::internal, "engine: the MTP bank arena is too small");
    } else {
      D.bank_mem = g::dalloc(need_b);
      if (!D.bank_mem) return gfail("engine: bank memory");
      dev_bytes += need_b;
      bank_bytes_total += need_b;
    }
    Arena A{static_cast<std::uint8_t*>(D.bank_mem), D.bank_cap > 0 ? D.bank_cap : need_b, 0};
    for (auto& ba : D.bank_at) ba.assign(z(ne), nullptr);
    auto stacks = [&](const WeightView& v, std::vector<g::PromptWeights>& dst, int proj) -> Status {
      std::map<std::uint32_t, std::vector<int>> by;
      for (int e = 0; e < ne; ++e)
        if (D.hot[z(e)]) by[exp_type(v, e)].push_back(e);
      const int64_t n_out = static_cast<int64_t>(v.ne[1]), k = static_cast<int64_t>(v.ne[0]);
      for (const auto& [t, ex] : by) {
        const quant::TypeInfo ti = quant::type_info(t);
        const std::size_t mat = static_cast<std::size_t>(n_out) * static_cast<std::size_t>(k / static_cast<int64_t>(ti.block_elements)) * ti.block_bytes;
        auto* d = static_cast<std::uint8_t*>(A.take(ex.size() * mat));
        if (!d) return Status::fail(Code::internal, "engine: bank arena (" + v.name + ")");
        for (std::size_t i = 0; i < ex.size(); ++i) {
          const ExpRows r = exp_rows(v, ex[i]);
          if (!g::to_device(d + i * mat, r.base, mat)) return gfail("engine: bank upload " + v.name);
          D.bank_at[proj][z(ex[i])] = d + i * mat;
        }
        const int64_t rows = static_cast<int64_t>(ex.size()) * n_out;
        g::PromptWeights pw;
        void* tb = A.take(g::prompt_table_bytes(t, rows, k));
        if (!tb || !g::prompt_prepare_at(t, d, rows, k, tb, &pw)) return gfail("engine: bank tables " + v.name);
        std::vector<int> map(z(ne), -1);
        for (std::size_t i = 0; i < ex.size(); ++i) map[z(ex[i])] = static_cast<int>(i);
        auto* em = static_cast<int*>(A.take(z(ne) * sizeof(int)));
        if (!em || !g::to_device(em, map.data(), z(ne) * sizeof(int))) return gfail("engine: bank map " + v.name);
        pw.emap = em;
        pw.n_mat = static_cast<int>(ex.size());
        dst.push_back(pw);
      }
      return Status::good();
    };
    Status st;
    if (!(st = stacks(l.gate_exps, D.bg, 0)) || !(st = stacks(l.up_exps, D.bu, 1)) || !(st = stacks(l.down_exps, D.bd, 2))) return st;
    if (D.bg.size() > 4 || D.bu.size() > 4 || D.bd.size() > 4) return Status::fail(Code::unsupported, "engine: more than 4 expert types in a layer's bank");
    return Status::good();
  }

  // ---- products: as GpuForward::prod (quantized weights round their input to gpu256x8 first)
  Status prod(const DevW& W, const float* xin, int64_t n, float* yout) {
    const WeightView& v = *W.v;
    if (is_float_type(v.type)) {
      if (v.rot32) return Status::fail(Code::unsupported, "engine: a rotated float weight (" + v.name + ")");
      if (!g::matmul(W.dtype, W.d, W.drb, W.n_out, W.k, xin, n, W.k, yout, W.n_out)) return gfail("engine: product " + v.name);
      return Status::good();
    }
    if (W.r3 && (n == 1 || (bi_now && n <= kReaderMax))) return prod_r3({{&W, yout}}, xin, n);
    if (!g::act_q8x32(xin, W.k, n, W.k, v.rot32, xr, W.k, q, scale, sum)) return gfail("engine: rounding for " + v.name);
    if (W.int8) {
      if (!g::act_digits(scale, sum, n, W.k, dig) || !g::prompt_gemm(W.pw, W.n_out, q, scale, dig, n, yout)) return gfail("engine: int8 product " + v.name);
      return Status::good();
    }
    if (!g::matmul(W.dtype, W.d, W.drb, W.n_out, W.k, xr, n, W.k, yout, W.n_out)) return gfail("engine: product " + v.name);
    return Status::good();
  }
  // one token through R3: the products of one input, grouped into one plan (built once per group, kept), outputs into a
  // fixed scratch and then copied to where each belongs (the attention cache's row moves with the position)
  struct R3Group {
    std::unique_ptr<g::StreamPlan> plan;
    std::vector<int64_t> off;  // each job's output offset in r3y
  };
  std::map<std::pair<const DevW*, int>, R3Group> r3plans;  // v26: per group and token count
  bool bi_now = false;  // v26 option B: this step's products of 2-4 tokens go through R3 (set by run())
  // v26 CUDA graph
  g::Stream stm = nullptr;
  int* pos_d = nullptr;
  int* pos_hp = nullptr;
  void* pos_hv = nullptr;
  float *knew = nullptr, *vnew = nullptr;
  bool graph_ok = false;
  bool trunk_dp = false;  // the trunk's step reads the device position (not the MTP block's own attention)
  std::string graph_note = "off";
  std::map<std::tuple<int, int, int, int>, void*> graphs;
  std::map<std::tuple<int, int, int, int>, int> graph_seen;
  std::uint64_t graph_replays = 0, graph_captures = 0;
  // v26 prefetch
  bool pf_ok = false;
  g::Stream side = nullptr;
  std::vector<void*> ev_a, ev_b;
  float *xp = nullptr, *lp = nullptr, *wselp = nullptr;
  std::int32_t* selp = nullptr;
  int* pfjob = nullptr;
  std::vector<std::uint8_t> pf_issued;  // per layer: a prefetch for it is in flight this step
  float* r3y = nullptr;
  std::size_t r3y_floats = 0;
  Status prod_r3(std::initializer_list<std::pair<const DevW*, float*>> jobs, const float* xin, int64_t n = 1) {
    const DevW& W0 = *jobs.begin()->first;
    if (n < 1 || n > kReaderMax) return Status::fail(Code::internal, "engine: R3 takes 1-4 tokens");
    R3Group& G = r3plans[{&W0, static_cast<int>(n)}];
    if (!G.plan) {
      std::vector<g::StreamJob> js;
      int64_t at = 0;
      for (const auto& [W, yo] : jobs) {
        (void)yo;
        if (!W->r3 || W->k != W0.k || W->v->rot32 != W0.v->rot32) return Status::fail(Code::internal, "engine: an R3 group with unlike inputs");
        g::StreamJob j;
        j.w = &W->sw;
        j.xq = q;
        j.xs = scale;
        j.xsum = sum;
        if (static_cast<std::size_t>(at + n * W->n_out) > r3y_floats) return Status::fail(Code::internal, "engine: R3 scratch too small");
        j.y = r3y + at;
        j.ntok = static_cast<int>(n);  // v26: rows of n tokens back to back (stride n_out)
        j.xq_stride = W0.k;
        j.xs_stride = W0.k / 32;
        j.y_stride = W->n_out;
        G.off.push_back(at);
        at += (n * W->n_out + 63) / 64 * 64;
        js.push_back(j);
      }
      const int stage_end[1] = {static_cast<int>(js.size())};
      G.plan = std::make_unique<g::StreamPlan>();
      if (!G.plan->build(js.data(), static_cast<int>(js.size()), stage_end, 1, 0))
        return Status::fail(Code::internal, std::string("engine: R3 plan: ") + g::stream_last_error());
    }
    if (!g::act_q8x32(xin, W0.k, n, W0.k, W0.v->rot32, xr, W0.k, q, scale, sum)) return gfail("engine: rounding for R3");
    if (!(o.graph ? G.plan->run_reset() : G.plan->run())) return Status::fail(Code::internal, std::string("engine: R3 run: ") + g::stream_last_error());
    std::size_t i = 0;
    for (const auto& [W, yo] : jobs)
      if (!g::dcopy(yo, r3y + G.off[i++], z(n * W->n_out) * sizeof(float))) return gfail("engine: R3 output");
    return Status::good();
  }

  Status attention_block(Layer& D, int64_t n, int64_t p0) {
    const int64_t hd = s.head_dim, nh = s.n_head, nkv = s.n_head_kv;
    const int64_t qh = s.attn_gate ? 2 * hd : hd, q_w = qh * nh;
    const bool gp = trunk_dp;  // v26: new rows in scratch, stored at the device position; rope / attention read it there
    float* kn = gp ? knew : D.K + p0 * kv_w;  // the new keys and values go straight into the cache
    float* vn = gp ? vnew : D.V + p0 * kv_w;
    Status st;
    if ((n == 1 || bi_now) && D.wq.r3 && D.wk.r3 && D.wv.r3 && D.wq.k == D.wk.k && D.wk.k == D.wv.k) {
      if (!(st = prod_r3({{&D.wq, qg}, {&D.wk, kn}, {&D.wv, vn}}, xn, n))) return st;
    } else if (!(st = prod(D.wq, xn, n, qg)) || !(st = prod(D.wk, xn, n, kn)) || !(st = prod(D.wv, xn, n, vn))) {
      return st;
    }
    if (!opstamp(2)) return gfail("engine: stamp");
    if (D.q_norm && !g::rms_norm(qg, D.q_norm, qg, hd, nh, qh, q_w, n, s.rms_eps)) return gfail("engine: q norm");
    if (D.k_norm && !g::rms_norm(kn, D.k_norm, kn, hd, nkv, hd, kv_w, n, s.rms_eps)) return gfail("engine: k norm");
    const int sec[4] = {s.rope_sections[0], s.rope_sections[1], s.rope_sections[2], s.rope_sections[3]};
    if (!g::rope(qg, nh, qh, q_w, n, p0, static_cast<int>(s.n_rot), s.rope_freq_base, sec, s.rope_neox, rope_ff, nullptr, gp ? pos_d : nullptr) ||
        !g::rope(kn, nkv, hd, kv_w, n, p0, static_cast<int>(s.n_rot), s.rope_freq_base, sec, s.rope_neox, rope_ff, nullptr, gp ? pos_d : nullptr))
      return gfail("engine: rope");
    if (gp && !g::kv_store(D.K, D.V, kn, vn, kv_w, n, pos_d)) return gfail("engine: K / V store");
    const float scl = s.attn_scale != 0.0f ? s.attn_scale : 1.0f / std::sqrt(static_cast<float>(hd));
    if (!opstamp(3)) return gfail("engine: stamp");
    if (!g::attention(qg, q_w, qh, D.K, D.V, kv_w, nh, nkv, hd, n, p0, scl, s.attn_gate, att, ascr, nullptr, gp ? pos_d : nullptr, o.max_ctx))
      return gfail("engine: attention");
    if (!opstamp(4) || !opstamp(5)) return gfail("engine: stamp");
    if (Status ps = prod(D.wo, att, n, y); !ps) return ps;
    return opstamp(6) ? Status::good() : gfail("engine: stamp");
  }

  Status deltanet_block(Layer& D, int64_t n) {
    const int64_t ch = s.conv_channels(), ds = s.ssm_d_state, nk = s.ssm_n_group, nv = s.ssm_dt_rank, taps = s.ssm_d_conv;
    Status st;
    if ((n == 1 || bi_now) && D.wqkv.r3 && D.wgate.r3 && D.wqkv.k == D.wgate.k) {
      if (!(st = prod_r3({{&D.wqkv, mixed}, {&D.wgate, zg}}, xn, n))) return st;
    } else if (!(st = prod(D.wqkv, xn, n, mixed)) || !(st = prod(D.wgate, xn, n, zg))) {
      return st;
    }
    if (!(st = prod(D.wbeta, xn, n, beta)) || !(st = prod(D.walpha, xn, n, alpha))) return st;
    if (!opstamp(2)) return gfail("engine: stamp");
    if (!g::deltanet_gates(beta, alpha, D.ssm_dt, D.ssm_a, n, nv)) return gfail("engine: deltanet gates");
    const bool ck = ckpt_mode && n > 1;
    if (ck && !g::dcopy(D.mixed_save, mixed, z(n * ch) * sizeof(float))) return gfail("engine: verify inputs");
    if (!g::deltanet_conv(mixed, D.tail[D.cur], D.conv1d, n, ch, taps, conv, D.tail[1 - D.cur])) return gfail("engine: deltanet conv");
    D.cur = 1 - D.cur;
    if (!g::l2_norm(conv, ds, 2 * nk, ds, ch, n, s.rms_eps)) return gfail("engine: deltanet l2 norm");
    if (!opstamp(3)) return gfail("engine: stamp");
    const bool pow2 = ds >= 32 && ds <= 1024 && (ds & (ds - 1)) == 0;
    const bool ok = ck ? g::deltanet_recurrent_reg_ckpt(conv, ch, beta, alpha, D.state, n, nk, nv, ds, core, D.ckpt)
                       : pow2 ? g::deltanet_recurrent_reg(conv, ch, beta, alpha, D.state, n, nk, nv, ds, core)
                              : g::deltanet_recurrent(conv, ch, beta, alpha, D.state, n, nk, nv, ds, core);
    if (!ok) return gfail("engine: deltanet recurrence");
    if (!opstamp(4)) return gfail("engine: stamp");
    if (!g::gated_norm(core, zg, D.ssm_norm, n, nv, ds, s.rms_eps)) return gfail("engine: gated norm");
    if (!opstamp(5)) return gfail("engine: stamp");
    if (Status ps = prod(D.ssm_out, core, n, y); !ps) return ps;
    return opstamp(6) ? Status::good() : gfail("engine: stamp");
  }

  // the cold picks on the CPU: ycold_h (n x e) = sum over cold picks of weight x down(round32(silu(gate x) * up x))
  void cold_experts(const Qwen35MoeLayer& l, int64_t n) { cold_experts_io(l, n, xn_h.data(), ycold_h.data()); }
  // tlp (v19 timeline, the doorbell thread only): each stage's wall time, item time over all threads, the first worker's
  // wake-up on the two big jobs, and the expert bytes read
  void cold_experts_io(const Qwen35MoeLayer& l, int64_t n, const float* xsrc, float* ydst, EngineTimeline* tlp = nullptr) {
    const int64_t e = s.n_embd, ff = s.n_ff_exp;
    const int64_t np = static_cast<int64_t>(cold.size());
    Pool::RunStats rs[5];
    Clock::time_point ts = Clock::now();
    auto mark = [&](int k) {
      if (!tlp) return;
      const Clock::time_point now = Clock::now();
      tlp->stage[k] += std::chrono::duration<double>(now - ts).count();
      tlp->busy[k] += static_cast<double>(rs[k].busy_ns.load(std::memory_order_relaxed)) * 1e-9;
      const std::int64_t w8 = rs[k].wake_ns.load(std::memory_order_relaxed);
      if ((k == 1 || k == 3) && w8 >= 0) {
        tlp->wake += static_cast<double>(w8) * 1e-9;
        ++tlp->wake_n;
      }
      ts = now;
    };
    Pool::RunStats* rp0 = tlp ? &rs[0] : nullptr;
    Pool::RunStats* rp1 = tlp ? &rs[1] : nullptr;
    Pool::RunStats* rp2 = tlp ? &rs[2] : nullptr;
    Pool::RunStats* rp3 = tlp ? &rs[3] : nullptr;
    Pool::RunStats* rp4 = tlp ? &rs[4] : nullptr;
    if (tlp)
      for (const ColdPick& c : cold)
        for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps}) tlp->cold_bytes += static_cast<double>(exp_rows(*v, c.e).rb) * static_cast<double>(v == &l.down_exps ? e : ff);
    std::fill(need.begin(), need.begin() + n, std::uint8_t{0});
    for (const ColdPick& c : cold) need[z(c.t)] = 1;
    pool->run(n, [&](int64_t t) {
      if (need[z(t)]) kern::act_q8x32_quantize(xsrc + t * e, e, actx[z(t)]);
    }, rp0);
    mark(0);
    // v20 (author 15:57, from the v19 timeline): the picks grouped by expert, up to 4 tokens per group, so a weight row is
    // read once for every token that picked it (dot_row_q8 takes 1-4 activation rows; each token's sum is the same as
    // alone); and the silu + rounding of a group done by the thread that finishes its last gate / up block (one thread
    // hand-off less per layer). Outputs stay per pick (gbuf / ubuf / dbuf at the pick's index), so the combine below is
    // unchanged and deterministic.
    constexpr int64_t kRowsUp = 64, kRowsDown = 128, kGroup = 4;
    grp.clear();
    {
      ord.resize(z(np));
      for (int64_t p = 0; p < np; ++p) ord[z(p)] = static_cast<int>(p);
      std::stable_sort(ord.begin(), ord.end(), [&](int pa, int pb) { return cold[z(pa)].e < cold[z(pb)].e; });
      for (int64_t i = 0; i < np;) {
        const int ex = cold[z(ord[z(i)])].e;
        PickGroup gp;
        gp.e = ex;
        while (i < np && cold[z(ord[z(i)])].e == ex && gp.m < kGroup) gp.p[gp.m++] = ord[z(i++)];
        grp.push_back(gp);
      }
    }
    const int64_t ng = static_cast<int64_t>(grp.size());
    const int64_t ub = (ff + kRowsUp - 1) / kRowsUp;  // row blocks per projection (gate, up)
    for (int64_t gi = 0; gi < ng; ++gi) grp_left[z(gi)].store(static_cast<int>(2 * ub), std::memory_order_relaxed);
    pool->run(ng * 2 * ub, [&](int64_t u) {
      const int64_t gi = u / (2 * ub), rem = u % (2 * ub), proj = rem / ub, r0 = (rem % ub) * kRowsUp, r1 = std::min(ff, r0 + kRowsUp);
      const PickGroup& gp = grp[z(gi)];
      const ExpRows R = exp_rows(proj == 0 ? l.gate_exps : l.up_exps, gp.e);
      const kern::ActQ8x32* acts[kGroup] = {};
      float* outs[kGroup] = {};
      for (int j = 0; j < gp.m; ++j) {
        acts[j] = &actx[z(cold[z(gp.p[j])].t)];
        outs[j] = (proj == 0 ? gbuf.data() : ubuf.data()) + static_cast<int64_t>(gp.p[j]) * ff;
      }
      float o4[kGroup];
      for (int64_t r = r0; r < r1; ++r) {
        kern::dot_row_q8(R.t, R.base + static_cast<std::uint64_t>(r) * R.rb, e, nullptr, acts, gp.m, o4);
        for (int j = 0; j < gp.m; ++j) outs[j][r] = o4[j];
      }
      if (grp_left[z(gi)].fetch_sub(1, std::memory_order_acq_rel) == 1)  // the group's last block: its silu and rounding
        for (int j = 0; j < gp.m; ++j) {
          const int64_t pk = gp.p[j];
          float* gg = gbuf.data() + pk * ff;
          const float* uu = ubuf.data() + pk * ff;
          for (int64_t i = 0; i < ff; ++i) gg[i] = silu(gg[i]) * uu[i];
          kern::act_q8x32_quantize(gg, ff, acth[z(pk)]);
        }
    }, rp1);
    mark(1);
    (void)rp2;
    mark(2);
    const int64_t db = (e + kRowsDown - 1) / kRowsDown;
    pool->run(ng * db, [&](int64_t u) {
      const int64_t gi = u / db, r0 = (u % db) * kRowsDown, r1 = std::min(e, r0 + kRowsDown);
      const PickGroup& gp = grp[z(gi)];
      const ExpRows R = exp_rows(l.down_exps, gp.e);
      const kern::ActQ8x32* acts[kGroup] = {};
      float* outs[kGroup] = {};
      for (int j = 0; j < gp.m; ++j) {
        acts[j] = &acth[z(gp.p[j])];
        outs[j] = dbuf.data() + static_cast<int64_t>(gp.p[j]) * e;
      }
      float o4[kGroup];
      for (int64_t r = r0; r < r1; ++r) {
        kern::dot_row_q8(R.t, R.base + static_cast<std::uint64_t>(r) * R.rb, ff, nullptr, acts, gp.m, o4);
        for (int j = 0; j < gp.m; ++j) outs[j][r] = o4[j];
      }
    }, rp3);
    mark(3);
    // per token, the picks in slot order (deterministic)
    pool->run(n, [&](int64_t t) {
      float* yt = ydst + t * e;
      std::fill(yt, yt + e, 0.0f);
      for (int64_t p = 0; p < np; ++p) {
        if (cold[z(p)].t != t) continue;
        const float wgt = cold[z(p)].w;
        const float* d = dbuf.data() + p * e;
        for (int64_t i = 0; i < e; ++i) yt[i] += wgt * d[i];
      }
    }, rp4);
    mark(4);
  }

  // v28b (author 23:14, the split's CPU side in one dispatch per layer): the inputs rounded by the calling thread (1-4 rows),
  // then one pool job in two phases: (1) gate / up row blocks of every expert group, the group's last block doing its silu
  // and rounding (as v20); (2) down rows in blocks, each block computing every group's rows and then summing them per token in
  // pick-slot order (the old combine's order and arithmetic, fused: no separate combine job). The pool is held hot for the
  // whole step by the doorbell thread, so no worker sleeps between layers.
  void cold_experts_fast(const Qwen35MoeLayer& l, int64_t n, const float* xsrc, float* ydst) {
    const int64_t e = s.n_embd, ff = s.n_ff_exp;
    const int64_t np = static_cast<int64_t>(cold.size());
    std::fill(need.begin(), need.begin() + n, std::uint8_t{0});
    for (const ColdPick& c : cold) need[z(c.t)] = 1;
    for (int64_t t = 0; t < n; ++t)
      if (need[z(t)]) kern::act_q8x32_quantize(xsrc + t * e, e, actx[z(t)]);
    constexpr int64_t kRowsUp = 64, kRowsDown = 128, kGroup = 4;
    grp.clear();
    ord.resize(z(np));
    for (int64_t p = 0; p < np; ++p) ord[z(p)] = static_cast<int>(p);
    std::stable_sort(ord.begin(), ord.end(), [&](int pa, int pb) { return cold[z(pa)].e < cold[z(pb)].e; });
    for (int64_t i = 0; i < np;) {
      const int ex = cold[z(ord[z(i)])].e;
      PickGroup gp;
      gp.e = ex;
      while (i < np && cold[z(ord[z(i)])].e == ex && gp.m < kGroup) gp.p[gp.m++] = ord[z(i++)];
      grp.push_back(gp);
    }
    const int64_t ng = static_cast<int64_t>(grp.size());
    const int64_t ub = (ff + kRowsUp - 1) / kRowsUp, db = (e + kRowsDown - 1) / kRowsDown;
    for (int64_t gi = 0; gi < ng; ++gi) grp_left[z(gi)].store(static_cast<int>(2 * ub), std::memory_order_relaxed);
    const std::function<void(int64_t)> up_fn = [&](int64_t u) {
      const int64_t gi = u / (2 * ub), rem = u % (2 * ub), proj = rem / ub, r0 = (rem % ub) * kRowsUp, r1 = std::min(ff, r0 + kRowsUp);
      const PickGroup& gp = grp[z(gi)];
      const ExpRows R = exp_rows(proj == 0 ? l.gate_exps : l.up_exps, gp.e);
      const kern::ActQ8x32* acts[kGroup] = {};
      float* outs[kGroup] = {};
      for (int j = 0; j < gp.m; ++j) {
        acts[j] = &actx[z(cold[z(gp.p[j])].t)];
        outs[j] = (proj == 0 ? gbuf.data() : ubuf.data()) + static_cast<int64_t>(gp.p[j]) * ff;
      }
      float o4[kGroup];
      for (int64_t r = r0; r < r1; ++r) {
        kern::dot_row_q8(R.t, R.base + static_cast<std::uint64_t>(r) * R.rb, e, nullptr, acts, gp.m, o4);
        for (int j = 0; j < gp.m; ++j) outs[j][r] = o4[j];
      }
      if (grp_left[z(gi)].fetch_sub(1, std::memory_order_acq_rel) == 1)
        for (int j = 0; j < gp.m; ++j) {
          const int64_t pk = gp.p[j];
          float* gg = gbuf.data() + pk * ff;
          const float* uu = ubuf.data() + pk * ff;
          for (int64_t i = 0; i < ff; ++i) gg[i] = silu(gg[i]) * uu[i];
          kern::act_q8x32_quantize(gg, ff, acth[z(pk)]);
        }
    };
    const std::function<void(int64_t)> down_fn = [&](int64_t u) {
      const int64_t r0 = u * kRowsDown, r1 = std::min(e, r0 + kRowsDown);
      for (int64_t gi = 0; gi < ng; ++gi) {
        const PickGroup& gp = grp[z(gi)];
        const ExpRows R = exp_rows(l.down_exps, gp.e);
        const kern::ActQ8x32* acts[kGroup] = {};
        float* outs[kGroup] = {};
        for (int j = 0; j < gp.m; ++j) {
          acts[j] = &acth[z(gp.p[j])];
          outs[j] = dbuf.data() + static_cast<int64_t>(gp.p[j]) * e;
        }
        float o4[kGroup];
        for (int64_t r = r0; r < r1; ++r) {
          kern::dot_row_q8(R.t, R.base + static_cast<std::uint64_t>(r) * R.rb, ff, nullptr, acts, gp.m, o4);
          for (int j = 0; j < gp.m; ++j) outs[j][r] = o4[j];
        }
      }
      for (int64_t t = 0; t < n; ++t) {  // this block's rows of each token's sum, picks in slot order (the combine's order)
        float* yt = ydst + t * e;
        std::fill(yt + r0, yt + r1, 0.0f);
        for (int64_t p = 0; p < np; ++p) {
          if (cold[z(p)].t != t) continue;
          const float wgt = cold[z(p)].w;
          const float* d = dbuf.data() + p * e;
          for (int64_t i = r0; i < r1; ++i) yt[i] += wgt * d[i];
        }
      }
    };
    pool->run2(ng * 2 * ub, up_fn, db, down_fn);
  }

  Status moe_block(const Qwen35MoeLayer& l, Layer& D, int64_t n, EngineStats& es) {
    const int64_t e = s.n_embd, ne = s.n_expert, nu = s.n_expert_used, ffs = s.n_ff_shexp;
    Status st;
    if (!(st = prod(D.gate_inp, xn, n, logits_r))) return st;
    if (!g::router_topk(logits_r, ne, n, static_cast<int>(nu), s.expert_weights_scale, sel, wsel, nullptr, o.router_tie)) return gfail("engine: router");
    // the picks and the normed rows to the host (waits for the router)
    Clock::time_point tw = Clock::now();
    if (!g::to_host(sel_h.data(), sel, z(n * nu) * sizeof(std::int32_t)) || !g::to_host(wsel_h.data(), wsel, z(n * nu) * sizeof(float)))
      return gfail("engine: router picks");
    es.wait_seconds += since(tw);
    cold.clear();
    int64_t hot_here = 0;
    for (int64_t t = 0; t < n; ++t)
      for (int64_t j = 0; j < nu; ++j) {
        const int ex = sel_h[z(t * nu + j)];
        if (ex < 0 || ex >= ne) return Status::fail(Code::internal, "engine: a router pick outside the experts");
        if (!D.picks.empty()) ++D.picks[z(ex)];
        if (D.hot[z(ex)]) ++hot_here;
        else cold.push_back(ColdPick{static_cast<int>(t), ex, wsel_h[z(t * nu + j)]});
      }
    es.hot_picks += static_cast<std::uint64_t>(hot_here);
    es.cold_picks += cold.size();
    if (!cold.empty()) {
      tw = Clock::now();
      if (!g::to_host(xn_h.data(), xn, z(n * e) * sizeof(float))) return gfail("engine: normed rows");
      es.wait_seconds += since(tw);
    }
    // the GPU: the bank's experts (cold experts have no unit in any stack: they add nothing here), then the shared expert
    if (hot_here > 0) {
      if (!g::act_q8x32(xn, e, n, e, false, xr, e, q, scale, sum) || !g::act_digits(scale, sum, n, e, dig)) return gfail("engine: expert input rounding");
      const g::MoeStacks sg{D.bg.data(), static_cast<int>(D.bg.size())}, su{D.bu.data(), static_cast<int>(D.bu.size())},
          sd{D.bd.data(), static_cast<int>(D.bd.size())};
      // the down projection folded into the combine in fixed point (deterministic: a step repeats bit for bit). int64 at 2^32:
      // resolution 2^-32 per addend, range +-2^31 (the locked int32 fold needs a per-layer scale with its overflow counter;
      // later, measured)
      if (!g::moe_prompt_multi(sg, su, sd, q, scale, dig, n, sel, wsel, moe, g::MoeCombine::kFixed64, 32, y)) return gfail("engine: bank experts");
    } else if (!g::dzero(y, z(n * e) * sizeof(float))) {
      return gfail("engine: clear y");
    }
    if (n == 1 && D.gate_sh.r3 && D.up_sh.r3 && D.gate_sh.k == D.up_sh.k) {
      if (!(st = prod_r3({{&D.gate_sh, gs}, {&D.up_sh, us}}, xn))) return st;
    } else if (!(st = prod(D.gate_sh, xn, n, gs)) || !(st = prod(D.up_sh, xn, n, us))) {
      return st;
    }
    if (!g::silu_mul(gs, us, gs, n * ffs)) return gfail("engine: shared silu");
    if (!(st = prod(D.down_sh, gs, n, sh))) return st;
    if (!g::matmul(static_cast<std::uint32_t>(quant::Type::f32), D.gate_inp_sh, static_cast<std::uint64_t>(e) * sizeof(float), 1, e, xn, n, e, gsh, 1) ||
        !g::deltanet_gates(gsh, galpha, zero2, zero2, n, 1))
      return gfail("engine: shared gate");
    if (!g::add_rows(y, e, sh, e, gsh, e, n)) return gfail("engine: shared add");
    // the CPU: the cold picks, while the GPU runs the above
    if (!cold.empty()) {
      const Clock::time_point tc = Clock::now();
      cold_experts(l, n);
      es.cold_seconds += since(tc);
      tw = Clock::now();
      if (!g::to_device(yc, ycold_h.data(), z(n * e) * sizeof(float))) return gfail("engine: cold sum");
      es.wait_seconds += since(tw);
      if (!g::add_rows(y, e, yc, e, nullptr, e, n)) return gfail("engine: cold add");
    }
    return Status::good();
  }

  Status run(const std::int32_t* tokens, int n, float* logits, bool last_only, EngineStats* stats, float* hidden_out);
  // embedding rows on the host (the reference's embed_row without planes), then one copy
  Status embed(const WeightView& et, const std::int32_t* tokens, int n, float* dst, int64_t e) {
    xe_h.resize(z(n) * z(e));
    for (int t = 0; t < n; ++t) {
      const std::int32_t tok = tokens[t];
      if (tok < 0 || static_cast<std::uint64_t>(tok) >= s.n_vocab) return Status::fail(Code::invalid_argument, "a token id is outside the vocabulary");
      const Status qs = quant::dequantize_row(static_cast<quant::Type>(et.type), et.data + static_cast<std::uint64_t>(tok) * et.row_bytes, xe_h.data() + z(t) * z(e), e);
      if (!qs.ok()) return Status::fail(Code::unsupported, "engine: the embedding type cannot be read on the host");
    }
    return g::to_device(dst, xe_h.data(), xe_h.size() * sizeof(float)) ? Status::good() : gfail("engine: embedding upload");
  }

  // the doorbell's CPU side: for each layer of the step, wait for the GPU's ring, run the cold picks into pinned memory,
  // ring back. On a cancel or a stall it still rings (zeros), so the GPU never waits forever.
  void bell_loop() {
    for (;;) {
      int n = 0;
      unsigned g0 = 0;
      bool split = false;
      {
        std::unique_lock<std::mutex> lk(bmu);
        bcv.wait(lk, [&] { return bstop || bjob; });
        if (bstop) return;
        bjob = false;
        n = bn;
        g0 = bg0;
        split = bsplit;
      }
      EngineStats es;
      std::string err;
      const int64_t e = s.n_embd, ne = s.n_expert, nu = s.n_expert_used;
      if (split) pool->set_hot(true);  // v28b: the workers spin through the whole step
      for (std::uint32_t il = 0; il < s.n_layer; ++il) {
        const unsigned g = g0 + il;
        const Clock::time_point tw = Clock::now();
        std::atomic_ref<unsigned> fa(*fa_h);
        for (std::uint64_t spin = 0; static_cast<int>(fa.load(std::memory_order_acquire) - g) < 0; ++spin) {
          if (bcancel.load(std::memory_order_relaxed)) break;
          if ((spin & 1023) == 1023 && since(tw) > 30.0) {
            err = "engine: doorbell stall (layer " + std::to_string(il) + ")";
            bcancel.store(true, std::memory_order_relaxed);
            break;
          }
        }
        const double wt_l = since(tw);
        es.wait_seconds += wt_l;
        EngineTimeline* tlp = tl_on ? &tline : nullptr;
        if (tlp) tlp->cpu_wait[il] += wt_l;
        float* yd = yc_m;
        if (bcancel.load(std::memory_order_relaxed)) {
          std::fill(yd, yd + n * e, 0.0f);
        } else {
          cold.clear();
          for (int64_t t = 0; t < n; ++t)
            for (int64_t j = 0; j < nu; ++j) {
              const int ex = sel_m[z(t * nu + j)];
              if (ex < 0 || ex >= ne) continue;
              if (split) {  // v28: only the light experts are the CPU's; the GPU counts the rest
                if (!L[il].light.empty() && L[il].light[z(ex)]) cold.push_back(ColdPick{static_cast<int>(t), ex, wsel_m[z(t * nu + j)]});
              } else if (L[il].hot[z(ex)]) {
                ++es.hot_picks;
              } else {
                cold.push_back(ColdPick{static_cast<int>(t), ex, wsel_m[z(t * nu + j)]});
              }
            }
          es.cold_picks += cold.size();
          {
            const Qwen35MoeLayer& lw = w->layers[il];
            for (const ColdPick& c : cold)
              for (const WeightView* v : {&lw.gate_exps, &lw.up_exps, &lw.down_exps}) es.cold_bytes += exp_rows(*v, c.e).rb * v->ne[1];
          }
          const Clock::time_point tc = Clock::now();
          if (cold.empty()) std::fill(yd, yd + n * e, 0.0f);
          else if (split && !o.split_slow) cold_experts_fast(w->layers[il], n, xn_m, yd);
          else cold_experts_io(w->layers[il], n, xn_m, yd, tlp);
          const double ct_l = since(tc);
          es.cold_seconds += ct_l;
          if (tlp) tlp->cpu_cold[il] += ct_l;
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::atomic_ref<unsigned>(*fb_h).store(g, std::memory_order_release);
      }
      if (split) pool->set_hot(false);
      {
        std::lock_guard<std::mutex> lk(bmu);
        bst = es;
        berr = err;
        bbusy = false;
      }
      bcv.notify_all();
    }
  }
  // the MoE block with the doorbell: everything enqueued, no host wait
  Status moe_block_bell(Layer& D, int64_t n, unsigned g, unsigned long long* tsd) {
    const int64_t e = s.n_embd, ne = s.n_expert, nu = s.n_expert_used, ffs = s.n_ff_shexp;
    Status st;
    if (!(st = prod(D.gate_inp, xn, n, logits_r))) return st;
    if (!g::router_topk(logits_r, ne, n, static_cast<int>(nu), s.expert_weights_scale, sel, wsel, nullptr, o.router_tie)) return gfail("engine: router");
    if (!opstamp(8)) return gfail("engine: stamp");
    if (!g::to_host_async(sel_m, sel, z(n * nu) * sizeof(std::int32_t)) || !g::to_host_async(wsel_m, wsel, z(n * nu) * sizeof(float)) ||
        !g::to_host_async(xn_m, xn, z(n * e) * sizeof(float)) || !g::bell_ring(fa_d, g, nullptr, tsd))
      return gfail("engine: doorbell (picks)");
    if (!opstamp(9)) return gfail("engine: stamp");
    if (!D.bg.empty()) {
      if (!g::act_q8x32(xn, e, n, e, false, xr, e, q, scale, sum) || !g::act_digits(scale, sum, n, e, dig)) return gfail("engine: expert input rounding");
      if (!opstamp(10)) return gfail("engine: stamp");
      const g::MoeStacks sg{D.bg.data(), static_cast<int>(D.bg.size())}, su{D.bu.data(), static_cast<int>(D.bu.size())},
          sd{D.bd.data(), static_cast<int>(D.bd.size())};
      if (!g::moe_prompt_multi(sg, su, sd, q, scale, dig, n, sel, wsel, moe, g::MoeCombine::kFixed64, 32, y)) return gfail("engine: bank experts");
    } else if (!opstamp(10) || !g::dzero(y, z(n * e) * sizeof(float))) {
      return gfail("engine: clear y");
    }
    if (!opstamp(11)) return gfail("engine: stamp");
    if (n == 1 && D.gate_sh.r3 && D.up_sh.r3 && D.gate_sh.k == D.up_sh.k) {
      if (!(st = prod_r3({{&D.gate_sh, gs}, {&D.up_sh, us}}, xn))) return st;
    } else if (!(st = prod(D.gate_sh, xn, n, gs)) || !(st = prod(D.up_sh, xn, n, us))) {
      return st;
    }
    if (!g::silu_mul(gs, us, gs, n * ffs)) return gfail("engine: shared silu");
    if (!(st = prod(D.down_sh, gs, n, sh))) return st;
    if (!g::matmul(static_cast<std::uint32_t>(quant::Type::f32), D.gate_inp_sh, static_cast<std::uint64_t>(e) * sizeof(float), 1, e, xn, n, e, gsh, 1) ||
        !g::deltanet_gates(gsh, galpha, zero2, zero2, n, 1))
      return gfail("engine: shared gate");
    if (!g::add_rows(y, e, sh, e, gsh, e, n)) return gfail("engine: shared add");
    if (!opstamp(12)) return gfail("engine: stamp");
    if (!g::bell_wait(fb_d, g, nullptr, tsd ? tsd + 1 : nullptr)) return gfail("engine: doorbell (cold sum)");
    if (!opstamp(13)) return gfail("engine: stamp");
    if (!g::add_rows(y, e, yc_md, e, nullptr, e, n)) return gfail("engine: doorbell (cold sum)");
    if (!opstamp(14)) return gfail("engine: stamp");
    return Status::good();
  }

  // v24: the MoE block of a 1-4 token step with the picked-expert reader: router, rounding, every routed expert on the GPU
  // (bank experts from VRAM, the rest from the registered file mapping), then the shared expert; nothing waits on the CPU
  Status moe_block_reader(Layer& D, int64_t n, std::uint32_t il) {
    const int64_t e = s.n_embd, ne = s.n_expert, nu = s.n_expert_used, ffs = s.n_ff_shexp;
    Status st;
    if (!(st = prod(D.gate_inp, xn, n, logits_r))) return st;
    if (!g::router_topk(logits_r, ne, n, static_cast<int>(nu), s.expert_weights_scale, sel, wsel, nullptr, o.router_tie)) return gfail("engine: router");
    if (!opstamp(8) || !opstamp(9)) return gfail("engine: stamp");
    if (!g::act_q8x32(xn, e, n, e, false, xr, e, q, scale, sum)) return gfail("engine: expert input rounding");
    if (!opstamp(10)) return gfail("engine: stamp");
    const bool fold = fold_ok;
    if (fold && (!g::matmul(static_cast<std::uint32_t>(quant::Type::f32), D.gate_inp_sh, static_cast<std::uint64_t>(e) * sizeof(float), 1, e, xn, n, e, gsh, 1) ||
                 !g::deltanet_gates(gsh, galpha, zero2, zero2, n, 1)))
      return gfail("engine: shared gate");
    unsigned long long* tsl = trc_ts_d ? trc_ts_d + 6 * static_cast<int64_t>(il) : nullptr;
    if (split_ok) {  // v28: the picks and the normed rows to the CPU, ring: it computes the light picks while the GPU reads the rest
      if (!g::to_host_async(sel_m, sel, z(n * nu) * sizeof(std::int32_t)) || !g::to_host_async(wsel_m, wsel, z(n * nu) * sizeof(float)) ||
          !g::to_host_async(xn_m, xn, z(n * e) * sizeof(float)) || !g::bell_ring_dev(fa_d, gen_d, il))
        return gfail("engine: split (picks to the CPU)");
    }
    if (pf_ok && il < pf_issued.size() && pf_issued[il]) {  // v26: this layer's prefetch (issued after the previous layer) is done
      if (!g::stream_wait(nullptr, ev_b[il - 1])) return gfail("engine: prefetch wait");
      pf_issued[il] = 0;
    }
    const bool rd_ok =
        o.reader_v3
            ? g::moe_reader3(rtab + static_cast<int64_t>(il) * ne, fold ? rshx + il : nullptr, sel, wsel, gsh, static_cast<int>(nu), xr, n, e, s.n_ff_exp,
                             rrow_max, rcache.empty() ? nullptr : &rcache[il], rgu, rh, ryd, rgrp, rgsrc, rgfill, y, tsl, nullptr, o.plane_tail,
                             split_ok ? lmask_d + static_cast<int64_t>(il) * ne : nullptr, o.reader_rows_per_warp)
            : g::moe_reader(rtab + static_cast<int64_t>(il) * ne, sel, wsel, static_cast<int>(nu), xr, n, e, s.n_ff_exp, rrow_max, rgu, rh, ryd, rgrp, y, tsl);
    if (!rd_ok) return gfail("engine: expert reader");
    if (trc_src_d && o.reader_v3) {
      const int64_t P = kReaderMax * (nu + 1);
      if (!g::dcopy(trc_src_d + static_cast<int64_t>(il) * P, rgsrc, z(n * (nu + 1)) * sizeof(g::ExpertPtr)) ||
          !g::dcopy(trc_fill_d + static_cast<int64_t>(il) * P, rgfill, z(n * (nu + 1)) * sizeof(g::ExpertPtr)))
        return gfail("engine: trace copy");
    }
    if (trc_sel_d && (!g::dcopy(trc_sel_d + static_cast<int64_t>(il) * kReaderMax * nu, sel, z(n * nu) * sizeof(std::int32_t)) ||
                      !g::dcopy(trc_u_d + il, rgrp, sizeof(int))))
      return gfail("engine: trace copy");
    if (!opstamp(11)) return gfail("engine: stamp");
    if (!fold) {
      if ((n == 1 || bi_now) && D.gate_sh.r3 && D.up_sh.r3 && D.gate_sh.k == D.up_sh.k) {
        if (!(st = prod_r3({{&D.gate_sh, gs}, {&D.up_sh, us}}, xn, n))) return st;
      } else if (!(st = prod(D.gate_sh, xn, n, gs)) || !(st = prod(D.up_sh, xn, n, us))) {
        return st;
      }
      if (!g::silu_mul(gs, us, gs, n * ffs)) return gfail("engine: shared silu");
      if (!(st = prod(D.down_sh, gs, n, sh))) return st;
      if (!g::matmul(static_cast<std::uint32_t>(quant::Type::f32), D.gate_inp_sh, static_cast<std::uint64_t>(e) * sizeof(float), 1, e, xn, n, e, gsh, 1) ||
          !g::deltanet_gates(gsh, galpha, zero2, zero2, n, 1))
        return gfail("engine: shared gate");
      if (!g::add_rows(y, e, sh, e, gsh, e, n)) return gfail("engine: shared add");
    }
    if (split_ok && (!g::bell_wait_dev(fb_d, gen_d, il) || !g::add_rows(y, e, yc_md, e, nullptr, e, n)))  // v28: the CPU's light sum
      return gfail("engine: split (the CPU's sum)");
    if (!opstamp(12) || !opstamp(13) || !opstamp(14)) return gfail("engine: stamp");
    return Status::good();
  }
  // v25 trace of one reader step (after the step's sync): S line, then one L line per layer
  void trace_step(int n, double wall_s) {
    const int64_t nu = s.n_expert_used, ne = s.n_expert, mp = kReaderMax * nu;
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    char b[256];
    std::snprintf(b, sizeof b, "S\t%llu\t%d\t%.3f\t%lld\n", static_cast<unsigned long long>(trace_seq++), n, wall_s * 1e3, static_cast<long long>(now_ms));
    trace_text += b;
    for (std::uint32_t il = 0; il < s.n_layer; ++il) {
      const std::int32_t* sp = trc_sel_h + static_cast<int64_t>(il) * mp;
      std::vector<int> seen;
      std::uint64_t bv = 0, bh = 0;
      int n_hot = 0, n_cold = 0;
      std::string picks;
      for (int64_t pi = 0; pi < n * nu; ++pi) {
        const int ex = sp[pi];
        picks += (pi > 0 ? (pi % nu == 0 ? ";" : ",") : "") + std::to_string(ex);
        if (ex < 0 || ex >= ne) continue;
        if (L[il].hot[z(ex)]) ++n_hot;
        else ++n_cold;
        if (std::find(seen.begin(), seen.end(), ex) != seen.end()) continue;
        seen.push_back(ex);
        const g::ExpertPtr& X = rtab_h[z(static_cast<int64_t>(il) * ne + ex)];
        const std::uint64_t bytes = X.rb[0] * s.n_ff_exp + X.rb[1] * s.n_ff_exp + X.rb[2] * s.n_embd;
        (L[il].hot[z(ex)] ? bv : bh) += bytes;
      }
      const unsigned long long* qs = trc_ts_h + 6 * il;
      auto dus = [&](int a, int c) { return static_cast<double>(static_cast<std::int64_t>(qs[c] - qs[a])) * 1e-3; };
      std::snprintf(b, sizeof b, "L\t%llu\t%u\t%d\t%d\t%d\t%d\t%llu\t%llu\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t", static_cast<unsigned long long>(trace_seq - 1), il,
                    trc_u_h[il], static_cast<int>(seen.size()), n_hot, n_cold, static_cast<unsigned long long>(bv), static_cast<unsigned long long>(bh), dus(0, 1), dus(1, 2),
                    dus(2, 3), dus(3, 4), dus(4, 5));
      trace_text += b;
      trace_text += picks;
      if (trc_src_h && o.reader_v3) {  // v26: groups served from VRAM by the cache, and misses written into a slot
        const int64_t P = kReaderMax * (nu + 1);
        const int U = trc_u_h[il];
        int hits = 0, fills = 0;
        std::uint64_t vram_b = 0, file_b = 0;
        for (int u = 0; u < U && u < P; ++u) {
          const g::ExpertPtr& X = trc_src_h[static_cast<int64_t>(il) * P + u];
          const std::uint64_t bytes = X.rb[0] * s.n_ff_exp + X.rb[1] * s.n_ff_exp + X.rb[2] * s.n_embd;
          (X.vram ? vram_b : file_b) += bytes;
          if (trc_fill_h[static_cast<int64_t>(il) * P + u].w[0]) ++fills;
        }
        for (int u = 0; u < U && u < P; ++u) {
          const g::ExpertPtr& X = trc_src_h[static_cast<int64_t>(il) * P + u];
          bool bank = false;
          for (int64_t ex = 0; ex < ne; ++ex)
            if (rtab_h[z(static_cast<int64_t>(il) * ne + ex)].w[0] == X.w[0] && rtab_h[z(static_cast<int64_t>(il) * ne + ex)].vram) bank = true;
          if (X.vram && !bank && u < static_cast<int>(seen.size())) ++hits;
        }
        std::snprintf(b, sizeof b, "\t%d\t%d\t%llu\t%llu", hits, fills, static_cast<unsigned long long>(vram_b), static_cast<unsigned long long>(file_b));
        trace_text += b;
      }
      trace_text += "\n";
    }
  }
  // v24: register the span of the model file holding the trunk's routed experts, and build the reader's tables
  void setup_reader() {
    const int64_t ne = s.n_expert;
    if (s.n_embd % 32 != 0 || s.n_ff_exp % 32 != 0) {
      reader_note = "off (shapes not a multiple of 32)";
      return;
    }
    std::uintptr_t lo = UINTPTR_MAX, hi = 0;
    for (std::uint32_t il = 0; il < s.n_layer; ++il)
      for (const WeightView* v : {&w->layers[il].gate_exps, &w->layers[il].up_exps, &w->layers[il].down_exps}) {
        if (v->rot32) {
          reader_note = "off (a rotated expert weight)";
          return;
        }
        for (int64_t ex = 0; ex < ne; ++ex) {
          const ExpRows r = exp_rows(*v, static_cast<int>(ex));
          if (!g::type_supported(static_cast<std::uint32_t>(r.t))) {
            reader_note = "off (an expert type the reader cannot decode)";
            return;
          }
          const auto a = reinterpret_cast<std::uintptr_t>(r.base);
          lo = std::min(lo, a);
          hi = std::max(hi, a + static_cast<std::uintptr_t>(r.rb * v->ne[1]));
        }
      }
    if (hi <= lo) return;
    // v28 the author's split: light experts (every matrix Q4_TG2, not in the bank) go to the CPU and are not pinned
    const bool split = o.cpu_light && o.doorbell && o.reader_v3;
    int n_light_all = 0;
    std::uint64_t light_b = 0;
    for (std::uint32_t il = 0; il < s.n_layer; ++il) {
      L[il].light.assign(z(ne), 0);
      L[il].n_light = 0;
      if (!split) continue;
      const WeightView* vv[3] = {&w->layers[il].gate_exps, &w->layers[il].up_exps, &w->layers[il].down_exps};
      for (int64_t ex = 0; ex < ne; ++ex) {
        if (L[il].hot[z(ex)]) continue;
        bool lt = true;
        std::uint64_t b = 0;
        for (const WeightView* v : vv) {
          const ExpRows r = exp_rows(*v, static_cast<int>(ex));
          if (r.t != quant::Type::q4_tg2) lt = false;
          b += r.rb * v->ne[1];
        }
        if (!lt) continue;
        L[il].light[z(ex)] = 1;
        ++L[il].n_light;
        ++n_light_all;
        light_b += b;
      }
    }
    // v27: pin only the experts the bank does not hold (the GPU reads the bank's from VRAM): their matrices as page-aligned
    // ranges, merged where the gap between them is under o.pin_gap (v27: 8 MiB; fewer, larger registrations); each range's
    // device address. v28: not the light experts either when the split is on.
    const std::uintptr_t pg = 4096;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> want;
    for (std::uint32_t il = 0; il < s.n_layer; ++il)
      for (const WeightView* v : {&w->layers[il].gate_exps, &w->layers[il].up_exps, &w->layers[il].down_exps})
        for (int64_t ex = 0; ex < ne; ++ex) {
          if (L[il].hot[z(ex)] || L[il].light[z(ex)]) continue;
          const ExpRows r = exp_rows(*v, static_cast<int>(ex));
          const auto a = reinterpret_cast<std::uintptr_t>(r.base);
          want.emplace_back(a & ~(pg - 1), (a + static_cast<std::uintptr_t>(r.rb * v->ne[1]) + pg - 1) & ~(pg - 1));
        }
    std::sort(want.begin(), want.end());
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> runs;
    for (const auto& iv : want) {
      if (!runs.empty() && iv.first <= runs.back().second + static_cast<std::uintptr_t>(o.pin_gap)) runs.back().second = std::max(runs.back().second, iv.second);
      else runs.push_back(iv);
    }
    std::vector<const std::uint8_t*> run_dev;
    std::uint64_t pinned = 0;
    std::string why;
    const Clock::time_point tr = Clock::now();
    std::size_t small_ok = 0;
    if (o.pin_small_pages)
      for (const auto& rn : runs)
        if (hal::advise_small_pages(reinterpret_cast<void*>(rn.first), static_cast<std::size_t>(rn.second - rn.first))) ++small_ok;
    for (const auto& rn : runs) {
      void* dv = nullptr;
      if (!g::host_register(reinterpret_cast<void*>(rn.first), static_cast<std::size_t>(rn.second - rn.first), &dv, &why)) {
        for (void* rb2 : reg_bases) g::host_unregister(rb2);
        reg_bases.clear();
        reader_note = "off (registering " + std::to_string((rn.second - rn.first) >> 20) + " MiB of the file mapping failed: " + why + ")";
        return;
      }
      reg_bases.push_back(reinterpret_cast<void*>(rn.first));
      run_dev.push_back(static_cast<const std::uint8_t*>(dv));
      pinned += rn.second - rn.first;
    }
    auto dev_of = [&](const std::uint8_t* hp) -> const std::uint8_t* {
      const auto a = reinterpret_cast<std::uintptr_t>(hp);
      auto it = std::upper_bound(runs.begin(), runs.end(), std::make_pair(a, UINTPTR_MAX));
      if (it == runs.begin()) return nullptr;
      --it;
      if (a >= it->second) return nullptr;
      return run_dev[z(it - runs.begin())] + (a - it->first);
    };
    const std::uintptr_t b0 = 0, b1 = pinned;
    const double reg_s = since(tr);
    std::vector<g::ExpertPtr> tab(z(static_cast<int64_t>(s.n_layer) * ne));
    for (std::uint32_t il = 0; il < s.n_layer; ++il) {
      const Qwen35MoeLayer& l = w->layers[il];
      const WeightView* vv[3] = {&l.gate_exps, &l.up_exps, &l.down_exps};
      for (int64_t ex = 0; ex < ne; ++ex) {
        g::ExpertPtr& X = tab[z(static_cast<int64_t>(il) * ne + ex)];
        for (int pj = 0; pj < 3; ++pj) {
          const ExpRows r = exp_rows(*vv[pj], static_cast<int>(ex));
          const std::uint8_t* bank = L[il].hot[z(ex)] && !L[il].bank_at[pj].empty() ? L[il].bank_at[pj][z(ex)] : nullptr;
          X.w[pj] = bank ? bank : L[il].light[z(ex)] ? nullptr : dev_of(r.base);
          X.type[pj] = static_cast<std::uint32_t>(r.t);
          X.rb[pj] = r.rb;
          if (bank) X.vram = 1;
        }
      }
    }
    for (std::size_t ti = 0; ti < tab.size(); ++ti) {
      const g::ExpertPtr& X = tab[ti];
      const std::uint32_t til = static_cast<std::uint32_t>(static_cast<int64_t>(ti) / ne);
      if (L[til].light[z(static_cast<int64_t>(ti) % ne)]) continue;  // v28: the CPU's, never read by the GPU
      if (!X.w[0] || !X.w[1] || !X.w[2]) {
        reader_note = "off (an expert outside the pinned ranges)";
        return;
      }
    }
    if (split && n_light_all > 0) {  // v28: the mask; the doorbell generation on the device (written before each step)
      std::vector<std::uint8_t> mask(z(static_cast<int64_t>(s.n_layer) * ne), 0);
      for (std::uint32_t il = 0; il < s.n_layer; ++il)
        for (int64_t ex = 0; ex < ne; ++ex) mask[z(static_cast<int64_t>(il) * ne + ex)] = L[il].light[z(ex)];
      lmask_d = static_cast<std::uint8_t*>(g::dalloc(mask.size()));
      gen_d = static_cast<unsigned*>(g::dalloc(256));
      void* gdev = nullptr;
      gen_hv = g::host_alloc_mapped(64, &gdev);
      if (!lmask_d || !gen_d || !gen_hv || !g::to_device(lmask_d, mask.data(), mask.size())) {
        reader_note = "off (device memory for the split's mask)";
        return;
      }
      gen_hp = static_cast<unsigned*>(gen_hv);
      dev_bytes += mask.size() + 256;
      char sb[200];
      std::snprintf(sb, sizeof sb, "on (%d light experts of %lld, %.2f GB, computed by the CPU from the unpinned file; %d CPU threads)", n_light_all,
                    static_cast<long long>(ne) * static_cast<long long>(s.n_layer), static_cast<double>(light_b) / 1e9,
                    o.cpu_threads > 0 ? o.cpu_threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
      split_note = sb;
      if (o.preload_light) {  // v30: touch every page of the light experts once, in place, while the engine runs
        std::vector<std::pair<const std::uint8_t*, std::uint64_t>> spans;
        for (std::uint32_t il = 0; il < s.n_layer; ++il)
          for (const WeightView* v : {&w->layers[il].gate_exps, &w->layers[il].up_exps, &w->layers[il].down_exps})
            for (int64_t ex = 0; ex < ne; ++ex)
              if (L[il].light[z(ex)]) {
                const ExpRows r = exp_rows(*v, static_cast<int>(ex));
                spans.emplace_back(r.base, r.rb * v->ne[1]);
              }
        preload_on = true;
        preload_th = std::thread([this, spans = std::move(spans)] {
          const Clock::time_point t0 = Clock::now();
          volatile std::uint8_t sink = 0;
          std::uint64_t b = 0;
          for (const auto& sp : spans) {
            if (preload_stop.load(std::memory_order_relaxed)) break;
            for (std::uint64_t i = 0; i < sp.second; i += 4096) sink = static_cast<std::uint8_t>(sink ^ sp.first[i]);
            sink = static_cast<std::uint8_t>(sink ^ sp.first[sp.second - 1]);
            b += sp.second;
          }
          (void)sink;
          preload_b = b;
          preload_s = since(t0);
        });
      }
    } else if (o.cpu_light) {
      split_note = split ? "off (no light experts outside the bank)" : "off (needs the doorbell and reader v3)";
    }
    // v26: the shared expert as a pick (its three matrices as the batched kernels read them, in VRAM)
    std::vector<g::ExpertPtr> shx_h;
    if (o.fold_shared && o.reader_v3 && s.n_ff_shexp == s.n_ff_exp) {
      bool ok = true;
      for (std::uint32_t il = 0; il < s.n_layer && ok; ++il) {
        const Layer& D = L[il];
        g::ExpertPtr X;
        const DevW* dw[3] = {&D.gate_sh, &D.up_sh, &D.down_sh};
        for (int pj = 0; pj < 3; ++pj) {
          const DevW& W = *dw[pj];
          if (!W.d || !W.v || W.v->rot32 || !g::type_supported(W.dtype) || W.k != (pj < 2 ? s.n_embd : s.n_ff_shexp)) ok = false;
          X.w[pj] = static_cast<const std::uint8_t*>(W.d);
          X.type[pj] = W.dtype;
          X.rb[pj] = W.drb;
        }
        X.vram = 1;
        shx_h.push_back(X);
      }
      if (ok) fold_ok = true;
      else shx_h.clear();
    }
    const int64_t mp = kReaderMax * static_cast<int64_t>(s.n_expert_used), ff = s.n_ff_exp;
    for (const g::ExpertPtr& X : tab)
      for (int pj = 0; pj < 3; ++pj) rrow_max = std::max<std::uint64_t>(rrow_max, X.rb[pj]);
    for (const g::ExpertPtr& X : shx_h)
      for (int pj = 0; pj < 3; ++pj) rrow_max = std::max<std::uint64_t>(rrow_max, X.rb[pj]);
    if (16384 + 8 * ((rrow_max + 32 + 15) & ~std::uint64_t{15}) > 48 * 1024) {
      reader_note = "off (expert rows too long for the reader's shared staging)";
      fold_ok = false;
      return;
    }
    const int64_t P = kReaderMax * (static_cast<int64_t>(s.n_expert_used) + 1);  // v26: room for the shared expert's picks
    (void)mp;
    rtab = static_cast<g::ExpertPtr*>(g::dalloc(tab.size() * sizeof(g::ExpertPtr)));
    rgu = static_cast<float*>(g::dalloc(z(P * 2 * ff) * sizeof(float)));
    rh = static_cast<float*>(g::dalloc(z(P * ff) * sizeof(float)));
    ryd = static_cast<float*>(g::dalloc(z(P * s.n_embd) * sizeof(float)));
    rgrp = static_cast<int*>(g::dalloc(z(1 + 6 * P) * sizeof(int)));
    rgsrc = static_cast<g::ExpertPtr*>(g::dalloc(z(P) * sizeof(g::ExpertPtr)));
    rgfill = static_cast<g::ExpertPtr*>(g::dalloc(z(P) * sizeof(g::ExpertPtr)));
    if (!shx_h.empty()) {
      rshx = static_cast<g::ExpertPtr*>(g::dalloc(shx_h.size() * sizeof(g::ExpertPtr)));
      if (!rshx || !g::to_device(rshx, shx_h.data(), shx_h.size() * sizeof(g::ExpertPtr))) fold_ok = false;
    }
    if (!rtab || !rgu || !rh || !ryd || !rgrp || !rgsrc || !rgfill || !g::to_device(rtab, tab.data(), tab.size() * sizeof(g::ExpertPtr))) {
      reader_note = "off (device memory for the reader's tables)";
      fold_ok = false;
      return;
    }
    dev_bytes += tab.size() * sizeof(g::ExpertPtr) + z(P * 3 * ff + P * s.n_embd) * sizeof(float) + z(1 + 6 * P) * sizeof(int) + z(2 * P) * sizeof(g::ExpertPtr);
    // v26 cache: the bytes reserved before the bank was filled
    std::string cache_note;
    if (cache_reserved > 0 && o.reader_v3) {
      const int K = o.reader_cache;
      const std::size_t meta_b = static_cast<std::size_t>(s.n_layer) * (static_cast<std::size_t>(K) * 8 + 16);
      cache_mem = g::dalloc(cache_reserved);
      cache_meta = g::dalloc(meta_b);
      if (cache_mem && cache_meta) {
        std::vector<std::uint8_t> init(meta_b, 0);
        auto* ip = reinterpret_cast<int*>(init.data());
        for (std::size_t i = 0; i < static_cast<std::size_t>(s.n_layer) * static_cast<std::size_t>(K); ++i) ip[i] = -1;  // tags: empty
        if (g::to_device(cache_meta, init.data(), meta_b)) {
          auto* mb = static_cast<std::uint8_t*>(cache_meta);
          const std::size_t nl = s.n_layer;
          for (std::size_t il = 0; il < nl; ++il) {
            g::ReaderCache c;
            c.K = K;
            c.tag = reinterpret_cast<int*>(mb) + il * static_cast<std::size_t>(K);
            c.use = reinterpret_cast<unsigned*>(mb + nl * static_cast<std::size_t>(K) * 4) + il * static_cast<std::size_t>(K);
            c.stamp = reinterpret_cast<unsigned*>(mb + nl * static_cast<std::size_t>(K) * 8) + il;
            c.base = static_cast<std::uint8_t*>(cache_mem) + il * static_cast<std::size_t>(K) * cslot;
            c.slot_bytes = cslot;
            rcache.push_back(c);
          }
          dev_bytes += cache_reserved + meta_b;
          char cb[128];
          std::snprintf(cb, sizeof cb, "; cache %d slots per layer, %.2f GB", K, static_cast<double>(cache_reserved) / 1e9);
          cache_note = cb;
        }
      }
      if (rcache.empty()) cache_note = "; cache off (device memory)";
    }
    rtab_h = tab;
    if (o.trace) {
      const std::size_t nl = s.n_layer, b_ts = nl * 6 * sizeof(unsigned long long), b_sel = nl * z(mp) * sizeof(std::int32_t), b_u = nl * sizeof(int);
      const std::size_t b_src = nl * z(P) * sizeof(g::ExpertPtr);
      void* dv = nullptr;
      trc_hv = g::host_alloc_mapped(b_ts + b_sel + b_u + 64 + 2 * b_src, &dv);
      if (trc_hv) {
        auto* hb = static_cast<std::uint8_t*>(trc_hv);
        auto* db = static_cast<std::uint8_t*>(dv);
        trc_ts_h = reinterpret_cast<unsigned long long*>(hb);
        trc_ts_d = reinterpret_cast<unsigned long long*>(db);
        trc_sel_h = reinterpret_cast<std::int32_t*>(hb + b_ts);
        trc_sel_d = reinterpret_cast<std::int32_t*>(db + b_ts);
        trc_u_h = reinterpret_cast<int*>(hb + b_ts + b_sel);
        trc_u_d = reinterpret_cast<int*>(db + b_ts + b_sel);
        const std::size_t o2 = (b_ts + b_sel + b_u + 63) & ~std::size_t{63};
        trc_src_h = reinterpret_cast<g::ExpertPtr*>(hb + o2);
        trc_src_d = reinterpret_cast<g::ExpertPtr*>(db + o2);
        trc_fill_h = reinterpret_cast<g::ExpertPtr*>(hb + o2 + b_src);
        trc_fill_d = reinterpret_cast<g::ExpertPtr*>(db + o2 + b_src);
      }
    }
    reader = true;
    char b[160];
    char b2[640];
    std::snprintf(b2, sizeof b2, "on (%s, %d rows per warp, steps of 1-%d tokens; %.2f GB of the file (%s) registered in %zu ranges (gap %llu KiB; small pages advised on %zu) in %.2f s%s%s)",
                  o.reader_v3 ? "v3" : "v25", o.reader_rows_per_warp, kReaderMax, static_cast<double>(b1 - b0) / 1e9,
                  lmask_d ? "the heavy experts outside the bank" : "the experts outside the bank", runs.size(),
                  static_cast<unsigned long long>(o.pin_gap >> 10), small_ok, reg_s, fold_ok ? "; shared expert folded in" : "", cache_note.c_str());
    reader_note = b2;
    (void)b;
  }

  Status reset_state() {
    pos = 0;
    mtp_pos = 0;
    last_n = 0;
    for (Layer& D : L) {
      D.cur = 0;
      if (D.state && !g::dzero(D.state, z(s.ssm_dt_rank * s.ssm_d_state * s.ssm_d_state) * sizeof(float))) return gfail("engine: state reset");
      for (float* tl : D.tail)
        if (tl && !g::dzero(tl, z((s.ssm_d_conv - 1) * s.conv_channels()) * sizeof(float))) return gfail("engine: conv tail reset");
    }
    return g::sync() ? Status::good() : gfail("engine: reset sync");
  }
};

GpuEngine::GpuEngine(std::unique_ptr<Impl> p) : p_(std::move(p)) {}

bool GpuEngine::preload_wait(double* seconds, std::uint64_t* bytes) {
  if (!p_ || !p_->preload_on) return false;
  if (p_->preload_th.joinable()) p_->preload_th.join();
  if (seconds) *seconds = p_->preload_s;
  if (bytes) *bytes = p_->preload_b;
  return true;
}

GpuEngine::~GpuEngine() {
  if (p_ && p_->preload_th.joinable()) {
    p_->preload_stop.store(true, std::memory_order_relaxed);
    p_->preload_th.join();
  }
  if (!p_) return;
  if (p_->bell_th.joinable()) {
    {
      std::lock_guard<std::mutex> lk(p_->bmu);
      p_->bstop = true;
    }
    p_->bcv.notify_all();
    p_->bell_th.join();
  }
  for (auto& kv : p_->graphs) g::graph_free(kv.second);
  p_->graphs.clear();
  for (void* rb2 : p_->reg_bases) g::host_unregister(rb2);
  g::dfree(p_->rtab);
  g::dfree(p_->rgu);
  g::dfree(p_->rh);
  g::dfree(p_->ryd);
  g::dfree(p_->rgrp);
  g::dfree(p_->rgsrc);
  g::dfree(p_->rgfill);
  g::dfree(p_->rshx);
  g::dfree(p_->cache_mem);
  g::dfree(p_->cache_meta);
  g::host_free(p_->trc_hv);
  g::dfree(p_->probe_bd);
  g::dfree(p_->probe_ld);
  g::dfree(p_->probe_md);
  g::host_free(p_->bell_h);
  g::host_free(p_->ts_hv);
  g::host_free(p_->ops_hv);
  g::dfree(p_->probe_xd);
  g::dfree(p_->probe_seld);
  g::dfree(p_->probe_wd);
  g::moe_scratch_free(&p_->moe);
  for (Layer& D : p_->L) g::dfree(D.bank_mem);
  g::dfree(p_->M.bank_mem);
  p_->r3plans.clear();
  auto rel = [](DevW& d) {
    if (d.r3) g::stream_release(&d.sw);
  };
  for (Layer& D : p_->L)
    for (DevW* d : {&D.wq, &D.wk, &D.wv, &D.wo, &D.wqkv, &D.wgate, &D.wbeta, &D.walpha, &D.ssm_out, &D.gate_inp, &D.gate_sh, &D.up_sh, &D.down_sh}) rel(*d);
  for (DevW* d : {&p_->M.wq, &p_->M.wk, &p_->M.wv, &p_->M.wo, &p_->M.gate_inp, &p_->M.gate_sh, &p_->M.up_sh, &p_->M.down_sh, &p_->eh, &p_->out}) rel(*d);
  if (p_->head.r3 && p_->head.sw.w != p_->out.sw.w) rel(p_->head);
  g::dfree(p_->act_mem);
  g::dfree(p_->dense_mem);
  g::dfree(p_->state_mem);
  if (p_->side) {  // v26 prefetch
    g::sync(p_->side);
    g::stream_free(p_->side);
  }
  for (void* ev : p_->ev_a) g::event_free(ev);
  for (void* ev : p_->ev_b) g::event_free(ev);
  g::dfree(p_->xp);
  g::dfree(p_->lp);
  g::dfree(p_->wselp);
  g::dfree(p_->selp);
  g::dfree(p_->pfjob);
  if (p_->stm) {  // v26
    g::sync();
    g::set_stream(nullptr);
    g::stream_free(p_->stm);
  }
  g::dfree(p_->pos_d);
  g::dfree(p_->knew);
  g::dfree(p_->vnew);
  g::host_free(p_->pos_hv);
  g::dfree(p_->lmask_d);
  g::dfree(p_->gen_d);
  g::host_free(p_->gen_hv);
}

std::int64_t GpuEngine::pos() const { return p_ ? p_->pos : 0; }
std::uint64_t GpuEngine::device_bytes() const { return p_ ? p_->dev_bytes : 0; }
int GpuEngine::bank_experts() const { return p_ ? p_->bank_count : 0; }
int GpuEngine::max_batch() const { return p_ ? p_->o.max_batch : 0; }
std::uint64_t GpuEngine::bank_bytes() const { return p_ ? p_->bank_bytes_total : 0; }
Status GpuEngine::reset() { return p_->reset_state(); }
const EngineTimeline& GpuEngine::timeline() const { return p_->tline; }
void GpuEngine::probe_set(std::int64_t pos) {
  Impl& I = *p_;
  I.probe_have = false;
  I.probe_pos = pos;
  if (pos >= 0 && !I.probe_xd) {
    const std::size_t nl = I.s.n_layer, nuq = I.s.n_expert_used;
    I.probe_xd = static_cast<float*>(g::dalloc(nl * I.s.n_embd * sizeof(float)));
    I.probe_seld = static_cast<std::int32_t*>(g::dalloc(nl * nuq * sizeof(std::int32_t)));
    I.probe_wd = static_cast<float*>(g::dalloc(nl * nuq * sizeof(float)));
    I.probe_bd = static_cast<float*>(g::dalloc(nl * I.s.n_embd * sizeof(float)));
    I.probe_ld = static_cast<float*>(g::dalloc(nl * I.s.n_expert * sizeof(float)));
    I.probe_md = static_cast<float*>(g::dalloc(nl * I.s.n_embd * sizeof(float)));
    if (!I.probe_xd || !I.probe_seld || !I.probe_wd || !I.probe_bd || !I.probe_ld || !I.probe_md) I.probe_pos = -1;
  }
}
bool GpuEngine::probe_get(std::vector<float>* x_layers, std::vector<std::int32_t>* picks, std::vector<float>* pick_w) {
  Impl& I = *p_;
  if (!I.probe_have) return false;
  if (x_layers) *x_layers = I.probe_xh;
  if (picks) *picks = I.probe_selh;
  if (pick_w) *pick_w = I.probe_wh;
  return true;
}
bool GpuEngine::probe_get_ops(std::vector<float>* block_out, std::vector<float>* router_logits, std::vector<float>* moe_out) {
  Impl& I = *p_;
  if (I.probe_bh.empty()) return false;
  if (block_out) *block_out = I.probe_bh;
  if (router_logits) *router_logits = I.probe_lh;
  if (moe_out) *moe_out = I.probe_mh;
  return true;
}
std::string GpuEngine::graph_report() const {
  return p_->graph_note + "; captures " + std::to_string(p_->graph_captures) + ", replays " + std::to_string(p_->graph_replays);
}
std::string GpuEngine::trace_take() {
  std::string out;
  out.swap(p_->trace_text);
  return out;
}

Status GpuEngine::bank_selftest(int combos, std::string* report, int* bad) {
  Impl& I = *p_;
  const Qwen35MoeSpec& s = I.s;
  const int64_t e = s.n_embd, ne = s.n_expert, nu = s.n_expert_used;
  std::uint32_t rng = 12345u;
  auto rnd = [&]() {
    rng = rng * 1664525u + 1013904223u;
    return rng;
  };
  std::vector<float> xh(z(2 * e)), yg(z(2 * e)), yc(z(2 * e));
  for (float& v : xh) v = static_cast<float>(static_cast<int>(rnd() >> 8) % 20001 - 10000) * 1e-4f;
  struct Worst {
    double rel;
    int layer, expert, rows;
  };
  std::vector<Worst> all;
  int checked = 0, nbad = 0, ncombo = 0;
  double max_rel = 0.0;
  for (std::uint32_t il = 0; il < s.n_layer; ++il) {
    Layer& D = I.L[il];
    if (D.bg.empty()) continue;
    const Qwen35MoeLayer& l = I.w->layers[il];
    std::vector<int> hot, cl;
    for (int x = 0; x < static_cast<int>(ne); ++x) (D.hot[z(x)] ? hot : cl).push_back(x);
    if (static_cast<int64_t>(cl.size()) < nu) continue;  // fillers must be experts outside the bank
    const g::MoeStacks sg{D.bg.data(), static_cast<int>(D.bg.size())}, su{D.bu.data(), static_cast<int>(D.bu.size())},
        sd{D.bd.data(), static_cast<int>(D.bd.size())};
    // one GPU call and the CPU path on the same picks; returns the worst row's relative L2 gap
    auto run = [&](int n, const std::vector<std::int32_t>& pk, const std::vector<float>& wt, double* rel) -> Status {
      if (!g::to_device(I.xn, xh.data(), z(n * e) * sizeof(float)) || !g::act_q8x32(I.xn, e, n, e, false, I.xr, e, I.q, I.scale, I.sum) ||
          !g::act_digits(I.scale, I.sum, n, e, I.dig) || !g::to_device(I.sel, pk.data(), pk.size() * sizeof(std::int32_t)) ||
          !g::to_device(I.wsel, wt.data(), wt.size() * sizeof(float)) ||
          !g::moe_prompt_multi(sg, su, sd, I.q, I.scale, I.dig, n, I.sel, I.wsel, I.moe, g::MoeCombine::kFixed64, 32, I.y) ||
          !g::to_host(yg.data(), I.y, z(n * e) * sizeof(float)))
        return gfail("engine: bank self-test (GPU)");
      I.cold.clear();
      for (int t = 0; t < n; ++t)
        for (int64_t j = 0; j < nu; ++j) {
          const int x = pk[z(t * nu + j)];
          if (D.hot[z(x)]) I.cold.push_back(ColdPick{t, x, wt[z(t * nu + j)]});
        }
      I.cold_experts_io(l, n, xh.data(), yc.data());
      double worst = 0.0;
      for (int t = 0; t < n; ++t) {
        double num = 0.0, den = 0.0;
        for (int64_t i = 0; i < e; ++i) {
          const double dd = static_cast<double>(yg[z(t * e + i)]) - yc[z(t * e + i)];
          num += dd * dd;
          den += static_cast<double>(yc[z(t * e + i)]) * yc[z(t * e + i)];
        }
        worst = std::max(worst, den > 0 ? std::sqrt(num / den) : std::sqrt(num));
      }
      *rel = worst;
      return Status::good();
    };
    for (const int x : hot)
      for (int n = 1; n <= 2; ++n) {
        std::vector<std::int32_t> pk(z(n * nu));
        std::vector<float> wt(z(n * nu), 0.0f);
        for (int t = 0; t < n; ++t) {
          pk[z(t * nu)] = x;
          wt[z(t * nu)] = 1.0f;
          for (int64_t j = 1; j < nu; ++j) pk[z(t * nu + j)] = cl[z(j - 1)];
        }
        double rel = 0.0;
        if (Status st = run(n, pk, wt, &rel); !st) return st;
        ++checked;
        max_rel = std::max(max_rel, rel);
        if (rel > 1e-2) ++nbad;
        all.push_back(Worst{rel, static_cast<int>(il), x, n});
      }
    for (int c = 0; c < combos; ++c) {  // random 8-pick mixes over the whole layer (bank and cold), 1 token
      std::vector<std::int32_t> pk;
      std::vector<float> wt;
      float tot = 0.0f;
      while (static_cast<int64_t>(pk.size()) < nu) {
        const int x = static_cast<int>(rnd() % static_cast<std::uint32_t>(ne));
        if (std::find(pk.begin(), pk.end(), x) != pk.end()) continue;
        pk.push_back(x);
        wt.push_back(static_cast<float>(rnd() % 1000 + 1));
        tot += wt.back();
      }
      for (float& v : wt) v /= tot;
      double rel = 0.0;
      if (Status st = run(1, pk, wt, &rel); !st) return st;
      ++ncombo;
      max_rel = std::max(max_rel, rel);
      if (rel > 1e-2) ++nbad;
      all.push_back(Worst{rel, static_cast<int>(il), -1, 1});
    }
  }
  std::sort(all.begin(), all.end(), [](const Worst& a, const Worst& b) { return a.rel > b.rel; });
  std::string r;
  char buf[256];
  std::snprintf(buf, sizeof buf, "bank self-test: %d single-expert checks (1 and 2 tokens) + %d mixes; worst relative L2 gap GPU vs CPU %.3e; %d above 1e-2",
                checked, ncombo, max_rel, nbad);
  r += buf;
  if (!all.empty()) {
    const double med = all[all.size() / 2].rel;
    std::snprintf(buf, sizeof buf, "; median %.3e", med);
    r += buf;
  }
  for (std::size_t i = 0; i < all.size() && i < 12; ++i) {
    const Worst& w8 = all[i];
    std::string tys;
    if (w8.expert >= 0)
      for (const WeightView* v : {&I.w->layers[z(w8.layer)].gate_exps, &I.w->layers[z(w8.layer)].up_exps, &I.w->layers[z(w8.layer)].down_exps})
        tys += std::string(tys.empty() ? "" : "/") + quant::type_info(exp_type(*v, w8.expert)).name;
    std::snprintf(buf, sizeof buf, "\n    layer %2d %s %3d, %d token%s: %.3e %s", w8.layer, w8.expert >= 0 ? "expert" : "mix   ", w8.expert, w8.rows,
                  w8.rows > 1 ? "s" : " ", w8.rel, tys.c_str());
    r += buf;
  }
  if (report) *report = r;
  if (bad) *bad = nbad;
  return g::sync() ? Status::good() : gfail("engine: bank self-test sync");
}
void GpuEngine::timeline_reset() {
  EngineTimeline& t = p_->tline;
  const int th = t.threads;
  const std::size_t nl = t.gpu_dense.size();
  t = EngineTimeline{};
  t.threads = th;
  for (std::vector<double>* v : {&t.gpu_dense, &t.gpu_hot, &t.gpu_idle, &t.cpu_wait, &t.cpu_cold}) v->assign(nl, 0.0);
}

std::string GpuEngine::summary() const {
  const Impl& I = *p_;
  char b[2048];
  std::snprintf(b, sizeof b, "engine: %.2f GB on the device (dense + output %.2f GB, bank %.2f GB = %d experts of %u, cache + state %.2f GB); %d CPU threads; ctx %d, batch %d; hand-off %s; router tie band %g; expert reader %s; batch-identical %s; graph %s; prefetch %s; split %s",
                static_cast<double>(I.dev_bytes) / 1e9, static_cast<double>(I.dense_bytes_total) / 1e9, static_cast<double>(I.bank_bytes_total) / 1e9, I.bank_count,
                I.s.n_layer * I.s.n_expert, static_cast<double>(I.cache_bytes) / 1e9, I.pool ? I.pool->threads() : 0, I.o.max_ctx, I.o.max_batch,
                I.bell ? "doorbell" : "device sync per layer", static_cast<double>(I.o.router_tie), I.reader_note.c_str(), I.o.batch_identical ? "on" : "off",
                I.graph_note.c_str(), I.pf_ok ? "on" : "off", I.split_note.c_str());
  return b;
}

Result<std::unique_ptr<GpuEngine>> GpuEngine::open(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, const EngineOptions& o) {
  using R = Result<std::unique_ptr<GpuEngine>>;
  if (!g::available()) return R::fail(Status::fail(Code::no_device, std::string("engine: ") + g::last_error()));
  if (s.family != Family::qwen35moe || s.dense) return R::fail(Status::fail(Code::unsupported, "engine: qwen35moe only in this first cut"));
  if (o.max_ctx < 1 || o.max_ctx > (1 << 20) || o.max_batch < 1 || o.max_batch > 4096)
    return R::fail(Status::fail(Code::invalid_argument, "engine: max_ctx 1 .. 2^20, max_batch 1 .. 4096"));
  if (w.layers.size() != s.n_layer) return R::fail(Status::fail(Code::invalid_argument, "engine: the weights were bound for another spec"));
  if (!w.output.present() || w.output_norm.empty()) return R::fail(Status::fail(Code::invalid_argument, "engine: no output layer"));
  if (!o.hot_order.empty() && o.hot_order.size() != s.n_layer) return R::fail(Status::fail(Code::invalid_argument, "engine: hot order for another layer count"));
  for (const Qwen35MoeLayer& l : w.layers) {
    if (!l.bq.empty() || !l.attn_sub_norm.empty()) return R::fail(Status::fail(Code::unsupported, "engine: biases / sub-norms (qwen35moe only)"));
    for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps})
      if (v->rot32) return R::fail(Status::fail(Code::unsupported, "engine: rotated routed experts (" + v->name + ")"));
  }
  auto p = std::make_unique<Impl>();
  Impl& I = *p;
  I.s = s;
  I.w = &w;
  I.o = o;
  I.npad = (o.max_batch + 63) / 64 * 64 + 64;
  const int64_t e = s.n_embd, hd = s.head_dim, nh = s.n_head, nkv = s.n_head_kv, qh = s.attn_gate ? 2 * hd : hd;
  const int64_t ch = s.conv_channels(), di = s.ssm_d_inner, ds = s.ssm_d_state, nv = s.ssm_dt_rank, ne = s.n_expert, nu = s.n_expert_used;
  const int64_t ffs = s.n_ff_shexp, ff = s.n_ff_exp, taps = s.ssm_d_conv;
  I.kv_w = hd * nkv;
  I.has_mtp = o.mtp;
  I.use_r3 = o.r3;
  if (o.mtp && !w.mtp.present) return R::fail(Status::fail(Code::unsupported, "engine: this file has no MTP block"));
  if (o.max_verify > 1 && !(ds >= 32 && ds <= 1024 && (ds & (ds - 1)) == 0))
    return R::fail(Status::fail(Code::unsupported, "engine: verify batches need the DeltaNet d_state a power of two (the checkpointing recurrence)"));
  if (o.max_verify > o.max_batch) return R::fail(Status::fail(Code::invalid_argument, "engine: max_verify above max_batch"));
  I.kmax = std::max<int64_t>({e, hd * nh, di, ffs, ff, o.mtp ? 2 * e : e});
  // activations
  const std::size_t np = z(I.npad);
  struct Buf {
    void** ptr;
    std::size_t bytes;
  };
  std::vector<Buf> bufs = {
      {reinterpret_cast<void**>(&I.x), np * z(e) * 4},          {reinterpret_cast<void**>(&I.xn), np * z(e) * 4},
      {reinterpret_cast<void**>(&I.y), np * z(e) * 4},          {reinterpret_cast<void**>(&I.xr), np * z(I.kmax) * 4},
      {reinterpret_cast<void**>(&I.q), np * z(I.kmax)},         {reinterpret_cast<void**>(&I.scale), np * z(I.kmax / 32) * 4},
      {reinterpret_cast<void**>(&I.sum), np * z(I.kmax / 32) * 4}, {reinterpret_cast<void**>(&I.dig), np * z(I.kmax / 32) * 4},
      {reinterpret_cast<void**>(&I.qg), np * z(qh * nh) * 4},   {reinterpret_cast<void**>(&I.att), np * z(hd * nh) * 4},
      {reinterpret_cast<void**>(&I.ascr), np * z(nh) * z(o.max_ctx) * 4},
      {reinterpret_cast<void**>(&I.mixed), np * z(ch) * 4},     {reinterpret_cast<void**>(&I.zg), np * z(di) * 4},
      {reinterpret_cast<void**>(&I.beta), np * z(nv) * 4},      {reinterpret_cast<void**>(&I.alpha), np * z(nv) * 4},
      {reinterpret_cast<void**>(&I.conv), np * z(ch) * 4},      {reinterpret_cast<void**>(&I.core), np * z(di) * 4},
      {reinterpret_cast<void**>(&I.logits_r), np * z(ne) * 4},  {reinterpret_cast<void**>(&I.sel), np * z(nu) * 4},
      {reinterpret_cast<void**>(&I.wsel), np * z(nu) * 4},      {reinterpret_cast<void**>(&I.gs), np * z(ffs) * 4},
      {reinterpret_cast<void**>(&I.us), np * z(ffs) * 4},       {reinterpret_cast<void**>(&I.sh), np * z(e) * 4},
      {reinterpret_cast<void**>(&I.gsh), np * 4},               {reinterpret_cast<void**>(&I.galpha), np * 4},
      {reinterpret_cast<void**>(&I.zero2), 256},                {reinterpret_cast<void**>(&I.yc), np * z(e) * 4},
      {reinterpret_cast<void**>(&I.cat), o.mtp ? np * z(2 * e) * 4 : 256},
      {reinterpret_cast<void**>(&I.r3y), 4 * (z(s.n_vocab) + z(qh * nh) + z(4 * I.kv_w) + z(2 * ch) + z(2 * di) + z(4 * ffs) + 1024) * 4},
      {reinterpret_cast<void**>(&I.lg), z(Impl::kOutRows) * z(s.n_vocab) * 4}};
  std::size_t total = 0;
  for (const Buf& b : bufs) total += al(b.bytes);
  I.r3y_floats = 4 * (z(s.n_vocab) + z(qh * nh) + z(4 * I.kv_w) + z(2 * ch) + z(2 * di) + z(4 * ffs) + 1024);  // v26: up to 4 tokens
  I.act_mem = g::dalloc(total);
  if (!I.act_mem || !g::dzero(I.act_mem, total)) return R::fail(Status::fail(Code::over_ceiling, "engine: activation memory"));
  auto* cur = static_cast<std::uint8_t*>(I.act_mem);
  for (const Buf& b : bufs) {
    *b.ptr = cur;
    cur += al(b.bytes);
  }
  I.dev_bytes += total;
  // the cache and the state
  I.L.resize(s.n_layer);
  {
    std::size_t sb = 0;
    const int64_t nck = std::max(0, o.max_verify - 1);
    for (const Qwen35MoeLayer& l : w.layers)
      sb += l.recurrent ? al(z(nv * ds * ds) * 4) + 2 * al(z((taps - 1) * ch) * 4) + (nck > 0 ? al(z(nck * nv * ds * ds) * 4) + al(z(o.max_verify) * z(ch) * 4) : 0)
                        : 2 * al(z(o.max_ctx) * z(I.kv_w) * 4);
    if (o.mtp) sb += 2 * al(z(o.max_ctx) * z(I.kv_w) * 4);
    I.state_mem = g::dalloc(sb);
    if (!I.state_mem) return R::fail(Status::fail(Code::over_ceiling, "engine: cache and state (" + std::to_string(sb >> 20) + " MiB)"));
    I.dev_bytes += sb;
    I.cache_bytes = sb;
    auto* c = static_cast<std::uint8_t*>(I.state_mem);
    for (std::uint32_t il = 0; il < s.n_layer; ++il) {
      Layer& D = I.L[il];
      if (w.layers[il].recurrent) {
        D.state = reinterpret_cast<float*>(c);
        c += al(z(nv * ds * ds) * 4);
        for (float*& tl : D.tail) {
          tl = reinterpret_cast<float*>(c);
          c += al(z((taps - 1) * ch) * 4);
        }
        if (nck > 0) {
          D.ckpt = reinterpret_cast<float*>(c);
          c += al(z(nck * nv * ds * ds) * 4);
          D.mixed_save = reinterpret_cast<float*>(c);
          c += al(z(o.max_verify) * z(ch) * 4);
        }
      } else {
        D.K = reinterpret_cast<float*>(c);
        c += al(z(o.max_ctx) * z(I.kv_w) * 4);
        D.V = reinterpret_cast<float*>(c);
        c += al(z(o.max_ctx) * z(I.kv_w) * 4);
      }
    }
    if (o.mtp) {
      I.M.K = reinterpret_cast<float*>(c);
      c += al(z(o.max_ctx) * z(I.kv_w) * 4);
      I.M.V = reinterpret_cast<float*>(c);
      c += al(z(o.max_ctx) * z(I.kv_w) * 4);
    }
  }
  // the experts' scratch (the bank's stacks all have an int8 step: no exact buffers)
  if (!g::moe_scratch_alloc(o.max_batch, static_cast<int>(nu), static_cast<int>(ne), e, ff, &I.moe, false))
    return R::fail(Status::fail(Code::over_ceiling, std::string("engine: expert scratch: ") + g::prompt_last_error()));
  // the dense weights of every layer and the output layer, resident
  {
    std::size_t db = I.dense_bytes(w.output) + Impl::vec_bytes(w.output_norm) + Impl::vec_bytes(w.rope_freqs) + 1024;
    for (const Qwen35MoeLayer& l : w.layers) db += I.layer_dense_bytes(l);
    const bool head_is_out = w.mtp.head.data == w.output.data;
    if (o.mtp)
      db += I.layer_dense_bytes(w.mtp.layer) + I.dense_bytes(w.mtp.eh_proj) + Impl::vec_bytes(w.mtp.enorm) + Impl::vec_bytes(w.mtp.hnorm) +
            Impl::vec_bytes(w.mtp.head_norm) + (head_is_out ? 0 : I.dense_bytes(w.mtp.head));
    I.dense_mem = g::dalloc(db);
    if (!I.dense_mem) return R::fail(Status::fail(Code::over_ceiling, "engine: dense weights (" + std::to_string(db >> 20) + " MiB)"));
    I.dev_bytes += db;
    I.dense_bytes_total = db;
    I.arena = Arena{static_cast<std::uint8_t*>(I.dense_mem), db, 0};
    Status st = I.put_dense(w.output, I.out);
    if (st) st = I.put_vec(w.output_norm, I.out_norm);
    if (st) st = I.put_vec(w.rope_freqs, I.rope_ff);
    for (std::uint32_t il = 0; st && il < s.n_layer; ++il) st = I.put_layer(w.layers[il], I.L[il]);
    if (st && o.mtp) {
      st = I.put_layer(w.mtp.layer, I.M);
      if (st) st = I.put_dense(w.mtp.eh_proj, I.eh);
      if (st) st = I.put_vec(w.mtp.enorm, I.enorm);
      if (st) st = I.put_vec(w.mtp.hnorm, I.hnorm);
      if (st) st = I.put_vec(w.mtp.head_norm, I.head_norm);
      if (st) {
        if (head_is_out) I.head = I.out;
        else st = I.put_dense(w.mtp.head, I.head);
      }
      I.M.hot.assign(z(ne), 0);  // the MTP block's experts run on the CPU path in this first cut
    }
    if (!st) return R::fail(st);
  }
  // the bank: what is left of the budget, filled rank by rank across the layers
  for (Layer& D : I.L) D.hot.assign(z(ne), 0);
  std::size_t free_b = 0, total_b = 0;
  if (!g::sync() || !g::mem_info(&free_b, &total_b)) return R::fail(Status::fail(Code::internal, std::string("engine: device memory: ") + g::last_error()));
  std::uint64_t left = free_b > o.vram_reserve ? free_b - o.vram_reserve : 0;
  if (o.vram_budget > 0) left = std::min<std::uint64_t>(left, o.vram_budget > I.dev_bytes ? o.vram_budget - I.dev_bytes : 0);
  if (o.mtp && o.mtp_bank != 0) {  // the MTP block's fixed arena, for promotion
    const Qwen35MoeLayer& ml = w.mtp.layer;
    std::uint64_t eb = 0;
    int bankable_n = 0;
    for (int ex = 0; ex < static_cast<int>(ne); ++ex)
      if (I.bankable(ml, ex)) {
        eb = std::max<std::uint64_t>(eb, I.expert_bytes(ml, ex));
        ++bankable_n;
      }
    int k = o.mtp_bank;
    if (k < 0 && eb > 0) k = static_cast<int>(std::min<std::uint64_t>(static_cast<std::uint64_t>(ne), left / (eb * (s.n_layer + 1))));
    k = std::min(k, bankable_n);
    if (k > 0 && eb > 0) {
      // matrices of k experts plus their side tables (per stack, rounded) and up to 3 x 4 stacks' maps and alignment
      const std::size_t cap = static_cast<std::size_t>(eb) * static_cast<std::size_t>(k) + 12 * (al(z(ne) * sizeof(int)) + 1024) + 3 * 4 * 4096;
      if (cap < left) {
        I.M.bank_mem = g::dalloc(cap);
        if (I.M.bank_mem) {
          I.M.bank_cap = cap;
          I.M.bank_slots = k;
          I.dev_bytes += cap;
          I.bank_bytes_total += cap;
          left -= cap;
        }
      }
    }
  }
  if (o.mtp) I.M.picks.assign(z(ne), 0);
  if (o.cold_on_gpu && o.reader_v3 && o.reader_cache > 0) {  // v26: the reader cache's bytes before the bank takes its share
    std::uint64_t mx = 0;
    for (std::uint32_t il = 0; il < s.n_layer; ++il)
      for (int ex = 0; ex < static_cast<int>(ne); ++ex) {
        std::uint64_t b = 0;
        for (const WeightView* v : {&w.layers[il].gate_exps, &w.layers[il].up_exps, &w.layers[il].down_exps}) b += exp_rows(*v, ex).rb * v->ne[1];
        mx = std::max(mx, b);
      }
    const std::uint64_t slot_b = (mx + 255) & ~std::uint64_t{255};
    const std::uint64_t need = slot_b * static_cast<std::uint64_t>(o.reader_cache) * s.n_layer;
    if (need < left) {
      left -= need;
      I.cache_reserved = need;
      I.cslot = slot_b;
    }
  }
  if (!o.hot_order.empty()) {
    // stack overhead per layer (alignment, maps, table rounding): ESTIMATE, kept out of the budget
    const std::uint64_t per_layer_slack = 3 * 4 * (2048 + 256 * 4) + 64 * 1024;
    left = left > per_layer_slack * s.n_layer ? left - per_layer_slack * s.n_layer : 0;
    std::size_t max_rank = 0;
    for (const auto& v : o.hot_order) max_rank = std::max(max_rank, v.size());
    const std::size_t cap = std::min<std::size_t>(max_rank, static_cast<std::size_t>(std::max(0, o.max_hot_per_layer)));
    for (std::size_t r = 0; r < cap && left > 0; ++r)
      for (std::uint32_t il = 0; il < s.n_layer; ++il) {
        if (r >= o.hot_order[il].size()) continue;
        const int ex = o.hot_order[il][r];
        if (ex < 0 || ex >= ne || I.L[il].hot[z(ex)] || !I.bankable(w.layers[il], ex)) continue;
        const std::uint64_t b = I.expert_bytes(w.layers[il], ex);
        if (b > left) continue;
        left -= b;
        I.L[il].hot[z(ex)] = 1;
        ++I.L[il].n_hot;
        ++I.bank_count;
      }
    for (std::uint32_t il = 0; il < s.n_layer; ++il) {
      const Status st = I.build_bank(w.layers[il], I.L[il]);
      if (!st) return R::fail(st);
    }
  }
  if (o.cold_on_gpu) I.setup_reader();  // v24
  // the host side of the cold path
  const int threads = o.cpu_threads > 0 ? o.cpu_threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  I.pool = std::make_unique<Pool>(threads);
  const int64_t mb = o.max_batch, maxp = mb * nu;
  I.xn_h.assign(z(mb * e), 0.0f);
  I.ycold_h.assign(z(mb * e), 0.0f);
  I.sel_h.assign(z(maxp), 0);
  I.wsel_h.assign(z(maxp), 0.0f);
  I.actx.resize(z(mb));
  I.acth.resize(z(maxp));
  I.gbuf.assign(z(maxp * ff), 0.0f);
  I.ubuf.assign(z(maxp * ff), 0.0f);
  I.dbuf.assign(z(maxp * e), 0.0f);
  I.cold.reserve(z(maxp));
  I.grp.reserve(z(maxp));
  I.ord.reserve(z(maxp));
  I.grp_left = std::make_unique<std::atomic<int>[]>(z(maxp));
  I.need.assign(z(mb), 0);
  if (o.doorbell) {
    const std::size_t sel_b = al(z(maxp) * 4), xn_b = al(z(mb * e) * 4);
    const std::size_t bell_b = 512 + 2 * sel_b + 2 * xn_b;
    void* dev = nullptr;
    I.bell_h = g::host_alloc_mapped(bell_b, &dev);
    if (I.bell_h) {
      auto* hb = static_cast<std::uint8_t*>(I.bell_h);
      auto* db2 = static_cast<std::uint8_t*>(dev);
      I.fa_h = reinterpret_cast<unsigned*>(hb);
      I.fa_d = reinterpret_cast<unsigned*>(db2);
      I.fb_h = reinterpret_cast<unsigned*>(hb + 256);
      I.fb_d = reinterpret_cast<unsigned*>(db2 + 256);
      I.sel_m = reinterpret_cast<std::int32_t*>(hb + 512);
      I.wsel_m = reinterpret_cast<float*>(hb + 512 + sel_b);
      I.xn_m = reinterpret_cast<float*>(hb + 512 + 2 * sel_b);
      I.yc_m = reinterpret_cast<float*>(hb + 512 + 2 * sel_b + xn_b);
      I.yc_md = reinterpret_cast<float*>(db2 + 512 + 2 * sel_b + xn_b);
      I.bell = true;
      if (o.timeline_ops) {
        void* odev = nullptr;
        I.ops_hv = g::host_alloc_mapped(z(s.n_layer) * z(EngineTimeline::kOps) * sizeof(unsigned long long), &odev);
        if (I.ops_hv) {
          I.ops_h = static_cast<unsigned long long*>(I.ops_hv);
          I.ops_d = static_cast<unsigned long long*>(odev);
        }
      }
      if (o.timeline || o.timeline_ops) {
        void* tdev = nullptr;
        I.ts_hv = g::host_alloc_mapped(z(s.n_layer) * 4 * sizeof(unsigned long long), &tdev);
        if (I.ts_hv) {
          I.ts_h = static_cast<unsigned long long*>(I.ts_hv);
          I.ts_d = static_cast<unsigned long long*>(tdev);
          I.tl_on = true;
        }
      }
      I.tline.threads = I.pool->threads();
      for (std::vector<double>* v : {&I.tline.gpu_dense, &I.tline.gpu_hot, &I.tline.gpu_idle, &I.tline.cpu_wait, &I.tline.cpu_cold}) v->assign(s.n_layer, 0.0);
      Impl* ip = &I;
      I.bell_th = std::thread([ip] { ip->bell_loop(); });
    }
  }
  if (I.lmask_d) {  // v28: the split runs through the doorbell's CPU side
    if (!I.bell) return R::fail(Status::fail(Code::internal, "engine: the split (cpu_light) needs the doorbell's mapped memory"));
    I.split_ok = true;
  }
  const Status rs = I.reset_state();
  if (!rs) return R::fail(rs);
  if (o.graph && I.reader) {  // v26: the engine's own stream (a graph cannot record the default stream), the device position
    I.graph_note = "off (setup failed)";
    void* pdev = nullptr;
    const std::size_t kvb = z(o.max_batch) * z(I.kv_w) * sizeof(float);
    if (g::sync() && (I.stm = g::stream_new()) != nullptr) {
      I.pos_hv = g::host_alloc_mapped(64, &pdev);
      I.pos_d = static_cast<int*>(g::dalloc(256));
      I.knew = static_cast<float*>(g::dalloc(kvb));
      I.vnew = static_cast<float*>(g::dalloc(kvb));
      if (I.pos_hv && I.pos_d && I.knew && I.vnew) {
        I.pos_hp = static_cast<int*>(I.pos_hv);
        g::set_stream(I.stm);
        I.graph_ok = true;
        I.graph_note = "on (reader steps of 1-4 tokens)";
        I.dev_bytes += 256 + 2 * kvb;
      }
    }
  } else if (o.graph) {
    I.graph_note = "off (needs the expert reader)";
  }
  if (o.prefetch && I.reader && !I.rcache.empty() && !I.split_ok) {  // v26 (not with the v28 split: light experts have no device address)
    const int64_t nuq = s.n_expert_used, np4 = Impl::kReaderMax;
    I.side = g::stream_new();
    I.xp = static_cast<float*>(g::dalloc(z(np4 * s.n_embd) * sizeof(float)));
    I.lp = static_cast<float*>(g::dalloc(z(np4 * s.n_expert) * sizeof(float)));
    I.wselp = static_cast<float*>(g::dalloc(z(np4 * nuq) * sizeof(float)));
    I.selp = static_cast<std::int32_t*>(g::dalloc(z(np4 * nuq) * sizeof(std::int32_t)));
    I.pfjob = static_cast<int*>(g::dalloc(z(1 + 2 * np4 * nuq) * sizeof(int)));
    bool ok = I.side && I.xp && I.lp && I.wselp && I.selp && I.pfjob;
    for (std::uint32_t il = 0; ok && il < s.n_layer; ++il) {
      I.ev_a.push_back(g::event_new());
      I.ev_b.push_back(g::event_new());
      ok = I.ev_a.back() && I.ev_b.back();
    }
    I.pf_issued.assign(s.n_layer, 0);
    I.pf_ok = ok;
  }
  return R::good(std::unique_ptr<GpuEngine>(new GpuEngine(std::move(p))));
}

Status GpuEngine::step(const std::int32_t* tokens, int n, float* logits, bool last_only, EngineStats* stats, float* hidden_out) {
  Impl& I = *p_;
  I.ckpt_mode = false;
  I.last_n = 0;
  return I.run(tokens, n, logits, last_only, stats, hidden_out);
}

Status GpuEngine::verify(const std::int32_t* tokens, int n, float* logits, float* hidden_out, EngineStats* stats) {
  Impl& I = *p_;
  if (n > std::max(1, I.o.max_verify)) return Status::fail(Code::invalid_argument, "engine: verify batch above max_verify");
  I.ckpt_mode = true;
  I.verify_pos0 = I.pos;
  const Status st = I.run(tokens, n, logits, false, stats, hidden_out);
  I.ckpt_mode = false;
  I.last_n = st ? n : 0;
  return st;
}

Status GpuEngine::keep(int k) {
  Impl& I = *p_;
  if (I.last_n == 0) return Status::fail(Code::invalid_argument, "engine: keep() without a verify batch");
  if (k < 1 || k > I.last_n) return Status::fail(Code::invalid_argument, "engine: keep 1 .. the verify batch's tokens");
  const int n = I.last_n;
  I.last_n = 0;
  if (k == n) return Status::good();
  const Qwen35MoeSpec& s = I.s;
  const int64_t ch = s.conv_channels(), taps = s.ssm_d_conv, ss = s.ssm_dt_rank * s.ssm_d_state * s.ssm_d_state;
  for (std::uint32_t il = 0; il < s.n_layer; ++il) {
    Layer& D = I.L[il];
    if (!D.state) continue;
    // the state after token k - 1 of the batch; the conv tail rebuilt from the tail before the batch and the kept inputs
    if (!g::dcopy(D.state, D.ckpt + static_cast<int64_t>(k - 1) * ss, z(ss) * sizeof(float))) return gfail("engine: keep state");
    if (!g::deltanet_conv(D.mixed_save, D.tail[1 - D.cur], D.conv1d, k, ch, taps, I.conv, D.tail[D.cur])) return gfail("engine: keep tail");
  }
  I.pos = I.verify_pos0 + k;
  return g::sync() ? Status::good() : gfail("engine: keep sync");
}

Status GpuEngine::Impl::run(const std::int32_t* tokens, int n, float* logits, bool last_only, EngineStats* stats, float* hidden_out) {
  Impl& I = *this;
  const Qwen35MoeWeights& wt = *I.w;
  if (n <= 0 || n > I.o.max_batch || !tokens || !logits) return Status::fail(Code::invalid_argument, "engine: 1 .. max_batch tokens and an output");
  if (I.pos + n > I.o.max_ctx) return Status::fail(Code::invalid_argument, "engine: the cache is full (max_ctx)");
  const Clock::time_point t0 = Clock::now();
  EngineStats es;
  const int64_t e = s.n_embd;
  if (Status es0 = I.embed(wt.tok_embd, tokens, n, I.x, e); !es0) return es0;
  const unsigned g0 = I.gen + 1;
  const bool rd = I.reader && n <= Impl::kReaderMax;  // v24: the reader's steps use no doorbell
  const bool bl = I.bell && !rd;
  const bool cl = rd && I.split_ok;  // v28: a reader step whose light picks the CPU computes
  I.bi_now = rd && I.o.batch_identical;  // v26 option B
  struct BiReset {
    bool& f;
    ~BiReset() { f = false; }
  } bi_reset{I.bi_now};
  if (bl || cl) {  // the CPU side starts waiting for this step's rings
    if (cl) {  // v28: the generation the reader steps' rings carry (device-resident, so a graph replay rings this step's)
      *I.gen_hp = g0;
      if (!g::h2d_async(I.gen_d, I.gen_hp, sizeof(unsigned))) return gfail("engine: split generation");
    }
    {
      std::lock_guard<std::mutex> lk(I.bmu);
      I.bn = n;
      I.bg0 = g0;
      I.bsplit = cl;
      I.bjob = true;
      I.bbusy = true;
      I.bcancel.store(false, std::memory_order_relaxed);
    }
    I.bcv.notify_all();
    I.gen += s.n_layer;
  }
  // on an early return the CPU side is told to stop waiting (it rings zeros, so nothing on the GPU waits forever)
  struct BellGuard {
    Impl& I;
    bool armed;
    ~BellGuard() {
      if (!armed) return;
      I.bcancel.store(true, std::memory_order_relaxed);
      std::unique_lock<std::mutex> lk(I.bmu);
      I.bcv.wait(lk, [&] { return !I.bbusy; });
    }
  } guard{I, bl || cl};
  Status st;
  if (I.graph_ok) {  // v26: the step's position for the device-side readers (on the engine's stream, before the layers)
    *I.pos_hp = static_cast<int>(I.pos);
    if (!g::h2d_async(I.pos_d, I.pos_hp, sizeof(int))) return gfail("engine: position");
  }
  std::fill(I.pf_issued.begin(), I.pf_issued.end(), std::uint8_t{0});  // v26: no prefetch in flight from an earlier step
  I.trunk_dp = I.graph_ok;
  struct DpReset {
    bool& f;
    ~DpReset() { f = false; }
  } dp_reset{I.trunk_dp};
  auto layers = [&]() -> Status {
  for (std::uint32_t il = 0; il < s.n_layer; ++il) {
    const Qwen35MoeLayer& l = wt.layers[il];
    Layer& D = I.L[il];
    I.cur_ops = (I.ops_d && (bl || rd) && n == 1) ? I.ops_d + z(EngineTimeline::kOps) * il : nullptr;
    if (!I.opstamp(0)) return gfail("engine: stamp");
    if (!g::rms_norm(I.x, D.attn_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: attn norm");
    if (!I.opstamp(1)) return gfail("engine: stamp");
    if (!(st = l.recurrent ? I.deltanet_block(D, n) : I.attention_block(D, n, I.pos))) return st;
    const bool probe_row = I.probe_pos >= I.pos && I.probe_pos < I.pos + n && I.probe_bd;
    if (probe_row && !g::dcopy(I.probe_bd + static_cast<int64_t>(il) * e, I.y + (I.probe_pos - I.pos) * e, z(e) * sizeof(float))) return gfail("engine: probe copy");
    if (!g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("engine: residual");
    if (!g::rms_norm(I.x, D.post_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: post norm");
    if (!I.opstamp(7)) return gfail("engine: stamp");
    if (!(st = rd ? I.moe_block_reader(D, n, il) : bl ? I.moe_block_bell(D, n, g0 + il, I.ts_d ? I.ts_d + 4 * il : nullptr) : I.moe_block(l, D, n, es))) return st;
    if (probe_row) {
      const int64_t pr = I.probe_pos - I.pos, ne2 = s.n_expert;
      if (!g::dcopy(I.probe_md + static_cast<int64_t>(il) * e, I.y + pr * e, z(e) * sizeof(float)) ||
          !g::dcopy(I.probe_ld + static_cast<int64_t>(il) * ne2, I.logits_r + pr * ne2, z(ne2) * sizeof(float)))
        return gfail("engine: probe copy");
    }
    if (!g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("engine: residual");
    if (rd && I.pf_ok && il + 1 < s.n_layer) {  // v26: predict the next layer's picks, prefetch its file experts on the side stream
      Layer& Dn = I.L[il + 1];
      const int64_t nuq = s.n_expert_used;
      if (!g::rms_norm(I.x, Dn.post_norm, I.xp, e, 1, e, e, n, s.rms_eps)) return gfail("engine: prefetch norm");
      if (!(st = I.prod(Dn.gate_inp, I.xp, n, I.lp))) return st;
      if (!g::router_topk(I.lp, s.n_expert, n, static_cast<int>(nuq), s.expert_weights_scale, I.selp, I.wselp, nullptr, I.o.router_tie) ||
          !g::event_record(I.ev_a[il], nullptr) || !g::stream_wait(I.side, I.ev_a[il]) ||
          !g::reader_prefetch(I.rtab + static_cast<int64_t>(il + 1) * s.n_expert, I.selp, static_cast<int>(n * nuq), &I.rcache[il + 1], e, s.n_ff_exp, I.pfjob,
                              I.side) ||
          !g::event_record(I.ev_b[il], I.side))
        return gfail("engine: prefetch");
      I.pf_issued[il + 1] = 1;
    }
    if (!I.opstamp(15)) return gfail("engine: stamp");
    if (I.probe_pos >= I.pos && I.probe_pos < I.pos + n && I.probe_xd) {  // v23: this row's residual and picks after the layer
      const int64_t r = I.probe_pos - I.pos, nuq = s.n_expert_used;
      if (!g::dcopy(I.probe_xd + static_cast<int64_t>(il) * e, I.x + r * e, z(e) * sizeof(float)) ||
          !g::dcopy(I.probe_seld + static_cast<int64_t>(il) * nuq, I.sel + r * nuq, z(nuq) * sizeof(std::int32_t)) ||
          !g::dcopy(I.probe_wd + static_cast<int64_t>(il) * nuq, I.wsel + r * nuq, z(nuq) * sizeof(float)))
        return gfail("engine: probe copy");
    }
  }
  return Status::good();
  };
  // v26: replay a recorded graph of this kind of step, record it the second time this kind is seen, else plain launches
  const bool probe_here = I.probe_pos >= I.pos && I.probe_pos < I.pos + n && I.probe_xd;
  if (I.graph_ok && rd && !probe_here) {
    int parity = 0;
    for (std::uint32_t il = 0; il < s.n_layer; ++il)
      if (wt.layers[il].recurrent) {
        parity = I.L[il].cur;
        break;
      }
    const std::tuple<int, int, int, int> key{n, parity, I.ckpt_mode && n > 1 ? 1 : 0, (I.ops_d && n == 1) ? 1 : 0};
    auto flip = [&]() {
      for (std::uint32_t il = 0; il < s.n_layer; ++il)
        if (wt.layers[il].recurrent) I.L[il].cur = 1 - I.L[il].cur;
    };
    auto it = I.graphs.find(key);
    if (it != I.graphs.end()) {
      if (!g::graph_launch(it->second, nullptr)) return gfail("engine: graph replay");
      flip();  // the host side of the DeltaNet tails, as the plain step would have done
      ++I.graph_replays;
    } else if (I.graph_seen[key]++ >= 1) {
      void* ex = nullptr;
      const bool began = g::capture_begin(nullptr);
      const Status cs = began ? layers() : Status::fail(Code::internal, "capture did not begin");
      const bool ended = began && g::capture_end(nullptr, &ex);
      if (cs.ok() && ended && ex) {
        I.graphs[key] = ex;
        ++I.graph_captures;
        if (!g::graph_launch(ex, nullptr)) return gfail("engine: graph launch");
      } else {  // the attempt ran the host side only: undo its tail flips, stop using graphs, run this step plainly
        if (began) flip();
        g::graph_free(ex);
        I.graph_ok = false;
        I.graph_note = std::string("off after a failed capture (") + (cs.ok() ? g::last_error() : cs.text()) + ")";
        if (!(st = layers())) return st;
      }
    } else if (!(st = layers())) {
      return st;
    }
  } else if (!(st = layers())) {
    return st;
  }
  if (I.probe_pos >= I.pos && I.probe_pos < I.pos + n && I.probe_xd) {
    const std::size_t nl = s.n_layer, nuq = s.n_expert_used;
    I.probe_xh.resize(nl * z(e));
    I.probe_selh.resize(nl * nuq);
    I.probe_wh.resize(nl * nuq);
    if (!g::to_host(I.probe_xh.data(), I.probe_xd, I.probe_xh.size() * sizeof(float)) ||
        !g::to_host(I.probe_selh.data(), I.probe_seld, I.probe_selh.size() * sizeof(std::int32_t)) ||
        !g::to_host(I.probe_wh.data(), I.probe_wd, I.probe_wh.size() * sizeof(float)))
      return gfail("engine: probe read");
    I.probe_bh.resize(nl * z(e));
    I.probe_lh.resize(nl * z(s.n_expert));
    I.probe_mh.resize(nl * z(e));
    if (!g::to_host(I.probe_bh.data(), I.probe_bd, I.probe_bh.size() * sizeof(float)) ||
        !g::to_host(I.probe_lh.data(), I.probe_ld, I.probe_lh.size() * sizeof(float)) ||
        !g::to_host(I.probe_mh.data(), I.probe_md, I.probe_mh.size() * sizeof(float)))
      return gfail("engine: probe read");
    I.probe_have = true;
    I.probe_pos = -1;
  }
  const bool ops_this = I.cur_ops != nullptr;
  I.cur_ops = nullptr;
  const int64_t nvoc = s.n_vocab;
  const int r_begin = last_only ? n - 1 : 0;
  for (int r0 = hidden_out ? 0 : r_begin; r0 < n; r0 += Impl::kOutRows) {
    const int m = std::min(Impl::kOutRows, n - r0);
    if (!g::rms_norm(I.x + static_cast<int64_t>(r0) * e, I.out_norm, I.xn, e, 1, e, e, m, s.rms_eps)) return gfail("engine: output norm");
    if (hidden_out && !g::to_host(hidden_out + static_cast<int64_t>(r0) * e, I.xn, z(m) * z(e) * sizeof(float))) return gfail("engine: hidden");
    if (r0 + m <= r_begin) continue;
    // rows before r_begin (asked for the hidden only) are run through the head with the rest of their chunk; only rows
    // from r_begin on are copied out
    if (!(st = I.prod(I.out, I.xn, m, I.lg))) return st;
    const int c0 = std::max(r0, r_begin);
    if (!g::to_host(logits + static_cast<int64_t>(c0 - r_begin) * nvoc, I.lg + static_cast<int64_t>(c0 - r0) * nvoc, z(r0 + m - c0) * z(nvoc) * sizeof(float)))
      return gfail("engine: logits");
  }
  if (!g::sync()) return gfail("engine: sync");
  if (rd) {  // v25: every pick of a reader step runs on the GPU
    es.hot_picks += static_cast<std::uint64_t>(n) * s.n_expert_used;
    if (I.trc_ts_h) I.trace_step(n, since(t0));
  }
  if (rd && I.tl_on) {  // v24: the reader's steps: wall time per step size, and the per-operation segments
    const int b = n > 8 ? 9 : n;
    I.tline.by_n_steps[b] += 1.0;
    I.tline.by_n_wall[b] += since(t0);
    ++I.tline.steps;
    if (ops_this && I.ops_h) {
      auto d = [](unsigned long long a, unsigned long long b2) { return static_cast<double>(static_cast<std::int64_t>(b2 - a)) * 1e-9; };
      for (std::uint32_t il = 0; il < s.n_layer; ++il) {
        const unsigned long long* orow = I.ops_h + z(EngineTimeline::kOps) * il;
        const int ty = wt.layers[il].recurrent ? 0 : 1;
        for (int k = 1; k < EngineTimeline::kOps; ++k) I.tline.ops[ty][k] += d(orow[k - 1], orow[k]);
      }
      ++I.tline.ops_steps;
    }
  }
  if (bl || cl) {
    guard.armed = false;
    std::unique_lock<std::mutex> lk(I.bmu);
    I.bcv.wait(lk, [&] { return !I.bbusy; });
    if (!I.berr.empty()) return Status::fail(Code::internal, I.berr);
    es.wait_seconds += I.bst.wait_seconds;
    es.cold_seconds += I.bst.cold_seconds;
    es.hot_picks += I.bst.hot_picks;
    es.cold_picks += I.bst.cold_picks;
    es.cold_bytes += I.bst.cold_bytes;
    if (cl) {  // v28b: per layer on both sides (the reader step had counted n x nu once for the whole step)
      const std::uint64_t all = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(s.n_expert_used) * s.n_layer;
      es.hot_picks = all - std::min<std::uint64_t>(all, I.bst.cold_picks);
    }
    if (bl && I.tl_on && I.ts_h) {  // the GPU's timestamps of this step (synced above), per layer
      auto d = [](unsigned long long a, unsigned long long b) { return static_cast<double>(static_cast<std::int64_t>(b - a)) * 1e-9; };
      for (std::uint32_t il = 0; il < s.n_layer; ++il) {
        const unsigned long long* q4 = I.ts_h + 4 * il;
        if (il > 0) I.tline.gpu_dense[il] += d(I.ts_h[4 * (il - 1) + 2], q4[0]);
        I.tline.gpu_hot[il] += d(q4[0], q4[1]);
        I.tline.gpu_idle[il] += d(q4[1], q4[2]);
      }
      I.tline.hot_picks += I.bst.hot_picks;
      I.tline.cold_picks += I.bst.cold_picks;
      ++I.tline.steps;
      double busy = 0.0, idle = 0.0;
      for (std::uint32_t il = 0; il < s.n_layer; ++il) {
        const unsigned long long* q4 = I.ts_h + 4 * il;
        if (il > 0) busy += d(I.ts_h[4 * (il - 1) + 2], q4[0]);
        busy += d(q4[0], q4[1]);
        idle += d(q4[1], q4[2]);
      }
      const int b = n > 8 ? 9 : n;
      I.tline.by_n_steps[b] += 1.0;
      I.tline.by_n_wall[b] += since(t0);
      I.tline.by_n_gpu_busy[b] += busy;
      I.tline.by_n_gpu_idle[b] += idle;
      I.tline.by_n_cpu_cold[b] += I.bst.cold_seconds;
      I.tline.by_n_cpu_wait[b] += I.bst.wait_seconds;
      if (ops_this && I.ops_h) {  // v21: the per-operation segments of this one-token step
        for (std::uint32_t il = 0; il < s.n_layer; ++il) {
          const unsigned long long* orow = I.ops_h + z(EngineTimeline::kOps) * il;
          const int ty = wt.layers[il].recurrent ? 0 : 1;
          for (int k = 1; k < EngineTimeline::kOps; ++k) I.tline.ops[ty][k] += d(orow[k - 1], orow[k]);
        }
        ++I.tline.ops_steps;
      }
    }
  }
  I.pos += n;
  es.seconds = since(t0);
  if (stats) *stats = es;
  return Status::good();
}

Status GpuEngine::mtp_step(const std::int32_t* tokens, const float* h_prev, int n, float* logits, float* h_out, EngineStats* stats) {
  Impl& I = *p_;
  const Qwen35MoeSpec& s = I.s;
  const Qwen35MoeWeights& w = *I.w;
  if (!I.has_mtp) return Status::fail(Code::unsupported, "engine: opened without the MTP block (EngineOptions::mtp)");
  if (n <= 0 || n > I.o.max_batch || !tokens || !h_prev) return Status::fail(Code::invalid_argument, "engine: MTP 1 .. max_batch tokens and their hidden rows");
  if (I.mtp_pos + n > I.o.max_ctx) return Status::fail(Code::invalid_argument, "engine: the MTP cache is full (max_ctx)");
  const Clock::time_point t0 = Clock::now();
  EngineStats es;
  const int64_t e = s.n_embd;
  // [enorm(embedding) | hnorm(h_prev)] -> eh_proj
  I.cat_h.resize(z(n) * z(2 * e));
  for (int t = 0; t < n; ++t) {
    const std::int32_t tok = tokens[t];
    if (tok < 0 || static_cast<std::uint64_t>(tok) >= s.n_vocab) return Status::fail(Code::invalid_argument, "a token id is outside the vocabulary");
    const WeightView& et = w.mtp.embed_tokens;
    if (!quant::dequantize_row(static_cast<quant::Type>(et.type), et.data + static_cast<std::uint64_t>(tok) * et.row_bytes, I.cat_h.data() + z(t) * z(2 * e), e).ok())
      return Status::fail(Code::unsupported, "engine: the MTP embedding type cannot be read on the host");
    std::copy_n(h_prev + static_cast<int64_t>(t) * e, e, I.cat_h.data() + z(t) * z(2 * e) + z(e));
  }
  if (!g::to_device(I.cat, I.cat_h.data(), I.cat_h.size() * sizeof(float))) return gfail("engine: MTP input upload");
  if (!g::rms_norm(I.cat, I.enorm, I.cat, e, 1, e, 2 * e, n, s.rms_eps) || !g::rms_norm(I.cat + e, I.hnorm, I.cat + e, e, 1, e, 2 * e, n, s.rms_eps))
    return gfail("engine: MTP norms");
  Status st;
  if (!(st = I.prod(I.eh, I.cat, n, I.x))) return st;
  Layer& D = I.M;
  if (!g::rms_norm(I.x, D.attn_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: MTP attn norm");
  if (!(st = I.attention_block(D, n, I.mtp_pos))) return st;
  if (!g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("engine: MTP residual");
  if (!g::rms_norm(I.x, D.post_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: MTP post norm");
  if (!(st = I.moe_block(w.mtp.layer, D, n, es))) return st;
  if (!g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("engine: MTP residual");
  const int64_t nvoc = s.n_vocab;
  for (int r0 = 0; r0 < n; r0 += Impl::kOutRows) {
    const int m = std::min(Impl::kOutRows, n - r0);
    if (!g::rms_norm(I.x + static_cast<int64_t>(r0) * e, I.head_norm, I.xn, e, 1, e, e, m, s.rms_eps)) return gfail("engine: MTP head norm");
    if (h_out && !g::to_host(h_out + static_cast<int64_t>(r0) * e, I.xn, z(m) * z(e) * sizeof(float))) return gfail("engine: MTP hidden");
    if (logits) {
      if (!(st = I.prod(I.head, I.xn, m, I.lg))) return st;
      if (!g::to_host(logits + static_cast<int64_t>(r0) * nvoc, I.lg, z(m) * z(nvoc) * sizeof(float))) return gfail("engine: MTP logits");
    }
  }
  if (!g::sync()) return gfail("engine: MTP sync");
  I.mtp_pos += n;
  es.seconds = since(t0);
  if (stats) *stats = es;
  return Status::good();
}

Result<int> GpuEngine::mtp_promote() {
  using R = Result<int>;
  Impl& I = *p_;
  Layer& M = I.M;
  if (!I.has_mtp || M.bank_cap == 0 || M.bank_slots <= 0) return R::good(0);
  const Qwen35MoeLayer& ml = I.w->mtp.layer;
  const int ne = static_cast<int>(I.s.n_expert);
  std::vector<int> order;
  for (int e = 0; e < ne; ++e)
    if (M.picks[static_cast<std::size_t>(e)] > 0 && I.bankable(ml, e)) order.push_back(e);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return M.picks[static_cast<std::size_t>(a)] > M.picks[static_cast<std::size_t>(b)]; });
  std::vector<std::uint8_t> want(static_cast<std::size_t>(ne), 0);
  std::uint64_t used = 0;
  int count = 0;
  for (const int e : order) {
    if (count >= M.bank_slots) break;
    const std::uint64_t b = I.expert_bytes(ml, e);
    if (used + b > M.bank_cap - 12 * (al(static_cast<std::size_t>(ne) * sizeof(int)) + 1024) - 3 * 4 * 4096) continue;
    used += b;
    want[static_cast<std::size_t>(e)] = 1;
    ++count;
  }
  int swapped = 0;
  for (int e = 0; e < ne; ++e) swapped += want[static_cast<std::size_t>(e)] && !M.hot[static_cast<std::size_t>(e)] ? 1 : 0;
  if (want == M.hot) return R::good(0);
  M.hot = want;
  M.n_hot = count;
  if (Status st = I.build_bank(ml, M); !st) return R::fail(st);
  if (!g::sync()) return R::fail(gfail("engine: MTP bank sync"));
  return R::good(swapped);
}

void GpuEngine::mtp_truncate(std::int64_t p) {
  if (p >= 0 && p < p_->mtp_pos) p_->mtp_pos = p;
}
std::int64_t GpuEngine::mtp_pos() const { return p_ ? p_->mtp_pos : 0; }

namespace {
std::int32_t argmax_of(const float* v, std::int64_t n) {
  std::int64_t b = 0;
  for (std::int64_t i = 1; i < n; ++i)
    if (v[i] > v[b]) b = i;
  return static_cast<std::int32_t>(b);
}
void add_stats(EngineStats& a, const EngineStats& b) {
  a.seconds += b.seconds;
  a.cold_seconds += b.cold_seconds;
  a.wait_seconds += b.wait_seconds;
  a.hot_picks += b.hot_picks;
  a.cold_picks += b.cold_picks;
}
}  // namespace

Status engine_generate_greedy_mtp(GpuEngine& eng, const Qwen35MoeSpec& s, const std::vector<std::int32_t>& prompt, int n_predict, int n_draft,
                                  const std::vector<std::int32_t>& stops, std::vector<std::int32_t>& out, MtpStats* stats, EngineStats* es,
                                  const std::function<void(std::int32_t)>& on_token, EngineMtpTiming* tm) {
  EngineMtpTiming tm_local;
  EngineMtpTiming& T = tm ? *tm : tm_local;
  auto secs = [](Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); };
  const Clock::time_point tp0 = Clock::now();
  out.clear();
  const auto is_stop = [&stops](std::int32_t t) { return std::find(stops.begin(), stops.end(), t) != stops.end(); };
  if (prompt.empty()) return Status::fail(Code::invalid_argument, "an empty prompt");
  if (n_predict <= 0) return Status::good();
  if (n_draft < 0) return Status::fail(Code::invalid_argument, "the draft length is 0 or more");
  const int64_t e = s.n_embd, nv = s.n_vocab, P = static_cast<int64_t>(prompt.size());
  MtpStats local;
  MtpStats& stt = stats ? *stats : local;
  EngineStats tot;
  if (Status r = eng.reset(); !r) return r;
  // the prompt in chunks: the trunk's last logits and every position's hidden
  const int kChunk = eng.max_batch();
  std::vector<float> lg(z(nv)), hid(z(P * e));
  for (int64_t t0 = 0; t0 < P; t0 += kChunk) {
    const int k = static_cast<int>(std::min<int64_t>(kChunk, P - t0));
    EngineStats x;
    if (Status r = eng.step(prompt.data() + t0, k, lg.data(), true, &x, hid.data() + t0 * e); !r) return r;
    add_stats(tot, x);
  }
  std::vector<float> pending(hid.end() - e, hid.end());
  if (n_draft > 0) {  // the MTP block catches up on the prompt: token p with h of p - 1 (zeros before the first)
    std::vector<float> hp(z(P * e), 0.0f);
    std::copy(hid.begin(), hid.end() - e, hp.begin() + e);
    for (int64_t t0 = 0; t0 < P; t0 += kChunk) {
      const int k = static_cast<int>(std::min<int64_t>(kChunk, P - t0));
      EngineStats x;
      if (Status r = eng.mtp_step(prompt.data() + t0, hp.data() + t0 * e, k, nullptr, nullptr, &x); !r) return r;
      add_stats(tot, x);
    }
    if (auto pr = eng.mtp_promote(); !pr.ok()) return pr.status;  // the MTP bank from the prompt's picks
  }
  T.prompt_s += secs(tp0);
  std::int32_t last = argmax_of(lg.data(), nv);
  out.push_back(last);
  if (on_token) on_token(last);
  std::vector<float> dl(z(nv)), hn(z(e)), h(z(e)), bl, bh;
  while (static_cast<int>(out.size()) < n_predict && !is_stop(last)) {
    ++stt.steps;
    if (n_draft > 0 && stt.steps % 32 == 0) {  // idle-time promotion: the MTP bank follows the block's own picks
      const Clock::time_point tq = Clock::now();
      if (auto pr = eng.mtp_promote(); !pr.ok()) return pr.status;
      T.promote_s += secs(tq);
    }
    const int room = n_predict - static_cast<int>(out.size()) - 1;
    const int k = std::max(0, std::min(n_draft, room));
    std::vector<std::int32_t> drafts;
    const std::int64_t m0 = eng.mtp_pos();
    if (k > 0) {
      h = pending;
      std::int32_t tok = last;
      for (int i = 0; i < k; ++i) {
        EngineStats x;
        const Clock::time_point tq = Clock::now();
        if (Status r = eng.mtp_step(&tok, h.data(), 1, dl.data(), hn.data(), &x); !r) return r;
        T.draft_s += secs(tq);
        ++T.drafts;
        add_stats(tot, x);
        const Clock::time_point ta = Clock::now();
        tok = argmax_of(dl.data(), nv);
        T.argmax_s += secs(ta);
        drafts.push_back(tok);
        h = hn;
      }
      eng.mtp_truncate(m0);
      stt.drafted += k;
    }
    const int nb = 1 + k;
    std::vector<std::int32_t> batch{last};
    batch.insert(batch.end(), drafts.begin(), drafts.end());
    bl.resize(z(nb * nv));
    bh.resize(z(nb * e));
    EngineStats x;
    const Clock::time_point tv = Clock::now();
    if (Status r = k > 0 ? eng.verify(batch.data(), nb, bl.data(), bh.data(), &x) : eng.step(batch.data(), 1, bl.data(), true, &x, bh.data()); !r) return r;
    T.verify_s[nb > 8 ? 9 : nb] += secs(tv);
    ++T.verify_n[nb > 8 ? 9 : nb];
    add_stats(tot, x);
    const Clock::time_point ta2 = Clock::now();
    int a = 0;
    while (a < k && argmax_of(bl.data() + a * nv, nv) == drafts[z(a)] && !is_stop(drafts[z(a)])) ++a;
    if (a < k && is_stop(drafts[z(a)]) && argmax_of(bl.data() + a * nv, nv) == drafts[z(a)]) ++a;
    const std::int32_t next = argmax_of(bl.data() + a * nv, nv);
    T.argmax_s += secs(ta2);
    const Clock::time_point tk = Clock::now();
    if (k > 0)
      if (Status r = eng.keep(a + 1); !r) return r;
    T.keep_s += secs(tk);
    stt.accepted += a;
    if (n_draft > 0) {
      std::vector<float> hp(z((a + 1) * e));
      std::copy(pending.begin(), pending.end(), hp.begin());
      std::copy(bh.begin(), bh.begin() + a * e, hp.begin() + e);
      EngineStats y;
      const Clock::time_point tc = Clock::now();
      if (Status r = eng.mtp_step(batch.data(), hp.data(), a + 1, nullptr, nullptr, &y); !r) return r;
      T.catchup_s += secs(tc);
      add_stats(tot, y);
    }
    pending.assign(bh.begin() + a * e, bh.begin() + (a + 1) * e);
    for (int i = 0; i < a && static_cast<int>(out.size()) < n_predict; ++i) {
      out.push_back(drafts[z(i)]);
      if (on_token) on_token(drafts[z(i)]);
    }
    if (!out.empty() && is_stop(out.back())) break;
    if (static_cast<int>(out.size()) < n_predict) {
      out.push_back(next);
      if (on_token) on_token(next);
    }
    last = next;
  }
  if (es) *es = tot;
  return Status::good();
}

Result<std::vector<std::vector<int>>> hot_order_from_routes(const std::string& path, std::uint32_t n_layer, std::uint32_t n_expert) {
  using R = Result<std::vector<std::vector<int>>>;
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return R::fail(Status::fail(Code::not_found, "routing record not found: " + path));
  char magic[8];
  std::uint32_t hdr[5];
  std::uint64_t hash = 0;
  bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "PELROUT1", 8) == 0 && std::fread(hdr, 4, 5, f) == 5 && std::fread(&hash, 8, 1, f) == 1;
  if (!ok || hdr[3] != n_layer || hdr[4] > 8) {
    std::fclose(f);
    return R::fail(Status::fail(Code::corrupt, "not a routing record for this model (PELROUT1, " + std::to_string(n_layer) + " layers): " + path));
  }
  const std::uint64_t records = static_cast<std::uint64_t>(hdr[1]) * hdr[2] * hdr[3];
  std::vector<std::vector<std::uint64_t>> cnt(n_layer, std::vector<std::uint64_t>(n_expert, 0));
  std::uint8_t rec[72];
  for (std::uint64_t i = 0; i < records && ok; ++i) {
    ok = std::fread(rec, 1, sizeof rec, f) == sizeof rec;
    if (!ok) break;
    const std::uint32_t il = static_cast<std::uint32_t>(i % n_layer);
    for (int j = 0; j < 8; ++j) {
      std::int16_t ex;
      std::memcpy(&ex, rec + 2 * j, 2);
      if (ex >= 0 && static_cast<std::uint32_t>(ex) < n_expert) ++cnt[il][static_cast<std::size_t>(ex)];
    }
  }
  std::fclose(f);
  if (!ok) return R::fail(Status::fail(Code::corrupt, "routing record cut short: " + path));
  std::vector<std::vector<int>> order(n_layer);
  for (std::uint32_t il = 0; il < n_layer; ++il) {
    std::vector<int>& v = order[il];
    v.resize(n_expert);
    for (std::uint32_t e = 0; e < n_expert; ++e) v[e] = static_cast<int>(e);
    std::stable_sort(v.begin(), v.end(), [&](int a, int b) { return cnt[il][static_cast<std::size_t>(a)] > cnt[il][static_cast<std::size_t>(b)]; });
  }
  return R::good(std::move(order));
}

}  // namespace pelias::model
