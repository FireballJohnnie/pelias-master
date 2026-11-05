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
#include "kern/tg2_planes.h"
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
// v40: a software prefetch of one row of weights (each 64-byte line), and a small index per pool thread for the stamps
inline void relax_cpu() {
#if defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#endif
}
inline void prefetch_row(const std::uint8_t* p, std::uint64_t bytes) {
#if defined(__x86_64__) || defined(_M_X64)
  for (std::uint64_t off = 0; off < bytes; off += 64) _mm_prefetch(reinterpret_cast<const char*>(p + off), _MM_HINT_T0);
#else
  (void)p;
  (void)bytes;
#endif
}
std::atomic<int> g_tid_next{0};
inline int pool_tid() {
  thread_local const int t = std::min(31, g_tid_next.fetch_add(1, std::memory_order_relaxed));
  return t;
}
inline int pool_tid_count() { return g_tid_next.load(std::memory_order_relaxed); }

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
// v57 (Windows build check wincheck57b): MSVC warns (C4324, an error under warnings-as-errors) that the cache-line-aligned
// counters below pad the struct, which is their purpose
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
class Pool {
 public:
  // v43: cpus[i] (when given) binds worker i to that logical CPU before it starts
  explicit Pool(int threads, const std::vector<int>& cpus = {}) {
    const int workers = std::max(0, threads - 1);
    for (int i = 0; i < workers; ++i) {
      const int cpu = i < static_cast<int>(cpus.size()) ? cpus[static_cast<std::size_t>(i)] : -1;
      th_.emplace_back([this, cpu] {
        if (cpu >= 0) pinned_.fetch_add(hal::pin_this_thread(cpu) ? 1 : 0, std::memory_order_relaxed);
        loop();
      });
    }
  }
  [[nodiscard]] int pinned() const { return pinned_.load(std::memory_order_relaxed); }
  // v45b (author 17:21, --spin-pause): a pause in every spin (the hot waits included); off = v45's spins (A/B for Hyper-V's
  // pause-loop exits)
  void set_pause(bool p) { pause_.store(p, std::memory_order_relaxed); }
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
    const bool pz = pause_.load(std::memory_order_relaxed);
    for (int spin = 0; active_.load(std::memory_order_acquire) != 0; ++spin) {
      if (pz) relax();
      if (spin > kSpin && !hot_.load(std::memory_order_relaxed)) std::this_thread::yield();
    }
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
    const bool pz = pause_.load(std::memory_order_relaxed);
    for (int spin = 0; active_.load(std::memory_order_acquire) != 0; ++spin) {
      if (pz) relax();
      if (spin > kSpin) std::this_thread::yield();
    }
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
        if (spin >= kSpin || pause_.load(std::memory_order_relaxed)) relax();
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
  // v45b (author 17:21): the four shared counters each on their own 64-byte line (spinning readers of one no longer pull
  // the line a working thread is adding to)
  alignas(64) std::atomic<int64_t> done1_{0};
  alignas(64) std::atomic<bool> hot_{false};
  std::atomic<bool> pause_{false};
  RunStats* rs_ = nullptr;  // written before the job's generation is released; read by the workers after they see it
  Clock::time_point t0_{};
  alignas(64) std::atomic<int64_t> next_{0};
  alignas(64) std::atomic<int> active_{0};
  alignas(64) std::atomic<std::uint64_t> gen_{0};
  alignas(64) std::atomic<int> pinned_{0};
  std::atomic<bool> stop_{false};
};
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

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
  bool lean = false;         // v55: only R3's copy is held (no raw copy, no int8 tables); every product of it must run on R3
};

struct Layer {
  DevW wq, wk, wv, wo, wqkv, wgate, wbeta, walpha, ssm_out, gate_inp, gate_sh, up_sh, down_sh;
  const float *attn_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *conv1d = nullptr, *ssm_dt = nullptr,
              *ssm_a = nullptr, *ssm_norm = nullptr, *gate_inp_sh = nullptr;
  // v58b: an F32 copy of the router for the fused launch when the file stores it otherwise (the MTP block's; 0: gate_inp
  // is F32 itself)
  const float* router_f32 = nullptr;
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
  std::int32_t* top_d = nullptr;    // v56: device argmax per output row (max_batch)
  std::int32_t* top_req = nullptr;  // v56: non-null during step_top / verify_top / mtp_step_top: the host destination
  unsigned* dn_hcnt = nullptr;  // v55g (L3): per value head, the dense half's units that finished it (zero between launches)
  std::uint64_t dn_half_runs = 0;
  float *logits_r = nullptr, *wsel = nullptr, *gs = nullptr, *us = nullptr, *sh = nullptr, *gsh = nullptr, *galpha = nullptr, *zero2 = nullptr;
  float* yc = nullptr;  // the cold sum, copied up
  std::int32_t* sel = nullptr;
  float* lg = nullptr;
  // v24 picked-expert reader
  static constexpr int kReaderMax = 4;   // steps of up to this many tokens use it
  bool reader = false;
  std::string reader_note = "off";
  std::vector<void*> reg_bases;          // v27: the registered ranges of the model file's mapping (host)
  void* planes_h = nullptr;              // v54: the RAM experts' Q4_TG2 matrices in bit planes (pinned, mapped)
  std::uint64_t planes_b = 0;
  std::string planes_note;
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
  int rc_eff = 0;      // v54b: the reader cache's slots per layer (--reader-cache, or the VRAM probe's)
  std::string probe_note;
  void* cache_mem = nullptr;
  void* cache_meta = nullptr;
  std::vector<g::ReaderCache> rcache;
  // v46a-1 (THE PLAN 46a): the R3 expert stage instead of the reader v3 kernels (EngineOptions::r3_experts): the int8
  // silu x up per pick, its scales and sums, the plan buffer; on only if every expert matrix is one the stage reads
  std::int8_t* rhq = nullptr;
  std::uint32_t *rxb = nullptr, *rhb = nullptr;  // v55f: the expert inputs' and the down inputs' bit planes (the 1-bit route)
  float* rhs = nullptr;
  int* rhsum = nullptr;
  void* rxplan = nullptr;
  int rx_maxp = 0;
  bool rx_ok = false;
  unsigned long long *ph_h = nullptr, *ph_d = nullptr;  // v55b: n_layer x 5 phase stamps of the fused MoE half (verbose_gpu)
  unsigned long long *dph_h = nullptr, *dph_d = nullptr;  // v56: n_layer x 5 stage stamps of the DeltaNet dense half (L3; verbose_gpu)
  double dph_sum[4] = {0, 0, 0, 0};
  std::uint64_t dph_steps = 0, dph_layers = 0;
  // v56: the same stamps of verify steps (2-4 tokens; MTP), per step size: [n - 2][phase]
  double vph_sum[3][4] = {}, vdph_sum[3][4] = {};
  std::uint64_t vph_steps[3] = {}, vdph_steps[3] = {};
  unsigned long long* rst_d = nullptr;                    // v55e3 diagnosis: the expert stage's counters (g::kRxStatN, verbose_gpu)
  double ph_sum[4] = {0, 0, 0, 0};
  std::uint64_t ph_steps = 0, ph_layers = 0;
  std::string rx_note = "off";
  g::ExpertPtr *trc_src_h = nullptr, *trc_src_d = nullptr, *trc_fill_h = nullptr, *trc_fill_d = nullptr;
  // v25 trace: pinned mapped, per layer 6 reader stamps, the picks (kReaderMax x nu) and the group count
  void* trc_hv = nullptr;
  unsigned long long *trc_ts_h = nullptr, *trc_ts_d = nullptr;
  std::int32_t *trc_sel_h = nullptr, *trc_sel_d = nullptr;
  int *trc_u_h = nullptr, *trc_u_d = nullptr;
  std::string trace_text;
  float *rlg_h = nullptr, *rlg_d = nullptr;  // v49: router logits per layer (kReaderMax x n_expert each), pinned mapped
  std::vector<float> rmarg;                  // v49: per position 2 x n_layer router margins
  std::vector<std::uint8_t> rmarg_have;
  std::uint64_t trace_seq = 0;
  static constexpr int kOutRows = 64;
  // host side of the cold path (sized once)
  std::unique_ptr<Pool> pool;
  std::vector<float> xn_h, ycold_h, wsel_h, xe_h;
  std::vector<std::int32_t> sel_h;
  std::vector<kern::ActQ8x32> actx, acth;
  std::vector<float> gbuf, ubuf, dbuf;
  std::atomic<std::uint64_t> down2_n{0}, down2_bad{0};  // v40 --cpu-check
  std::atomic<std::uint64_t> gu2_n{0}, gu2_bad{0}, round_n{0}, round_bad{0};  // v45 --cpu-check
  std::atomic<std::uint64_t> gx_n{0}, gx_bad{0};  // v45b --cpu-check: GPU rounded input vs the CPU's
  void* xr_h = nullptr;  // v45b: pinned mapped copy of the GPU's rounded input (--cpu-check only)
  float* xr_m = nullptr;
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
  bool in_prompt = false;   // v48b: a step of a split prompt (its products of 2-4 tokens go through R3, as option B's)
  bool no_head = false;      // v48: a prompt step whose logits nobody reads (an inner step of a split prompt): skip the head
  int last_n = 0;            // tokens of the last verify batch (0: none pending)
  std::int64_t verify_pos0 = 0;
  // the MTP block (stage 3)
  bool has_mtp = false;
  Layer M;
  // v58: the MTP block's MoE half on the reader as its layer n_layer (EngineOptions::mtp_fused); the reader's per-layer
  // tables (expert table, shared expert, cache, planes) then have n_layer + 1 rows
  bool mtp_rd = false;
  std::uint64_t mtp_fused_steps = 0, mtp_old_steps = 0;
  std::string mtp_why;          // v58b: why the first MTP step that took the old path did
  std::string mtp_router_note;  // v58b: the MTP router's stored type, and whether an F32 copy was made
  std::uint32_t nlr() const { return s.n_layer + (mtp_rd ? 1u : 0u); }
  const Qwen35MoeLayer& WL(std::uint32_t il) const { return il < s.n_layer ? w->layers[il] : w->mtp.layer; }
  Layer& LL(std::uint32_t il) { return il < s.n_layer ? L[il] : M; }
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
  bool pin_bell_ok = false, pin_host_ok = false;  // v43
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
  const WeightView* lean_w = nullptr;  // v55: a weight held only in R3's copy (the output head when every head product runs on R3)
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
    if (&v == lean_w && r3_takes(v)) return 256;  // v55: R3's copy only (counted by put_dense from stream_upload)
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
      if (&v == lean_w) {  // v55: no raw copy and no int8 tables; prod() refuses any product of it that is not on R3
        dw.lean = true;
        return Status::good();
      }
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
  // v54c: the plane format a stored matrix takes when the slider is on (Q4_TG2 / Q5_K rows), or 0
  std::uint32_t planed_type(quant::Type t, std::uint64_t rb) const {
    if (o.plane_bits <= 0) return 0;
    if (t == quant::Type::q4_tg2 && rb % 144 == 0) return kern::gpu::g_q4_tg2p;
    if (t == quant::Type::q5_k && rb % 176 == 0) return kern::gpu::g_q5_kp;
    return 0;
  }
  // v54c: an expert's matrices at the slider's bits (the bytes a bank copy or a cache slot of it takes)
  std::size_t planed_bytes(const Qwen35MoeLayer& l, int e) const {
    std::size_t b = 0;
    for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
      const ExpRows r = exp_rows(*v, e);
      const std::uint32_t pt = planed_type(r.t, r.rb);
      b += z(static_cast<int64_t>(pt ? kern::gpu::plane_prefix(pt, r.rb, o.plane_bits) : r.rb)) * z(static_cast<int64_t>(v->ne[1]));
    }
    return b;
  }
  // v56: planed false for the MTP block's bank, which build_bank fills in the stored types with their int8 tables whatever
  // the slider (v56 M1-M3 MEASURED: "the MTP bank arena is too small" when its arena was sized with the planed bytes)
  std::size_t expert_bytes(const Qwen35MoeLayer& l, int e, bool planed = true) const {
    if (planed && o.plane_bits > 0) return al(planed_bytes(l, e));  // v54c: the planed bank holds the matrices alone
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
  // v54c: the bank in bit planes (the slider on): each hot expert's matrices back to back in VRAM, plane matrices as the
  // slider's prefix of each row (packed), the rest as stored. Only the R3 expert stage reads it (no stacks, no tables)
  Status build_bank_planes(const Qwen35MoeLayer& l, Layer& D) {
    const int ne = static_cast<int>(s.n_expert);
    std::size_t need_b = 0;
    for (int e = 0; e < ne; ++e)
      if (D.hot[z(e)]) need_b += al(planed_bytes(l, e));
    D.bg.clear();
    D.bu.clear();
    D.bd.clear();
    for (auto& ba : D.bank_at) ba.assign(z(ne), nullptr);
    if (need_b == 0) return Status::good();
    D.bank_mem = g::dalloc(need_b);
    if (!D.bank_mem) return gfail("engine: bank memory (planes)");
    dev_bytes += need_b;
    bank_bytes_total += need_b;
    auto* base = static_cast<std::uint8_t*>(D.bank_mem);
    std::size_t off = 0;
    std::vector<std::uint8_t> row, host;
    for (int e = 0; e < ne; ++e) {
      if (!D.hot[z(e)]) continue;
      host.clear();
      const WeightView* vv[3] = {&l.gate_exps, &l.up_exps, &l.down_exps};
      std::size_t moff[3];
      for (int pj = 0; pj < 3; ++pj) {
        const ExpRows r = exp_rows(*vv[pj], e);
        const std::uint32_t pt = planed_type(r.t, r.rb);
        const std::uint64_t rows = static_cast<std::uint64_t>(vv[pj]->ne[1]);
        moff[pj] = host.size();
        if (!pt) {
          host.insert(host.end(), r.base, r.base + r.rb * rows);
          continue;
        }
        const std::uint64_t pf = kern::gpu::plane_prefix(pt, r.rb, o.plane_bits);
        row.resize(z(static_cast<int64_t>(r.rb)));
        for (std::uint64_t rr = 0; rr < rows; ++rr) {
          kern::gpu::plane_row(static_cast<std::uint32_t>(r.t), r.base + rr * r.rb, row.data(), r.rb);
          host.insert(host.end(), row.begin(), row.begin() + static_cast<std::ptrdiff_t>(pf));
        }
      }
      if (!g::to_device(base + off, host.data(), host.size())) return gfail("engine: bank upload (planes)");
      for (int pj = 0; pj < 3; ++pj) D.bank_at[z(pj)][z(e)] = base + off + moff[pj];
      off += al(host.size());
    }
    return Status::good();
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
    if (W.lean) return Status::fail(Code::internal, "engine: " + v.name + " is held only in R3's copy (v55) and this product of " + std::to_string(n) + " rows is not on R3");
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
    std::vector<float*> yo;    // v55: direct plans: the outputs the plan writes (a call with other outputs takes the scratch plan)
  };
  // v55 (L1): a product's outputs go straight to where they belong when the call's outputs are the plan's (no scratch + copy),
  // and a one-stage group with no waits runs as a plain launch (no counter reset, no cooperative launch); same tasks, same sums
  std::uint64_t r3_direct_runs = 0, r3_copy_runs = 0;
  std::string r3_direct_note() const {
    return use_r3 ? "; R3 products: outputs written in place, one-stage groups as plain launches (v55)" : "";
  }
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
    // v55: the direct plan (built with this call's outputs) when the outputs match; otherwise the scratch plan and copies
    R3Group* Dp = nullptr;
    if (Status st = r3_group(jobs, n, 0x200, &Dp); !st) return st;
    bool same = Dp->yo.size() == jobs.size();
    {
      std::size_t i = 0;
      for (const auto& [W, yo] : jobs) {
        (void)W;
        same = same && Dp->yo[i++] == yo;
      }
    }
    if (!g::act_q8x32(xin, W0.k, n, W0.k, W0.v->rot32, xr, W0.k, q, scale, sum)) return gfail("engine: rounding for R3");
    if (same) {
      if (!Dp->plan->run_unstaged()) return Status::fail(Code::internal, std::string("engine: R3 run: ") + g::stream_last_error());
      ++r3_direct_runs;
      return Status::good();
    }
    R3Group* Gp = nullptr;
    if (Status st = r3_group(jobs, n, 0, &Gp); !st) return st;
    R3Group& G = *Gp;
    if (!G.plan->run_unstaged()) return Status::fail(Code::internal, std::string("engine: R3 run: ") + g::stream_last_error());
    std::size_t i = 0;
    for (const auto& [W, yo] : jobs)
      if (!g::dcopy(yo, r3y + G.off[i++], z(n * W->n_out) * sizeof(float))) return gfail("engine: R3 output");
    ++r3_copy_runs;
    return Status::good();
  }
  // v26 / v50: the R3 plan of a group for `rows` input rows (tag separates the halves' plans from plain ones of the same size)
  Status r3_group(std::initializer_list<std::pair<const DevW*, float*>> jobs, int64_t n, int tag, R3Group** gout) {
    const DevW& W0 = *jobs.begin()->first;
    R3Group& G = r3plans[{&W0, static_cast<int>(n) | tag}];
    *gout = &G;
    if (!G.plan) {
      std::vector<g::StreamJob> js;
      int64_t at = 0;
      for (const auto& [W, yo] : jobs) {
        if (!W->r3 || W->k != W0.k || W->v->rot32 != W0.v->rot32) return Status::fail(Code::internal, "engine: an R3 group with unlike inputs");
        g::StreamJob j;
        j.w = &W->sw;
        j.xq = q;
        j.xs = scale;
        j.xsum = sum;
        if (tag & 0x200) {  // v55: a direct plan writes the call's outputs
          j.y = yo;
          G.yo.push_back(yo);
        } else {
          if (static_cast<std::size_t>(at + n * W->n_out) > r3y_floats) return Status::fail(Code::internal, "engine: R3 scratch too small");
          j.y = r3y + at;
        }
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

  // v55g (L3): a DeltaNet layer's dense half for one token in one cooperative launch (attn norm + rounding, qkv + gate,
  // beta / alpha + gates, conv + tail, l2 norms, recurrence, gated norm + rounding, ssm_out), the same bits as the separate
  // launches; *took false (nothing run) where it does not apply, and the caller runs the attn norm and deltanet_block
  Status dn_half_layer(Layer& D, int64_t n, bool* took) {
    *took = false;
    // v56: 2-4 tokens where the products of n take R3; v57: only with dn_half_mt (v56b MEASURED: L3 for 3-token verify steps 33.5 ms
    // vs 24.8 ms without, the prompt 5.06 vs 3.82 s: its several-token stages still read the inputs from L2)
    if (n < 1 || n > 4 || (n > 1 && (!o.dn_half_mt || !bi_now))) return Status::good();
    const int64_t ch = s.conv_channels(), ds = s.ssm_d_state, nk = s.ssm_n_group, nv = s.ssm_dt_rank, taps = s.ssm_d_conv, e = s.n_embd;
    if (!o.dn_half || !dn_hcnt || !D.wqkv.r3 || !D.wgate.r3 || !D.ssm_out.r3 || D.wqkv.k != e || D.wgate.k != e || D.ssm_out.k != nv * ds ||
        D.wqkv.v->rot32 != D.wgate.v->rot32 || D.wbeta.v->rot32 || D.walpha.v->rot32 || D.wbeta.k != e || D.walpha.k != e || D.wbeta.n_out != nv ||
        D.walpha.n_out != nv || !is_float_type(D.wbeta.v->type) || !is_float_type(D.walpha.v->type) ||
        !g::deltanet_ba_fast(D.wbeta.dtype, D.wbeta.d, D.wbeta.drb, D.walpha.dtype, D.walpha.d, D.walpha.drb, nv, e, xn, n, e))
      return Status::good();
    R3Group* G1 = nullptr;
    R3Group* G2 = nullptr;
    if (Status st = r3_group({{&D.wqkv, mixed}, {&D.wgate, zg}}, n, 0x200, &G1); !st) return st;
    if (Status st = r3_group({{&D.ssm_out, y}}, n, 0x200, &G2); !st) return st;
    if (G1->yo.size() != 2 || G1->yo[0] != mixed || G1->yo[1] != zg || G2->yo.size() != 1 || G2->yo[0] != y) return Status::good();
    g::DnHalfArgs A;
    A.x = x;
    A.norm_w = D.attn_norm;
    A.eps = s.rms_eps;
    A.H = static_cast<int>(e);
    A.rot_in = D.wqkv.v->rot32 ? 1 : 0;
    A.rot_out = D.ssm_out.v->rot32 ? 1 : 0;
    A.xn = xn;
    A.xr = xr;
    A.q = q;
    A.scale = scale;
    A.sum = sum;
    A.p_in = G1->plan.get();
    A.p_out = G2->plan.get();
    A.wb = static_cast<const float*>(D.wbeta.d);
    A.wa = static_cast<const float*>(D.walpha.d);
    A.rbb = D.wbeta.drb;
    A.rba = D.walpha.drb;
    A.beta = beta;
    A.alpha = alpha;
    A.dt = D.ssm_dt;
    A.ssm_a = D.ssm_a;
    A.nk = static_cast<int>(nk);
    A.nv = static_cast<int>(nv);
    A.ds = static_cast<int>(ds);
    A.ch = static_cast<int>(ch);
    A.taps = static_cast<int>(taps);
    A.mixed = mixed;
    A.zg = zg;
    A.tail = D.tail[D.cur];
    A.new_tail = D.tail[1 - D.cur];
    A.conv_w = D.conv1d;
    A.state = D.state;
    A.core = core;
    A.gn_w = D.ssm_norm;
    A.qscale = 1.0f / std::sqrt(static_cast<float>(ds));  // as deltanet_recurrent_reg computes it
    A.hcnt = dn_hcnt;
    A.n = static_cast<int>(n);
    A.smact = o.dn_smem && A.rot_in == 0 ? 1 : 0;  // v56b; v57b: the several-token kernel too
    const bool ck = ckpt_mode && n > 1;  // v56: a verify step keeps every prefix's state (k_dn_rec_reg_ck) and the batch's conv inputs
    A.ckpt = ck ? D.ckpt : nullptr;
    A.msave = ck ? D.mixed_save : nullptr;
    if (ck && (!D.ckpt || !D.mixed_save)) return Status::good();
    A.ph = dph_d ? dph_d + static_cast<int64_t>(&D - L.data()) * 5 : nullptr;  // v56: stage stamps (verbose_gpu)
    if (!g::dn_half_ok(A)) return Status::good();
    if (!g::dn_half(A)) return Status::fail(Code::internal, std::string("engine: dense half (L3): ") + g::stream_last_error());
    D.cur = 1 - D.cur;
    ++dn_half_runs;
    *took = true;
    return Status::good();
  }

  Status deltanet_block(Layer& D, int64_t n) {
    const int64_t ch = s.conv_channels(), ds = s.ssm_d_state, nk = s.ssm_n_group, nv = s.ssm_dt_rank, taps = s.ssm_d_conv;
    Status st;
    if ((n == 1 || bi_now) && D.wqkv.r3 && D.wgate.r3 && D.wqkv.k == D.wgate.k) {
      if (!(st = prod_r3({{&D.wqkv, mixed}, {&D.wgate, zg}}, xn, n))) return st;
    } else if (!(st = prod(D.wqkv, xn, n, mixed)) || !(st = prod(D.wgate, xn, n, zg))) {
      return st;
    }
    // v55g (L2): beta, alpha and the gates in one launch where both products take the F32 fast path (the same sums)
    const bool ba = !D.wbeta.v->rot32 && !D.walpha.v->rot32 && D.wbeta.k == D.walpha.k && D.wbeta.n_out == nv && D.walpha.n_out == nv &&
                    is_float_type(D.wbeta.v->type) && is_float_type(D.walpha.v->type) &&
                    g::deltanet_beta_alpha_gates(D.wbeta.dtype, D.wbeta.d, D.wbeta.drb, D.walpha.dtype, D.walpha.d, D.walpha.drb, nv, D.wbeta.k, xn, n,
                                                 D.wbeta.k, beta, alpha, D.ssm_dt, D.ssm_a);
    if (!ba) {
      if (!(st = prod(D.wbeta, xn, n, beta)) || !(st = prod(D.walpha, xn, n, alpha))) return st;
    }
    if (!opstamp(2)) return gfail("engine: stamp");
    if (!ba && !g::deltanet_gates(beta, alpha, D.ssm_dt, D.ssm_a, n, nv)) return gfail("engine: deltanet gates");
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
  // v39 (author 01:57): tlp / il (the doorbell thread only, --timeline): stage[0] input rounding + grouping, stage[1] the
  // gate/up phase (silu fused in), stage[3] the down phase (combine fused in), wall time; busy[1] / busy[3] item time over all
  // threads; wake = the first item's start after the job opened; per-layer picks and groups.
  void cold_experts_fast(const Qwen35MoeLayer& l, int64_t n, const float* xsrc, float* ydst, EngineTimeline* tlp = nullptr, std::uint32_t il = 0) {
    const int64_t e = s.n_embd, ff = s.n_ff_exp;
    const int64_t np = static_cast<int64_t>(cold.size());
    const Clock::time_point tq = Clock::now();
    std::fill(need.begin(), need.begin() + n, std::uint8_t{0});
    for (const ColdPick& c : cold) need[z(c.t)] = 1;
    for (int64_t t = 0; t < n; ++t)
      if (need[z(t)]) {
        kern::act_q8x32_quantize(xsrc + t * e, e, actx[z(t)]);
        if (o.cpu_check) {  // v45: the AVX2 rounding against the scalar one, bit for bit (q, scales, rounded values)
          kern::ActQ8x32 rf;
          kern::act_q8x32_quantize_ref(xsrc + t * e, e, rf);
          const kern::ActQ8x32& av = actx[z(t)];
          const bool same = rf.q == av.q && rf.sum16 == av.sum16 && std::memcmp(rf.s.data(), av.s.data(), rf.s.size() * sizeof(float)) == 0 &&
                            std::memcmp(rf.x.data(), av.x.data(), rf.x.size() * sizeof(float)) == 0;
          round_n.fetch_add(1, std::memory_order_relaxed);
          if (!same) round_bad.fetch_add(1, std::memory_order_relaxed);
          if (xr_m) {  // v45b (author 17:21): the GPU's rounded input (act_q8x32's xr) against the CPU's, bit for bit (B5's question)
            gx_n.fetch_add(1, std::memory_order_relaxed);
            if (std::memcmp(xr_m + t * e, av.x.data(), static_cast<std::size_t>(e) * sizeof(float)) != 0) gx_bad.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    constexpr int64_t kRowsDown = 128, kGroup = 4;
    const int64_t kRowsUp = std::max<int64_t>(8, o.cpu_up_rows);  // v45: the split path (v40 put this in cold_experts_io by mistake)
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
    // v45 (Fable review): --cpu-gu2 on one-token steps computes gate row r and up row r in one call (same x, same grids;
    // each row's arithmetic as dot_row: bit for bit); a group then has ub items instead of 2 ub
    const bool gu2 = o.cpu_gu2 && n == 1;
    const int64_t per = gu2 ? ub : 2 * ub;
    for (int64_t gi = 0; gi < ng; ++gi) grp_left[z(gi)].store(static_cast<int>(per), std::memory_order_relaxed);
    const std::function<void(int64_t)> up_fn = [&](int64_t u) {
      const int64_t gi = u / per, rem = u % per, proj = gu2 ? 0 : rem / ub, r0 = (gu2 ? rem : rem % ub) * kRowsUp, r1 = std::min(ff, r0 + kRowsUp);
      const PickGroup& gp = grp[z(gi)];
      const ExpRows R = exp_rows(proj == 0 ? l.gate_exps : l.up_exps, gp.e);
      const kern::ActQ8x32* acts[kGroup] = {};
      float* outs[kGroup] = {};
      for (int j = 0; j < gp.m; ++j) {
        acts[j] = &actx[z(cold[z(gp.p[j])].t)];
        outs[j] = (proj == 0 ? gbuf.data() : ubuf.data()) + static_cast<int64_t>(gp.p[j]) * ff;
      }
      float o4[kGroup];
      if (gu2) {
        const ExpRows U = exp_rows(l.up_exps, gp.e);
        const bool pair_ok = gp.m == 1 && U.t == R.t && !kern::dot_q8_has_int_path(R.t);
        float* uo = ubuf.data() + static_cast<int64_t>(gp.p[0]) * ff;
        int64_t r = r0;
        if (o.cpu_tile && pair_ok && R.rb == U.rb) {  // v45b: the whole block of pairs in one call, the grids loaded once
          float* uo2 = ubuf.data() + static_cast<int64_t>(gp.p[0]) * ff;
          kern::dot_tile2_f32(R.t, R.base + static_cast<std::uint64_t>(r0) * R.rb, U.base + static_cast<std::uint64_t>(r0) * U.rb, R.rb, r1 - r0, e,
                              acts[0]->x.data(), outs[0] + r0, uo2 + r0, 1);
          if (o.cpu_check)
            for (int64_t rr = r0; rr < r1; ++rr) {
              float c0[kGroup], c1[kGroup];
              kern::dot_row_q8(R.t, R.base + static_cast<std::uint64_t>(rr) * R.rb, e, nullptr, acts, 1, c0);
              kern::dot_row_q8(U.t, U.base + static_cast<std::uint64_t>(rr) * U.rb, e, nullptr, acts, 1, c1);
              if (std::memcmp(&c0[0], outs[0] + rr, sizeof(float)) != 0 || std::memcmp(&c1[0], uo2 + rr, sizeof(float)) != 0)
                gu2_bad.fetch_add(1, std::memory_order_relaxed);
              gu2_n.fetch_add(2, std::memory_order_relaxed);
            }
          r = r1;
        }
        for (; r < r1; ++r) {
          const std::uint8_t* gr = R.base + static_cast<std::uint64_t>(r) * R.rb;
          const std::uint8_t* ur = U.base + static_cast<std::uint64_t>(r) * U.rb;
          if (o.cpu_prefetch && r + 2 < r1) {
            prefetch_row(gr + 2 * R.rb, R.rb);
            prefetch_row(ur + 2 * U.rb, U.rb);
          }
          if (pair_ok) {
            float o2[2];
            kern::dot_rows2_f32(R.t, gr, ur, e, acts[0]->x.data(), o2);
            if (o.cpu_check) {
              float c0[kGroup], c1[kGroup];
              kern::dot_row_q8(R.t, gr, e, nullptr, acts, 1, c0);
              kern::dot_row_q8(U.t, ur, e, nullptr, acts, 1, c1);
              if (std::memcmp(&c0[0], &o2[0], sizeof(float)) != 0 || std::memcmp(&c1[0], &o2[1], sizeof(float)) != 0)
                gu2_bad.fetch_add(1, std::memory_order_relaxed);
              gu2_n.fetch_add(2, std::memory_order_relaxed);
            }
            outs[0][r] = o2[0];
            uo[r] = o2[1];
          } else {
            const kern::ActQ8x32* au[kGroup] = {};
            for (int j = 0; j < gp.m; ++j) au[j] = acts[j];
            kern::dot_row_q8(R.t, gr, e, nullptr, acts, gp.m, o4);
            for (int j = 0; j < gp.m; ++j) outs[j][r] = o4[j];
            kern::dot_row_q8(U.t, ur, e, nullptr, au, gp.m, o4);
            for (int j = 0; j < gp.m; ++j) ubuf[z(static_cast<int64_t>(gp.p[j]) * ff + r)] = o4[j];
          }
        }
      } else {
        for (int64_t r = r0; r < r1; ++r) {
          if (o.cpu_prefetch && r + 2 < r1) prefetch_row(R.base + static_cast<std::uint64_t>(r + 2) * R.rb, R.rb);
          kern::dot_row_q8(R.t, R.base + static_cast<std::uint64_t>(r) * R.rb, e, nullptr, acts, gp.m, o4);
          for (int j = 0; j < gp.m; ++j) outs[j][r] = o4[j];
        }
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
        int64_t r = r0;
        if (o.cpu_tile && gp.m == 1 && kern::dot_q8_has_int_path(R.t) == false && r1 - r0 >= 2) {  // v45b: the block's row pairs in one call
          const int64_t np2 = (r1 - r0) / 2;
          const std::uint8_t* q0 = R.base + static_cast<std::uint64_t>(r0) * R.rb;
          kern::dot_tile2_f32(R.t, q0, q0 + R.rb, 2 * R.rb, np2, ff, acts[0]->x.data(), outs[0] + r0, outs[0] + r0 + 1, 2);
          if (o.cpu_check)
            for (int64_t rr = r0; rr < r0 + 2 * np2; ++rr) {
              float c0[kGroup];
              kern::dot_row_q8(R.t, R.base + static_cast<std::uint64_t>(rr) * R.rb, ff, nullptr, acts, 1, c0);
              if (std::memcmp(&c0[0], outs[0] + rr, sizeof(float)) != 0) down2_bad.fetch_add(1, std::memory_order_relaxed);
              down2_n.fetch_add(1, std::memory_order_relaxed);
            }
          r = r0 + 2 * np2;
        }
        if (o.cpu_down2 && gp.m == 1 && kern::dot_q8_has_int_path(R.t) == false) {  // v40: two rows per call, one token
          const float* xv = acts[0]->x.data();
          for (; r + 1 < r1; r += 2) {
            const std::uint8_t* q0 = R.base + static_cast<std::uint64_t>(r) * R.rb;
            if (o.cpu_prefetch && r + 5 < r1) {
              prefetch_row(q0 + 4 * R.rb, R.rb);
              prefetch_row(q0 + 5 * R.rb, R.rb);
            }
            float o2[2];
            kern::dot_rows2_f32(R.t, q0, q0 + R.rb, ff, xv, o2);
            if (o.cpu_check) {
              float c0[kGroup], c1[kGroup];
              kern::dot_row_q8(R.t, q0, ff, nullptr, acts, 1, c0);
              kern::dot_row_q8(R.t, q0 + R.rb, ff, nullptr, acts, 1, c1);
              if (std::memcmp(&c0[0], &o2[0], sizeof(float)) != 0 || std::memcmp(&c1[0], &o2[1], sizeof(float)) != 0)
                down2_bad.fetch_add(1, std::memory_order_relaxed);
              down2_n.fetch_add(2, std::memory_order_relaxed);
            }
            outs[0][r] = o2[0];
            outs[0][r + 1] = o2[1];
          }
        }
        for (; r < r1; ++r) {
          if (o.cpu_prefetch && r + 2 < r1) prefetch_row(R.base + static_cast<std::uint64_t>(r + 2) * R.rb, R.rb);
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
    if (!tlp) {
      pool->run2(ng * per, up_fn, db, down_fn);
      return;
    }
    // stamped run: the same items, each timed; the phase-1 end is the time the last up item finished
    const int64_t n1 = ng * per;
    std::atomic<std::int64_t> b1{0}, b2{0}, first{-1}, end1{0};
    std::atomic<int64_t> left1{n1};
    const Clock::time_point t0 = Clock::now();
    auto tns = [&](Clock::time_point t) { return std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0).count(); };
    std::atomic<std::int64_t> tu[32] = {}, td[32] = {}, fd[32];
    for (std::atomic<std::int64_t>& q3 : fd) q3.store(-1, std::memory_order_relaxed);
    const std::function<void(int64_t)> up_t = [&](int64_t u) {
      const Clock::time_point a = Clock::now();
      std::int64_t none = -1;
      first.compare_exchange_strong(none, tns(a), std::memory_order_relaxed);
      up_fn(u);
      const Clock::time_point b = Clock::now();
      b1.fetch_add(tns(b) - tns(a), std::memory_order_relaxed);
      tu[z(pool_tid())].fetch_add(tns(b) - tns(a), std::memory_order_relaxed);
      if (left1.fetch_sub(1, std::memory_order_acq_rel) == 1) end1.store(tns(b), std::memory_order_relaxed);
    };
    const std::function<void(int64_t)> down_t = [&](int64_t u) {
      const Clock::time_point a = Clock::now();
      {
        std::int64_t none = -1;
        fd[z(pool_tid())].compare_exchange_strong(none, tns(a), std::memory_order_relaxed);
      }
      down_fn(u);
      const std::int64_t dt = tns(Clock::now()) - tns(a);
      b2.fetch_add(dt, std::memory_order_relaxed);
      td[z(pool_tid())].fetch_add(dt, std::memory_order_relaxed);
    };
    const double tround = std::chrono::duration<double>(t0 - tq).count();
    pool->run2(n1, up_t, db, down_t);
    const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
    const double p1 = static_cast<double>(end1.load()) * 1e-9;
    tlp->split_stamps = true;
    for (int q2 = 0; q2 < 32; ++q2) {
      tlp->thr_up[q2] += static_cast<double>(tu[q2].load()) * 1e-9;
      tlp->thr_dn[q2] += static_cast<double>(td[q2].load()) * 1e-9;
    }
    tlp->thr_n = std::max(tlp->thr_n, std::min(32, pool_tid_count()));
    {  // v45: each thread's first down item after the gate/up phase ended (end1): the hand-over between the phases
      const std::int64_t e1 = end1.load();
      for (int q2 = 0; q2 < 32; ++q2) {
        const std::int64_t f2 = fd[q2].load();
        if (f2 < 0) continue;
        const double lag = static_cast<double>(f2 - e1) * 1e-9;
        tlp->lag_sum += lag;
        tlp->lag_max = std::max(tlp->lag_max, lag);
        ++tlp->lag_n;
      }
    }
    tlp->gu2_rows = gu2_n.load();
    tlp->gu2_mismatch = gu2_bad.load();
    tlp->round_n = round_n.load();
    tlp->round_mismatch = round_bad.load();
    tlp->gx_n = gx_n.load();
    tlp->gx_mismatch = gx_bad.load();
    tlp->down2_rows = down2_n.load();
    tlp->down2_mismatch = down2_bad.load();
    tlp->stage[0] += tround;
    tlp->stage[1] += p1;
    tlp->stage[3] += wall - p1;
    tlp->busy[1] += static_cast<double>(b1.load()) * 1e-9;
    tlp->busy[3] += static_cast<double>(b2.load()) * 1e-9;
    if (first.load() >= 0) {
      tlp->wake += static_cast<double>(first.load()) * 1e-9;
      ++tlp->wake_n;
    }
    for (const ColdPick& c : cold)
      for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps}) tlp->cold_bytes += static_cast<double>(exp_rows(*v, c.e).rb) * static_cast<double>(v == &l.down_exps ? e : ff);
    if (il < tlp->cpu_picks.size()) {
      tlp->cpu_picks[il] += static_cast<double>(np);
      tlp->cpu_groups[il] += static_cast<double>(ng);
    }
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
          if (o.spin_pause) relax_cpu();  // v45b --spin-pause
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
          else if (split && !o.split_slow) cold_experts_fast(w->layers[il], n, xn_m, yd, tlp, il);
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
  // v52: whether this layer's MoE half can run as one launch (g::moe_half): the R3 expert stage with the shared expert folded
  // in, an F32 router, the shared-memory picks; none of the side outputs the separate kernels feed (router margins, op
  // stamps, traces, probes, the CPU split, the plane tail, prefetch, the halves)
  // v58b: the first condition that keeps a layer's MoE half off the fused launch, or nullptr when it can run there (v58
  // MEASURED: the MTP block's took the old path on every step; this names why)
  const char* fused_why(const Layer& D, int64_t n, bool probe_row) const {
    const int64_t e = s.n_embd;
    auto k_ok = [](int64_t k) { return k == 256 || k == 512 || k == 1024 || k == 2048 || k == 4096; };
    if (!o.fused_moe) return "--no-fused-moe";
    if (!k_ok(e) || !g::fused_width_ok(static_cast<int>(e)) || !k_ok(s.n_ff_exp)) return "a width the fused kernels are not built for";
    if (s.n_expert > 1024 || s.n_expert_used > 8) return "more than 1024 experts or 8 picks";
    if (!rx_ok) return "the R3 expert stage is off";
    if (!fold_ok) return "the shared expert is not folded";
    if (split_ok || pf_ok || o.plane_tail != 0) return "the CPU split, prefetch or the plane tail";
    if (rlg_d || cur_ops || trc_src_d || trc_sel_d || probe_row) return "a side output (router margins, op stamps, trace, probe)";
    if (n < 1 || n > 4) return "a step of more than 4 tokens";
    const float* wr = router_of(D);
    if (!wr) return "the router is not F32 (and has no F32 copy)";
    if (!D.gate_inp_sh) return "no shared-expert gate";
    if (!D.post_norm) return "no post-attention norm";
    if ((reinterpret_cast<std::uintptr_t>(wr) & 15u) != 0 || (reinterpret_cast<std::uintptr_t>(D.gate_inp_sh) & 15u) != 0) return "router rows not 16-byte aligned";
    if (!g::router_fast_for(s.n_expert, o.router_tie)) return "the fast router does not take this expert count / tie band";
    return nullptr;
  }
  // the router rows the fused launch reads: gate_inp itself when stored as plain F32 rows, else the layer's F32 copy
  const float* router_of(const Layer& D) const {
    if (D.router_f32) return D.router_f32;
    const int64_t e = s.n_embd;
    if (D.gate_inp.v && !D.gate_inp.v->rot32 && D.gate_inp.dtype == static_cast<std::uint32_t>(quant::Type::f32) &&
        D.gate_inp.drb == static_cast<std::uint64_t>(e) * sizeof(float))
      return static_cast<const float*>(D.gate_inp.d);
    return nullptr;
  }
  bool fused_ok(const Layer& D, int64_t n, bool probe_row) const { return fused_why(D, n, probe_row) == nullptr; }

  // v52: x += y (the block's output, in y), then the MoE half, then x += y, in one launch
  Status moe_half_fused(Layer& D, int64_t n, std::uint32_t il) {
    g::MoeHalfArgs A;
    A.x = x;
    A.a = y;
    A.post_w = D.post_norm;
    A.eps = s.rms_eps;
    A.n = static_cast<int>(n);
    A.H = static_cast<int>(s.n_embd);
    A.F = static_cast<int>(s.n_ff_exp);
    A.ne = static_cast<int>(s.n_expert);
    A.nu = static_cast<int>(s.n_expert_used);
    A.xn = xn;
    A.xr = xr;
    A.xq = q;
    A.xs = scale;
    A.xsum = sum;
    A.wr = router_of(D);  // v58b
    A.logits = logits_r;
    A.wsh = D.gate_inp_sh;
    A.gsh = gsh;
    A.wscale = s.expert_weights_scale;
    A.band = o.router_tie;
    A.sel = sel;
    A.wsel = wsel;
    A.tab = rtab + static_cast<int64_t>(il) * s.n_expert;
    A.shx = rshx + il;
    if (!rcache.empty()) A.cache = rcache[il];
    A.grp = rgrp;
    A.gsrc = rgsrc;
    A.gfill = rgfill;
    A.gu = rgu;
    A.hq = rhq;
    A.hs = rhs;
    A.hsum = rhsum;
    A.yd = ryd;
    A.plan = rxplan;
    A.max_picks = rx_maxp;
    A.load_mode = o.rx_load;
    A.y = y;
    A.ph = ph_d && il < s.n_layer ? ph_d + static_cast<int64_t>(il) * 5 : nullptr;  // v55b (v58: not the MTP row)
    A.pipe = o.rx_pipe && o.rx_load > 0 ? 1 : 0;  // v57
    A.sact = o.rx_sact ? 1 : 0;  // v57
    A.st = rst_d;                                                 // v55e3
    // v55f / v55f2: the bit planes only feed planed experts' 1-bit route; with no planes (--plane-bits 0) nothing reads
    // them, so they are not built (v55f2 S0 +0.33 ms vs g55e3, MEASURED)
    A.xb = o.plane_bits > 0 ? rxb : nullptr;
    A.hb = o.plane_bits > 0 ? rhb : nullptr;
    if (!g::moe_half(A)) return Status::fail(Code::internal, std::string("engine: fused MoE half: ") + g::stream_last_error());
    return Status::good();
  }
  Status moe_block_reader(Layer& D, int64_t n, std::uint32_t il) {
    const int64_t e = s.n_embd, ne = s.n_expert, nu = s.n_expert_used, ffs = s.n_ff_shexp;
    Status st;
    if (!(st = prod(D.gate_inp, xn, n, logits_r))) return st;
    if (rlg_d && !g::dcopy(rlg_d + static_cast<int64_t>(il) * kReaderMax * ne, logits_r, z(n * ne) * sizeof(float))) return gfail("engine: router margins");  // v49
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
          !g::to_host_async(xn_m, xn, z(n * e) * sizeof(float)) || (xr_m && !g::to_host_async(xr_m, xr, z(n * e) * sizeof(float))) ||
          !g::bell_ring_dev(fa_d, gen_d, il))
        return gfail("engine: split (picks to the CPU)");
    }
    if (pf_ok && il < pf_issued.size() && pf_issued[il]) {  // v26: this layer's prefetch (issued after the previous layer) is done
      if (!g::stream_wait(nullptr, ev_b[il - 1])) return gfail("engine: prefetch wait");
      pf_issued[il] = 0;
    }
    // v46a-1: the R3 expert stage (not with the split, whose CPU picks it does not skip, or the plane test)
    const bool use_rx = rx_ok && !split_ok && o.plane_tail == 0;
    const bool rd_ok =
        use_rx ? g::moe_r3(rtab + static_cast<int64_t>(il) * ne, fold ? rshx + il : nullptr, sel, wsel, gsh, static_cast<int>(nu), q, scale, sum, n, e,
                           s.n_ff_exp, rcache.empty() ? nullptr : &rcache[il], rgu, rhq, rhs, rhsum, ryd, rgrp, rgsrc, rgfill, rxplan, rx_maxp, y, tsl,
                           nullptr, o.rx_load)
        : g::moe_reader3(rtab + static_cast<int64_t>(il) * ne, fold ? rshx + il : nullptr, sel, wsel, gsh, static_cast<int>(nu), xr, n, e, s.n_ff_exp,
                         rrow_max, rcache.empty() ? nullptr : &rcache[il], rgu, rh, ryd, rgrp, rgsrc, rgfill, y, tsl, nullptr, o.plane_tail,
                         split_ok ? lmask_d + static_cast<int64_t>(il) * ne : nullptr, o.reader_rows_per_warp);
    if (!rd_ok) return use_rx ? Status::fail(Code::internal, std::string("engine: R3 expert stage: ") + g::stream_last_error()) : gfail("engine: expert reader");
    if (trc_src_d) {
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
      if (trc_src_h) {  // v26: groups served from VRAM by the cache, and misses written into a slot
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
    for (std::uint32_t il = 0; il < nlr(); ++il)  // v58: nlr() = n_layer (+ the MTP block's row with mtp_fused)
      for (const WeightView* v : {&WL(il).gate_exps, &WL(il).up_exps, &WL(il).down_exps}) {
        if (v->rot32) {
          reader_note = "off (a rotated expert weight)";
          return;
        }
        for (int64_t ex = 0; ex < ne; ++ex) {
          const ExpRows r = exp_rows(*v, static_cast<int>(ex));
          if (!g::type_supported(static_cast<std::uint32_t>(r.t)) && il >= s.n_layer) {  // v58: the MTP block keeps the old path
            mtp_rd = false;
            break;
          }
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
    const bool split = o.cpu_light && o.doorbell;
    int n_light_all = 0;
    std::uint64_t light_b = 0;
    for (std::uint32_t il = 0; il < nlr(); ++il) {
      LL(il).light.assign(z(ne), 0);
      LL(il).n_light = 0;
      if (!split || il >= s.n_layer) continue;
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
    // v54: the bit planes: every Q4_TG2 matrix of an expert outside the bank (and not the CPU's) copied row by row into the
    // plane layout, in pinned mapped memory; those matrices' file bytes are then not registered
    std::vector<std::uint64_t> poff;  // per (layer, matrix, expert): offset in planes_h + 1 (0: not in planes)
    void* planes_dev = nullptr;
    if (o.plane_bits > 0) {
      poff.assign(z(static_cast<int64_t>(nlr()) * 3 * ne), 0);
      std::uint64_t tot = 0;
      struct PJob {
        const std::uint8_t* src;
        std::uint64_t off, rb, rows;
        std::uint32_t t;
      };
      std::vector<PJob> jobs;
      for (std::uint32_t il = 0; il < nlr(); ++il) {
        const WeightView* vv[3] = {&WL(il).gate_exps, &WL(il).up_exps, &WL(il).down_exps};
        for (int pj = 0; pj < 3; ++pj)
          for (int64_t ex = 0; ex < ne; ++ex) {
            if (LL(il).hot[z(ex)] || LL(il).light[z(ex)]) continue;
            const ExpRows r = exp_rows(*vv[pj], static_cast<int>(ex));
            if (!planed_type(r.t, r.rb)) continue;  // v54c: Q4_TG2 and Q5_K
            const std::uint64_t rows = static_cast<std::uint64_t>(vv[pj]->ne[1]), b = r.rb * rows;
            poff[z((static_cast<int64_t>(il) * 3 + pj) * ne + ex)] = tot + 1;
            jobs.push_back({r.base, tot, r.rb, rows, static_cast<std::uint32_t>(r.t)});
            tot += (b + 255) / 256 * 256;  // each matrix 256-byte aligned
          }
      }
      if (tot > 0) {
        const Clock::time_point tp = Clock::now();
        planes_h = g::host_alloc_mapped(static_cast<std::size_t>(tot), &planes_dev);
        if (!planes_h || !planes_dev) {
          reader_note = "off (pinned memory for the bit planes: " + std::to_string(tot >> 20) + " MiB)";
          planes_h = nullptr;
          return;
        }
        planes_b = tot;
        auto* ph = static_cast<std::uint8_t*>(planes_h);
        const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nt; ++t)
          th.emplace_back([&]() {
            for (std::size_t i = next.fetch_add(1); i < jobs.size(); i = next.fetch_add(1)) {
              const PJob& J = jobs[i];
              for (std::uint64_t rr = 0; rr < J.rows; ++rr) kern::gpu::plane_row(J.t, J.src + rr * J.rb, ph + J.off + rr * J.rb, J.rb);
            }
          });
        for (auto& t : th) t.join();
        char pb[200];
        std::snprintf(pb, sizeof pb, "; bit planes: %zu Q4_TG2 / Q5_K matrices outside the bank, %.2f GB pinned, built in %.2f s; every pick read with %d bits (at most each matrix's own), bank and slots sized for them",
                      jobs.size(), static_cast<double>(tot) / 1e9, since(tp), o.plane_bits);
        planes_note = pb;
      }
    }
    // v27: pin only the experts the bank does not hold (the GPU reads the bank's from VRAM): their matrices as page-aligned
    // ranges, merged where the gap between them is under o.pin_gap (v27: 8 MiB; fewer, larger registrations); each range's
    // device address. v28: not the light experts either when the split is on. v54: not the matrices held in bit planes.
    const std::uintptr_t pg = 4096;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> want;
    for (std::uint32_t il = 0; il < nlr(); ++il)
      for (const WeightView* v : {&WL(il).gate_exps, &WL(il).up_exps, &WL(il).down_exps})
        for (int64_t ex = 0; ex < ne; ++ex) {
          if (LL(il).hot[z(ex)] || LL(il).light[z(ex)]) continue;
          const int pjv = v == &WL(il).gate_exps ? 0 : v == &WL(il).up_exps ? 1 : 2;
          if (!poff.empty() && poff[z((static_cast<int64_t>(il) * 3 + pjv) * ne + ex)]) continue;
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
    std::vector<g::ExpertPtr> tab(z(static_cast<int64_t>(nlr()) * ne));
    for (std::uint32_t il = 0; il < nlr(); ++il) {
      const Qwen35MoeLayer& l = WL(il);
      Layer& Li = LL(il);
      const WeightView* vv[3] = {&l.gate_exps, &l.up_exps, &l.down_exps};
      for (int64_t ex = 0; ex < ne; ++ex) {
        g::ExpertPtr& X = tab[z(static_cast<int64_t>(il) * ne + ex)];
        for (int pj = 0; pj < 3; ++pj) {
          const ExpRows r = exp_rows(*vv[pj], static_cast<int>(ex));
          const std::uint8_t* bank = Li.hot[z(ex)] && !Li.bank_at[pj].empty() ? Li.bank_at[pj][z(ex)] : nullptr;
          X.w[pj] = bank ? bank : Li.light[z(ex)] ? nullptr : dev_of(r.base);
          X.type[pj] = static_cast<std::uint32_t>(r.t);
          X.rb[pj] = r.rb;
          if (bank) X.vram = 1;
          if (!bank && !poff.empty() && poff[z((static_cast<int64_t>(il) * 3 + pj) * ne + ex)]) {  // v54: from the bit planes
            X.w[pj] = static_cast<const std::uint8_t*>(planes_dev) + (poff[z((static_cast<int64_t>(il) * 3 + pj) * ne + ex)] - 1);
            X.type[pj] = planed_type(r.t, r.rb);
          }
          if (bank && planed_type(r.t, r.rb)) {  // v54c: the planed bank holds the slider's prefix of each row
            X.type[pj] = planed_type(r.t, r.rb);
            X.rb[pj] = kern::gpu::plane_prefix(X.type[pj], r.rb, o.plane_bits);
          }
        }
      }
    }
    for (std::size_t ti = 0; ti < tab.size(); ++ti) {
      const g::ExpertPtr& X = tab[ti];
      const std::uint32_t til = static_cast<std::uint32_t>(static_cast<int64_t>(ti) / ne);
      if (LL(til).light[z(static_cast<int64_t>(ti) % ne)]) continue;  // v28: the CPU's, never read by the GPU
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
    if (o.fold_shared && s.n_ff_shexp == s.n_ff_exp) {
      bool ok = true;
      for (std::uint32_t il = 0; il < nlr() && ok; ++il) {
        const Layer& D = LL(il);
        g::ExpertPtr X;
        const DevW* dw[3] = {&D.gate_sh, &D.up_sh, &D.down_sh};
        for (int pj = 0; pj < 3; ++pj) {
          const DevW& W = *dw[pj];
          if (!W.d || !W.v || W.v->rot32 || !g::type_supported(W.dtype) || W.k != (pj < 2 ? s.n_embd : s.n_ff_shexp)) ok = false;
          X.w[pj] = static_cast<const std::uint8_t*>(W.d);
          X.type[pj] = W.dtype;
          X.rb[pj] = W.drb;
        }
        if (!ok && il >= s.n_layer) {  // v58: the MTP block's shared expert cannot be folded: its MoE half stays on the old path
          mtp_rd = false;
          ok = true;
          break;
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
    if (cache_reserved > 0) {
      int K = rc_eff;
      const std::size_t nl = nlr();
      const std::uint64_t budget = cslot * static_cast<std::uint64_t>(rc_eff);  // per layer, as before (v54b: rc_eff, the probe's or --reader-cache)
      // v51 size pools: per layer the file experts (not in the bank) by real size; each size's share of the budget follows
      // its experts' routes counts x size (counts + 1); at most kMaxPools sizes (the smallest-demand size merges upward)
      std::vector<g::ReaderCache> pc(nl);
      bool pools_ok = true;  // v51b
      std::vector<signed char> pool_h(nl * z(ne), -1);
      int kpool_max = K;
      if (o.cache_pools) {
        kpool_max = 0;
        for (std::size_t il = 0; il < nl; ++il) {
          const Qwen35MoeLayer& l = WL(static_cast<std::uint32_t>(il));
          Layer& Li = LL(static_cast<std::uint32_t>(il));
          std::vector<std::uint64_t> esz(z(ne), 0);
          std::map<std::uint64_t, double> dem;
          for (int ex = 0; ex < static_cast<int>(ne); ++ex) {
            if (Li.hot[z(ex)]) continue;
            std::uint64_t b = 0;
            for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
              const ExpRows r = exp_rows(*v, ex);
              // v54b: a matrix held in bit planes takes the slider's prefix of each row in its slot (as rd3_group_one lays it)
              const std::uint32_t pt = Li.light[z(ex)] ? 0u : planed_type(r.t, r.rb);
              b += (pt ? kern::gpu::plane_prefix(pt, r.rb, o.plane_bits) : r.rb) * static_cast<std::uint64_t>(v->ne[1]);
            }
            esz[z(ex)] = (b + 255) & ~std::uint64_t{255};
            const double cn = il < o.hot_count.size() && z(ex) < o.hot_count[il].size() ? static_cast<double>(o.hot_count[il][z(ex)]) : 0.0;
            dem[esz[z(ex)]] += (cn + 1.0) * static_cast<double>(esz[z(ex)]);
          }
          while (dem.size() > static_cast<std::size_t>(g::ReaderCache::kMaxPools)) {  // merge the smallest-demand size (not the largest) upward
            auto lw = dem.begin();
            for (auto it = dem.begin(); std::next(it) != dem.end(); ++it)
              if (it->second < lw->second) lw = it;
            const double d = lw->second;
            auto up = std::next(lw);
            dem.erase(lw);
            up->second += d;
          }
          double tot = 0.0;
          for (const auto& kv : dem) tot += kv.second;
          // v51b: slot counts first, then the layout. Each size gets floor(budget x share / size) slots (at least 1 while it
          // fits); if the forced ones push the layer past its budget, the size with the most slots gives one back until it fits
          // (v51 laid the pools out as it went and could run past the budget into the next layer's region: G51's garbage)
          std::vector<std::uint64_t> szs;
          std::vector<double> dms;
          std::vector<int> cnt;
          for (const auto& kv : dem) {
            szs.push_back(kv.first);
            dms.push_back(kv.second);
            cnt.push_back(tot > 0 ? static_cast<int>(static_cast<double>(budget) * kv.second / tot / static_cast<double>(kv.first)) : 0);
          }
          auto total = [&]() {
            std::uint64_t u = 0;
            for (std::size_t pi = 0; pi < szs.size(); ++pi) u += static_cast<std::uint64_t>(cnt[pi]) * szs[pi];
            return u;
          };
          for (std::size_t pi = 0; pi < szs.size(); ++pi)
            if (cnt[pi] < 1 && total() + szs[pi] <= budget) cnt[pi] = 1;
          while (total() > budget) {
            std::size_t mx = 0;
            for (std::size_t pi = 1; pi < szs.size(); ++pi)
              if (cnt[pi] > cnt[mx]) mx = pi;
            if (cnt[mx] == 0) break;
            --cnt[mx];
          }
          // leftover bytes: one more slot at a time to the size of most demand per byte per slot that still fits
          for (bool more = true; more;) {
            more = false;
            int best = -1;
            double bd = -1.0;
            for (std::size_t pi = 0; pi < szs.size(); ++pi)
              if (total() + szs[pi] <= budget && dms[pi] / static_cast<double>(szs[pi]) / (cnt[pi] + 1) > bd) {
                bd = dms[pi] / static_cast<double>(szs[pi]) / (cnt[pi] + 1);
                best = static_cast<int>(pi);
              }
            if (best >= 0) {
              ++cnt[static_cast<std::size_t>(best)];
              more = true;
            }
          }
          g::ReaderCache& c = pc[il];
          std::uint64_t used = 0;
          int k = 0;
          for (std::size_t pi = 0; pi < szs.size(); ++pi) {
            c.pbytes[pi] = szs[pi];
            c.pboff[pi] = used;
            c.pk0[pi] = k;
            used += static_cast<std::uint64_t>(cnt[pi]) * szs[pi];
            k += cnt[pi];
          }
          c.np = static_cast<int>(szs.size());
          c.pk0[c.np] = k;
          if (used > budget) pools_ok = false;  // never expected after the loop above; checked, not assumed
          c.K = c.pk0[c.np];
          kpool_max = std::max(kpool_max, c.K);
          for (int ex = 0; ex < static_cast<int>(ne); ++ex) {
            if (esz[z(ex)] == 0) continue;
            for (int pq = 0; pq < c.np; ++pq)
              if (c.pbytes[pq] >= esz[z(ex)] && c.pk0[pq + 1] > c.pk0[pq]) {
                pool_h[il * z(ne) + z(ex)] = static_cast<signed char>(pq);
                break;
              }
          }
        }
        K = kpool_max;
        if (!pools_ok) {  // v51b: fall back to the fixed slots rather than run past a layer's budget
          K = rc_eff;
          pc.assign(nl, g::ReaderCache{});
        }
      }
      const bool use_pools = o.cache_pools && pools_ok;
      const std::size_t meta_b = nl * (static_cast<std::size_t>(K) * 8 + 16) + (use_pools ? nl * z(ne) : 0);
      cache_mem = g::dalloc(cache_reserved);
      cache_meta = g::dalloc(meta_b);
      if (cache_mem && cache_meta) {
        std::vector<std::uint8_t> init(meta_b, 0);
        auto* ip = reinterpret_cast<int*>(init.data());
        for (std::size_t i = 0; i < nl * static_cast<std::size_t>(K); ++i) ip[i] = -1;  // tags: empty
        if (use_pools) std::memcpy(init.data() + nl * (static_cast<std::size_t>(K) * 8 + 16), pool_h.data(), pool_h.size());
        if (g::to_device(cache_meta, init.data(), meta_b)) {
          auto* mb = static_cast<std::uint8_t*>(cache_meta);
          int kmin = K, kall = 0;
          for (std::size_t il = 0; il < nl; ++il) {
            g::ReaderCache c = use_pools ? pc[il] : g::ReaderCache{};
            if (!use_pools) c.K = K;
            c.tag = reinterpret_cast<int*>(mb) + il * static_cast<std::size_t>(K);
            c.use = reinterpret_cast<unsigned*>(mb + nl * static_cast<std::size_t>(K) * 4) + il * static_cast<std::size_t>(K);
            c.stamp = reinterpret_cast<unsigned*>(mb + nl * static_cast<std::size_t>(K) * 8) + il;
            c.base = static_cast<std::uint8_t*>(cache_mem) + il * budget;
            c.slot_bytes = cslot;
            if (use_pools) c.pool_of = reinterpret_cast<const signed char*>(mb + nl * (static_cast<std::size_t>(K) * 8 + 16)) + il * z(ne);
            if (o.plane_bits > 0) c.plane_bits = o.plane_bits;  // v54b
            kmin = std::min(kmin, c.K);
            kall += c.K;
            rcache.push_back(c);
          }
          dev_bytes += cache_reserved + meta_b;
          char cb[200];
          if (use_pools)
            std::snprintf(cb, sizeof cb, "; cache %.2f GB in size pools (%d slots of %d before; %d-%d slots per layer, %.1f average)", static_cast<double>(cache_reserved) / 1e9,
                          rc_eff, rc_eff, kmin, K, static_cast<double>(kall) / static_cast<double>(nl));
          else
            std::snprintf(cb, sizeof cb, "; cache %d slots per layer, %.2f GB%s", K, static_cast<double>(cache_reserved) / 1e9,
                          o.cache_pools ? " (size pools off: layout over budget)" : "");
          cache_note = cb;
        }
      }
      if (rcache.empty()) cache_note = "; cache off (device memory)";
    }
    if (o.plane_bits > 0 && rcache.empty()) {  // v54b: no slots, but the grouping still needs the slider
      g::ReaderCache c0;
      c0.plane_bits = o.plane_bits;
      rcache.assign(nlr(), c0);
    }
    rtab_h = tab;
    // v46a-1: the R3 expert stage. It reads the experts as stored (Q4_TG2 / Q5_K / Q4_K with 16-byte aligned rows, Q6_K
    // 2-byte aligned) and the shared expert in R3's Q8_0 layout; each matrix's start 16-byte aligned (a miss's bytes go to
    // the same offsets in its slot, so the slots keep that alignment too)
    if (o.r3_experts) {
      const int64_t H = s.n_embd, Fx = s.n_ff_exp;
      auto mat_ok = [&](const g::ExpertPtr& X) {
        for (int pj = 0; pj < 3; ++pj) {
          if (!g::rx_type_ok(X.type[pj], pj < 2 ? H : Fx) || !X.w[pj]) return false;
          if (reinterpret_cast<std::uintptr_t>(X.w[pj]) % 16 != 0) return false;
          if (X.type[pj] == g::g_q6_k ? X.rb[pj] % 2 != 0 : X.rb[pj] % 16 != 0) return false;
        }
        return true;
      };
      bool ok = true;
      for (std::size_t ti = 0; ti < tab.size(); ++ti) {
        if (mat_ok(tab[ti])) continue;
        if (ti >= z(static_cast<int64_t>(s.n_layer) * s.n_expert)) mtp_rd = false;  // v58: only the MTP row: its MoE half keeps the old path
        else ok = false;
      }
      if (!ok) rx_note = "off (an expert matrix the stage cannot read: type or alignment)";
      for (const g::ExpertPtr& X : shx_h)
        if (ok && !mat_ok(X)) {
          ok = false;
          rx_note = "off (the shared expert's type or alignment)";
        }
      for (const g::ReaderCache& c : rcache)
        if (ok && (c.slot_bytes % 16 != 0 || reinterpret_cast<std::uintptr_t>(c.base) % 16 != 0)) {
          ok = false;
          rx_note = "off (cache slots not 16-byte aligned)";
        }
      if (ok) {
        rx_maxp = static_cast<int>(P);
        rhq = static_cast<std::int8_t*>(g::dalloc(z(P * Fx)));
        rhs = static_cast<float*>(g::dalloc(z(P * Fx / 32) * sizeof(float)));
        rhsum = static_cast<int*>(g::dalloc(z(P * Fx / 32) * sizeof(int)));
        const std::uint64_t pb = g::rx_scratch_bytes(rx_maxp);
        rxplan = g::dalloc(pb);
        // v55f: bit planes for the 1-bit route (8 words per 32 inputs = 1 byte per input): up to 4 tokens of H, every pick's
        // F; without them the stage reads planed experts by dp4a as before
        rxb = static_cast<std::uint32_t*>(g::dalloc(z(4 * s.n_embd)));
        rhb = static_cast<std::uint32_t*>(g::dalloc(z(P * Fx)));
        if (!rxb || !rhb) {
          g::dfree(rxb);
          g::dfree(rhb);
          rxb = rhb = nullptr;
        } else {
          dev_bytes += z(4 * s.n_embd) + z(P * Fx);
        }
        if (rhq && rhs && rhsum && rxplan) {
          dev_bytes += z(P * Fx) + z(P * Fx / 32) * 8 + pb;
          rx_ok = true;
          rx_note = "on (gate / up, silu x up, down of every pick in one R3 launch per layer)";
          // v55b: the stage's one grid in shared memory: the slider's bits for every Q4_TG2 job (planes), else the stored grid
          {  // v55f: the staging buffer sized to the largest block of any matrix the stage stages (load mode 1: the experts
             // not in VRAM; 2: every expert and the shared one), at the slider's bits for the plane matrices
            int mb = 0;
            auto take = [&](const g::ExpertPtr& X) {
              for (int pj = 0; pj < 3; ++pj) mb = std::max(mb, g::rx_stage_block_bytes(X.type[pj], o.plane_bits));
            };
            for (const g::ExpertPtr& X : tab)
              if (o.rx_load == 2 || o.rx_pipe || !X.vram) take(X);  // v57: the pipeline stages every expert
            if (o.rx_load == 2 || o.rx_pipe)
              for (const g::ExpertPtr& X : shx_h) take(X);
            g::rx_set_stage_bytes(mb);
          }
          if (!g::rx_set_grid_bits(o.plane_bits > 0 ? std::min(o.plane_bits, 4) : 4)) {
            rx_ok = false;
            rx_note = std::string("off (grid bits: ") + g::stream_last_error() + ")";
          }
          if (rx_ok && o.verbose_gpu) {
            g::stream_kernel_report();
            void* phd = nullptr;
            ph_h = static_cast<unsigned long long*>(g::host_alloc_mapped(static_cast<std::size_t>(s.n_layer) * 5 * sizeof(unsigned long long), &phd));
            ph_d = static_cast<unsigned long long*>(phd);
            if (!ph_h || !ph_d) ph_h = ph_d = nullptr;
            void* dphd = nullptr;  // v56
            dph_h = static_cast<unsigned long long*>(g::host_alloc_mapped(static_cast<std::size_t>(s.n_layer) * 5 * sizeof(unsigned long long), &dphd));
            dph_d = static_cast<unsigned long long*>(dphd);
            if (!dph_h || !dph_d) dph_h = dph_d = nullptr;
            rst_d = static_cast<unsigned long long*>(g::dalloc(g::kRxStatN * sizeof(unsigned long long)));  // v55e3
            if (rst_d && !g::dzero(rst_d, g::kRxStatN * sizeof(unsigned long long))) {
              g::dfree(rst_d);
              rst_d = nullptr;
            }
          }
        } else {
          rx_note = "off (device memory)";
        }
      }
    }
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
    if (o.router_margins) {  // v49
      void* dv = nullptr;
      rlg_h = static_cast<float*>(g::host_alloc_mapped(z(s.n_layer) * kReaderMax * z(ne) * sizeof(float), &dv));
      rlg_d = static_cast<float*>(dv);
      if (rlg_h) {
        rmarg.assign(z(o.max_ctx) * 2 * z(s.n_layer), 0.0f);
        rmarg_have.assign(z(o.max_ctx), 0);
      }
    }
    reader = true;
    char b[160];
    char b2[640];
    std::snprintf(b2, sizeof b2, "on (%s, %d rows per warp, steps of 1-%d tokens; %.2f GB of the file (%s) registered in %zu ranges (gap %llu KiB; small pages advised on %zu) in %.2f s%s%s)",
                  "v3", o.reader_rows_per_warp, kReaderMax, static_cast<double>(b1 - b0) / 1e9,
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
  g::dfree(p_->rhq);
  g::dfree(p_->rxb);  // v55f
  g::dfree(p_->rhb);
  g::dfree(p_->rhs);
  g::dfree(p_->rhsum);
  g::dfree(p_->rxplan);
  g::dfree(p_->cache_mem);
  g::dfree(p_->cache_meta);
  g::host_free(p_->trc_hv);
  g::host_free(p_->planes_h);  // v54
  g::host_free(p_->ph_h);  // v55b
  g::host_free(p_->dph_h);  // v56
  g::dfree(p_->rst_d);      // v55e3
  g::host_free(p_->rlg_h);  // v49
  g::dfree(p_->probe_bd);
  g::dfree(p_->probe_ld);
  g::dfree(p_->probe_md);
  g::host_free(p_->bell_h);
  g::host_free(p_->xr_h);
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
  if (p_->M.router_f32) g::dfree(const_cast<float*>(p_->M.router_f32));  // v58b: the MTP router's F32 copy (device memory)
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
std::string GpuEngine::moe_phase_report() const {  // v55b
  const Impl& I = *p_;
  if (!I.ph_h || I.ph_steps == 0) return "";
  const double st = static_cast<double>(I.ph_steps);
  char b[400];
  std::snprintf(b, sizeof b,
                "fused MoE half, ms per token over %llu decode steps (%.1f layers per step; GPU clock, block 0): A norm + router + shared gate %.3f, "
                "B picks + grouping + plan (block 0) %.3f, C expert stage %.3f, D combine + residual %.3f; total %.3f",
                static_cast<unsigned long long>(I.ph_steps), static_cast<double>(I.ph_layers) / st, I.ph_sum[0] / st, I.ph_sum[1] / st, I.ph_sum[2] / st,
                I.ph_sum[3] / st, (I.ph_sum[0] + I.ph_sum[1] + I.ph_sum[2] + I.ph_sum[3]) / st);
  std::string r = b;
  if (I.dph_h && I.dph_steps > 0) {  // v56
    const double ds2 = static_cast<double>(I.dph_steps);
    char d[400];
    std::snprintf(d, sizeof d,
                  "\nDeltaNet dense half (L3), ms per token over %llu decode steps (%.1f layers per step; GPU clock, block 0): 0 norm + rounding %.3f, "
                  "1 qkv + gate + beta / alpha %.3f, 2 conv + l2 + recurrence + gated norm %.3f, 3 ssm_out %.3f; total %.3f",
                  static_cast<unsigned long long>(I.dph_steps), static_cast<double>(I.dph_layers) / ds2, I.dph_sum[0] / ds2, I.dph_sum[1] / ds2,
                  I.dph_sum[2] / ds2, I.dph_sum[3] / ds2, (I.dph_sum[0] + I.dph_sum[1] + I.dph_sum[2] + I.dph_sum[3]) / ds2);
    r += d;
  }
  for (int k = 0; k < 3; ++k) {  // v56: verify steps (MTP) by size
    char d[400];
    if (I.vph_steps[k] > 0) {
      const double st2 = static_cast<double>(I.vph_steps[k]);
      std::snprintf(d, sizeof d, "\nverify steps of %d tokens (%llu): fused MoE half ms per step: A %.3f, B %.3f, C expert stage %.3f, D %.3f; total %.3f", k + 2,
                    static_cast<unsigned long long>(I.vph_steps[k]), I.vph_sum[k][0] / st2, I.vph_sum[k][1] / st2, I.vph_sum[k][2] / st2, I.vph_sum[k][3] / st2,
                    (I.vph_sum[k][0] + I.vph_sum[k][1] + I.vph_sum[k][2] + I.vph_sum[k][3]) / st2);
      r += d;
    }
    if (I.vdph_steps[k] > 0) {
      const double st2 = static_cast<double>(I.vdph_steps[k]);
      std::snprintf(d, sizeof d, "\nverify steps of %d tokens (%llu): DeltaNet dense half ms per step: 0 %.3f, 1 %.3f, 2 %.3f, 3 %.3f; total %.3f", k + 2,
                    static_cast<unsigned long long>(I.vdph_steps[k]), I.vdph_sum[k][0] / st2, I.vdph_sum[k][1] / st2, I.vdph_sum[k][2] / st2,
                    I.vdph_sum[k][3] / st2, (I.vdph_sum[k][0] + I.vdph_sum[k][1] + I.vdph_sum[k][2] + I.vdph_sum[k][3]) / st2);
      r += d;
    }
  }
  if (I.rst_d) {  // v55e3 diagnosis: the one-token expert stage's counters (SM cycles; every one-token step since open)
    unsigned long long c[g::kRxStatN] = {};
    if (g::sync() && g::to_host(c, I.rst_d, sizeof c)) {
      const double wt = static_cast<double>(c[0]), wk = static_cast<double>(c[1]), sp = static_cast<double>(c[3]);
      const double tk = static_cast<double>(c[2]), wp = static_cast<double>(c[4]);
      char e[600];
      std::snprintf(e, sizeof e,
                    "\nexpert stage counters (one-token launches, SM cycles summed over warps): warps %llu, tasks %llu (%.1f per warp), loop span %.0f cycles "
                    "per warp: running tasks %.1f %%, waiting for a group %.1f %%, rest (job lookup, signals) %.1f %%; %.0f cycles per task; staged tasks (host "
                    "RAM over PCIe) %llu at %.0f cycles each = %.1f %% of task cycles (VRAM tasks %.0f cycles each); filling tasks %llu",
                    c[4], c[2], wp > 0 ? tk / wp : 0.0, wp > 0 ? sp / wp : 0.0, sp > 0 ? 100.0 * wk / sp : 0.0, sp > 0 ? 100.0 * wt / sp : 0.0,
                    sp > 0 ? 100.0 * (sp - wk - wt) / sp : 0.0, tk > 0 ? wk / tk : 0.0, c[5], c[5] ? static_cast<double>(c[7]) / static_cast<double>(c[5]) : 0.0,
                    wk > 0 ? 100.0 * static_cast<double>(c[7]) / wk : 0.0, tk > static_cast<double>(c[5]) ? (wk - static_cast<double>(c[7])) / (tk - static_cast<double>(c[5])) : 0.0,
                    c[6]);
      r += e;
      static const char* fn[12] = {"Q7_ZT", "Q4_TG2", "Q5_K", "Q6_K dense", "Q8_0", "Q4_K", "IQ4_XS", "Q6_K", "silu x up", "Q4_TG2 planes", "Q5_K planes", "other"};
      for (int f = 0; f < 12; ++f) {
        if (c[20 + f] == 0) continue;
        std::snprintf(e, sizeof e, "\n  %-13s tasks %10llu, %8.0f cycles per task, %5.1f %% of task cycles", fn[f], c[20 + f],
                      static_cast<double>(c[8 + f]) / static_cast<double>(c[20 + f]), wk > 0 ? 100.0 * static_cast<double>(c[8 + f]) / wk : 0.0);
        r += e;
      }
    }
  }
  return r;
}
std::string GpuEngine::mtp_path_report() const {
  const Impl& I = *p_;
  if (!I.has_mtp) return {};
  char b[200];
  std::snprintf(b, sizeof b, "MTP block MoE half (v58): %s; %llu steps on the fused launch, %llu on the old path", I.mtp_rd && I.rx_ok ? "set up on the fused launch (reader row n_layer)" : "old path",
                static_cast<unsigned long long>(I.mtp_fused_steps), static_cast<unsigned long long>(I.mtp_old_steps));
  std::string r = b;
  if (!I.mtp_why.empty()) r += std::string("; first old-path step: ") + I.mtp_why;  // v58b
  if (!I.mtp_router_note.empty()) r += "; " + I.mtp_router_note;
  return r;
}

std::string GpuEngine::graph_report() const {
  return p_->graph_note + "; captures " + std::to_string(p_->graph_captures) + ", replays " + std::to_string(p_->graph_replays) +
         "; DeltaNet dense halves in one launch (L3, v55g) " + std::to_string(p_->dn_half_runs);
}
bool GpuEngine::router_margins(std::vector<float>* m, std::vector<std::uint8_t>* have) const {
  if (!p_->rlg_h) return false;
  if (m) *m = p_->rmarg;
  if (have) *have = p_->rmarg_have;
  return true;
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
  for (std::vector<double>* v : {&t.gpu_dense, &t.gpu_hot, &t.gpu_idle, &t.cpu_wait, &t.cpu_cold, &t.cpu_picks, &t.cpu_groups}) v->assign(nl, 0.0);
}

std::string GpuEngine::summary() const {
  const Impl& I = *p_;
  char b[2048];
  std::snprintf(b, sizeof b, "engine: %.2f GB on the device (dense + output %.2f GB, bank %.2f GB = %d experts of %u, cache + state %.2f GB); %d CPU threads; ctx %d, batch %d; hand-off %s; router tie band %g; expert reader %s; batch-identical %s; graph %s; prefetch %s; split %s",
                static_cast<double>(I.dev_bytes) / 1e9, static_cast<double>(I.dense_bytes_total) / 1e9, static_cast<double>(I.bank_bytes_total) / 1e9, I.bank_count,
                I.s.n_layer * I.s.n_expert, static_cast<double>(I.cache_bytes) / 1e9, I.pool ? I.pool->threads() : 0, I.o.max_ctx, I.o.max_batch,
                I.bell ? "doorbell" : "device sync per layer", static_cast<double>(I.o.router_tie), I.reader_note.c_str(), I.o.batch_identical ? "on" : "off",
                I.graph_note.c_str(), I.pf_ok ? "on" : "off", I.split_note.c_str());
  std::string r = b;
  if (I.reader) r += I.o.gpu_prompt ? "; prompt on the GPU in steps of 1-4 tokens" : "; prompt batches on the bank + CPU cold path";  // v48
  if (I.o.r3_experts) r += "; R3 expert stage " + I.rx_note + (I.o.rx_load == 2 ? ", rows staged for every expert" : I.o.rx_load == 1 ? ", rows staged for experts read from RAM" : ", lane loads");  // v46a-1 / v46a-2
  r += I.planes_note;  // v54
  r += I.probe_note;   // v54b
  if (I.out.lean) r += "; output head held once (R3's copy; no raw copy or int8 tables)";  // v55
  r += I.r3_direct_note();  // v55
  if (I.o.fused_moe && I.rx_ok && I.fold_ok) r += "; MoE half fused (one launch per layer where no side output needs the separate kernels)";  // v52
  if (!I.o.pin_pool.empty() || I.o.pin_host >= 0) {  // v43
    char c[256];
    std::snprintf(c, sizeof c, "; pinned: %d of %zu workers, doorbell %s, host %s", I.pool ? I.pool->pinned() : 0, I.o.pin_pool.empty() ? std::size_t{0} : I.o.pin_pool.size() - 1,
                  I.o.pin_pool.empty() ? "not asked" : I.pin_bell_ok ? "ok" : "failed (or not started yet)", I.o.pin_host < 0 ? "not asked" : I.pin_host_ok ? "ok" : "failed");
    r += c;
  }
  return r;
}

Result<std::unique_ptr<GpuEngine>> GpuEngine::open(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, const EngineOptions& o) {
  g::set_fast_f32(o.fast_f32);  // v52
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
  // v55 (author 02:36-02:41, build audit A): with planes on, every product of the output head runs on R3 (one-token steps,
  // the GPU prompt's steps of 1-4 tokens on R3, no MTP block, no verify batches off R3), so the head's raw copy and its int8
  // tables (~0.48 GB for this model's Q6_K head) are not loaded; the VRAM probe gives that room to the experts
  if (o.r3 && o.plane_bits > 0 && (o.max_verify <= 1 || o.batch_identical) && o.gpu_prompt && o.prompt_r3) I.lean_w = &w.output;
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
      {reinterpret_cast<void**>(&I.dn_hcnt), z(nv > 0 ? nv : 1) * 4},
      {reinterpret_cast<void**>(&I.top_d), np * 4 + 64},
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
  // v58: the MTP block's MoE half on the reader (its row n_layer: planes / file mapping and its own cache slots, no bank)
  I.mtp_rd = o.mtp && o.mtp_fused && w.mtp.present && o.cold_on_gpu && o.r3_experts && o.rx_load >= 1 && o.fused_moe &&
             !o.cpu_light && !o.prefetch && o.plane_tail == 0;
  if (I.mtp_rd) {  // v58b: the fused launch reads the router as F32 rows; the MTP block's may be stored otherwise
    const WeightView& rv = w.mtp.layer.gate_inp;
    I.mtp_router_note = std::string("MTP router stored as ") + quant::type_info(rv.type).name + (rv.rot32 ? " (rotated)" : "");
    if (!I.router_of(I.M) && rv.present()) {
      std::vector<float> rf(z(ne) * z(e));
      bool ok = true;
      for (int64_t r = 0; r < ne && ok; ++r) {
        float* rptr = rf.data() + z(r) * z(e);
        ok = quant::dequantize_row(static_cast<quant::Type>(rv.type), rv.data + static_cast<std::uint64_t>(r) * rv.row_bytes, rptr, e).ok();
        if (ok && rv.rot32) {
          quant::hadamard32(rptr, e);
        }
      }
      void* rd = ok ? g::dalloc(rf.size() * sizeof(float)) : nullptr;
      if (rd && g::to_device(rd, rf.data(), rf.size() * sizeof(float))) {
        I.M.router_f32 = static_cast<const float*>(rd);
        I.dev_bytes += rf.size() * sizeof(float);
        left = left > rf.size() * sizeof(float) ? left - rf.size() * sizeof(float) : 0;
        I.mtp_router_note += ", an F32 copy made for the fused launch (decoded from the stored rows; draft picks only: verify still decides every token)";
      } else {
        I.mtp_router_note += ", no F32 copy (dequantize or device memory)";
      }
    }
  }
  if (o.mtp && o.mtp_bank != 0 && !I.mtp_rd) {  // the MTP block's fixed arena, for promotion
    const Qwen35MoeLayer& ml = w.mtp.layer;
    std::uint64_t eb = 0;
    int bankable_n = 0;
    for (int ex = 0; ex < static_cast<int>(ne); ++ex)
      if (I.bankable(ml, ex)) {
        eb = std::max<std::uint64_t>(eb, I.expert_bytes(ml, ex, false));
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
  I.rc_eff = o.reader_cache;
  int hot_cap = o.max_hot_per_layer;
  if (o.cold_on_gpu && (o.reader_cache > 0 || o.auto_vram)) {  // v26: the reader cache's bytes before the bank takes its share
    std::uint64_t mx = 0;
    for (std::uint32_t il = 0; il < I.nlr(); ++il)  // v58: with the MTP row too
      for (int ex = 0; ex < static_cast<int>(ne); ++ex) {
        std::uint64_t b = 0;
        for (const WeightView* v : {&I.WL(il).gate_exps, &I.WL(il).up_exps, &I.WL(il).down_exps}) b += exp_rows(*v, ex).rb * v->ne[1];
        mx = std::max(mx, b);
      }
    const std::uint64_t slot_b = (mx + 255) & ~std::uint64_t{255};
    if (o.auto_vram) {  // v54b VRAM probe: the free bytes after the dense weights split cache_share / the rest
      const std::uint64_t slack = (3 * 4 * (2048 + 256 * 4) + 64 * 1024) * static_cast<std::uint64_t>(s.n_layer);
      const std::uint64_t ex_b = left > slack ? left - slack : 0;
      // the cache share, capped so bank + cache never ask for more than every expert's bytes (a model whose experts all fit
      // in the bank gets no cache)
      std::uint64_t all_b = 0;
      for (std::uint32_t il = 0; il < s.n_layer; ++il)
        for (int ex = 0; ex < static_cast<int>(ne); ++ex) all_b += I.expert_bytes(w.layers[il], ex);
      double cache_b = static_cast<double>(ex_b) * o.cache_share;
      if (static_cast<double>(all_b) <= static_cast<double>(ex_b)) cache_b = 0.0;
      else cache_b = std::min(cache_b, static_cast<double>(all_b) - static_cast<double>(ex_b) * (1.0 - o.cache_share));
      I.rc_eff = static_cast<int>(cache_b / static_cast<double>(slot_b * I.nlr()));
      hot_cap = static_cast<int>(ne);
      char pn[240];
      std::snprintf(pn, sizeof pn, "; VRAM probe: %.2f GB free after the dense weights (reserve %.2f GB kept), %.2f GB for experts: cache %.1f %% = %d slots of %.2f MB per layer, the bank the rest",
                    static_cast<double>(free_b) / 1e9, static_cast<double>(o.vram_reserve) / 1e9, static_cast<double>(ex_b) / 1e9, 100.0 * o.cache_share, I.rc_eff,
                    static_cast<double>(slot_b) / 1e6);
      I.probe_note = pn;
    }
    const std::uint64_t need = slot_b * static_cast<std::uint64_t>(I.rc_eff) * I.nlr();
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
    const std::size_t cap = std::min<std::size_t>(max_rank, static_cast<std::size_t>(std::max(0, hot_cap)));  // v54b: the probe lifts the cap
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
      const Status st = o.plane_bits > 0 ? I.build_bank_planes(w.layers[il], I.L[il]) : I.build_bank(w.layers[il], I.L[il]);  // v54c
      if (!st) return R::fail(st);
    }
  }
  if (o.plane_bits != 0) {  // v54b: only the R3 expert stage reads the planes; misses fill their slots through staging
    if (o.plane_bits < 2 || o.plane_bits > 5) return R::fail(Status::fail(Code::invalid_argument, "engine: --plane-bits 2, 3, 4 or 5 (5: every code as stored)"));
    if (!o.cold_on_gpu || !o.r3_experts || o.rx_load < 1 || o.prefetch || o.cpu_light || o.plane_tail != 0 || !o.gpu_prompt)
      return R::fail(Status::fail(Code::invalid_argument,
                                  "engine: --plane-bits needs the R3 expert stage with staged loads (--cold-on-gpu --r3-experts --rx-load miss|all), "
                                  "no --prefetch, --cpu-light, --plane-tail or --cpu-prompt"));
  }
  if (o.cold_on_gpu) I.setup_reader();  // v24
  if (o.plane_bits > 0 && !I.rx_ok)  // (no plane matrices at all is fine: every expert in the bank)
    return R::fail(Status::fail(Code::internal, "engine: the bit planes need the R3 expert stage on (" + I.rx_note + "; reader " + I.reader_note + ")"));
  // the host side of the cold path
  const int threads = o.cpu_threads > 0 ? o.cpu_threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  {  // v43: workers take pin_pool[1..]; the doorbell thread takes pin_pool[0] when it starts; the driving thread pin_host
    std::vector<int> wk;
    for (std::size_t q2 = 1; q2 < o.pin_pool.size(); ++q2) wk.push_back(o.pin_pool[q2]);
    I.pool = std::make_unique<Pool>(threads, wk);
    I.pool->set_pause(o.spin_pause);  // v45b
    if (o.pin_host >= 0) I.pin_host_ok = hal::pin_this_thread(o.pin_host);
  }
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
      if (o.cpu_check) {  // v45b
        void* xdev = nullptr;
        I.xr_h = g::host_alloc_mapped(xn_b, &xdev);
        I.xr_m = static_cast<float*>(I.xr_h);
      }
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
      for (std::vector<double>* v : {&I.tline.gpu_dense, &I.tline.gpu_hot, &I.tline.gpu_idle, &I.tline.cpu_wait, &I.tline.cpu_cold, &I.tline.cpu_picks, &I.tline.cpu_groups}) v->assign(s.n_layer, 0.0);
      Impl* ip = &I;
      I.bell_th = std::thread([ip] {
        if (!ip->o.pin_pool.empty()) ip->pin_bell_ok = hal::pin_this_thread(ip->o.pin_pool[0]);
        ip->bell_loop();
      });
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
  if (!(I.reader && I.o.gpu_prompt && n > Impl::kReaderMax)) return I.run(tokens, n, logits, last_only, stats, hidden_out);
  // v48: the prompt as steps of up to kReaderMax tokens on the GPU-only path; each step's rows land where the one batch
  // would have put them (logits row c0 + r, hidden row c0 + r); with last_only only the final step runs the head
  if (n > I.o.max_batch || !tokens || !logits) return Status::fail(Code::invalid_argument, "engine: 1 .. max_batch tokens and an output");
  const std::int64_t nvoc = I.s.n_vocab, e = I.s.n_embd;
  EngineStats sum;
  for (int c0 = 0; c0 < n; c0 += Impl::kReaderMax) {
    const int m = std::min(Impl::kReaderMax, n - c0);
    const bool last = c0 + m == n;
    EngineStats es;
    I.no_head = last_only && !last;
    I.in_prompt = true;
    const Status st = I.run(tokens + c0, m, last_only ? logits : logits + static_cast<std::int64_t>(c0) * nvoc, last_only, &es,
                            hidden_out ? hidden_out + static_cast<std::int64_t>(c0) * e : nullptr);
    I.no_head = false;
    I.in_prompt = false;
    if (!st) return st;
    sum.seconds += es.seconds;
    sum.cold_seconds += es.cold_seconds;
    sum.wait_seconds += es.wait_seconds;
    sum.hot_picks += es.hot_picks;
    sum.cold_picks += es.cold_picks;
    sum.cold_bytes += es.cold_bytes;
  }
  if (stats) *stats = sum;
  return {};
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
  I.bi_now = rd && (I.o.batch_identical || (I.in_prompt && I.o.prompt_r3));  // v26 option B; v48b: the GPU prompt's steps too (MEASURED v48: 99.0 vs 49.9 tok/s)
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
    bool dnh = false;  // v55g (L3): the DeltaNet dense half in one launch (1-4 tokens since v56, no per-op stamps)
    if (l.recurrent && n <= 4 && !I.cur_ops && !(st = I.dn_half_layer(D, n, &dnh))) return st;
    if (!dnh) {
    if (!I.opstamp(0)) return gfail("engine: stamp");
    if (!g::rms_norm(I.x, D.attn_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: attn norm");
    if (!I.opstamp(1)) return gfail("engine: stamp");
    if (!(st = l.recurrent ? I.deltanet_block(D, n) : I.attention_block(D, n, I.pos))) return st;
    }
    const bool probe_row = I.probe_pos >= I.pos && I.probe_pos < I.pos + n && I.probe_bd;
    if (probe_row && !g::dcopy(I.probe_bd + static_cast<int64_t>(il) * e, I.y + (I.probe_pos - I.pos) * e, z(e) * sizeof(float))) return gfail("engine: probe copy");
    const bool fused = rd && I.fused_ok(D, n, probe_row);  // v52: residual + MoE half + residual in one launch
    if (fused) {
      if (!(st = I.moe_half_fused(D, n, il))) return st;
    } else {
    if (!g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("engine: residual");
    if (!g::rms_norm(I.x, D.post_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: post norm");
    if (!I.opstamp(7)) return gfail("engine: stamp");
    if (!(st = rd ? I.moe_block_reader(D, n, il) : bl ? I.moe_block_bell(D, n, g0 + il, I.ts_d ? I.ts_d + 4 * il : nullptr) : I.moe_block(l, D, n, es))) return st;
    }
    if (probe_row) {
      const int64_t pr = I.probe_pos - I.pos, ne2 = s.n_expert;
      if (!g::dcopy(I.probe_md + static_cast<int64_t>(il) * e, I.y + pr * e, z(e) * sizeof(float)) ||
          !g::dcopy(I.probe_ld + static_cast<int64_t>(il) * ne2, I.logits_r + pr * ne2, z(ne2) * sizeof(float)))
        return gfail("engine: probe copy");
    }
    if (!fused && !g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("engine: residual");
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
    const std::tuple<int, int, int, int> key{n, parity, I.ckpt_mode && n > 1 ? 1 : (I.bi_now && n > 1 ? 2 : 0), (I.ops_d && n == 1) ? 1 : 0};  // v48b: R3 (bi) and batched steps of the same size are different graphs
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
  const int r_begin = I.no_head ? n : last_only ? n - 1 : 0;  // v48: no_head runs no row through the head
  for (int r0 = hidden_out ? 0 : r_begin; r0 < n; r0 += Impl::kOutRows) {
    const int m = std::min(Impl::kOutRows, n - r0);
    if (!g::rms_norm(I.x + static_cast<int64_t>(r0) * e, I.out_norm, I.xn, e, 1, e, e, m, s.rms_eps)) return gfail("engine: output norm");
    if (hidden_out && !g::to_host(hidden_out + static_cast<int64_t>(r0) * e, I.xn, z(m) * z(e) * sizeof(float))) return gfail("engine: hidden");
    if (r0 + m <= r_begin) continue;
    // rows before r_begin (asked for the hidden only) are run through the head with the rest of their chunk; only rows
    // from r_begin on are copied out
    if (!(st = I.prod(I.out, I.xn, m, I.lg))) return st;
    const int c0 = std::max(r0, r_begin);
    if (I.top_req) {  // v56: the argmax on the device
      if (!g::argmax_rows(I.lg + static_cast<int64_t>(c0 - r0) * nvoc, nvoc, r0 + m - c0, nvoc, I.top_d + (c0 - r_begin))) return gfail("engine: argmax");
    } else if (!g::to_host(logits + static_cast<int64_t>(c0 - r_begin) * nvoc, I.lg + static_cast<int64_t>(c0 - r0) * nvoc, z(r0 + m - c0) * z(nvoc) * sizeof(float))) {
      return gfail("engine: logits");
    }
  }
  if (I.top_req && n > r_begin && !g::to_host(I.top_req, I.top_d, z(n - r_begin) * sizeof(std::int32_t))) return gfail("engine: argmax out");
  if (!g::sync()) return gfail("engine: sync");
  if (rd) {  // v25: every pick of a reader step runs on the GPU
    es.hot_picks += static_cast<std::uint64_t>(n) * s.n_expert_used;
    if (I.trc_ts_h) I.trace_step(n, since(t0));
    // v55b / v56: the fused halves' stamps of this step (layers that ran one fill stamp 0): decode steps into ph / dph,
    // verify steps of 2-4 tokens into vph / vdph by size; prompt steps are not counted (stamps cleared)
    auto take = [&](unsigned long long* base, double* acc, std::uint64_t* nlay) {
      bool any = false;
      for (std::uint32_t il = 0; il < s.n_layer; ++il) {
        unsigned long long* pq = base + static_cast<int64_t>(il) * 5;
        if (pq[0] == 0 || pq[4] < pq[0]) {
          pq[0] = 0;
          continue;
        }
        if (acc)
          for (int k = 0; k < 4; ++k) acc[k] += static_cast<double>(pq[k + 1] - pq[k]) * 1e-6;
        if (nlay) ++*nlay;
        pq[0] = 0;
        any = true;
      }
      return any;
    };
    const bool dec = !I.in_prompt && n >= 1 && n <= 4;
    if (I.ph_h) {
      const bool any = take(I.ph_h, dec ? (n == 1 ? I.ph_sum : I.vph_sum[n - 2]) : nullptr, dec && n == 1 ? &I.ph_layers : nullptr);
      if (any && dec) ++(n == 1 ? I.ph_steps : I.vph_steps[n - 2]);
    }
    if (I.dph_h) {
      const bool any = take(I.dph_h, dec ? (n == 1 ? I.dph_sum : I.vdph_sum[n - 2]) : nullptr, dec && n == 1 ? &I.dph_layers : nullptr);
      if (any && dec) ++(n == 1 ? I.dph_steps : I.vdph_steps[n - 2]);
    }
    if (I.rlg_h) {  // v49: this step's router margins, per token and layer (8th - 9th logit, 1st - 2nd)
      const int64_t ne2 = s.n_expert, nu2 = s.n_expert_used, nl = s.n_layer;
      std::vector<float> row(z(ne2));
      for (int t = 0; t < n; ++t) {
        const int64_t p = I.pos + t;  // pos advances after this (I.pos += n below)
        if (p < 0 || p >= I.o.max_ctx) continue;
        for (int64_t il = 0; il < nl; ++il) {
          const float* lr = I.rlg_h + (il * Impl::kReaderMax + t) * ne2;
          row.assign(lr, lr + ne2);
          std::partial_sort(row.begin(), row.begin() + nu2 + 1, row.end(), std::greater<float>());
          I.rmarg[z(p) * 2 * z(nl) + 2 * z(il)] = row[z(nu2 - 1)] - row[z(nu2)];
          I.rmarg[z(p) * 2 * z(nl) + 2 * z(il) + 1] = row[0] - row[1];
        }
        I.rmarg_have[z(p)] = 1;
      }
    }
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
  const bool prev_bi = I.bi_now;
  if (n <= Impl::kReaderMax && (I.o.batch_identical || I.o.prompt_r3)) I.bi_now = true;
  struct MtpBiReset { bool& f; bool v; ~MtpBiReset() { f = v; } } mtp_bi_reset{I.bi_now, prev_bi};
  Layer& D = I.M;
  if (!g::rms_norm(I.x, D.attn_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: MTP attn norm");
  if (!(st = I.attention_block(D, n, I.mtp_pos))) return st;
  if (I.mtp_rd && I.fused_ok(D, n, false)) {  // v58: residual + MoE half + residual in one launch, the reader's row n_layer
    if (!(st = I.moe_half_fused(D, n, s.n_layer))) return st;
    ++I.mtp_fused_steps;
  } else {
    if (I.mtp_rd && I.mtp_why.empty()) {
      const char* why = I.fused_why(D, n, false);
      I.mtp_why = why ? why : "?";
    }
    if (!g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("engine: MTP residual");
    if (!g::rms_norm(I.x, D.post_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: MTP post norm");
    if (!(st = I.moe_block(w.mtp.layer, D, n, es))) return st;
    if (!g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("engine: MTP residual");
    ++I.mtp_old_steps;
  }
  const int64_t nvoc = s.n_vocab;
  for (int r0 = 0; r0 < n; r0 += Impl::kOutRows) {
    const int m = std::min(Impl::kOutRows, n - r0);
    if (!g::rms_norm(I.x + static_cast<int64_t>(r0) * e, I.head_norm, I.xn, e, 1, e, e, m, s.rms_eps)) return gfail("engine: MTP head norm");
    if (h_out && !g::to_host(h_out + static_cast<int64_t>(r0) * e, I.xn, z(m) * z(e) * sizeof(float))) return gfail("engine: MTP hidden");
    if (logits) {
      if (!(st = I.prod(I.head, I.xn, m, I.lg))) return st;
      if (I.top_req) {  // v56
        if (!g::argmax_rows(I.lg, nvoc, m, nvoc, I.top_d + r0)) return gfail("engine: MTP argmax");
      } else if (!g::to_host(logits + static_cast<int64_t>(r0) * nvoc, I.lg, z(m) * z(nvoc) * sizeof(float))) {
        return gfail("engine: MTP logits");
      }
    }
  }
  if (I.top_req && logits && !g::to_host(I.top_req, I.top_d, z(n) * sizeof(std::int32_t))) return gfail("engine: MTP argmax out");
  if (!g::sync()) return gfail("engine: MTP sync");
  I.mtp_pos += n;
  es.seconds = since(t0);
  if (stats) *stats = es;
  return Status::good();
}

// v56: the argmax variants (Impl::top_req routes the head's rows to the device argmax; the logits pointer is never written)
namespace {
struct TopReq {
  std::int32_t*& r;
  TopReq(std::int32_t*& slot, std::int32_t* v) : r(slot) { r = v; }
  ~TopReq() { r = nullptr; }
};
}  // namespace
Status GpuEngine::step_top(const std::int32_t* tokens, int n, std::int32_t* top, bool last_only, EngineStats* st, float* hidden_out) {
  if (!top) return Status::fail(Code::invalid_argument, "engine: step_top needs an output");
  TopReq g{p_->top_req, top};
  return step(tokens, n, reinterpret_cast<float*>(top), last_only, st, hidden_out);
}
Status GpuEngine::verify_top(const std::int32_t* tokens, int n, std::int32_t* top, float* hidden_out, EngineStats* st) {
  if (!top) return Status::fail(Code::invalid_argument, "engine: verify_top needs an output");
  TopReq g{p_->top_req, top};
  return verify(tokens, n, reinterpret_cast<float*>(top), hidden_out, st);
}
Status GpuEngine::mtp_step_top(const std::int32_t* tokens, const float* h_prev, int n, std::int32_t* top, float* h_out, EngineStats* st) {
  if (!top) return Status::fail(Code::invalid_argument, "engine: mtp_step_top needs an output");
  TopReq g{p_->top_req, top};
  return mtp_step(tokens, h_prev, n, reinterpret_cast<float*>(top), h_out, st);
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
    const std::uint64_t b = I.expert_bytes(ml, e, false);
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
  if (n_draft > 0) {  // v58c: catch up in 4-token R3 fused steps so 0 picks go to the slow CPU path
    std::vector<float> hp(z(P * e), 0.0f);
    std::copy(hid.begin(), hid.end() - e, hp.begin() + e);
    const int kMtpChunk = 4;
    for (int64_t t0 = 0; t0 < P; t0 += kMtpChunk) {
      const int k = static_cast<int>(std::min<int64_t>(kMtpChunk, P - t0));
      EngineStats x;
      if (Status r = eng.mtp_step(prompt.data() + t0, hp.data() + t0 * e, k, nullptr, nullptr, &x); !r) return r;
      add_stats(tot, x);
    }
    if (auto pr = eng.mtp_promote(); !pr.ok()) return pr.status;
  }
  T.prompt_s += secs(tp0);
  std::int32_t last = argmax_of(lg.data(), nv);
  out.push_back(last);
  if (on_token) on_token(last);
  std::vector<float> hn(z(e)), h(z(e)), bh;
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
        std::int32_t nt = 0;  // v56: the draft's argmax on the device (one int to the host instead of n_vocab floats)
        if (Status r = eng.mtp_step_top(&tok, h.data(), 1, &nt, hn.data(), &x); !r) return r;
        T.draft_s += secs(tq);
        ++T.drafts;
        add_stats(tot, x);
        tok = nt;
        drafts.push_back(tok);
        h = hn;
      }
      // v58c: draft[0] at position m0 used exact (last, pending), so m0 + 1 is ALREADY valid in the MTP KV cache!
      eng.mtp_truncate(m0 + 1);
      stt.drafted += k;
    }
    const int nb = 1 + k;
    std::vector<std::int32_t> batch{last};
    batch.insert(batch.end(), drafts.begin(), drafts.end());
    std::vector<std::int32_t> tops(z(nb));
    bh.resize(z(nb * e));
    EngineStats x;
    const Clock::time_point tv = Clock::now();
    if (Status r = k > 0 ? eng.verify_top(batch.data(), nb, tops.data(), bh.data(), &x) : eng.step_top(batch.data(), 1, tops.data(), true, &x, bh.data()); !r)
      return r;
    T.verify_s[nb > 8 ? 9 : nb] += secs(tv);
    ++T.verify_n[nb > 8 ? 9 : nb];
    add_stats(tot, x);
    const Clock::time_point ta2 = Clock::now();
    int a = 0;
    while (a < k && tops[z(a)] == drafts[z(a)] && !is_stop(drafts[z(a)])) ++a;
    if (a < k && is_stop(drafts[z(a)]) && tops[z(a)] == drafts[z(a)]) ++a;
    const std::int32_t next = tops[z(a)];
    T.argmax_s += secs(ta2);
    const Clock::time_point tk = Clock::now();
    if (k > 0)
      if (Status r = eng.keep(a + 1); !r) return r;
    T.keep_s += secs(tk);
    stt.accepted += a;
    if (n_draft > 0) {
      if (k == 0) {
        EngineStats y;
        const Clock::time_point tc = Clock::now();
        if (Status r = eng.mtp_step(batch.data(), pending.data(), 1, nullptr, nullptr, &y); !r) return r;
        T.catchup_s += secs(tc);
        add_stats(tot, y);
      } else if (a > 0) {
        EngineStats y;
        const Clock::time_point tc = Clock::now();
        if (Status r = eng.mtp_step(batch.data() + 1, bh.data(), a, nullptr, nullptr, &y); !r) return r;
        T.catchup_s += secs(tc);
        add_stats(tot, y);
      }
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

Result<std::vector<std::vector<int>>> hot_order_from_routes(const std::string& path, std::uint32_t n_layer, std::uint32_t n_expert,
                                                            std::vector<std::vector<std::uint64_t>>* counts) {
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
  if (counts) *counts = std::move(cnt);  // v51
  return R::good(std::move(order));
}

}  // namespace pelias::model
