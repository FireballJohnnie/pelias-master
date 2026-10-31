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

#include "kern/cpu_dot.h"
#include "kern/gpu_ops.h"
#include "kern/gpu_prompt.h"
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
  void run(int64_t n, const std::function<void(int64_t)>& fn) {
    if (n <= 0) return;
    if (th_.empty() || n == 1) {
      for (int64_t i = 0; i < n; ++i) fn(i);
      return;
    }
    fn_ = &fn;
    n_ = n;
    next_.store(0, std::memory_order_relaxed);
    active_.store(static_cast<int>(th_.size()), std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(mu_);  // a sleeping worker sees the new generation under the lock
      gen_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_all();
    work(fn, n);
    for (int spin = 0; active_.load(std::memory_order_acquire) != 0; ++spin)
      if (spin > kSpin) std::this_thread::yield();
  }

 private:
  static constexpr int kSpin = 4000;
  void work(const std::function<void(int64_t)>& fn, int64_t n) {
    for (int64_t i = next_.fetch_add(1, std::memory_order_relaxed); i < n; i = next_.fetch_add(1, std::memory_order_relaxed)) fn(i);
  }
  void loop() {
    std::uint64_t seen = 0;
    for (;;) {
      bool got = false;
      for (int spin = 0; spin < kSpin && !got; ++spin) got = gen_.load(std::memory_order_acquire) != seen || stop_.load(std::memory_order_acquire);
      if (!got) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_.load(std::memory_order_acquire) || gen_.load(std::memory_order_acquire) != seen; });
      }
      if (stop_.load(std::memory_order_acquire)) return;
      seen = gen_.load(std::memory_order_acquire);
      work(*fn_, n_);
      active_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }
  std::vector<std::thread> th_;
  std::mutex mu_;
  std::condition_variable cv_;
  const std::function<void(int64_t)>* fn_ = nullptr;
  int64_t n_ = 0;
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
};

struct Layer {
  DevW wq, wk, wv, wo, wqkv, wgate, wbeta, walpha, ssm_out, gate_inp, gate_sh, up_sh, down_sh;
  const float *attn_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *conv1d = nullptr, *ssm_dt = nullptr,
              *ssm_a = nullptr, *ssm_norm = nullptr, *gate_inp_sh = nullptr;
  // the bank: typed stacks per projection (each with its expert map), and which experts it holds
  std::vector<g::PromptWeights> bg, bu, bd;
  std::vector<std::uint8_t> hot;  // n_expert flags
  int n_hot = 0;
  void* bank_mem = nullptr;
  // state
  float *K = nullptr, *V = nullptr;          // full attention: max_ctx x kv_w each
  float* state = nullptr;                    // DeltaNet: nv x ds x ds
  float* tail[2] = {nullptr, nullptr};       // DeltaNet: (taps - 1) x ch, ping-pong
  int cur = 0;
  float* ckpt = nullptr;        // DeltaNet, verify: the state after each of the first max_verify - 1 tokens
  float* mixed_save = nullptr;  // DeltaNet, verify: the batch's conv inputs (max_verify x ch), to rebuild the tail of a prefix
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
  static constexpr int kOutRows = 64;
  // host side of the cold path (sized once)
  std::unique_ptr<Pool> pool;
  std::vector<float> xn_h, ycold_h, wsel_h, xe_h;
  std::vector<std::int32_t> sel_h;
  std::vector<kern::ActQ8x32> actx, acth;
  std::vector<float> gbuf, ubuf, dbuf;
  std::vector<ColdPick> cold;
  std::vector<std::uint8_t> need;
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
  static std::size_t dense_bytes(const WeightView& v) {
    if (!v.present()) return 0;
    const int64_t n_out = static_cast<int64_t>(v.ne[1]), k = static_cast<int64_t>(v.ne[0]);
    std::size_t b = al(view_bytes(v)) + 256;
    if (int8_ok(v.type, n_out, k)) b += al(g::prompt_table_bytes(v.type, n_out, k)) + 256;
    return b;
  }
  static std::size_t vec_bytes(const std::vector<float>& f) { return f.empty() ? 0 : al(f.size() * sizeof(float)) + 256; }
  static std::size_t layer_dense_bytes(const Qwen35MoeLayer& l) {
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
    const std::size_t bytes = view_bytes(v);
    void* d = arena.take(bytes);
    if (!d || !g::to_device(d, v.data, bytes)) return Status::fail(Code::internal, "engine: weight upload " + v.name);
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
    if (need_b == 0) return Status::good();
    D.bank_mem = g::dalloc(need_b);
    if (!D.bank_mem) return gfail("engine: bank memory");
    dev_bytes += need_b;
    bank_bytes_total += need_b;
    Arena A{static_cast<std::uint8_t*>(D.bank_mem), need_b, 0};
    auto stacks = [&](const WeightView& v, std::vector<g::PromptWeights>& dst) -> Status {
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
    if (!(st = stacks(l.gate_exps, D.bg)) || !(st = stacks(l.up_exps, D.bu)) || !(st = stacks(l.down_exps, D.bd))) return st;
    if (D.bg.size() > 4 || D.bu.size() > 4 || D.bd.size() > 4) return Status::fail(Code::unsupported, "engine: more than 4 expert types in a layer's bank");
    return Status::good();
  }

  // ---- products: as GpuForward::prod (quantized weights round their input to gpu256x8 first)
  Status prod(const DevW& W, const float* xin, int64_t n, float* yout) {
    const WeightView& v = *W.v;
    if (is_float_type(v.type)) {
      if (v.rot32) return Status::fail(Code::unsupported, "engine: a rotated float weight (" + v.name + ")");
      if (!g::matmul(v.type, W.d, v.row_bytes, W.n_out, W.k, xin, n, W.k, yout, W.n_out)) return gfail("engine: product " + v.name);
      return Status::good();
    }
    if (!g::act_q8x32(xin, W.k, n, W.k, v.rot32, xr, W.k, q, scale, sum)) return gfail("engine: rounding for " + v.name);
    if (W.int8) {
      if (!g::act_digits(scale, sum, n, W.k, dig) || !g::prompt_gemm(W.pw, W.n_out, q, scale, dig, n, yout)) return gfail("engine: int8 product " + v.name);
      return Status::good();
    }
    if (!g::matmul(v.type, W.d, v.row_bytes, W.n_out, W.k, xr, n, W.k, yout, W.n_out)) return gfail("engine: product " + v.name);
    return Status::good();
  }

  Status attention_block(Layer& D, int64_t n, int64_t p0) {
    const int64_t hd = s.head_dim, nh = s.n_head, nkv = s.n_head_kv;
    const int64_t qh = s.attn_gate ? 2 * hd : hd, q_w = qh * nh;
    float* kn = D.K + p0 * kv_w;  // the new keys and values go straight into the cache
    float* vn = D.V + p0 * kv_w;
    Status st;
    if (!(st = prod(D.wq, xn, n, qg)) || !(st = prod(D.wk, xn, n, kn)) || !(st = prod(D.wv, xn, n, vn))) return st;
    if (D.q_norm && !g::rms_norm(qg, D.q_norm, qg, hd, nh, qh, q_w, n, s.rms_eps)) return gfail("engine: q norm");
    if (D.k_norm && !g::rms_norm(kn, D.k_norm, kn, hd, nkv, hd, kv_w, n, s.rms_eps)) return gfail("engine: k norm");
    const int sec[4] = {s.rope_sections[0], s.rope_sections[1], s.rope_sections[2], s.rope_sections[3]};
    if (!g::rope(qg, nh, qh, q_w, n, p0, static_cast<int>(s.n_rot), s.rope_freq_base, sec, s.rope_neox, rope_ff) ||
        !g::rope(kn, nkv, hd, kv_w, n, p0, static_cast<int>(s.n_rot), s.rope_freq_base, sec, s.rope_neox, rope_ff))
      return gfail("engine: rope");
    const float scl = s.attn_scale != 0.0f ? s.attn_scale : 1.0f / std::sqrt(static_cast<float>(hd));
    if (!g::attention(qg, q_w, qh, D.K, D.V, kv_w, nh, nkv, hd, n, p0, scl, s.attn_gate, att, ascr)) return gfail("engine: attention");
    return prod(D.wo, att, n, y);
  }

  Status deltanet_block(Layer& D, int64_t n) {
    const int64_t ch = s.conv_channels(), ds = s.ssm_d_state, nk = s.ssm_n_group, nv = s.ssm_dt_rank, taps = s.ssm_d_conv;
    Status st;
    if (!(st = prod(D.wqkv, xn, n, mixed)) || !(st = prod(D.wgate, xn, n, zg)) || !(st = prod(D.wbeta, xn, n, beta)) || !(st = prod(D.walpha, xn, n, alpha)))
      return st;
    if (!g::deltanet_gates(beta, alpha, D.ssm_dt, D.ssm_a, n, nv)) return gfail("engine: deltanet gates");
    const bool ck = ckpt_mode && n > 1;
    if (ck && !g::dcopy(D.mixed_save, mixed, z(n * ch) * sizeof(float))) return gfail("engine: verify inputs");
    if (!g::deltanet_conv(mixed, D.tail[D.cur], D.conv1d, n, ch, taps, conv, D.tail[1 - D.cur])) return gfail("engine: deltanet conv");
    D.cur = 1 - D.cur;
    if (!g::l2_norm(conv, ds, 2 * nk, ds, ch, n, s.rms_eps)) return gfail("engine: deltanet l2 norm");
    const bool pow2 = ds >= 32 && ds <= 1024 && (ds & (ds - 1)) == 0;
    const bool ok = ck ? g::deltanet_recurrent_reg_ckpt(conv, ch, beta, alpha, D.state, n, nk, nv, ds, core, D.ckpt)
                       : pow2 ? g::deltanet_recurrent_reg(conv, ch, beta, alpha, D.state, n, nk, nv, ds, core)
                              : g::deltanet_recurrent(conv, ch, beta, alpha, D.state, n, nk, nv, ds, core);
    if (!ok) return gfail("engine: deltanet recurrence");
    if (!g::gated_norm(core, zg, D.ssm_norm, n, nv, ds, s.rms_eps)) return gfail("engine: gated norm");
    return prod(D.ssm_out, core, n, y);
  }

  // the cold picks on the CPU: ycold_h (n x e) = sum over cold picks of weight x down(round32(silu(gate x) * up x))
  void cold_experts(const Qwen35MoeLayer& l, int64_t n) {
    const int64_t e = s.n_embd, ff = s.n_ff_exp;
    const int64_t np = static_cast<int64_t>(cold.size());
    std::fill(need.begin(), need.begin() + n, std::uint8_t{0});
    for (const ColdPick& c : cold) need[z(c.t)] = 1;
    pool->run(n, [&](int64_t t) {
      if (need[z(t)]) kern::act_q8x32_quantize(xn_h.data() + t * e, e, actx[z(t)]);
    });
    constexpr int64_t kRowsUp = 64, kRowsDown = 128;
    const int64_t ub = (ff + kRowsUp - 1) / kRowsUp;  // row blocks per projection (gate, up)
    pool->run(np * 2 * ub, [&](int64_t u) {
      const int64_t p = u / (2 * ub), rem = u % (2 * ub), proj = rem / ub, r0 = (rem % ub) * kRowsUp, r1 = std::min(ff, r0 + kRowsUp);
      const ColdPick& c = cold[z(p)];
      const ExpRows R = exp_rows(proj == 0 ? l.gate_exps : l.up_exps, c.e);
      float* o_row = (proj == 0 ? gbuf.data() : ubuf.data()) + p * ff;
      const kern::ActQ8x32* a = &actx[z(c.t)];
      for (int64_t r = r0; r < r1; ++r) kern::dot_row_q8(R.t, R.base + static_cast<std::uint64_t>(r) * R.rb, e, nullptr, &a, 1, o_row + r);
    });
    pool->run(np, [&](int64_t p) {
      float* gg = gbuf.data() + p * ff;
      const float* uu = ubuf.data() + p * ff;
      for (int64_t i = 0; i < ff; ++i) gg[i] = silu(gg[i]) * uu[i];
      kern::act_q8x32_quantize(gg, ff, acth[z(p)]);
    });
    const int64_t db = (e + kRowsDown - 1) / kRowsDown;
    pool->run(np * db, [&](int64_t u) {
      const int64_t p = u / db, r0 = (u % db) * kRowsDown, r1 = std::min(e, r0 + kRowsDown);
      const ExpRows R = exp_rows(l.down_exps, cold[z(p)].e);
      const kern::ActQ8x32* a = &acth[z(p)];
      float* o_row = dbuf.data() + p * e;
      for (int64_t r = r0; r < r1; ++r) kern::dot_row_q8(R.t, R.base + static_cast<std::uint64_t>(r) * R.rb, ff, nullptr, &a, 1, o_row + r);
    });
    // per token, the picks in slot order (deterministic)
    pool->run(n, [&](int64_t t) {
      float* yt = ycold_h.data() + t * e;
      std::fill(yt, yt + e, 0.0f);
      for (int64_t p = 0; p < np; ++p) {
        if (cold[z(p)].t != t) continue;
        const float wgt = cold[z(p)].w;
        const float* d = dbuf.data() + p * e;
        for (int64_t i = 0; i < e; ++i) yt[i] += wgt * d[i];
      }
    });
  }

  Status moe_block(const Qwen35MoeLayer& l, Layer& D, int64_t n, EngineStats& es) {
    const int64_t e = s.n_embd, ne = s.n_expert, nu = s.n_expert_used, ffs = s.n_ff_shexp;
    Status st;
    if (!(st = prod(D.gate_inp, xn, n, logits_r))) return st;
    if (!g::router_topk(logits_r, ne, n, static_cast<int>(nu), s.expert_weights_scale, sel, wsel)) return gfail("engine: router");
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
    if (!(st = prod(D.gate_sh, xn, n, gs)) || !(st = prod(D.up_sh, xn, n, us))) return st;
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

GpuEngine::~GpuEngine() {
  if (!p_) return;
  g::moe_scratch_free(&p_->moe);
  for (Layer& D : p_->L) g::dfree(D.bank_mem);
  g::dfree(p_->act_mem);
  g::dfree(p_->dense_mem);
  g::dfree(p_->state_mem);
}

std::int64_t GpuEngine::pos() const { return p_ ? p_->pos : 0; }
std::uint64_t GpuEngine::device_bytes() const { return p_ ? p_->dev_bytes : 0; }
int GpuEngine::bank_experts() const { return p_ ? p_->bank_count : 0; }
int GpuEngine::max_batch() const { return p_ ? p_->o.max_batch : 0; }
std::uint64_t GpuEngine::bank_bytes() const { return p_ ? p_->bank_bytes_total : 0; }
Status GpuEngine::reset() { return p_->reset_state(); }

std::string GpuEngine::summary() const {
  const Impl& I = *p_;
  char b[512];
  std::snprintf(b, sizeof b, "engine: %.2f GB on the device (dense + output %.2f GB, bank %.2f GB = %d experts of %u, cache + state %.2f GB); %d CPU threads; ctx %d, batch %d",
                static_cast<double>(I.dev_bytes) / 1e9, static_cast<double>(I.dense_bytes_total) / 1e9, static_cast<double>(I.bank_bytes_total) / 1e9, I.bank_count,
                I.s.n_layer * I.s.n_expert, static_cast<double>(I.cache_bytes) / 1e9, I.pool ? I.pool->threads() : 0, I.o.max_ctx, I.o.max_batch);
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
      {reinterpret_cast<void**>(&I.lg), z(Impl::kOutRows) * z(s.n_vocab) * 4}};
  std::size_t total = 0;
  for (const Buf& b : bufs) total += al(b.bytes);
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
    std::size_t db = Impl::dense_bytes(w.output) + Impl::vec_bytes(w.output_norm) + Impl::vec_bytes(w.rope_freqs) + 1024;
    for (const Qwen35MoeLayer& l : w.layers) db += Impl::layer_dense_bytes(l);
    const bool head_is_out = w.mtp.head.data == w.output.data;
    if (o.mtp)
      db += Impl::layer_dense_bytes(w.mtp.layer) + Impl::dense_bytes(w.mtp.eh_proj) + Impl::vec_bytes(w.mtp.enorm) + Impl::vec_bytes(w.mtp.hnorm) +
            Impl::vec_bytes(w.mtp.head_norm) + (head_is_out ? 0 : Impl::dense_bytes(w.mtp.head));
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
  if (!o.hot_order.empty()) {
    std::size_t free_b = 0, total_b = 0;
    if (!g::sync() || !g::mem_info(&free_b, &total_b)) return R::fail(Status::fail(Code::internal, std::string("engine: device memory: ") + g::last_error()));
    std::uint64_t left = free_b > o.vram_reserve ? free_b - o.vram_reserve : 0;
    if (o.vram_budget > 0) left = std::min<std::uint64_t>(left, o.vram_budget > I.dev_bytes ? o.vram_budget - I.dev_bytes : 0);
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
  I.need.assign(z(mb), 0);
  const Status rs = I.reset_state();
  if (!rs) return R::fail(rs);
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
  Status st;
  for (std::uint32_t il = 0; il < s.n_layer; ++il) {
    const Qwen35MoeLayer& l = wt.layers[il];
    Layer& D = I.L[il];
    if (!g::rms_norm(I.x, D.attn_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: attn norm");
    if (!(st = l.recurrent ? I.deltanet_block(D, n) : I.attention_block(D, n, I.pos))) return st;
    if (!g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("engine: residual");
    if (!g::rms_norm(I.x, D.post_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("engine: post norm");
    if (!(st = I.moe_block(l, D, n, es))) return st;
    if (!g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("engine: residual");
  }
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
                                  const std::function<void(std::int32_t)>& on_token) {
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
  }
  std::int32_t last = argmax_of(lg.data(), nv);
  out.push_back(last);
  if (on_token) on_token(last);
  std::vector<float> dl(z(nv)), hn(z(e)), h(z(e)), bl, bh;
  while (static_cast<int>(out.size()) < n_predict && !is_stop(last)) {
    ++stt.steps;
    const int room = n_predict - static_cast<int>(out.size()) - 1;
    const int k = std::max(0, std::min(n_draft, room));
    std::vector<std::int32_t> drafts;
    const std::int64_t m0 = eng.mtp_pos();
    if (k > 0) {
      h = pending;
      std::int32_t tok = last;
      for (int i = 0; i < k; ++i) {
        EngineStats x;
        if (Status r = eng.mtp_step(&tok, h.data(), 1, dl.data(), hn.data(), &x); !r) return r;
        add_stats(tot, x);
        tok = argmax_of(dl.data(), nv);
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
    if (Status r = k > 0 ? eng.verify(batch.data(), nb, bl.data(), bh.data(), &x) : eng.step(batch.data(), 1, bl.data(), true, &x, bh.data()); !r) return r;
    add_stats(tot, x);
    int a = 0;
    while (a < k && argmax_of(bl.data() + a * nv, nv) == drafts[z(a)] && !is_stop(drafts[z(a)])) ++a;
    if (a < k && is_stop(drafts[z(a)]) && argmax_of(bl.data() + a * nv, nv) == drafts[z(a)]) ++a;
    const std::int32_t next = argmax_of(bl.data() + a * nv, nv);
    if (k > 0)
      if (Status r = eng.keep(a + 1); !r) return r;
    stt.accepted += a;
    if (n_draft > 0) {
      std::vector<float> hp(z((a + 1) * e));
      std::copy(pending.begin(), pending.end(), hp.begin());
      std::copy(bh.begin(), bh.begin() + a * e, hp.begin() + e);
      EngineStats y;
      if (Status r = eng.mtp_step(batch.data(), hp.data(), a + 1, nullptr, nullptr, &y); !r) return r;
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
