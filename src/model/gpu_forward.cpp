// provenance: original (the layer order and every step are forward_impl's in model/arch_qwen35moe.cpp, run on the GPU
// with the rung-1 kernels)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// gpu_forward.cpp - see gpu_forward.h.
#include "model/gpu_forward.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

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

bool is_float_type(std::uint32_t t) {
  return t == static_cast<std::uint32_t>(quant::Type::f32) || t == static_cast<std::uint32_t>(quant::Type::f16) ||
         t == static_cast<std::uint32_t>(quant::Type::bf16);
}
Status gfail(const std::string& what) { return Status::fail(Code::internal, what + ": " + g::last_error()); }

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

// one dense weight on the device: its bytes (and int8 side tables when its product runs on the int8 tiles)
struct DevW {
  const WeightView* v = nullptr;
  const void* d = nullptr;
  int64_t n_out = 0, k = 0;
  bool int8 = false;
  g::PromptWeights pw;
};
// one expert projection as typed stacks
struct DevMoe {
  std::vector<g::PromptWeights> st;  // one per type
};

struct Layer {
  DevW wq, wk, wv, wo, wqkv, wgate, wbeta, walpha, ssm_out, gate_inp, gate_sh, up_sh, down_sh;
  DevMoe gate, up, down;
  const float *attn_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *conv1d = nullptr, *ssm_dt = nullptr,
              *ssm_a = nullptr, *ssm_norm = nullptr, *gate_inp_sh = nullptr;
};

}  // namespace

struct GpuForward::Impl {
  Qwen35MoeSpec s;
  const Qwen35MoeWeights* w = nullptr;
  int max_tokens = 0, npad = 0;
  int64_t kmax = 0;
  std::uint64_t dev_bytes = 0;
  // device memory
  void* act_mem = nullptr;  // every activation buffer, one block
  Arena arena;              // one layer's weights and side tables, refilled per layer
  void* arena_mem = nullptr;
  void* out_mem = nullptr;  // the output layer and its tables (resident)
  DevW out;
  const float* out_norm = nullptr;
  const float* rope_ff = nullptr;
  g::MoeScratch moe;
  // activations
  float *x = nullptr, *xn = nullptr, *y = nullptr, *xr = nullptr, *scale = nullptr, *dig = nullptr;
  std::int8_t* q = nullptr;
  int* sum = nullptr;
  float *qg = nullptr, *kk = nullptr, *vv = nullptr, *att = nullptr, *ascr = nullptr;
  float *mixed = nullptr, *zg = nullptr, *beta = nullptr, *alpha = nullptr, *conv = nullptr, *core = nullptr, *state = nullptr;
  float *logits_r = nullptr, *wsel = nullptr, *gs = nullptr, *us = nullptr, *sh = nullptr, *gsh = nullptr, *galpha = nullptr, *zero2 = nullptr;
  std::int32_t* sel = nullptr;
  float* lg = nullptr;  // output rows (kOutRows x n_vocab)
  static constexpr int kOutRows = 64;
  std::uint64_t up_bytes = 0;
  double up_s = 0.0;

  bool upload(void* dst, const void* src, std::size_t bytes) {
    const Clock::time_point t0 = Clock::now();
    const bool ok = g::to_device(dst, src, bytes);
    up_s += since(t0);
    up_bytes += bytes;
    return ok;
  }
  // bytes of a weight's rows (one matrix, or every expert matrix)
  static std::size_t view_bytes(const WeightView& v) {
    if (!v.slots.empty()) {
      std::size_t b = 0;
      for (const ExpertSlot& sl : v.slots) b += al(static_cast<std::size_t>(v.ne[1] * sl.row_bytes));
      return b;
    }
    return static_cast<std::size_t>(v.ne[1] * v.ne[2] * v.row_bytes);
  }
  static bool int8_ok(std::uint32_t type, int64_t n_out, int64_t k) { return g::prompt_type_ok(type, k) && n_out % 128 == 0; }
  // device bytes one dense weight takes in the arena
  static std::size_t dense_bytes(const WeightView& v) {
    if (!v.present()) return 0;
    const int64_t n_out = static_cast<int64_t>(v.ne[1]), k = static_cast<int64_t>(v.ne[0]);
    std::size_t b = al(view_bytes(v));
    if (int8_ok(v.type, n_out, k)) b += al(g::prompt_table_bytes(v.type, n_out, k));
    return b;
  }
  // the typed stacks of an expert weight: type -> experts (in expert order)
  static std::map<std::uint32_t, std::vector<int>> stacks_of(const WeightView& v, int n_expert) {
    std::map<std::uint32_t, std::vector<int>> m;
    for (int e = 0; e < n_expert; ++e) m[v.slots.empty() ? v.type : v.slots[z(e)].type].push_back(e);
    return m;
  }
  static std::size_t moe_bytes(const WeightView& v, int n_expert) {
    std::size_t b = 0;
    const int64_t n_out = static_cast<int64_t>(v.ne[1]), k = static_cast<int64_t>(v.ne[0]);
    for (const auto& [t, ex] : stacks_of(v, n_expert)) {
      const int64_t rows = static_cast<int64_t>(ex.size()) * n_out;
      b += al(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k / static_cast<int64_t>(quant::type_info(t).block_elements) *
                                                                       static_cast<int64_t>(quant::type_info(t).block_bytes)));
      b += al(g::prompt_table_bytes(t, rows, k)) + al(static_cast<std::size_t>(n_expert) * sizeof(int));
    }
    return b;
  }
  std::size_t layer_bytes(const Qwen35MoeLayer& l) const {
    std::size_t b = 0;
    for (const WeightView* v : {&l.wq, &l.wk, &l.wv, &l.wo, &l.wqkv, &l.wgate, &l.wbeta, &l.walpha, &l.ssm_out, &l.gate_inp, &l.gate_shexp, &l.up_shexp,
                                &l.down_shexp})
      b += dense_bytes(*v);
    for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps}) b += moe_bytes(*v, static_cast<int>(s.n_expert));
    for (const std::vector<float>* f : {&l.attn_norm, &l.post_norm, &l.q_norm, &l.k_norm, &l.conv1d, &l.ssm_dt, &l.ssm_a, &l.ssm_norm, &l.gate_inp_shexp})
      b += al(f->size() * sizeof(float));
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
    if (!d || !upload(d, v.data, bytes)) return Status::fail(Code::internal, "gpu forward: weight upload " + v.name);
    dw.d = d;
    if (int8_ok(v.type, dw.n_out, dw.k)) {
      void* tb = arena.take(g::prompt_table_bytes(v.type, dw.n_out, dw.k));
      if (!tb || !g::prompt_prepare_at(v.type, d, dw.n_out, dw.k, tb, &dw.pw)) return gfail("gpu forward: side tables " + v.name);
      dw.int8 = true;
    }
    return Status::good();
  }
  Status put_moe(const WeightView& v, DevMoe& dm) {
    dm.st.clear();
    const int ne = static_cast<int>(s.n_expert);
    const int64_t n_out = static_cast<int64_t>(v.ne[1]), k = static_cast<int64_t>(v.ne[0]);
    const auto stacks = stacks_of(v, ne);
    for (const auto& [t, ex] : stacks) {
      const quant::TypeInfo ti = quant::type_info(t);
      const std::size_t rb = static_cast<std::size_t>(k / static_cast<int64_t>(ti.block_elements)) * ti.block_bytes;
      const std::size_t mat = static_cast<std::size_t>(n_out) * rb;
      auto* d = static_cast<std::uint8_t*>(arena.take(ex.size() * mat));
      if (!d) return Status::fail(Code::internal, "gpu forward: arena too small for " + v.name);
      for (std::size_t i = 0; i < ex.size(); ++i) {
        const int e = ex[i];
        const std::uint8_t* src = v.slots.empty() ? v.data + static_cast<std::uint64_t>(e) * static_cast<std::uint64_t>(mat) : v.slots[z(e)].data;
        if (!upload(d + i * mat, src, mat)) return gfail("gpu forward: expert upload " + v.name);
      }
      const int64_t rows = static_cast<int64_t>(ex.size()) * n_out;
      g::PromptWeights pw;
      void* tb = g::prompt_table_bytes(t, rows, k) ? arena.take(g::prompt_table_bytes(t, rows, k)) : nullptr;
      if (!g::prompt_prepare_at(t, d, rows, k, tb, &pw)) return gfail("gpu forward: expert tables " + v.name);
      if (stacks.size() > 1) {
        std::vector<int> map(z(ne), -1);
        for (std::size_t i = 0; i < ex.size(); ++i) map[z(ex[i])] = static_cast<int>(i);
        auto* em = static_cast<int*>(arena.take(z(ne) * sizeof(int)));
        if (!em || !upload(em, map.data(), z(ne) * sizeof(int))) return gfail("gpu forward: expert map " + v.name);
        pw.emap = em;
        pw.n_mat = static_cast<int>(ex.size());
      }
      dm.st.push_back(pw);
    }
    return Status::good();
  }
  Status put_vec(const std::vector<float>& f, const float*& out_ptr) {
    out_ptr = nullptr;
    if (f.empty()) return Status::good();
    void* d = arena.take(f.size() * sizeof(float));
    if (!d || !upload(d, f.data(), f.size() * sizeof(float))) return Status::fail(Code::internal, "gpu forward: vector upload");
    out_ptr = static_cast<const float*>(d);
    return Status::good();
  }
  Status load_layer(const Qwen35MoeLayer& l, Layer& L) {
    arena.used = 0;
    Status st;
    for (auto [v, d] : {std::pair{&l.wq, &L.wq}, {&l.wk, &L.wk}, {&l.wv, &L.wv}, {&l.wo, &L.wo}, {&l.wqkv, &L.wqkv}, {&l.wgate, &L.wgate},
                        {&l.wbeta, &L.wbeta}, {&l.walpha, &L.walpha}, {&l.ssm_out, &L.ssm_out}, {&l.gate_inp, &L.gate_inp},
                        {&l.gate_shexp, &L.gate_sh}, {&l.up_shexp, &L.up_sh}, {&l.down_shexp, &L.down_sh}})
      if (!(st = put_dense(*v, *d))) return st;
    if (!(st = put_moe(l.gate_exps, L.gate)) || !(st = put_moe(l.up_exps, L.up)) || !(st = put_moe(l.down_exps, L.down))) return st;
    for (auto [f, d] : {std::pair{&l.attn_norm, &L.attn_norm}, {&l.post_norm, &L.post_norm}, {&l.q_norm, &L.q_norm}, {&l.k_norm, &L.k_norm},
                        {&l.conv1d, &L.conv1d}, {&l.ssm_dt, &L.ssm_dt}, {&l.ssm_a, &L.ssm_a}, {&l.ssm_norm, &L.ssm_norm},
                        {&l.gate_inp_shexp, &L.gate_inp_sh}})
      if (!(st = put_vec(*f, *d))) return st;
    return Status::good();
  }

  // y (n x W.n_out) = W x, x n rows of W.k unrounded floats: quantized weights round x to gpu256x8 first (rotated when
  // the weight is stored rotated), as the reference's mm() does
  Status prod(const DevW& W, const float* xin, int64_t n, float* yout) {
    const WeightView& v = *W.v;
    if (is_float_type(v.type)) {
      if (v.rot32) return Status::fail(Code::unsupported, "gpu forward: a rotated float weight (" + v.name + ")");
      if (!g::matmul(v.type, W.d, v.row_bytes, W.n_out, W.k, xin, n, W.k, yout, W.n_out)) return gfail("gpu forward: product " + v.name);
      return Status::good();
    }
    if (!g::act_q8x32(xin, W.k, n, W.k, v.rot32, xr, W.k, q, scale, sum)) return gfail("gpu forward: rounding for " + v.name);
    if (W.int8) {
      if (!g::act_digits(scale, sum, n, W.k, dig) || !g::prompt_gemm(W.pw, W.n_out, q, scale, dig, n, yout))
        return gfail("gpu forward: int8 product " + v.name);
      return Status::good();
    }
    if (!g::matmul(v.type, W.d, v.row_bytes, W.n_out, W.k, xr, n, W.k, yout, W.n_out)) return gfail("gpu forward: product " + v.name);
    return Status::good();
  }

  Status attention_block(const Layer& L, int64_t n) {
    const int64_t hd = s.head_dim, nh = s.n_head, nkv = s.n_head_kv, e = s.n_embd;
    const int64_t qh = s.attn_gate ? 2 * hd : hd, q_w = qh * nh, kv_w = hd * nkv;
    Status st;
    if (!(st = prod(L.wq, xn, n, qg)) || !(st = prod(L.wk, xn, n, kk)) || !(st = prod(L.wv, xn, n, vv))) return st;
    if (L.q_norm && !g::rms_norm(qg, L.q_norm, qg, hd, nh, qh, q_w, n, s.rms_eps)) return gfail("gpu forward: q norm");
    if (L.k_norm && !g::rms_norm(kk, L.k_norm, kk, hd, nkv, hd, kv_w, n, s.rms_eps)) return gfail("gpu forward: k norm");
    const int sec[4] = {s.rope_sections[0], s.rope_sections[1], s.rope_sections[2], s.rope_sections[3]};
    if (!g::rope(qg, nh, qh, q_w, n, 0, static_cast<int>(s.n_rot), s.rope_freq_base, sec, s.rope_neox, rope_ff) ||
        !g::rope(kk, nkv, hd, kv_w, n, 0, static_cast<int>(s.n_rot), s.rope_freq_base, sec, s.rope_neox, rope_ff))
      return gfail("gpu forward: rope");
    const float scl = s.attn_scale != 0.0f ? s.attn_scale : 1.0f / std::sqrt(static_cast<float>(hd));
    if (!g::attention(qg, q_w, qh, kk, vv, kv_w, nh, nkv, hd, n, 0, scl, s.attn_gate, att, ascr)) return gfail("gpu forward: attention");
    (void)e;
    return prod(L.wo, att, n, y);
  }

  Status deltanet_block(const Layer& L, int64_t n) {
    const int64_t ch = s.conv_channels(), di = s.ssm_d_inner, ds = s.ssm_d_state, nk = s.ssm_n_group, nv = s.ssm_dt_rank, taps = s.ssm_d_conv;
    Status st;
    if (!(st = prod(L.wqkv, xn, n, mixed)) || !(st = prod(L.wgate, xn, n, zg)) || !(st = prod(L.wbeta, xn, n, beta)) ||
        !(st = prod(L.walpha, xn, n, alpha)))
      return st;
    if (!g::deltanet_gates(beta, alpha, L.ssm_dt, L.ssm_a, n, nv)) return gfail("gpu forward: deltanet gates");
    if (!g::deltanet_conv(mixed, nullptr, L.conv1d, n, ch, taps, conv, nullptr)) return gfail("gpu forward: deltanet conv");
    if (!g::l2_norm(conv, ds, 2 * nk, ds, ch, n, s.rms_eps)) return gfail("gpu forward: deltanet l2 norm");
    if (!g::dzero(state, z(nv * ds * ds) * sizeof(float))) return gfail("gpu forward: deltanet state");
    const bool pow2 = ds >= 32 && ds <= 1024 && (ds & (ds - 1)) == 0;
    const bool ok = pow2 ? g::deltanet_recurrent_reg(conv, ch, beta, alpha, state, n, nk, nv, ds, core)
                         : g::deltanet_recurrent(conv, ch, beta, alpha, state, n, nk, nv, ds, core);
    if (!ok) return gfail("gpu forward: deltanet recurrence");
    if (!g::gated_norm(core, zg, L.ssm_norm, n, nv, ds, s.rms_eps)) return gfail("gpu forward: gated norm");
    (void)di;
    return prod(L.ssm_out, core, n, y);
  }

  Status moe_block(const Layer& L, int64_t n) {
    const int64_t e = s.n_embd, ne = s.n_expert, ffs = s.n_ff_shexp;
    Status st;
    if (!(st = prod(L.gate_inp, xn, n, logits_r))) return st;  // the router first: its product may use the rounding buffers
    if (!g::router_topk(logits_r, ne, n, static_cast<int>(s.n_expert_used), s.expert_weights_scale, sel, wsel)) return gfail("gpu forward: router");
    // the routed experts' input: post-norm rows rounded once (gate and up share it), unrotated (experts stay unrotated)
    if (!g::act_q8x32(xn, e, n, e, false, xr, e, q, scale, sum) || !g::act_digits(scale, sum, n, e, dig)) return gfail("gpu forward: expert input rounding");
    const g::MoeStacks sg{L.gate.st.data(), static_cast<int>(L.gate.st.size())}, su{L.up.st.data(), static_cast<int>(L.up.st.size())},
        sd{L.down.st.data(), static_cast<int>(L.down.st.size())};
    if (!g::moe_prompt_multi(sg, su, sd, q, scale, dig, n, sel, wsel, moe, g::MoeCombine::kFloat, 0, y)) return gfail("gpu forward: routed experts");
    // the gated shared expert: y += sigmoid(gate_inp_shexp . xn) x down(silu(gate xn) * up xn)
    if (!(st = prod(L.gate_sh, xn, n, gs)) || !(st = prod(L.up_sh, xn, n, us))) return st;
    if (!g::silu_mul(gs, us, gs, n * ffs)) return gfail("gpu forward: shared silu");
    if (!(st = prod(L.down_sh, gs, n, sh))) return st;
    if (!g::matmul(static_cast<std::uint32_t>(quant::Type::f32), L.gate_inp_sh, static_cast<std::uint64_t>(e) * sizeof(float), 1, e, xn, n, e, gsh, 1) ||
        !g::deltanet_gates(gsh, galpha, zero2, zero2, n, 1))  // its first output is sigmoid(gsh), as the shared gate needs
      return gfail("gpu forward: shared gate");
    if (!g::add_rows(y, e, sh, e, gsh, e, n)) return gfail("gpu forward: shared add");
    return Status::good();
  }
};

GpuForward::GpuForward(std::unique_ptr<Impl> p) : p_(std::move(p)) {}

GpuForward::~GpuForward() {
  if (!p_) return;
  g::moe_scratch_free(&p_->moe);
  g::dfree(p_->act_mem);
  g::dfree(p_->arena_mem);
  g::dfree(p_->out_mem);
}

std::uint64_t GpuForward::device_bytes() const { return p_ ? p_->dev_bytes : 0; }

ForwardOptions GpuForward::matching_reference_options() {
  ForwardOptions o;
  o.act_int8 = {"all"};
  o.act_int8_gpu = true;
  o.act_gpu256 = true;
  o.act_gpu256_sub = 1;
  o.planes = false;
  return o;
}

Result<std::unique_ptr<GpuForward>> GpuForward::open(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, int max_tokens) {
  using R = Result<std::unique_ptr<GpuForward>>;
  if (!g::available()) return R::fail(Status::fail(Code::no_device, std::string("gpu forward: ") + g::last_error()));
  if (s.family != Family::qwen35moe || s.dense) return R::fail(Status::fail(Code::unsupported, "gpu forward: qwen35moe only in this first cut"));
  if (max_tokens <= 0 || max_tokens > 8192) return R::fail(Status::fail(Code::invalid_argument, "gpu forward: max_tokens 1-8192"));
  if (w.layers.size() != s.n_layer) return R::fail(Status::fail(Code::invalid_argument, "gpu forward: the weights were bound for another spec"));
  if (!w.output.present() || w.output_norm.empty()) return R::fail(Status::fail(Code::invalid_argument, "gpu forward: no output layer"));
  auto p = std::make_unique<Impl>();
  Impl& I = *p;
  I.s = s;
  I.w = &w;
  I.max_tokens = max_tokens;
  I.npad = (max_tokens + 63) / 64 * 64 + 64;
  const int64_t e = s.n_embd, hd = s.head_dim, nh = s.n_head, nkv = s.n_head_kv, qh = s.attn_gate ? 2 * hd : hd;
  const int64_t ch = s.conv_channels(), di = s.ssm_d_inner, ds = s.ssm_d_state, nv = s.ssm_dt_rank, ne = s.n_expert, nu = s.n_expert_used;
  const int64_t ffs = s.n_ff_shexp;
  for (const Qwen35MoeLayer& l : w.layers) {
    for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps})
      if (v->rot32) return R::fail(Status::fail(Code::unsupported, "gpu forward: rotated routed experts (" + v->name + ")"));
    for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
      const auto st = Impl::stacks_of(*v, static_cast<int>(ne));
      if (st.size() > 4) return R::fail(Status::fail(Code::unsupported, "gpu forward: more than 4 expert types in " + v->name));
      for (const auto& [t, ex] : st)
        if (!g::prompt_type_ok(t, static_cast<int64_t>(v->ne[0])) && st.size() > 1)
          return R::fail(Status::fail(Code::unsupported, "gpu forward: a tiered weight with a type that has no int8 step (" + v->name + ")"));
    }
    if (!l.bq.empty() || !l.attn_sub_norm.empty()) return R::fail(Status::fail(Code::unsupported, "gpu forward: biases / sub-norms (qwen35moe only)"));
  }
  // the largest product input and the largest layer
  I.kmax = std::max<int64_t>({e, hd * nh, di, ffs, static_cast<int64_t>(s.n_ff_exp)});
  std::size_t lmax = 0;
  for (const Qwen35MoeLayer& l : w.layers) lmax = std::max(lmax, I.layer_bytes(l));
  // activations (rows padded for the int8 tiles' 64-row reads)
  const std::size_t np = z(I.npad);
  struct Buf {
    void** ptr;
    std::size_t bytes;
  };
  const int64_t att_rows = static_cast<int64_t>(np);
  std::vector<Buf> bufs = {
      {reinterpret_cast<void**>(&I.x), np * z(e) * 4},          {reinterpret_cast<void**>(&I.xn), np * z(e) * 4},
      {reinterpret_cast<void**>(&I.y), np * z(e) * 4},          {reinterpret_cast<void**>(&I.xr), np * z(I.kmax) * 4},
      {reinterpret_cast<void**>(&I.q), np * z(I.kmax)},         {reinterpret_cast<void**>(&I.scale), np * z(I.kmax / 32) * 4},
      {reinterpret_cast<void**>(&I.sum), np * z(I.kmax / 32) * 4}, {reinterpret_cast<void**>(&I.dig), np * z(I.kmax / 32) * 4},
      {reinterpret_cast<void**>(&I.qg), np * z(qh * nh) * 4},   {reinterpret_cast<void**>(&I.kk), np * z(hd * nkv) * 4},
      {reinterpret_cast<void**>(&I.vv), np * z(hd * nkv) * 4},  {reinterpret_cast<void**>(&I.att), np * z(hd * nh) * 4},
      {reinterpret_cast<void**>(&I.ascr), z(att_rows) * z(nh) * z(att_rows) * 4},
      {reinterpret_cast<void**>(&I.mixed), np * z(ch) * 4},     {reinterpret_cast<void**>(&I.zg), np * z(di) * 4},
      {reinterpret_cast<void**>(&I.beta), np * z(nv) * 4},      {reinterpret_cast<void**>(&I.alpha), np * z(nv) * 4},
      {reinterpret_cast<void**>(&I.conv), np * z(ch) * 4},      {reinterpret_cast<void**>(&I.core), np * z(di) * 4},
      {reinterpret_cast<void**>(&I.state), z(nv) * z(ds) * z(ds) * 4},
      {reinterpret_cast<void**>(&I.logits_r), np * z(ne) * 4},  {reinterpret_cast<void**>(&I.sel), np * z(nu) * 4},
      {reinterpret_cast<void**>(&I.wsel), np * z(nu) * 4},      {reinterpret_cast<void**>(&I.gs), np * z(ffs) * 4},
      {reinterpret_cast<void**>(&I.us), np * z(ffs) * 4},       {reinterpret_cast<void**>(&I.sh), np * z(e) * 4},
      {reinterpret_cast<void**>(&I.gsh), np * 4},               {reinterpret_cast<void**>(&I.galpha), np * 4},
      {reinterpret_cast<void**>(&I.zero2), 256},
      {reinterpret_cast<void**>(&I.lg), z(Impl::kOutRows) * z(s.n_vocab) * 4}};
  std::size_t total = 0;
  for (const Buf& b : bufs) total += al(b.bytes);
  I.act_mem = g::dalloc(total);
  if (!I.act_mem || !g::dzero(I.act_mem, total)) return R::fail(Status::fail(Code::over_ceiling, "gpu forward: activation memory"));
  auto* cur = static_cast<std::uint8_t*>(I.act_mem);
  for (const Buf& b : bufs) {
    *b.ptr = cur;
    cur += al(b.bytes);
  }
  I.dev_bytes += total;
  // the per-layer arena
  I.arena_mem = g::dalloc(lmax);
  if (!I.arena_mem) return R::fail(Status::fail(Code::over_ceiling, "gpu forward: layer arena (" + std::to_string(lmax >> 20) + " MiB)"));
  I.arena = Arena{static_cast<std::uint8_t*>(I.arena_mem), lmax, 0};
  I.dev_bytes += lmax;
  // the experts' scratch (the exact paths' buffers only when some expert type has no int8 step)
  bool exact = false;
  for (const Qwen35MoeLayer& l : w.layers)
    for (const WeightView* v : {&l.gate_exps, &l.up_exps, &l.down_exps})
      for (const auto& [t, ex] : Impl::stacks_of(*v, static_cast<int>(ne))) exact = exact || !g::prompt_type_ok(t, static_cast<int64_t>(v->ne[0]));
  if (!g::moe_scratch_alloc(max_tokens, static_cast<int>(nu), static_cast<int>(ne), e, s.n_ff_exp, &I.moe, exact))
    return R::fail(Status::fail(Code::over_ceiling, std::string("gpu forward: expert scratch: ") + g::prompt_last_error()));
  // the output layer, its norm and the rope factors (resident)
  {
    const std::size_t ob = Impl::dense_bytes(w.output) + al(w.output_norm.size() * 4) + al(w.rope_freqs.size() * 4) + 512;
    I.out_mem = g::dalloc(ob);
    if (!I.out_mem) return R::fail(Status::fail(Code::over_ceiling, "gpu forward: output layer"));
    I.dev_bytes += ob;
    Arena keep = I.arena;
    I.arena = Arena{static_cast<std::uint8_t*>(I.out_mem), ob, 0};
    Status st = I.put_dense(w.output, I.out);
    if (st) st = I.put_vec(w.output_norm, I.out_norm);
    if (st) st = I.put_vec(w.rope_freqs, I.rope_ff);
    I.arena = keep;
    if (!st) return R::fail(st);
  }
  return R::good(std::unique_ptr<GpuForward>(new GpuForward(std::move(p))));
}

Status GpuForward::prefill(const std::int32_t* tokens, int n, int out_begin, int out_end, float* logits, GpuForwardStats* stats) {
  Impl& I = *p_;
  const Qwen35MoeSpec& s = I.s;
  const Qwen35MoeWeights& w = *I.w;
  if (n <= 0 || n > I.max_tokens || out_begin < 0 || out_end > n || out_begin >= out_end || !logits)
    return Status::fail(Code::invalid_argument, "gpu forward: token count or output range");
  const Clock::time_point t0 = Clock::now();
  I.up_s = 0.0;
  I.up_bytes = 0;
  const int64_t e = s.n_embd;
  // embedding rows on the host (the reference's embed_row without planes), then one copy
  {
    std::vector<float> xh(z(n) * z(e));
    for (int t = 0; t < n; ++t) {
      const std::int32_t tok = tokens[t];
      if (tok < 0 || static_cast<std::uint64_t>(tok) >= s.n_vocab) return Status::fail(Code::invalid_argument, "a token id is outside the vocabulary");
      const Status qs =
          quant::dequantize_row(static_cast<quant::Type>(w.tok_embd.type), w.tok_embd.data + static_cast<std::uint64_t>(tok) * w.tok_embd.row_bytes,
                                xh.data() + z(t) * z(e), e);
      if (!qs.ok()) return Status::fail(Code::unsupported, "gpu forward: the embedding type cannot be read on the host");
    }
    if (!I.upload(I.x, xh.data(), xh.size() * sizeof(float))) return gfail("gpu forward: embedding upload");
  }
  Layer L;
  Status st;
  for (std::uint32_t il = 0; il < s.n_layer; ++il) {
    const Qwen35MoeLayer& l = w.layers[il];
    if (!(st = I.load_layer(l, L))) return st;
    if (!g::rms_norm(I.x, L.attn_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("gpu forward: attn norm");
    if (!(st = l.recurrent ? I.deltanet_block(L, n) : I.attention_block(L, n))) return st;
    if (!g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("gpu forward: residual");
    if (!g::rms_norm(I.x, L.post_norm, I.xn, e, 1, e, e, n, s.rms_eps)) return gfail("gpu forward: post norm");
    if (!(st = I.moe_block(L, n))) return st;
    if (!g::add_rows(I.x, e, I.y, e, nullptr, e, n)) return gfail("gpu forward: residual");
  }
  // the output layer over the requested rows, in chunks of kOutRows
  const int64_t nvoc = s.n_vocab;
  for (int r0 = out_begin; r0 < out_end; r0 += Impl::kOutRows) {
    const int m = std::min(Impl::kOutRows, out_end - r0);
    if (!g::rms_norm(I.x + static_cast<int64_t>(r0) * e, I.out_norm, I.xn, e, 1, e, e, m, s.rms_eps)) return gfail("gpu forward: output norm");
    if (!(st = I.prod(I.out, I.xn, m, I.lg))) return st;
    if (!g::to_host(logits + static_cast<int64_t>(r0 - out_begin) * nvoc, I.lg, z(m) * z(nvoc) * sizeof(float))) return gfail("gpu forward: logits");
  }
  if (!g::sync()) return gfail("gpu forward: sync");
  if (stats) {
    stats->seconds = since(t0);
    stats->upload_seconds = I.up_s;
    stats->bytes_uploaded = I.up_bytes;
  }
  return Status::good();
}

}  // namespace pelias::model
