// provenance: original (the computation READ-SRC from llama.cpp src/models/qwen35moe.cpp, src/llama-graph.cpp
//   build_moe_ffn / build_ffn / build_norm, src/models/models.h build_gdn_l2_norm and ggml/src/ggml-cpu/ops.cpp
//   (rms_norm, rope multi / IMRoPE, ssm_conv, gated_delta_net, softplus) at commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a, read as a specification; no code copied)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "model/arch_qwen35moe.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>

#include "quant/block_types.h"
#include "quant/quant_cpu.h"

namespace pelias::model {

namespace {

using std::int64_t;
using std::size_t;
using std::uint64_t;

inline size_t z(int64_t v) { return static_cast<size_t>(v); }

// ------------------------------------------------------------------------------------------------ binding

Status shape_error(const std::string& name, const std::string& want) {
  return Status::fail(Code::corrupt, "the tensor's shape is not the one the spec implies (" + want + ")", 0, name);
}

Status bind_view(const io::GgufReader& r, const std::uint8_t* base, const std::string& name, std::initializer_list<uint64_t> want,
                 WeightView& out, bool required = true) {
  const io::GgufTensor* t = r.find_tensor(name);
  if (!t) return required ? Status::fail(Code::not_found, "a tensor the model needs is missing", 0, name) : Status::good();
  uint64_t ne[3] = {1, 1, 1};
  std::string want_text;
  size_t k = 0;
  for (uint64_t v : want) {
    ne[k++] = v;
    want_text += (want_text.empty() ? "" : " x ") + std::to_string(v);
  }
  if (t->ne[0] != ne[0] || t->ne[1] != ne[1] || t->ne[2] != ne[2] || t->ne[3] != 1) return shape_error(name, want_text);
  const auto type = static_cast<quant::Type>(t->type);
  if (!quant::can_dequantize(type))
    return Status::fail(Code::unsupported, std::string("the reference pass cannot read tensor type ") + quant::type_info(t->type).name + " yet", 0, name);
  const quant::TypeInfo ti = quant::type_info(t->type);
  if (ne[0] % ti.block_elements != 0) return shape_error(name, "rows a whole number of blocks");
  out.data = base + t->file_offset;
  out.type = t->type;
  out.ne[0] = ne[0];
  out.ne[1] = ne[1];
  out.ne[2] = ne[2];
  out.row_bytes = quant::row_bytes(t->type, ne[0]);
  out.name = name;
  return Status::good();
}

Status bind_vector(const io::GgufReader& r, const std::uint8_t* base, const std::string& name, std::initializer_list<uint64_t> want,
                   std::vector<float>& out) {
  WeightView v;
  Status s = bind_view(r, base, name, want, v);
  if (!s) return s;
  const int64_t n = static_cast<int64_t>(v.ne[0] * v.ne[1] * v.ne[2]);
  out.assign(z(n), 0.0f);
  for (uint64_t row = 0; row < v.ne[1] * v.ne[2]; ++row) {
    s = quant::dequantize_row(static_cast<quant::Type>(v.type), v.data + row * v.row_bytes, out.data() + row * v.ne[0],
                              static_cast<int64_t>(v.ne[0]));
    if (!s) return Status::fail(s.code, s.message, 0, name);
  }
  return Status::good();
}

// ------------------------------------------------------------------------------------------------ threads

// Runs fn(item, thread) for item in [0, n), items handed out one at a time (the work per item is uneven: experts).
template <typename F>
void parallel_items(int64_t n, int threads, F&& fn) {
  const int t = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(threads, n)));
  std::atomic<int64_t> next{0};
  auto work = [&](int tid) {
    for (int64_t i = next.fetch_add(1); i < n; i = next.fetch_add(1)) fn(i, tid);
  };
  if (t == 1) {
    work(0);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(z(t - 1));
  for (int k = 1; k < t; ++k) pool.emplace_back(work, k);
  work(0);
  for (auto& th : pool) th.join();
}

// ------------------------------------------------------------------------------------------------ matrix x activations

constexpr int kTile = 16;  // weight rows per tile (one fp32 vector lane group per row position)

struct Scratch {
  std::vector<float> row, wt;            // one dequantized row; a tile transposed to [k][kTile]
  std::vector<float> xe, ge, ue, ye;     // expert inputs and intermediates
  std::vector<float> scores;             // attention scores of one query
  std::vector<float> state, delta;       // DeltaNet state and its per-token update
};

void load_tile(const WeightView& w, uint64_t first_row, int nr, int64_t k, Scratch& sc) {
  if (sc.row.size() < z(k)) sc.row.resize(z(k));
  if (sc.wt.size() < z(k * kTile)) sc.wt.resize(z(k * kTile));
  float* wt = sc.wt.data();
  for (int r = 0; r < nr; ++r) {
    // the type and the row length were checked when the tensor was bound
    (void)quant::dequantize_row(static_cast<quant::Type>(w.type), w.data + (first_row + static_cast<uint64_t>(r)) * w.row_bytes,
                                sc.row.data(), k);
    const float* src = sc.row.data();
    for (int64_t i = 0; i < k; ++i) wt[z(i * kTile + r)] = src[z(i)];
  }
  for (int r = nr; r < kTile; ++r)
    for (int64_t i = 0; i < k; ++i) wt[z(i * kTile + r)] = 0.0f;
}

// acc[t][j] = sum_i wt[i][j] * x_t[i], for NT activation rows at once (each output lane its own running sum, so the
// order of the sum is fixed: i = 0, 1, ..., k-1, whatever the thread count)
template <int NT>
void tile_dot(const float* wt, int64_t k, const float* const* xs, float (&acc)[4][kTile]) {
  for (int t = 0; t < NT; ++t)
    for (int j = 0; j < kTile; ++j) acc[t][j] = 0.0f;
  for (int64_t i = 0; i < k; ++i) {
    const float* wrow = wt + i * kTile;
    for (int t = 0; t < NT; ++t) {
      const float xv = xs[t][i];
      float* a = acc[t];
      for (int j = 0; j < kTile; ++j) a[j] += wrow[j] * xv;
    }
  }
}

// Y[t][o] = W_e[o] . X[t] for o in [o_begin, o_end) of matrix e, t in [0, n_tok). X rows ldx apart, Y rows ldy apart.
void matmul_rows(const WeightView& w, uint64_t e, int64_t o_begin, int64_t o_end, const float* x, int64_t n_tok, int64_t ldx, float* y,
                 int64_t ldy, Scratch& sc) {
  const int64_t k = static_cast<int64_t>(w.ne[0]);
  const uint64_t base_row = e * w.ne[1];
  for (int64_t o0 = o_begin; o0 < o_end; o0 += kTile) {
    const int nr = static_cast<int>(std::min<int64_t>(kTile, o_end - o0));
    load_tile(w, base_row + static_cast<uint64_t>(o0), nr, k, sc);
    float acc[4][kTile];
    for (int64_t t = 0; t < n_tok; t += 4) {
      const int nt = static_cast<int>(std::min<int64_t>(4, n_tok - t));
      const float* xs[4] = {x + t * ldx, x + std::min(t + 1, n_tok - 1) * ldx, x + std::min(t + 2, n_tok - 1) * ldx,
                            x + std::min(t + 3, n_tok - 1) * ldx};
      switch (nt) {
        case 4: tile_dot<4>(sc.wt.data(), k, xs, acc); break;
        case 3: tile_dot<3>(sc.wt.data(), k, xs, acc); break;
        case 2: tile_dot<2>(sc.wt.data(), k, xs, acc); break;
        default: tile_dot<1>(sc.wt.data(), k, xs, acc); break;
      }
      for (int tt = 0; tt < nt; ++tt)
        for (int j = 0; j < nr; ++j) y[(t + tt) * ldy + o0 + j] = acc[tt][j];
    }
  }
}

// The whole matrix (matrix 0 of the view), output rows split among the threads tile by tile.
void matmul(const WeightView& w, const float* x, int64_t n_tok, int64_t ldx, float* y, int64_t ldy, int threads, std::vector<Scratch>& sc) {
  const int64_t n_out = static_cast<int64_t>(w.ne[1]);
  const int64_t tiles = (n_out + kTile - 1) / kTile;
  // a few tiles per item keeps the hand-out cheap without unbalancing the end
  const int64_t per = std::max<int64_t>(1, tiles / (static_cast<int64_t>(threads) * 16));
  const int64_t items = (tiles + per - 1) / per;
  parallel_items(items, threads, [&](int64_t it, int tid) {
    const int64_t o_begin = it * per * kTile;
    const int64_t o_end = std::min(n_out, (it + 1) * per * kTile);
    matmul_rows(w, 0, o_begin, o_end, x, n_tok, ldx, y, ldy, sc[z(tid)]);
  });
}

// ------------------------------------------------------------------------------------------------ elementwise

// y = (x * 1/sqrt(mean(x^2) + eps)) * w, the mean summed in double as the reference's rms_norm does
void rms_norm(const float* x, const float* w, float* y, int64_t n, float eps) {
  double sum = 0.0;
  for (int64_t i = 0; i < n; ++i) sum += static_cast<double>(x[i] * x[i]);
  const float mean = static_cast<float>(sum / static_cast<double>(n));
  const float scale = 1.0f / std::sqrt(mean + eps);
  for (int64_t i = 0; i < n; ++i) y[i] = (x[i] * scale) * w[i];
}

// x / sqrt(sum(x^2) + eps), as the reference builds it: rms_norm with eps/n, then a scale by 1/sqrt(n)
void l2_norm(float* x, int64_t n, float eps) {
  const float nf = static_cast<float>(n);
  double sum = 0.0;
  for (int64_t i = 0; i < n; ++i) sum += static_cast<double>(x[i] * x[i]);
  const float mean = static_cast<float>(sum / static_cast<double>(n));
  const float scale = 1.0f / std::sqrt(mean + eps / nf);
  const float post = 1.0f / std::sqrt(nf);
  for (int64_t i = 0; i < n; ++i) x[i] = (x[i] * scale) * post;
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softplus(float x) { return x > 20.0f ? x : std::log(1.0f + std::exp(x)); }

// softmax in place: max, exp(x - max), the sum in double, then a multiply by 1/sum
void softmax(float* x, int64_t n) {
  float mx = x[0];
  for (int64_t i = 1; i < n; ++i) mx = std::max(mx, x[i]);
  double sum = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - mx);
    sum += static_cast<double>(x[i]);
  }
  const float inv = static_cast<float>(1.0 / sum);
  for (int64_t i = 0; i < n; ++i) x[i] *= inv;
}

// ------------------------------------------------------------------------------------------------ rope (IMRoPE)

// cos / sin per rotated pair for one text position: all three position components equal `pos`, the fourth 0.
// The angle grows by one multiply per pair (theta *= base^(-2/n_rot)), as the reference's cache does.
void rope_cache(const Qwen35MoeSpec& s, int64_t pos, std::vector<float>& cs) {
  const int64_t n_rot = s.n_rot;
  cs.assign(z(n_rot), 0.0f);
  const float theta_scale = std::pow(s.rope_freq_base, -2.0f / static_cast<float>(n_rot));
  const int sec0 = s.rope_sections[0], sec1 = s.rope_sections[1], sec2 = s.rope_sections[2];
  const int sect_dims = sec0 + sec1 + sec2 + s.rope_sections[3];
  float theta_tpos = static_cast<float>(pos);  // t, h and w components share the position for text
  float theta_e = 0.0f;                         // the fourth component's position is 0
  for (int64_t i0 = 0; i0 < n_rot; i0 += 2) {
    const int sector = sect_dims > 0 ? static_cast<int>((i0 / 2) % sect_dims) : 0;
    bool use_e = false;
    if (sect_dims > 0) {
      if (sector % 3 == 1 && sector < 3 * sec1) use_e = false;
      else if (sector % 3 == 2 && sector < 3 * sec2) use_e = false;
      else if (sector % 3 == 0 && sector < 3 * sec0) use_e = false;
      else use_e = true;
    }
    const float theta = use_e ? theta_e : theta_tpos;
    cs[z(i0)] = std::cos(theta);
    cs[z(i0 + 1)] = std::sin(theta);
    theta_tpos *= theta_scale;
    theta_e *= theta_scale;
  }
}

// NEOX pairing over the first n_rot dimensions: (i, i + n_rot/2)
void rope_apply(float* v, const std::vector<float>& cs, int64_t n_rot) {
  const int64_t half = n_rot / 2;
  for (int64_t i0 = 0; i0 < n_rot; i0 += 2) {
    const int64_t ic = i0 / 2;
    const float c = cs[z(i0)], sn = cs[z(i0 + 1)];
    const float x0 = v[ic], x1 = v[ic + half];
    v[ic] = x0 * c - x1 * sn;
    v[ic + half] = x0 * sn + x1 * c;
  }
}

// ------------------------------------------------------------------------------------------------ the blocks

struct Ctx {
  const Qwen35MoeSpec& s;
  int threads;
  std::vector<Scratch>& sc;
  int64_t n;  // tokens
  const ForwardOptions& opt;
  std::uint32_t il = 0;  // the layer being run
};

// the 8-bit activation format the inputs of `w` are rounded to (ForwardOptions::act_int8), or none; float weights never
// are, as in the reference
enum class ActFmt { none, q8_0, q8_1, q8_k, q4_32, q8_256 };

ActFmt act_format(const ForwardOptions& o, const WeightView& w) {
  if (o.act_int8.empty() || !w.present()) return ActFmt::none;
  const auto t = static_cast<quant::Type>(w.type);
  if (t == quant::Type::f32 || t == quant::Type::f16 || t == quant::Type::bf16) return ActFmt::none;
  bool hit = false;
  for (const std::string& g : o.act_int8)
    if (g == "all" || w.name.find(g) != std::string::npos) hit = true;
  if (!hit) return ActFmt::none;
  if (o.act_int8_gpu) return o.act_bits == 4 ? ActFmt::q4_32 : (o.act_gpu256 ? ActFmt::q8_256 : ActFmt::q8_1);
  switch (t) {
    case quant::Type::q4_0: case quant::Type::q5_0: case quant::Type::q8_0: case quant::Type::iq4_nl: case quant::Type::mxfp4:
      return ActFmt::q8_0;
    case quant::Type::q4_1: case quant::Type::q5_1:
      return ActFmt::q8_1;
    default:
      return ActFmt::q8_k;
  }
}

int act_block(ActFmt f) { return f == ActFmt::q8_k || f == ActFmt::q8_256 ? 256 : 32; }

// one row to the 8-bit format and back: Q8_0 (32: d fp16, qs int8), q8_1 (32: d fp16, s fp16, qs int8) or Q8_K
// (256: d fp32, qs int8, sums). The row length is a multiple of the block (checked once by check_act_rounding).
void act_round_plain(float* x, int64_t n, ActFmt f) {
  if (f == ActFmt::q4_32) {  // 4 bits per value, -7..7, one fp16 scale per 32 (Block M, W4A4)
    for (int64_t i = 0; i < n; i += 32) {
      float amax = 0.0f;
      for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[i + j]));
      const float d = quant::fp16_to_fp32(quant::fp32_to_fp16(amax / 7.0f));
      for (int j = 0; j < 32; ++j) {
        const float q = d > 0.0f ? std::nearbyint(x[i + j] / d) : 0.0f;
        x[i + j] = d * std::min(7.0f, std::max(-7.0f, q));
      }
    }
    return;
  }
  if (f == ActFmt::q8_256) {  // the Pelias int8 kernel's format
    act_round_q8_256(x, n);
    return;
  }
  if (f == ActFmt::q8_k) {
    std::uint8_t b[292];
    for (int64_t i = 0; i < n; i += 256) {
      quant::quantize_q8_k(x + i, b, 256);
      float d;
      std::memcpy(&d, b, 4);
      for (int j = 0; j < 256; ++j) x[i + j] = d * static_cast<float>(static_cast<std::int8_t>(b[4 + j]));
    }
    return;
  }
  std::uint8_t b[36];
  const int qs = f == ActFmt::q8_1 ? 4 : 2;  // q8_1 carries d and s (2 x fp16) before the values; Q8_0 only d
  for (int64_t i = 0; i < n; i += 32) {
    if (f == ActFmt::q8_1) quant::quantize_q8_1(x + i, b, 32);
    else quant::quantize_q8_0(x + i, b, 32);
    std::uint16_t dh;
    std::memcpy(&dh, b, 2);
    const float d = quant::fp16_to_fp32(dh);
    for (int j = 0; j < 32; ++j) x[i + j] = d * static_cast<float>(static_cast<std::int8_t>(b[qs + j]));
  }
}

// the rounding, with the optional Walsh-Hadamard rotation of each 32-value block around it (ForwardOptions::act_rot32)
void act_round_row(float* x, int64_t n, ActFmt f, bool rot) {
  if (rot) act_rotate32(x, n);
  act_round_plain(x, n, f);
  if (rot) act_rotate32(x, n);
}

// one input row of a product with `w`: rotated when the file stores `w` rotated (then rounded in the rotated basis, and
// not rotated back), else rounded with the optional simulated rotation around it
void prep_row(float* x, int64_t n, ActFmt f, bool w_rot, bool sim_rot) {
  if (w_rot) {
    act_rotate32(x, n);
    if (f != ActFmt::none) act_round_plain(x, n, f);
  } else if (f != ActFmt::none) {
    act_round_row(x, n, f, sim_rot);
  }
}

// every rounded input row must be a whole number of blocks (the reference's own requirement for these products)
Status check_act_rounding(const ForwardOptions& o, const Qwen35MoeWeights& w) {
  if (o.act_int8.empty()) return Status::good();
  std::vector<const WeightView*> v{&w.output};
  for (const Qwen35MoeLayer& l : w.layers)
    for (const WeightView* p : {&l.wq, &l.wk, &l.wv, &l.wo, &l.wqkv, &l.wgate, &l.wbeta, &l.walpha, &l.ssm_out, &l.gate_inp, &l.up_exps,
                                &l.gate_exps, &l.down_exps, &l.up_shexp, &l.gate_shexp, &l.down_shexp})
      v.push_back(p);
  for (const WeightView* p : v) {
    const ActFmt f = act_format(o, *p);
    if (f != ActFmt::none && p->ne[0] % static_cast<std::uint64_t>(act_block(f)) != 0)
      return Status::fail(Code::invalid_argument, "8-bit activation rounding: the row length is not a whole number of blocks", 0, p->name);
  }
  return Status::good();
}

// matmul over the whole matrix, recording the activations for the importance matrix when asked
void mm(const Ctx& c, const WeightView& w, const float* x, int64_t n_tok, int64_t ldx, float* y, int64_t ldy) {
  if (c.opt.imatrix && !w.rot32) c.opt.imatrix->add_dense(w.name, x, n_tok, ldx, static_cast<int64_t>(w.ne[0]));
  if (c.opt.imatrix && w.rot32) {  // stored rotated: the quantizer sees the rotated inputs, so collect those
    const int64_t k = static_cast<int64_t>(w.ne[0]);
    std::vector<float> xh(z(n_tok * k));
    for (int64_t t = 0; t < n_tok; ++t) {
      std::memcpy(xh.data() + t * k, x + t * ldx, z(k) * sizeof(float));
      act_rotate32(xh.data() + t * k, k);
    }
    c.opt.imatrix->add_dense(w.name, xh.data(), n_tok, k, k);
    c.opt.imatrix->mark_rotated(w.name);
  }
  if (const ActFmt f = act_format(c.opt, w); f != ActFmt::none || w.rot32) {
    const int64_t k = static_cast<int64_t>(w.ne[0]);
    std::vector<float> xr(z(n_tok * k));
    for (int64_t t = 0; t < n_tok; ++t) {
      std::memcpy(xr.data() + t * k, x + t * ldx, z(k) * sizeof(float));
      prep_row(xr.data() + t * k, k, f, w.rot32, c.opt.act_rot32);
    }
    matmul(w, xr.data(), n_tok, k, y, ldy, c.threads, c.sc);
    return;
  }
  matmul(w, x, n_tok, ldx, y, ldy, c.threads, c.sc);
}

void full_attention(const Ctx& c, const Qwen35MoeLayer& l, const float* xn, float* out) {
  const Qwen35MoeSpec& s = c.s;
  const int64_t n = c.n, hd = s.head_dim, nh = s.n_head, nkv = s.n_head_kv, e = s.n_embd;
  const int64_t q_w = 2 * hd * nh, kv_w = hd * nkv;
  std::vector<float> qg(z(n * q_w)), k(z(n * kv_w)), v(z(n * kv_w)), att(z(n * hd * nh));
  mm(c, l.wq, xn, n, e, qg.data(), q_w);
  mm(c, l.wk, xn, n, e, k.data(), kv_w);
  mm(c, l.wv, xn, n, e, v.data(), kv_w);
  // per-head norms, then rope at each token's position
  parallel_items(n, c.threads, [&](int64_t t, int) {
    std::vector<float> cs, tmp(z(hd));
    rope_cache(s, t, cs);
    for (int64_t h = 0; h < nh; ++h) {
      float* q = qg.data() + t * q_w + h * 2 * hd;
      rms_norm(q, l.q_norm.data(), tmp.data(), hd, s.rms_eps);
      std::copy(tmp.begin(), tmp.end(), q);
      rope_apply(q, cs, s.n_rot);
    }
    for (int64_t h = 0; h < nkv; ++h) {
      float* kk = k.data() + t * kv_w + h * hd;
      rms_norm(kk, l.k_norm.data(), tmp.data(), hd, s.rms_eps);
      std::copy(tmp.begin(), tmp.end(), kk);
      rope_apply(kk, cs, s.n_rot);
    }
  });
  const float scale = s.attn_scale != 0.0f ? s.attn_scale : 1.0f / std::sqrt(static_cast<float>(hd));
  const int64_t group = nh / nkv;
  // causal attention, one (head, query) per item; then the output is gated by sigmoid(gate) of the same head
  parallel_items(nh * n, c.threads, [&](int64_t item, int tid) {
    const int64_t h = item / n, t = item % n, kvh = h / group;
    Scratch& sc = c.sc[z(tid)];
    if (sc.scores.size() < z(n)) sc.scores.resize(z(n));
    float* sco = sc.scores.data();
    const float* q = qg.data() + t * q_w + h * 2 * hd;
    for (int64_t u = 0; u <= t; ++u) {
      const float* kk = k.data() + u * kv_w + kvh * hd;
      float d = 0.0f;
      for (int64_t i = 0; i < hd; ++i) d += q[i] * kk[i];
      sco[u] = d * scale;
    }
    softmax(sco, t + 1);
    float* o = att.data() + t * hd * nh + h * hd;
    for (int64_t i = 0; i < hd; ++i) o[i] = 0.0f;
    for (int64_t u = 0; u <= t; ++u) {
      const float p = sco[u];
      const float* vv = v.data() + u * kv_w + kvh * hd;
      for (int64_t i = 0; i < hd; ++i) o[i] += p * vv[i];
    }
    const float* gate = q + hd;
    for (int64_t i = 0; i < hd; ++i) o[i] *= sigmoid(gate[i]);
  });
  mm(c, l.wo, att.data(), n, hd * nh, out, e);
}

void delta_net(const Ctx& c, const Qwen35MoeLayer& l, const float* xn, float* out) {
  const Qwen35MoeSpec& s = c.s;
  const int64_t n = c.n, e = s.n_embd;
  const int64_t ch = s.conv_channels(), d_inner = s.ssm_d_inner, ds = s.ssm_d_state, nk = s.ssm_n_group, nv = s.ssm_dt_rank;
  const int64_t taps = s.ssm_d_conv;
  std::vector<float> mixed(z(n * ch)), zg(z(n * d_inner)), beta(z(n * nv)), alpha(z(n * nv));
  mm(c, l.wqkv, xn, n, e, mixed.data(), ch);
  mm(c, l.wgate, xn, n, e, zg.data(), d_inner);
  mm(c, l.wbeta, xn, n, e, beta.data(), nv);
  mm(c, l.walpha, xn, n, e, alpha.data(), nv);
  // beta = sigmoid(.); g = softplus(alpha + dt) * A
  for (int64_t t = 0; t < n; ++t)
    for (int64_t h = 0; h < nv; ++h) {
      beta[z(t * nv + h)] = sigmoid(beta[z(t * nv + h)]);
      alpha[z(t * nv + h)] = softplus(alpha[z(t * nv + h) ] + l.ssm_dt[z(h)]) * l.ssm_a[z(h)];
    }
  // causal depthwise conv over [taps-1 zeros, tokens], then SiLU
  std::vector<float> conv(z(n * ch));
  parallel_items((ch + 255) / 256, c.threads, [&](int64_t blk, int) {
    const int64_t c0 = blk * 256, c1 = std::min(ch, c0 + 256);
    for (int64_t t = 0; t < n; ++t)
      for (int64_t cc = c0; cc < c1; ++cc) {
        float sum = 0.0f;
        for (int64_t kk = 0; kk < taps; ++kk) {
          const int64_t src = t + kk - (taps - 1);
          const float xv = src >= 0 ? mixed[z(src * ch + cc)] : 0.0f;
          sum += xv * l.conv1d[z(cc * taps + kk)];
        }
        conv[z(t * ch + cc)] = silu(sum);
      }
  });
  // q, k: L2-normalized per key head
  for (int64_t t = 0; t < n; ++t)
    for (int64_t h = 0; h < 2 * nk; ++h) l2_norm(conv.data() + t * ch + h * ds, ds, s.rms_eps);
  // the recurrence, one value head per item (value head h reads key head h % nk), zero state at the start
  std::vector<float> core(z(n * d_inner));
  const float qscale = 1.0f / std::sqrt(static_cast<float>(ds));
  parallel_items(nv, c.threads, [&](int64_t hv, int tid) {
    Scratch& sc = c.sc[z(tid)];
    sc.state.assign(z(ds * ds), 0.0f);  // M[j][i]: row j = value index, column i = key index
    if (sc.delta.size() < z(ds)) sc.delta.resize(z(ds));
    float* m = sc.state.data();
    float* delta = sc.delta.data();
    const int64_t hk = hv % nk;
    for (int64_t t = 0; t < n; ++t) {
      const float* q = conv.data() + t * ch + hk * ds;
      const float* kk = conv.data() + t * ch + nk * ds + hk * ds;
      const float* vv = conv.data() + t * ch + 2 * nk * ds + hv * ds;
      const float b = beta[z(t * nv + hv)];
      const float decay = std::exp(alpha[z(t * nv + hv)]);
      for (int64_t i = 0; i < ds * ds; ++i) m[i] *= decay;
      for (int64_t j = 0; j < ds; ++j) {
        float sum = 0.0f;
        for (int64_t i = 0; i < ds; ++i) sum += m[j * ds + i] * kk[i];
        delta[j] = (vv[j] - sum) * b;
      }
      for (int64_t j = 0; j < ds; ++j)
        for (int64_t i = 0; i < ds; ++i) m[j * ds + i] += kk[i] * delta[j];
      float* o = core.data() + t * d_inner + hv * ds;
      for (int64_t j = 0; j < ds; ++j) {
        float sum = 0.0f;
        for (int64_t i = 0; i < ds; ++i) sum += m[j * ds + i] * q[i];
        o[j] = sum * qscale;
      }
    }
  });
  // gated norm per value head: rms_norm(o) * w * silu(z)
  std::vector<float> tmp(z(ds));
  for (int64_t t = 0; t < n; ++t)
    for (int64_t h = 0; h < nv; ++h) {
      float* o = core.data() + t * d_inner + h * ds;
      const float* zz = zg.data() + t * d_inner + h * ds;
      rms_norm(o, l.ssm_norm.data(), tmp.data(), ds, s.rms_eps);
      for (int64_t i = 0; i < ds; ++i) o[i] = tmp[z(i)] * silu(zz[i]);
    }
  mm(c, l.ssm_out, core.data(), n, d_inner, out, e);
}

void moe(const Ctx& c, const Qwen35MoeLayer& l, const float* xn, float* out, float* deferred = nullptr) {
  const Qwen35MoeSpec& s = c.s;
  const int64_t n = c.n, e = s.n_embd, ne = s.n_expert, nu = s.n_expert_used, ff = s.n_ff_exp, ffs = s.n_ff_shexp;
  // routing: softmax over every expert, the top n_expert_used by probability (ties: lower index first), their
  // probabilities renormalized to sum 1 (the sum clamped below at 2^-14)
  std::vector<float> probs(z(n * ne));
  mm(c, l.gate_inp, xn, n, e, probs.data(), ne);
  std::vector<int32_t> sel(z(n * nu));
  std::vector<float> wsel(z(n * nu));
  std::vector<int32_t> order(z(ne));
  for (int64_t t = 0; t < n; ++t) {
    float* p = probs.data() + t * ne;
    softmax(p, ne);
    for (int64_t i = 0; i < ne; ++i) order[z(i)] = static_cast<int32_t>(i);
    // one more than picked, so the first unpicked expert is known (the picks themselves are the same)
    const int64_t nsort = std::min<int64_t>(nu + 1, ne);
    std::partial_sort(order.begin(), order.begin() + nsort, order.end(), [p](int32_t a, int32_t b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
    if (c.opt.route) {
      RouteRecord& rr = c.opt.route[z(t * static_cast<int64_t>(s.n_layer) + c.il)];
      rr.p_last = p[order[z(nu - 1)]];
      rr.p_next = nsort > nu ? p[order[z(nu)]] : 0.0f;
    }
    if (c.opt.swap_margin > 0.0f && nsort > nu && !c.opt.hot.empty()) {  // Block M: near-tie swap toward a resident expert
      const std::uint8_t* hl = c.opt.hot.data() + static_cast<int64_t>(c.il) * ne;
      const int32_t last = order[z(nu - 1)], next = order[z(nu)];
      if (!hl[last]) {
        if (c.opt.m_stats) ++c.opt.m_stats[1];
        if (hl[next] && p[last] - p[next] < c.opt.swap_margin) {
          std::swap(order[z(nu - 1)], order[z(nu)]);
          if (c.opt.m_stats) ++c.opt.m_stats[0];
        }
      }
    }
    float sum = 0.0f;
    for (int64_t k = 0; k < nu; ++k) {
      sel[z(t * nu + k)] = order[z(k)];
      wsel[z(t * nu + k)] = p[order[z(k)]];
      sum += p[order[z(k)]];
    }
    sum = std::max(sum, 6.103515625e-5f);
    for (int64_t k = 0; k < nu; ++k) {
      wsel[z(t * nu + k)] /= sum;
      if (s.expert_weights_scale != 0.0f && s.expert_weights_scale != 1.0f) wsel[z(t * nu + k)] *= s.expert_weights_scale;
    }
    if (c.opt.route) {
      RouteRecord& rr = c.opt.route[z(t * static_cast<int64_t>(s.n_layer) + c.il)];
      for (int64_t k = 0; k < nu; ++k) {
        rr.sel[k] = static_cast<std::int16_t>(sel[z(t * nu + k)]);
        rr.w[k] = wsel[z(t * nu + k)];
      }
    }
  }
  if (c.opt.expert_picks || c.opt.expert_weight) {
    const int64_t base = static_cast<int64_t>(c.il) * ne;
    for (int64_t i = 0; i < n * nu; ++i) {
      if (c.opt.expert_picks) ++c.opt.expert_picks[base + sel[z(i)]];
      if (c.opt.expert_weight) c.opt.expert_weight[base + sel[z(i)]] += static_cast<double>(wsel[z(i)]);
    }
  }
  // tokens per expert, in token order
  std::vector<std::vector<int64_t>> slots(z(ne));  // entries t * nu + k
  for (int64_t t = 0; t < n; ++t)
    for (int64_t k = 0; k < nu; ++k) slots[z(sel[z(t * nu + k)])].push_back(t * nu + k);
  std::vector<float> ys(z(n * nu * e));  // each (token, slot)'s weighted expert output
  // importance matrix of the experts, in the reference's order: slot by slot, then token by token
  std::vector<float> hs(c.opt.imatrix ? z(n * nu * ff) : 0);
  if (c.opt.imatrix) {
    ImatrixEntry& eg = c.opt.imatrix->entry(l.gate_exps.name, e, ne);
    ImatrixEntry& eu = c.opt.imatrix->entry(l.up_exps.name, e, ne);
    for (int64_t k = 0; k < nu; ++k)
      for (int64_t t = 0; t < n; ++t) {
        ImatrixCollector::add_row(eg, sel[z(t * nu + k)], xn + t * e);
        ImatrixCollector::add_row(eu, sel[z(t * nu + k)], xn + t * e);
      }
  }
  std::vector<int64_t> busy;
  for (int64_t x = 0; x < ne; ++x)
    if (!slots[z(x)].empty()) busy.push_back(x);
  parallel_items(static_cast<int64_t>(busy.size()), c.threads, [&](int64_t bi, int tid) {
    const int64_t ex = busy[z(bi)];
    const std::vector<int64_t>& list = slots[z(ex)];
    const int64_t m = static_cast<int64_t>(list.size());
    Scratch& sc = c.sc[z(tid)];
    sc.xe.resize(z(m * e));
    sc.ge.resize(z(m * ff));
    sc.ue.resize(z(m * ff));
    sc.ye.resize(z(m * e));
    for (int64_t r = 0; r < m; ++r) std::memcpy(sc.xe.data() + r * e, xn + (list[z(r)] / nu) * e, z(e) * sizeof(float));
    // one input feeds both gate and up, rounded once (in llama.cpp both are the same type, so the same format)
    const ActFmt fgu = act_format(c.opt, l.gate_exps) != ActFmt::none ? act_format(c.opt, l.gate_exps) : act_format(c.opt, l.up_exps);
    if (fgu != ActFmt::none || l.gate_exps.rot32)
      for (int64_t r = 0; r < m; ++r) prep_row(sc.xe.data() + r * e, e, fgu, l.gate_exps.rot32, c.opt.act_rot32);
    matmul_rows(l.gate_exps, static_cast<uint64_t>(ex), 0, ff, sc.xe.data(), m, e, sc.ge.data(), ff, sc);
    matmul_rows(l.up_exps, static_cast<uint64_t>(ex), 0, ff, sc.xe.data(), m, e, sc.ue.data(), ff, sc);
    for (int64_t i = 0; i < m * ff; ++i) sc.ge[z(i)] = silu(sc.ge[z(i)]) * sc.ue[z(i)];
    if (c.opt.neuron_skip > 0.0) {  // Block M: oracle neuron skipping, the smallest |h| per token set to zero
      const auto kz = static_cast<int64_t>(c.opt.neuron_skip * static_cast<double>(ff));
      if (kz > 0) {
        std::vector<float> mag(z(ff));
        for (int64_t r = 0; r < m; ++r) {
          float* h = sc.ge.data() + r * ff;
          for (int64_t i = 0; i < ff; ++i) mag[z(i)] = std::fabs(h[i]);
          std::nth_element(mag.begin(), mag.begin() + (kz - 1), mag.end());
          const float thr = mag[z(kz - 1)];
          int64_t zeroed = 0;
          for (int64_t i = 0; i < ff && zeroed < kz; ++i)
            if (std::fabs(h[i]) <= thr) { h[i] = 0.0f; ++zeroed; }
        }
        if (c.opt.m_stats) {
          std::atomic_ref<std::uint64_t>(c.opt.m_stats[2]).fetch_add(static_cast<std::uint64_t>(kz * m));
          std::atomic_ref<std::uint64_t>(c.opt.m_stats[3]).fetch_add(static_cast<std::uint64_t>(ff * m));
        }
      }
    }
    if (!hs.empty())
      for (int64_t r = 0; r < m; ++r) std::memcpy(hs.data() + list[z(r)] * ff, sc.ge.data() + r * ff, z(ff) * sizeof(float));
    if (const ActFmt fd = act_format(c.opt, l.down_exps); fd != ActFmt::none || l.down_exps.rot32)
      for (int64_t r = 0; r < m; ++r) prep_row(sc.ge.data() + r * ff, ff, fd, l.down_exps.rot32, c.opt.act_rot32);
    matmul_rows(l.down_exps, static_cast<uint64_t>(ex), 0, e, sc.ge.data(), m, ff, sc.ye.data(), e, sc);
    for (int64_t r = 0; r < m; ++r) {
      const int64_t slot = list[z(r)];
      const float wgt = wsel[z(slot)];
      float* dst = ys.data() + slot * e;
      const float* src = sc.ye.data() + r * e;
      for (int64_t i = 0; i < e; ++i) dst[i] = src[i] * wgt;
    }
  });
  if (c.opt.imatrix) {
    ImatrixEntry& ed = c.opt.imatrix->entry(l.down_exps.name, ff, ne);
    for (int64_t k = 0; k < nu; ++k)
      for (int64_t t = 0; t < n; ++t) ImatrixCollector::add_row(ed, sel[z(t * nu + k)], hs.data() + (t * nu + k) * ff);
  }
  // shared expert, gated per token by sigmoid(gate_inp_shexp . x)
  std::vector<float> gs(z(n * ffs)), us(z(n * ffs)), sh(z(n * e));
  mm(c, l.gate_shexp, xn, n, e, gs.data(), ffs);
  mm(c, l.up_shexp, xn, n, e, us.data(), ffs);
  for (int64_t i = 0; i < n * ffs; ++i) gs[z(i)] = silu(gs[z(i)]) * us[z(i)];
  mm(c, l.down_shexp, gs.data(), n, ffs, sh.data(), e);
  if (c.opt.imatrix) c.opt.imatrix->add_dense("blk." + std::to_string(c.il) + ".ffn_gate_inp_shexp.weight", xn, n, e, e);
  for (int64_t t = 0; t < n; ++t) {
    float g = 0.0f;
    for (int64_t i = 0; i < e; ++i) g += l.gate_inp_shexp[z(i)] * xn[t * e + i];
    g = sigmoid(g);
    float* o = out + t * e;
    const float* y0 = ys.data() + (t * nu) * e;
    // Block M expert deferral: the last d slots (lowest probability) go to `deferred`, added one layer later
    const int64_t keep = deferred ? std::max<int64_t>(1, nu - c.opt.defer_experts) : nu;
    for (int64_t i = 0; i < e; ++i) o[i] = y0[i];
    for (int64_t k = 1; k < keep; ++k) {
      const float* yk = ys.data() + (t * nu + k) * e;
      for (int64_t i = 0; i < e; ++i) o[i] += yk[i];
    }
    if (deferred) {
      float* dd = deferred + t * e;
      for (int64_t i = 0; i < e; ++i) dd[i] = 0.0f;
      for (int64_t k = keep; k < nu; ++k) {
        const float* yk = ys.data() + (t * nu + k) * e;
        for (int64_t i = 0; i < e; ++i) dd[i] += yk[i];
      }
    }
    const float* shr = sh.data() + t * e;
    for (int64_t i = 0; i < e; ++i) o[i] += shr[i] * g;
  }
}

}  // namespace

void act_rotate32(float* x, std::int64_t n) { quant::hadamard32(x, n); }

Result<Qwen35MoeWeights> bind_qwen35moe(const io::GgufReader& r, const std::uint8_t* base, const Qwen35MoeSpec& s) {
  using R = Result<Qwen35MoeWeights>;
  Qwen35MoeWeights w;
  const uint64_t e = s.n_embd, nv = s.n_vocab;
  Status st = bind_view(r, base, "token_embd.weight", {e, nv}, w.tok_embd);
  if (!st) return R::fail(st);
  st = bind_view(r, base, "output.weight", {e, nv}, w.output, false);
  if (!st) return R::fail(st);
  if (!w.output.present()) w.output = w.tok_embd;  // tied embeddings
  st = bind_vector(r, base, "output_norm.weight", {e}, w.output_norm);
  if (!st) return R::fail(st);
  const uint64_t hd = s.head_dim, nh = s.n_head, nkv = s.n_head_kv;
  const uint64_t ch = s.conv_channels(), di = s.ssm_d_inner, ds = s.ssm_d_state, nvh = s.ssm_dt_rank;
  const uint64_t ne = s.n_expert, ff = s.n_ff_exp, ffs = s.n_ff_shexp;
  w.layers.resize(s.n_layer);
  for (std::uint32_t il = 0; il < s.n_layer; ++il) {
    Qwen35MoeLayer& l = w.layers[il];
    l.recurrent = s.is_recurrent(il);
    const std::string p = "blk." + std::to_string(il) + ".";
    std::vector<Status> all;
    all.push_back(bind_vector(r, base, p + "attn_norm.weight", {e}, l.attn_norm));
    all.push_back(bind_vector(r, base, p + "post_attention_norm.weight", {e}, l.post_norm));
    if (!l.recurrent) {
      if (r.find_tensor(p + "attn_qkv.weight"))
        return R::fail(Status::fail(Code::unsupported, "a fused q/k/v projection on a full-attention layer is not read yet", 0, p + "attn_qkv.weight"));
      all.push_back(bind_view(r, base, p + "attn_q.weight", {e, 2 * hd * nh}, l.wq));
      all.push_back(bind_view(r, base, p + "attn_k.weight", {e, hd * nkv}, l.wk));
      all.push_back(bind_view(r, base, p + "attn_v.weight", {e, hd * nkv}, l.wv));
      all.push_back(bind_view(r, base, p + "attn_output.weight", {hd * nh, e}, l.wo));
      all.push_back(bind_vector(r, base, p + "attn_q_norm.weight", {hd}, l.q_norm));
      all.push_back(bind_vector(r, base, p + "attn_k_norm.weight", {hd}, l.k_norm));
    } else {
      all.push_back(bind_view(r, base, p + "attn_qkv.weight", {e, ch}, l.wqkv));
      all.push_back(bind_view(r, base, p + "attn_gate.weight", {e, di}, l.wgate));
      all.push_back(bind_view(r, base, p + "ssm_beta.weight", {e, nvh}, l.wbeta));
      all.push_back(bind_view(r, base, p + "ssm_alpha.weight", {e, nvh}, l.walpha));
      all.push_back(bind_view(r, base, p + "ssm_out.weight", {di, e}, l.ssm_out));
      all.push_back(bind_vector(r, base, p + "ssm_conv1d.weight", {s.ssm_d_conv, ch}, l.conv1d));
      all.push_back(bind_vector(r, base, p + "ssm_dt.bias", {nvh}, l.ssm_dt));
      all.push_back(bind_vector(r, base, p + "ssm_a", {nvh}, l.ssm_a));
      all.push_back(bind_vector(r, base, p + "ssm_norm.weight", {ds}, l.ssm_norm));
    }
    if (r.find_tensor(p + "ffn_gate_up_exps.weight"))
      return R::fail(Status::fail(Code::unsupported, "fused expert gate/up tensors are not read yet", 0, p + "ffn_gate_up_exps.weight"));
    all.push_back(bind_view(r, base, p + "ffn_gate_inp.weight", {e, ne}, l.gate_inp));
    all.push_back(bind_view(r, base, p + "ffn_up_exps.weight", {e, ff, ne}, l.up_exps));
    all.push_back(bind_view(r, base, p + "ffn_gate_exps.weight", {e, ff, ne}, l.gate_exps));
    all.push_back(bind_view(r, base, p + "ffn_down_exps.weight", {ff, e, ne}, l.down_exps));
    all.push_back(bind_view(r, base, p + "ffn_up_shexp.weight", {e, ffs}, l.up_shexp));
    all.push_back(bind_view(r, base, p + "ffn_gate_shexp.weight", {e, ffs}, l.gate_shexp));
    all.push_back(bind_view(r, base, p + "ffn_down_shexp.weight", {ffs, e}, l.down_shexp));
    all.push_back(bind_vector(r, base, p + "ffn_gate_inp_shexp.weight", {e}, l.gate_inp_shexp));
    for (const Status& x : all)
      if (!x) return R::fail(x);
  }
  // weights the file stores rotated (pelias.rot32.tensors, written by `pelias quantize --rotate32`)
  for (const io::GgufKv& kv : r.kv()) {
    if (kv.key != "pelias.rot32.tensors") continue;
    if (kv.value.type != io::GgufValueType::array || kv.value.array_type != io::GgufValueType::string)
      return R::fail(Status::fail(Code::corrupt, "pelias.rot32.tensors is not a list of tensor names"));
    std::vector<std::string> names;
    if (Status as = r.array_strings(kv.value, names); !as) return R::fail(as);
    std::vector<WeightView*> views{&w.output};
    for (Qwen35MoeLayer& l : w.layers)
      for (WeightView* p : {&l.wq, &l.wk, &l.wv, &l.wo, &l.wqkv, &l.wgate, &l.wbeta, &l.walpha, &l.ssm_out, &l.up_exps, &l.gate_exps, &l.down_exps,
                            &l.up_shexp, &l.gate_shexp, &l.down_shexp})
        views.push_back(p);
    for (const std::string& nm : names) {
      WeightView* hit = nullptr;
      for (WeightView* p : views)
        if (p->present() && p->name == nm) hit = p;
      if (!hit) {
        // a block the reference does not run (the MTP block) may be stored rotated; anything else must be a product here
        const bool mtp = nm.rfind("blk." + std::to_string(s.n_layer) + ".", 0) == 0;
        if (mtp) continue;
        return R::fail(Status::fail(Code::unsupported, "the file marks a tensor as rotated that is not a product weight of this model", 0, nm));
      }
      if (hit == &w.output && w.output.data == w.tok_embd.data)
        return R::fail(Status::fail(Code::unsupported, "a rotated output layer tied to the embedding table", 0, nm));
      if (hit->ne[0] % 32 != 0) return R::fail(Status::fail(Code::corrupt, "a rotated tensor's row length is not a multiple of 32", 0, nm));
      hit->rot32 = true;
    }
    for (const Qwen35MoeLayer& l : w.layers)
      if (l.gate_exps.rot32 != l.up_exps.rot32)
        return R::fail(Status::fail(Code::unsupported, "expert gate and up share one input, so both or neither must be rotated", 0, l.gate_exps.name));
  }
  return R::good(std::move(w));
}

Status forward_reference(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, const std::int32_t* tokens, int n_tokens, int out_begin,
                         int out_end, float* logits, const ForwardOptions& opt) {
  if (n_tokens <= 0 || out_begin < 0 || out_end > n_tokens || out_begin >= out_end)
    return Status::fail(Code::invalid_argument, "the token count or the output range is empty or out of range");
  if (w.layers.size() != s.n_layer) return Status::fail(Code::invalid_argument, "the weights were bound for another spec");
  if (Status st = check_act_rounding(opt, w); !st) return st;
  if (opt.act_gpu256 && (!opt.act_int8_gpu || opt.act_bits != 8))
    return Status::fail(Code::invalid_argument, "the 256-value int8 format is an 8-bit gpu-path format (--act-int8-path gpu256)");
  if (opt.act_bits != 8 && (opt.act_bits != 4 || !opt.act_int8_gpu))
    return Status::fail(Code::invalid_argument, "4-bit activations need the gpu path (--act-int8-path gpu); only 8 and 4 bits exist");
  if (opt.neuron_skip < 0.0 || opt.neuron_skip >= 1.0) return Status::fail(Code::invalid_argument, "neuron skipping takes a fraction in [0, 1)");
  if (opt.swap_margin > 0.0f && opt.hot.size() != static_cast<std::size_t>(s.n_layer) * s.n_expert)
    return Status::fail(Code::invalid_argument, "the near-tie swap needs a hot set of n_layer x n_expert entries");
  if (opt.defer_experts < 0 || opt.defer_experts >= static_cast<int>(s.n_expert_used))
    return Status::fail(Code::invalid_argument, "expert deferral must keep at least one expert immediate");
  if (opt.route && (s.n_expert_used > 8 || s.n_expert > 32767))
    return Status::fail(Code::unsupported, "the routing record holds at most 8 picked experts of at most 32767");
  if (opt.imatrix) {  // rotated dense weights are collected in the rotated basis (mm); rotated routed experts are not built
    bool exp_rot = false;
    for (const Qwen35MoeLayer& l : w.layers) exp_rot = exp_rot || l.up_exps.rot32 || l.gate_exps.rot32 || l.down_exps.rot32;
    if (exp_rot) return Status::fail(Code::unsupported, "collecting an importance matrix on a file with rotated routed experts is not built (game plan Block J: experts stay unrotated)");
  }
  const int threads = opt.threads > 0 ? opt.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  const int64_t n = n_tokens, e = s.n_embd;
  std::vector<Scratch> sc(z(threads));
  Ctx c{s, threads, sc, n, opt};

  // embedding rows
  std::vector<float> x(z(n * e)), xn(z(n * e)), y(z(n * e));
  std::vector<float> ydef_now(opt.defer_experts > 0 ? z(n * e) : 0), ydef_prev(opt.defer_experts > 0 ? z(n * e) : 0);
  for (int64_t t = 0; t < n; ++t) {
    const std::int32_t tok = tokens[t];
    if (tok < 0 || static_cast<uint64_t>(tok) >= s.n_vocab) return Status::fail(Code::invalid_argument, "a token id is outside the vocabulary");
    (void)quant::dequantize_row(static_cast<quant::Type>(w.tok_embd.type), w.tok_embd.data + static_cast<uint64_t>(tok) * w.tok_embd.row_bytes,
                                x.data() + t * e, e);
  }
  for (std::uint32_t il = 0; il < s.n_layer; ++il) {
    const Qwen35MoeLayer& l = w.layers[il];
    c.il = il;
    for (int64_t t = 0; t < n; ++t) rms_norm(x.data() + t * e, l.attn_norm.data(), xn.data() + t * e, e, s.rms_eps);
    if (l.recurrent) delta_net(c, l, xn.data(), y.data());
    else full_attention(c, l, xn.data(), y.data());
    for (int64_t i = 0; i < n * e; ++i) x[z(i)] += y[z(i)];
    for (int64_t t = 0; t < n; ++t) rms_norm(x.data() + t * e, l.post_norm.data(), xn.data() + t * e, e, s.rms_eps);
    if (opt.defer_experts > 0) {  // Block M: this layer's deferred experts wait; the previous layer's arrive now
      moe(c, l, xn.data(), y.data(), ydef_now.data());
      for (int64_t i = 0; i < n * e; ++i) x[z(i)] += y[z(i)] + ydef_prev[z(i)];
      std::swap(ydef_now, ydef_prev);
    } else {
      moe(c, l, xn.data(), y.data());
      for (int64_t i = 0; i < n * e; ++i) x[z(i)] += y[z(i)];
    }
    if (opt.route && il + 1 < s.n_layer) {
      // pre-gating prediction: the next layer's router on this layer's output, through the next layer's norm
      const Qwen35MoeLayer& nl = w.layers[il + 1];
      const int64_t ne = s.n_expert, nu = s.n_expert_used;
      for (int64_t t = 0; t < n; ++t) rms_norm(x.data() + t * e, nl.post_norm.data(), xn.data() + t * e, e, s.rms_eps);
      std::vector<float> pr(z(n * ne));
      mm(c, nl.gate_inp, xn.data(), n, e, pr.data(), ne);
      std::vector<int32_t> ord(z(ne));
      for (int64_t t = 0; t < n; ++t) {
        const float* p = pr.data() + t * ne;
        for (int64_t i = 0; i < ne; ++i) ord[z(i)] = static_cast<int32_t>(i);
        std::partial_sort(ord.begin(), ord.begin() + nu, ord.end(), [p](int32_t a, int32_t b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
        RouteRecord& rr = opt.route[z(t * static_cast<int64_t>(s.n_layer) + il)];
        for (int64_t k = 0; k < nu; ++k) rr.pred[k] = static_cast<std::int16_t>(ord[z(k)]);
      }
    }
  }
  if (opt.defer_experts > 0)  // the last layer's deferred experts arrive before the output norm
    for (int64_t i = 0; i < n * e; ++i) x[z(i)] += ydef_prev[z(i)];
  if (!logits) return Status::good();  // the activations were all the caller wanted (importance matrix)
  const int64_t m = out_end - out_begin;
  for (int64_t t = 0; t < m; ++t) rms_norm(x.data() + (out_begin + t) * e, w.output_norm.data(), xn.data() + t * e, e, s.rms_eps);
  mm(c, w.output, xn.data(), m, e, logits, s.n_vocab);  // through mm(), so --act-int8 reaches the output layer too
  return Status::good();
}

}  // namespace pelias::model

namespace pelias::model {
// value for value what q8k_warp in tools/probes/batch_c/kernels_c.cu (int8 kernels v7-v12) gives the GEMM
void act_round_q8_256(float* x, std::int64_t n) {
  for (std::int64_t i = 0; i < n; i += 256) {
    float amax = 0.0f;
    for (int j = 0; j < 256; ++j) amax = std::max(amax, std::fabs(x[i + j]));
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / scale;
    for (int j = 0; j < 256; ++j) {
      const float q = std::min(127.0f, std::max(-127.0f, std::nearbyint(x[i + j] * inv)));
      x[i + j] = scale * q;
    }
  }
}
}  // namespace pelias::model
