// provenance: original (the computation READ-SRC from llama.cpp src/models/qwen35moe.cpp, src/llama-graph.cpp
//   build_moe_ffn / build_ffn / build_norm, src/models/models.h build_gdn_l2_norm and ggml/src/ggml-cpu/ops.cpp
//   (rms_norm, rope multi / IMRoPE, ssm_conv, gated_delta_net, softplus) at commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a, read as a specification; no code copied)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "model/arch_qwen35moe.h"

#include <algorithm>
#include <bit>
#include <atomic>
#include <cmath>
#include <cstring>
#include <map>
#include <thread>

#include "io/tier_build.h"
#include "kern/cpu_dot.h"

#include "quant/block_types.h"
#include "quant/planes.h"
#include "quant/quant_cpu.h"
#include "quant/tdecode.h"

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

// Routed experts stored with gate and up fused (newer converters, [n_embd, 2 n_ff, n_expert]: each expert's n_ff gate rows,
// then its n_ff up rows): two views whose experts are slots into the one tensor.
Status bind_gate_up_exps(const io::GgufReader& r, const std::uint8_t* base, const std::string& name, uint64_t ne0, uint64_t ff, uint64_t n_exp,
                         WeightView& gate, WeightView& up) {
  WeightView f;
  if (Status st = bind_view(r, base, name, {ne0, 2 * ff, n_exp}, f); !st) return st;
  for (WeightView* v : {&gate, &up}) {
    *v = f;
    v->ne[1] = ff;
    v->slots.resize(static_cast<std::size_t>(n_exp));
  }
  for (uint64_t ex = 0; ex < n_exp; ++ex) {
    const std::uint8_t* d = f.data + ex * 2 * ff * f.row_bytes;
    gate.slots[static_cast<std::size_t>(ex)] = {d, f.type, f.row_bytes};
    up.slots[static_cast<std::size_t>(ex)] = {d + ff * f.row_bytes, f.type, f.row_bytes};
  }
  up.data = f.data + ff * f.row_bytes;  // expert 0's up rows (the slots carry every expert)
  return Status::good();
}

// Routed experts: the stock tensor, or a usage-tiered weight (io/tier_build.h) read as one view whose experts each keep
// their own type and rows.
Status bind_experts(const io::GgufReader& r, const std::uint8_t* base, const std::string& name, uint64_t ne0, uint64_t ne1, uint64_t n_exp,
                    WeightView& out) {
  const io::GgufValue* tv = r.find(io::tier_types_key(name));
  if (!tv) return bind_view(r, base, name, {ne0, ne1, n_exp}, out);
  if (r.find_tensor(name)) return Status::fail(Code::corrupt, "a tiered weight is also stored whole", 0, name);
  std::vector<std::int64_t> types;
  if (tv->type != io::GgufValueType::array) return Status::fail(Code::corrupt, "the tier's types are not a list", 0, name);
  if (Status s = r.array_ints(*tv, types); !s) return s;
  if (types.size() != n_exp) return Status::fail(Code::corrupt, "the tier names a type for " + std::to_string(types.size()) + " experts, the layer has " + std::to_string(n_exp), 0, name);
  std::map<uint32_t, uint64_t> count;
  for (const std::int64_t t : types) {
    if (t < 0 || t > 0xFFFF) return Status::fail(Code::corrupt, "the tier names an invalid type id", 0, name);
    ++count[static_cast<uint32_t>(t)];
  }
  std::map<uint32_t, WeightView> part;
  for (const auto& [t, n] : count) {
    WeightView v;
    const std::string pn = io::tier_part_name(name, t);
    if (Status s = bind_view(r, base, pn, {ne0, ne1, n}, v); !s) return s;
    part[t] = v;
  }
  out = WeightView{};
  out.slots.resize(n_exp);
  std::map<uint32_t, uint64_t> next;
  for (uint64_t e = 0; e < n_exp; ++e) {
    const auto t = static_cast<uint32_t>(types[e]);
    const WeightView& v = part[t];
    out.slots[e] = ExpertSlot{v.data + next[t]++ * ne1 * v.row_bytes, t, v.row_bytes};
  }
  uint32_t most = count.begin()->first;
  for (const auto& [t, n] : count)
    if (n > count[most]) most = t;
  out.data = part[most].data;
  out.type = most;
  out.row_bytes = part[most].row_bytes;
  out.ne[0] = ne0;
  out.ne[1] = ne1;
  out.ne[2] = n_exp;
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
  bool planes = true;                    // ForwardOptions::planes for this call
  bool fast = false;                     // ForwardOptions::fast_cpu: rows through kern::dot_row
  // with fast: the inputs of this product in the engine's int8 format (gpu256x8, no simulated rotation), one per input row;
  // the rows then go through kern::dot_row_q8 (integer products per 32-block)
  const std::vector<kern::ActQ8x32>* acts = nullptr;
  std::vector<kern::ActQ8x32> own_acts;  // the per-thread inputs of an expert group
};

void load_tile(const WeightView& w, uint64_t first_row, int nr, int64_t k, Scratch& sc) {
  if (sc.row.size() < z(k)) sc.row.resize(z(k));
  if (sc.wt.size() < z(k * kTile)) sc.wt.resize(z(k * kTile));
  float* wt = sc.wt.data();
  for (int r = 0; r < nr; ++r) {
    // the type and the row length were checked when the tensor was bound
    const uint64_t row = first_row + static_cast<uint64_t>(r);
    if (w.slots.empty()) {
      (void)quant::dequantize_row(static_cast<quant::Type>(w.type), w.data + row * w.row_bytes, sc.row.data(), k);
    } else {  // a usage-tiered expert: its own type and rows
      const ExpertSlot& s = w.slots[row / w.ne[1]];
      (void)quant::dequantize_row(static_cast<quant::Type>(s.type), s.data + (row % w.ne[1]) * s.row_bytes, sc.row.data(), k);
    }
    if (sc.planes && w.planes) {  // residual sign planes: the row as the original weights, to the planes read
      if (w.plane_mask)  // block-selective: only the selected blocks of this row carry planes
        quant::plane_add_row_sparse(w.planes + static_cast<uint64_t>(w.plane_off[row]) * 34u * static_cast<uint64_t>(w.plane_k_stored),
                                    w.plane_mask + row * w.plane_mask_bytes, k, w.plane_k_stored, w.plane_k, sc.row.data());
      else
        quant::plane_add_row(w.planes + row * w.plane_row_bytes, k, w.plane_k_stored, w.plane_k, sc.row.data());
    }
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
// The fast CPU path (GP3-10): each row straight from its storage through kern::dot_row, planes included. The same products
// as load_tile + tile_dot; the sum runs in 32 lanes with fused multiply-adds, so outputs match within the rung-1 tolerance.
void fast_rows(const WeightView& w, uint64_t e, int64_t o_begin, int64_t o_end, const float* x, int64_t n_tok, int64_t ldx, float* y,
               int64_t ldy, const Scratch& sc) {
  const int64_t k = static_cast<int64_t>(w.ne[0]);
  const uint64_t base_row = e * w.ne[1];
  for (int64_t o = o_begin; o < o_end; ++o) {
    const uint64_t row = base_row + static_cast<uint64_t>(o);
    quant::Type t;
    const std::uint8_t* p;
    if (w.slots.empty()) {
      t = static_cast<quant::Type>(w.type);
      p = w.data + row * w.row_bytes;
    } else {
      const ExpertSlot& s = w.slots[row / w.ne[1]];
      t = static_cast<quant::Type>(s.type);
      p = s.data + (row % w.ne[1]) * s.row_bytes;
    }
    kern::RowPlanes rp;
    const kern::RowPlanes* pp = nullptr;
    if (sc.planes && w.planes) {
      if (w.plane_mask)
        rp = {w.planes + static_cast<uint64_t>(w.plane_off[row]) * 34u * static_cast<uint64_t>(w.plane_k_stored),
              w.plane_mask + row * w.plane_mask_bytes, w.plane_k_stored, w.plane_k};
      else
        rp = {w.planes + row * w.plane_row_bytes, nullptr, w.plane_k_stored, w.plane_k};
      pp = &rp;
    }
    for (int64_t t0 = 0; t0 < n_tok; t0 += 4) {
      const int nt = static_cast<int>(std::min<int64_t>(4, n_tok - t0));
      float out[4];
      if (sc.acts && static_cast<int64_t>(sc.acts->size()) == n_tok) {
        const kern::ActQ8x32* as[4];
        for (int tt = 0; tt < 4; ++tt) as[tt] = &(*sc.acts)[z(std::min(t0 + tt, n_tok - 1))];
        kern::dot_row_q8(t, p, k, pp, as, nt, out);
        for (int tt = 0; tt < nt; ++tt) y[(t0 + tt) * ldy + o] = out[tt];
        continue;
      }
      const float* xs[4];
      for (int tt = 0; tt < 4; ++tt) xs[tt] = x + std::min(t0 + tt, n_tok - 1) * ldx;
      kern::dot_row(t, p, k, pp, xs, nt, out);
      for (int tt = 0; tt < nt; ++tt) y[(t0 + tt) * ldy + o] = out[tt];
    }
  }
}

void matmul_rows(const WeightView& w, uint64_t e, int64_t o_begin, int64_t o_end, const float* x, int64_t n_tok, int64_t ldx, float* y,
                 int64_t ldy, Scratch& sc) {
  const int64_t k = static_cast<int64_t>(w.ne[0]);
  const uint64_t base_row = e * w.ne[1];
  if (sc.fast) {
    fast_rows(w, e, o_begin, o_end, x, n_tok, ldx, y, ldy, sc);
    return;
  }
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
  std::uint32_t ai = 0;  // full-attention layers run before this one (ForwardOptions::qk_dump)
  int64_t pos0 = 0;               // GP3-07: the position of the first token of this call
  DecodeState* st = nullptr;      // GP3-07: the carried state (null: a fresh sequence from position 0)
};

// the 8-bit activation format the inputs of `w` are rounded to (ForwardOptions::act_int8), or none; float weights never
// are, as in the reference
enum class ActFmt { none, q8_0, q8_1, q8_k, q4_32, q8_256, q8_256x8, q8_256e };

ActFmt act_format(const ForwardOptions& o, const WeightView& w) {
  if (o.act_int8.empty() || !w.present()) return ActFmt::none;
  const auto t = static_cast<quant::Type>(w.type);
  if (t == quant::Type::f32 || t == quant::Type::f16 || t == quant::Type::bf16) return ActFmt::none;
  bool hit = false;
  for (const std::string& g : o.act_int8)
    if (g == "all" || w.name.find(g) != std::string::npos) hit = true;
  if (!hit) return ActFmt::none;
  if (o.act_int8_gpu) return o.act_bits == 4 ? ActFmt::q4_32 : (o.act_gpu256 ? (o.act_gpu256_sub == 1 ? ActFmt::q8_256x8 : o.act_gpu256_sub == 2 ? ActFmt::q8_256e : ActFmt::q8_256) : ActFmt::q8_1);
  switch (t) {
    case quant::Type::q4_0: case quant::Type::q5_0: case quant::Type::q8_0: case quant::Type::iq4_nl: case quant::Type::mxfp4:
      return ActFmt::q8_0;
    case quant::Type::q4_1: case quant::Type::q5_1:
      return ActFmt::q8_1;
    default:
      return ActFmt::q8_k;
  }
}

int act_block(ActFmt f) { return f == ActFmt::q8_k || f == ActFmt::q8_256 || f == ActFmt::q8_256x8 || f == ActFmt::q8_256e ? 256 : 32; }

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
  if (f == ActFmt::q8_256x8) {
    act_round_q8_256x8(x, n);
    return;
  }
  if (f == ActFmt::q8_256e) {
    act_round_q8_256e(x, n);
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

void h_gate_impl(const std::uint8_t* data, std::uint64_t row_bytes, int64_t n_out, int64_t k, const float* x, int64_t n_tok,
                 int64_t ldx, float* y, int64_t ldy, double f, int threads) {
  const int64_t nb = k / 32;
  const int64_t keep = std::min<int64_t>(nb, std::max<int64_t>(0, static_cast<int64_t>(std::llround(f * static_cast<double>(nb)))));
  if (keep == nb) return;
  std::vector<float> X(z(n_tok * nb));  // sum of x^2 per token and 32-block
  for (int64_t t = 0; t < n_tok; ++t)
    for (int64_t j = 0; j < nb; ++j) {
      float s2 = 0.0f;
      for (int i = 0; i < 32; ++i) s2 += x[t * ldx + 32 * j + i] * x[t * ldx + 32 * j + i];
      X[z(t * nb + j)] = s2;
    }
  parallel_items(n_out, threads, [&](int64_t r, int) {
    const std::uint8_t* row = data + static_cast<uint64_t>(r) * row_bytes;
    std::vector<float> s(z(nb)), hm(z(k)), score(z(nb));
    std::vector<int64_t> idx(z(nb));
    for (int64_t b = 0; b < k / 256; ++b) {
      const std::uint8_t* p = row + 264 * b;
      const float d = quant::fp16_to_fp32(static_cast<std::uint16_t>(p[0] | (p[1] << 8)));
      std::uint64_t lv = 0;
      for (int i = 0; i < 6; ++i) lv |= static_cast<std::uint64_t>(p[2 + i]) << (8 * i);
      for (int j = 0; j < 8; ++j) {
        s[z(8 * b + j)] = quant::q8_z_scale(d, static_cast<int>((lv >> (6 * j)) & 63u));
        for (int i = 0; i < 32; ++i) {
          const int q = static_cast<int>(static_cast<std::int8_t>(p[8 + 32 * j + i]));
          hm[z(256 * b + 32 * j + i)] = static_cast<float>((q + 127) & 1) - 0.5f;
        }
      }
    }
    for (int64_t t = 0; t < n_tok; ++t) {
      for (int64_t j = 0; j < nb; ++j) {
        score[z(j)] = s[z(j)] * s[z(j)] * X[z(t * nb + j)];
        idx[z(j)] = j;
      }
      // the nb - keep blocks with the smallest expected loss drop their 8th bit
      std::nth_element(idx.begin(), idx.begin() + (nb - keep), idx.end(), [&](int64_t a, int64_t b2) {
        return score[z(a)] < score[z(b2)] || (score[z(a)] == score[z(b2)] && a < b2);
      });
      float corr = 0.0f;
      for (int64_t q = 0; q < nb - keep; ++q) {
        const int64_t j = idx[z(q)];
        float acc = 0.0f;
        for (int i = 0; i < 32; ++i) acc += hm[z(32 * j + i)] * x[t * ldx + 32 * j + i];
        corr += s[z(j)] * acc;
      }
      y[t * ldy + r] -= corr;
    }
  });
}

void t_decode_impl(const std::uint8_t* data, std::uint64_t row_bytes, int64_t n_out, int64_t k, const float* x, int64_t n_tok, int64_t ldx,
                   float* y, int64_t ldy, int threads, bool fold = false) {
  const int64_t nb = k / 32;
  std::vector<quant::ActBlock32> act(z(n_tok * nb));
  for (int64_t t = 0; t < n_tok; ++t) quant::act_quantize_32(x + t * ldx, k, act.data() + t * nb);
  parallel_items(n_out, threads, [&](int64_t r, int) {
    // the row as a kernel would hold it: the 7-bit base as integers b (0..127), the H word of each gated block, the scales
    const std::uint8_t* row = data + static_cast<uint64_t>(r) * row_bytes;
    std::vector<std::int8_t> bb(z(k));
    std::vector<std::uint32_t> hw(z(nb), 0u);
    std::vector<std::uint8_t> gated(z(nb), 0);
    std::vector<float> s(z(nb));
    for (int64_t i = 0; i < k / 256; ++i) {
      const std::uint8_t* p = row + 241 * i;
      const float d = quant::fp16_to_fp32(static_cast<std::uint16_t>(p[0] | (p[1] << 8)));
      std::uint64_t lv = 0;
      for (int b = 0; b < 6; ++b) lv |= static_cast<std::uint64_t>(p[2 + b]) << (8 * b);
      const std::uint8_t mask = p[8];
      int slot = 0;
      for (int j = 0; j < 8; ++j) {
        s[z(8 * i + j)] = quant::q8_z_scale(d, static_cast<int>((lv >> (6 * j)) & 63u));
        if (((mask >> j) & 1u) && slot < 2) {  // two H words per block (a corrupt mask reads no further)
          const std::uint8_t* hp = p + 233 + 4 * slot++;
          hw[z(8 * i + j)] = static_cast<std::uint32_t>(hp[0]) | (static_cast<std::uint32_t>(hp[1]) << 8) | (static_cast<std::uint32_t>(hp[2]) << 16) |
                             (static_cast<std::uint32_t>(hp[3]) << 24);
          gated[z(8 * i + j)] = 1;
        }
        for (int g = 0; g < 4; ++g) {
          std::uint64_t wv = 0;
          for (int b = 0; b < 7; ++b) wv |= static_cast<std::uint64_t>(p[9 + 7 * (4 * j + g) + b]) << (8 * b);
          for (int kk = 0; kk < 8; ++kk) bb[z(256 * i + 32 * j + 8 * g + kk)] = static_cast<std::int8_t>((wv >> (7 * kk)) & 127u);
        }
      }
    }
    for (int64_t t = 0; t < n_tok; ++t) {
      float acc = 0.0f;
      for (int64_t j = 0; j < nb; ++j) {
        const quant::ActBlock32& a = act[z(t * nb + j)];
        std::int32_t pb = 0;
        for (int i = 0; i < 32; ++i) pb += static_cast<std::int32_t>(bb[z(32 * j + i)]) * a.q[i];
        std::int64_t P2 = 2 * (2 * static_cast<std::int64_t>(pb) - 127 * static_cast<std::int64_t>(a.sum));
        if (gated[z(j)]) {
          const std::uint32_t h = hw[z(j)];
          std::int64_t hc = 0;
          if (fold) {  // control: h folded into the values, sum h x directly
            for (int i = 0; i < 32; ++i) hc += static_cast<std::int64_t>((h >> i) & 1u) * a.q[i];
          } else {
            hc = -128 * std::popcount(h);
            for (int kk = 0; kk < 8; ++kk) hc += static_cast<std::int64_t>(std::popcount(h & a.plane[kk])) << kk;
          }
          P2 += 2 * hc;
        }
        acc += s[z(j)] * a.scale * (0.5f * static_cast<float>(P2));
      }
      y[t * ldy + r] = acc;
    }
  });
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
    if (c.opt.t_decode && f == ActFmt::q8_256x8 && static_cast<quant::Type>(w.type) == quant::Type::q7_zt && !c.opt.act_rot32) {
      for (int64_t t = 0; t < n_tok; ++t) {  // rotated as the weight is, rounded inside the T-decode (same rounding as gpu256x8)
        std::memcpy(xr.data() + t * k, x + t * ldx, z(k) * sizeof(float));
        if (w.rot32) act_rotate32(xr.data() + t * k, k);
      }
      t_decode_impl(w.data, w.row_bytes, static_cast<int64_t>(w.ne[1]), k, xr.data(), n_tok, k, y, ldy, c.threads, c.opt.t_decode_fold);
      return;
    }
    for (int64_t t = 0; t < n_tok; ++t) {
      std::memcpy(xr.data() + t * k, x + t * ldx, z(k) * sizeof(float));
      prep_row(xr.data() + t * k, k, f, w.rot32, c.opt.act_rot32);
    }
    std::vector<kern::ActQ8x32> acts;
    if (c.opt.fast_cpu && f == ActFmt::q8_256x8 && !c.opt.act_rot32) {  // the engine's int8 path
      acts.resize(z(n_tok));
      for (int64_t t = 0; t < n_tok; ++t) kern::act_q8x32_from_rounded(xr.data() + t * k, k, acts[z(t)]);
      for (Scratch& sx : c.sc) sx.acts = &acts;
    }
    matmul(w, xr.data(), n_tok, k, y, ldy, c.threads, c.sc);
    for (Scratch& sx : c.sc) sx.acts = nullptr;
    if (c.opt.h_gate > 0.0 && static_cast<quant::Type>(w.type) == quant::Type::q8_z)
      h_gate_impl(w.data, w.row_bytes, static_cast<int64_t>(w.ne[1]), k, xr.data(), n_tok, k, y, ldy, c.opt.h_gate, c.threads);
    return;
  }
  matmul(w, x, n_tok, ldx, y, ldy, c.threads, c.sc);
  if (c.opt.h_gate > 0.0 && static_cast<quant::Type>(w.type) == quant::Type::q8_z)
    h_gate_impl(w.data, w.row_bytes, static_cast<int64_t>(w.ne[1]), static_cast<int64_t>(w.ne[0]), x, n_tok, ldx, y, ldy, c.opt.h_gate, c.threads);
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
    rope_cache(s, c.pos0 + t, cs);
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
  if (c.opt.kv_store != ForwardOptions::StateStore::exact) {
    state_store_roundtrip(k.data(), n * kv_w, c.opt.kv_store);
    state_store_roundtrip(v.data(), n * kv_w, c.opt.kv_store);
  }
  if (c.opt.qk_dump) {
    const int64_t row = (nh + nkv) * hd;
    std::uint16_t* dst = c.opt.qk_dump + static_cast<int64_t>(c.ai) * n * row;
    for (int64_t t = 0; t < n; ++t) {
      for (int64_t h = 0; h < nh; ++h)
        for (int64_t i = 0; i < hd; ++i) dst[t * row + h * hd + i] = quant::fp32_to_fp16(qg[z(t * q_w + h * 2 * hd + i)]);
      for (int64_t i = 0; i < nkv * hd; ++i) dst[t * row + nh * hd + i] = quant::fp32_to_fp16(k[z(t * kv_w + i)]);
    }
  }
  // GP3-07: with a carried state the new keys and values are appended to the layer's cache and every query attends to all
  // cached positions before it; without one the cache is this call's own k and v
  const float* kall = k.data();
  const float* vall = v.data();
  const int64_t p0 = c.st ? c.pos0 : 0;
  if (c.st) {
    std::vector<float>& K = c.st->k[c.il];
    std::vector<float>& V = c.st->v[c.il];
    K.insert(K.end(), k.begin(), k.end());
    V.insert(V.end(), v.begin(), v.end());
    kall = K.data();
    vall = V.data();
  }
  const float scale = s.attn_scale != 0.0f ? s.attn_scale : 1.0f / std::sqrt(static_cast<float>(hd));
  const int64_t group = nh / nkv;
  // causal attention, one (head, query) per item; then the output is gated by sigmoid(gate) of the same head
  parallel_items(nh * n, c.threads, [&](int64_t item, int tid) {
    const int64_t h = item / n, t = item % n, kvh = h / group;
    Scratch& sc = c.sc[z(tid)];
    if (sc.scores.size() < z(p0 + n)) sc.scores.resize(z(p0 + n));
    float* sco = sc.scores.data();
    const float* q = qg.data() + t * q_w + h * 2 * hd;
    const int64_t last = p0 + t;  // the query's own position: it attends to 0..last
    for (int64_t u = 0; u <= last; ++u) {
      const float* kk = kall + u * kv_w + kvh * hd;
      float d = 0.0f;
      for (int64_t i = 0; i < hd; ++i) d += q[i] * kk[i];
      sco[u] = d * scale;
    }
    softmax(sco, last + 1);
    float* o = att.data() + t * hd * nh + h * hd;
    for (int64_t i = 0; i < hd; ++i) o[i] = 0.0f;
    for (int64_t u = 0; u <= last; ++u) {
      const float p = sco[u];
      const float* vv = vall + u * kv_w + kvh * hd;
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
  // ForwardOptions::state_store: a checkpoint taken just before position at and restored: the conv inputs it holds (the
  // last taps-1 positions) and the recurrent state are read back from their lossy form for every later position
  const bool ckpt = c.opt.state_store != ForwardOptions::StateStore::exact;
  const int64_t at = c.opt.state_at >= 0 ? c.opt.state_at : n / 2;
  std::vector<float> mixed_ck;
  if (ckpt && at > 0 && at < n) {
    const int64_t r0 = std::max<int64_t>(0, at - (taps - 1));
    mixed_ck.assign(mixed.begin() + r0 * ch, mixed.begin() + at * ch);
    state_store_roundtrip(mixed_ck.data(), static_cast<int64_t>(mixed_ck.size()), c.opt.state_store);
  }
  // causal depthwise conv over [taps-1 rows before the call, tokens], then SiLU; the rows before are zeros for a fresh
  // sequence and the carried conv tail with a decode state (GP3-07)
  const float* tail = c.st ? c.st->conv_tail[c.il].data() : nullptr;
  std::vector<float> conv(z(n * ch));
  parallel_items((ch + 255) / 256, c.threads, [&](int64_t blk, int) {
    const int64_t c0 = blk * 256, c1 = std::min(ch, c0 + 256);
    for (int64_t t = 0; t < n; ++t)
      for (int64_t cc = c0; cc < c1; ++cc) {
        float sum = 0.0f;
        for (int64_t kk = 0; kk < taps; ++kk) {
          const int64_t src = t + kk - (taps - 1);
          float xv = src >= 0 ? mixed[z(src * ch + cc)] : (tail ? tail[z((taps - 1 + src) * ch + cc)] : 0.0f);
          if (!mixed_ck.empty() && t >= at && src >= 0 && src < at) xv = mixed_ck[z((src - (at - (taps - 1) < 0 ? 0 : at - (taps - 1))) * ch + cc)];
          sum += xv * l.conv1d[z(cc * taps + kk)];
        }
        conv[z(t * ch + cc)] = silu(sum);
      }
  });
  if (c.st) {  // the new tail: the last taps-1 rows of [old tail, this call's conv input]
    std::vector<float>& T = c.st->conv_tail[c.il];
    std::vector<float> nt(z((taps - 1) * ch));
    for (int64_t r = 0; r < taps - 1; ++r) {
      const int64_t src = n - (taps - 1) + r;  // row index in this call; negative = from the old tail
      for (int64_t cc = 0; cc < ch; ++cc) nt[z(r * ch + cc)] = src >= 0 ? mixed[z(src * ch + cc)] : T[z((taps - 1 + src) * ch + cc)];
    }
    T.swap(nt);
  }
  // q, k: L2-normalized per key head
  for (int64_t t = 0; t < n; ++t)
    for (int64_t h = 0; h < 2 * nk; ++h) l2_norm(conv.data() + t * ch + h * ds, ds, s.rms_eps);
  // the recurrence, one value head per item (value head h reads key head h % nk), zero state at the start
  std::vector<float> core(z(n * d_inner));
  const float qscale = 1.0f / std::sqrt(static_cast<float>(ds));
  parallel_items(nv, c.threads, [&](int64_t hv, int tid) {
    Scratch& sc = c.sc[z(tid)];
    sc.state.assign(z(ds * ds), 0.0f);  // M[j][i]: row j = value index, column i = key index
    if (c.st) std::copy_n(c.st->rec[c.il].data() + hv * ds * ds, ds * ds, sc.state.data());  // GP3-07: the carried state
    if (sc.delta.size() < z(ds)) sc.delta.resize(z(ds));
    float* m = sc.state.data();
    float* delta = sc.delta.data();
    const int64_t hk = hv % nk;
    for (int64_t t = 0; t < n; ++t) {
      if (ckpt && t == at) state_store_roundtrip(m, ds * ds, c.opt.state_store);  // the restored checkpoint
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
    if (c.st) std::copy_n(m, ds * ds, c.st->rec[c.il].data() + hv * ds * ds);
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
  // plane_picks: an expert's rows are run in two groups, the picks that read their planes (rank < plane_picks) and the rest
  const bool split_picks = c.opt.plane_picks > 0 && c.opt.plane_picks < nu && c.opt.planes;
  std::vector<std::vector<int64_t>> groups;
  std::vector<char> group_planes;
  for (const int64_t x : busy) {
    if (!split_picks) {
      groups.push_back(slots[z(x)]);
      group_planes.push_back(1);
      continue;
    }
    std::vector<int64_t> a, b;  // slot = t * nu + k, and k is the pick's rank (the picks are kept in descending order)
    for (const int64_t slot : slots[z(x)]) ((slot % nu) < c.opt.plane_picks ? a : b).push_back(slot);
    for (int g = 0; g < 2; ++g) {
      std::vector<int64_t>& v = g == 0 ? a : b;
      if (v.empty()) continue;
      groups.push_back(std::move(v));
      group_planes.push_back(g == 0 ? 1 : 0);
    }
  }
  std::vector<int64_t> group_ex;
  for (const int64_t x : busy) {
    if (!split_picks) { group_ex.push_back(x); continue; }
    int cnt = 0;
    for (const int64_t slot : slots[z(x)]) cnt |= ((slot % nu) < c.opt.plane_picks) ? 1 : 2;
    if (cnt & 1) group_ex.push_back(x);
    if (cnt & 2) group_ex.push_back(x);
  }
  parallel_items(static_cast<int64_t>(groups.size()), c.threads, [&](int64_t bi, int tid) {
    const int64_t ex = group_ex[z(bi)];
    const std::vector<int64_t>& list = groups[z(bi)];
    const int64_t m = static_cast<int64_t>(list.size());
    Scratch& sc = c.sc[z(tid)];
    const bool planes_before = sc.planes;
    sc.planes = planes_before && group_planes[z(bi)] != 0;
    sc.xe.resize(z(m * e));
    sc.ge.resize(z(m * ff));
    sc.ue.resize(z(m * ff));
    sc.ye.resize(z(m * e));
    for (int64_t r = 0; r < m; ++r) std::memcpy(sc.xe.data() + r * e, xn + (list[z(r)] / nu) * e, z(e) * sizeof(float));
    // one input feeds both gate and up, rounded once (in llama.cpp both are the same type, so the same format)
    ActFmt fgu = act_format(c.opt, l.gate_exps) != ActFmt::none ? act_format(c.opt, l.gate_exps) : act_format(c.opt, l.up_exps);
    // the engine's split: a cold expert runs on the CPU, whose kernel takes one scale per 256
    const bool cold = c.opt.cold_gpu256 && !c.opt.hot.empty() && c.opt.hot[z(static_cast<int64_t>(c.il) * ne + ex)] == 0;
    if (cold && fgu != ActFmt::none) fgu = ActFmt::q8_256;
    if (c.opt.cold_gpu256 && c.opt.m_stats) {
      if (cold) std::atomic_ref<std::uint64_t>(c.opt.m_stats[4]).fetch_add(static_cast<std::uint64_t>(m));
      std::atomic_ref<std::uint64_t>(c.opt.m_stats[5]).fetch_add(static_cast<std::uint64_t>(m));
    }
    if (fgu != ActFmt::none || l.gate_exps.rot32)
      for (int64_t r = 0; r < m; ++r) prep_row(sc.xe.data() + r * e, e, fgu, l.gate_exps.rot32, c.opt.act_rot32);
    const bool int8_in = c.opt.fast_cpu && !c.opt.act_rot32;  // the engine's int8 path for gpu256x8 inputs
    auto own = [&](const float* xr, int64_t kk) {
      sc.own_acts.resize(z(m));
      for (int64_t r = 0; r < m; ++r) kern::act_q8x32_from_rounded(xr + r * kk, kk, sc.own_acts[z(r)]);
      sc.acts = &sc.own_acts;
    };
    if (int8_in && fgu == ActFmt::q8_256x8) own(sc.xe.data(), e);
    matmul_rows(l.gate_exps, static_cast<uint64_t>(ex), 0, ff, sc.xe.data(), m, e, sc.ge.data(), ff, sc);
    matmul_rows(l.up_exps, static_cast<uint64_t>(ex), 0, ff, sc.xe.data(), m, e, sc.ue.data(), ff, sc);
    sc.acts = nullptr;
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
    ActFmt fd = act_format(c.opt, l.down_exps);
    if (cold && fd != ActFmt::none) fd = ActFmt::q8_256;
    if (fd != ActFmt::none || l.down_exps.rot32)
      for (int64_t r = 0; r < m; ++r) prep_row(sc.ge.data() + r * ff, ff, fd, l.down_exps.rot32, c.opt.act_rot32);
    if (int8_in && fd == ActFmt::q8_256x8) own(sc.ge.data(), ff);
    matmul_rows(l.down_exps, static_cast<uint64_t>(ex), 0, e, sc.ge.data(), m, ff, sc.ye.data(), e, sc);
    sc.acts = nullptr;
    for (int64_t r = 0; r < m; ++r) {
      const int64_t slot = list[z(r)];
      const float wgt = wsel[z(slot)];
      float* dst = ys.data() + slot * e;
      const float* src = sc.ye.data() + r * e;
      for (int64_t i = 0; i < e; ++i) dst[i] = src[i] * wgt;
    }
    sc.planes = planes_before;
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

void state_store_roundtrip(float* x, std::int64_t n, ForwardOptions::StateStore kind) {
  using K = ForwardOptions::StateStore;
  if (kind == K::exact) return;
  if (kind == K::int8row) {
    for (std::int64_t r = 0; r < n; r += 128) {
      float amax = 0.0f;
      for (std::int64_t i = r; i < r + 128; ++i) amax = std::max(amax, std::fabs(x[i]));
      const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
      for (std::int64_t i = r; i < r + 128; ++i) x[i] = sc * std::min(127.0f, std::max(-127.0f, std::nearbyint(x[i] / sc)));
    }
    return;
  }
  static const float sign[32] = {1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1,
                                 -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1};
  const float top = kind == K::q8 ? 127.0f : 7.0f;
  for (std::int64_t b = 0; b < n; b += 32) {
    float* v = x + b;
    if (kind == K::int4r) {
      for (int i = 0; i < 32; ++i) v[i] *= sign[i];
      quant::hadamard32(v, 32);
    }
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[i]));
    const float sc = quant::fp16_to_fp32(quant::fp32_to_fp16(amax > 0.0f ? amax / top : 1.0f));
    for (int i = 0; i < 32; ++i) v[i] = sc * std::min(top, std::max(-top, std::nearbyint(v[i] / sc)));
    if (kind == K::int4r) {
      quant::hadamard32(v, 32);  // orthonormal and its own inverse
      for (int i = 0; i < 32; ++i) v[i] *= sign[i];
    }
  }
}

namespace {
// one decoder block's tensors under prefix p ("blk.N."): attention (full or DeltaNet), norms, router, experts, shared expert
Status bind_layer(const io::GgufReader& r, const std::uint8_t* base, const Qwen35MoeSpec& s, const std::string& p, bool recurrent,
                  Qwen35MoeLayer& l) {
  const uint64_t e = s.n_embd, hd = s.head_dim, nh = s.n_head, nkv = s.n_head_kv;
  const uint64_t ch = s.conv_channels(), di = s.ssm_d_inner, ds = s.ssm_d_state, nvh = s.ssm_dt_rank;
  const uint64_t ne = s.n_expert, ff = s.n_ff_exp, ffs = s.n_ff_shexp;
  l.recurrent = recurrent;
  std::vector<Status> all;
  all.push_back(bind_vector(r, base, p + "attn_norm.weight", {e}, l.attn_norm));
  all.push_back(bind_vector(r, base, p + "post_attention_norm.weight", {e}, l.post_norm));
  if (!l.recurrent) {
    if (r.find_tensor(p + "attn_qkv.weight")) {
      // a fused projection (newer converters): the rows are [query | gate] per head, then the keys, then the values, so
      // the three views are consecutive row ranges of one tensor
      WeightView f;
      if (Status fs = bind_view(r, base, p + "attn_qkv.weight", {e, 2 * hd * nh + 2 * hd * nkv}, f); !fs) return fs;
      l.wq = l.wk = l.wv = f;
      l.wq.ne[1] = 2 * hd * nh;
      l.wk.ne[1] = l.wv.ne[1] = hd * nkv;
      l.wk.data = f.data + 2 * hd * nh * f.row_bytes;
      l.wv.data = l.wk.data + hd * nkv * f.row_bytes;
    } else {
      all.push_back(bind_view(r, base, p + "attn_q.weight", {e, 2 * hd * nh}, l.wq));
      all.push_back(bind_view(r, base, p + "attn_k.weight", {e, hd * nkv}, l.wk));
      all.push_back(bind_view(r, base, p + "attn_v.weight", {e, hd * nkv}, l.wv));
    }
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
  all.push_back(bind_view(r, base, p + "ffn_gate_inp.weight", {e, ne}, l.gate_inp));
  if (r.find_tensor(p + "ffn_gate_up_exps.weight")) {
    all.push_back(bind_gate_up_exps(r, base, p + "ffn_gate_up_exps.weight", e, ff, ne, l.gate_exps, l.up_exps));
  } else {
    all.push_back(bind_experts(r, base, p + "ffn_up_exps.weight", e, ff, ne, l.up_exps));
    all.push_back(bind_experts(r, base, p + "ffn_gate_exps.weight", e, ff, ne, l.gate_exps));
  }
  all.push_back(bind_experts(r, base, p + "ffn_down_exps.weight", ff, e, ne, l.down_exps));
  all.push_back(bind_view(r, base, p + "ffn_up_shexp.weight", {e, ffs}, l.up_shexp));
  all.push_back(bind_view(r, base, p + "ffn_gate_shexp.weight", {e, ffs}, l.gate_shexp));
  all.push_back(bind_view(r, base, p + "ffn_down_shexp.weight", {ffs, e}, l.down_shexp));
  all.push_back(bind_vector(r, base, p + "ffn_gate_inp_shexp.weight", {e}, l.gate_inp_shexp));
  for (const Status& x : all)
    if (!x) return x;
  return Status::good();
}
}  // namespace

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
  w.layers.resize(s.n_layer);
  for (std::uint32_t il = 0; il < s.n_layer; ++il)
    if (Status ls = bind_layer(r, base, s, "blk." + std::to_string(il) + ".", s.is_recurrent(il), w.layers[il]); !ls) return R::fail(ls);
  // GP3-08: the MTP block, when the file has one. A file may leave its tensors out (the MTP piece is optional to a
  // converter); then the model runs without drafting. One that has some of them must have all of them.
  if (s.n_layer_all > s.n_layer) {
    if (s.n_layer_all - s.n_layer != 1)
      return R::fail(Status::fail(Code::unsupported, "more than one MTP block (nextn_predict_layers > 1) is not read yet"));
    const std::string p = "blk." + std::to_string(s.n_layer) + ".";
    if (r.find_tensor(p + "nextn.eh_proj.weight")) {
      Qwen35MoeMtp& m = w.mtp;
      if (Status ls = bind_layer(r, base, s, p, false, m.layer); !ls) return R::fail(ls);
      std::vector<Status> all;
      all.push_back(bind_view(r, base, p + "nextn.eh_proj.weight", {2 * e, e}, m.eh_proj));
      all.push_back(bind_vector(r, base, p + "nextn.enorm.weight", {e}, m.enorm));
      all.push_back(bind_vector(r, base, p + "nextn.hnorm.weight", {e}, m.hnorm));
      all.push_back(bind_view(r, base, p + "nextn.embed_tokens.weight", {e, nv}, m.embed_tokens, false));
      all.push_back(bind_view(r, base, p + "nextn.shared_head_head.weight", {e, nv}, m.head, false));
      if (r.find_tensor(p + "nextn.shared_head_norm.weight")) all.push_back(bind_vector(r, base, p + "nextn.shared_head_norm.weight", {e}, m.head_norm));
      for (const Status& x : all)
        if (!x) return R::fail(x);
      if (!m.embed_tokens.present()) m.embed_tokens = w.tok_embd;
      if (!m.head.present()) m.head = w.output;
      if (m.head_norm.empty()) m.head_norm = w.output_norm;
      m.present = true;
    }
  }
  // weights the file stores rotated (pelias.rot32.tensors, written by `pelias quantize --rotate32`)
  for (const io::GgufKv& kv : r.kv()) {
    if (kv.key != "pelias.rot32.tensors") continue;
    if (kv.value.type != io::GgufValueType::array || kv.value.array_type != io::GgufValueType::string)
      return R::fail(Status::fail(Code::corrupt, "pelias.rot32.tensors is not a list of tensor names"));
    std::vector<std::string> names;
    if (Status as = r.array_strings(kv.value, names); !as) return R::fail(as);
    std::vector<WeightView*> views{&w.output};
    std::vector<Qwen35MoeLayer*> layers;
    for (Qwen35MoeLayer& l : w.layers) layers.push_back(&l);
    if (w.mtp.present) {
      layers.push_back(&w.mtp.layer);
      views.push_back(&w.mtp.eh_proj);
      if (w.mtp.head.data != w.output.data) views.push_back(&w.mtp.head);
    }
    for (Qwen35MoeLayer* l : layers)
      for (WeightView* p : {&l->wq, &l->wk, &l->wv, &l->wo, &l->wqkv, &l->wgate, &l->wbeta, &l->walpha, &l->ssm_out, &l->up_exps, &l->gate_exps,
                            &l->down_exps, &l->up_shexp, &l->gate_shexp, &l->down_shexp})
        views.push_back(p);
    for (const std::string& nm : names) {
      WeightView* hit = nullptr;
      for (WeightView* p : views)
        if (p->present() && p->name == nm) {
          if (hit && hit != p) p->rot32 = true;  // a fused tensor's other views (q/k/v, gate/up) share its name
          hit = p;
        }
      if (!hit) {
        // an MTP block the file left incomplete is not bound (no drafting); its rotated tensors are skipped with it
        const bool mtp = nm.rfind("blk." + std::to_string(s.n_layer) + ".", 0) == 0;
        if (mtp && !w.mtp.present) continue;
        return R::fail(Status::fail(Code::unsupported, "the file marks a tensor as rotated that is not a product weight of this model", 0, nm));
      }
      if (hit == &w.output && w.output.data == w.tok_embd.data)
        return R::fail(Status::fail(Code::unsupported, "a rotated output layer tied to the embedding table", 0, nm));
      if (hit->ne[0] % 32 != 0) return R::fail(Status::fail(Code::corrupt, "a rotated tensor's row length is not a multiple of 32", 0, nm));
      hit->rot32 = true;
    }
    for (const Qwen35MoeLayer* l : layers)
      if (l->gate_exps.rot32 != l->up_exps.rot32)
        return R::fail(Status::fail(Code::unsupported, "expert gate and up share one input, so both or neither must be rotated", 0, l->gate_exps.name));
  }
  return R::good(std::move(w));
}

std::int64_t plane_signature(const WeightView& v) {
  if (v.slots.empty()) return static_cast<std::int64_t>(v.type);
  std::vector<std::uint32_t> t;  // a tiered weight: its per-expert types (io::tier_signature, as the plane builder writes it)
  for (const ExpertSlot& x : v.slots) t.push_back(x.type);
  return io::tier_signature(t);
}

Result<int> attach_planes(const io::GgufReader& pr, const std::uint8_t* pb, Qwen35MoeWeights& w, int k_use) {
  using R = Result<int>;
  bool found = false;
  const uint64_t k = pr.get_uint("pelias.planes.k", &found);
  if (!found || k < 1 || k > static_cast<uint64_t>(quant::kPlaneMaxK))
    return R::fail(Status::fail(Code::corrupt, "not a plane side file (pelias.planes.k missing or out of range)"));
  if (pr.get_uint("pelias.planes.block", &found) != static_cast<uint64_t>(quant::kPlaneBlock) || !found)
    return R::fail(Status::fail(Code::unsupported, "the plane side file uses another block size"));
  if (k_use < 0 || static_cast<uint64_t>(k_use) > k)
    return R::fail(Status::fail(Code::invalid_argument, "more planes asked for than the side file stores (" + std::to_string(k) + ")"));
  auto ints = [&](const char* key, std::vector<std::int64_t>& out) -> Status {
    const io::GgufValue* v = pr.find(key);
    if (!v) return Status::good();
    if (v->type != io::GgufValueType::array) return Status::fail(Code::corrupt, std::string(key) + " is not a list");
    return pr.array_ints(*v, out);
  };
  std::vector<std::int64_t> types, ks, sparse;
  if (!pr.find("pelias.planes.types")) return R::fail(Status::fail(Code::corrupt, "the plane side file names no source types"));
  for (const auto& [key, vec] : {std::pair<const char*, std::vector<std::int64_t>*>{"pelias.planes.types", &types}, {"pelias.planes.ks", &ks},
                                 {"pelias.planes.sparse", &sparse}})
    if (Status st = ints(key, *vec); !st) return R::fail(st);
  const bool tied = w.output.data == w.tok_embd.data;
  std::vector<WeightView*> views{&w.output, &w.tok_embd};  // a tied output layer shares the embedding's planes (below)
  for (Qwen35MoeLayer& l : w.layers)
    for (WeightView* p : {&l.wq, &l.wk, &l.wv, &l.wo, &l.wqkv, &l.wgate, &l.wbeta, &l.walpha, &l.ssm_out, &l.up_exps, &l.gate_exps, &l.down_exps,
                          &l.up_shexp, &l.gate_shexp, &l.down_shexp})
      views.push_back(p);
  auto suffix = [](const std::string& a, const char* sfx) {
    const std::size_t n = std::char_traits<char>::length(sfx);
    return a.size() >= n && a.compare(a.size() - n, n, sfx) == 0;
  };
  int n = 0;
  size_t ti = 0;
  for (const io::GgufTensor& t : pr.tensors()) {
    if (t.name == "output.weight.exact") {
      if (t.type != static_cast<uint32_t>(quant::Type::bf16) || t.ne[0] != w.output.ne[0] || t.ne[1] != w.output.ne[1] || t.ne[2] != 1)
        return R::fail(Status::fail(Code::corrupt, "the exact output rows are not BF16 rows of the output layer's shape", 0, t.name));
      w.output_exact.data = pb + t.file_offset;
      w.output_exact.type = t.type;
      w.output_exact.ne[0] = t.ne[0];
      w.output_exact.ne[1] = t.ne[1];
      w.output_exact.row_bytes = quant::row_bytes(t.type, t.ne[0]);
      w.output_exact.name = t.name;
      continue;
    }
    if (suffix(t.name, ".planemask") || suffix(t.name, ".planeoff")) continue;  // read with their plane tensor below
    if (ti >= types.size()) return R::fail(Status::fail(Code::corrupt, "the plane side file names fewer source types than it has plane tensors"));
    const std::int64_t src_type = types[ti];
    const int kt = ks.empty() ? static_cast<int>(k) : static_cast<int>(ks[ti]);
    const bool sp = !sparse.empty() && sparse[ti] != 0;
    ++ti;
    if (kt < 1 || kt > quant::kPlaneMaxK) return R::fail(Status::fail(Code::corrupt, "a plane tensor's plane count is out of range", 0, t.name));
    WeightView* hit = nullptr;
    for (WeightView* p : views)
      if (p->present() && p->name == t.name) hit = p;
    if (!hit) continue;  // a block the reference does not run (MTP), or a tensor that is not a product here
    if (src_type != plane_signature(*hit))
      return R::fail(Status::fail(Code::invalid_argument, "the plane side file was built for another type of this tensor (another model file)", 0, t.name));
    const uint64_t rows = hit->ne[1] * hit->ne[2];
    if (hit->ne[0] % quant::kPlaneBlock != 0 || t.type != static_cast<uint32_t>(quant::Type::i8))
      return R::fail(Status::fail(Code::corrupt, "a plane tensor does not match its weight's rows", 0, t.name));
    if (!sp) {
      const uint64_t prb = static_cast<uint64_t>(quant::plane_row_bytes(static_cast<int64_t>(hit->ne[0]), kt));
      if (t.ne[0] != prb || t.ne[1] != rows || t.ne[2] != 1)
        return R::fail(Status::fail(Code::corrupt, "a plane tensor does not match its weight's rows", 0, t.name));
      hit->plane_row_bytes = prb;
      hit->plane_mask = nullptr;
      hit->plane_off = nullptr;
    } else {
      const io::GgufTensor* mk = pr.find_tensor(t.name + ".planemask");
      const io::GgufTensor* of = pr.find_tensor(t.name + ".planeoff");
      const uint64_t mb = (hit->ne[0] / quant::kPlaneBlock + 7) / 8;
      if (!mk || !of || mk->type != static_cast<uint32_t>(quant::Type::i8) || mk->ne[0] != mb || mk->ne[1] != rows ||
          of->type != static_cast<uint32_t>(quant::Type::i32) || of->ne[0] != rows || t.ne[0] != static_cast<uint64_t>(34 * kt))
        return R::fail(Status::fail(Code::corrupt, "a block-selective plane tensor's mask or offsets do not match its weight", 0, t.name));
      hit->plane_mask = pb + mk->file_offset;
      hit->plane_off = reinterpret_cast<const std::uint32_t*>(pb + of->file_offset);
      hit->plane_mask_bytes = mb;
      hit->plane_row_bytes = 0;
    }
    hit->planes = pb + t.file_offset;
    hit->plane_k_stored = kt;
    hit->plane_k = k_use == 0 ? kt : std::min(k_use, kt);
    ++n;
  }
  if (tied && w.tok_embd.planes) {
    w.output.planes = w.tok_embd.planes;
    w.output.plane_row_bytes = w.tok_embd.plane_row_bytes;
    w.output.plane_k_stored = w.tok_embd.plane_k_stored;
    w.output.plane_k = w.tok_embd.plane_k;
    w.output.plane_mask = w.tok_embd.plane_mask;
    w.output.plane_off = w.tok_embd.plane_off;
    w.output.plane_mask_bytes = w.tok_embd.plane_mask_bytes;
  }
  return R::good(n);
}

namespace {
Status forward_impl(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, const std::int32_t* tokens, int n_tokens, int out_begin, int out_end,
                    float* logits, const ForwardOptions& opt, DecodeState* st);

// one token's embedding row (and its planes, when the table has them and they are read)
void embed_row(const WeightView& te, std::int32_t tok, int64_t e, bool planes, float* out) {
  (void)quant::dequantize_row(static_cast<quant::Type>(te.type), te.data + static_cast<uint64_t>(tok) * te.row_bytes, out, e);
  if (planes && te.planes) {
    if (te.plane_mask)
      quant::plane_add_row_sparse(te.planes + static_cast<uint64_t>(te.plane_off[tok]) * 34u * static_cast<uint64_t>(te.plane_k_stored),
                                  te.plane_mask + static_cast<uint64_t>(tok) * te.plane_mask_bytes, e, te.plane_k_stored, te.plane_k, out);
    else
      quant::plane_add_row(te.planes + static_cast<uint64_t>(tok) * te.plane_row_bytes, e, te.plane_k_stored, te.plane_k, out);
  }
}
}  // namespace

Status forward_reference(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, const std::int32_t* tokens, int n_tokens, int out_begin,
                         int out_end, float* logits, const ForwardOptions& opt) {
  return forward_impl(s, w, tokens, n_tokens, out_begin, out_end, logits, opt, nullptr);
}

void decode_state_reset(const Qwen35MoeSpec& s, DecodeState& st) {
  st.pos = 0;
  const std::size_t L = s.n_layer;
  st.k.assign(L, {});
  st.v.assign(L, {});
  st.conv_tail.assign(L, {});
  st.rec.assign(L, {});
  for (std::size_t il = 0; il < L; ++il)
    if (s.is_recurrent(static_cast<std::uint32_t>(il))) {
      st.conv_tail[il].assign(static_cast<std::size_t>(s.ssm_d_conv - 1) * s.conv_channels(), 0.0f);
      st.rec[il].assign(static_cast<std::size_t>(s.ssm_dt_rank) * s.ssm_d_state * s.ssm_d_state, 0.0f);
    }
}

Status forward_decode(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, DecodeState& st, const std::int32_t* tokens, int n_new, float* logits,
                      bool last_only, const ForwardOptions& opt) {
  if (opt.state_store != ForwardOptions::StateStore::exact || opt.qk_dump || opt.state_codes)
    return Status::fail(Code::invalid_argument, "the checkpoint, q/k and state-code records need a whole chunk; they are not built for decoding");
  if (st.k.size() != s.n_layer) return Status::fail(Code::invalid_argument, "the decode state was not sized for this model (decode_state_reset)");
  if (n_new <= 0) return Status::fail(Code::invalid_argument, "no new tokens");
  return forward_impl(s, w, tokens, n_new, last_only ? n_new - 1 : 0, n_new, logits, opt, &st);
}

void mtp_state_reset(MtpState& st) {
  st.pos = 0;
  st.kv = {};
  st.kv.k.assign(1, {});
  st.kv.v.assign(1, {});
}

void mtp_state_truncate(const Qwen35MoeSpec& s, MtpState& st, std::int64_t pos) {
  if (pos >= st.pos || pos < 0) return;
  const std::size_t kv_w = static_cast<std::size_t>(s.head_dim) * s.n_head_kv;
  st.kv.k[0].resize(static_cast<std::size_t>(pos) * kv_w);
  st.kv.v[0].resize(static_cast<std::size_t>(pos) * kv_w);
  st.pos = pos;
  st.kv.pos = pos;
}

Status mtp_forward(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, MtpState& st, const std::int32_t* tokens, const float* h_prev, int n,
                   float* logits, float* h_out, const ForwardOptions& opt) {
  if (!w.mtp.present) return Status::fail(Code::unsupported, "this file has no MTP block (nextn_predict_layers 0, or its tensors left out)");
  if (n <= 0) return Status::fail(Code::invalid_argument, "no tokens");
  if (st.kv.k.size() != 1) return Status::fail(Code::invalid_argument, "the MTP state was not reset (mtp_state_reset)");
  if (Status cs = check_act_rounding(opt, w); !cs) return cs;
  const Qwen35MoeMtp& m = w.mtp;
  // the per-layer records and the measurement options belong to the trunk's layers; the MTP block runs plain
  ForwardOptions o = opt;
  o.expert_picks = nullptr;
  o.expert_weight = nullptr;
  o.imatrix = nullptr;
  o.route = nullptr;
  o.state_codes = nullptr;
  o.qk_dump = nullptr;
  o.hot.clear();
  o.cold_gpu256 = false;
  o.swap_margin = 0.0f;
  o.defer_experts = 0;
  o.m_stats = nullptr;
  o.neuron_skip = 0.0;
  o.kv_store = ForwardOptions::StateStore::exact;
  o.hidden_out = nullptr;
  const int threads = o.threads > 0 ? o.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  const int64_t e = s.n_embd;
  std::vector<Scratch> sc(z(threads));
  for (Scratch& x : sc) {
    x.planes = o.planes;
    x.fast = o.fast_cpu;
  }
  Ctx c{s, threads, sc, n, o};
  c.st = &st.kv;
  c.pos0 = st.pos;
  c.il = 0;  // the MTP state holds one layer
  // [enorm(embedding) | hnorm(h_prev)] -> eh_proj
  std::vector<float> cat(z(2 * n * e)), emb(z(e)), x(z(n * e)), xn(z(n * e)), y(z(n * e));
  for (int64_t t = 0; t < n; ++t) {
    const std::int32_t tok = tokens[t];
    if (tok < 0 || static_cast<uint64_t>(tok) >= s.n_vocab) return Status::fail(Code::invalid_argument, "a token id is outside the vocabulary");
    embed_row(m.embed_tokens, tok, e, o.planes, emb.data());
    rms_norm(emb.data(), m.enorm.data(), cat.data() + t * 2 * e, e, s.rms_eps);
    rms_norm(h_prev + t * e, m.hnorm.data(), cat.data() + t * 2 * e + e, e, s.rms_eps);
  }
  mm(c, m.eh_proj, cat.data(), n, 2 * e, x.data(), e);
  // one full-attention decoder block: x += attn(attn_norm(x)); x += moe(post_norm(x))
  const Qwen35MoeLayer& l = m.layer;
  for (int64_t t = 0; t < n; ++t) rms_norm(x.data() + t * e, l.attn_norm.data(), xn.data() + t * e, e, s.rms_eps);
  st.kv.pos = st.pos;
  full_attention(c, l, xn.data(), y.data());
  for (int64_t i = 0; i < n * e; ++i) x[z(i)] += y[z(i)];
  for (int64_t t = 0; t < n; ++t) rms_norm(x.data() + t * e, l.post_norm.data(), xn.data() + t * e, e, s.rms_eps);
  moe(c, l, xn.data(), y.data());
  for (int64_t i = 0; i < n * e; ++i) x[z(i)] += y[z(i)];
  st.pos += n;
  st.kv.pos = st.pos;
  // the head norm gives the block's hidden (the next draft step's seed) and the logits' input
  for (int64_t t = 0; t < n; ++t) rms_norm(x.data() + t * e, m.head_norm.data(), xn.data() + t * e, e, s.rms_eps);
  if (h_out) std::copy_n(xn.data(), z(n * e), h_out);
  if (logits) mm(c, m.head, xn.data(), n, e, logits, s.n_vocab);
  return Status::good();
}

namespace {
std::int32_t argmax_row(const float* lg, int64_t nv) {  // the first of equal largest logits, as greedy sampling picks
  int64_t best = 0;
  for (int64_t i = 1; i < nv; ++i)
    if (lg[i] > lg[best]) best = i;
  return static_cast<std::int32_t>(best);
}
}  // namespace

Status generate_greedy_mtp(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, const std::vector<std::int32_t>& prompt, int n_predict, int n_draft,
                           const std::vector<std::int32_t>& stops, std::vector<std::int32_t>& out, MtpStats* stats,
                           const ForwardOptions& opt) {
  out.clear();
  const auto is_stop = [&stops](std::int32_t t) { return std::find(stops.begin(), stops.end(), t) != stops.end(); };
  if (prompt.empty()) return Status::fail(Code::invalid_argument, "an empty prompt");
  if (n_predict <= 0) return Status::good();
  if (n_draft < 0) return Status::fail(Code::invalid_argument, "the draft length is 0 or more");
  if (n_draft > 0 && !w.mtp.present) return Status::fail(Code::unsupported, "this file has no MTP block to draft with");
  const int64_t e = s.n_embd, nv = s.n_vocab, P = static_cast<int64_t>(prompt.size());
  MtpStats local;
  MtpStats& stt = stats ? *stats : local;
  DecodeState ts;
  decode_state_reset(s, ts);
  MtpState ms;
  mtp_state_reset(ms);
  // prompt: the trunk's last logits and every position's hidden
  std::vector<float> lg(z(nv)), hid(z(P * e));
  ForwardOptions o = opt;
  o.hidden_out = hid.data();
  if (Status st = forward_decode(s, w, ts, prompt.data(), static_cast<int>(P), lg.data(), true, o); !st) return st;
  std::vector<float> pending(hid.end() - e, hid.end());  // h of the last position run by the trunk
  if (n_draft > 0) {  // the MTP block catches up on the prompt: token p with h of p - 1 (zeros before the first)
    std::vector<float> hp(z(P * e), 0.0f);
    std::copy(hid.begin(), hid.end() - e, hp.begin() + e);
    if (Status st = mtp_forward(s, w, ms, prompt.data(), hp.data(), static_cast<int>(P), nullptr, nullptr, opt); !st) return st;
  }
  std::int32_t last = argmax_row(lg.data(), nv);
  out.push_back(last);
  while (static_cast<int>(out.size()) < n_predict && !is_stop(last)) {
    ++stt.steps;
    // draft up to n_draft tokens from `last` (never more than the tokens still wanted after the trunk's own next one)
    const int room = n_predict - static_cast<int>(out.size()) - 1;
    const int k = std::max(0, std::min(n_draft, room));
    std::vector<std::int32_t> drafts;
    const std::int64_t m0 = ms.pos;
    if (k > 0) {
      std::vector<float> h(pending), dl(z(nv)), hn(z(e));
      std::int32_t tok = last;
      for (int i = 0; i < k; ++i) {
        if (Status st = mtp_forward(s, w, ms, &tok, h.data(), 1, dl.data(), hn.data(), opt); !st) return st;
        tok = argmax_row(dl.data(), nv);
        drafts.push_back(tok);
        h = hn;
      }
      mtp_state_truncate(s, ms, m0);  // the drafts' cache entries go; the kept tokens are run again below with the trunk's h
      stt.drafted += k;
    }
    // verify: the trunk runs `last` and the drafts as one batch (every row's logits and hidden)
    const int nb = 1 + k;
    std::vector<std::int32_t> batch{last};
    batch.insert(batch.end(), drafts.begin(), drafts.end());
    std::vector<float> bl(z(nb * nv)), bh(z(nb * e));
    const DecodeState saved = k > 0 ? ts : DecodeState{};
    o.hidden_out = bh.data();
    if (Status st = forward_decode(s, w, ts, batch.data(), nb, bl.data(), false, o); !st) return st;
    int a = 0;  // drafts kept: each must equal the trunk's own choice at the row before it
    while (a < k && argmax_row(bl.data() + a * nv, nv) == drafts[z(a)] && !is_stop(drafts[z(a)])) ++a;
    if (a < k && is_stop(drafts[z(a)]) && argmax_row(bl.data() + a * nv, nv) == drafts[z(a)]) ++a;  // a kept stop token ends the run
    const std::int32_t next = argmax_row(bl.data() + a * nv, nv);
    if (a < k) {  // the trunk's state must not hold the rejected drafts: back to before the batch, then the kept rows again
      ts = saved;
      std::vector<float> rl(z(nv));
      o.hidden_out = bh.data();
      if (Status st = forward_decode(s, w, ts, batch.data(), a + 1, rl.data(), true, o); !st) return st;
    }
    stt.accepted += a;
    if (n_draft > 0) {  // the MTP block catches up on the kept tokens, each paired with the trunk's h of the position before
      std::vector<float> hp(z((a + 1) * e));
      std::copy(pending.begin(), pending.end(), hp.begin());
      std::copy(bh.begin(), bh.begin() + a * e, hp.begin() + e);
      if (Status st = mtp_forward(s, w, ms, batch.data(), hp.data(), a + 1, nullptr, nullptr, opt); !st) return st;
    }
    pending.assign(bh.begin() + a * e, bh.begin() + (a + 1) * e);
    for (int i = 0; i < a && static_cast<int>(out.size()) < n_predict; ++i) out.push_back(drafts[z(i)]);
    if (!out.empty() && is_stop(out.back())) break;
    if (static_cast<int>(out.size()) < n_predict) out.push_back(next);
    last = next;
  }
  return Status::good();
}

namespace {
Status forward_impl(const Qwen35MoeSpec& s, const Qwen35MoeWeights& w, const std::int32_t* tokens, int n_tokens, int out_begin, int out_end,
                    float* logits, const ForwardOptions& opt, DecodeState* dst) {
  if (n_tokens <= 0 || out_begin < 0 || out_end > n_tokens || out_begin >= out_end)
    return Status::fail(Code::invalid_argument, "the token count or the output range is empty or out of range");
  if (w.layers.size() != s.n_layer) return Status::fail(Code::invalid_argument, "the weights were bound for another spec");
  if (Status st = check_act_rounding(opt, w); !st) return st;
  if (opt.act_gpu256_sub < 0 || opt.act_gpu256_sub > 2 || (opt.act_gpu256_sub != 0 && !opt.act_gpu256))
    return Status::fail(Code::invalid_argument, "the per-32 refinement needs the 256-value int8 format");
  if (opt.act_gpu256 && (!opt.act_int8_gpu || opt.act_bits != 8))
    return Status::fail(Code::invalid_argument, "the 256-value int8 format is an 8-bit gpu-path format (--act-int8-path gpu256)");
  if (opt.act_bits != 8 && (opt.act_bits != 4 || !opt.act_int8_gpu))
    return Status::fail(Code::invalid_argument, "4-bit activations need the gpu path (--act-int8-path gpu); only 8 and 4 bits exist");
  if (opt.neuron_skip < 0.0 || opt.neuron_skip >= 1.0) return Status::fail(Code::invalid_argument, "neuron skipping takes a fraction in [0, 1)");
  if (opt.h_gate < 0.0 || opt.h_gate > 1.0) return Status::fail(Code::invalid_argument, "the live H gate takes a fraction in (0, 1]");
  if (opt.cold_gpu256 && (!opt.act_int8_gpu || opt.act_bits != 8 || opt.hot.size() != static_cast<std::size_t>(s.n_layer) * s.n_expert))
    return Status::fail(Code::invalid_argument, "the cold-expert format needs 8-bit activations on the gpu path and a hot set of n_layer x n_expert entries");
  if (opt.swap_margin > 0.0f && opt.hot.size() != static_cast<std::size_t>(s.n_layer) * s.n_expert)
    return Status::fail(Code::invalid_argument, "the near-tie swap needs a hot set of n_layer x n_expert entries");
  if (opt.defer_experts < 0 || opt.defer_experts >= static_cast<int>(s.n_expert_used))
    return Status::fail(Code::invalid_argument, "expert deferral must keep at least one expert immediate");
  if (opt.planes && opt.t_decode) {
    bool any = w.output.planes != nullptr;
    for (const Qwen35MoeLayer& l : w.layers)
      for (const WeightView* p : {&l.wq, &l.wk, &l.wv, &l.wo, &l.wqkv, &l.wgate, &l.wbeta, &l.walpha, &l.ssm_out, &l.up_exps, &l.gate_exps, &l.down_exps,
                                  &l.up_shexp, &l.gate_shexp, &l.down_shexp})
        any = any || p->planes != nullptr;
    if (any) return Status::fail(Code::unsupported, "residual planes with the T-decode route are not built yet (the dequantized path reads them)");
  }
  if (opt.head_rescore < 0) return Status::fail(Code::invalid_argument, "the exact head re-score takes a count of 0 or more");
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
  for (Scratch& x : sc) {
    x.planes = opt.planes;
    x.fast = opt.fast_cpu;
  }
  Ctx c{s, threads, sc, n, opt};
  c.st = dst;
  c.pos0 = dst ? dst->pos : 0;

  // embedding rows
  std::vector<float> x(z(n * e)), xn(z(n * e)), y(z(n * e));
  std::vector<float> ydef_now(opt.defer_experts > 0 ? z(n * e) : 0), ydef_prev(opt.defer_experts > 0 ? z(n * e) : 0);
  for (int64_t t = 0; t < n; ++t) {
    const std::int32_t tok = tokens[t];
    if (tok < 0 || static_cast<uint64_t>(tok) >= s.n_vocab) return Status::fail(Code::invalid_argument, "a token id is outside the vocabulary");
    embed_row(w.tok_embd, tok, e, opt.planes, x.data() + t * e);
  }
  for (std::uint32_t il = 0; il < s.n_layer; ++il) {
    const Qwen35MoeLayer& l = w.layers[il];
    c.il = il;
    for (int64_t t = 0; t < n; ++t) rms_norm(x.data() + t * e, l.attn_norm.data(), xn.data() + t * e, e, s.rms_eps);
    if (opt.state_codes) {
      const int64_t cb = quant::state_code_bytes(e);
      for (int64_t t = 0; t < n; ++t) quant::state_code(xn.data() + t * e, e, opt.state_codes + (t * static_cast<int64_t>(s.n_layer) + il) * cb);
    }
    if (l.recurrent) delta_net(c, l, xn.data(), y.data());
    else {
      full_attention(c, l, xn.data(), y.data());
      ++c.ai;
    }
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
  if (dst) dst->pos += n;
  if (opt.hidden_out)  // GP3-08: every token's post-norm hidden, the MTP block's seed
    for (int64_t t = 0; t < n; ++t) rms_norm(x.data() + t * e, w.output_norm.data(), opt.hidden_out + t * e, e, s.rms_eps);
  if (!logits) return Status::good();  // the activations were all the caller wanted (importance matrix)
  const int64_t m = out_end - out_begin;
  for (int64_t t = 0; t < m; ++t) rms_norm(x.data() + (out_begin + t) * e, w.output_norm.data(), xn.data() + t * e, e, s.rms_eps);
  mm(c, w.output, xn.data(), m, e, logits, s.n_vocab);  // through mm(), so --act-int8 reaches the output layer too
  if (opt.planes && opt.head_rescore > 0 && w.output_exact.present()) {
    // the largest logits again, exactly: the original rows (BF16) times the unrounded fp32 input
    const int64_t nv = s.n_vocab, kk = std::min<int64_t>(opt.head_rescore, nv);
    parallel_items(m, threads, [&](int64_t t, int tid) {
      Scratch& hs = sc[z(tid)];
      float* lg = logits + t * nv;
      std::vector<int32_t> ord(z(nv));
      for (int64_t i = 0; i < nv; ++i) ord[z(i)] = static_cast<int32_t>(i);
      std::partial_sort(ord.begin(), ord.begin() + kk, ord.end(), [lg](int32_t a, int32_t b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
      if (hs.row.size() < z(e)) hs.row.resize(z(e));
      const float* h = xn.data() + t * e;
      for (int64_t j = 0; j < kk; ++j) {
        const int32_t v = ord[z(j)];
        (void)quant::dequantize_row(static_cast<quant::Type>(w.output_exact.type), w.output_exact.data + static_cast<uint64_t>(v) * w.output_exact.row_bytes,
                                    hs.row.data(), e);
        double acc = 0.0;
        for (int64_t i = 0; i < e; ++i) acc += static_cast<double>(hs.row[z(i)]) * h[i];
        lg[v] = static_cast<float>(acc);
      }
    });
  }
  return Status::good();
}
}  // namespace

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

namespace {
// one 32-value block: amax / 127 as in act_round_q8_256, round half to even, clamp to -127..127
void round32(float* x, float scale, float inv) {
  for (int j = 0; j < 32; ++j) x[j] = scale * std::min(127.0f, std::max(-127.0f, std::nearbyint(x[j] * inv)));
}
float amax_of(const float* x, int n) {
  float a = 0.0f;
  for (int j = 0; j < n; ++j) a = std::max(a, std::fabs(x[j]));
  return a;
}
}  // namespace

void act_round_q8_256x8(float* x, std::int64_t n) {
  for (std::int64_t i = 0; i < n; i += 32) {
    const float a = amax_of(x + i, 32);
    const float scale = a > 0.0f ? a / 127.0f : 1.0f;
    round32(x + i, scale, 1.0f / scale);
  }
}

void act_round_q8_256e(float* x, std::int64_t n) {
  for (std::int64_t i = 0; i < n; i += 256) {
    const float a256 = amax_of(x + i, 256);
    const float s = a256 > 0.0f ? a256 / 127.0f : 1.0f;
    const float inv = 1.0f / s;
    for (int b = 0; b < 8; ++b) {
      float* p = x + i + 32 * b;
      const float a = amax_of(p, 32);
      int e = 0;  // the largest e <= 3 whose scale s / 2^e still covers the block: a * 2^(e+1) <= a256
      while (e < 3 && std::ldexp(a, e + 1) <= a256) ++e;
      round32(p, std::ldexp(s, -e), std::ldexp(inv, e));
    }
  }
}
void h_gate_correct_q8z(const std::uint8_t* data, std::uint64_t row_bytes, std::int64_t n_out, std::int64_t k, const float* x,
                        std::int64_t n_tok, std::int64_t ldx, float* y, std::int64_t ldy, double f, int threads) {
  h_gate_impl(data, row_bytes, n_out, k, x, n_tok, ldx, y, ldy, f, threads);
}
void t_decode_q7zt(const std::uint8_t* data, std::uint64_t row_bytes, std::int64_t n_out, std::int64_t k, const float* x, std::int64_t n_tok,
                   std::int64_t ldx, float* y, std::int64_t ldy, int threads) {
  t_decode_impl(data, row_bytes, n_out, k, x, n_tok, ldx, y, ldy, threads);
}

}  // namespace pelias::model
