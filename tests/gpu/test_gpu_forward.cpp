// Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
// provenance: original
// GP3-12 rungs 2-3 (masterwork 16.1): the trunk on the GPU (model/gpu_forward.h) against the CPU reference forward pass
// (model/arch_qwen35moe.h forward_reference) with the matching options (the same per-32 int8 rounding before every
// quantized product). A small qwen35moe built here with quantized weights of the kinds the engine meets: Q4_K / Q5_K / Q6_K /
// Q4_TG2 products on the int8 tiles, Q8_0 and outputs not a multiple of 128 on the exact path, float router and vectors,
// routed experts of another type in each layer. Rung 2: one layer (a DeltaNet layer; the attention layer); rung 3: all four
// layers. Gates: mean KL(reference || GPU) over the positions <= 1e-4 and the top-1 token equal at >= 95 % of positions; the
// relative L2 of the logits and the largest difference are printed. Why not 1e-6 as for single products: every product's
// input is rounded to int8, and a value within float noise of a rounding boundary flips by one step (1/127 of its block's
// largest value) in one path and not the other; such flips are expected and grow layer by layer, so the whole-model gate is
// on what the outputs mean (the KL of the next-token distribution, 1e-4 = a tenth of the measured per-256 vs per-32 format
// difference 0.000999), not on bits.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../unit/gguf_builder.h"
#include "io/gguf_reader.h"
#include "kern/gpu_ops.h"
#include "model/arch_qwen35moe.h"
#include "model/gpu_forward.h"
#include "model/gpu_engine.h"
#include "model/spec.h"
#include "os/file_map.h"
#include "pelias_test.h"
#include "quant/block_types.h"
#include "quant/quant_cpu.h"

namespace q = pelias::quant;
namespace g = pelias::kern::gpu;
namespace m = pelias::model;

namespace {

struct Small {
  pelias::os::FileMap map;
  pelias::io::GgufReader reader;
  m::Qwen35MoeSpec spec;
  m::Qwen35MoeWeights w;
  bool ok = false;
};

std::uint32_t ty(q::Type t) { return static_cast<std::uint32_t>(t); }

// 4 blocks (DeltaNet x 3, full attention at block 3), n_embd 256, 8 experts of 256, 2 used; quantized as listed
void small_model(Small& sm) {
  ptest::GgufBuilder b;
  const std::string a = "qwen35moe.";
  b.kv_string("general.architecture", "qwen35moe")
      .kv_u32(a + "block_count", 4)
      .kv_u32(a + "nextn_predict_layers", 0)
      .kv_u32(a + "embedding_length", 256)
      .kv_u32(a + "attention.head_count", 4)
      .kv_u32(a + "attention.head_count_kv", 2)
      .kv_u32(a + "attention.key_length", 64)
      .kv_u32(a + "attention.value_length", 64)
      .kv_u32(a + "rope.dimension_count", 16)
      .kv_i32_array(a + "rope.dimension_sections", {3, 3, 2, 0})
      .kv_f32(a + "rope.freq_base", 1e7f)
      .kv_f32(a + "attention.layer_norm_rms_epsilon", 1e-6f)
      .kv_u32(a + "expert_count", 8)
      .kv_u32(a + "expert_used_count", 2)
      .kv_u32(a + "expert_feed_forward_length", 256)
      .kv_u32(a + "expert_shared_feed_forward_length", 256)
      .kv_u32(a + "ssm.conv_kernel", 4)
      .kv_u32(a + "ssm.state_size", 32)
      .kv_u32(a + "ssm.group_count", 4)
      .kv_u32(a + "ssm.time_step_rank", 8)
      .kv_u32(a + "ssm.inner_size", 256)
      .kv_u32(a + "full_attention_interval", 4);
  const std::uint64_t e = 256, hd = 64, nh = 4, nkv = 2, ch = 512, di = 256, ds = 32, nvh = 8, ne = 8, ff = 256, ffs = 256;
  const std::uint32_t f32 = 0;
  b.tensor("token_embd.weight", {e, 1000}, f32).tensor("output_norm.weight", {e}, f32).tensor("output.weight", {e, 1000}, ty(q::Type::q4_k));
  const q::Type exp_t[4] = {q::Type::q4_tg2, q::Type::q5_k, q::Type::q6_k, q::Type::q4_k};
  for (int il = 0; il < 4; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    b.tensor(p + "attn_norm.weight", {e}, f32).tensor(p + "post_attention_norm.weight", {e}, f32);
    if (il == 3) {
      b.tensor(p + "attn_q.weight", {e, 2 * hd * nh}, ty(q::Type::q4_k)).tensor(p + "attn_k.weight", {e, hd * nkv}, ty(q::Type::q5_k));
      b.tensor(p + "attn_v.weight", {e, hd * nkv}, ty(q::Type::q6_k)).tensor(p + "attn_output.weight", {hd * nh, e}, ty(q::Type::q4_tg2));
      b.tensor(p + "attn_q_norm.weight", {hd}, f32).tensor(p + "attn_k_norm.weight", {hd}, f32);
    } else {
      b.tensor(p + "attn_qkv.weight", {e, ch}, ty(q::Type::q4_k)).tensor(p + "attn_gate.weight", {e, di}, ty(q::Type::q5_k));
      b.tensor(p + "ssm_beta.weight", {e, nvh}, ty(q::Type::q8_0)).tensor(p + "ssm_alpha.weight", {e, nvh}, ty(q::Type::q8_0));
      b.tensor(p + "ssm_out.weight", {di, e}, ty(q::Type::q6_k)).tensor(p + "ssm_conv1d.weight", {4, ch}, f32);
      b.tensor(p + "ssm_dt.bias", {nvh}, f32).tensor(p + "ssm_a", {nvh}, f32).tensor(p + "ssm_norm.weight", {ds}, f32);
    }
    const std::uint32_t xt = ty(exp_t[il]);
    b.tensor(p + "ffn_gate_inp.weight", {e, ne}, f32).tensor(p + "ffn_up_exps.weight", {e, ff, ne}, xt).tensor(p + "ffn_gate_exps.weight", {e, ff, ne}, xt);
    b.tensor(p + "ffn_down_exps.weight", {ff, e, ne}, xt).tensor(p + "ffn_up_shexp.weight", {e, ffs}, ty(q::Type::q4_k));
    b.tensor(p + "ffn_gate_shexp.weight", {e, ffs}, ty(q::Type::q4_k)).tensor(p + "ffn_down_shexp.weight", {ffs, e}, ty(q::Type::q8_0));
    b.tensor(p + "ffn_gate_inp_shexp.weight", {e}, f32);
  }
  std::vector<std::uint8_t> bytes = b.build();
  pelias::os::FileMap m0;
  pelias::io::GgufReader r0;
  if (!ptest::write_and_map(ptest::data_dir() + "/gpu_forward_small_layout.gguf", bytes, m0) || !r0.parse(m0).ok()) return;
  std::uint32_t h = 20261031u;
  auto unit = [&h]() {  // xorshift, uniform in [-1, 1)
    h ^= h << 13;
    h ^= h >> 17;
    h ^= h << 5;
    return static_cast<float>(h) / 2147483648.0f - 1.0f;
  };
  for (const pelias::io::GgufTensor& t : r0.tensors()) {
    const std::string& nm = t.name;
    const std::uint64_t n = t.elements(), k = t.ne[0];
    std::vector<float> v(static_cast<std::size_t>(n));
    const float row = 1.0f / std::sqrt(static_cast<float>(k));
    for (std::uint64_t i = 0; i < n; ++i) {
      const float u = unit();
      float x = u * row;
      if (nm.find("norm") != std::string::npos) x = 1.0f + 0.1f * u;
      else if (nm.find("ssm_a") != std::string::npos) x = -(0.5f + 0.5f * std::fabs(u));
      else if (nm.find("ssm_dt") != std::string::npos) x = 0.1f * u;
      else if (nm.find("conv1d") != std::string::npos) x = 0.3f * u;
      else if (nm == "token_embd.weight") x = 0.5f * u;
      v[static_cast<std::size_t>(i)] = x;
    }
    std::uint8_t* d = bytes.data() + t.file_offset;
    if (t.type == 0) {
      std::memcpy(d, v.data(), v.size() * sizeof(float));
      continue;
    }
    const std::uint64_t rb = q::row_bytes(t.type, k);
    for (std::uint64_t r = 0; r < n / k; ++r)
      if (!q::quantize_row(static_cast<q::Type>(t.type), v.data() + r * k, d + r * rb, static_cast<std::int64_t>(k)).ok()) return;
  }
  if (!ptest::write_and_map(ptest::data_dir() + "/gpu_forward_small.gguf", bytes, sm.map) || !sm.reader.parse(sm.map).ok()) return;
  auto s = m::read_qwen35moe_spec(sm.reader);
  if (!s.ok()) return;
  sm.spec = s.value;
  auto w = m::bind_qwen35moe(sm.reader, sm.map.data(), sm.spec);
  if (!w.ok()) return;
  sm.w = std::move(w.value);
  sm.ok = true;
}

double kl(const float* p_log, const float* q_log, int n) {  // KL(softmax(p) || softmax(q)) in double
  double mp = p_log[0], mq = q_log[0];
  for (int i = 1; i < n; ++i) {
    mp = std::max(mp, static_cast<double>(p_log[i]));
    mq = std::max(mq, static_cast<double>(q_log[i]));
  }
  double sp = 0.0, sq = 0.0;
  for (int i = 0; i < n; ++i) {
    sp += std::exp(p_log[i] - mp);
    sq += std::exp(q_log[i] - mq);
  }
  const double lp = mp + std::log(sp), lq = mq + std::log(sq);
  double d = 0.0;
  for (int i = 0; i < n; ++i) {
    const double a = p_log[i] - lp, b = q_log[i] - lq;
    d += std::exp(a) * (a - b);
  }
  return d;
}

// one comparison: the model restricted to `layers` (indices into the trunk), n tokens
void compare(const Small& sm, const std::vector<int>& layers, int n, const char* what) {
  m::Qwen35MoeSpec s = sm.spec;
  m::Qwen35MoeWeights w = sm.w;
  w.layers.clear();
  for (const int il : layers) w.layers.push_back(sm.w.layers[static_cast<std::size_t>(il)]);
  s.n_layer = static_cast<std::uint32_t>(layers.size());
  std::vector<std::int32_t> tok(static_cast<std::size_t>(n));
  for (int t = 0; t < n; ++t) tok[static_cast<std::size_t>(t)] = static_cast<std::int32_t>((t * 37 + 11) % 1000);
  const int nv = static_cast<int>(s.n_vocab);
  std::vector<float> ref(static_cast<std::size_t>(n) * static_cast<std::size_t>(nv)), gpu(ref.size());
  m::ForwardOptions o = m::GpuForward::matching_reference_options();
  o.threads = 4;
  CHECK(m::forward_reference(s, w, tok.data(), n, 0, n, ref.data(), o).ok());
  auto f = m::GpuForward::open(s, w, 128);
  CHECK(f.ok());
  if (!f.ok()) {
    std::printf("    %s: open failed: %s\n", what, f.status.text().c_str());
    return;
  }
  m::GpuForwardStats st;
  const pelias::Status ps = f.value->prefill(tok.data(), n, 0, n, gpu.data(), &st);
  CHECK(ps.ok());
  if (!ps.ok()) {
    std::printf("    %s: prefill failed: %s\n", what, ps.text().c_str());
    return;
  }
  double num = 0.0, den = 0.0, worst = 0.0, kl_sum = 0.0, kl_max = 0.0;
  int top_same = 0;
  bool finite = true;
  for (int t = 0; t < n; ++t) {
    const float* r = ref.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(nv);
    const float* gq = gpu.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(nv);
    int ar = 0, ag = 0;
    for (int i = 0; i < nv; ++i) {
      finite = finite && std::isfinite(gq[i]);
      const double d = static_cast<double>(gq[i]) - r[i];
      num += d * d;
      den += static_cast<double>(r[i]) * r[i];
      worst = std::max(worst, std::fabs(d));
      if (r[i] > r[ar]) ar = i;
      if (gq[i] > gq[ag]) ag = i;
    }
    top_same += ar == ag ? 1 : 0;
    const double k = kl(r, gq, nv);
    kl_sum += k;
    kl_max = std::max(kl_max, k);
  }
  const double rel = den > 0 ? std::sqrt(num / den) : 0.0, kl_mean = kl_sum / n, top = static_cast<double>(top_same) / n;
  CHECK(finite);
  CHECK(kl_mean <= 1e-4);
  CHECK(top >= 0.95);
  std::printf("    %-34s %3d tokens: logits rel L2 %.3e, largest |diff| %.3e; KL mean %.3e max %.3e; top-1 equal %d / %d; GPU %.3f s (uploads %.3f s, %.1f MB)\n", what,
              n, rel, worst, kl_mean, kl_max, top_same, n, st.seconds, st.upload_seconds, static_cast<double>(st.bytes_uploaded) / 1e6);
}

}  // namespace

TEST(rung2_one_layer_on_the_gpu_against_the_reference) {
  if (!g::available()) return;
  Small sm;
  small_model(sm);
  CHECK(sm.ok);
  if (!sm.ok) return;
  const int n = std::getenv("PELIAS_GPU_QUICK") ? 24 : 96;
  compare(sm, {0}, n, "one DeltaNet layer (experts Q4_TG2)");
  compare(sm, {3}, n, "one attention layer (experts Q4_K)");
}

TEST(rung3_whole_trunk_on_the_gpu_against_the_reference) {
  if (!g::available()) return;
  Small sm;
  small_model(sm);
  CHECK(sm.ok);
  if (!sm.ok) return;
  const int n = std::getenv("PELIAS_GPU_QUICK") ? 24 : 96;
  compare(sm, {0, 1, 2, 3}, n, "four layers (all expert types)");
}

namespace {

// GP3-14 stages 1-2: the engine over a sequence fed in pieces (prefill chunks, then one token at a time) against the
// reference's fresh pass with the same roundings. hot: experts 0 .. hot-1 of every layer in the GPU bank, the rest read
// from the registered file mapping / the bit planes (v60: every pick on the GPU; 8: every expert in the bank).
std::vector<float> engine_case(const Small& sm, int hot, const char* what, float band = 0.0f, int cache = 0, bool graph = false, bool trace = false) {
  const m::Qwen35MoeSpec& s = sm.spec;
  const int n = std::getenv("PELIAS_GPU_QUICK") ? 20 : 48, nv = static_cast<int>(s.n_vocab);
  std::vector<std::int32_t> tok(static_cast<std::size_t>(n));
  for (int t = 0; t < n; ++t) tok[static_cast<std::size_t>(t)] = static_cast<std::int32_t>((t * 53 + 7) % 1000);
  std::vector<float> ref(static_cast<std::size_t>(n) * static_cast<std::size_t>(nv)), eng(ref.size());
  m::ForwardOptions o = m::GpuForward::matching_reference_options();
  o.threads = 4;
  o.router_tie_band = band;  // v24: the reference orders by the same band
  CHECK(m::forward_reference(s, sm.w, tok.data(), n, 0, n, ref.data(), o).ok());
  m::EngineOptions eo;
  eo.max_ctx = 128;
  eo.max_batch = 16;
  eo.trace = trace;  // v25: the trace only reads stamps and picks (v60: a side output, so the layers run the separate kernels)
  eo.reader_cache = cache;  // v26
  eo.graph = graph;
  eo.router_tie = band;
  if (hot > 0) {
    eo.hot_order.assign(s.n_layer, {});
    for (auto& v : eo.hot_order)
      for (int e = 0; e < static_cast<int>(s.n_expert); ++e) v.push_back(e);
    eo.max_hot_per_layer = hot;
  }
  auto f = m::GpuEngine::open(s, sm.w, eo);
  CHECK(f.ok());
  if (!f.ok()) {
    std::printf("    %s: open failed: %s\n", what, f.status.text().c_str());
    return {};
  }
  CHECK(f.value->bank_experts() == static_cast<int>(s.n_layer) * hot);
  {  // v24: the reader must be on (the small model's file mapping registered)
    const std::string sm_sum = f.value->summary();
    CHECK(sm_sum.find("expert reader on") != std::string::npos);
    std::printf("    %s: %s\n", what, sm_sum.substr(sm_sum.find("expert reader")).c_str());
  }
  // two passes: the second after reset() must give the same logits bit for bit (the state is carried, not leaked)
  std::vector<float> first;
  std::uint64_t hp = 0;
  for (int pass = 0; pass < 2; ++pass) {
    CHECK(f.value->reset().ok());
    int t = 0;
    const int chunk[2] = {16, 13};  // prefill pieces, then single tokens
    for (int c = 0; c < 2 && t < n; ++c) {
      const int k = std::min(chunk[c], n - t);
      m::EngineStats es;
      const pelias::Status ps = f.value->step(tok.data() + t, k, eng.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(nv), false, &es);
      CHECK(ps.ok());
      if (!ps.ok()) {
        std::printf("    %s: step failed: %s\n", what, ps.text().c_str());
        return {};
      }
      hp += es.hot_picks;
      t += k;
    }
    for (; t < n; ++t) {
      m::EngineStats es;
      CHECK(f.value->step(tok.data() + t, 1, eng.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(nv), true, &es).ok());
      hp += es.hot_picks;
    }
    CHECK(f.value->pos() == n);
    if (trace) {  // v25: one S line per step, one L line per layer
      const std::string tr = f.value->trace_take();
      std::size_t s_lines = 0, l_lines = 0;
      for (std::size_t i = 0; i < tr.size();) {
        const std::size_t j = tr.find('\n', i);
        if (tr.compare(i, 2, "S\t") == 0) ++s_lines;
        if (tr.compare(i, 2, "L\t") == 0) ++l_lines;
        i = j == std::string::npos ? tr.size() : j + 1;
      }
      // v48: with the prompt on the GPU, the prefill pieces (16, 13) are steps of up to 4 too: 4 + 4 steps
      const std::size_t pre = n >= 29 ? 8 : 0;
      CHECK(s_lines == pre + static_cast<std::size_t>(std::max(0, n - 29)) && l_lines == s_lines * s.n_layer);
      if (pass == 0) std::printf("    %s: trace %zu steps, %zu layer lines\n", what, s_lines, l_lines);
    }
    if (pass == 0) first = eng;
    else CHECK(std::memcmp(first.data(), eng.data(), eng.size() * sizeof(float)) == 0);
  }
  double num = 0.0, den = 0.0, kl_sum = 0.0, kl_max = 0.0;
  int top_same = 0;
  bool finite = true;
  for (int t = 0; t < n; ++t) {
    const float* r = ref.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(nv);
    const float* gq = eng.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(nv);
    int ar = 0, ag = 0;
    for (int i = 0; i < nv; ++i) {
      finite = finite && std::isfinite(gq[i]);
      const double d = static_cast<double>(gq[i]) - r[i];
      num += d * d;
      den += static_cast<double>(r[i]) * r[i];
      if (r[i] > r[ar]) ar = i;
      if (gq[i] > gq[ag]) ag = i;
    }
    top_same += ar == ag ? 1 : 0;
    const double k = kl(r, gq, nv);
    kl_sum += k;
    kl_max = std::max(kl_max, k);
  }
  const double kl_mean = kl_sum / n, top = static_cast<double>(top_same) / n;
  CHECK(finite);
  // v60: with a tie band the picks depend on bucket boundaries (floor((logit - max) / band)); the engine's router logits
  // differ from the reference's by sum order alone, and on this synthetic model (near-ties everywhere) one boundary crossing
  // flips an expert at one position (MEASURED v60at: KL mean 1.156e-04, max 5.0e-03, top-1 48 / 48; the fused router and
  // router_topk give the same bits). The KL gate holds for the band-less cases; a band case keeps the top-1 gate and the
  // GPU-vs-GPU comparison its caller makes.
  if (band == 0.0f) CHECK(kl_mean <= 1e-4);
  CHECK(top >= 0.95);
  CHECK(hp == 2 * static_cast<std::uint64_t>(n) * s.n_expert_used);  // v25 / v60: every pick of every step on the GPU (counted per step), both passes
  std::printf("    %-34s %3d tokens: logits rel L2 %.3e; KL mean %.3e max %.3e; top-1 equal %d / %d; picks %llu (all on the GPU); reset repeat bit for bit\n", what, n,
              den > 0 ? std::sqrt(num / den) : 0.0, kl_mean, kl_max, top_same, n, static_cast<unsigned long long>(hp / 2));
  return eng;
}

}  // namespace

TEST(rung4_engine_decode_against_the_reference) {
  if (!g::available()) return;
  Small sm;
  small_model(sm);
  CHECK(sm.ok);
  if (!sm.ok) return;
  engine_case(sm, 0, "engine, no bank (planes / file mapping)");
  engine_case(sm, 0, "engine, no bank, trace (separate kernels)", 0.0f, 0, false, true);
  engine_case(sm, 4, "engine, half the experts in the bank");
  engine_case(sm, 8, "engine, every expert in the bank");
  // v60: the tie band through the fused half's fast router vs through router_topk (the separate kernels, forced by the trace):
  // the reference orders by the same band; a gap between the two is the fused router's tie rule
  const std::vector<float> b0 = engine_case(sm, 4, "engine, tie band 0.01 (fused router)", 0.01f);
  const std::vector<float> b1 = engine_case(sm, 4, "engine, tie band 0.01, trace (router_topk)", 0.01f, 0, false, true);
  const bool band_eq = !b0.empty() && b0.size() == b1.size() && std::memcmp(b0.data(), b1.data(), b0.size() * sizeof(float)) == 0;
  CHECK(band_eq);
  std::printf("    tie band 0.01: fused router vs router_topk logits bit for bit %s\n", band_eq ? "equal" : "DIFFERENT");
  // v26: the cache changes where bytes come from, never the arithmetic: logits bit for bit with and without it
  const std::vector<float> r0 = engine_case(sm, 2, "engine, bank 2", 0.0f, 0);
  const std::vector<float> r1 = engine_case(sm, 2, "engine, bank 2 + cache 3", 0.0f, 3);
  CHECK(!r0.empty() && r0.size() == r1.size() && std::memcmp(r0.data(), r1.data(), r0.size() * sizeof(float)) == 0);
  std::printf("    reader cache on vs off: logits bit for bit %s\n",
              !r0.empty() && r0.size() == r1.size() && std::memcmp(r0.data(), r1.data(), r0.size() * sizeof(float)) == 0 ? "equal" : "DIFFERENT");
  // v26: the graph moves work, never the arithmetic: bit for bit with the plain launches
  const std::vector<float> r3 = engine_case(sm, 2, "engine, bank 2 + cache 3 + graph", 0.0f, 3, true);
  const bool e3 = !r1.empty() && r1.size() == r3.size() && std::memcmp(r1.data(), r3.data(), r1.size() * sizeof(float)) == 0;
  CHECK(e3);
  std::printf("    graph: logits bit for bit %s\n", e3 ? "equal" : "DIFFERENT");
}

// GP3-14 stage 3: verify batches and keep() (the MTP rollback) on the small model: after a 16-token prefill, batches of 4
// are verified and only their first k tokens kept (k cycling 1 .. 4); the kept rows' logits against the reference's
// fresh pass over the kept sequence, and the position after each keep.
// v26 option B and the CUDA graph: tokens decoded one at a time and the same tokens verified 4 at a time give the same
// logits bit for bit (each product's per-token arithmetic is the one-token path's; v60: every 1-4-token step is on R3);
// with the graph on, the same again and equal to the plain launches (replays from the third step of each kind)
TEST(rung4_engine_batch_identical_and_graph) {
  if (!g::available()) return;
  Small sm;
  small_model(sm);
  CHECK(sm.ok);
  if (!sm.ok) return;
  const m::Qwen35MoeSpec& s = sm.spec;
  const int nv = static_cast<int>(s.n_vocab), npre = 16, ndec = 12;
  const std::size_t row = static_cast<std::size_t>(nv);
  std::vector<std::int32_t> tok(static_cast<std::size_t>(npre + ndec));
  for (std::size_t i = 0; i < tok.size(); ++i) tok[i] = static_cast<std::int32_t>((i * 37 + 11) % 1000);
  std::vector<std::vector<float>> one(2), ver(2);
  // (v47) the R3 expert stage with staged RAM experts: verify batches run the kernels that load each row once for every
  // token (lane_part_n), one-token steps the one-token kernels; both must give the same logits
  for (int cfg = 0; cfg < 2; ++cfg) {  // 0: plain launches; 1: the graph
    m::EngineOptions eo;
    eo.max_ctx = 128;
    eo.max_batch = 16;
    eo.max_verify = 4;
    eo.hot_order.assign(s.n_layer, {});
    for (auto& v : eo.hot_order)
      for (int e = 0; e < static_cast<int>(s.n_expert); ++e) v.push_back(e);
    eo.max_hot_per_layer = 2;
    eo.graph = cfg == 1;
    auto f = m::GpuEngine::open(s, sm.w, eo);
    CHECK(f.ok());
    if (!f.ok()) {
      std::printf("    batch-identical / graph: open failed: %s\n", f.status.text().c_str());
      return;
    }
    m::GpuEngine& E = *f.value;
    std::vector<float> pre(static_cast<std::size_t>(npre) * row);
    one[static_cast<std::size_t>(cfg)].assign(static_cast<std::size_t>(ndec) * row, 0.0f);
    ver[static_cast<std::size_t>(cfg)].assign(static_cast<std::size_t>(ndec) * row, 0.0f);
    CHECK(E.reset().ok());
    CHECK(E.step(tok.data(), npre, pre.data(), false).ok());
    for (int i = 0; i < ndec; ++i)
      CHECK(E.step(tok.data() + npre + i, 1, one[static_cast<std::size_t>(cfg)].data() + static_cast<std::size_t>(i) * row, true).ok());
    CHECK(E.reset().ok());
    CHECK(E.step(tok.data(), npre, pre.data(), false).ok());
    std::vector<float> bh(4 * static_cast<std::size_t>(s.n_embd));
    for (int b = 0; b < ndec / 4; ++b) {
      CHECK(E.verify(tok.data() + npre + 4 * b, 4, ver[static_cast<std::size_t>(cfg)].data() + static_cast<std::size_t>(4 * b) * row, bh.data()).ok());
      CHECK(E.keep(4).ok());
    }
    std::printf("    config %d (%s): %s\n", cfg, cfg == 0 ? "plain launches" : "graph", E.graph_report().c_str());
    if (cfg == 0) std::printf("      %s\n", E.summary().c_str());
    if (cfg == 1) CHECK(E.graph_report().find("replays 0") == std::string::npos);
  }
  const std::size_t nb = static_cast<std::size_t>(ndec) * row * sizeof(float);
  const bool bi_eq = std::memcmp(one[0].data(), ver[0].data(), nb) == 0;
  const bool gr_eq = std::memcmp(one[1].data(), ver[1].data(), nb) == 0 && std::memcmp(one[0].data(), one[1].data(), nb) == 0;
  CHECK(bi_eq);
  CHECK(gr_eq);
  std::printf("    one token at a time vs verify batches of 4 (R3 expert stage, multi-token loads): plain launches %s; with the graph %s (and equal to plain launches)\n",
              bi_eq ? "bit for bit" : "DIFFER", gr_eq ? "bit for bit" : "DIFFER");
}

// v52 (token kernel 3a): the fused MoE half (one cooperative launch per layer: residual, post norm, rounding, router, shared
// gate, picks, grouping with the reader cache, plan, R3 expert stage, combine, residual) against the separate kernels (v60:
// a layer runs them when a side output needs them; here the trace): the prompt (GPU steps of up to 4 tokens), one-token
// decode and verify batches of 4, with the graph, logits bit for bit
TEST(rung4_engine_fused_moe_half) {
  if (!g::available()) return;
  Small sm;
  small_model(sm);
  CHECK(sm.ok);
  if (!sm.ok) return;
  const m::Qwen35MoeSpec& s = sm.spec;
  const int nv = static_cast<int>(s.n_vocab), npre = 16, ndec = 12;
  const std::size_t row = static_cast<std::size_t>(nv);
  std::vector<std::int32_t> tok(static_cast<std::size_t>(npre + ndec));
  for (std::size_t i = 0; i < tok.size(); ++i) tok[i] = static_cast<std::int32_t>((i * 41 + 3) % 1000);
  std::vector<float> out[2];
  std::string summ[2];
  for (int fz = 0; fz < 2; ++fz) {
    m::EngineOptions eo;
    eo.max_ctx = 128;
    eo.max_batch = 16;
    eo.max_verify = 4;
    eo.hot_order.assign(s.n_layer, {});
    for (auto& v : eo.hot_order)
      for (int e = 0; e < static_cast<int>(s.n_expert); ++e) v.push_back(e);
    eo.max_hot_per_layer = 2;
    eo.reader_cache = 3;
    eo.graph = true;
    eo.router_tie = 0.01f;
    eo.trace = fz == 0;  // v60: the trace is a side output, so every layer runs the separate kernels
    auto f = m::GpuEngine::open(s, sm.w, eo);
    CHECK(f.ok());
    if (!f.ok()) {
      std::printf("    fused MoE half: open failed: %s\n", f.status.text().c_str());
      return;
    }
    m::GpuEngine& E = *f.value;
    std::vector<float>& o = out[fz];
    o.assign(static_cast<std::size_t>(npre + 2 * ndec) * row, 0.0f);
    CHECK(E.reset().ok());
    CHECK(E.step(tok.data(), npre, o.data(), false).ok());
    for (int i = 0; i < ndec; ++i) CHECK(E.step(tok.data() + npre + i, 1, o.data() + static_cast<std::size_t>(npre + i) * row, false).ok());
    CHECK(E.reset().ok());
    std::vector<float> pre(static_cast<std::size_t>(npre) * row);
    CHECK(E.step(tok.data(), npre, pre.data(), false).ok());
    std::vector<float> bh(4 * static_cast<std::size_t>(s.n_embd));
    for (int b = 0; b < ndec / 4; ++b) {
      CHECK(E.verify(tok.data() + npre + 4 * b, 4, o.data() + static_cast<std::size_t>(npre + ndec + 4 * b) * row, bh.data()).ok());
      CHECK(E.keep(4).ok());
    }
    summ[fz] = E.summary();
    if (fz == 0) CHECK(!E.trace_take().empty());
  }
  const bool noted = summ[1].find("MoE half fused") != std::string::npos;
  const bool eq = out[0].size() == out[1].size() && std::memcmp(out[0].data(), out[1].data(), out[0].size() * sizeof(float)) == 0;
  CHECK(noted);
  CHECK(eq);
  std::printf("    fused MoE half vs separate kernels (prompt 16 in GPU steps, 12 one-token steps, 3 verify batches of 4, graph): logits %s; summary %s\n",
              eq ? "bit for bit" : "DIFFER", noted ? "names the fused path" : "MISSING the fused note");
}

// v54b / v54c: the sliding bit planes (kern/tg2_planes.h) in the R3 expert stage, one slider for every pick (--plane-bits),
// Q4_TG2 (layer 0) and Q5_K (layer 1) matrices, bank and cache. 5 bits = the stored layout bit for bit; 4 (Q5_K cut), 3 and
// 2 bits change the logits (finite), one token = verify batches of 4; the slots hold the slider's prefix (packed rows): no
// cache, 1 slot or 4 slots, staged misses or every expert staged give the same bits; the VRAM probe (--auto-vram) runs
// and, at 5 bits, changes nothing
TEST(rung4_engine_bit_planes) {
  if (!g::available()) return;
  Small sm;
  small_model(sm);
  CHECK(sm.ok);
  if (!sm.ok) return;
  const m::Qwen35MoeSpec& s = sm.spec;
  const int nv = static_cast<int>(s.n_vocab), npre = 12, ndec = 12;
  const std::size_t row = static_cast<std::size_t>(nv);
  std::vector<std::int32_t> tok(static_cast<std::size_t>(npre + ndec));
  for (std::size_t i = 0; i < tok.size(); ++i) tok[i] = static_cast<std::int32_t>((i * 29 + 5) % 1000);
  struct Cfg {
    int bits, cache;
    bool probe;
  };
  auto run = [&](const Cfg& c, std::string* summ) {
    m::EngineOptions eo;
    eo.max_ctx = 128;
    eo.max_batch = 16;
    eo.max_verify = 4;
    eo.hot_order.assign(s.n_layer, {});
    for (auto& v : eo.hot_order)
      for (int e = 0; e < static_cast<int>(s.n_expert); ++e) v.push_back(e);
    eo.max_hot_per_layer = 2;
    eo.reader_cache = c.cache;
    eo.graph = true;
    eo.router_tie = 0.01f;
    eo.plane_bits = c.bits;
    eo.auto_vram = c.probe;
    std::vector<float> o;
    auto f = m::GpuEngine::open(s, sm.w, eo);
    CHECK(f.ok());
    if (!f.ok()) {
      std::printf("    bit planes: open failed: %s\n", f.status.text().c_str());
      return o;
    }
    m::GpuEngine& E = *f.value;
    o.assign(static_cast<std::size_t>(npre + 2 * ndec) * row, 0.0f);
    bool ok = E.reset().ok() && E.step(tok.data(), npre, o.data(), false).ok();
    for (int i = 0; i < ndec; ++i) ok = ok && E.step(tok.data() + npre + i, 1, o.data() + static_cast<std::size_t>(npre + i) * row, false).ok();
    std::vector<float> pre(static_cast<std::size_t>(npre) * row);
    ok = ok && E.reset().ok() && E.step(tok.data(), npre, pre.data(), false).ok();
    std::vector<float> bh(4 * static_cast<std::size_t>(s.n_embd));
    for (int b = 0; b < ndec / 4; ++b) {
      ok = ok && E.verify(tok.data() + npre + 4 * b, 4, o.data() + static_cast<std::size_t>(npre + ndec + 4 * b) * row, bh.data()).ok();
      ok = ok && E.keep(4).ok();
    }
    CHECK(ok);
    if (summ) *summ = E.summary();
    return o;
  };
  auto same = [](const std::vector<float>& a, const std::vector<float>& b) {
    return !a.empty() && a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
  };
  auto finite = [](const std::vector<float>& a) {
    for (const float x : a)
      if (!std::isfinite(x)) return false;
    return !a.empty();
  };
  auto one_vs_ver = [&](const std::vector<float>& a) {
    return !a.empty() && std::memcmp(a.data() + static_cast<std::size_t>(npre) * row, a.data() + static_cast<std::size_t>(npre + ndec) * row,
                                     static_cast<std::size_t>(ndec) * row * sizeof(float)) == 0;
  };
  std::string summ, summp;
  const std::vector<float> base = run({5, 3, false}, &summ);  // v60: 5 bits (every code as stored) is the base
  const bool full_eq = finite(base) && one_vs_ver(base);
  CHECK(full_eq);
  CHECK(summ.find("bit planes:") != std::string::npos);
  const std::vector<float> p4 = run({4, 3, false}, nullptr), p2 = run({2, 3, false}, nullptr), p3 = run({3, 3, false}, nullptr);
  const bool lossy = !same(p4, base) && !same(p2, base) && !same(p3, base) && !same(p2, p3) && !same(p3, p4) && finite(p2) && finite(p3) && finite(p4);
  CHECK(lossy);
  const bool det = one_vs_ver(p2) && one_vs_ver(p3) && one_vs_ver(p4);
  CHECK(det);
  const std::vector<float> c0 = run({2, 0, false}, nullptr), c1 = run({2, 1, false}, nullptr), c4 = run({2, 4, false}, nullptr);
  const bool cache_free = same(p2, c0) && same(p2, c1) && same(p2, c4);
  CHECK(cache_free);
  const std::vector<float> pr = run({5, 3, true}, &summp);
  const bool probe = same(base, pr) && summp.find("VRAM probe:") != std::string::npos;
  CHECK(probe);
  std::printf("    bit planes: 5 bits (stored layout) finite, one token = verify batches %s; 4 / 3 / 2 bits change the logits %s, one token = verify batches %s; "
              "2 bits with no cache / 1 / 3 / 4 slots %s; VRAM probe %s\n",
              full_eq ? "bit for bit" : "DIFFER", lossy ? "(finite)" : "NOT AS EXPECTED", det ? "bit for bit" : "DIFFER",
              cache_free ? "bit for bit" : "DIFFER", probe ? "runs, same bits at 5" : "WRONG");
  std::printf("      %s\n", summp.c_str());
}

TEST(rung4_engine_verify_and_keep) {
  if (!g::available()) return;
  Small sm;
  small_model(sm);
  CHECK(sm.ok);
  if (!sm.ok) return;
  const m::Qwen35MoeSpec& s = sm.spec;
  const int nv = static_cast<int>(s.n_vocab), n = std::getenv("PELIAS_GPU_QUICK") ? 26 : 46;
  std::vector<std::int32_t> tok(static_cast<std::size_t>(n));
  for (int t = 0; t < n; ++t) tok[static_cast<std::size_t>(t)] = static_cast<std::int32_t>((t * 71 + 5) % 1000);
  std::vector<float> ref(static_cast<std::size_t>(n) * static_cast<std::size_t>(nv)), eng(ref.size());
  m::ForwardOptions o = m::GpuForward::matching_reference_options();
  o.threads = 4;
  CHECK(m::forward_reference(s, sm.w, tok.data(), n, 0, n, ref.data(), o).ok());
  m::EngineOptions eo;
  eo.max_ctx = 128;
  eo.max_batch = 16;
  eo.max_verify = 4;
  eo.hot_order.assign(s.n_layer, {});
  for (auto& v : eo.hot_order)
    for (int e = 0; e < static_cast<int>(s.n_expert); ++e) v.push_back(e);
  eo.max_hot_per_layer = 4;
  auto f = m::GpuEngine::open(s, sm.w, eo);
  CHECK(f.ok());
  if (!f.ok()) {
    std::printf("    verify / keep: open failed: %s\n", f.status.text().c_str());
    return;
  }
  m::GpuEngine& E = *f.value;
  const std::size_t row = static_cast<std::size_t>(nv);
  CHECK(E.step(tok.data(), 16, eng.data(), false).ok());
  CHECK(!E.keep(1).ok());  // no verify batch pending
  int t = 16, kk = 1, batches = 0;
  std::vector<float> bl(4 * row), bh(4 * static_cast<std::size_t>(s.n_embd));
  while (t < n) {
    const int nb = std::min(4, n - t + 0), keepk = std::min(kk, nb);
    // the batch: the next nb tokens of the sequence (as if drafted), of which keepk are kept
    CHECK(E.verify(tok.data() + t, nb, bl.data(), bh.data()).ok());
    CHECK(!E.keep(nb + 1).ok());
    CHECK(E.keep(keepk).ok());
    CHECK(E.pos() == t + keepk);
    std::memcpy(eng.data() + static_cast<std::size_t>(t) * row, bl.data(), static_cast<std::size_t>(keepk) * row * sizeof(float));
    t += keepk;
    kk = kk % 4 + 1;
    ++batches;
  }
  CHECK(E.verify(tok.data(), 5, bl.data(), bh.data()).ok() == false);  // above max_verify
  double kl_sum = 0.0, kl_max = 0.0;
  int top_same = 0;
  for (int i = 0; i < n; ++i) {
    const float* r = ref.data() + static_cast<std::size_t>(i) * row;
    const float* gq = eng.data() + static_cast<std::size_t>(i) * row;
    int ar = 0, ag = 0;
    for (int j = 0; j < nv; ++j) {
      if (r[j] > r[ar]) ar = j;
      if (gq[j] > gq[ag]) ag = j;
    }
    top_same += ar == ag ? 1 : 0;
    const double k = kl(r, gq, nv);
    kl_sum += k;
    kl_max = std::max(kl_max, k);
  }
  CHECK(kl_sum / n <= 1e-4);
  CHECK(static_cast<double>(top_same) / n >= 0.95);
  std::printf("    verify / keep: %d batches of up to 4 (kept 1, 2, 3, 4 in turn), %d tokens: KL mean %.3e max %.3e; top-1 equal %d / %d\n", batches, n,
              kl_sum / n, kl_max, top_same, n);
}
