// provenance: original (the qwen35moe conversion READ-SRC from llama.cpp conversion/qwen.py (Qwen3_5MoeTextModel and its
//   bases _Qwen35MRopeMixin, _LinearAttentionVReorderBase, Qwen3NextModel, _QwenMtpMixin, Qwen2MoeModel) and
//   conversion/base.py (TextModel.set_gguf_parameters, prepare_tensors) at commit 2ca15f5404760548c39e7b92bd43116a09414a1a,
//   and the tensor names from gguf-py/gguf/tensor_mapping.py there; used as a specification, no code copied)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "ingest/stage_a.h"
#include "quant/block_types.h"

namespace pelias::ingest {
namespace {

using io::GgufKvBlob;
using io::Json;

constexpr auto kBF16 = static_cast<std::uint32_t>(quant::Type::bf16);
constexpr auto kF32 = static_cast<std::uint32_t>(quant::Type::f32);

Status bad(const std::string& msg, const std::string& name = {}) { return Status::fail(Code::unsupported, msg, 0, name); }

// find_hparam: the first of `keys` present in the hyperparameters (present with a null value counts as present)
const Json* find(const Json& h, std::initializer_list<const char*> keys) {
  for (const char* k : keys)
    if (const Json* v = h.get(k)) return v;
  return nullptr;
}

Result<std::uint32_t> as_u32(const Json* v, const char* what) {
  if (!v || !v->is_integer() || v->as_int() < 0 || v->as_uint() > 0xFFFFFFFFull)
    return Result<std::uint32_t>::fail(bad(std::string("hyperparameter ") + what + " is not a whole number that fits 32 bits", "config.json"));
  return Result<std::uint32_t>::good(static_cast<std::uint32_t>(v->as_uint()));
}

Result<float> as_f32(const Json* v, const char* what) {
  if (!v || !v->is_number() || v->is_bool() || !(std::fabs(v->as_double()) <= 3.4028234663852886e38))
    return Result<float>::fail(bad(std::string("hyperparameter ") + what + " is not a number a 32-bit float holds", "config.json"));
  return Result<float>::good(static_cast<float>(v->as_double()));
}

}  // namespace

Result<ArchPlan> plan_qwen35moe(const Json& h, const std::vector<SourceRef>& sources, std::uint32_t converter_file_type) {
  using R = Result<ArchPlan>;
  ArchPlan p;
  p.arch = "qwen35moe";
  const std::string a = p.arch + ".";

  // ---- block count: the layers plus the MTP block(s)
  const Json* nl = find(h, {"n_layers", "num_hidden_layers", "n_layer", "num_layers"});
  const Json* nhl = h.get("num_hidden_layers");
  auto n_layer_r = as_u32(nhl, "num_hidden_layers");
  if (!nl || !n_layer_r.ok()) return R::fail(n_layer_r.ok() ? bad("no layer count", "config.json") : n_layer_r.status);
  const std::uint32_t n_layer = n_layer_r.value;
  std::uint32_t n_mtp = 0;
  if (const Json* m = h.get("mtp_num_hidden_layers"); m && !m->is_null()) {
    auto r = as_u32(m, "mtp_num_hidden_layers");
    if (!r.ok()) return R::fail(r.status);
    n_mtp = r.value;
  }
  if (n_mtp == 0) {  // the reference counts the mtp.layers.N tensors instead
    for (const SourceRef& s : sources)
      if (s.name.rfind("model.layers.", 0) == 0) {
        const std::uint64_t bid = std::strtoull(s.name.c_str() + 13, nullptr, 10);
        if (bid >= n_layer) n_mtp = std::max<std::uint32_t>(n_mtp, static_cast<std::uint32_t>(bid - n_layer + 1));
      }
    if (n_mtp == 0) return R::fail(bad("the model has no MTP block (the reference requires one for this architecture)"));
  }
  const std::uint32_t block_count = n_layer + n_mtp;

  // ---- rope parameters, as TextModel.__init__ completes them
  const Json* rp_src = h.get("rope_parameters");
  if (!rp_src) rp_src = h.get("rope_scaling");
  Json rp = rp_src && rp_src->is_object() ? *rp_src : Json::object();
  if (rp_src && !rp_src->is_null() && !rp_src->is_object()) return R::fail(bad("rope_parameters is not an object", "config.json"));
  if (!rp.get("full_attention") && !rp.get("sliding_attention")) {
    if (find(h, {"local_rope_theta", "rope_local_theta", "rope_theta_local", "swa_rope_theta", "rope_local_base_freq"}))
      return R::fail(bad("a sliding-window rope theta is not read yet", "config.json"));
    const Json* theta = find(h, {"global_rope_theta", "rope_global_theta", "rope_theta_global", "rope_theta", "rotary_emb_base"});
    if (!rp.get("rope_theta") && theta && !theta->is_null()) rp.set("rope_theta", *theta);
    if (!rp.get("rope_type") && rp.get("type") && !rp.get("type")->is_null()) rp.set("rope_type", *rp.get("type"));
    const Json* prf = find(h, {"partial_rotary_factor", "rope_pct", "rope_percent"});
    if (!rp.get("partial_rotary_factor") && prf && !prf->is_null()) rp.set("partial_rotary_factor", *prf);
    const Json* omp = h.get("original_max_position_embeddings");
    if (!rp.get("original_max_position_embeddings") && omp && !omp->is_null()) rp.set("original_max_position_embeddings", *omp);
  } else {
    return R::fail(bad("per-attention-type rope parameters are not read yet", "config.json"));
  }

  // ---- metadata entries in the reference's order
  std::vector<GgufKvBlob>& kv = p.kv;
  kv.push_back(io::gguf_kv_u32(a + "block_count", block_count));
  auto opt_u32 = [&](std::initializer_list<const char*> keys, const std::string& key, const char* what) -> Status {
    const Json* v = find(h, keys);
    if (!v || v->is_null()) return Status::good();
    auto r = as_u32(v, what);
    if (!r.ok()) return r.status;
    kv.push_back(io::gguf_kv_u32(key, r.value));
    return Status::good();
  };
  Status s = opt_u32({"max_position_embeddings", "n_ctx", "n_positions", "max_length", "max_sequence_length", "model_max_length"}, a + "context_length", "context length");
  if (!s) return R::fail(s);
  s = opt_u32({"hidden_size", "n_embd", "dim"}, a + "embedding_length", "hidden_size");
  if (!s) return R::fail(s);
  s = opt_u32({"prefix_dense_intermediate_size", "dense_intermediate_size", "intermediate_size", "n_inner", "hidden_dim"}, a + "feed_forward_length", "intermediate_size");
  if (!s) return R::fail(s);
  s = opt_u32({"num_attention_heads", "n_head", "n_heads"}, a + "attention.head_count", "num_attention_heads");
  if (!s) return R::fail(s);
  s = opt_u32({"num_key_value_heads", "n_kv_heads"}, a + "attention.head_count_kv", "num_key_value_heads");
  if (!s) return R::fail(s);
  if (const Json* c = h.get("is_causal"); c && c->is_bool() && !c->as_bool()) kv.push_back(io::gguf_kv_bool(a + "attention.causal", false));
  // rope scaling: only "no scaling" ("default" or absent) is read yet
  if (const Json* rt = rp.get("rope_type"); rt && !rt->is_null()) {
    if (!rt->is_string() || rt->as_string() != "default") return R::fail(bad("rope scaling '" + (rt->is_string() ? rt->as_string() : std::string("?")) + "' is not read yet", "config.json"));
  }
  if (const Json* ms = rp.get("mrope_section"); ms && !ms->is_null()) {
    if (!ms->is_array()) return R::fail(bad("mrope_section is not a list", "config.json"));
    std::vector<std::int32_t> sec;
    for (const Json& x : ms->items()) {
      if (!x.is_integer() || x.as_int() < INT32_MIN || x.as_int() > INT32_MAX) return R::fail(bad("mrope_section holds a value that is not a 32-bit integer", "config.json"));
      sec.push_back(static_cast<std::int32_t>(x.as_int()));
    }
    while (sec.size() < 4) sec.push_back(0);
    sec.resize(4);
    kv.push_back(io::gguf_kv_arr_i32(a + "rope.dimension_sections", sec));
  }
  if (const Json* th = rp.get("rope_theta"); th && !th->is_null()) {
    auto r = as_f32(th, "rope_theta");
    if (!r.ok()) return R::fail(r.status);
    kv.push_back(io::gguf_kv_f32(a + "rope.freq_base", r.value));
  }
  if (const Json* e = find(h, {"rms_norm_eps", "norm_eps"}); e && !e->is_null()) {
    auto r = as_f32(e, "rms_norm_eps");
    if (!r.ok()) return R::fail(r.status);
    kv.push_back(io::gguf_kv_f32(a + "attention.layer_norm_rms_epsilon", r.value));
  }
  if (const Json* e = find(h, {"layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"}); e && !e->is_null()) {
    auto r = as_f32(e, "layer_norm_eps");
    if (!r.ok()) return R::fail(r.status);
    kv.push_back(io::gguf_kv_f32(a + "attention.layer_norm_epsilon", r.value));
  }
  const Json* ne = find(h, {"num_local_experts", "num_experts", "n_routed_experts"});
  auto n_expert_r = as_u32(ne, "num_experts");
  if (!n_expert_r.ok()) return R::fail(n_expert_r.status);
  kv.push_back(io::gguf_kv_u32(a + "expert_count", n_expert_r.value));
  s = opt_u32({"num_experts_per_tok", "num_experts_per_token", "top_k_experts"}, a + "expert_used_count", "num_experts_per_tok");
  if (!s) return R::fail(s);
  s = opt_u32({"n_group"}, a + "expert_group_count", "n_group");
  if (!s) return R::fail(s);
  s = opt_u32({"topk_group"}, a + "expert_group_used_count", "topk_group");
  if (!s) return R::fail(s);
  if (const Json* sf = find(h, {"score_function", "scoring_func", "score_func", "moe_router_activation", "moe_router_activation_func", "expert_selection_fn"}); sf && !sf->is_null())
    return R::fail(bad("an expert gating function setting is not read yet", "config.json"));
  const Json* head_dim = h.get("head_dim");
  if (head_dim && !head_dim->is_null()) {
    auto r = as_u32(head_dim, "head_dim");
    if (!r.ok()) return R::fail(r.status);
    kv.push_back(io::gguf_kv_u32(a + "attention.key_length", r.value));
    kv.push_back(io::gguf_kv_u32(a + "attention.value_length", r.value));
  }
  kv.push_back(io::gguf_kv_u32("general.file_type", converter_file_type));
  // Qwen2MoeModel
  s = opt_u32({"moe_intermediate_size"}, a + "expert_feed_forward_length", "moe_intermediate_size");
  if (!s) return R::fail(s);
  s = opt_u32({"shared_expert_intermediate_size"}, a + "expert_shared_feed_forward_length", "shared_expert_intermediate_size");
  if (!s) return R::fail(s);
  // _QwenMtpMixin
  kv.push_back(io::gguf_kv_u32(a + "nextn_predict_layers", block_count - n_layer));
  // Qwen3NextModel
  auto req = [&](const char* key) { return as_u32(h.get(key), key); };
  const auto conv = req("linear_conv_kernel_dim"), hk = req("linear_key_head_dim"), nk = req("linear_num_key_heads"), nv = req("linear_num_value_heads"),
             hv = req("linear_value_head_dim");
  for (const auto* r : {&conv, &hk, &nk, &nv, &hv})
    if (!r->ok()) return R::fail(r->status);
  kv.push_back(io::gguf_kv_u32(a + "ssm.conv_kernel", conv.value));
  kv.push_back(io::gguf_kv_u32(a + "ssm.state_size", hk.value));
  kv.push_back(io::gguf_kv_u32(a + "ssm.group_count", nk.value));
  kv.push_back(io::gguf_kv_u32(a + "ssm.time_step_rank", nv.value));
  kv.push_back(io::gguf_kv_u32(a + "ssm.inner_size", hv.value * nv.value));
  std::vector<bool> recurrent;
  if (const Json* lt = h.get("layer_types"); lt && !lt->is_null()) {
    if (!lt->is_array() || lt->size() != n_layer) return R::fail(bad("layer_types does not list one type per layer", "config.json"));
    for (const Json& x : lt->items()) recurrent.push_back(x.is_string() && x.as_string() == "linear_attention");
    recurrent.resize(block_count, false);
    kv.push_back(io::gguf_kv_arr_bool(a + "attention.recurrent_layers", recurrent));
  }
  std::uint32_t fai = 4;
  if (const Json* f = h.get("full_attention_interval")) {
    auto r = as_u32(f, "full_attention_interval");
    if (!r.ok()) return R::fail(r.status);
    fai = r.value;
  }
  kv.push_back(io::gguf_kv_u32(a + "full_attention_interval", fai));
  double rope_dim = 0;
  if (head_dim && !head_dim->is_null()) {
    rope_dim = static_cast<double>(head_dim->as_uint());
  } else {
    auto hs = req("hidden_size"), nh = req("num_attention_heads");
    if (!hs.ok() || !nh.ok() || nh.value == 0) return R::fail(bad("no head size", "config.json"));
    rope_dim = static_cast<double>(hs.value / nh.value);  // integer division, as the reference
  }
  double prf = 0.25;
  if (const Json* x = rp.get("partial_rotary_factor"); x && !x->is_null()) {
    if (!x->is_number()) return R::fail(bad("partial_rotary_factor is not a number", "config.json"));
    prf = x->as_double();
  }
  kv.push_back(io::gguf_kv_u32(a + "rope.dimension_count", static_cast<std::uint32_t>(rope_dim * prf)));  // int() truncates
  // _Qwen35MRopeMixin
  if (!rp.get("mrope_section")) kv.push_back(io::gguf_kv_arr_i32(a + "rope.dimension_sections", {11, 11, 10, 0}));

  p.facts.arch = p.arch;
  p.facts.n_expert = n_expert_r.value;
  p.facts.n_layer = block_count;

  // ---- tensors, in reading order
  const std::uint64_t K = nk.value, V = nv.value, HK = hk.value, HV = hv.value;
  const bool reorder = K > 0 && V > 0 && K != V;
  if (reorder && V % K != 0) return R::fail(bad("linear_num_value_heads is not a multiple of linear_num_key_heads", "config.json"));
  const std::uint64_t VPK = reorder ? V / K : 1;
  const std::uint64_t qk_rows = 2 * K * HK;
  for (const SourceRef& src : sources) {
    const io::SafetensorsTensor& t = *src.t;
    if (t.dtype != "BF16" && t.dtype != "F32") return R::fail(bad("source type " + t.dtype + " is not converted yet (BF16 and F32 are)", src.name));
    if (t.dtype == "F32") return R::fail(bad("F32 source tensors are not converted yet", src.name));
    const std::string& n = src.name;
    std::string rest;
    long long bid = -1;
    if (n.rfind("model.layers.", 0) == 0) {
      const std::size_t dot = n.find('.', 13);
      if (dot == std::string::npos) return R::fail(bad("a layer tensor name without a field", n));
      bid = std::stoll(n.substr(13, dot - 13));
      rest = n.substr(dot + 1);
      if (bid < 0 || bid >= static_cast<long long>(block_count)) return R::fail(bad("a layer outside the model's blocks", n));
    }
    const std::string blk = "blk." + std::to_string(bid) + ".";
    TensorPlan tp;
    tp.shard = src.shard;
    tp.source = src.index;
    tp.shape = t.shape;
    auto add = [&](TensorPlan x, const std::string& gname) {
      x.name = gname;
      p.tensors.push_back(std::move(x));
    };
    auto expect_shape = [&](std::initializer_list<std::uint64_t> want) -> Status {
      if (std::vector<std::uint64_t>(want) != t.shape) return bad("unexpected shape", n);
      return Status::good();
    };
    auto heads_rows = [&](TensorPlan& x, std::uint64_t start, std::uint64_t hd) {
      if (!reorder) return;
      x.rows = TensorPlan::Rows::heads;
      x.head_start = start;
      x.nk = K;
      x.nvpk = VPK;
      x.hd = hd;
    };
    if (bid < 0) {
      if (n == "model.embed_tokens.weight") add(tp, "token_embd.weight");
      else if (n == "lm_head.weight") add(tp, "output.weight");
      else if (n == "model.norm.weight") {
        tp.op = TensorPlan::Op::plus_one;
        add(tp, "output_norm.weight");
      } else {
        return R::fail(bad("a tensor this architecture does not have", n));
      }
      continue;
    }
    const bool mtp = static_cast<std::uint32_t>(bid) >= n_layer;
    if (rest == "input_layernorm.weight") {
      tp.op = TensorPlan::Op::plus_one;
      add(tp, blk + "attn_norm.weight");
    } else if (rest == "post_attention_layernorm.weight") {
      tp.op = TensorPlan::Op::plus_one;
      add(tp, blk + "post_attention_norm.weight");
    } else if (rest == "self_attn.q_proj.weight") add(tp, blk + "attn_q.weight");
    else if (rest == "self_attn.k_proj.weight") add(tp, blk + "attn_k.weight");
    else if (rest == "self_attn.v_proj.weight") add(tp, blk + "attn_v.weight");
    else if (rest == "self_attn.o_proj.weight") add(tp, blk + "attn_output.weight");
    else if (rest == "self_attn.q_norm.weight") {
      tp.op = TensorPlan::Op::plus_one;
      add(tp, blk + "attn_q_norm.weight");
    } else if (rest == "self_attn.k_norm.weight") {
      tp.op = TensorPlan::Op::plus_one;
      add(tp, blk + "attn_k_norm.weight");
    } else if (rest == "linear_attn.in_proj_qkv.weight") {
      if (t.shape.size() != 2 || t.shape[0] != qk_rows + V * HV) return R::fail(bad("unexpected shape", n));
      heads_rows(tp, qk_rows, HV);
      add(tp, blk + "attn_qkv.weight");
    } else if (rest == "linear_attn.in_proj_z.weight") {
      if (t.shape.size() != 2 || t.shape[0] != V * HV) return R::fail(bad("unexpected shape", n));
      heads_rows(tp, 0, HV);
      add(tp, blk + "attn_gate.weight");
    } else if (rest == "linear_attn.in_proj_a.weight" || rest == "linear_attn.in_proj_b.weight") {
      if (t.shape.size() != 2 || t.shape[0] != V) return R::fail(bad("unexpected shape", n));
      heads_rows(tp, 0, 1);
      add(tp, blk + (rest == "linear_attn.in_proj_a.weight" ? "ssm_alpha.weight" : "ssm_beta.weight"));
    } else if (rest == "linear_attn.A_log" || rest == "linear_attn.dt_bias") {
      const Status st = expect_shape({V});
      if (!st) return R::fail(st);
      if (reorder) {
        heads_rows(tp, 0, 1);
        tp.head_columns = true;
      }
      if (rest == "linear_attn.A_log") {
        tp.op = TensorPlan::Op::neg_exp;
        add(tp, blk + "ssm_a");
      } else {
        add(tp, blk + "ssm_dt.bias");
      }
    } else if (rest == "linear_attn.conv1d.weight") {
      if (t.shape.size() != 3 || t.shape[0] != qk_rows + V * HV || t.shape[1] != 1) return R::fail(bad("unexpected shape", n));
      tp.shape = {t.shape[0], t.shape[2]};  // squeezed
      heads_rows(tp, qk_rows, HV);
      add(tp, blk + "ssm_conv1d.weight");
    } else if (rest == "linear_attn.norm.weight") add(tp, blk + "ssm_norm.weight");
    else if (rest == "linear_attn.out_proj.weight") {
      if (t.shape.size() != 2 || t.shape[1] != V * HV) return R::fail(bad("unexpected shape", n));
      if (reorder) {
        heads_rows(tp, 0, HV);
        tp.head_columns = true;
      }
      add(tp, blk + "ssm_out.weight");
    } else if (rest == "mlp.gate.weight") add(tp, blk + "ffn_gate_inp.weight");
    else if (rest == "mlp.shared_expert_gate.weight") add(tp, blk + "ffn_gate_inp_shexp.weight");
    else if (rest == "mlp.shared_expert.gate_proj.weight") add(tp, blk + "ffn_gate_shexp.weight");
    else if (rest == "mlp.shared_expert.up_proj.weight") add(tp, blk + "ffn_up_shexp.weight");
    else if (rest == "mlp.shared_expert.down_proj.weight") add(tp, blk + "ffn_down_shexp.weight");
    else if (rest == "mlp.experts.down_proj" || rest == "mlp.experts.down_proj.weight") add(tp, blk + "ffn_down_exps.weight");
    else if (rest == "mlp.experts.gate_up_proj" || rest == "mlp.experts.gate_up_proj.weight") {
      if (t.shape.size() < 3 || t.shape[t.shape.size() - 2] % 2 != 0) return R::fail(bad("unexpected gate_up_proj shape", n));
      const std::uint64_t ff = t.shape[t.shape.size() - 2] / 2;
      TensorPlan g = tp, u = tp;
      g.shape[g.shape.size() - 2] = ff;
      u.shape[u.shape.size() - 2] = ff;
      g.rows = u.rows = TensorPlan::Rows::expert_part;
      g.part_rows = u.part_rows = ff;
      g.part_offset = 0;
      u.part_offset = ff;
      add(g, blk + "ffn_gate_exps.weight");
      add(u, blk + "ffn_up_exps.weight");
    } else if (mtp && rest == "eh_proj.weight") add(tp, blk + "nextn.eh_proj.weight");
    else if (mtp && rest == "enorm.weight") {
      tp.op = TensorPlan::Op::plus_one;
      add(tp, blk + "nextn.enorm.weight");
    } else if (mtp && rest == "hnorm.weight") {
      tp.op = TensorPlan::Op::plus_one;
      add(tp, blk + "nextn.hnorm.weight");
    } else if (mtp && rest == "shared_head.norm.weight") {
      tp.op = TensorPlan::Op::plus_one;
      add(tp, blk + "nextn.shared_head_norm.weight");
    } else if (rest.rfind("mlp.experts.", 0) == 0) {
      return R::fail(bad("per-expert tensors are not converted yet (this model stores its experts stacked)", n));
    } else {
      return R::fail(bad("a tensor this architecture does not have", n));
    }
  }

  // ---- output types (prepare_tensors with --outtype bf16)
  for (TensorPlan& x : p.tensors) {
    const bool one_dim = x.shape.size() <= 1;
    const bool norm = x.name.size() >= 12 && x.name.compare(x.name.size() - 12, 12, "_norm.weight") == 0;
    const bool forced_f32 = x.name.find(".ffn_gate_inp.weight") != std::string::npos || x.name.find(".ffn_gate_inp_shexp.weight") != std::string::npos ||
                            x.name.find(".ssm_conv1d.weight") != std::string::npos;
    const bool weight_suffix = x.name.size() >= 7 && x.name.compare(x.name.size() - 7, 7, ".weight") == 0;
    x.type = one_dim || norm || forced_f32 || !weight_suffix ? kF32 : kBF16;
  }
  return R::good(std::move(p));
}

}  // namespace pelias::ingest
