// provenance: original (keys and defaults READ-SRC from llama.cpp src/models/qwen35moe.cpp load_arch_hparams and
//   src/llama-model.cpp at commit 2ca15f5404760548c39e7b92bd43116a09414a1a, read as a specification)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "model/spec.h"

#include <cmath>

namespace pelias::model {

namespace {

Status missing(const std::string& key) { return Status::fail(Code::corrupt, "a metadata entry the model needs is missing", 0, key); }
Status bad(const std::string& key, const std::string& why) { return Status::fail(Code::corrupt, why, 0, key); }

}  // namespace

namespace {
// GP3-09: a dense decoder (qwen3 / qwen2 / llama). Keys READ-SRC from llama.cpp src/llama-model.cpp (load_hparams) and
// src/models/{qwen3,qwen2,llama}.cpp, read as a specification: head size defaults to n_embd / n_head, the rotated size to
// the head size, head_count_kv to head_count, the rope base to 10000.
Result<Qwen35MoeSpec> read_dense_spec(const io::GgufReader& r, const std::string& arch, Family fam) {
  using R = Result<Qwen35MoeSpec>;
  Qwen35MoeSpec s;
  s.family = fam;
  s.arch = arch;
  s.dense = true;
  s.attn_gate = false;
  s.qk_norm = fam == Family::qwen3;
  s.qkv_bias = fam == Family::qwen2;
  // llama.cpp llama_model_rope_type: LLAMA -> NORM, QWEN2 / QWEN3 / BITNET -> NEOX. BitNet b1.58 2B4T: the I2_S file's q/k
  // rows are in the original's order (MEASURED, tools/oracle/i2s_check.py), so the original's rotate_half pairs: NEOX.
  s.rope_neox = fam != Family::llama;
  if (fam == Family::bitnet) {
    s.sub_norms = true;
    s.bitlinear_act8 = true;
    s.ffn_relu2 = arch == "bitnet-b1.58";
  }
  s.rope_freq_base = 10000.0f;
  const std::string a = arch + ".";
  bool found = false;
  auto need_u32 = [&](const char* key, std::uint32_t& out) -> Status {
    const std::uint64_t v = r.get_uint(a + key, &found);
    if (!found) return missing(a + key);
    if (v == 0 || v > 0xFFFFFFFFull) return bad(a + key, "the value is out of range");
    out = static_cast<std::uint32_t>(v);
    return Status::good();
  };
  auto opt_u32 = [&](const char* key, std::uint32_t& out) {
    const std::uint64_t v = r.get_uint(a + key, &found);
    if (found && v > 0 && v <= 0xFFFFFFFFull) out = static_cast<std::uint32_t>(v);
  };
  for (Status st : {need_u32("block_count", s.n_layer_all), need_u32("embedding_length", s.n_embd), need_u32("feed_forward_length", s.n_ff),
                    need_u32("attention.head_count", s.n_head)}) {
    if (!st) return R::fail(st);
  }
  {
    const double eps = r.get_float(a + "attention.layer_norm_rms_epsilon", &found);
    if (!found) return R::fail(missing(a + "attention.layer_norm_rms_epsilon"));
    if (!std::isfinite(eps) || eps <= 0.0) return R::fail(bad(a + "attention.layer_norm_rms_epsilon", "the value is not a positive number"));
    s.rms_eps = static_cast<float>(eps);
  }
  s.n_head_kv = s.n_head;
  opt_u32("attention.head_count_kv", s.n_head_kv);
  if (s.n_embd % s.n_head != 0 && !r.find(a + "attention.key_length"))
    return R::fail(bad(a + "attention.head_count", "embedding_length is not a multiple of head_count and no key_length is given"));
  s.head_dim = s.n_embd / s.n_head;
  opt_u32("attention.key_length", s.head_dim);
  std::uint32_t value_length = s.head_dim;
  opt_u32("attention.value_length", value_length);
  if (value_length != s.head_dim) return R::fail(bad(a + "attention.value_length", "key and value head sizes differ (not read yet)"));
  s.n_rot = s.head_dim;
  opt_u32("rope.dimension_count", s.n_rot);
  opt_u32("context_length", s.n_ctx_train);
  {
    const double fb = r.get_float(a + "rope.freq_base", &found);
    if (found) s.rope_freq_base = static_cast<float>(fb);
    const double sc = r.get_float(a + "attention.scale", &found);
    if (found) s.attn_scale = static_cast<float>(sc);
    const std::string st = r.get_string(a + "rope.scaling.type", &found);
    if (found && st != "none" && st != "linear") return R::fail(Status::fail(Code::unsupported, "rope scaling '" + st + "' is not read yet", 0, a + "rope.scaling.type"));
    const double sf = r.get_float(a + "rope.scaling.factor", &found);
    if (found && sf != 0.0 && sf != 1.0) return R::fail(Status::fail(Code::unsupported, "rope scaling is not read yet", 0, a + "rope.scaling.factor"));
  }
  s.n_layer = s.n_layer_all;
  s.recurrent.assign(s.n_layer_all, false);
  const io::GgufTensor* emb = r.find_tensor("token_embd.weight");
  if (!emb) return R::fail(Status::fail(Code::not_found, "the token embedding is missing", 0, "token_embd.weight"));
  if (emb->ne[0] != s.n_embd || emb->ne[1] == 0 || emb->ne[1] > 0x7FFFFFFFull)
    return R::fail(bad("token_embd.weight", "the shape does not match embedding_length"));
  s.n_vocab = static_cast<std::uint32_t>(emb->ne[1]);
  if (s.n_rot > s.head_dim || s.n_rot % 2 != 0) return R::fail(bad(a + "rope.dimension_count", "the rotated size does not fit the head"));
  if (s.n_head % s.n_head_kv != 0) return R::fail(bad(a + "attention.head_count_kv", "query heads are not a multiple of key/value heads"));
  return R::good(std::move(s));
}
}  // namespace

Result<Qwen35MoeSpec> read_qwen35moe_spec(const io::GgufReader& r) {
  using R = Result<Qwen35MoeSpec>;
  Qwen35MoeSpec s;
  const std::string arch = r.get_string("general.architecture");
  if (arch == "qwen3") return read_dense_spec(r, arch, Family::qwen3);
  if (arch == "qwen2") return read_dense_spec(r, arch, Family::qwen2);
  if (arch == "llama") return read_dense_spec(r, arch, Family::llama);
  if (arch == "bitnet-b1.58" || arch == "bitnet") return read_dense_spec(r, arch, Family::bitnet);
  if (arch != "qwen35moe")
    return R::fail(Status::fail(Code::unsupported, "the reference pass reads qwen35moe, qwen3, qwen2, llama and bitnet (this file is '" + arch + "')", 0, "general.architecture"));
  s.arch = arch;
  const std::string a = "qwen35moe.";

  bool found = false;
  auto need_u32 = [&](const char* key, std::uint32_t& out) -> Status {
    const std::uint64_t v = r.get_uint(a + key, &found);
    if (!found) return missing(a + key);
    if (v == 0 || v > 0xFFFFFFFFull) return bad(a + key, "the value is out of range");
    out = static_cast<std::uint32_t>(v);
    return Status::good();
  };
  auto opt_u32 = [&](const char* key, std::uint32_t& out) {
    const std::uint64_t v = r.get_uint(a + key, &found);
    if (found && v <= 0xFFFFFFFFull) out = static_cast<std::uint32_t>(v);
  };
  auto need_f32 = [&](const char* key, float& out) -> Status {
    const double v = r.get_float(a + key, &found);
    if (!found) return missing(a + key);
    if (!std::isfinite(v)) return bad(a + key, "the value is not finite");
    out = static_cast<float>(v);
    return Status::good();
  };

  for (Status st : {need_u32("block_count", s.n_layer_all), need_u32("embedding_length", s.n_embd),
                    need_u32("attention.head_count", s.n_head), need_u32("attention.head_count_kv", s.n_head_kv),
                    need_u32("attention.key_length", s.head_dim), need_u32("rope.dimension_count", s.n_rot),
                    need_f32("attention.layer_norm_rms_epsilon", s.rms_eps), need_u32("expert_count", s.n_expert),
                    need_u32("expert_used_count", s.n_expert_used), need_u32("expert_feed_forward_length", s.n_ff_exp),
                    need_u32("expert_shared_feed_forward_length", s.n_ff_shexp), need_u32("ssm.conv_kernel", s.ssm_d_conv),
                    need_u32("ssm.inner_size", s.ssm_d_inner), need_u32("ssm.state_size", s.ssm_d_state),
                    need_u32("ssm.group_count", s.ssm_n_group), need_u32("ssm.time_step_rank", s.ssm_dt_rank)}) {
    if (!st) return R::fail(st);
  }
  std::uint32_t value_length = s.head_dim;
  opt_u32("attention.value_length", value_length);
  if (value_length != s.head_dim) return R::fail(bad(a + "attention.value_length", "key and value head sizes differ (not read yet)"));
  {
    const double fb = r.get_float(a + "rope.freq_base", &found);
    if (found) s.rope_freq_base = static_cast<float>(fb);
    const double sc = r.get_float(a + "attention.scale", &found);
    if (found) s.attn_scale = static_cast<float>(sc);
    const double ws = r.get_float(a + "expert_weights_scale", &found);
    if (found) s.expert_weights_scale = static_cast<float>(ws);
  }
  {
    const std::string st = r.get_string(a + "rope.scaling.type", &found);
    if (found && st != "none") return R::fail(Status::fail(Code::unsupported, "rope scaling is not read yet", 0, a + "rope.scaling.type"));
    const double sf = r.get_float(a + "rope.scaling.factor", &found);
    if (found && sf != 0.0 && sf != 1.0) return R::fail(Status::fail(Code::unsupported, "rope scaling is not read yet", 0, a + "rope.scaling.factor"));
  }
  std::uint32_t nextn = 0;
  opt_u32("nextn_predict_layers", nextn);
  if (nextn >= s.n_layer_all) return R::fail(bad(a + "nextn_predict_layers", "the MTP blocks leave no trunk layers"));
  s.n_layer = s.n_layer_all - nextn;

  // rope sections (IMRoPE). For text every position component is the same, so only their sum matters for the check.
  if (const io::GgufValue* v = r.find(a + "rope.dimension_sections")) {
    std::vector<std::int64_t> sec;
    Status st = r.array_ints(*v, sec);
    if (!st) return R::fail(Status::fail(Code::corrupt, st.message, 0, a + "rope.dimension_sections"));
    for (std::size_t k = 0; k < 4 && k < sec.size(); ++k) s.rope_sections[k] = static_cast<std::int32_t>(sec[k]);
  } else {
    return R::fail(missing(a + "rope.dimension_sections"));
  }

  // which blocks are DeltaNet layers: the explicit list, else every layer but each full_attention_interval-th
  s.recurrent.assign(s.n_layer_all, false);
  if (const io::GgufValue* v = r.find(a + "attention.recurrent_layers")) {
    std::vector<std::int64_t> rl;
    Status st = r.array_ints(*v, rl);
    if (!st) return R::fail(Status::fail(Code::corrupt, st.message, 0, a + "attention.recurrent_layers"));
    if (rl.size() != s.n_layer_all) return R::fail(bad(a + "attention.recurrent_layers", "the list does not have one entry per block"));
    for (std::uint32_t i = 0; i < s.n_layer_all; ++i) s.recurrent[i] = rl[i] != 0;
  } else {
    std::uint32_t interval = 4;
    opt_u32("full_attention_interval", interval);
    if (interval == 0) return R::fail(bad(a + "full_attention_interval", "the value is 0"));
    for (std::uint32_t i = 0; i < s.n_layer_all; ++i) s.recurrent[i] = i < s.n_layer && (i + 1) % interval != 0;
  }

  // shapes that must agree
  const io::GgufTensor* emb = r.find_tensor("token_embd.weight");
  if (!emb) return R::fail(Status::fail(Code::not_found, "the token embedding is missing", 0, "token_embd.weight"));
  if (emb->ne[0] != s.n_embd || emb->ne[1] == 0 || emb->ne[1] > 0x7FFFFFFFull)
    return R::fail(bad("token_embd.weight", "the shape does not match embedding_length"));
  s.n_vocab = static_cast<std::uint32_t>(emb->ne[1]);
  if (s.n_rot > s.head_dim || s.n_rot % 2 != 0) return R::fail(bad(a + "rope.dimension_count", "the rotated size does not fit the head"));
  if (s.n_head % s.n_head_kv != 0) return R::fail(bad(a + "attention.head_count_kv", "query heads are not a multiple of key/value heads"));
  if (s.ssm_dt_rank % s.ssm_n_group != 0) return R::fail(bad(a + "ssm.time_step_rank", "value heads are not a multiple of key heads"));
  if (s.ssm_d_inner % s.ssm_dt_rank != 0 || s.ssm_d_inner / s.ssm_dt_rank != s.ssm_d_state)
    return R::fail(bad(a + "ssm.inner_size", "the value head size differs from the key head size (not read yet)"));
  if (s.n_expert_used > s.n_expert) return R::fail(bad(a + "expert_used_count", "more experts used than exist"));
  return R::good(std::move(s));
}

}  // namespace pelias::model
