// provenance: re-expressed (not copied) from llama.cpp src/llama-quant.cpp at commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a (MIT; THIRD_PARTY_NOTICES.md), read as a specification
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// The branches below are the reference's, kept for the file types this build offers (F32, F16, BF16, Q8_0, Q6_K,
// Q4_K_M). Branches that only other mixes reach are left out; nothing in them can change a type for these six.
#include "quant/type_rules.h"

#include <algorithm>
#include <cctype>
#include <cstdio>

#include "quant/block_types.h"

namespace pelias::quant {
namespace {

enum class Category { token_embd, attn_v, attn_qkv, attn_kv_b, attn_k, attn_q, attn_output, ffn_up, ffn_gate, ffn_down, output, other };

bool contains(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

Category category_of(const std::string& n) {
  if (n == "output.weight") return Category::output;
  if (n == "token_embd.weight" || n == "per_layer_token_embd.weight") return Category::token_embd;
  if (contains(n, "attn_qkv.weight")) return Category::attn_qkv;
  if (contains(n, "attn_kv_b.weight")) return Category::attn_kv_b;
  if (contains(n, "attn_v.weight")) return Category::attn_v;
  if (contains(n, "attn_k.weight")) return Category::attn_k;
  if (contains(n, "attn_q.weight")) return Category::attn_q;
  if (contains(n, "attn_output.weight")) return Category::attn_output;
  if (contains(n, "ffn_up")) return Category::ffn_up;
  if (contains(n, "ffn_gate")) return Category::ffn_gate;
  if (contains(n, "ffn_down")) return Category::ffn_down;
  return Category::other;
}

bool attn_v_like(Category c) { return c == Category::attn_v || c == Category::attn_qkv || c == Category::attn_kv_b; }

bool use_more_bits(long long i_layer, long long n_layers) {
  return i_layer < n_layers / 8 || i_layer >= 7 * n_layers / 8 || (i_layer - n_layers / 8) % 3 == 2;
}

std::uint32_t ftype_default(FileType f) {
  switch (f) {
    case FileType::all_f32: return static_cast<std::uint32_t>(Type::f32);
    case FileType::f16: return static_cast<std::uint32_t>(Type::f16);
    case FileType::bf16: return static_cast<std::uint32_t>(Type::bf16);
    case FileType::q8_0: return static_cast<std::uint32_t>(Type::q8_0);
    case FileType::q6_k: return static_cast<std::uint32_t>(Type::q6_k);
    case FileType::q4_k_m: return static_cast<std::uint32_t>(Type::q4_k);
  }
  return static_cast<std::uint32_t>(Type::f32);
}

bool is_quantized(std::uint32_t t) {
  const TypeInfo ti = type_info(t);
  return ti.known && ti.block_elements > 1;
}

constexpr auto T_Q8_0 = static_cast<std::uint32_t>(Type::q8_0);
constexpr auto T_Q6_K = static_cast<std::uint32_t>(Type::q6_k);
constexpr auto T_Q5_K = static_cast<std::uint32_t>(Type::q5_k);
constexpr auto T_Q4_K = static_cast<std::uint32_t>(Type::q4_k);
constexpr auto T_Q3_K = static_cast<std::uint32_t>(Type::q3_k);
constexpr auto T_Q5_0 = static_cast<std::uint32_t>(Type::q5_0);
constexpr auto T_Q5_1 = static_cast<std::uint32_t>(Type::q5_1);
constexpr auto T_Q4_0 = static_cast<std::uint32_t>(Type::q4_0);
constexpr auto T_F16 = static_cast<std::uint32_t>(Type::f16);
constexpr auto T_F32 = static_cast<std::uint32_t>(Type::f32);
constexpr auto T_BF16 = static_cast<std::uint32_t>(Type::bf16);

// the reference's shape fallback (an incompatible row length moves to a 32-block type, and to F16 if even that fails)
std::uint32_t shape_fallback(const TensorDesc& t, std::uint32_t target) {
  const std::uint64_t ncols = t.ne[0];
  const std::uint64_t qk = type_info(target).block_elements;
  if (qk == 0 || ncols % qk == 0) return target;
  std::uint32_t r = target;
  switch (static_cast<Type>(target)) {
    case Type::q2_k: case Type::q3_k: case Type::tq1_0: case Type::tq2_0: case Type::q2_0: r = T_Q4_0; break;
    case Type::q4_k: r = T_Q5_0; break;
    case Type::q5_k: r = T_Q5_1; break;
    case Type::q6_k: r = T_Q8_0; break;
    default: r = target; break;  // a 32-block type stays, then the check below may move it to F16
  }
  if (ncols % type_info(r).block_elements != 0) r = T_F16;
  return r;
}

struct Counters {
  long long n_attention_wv = 0, i_attention_wv = 0;
  long long n_ffn = 0, i_ffn_down = 0, i_ffn_gate = 0, i_ffn_up = 0;
  bool tied_embeddings = true;
};

}  // namespace

Result<FileType> file_type_from_name(const std::string& name) {
  std::string u;
  for (char c : name) u += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  if (u == "Q4_K_M") return Result<FileType>::good(FileType::q4_k_m);
  if (u == "Q8_0") return Result<FileType>::good(FileType::q8_0);
  if (u == "Q6_K") return Result<FileType>::good(FileType::q6_k);
  if (u == "F16") return Result<FileType>::good(FileType::f16);
  if (u == "BF16") return Result<FileType>::good(FileType::bf16);
  if (u == "F32") return Result<FileType>::good(FileType::all_f32);
  return Result<FileType>::fail(Status::fail(Code::unsupported, "type '" + name + "' is not offered by this build (offered: Q4_K_M, Q6_K, Q8_0, BF16, F16, F32)"));
}

const char* file_type_name(FileType f) {
  switch (f) {
    case FileType::all_f32: return "F32";
    case FileType::f16: return "F16";
    case FileType::bf16: return "BF16";
    case FileType::q8_0: return "Q8_0";
    case FileType::q6_k: return "Q6_K";
    case FileType::q4_k_m: return "Q4_K_M";
  }
  return "?";
}

bool tensor_allows_quantization(const TensorDesc& t, const std::string& arch) {
  // the reference's ggml_n_dims: dimensions up to the last one larger than 1
  std::uint32_t nd = 1;
  for (std::uint32_t d = 0; d < 4; ++d)
    if (t.ne[d] > 1) nd = d + 1;
  if (nd < 2) return false;
  const std::string& name = t.name;
  bool q = name.size() >= 6 && name.rfind("weight") == name.size() - 6;
  q = q && !contains(name, "_norm.weight");
  q = q && !contains(name, "ffn_gate_inp.weight");
  q = q && !contains(name, "ffn_gate_tid2eid.weight");
  q = q && !contains(name, "altup") && !contains(name, "laurel");
  q = q && !contains(name, "per_layer_model_proj");
  q = q && name != "position_embd.weight" && name != "token_types.weight";
  q = q && !contains(name, "ssm_conv1d") && !contains(name, "shortconv.conv.weight");
  q = q && !contains(name, "indexer.k_proj.weight") && !contains(name, "indexer.q_proj.weight");
  if (arch == "glm5-next") {
    for (const char* p : {"hc_", "indexer.attn_q_b", "indexer.attn_k", "indexer.proj", "indexer_compressor_gate", "indexer_compressor_ape",
                          "ssm_f_a.weight", "ssm_f_b.weight", "ssm_g_a.weight", "ssm_g_b.weight", "ssm_beta.weight", "attn_kv_a_mqa.weight",
                          "attn_k_b.weight", "attn_v_b.weight"})
      q = q && !contains(name, p);
  }
  for (const char* p : {"time_mix_first.weight", "time_mix_w0.weight", "time_mix_w1.weight", "time_mix_w2.weight", "time_mix_v0.weight",
                        "time_mix_v1.weight", "time_mix_v2.weight", "time_mix_a0.weight", "time_mix_a1.weight", "time_mix_a2.weight",
                        "time_mix_g1.weight", "time_mix_g2.weight", "time_mix_decay_w1.weight", "time_mix_decay_w2.weight",
                        "time_mix_lerp_fused.weight", "attn_rel_b.weight", ".position_embd", "sam.pos_embd", "sam.neck.", "sam.net_",
                        ".rel_pos", ".patch_embd", ".patch_merger", "a.rvq.codebook", "mm.a.code_embd"})
    q = q && !contains(name, p);
  return q;
}

Result<std::vector<std::uint32_t>> choose_types(const std::vector<TensorDesc>& tensors, const ModelFacts& model, FileType ftype) {
  const bool is_q4km = ftype == FileType::q4_k_m;
  const bool falcon = model.arch == "falcon";
  const long long n_expert = std::max<long long>(1, static_cast<long long>(model.n_expert));
  // only these architectures and layer counts are the reference's "70B" type, which moves attn_v up a step
  const bool type_70b = ((model.arch == "qwen2" || model.arch == "deci" || model.arch == "olmo") && model.n_layer == 80) ||
                        (model.arch == "llama" && model.n_layer == 80) || (model.arch == "jais2" && model.n_layer == 68);

  Counters c;
  for (const TensorDesc& t : tensors) {
    const Category cat = category_of(t.name);
    if (attn_v_like(cat)) ++c.n_attention_wv;
    if (cat == Category::output) c.tied_embeddings = false;
  }
  c.n_ffn = static_cast<long long>(model.n_layer);

  std::vector<std::uint32_t> out;
  out.reserve(tensors.size());
  const std::uint32_t default_type = ftype_default(ftype);
  for (const TensorDesc& t : tensors) {
    if (!tensor_allows_quantization(t, model.arch)) {
      out.push_back(t.type);
      continue;
    }
    std::uint32_t nt = default_type;
    if (is_quantized(default_type)) {
      const Category cat = category_of(t.name);
      auto layer_info = [&](long long i_counter, long long n_counter, long long& i_layer, long long& n_layer) -> Status {
        i_layer = i_counter;
        n_layer = n_counter;
        if (n_expert > 1) {
          int l = -1;
          if (std::sscanf(t.name.c_str(), "blk.%d.", &l) != 1) return Status::fail(Code::corrupt, "cannot find the layer of an expert tensor", 0, t.name);
          if (l < 0 || l >= n_counter) return Status::fail(Code::corrupt, "layer " + std::to_string(l) + " is outside the block count", 0, t.name);
          i_layer = l;
        }
        return Status::good();
      };
      if (model.arch == "glm5-next" && (contains(t.name, "attn_q_a") || contains(t.name, "attn_q_b") || contains(t.name, "nextn.eh_proj"))) {
        if (nt != T_F32 && nt != T_BF16 && nt != T_F16) {
          out.push_back(T_Q8_0);
          continue;
        }
      }
      if (cat == Category::output || (c.tied_embeddings && cat == Category::token_embd)) {
        const std::uint64_t qk = type_info(nt).block_elements;
        if (falcon || (qk != 0 && t.ne[0] % qk != 0)) nt = T_Q8_0;
        else if (nt != T_Q8_0) nt = T_Q6_K;
      } else if (cat == Category::token_embd) {
        // no change for the offered mixes
      } else if (attn_v_like(cat)) {
        if (is_q4km && use_more_bits(c.i_attention_wv, c.n_attention_wv)) nt = T_Q6_K;
        if (type_70b && (nt == T_Q3_K || nt == T_Q4_K)) nt = T_Q5_K;
        if (n_expert == 8) nt = T_Q8_0;
        ++c.i_attention_wv;
      } else if (cat == Category::attn_k) {
        if (n_expert == 8) nt = T_Q8_0;
      } else if (cat == Category::attn_q) {
        // no change for the offered mixes
      } else if (cat == Category::ffn_down) {
        long long i_layer = 0, n_layer = 0;
        const Status s = layer_info(c.i_ffn_down, c.n_ffn, i_layer, n_layer);
        if (!s) return Result<std::vector<std::uint32_t>>::fail(s);
        if (is_q4km) {
          if (falcon) nt = i_layer < n_layer / 16 ? T_Q6_K : use_more_bits(i_layer, n_layer) ? T_Q5_K : T_Q4_K;
          else if (use_more_bits(i_layer, n_layer)) nt = T_Q6_K;
        }
        ++c.i_ffn_down;
      } else if (cat == Category::attn_output) {
        if (!falcon && n_expert == 8 && is_q4km) nt = T_Q5_K;
      } else if (cat == Category::attn_qkv) {
        if (is_q4km) nt = T_Q5_K;  // the reference's branch; attn_qkv is taken by the attn_v rule above first
      } else if (cat == Category::ffn_gate) {
        long long i_layer = 0, n_layer = 0;
        const Status s = layer_info(c.i_ffn_gate, c.n_ffn, i_layer, n_layer);
        if (!s) return Result<std::vector<std::uint32_t>>::fail(s);
        ++c.i_ffn_gate;
      } else if (cat == Category::ffn_up) {
        long long i_layer = 0, n_layer = 0;
        const Status s = layer_info(c.i_ffn_up, c.n_ffn, i_layer, n_layer);
        if (!s) return Result<std::vector<std::uint32_t>>::fail(s);
        ++c.i_ffn_up;
      }
      nt = shape_fallback(t, nt);
    }
    out.push_back(nt);
  }
  return Result<std::vector<std::uint32_t>>::good(std::move(out));
}

}  // namespace pelias::quant
