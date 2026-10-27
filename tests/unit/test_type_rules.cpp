// provenance: original (expected types READ-SRC from the served Q4_K_M file's header, read 2026-10-26: the stock
//   reference quantizer's choices on qwen35moe, 41 blocks with the MTP block, 256 experts)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "io/gguf_writer.h"
#include "pelias_test.h"
#include "quant/block_types.h"
#include "quant/type_rules.h"
#include "unit/synth_model.h"

using namespace pelias;
namespace q = pelias::quant;

namespace {

constexpr auto BF16 = static_cast<std::uint32_t>(q::Type::bf16);
constexpr auto F32 = static_cast<std::uint32_t>(q::Type::f32);
constexpr auto Q4K = static_cast<std::uint32_t>(q::Type::q4_k);
constexpr auto Q6K = static_cast<std::uint32_t>(q::Type::q6_k);

q::TensorDesc t(const std::string& name, std::vector<std::uint64_t> ne, std::uint32_t type) {
  q::TensorDesc d;
  d.name = name;
  d.n_dims = static_cast<std::uint32_t>(ne.size());
  for (std::size_t i = 0; i < ne.size(); ++i) d.ne[i] = ne[i];
  d.type = type;
  return d;
}

}  // namespace

TEST(type_rule_q4_k_m_matches_the_served_file) {
  std::vector<q::TensorDesc> in = ptest::qwen35moe_served_tensors();
  CHECK_EQ(in.size(), 753u);
  // write order, as the writer sorts
  std::vector<std::string> names;
  for (const auto& d : in) names.push_back(d.name);
  std::vector<q::TensorDesc> ordered;
  for (const std::size_t i : io::reference_order(names)) ordered.push_back(in[i]);
  CHECK_EQ(ordered.front().name, std::string("output.weight"));  // outside any block, by name: output, output_norm, token_embd
  CHECK_EQ(ordered[2].name, std::string("token_embd.weight"));
  CHECK_EQ(ordered[3].name, std::string("blk.0.attn_gate.weight"));
  CHECK_EQ(ordered.back().name, std::string("blk.40.post_attention_norm.weight"));

  q::ModelFacts m;
  m.arch = "qwen35moe";
  m.n_expert = 256;
  m.n_layer = 41;
  const auto r = q::choose_types(ordered, m, q::FileType::q4_k_m);
  CHECK(r.ok());
  // the served file (READ-SRC): Q6_K for these layers in attn_qkv / attn_v (one shared counter), ffn_down_exps and
  // ffn_down_shexp; output Q6_K; every other quantizable tensor Q4_K; the rest keep F32
  const std::set<int> more = {0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39, 40};
  int q6 = 0, q4 = 0, f32 = 0, wrong = 0;
  for (std::size_t i = 0; i < ordered.size(); ++i) {
    const std::string& n = ordered[i].name;
    const std::uint32_t got = r.value[i];
    std::uint32_t want = Q4K;
    if (ordered[i].type == F32) want = F32;
    else if (n == "output.weight") want = Q6K;
    else {
      int b = -1;
      if (n.rfind("blk.", 0) == 0) b = std::stoi(n.substr(4));
      const bool v_like = n.find("attn_qkv.weight") != std::string::npos || n.find("attn_v.weight") != std::string::npos;
      const bool down = n.find("ffn_down") != std::string::npos;
      if ((v_like || down) && more.count(b)) want = Q6K;
    }
    if (got != want) {
      ++wrong;
      if (wrong <= 5) std::printf("      %s: got %s want %s\n", n.c_str(), q::type_info(got).name, q::type_info(want).name);
    }
    q6 += got == Q6K;
    q4 += got == Q4K;
    f32 += got == F32;
  }
  CHECK_EQ(wrong, 0);
  CHECK_EQ(q6, 64);   // the served file: attn_qkv 14 + attn_v 7 + ffn_down_exps 21 + ffn_down_shexp 21 + output 1
  CHECK_EQ(q4, 379);
  CHECK_EQ(f32, 310);
}

TEST(type_rule_basics) {
  CHECK(q::file_type_from_name("q4_k_m").ok());
  CHECK(q::file_type_from_name("Q5_K_M").status.code == Code::unsupported);
  CHECK(!q::tensor_allows_quantization(t("blk.0.attn_norm.weight", {2048}, F32), "qwen35moe"));        // 1-D
  CHECK(!q::tensor_allows_quantization(t("blk.0.ffn_gate_inp.weight", {2048, 256}, F32), "qwen35moe")); // router
  CHECK(!q::tensor_allows_quantization(t("blk.0.ssm_conv1d.weight", {4, 8192}, F32), "qwen35moe"));
  CHECK(!q::tensor_allows_quantization(t("blk.0.ssm_dt.bias", {32, 4}, F32), "qwen35moe"));             // not a weight
  CHECK(q::tensor_allows_quantization(t("blk.0.ssm_alpha.weight", {2048, 32}, BF16), "qwen35moe"));
  // an 8-expert model moves attn_v and attn_k to Q8_0 (the reference's rule)
  std::vector<q::TensorDesc> v = {t("blk.0.attn_k.weight", {256, 64}, BF16), t("blk.0.attn_v.weight", {256, 64}, BF16)};
  q::ModelFacts m;
  m.arch = "llama";
  m.n_expert = 8;
  m.n_layer = 1;
  const auto r = q::choose_types(v, m, q::FileType::q4_k_m);
  CHECK(r.ok());
  CHECK_EQ(r.value[0], static_cast<std::uint32_t>(q::Type::q8_0));
  CHECK_EQ(r.value[1], static_cast<std::uint32_t>(q::Type::q8_0));
  // a row length that is not a multiple of 256 falls back (Q4_K -> Q5_0), as the reference does
  std::vector<q::TensorDesc> odd = {t("blk.0.ffn_up.weight", {320, 64}, BF16)};
  m.n_expert = 0;
  const auto r2 = q::choose_types(odd, m, q::FileType::q4_k_m);
  CHECK(r2.ok());
  CHECK_EQ(r2.value[0], static_cast<std::uint32_t>(q::Type::q5_0));
}
