// provenance: original (the base-record encoding used to build test input READ-SRC from llama.cpp
//   tools/perplexity/perplexity.cpp log_softmax at commit 2ca15f5404760548c39e7b92bd43116a09414a1a)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "gguf_builder.h"
#include "io/gguf_reader.h"
#include "model/spec.h"
#include "pelias_test.h"
#include "verify/kl.h"

namespace {

ptest::GgufBuilder qwen35moe_meta(std::uint32_t blocks, std::uint32_t nextn) {
  ptest::GgufBuilder b;
  const std::string a = "qwen35moe.";
  b.kv_string("general.architecture", "qwen35moe")
      .kv_u32(a + "block_count", blocks)
      .kv_u32(a + "nextn_predict_layers", nextn)
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
      .kv_u32(a + "full_attention_interval", 4)
      .tensor("token_embd.weight", {256, 1000}, 0);
  return b;
}

// one base-file record for `logits`, encoded as the reference's perplexity tool writes it
std::vector<std::uint16_t> base_record(const std::vector<float>& logits) {
  const std::size_t n = logits.size();
  float mx = logits[0], mn = logits[0];
  for (float v : logits) {
    mx = std::max(mx, v);
    mn = std::min(mn, v);
  }
  mn = std::max(mn, mx - 16);
  double sum = 0;
  for (float v : logits) sum += static_cast<double>(std::exp(v - mx));
  const float lse = static_cast<float>(std::log(sum));
  const float min_log_prob = mn - mx - lse;
  const float scale = (mx - mn) / 65535.f;
  std::vector<std::uint16_t> rec(2 * ((n + 1) / 2) + 4, 0);
  std::memcpy(rec.data(), &scale, 4);
  std::memcpy(rec.data() + 2, &min_log_prob, 4);
  for (std::size_t i = 0; i < n; ++i)
    rec[i + 4] = logits[i] > mn ? static_cast<std::uint16_t>(std::lround((logits[i] - mn) / scale)) : 0;
  return rec;
}

}  // namespace

TEST(model_spec_reads_qwen35moe) {
  const std::vector<std::uint8_t> bytes = qwen35moe_meta(5, 1).build();
  pelias::os::FileMap m;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/model_spec.gguf", bytes, m));
  pelias::io::GgufReader r;
  CHECK(r.parse(m).ok());
  const auto s = pelias::model::read_qwen35moe_spec(r);
  CHECK(s.ok());
  if (!s.ok()) return;
  CHECK(s.value.n_layer == 4 && s.value.n_layer_all == 5);
  CHECK(s.value.n_vocab == 1000 && s.value.n_embd == 256);
  CHECK(s.value.is_recurrent(0) && s.value.is_recurrent(2) && !s.value.is_recurrent(3) && !s.value.is_recurrent(4));
  CHECK(s.value.conv_channels() == 512);
  CHECK(s.value.rope_sections[0] == 3 && s.value.rope_sections[2] == 2);
}

TEST(model_spec_refuses_what_it_cannot_run) {
  pelias::os::FileMap m;
  pelias::io::GgufReader r;
  // MTP blocks that leave no trunk
  CHECK(ptest::write_and_map(ptest::data_dir() + "/model_spec.gguf", qwen35moe_meta(2, 2).build(), m));
  CHECK(r.parse(m).ok());
  CHECK(!pelias::model::read_qwen35moe_spec(r).ok());
  // another architecture
  ptest::GgufBuilder b;
  b.kv_string("general.architecture", "llama").tensor("token_embd.weight", {256, 10}, 0);
  pelias::os::FileMap m2;
  pelias::io::GgufReader r2;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/model_spec2.gguf", b.build(), m2));
  CHECK(r2.parse(m2).ok());
  const auto s = pelias::model::read_qwen35moe_spec(r2);
  CHECK(!s.ok() && s.status.code == pelias::Code::unsupported);
}

TEST(kl_token_matches_itself_and_a_known_case) {
  std::vector<float> logits(1001);
  for (std::size_t i = 0; i < logits.size(); ++i) logits[i] = std::sin(static_cast<float>(i) * 0.37f) * 6.0f;
  const std::vector<std::uint16_t> rec = base_record(logits);
  const pelias::verify::KlToken t = pelias::verify::kl_token(1001, logits.data(), rec.data(), 17);
  CHECK(std::fabs(t.kld) < 1e-6);
  CHECK(t.same_top);
  CHECK(std::fabs(t.nll - t.nll_base) < 1e-3f);
  // two tokens: base p = (0.5, 0.5); model p = (e/(e+1), 1/(e+1)): KL = ln 0.5 - 0.5 ln p0 - 0.5 ln p1
  const std::vector<float> b2 = {0.f, 0.f}, m2 = {1.f, 0.f};
  const std::vector<std::uint16_t> r2 = base_record(b2);
  const pelias::verify::KlToken k = pelias::verify::kl_token(2, m2.data(), r2.data(), 0);
  const double e = std::exp(1.0), want = std::log(0.5) - 0.5 * std::log(e / (e + 1)) - 0.5 * std::log(1 / (e + 1));
  CHECK(std::fabs(k.kld - want) < 1e-5);
}

TEST(kl_stats_report_layout) {
  pelias::verify::KlStats st;
  for (int i = 0; i < 200; ++i) {
    pelias::verify::KlToken t;
    t.nll = 2.0f + static_cast<float>(i % 7) * 0.1f;
    t.nll_base = 2.0f;
    t.kld = 0.01 * (i % 5);
    t.p_diff = 0.001f * static_cast<float>(i % 3);
    t.same_top = i % 10 != 0;
    st.add(t);
  }
  const std::string rep = pelias::verify::kl_report(st);
  CHECK(rep.find("Mean    KLD:   0.020000") != std::string::npos);
  CHECK(rep.find("Same top p: 90.000") != std::string::npos);
  CHECK(pelias::verify::kl_chunk_line(1, st).rfind("   1", 0) == 0);
}
