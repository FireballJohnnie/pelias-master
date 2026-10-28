// provenance: original (the base-record encoding used to build test input READ-SRC from llama.cpp
//   tools/perplexity/perplexity.cpp log_softmax at commit 2ca15f5404760548c39e7b92bd43116a09414a1a)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf_builder.h"
#include "io/gguf_reader.h"
#include "model/arch_qwen35moe.h"
#include "model/imatrix.h"
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

namespace {
pelias::verify::KlDump dump_of(const std::vector<double>& klds, std::uint32_t per, std::uint64_t hash) {
  pelias::verify::KlDump d;
  d.n_ctx = 2 * per + 2;
  d.scored = per;
  d.chunks = static_cast<std::uint32_t>(klds.size() / per);
  d.token_hash = hash;
  for (std::size_t i = 0; i < klds.size(); ++i) {
    pelias::verify::KlToken t;
    t.kld = klds[i];
    t.nll = 1.5f + static_cast<float>(i) * 0.25f;
    t.nll_base = 1.25f;
    t.p_diff = -0.125f;
    t.same_top = i % 3 != 0;
    d.tokens.push_back(t);
  }
  return d;
}
}  // namespace

TEST(kl_dump_round_trips_and_refuses_overwrite) {
  const std::vector<std::int32_t> toks = {1, 2, 3, 70000};
  const std::uint64_t h = pelias::verify::kl_token_hash(toks.data(), toks.size());
  CHECK(h != pelias::verify::kl_token_hash(toks.data(), 3));
  const pelias::verify::KlDump d = dump_of({0.5, 0.25, 0.125, 1.0, 2.0, 0.0}, 3, h);
  const std::string path = ptest::data_dir() + "/kl_dump_rt.bin";
  std::remove(path.c_str());
  CHECK(pelias::verify::kl_dump_write(path, d).ok());
  CHECK(!pelias::verify::kl_dump_write(path, d).ok());  // never overwrites
  const auto r = pelias::verify::kl_dump_read(path);
  CHECK(r.ok());
  CHECK(r.value.n_ctx == d.n_ctx && r.value.chunks == 2 && r.value.scored == 3 && r.value.token_hash == h);
  CHECK(r.value.tokens.size() == 6);
  for (std::size_t i = 0; i < 6; ++i) {
    CHECK(r.value.tokens[i].kld == d.tokens[i].kld);
    CHECK(r.value.tokens[i].nll == d.tokens[i].nll);
    CHECK(r.value.tokens[i].p_diff == d.tokens[i].p_diff);
    CHECK(r.value.tokens[i].same_top == d.tokens[i].same_top);
  }
  std::remove(path.c_str());
}

TEST(kl_compare_paired_statistics) {
  // A = B - 0.1 on every token, with chunk-level variation: difference -0.1 exactly, zero spread -> resolved
  std::vector<double> b, a;
  for (int i = 0; i < 40; ++i) {
    b.push_back(0.2 + 0.01 * (i % 7));
    a.push_back(b.back() - 0.1);
  }
  const auto da = dump_of(a, 10, 7), db = dump_of(b, 10, 7);
  const auto r = pelias::verify::kl_compare(da, db, "A", "B");
  CHECK(r.ok());
  CHECK(r.value.find("difference (A - B) -0.100000") != std::string::npos);
  CHECK(r.value.find("(t 3.182, 3 df)") != std::string::npos);
  CHECK(r.value.find("VERDICT: A has less drift") != std::string::npos);
  // identical files: no difference
  const auto same = pelias::verify::kl_compare(db, db, "B", "B");
  CHECK(same.ok() && same.value.find("contains zero") != std::string::npos);
  // chunk means of the difference +0.1, -0.1, +0.1, -0.1: mean 0, unresolved
  std::vector<double> c;
  for (int i = 0; i < 40; ++i) c.push_back(b[static_cast<std::size_t>(i)] + ((i / 10) % 2 ? -0.1 : 0.1));
  const auto mixed = pelias::verify::kl_compare(dump_of(c, 10, 7), db, "C", "B");
  CHECK(mixed.ok() && mixed.value.find("contains zero") != std::string::npos);
  // different tokens, or a single chunk: refused
  CHECK(!pelias::verify::kl_compare(da, dump_of(b, 10, 8), "A", "B").ok());
  CHECK(!pelias::verify::kl_compare(dump_of(a, 40, 7), dump_of(b, 40, 7), "A", "B").ok());
}

TEST(act_rotate32_is_orthonormal_and_spreads_a_spike) {
  std::vector<float> x(64), y;
  for (int i = 0; i < 64; ++i) x[static_cast<std::size_t>(i)] = std::sin(0.37f * static_cast<float>(i)) * static_cast<float>(i % 7);
  y = x;
  pelias::model::act_rotate32(y.data(), 64);
  double n0 = 0, n1 = 0;
  for (int i = 0; i < 64; ++i) { n0 += double(x[size_t(i)]) * x[size_t(i)]; n1 += double(y[size_t(i)]) * y[size_t(i)]; }
  CHECK(std::fabs(n0 - n1) < 1e-4 * n0);  // the length is kept
  pelias::model::act_rotate32(y.data(), 64);
  for (int i = 0; i < 64; ++i) CHECK(std::fabs(y[size_t(i)] - x[size_t(i)]) < 1e-5f);  // its own inverse
  std::vector<float> s(32, 0.0f);
  s[5] = 100.0f;  // one spike: spread evenly over the block
  pelias::model::act_rotate32(s.data(), 32);
  for (float v : s) CHECK(std::fabs(std::fabs(v) - 100.0f / std::sqrt(32.0f)) < 1e-4f);
}

TEST(imatrix_lists_weights_collected_rotated) {
  pelias::model::ImatrixCollector col;
  std::vector<float> rows(16 * 32, 1.0f);
  col.add_dense("blk.0.attn_qkv.weight", rows.data(), 16, 32, 32);
  col.add_dense("blk.0.ffn_gate_inp.weight", rows.data(), 16, 32, 32);
  col.mark_rotated("blk.0.attn_qkv.weight");
  col.mark_rotated("blk.0.never_collected.weight");  // marked but no entry: not listed
  const std::vector<std::string> r = col.rotated();
  CHECK(r.size() == 1 && r[0] == "blk.0.attn_qkv.weight");
}
