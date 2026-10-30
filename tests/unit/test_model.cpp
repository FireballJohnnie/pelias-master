// provenance: original (the base-record encoding used to build test input READ-SRC from llama.cpp
//   tools/perplexity/perplexity.cpp log_softmax at commit 2ca15f5404760548c39e7b92bd43116a09414a1a)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <filesystem>
#if defined(_WIN32)
#include <process.h>
#define PELIAS_TEST_GETPID _getpid
#else
#include <unistd.h>
#define PELIAS_TEST_GETPID getpid
#endif

#include "gguf_builder.h"
#include "io/planes_build.h"
#include "io/tier_build.h"
#include "io/requantize.h"
#include "io/gguf_reader.h"
#include "model/arch_qwen35moe.h"
#include "quant/block_types.h"
#include "quant/quant_cpu.h"
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
  // another architecture (llama, qwen2 and qwen3 are read since GP3-09)
  ptest::GgufBuilder b;
  b.kv_string("general.architecture", "falcon").tensor("token_embd.weight", {256, 10}, 0);
  pelias::os::FileMap m2;
  pelias::io::GgufReader r2;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/model_spec2.gguf", b.build(), m2));
  CHECK(r2.parse(m2).ok());
  const auto s = pelias::model::read_qwen35moe_spec(r2);
  CHECK(!s.ok() && s.status.code == pelias::Code::unsupported);
  // a dense family without its keys: refused as incomplete, naming the key
  ptest::GgufBuilder b3;
  b3.kv_string("general.architecture", "llama").tensor("token_embd.weight", {256, 10}, 0);
  pelias::os::FileMap m3;
  pelias::io::GgufReader r3;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/model_spec3.gguf", b3.build(), m3));
  CHECK(r3.parse(m3).ok());
  const auto s3 = pelias::model::read_qwen35moe_spec(r3);
  CHECK(!s3.ok() && s3.status.code == pelias::Code::corrupt);
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

TEST(act_round_q8_256_matches_the_int8_kernel_format) {
  // one scale per 256 = amax / 127; round half to even; clamp to -127..127 (q8k_warp, tools/probes/batch_c)
  std::vector<float> x(512, 0.0f);
  x[0] = 127.0f;  // block 0: amax 127, scale exactly 1
  x[1] = 2.5f;    // ties go to even
  x[2] = 3.5f;
  x[3] = -2.5f;
  x[4] = 0.4f;
  x[5] = -126.6f;
  // block 1 all zero: scale 1, everything stays 0
  std::vector<float> y = x;
  pelias::model::act_round_q8_256(y.data(), 512);
  CHECK(y[0] == 127.0f);
  CHECK(y[1] == 2.0f);
  CHECK(y[2] == 4.0f);
  CHECK(y[3] == -2.0f);
  CHECK(y[4] == 0.0f);
  CHECK(y[5] == -127.0f);
  for (int i = 256; i < 512; ++i) CHECK(y[size_t(i)] == 0.0f);
  // a general row: every value is a multiple of its block's scale, within half a step, and the largest is exact
  std::vector<float> r(256);
  for (int i = 0; i < 256; ++i) r[size_t(i)] = std::sin(0.11f * float(i)) * (1.0f + float(i % 13));
  float amax = 0.0f;
  for (float v : r) amax = std::max(amax, std::fabs(v));
  const float scale = amax / 127.0f;
  std::vector<float> q = r;
  pelias::model::act_round_q8_256(q.data(), 256);
  for (int i = 0; i < 256; ++i) {
    CHECK(std::fabs(q[size_t(i)] - r[size_t(i)]) <= 0.5f * scale * 1.0001f);
    const float k = q[size_t(i)] / scale;
    CHECK(std::fabs(k - std::nearbyint(k)) < 1e-3f);
  }
}

TEST(act_round_per32_refinements_of_the_256_layout) {
  // two 32-blocks inside one 256-block: a large one (amax 127) and a small one (amax 10)
  std::vector<float> x(256, 0.0f);
  for (int j = 0; j < 32; ++j) x[size_t(j)] = 127.0f * std::sin(0.3f * float(j + 1));
  x[0] = 127.0f;
  for (int j = 0; j < 32; ++j) x[size_t(32 + j)] = 10.0f * std::cos(0.7f * float(j));
  x[32] = 10.0f;
  // eight scales: each block uses its own amax / 127, so the small block's largest value is exact and every value is
  // within half of its own step
  std::vector<float> a = x;
  pelias::model::act_round_q8_256x8(a.data(), 256);
  CHECK(a[0] == 127.0f);
  CHECK(std::fabs(a[32] - 10.0f) < 1e-5f);
  for (int j = 32; j < 64; ++j) CHECK(std::fabs(a[size_t(j)] - x[size_t(j)]) <= 0.5f * (10.0f / 127.0f) * 1.0001f);
  // power-of-two exponent: the small block (127 / 10 = 12.7, so e = 3, scale 1/8) sits between the 256 and the 32 scale
  std::vector<float> e = x;
  pelias::model::act_round_q8_256e(e.data(), 256);
  for (int j = 32; j < 64; ++j) {
    CHECK(std::fabs(e[size_t(j)] - x[size_t(j)]) <= 0.5f * 0.125f * 1.0001f);
    const float k = e[size_t(j)] * 8.0f;  // multiples of the 1/8 step
    CHECK(std::fabs(k - std::nearbyint(k)) < 1e-4f);
  }
  // the large block keeps the 256 scale (e = 0) in both refinements: identical to one scale per 256 there
  std::vector<float> o = x;
  pelias::model::act_round_q8_256(o.data(), 256);
  for (int j = 0; j < 32; ++j) CHECK(e[size_t(j)] == o[size_t(j)] && a[size_t(j)] == o[size_t(j)]);
  // error ordering on the small block: eight scales <= exponent <= one scale per 256
  double ea = 0, ee = 0, eo = 0;
  for (int j = 32; j < 64; ++j) {
    ea += std::pow(a[size_t(j)] - x[size_t(j)], 2);
    ee += std::pow(e[size_t(j)] - x[size_t(j)], 2);
    eo += std::pow(o[size_t(j)] - x[size_t(j)], 2);
  }
  CHECK(ea <= ee && ee <= eo);
}

TEST(h_gate_live_drops_the_8th_bit_on_the_smallest_expected_loss) {
  // 3 rows x 512 inputs of Q8_Z, 2 tokens: f = 1 changes nothing; f = 0 is the base with H = 0.5 everywhere; f = 0.5
  // keeps the 8th bit on the 8 of 16 blocks with the largest s^2 * sum(x^2) per row and token
  const int64_t k = 512, n_out = 3, n_tok = 2, nb = k / 32;
  std::vector<float> w(static_cast<size_t>(n_out * k)), x(static_cast<size_t>(n_tok * k));
  std::uint32_t s = 5;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return (static_cast<float>(s >> 8) / 16777216.0f) - 0.5f; };
  for (float& v : w) v = 0.05f * rnd();
  for (int64_t i = 0; i < n_tok * k; ++i) x[static_cast<size_t>(i)] = rnd() * (i % 97 == 0 ? 8.0f : 1.0f);
  const std::uint64_t rb = pelias::quant::row_bytes(200, static_cast<std::uint64_t>(k));
  std::vector<std::uint8_t> q(static_cast<size_t>(rb * static_cast<std::uint64_t>(n_out)));
  for (int64_t r = 0; r < n_out; ++r)
    CHECK(pelias::quant::quantize_row(pelias::quant::Type::q8_z, w.data() + r * k, q.data() + static_cast<std::uint64_t>(r) * rb, k).ok());
  // reference values per element: full, and base with H = 0.5
  std::vector<double> full(static_cast<size_t>(n_out * k)), half(full.size()), sc(static_cast<size_t>(n_out * nb));
  for (int64_t r = 0; r < n_out; ++r)
    for (int64_t b = 0; b < k / 256; ++b) {
      const std::uint8_t* p = q.data() + static_cast<std::uint64_t>(r) * rb + static_cast<std::uint64_t>(264 * b);
      const float d = pelias::quant::fp16_to_fp32(static_cast<std::uint16_t>(p[0] | (p[1] << 8)));
      std::uint64_t lv = 0;
      for (int i = 0; i < 6; ++i) lv |= static_cast<std::uint64_t>(p[2 + i]) << (8 * i);
      for (int j = 0; j < 8; ++j) {
        const double sj = pelias::quant::q8_z_scale(d, static_cast<int>((lv >> (6 * j)) & 63u));
        sc[static_cast<size_t>(r * nb + 8 * b + j)] = sj;
        for (int i = 0; i < 32; ++i) {
          const int qv = static_cast<std::int8_t>(p[8 + 32 * j + i]);
          const size_t at = static_cast<size_t>(r * k + 256 * b + 32 * j + i);
          full[at] = sj * qv;
          half[at] = sj * (qv - ((qv + 127) & 1) + 0.5);
        }
      }
    }
  auto run = [&](double f) {
    std::vector<float> y(static_cast<size_t>(n_tok * n_out));
    for (int64_t t = 0; t < n_tok; ++t)
      for (int64_t r = 0; r < n_out; ++r) {
        double a = 0;
        for (int64_t i = 0; i < k; ++i) a += full[static_cast<size_t>(r * k + i)] * x[static_cast<size_t>(t * k + i)];
        y[static_cast<size_t>(t * n_out + r)] = static_cast<float>(a);
      }
    pelias::model::h_gate_correct_q8z(q.data(), rb, n_out, k, x.data(), n_tok, k, y.data(), n_out, f, 2);
    return y;
  };
  const std::vector<float> y1 = run(1.0), y0 = run(0.0), yh = run(0.5);
  for (int64_t t = 0; t < n_tok; ++t)
    for (int64_t r = 0; r < n_out; ++r) {
      double a1 = 0, a0 = 0;
      std::vector<std::pair<double, int64_t>> sco;
      for (int64_t j = 0; j < nb; ++j) {
        double x2 = 0;
        for (int i = 0; i < 32; ++i) x2 += double(x[static_cast<size_t>(t * k + 32 * j + i)]) * x[static_cast<size_t>(t * k + 32 * j + i)];
        sco.push_back({sc[static_cast<size_t>(r * nb + j)] * sc[static_cast<size_t>(r * nb + j)] * x2, j});
      }
      std::sort(sco.begin(), sco.end());
      std::vector<bool> kept(static_cast<size_t>(nb), false);
      for (int64_t q2 = nb / 2; q2 < nb; ++q2) kept[static_cast<size_t>(sco[static_cast<size_t>(q2)].second)] = true;
      double ah = 0;
      for (int64_t i = 0; i < k; ++i) {
        const size_t at = static_cast<size_t>(r * k + i);
        const double xv = x[static_cast<size_t>(t * k + i)];
        a1 += full[at] * xv;
        a0 += half[at] * xv;
        ah += (kept[static_cast<size_t>(i / 32)] ? full[at] : half[at]) * xv;
      }
      const size_t o = static_cast<size_t>(t * n_out + r);
      CHECK(std::fabs(y1[o] - a1) < 1e-4 * (1 + std::fabs(a1)));
      CHECK(std::fabs(y0[o] - a0) < 1e-4 * (1 + std::fabs(a0)));
      CHECK(std::fabs(yh[o] - ah) < 1e-4 * (1 + std::fabs(ah)));
      CHECK(std::fabs(a0 - a1) > 1e-6);
    }
}

// Run D: the T-decode route (7-bit base as integers + the gated 8th bit by AND + popcount) equals the dequantized Q7_ZT
// weights times the gpu256x8-rounded activations (the same integers in another summation order).
TEST(t_decode_q7zt_equals_dequantized_product) {
  const std::int64_t k = 512, n_out = 24, n_tok = 3;
  std::vector<float> w(static_cast<std::size_t>(n_out * k)), x(static_cast<std::size_t>(n_tok * k));
  std::uint32_t seed = 99u;
  auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u; return (static_cast<float>(seed >> 8) / 16777216.0f) - 0.5f; };
  for (float& v : w) v = 0.05f * rnd() / (0.05f + std::fabs(rnd()));
  for (float& v : x) v = rnd() * (1.0f + 4.0f * std::fabs(rnd()));
  const std::uint64_t rb = static_cast<std::uint64_t>(k / 256) * 241;
  std::vector<std::uint8_t> q(static_cast<std::size_t>(n_out) * rb);
  for (std::int64_t r = 0; r < n_out; ++r) pelias::quant::quantize_q7_zt(w.data() + r * k, q.data() + r * static_cast<std::int64_t>(rb), k);
  std::vector<float> y(static_cast<std::size_t>(n_tok * n_out));
  pelias::model::t_decode_q7zt(q.data(), rb, n_out, k, x.data(), n_tok, k, y.data(), n_out, 2);
  std::vector<float> wd(static_cast<std::size_t>(k)), xr(x);
  for (std::int64_t t = 0; t < n_tok; ++t) pelias::model::act_round_q8_256x8(xr.data() + t * k, k);
  double worst = 0.0, scale = 0.0;
  for (std::int64_t r = 0; r < n_out; ++r) {
    pelias::quant::dequantize_q7_zt(q.data() + r * static_cast<std::int64_t>(rb), wd.data(), k);
    for (std::int64_t t = 0; t < n_tok; ++t) {
      double ref = 0.0;
      for (std::int64_t i = 0; i < k; ++i) ref += static_cast<double>(wd[static_cast<std::size_t>(i)]) * xr[static_cast<std::size_t>(t * k + i)];
      scale = std::max(scale, std::fabs(ref));
      worst = std::max(worst, std::fabs(ref - static_cast<double>(y[static_cast<std::size_t>(t * n_out + r)])));
    }
  }
  std::printf("      T-decode vs dequantized product: max error %.2e of max |y| %.3f\n", worst / scale, scale);
  CHECK(worst / scale < 1e-5);
}

TEST(state_store_roundtrip_formats) {
  using K = pelias::model::ForwardOptions::StateStore;
  std::vector<float> x(256);
  std::uint32_t seed = 7u;
  auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u; return (static_cast<float>(seed >> 8) / 16777216.0f) - 0.5f; };
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = rnd() + (i % 32 == 5 ? 8.0f : 0.0f);  // one outlier per 32-block
  auto err = [&](K kind) {
    std::vector<float> y(x);
    pelias::model::state_store_roundtrip(y.data(), static_cast<std::int64_t>(y.size()), kind);
    double e = 0, r = 0;
    for (std::size_t i = 0; i < x.size(); ++i) { e += (x[i] - y[i]) * (x[i] - y[i]); r += x[i] * x[i]; }
    return e / r;
  };
  const double e8 = err(K::q8), e4 = err(K::int4), e4r = err(K::int4r), erow = err(K::int8row);
  std::printf("      store error: q8 %.2e, int4 %.2e, int4r %.2e, int8row %.2e\n", e8, e4, e4r, erow);
  CHECK(err(K::exact) == 0.0);
  CHECK(e8 < 1e-3);  // an 8.0 outlier per block sets the 8-bit step (about 3e-4 here, by step^2 / 12)
  CHECK(e4r < e4);  // rotation spreads the outlier, so the 4-bit scale is finer
  CHECK(erow < 1e-3);
  std::vector<float> zero(64, 0.0f);
  pelias::model::state_store_roundtrip(zero.data(), 64, K::int4r);
  for (float v : zero) CHECK(v == 0.0f);
}

// GP3-07: a tiny bindable qwen35moe (4 trunk blocks: three DeltaNet, one full attention; every tensor f32) whose data is
// small deterministic values instead of the builder's byte pattern, so the forward pass is finite and well scaled
namespace {
struct TinyModel {
  pelias::os::FileMap map;
  pelias::io::GgufReader reader;
  pelias::model::Qwen35MoeSpec spec;
  pelias::model::Qwen35MoeWeights w;
  bool ok = false;
};

// mtp: a fifth block blk.4 is the MTP block (nextn_predict_layers 1), as the served file has blk.40. mtp_echo: that block
// is built to give back the trunk's own last prediction (attention and expert outputs zero, eh_proj selecting the hidden,
// unit norms), so its drafts repeat the last token and a repeating greedy run accepts them (the acceptance path in tests)
void tiny_model(TinyModel& tm, bool mtp = false, bool mtp_echo = false) {
  ptest::GgufBuilder b = qwen35moe_meta(mtp ? 5 : 4, mtp ? 1 : 0);
  // a byte-level tokenizer over the 1,000-token vocabulary, so `pelias generate` runs on this file too: ids 0..255 are
  // the bytes (GPT-2 byte-to-character map), a few merges, fillers, then <|endoftext|> <|im_start|> <|im_end|> as control
  {
    std::vector<std::string> toks;
    std::vector<std::int32_t> types;
    int extra = 0;
    for (int by = 0; by < 256; ++by) {
      const bool keep = (by >= 33 && by <= 126) || (by >= 161 && by <= 172) || (by >= 174 && by <= 255);
      const char32_t cp = keep ? static_cast<char32_t>(by) : static_cast<char32_t>(256 + extra++);
      std::string u;
      if (cp < 0x80) u += static_cast<char>(cp);
      else {
        u += static_cast<char>(0xC0 | (cp >> 6));
        u += static_cast<char>(0x80 | (cp & 0x3F));
      }
      toks.push_back(u);
      types.push_back(6);
    }
    const std::vector<std::string> merges{"H e", "l l", "He ll", "o r"};
    for (const char* t : {"He", "ll", "Hell", "or"}) {
      toks.emplace_back(t);
      types.push_back(1);
    }
    while (toks.size() < 997) {
      toks.push_back("<|filler_" + std::to_string(toks.size()) + "|>");
      types.push_back(5);
    }
    for (const char* t : {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}) {
      toks.emplace_back(t);
      types.push_back(3);
    }
    b.kv_string("tokenizer.ggml.model", "gpt2").kv_string("tokenizer.ggml.pre", "qwen35").kv_string_array("tokenizer.ggml.tokens", toks)
        .kv_i32_array("tokenizer.ggml.token_type", types).kv_string_array("tokenizer.ggml.merges", merges)
        .kv_u32("tokenizer.ggml.eos_token_id", 999).kv_bool("tokenizer.ggml.add_bos_token", false);
  }
  const std::uint64_t e = 256, hd = 64, nh = 4, nkv = 2, ch = 512, di = 256, ds = 32, nvh = 8, ne = 8, ff = 256, ffs = 256;
  b.tensor("output_norm.weight", {e}, 0);
  for (int il = 0; il < 4; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    b.tensor(p + "attn_norm.weight", {e}, 0).tensor(p + "post_attention_norm.weight", {e}, 0);
    if (il == 3) {
      b.tensor(p + "attn_q.weight", {e, 2 * hd * nh}, 0).tensor(p + "attn_k.weight", {e, hd * nkv}, 0).tensor(p + "attn_v.weight", {e, hd * nkv}, 0);
      b.tensor(p + "attn_output.weight", {hd * nh, e}, 0).tensor(p + "attn_q_norm.weight", {hd}, 0).tensor(p + "attn_k_norm.weight", {hd}, 0);
    } else {
      b.tensor(p + "attn_qkv.weight", {e, ch}, 0).tensor(p + "attn_gate.weight", {e, di}, 0).tensor(p + "ssm_beta.weight", {e, nvh}, 0);
      b.tensor(p + "ssm_alpha.weight", {e, nvh}, 0).tensor(p + "ssm_out.weight", {di, e}, 0).tensor(p + "ssm_conv1d.weight", {4, ch}, 0);
      b.tensor(p + "ssm_dt.bias", {nvh}, 0).tensor(p + "ssm_a", {nvh}, 0).tensor(p + "ssm_norm.weight", {ds}, 0);
    }
    b.tensor(p + "ffn_gate_inp.weight", {e, ne}, 0).tensor(p + "ffn_up_exps.weight", {e, ff, ne}, 0).tensor(p + "ffn_gate_exps.weight", {e, ff, ne}, 0);
    b.tensor(p + "ffn_down_exps.weight", {ff, e, ne}, 0).tensor(p + "ffn_up_shexp.weight", {e, ffs}, 0).tensor(p + "ffn_gate_shexp.weight", {e, ffs}, 0);
    b.tensor(p + "ffn_down_shexp.weight", {ffs, e}, 0).tensor(p + "ffn_gate_inp_shexp.weight", {e}, 0);
  }
  if (mtp) {
    const std::string p = "blk.4.";
    b.tensor(p + "attn_norm.weight", {e}, 0).tensor(p + "post_attention_norm.weight", {e}, 0);
    b.tensor(p + "attn_q.weight", {e, 2 * hd * nh}, 0).tensor(p + "attn_k.weight", {e, hd * nkv}, 0).tensor(p + "attn_v.weight", {e, hd * nkv}, 0);
    b.tensor(p + "attn_output.weight", {hd * nh, e}, 0).tensor(p + "attn_q_norm.weight", {hd}, 0).tensor(p + "attn_k_norm.weight", {hd}, 0);
    b.tensor(p + "ffn_gate_inp.weight", {e, ne}, 0).tensor(p + "ffn_up_exps.weight", {e, ff, ne}, 0).tensor(p + "ffn_gate_exps.weight", {e, ff, ne}, 0);
    b.tensor(p + "ffn_down_exps.weight", {ff, e, ne}, 0).tensor(p + "ffn_up_shexp.weight", {e, ffs}, 0).tensor(p + "ffn_gate_shexp.weight", {e, ffs}, 0);
    b.tensor(p + "ffn_down_shexp.weight", {ffs, e}, 0).tensor(p + "ffn_gate_inp_shexp.weight", {e}, 0);
    b.tensor(p + "nextn.eh_proj.weight", {2 * e, e}, 0).tensor(p + "nextn.enorm.weight", {e}, 0).tensor(p + "nextn.hnorm.weight", {e}, 0);
    b.tensor(p + "nextn.shared_head_norm.weight", {e}, 0);
  }
  std::vector<std::uint8_t> bytes = b.build();
  pelias::os::FileMap m0;
  pelias::io::GgufReader r0;
  const std::string stem = mtp ? (mtp_echo ? "/decode_tiny_mtp_echo" : "/decode_tiny_mtp") : "/decode_tiny";
  if (!ptest::write_and_map(ptest::data_dir() + stem + "_layout.gguf", bytes, m0) || !r0.parse(m0).ok()) return;
  std::uint32_t h = 20261019u;
  auto unit = [&h]() {  // xorshift, uniform in [-1, 1)
    h ^= h << 13;
    h ^= h >> 17;
    h ^= h << 5;
    return static_cast<float>(h) / 2147483648.0f - 1.0f;
  };
  for (const pelias::io::GgufTensor& t : r0.tensors()) {
    float* d = reinterpret_cast<float*>(bytes.data() + t.file_offset);
    const std::string& nm = t.name;
    const float row = 1.0f / std::sqrt(static_cast<float>(t.ne[0]));
    for (std::uint64_t i = 0; i < t.elements(); ++i) {
      const float u = unit();
      float v = u * row;
      if (nm.find("norm") != std::string::npos) v = 1.0f + 0.1f * u;
      else if (nm.find("ssm_a") != std::string::npos) v = -(0.5f + 0.5f * std::fabs(u));
      else if (nm.find("ssm_dt") != std::string::npos) v = 0.1f * u;
      else if (nm.find("conv1d") != std::string::npos) v = 0.3f * u;
      else if (nm == "token_embd.weight") v = 0.5f * u;
      if (mtp_echo && nm.rfind("blk.4.", 0) == 0) {
        if (nm == "blk.4.attn_output.weight" || nm == "blk.4.ffn_down_exps.weight" || nm == "blk.4.ffn_down_shexp.weight") v = 0.0f;
        if (nm == "blk.4.nextn.hnorm.weight" || nm == "blk.4.nextn.shared_head_norm.weight") v = 1.0f;
        if (nm == "blk.4.nextn.eh_proj.weight") v = (i % (2 * e) == e + i / (2 * e)) ? 1.0f : 0.0f;  // row o reads input e + o
      }
      std::memcpy(d + i, &v, 4);
    }
  }
  if (!ptest::write_and_map(ptest::data_dir() + stem + ".gguf", bytes, tm.map) || !tm.reader.parse(tm.map).ok()) return;
  auto s = pelias::model::read_qwen35moe_spec(tm.reader);
  if (!s.ok()) return;
  tm.spec = s.value;
  auto w = pelias::model::bind_qwen35moe(tm.reader, tm.map.data(), tm.spec);
  if (!w.ok()) return;
  tm.w = std::move(w.value);
  tm.ok = true;
}
}  // namespace

TEST(decode_incremental_equals_the_fresh_pass) {
  TinyModel tm;
  tiny_model(tm);
  CHECK(tm.ok);
  if (!tm.ok) return;
  const std::vector<std::int32_t> tok{11, 427, 3, 999, 58, 58, 0, 731, 210, 64, 5};
  const int n = static_cast<int>(tok.size()), v = static_cast<int>(tm.spec.n_vocab);
  pelias::model::ForwardOptions opt;
  opt.threads = 3;
  std::vector<float> full(static_cast<std::size_t>(n) * static_cast<std::size_t>(v));
  CHECK(pelias::model::forward_reference(tm.spec, tm.w, tok.data(), n, 0, n, full.data(), opt).ok());
  float amax = 0.0f;
  for (float x : full) {
    CHECK(std::isfinite(x));
    amax = std::max(amax, std::fabs(x));
  }
  CHECK(amax > 1e-3f);
  // prefill 4, then one token, then a chunk of 3 (all rows), then one at a time with only the last row
  pelias::model::DecodeState st;
  pelias::model::decode_state_reset(tm.spec, st);
  std::vector<float> got(full.size(), 0.0f);
  const int chunks[][2] = {{0, 4}, {4, 1}, {5, 3}, {8, 1}, {9, 1}, {10, 1}};
  for (const auto& ck : chunks) {
    const bool last_only = ck[0] >= 8;
    std::vector<float> lg(static_cast<std::size_t>(last_only ? 1 : ck[1]) * static_cast<std::size_t>(v));
    CHECK(pelias::model::forward_decode(tm.spec, tm.w, st, tok.data() + ck[0], ck[1], lg.data(), last_only, opt).ok());
    const int first = last_only ? ck[0] + ck[1] - 1 : ck[0];
    std::copy(lg.begin(), lg.end(), got.begin() + static_cast<std::ptrdiff_t>(first) * v);
  }
  CHECK(st.pos == n);
  double worst = 0.0;
  for (std::size_t i = 0; i < full.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(full[i] - got[i])));
  std::printf("    decode vs fresh: max |diff| %.3g of max |logit| %.3g\n", worst, static_cast<double>(amax));
  CHECK(worst <= 1e-5 * static_cast<double>(amax));
  // the whole-chunk records are refused, and an unsized state is refused
  pelias::model::ForwardOptions bad = opt;
  bad.state_store = pelias::model::ForwardOptions::StateStore::q8;
  std::vector<float> one(static_cast<std::size_t>(v));
  CHECK(!pelias::model::forward_decode(tm.spec, tm.w, st, tok.data(), 1, one.data(), true, bad).ok());
  pelias::model::DecodeState empty;
  CHECK(!pelias::model::forward_decode(tm.spec, tm.w, empty, tok.data(), 1, one.data(), true, opt).ok());
}

// The same on the fast CPU path (GP3-10): a batch of 11 (rows decoded once, kern::dot_row_n) against the same tokens fed
// 4 + 1 + 3 + 1 + 1 + 1; batches of four or fewer take the per-group path, so the two must agree within the rounding of
// the attention and recurrence (the matrix products are the same bits per token).
TEST(decode_incremental_equals_the_fresh_pass_on_the_fast_path) {
  TinyModel tm;
  tiny_model(tm);
  CHECK(tm.ok);
  if (!tm.ok) return;
  const std::vector<std::int32_t> tok{11, 427, 3, 999, 58, 58, 0, 731, 210, 64, 5};
  const int n = static_cast<int>(tok.size()), v = static_cast<int>(tm.spec.n_vocab);
  pelias::model::ForwardOptions opt;
  opt.threads = 3;
  opt.fast_cpu = true;
  std::vector<float> full(static_cast<std::size_t>(n) * static_cast<std::size_t>(v));
  CHECK(pelias::model::forward_reference(tm.spec, tm.w, tok.data(), n, 0, n, full.data(), opt).ok());
  float amax = 0.0f;
  for (float x : full) amax = std::max(amax, std::fabs(x));
  pelias::model::DecodeState st;
  pelias::model::decode_state_reset(tm.spec, st);
  std::vector<float> got(full.size(), 0.0f);
  const int chunks[][2] = {{0, 4}, {4, 1}, {5, 3}, {8, 1}, {9, 1}, {10, 1}};
  for (const auto& ck : chunks) {
    std::vector<float> lg(static_cast<std::size_t>(ck[1]) * static_cast<std::size_t>(v));
    CHECK(pelias::model::forward_decode(tm.spec, tm.w, st, tok.data() + ck[0], ck[1], lg.data(), false, opt).ok());
    std::copy(lg.begin(), lg.end(), got.begin() + static_cast<std::ptrdiff_t>(ck[0]) * v);
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < full.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(full[i] - got[i])));
  std::printf("    fast path, batch of 11 vs pieces: max |diff| %.3g of max |logit| %.3g\n", worst, static_cast<double>(amax));
  CHECK(amax > 1e-3f);
  CHECK(worst <= 1e-5 * static_cast<double>(amax));
}

// Residual planes end to end on the tiny model: its F32 file is the original, a Q4_K_M copy of it (Pelias's quantizer) is
// the model file; `pelias planes` (build_planes) fits 2 planes per row, attach_planes reads them, and the forward pass
// with 1 then 2 planes comes closer to the original's logits each time; planes off gives the file alone back exactly;
// the exact head re-score changes only the largest logits. Files go to a fresh temporary folder (nothing is removed).
TEST(planes_bring_a_quantized_file_toward_its_original) {
  TinyModel tm;
  tiny_model(tm);
  CHECK(tm.ok);
  if (!tm.ok) return;
  namespace fs = std::filesystem;
  static int run = 0;
  const fs::path dir = fs::temp_directory_path() / ("pelias_planes_test_" + std::to_string(PELIAS_TEST_GETPID()) + "_" + std::to_string(run++));
  std::error_code ec;
  fs::create_directories(dir, ec);
  const std::string q4 = (dir / "tiny_Q4_K_M.gguf").string(), pl = (dir / "tiny_Q4_K_M.planes2.gguf").string();
  pelias::io::RequantOptions ro;
  ro.ftype = pelias::quant::FileType::q4_k_m;
  ro.threads = 2;
  const auto rq = pelias::io::requantize(tm.map, tm.reader, q4, ro);
  CHECK(rq.ok());
  if (!rq.ok()) return;
  pelias::os::FileMap qm;
  pelias::io::GgufReader qr;
  CHECK(qm.open(q4).ok() && qr.parse(qm).ok());
  pelias::io::PlaneBuildOptions po;
  po.k = 2;
  po.threads = 2;
  po.exact_head = true;
  const auto rep = pelias::io::build_planes(qm, qr, tm.map, tm.reader, pl, po);
  CHECK(rep.ok());
  if (!rep.ok()) {
    std::printf("      %s\n", rep.status.text().c_str());
    return;
  }
  std::printf("    planes for %zu weights, %llu bytes; residual %.4g -> %.4g\n", rep.value.tensors, static_cast<unsigned long long>(rep.value.bytes),
              rep.value.err_before, rep.value.err_after);
  CHECK(rep.value.err_after < 0.2 * rep.value.err_before);
  auto qw = pelias::model::bind_qwen35moe(qr, qm.data(), tm.spec);
  CHECK(qw.ok());
  if (!qw.ok()) return;
  pelias::os::FileMap pm;
  pelias::io::GgufReader pr;
  CHECK(pm.load(pl).ok() && pr.parse(pm).ok());
  const std::vector<std::int32_t> tok{11, 427, 3, 999, 58, 58, 0, 731, 210, 64, 5, 300, 17};
  const int n = static_cast<int>(tok.size()), v = static_cast<int>(tm.spec.n_vocab);
  pelias::model::ForwardOptions opt;
  opt.threads = 2;
  auto logits = [&](const pelias::model::Qwen35MoeWeights& w, const pelias::model::ForwardOptions& o) {
    std::vector<float> lg(static_cast<std::size_t>(n) * static_cast<std::size_t>(v));
    CHECK(pelias::model::forward_reference(tm.spec, w, tok.data(), n, 0, n, lg.data(), o).ok());
    return lg;
  };
  auto kl = [&](const std::vector<float>& p, const std::vector<float>& q) {  // mean KL(p || q) over positions
    double total = 0.0;
    for (int t = 0; t < n; ++t) {
      const float* a = p.data() + static_cast<std::ptrdiff_t>(t) * v;
      const float* b = q.data() + static_cast<std::ptrdiff_t>(t) * v;
      double ma = a[0], mb = b[0], sa = 0.0, sb = 0.0;
      for (int i = 1; i < v; ++i) ma = std::max(ma, static_cast<double>(a[i])), mb = std::max(mb, static_cast<double>(b[i]));
      for (int i = 0; i < v; ++i) sa += std::exp(a[i] - ma), sb += std::exp(b[i] - mb);
      for (int i = 0; i < v; ++i) {
        const double la = a[i] - ma - std::log(sa), lb = b[i] - mb - std::log(sb);
        total += std::exp(la) * (la - lb);
      }
    }
    return total / n;
  };
  const std::vector<float> ref = logits(tm.w, opt);
  const std::vector<float> l0 = logits(qw.value, opt);
  pelias::model::Qwen35MoeWeights w1 = qw.value, w2 = qw.value;
  const auto a1 = pelias::model::attach_planes(pr, pm.data(), w1, 1);
  const auto a2 = pelias::model::attach_planes(pr, pm.data(), w2, 0);
  CHECK(a1.ok() && a2.ok());
  if (!a1.ok() || !a2.ok()) return;
  CHECK_EQ(static_cast<std::size_t>(a2.value), rep.value.tensors);
  const std::vector<float> l1 = logits(w1, opt), l2 = logits(w2, opt);
  // the tiny model's random weights give near-flat distributions (KL about 1e-15), so the logits themselves are compared
  auto mse = [&](const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) e += static_cast<double>(a[i] - b[i]) * (a[i] - b[i]);
    return e / static_cast<double>(a.size());
  };
  const double k0 = mse(ref, l0), k1 = mse(ref, l1), k2 = mse(ref, l2);
  std::printf("    logit error to the original: file %.4g, + 1 plane %.4g (%.3f), + 2 planes %.4g (%.3f); KL %.3g / %.3g / %.3g\n", k0, k1, k1 / k0, k2,
              k2 / k0, kl(ref, l0), kl(ref, l1), kl(ref, l2));
  CHECK(k1 < 0.6 * k0);
  CHECK(k2 < 0.6 * k1);
  pelias::model::ForwardOptions off = opt;
  off.planes = false;
  CHECK(logits(w2, off) == l0);
  // more planes than stored, and a side file built for another model file, are refused
  pelias::model::Qwen35MoeWeights w3 = qw.value;
  CHECK(!pelias::model::attach_planes(pr, pm.data(), w3, 3).ok());
  pelias::model::Qwen35MoeWeights wf = tm.w;
  CHECK(!pelias::model::attach_planes(pr, pm.data(), wf, 0).ok());
  // the exact head re-score: only the largest logits of each row change
  CHECK(w2.output_exact.present());
  pelias::model::ForwardOptions hr = opt;
  hr.head_rescore = 8;
  const std::vector<float> lh = logits(w2, hr);
  int worst = 0;
  for (int t = 0; t < n; ++t) {
    int changed = 0;
    for (int i = 0; i < v; ++i) {
      const std::size_t at = static_cast<std::size_t>(t) * static_cast<std::size_t>(v) + static_cast<std::size_t>(i);
      changed += lh[at] != l2[at];
    }
    worst = std::max(worst, changed);
  }
  CHECK(worst <= 8);
  // every row re-scored exactly: no worse than the planed output layer (here the tied embedding, planed too)
  hr.head_rescore = v;
  const double kall = mse(ref, logits(w2, hr));
  std::printf("    exact head re-score: 8 rows %.4g, all rows %.4g (%.3f of the file alone)\n", mse(ref, lh), kall, kall / k0);
  CHECK(kall <= k2);
}

// Usage-tiered experts (io/tier_build.h, docs/PELIAS_RESIDUAL_PLANES.md 6.7): (1) a tier that gives every expert its base
// file's own type rebuilds that file's expert bytes exactly and the same logits bit for bit; (2) a mixed tier reads each
// expert in its own type (every row equals quantize + dequantize of the original row in that type), its size is the sum of
// its parts, finer experts move the logits toward the original, and an existing output is refused.
TEST(tiered_experts_read_each_expert_in_its_own_type) {
  TinyModel tm;
  tiny_model(tm);
  CHECK(tm.ok);
  if (!tm.ok) return;
  namespace fs = std::filesystem;
  namespace q = pelias::quant;
  static int run = 0;
  const fs::path dir = fs::temp_directory_path() / ("pelias_tier_test_" + std::to_string(PELIAS_TEST_GETPID()) + "_" + std::to_string(run++));
  std::error_code ec;
  fs::create_directories(dir, ec);
  const std::string q4 = (dir / "tiny_Q4_K_M.gguf").string();
  pelias::io::RequantOptions ro;
  ro.ftype = q::FileType::q4_k_m;
  ro.threads = 2;
  CHECK(pelias::io::requantize(tm.map, tm.reader, q4, ro).ok());
  pelias::os::FileMap qm;
  pelias::io::GgufReader qr;
  CHECK(qm.open(q4).ok() && qr.parse(qm).ok());
  const std::uint32_t L = tm.spec.n_layer, E = tm.spec.n_expert;
  const std::vector<std::int32_t> tok{11, 427, 3, 999, 58, 58, 0, 731, 210, 64, 5, 300, 17};
  const int n = static_cast<int>(tok.size()), v = static_cast<int>(tm.spec.n_vocab);
  pelias::model::ForwardOptions opt;
  opt.threads = 2;
  auto logits = [&](const pelias::model::Qwen35MoeWeights& w) {
    std::vector<float> lg(static_cast<std::size_t>(n) * static_cast<std::size_t>(v));
    CHECK(pelias::model::forward_reference(tm.spec, w, tok.data(), n, 0, n, lg.data(), opt).ok());
    return lg;
  };
  auto mse = [&](const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) e += static_cast<double>(a[i] - b[i]) * (a[i] - b[i]);
    return e / static_cast<double>(a.size());
  };
  const char* kinds[] = {"gate", "up", "down"};
  // (1) every expert in the base file's own type
  std::map<std::string, std::vector<std::uint32_t>> same;
  for (std::uint32_t l = 0; l < L; ++l)
    for (const char* k : kinds) {
      const std::string nm = "blk." + std::to_string(l) + ".ffn_" + k + "_exps.weight";
      same[nm] = std::vector<std::uint32_t>(E, qr.find_tensor(nm)->type);
    }
  const std::string t1 = (dir / "tiny_tier_same.gguf").string();
  pelias::io::TierBuildOptions to;
  to.threads = 2;
  const auto r1 = pelias::io::build_tiered(qm, qr, tm.map, tm.reader, same, t1, to);
  CHECK(r1.ok());
  if (!r1.ok()) {
    std::printf("      %s\n", r1.status.text().c_str());
    return;
  }
  CHECK_EQ(r1.value.expert_bytes_out, r1.value.expert_bytes_base);
  pelias::os::FileMap m1;
  pelias::io::GgufReader g1;
  CHECK(m1.open(t1).ok() && g1.parse(m1).ok());
  for (const auto& [nm, types] : same) {  // one part per weight, byte for byte the base file's tensor
    const auto* a = qr.find_tensor(nm);
    const auto* b = g1.find_tensor(pelias::io::tier_part_name(nm, types[0]));
    CHECK(a && b && a->bytes == b->bytes && std::memcmp(qm.data() + a->file_offset, m1.data() + b->file_offset, a->bytes) == 0);
  }
  auto w0 = pelias::model::bind_qwen35moe(qr, qm.data(), tm.spec);
  auto w1 = pelias::model::bind_qwen35moe(g1, m1.data(), tm.spec);
  CHECK(w0.ok() && w1.ok());
  if (!w0.ok() || !w1.ok()) return;
  CHECK(!w1.value.layers[0].gate_exps.slots.empty());
  const std::vector<float> ref = logits(tm.w), l0 = logits(w0.value), l1 = logits(w1.value);
  CHECK(l0 == l1);
  // (2) a mixed tier: experts cycle through four types
  const std::uint32_t cyc[4] = {static_cast<std::uint32_t>(q::Type::q4_tg2), static_cast<std::uint32_t>(q::Type::q5_k),
                                static_cast<std::uint32_t>(q::Type::q6_k), static_cast<std::uint32_t>(q::Type::q8_0)};
  std::map<std::string, std::vector<std::uint32_t>> mixed, fine, coarse;
  std::uint64_t want_bytes = 0;
  for (const auto& [nm, types] : same) {
    const auto* t = qr.find_tensor(nm);
    for (std::uint32_t e = 0; e < E; ++e) {
      mixed[nm].push_back(cyc[(e + nm.size()) % 4]);
      want_bytes += q::row_bytes(mixed[nm].back(), t->ne[0]) * t->ne[1];
    }
    fine[nm] = std::vector<std::uint32_t>(E, cyc[3]);
    coarse[nm] = std::vector<std::uint32_t>(E, cyc[0]);
  }
  const std::string t2 = (dir / "tiny_tier_mixed.gguf").string(), t3 = (dir / "tiny_tier_fine.gguf").string(), t4 = (dir / "tiny_tier_coarse.gguf").string();
  const auto r2 = pelias::io::build_tiered(qm, qr, tm.map, tm.reader, mixed, t2, to);
  CHECK(r2.ok() && r2.value.expert_bytes_out == want_bytes);
  CHECK(pelias::io::build_tiered(qm, qr, tm.map, tm.reader, fine, t3, to).ok());
  CHECK(pelias::io::build_tiered(qm, qr, tm.map, tm.reader, coarse, t4, to).ok());
  CHECK(!pelias::io::build_tiered(qm, qr, tm.map, tm.reader, mixed, t2, to).ok());  // an existing output is refused
  pelias::os::FileMap m2, m3, m4;
  pelias::io::GgufReader g2, g3, g4;
  CHECK(m2.open(t2).ok() && g2.parse(m2).ok() && m3.open(t3).ok() && g3.parse(m3).ok() && m4.open(t4).ok() && g4.parse(m4).ok());
  auto w2 = pelias::model::bind_qwen35moe(g2, m2.data(), tm.spec);
  auto w3 = pelias::model::bind_qwen35moe(g3, m3.data(), tm.spec);
  auto w4 = pelias::model::bind_qwen35moe(g4, m4.data(), tm.spec);
  CHECK(w2.ok() && w3.ok() && w4.ok());
  if (!w2.ok() || !w3.ok() || !w4.ok()) return;
  // every expert row through its slot = quantize + dequantize of the original row in that expert's type
  bool rows_ok = true;
  for (std::uint32_t l = 0; l < L && rows_ok; ++l) {
    const pelias::model::Qwen35MoeLayer& ly = w2.value.layers[l];
    for (const pelias::model::WeightView* wv : {&ly.gate_exps, &ly.up_exps, &ly.down_exps}) {
      const auto* src = tm.reader.find_tensor(wv->name);
      const std::int64_t k = static_cast<std::int64_t>(wv->ne[0]);
      std::vector<float> orig(static_cast<std::size_t>(k)), got(orig.size()), want(orig.size());
      for (std::uint32_t e = 0; e < E && rows_ok; ++e) {
        const pelias::model::ExpertSlot& sl = wv->slots[e];
        rows_ok = rows_ok && sl.type == mixed[wv->name][e];
        std::vector<std::uint8_t> qb(q::row_bytes(sl.type, wv->ne[0]));
        for (std::uint64_t r = 0; r < wv->ne[1] && rows_ok; r += 37) {
          (void)q::dequantize_row(static_cast<q::Type>(src->type), tm.map.data() + src->file_offset + (e * wv->ne[1] + r) * q::row_bytes(src->type, wv->ne[0]), orig.data(), k);
          (void)q::quantize_rows(static_cast<q::Type>(sl.type), orig.data(), qb.data(), 1, k, nullptr);
          (void)q::dequantize_row(static_cast<q::Type>(sl.type), qb.data(), want.data(), k);
          (void)q::dequantize_row(static_cast<q::Type>(sl.type), sl.data + r * sl.row_bytes, got.data(), k);
          rows_ok = got == want;
        }
      }
    }
  }
  CHECK(rows_ok);
  const double e_coarse = mse(logits(w4.value), ref), e_mixed = mse(logits(w2.value), ref), e_fine = mse(logits(w3.value), ref), e_base = mse(l0, ref);
  std::printf("    logit error vs the original: Q4_K_M %.3g; tiered all Q4_TG2 %.3g, mixed %.3g, all Q8_0 %.3g\n", e_base, e_coarse, e_mixed, e_fine);
  CHECK(e_fine < e_mixed && e_mixed < e_coarse);
  // the allocation file (tools/tier_alloc.py) reads back as written; unknown types and repeated names are refused
  const std::string a1 = (dir / "alloc_ok.tsv").string(), a2 = (dir / "alloc_bad.tsv").string(), a3 = (dir / "alloc_twice.tsv").string();
  if (std::FILE* f = std::fopen(a1.c_str(), "wbx")) { std::fputs("blk.0.ffn_gate_exps.weight\tQ4_TG2,Q5_K,Q6_K,Q8_0\r\n\nblk.0.ffn_up_exps.weight\tQ8_0\n", f); std::fclose(f); }
  if (std::FILE* f = std::fopen(a2.c_str(), "wbx")) { std::fputs("blk.0.ffn_gate_exps.weight\tQ4_TG2,Q9_X\n", f); std::fclose(f); }
  if (std::FILE* f = std::fopen(a3.c_str(), "wbx")) { std::fputs("x\tQ8_0\nx\tQ8_0\n", f); std::fclose(f); }
  const auto ra = pelias::io::read_tier_alloc(a1);
  CHECK(ra.ok() && ra.value.size() == 2 && ra.value.at("blk.0.ffn_gate_exps.weight") == std::vector<std::uint32_t>(cyc, cyc + 4) &&
        ra.value.at("blk.0.ffn_up_exps.weight") == std::vector<std::uint32_t>{cyc[3]});
  CHECK(!pelias::io::read_tier_alloc(a2).ok() && !pelias::io::read_tier_alloc(a3).ok());
}

// Planes on a usage-tiered file, block selection, per-tensor plane counts and per-pick reading (docs/PELIAS_RESIDUAL_PLANES.md
// 6.7-6.9): (1) planes fitted expert by expert on a tiered file move the logits toward the original; (2) selecting every block
// gives exactly the full-row planes' logits, a half budget keeps exactly that many blocks and lands between none and all;
// (3) --k-rules set each tensor's plane count; (4) per-pick reading: all picks = every pick, fewer picks lands between;
// (5) planes fitted to one allocation are refused on another.
TEST(planes_on_tiered_files_select_blocks_and_picks) {
  TinyModel tm;
  tiny_model(tm);
  CHECK(tm.ok);
  if (!tm.ok) return;
  namespace fs = std::filesystem;
  namespace q = pelias::quant;
  static int run = 0;
  const fs::path dir = fs::temp_directory_path() / ("pelias_tierplanes_test_" + std::to_string(PELIAS_TEST_GETPID()) + "_" + std::to_string(run++));
  std::error_code ec;
  fs::create_directories(dir, ec);
  const std::string q4 = (dir / "tiny_Q4_K_M.gguf").string();
  pelias::io::RequantOptions ro;
  ro.ftype = q::FileType::q4_k_m;
  ro.threads = 2;
  CHECK(pelias::io::requantize(tm.map, tm.reader, q4, ro).ok());
  pelias::os::FileMap qm;
  pelias::io::GgufReader qr;
  CHECK(qm.open(q4).ok() && qr.parse(qm).ok());
  const std::uint32_t L = tm.spec.n_layer, E = tm.spec.n_expert;
  const std::uint32_t cyc[3] = {static_cast<std::uint32_t>(q::Type::q4_tg2), static_cast<std::uint32_t>(q::Type::q5_k), static_cast<std::uint32_t>(q::Type::q6_k)};
  std::map<std::string, std::vector<std::uint32_t>> mixed, other;
  for (std::uint32_t l = 0; l < L; ++l)
    for (const char* k : {"gate", "up", "down"}) {
      const std::string nm = "blk." + std::to_string(l) + ".ffn_" + k + "_exps.weight";
      for (std::uint32_t e = 0; e < E; ++e) {
        mixed[nm].push_back(cyc[(e + l) % 3]);
        other[nm].push_back(cyc[(e + l + 1) % 3]);
      }
    }
  const std::string tf = (dir / "tiny_tier.gguf").string(), tf2 = (dir / "tiny_tier_other.gguf").string();
  pelias::io::TierBuildOptions to;
  to.threads = 2;
  CHECK(pelias::io::build_tiered(qm, qr, tm.map, tm.reader, mixed, tf, to).ok());
  CHECK(pelias::io::build_tiered(qm, qr, tm.map, tm.reader, other, tf2, to).ok());
  pelias::os::FileMap tmap, tmap2;
  pelias::io::GgufReader tr, tr2;
  CHECK(tmap.open(tf).ok() && tr.parse(tmap).ok() && tmap2.open(tf2).ok() && tr2.parse(tmap2).ok());
  auto tw = pelias::model::bind_qwen35moe(tr, tmap.data(), tm.spec);
  auto tw2 = pelias::model::bind_qwen35moe(tr2, tmap2.data(), tm.spec);
  CHECK(tw.ok() && tw2.ok());
  if (!tw.ok() || !tw2.ok()) return;
  const std::vector<std::int32_t> tok{11, 427, 3, 999, 58, 58, 0, 731, 210, 64, 5, 300, 17};
  const int n = static_cast<int>(tok.size()), v = static_cast<int>(tm.spec.n_vocab);
  auto logits = [&](const pelias::model::Qwen35MoeWeights& w, int picks, bool fast = false, bool int8 = false) {
    pelias::model::ForwardOptions o;
    o.threads = 2;
    o.plane_picks = picks;
    o.fast_cpu = fast;
    if (int8) {  // the engine's activation format, every quantized weight (--act-int8 all --act-int8-path gpu256x8)
      o.act_int8 = {"all"};
      o.act_int8_gpu = true;
      o.act_gpu256 = true;
      o.act_gpu256_sub = 1;
    }
    std::vector<float> lg(static_cast<std::size_t>(n) * static_cast<std::size_t>(v));
    CHECK(pelias::model::forward_reference(tm.spec, w, tok.data(), n, 0, n, lg.data(), o).ok());
    return lg;
  };
  auto mse = [&](const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) e += static_cast<double>(a[i] - b[i]) * (a[i] - b[i]);
    return e / static_cast<double>(a.size());
  };
  auto planes = [&](const std::string& name, pelias::io::PlaneBuildOptions po, pelias::io::PlaneBuildReport* rep_out) {
    const std::string pf = (dir / name).string();
    po.threads = 2;
    const auto rep = pelias::io::build_planes(tmap, tr, tm.map, tm.reader, pf, po);
    CHECK(rep.ok());
    if (!rep.ok()) std::printf("      %s\n", rep.status.text().c_str());
    if (rep.ok() && rep_out) *rep_out = rep.value;
    return pf;
  };
  struct Attached { pelias::os::FileMap m; pelias::io::GgufReader g; pelias::model::Qwen35MoeWeights w; bool ok = false; };
  auto attach = [&](const std::string& pf, const pelias::model::Qwen35MoeWeights& base, Attached& at) {
    at.ok = at.m.load(pf).ok() && at.g.parse(at.m).ok();
    at.w = base;
    if (at.ok) at.ok = pelias::model::attach_planes(at.g, at.m.data(), at.w, 0).ok();
    return at.ok;
  };
  const std::vector<float> ref = logits(tm.w, 0), l0 = logits(tw.value, 0);
  // (1) one plane on everything, the tiered experts fitted expert by expert
  pelias::io::PlaneBuildOptions full;
  full.k = 1;
  pelias::io::PlaneBuildReport r1;
  const std::string p1 = planes("full_k1.gguf", full, &r1);
  CHECK_EQ(r1.tiered, static_cast<std::size_t>(L * 3));
  Attached a1;
  CHECK(attach(p1, tw.value, a1));
  if (!a1.ok) return;
  const std::vector<float> l1 = logits(a1.w, 0);
  // (2) block selection: every block = the full-row planes exactly; half the budget = that many blocks, between
  std::uint64_t exp_blocks = 0;
  for (const auto& [nm, ty] : mixed) {
    const auto* t = tm.reader.find_tensor(nm);
    exp_blocks += t->ne[1] * t->ne[2] * (t->ne[0] / 256);
  }
  pelias::io::PlaneBuildOptions all = full, half = full;
  all.select = "_exps";
  all.budget_bytes = exp_blocks * 34;
  half.select = "_exps";
  half.budget_bytes = (exp_blocks / 2) * 34;
  pelias::io::PlaneBuildReport r2, r3;
  const std::string p2 = planes("sel_all.gguf", all, &r2), p3 = planes("sel_half.gguf", half, &r3);
  CHECK_EQ(r2.blocks_selected, exp_blocks);
  CHECK_EQ(r3.blocks_selected, exp_blocks / 2);
  Attached a2, a3;
  CHECK(attach(p2, tw.value, a2) && attach(p3, tw.value, a3));
  if (!a2.ok || !a3.ok) return;
  CHECK(a2.w.layers[0].gate_exps.plane_mask != nullptr);
  const std::vector<float> l2 = logits(a2.w, 0), l3 = logits(a3.w, 0);
  CHECK(l2 == l1);
  const double e0 = mse(l0, ref), e1 = mse(l1, ref), e3 = mse(l3, ref);
  std::printf("    logit error vs the original: tiered %.4g; + half the expert blocks %.4g; + one plane everywhere %.4g\n", e0, e3, e1);
  CHECK(e1 < e0 && e1 < e3 && e3 < e0);
  CHECK(r3.select_err_after < r3.select_err_before);
  // (3) per-tensor plane counts
  pelias::io::PlaneBuildOptions rules = full;
  rules.k_rules = {{"token_embd", 3}, {"_exps", 1}, {"", 0}};  // the tiny model ties its output layer to the embedding
  const std::string p4 = planes("rules.gguf", rules, nullptr);
  Attached a4;
  CHECK(attach(p4, tw.value, a4));
  if (!a4.ok) return;
  CHECK(a4.w.tok_embd.plane_k_stored == 3 && a4.w.output.plane_k_stored == 3 && a4.w.layers[1].down_exps.plane_k_stored == 1 &&
        a4.w.layers[1].down_shexp.planes == nullptr);
  // (4) per-pick reading: as many picks as the model uses = every pick; one pick lands between none and every
  const int nu = static_cast<int>(tm.spec.n_expert_used);
  CHECK(logits(a1.w, nu) == l1);
  const double ep = mse(logits(a1.w, 1), ref);
  std::printf("    per-pick planes (tiny model, %d picks): every pick %.4g, top 1 %.4g\n", nu, e1, ep);
  CHECK(ep > e1 && ep < e0);
  // (5) the fast CPU kernels (GP3-10): the same products, another order of the sum; tiered types, dense and
  // block-selective planes, per-tensor plane counts and per-pick reading all agree with the reference forward pass far
  // below the quantization error
  const std::vector<float> f0 = logits(tw.value, 0, true), f1 = logits(a1.w, 0, true), f3 = logits(a3.w, 0, true),
                           f4 = logits(a4.w, 0, true), fp = logits(a1.w, 1, true), fr = logits(tm.w, 0, true);
  const double d0 = mse(f0, l0), d1 = mse(f1, l1), d3 = mse(f3, l3), d4 = mse(f4, logits(a4.w, 0)), dp = mse(fp, logits(a1.w, 1)),
               dr = mse(fr, ref);
  std::printf("    fast CPU kernels vs the reference (logit mse): tiered %.3g, + planes %.3g, + half blocks %.3g, rules %.3g, top 1 %.3g, original %.3g\n",
              d0, d1, d3, d4, dp, dr);
  for (const double d : {d0, d1, d3, d4, dp}) CHECK(d < 1e-4 * e1);
  CHECK(dr < 1e-4 * e1);
  // the int8-activation path: the same rounded inputs as the reference with gpu256x8 rounding, integer products
  const double i0 = mse(logits(tw.value, 0, true, true), logits(tw.value, 0, false, true)),
               i1 = mse(logits(a1.w, 0, true, true), logits(a1.w, 0, false, true)), i3 = mse(logits(a3.w, 1, true, true), logits(a3.w, 1, false, true));
  std::printf("    int8-activation path vs the reference with the same rounding (logit mse): tiered %.3g, + planes %.3g, half blocks top 1 %.3g\n", i0, i1, i3);
  // Per row the integer path is within 2e-7 of fp64 (tests/unit/test_cpu_dot.cpp); through the model, a value that lands on
  // a rounding boundary of the 8-bit activations can round one step the other way when an earlier sum differs in its last
  // bit, so a position can differ by one activation step. That stays far below the weights' own error.
  for (const double d : {i0, i1, i3}) CHECK(d < 2e-2 * e1);
  // (6) the same planes on another allocation are refused
  Attached a5;
  CHECK(!attach(p1, tw2.value, a5));
}


// GP3-08: the MTP block. Bound from the file; one call over n tokens equals n one-token calls; greedy decoding with drafting
// gives exactly the tokens of plain greedy decoding for every draft length, on the reference and the fast CPU path; the
// drafts are counted and, on a block built to echo the trunk, accepted; a file without the block refuses to draft.
namespace {
std::vector<std::int32_t> plain_greedy(const TinyModel& tm, const std::vector<std::int32_t>& prompt, int n, std::int32_t eos,
                                       const pelias::model::ForwardOptions& o) {
  const int v = static_cast<int>(tm.spec.n_vocab);
  pelias::model::DecodeState st;
  pelias::model::decode_state_reset(tm.spec, st);
  std::vector<float> lg(static_cast<std::size_t>(v));
  std::vector<std::int32_t> out;
  if (!pelias::model::forward_decode(tm.spec, tm.w, st, prompt.data(), static_cast<int>(prompt.size()), lg.data(), true, o).ok()) return out;
  while (static_cast<int>(out.size()) < n) {
    int best = 0;
    for (int i = 1; i < v; ++i)
      if (lg[static_cast<std::size_t>(i)] > lg[static_cast<std::size_t>(best)]) best = i;
    out.push_back(best);
    if (best == eos || static_cast<int>(out.size()) == n) break;
    if (!pelias::model::forward_decode(tm.spec, tm.w, st, &out.back(), 1, lg.data(), true, o).ok()) return {};
  }
  return out;
}
}  // namespace

TEST(mtp_block_drafts_and_greedy_equals_plain) {
  for (int echo = 0; echo < 2; ++echo) {
    TinyModel tm;
    tiny_model(tm, true, echo == 1);
    CHECK(tm.ok);
    if (!tm.ok) return;
    CHECK(tm.w.mtp.present);
    CHECK(tm.spec.n_layer == 4 && tm.spec.n_layer_all == 5);
    const std::size_t e = tm.spec.n_embd, v = tm.spec.n_vocab;
    const auto pe = static_cast<std::ptrdiff_t>(e);
    pelias::model::ForwardOptions o;
    o.threads = 2;
    const std::vector<std::int32_t> tok{11, 427, 3, 999, 58, 58, 0, 731, 210, 64, 5};
    const int n = static_cast<int>(tok.size());
    // the trunk's hidden rows, then the block over all tokens at once and one at a time
    std::vector<float> lg(static_cast<std::size_t>(n) * v), hid(static_cast<std::size_t>(n) * e), hp(static_cast<std::size_t>(n) * e, 0.0f);
    pelias::model::ForwardOptions oh = o;
    oh.hidden_out = hid.data();
    CHECK(pelias::model::forward_reference(tm.spec, tm.w, tok.data(), n, 0, n, lg.data(), oh).ok());
    std::copy(hid.begin(), hid.end() - pe, hp.begin() + pe);
    pelias::model::MtpState a, b;
    pelias::model::mtp_state_reset(a);
    pelias::model::mtp_state_reset(b);
    std::vector<float> la(static_cast<std::size_t>(n) * v), lb(la.size()), ha(hid.size()), hb(hid.size());
    CHECK(pelias::model::mtp_forward(tm.spec, tm.w, a, tok.data(), hp.data(), n, la.data(), ha.data(), o).ok());
    for (int t = 0; t < n; ++t)
      CHECK(pelias::model::mtp_forward(tm.spec, tm.w, b, tok.data() + t, hp.data() + static_cast<std::size_t>(t) * e, 1,
                                       lb.data() + static_cast<std::size_t>(t) * v, hb.data() + static_cast<std::size_t>(t) * e, o).ok());
    CHECK(a.pos == n && b.pos == n);
    CHECK(la == lb);
    CHECK(ha == hb);
    bool finite = true;
    for (float x : la) finite = finite && std::isfinite(x);
    CHECK(finite);
    // truncation drops cached positions: run the last 3 again from position n - 3 and get the same rows
    pelias::model::mtp_state_truncate(tm.spec, b, n - 3);
    std::vector<float> lc(3 * v);
    CHECK(pelias::model::mtp_forward(tm.spec, tm.w, b, tok.data() + n - 3, hp.data() + static_cast<std::size_t>(n - 3) * e, 3, lc.data(), nullptr, o).ok());
    CHECK(std::equal(lc.begin(), lc.end(), la.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(n - 3) * v)));
    // greedy with drafting = plain greedy, for every draft length, reference and fast path, with and without an end token
    for (int fast = 0; fast < 2; ++fast) {
      pelias::model::ForwardOptions of = o;
      of.fast_cpu = fast == 1;
      for (std::int32_t eos : {-1, 5}) {
        const std::vector<std::int32_t> want = plain_greedy(tm, tok, 24, eos, of);
        CHECK(!want.empty());
        for (int nd : {0, 1, 2, 4}) {
          std::vector<std::int32_t> got;
          pelias::model::MtpStats ms;
          CHECK(pelias::model::generate_greedy_mtp(tm.spec, tm.w, tok, 24, nd, eos >= 0 ? std::vector<std::int32_t>{eos} : std::vector<std::int32_t>{}, got, &ms, of).ok());
          CHECK(got == want);
          if (got != want) {
            std::printf("    mtp: echo %d fast %d eos %d draft %d differs from plain greedy\n", echo, fast, eos, nd);
          }
          if (fast == 0 && eos == -1 && nd == 4)
            std::printf("    mtp (%s block): 24 tokens, %lld steps, %lld drafted, %lld accepted\n", echo ? "echo" : "random",
                        static_cast<long long>(ms.steps), static_cast<long long>(ms.drafted), static_cast<long long>(ms.accepted));
          if (echo == 1 && fast == 0 && eos == -1 && nd > 0) CHECK(ms.accepted > 0);
          if (nd == 0) CHECK(ms.drafted == 0);
        }
      }
    }
  }
  // a file without the block refuses to draft, and still decodes plainly
  TinyModel plain;
  tiny_model(plain);
  CHECK(plain.ok && !plain.w.mtp.present);
  std::vector<std::int32_t> got;
  const std::vector<std::int32_t> tok{11, 427, 3};
  CHECK(!pelias::model::generate_greedy_mtp(plain.spec, plain.w, tok, 8, 2, {}, got, nullptr).ok());
  CHECK(pelias::model::generate_greedy_mtp(plain.spec, plain.w, tok, 8, 0, {}, got, nullptr).ok() && got.size() == 8);
}

// Newer converters fuse tensors: a full-attention layer's q/k/v into one attn_qkv ([query | gate] per head, then keys, then
// values) and each layer's expert gate and up into one ffn_gate_up_exps (per expert, its gate rows then its up rows). A
// copy of the tiny model written that way binds and gives the same logits, bit for bit, on both CPU paths.
TEST(fused_qkv_and_gate_up_files_read_the_same) {
  TinyModel tm;
  tiny_model(tm);
  CHECK(tm.ok);
  if (!tm.ok) return;
  ptest::GgufBuilder b = qwen35moe_meta(4, 0);
  auto is_split = [](const std::string& nm) {
    return nm.find("ffn_gate_exps") != std::string::npos || nm.find("ffn_up_exps") != std::string::npos ||
           (nm.rfind("blk.3.", 0) == 0 && (nm.find("attn_q.weight") != std::string::npos || nm.find("attn_k.weight") != std::string::npos ||
                                           nm.find("attn_v.weight") != std::string::npos));
  };
  for (const pelias::io::GgufTensor& t : tm.reader.tensors()) {
    if (t.name == "token_embd.weight" || is_split(t.name)) continue;  // the meta builder adds the embedding table
    std::vector<std::uint64_t> ne(t.ne, t.ne + t.n_dims);
    b.tensor(t.name, ne, t.type);
  }
  const auto* q = tm.reader.find_tensor("blk.3.attn_q.weight");
  const auto* k = tm.reader.find_tensor("blk.3.attn_k.weight");
  const auto* v = tm.reader.find_tensor("blk.3.attn_v.weight");
  b.tensor("blk.3.attn_qkv.weight", {q->ne[0], q->ne[1] + k->ne[1] + v->ne[1]}, 0);
  for (int il = 0; il < 4; ++il) {
    const auto* g = tm.reader.find_tensor("blk." + std::to_string(il) + ".ffn_gate_exps.weight");
    b.tensor("blk." + std::to_string(il) + ".ffn_gate_up_exps.weight", {g->ne[0], 2 * g->ne[1], g->ne[2]}, 0);
  }
  std::vector<std::uint8_t> bytes = b.build();
  pelias::os::FileMap m0;
  pelias::io::GgufReader r0;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/decode_tiny_fused_layout.gguf", bytes, m0) && r0.parse(m0).ok());
  auto src = [&](const std::string& nm) { return tm.map.data() + tm.reader.find_tensor(nm)->file_offset; };
  for (const pelias::io::GgufTensor& t : r0.tensors()) {
    std::uint8_t* d = bytes.data() + t.file_offset;
    if (t.name == "blk.3.attn_qkv.weight") {
      std::memcpy(d, src("blk.3.attn_q.weight"), q->bytes);
      std::memcpy(d + q->bytes, src("blk.3.attn_k.weight"), k->bytes);
      std::memcpy(d + q->bytes + k->bytes, src("blk.3.attn_v.weight"), v->bytes);
    } else if (t.name.find("ffn_gate_up_exps") != std::string::npos) {
      const std::string p = t.name.substr(0, t.name.find("ffn_gate_up_exps"));
      const auto* g = tm.reader.find_tensor(p + "ffn_gate_exps.weight");
      const std::uint64_t per = g->bytes / g->ne[2];  // one expert's gate (or up) bytes
      for (std::uint64_t ex = 0; ex < g->ne[2]; ++ex) {
        std::memcpy(d + 2 * ex * per, src(p + "ffn_gate_exps.weight") + ex * per, per);
        std::memcpy(d + (2 * ex + 1) * per, src(p + "ffn_up_exps.weight") + ex * per, per);
      }
    } else {
      std::memcpy(d, src(t.name), t.bytes);
    }
  }
  pelias::os::FileMap fm;
  pelias::io::GgufReader fr;
  CHECK(ptest::write_and_map(ptest::data_dir() + "/decode_tiny_fused.gguf", bytes, fm) && fr.parse(fm).ok());
  auto fw = pelias::model::bind_qwen35moe(fr, fm.data(), tm.spec);
  CHECK(fw.ok());
  if (!fw.ok()) {
    std::printf("      %s\n", fw.status.text().c_str());
    return;
  }
  const std::vector<std::int32_t> tok{11, 427, 3, 999, 58, 58, 0, 731, 210, 64, 5};
  const int n = static_cast<int>(tok.size());
  const std::size_t nv = tm.spec.n_vocab;
  for (int fast = 0; fast < 2; ++fast) {
    pelias::model::ForwardOptions o;
    o.threads = 2;
    o.fast_cpu = fast == 1;
    std::vector<float> a(static_cast<std::size_t>(n) * nv), c(a.size());
    CHECK(pelias::model::forward_reference(tm.spec, tm.w, tok.data(), n, 0, n, a.data(), o).ok());
    CHECK(pelias::model::forward_reference(tm.spec, fw.value, tok.data(), n, 0, n, c.data(), o).ok());
    CHECK(a == c);
  }
}

// GP3-09: the dense families (qwen3: q/k norms, tied output; qwen2: q/k/v biases; llama: adjacent-pair rope and
// rope_freqs) on tiny F32 files with small deterministic values: the spec reads, the weights bind, the logits are finite;
// the fast CPU path agrees with the reference within the rung-1 tolerance; the decode path one token at a time equals
// the fresh pass (fast path: bit for bit). The values against the reference library are checked by
// tools/oracle/logits_oracle.cpp + make_tiny_dense.py (sandbox, test-only).
namespace {
bool dense_model(const std::string& fam, bool rope_freqs, TinyModel& tm) {
  const std::uint32_t E = 128, H = 4, HKV = 2, HD = 32, FF = 192, L = 2, V = 300;
  ptest::GgufBuilder b;
  const std::string a = fam + ".";
  b.kv_string("general.architecture", fam)
      .kv_u32(a + "block_count", L)
      .kv_u32(a + "embedding_length", E)
      .kv_u32(a + "feed_forward_length", FF)
      .kv_u32(a + "attention.head_count", H)
      .kv_u32(a + "attention.head_count_kv", HKV)
      .kv_f32(a + "rope.freq_base", 500000.0f)
      .kv_f32(a + "attention.layer_norm_rms_epsilon", 1e-6f)
      .tensor("token_embd.weight", {E, V}, 0)
      .tensor("output_norm.weight", {E}, 0);
  if (fam != "qwen3") b.tensor("output.weight", {E, V}, 0);
  if (rope_freqs) b.tensor("rope_freqs.weight", {HD / 2}, 0);
  for (std::uint32_t il = 0; il < L; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    b.tensor(p + "attn_norm.weight", {E}, 0).tensor(p + "attn_q.weight", {E, H * HD}, 0).tensor(p + "attn_k.weight", {E, HKV * HD}, 0);
    b.tensor(p + "attn_v.weight", {E, HKV * HD}, 0).tensor(p + "attn_output.weight", {H * HD, E}, 0).tensor(p + "ffn_norm.weight", {E}, 0);
    b.tensor(p + "ffn_gate.weight", {E, FF}, 0).tensor(p + "ffn_up.weight", {E, FF}, 0).tensor(p + "ffn_down.weight", {FF, E}, 0);
    if (fam == "qwen3") b.tensor(p + "attn_q_norm.weight", {HD}, 0).tensor(p + "attn_k_norm.weight", {HD}, 0);
    if (fam == "qwen2") b.tensor(p + "attn_q.bias", {H * HD}, 0).tensor(p + "attn_k.bias", {HKV * HD}, 0).tensor(p + "attn_v.bias", {HKV * HD}, 0);
  }
  std::vector<std::uint8_t> bytes = b.build();
  pelias::os::FileMap m0;
  pelias::io::GgufReader r0;
  const std::string stem = ptest::data_dir() + "/dense_" + fam + (rope_freqs ? "_rf" : "");
  if (!ptest::write_and_map(stem + "_layout.gguf", bytes, m0) || !r0.parse(m0).ok()) return false;
  std::uint32_t h = 20261030u;
  auto unit = [&h]() {
    h ^= h << 13;
    h ^= h >> 17;
    h ^= h << 5;
    return static_cast<float>(h) / 2147483648.0f - 1.0f;
  };
  for (const pelias::io::GgufTensor& t : r0.tensors()) {
    float* d = reinterpret_cast<float*>(bytes.data() + t.file_offset);
    const float row = 1.0f / std::sqrt(static_cast<float>(t.ne[0]));
    for (std::uint64_t i = 0; i < t.elements(); ++i) {
      const float u = unit();
      float v = u * row;
      if (t.name.find("norm") != std::string::npos) v = 1.0f + 0.1f * u;
      else if (t.name.find(".bias") != std::string::npos) v = 0.3f * u;
      else if (t.name == "rope_freqs.weight") v = 1.0f + 3.5f * (u + 1.0f);
      else if (t.name == "token_embd.weight") v = 0.5f * u;
      std::memcpy(d + i, &v, 4);
    }
  }
  if (!ptest::write_and_map(stem + ".gguf", bytes, tm.map) || !tm.reader.parse(tm.map).ok()) return false;
  auto s = pelias::model::read_qwen35moe_spec(tm.reader);
  if (!s.ok()) return false;
  tm.spec = s.value;
  auto w = pelias::model::bind_qwen35moe(tm.reader, tm.map.data(), tm.spec);
  if (!w.ok()) return false;
  tm.w = std::move(w.value);
  tm.ok = true;
  return true;
}
}  // namespace

TEST(dense_families_run_fast_and_decode_paths) {
  struct F {
    const char* fam;
    bool rf;
  };
  for (const F f : {F{"qwen3", false}, F{"qwen2", false}, F{"llama", false}, F{"llama", true}}) {
    TinyModel tm;
    CHECK(dense_model(f.fam, f.rf, tm));
    if (!tm.ok) continue;
    CHECK(tm.spec.dense);
    CHECK(tm.spec.rope_neox == (std::string(f.fam) != "llama"));
    CHECK(tm.w.rope_freqs.empty() == !f.rf);
    CHECK((tm.w.output.data == tm.w.tok_embd.data) == (std::string(f.fam) == "qwen3"));
    const std::vector<std::int32_t> tok{11, 27, 3, 299, 58, 58, 0, 131, 210, 64, 5};
    const int n = static_cast<int>(tok.size()), v = static_cast<int>(tm.spec.n_vocab);
    pelias::model::ForwardOptions ref, fast;
    ref.threads = fast.threads = 3;
    fast.fast_cpu = true;
    std::vector<float> a(static_cast<std::size_t>(n) * static_cast<std::size_t>(v)), bfast(a.size()), dec(a.size());
    CHECK(pelias::model::forward_reference(tm.spec, tm.w, tok.data(), n, 0, n, a.data(), ref).ok());
    CHECK(pelias::model::forward_reference(tm.spec, tm.w, tok.data(), n, 0, n, bfast.data(), fast).ok());
    pelias::model::DecodeState st;
    pelias::model::decode_state_reset(tm.spec, st);
    for (int i = 0; i < n; ++i)
      CHECK(pelias::model::forward_decode(tm.spec, tm.w, st, tok.data() + i, 1, dec.data() + static_cast<std::ptrdiff_t>(i) * v, true, fast).ok());
    float amax = 0.0f;
    double d_fast = 0.0, d_dec = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < a.size(); ++i) {
      finite = finite && std::isfinite(a[i]);
      amax = std::max(amax, std::fabs(a[i]));
      d_fast = std::max(d_fast, static_cast<double>(std::fabs(a[i] - bfast[i])));
      d_dec = std::max(d_dec, static_cast<double>(std::fabs(bfast[i] - dec[i])));
    }
    std::printf("    %s%s: max |logit| %.3g, fast vs reference %.3g, decode vs batch (fast) %.3g\n", f.fam, f.rf ? " + rope_freqs" : "",
                static_cast<double>(amax), d_fast, d_dec);
    CHECK(finite);
    CHECK(amax > 1e-3f);
    CHECK(d_fast <= 1e-5 * static_cast<double>(amax));
    CHECK(d_dec <= 1e-5 * static_cast<double>(amax));
    pelias::model::ForwardOptions bad = ref;
    bad.defer_experts = 1;  // expert options are refused on a dense model
    CHECK(!pelias::model::forward_reference(tm.spec, tm.w, tok.data(), n, 0, n, a.data(), bad).ok());
  }
}

// BitNet b1.58 (pre-build list item 5): a tiny bitnet-b1.58 file whose projections are I2_S (ternary codes + one fp32
// scale per tensor) and the same model with those weights written out as F32 (t x scale): the two give the same logits
// within the rung-1 tolerance (the I2_S decode, its code mapping and the per-tensor scale), the fast path agrees with the
// reference, and the decode path equals the batch. relu^2 FFN, sub-norms and the per-token int8 BitLinear inputs run in
// both. The code mapping itself was MEASURED against the real model's BF16 original (tools/oracle/i2s_check.py).
namespace {
bool bitnet_model(bool i2s, TinyModel& tm) {
  const std::uint32_t E = 256, H = 4, HKV = 2, HD = 64, FF = 384, L = 2, V = 300;
  const std::uint32_t wt = i2s ? 36u : 0u;
  ptest::GgufBuilder b;
  const std::string a = "bitnet-b1.58.";
  b.kv_string("general.architecture", "bitnet-b1.58")
      .kv_u32(a + "block_count", L)
      .kv_u32(a + "embedding_length", E)
      .kv_u32(a + "feed_forward_length", FF)
      .kv_u32(a + "attention.head_count", H)
      .kv_u32(a + "attention.head_count_kv", HKV)
      .kv_f32(a + "rope.freq_base", 500000.0f)
      .kv_f32(a + "attention.layer_norm_rms_epsilon", 1e-5f)
      .tensor("token_embd.weight", {E, V}, 0)
      .tensor("output_norm.weight", {E}, 0);
  for (std::uint32_t il = 0; il < L; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    b.tensor(p + "attn_norm.weight", {E}, 0).tensor(p + "attn_q.weight", {E, H * HD}, wt).tensor(p + "attn_k.weight", {E, HKV * HD}, wt);
    b.tensor(p + "attn_v.weight", {E, HKV * HD}, wt).tensor(p + "attn_output.weight", {H * HD, E}, wt).tensor(p + "attn_sub_norm.weight", {E}, 0);
    b.tensor(p + "ffn_norm.weight", {E}, 0).tensor(p + "ffn_gate.weight", {E, FF}, wt).tensor(p + "ffn_up.weight", {E, FF}, wt);
    b.tensor(p + "ffn_down.weight", {FF, E}, wt).tensor(p + "ffn_sub_norm.weight", {FF}, 0);
  }
  std::vector<std::uint8_t> bytes = b.build();
  pelias::os::FileMap m0;
  pelias::io::GgufReader r0;
  const std::string stem = ptest::data_dir() + (i2s ? "/bitnet_i2s" : "/bitnet_f32");
  if (!ptest::write_and_map(stem + "_layout.gguf", bytes, m0) || !r0.parse(m0).ok()) return false;
  std::uint32_t h = 20261030u;
  auto unit = [&h]() {
    h ^= h << 13;
    h ^= h >> 17;
    h ^= h << 5;
    return static_cast<float>(h) / 2147483648.0f - 1.0f;
  };
  for (const pelias::io::GgufTensor& t : r0.tensors()) {
    std::uint8_t* d = bytes.data() + t.file_offset;
    const bool proj = t.name.find("attn_q.") != std::string::npos || t.name.find("attn_k.") != std::string::npos ||
                      t.name.find("attn_v.") != std::string::npos || t.name.find("attn_output.") != std::string::npos ||
                      t.name.find("ffn_gate.") != std::string::npos || t.name.find("ffn_up.") != std::string::npos ||
                      t.name.find("ffn_down.") != std::string::npos;
    if (proj) {  // ternary codes from the same stream for both files; scale 1/sqrt(row) per tensor
      const float scale = 1.5f / std::sqrt(static_cast<float>(t.ne[0]));
      const std::uint64_t n = t.elements();
      std::vector<std::uint8_t> codes(n);
      for (std::uint64_t i = 0; i < n; ++i) {
        const float u = unit();
        codes[i] = static_cast<std::uint8_t>(u < -0.33f ? 0 : (u < 0.33f ? 1 : 2));  // 0 = -1, 1 = 0, 2 = +1
      }
      if (i2s) {
        for (std::uint64_t blk = 0; blk < n / 128; ++blk)
          for (int j = 0; j < 32; ++j) {
            std::uint8_t by = 0;
            for (int g = 0; g < 4; ++g) by = static_cast<std::uint8_t>(by | (codes[blk * 128 + 32 * static_cast<std::uint64_t>(g) + static_cast<std::uint64_t>(j)] << (6 - 2 * g)));
            d[blk * 32 + static_cast<std::uint64_t>(j)] = by;
          }
        std::memcpy(d + n / 4, &scale, 4);
      } else {
        for (std::uint64_t i = 0; i < n; ++i) {
          const float v = static_cast<float>(static_cast<int>(codes[i]) - 1) * scale;
          std::memcpy(d + 4 * i, &v, 4);
        }
      }
      continue;
    }
    float* f = reinterpret_cast<float*>(d);
    for (std::uint64_t i = 0; i < t.elements(); ++i) {
      const float u = unit();
      const float v = t.name.find("norm") != std::string::npos ? 1.0f + 0.1f * u : 0.5f * u;
      std::memcpy(f + i, &v, 4);
    }
  }
  if (!ptest::write_and_map(stem + ".gguf", bytes, tm.map) || !tm.reader.parse(tm.map).ok()) return false;
  auto s = pelias::model::read_qwen35moe_spec(tm.reader);
  if (!s.ok()) return false;
  tm.spec = s.value;
  auto w = pelias::model::bind_qwen35moe(tm.reader, tm.map.data(), tm.spec);
  if (!w.ok()) return false;
  tm.w = std::move(w.value);
  tm.ok = true;
  return true;
}
}  // namespace

TEST(bitnet_i2s_equals_its_f32_copy_and_runs_every_path) {
  TinyModel a, f;
  CHECK(bitnet_model(true, a));
  CHECK(bitnet_model(false, f));
  if (!a.ok || !f.ok) return;
  CHECK(a.spec.family == pelias::model::Family::bitnet && a.spec.ffn_relu2 && a.spec.sub_norms && a.spec.bitlinear_act8 && a.spec.rope_neox);
  CHECK(a.w.layers[0].wq.type == 36u && a.w.layers[0].wq.tensor_scale > 0.0f && a.w.layers[0].wq.bitlinear);
  CHECK(a.w.output.data == a.w.tok_embd.data);
  const std::vector<std::int32_t> tok{11, 27, 3, 299, 58, 58, 0, 131, 210, 64, 5};
  const int n = static_cast<int>(tok.size()), v = static_cast<int>(a.spec.n_vocab);
  pelias::model::ForwardOptions ref, fast;
  ref.threads = fast.threads = 3;
  fast.fast_cpu = true;
  std::vector<float> la(static_cast<std::size_t>(n) * static_cast<std::size_t>(v)), lf(la.size()), lfast(la.size()), dec(la.size());
  CHECK(pelias::model::forward_reference(a.spec, a.w, tok.data(), n, 0, n, la.data(), ref).ok());
  CHECK(pelias::model::forward_reference(f.spec, f.w, tok.data(), n, 0, n, lf.data(), ref).ok());
  CHECK(pelias::model::forward_reference(a.spec, a.w, tok.data(), n, 0, n, lfast.data(), fast).ok());
  pelias::model::DecodeState st;
  pelias::model::decode_state_reset(a.spec, st);
  for (int i = 0; i < n; ++i)
    CHECK(pelias::model::forward_decode(a.spec, a.w, st, tok.data() + i, 1, dec.data() + static_cast<std::ptrdiff_t>(i) * v, true, fast).ok());
  float amax = 0.0f;
  double d_f32 = 0.0, d_fast = 0.0, d_dec = 0.0;
  for (std::size_t i = 0; i < la.size(); ++i) {
    amax = std::max(amax, std::fabs(la[i]));
    d_f32 = std::max(d_f32, static_cast<double>(std::fabs(la[i] - lf[i])));
    d_fast = std::max(d_fast, static_cast<double>(std::fabs(la[i] - lfast[i])));
    d_dec = std::max(d_dec, static_cast<double>(std::fabs(lfast[i] - dec[i])));
  }
  std::printf("    bitnet I2_S: max |logit| %.3g, vs its F32 copy %.3g, fast vs reference %.3g, decode vs batch (fast) %.3g\n", static_cast<double>(amax),
              d_f32, d_fast, d_dec);
  CHECK(amax > 1e-3f);
  CHECK(d_f32 <= 1e-4 * static_cast<double>(amax));
  CHECK(d_fast <= 1e-4 * static_cast<double>(amax));
  CHECK(d_dec <= 1e-5 * static_cast<double>(amax));
}
