// provenance: original (the base-record encoding used to build test input READ-SRC from llama.cpp
//   tools/perplexity/perplexity.cpp log_softmax at commit 2ca15f5404760548c39e7b92bd43116a09414a1a)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf_builder.h"
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
