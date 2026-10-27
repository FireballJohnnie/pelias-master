// provenance: original (the expected names, size labels and titles MEASURED in the sandbox 2026-10-26 by running the
//   reference's own gguf-py metadata.py and utility.py at commit 2ca15f5 on the same inputs)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include <cmath>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "ingest/hf_metadata.h"
#include "ingest/stage_a.h"
#include "io/gguf_compare.h"
#include "io/gguf_reader.h"
#include "io/requantize.h"
#include "io/sha256.h"
#include "os/file_map.h"
#include "pelias_test.h"
#include "quant/block_types.h"
#include "quant/quant_cpu.h"
#include "unit/synth_hf.h"

using namespace pelias;

namespace {

std::string opt(const std::optional<std::string>& s) { return s ? *s : std::string("None"); }

std::string parts_text(const std::string& id, std::int64_t total) {
  const auto r = ingest::model_id_components(id, total);
  if (!r.ok()) return "error";
  const auto& p = r.value;
  return opt(p.full_name) + "|" + opt(p.org) + "|" + opt(p.basename) + "|" + opt(p.finetune) + "|" + opt(p.version) + "|" + opt(p.size_label);
}

std::string kv_key(const io::GgufKvBlob& b) { return io::gguf_kv_key(b); }

std::string file_sha(const std::string& path) {
  os::FileMap m;
  if (!m.open(path)) return "unreadable";
  return io::Sha256::hex_of(m.data(), static_cast<std::size_t>(m.size()));
}

}  // namespace

TEST(ingest_names_follow_the_reference) {
  // (id, total params) -> full name | org | basename | finetune | version | size label, from the reference's code
  struct C {
    const char* id;
    std::int64_t total;
    const char* want;
  };
  const C cases[] = {
      {"2ab40a9acc6d567889ca4d4e59feb2da56121454", 0, "2ab40a9acc6d567889ca4d4e59feb2da56121454|None|None|2ab40a9acc6d567889ca4d4e59feb2da56121454|None|None"},
      {"Qwen3.6-35B-A3B", 0, "Qwen3.6-35B-A3B|None|Qwen3.6|None|None|35B-A3B"},
      {"Qwen/Qwen3.6-35B-A3B", 35951000000, "Qwen3.6-35B-A3B|Qwen|Qwen3.6|None|None|35B-A3B"},
      {"Merged_Bf16", 0, "Merged_Bf16|None|None|None|None|None"},
      {"Mixtral-8x7B-Instruct-v0.1", 0, "Mixtral-8x7B-Instruct-v0.1|None|Mixtral|Instruct|v0.1|8x7B"},
      {"mistralai/Mistral-7B-Instruct-v0.2", 0, "Mistral-7B-Instruct-v0.2|mistralai|Mistral|Instruct|v0.2|7B"},
      {"bloom-7b1", 0, "bloom-7b1|None|bloom|None|None|7.1B"},
      {"Meta-Llama-3.1-8B-Instruct-q4_k_m", 0, "Meta-Llama-3.1-8B-Instruct-q4_k_m|None|Meta-Llama-3.1|Instruct|None|8B"},
      {"llama-2-70b-chat-hf", 0, "llama-2-70b-chat-hf|None|llama-2|chat-hf|None|70B"},
      {"llama-2-70b-chat-hf", 35951000000, "llama-2-70b-chat-hf|None|llama-2|70b-chat-hf|None|None"},
      {"./models/x", 0, "models/x|None|None|None|None|None"},
      {"phi-3-mini-128k-instruct", 0, "phi-3-mini-128k-instruct|None|phi-3-mini|instruct|None|128K"},
      {"phi-3-mini-128k-instruct", 35951000000, "phi-3-mini-128k-instruct|None|phi-3|128k-instruct|None|mini"},
      {"gemma-2-9b-it-bf16", 0, "gemma-2-9b-it-bf16|None|gemma-2|it|None|9B"},
      {"v2-small-lora", 0, "v2-small-lora|None|v2|lora|None|small"},
      {"v2-small-lora", -1000, "v2-small-lora|None|v2|None|None|small"},
      {"Model Name With Spaces", 0, "Model Name With Spaces|None|None|None|None|None"},
      {"qwen3-30b-a3b", 0, "qwen3-30b-a3b|None|qwen3|None|None|30B-a3B"},
      {"deepseek-ai/DeepSeek-V3", 0, "DeepSeek-V3|deepseek-ai|DeepSeek|None|V3|None"},
      {"TinyLlama-1.1B-Chat-v1.0", 35951000000, "TinyLlama-1.1B-Chat-v1.0|None|TinyLlama|1.1b-Chat|v1.0|None"},
      {"SmolLM2-135M", 35951000000, "SmolLM2-135M|None|SmolLM2|135m|None|None"},
  };
  for (const C& c : cases) {
    const std::string got = parts_text(c.id, c.total);
    CHECK_EQ(got, std::string(c.want));
    if (got != c.want) std::printf("      %s: got %s\n", c.id, got.c_str());
  }
  CHECK_EQ(ingest::id_to_title("qwen3.6-35b-a3b"), std::string("Qwen3.6 35b A3B"));
  CHECK_EQ(ingest::id_to_title("v1.2"), std::string("v1.2"));
  CHECK_EQ(ingest::id_to_title("hello world-foo"), std::string("Hello World Foo"));
  CHECK_EQ(ingest::id_to_title("x86_64-bit"), std::string("X86_64 Bit"));
  CHECK_EQ(ingest::id_to_title("Qwen3.6-35B-A3B"), std::string("Qwen3.6 35B A3B"));
  const std::pair<std::int64_t, const char*> counts[] = {
      {0, "0.00K"}, {999, "1.0K"}, {1001, "1.0K"}, {999999, "1000K"}, {1000000, "1000K"}, {1500000, "1.5M"}, {2600000000LL, "2.6B"},
      {35951745024LL, "36B"}, {999500000LL, "1000M"}, {1000000001LL, "1.0B"}, {2500000000LL, "2.5B"}, {12500000000LL, "12B"}, {1250000000000LL, "1.2T"}};
  for (const auto& [n, want] : counts) CHECK_EQ(ingest::rounded_count(n, 2), std::string(want));
  ingest::ParamCount pc;
  pc.shared = 4;
  pc.expert = 6;
  pc.expert_count = 256;
  CHECK_EQ(ingest::size_label(pc), std::string("256x0.01K"));
}

TEST(ingest_model_card_subset) {
  const auto r = ingest::parse_model_card(
      "---\nlicense: apache-2.0\nbase_model:\n- Org/Model-7B\n  # comment\ntags: [a, 'b c', \"d\"]\nversion: 1.0\nlanguage:\n- en\n- no\nname: ~\n"
      "nested:\n  key: value\nquoted: 'it''s'\n---\nbody\n");
  CHECK(r.ok());
  if (!r.ok()) return;
  const auto& m = r.value;
  CHECK(m.at("license").kind == ingest::CardValue::Kind::string && m.at("license").text == "apache-2.0");
  CHECK(m.at("base_model").kind == ingest::CardValue::Kind::sequence && m.at("base_model").items.size() == 1);
  CHECK(m.at("tags").items.size() == 3 && m.at("tags").items[1].text == "b c" && m.at("tags").items[2].text == "d");
  CHECK(m.at("version").kind == ingest::CardValue::Kind::other);  // a float to the reference's YAML reader
  CHECK(m.at("language").items.size() == 2 && m.at("language").items[1].kind == ingest::CardValue::Kind::string &&
        m.at("language").items[1].text == "no");                 // the reference's "Norway" fix
  CHECK(m.at("name").kind == ingest::CardValue::Kind::null);
  CHECK(m.at("nested").kind == ingest::CardValue::Kind::other);
  CHECK(m.at("quoted").text == "it's");
  CHECK(ingest::parse_model_card("no front matter\n---\n").value.empty());
}

TEST(ingest_exp_is_the_same_everywhere) {
  // -exp(x) rounded once to f32, for every finite bf16 input below 88: one hash, checked on every platform
  io::Sha256 h;
  std::uint64_t n = 0;
  for (std::uint32_t b = 0; b < 65536; ++b) {
    const float x = quant::bf16_to_fp32(static_cast<std::uint16_t>(b));
    if (!std::isfinite(x) || x >= 88.0f) continue;
    const float y = -static_cast<float>(std::exp(static_cast<double>(x)));
    h.update(&y, 4);
    ++n;
  }
  CHECK_EQ(n, 49712u);
  const std::string got = io::Sha256::hex(h.finish());
  // MEASURED in the sandbox (glibc); a platform whose exp rounds differently fails here first
  CHECK_EQ(got, std::string("e09ffa65c60974875268189d8ee59df05b2107a4ee653ba52daefed22ddff4a4"));
  if (got != "e09ffa65c60974875268189d8ee59df05b2107a4ee653ba52daefed22ddff4a4") std::printf("      exp table sha %s\n", got.c_str());
}

TEST(ingest_converts_the_synthetic_folder) {
  const std::string dir = ptest::data_dir() + "/synth_hf/Tiny-MoE-2B-A1B";
  CHECK(ptest::write_synthetic_hf_folder(dir).ok());
  const auto pv = ingest::preview_folder(dir);
  CHECK(pv.ok());
  if (!pv.ok()) {
    std::printf("      %s\n", pv.status.text().c_str());
    return;
  }
  std::vector<std::string> keys;
  for (const auto& b : pv.value.kv) keys.push_back(kv_key(b));
  const std::vector<std::string> want = {
      "general.architecture", "general.type", "general.sampling.top_k", "general.sampling.top_p", "general.sampling.temp", "general.name",
      "general.finetune", "general.basename", "general.size_label", "general.license", "general.license.link", "general.base_model.count",
      "general.base_model.0.name", "general.base_model.0.organization", "general.base_model.0.repo_url", "general.tags", "general.languages",
      "qwen35moe.block_count", "qwen35moe.context_length", "qwen35moe.embedding_length", "qwen35moe.attention.head_count",
      "qwen35moe.attention.head_count_kv", "qwen35moe.rope.dimension_sections", "qwen35moe.rope.freq_base",
      "qwen35moe.attention.layer_norm_rms_epsilon", "qwen35moe.expert_count", "qwen35moe.expert_used_count", "qwen35moe.attention.key_length",
      "qwen35moe.attention.value_length", "general.file_type", "qwen35moe.expert_feed_forward_length", "qwen35moe.expert_shared_feed_forward_length",
      "qwen35moe.nextn_predict_layers", "qwen35moe.ssm.conv_kernel", "qwen35moe.ssm.state_size", "qwen35moe.ssm.group_count",
      "qwen35moe.ssm.time_step_rank", "qwen35moe.ssm.inner_size", "qwen35moe.attention.recurrent_layers", "qwen35moe.full_attention_interval",
      "qwen35moe.rope.dimension_count", "general.quantization_version", "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens",
      "tokenizer.ggml.token_type", "tokenizer.ggml.merges", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id",
      "tokenizer.ggml.bos_token_id", "tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token", "tokenizer.chat_template"};
  CHECK(keys == want);
  if (keys != want)
    for (std::size_t i = 0; i < keys.size(); ++i) std::printf("      kv %zu %s\n", i, keys[i].c_str());

  // the BF16 file
  const std::string bf16 = ptest::data_dir() + "/synth_hf_bf16.gguf";
  std::error_code ec;
  std::filesystem::remove(bf16, ec);
  ingest::ConvertOptions o;
  o.ftype = quant::FileType::bf16;
  const auto rep = ingest::convert_folder(dir, bf16, o);
  CHECK(rep.ok());
  if (!rep.ok()) {
    std::printf("      %s\n", rep.status.text().c_str());
    return;
  }
  os::FileMap m;
  CHECK(m.open(bf16).ok());
  io::GgufReader r;
  CHECK(r.parse(m).ok());
  const std::size_t n_tensors = r.tensors().size();
  CHECK_EQ(n_tensors, pv.value.tensors.size());
  // three non-block tensors, 3 x 19 DeltaNet tensors, one full-attention layer of 16, the MTP block of 20
  CHECK_EQ(n_tensors, 3u + 3u * 19u + 16u + 20u);
  CHECK_EQ(r.tensors().front().name, std::string("token_embd.weight"));
  // the reference's reading order: shard 1 sorted (embeddings, layer 0, layer 1), then shard 2 sorted
  CHECK_EQ(r.tensors()[1].name, std::string("blk.0.attn_norm.weight"));
  const io::GgufTensor* ssm_a = r.find_tensor("blk.0.ssm_a");
  const io::GgufTensor* inp_shexp = r.find_tensor("blk.0.ffn_gate_inp_shexp.weight");
  const io::GgufTensor* conv = r.find_tensor("blk.0.ssm_conv1d.weight");
  const io::GgufTensor* gate = r.find_tensor("blk.0.ffn_gate_exps.weight");
  CHECK(ssm_a && inp_shexp && conv && gate);
  if (!ssm_a || !inp_shexp || !conv || !gate) return;
  CHECK_EQ(ssm_a->type, 0u);
  CHECK_EQ(ssm_a->n_dims, 1u);
  CHECK_EQ(inp_shexp->n_dims, 2u);  // the array's rank: [1, 256] keeps both dimensions
  CHECK_EQ(inp_shexp->ne[1], 1u);
  CHECK_EQ(conv->type, 0u);
  CHECK(conv->ne[0] == 4 && conv->ne[1] == 512);
  CHECK(gate->ne[0] == 256 && gate->ne[1] == 256 && gate->ne[2] == 16 && gate->type == 30u);
  CHECK(r.find_tensor("blk.4.nextn.eh_proj.weight") != nullptr);
  CHECK(r.find_tensor("blk.4.nextn.shared_head_norm.weight") != nullptr);
  CHECK_EQ(r.get_uint("qwen35moe.block_count"), 5u);

  // the transforms, against the source shard read directly
  os::FileMap s1;
  CHECK(s1.open(dir + "/model-00001-of-00002.safetensors").ok());
  io::SafetensorsReader sr;
  CHECK(sr.parse(s1).ok());
  auto src = [&](const std::string& name) -> const std::uint8_t* {
    for (const auto& t : sr.tensors())
      if (t.name == name) return s1.data() + t.file_offset;
    return nullptr;
  };
  auto bf = [](const std::uint8_t* p, std::uint64_t i) {
    std::uint16_t h;
    std::memcpy(&h, p + i * 2, 2);
    return quant::bf16_to_fp32(h);
  };
  auto f32 = [](const std::uint8_t* p, std::uint64_t i) {
    float x;
    std::memcpy(&x, p + i * 4, 4);
    return x;
  };
  auto u16 = [](const std::uint8_t* p, std::uint64_t i) {
    std::uint16_t h;
    std::memcpy(&h, p + i * 2, 2);
    return h;
  };
  const std::uint8_t* out = m.data();
  // ssm_a = -exp(A_log), V heads tiled: new index j*K + i takes old i*VPK + j (K = 4, VPK = 2)
  const std::uint8_t* alog = src("model.language_model.layers.0.linear_attn.A_log");
  bool ok_a = alog != nullptr;
  for (int j = 0; ok_a && j < 2; ++j)
    for (int i = 0; i < 4; ++i) {
      const float want_v = -static_cast<float>(std::exp(static_cast<double>(bf(alog, static_cast<std::uint64_t>(i * 2 + j)))));
      if (f32(out + ssm_a->file_offset, static_cast<std::uint64_t>(j * 4 + i)) != want_v) ok_a = false;
    }
  CHECK(ok_a);
  // attn_norm = weight + 1 in f32
  const io::GgufTensor* an = r.find_tensor("blk.0.attn_norm.weight");
  const std::uint8_t* ln = src("model.language_model.layers.0.input_layernorm.weight");
  CHECK(an && ln && an->type == 0u && f32(out + an->file_offset, 7) == bf(ln, 7) + 1.0f);
  // ssm_norm keeps its value
  const io::GgufTensor* sn = r.find_tensor("blk.0.ssm_norm.weight");
  const std::uint8_t* lnn = src("model.language_model.layers.0.linear_attn.norm.weight");
  CHECK(sn && lnn && f32(out + sn->file_offset, 5) == bf(lnn, 5));
  // attn_qkv: rows below 256 (q, k) unchanged; V rows 256.. tiled with head size 32
  const io::GgufTensor* qkv = r.find_tensor("blk.0.attn_qkv.weight");
  const std::uint8_t* sq = src("model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
  bool ok_q = qkv && sq;
  for (std::uint64_t row = 0; ok_q && row < 512; row += 7) {
    std::uint64_t srow = row;
    if (row >= 256) {
      const std::uint64_t n = row - 256, j = n / 128, rem = n % 128, i = rem / 32, h = rem % 32;
      srow = 256 + i * 64 + j * 32 + h;
    }
    for (std::uint64_t c = 0; c < 256; c += 31)
      if (u16(out + qkv->file_offset, row * 256 + c) != u16(sq, srow * 256 + c)) ok_q = false;
  }
  CHECK(ok_q);
  // ssm_out: columns tiled
  const io::GgufTensor* so = r.find_tensor("blk.0.ssm_out.weight");
  const std::uint8_t* sso = src("model.language_model.layers.0.linear_attn.out_proj.weight");
  bool ok_o = so && sso;
  for (std::uint64_t row = 0; ok_o && row < 256; row += 13)
    for (std::uint64_t c = 0; c < 256; c += 5) {
      const std::uint64_t j = c / 128, rem = c % 128, i = rem / 32, h = rem % 32;
      if (u16(out + so->file_offset, row * 256 + c) != u16(sso, row * 256 + i * 64 + j * 32 + h)) ok_o = false;
    }
  CHECK(ok_o);
  // gate / up: the two halves of each expert's 512 rows
  const io::GgufTensor* up = r.find_tensor("blk.0.ffn_up_exps.weight");
  const std::uint8_t* gu = src("model.language_model.layers.0.mlp.experts.gate_up_proj");
  CHECK(up && gu && u16(out + gate->file_offset, (3 * 256 + 5) * 256 + 9) == u16(gu, (3 * 512 + 5) * 256 + 9) &&
        u16(out + up->file_offset, (3 * 256 + 5) * 256 + 9) == u16(gu, (3 * 512 + 256 + 5) * 256 + 9));

  // Q4_K_M in one pass equals the reference's two passes (convert to BF16, then quantize)
  const std::string q1 = ptest::data_dir() + "/synth_hf_Q4_K_M.onepass.gguf", q2 = ptest::data_dir() + "/synth_hf_Q4_K_M.twopass.gguf";
  std::filesystem::remove(q1, ec);
  std::filesystem::remove(q2, ec);
  o.ftype = quant::FileType::q4_k_m;
  o.threads = 2;
  const auto r1 = ingest::convert_folder(dir, q1, o);
  CHECK(r1.ok());
  if (!r1.ok()) std::printf("      %s\n", r1.status.text().c_str());
  io::RequantOptions ro;
  ro.ftype = quant::FileType::q4_k_m;
  ro.threads = 2;
  CHECK(io::requantize(m, r, q2, ro).ok());
  CHECK_EQ(file_sha(q1), file_sha(q2));
  // regression pins (MEASURED in the sandbox; the reference's own run on this folder is P-X6's job)
  const std::string bf16_sha = file_sha(bf16), q_sha = file_sha(q1);
  const char* kBf16Sha = "9c4b52f164d9367866f925338165b18bbb50cf407a8ac132cf76f79ea4673bf7";
  const char* kQ4KMSha = "9e02a016ad346e708765d0ddce199b8407b8a890d57e398c85885ee2e5d78ff6";
  CHECK_EQ(bf16_sha, std::string(kBf16Sha));
  CHECK_EQ(q_sha, std::string(kQ4KMSha));
  if (bf16_sha != kBf16Sha || q_sha != kQ4KMSha) std::printf("      bf16 sha %s\n      q4_k_m sha %s\n", bf16_sha.c_str(), q_sha.c_str());
  // nothing is overwritten
  CHECK(ingest::convert_folder(dir, q1, o).status.code == Code::invalid_argument);
}
