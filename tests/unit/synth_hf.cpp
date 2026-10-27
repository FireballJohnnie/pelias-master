// provenance: original (the folder's layout READ-SRC from the reference model's config.json, index and shard headers,
//   read 2026-10-26; sizes scaled down)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "unit/synth_hf.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <vector>

#include "io/json.h"

namespace ptest {
namespace {

using pelias::Code;
using pelias::Status;
using pelias::io::Json;

Status write_file(const std::string& path, const std::string& bytes) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return Status::fail(Code::io_error, "cannot create " + path);
  const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
  std::fclose(f);
  return ok ? Status::good() : Status::fail(Code::io_error, "short write " + path);
}

Json arr_u(std::initializer_list<std::uint64_t> v) {
  Json a = Json::array();
  for (const auto x : v) a.push(Json::unsigned_integer(x));
  return a;
}

struct T {
  std::string name;
  std::vector<std::uint64_t> shape;
  int kind;  // 0 weight-like, 1 norm-like (around 0), 2 A_log-like (0.5 .. 3), 3 dt-like
};

std::uint16_t to_bf16(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<std::uint16_t>((u + (0x7FFFu + ((u >> 16) & 1u))) >> 16);
}

}  // namespace

Status write_synthetic_hf_folder(const std::string& dir) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::create_directories(dir, ec);
  using S = SynthHf;

  // ---- config.json (the real model's keys)
  Json tc = Json::object();
  tc.set("attention_bias", Json::boolean(false));
  tc.set("attn_output_gate", Json::boolean(true));
  tc.set("bos_token_id", Json::integer(300));
  tc.set("torch_dtype", Json::string("bfloat16"));
  tc.set("eos_token_id", Json::integer(300));
  tc.set("full_attention_interval", Json::integer(4));
  tc.set("head_dim", Json::integer(S::head_dim));
  tc.set("hidden_act", Json::string("silu"));
  tc.set("hidden_size", Json::integer(S::hidden));
  Json lt = Json::array();
  for (int i = 0; i < S::layers; ++i) lt.push(Json::string(i % 4 == 3 ? "full_attention" : "linear_attention"));
  tc.set("layer_types", lt);
  tc.set("linear_conv_kernel_dim", Json::integer(4));
  tc.set("linear_key_head_dim", Json::integer(S::head_k));
  tc.set("linear_num_key_heads", Json::integer(S::k_heads));
  tc.set("linear_num_value_heads", Json::integer(S::v_heads));
  tc.set("linear_value_head_dim", Json::integer(S::head_v));
  tc.set("max_position_embeddings", Json::integer(262144));
  tc.set("model_type", Json::string("qwen3_5_moe_text"));
  tc.set("moe_intermediate_size", Json::integer(S::ff));
  tc.set("mtp_num_hidden_layers", Json::integer(1));
  tc.set("num_attention_heads", Json::integer(S::attn_heads));
  tc.set("num_experts", Json::integer(S::experts));
  tc.set("num_experts_per_tok", Json::integer(4));
  tc.set("num_hidden_layers", Json::integer(S::layers));
  tc.set("num_key_value_heads", Json::integer(S::kv_heads));
  tc.set("pad_token_id", Json::null());
  tc.set("partial_rotary_factor", Json::number(0.25));
  tc.set("rms_norm_eps", Json::number(1e-06));
  Json rp = Json::object();
  rp.set("mrope_interleaved", Json::boolean(true));
  rp.set("mrope_section", arr_u({11, 11, 10}));
  rp.set("partial_rotary_factor", Json::number(0.25));
  rp.set("rope_theta", Json::integer(10000000));
  rp.set("rope_type", Json::string("default"));
  tc.set("rope_parameters", rp);
  tc.set("shared_expert_intermediate_size", Json::integer(S::ff));
  tc.set("tie_word_embeddings", Json::boolean(false));
  tc.set("vocab_size", Json::integer(S::vocab));
  Json cfg = Json::object();
  Json archs = Json::array();
  archs.push(Json::string("Qwen3_5MoeForConditionalGeneration"));
  cfg.set("architectures", archs);
  cfg.set("model_type", Json::string("qwen3_5_moe"));
  cfg.set("pad_token_id", Json::integer(303));
  cfg.set("text_config", tc);
  cfg.set("tie_word_embeddings", Json::boolean(false));
  cfg.set("transformers_version", Json::string("4.57.1"));
  Json vc = Json::object();
  vc.set("depth", Json::integer(1));
  cfg.set("vision_config", vc);
  Status s = write_file(dir + "/config.json", cfg.dump(2));
  if (!s) return s;

  // ---- generation_config.json, README.md, chat_template.jinja
  s = write_file(dir + "/generation_config.json",
                 "{\n  \"bos_token_id\": 300,\n  \"do_sample\": true,\n  \"eos_token_id\": [302, 300],\n  \"pad_token_id\": 300,\n"
                 "  \"temperature\": 0.6,\n  \"top_k\": 20,\n  \"top_p\": 0.95\n}");
  if (!s) return s;
  s = write_file(dir + "/README.md",
                 "---\nlibrary_name: transformers\nlicense: apache-2.0\nlicense_link: https://huggingface.co/Example/Tiny-MoE-2B-A1B/blob/main/LICENSE\n"
                 "pipeline_tag: text-generation\nbase_model:\n- Example/Tiny-MoE-2B-A1B\ntags: [qwen3_5_moe, 'tiny']\nlanguage:\n- en\n- no\n---\n\n# Tiny\n");
  if (!s) return s;
  const std::string chat = "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}";
  s = write_file(dir + "/chat_template.jinja", chat);
  if (!s) return s;

  // ---- tokenizer.json: 300 model tokens, 6 added tokens (ids 300..305), vocabulary size 512
  Json vocab = Json::object();
  for (int i = 0; i < 300; ++i) {
    std::string t;
    if (i < 94) t = std::string(1, static_cast<char>('!' + i));        // printable ASCII
    else if (i == 94) t = "\xC4\xA0";                                     // U+0120 (a byte-level space)
    else if (i == 95) t = "\xC3\xA9";                                     // U+00E9
    else t = "tok" + std::to_string(i);
    vocab.set(t, Json::integer(i));
  }
  Json merges = Json::array();
  for (int i = 0; i < 20; ++i) {
    Json pair = Json::array();
    pair.push(Json::string(std::string(1, static_cast<char>('a' + i))));
    pair.push(Json::string(i == 3 ? "x y" : std::string(1, static_cast<char>('b' + i))));  // one with a space
    merges.push(pair);
  }
  Json model = Json::object();
  model.set("type", Json::string("BPE"));
  model.set("dropout", Json::null());
  model.set("unk_token", Json::null());
  model.set("continuing_subword_prefix", Json::string(""));
  model.set("end_of_word_suffix", Json::string(""));
  model.set("fuse_unk", Json::boolean(false));
  model.set("byte_fallback", Json::boolean(false));
  model.set("ignore_merges", Json::boolean(false));
  model.set("vocab", vocab);
  model.set("merges", merges);
  struct A {
    int id;
    const char* content;
    bool special;
  };
  const A added_list[] = {{300, "<|endoftext|>", true}, {301, "<|im_start|>", true}, {302, "<|im_end|>", true},
                          {303, "<|fim_pad|>", false},  {304, "<think>", false},      {305, "<tool\xE2\x96\x81" "call>", false}};
  Json added = Json::array();
  Json decoder = Json::object();
  for (const A& a : added_list) {
    Json x = Json::object();
    x.set("id", Json::integer(a.id));
    x.set("content", Json::string(a.content));
    x.set("single_word", Json::boolean(false));
    x.set("lstrip", Json::boolean(false));
    x.set("rstrip", Json::boolean(false));
    x.set("normalized", Json::boolean(false));
    x.set("special", Json::boolean(a.special));
    added.push(x);
    Json d = Json::object();
    d.set("content", Json::string(a.content));
    d.set("lstrip", Json::boolean(false));
    d.set("normalized", Json::boolean(false));
    d.set("rstrip", Json::boolean(false));
    d.set("single_word", Json::boolean(false));
    d.set("special", Json::boolean(a.special));
    decoder.set(std::to_string(a.id), d);
  }
  const char* pre =
      R"({"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]})";
  Json tok = Json::object();
  tok.set("version", Json::string("1.0"));
  tok.set("truncation", Json::null());
  tok.set("padding", Json::null());
  tok.set("added_tokens", added);
  Json nfc = Json::object();
  nfc.set("type", Json::string("NFC"));
  tok.set("normalizer", nfc);
  tok.set("pre_tokenizer", Json::parse(pre).value);
  Json bl = Json::object();
  bl.set("type", Json::string("ByteLevel"));
  bl.set("add_prefix_space", Json::boolean(false));
  bl.set("trim_offsets", Json::boolean(false));
  bl.set("use_regex", Json::boolean(false));
  tok.set("post_processor", bl);
  tok.set("decoder", bl);
  tok.set("model", model);
  s = write_file(dir + "/tokenizer.json", tok.dump(2));
  if (!s) return s;
  Json tcfg = Json::object();
  tcfg.set("add_bos_token", Json::boolean(false));
  tcfg.set("add_prefix_space", Json::boolean(false));
  tcfg.set("added_tokens_decoder", decoder);
  tcfg.set("bos_token", Json::null());
  tcfg.set("eos_token", Json::string("<|im_end|>"));
  tcfg.set("pad_token", Json::string("<|endoftext|>"));
  tcfg.set("chat_template", Json::string(chat));
  tcfg.set("tokenizer_class", Json::string("Qwen2Tokenizer"));
  s = write_file(dir + "/tokenizer_config.json", tcfg.dump(2));
  if (!s) return s;

  // ---- tensors: two shards, names as the real model's
  std::vector<T> shard1, shard2;
  const std::uint64_t H = S::hidden, E = S::experts, F = S::ff, V = S::vocab;
  const std::uint64_t qk = 2 * S::k_heads * S::head_k, vdim = static_cast<std::uint64_t>(S::v_heads) * S::head_v;
  shard1.push_back({"model.language_model.embed_tokens.weight", {V, H}, 0});
  auto layer = [&](std::vector<T>& out, const std::string& p, bool full_attn) {
    out.push_back({p + "input_layernorm.weight", {H}, 1});
    out.push_back({p + "post_attention_layernorm.weight", {H}, 1});
    if (full_attn) {
      const std::uint64_t q = static_cast<std::uint64_t>(S::attn_heads) * S::head_dim;
      out.push_back({p + "self_attn.q_proj.weight", {2 * q, H}, 0});
      out.push_back({p + "self_attn.k_proj.weight", {static_cast<std::uint64_t>(S::kv_heads) * S::head_dim, H}, 0});
      out.push_back({p + "self_attn.v_proj.weight", {static_cast<std::uint64_t>(S::kv_heads) * S::head_dim, H}, 0});
      out.push_back({p + "self_attn.o_proj.weight", {H, q}, 0});
      out.push_back({p + "self_attn.q_norm.weight", {static_cast<std::uint64_t>(S::head_dim)}, 1});
      out.push_back({p + "self_attn.k_norm.weight", {static_cast<std::uint64_t>(S::head_dim)}, 1});
    } else {
      out.push_back({p + "linear_attn.in_proj_qkv.weight", {qk + vdim, H}, 0});
      out.push_back({p + "linear_attn.in_proj_z.weight", {vdim, H}, 0});
      out.push_back({p + "linear_attn.in_proj_a.weight", {static_cast<std::uint64_t>(S::v_heads), H}, 0});
      out.push_back({p + "linear_attn.in_proj_b.weight", {static_cast<std::uint64_t>(S::v_heads), H}, 0});
      out.push_back({p + "linear_attn.A_log", {static_cast<std::uint64_t>(S::v_heads)}, 2});
      out.push_back({p + "linear_attn.dt_bias", {static_cast<std::uint64_t>(S::v_heads)}, 3});
      out.push_back({p + "linear_attn.conv1d.weight", {qk + vdim, 1, 4}, 0});
      out.push_back({p + "linear_attn.norm.weight", {static_cast<std::uint64_t>(S::head_v)}, 1});
      out.push_back({p + "linear_attn.out_proj.weight", {H, vdim}, 0});
    }
    out.push_back({p + "mlp.gate.weight", {E, H}, 0});
    out.push_back({p + "mlp.shared_expert_gate.weight", {1, H}, 0});
    out.push_back({p + "mlp.shared_expert.gate_proj.weight", {F, H}, 0});
    out.push_back({p + "mlp.shared_expert.up_proj.weight", {F, H}, 0});
    out.push_back({p + "mlp.shared_expert.down_proj.weight", {H, F}, 0});
    out.push_back({p + "mlp.experts.gate_up_proj", {E, 2 * F, H}, 0});
    out.push_back({p + "mlp.experts.down_proj", {E, H, F}, 0});
  };
  for (int l = 0; l < 2; ++l) layer(shard1, "model.language_model.layers." + std::to_string(l) + ".", l % 4 == 3);
  for (int l = 2; l < S::layers; ++l) layer(shard2, "model.language_model.layers." + std::to_string(l) + ".", l % 4 == 3);
  layer(shard2, "mtp.layers.0.", true);
  shard2.push_back({"mtp.fc.weight", {H, 2 * H}, 0});
  shard2.push_back({"mtp.pre_fc_norm_embedding.weight", {H}, 1});
  shard2.push_back({"mtp.pre_fc_norm_hidden.weight", {H}, 1});
  shard2.push_back({"mtp.norm.weight", {H}, 1});
  shard2.push_back({"lm_head.weight", {V, H}, 0});
  shard2.push_back({"model.language_model.norm.weight", {H}, 1});
  shard1.push_back({"model.visual.blocks.0.attn.qkv.weight", {24, 16}, 0});

  std::uint64_t st = 0x9E3779B97F4A7C15ull;
  auto next = [&st]() {
    st ^= st << 13;
    st ^= st >> 7;
    st ^= st << 17;
    return st;
  };
  Json weight_map = Json::object();
  std::uint64_t total = 0;
  auto write_shard = [&](const std::string& file, const std::vector<T>& ts) -> Status {
    Json header = Json::object();
    Json meta = Json::object();
    meta.set("format", Json::string("pt"));
    header.set("__metadata__", meta);
    std::string data;
    for (const T& t : ts) {
      std::uint64_t n = 1;
      for (const auto d : t.shape) n *= d;
      Json e = Json::object();
      e.set("dtype", Json::string("BF16"));
      Json shape = Json::array();
      for (const auto d : t.shape) shape.push(Json::unsigned_integer(d));
      e.set("shape", shape);
      Json off = Json::array();
      off.push(Json::unsigned_integer(data.size()));
      off.push(Json::unsigned_integer(data.size() + n * 2));
      e.set("data_offsets", off);
      header.set(t.name, e);
      weight_map.set(t.name, Json::string(file));
      for (std::uint64_t i = 0; i < n; ++i) {
        const auto r = static_cast<std::int32_t>(next() >> 44) - 524288;  // integer in [-2^19, 2^19)
        float x;
        switch (t.kind) {
          case 1: x = static_cast<float>(r) * (1.0f / 4194304.0f); break;                       // about +-0.125
          case 2: x = 0.5f + static_cast<float>(r + 524288) * (2.5f / 1048576.0f); break;      // 0.5 .. 3
          case 3: x = static_cast<float>(r) * (1.0f / 262144.0f); break;                        // about +-2
          default: x = static_cast<float>(r) * (1.0f / 16777216.0f); break;                     // about +-0.03
        }
        const std::uint16_t h = to_bf16(x);
        data.append(reinterpret_cast<const char*>(&h), 2);
      }
      total += n * 2;
    }
    std::string hj = header.dump();
    while ((8 + hj.size()) % 8 != 0) hj.push_back(' ');
    const std::uint64_t hn = hj.size();
    std::string bytes(reinterpret_cast<const char*>(&hn), 8);
    bytes += hj;
    bytes += data;
    return write_file(dir + "/" + file, bytes);
  };
  s = write_shard("model-00001-of-00002.safetensors", shard1);
  if (!s) return s;
  s = write_shard("model-00002-of-00002.safetensors", shard2);
  if (!s) return s;
  Json index = Json::object();
  Json im = Json::object();
  im.set("total_size", Json::unsigned_integer(total));
  index.set("metadata", im);
  index.set("weight_map", weight_map);
  return write_file(dir + "/model.safetensors.index.json", index.dump(2));
}

}  // namespace ptest
