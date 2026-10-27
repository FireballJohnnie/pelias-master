// provenance: original (tensor names, shapes, types and per-block order READ-SRC from the served bf16 export's header,
//   read 2026-10-26)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "unit/synth_model.h"

#include <cstring>

#include "io/gguf_reader.h"
#include "io/gguf_writer.h"
#include "os/file_map.h"
#include "quant/block_types.h"

namespace ptest {
namespace {

using pelias::quant::TensorDesc;
constexpr std::uint32_t kBF16 = 30, kF32 = 0;

TensorDesc td(const std::string& name, std::vector<std::uint64_t> ne, std::uint32_t type) {
  TensorDesc d;
  d.name = name;
  d.n_dims = static_cast<std::uint32_t>(ne.size());
  for (std::size_t i = 0; i < ne.size(); ++i) d.ne[i] = ne[i];
  d.type = type;
  return d;
}

std::uint64_t scale_dim(std::size_t i, std::uint64_t d) {
  if (i == 0) return d == 2048 ? 256 : d == 4096 ? 512 : d == 512 ? 256 : d;
  switch (d) {
    case 248320: return 512;
    case 8192: case 4096: case 2048: return 64;
    case 512: return 32;
    case 256: return 4;
    default: return d;
  }
}

}  // namespace

std::vector<TensorDesc> qwen35moe_served_tensors() {
  std::vector<TensorDesc> v;
  v.push_back(td("token_embd.weight", {2048, 248320}, kBF16));
  for (int b = 0; b <= 40; ++b) {
    const std::string p = "blk." + std::to_string(b) + ".";
    const bool full = b % 4 == 3 || b == 40;
    auto ffn_tail = [&]() {
      v.push_back(td(p + "ffn_gate_inp.weight", {2048, 256}, kF32));
      v.push_back(td(p + "ffn_down_shexp.weight", {512, 2048}, kBF16));
      v.push_back(td(p + "ffn_gate_shexp.weight", {2048, 512}, kBF16));
      v.push_back(td(p + "ffn_up_shexp.weight", {2048, 512}, kBF16));
      v.push_back(td(p + "ffn_gate_inp_shexp.weight", {2048, 1}, kF32));
      v.push_back(td(p + "post_attention_norm.weight", {2048}, kF32));
    };
    auto attn_tail = [&]() {
      v.push_back(td(p + "attn_k_norm.weight", {256}, kF32));
      v.push_back(td(p + "attn_k.weight", {2048, 512}, kBF16));
      v.push_back(td(p + "attn_output.weight", {4096, 2048}, kBF16));
      v.push_back(td(p + "attn_q_norm.weight", {256}, kF32));
      v.push_back(td(p + "attn_q.weight", {2048, 8192}, kBF16));
      v.push_back(td(p + "attn_v.weight", {2048, 512}, kBF16));
    };
    if (b == 40) {  // the MTP block
      v.push_back(td(p + "ffn_gate_exps.weight", {2048, 512, 256}, kBF16));
      v.push_back(td(p + "ffn_up_exps.weight", {2048, 512, 256}, kBF16));
      v.push_back(td(p + "nextn.eh_proj.weight", {4096, 2048}, kBF16));
      v.push_back(td(p + "attn_norm.weight", {2048}, kF32));
      v.push_back(td(p + "ffn_down_exps.weight", {512, 2048, 256}, kBF16));
      ffn_tail();
      attn_tail();
      v.push_back(td(p + "nextn.shared_head_norm.weight", {2048}, kF32));
      v.push_back(td(p + "nextn.enorm.weight", {2048}, kF32));
      v.push_back(td(p + "nextn.hnorm.weight", {2048}, kF32));
    } else if (full) {
      v.push_back(td(p + "attn_norm.weight", {2048}, kF32));
      v.push_back(td(p + "ffn_down_exps.weight", {512, 2048, 256}, kBF16));
      v.push_back(td(p + "ffn_gate_exps.weight", {2048, 512, 256}, kBF16));
      v.push_back(td(p + "ffn_up_exps.weight", {2048, 512, 256}, kBF16));
      ffn_tail();
      attn_tail();
    } else {  // DeltaNet block
      v.push_back(td(p + "attn_qkv.weight", {2048, 8192}, kBF16));
      v.push_back(td(p + "attn_gate.weight", {2048, 4096}, kBF16));
      v.push_back(td(p + "ssm_out.weight", {4096, 2048}, kBF16));
      v.push_back(td(p + "ffn_gate_exps.weight", {2048, 512, 256}, kBF16));
      v.push_back(td(p + "ffn_up_exps.weight", {2048, 512, 256}, kBF16));
      v.push_back(td(p + "attn_norm.weight", {2048}, kF32));
      v.push_back(td(p + "ssm_a", {32}, kF32));
      v.push_back(td(p + "ssm_conv1d.weight", {4, 8192}, kF32));
      v.push_back(td(p + "ssm_dt.bias", {32}, kF32));
      v.push_back(td(p + "ssm_alpha.weight", {2048, 32}, kBF16));
      v.push_back(td(p + "ssm_beta.weight", {2048, 32}, kBF16));
      v.push_back(td(p + "ssm_norm.weight", {128}, kF32));
      v.push_back(td(p + "ffn_down_exps.weight", {512, 2048, 256}, kBF16));
      ffn_tail();
    }
  }
  v.push_back(td("output.weight", {2048, 248320}, kBF16));
  v.push_back(td("output_norm.weight", {2048}, kF32));
  return v;
}

pelias::Status write_synthetic_qwen35moe(const std::string& header_path, const std::string& out_path) {
  using pelias::Status;
  pelias::os::FileMap m;
  Status s = m.open(header_path);
  if (!s) return s;
  pelias::io::GgufReader r;
  s = r.parse(m, false);
  if (!s) return s;
  std::vector<pelias::io::GgufKvBlob> kv;
  for (const auto& e : r.kv()) {
    if (e.key.rfind("quantize.imatrix", 0) == 0) continue;
    if (e.key == "qwen35moe.block_count") {
      kv.push_back(pelias::io::gguf_kv_u32(e.key, 41));
      kv.push_back(pelias::io::gguf_kv_u32("qwen35moe.nextn_predict_layers", 1));
      continue;
    }
    kv.emplace_back(m.data() + e.file_begin, m.data() + e.file_end);
  }

  // integer generator: every value is exact in float, so all platforms produce the same bytes
  std::uint64_t st = 0x2545F4914F6CDD1Dull;
  auto next = [&st]() {
    st ^= st << 13;
    st ^= st >> 7;
    st ^= st << 17;
    return st;
  };
  auto bell = [&next]() {  // a sum of four 14-bit uniforms, centred: roughly normal, integer in [-32768, 32764]
    const std::uint64_t x = next();
    const auto s4 = static_cast<std::int32_t>((x & 0x3FFF) + ((x >> 14) & 0x3FFF) + ((x >> 28) & 0x3FFF) + ((x >> 42) & 0x3FFF));
    return s4 - 32768;
  };
  auto to_bf16 = [](float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + (0x7FFFu + ((u >> 16) & 1u))) >> 16);
  };

  const std::vector<TensorDesc> list = qwen35moe_served_tensors();
  std::vector<pelias::io::GgufOutTensor> out;
  std::vector<std::vector<std::uint8_t>> data;
  for (std::size_t ti = 0; ti < list.size(); ++ti) {
    pelias::io::GgufOutTensor o;
    o.name = list[ti].name;
    std::uint64_t n = 1;
    for (std::uint32_t d = 0; d < list[ti].n_dims; ++d) {
      o.ne[d] = scale_dim(d, list[ti].ne[d]);
      n *= o.ne[d];
    }
    o.type = list[ti].type;
    std::vector<std::uint8_t> bytes;
    if (o.type == kBF16) {
      bytes.resize(static_cast<std::size_t>(n * 2));
      for (std::uint64_t i = 0; i < n; ++i) {
        float x = static_cast<float>(bell()) * (1.0f / 1048576.0f);  // about 0.01 in size, exact
        if (i % 997 == 0) x *= 32.0f;                                // outliers, exact
        if (ti % 7 == 0 && i < 256) x = 0.0f;                        // an all-zero block now and then
        const std::uint16_t h = to_bf16(x);
        std::memcpy(bytes.data() + i * 2, &h, 2);
      }
    } else {
      bytes.resize(static_cast<std::size_t>(n * 4));
      for (std::uint64_t i = 0; i < n; ++i) {
        const float x = 1.0f + static_cast<float>(bell()) * (1.0f / 262144.0f);  // the product is exact; one rounding
        std::memcpy(bytes.data() + i * 4, &x, 4);
      }
    }
    o.bytes = bytes.size();
    out.push_back(o);
    data.push_back(std::move(bytes));
  }
  pelias::io::GgufFileWriter w;
  s = w.begin(out_path, kv, out, 32);
  if (!s) return s;
  for (const auto& d : data) {
    s = w.write(d.data(), d.size());
    if (!s) return s;
  }
  return w.finish();
}

}  // namespace ptest
