// provenance: original (the metadata keys and their meaning READ-SRC from llama.cpp src/models/qwen35moe.cpp,
//   src/llama-model.cpp and src/llama-arch.cpp at commit 2ca15f5404760548c39e7b92bd43116a09414a1a, read as a
//   specification)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// spec.h - the shape of a model as its GGUF file states it (masterwork 14.2, model/spec). One struct per architecture
// family; qwen35moe (Qwen3.5 / 3.6 MoE: gated DeltaNet and full-attention layers, 256 routed experts
// with a gated shared expert, one MTP block after the trunk). Every value comes from the file's metadata or from a
// tensor's shape; nothing is assumed from the model's name.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "io/gguf_reader.h"
#include "pelias/status.h"

namespace pelias::model {

// The architecture families the reference pass runs (GP3-09, author 2026-10-30 11:24: dense families before the first
// build). qwen35moe is the hybrid MoE; the others are dense decoders: every layer attention + one gated FFN.
enum class Family { qwen35moe, qwen3, qwen2, llama };

struct Qwen35MoeSpec {
  Family family = Family::qwen35moe;
  std::string arch;                // general.architecture as the file names it
  bool dense = false;              // dense decoder: no DeltaNet, no experts; ffn_gate / ffn_up / ffn_down per layer
  std::uint32_t n_ff = 0;          // feed_forward_length (dense)
  bool rope_neox = true;           // NEOX pairs (i, i + n_rot/2); false: adjacent pairs (2i, 2i+1), the llama order
  bool attn_gate = true;           // the query projection carries a per-head output gate [query | gate] (qwen35moe)
  bool qk_norm = true;             // per-head RMS norms on q and k (qwen35moe, qwen3)
  bool qkv_bias = false;           // q / k / v biases (qwen2)
  std::uint32_t n_ctx_train = 0;   // context_length (informational)
  std::uint32_t n_layer = 0;       // trunk layers run in the main pass (block_count - nextn_predict_layers)
  std::uint32_t n_layer_all = 0;   // block_count (the MTP block(s) included)
  std::uint32_t n_embd = 0;
  std::uint32_t n_vocab = 0;       // rows of token_embd (and of output)
  // full attention
  std::uint32_t n_head = 0;
  std::uint32_t n_head_kv = 0;
  std::uint32_t head_dim = 0;      // key_length == value_length
  std::uint32_t n_rot = 0;         // rope.dimension_count: the first n_rot dimensions of each head are rotated
  std::int32_t rope_sections[4] = {0, 0, 0, 0};
  float rope_freq_base = 10000.0f;
  float attn_scale = 0.0f;         // 0: 1/sqrt(head_dim)
  float rms_eps = 0.0f;
  // gated DeltaNet
  std::uint32_t ssm_d_conv = 0;    // conv kernel taps
  std::uint32_t ssm_d_inner = 0;   // value heads x value head dim
  std::uint32_t ssm_d_state = 0;   // key (and value) head dim
  std::uint32_t ssm_n_group = 0;   // key heads
  std::uint32_t ssm_dt_rank = 0;   // value heads
  std::vector<bool> recurrent;     // per block: true = DeltaNet layer, false = full attention
  // MoE
  std::uint32_t n_expert = 0;
  std::uint32_t n_expert_used = 0;
  std::uint32_t n_ff_exp = 0;
  std::uint32_t n_ff_shexp = 0;
  float expert_weights_scale = 0.0f;  // applied only when neither 0 nor 1 (as the reference does)

  [[nodiscard]] bool is_recurrent(std::uint32_t il) const { return il < recurrent.size() && recurrent[il]; }
  [[nodiscard]] std::uint32_t conv_channels() const { return ssm_d_inner + 2 * ssm_n_group * ssm_d_state; }
};

// Reads and checks the spec. Refuses what the reference pass does not handle (another architecture, rope scaling,
// shapes that do not agree with each other), naming the key. Reads qwen35moe and the dense families qwen3, qwen2 and llama.
Result<Qwen35MoeSpec> read_qwen35moe_spec(const io::GgufReader& r);

}  // namespace pelias::model
