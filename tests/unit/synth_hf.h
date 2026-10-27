// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// synth_hf.h - a small Hugging Face folder of the Qwen3.5/3.6 MoE layout for Stage A's tests: config.json (the real
// model's keys, small sizes: hidden 256, 4 layers of which the 4th is full attention, one MTP block, 16 experts,
// 4 key heads and 8 value heads so the V heads are reordered), two safetensors shards with an index (BF16 data from an
// integer generator, one vision tensor that must be skipped), a byte-level BPE tokenizer.json with the qwen35
// pre-tokenizer, tokenizer_config.json, generation_config.json, chat_template.jinja and a README.md with front matter.
#pragma once

#include <string>

#include "pelias/status.h"

namespace ptest {

// Writes the folder (created if absent; existing files replaced).
pelias::Status write_synthetic_hf_folder(const std::string& dir);

// Facts of the synthetic model the tests check against.
struct SynthHf {
  static constexpr int hidden = 256, layers = 4, experts = 16, ff = 256, k_heads = 4, v_heads = 8, head_k = 32, head_v = 32;
  static constexpr int vocab = 512, attn_heads = 4, kv_heads = 1, head_dim = 64;
};

}  // namespace ptest
