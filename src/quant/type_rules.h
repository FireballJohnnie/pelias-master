// provenance: re-expressed (not copied) from llama.cpp src/llama-quant.cpp at commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a (MIT; THIRD_PARTY_NOTICES.md): which tensors are quantized and to which
//   type, read as a specification so a stock mix written by Pelias is identical to the reference's (5.3, P-E6)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// type_rules.h - the per-tensor type rule of the stock quantization mixes ("Q4_K_M" and others are mixes, not single
// types). Given the tensors in the order they will be written and three facts about the model, it returns the type of
// every tensor. Only the file types whose tensor types this build can produce are offered; a rule that would pick a
// type not built yet (for example Q5_K for an 8-expert model, or a fallback for an odd row length) is refused by the
// quantizer with the type named, never silently changed.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pelias/status.h"

namespace pelias::quant {

// The GGUF "general.file_type" numbers of the reference (llama.h), for the mixes this build offers.
enum class FileType : std::uint32_t {
  all_f32 = 0,
  f16 = 1,
  q8_0 = 7,
  q4_k_m = 15,
  q6_k = 18,
  bf16 = 32,
};

// "Q4_K_M", "Q8_0", "Q6_K", "F16", "BF16", "F32" (case-insensitive) -> the file type.
Result<FileType> file_type_from_name(const std::string& name);
const char* file_type_name(FileType f);

struct TensorDesc {
  std::string name;
  std::uint32_t n_dims = 0;
  std::uint64_t ne[4] = {1, 1, 1, 1};
  std::uint32_t type = 0;  // the type in the input file
};

struct ModelFacts {
  std::string arch;            // general.architecture
  std::uint64_t n_expert = 0;  // <arch>.expert_count (0 when absent)
  std::uint64_t n_layer = 0;   // <arch>.block_count (all blocks, the MTP block included)
};

// The output type of each tensor, in the same order. `tensors` must be in write order (gguf_writer's sort): the rule
// counts some tensor kinds in that order.
Result<std::vector<std::uint32_t>> choose_types(const std::vector<TensorDesc>& tensors, const ModelFacts& model, FileType ftype);

// Whether the stock rule quantizes this tensor at all (2-D and 3-D weights, minus norms, routers and a list of small
// or special tensors). Exposed for tests.
bool tensor_allows_quantization(const TensorDesc& t, const std::string& arch);

}  // namespace pelias::quant
