// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// memory_plan.h - the device memory plan for serving a GGUF model (10.3, 5.4): what stays in VRAM (spine, embedding,
// head, the hot expert bank) against the measured ceiling. A plan over the ceiling is refused (gate PG3), and the
// largest bank that fits is proposed instead (7.6). The byte map comes straight from the tensor table, so every number
// in the plan is DERIVED from the file; the working-set term is an ESTIMATE until P-M2 runs and is labelled so.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "io/gguf_reader.h"
#include "pelias/status.h"

namespace pelias::plan {

enum class TensorClass { embedding, head, spine, expert, mtp };
const char* class_name(TensorClass c);

// No MTP block in the file.
constexpr std::uint64_t kNoMtp = ~std::uint64_t{0};
// The first block index that belongs to the multi-token-prediction (MTP) head, or kNoMtp. llama.cpp stores the head as
// the last `<arch>.nextn_predict_layers` blocks of `<arch>.block_count` (READ-SRC: the served file has block_count 41,
// nextn_predict_layers 1, and blk.40 holds the head's 20 tensors, nextn.* among them).
std::uint64_t first_mtp_block(const io::GgufReader& r);

// Classification by the GGUF tensor name (qwen35moe and the common llama.cpp names; 5.4). Blocks at or after
// `first_mtp` are the MTP head: their routed experts stay in the expert pool (the bank decides by hits, like any
// layer's), every other tensor of the head is `mtp` and always resident (the head runs for every drafted token).
TensorClass classify(const std::string& name, std::uint64_t first_mtp = kNoMtp);

struct ByteMap {
  std::uint64_t embedding = 0, head = 0, spine = 0, experts = 0;
  std::uint64_t mtp = 0;                 // the MTP head without its routed experts (always resident)
  std::uint64_t mtp_experts = 0;         // the MTP head's routed experts (part of `experts`)
  std::uint64_t mtp_blocks = 0;          // blocks that form the MTP head (0: the file has none)
  std::uint64_t expert_layers = 0;       // layers that have routed experts
  std::uint64_t per_layer_experts = 0;   // expert bytes of the largest layer
  std::uint64_t tensors = 0;
  bool head_tied = false;                // no output.weight: the embedding serves as the head
  [[nodiscard]] std::uint64_t always_resident() const { return embedding + (head_tied ? 0 : head) + spine + mtp; }
  [[nodiscard]] std::uint64_t total() const { return embedding + head + spine + experts + mtp; }
};
ByteMap byte_map(const io::GgufReader& r);

struct PlanRequest {
  std::uint64_t ceiling_bytes = 0;       // from hal::Ceiling
  std::string ceiling_basis;             // "cuda", "nvml", "stated", "reference"
  std::uint64_t bank_bytes = 0;          // requested hot bank; 0 = as large as fits
  std::uint64_t working_set_bytes = 0;   // 0 = the estimate
};

struct Plan {
  ByteMap map;
  std::uint64_t ceiling = 0;
  std::string ceiling_basis;
  std::uint64_t resident_fixed = 0;      // embedding + head + spine + MTP head
  std::uint64_t working_set = 0;
  std::string working_set_label;
  std::uint64_t bank = 0;                // the bank in this plan
  std::uint64_t peak = 0;                // resident_fixed + bank + working_set
  std::uint64_t headroom = 0;            // ceiling - peak when it fits
  double expert_bytes_resident_fraction = 0.0;  // bank / experts
  bool fits = false;
  std::uint64_t largest_bank_that_fits = 0;     // proposal when refused, or the maximum when bank was 0
  std::string verdict;                   // one line
};

// Builds the plan; the Status is over_ceiling when the request does not fit (the Plan still describes it and the
// proposal), invalid_argument when the ceiling is 0 or the bank exceeds the experts.
Result<Plan> make_plan(const io::GgufReader& r, const PlanRequest& req);

std::string format_bytes(std::uint64_t b);  // "7.50 GiB"
std::string plan_text(const Plan& p);        // the plan card, several lines

}  // namespace pelias::plan
