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

enum class TensorClass { embedding, head, spine, expert };
const char* class_name(TensorClass c);
// Classification by the GGUF tensor name (qwen35moe and the common llama.cpp names; 5.4).
TensorClass classify(const std::string& name);

struct ByteMap {
  std::uint64_t embedding = 0, head = 0, spine = 0, experts = 0;
  std::uint64_t expert_layers = 0;       // layers that have routed experts
  std::uint64_t per_layer_experts = 0;   // expert bytes of the largest layer
  std::uint64_t tensors = 0;
  bool head_tied = false;                // no output.weight: the embedding serves as the head
  [[nodiscard]] std::uint64_t always_resident() const { return embedding + (head_tied ? 0 : head) + spine; }
  [[nodiscard]] std::uint64_t total() const { return embedding + head + spine + experts; }
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
  std::uint64_t resident_fixed = 0;      // embedding + head + spine
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
