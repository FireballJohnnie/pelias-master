// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// probe_constants.h - numbers the planner uses before a machine's own probes have run, each with its label and source.
// Every value here is replaced by a measurement when one exists (Section 4.1, 17.2); nothing is a guess.
#pragma once

#include <cstdint>

namespace pelias::plan {

inline constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;

// Working set of the decode loop beyond the weights (KV cache, activations, scratch) at the reference window:
// 2.50 GiB, an ESTIMATE from the masterwork 8.8 that probe P-M2 measures. Used only until P-M2 has run.
inline constexpr std::uint64_t kWorkingSetEstimateBytes = std::uint64_t{2684354560};  // 2.50 GiB, ESTIMATE (8.8, P-M2)
inline constexpr const char* kWorkingSetLabel = "ESTIMATE (masterwork 8.8; P-M2 measures it)";

// Reference machine ceiling, MEASURED (P-H8, D26/D28): 10.7 GiB. Used only when the user states no ceiling and no
// device is present (the plan then says so).
inline constexpr std::uint64_t kReferenceCeilingBytes = std::uint64_t{11488709837};  // 10.7 GiB
inline constexpr const char* kReferenceCeilingLabel = "MEASURED on the reference machine (P-H8); not this machine's";

}  // namespace pelias::plan
