// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// probe_constants.h - numbers the planner uses before a machine's own probes have run, each with its label and source.
// Every value here is replaced by a measurement when one exists (Section 4.1, 17.2); nothing is a guess.
#pragma once

#include <cstdint>

namespace pelias::plan {

inline constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;

// Working set of SERVING (decode and prefill) beyond the weights: KV cache, recurrent state, activations, scratch.
// DERIVED upper bound from the tier build's MEASURED peak on the served file (masterwork 17.4, 5.3):
//   whole-card peak at a 7.5 GiB bank and -ub 2048 (the reference profile, C9)   10.69 GiB  = 10,946.6 MiB  MEASURED
//   minus what Windows held before the process started (P-H8: 914 / 1,031 MiB)   -   914 MiB                MEASURED
//   minus the bank                                                               - 7,680 MiB
//   minus the weights certainly on the card (head 397.9 + spine 905.0 + MTP head 23.4; the embedding,
//   272.8 MiB, is left in: where that run kept it is not recorded)               - 1,326.3 MiB              DERIVED
//   = 1,026.3 MiB, rounded up to 1.00 GiB.
// Each choice makes the figure larger, not smaller (the smaller Windows reading; a bank of exactly 7.5 GiB, although
// the run's cap was 7,800 MiB; the embedding counted as working set if it was on the card; the llama.cpp process's
// own CUDA context counted
// as working set, while the planner's ceiling already excludes Pelias's context), so it is an upper bound for that
// run. It is llama.cpp's working set, not Pelias's: the engine's own peak (P-E4, step 10) replaces it.
// The 2.50 GiB of masterwork 8.8 is the TRAINING working set (8,192-token window, up to 32 k tokens per step); the
// serving planner used it by mistake until 2026-10-26 19:0x.
inline constexpr std::uint64_t kServingWorkingSetBytes = kGiB;  // 1.00 GiB, DERIVED upper bound (17.4, P-H8)
inline constexpr const char* kServingWorkingSetLabel =
    "DERIVED upper bound from the tier build's measured peak (17.4, P-H8); P-E4 measures the engine's own";

// Reference machine ceiling, MEASURED (P-H8, D26/D28): 10.7 GiB. Used only when the user states no ceiling and no
// device is present (the plan then says so).
inline constexpr std::uint64_t kReferenceCeilingBytes = std::uint64_t{11488709837};  // 10.7 GiB
inline constexpr const char* kReferenceCeilingLabel = "MEASURED on the reference machine (P-H8); not this machine's";

}  // namespace pelias::plan
