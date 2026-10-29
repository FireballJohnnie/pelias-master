// provenance: original (the penalty and filter rules follow llama.cpp src/llama-sampler.cpp at commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a, read as a specification; Philox4x32-10 after Salmon et al., "Parallel random
//   numbers: as easy as 1, 2, 3", SC 2011)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// sampler.h - picks the next token from one row of logits (GP3-07). Order: penalties on the recent tokens, then greedy
// (temperature 0) or top-k, top-p, min-p, temperature and one draw. The draw is Philox4x32-10 keyed by the seed with the
// draw number as the counter, so a seed gives the same tokens on any machine and thread count. Defaults are the
// parameter dictionary's (docs/inputs/pelias-parameter-dictionary.md, sampling.*); repeat_last_n is not in the dictionary
// and takes llama.cpp's 64.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "pelias/status.h"

namespace pelias::model {

// Philox4x32-10: the counter (four 32-bit words) under the key (two words) after ten rounds.
std::array<std::uint32_t, 4> philox4x32_10(std::array<std::uint32_t, 4> ctr, std::array<std::uint32_t, 2> key);

struct SamplerConfig {
  float temperature = 1.0f;        // 0 = greedy
  int top_k = 20;                  // 0 = off
  float top_p = 0.95f;             // 1 = off
  float min_p = 0.0f;              // 0 = off
  float repeat_penalty = 1.0f;     // 1 = off
  int repeat_last_n = 64;          // the recent tokens the penalties look at (0 = none, -1 = all)
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  std::uint64_t seed = 20261019u;
};

// The range checks of the dictionary (temperature 0..2, top-k 0..1000, top-p 0.01..1, min-p 0..1, repeat 1..2,
// presence and frequency -2..2, repeat_last_n -1..65536).
Status check_sampler_config(const SamplerConfig& c);

class Sampler {
 public:
  explicit Sampler(const SamplerConfig& c) : c_(c) {}
  // The next token for `logits` (n_vocab values; not changed) given the tokens so far (prompt and output), which the
  // penalties read.
  std::int32_t sample(const float* logits, std::int64_t n_vocab, const std::vector<std::int32_t>& history);
  [[nodiscard]] std::uint64_t draws() const { return draw_; }

 private:
  SamplerConfig c_;
  std::uint64_t draw_ = 0;  // Philox counter: one per sampled (non-greedy) token
  std::vector<float> work_;
  std::vector<std::int32_t> ids_;
};

}  // namespace pelias::model
