// provenance: original (see sampler.h)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "model/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace pelias::model {

std::array<std::uint32_t, 4> philox4x32_10(std::array<std::uint32_t, 4> c, std::array<std::uint32_t, 2> k) {
  constexpr std::uint64_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
  constexpr std::uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
  for (int r = 0; r < 10; ++r) {
    const std::uint64_t p0 = M0 * c[0], p1 = M1 * c[2];
    const auto hi0 = static_cast<std::uint32_t>(p0 >> 32), lo0 = static_cast<std::uint32_t>(p0);
    const auto hi1 = static_cast<std::uint32_t>(p1 >> 32), lo1 = static_cast<std::uint32_t>(p1);
    c = {hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0};
    k[0] += W0;
    k[1] += W1;
  }
  return c;
}

Status check_sampler_config(const SamplerConfig& c) {
  auto bad = [](const char* what) { return Status::fail(Code::invalid_argument, what); };
  if (!(c.temperature >= 0.0f && c.temperature <= 2.0f)) return bad("the temperature is 0 (greedy) to 2");
  if (c.top_k < 0 || c.top_k > 1000) return bad("top-k is 0 (off) to 1000");
  if (!(c.top_p >= 0.01f && c.top_p <= 1.0f)) return bad("top-p is 0.01 to 1 (off)");
  if (!(c.min_p >= 0.0f && c.min_p <= 1.0f)) return bad("min-p is 0 (off) to 1");
  if (!(c.repeat_penalty >= 1.0f && c.repeat_penalty <= 2.0f)) return bad("the repeat penalty is 1 (off) to 2");
  if (!(c.presence_penalty >= -2.0f && c.presence_penalty <= 2.0f)) return bad("the presence penalty is -2 to 2");
  if (!(c.frequency_penalty >= -2.0f && c.frequency_penalty <= 2.0f)) return bad("the frequency penalty is -2 to 2");
  if (c.repeat_last_n < -1 || c.repeat_last_n > 65536) return bad("repeat-last-n is -1 (all), 0 (none) or up to 65536");
  return Status::good();
}

std::int32_t Sampler::sample(const float* logits, std::int64_t n_vocab, const std::vector<std::int32_t>& history) {
  const auto n = static_cast<std::size_t>(n_vocab);
  work_.assign(logits, logits + n);
  // penalties (llama.cpp's penalties sampler): repeat divides a positive logit and multiplies a negative one; then
  // count x frequency + presence is subtracted, for every token seen in the last repeat_last_n
  const bool pen = c_.repeat_penalty != 1.0f || c_.presence_penalty != 0.0f || c_.frequency_penalty != 0.0f;
  if (pen && c_.repeat_last_n != 0 && !history.empty()) {
    const std::size_t take = c_.repeat_last_n < 0 ? history.size() : std::min(history.size(), static_cast<std::size_t>(c_.repeat_last_n));
    std::unordered_map<std::int32_t, int> count;
    for (std::size_t i = history.size() - take; i < history.size(); ++i) ++count[history[i]];
    for (const auto& [id, cnt] : count) {
      if (id < 0 || static_cast<std::size_t>(id) >= n) continue;
      float& l = work_[static_cast<std::size_t>(id)];
      if (c_.repeat_penalty != 1.0f) l = l > 0.0f ? l / c_.repeat_penalty : l * c_.repeat_penalty;
      l -= static_cast<float>(cnt) * c_.frequency_penalty + c_.presence_penalty;
    }
  }
  // greedy: the largest logit, the lowest id on a tie
  if (c_.temperature <= 0.0f) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i)
      if (work_[i] > work_[best]) best = i;
    return static_cast<std::int32_t>(best);
  }
  // candidates by logit, largest first (ties by id), cut to top-k
  ids_.resize(n);
  std::iota(ids_.begin(), ids_.end(), 0);
  auto order = [this](std::int32_t a, std::int32_t b) {
    const float la = work_[static_cast<std::size_t>(a)], lb = work_[static_cast<std::size_t>(b)];
    return la > lb || (la == lb && a < b);
  };
  std::size_t keep = n;
  if (c_.top_k > 0 && static_cast<std::size_t>(c_.top_k) < n) {
    keep = static_cast<std::size_t>(c_.top_k);
    std::partial_sort(ids_.begin(), ids_.begin() + static_cast<std::ptrdiff_t>(keep), ids_.end(), order);
  } else {
    std::sort(ids_.begin(), ids_.end(), order);
  }
  // top-p and min-p on the probabilities before temperature (llama.cpp's chain order: top-k, top-p, min-p, temperature)
  const float top = work_[static_cast<std::size_t>(ids_[0])];
  std::vector<double> p(keep);
  double sum = 0.0;
  for (std::size_t i = 0; i < keep; ++i) sum += p[i] = std::exp(static_cast<double>(work_[static_cast<std::size_t>(ids_[i])] - top));
  if (c_.top_p < 1.0f) {
    double cum = 0.0;
    std::size_t cut = keep;
    for (std::size_t i = 0; i < keep; ++i) {
      cum += p[i] / sum;
      if (cum >= static_cast<double>(c_.top_p)) {
        cut = i + 1;
        break;
      }
    }
    keep = cut;
  }
  if (c_.min_p > 0.0f) {
    std::size_t cut = 1;  // p[0] is the largest (exp(0) = 1)
    while (cut < keep && p[cut] >= static_cast<double>(c_.min_p)) ++cut;
    keep = cut;
  }
  // temperature, then one draw by the cumulative probability
  double tsum = 0.0;
  for (std::size_t i = 0; i < keep; ++i)
    tsum += p[i] = std::exp(static_cast<double>(work_[static_cast<std::size_t>(ids_[i])] - top) / static_cast<double>(c_.temperature));
  const std::array<std::uint32_t, 4> r = philox4x32_10({static_cast<std::uint32_t>(draw_), static_cast<std::uint32_t>(draw_ >> 32), 0u, 0u},
                                                      {static_cast<std::uint32_t>(c_.seed), static_cast<std::uint32_t>(c_.seed >> 32)});
  ++draw_;
  const double u = static_cast<double>((static_cast<std::uint64_t>(r[0]) << 21) ^ (r[1] >> 11)) / 9007199254740992.0 * tsum;  // 53 bits in [0, tsum)
  double cum = 0.0;
  for (std::size_t i = 0; i < keep; ++i) {
    cum += p[i];
    if (u < cum) return ids_[i];
  }
  return ids_[keep - 1];
}

}  // namespace pelias::model
