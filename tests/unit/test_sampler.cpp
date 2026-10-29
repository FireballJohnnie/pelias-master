// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
// test_sampler.cpp - GP3-07: the Philox generator against its published known answers, and the sampler's rules.
#include <cmath>
#include <cstdio>
#include <vector>

#include "model/sampler.h"
#include "pelias_test.h"

using pelias::model::Sampler;
using pelias::model::SamplerConfig;

TEST(philox4x32_10_known_answers) {
  // Random123 kat_vectors (philox4x32 10 rounds): zero, all-ones and pi-digit inputs
  using A4 = std::array<std::uint32_t, 4>;
  CHECK((pelias::model::philox4x32_10({0, 0, 0, 0}, {0, 0}) == A4{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}));
  CHECK((pelias::model::philox4x32_10({0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu}) ==
         A4{0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}));
  CHECK((pelias::model::philox4x32_10({0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u}) ==
         A4{0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}));
}

TEST(sampler_greedy_penalties_and_filters) {
  const std::vector<float> lg{1.0f, 3.0f, 2.5f, -1.0f, 3.0f};
  SamplerConfig g;
  g.temperature = 0.0f;
  CHECK(Sampler(g).sample(lg.data(), 5, {}) == 1);  // a tie goes to the lowest id
  // repeat penalty 2 on token 1 (3 -> 1.5) and 4 (3 -> 1.5): token 2 wins; a negative logit is multiplied
  g.repeat_penalty = 2.0f;
  CHECK(Sampler(g).sample(lg.data(), 5, {1, 4}) == 2);
  // the window: only the last token (4) is penalized, so 1 wins again
  g.repeat_last_n = 1;
  CHECK(Sampler(g).sample(lg.data(), 5, {1, 4}) == 1);
  // presence and frequency: 1 seen twice (3 - 2 x 0.4 - 0.1 = 2.1), so 4 wins
  SamplerConfig pf;
  pf.temperature = 0.0f;
  pf.frequency_penalty = 0.4f;
  pf.presence_penalty = 0.1f;
  CHECK(Sampler(pf).sample(lg.data(), 5, {1, 1}) == 4);
  // top-k 1 is greedy whatever the temperature and seed
  SamplerConfig k1;
  k1.top_k = 1;
  k1.temperature = 2.0f;
  for (std::uint64_t seed = 0; seed < 20; ++seed) {
    k1.seed = seed;
    CHECK(Sampler(k1).sample(lg.data(), 5, {}) == 1);
  }
  // min-p 0.5: only tokens within log(2) of the top survive (ids 1, 4, 2: 2.5 > 3 - 0.693)
  SamplerConfig mp;
  mp.top_k = 0;
  mp.top_p = 1.0f;
  mp.min_p = 0.5f;
  Sampler smp(mp);
  for (int i = 0; i < 300; ++i) {
    const std::int32_t t = smp.sample(lg.data(), 5, {});
    CHECK(t == 1 || t == 2 || t == 4);
  }
}

TEST(sampler_draws_follow_the_probabilities_and_the_seed) {
  // three tokens with probabilities 0.5, 0.3, 0.2 (temperature 1, no filters)
  const std::vector<float> lg{std::log(0.5f), std::log(0.3f), std::log(0.2f)};
  SamplerConfig c;
  c.top_k = 0;
  c.top_p = 1.0f;
  Sampler s1(c), s2(c);
  int hits[3] = {0, 0, 0};
  const int n = 20000;
  bool same = true;
  for (int i = 0; i < n; ++i) {
    const std::int32_t a = s1.sample(lg.data(), 3, {}), b = s2.sample(lg.data(), 3, {});
    same = same && a == b;
    ++hits[a];
  }
  CHECK(same);  // the same seed gives the same tokens
  CHECK(s1.draws() == static_cast<std::uint64_t>(n));
  // within 4 standard errors (sd of a share p over n draws: sqrt(p (1 - p) / n), about 0.0035 at p = 0.5)
  const double want[3] = {0.5, 0.3, 0.2};
  for (int t = 0; t < 3; ++t) {
    const double got = hits[t] / static_cast<double>(n), se = std::sqrt(want[t] * (1 - want[t]) / n);
    std::printf("    token %d: %.4f of draws (probability %.1f)\n", t, got, want[t]);
    CHECK(std::fabs(got - want[t]) < 4 * se);
  }
  // another seed gives another sequence
  c.seed = 7;
  Sampler s3(c);
  int differ = 0;
  Sampler s5(SamplerConfig{c.temperature, 0, 1.0f, 0.0f, 1.0f, 64, 0.0f, 0.0f, 20261019u});
  for (int i = 0; i < 200; ++i) differ += s3.sample(lg.data(), 3, {}) != s5.sample(lg.data(), 3, {});
  CHECK(differ > 20);
  // top-p 0.6: the first two (0.5 + 0.3 >= 0.6) survive
  SamplerConfig tp;
  tp.top_k = 0;
  tp.top_p = 0.6f;
  Sampler stp(tp);
  for (int i = 0; i < 300; ++i) CHECK(stp.sample(lg.data(), 3, {}) != 2);
  // the dictionary ranges
  SamplerConfig bad;
  bad.temperature = 2.5f;
  CHECK(!pelias::model::check_sampler_config(bad));
  CHECK(pelias::model::check_sampler_config(SamplerConfig{}));
}
