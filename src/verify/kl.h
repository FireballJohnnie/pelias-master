// provenance: original (the base-file layout and every statistic READ-SRC from llama.cpp tools/perplexity/perplexity.cpp
//   (kl_divergence, log_softmax, process_logits) at commit 2ca15f5404760548c39e7b92bd43116a09414a1a, read as a
//   specification so the numbers mean the same thing; no code copied)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// kl.h - drift against a base model's saved log-probabilities (masterwork 14.2, verify/kl). Reads the reference's
// KL-divergence base file (written by its perplexity tool with --kl-divergence-base) and accumulates, token by token,
// the same sums the reference does: NLL of the model and of the base, KL divergence, the probability of the right
// token, and whether the top token agrees. The report prints the reference's statistics in its layout, so a Pelias
// run and a reference run on the same base can be read side by side.
//
// Base file: "_logits_", u32 n_ctx, i32 n_vocab, i32 n_chunk, i32 tokens[n_ctx * n_chunk]; then per chunk,
// n_ctx - 1 - n_ctx/2 records (positions n_ctx/2 .. n_ctx-2) of nv = 2*((n_vocab+1)/2) + 4 u16: two floats (scale,
// min_log_prob) then the log-probabilities quantized to 16 bits.
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "pelias/status.h"

namespace pelias::verify {

class KlBaseFile {
 public:
  Status open(const std::string& path);
  [[nodiscard]] std::uint32_t n_ctx() const { return n_ctx_; }
  [[nodiscard]] std::int32_t n_vocab() const { return n_vocab_; }
  [[nodiscard]] std::int32_t n_chunk() const { return n_chunk_; }
  [[nodiscard]] std::int64_t record_u16() const { return 2 * ((static_cast<std::int64_t>(n_vocab_) + 1) / 2) + 4; }
  [[nodiscard]] int first() const { return static_cast<int>(n_ctx_ / 2); }               // first scored position
  [[nodiscard]] int scored() const { return static_cast<int>(n_ctx_) - 1 - first(); }   // positions per chunk
  [[nodiscard]] const std::vector<std::int32_t>& tokens() const { return tokens_; }
  // The next chunk's records (scored() * record_u16() values), in file order.
  Status read_chunk(std::vector<std::uint16_t>& out);

 private:
  std::ifstream in_;
  std::uint32_t n_ctx_ = 0;
  std::int32_t n_vocab_ = 0;
  std::int32_t n_chunk_ = 0;
  std::vector<std::int32_t> tokens_;
};

struct KlToken {  // one scored position, in the reference's own precisions (float, except the KL sum)
  float nll = 0, nll_base = 0, p_diff = 0;
  double kld = 0;
  bool same_top = false;
};

// One position: `logits` of the model (n_vocab floats), `base` its record in the base file, `tok` the next token.
KlToken kl_token(std::int32_t n_vocab, const float* logits, const std::uint16_t* base, std::int32_t tok);

struct KlStats {
  double sum_nll = 0, sum_nll2 = 0, sum_nll_base = 0, sum_nll_base2 = 0, sum_nll_nll_base = 0;
  double sum_kld = 0, sum_kld2 = 0, sum_p_diff = 0, sum_p_diff2 = 0, sum_p_diff4 = 0;
  float max_p_diff = 0;
  std::uint64_t n_same_top = 0, count = 0;
  std::vector<float> kld_values, p_diff_values;
  void add(const KlToken& t);
};

// The reference's per-chunk line: chunk, PPL, ln(PPL(Q)/PPL(base)), KL divergence, delta-p RMS, same top p.
std::string kl_chunk_line(int chunk, const KlStats& s);
std::string kl_chunk_header();
// The reference's closing statistics (perplexity, KL divergence, token probability), in its layout.
std::string kl_report(const KlStats& s);

}  // namespace pelias::verify
