// provenance: original (what is collected and the file layout READ-SRC from llama.cpp tools/imatrix/imatrix.cpp
//   (collect_imatrix, save_imatrix) and common/imatrix-loader.cpp at commit 2ca15f5404760548c39e7b92bd43116a09414a1a,
//   read as a specification so the reference quantizer and Pelias's read the same file)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// imatrix.h - the importance matrix: for every weight matrix the reference pass multiplies, the sum over tokens of each
// input column's squared activation (and of the activation itself), with the number of tokens it saw. For a stack of
// experts there is one row of sums and one count per expert, added to only when the router picked that expert.
// The quantizer weighs each column's rounding error by sum / count, so columns that carry large activations are kept
// more exactly.
//
// File (GGUF): general.type "imatrix"; imatrix.datasets (string array), imatrix.chunk_count, imatrix.chunk_size (u32);
// per weight W, tensors "W.in_sum2" (F32, [columns, matrices]), "W.counts" (F32, [1, matrices]) and "W.in_sum" (F32,
// [columns, matrices]), W in name order. When weights stored rotated were collected (their inputs rotated per 32 values
// first, so the sums are of the rotated activations), pelias.rot32.kind and pelias.rot32.tensors name them. What is collected matches the reference tool's defaults: every "blk." matrix
// multiply over at least 16 tokens (output.weight left out), sums in float, experts slot by slot then token by token.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "pelias/status.h"

namespace pelias::model {

struct ImatrixEntry {
  std::int64_t columns = 0;
  std::vector<float> sum2, sum;      // [matrix][column]
  std::vector<std::uint64_t> counts;  // [matrix]
};

class ImatrixCollector {
 public:
  // The entry for weight `name` with `columns` inputs and `matrices` matrices, created empty on first use.
  ImatrixEntry& entry(const std::string& name, std::int64_t columns, std::int64_t matrices);
  // n_tok activation rows (ldx floats apart) into matrix 0 of a dense weight; fewer than 16 rows are not collected.
  void add_dense(const std::string& name, const float* x, std::int64_t n_tok, std::int64_t ldx, std::int64_t columns);
  // one activation row into matrix `m` of an entry (experts; the caller keeps the reference's order)
  static void add_row(ImatrixEntry& e, std::int64_t m, const float* x);
  // Records that `name`'s activations were collected after the 32-value Walsh-Hadamard rotation (the weight is stored
  // rotated); the names are written as pelias.rot32.tensors so a quantizer can check it rotates the same weights.
  void mark_rotated(const std::string& name) { rotated_[name] = true; }
  [[nodiscard]] std::vector<std::string> rotated() const;
  void finish_chunk() { ++chunks_; }
  [[nodiscard]] std::uint32_t chunks() const { return chunks_; }
  [[nodiscard]] const std::map<std::string, ImatrixEntry>& entries() const { return entries_; }
  // Writes the GGUF imatrix file (refuses an existing path).
  Status write(const std::string& path, const std::vector<std::string>& datasets, std::uint32_t chunk_size) const;

 private:
  std::map<std::string, ImatrixEntry> entries_;
  std::map<std::string, bool> rotated_;
  std::uint32_t chunks_ = 0;
};

}  // namespace pelias::model
