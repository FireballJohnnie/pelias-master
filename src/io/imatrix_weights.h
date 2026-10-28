// provenance: original (what is read and how it is normalised READ-SRC from llama.cpp common/imatrix-loader.cpp
//   (common_imatrix_load) and tools/quantize/quantize.cpp (load_imatrix) at commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a, read as a specification so `pelias quantize --imatrix` weighs every column
//   as llama-quantize --imatrix does)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// imatrix_weights.h - an importance-matrix file (GGUF, as `pelias imatrix` and llama-imatrix write it) read into the
// per-column weights the quantizer uses: for every weight W, in_sum2 divided by the token count of its matrix (one
// matrix per expert for a stack of experts; a matrix that saw no token gets weight 1 for every column). The counts are
// rounded to whole numbers first, as the reference does. The legacy (non-GGUF) format is refused.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "pelias/status.h"

namespace pelias::io {

struct ImatrixWeights {
  std::map<std::string, std::vector<float>> weights;  // W -> [matrix][column]
  std::vector<std::string> datasets;
  std::uint32_t chunk_count = 0;
  std::uint32_t chunk_size = 0;
  // pelias.rot32.tensors of the file: the weights whose inputs were rotated (per 32, Walsh-Hadamard) when collected
  bool has_rot32 = false;
  std::vector<std::string> rot32;
};

// Reads `path`. Refuses a file without imatrix.datasets / chunk_count / chunk_size (the reference's "missing imatrix
// metadata"), mismatched or non-F32 sums and counts, and any non-finite weight.
Result<ImatrixWeights> read_imatrix_weights(const std::string& path);

}  // namespace pelias::io
