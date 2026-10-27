// provenance: original (the procedure follows llama.cpp src/llama-quant.cpp at commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a as a specification: metadata copied with the quantization version and
//   file type re-set at the end, tensors in its order, each either copied unchanged or converted to f32 and quantized)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// requantize.h - `pelias quantize` for a GGUF input (path Q from GGUF, 5.5 and 9.2): reads a GGUF through the memory
// map, chooses every tensor's type by the stock rule of the chosen mix (quant/type_rules), and writes a new GGUF next to
// nothing it would overwrite. The output is written under "<out>.partial" and renamed only when complete.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "io/gguf_reader.h"
#include "os/file_map.h"
#include "pelias/status.h"
#include "quant/type_rules.h"

namespace pelias::io {

struct RequantOptions {
  quant::FileType ftype = quant::FileType::q4_k_m;
  int threads = 0;  // 0 = all hardware threads
};

struct RequantReport {
  std::uint64_t tensors = 0, quantized = 0, copied = 0;
  std::uint64_t bytes_in = 0, bytes_out = 0;
  double seconds = 0;
  std::uint64_t by_type[64] = {};  // output tensors per type id (ids below 64)
};

// Called after each tensor: (index, count, name, output type id).
using RequantProgress = std::function<void(std::size_t, std::size_t, const std::string&, std::uint32_t)>;

Result<RequantReport> requantize(const os::FileMap& map, const GgufReader& in, const std::string& out_path, const RequantOptions& opt,
                                 const RequantProgress& progress = {});

}  // namespace pelias::io
