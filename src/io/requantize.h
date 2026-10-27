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
#include "io/gguf_writer.h"
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

// A tensor of the source, as a GGUF of it would describe it.
struct SourceTensor {
  std::string name;
  std::uint64_t ne[4] = {1, 1, 1, 1};
  std::uint32_t type = 0;   // type id of the source data
  std::uint64_t bytes = 0;
};
// Fills `out` with `n` rows (ne[0] elements each) of tensor `i`, starting at row `row0`, in the source type.
using ReadRows = std::function<Status(std::size_t i, std::uint64_t row0, std::uint64_t n, std::uint8_t* out)>;

// Anything that can be read as a GGUF: its metadata entries in order as encoded (booleans already 0 or 1), the facts
// the type rule needs, its tensors in their stored order, and a row reader. A GGUF file is one such source; a
// safetensors folder being converted (convert/) is another.
struct QuantSource {
  std::vector<GgufKvBlob> kv;
  quant::ModelFacts facts;
  std::vector<SourceTensor> tensors;
  ReadRows read;
};

// The reference quantizer's procedure on any source: metadata copied in order without the quantization version, the
// file type and the split keys, those two re-set at the end; tensors in the reference order; each tensor either
// copied unchanged or converted to f32 and quantized, by the mix's stock rule.
Result<RequantReport> quantize_source(const QuantSource& src, const std::string& out_path, const RequantOptions& opt,
                                      const RequantProgress& progress = {});

Result<RequantReport> requantize(const os::FileMap& map, const GgufReader& in, const std::string& out_path, const RequantOptions& opt,
                                 const RequantProgress& progress = {});

// Writes `out` under "<out_path>.partial", calling fill(j, buffer) for each tensor's data in turn (whole tensors or
// pieces, in order), checks the free space first, and renames when complete. Used by both paths.
Status write_gguf_file(const std::string& out_path, const std::vector<GgufKvBlob>& kv, const std::vector<GgufOutTensor>& out,
                       const std::function<Status(std::size_t j, GgufFileWriter& w)>& fill);

}  // namespace pelias::io
