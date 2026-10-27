// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// gguf_compare.h - compares two GGUF files for gate P-E6 (Pelias's quantization against the reference's on every
// tensor) and P-X6 (whole files): tensor by tensor by name (type, shape, data bytes), the metadata entries in order,
// and the files byte for byte.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "io/gguf_reader.h"
#include "os/file_map.h"

namespace pelias::io {

struct GgufCompareReport {
  std::uint64_t tensors_a = 0, tensors_b = 0;
  std::uint64_t identical = 0;       // same name, type, shape and data bytes
  std::uint64_t only_in_a = 0, only_in_b = 0;
  std::uint64_t type_differs = 0, shape_differs = 0, data_differs = 0;
  bool kv_identical = false;         // same entries, same order, same encoded bytes
  bool order_identical = false;      // tensors in the same order
  bool files_identical = false;      // same size and every byte equal
  std::uint64_t first_file_difference = 0;  // byte offset, when the files differ
  std::vector<std::string> notes;    // the first differences, one line each
};

GgufCompareReport compare_gguf(const os::FileMap& ma, const GgufReader& a, const os::FileMap& mb, const GgufReader& b, std::size_t max_notes = 20);

}  // namespace pelias::io
