// provenance: original (the safetensors format is a public specification: u64 little-endian header length, a JSON
//   header mapping tensor names to {dtype, shape, data_offsets}, then the data; no code copied)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// safetensors.h - reads one .safetensors shard's header from a memory map with the same discipline as the GGUF reader:
// every offset checked against the file, every tensor named in its error. Shard folders (config.json, several shards)
// are handled by model/ later; this reads one file.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "io/json.h"
#include "os/file_map.h"
#include "pelias/status.h"

namespace pelias::io {

struct SafetensorsTensor {
  std::string name;
  std::string dtype;            // as written: "BF16", "F16", "F32", "I64", ...
  std::vector<std::uint64_t> shape;
  std::uint64_t begin = 0;      // relative to the data section
  std::uint64_t end = 0;
  std::uint64_t file_offset = 0;
  [[nodiscard]] std::uint64_t bytes() const { return end - begin; }
  [[nodiscard]] std::uint64_t elements() const;
};

// Bytes per element of a safetensors dtype, or 0 if unknown.
std::uint32_t safetensors_dtype_bytes(std::string_view dtype);

class SafetensorsReader {
 public:
  Status parse(const os::FileMap& map);
  [[nodiscard]] const std::vector<SafetensorsTensor>& tensors() const { return tensors_; }
  [[nodiscard]] const Json& metadata() const { return metadata_; }  // the "__metadata__" object, if any
  [[nodiscard]] std::uint64_t data_start() const { return data_start_; }
  [[nodiscard]] std::uint64_t header_bytes() const { return header_bytes_; }

 private:
  std::vector<SafetensorsTensor> tensors_;
  Json metadata_;
  std::uint64_t data_start_ = 0;
  std::uint64_t header_bytes_ = 0;
};

// One shard opened for reading: its memory map and its parsed header (a model folder holds several; ingest/ opens one
// per file named in the folder's index).
class SafetensorsShard {
 public:
  Status open(const std::string& path);
  [[nodiscard]] const SafetensorsReader& reader() const { return reader_; }
  // The tensor's data (its bytes are reader-checked to lie inside the file).
  [[nodiscard]] const std::uint8_t* data(const SafetensorsTensor& t) const { return map_.data() + t.file_offset; }

 private:
  os::FileMap map_;
  SafetensorsReader reader_;
};

}  // namespace pelias::io
