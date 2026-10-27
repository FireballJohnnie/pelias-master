// provenance: original (the file layout READ-SRC from llama.cpp ggml/src/gguf.cpp at commit
//   2ca15f5404760548c39e7b92bd43116a09414a1a, the writer's order and padding, so a file written here is laid out
//   byte for byte as the reference lays it out)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// gguf_writer.h - writes a GGUF version 3 file: header, key-value pairs (copied as encoded bytes, or new u32 values),
// tensor infos with dense offsets, the data section padded to the alignment, and each tensor's data padded with zeros
// to the alignment. Tensors are written in the order the caller gives; reference_order() gives the reference
// quantizer's order (tensors outside any block first, then block by block, each by name).
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "pelias/status.h"

namespace pelias::io {

struct GgufOutTensor {
  std::string name;
  std::uint64_t ne[4] = {1, 1, 1, 1};
  std::uint32_t type = 0;
  std::uint64_t bytes = 0;  // size of the data (without padding)
  // Dimensions written to the file. 0 = the reference quantizer's rule (ggml_n_dims: up to the last dimension larger
  // than 1). The reference converter writes the source array's own rank instead, so a [1, 2048] tensor keeps 2.
  std::uint32_t n_dims = 0;
};

// One key-value entry already encoded as the file stores it (key string, type, value).
using GgufKvBlob = std::vector<std::uint8_t>;
GgufKvBlob gguf_kv_u32(const std::string& key, std::uint32_t value);
GgufKvBlob gguf_kv_i32(const std::string& key, std::int32_t value);
GgufKvBlob gguf_kv_f32(const std::string& key, float value);
GgufKvBlob gguf_kv_bool(const std::string& key, bool value);
GgufKvBlob gguf_kv_str(const std::string& key, const std::string& value);
GgufKvBlob gguf_kv_arr_str(const std::string& key, const std::vector<std::string>& values);
GgufKvBlob gguf_kv_arr_i32(const std::string& key, const std::vector<std::int32_t>& values);
GgufKvBlob gguf_kv_arr_bool(const std::string& key, const std::vector<bool>& values);
// The key of an encoded entry (the string at its start).
std::string gguf_kv_key(const GgufKvBlob& blob);

// The reference quantizer's tensor order: by the block number in "blk.N." (names without one count as -1), then by
// name. Returns the indices of `names` in that order.
std::vector<std::size_t> reference_order(const std::vector<std::string>& names);

// Size of the header (everything before the data section, padded to the alignment).
std::uint64_t gguf_meta_bytes(const std::vector<GgufKvBlob>& kv, const std::vector<GgufOutTensor>& tensors, std::uint32_t alignment);
// Size of the whole file.
std::uint64_t gguf_file_bytes(const std::vector<GgufKvBlob>& kv, const std::vector<GgufOutTensor>& tensors, std::uint32_t alignment);

class GgufFileWriter {
 public:
  GgufFileWriter() = default;
  GgufFileWriter(const GgufFileWriter&) = delete;
  GgufFileWriter& operator=(const GgufFileWriter&) = delete;
  ~GgufFileWriter();

  // Creates `path` (refuses if it exists) and writes the header. Tensors' data must then follow in the same order.
  Status begin(const std::string& path, const std::vector<GgufKvBlob>& kv, const std::vector<GgufOutTensor>& tensors, std::uint32_t alignment);
  // Appends part of the current tensor's data. When a tensor's bytes are complete its padding is written.
  Status write(const std::uint8_t* data, std::uint64_t bytes);
  // Checks every tensor was written in full, flushes and closes.
  Status finish();
  void abandon();  // closes without finishing (the caller removes the partial file)

  [[nodiscard]] std::uint64_t written() const { return written_; }

 private:
  Status put(const void* p, std::uint64_t n);
  Status pad_to_alignment();
  std::FILE* f_ = nullptr;
  std::vector<GgufOutTensor> tensors_;
  std::uint32_t alignment_ = 32;
  std::size_t current_ = 0;           // tensor being written
  std::uint64_t current_done_ = 0;    // bytes of it written so far
  std::uint64_t written_ = 0;
  std::vector<char> buffer_;
};

}  // namespace pelias::io
