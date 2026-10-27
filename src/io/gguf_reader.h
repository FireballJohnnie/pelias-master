// provenance: original (the file layout READ-SRC from the comment block of llama.cpp ggml/include/gguf.h at commit
//   9bc6578a574d1dda88610ad8532247d74779bd9e; the validation rules follow its reader's as a specification)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// gguf_reader.h - reads a GGUF file's header, key-value metadata and tensor table from a memory map, checking every
// byte it touches against the file's size and every value against the format's rules, so that a damaged file is refused
// with the tensor or key name and the byte offset of the problem (5.1, gate PG2). The tensor data itself is not read
// here: each tensor gets a byte range inside the file that the caller may map or stream.
//
// Format (version 3; version 2 accepted, 1 refused): "GGUF", u32 version, i64 n_tensors, i64 n_kv; then n_kv of
// (string key, u32 type, value); then n_tensors of (string name, u32 n_dims, u64 ne[n_dims], u32 type, u64 offset);
// then the data blob at the next multiple of the alignment (general.alignment, default 32). Strings are u64 length +
// bytes. Arrays are u32 element type, u64 count, elements.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "os/file_map.h"
#include "pelias/status.h"

namespace pelias::io {

enum class GgufValueType : std::uint32_t {
  u8 = 0, i8 = 1, u16 = 2, i16 = 3, u32 = 4, i32 = 5, f32 = 6, boolean = 7, string = 8, array = 9, u64 = 10, i64 = 11, f64 = 12,
};

struct GgufValue {
  GgufValueType type = GgufValueType::u8;
  // Scalars are stored widened; strings as bytes; arrays keep their element type, count and the byte range of their
  // elements in the file (strings inside arrays are read on demand through the reader).
  std::int64_t i = 0;
  std::uint64_t u = 0;
  double f = 0.0;
  bool b = false;
  std::string s;
  GgufValueType array_type = GgufValueType::u8;
  std::uint64_t array_count = 0;
  std::uint64_t array_offset = 0;  // byte offset of the first element in the file
};

struct GgufKv {
  std::string key;
  GgufValue value;
  std::uint64_t file_begin = 0;  // byte range of the whole entry (key, type, value) in the file, so a writer can copy it
  std::uint64_t file_end = 0;
};

struct GgufTensor {
  std::string name;
  std::uint32_t n_dims = 0;
  std::uint64_t ne[4] = {1, 1, 1, 1};
  std::uint32_t type = 0;          // quant::Type id as written in the file
  std::uint64_t data_offset = 0;   // relative to the data section start, as written
  std::uint64_t file_offset = 0;   // absolute byte offset in the file (data_start + data_offset)
  std::uint64_t bytes = 0;         // size of the tensor's data
  [[nodiscard]] std::uint64_t elements() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};

class GgufReader {
 public:
  // Parses the header, metadata and tensor table of an open map. On failure the Status names what is wrong and where.
  // With require_data = false the tensor table is checked for consistency but not against the file's size, so a
  // header alone (the first bytes of a file) can be inspected; file_offset is still filled in.
  Status parse(const os::FileMap& map, bool require_data = true);

  [[nodiscard]] std::uint32_t version() const { return version_; }
  [[nodiscard]] std::uint32_t alignment() const { return alignment_; }
  [[nodiscard]] std::uint64_t data_start() const { return data_start_; }
  [[nodiscard]] std::uint64_t data_bytes() const { return data_bytes_; }  // padded sum of every tensor
  [[nodiscard]] std::uint64_t file_bytes() const { return file_bytes_; }
  [[nodiscard]] const std::vector<GgufKv>& kv() const { return kv_; }
  [[nodiscard]] const std::vector<GgufTensor>& tensors() const { return tensors_; }

  [[nodiscard]] const GgufValue* find(std::string_view key) const;
  [[nodiscard]] const GgufTensor* find_tensor(std::string_view name) const;
  // Convenience readers; `found` false when the key is absent or of another type.
  [[nodiscard]] std::string get_string(std::string_view key, bool* found = nullptr) const;
  [[nodiscard]] std::uint64_t get_uint(std::string_view key, bool* found = nullptr) const;  // any integer type, widened
  [[nodiscard]] double get_float(std::string_view key, bool* found = nullptr) const;
  // The i-th string of a string array (bounds-checked; empty and `found` false otherwise).
  [[nodiscard]] std::string array_string(const GgufValue& v, std::uint64_t i, bool* found = nullptr) const;
  // Every element of a string array, read in one pass (the i-th accessor above walks from the start each call).
  [[nodiscard]] Status array_strings(const GgufValue& v, std::vector<std::string>& out) const;
  // Every element of an integer array (any integer element type, or bool as 0 / 1, widened to int64).
  [[nodiscard]] Status array_ints(const GgufValue& v, std::vector<std::int64_t>& out) const;

 private:
  const os::FileMap* map_ = nullptr;
  std::uint32_t version_ = 0;
  std::uint32_t alignment_ = 32;
  std::uint64_t data_start_ = 0;
  std::uint64_t data_bytes_ = 0;
  std::uint64_t file_bytes_ = 0;
  std::vector<GgufKv> kv_;
  std::vector<GgufTensor> tensors_;
};

}  // namespace pelias::io
