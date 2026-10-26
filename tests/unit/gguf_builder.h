// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// gguf_builder.h - writes small GGUF files in memory for the tests (the product's own writer, io/gguf_writer, comes at
// step 3; this one is test-only and deliberately simple).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "os/file_map.h"

namespace ptest {

struct TensorSpec {
  std::string name;
  std::vector<std::uint64_t> ne;
  std::uint32_t type;
};

class GgufBuilder {
 public:
  GgufBuilder& version(std::uint32_t v) { version_ = v; return *this; }
  GgufBuilder& alignment(std::uint32_t a) { alignment_ = a; return *this; }
  GgufBuilder& kv_string(const std::string& k, const std::string& v);
  GgufBuilder& kv_u32(const std::string& k, std::uint32_t v);
  GgufBuilder& kv_f32(const std::string& k, float v);
  GgufBuilder& kv_bool(const std::string& k, bool v);
  GgufBuilder& kv_string_array(const std::string& k, const std::vector<std::string>& v);
  GgufBuilder& kv_i32_array(const std::string& k, const std::vector<std::int32_t>& v);
  GgufBuilder& tensor(const std::string& name, std::vector<std::uint64_t> ne, std::uint32_t type);
  // Serializes; data bytes are a deterministic pattern. header_end receives the byte where the data section starts.
  std::vector<std::uint8_t> build(std::uint64_t* data_start = nullptr) const;

 private:
  std::uint32_t version_ = 3;
  std::uint32_t alignment_ = 32;
  std::vector<std::uint8_t> kv_;
  std::uint64_t n_kv_ = 0;
  std::vector<TensorSpec> tensors_;
};

// Writes bytes to a file in the test data directory and maps it. Returns false if the file could not be written.
bool write_and_map(const std::string& path, const std::vector<std::uint8_t>& bytes, pelias::os::FileMap& map);

}  // namespace ptest
