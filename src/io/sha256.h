// provenance: original (the algorithm is FIPS 180-4, a public standard; no code copied)
// Copyright (c) 2026 John Lipka and the Pelias contributors
//
// sha256.h - SHA-256 for the model key (5.2) and hash proofs (P-X1). Streaming: feed any number of chunks.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace pelias::io {

class Sha256 {
 public:
  Sha256();
  void update(const void* data, std::size_t bytes);
  std::array<std::uint8_t, 32> finish();  // after finish() the object must not be updated again
  static std::string hex(const std::array<std::uint8_t, 32>& d);
  // Convenience: hash of a whole buffer, as 64 lower-case hex characters.
  static std::string hex_of(const void* data, std::size_t bytes);

 private:
  void block(const std::uint8_t* p);
  std::uint32_t h_[8];
  std::uint8_t buf_[64];
  std::size_t buf_len_ = 0;
  std::uint64_t total_ = 0;
};

}  // namespace pelias::io
