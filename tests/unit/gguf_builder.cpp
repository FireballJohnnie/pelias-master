// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "unit/gguf_builder.h"

#include <cstdio>
#include <cstring>

#include "quant/block_types.h"

namespace ptest {
namespace {
template <typename T>
void put(std::vector<std::uint8_t>& v, T x) {
  std::uint8_t b[sizeof(T)];
  std::memcpy(b, &x, sizeof(T));
  for (std::size_t i = 0; i < sizeof(T); ++i) v.push_back(b[i]);
}
void put_str(std::vector<std::uint8_t>& v, const std::string& s) {
  put<std::uint64_t>(v, s.size());
  v.insert(v.end(), s.begin(), s.end());
}
}  // namespace

GgufBuilder& GgufBuilder::kv_string(const std::string& k, const std::string& v) {
  put_str(kv_, k); put<std::uint32_t>(kv_, 8); put_str(kv_, v); ++n_kv_; return *this;
}
GgufBuilder& GgufBuilder::kv_u32(const std::string& k, std::uint32_t v) {
  put_str(kv_, k); put<std::uint32_t>(kv_, 4); put<std::uint32_t>(kv_, v); ++n_kv_; return *this;
}
GgufBuilder& GgufBuilder::kv_f32(const std::string& k, float v) {
  put_str(kv_, k); put<std::uint32_t>(kv_, 6); put<float>(kv_, v); ++n_kv_; return *this;
}
GgufBuilder& GgufBuilder::kv_bool(const std::string& k, bool v) {
  put_str(kv_, k); put<std::uint32_t>(kv_, 7); put<std::uint8_t>(kv_, v ? 1 : 0); ++n_kv_; return *this;
}
GgufBuilder& GgufBuilder::kv_string_array(const std::string& k, const std::vector<std::string>& v) {
  put_str(kv_, k); put<std::uint32_t>(kv_, 9); put<std::uint32_t>(kv_, 8); put<std::uint64_t>(kv_, v.size());
  for (const auto& s : v) put_str(kv_, s);
  ++n_kv_; return *this;
}
GgufBuilder& GgufBuilder::kv_i32_array(const std::string& k, const std::vector<std::int32_t>& v) {
  put_str(kv_, k); put<std::uint32_t>(kv_, 9); put<std::uint32_t>(kv_, 5); put<std::uint64_t>(kv_, v.size());
  for (const auto x : v) put<std::int32_t>(kv_, x);
  ++n_kv_; return *this;
}
GgufBuilder& GgufBuilder::tensor(const std::string& name, std::vector<std::uint64_t> ne, std::uint32_t type) {
  tensors_.push_back({name, std::move(ne), type});
  return *this;
}

std::vector<std::uint8_t> GgufBuilder::build(std::uint64_t* data_start) const {
  std::vector<std::uint8_t> v;
  v.insert(v.end(), {'G', 'G', 'U', 'F'});
  put<std::uint32_t>(v, version_);
  put<std::int64_t>(v, static_cast<std::int64_t>(tensors_.size()));
  put<std::int64_t>(v, static_cast<std::int64_t>(n_kv_ + (alignment_ != 32 ? 1 : 0)));
  v.insert(v.end(), kv_.begin(), kv_.end());
  if (alignment_ != 32) {
    put_str(v, "general.alignment"); put<std::uint32_t>(v, 4); put<std::uint32_t>(v, alignment_);
  }
  std::uint64_t off = 0;
  std::vector<std::uint64_t> sizes;
  for (const TensorSpec& t : tensors_) {
    put_str(v, t.name);
    put<std::uint32_t>(v, static_cast<std::uint32_t>(t.ne.size()));
    std::uint64_t elems = 1;
    for (const auto d : t.ne) { put<std::uint64_t>(v, d); elems *= d; }
    put<std::uint32_t>(v, t.type);
    put<std::uint64_t>(v, off);
    const pelias::quant::TypeInfo ti = pelias::quant::type_info(t.type);
    const std::uint64_t bytes = ti.block_elements ? elems / ti.block_elements * ti.block_bytes : 0;
    sizes.push_back(bytes);
    off += (bytes + alignment_ - 1) / alignment_ * alignment_;
  }
  const std::uint64_t ds = (v.size() + alignment_ - 1) / alignment_ * alignment_;
  v.resize(static_cast<std::size_t>(ds), 0);
  if (data_start) *data_start = ds;
  std::uint64_t pos = ds;
  for (std::size_t i = 0; i < tensors_.size(); ++i) {
    const std::uint64_t padded = (sizes[i] + alignment_ - 1) / alignment_ * alignment_;
    v.resize(static_cast<std::size_t>(pos + padded), 0);
    for (std::uint64_t k = 0; k < sizes[i]; ++k) v[static_cast<std::size_t>(pos + k)] = static_cast<std::uint8_t>((i * 31 + k) & 0xFF);
    pos += padded;
  }
  return v;
}

bool write_and_map(const std::string& path, const std::vector<std::uint8_t>& bytes, pelias::os::FileMap& map) {
  map.close();  // Windows refuses to open a file for writing while a mapping of it is held (Linux allows it)
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  if (!bytes.empty() && std::fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size()) { std::fclose(f); return false; }
  std::fclose(f);
  return map.open(path).ok();
}

}  // namespace ptest
