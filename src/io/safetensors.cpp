// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "io/safetensors.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace pelias::io {

std::uint64_t SafetensorsTensor::elements() const {
  std::uint64_t n = 1;
  for (const std::uint64_t d : shape) n *= d;
  return n;
}

std::uint32_t safetensors_dtype_bytes(std::string_view d) {
  if (d == "F64" || d == "I64" || d == "U64") return 8;
  if (d == "F32" || d == "I32" || d == "U32") return 4;
  if (d == "F16" || d == "BF16" || d == "I16" || d == "U16") return 2;
  if (d == "I8" || d == "U8" || d == "BOOL" || d == "F8_E4M3" || d == "F8_E5M2") return 1;
  return 0;
}

Status SafetensorsReader::parse(const os::FileMap& map) {
  tensors_.clear();
  metadata_ = Json();
  if (map.size() < 8) return Status::fail(Code::corrupt, "file shorter than the 8-byte header length", 0);
  std::uint64_t n = 0;
  std::memcpy(&n, map.data(), 8);
  header_bytes_ = n;
  if (n > map.size() - 8) return Status::fail(Code::corrupt, "header length " + std::to_string(n) + " runs past the end of the file", 0);
  if (n > (std::uint64_t{1} << 31)) return Status::fail(Code::corrupt, "header length " + std::to_string(n) + " exceeds the limit", 0);
  data_start_ = 8 + n;
  const Result<Json> parsed = Json::parse(std::string_view(reinterpret_cast<const char*>(map.data() + 8), static_cast<std::size_t>(n)));
  if (!parsed.ok()) {
    Status s = parsed.status;
    s.offset += 8;
    return s;
  }
  const Json& root = parsed.value;
  if (!root.is_object()) return Status::fail(Code::corrupt, "header is not a JSON object", 8);
  const std::uint64_t data_bytes = map.size() - data_start_;
  for (const auto& [name, v] : root.members()) {
    if (name == "__metadata__") {
      metadata_ = v;
      continue;
    }
    if (!v.is_object()) return Status::fail(Code::corrupt, "tensor entry is not an object", 8, name);
    SafetensorsTensor t;
    t.name = name;
    const Json* dt = v.get("dtype");
    const Json* sh = v.get("shape");
    const Json* off = v.get("data_offsets");
    if (!dt || !dt->is_string()) return Status::fail(Code::corrupt, "missing or non-string dtype", 8, name);
    if (!sh || !sh->is_array()) return Status::fail(Code::corrupt, "missing or non-array shape", 8, name);
    if (!off || !off->is_array() || off->size() != 2) return Status::fail(Code::corrupt, "data_offsets must be [begin, end]", 8, name);
    t.dtype = dt->as_string();
    const std::uint32_t es = safetensors_dtype_bytes(t.dtype);
    if (es == 0) return Status::fail(Code::unsupported, "unknown dtype '" + t.dtype + "'", 8, name);
    std::uint64_t elems = 1;
    for (const Json& d : sh->items()) {
      if (!d.is_integer() || d.as_int() < 0) return Status::fail(Code::corrupt, "shape entry is not a non-negative integer", 8, name);
      const std::uint64_t x = d.as_uint();
      if (x != 0 && elems > std::numeric_limits<std::uint64_t>::max() / x) return Status::fail(Code::corrupt, "shape overflows", 8, name);
      elems *= x;
      t.shape.push_back(x);
    }
    const Json* b = off->at(0);
    const Json* e = off->at(1);
    if (!b->is_integer() || !e->is_integer() || b->as_int() < 0 || e->as_int() < 0) return Status::fail(Code::corrupt, "data_offsets are not non-negative integers", 8, name);
    t.begin = b->as_uint();
    t.end = e->as_uint();
    if (t.end < t.begin) return Status::fail(Code::corrupt, "data_offsets end before begin", 8, name);
    if (elems > std::numeric_limits<std::uint64_t>::max() / es || t.end - t.begin != elems * es)
      return Status::fail(Code::corrupt, "data_offsets span " + std::to_string(t.end - t.begin) + " bytes but the shape needs " + std::to_string(elems * es), 8, name);
    if (t.end > data_bytes) return Status::fail(Code::corrupt, "tensor data runs past the end of the file", data_start_ + t.begin, name);
    t.file_offset = data_start_ + t.begin;
    tensors_.push_back(std::move(t));
  }
  // the ranges must tile the data section without gaps or overlaps (the format's rule)
  std::vector<const SafetensorsTensor*> order;
  for (const auto& t : tensors_) order.push_back(&t);
  std::sort(order.begin(), order.end(), [](const SafetensorsTensor* a, const SafetensorsTensor* b) {
    // by begin, then end: a zero-length tensor sharing a begin with a real one sorts first, so the walk stays in order
    return a->begin != b->begin ? a->begin < b->begin : a->end < b->end;
  });
  std::uint64_t expect = 0;
  for (const SafetensorsTensor* t : order) {
    if (t->begin != expect) return Status::fail(Code::corrupt, "tensor data does not follow the previous tensor (gap or overlap at " + std::to_string(t->begin) + ")", data_start_ + t->begin, t->name);
    expect = t->end;
  }
  if (expect != data_bytes) return Status::fail(Code::corrupt, "data section has " + std::to_string(data_bytes - expect) + " trailing bytes no tensor claims", data_start_ + expect);
  return Status::good();
}

Status SafetensorsShard::open(const std::string& path) {
  Status s = map_.open(path);
  if (!s) return s;
  return reader_.parse(map_);
}

}  // namespace pelias::io
