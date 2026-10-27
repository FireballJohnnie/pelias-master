// provenance: original
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "io/gguf_compare.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include "quant/block_types.h"

namespace pelias::io {
namespace {

std::string shape_of(const GgufTensor& t) {
  std::string s = "[";
  for (std::uint32_t i = 0; i < t.n_dims; ++i) s += (i ? "," : "") + std::to_string(t.ne[i]);
  return s + "]";
}

}  // namespace

GgufCompareReport compare_gguf(const os::FileMap& ma, const GgufReader& a, const os::FileMap& mb, const GgufReader& b, std::size_t max_notes) {
  GgufCompareReport r;
  auto note = [&](const std::string& s) {
    if (r.notes.size() < max_notes) r.notes.push_back(s);
  };
  r.tensors_a = a.tensors().size();
  r.tensors_b = b.tensors().size();

  // metadata: same keys, same order, same encoded bytes
  r.kv_identical = a.kv().size() == b.kv().size();
  for (std::size_t i = 0; r.kv_identical && i < a.kv().size(); ++i) {
    const GgufKv& x = a.kv()[i];
    const GgufKv& y = b.kv()[i];
    const std::uint64_t nx = x.file_end - x.file_begin, ny = y.file_end - y.file_begin;
    if (x.key != y.key || nx != ny || std::memcmp(ma.data() + x.file_begin, mb.data() + y.file_begin, static_cast<std::size_t>(nx)) != 0) {
      r.kv_identical = false;
      note("metadata entry " + std::to_string(i) + " differs: '" + x.key + "' / '" + y.key + "'");
    }
  }
  if (!r.kv_identical && a.kv().size() != b.kv().size())
    note("metadata entries: " + std::to_string(a.kv().size()) + " / " + std::to_string(b.kv().size()));

  // tensors by name
  std::unordered_map<std::string, std::size_t> in_b;
  for (std::size_t i = 0; i < b.tensors().size(); ++i) in_b[b.tensors()[i].name] = i;
  r.order_identical = r.tensors_a == r.tensors_b;
  for (std::size_t i = 0; i < a.tensors().size(); ++i) {
    const GgufTensor& x = a.tensors()[i];
    const auto it = in_b.find(x.name);
    if (it == in_b.end()) {
      ++r.only_in_a;
      note("only in the first file: " + x.name);
      r.order_identical = false;
      continue;
    }
    if (it->second != i) r.order_identical = false;
    const GgufTensor& y = b.tensors()[it->second];
    bool same = true;
    if (x.type != y.type) {
      ++r.type_differs;
      same = false;
      note(x.name + ": type " + quant::type_info(x.type).name + " / " + quant::type_info(y.type).name);
    }
    if (x.n_dims != y.n_dims || std::memcmp(x.ne, y.ne, sizeof x.ne) != 0) {
      ++r.shape_differs;
      same = false;
      note(x.name + ": shape " + shape_of(x) + " / " + shape_of(y));
    }
    if (same) {
      const bool in_range_a = x.file_offset <= ma.size() && x.bytes <= ma.size() - x.file_offset;
      const bool in_range_b = y.file_offset <= mb.size() && y.bytes <= mb.size() - y.file_offset;
      if (!in_range_a || !in_range_b || x.bytes != y.bytes || std::memcmp(ma.data() + x.file_offset, mb.data() + y.file_offset, static_cast<std::size_t>(x.bytes)) != 0) {
        ++r.data_differs;
        same = false;
        // locate the first differing block for the note
        std::uint64_t k = 0;
        const std::uint64_t n = std::min(x.bytes, y.bytes);
        if (in_range_a && in_range_b)
          while (k < n && ma.data()[x.file_offset + k] == mb.data()[y.file_offset + k]) ++k;
        const quant::TypeInfo ti = quant::type_info(x.type);
        const std::uint64_t bb = ti.block_bytes ? ti.block_bytes : 1;
        note(x.name + ": data differs from byte " + std::to_string(k) + " (block " + std::to_string(k / bb) + ")");
      }
    }
    if (same) ++r.identical;
  }
  for (const GgufTensor& y : b.tensors())
    if (a.find_tensor(y.name) == nullptr) {
      ++r.only_in_b;
      note("only in the second file: " + y.name);
    }

  // the files themselves
  r.files_identical = ma.size() == mb.size();
  if (r.files_identical) {
    constexpr std::uint64_t kChunk = std::uint64_t{64} << 20;
    for (std::uint64_t off = 0; off < ma.size(); off += kChunk) {
      const auto n = static_cast<std::size_t>(std::min(kChunk, ma.size() - off));
      if (std::memcmp(ma.data() + off, mb.data() + off, n) != 0) {
        r.files_identical = false;
        std::uint64_t k = off;
        while (ma.data()[k] == mb.data()[k]) ++k;
        r.first_file_difference = k;
        break;
      }
    }
  } else {
    note("file sizes: " + std::to_string(ma.size()) + " / " + std::to_string(mb.size()) + " bytes");
  }
  return r;
}

}  // namespace pelias::io
