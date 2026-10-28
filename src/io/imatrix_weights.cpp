// provenance: original (procedure READ-SRC, see imatrix_weights.h)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "io/imatrix_weights.h"

#include <cmath>
#include <cstring>

#include "io/gguf_reader.h"
#include "os/file_map.h"

namespace pelias::io {
namespace {

bool strip(std::string& s, const std::string& suffix) {
  if (s.size() < suffix.size() || s.compare(s.size() - suffix.size(), suffix.size(), suffix) != 0) return false;
  s.resize(s.size() - suffix.size());
  return true;
}

std::uint64_t elements(const GgufTensor& t) { return t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]; }

std::vector<float> floats(const os::FileMap& m, const GgufTensor& t) {
  std::vector<float> v(static_cast<std::size_t>(elements(t)));
  std::memcpy(v.data(), m.data() + t.file_offset, v.size() * 4);  // little-endian hosts only (the build checks)
  return v;
}

}  // namespace

Result<ImatrixWeights> read_imatrix_weights(const std::string& path) {
  using R = Result<ImatrixWeights>;
  os::FileMap m;
  Status s = m.open(path);
  if (!s) return R::fail(s);
  if (m.size() < 4 || std::memcmp(m.data(), "GGUF", 4) != 0)
    return R::fail(Status::fail(Code::unsupported, "the importance matrix is not a GGUF file (the legacy format is not read)"));
  GgufReader r;
  s = r.parse(m);
  if (!s) return R::fail(s);
  ImatrixWeights w;
  bool has_ds = false, has_cc = false, has_cs = false;
  if (const GgufValue* v = r.find("imatrix.datasets"); v && v->type == GgufValueType::array && v->array_type == GgufValueType::string) {
    if (Status as = r.array_strings(*v, w.datasets); !as) return R::fail(as);
    has_ds = true;
  } else if (v) {
    has_ds = true;  // present with another type: the reference counts the key as present but reads no names
  }
  w.chunk_count = static_cast<std::uint32_t>(r.get_uint("imatrix.chunk_count", &has_cc));
  w.chunk_size = static_cast<std::uint32_t>(r.get_uint("imatrix.chunk_size", &has_cs));
  if (!has_ds || !has_cc || !has_cs) return R::fail(Status::fail(Code::corrupt, "missing imatrix metadata (imatrix.datasets, chunk_count, chunk_size)"));
  if (const GgufValue* v = r.find("pelias.rot32.tensors")) {
    if (v->type != GgufValueType::array || v->array_type != GgufValueType::string)
      return R::fail(Status::fail(Code::corrupt, "pelias.rot32.tensors is not a list of tensor names"));
    if (Status as = r.array_strings(*v, w.rot32); !as) return R::fail(as);
    w.has_rot32 = true;
  }

  struct Parts {
    const GgufTensor* sum = nullptr;
    const GgufTensor* sum2 = nullptr;
    const GgufTensor* counts = nullptr;
  };
  std::map<std::string, Parts> parts;
  for (const GgufTensor& t : r.tensors()) {
    std::string n = t.name;
    if (n.empty()) continue;
    if (strip(n, ".in_sum")) parts[n].sum = &t;
    else if (strip(n, ".in_sum2")) parts[n].sum2 = &t;
    else if (strip(n, ".counts")) parts[n].counts = &t;
  }
  if (parts.empty()) return R::fail(Status::fail(Code::corrupt, "the importance matrix holds no entries"));
  for (const auto& [name, p] : parts) {
    if (!p.sum2 || !p.counts || (p.sum && elements(*p.sum) != elements(*p.sum2)))
      return R::fail(Status::fail(Code::corrupt, "mismatched sums and counts", 0, name));
    if (p.sum2->type != 0 || p.counts->type != 0) return R::fail(Status::fail(Code::corrupt, "sums and counts must be F32", 0, name));
    const std::vector<float> sums = floats(m, *p.sum2), cf = floats(m, *p.counts);
    const std::size_t ncounts = cf.size();
    if (ncounts == 0 || sums.size() % ncounts != 0) return R::fail(Status::fail(Code::corrupt, "sums are not a whole number of matrices", 0, name));
    const std::size_t ne0 = sums.size() / ncounts;
    std::vector<float>& e = w.weights[name];
    e.resize(sums.size());
    for (std::size_t j = 0; j < ncounts; ++j) {
      const float count = static_cast<float>(std::lround(cf[j]));  // the loader keeps whole counts, quantize divides by them
      for (std::size_t i = 0; i < ne0; ++i) e[j * ne0 + i] = count > 0.0f ? sums[j * ne0 + i] / count : 1.0f;
    }
    for (float f : e)
      if (!std::isfinite(f)) return R::fail(Status::fail(Code::corrupt, "the importance matrix holds a non-finite value", 0, name));
  }
  return R::good(std::move(w));
}

}  // namespace pelias::io
