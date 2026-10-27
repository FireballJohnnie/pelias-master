// provenance: original (layout READ-SRC from llama.cpp tools/imatrix/imatrix.cpp and common/imatrix-loader.cpp at
//   commit 2ca15f5404760548c39e7b92bd43116a09414a1a)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "model/imatrix.h"

#include <cstring>

#include "io/gguf_writer.h"

namespace pelias::model {

ImatrixEntry& ImatrixCollector::entry(const std::string& name, std::int64_t columns, std::int64_t matrices) {
  ImatrixEntry& e = entries_[name];
  if (e.columns == 0) {
    e.columns = columns;
    e.sum2.assign(static_cast<std::size_t>(columns * matrices), 0.0f);
    e.sum.assign(static_cast<std::size_t>(columns * matrices), 0.0f);
    e.counts.assign(static_cast<std::size_t>(matrices), 0);
  }
  return e;
}

void ImatrixCollector::add_row(ImatrixEntry& e, std::int64_t m, const float* x) {
  float* acc = e.sum2.data() + m * e.columns;
  float* act = e.sum.data() + m * e.columns;
  for (std::int64_t j = 0; j < e.columns; ++j) acc[j] += x[j] * x[j];
  for (std::int64_t j = 0; j < e.columns; ++j) act[j] += x[j];
  ++e.counts[static_cast<std::size_t>(m)];
}

void ImatrixCollector::add_dense(const std::string& name, const float* x, std::int64_t n_tok, std::int64_t ldx, std::int64_t columns) {
  if (n_tok < 16 || name.rfind("blk.", 0) != 0) return;
  ImatrixEntry& e = entry(name, columns, 1);
  float* acc = e.sum2.data();
  float* act = e.sum.data();
  for (std::int64_t t = 0; t < n_tok; ++t) {
    const float* row = x + t * ldx;
    for (std::int64_t j = 0; j < columns; ++j) acc[j] += row[j] * row[j];
    for (std::int64_t j = 0; j < columns; ++j) act[j] += row[j];
  }
  e.counts[0] += static_cast<std::uint64_t>(n_tok);
}

Status ImatrixCollector::write(const std::string& path, const std::vector<std::string>& datasets, std::uint32_t chunk_size) const {
  std::vector<io::GgufKvBlob> kv = {io::gguf_kv_str("general.type", "imatrix"), io::gguf_kv_arr_str("imatrix.datasets", datasets),
                                    io::gguf_kv_u32("imatrix.chunk_count", chunks_), io::gguf_kv_u32("imatrix.chunk_size", chunk_size)};
  std::vector<io::GgufOutTensor> out;
  std::vector<const std::vector<float>*> data;
  std::vector<std::vector<float>> counts_f;  // counts as floats, kept alive until written
  counts_f.reserve(entries_.size());
  for (const auto& [name, e] : entries_) {  // std::map: name order, as the reference sorts
    const auto nmat = static_cast<std::uint64_t>(e.counts.size());
    io::GgufOutTensor s2;
    s2.name = name + ".in_sum2";
    s2.ne[0] = static_cast<std::uint64_t>(e.columns);
    s2.ne[1] = nmat;
    s2.type = 0;
    s2.bytes = e.sum2.size() * 4;
    s2.n_dims = 2;
    io::GgufOutTensor c;
    c.name = name + ".counts";
    c.ne[0] = 1;
    c.ne[1] = nmat;
    c.type = 0;
    c.bytes = nmat * 4;
    c.n_dims = 2;
    io::GgufOutTensor s1 = s2;
    s1.name = name + ".in_sum";
    counts_f.emplace_back();
    for (std::uint64_t k : e.counts) counts_f.back().push_back(static_cast<float>(k));
    out.push_back(s2);  // the reference adds in_sum2, counts, then in_sum
    data.push_back(&e.sum2);
    out.push_back(c);
    data.push_back(&counts_f.back());
    out.push_back(s1);
    data.push_back(&e.sum);
  }
  io::GgufFileWriter w;
  Status s = w.begin(path, kv, out, 32);
  if (!s) return s;
  for (const std::vector<float>* d : data) {
    // little-endian float bytes
    std::vector<std::uint8_t> bytes(d->size() * 4);
    for (std::size_t i = 0; i < d->size(); ++i) {
      std::uint32_t u;
      std::memcpy(&u, &(*d)[i], 4);
      for (int b = 0; b < 4; ++b) bytes[i * 4 + static_cast<std::size_t>(b)] = static_cast<std::uint8_t>(u >> (8 * b));
    }
    s = w.write(bytes.data(), bytes.size());
    if (!s) {
      w.abandon();
      return s;
    }
  }
  return w.finish();
}

}  // namespace pelias::model
