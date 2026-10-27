// provenance: original (layout READ-SRC, see gguf_writer.h)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "io/gguf_writer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace pelias::io {
namespace {

void append(std::vector<std::uint8_t>& v, const void* p, std::size_t n) {
  const auto* b = static_cast<const std::uint8_t*>(p);
  v.insert(v.end(), b, b + n);
}
void append_string(std::vector<std::uint8_t>& v, const std::string& s) {
  const std::uint64_t n = s.size();
  append(v, &n, 8);
  append(v, s.data(), s.size());
}

// the reference's ggml_n_dims: dimensions up to the last one larger than 1, at least 1
std::uint32_t n_dims_of(const std::uint64_t ne[4]) {
  std::uint32_t nd = 1;
  for (std::uint32_t d = 0; d < 4; ++d)
    if (ne[d] > 1) nd = d + 1;
  return nd;
}

std::uint64_t padded(std::uint64_t n, std::uint64_t a) { return (n + a - 1) / a * a; }

int block_number(const std::string& name) {
  int l = -1;
  if (name.rfind("blk.", 0) == 0) {
    std::size_t i = 4;
    long long v = 0;
    bool any = false;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9' && v < 100000000) {
      v = v * 10 + (name[i] - '0');
      ++i;
      any = true;
    }
    // the reference reads "blk.%d." with sscanf: digits after "blk." are enough, the dot is not required to match
    if (any) l = static_cast<int>(v);
  }
  return l;
}

}  // namespace

GgufKvBlob gguf_kv_u32(const std::string& key, std::uint32_t value) {
  GgufKvBlob b;
  append_string(b, key);
  const std::int32_t type = 4;  // u32
  append(b, &type, 4);
  append(b, &value, 4);
  return b;
}

std::vector<std::size_t> reference_order(const std::vector<std::string>& names) {
  std::vector<std::size_t> idx(names.size());
  for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::vector<int> layer(names.size());
  for (std::size_t i = 0; i < names.size(); ++i) layer[i] = block_number(names[i]);
  std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
    if (layer[a] != layer[b]) return layer[a] < layer[b];
    return names[a] < names[b];
  });
  return idx;
}

std::uint64_t gguf_meta_bytes(const std::vector<GgufKvBlob>& kv, const std::vector<GgufOutTensor>& tensors, std::uint32_t alignment) {
  std::uint64_t n = 4 + 4 + 8 + 8;
  for (const auto& b : kv) n += b.size();
  for (const auto& t : tensors) n += 8 + t.name.size() + 4 + 8ull * n_dims_of(t.ne) + 4 + 8;
  return padded(n, alignment);
}

std::uint64_t gguf_file_bytes(const std::vector<GgufKvBlob>& kv, const std::vector<GgufOutTensor>& tensors, std::uint32_t alignment) {
  std::uint64_t n = gguf_meta_bytes(kv, tensors, alignment);
  for (const auto& t : tensors) n += padded(t.bytes, alignment);
  return n;
}

GgufFileWriter::~GgufFileWriter() { abandon(); }

void GgufFileWriter::abandon() {
  if (f_) std::fclose(f_);
  f_ = nullptr;
}

Status GgufFileWriter::put(const void* p, std::uint64_t n) {
  if (n && std::fwrite(p, 1, static_cast<std::size_t>(n), f_) != n) return Status::fail(Code::io_error, "write failed after " + std::to_string(written_) + " bytes (disk full?)", written_);
  written_ += n;
  return Status::good();
}

Status GgufFileWriter::pad_to_alignment() {
  static const std::uint8_t zeros[64] = {};
  while (written_ % alignment_ != 0) {
    const std::uint64_t n = std::min<std::uint64_t>(alignment_ - written_ % alignment_, sizeof zeros);
    const Status s = put(zeros, n);
    if (!s) return s;
  }
  return Status::good();
}

Status GgufFileWriter::begin(const std::string& path, const std::vector<GgufKvBlob>& kv, const std::vector<GgufOutTensor>& tensors,
                             std::uint32_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) return Status::fail(Code::invalid_argument, "alignment must be a power of two");
  std::error_code ec;
  if (std::filesystem::exists(path, ec)) return Status::fail(Code::invalid_argument, "the output file already exists (nothing is overwritten)");
  f_ = std::fopen(path.c_str(), "wb");
  if (!f_) return Status::fail(Code::io_error, "cannot create the output file");
  buffer_.resize(std::size_t{8} << 20);
  std::setvbuf(f_, buffer_.data(), _IOFBF, buffer_.size());
  tensors_ = tensors;
  alignment_ = alignment;
  current_ = 0;
  current_done_ = 0;
  written_ = 0;

  std::vector<std::uint8_t> h;
  append(h, "GGUF", 4);
  const std::uint32_t version = 3;
  const auto n_tensors = static_cast<std::int64_t>(tensors.size());
  const auto n_kv = static_cast<std::int64_t>(kv.size());
  append(h, &version, 4);
  append(h, &n_tensors, 8);
  append(h, &n_kv, 8);
  for (const auto& b : kv) append(h, b.data(), b.size());
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    append_string(h, t.name);
    const std::uint32_t nd = n_dims_of(t.ne);
    append(h, &nd, 4);
    for (std::uint32_t d = 0; d < nd; ++d) append(h, &t.ne[d], 8);
    const auto type = static_cast<std::int32_t>(t.type);
    append(h, &type, 4);
    append(h, &offset, 8);
    offset += padded(t.bytes, alignment);
  }
  Status s = put(h.data(), h.size());
  if (!s) return s;
  s = pad_to_alignment();
  if (!s) return s;
  // a tensor with no data is complete at once
  while (current_ < tensors_.size() && tensors_[current_].bytes == 0) ++current_;
  return Status::good();
}

Status GgufFileWriter::write(const std::uint8_t* data, std::uint64_t bytes) {
  if (!f_) return Status::fail(Code::internal, "writer not open");
  while (bytes > 0) {
    if (current_ >= tensors_.size()) return Status::fail(Code::internal, "more data than the tensors hold");
    const std::uint64_t room = tensors_[current_].bytes - current_done_;
    const std::uint64_t n = std::min(room, bytes);
    Status s = put(data, n);
    if (!s) return s;
    data += n;
    bytes -= n;
    current_done_ += n;
    if (current_done_ == tensors_[current_].bytes) {
      s = pad_to_alignment();
      if (!s) return s;
      ++current_;
      current_done_ = 0;
      while (current_ < tensors_.size() && tensors_[current_].bytes == 0) ++current_;
    }
  }
  return Status::good();
}

Status GgufFileWriter::finish() {
  if (!f_) return Status::fail(Code::internal, "writer not open");
  if (current_ != tensors_.size()) {
    abandon();
    return Status::fail(Code::internal, "not every tensor was written", 0, current_ < tensors_.size() ? tensors_[current_].name : std::string());
  }
  const bool ok = std::fflush(f_) == 0;
  const bool closed = std::fclose(f_) == 0;
  f_ = nullptr;
  if (!ok || !closed) return Status::fail(Code::io_error, "could not finish writing the output file (disk full?)");
  return Status::good();
}

}  // namespace pelias::io
