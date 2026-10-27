// provenance: original (layout and rules READ-SRC, see gguf_reader.h)
// Copyright (c) 2026 John Lipka and the Pelias contributors
#include "io/gguf_reader.h"

#include <cstring>
#include <limits>
#include <unordered_set>

#include "quant/block_types.h"

namespace pelias::io {
namespace {

constexpr std::uint64_t kMaxStringBytes = std::uint64_t{1} << 30;   // same limit as the reference reader
constexpr std::uint64_t kMaxArrayElements = std::uint64_t{1} << 30;
constexpr std::uint64_t kMaxNameLength = 63;                        // GGML_MAX_NAME - 1
constexpr std::uint32_t kMaxDims = 4;
constexpr std::uint64_t kMaxCount = std::uint64_t{1} << 40;         // n_tensors / n_kv sanity bound (far above any real file)

// A cursor over the mapped bytes that refuses to read past the end.
class Cursor {
 public:
  Cursor(const std::uint8_t* p, std::uint64_t n) : p_(p), n_(n) {}
  [[nodiscard]] std::uint64_t pos() const { return pos_; }
  [[nodiscard]] std::uint64_t remaining() const { return n_ - pos_; }

  Status need(std::uint64_t bytes, const char* what, const std::string& name = {}) const {
    if (bytes > remaining())
      return Status::fail(Code::corrupt, std::string("file ends inside ") + what + " (needs " + std::to_string(bytes) + " more bytes)", pos_, name);
    return Status::good();
  }
  template <typename T>
  Status read(T& out, const char* what, const std::string& name = {}) {
    const Status s = need(sizeof(T), what, name);
    if (!s) return s;
    std::memcpy(&out, p_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return Status::good();
  }
  Status skip(std::uint64_t bytes, const char* what, const std::string& name = {}) {
    const Status s = need(bytes, what, name);
    if (!s) return s;
    pos_ += bytes;
    return Status::good();
  }
  Status read_string(std::string& out, const char* what, const std::string& name = {}) {
    std::uint64_t len = 0;
    const std::uint64_t at = pos_;
    Status s = read(len, what, name);
    if (!s) return s;
    if (len > kMaxStringBytes) return Status::fail(Code::corrupt, std::string("string length ") + std::to_string(len) + " in " + what + " exceeds the limit", at, name);
    s = need(len, what, name);
    if (!s) return s;
    out.assign(reinterpret_cast<const char*>(p_ + pos_), static_cast<std::size_t>(len));
    pos_ += len;
    return Status::good();
  }

 private:
  const std::uint8_t* p_;
  std::uint64_t n_;
  std::uint64_t pos_ = 0;
};

std::uint64_t scalar_size(GgufValueType t) {
  switch (t) {
    case GgufValueType::u8: case GgufValueType::i8: case GgufValueType::boolean: return 1;
    case GgufValueType::u16: case GgufValueType::i16: return 2;
    case GgufValueType::u32: case GgufValueType::i32: case GgufValueType::f32: return 4;
    case GgufValueType::u64: case GgufValueType::i64: case GgufValueType::f64: return 8;
    case GgufValueType::string: case GgufValueType::array: return 0;
  }
  return 0;
}

bool valid_type(std::uint32_t t) { return t <= static_cast<std::uint32_t>(GgufValueType::f64); }

Status read_scalar(Cursor& c, GgufValueType t, GgufValue& v, const std::string& key) {
  v.type = t;
  switch (t) {
    case GgufValueType::u8: { std::uint8_t x = 0; Status s = c.read(x, "value", key); if (!s) return s; v.u = x; v.i = x; v.f = x; return s; }
    case GgufValueType::i8: { std::int8_t x = 0; Status s = c.read(x, "value", key); if (!s) return s; v.i = x; v.u = static_cast<std::uint64_t>(x); v.f = x; return s; }
    case GgufValueType::u16: { std::uint16_t x = 0; Status s = c.read(x, "value", key); if (!s) return s; v.u = x; v.i = x; v.f = x; return s; }
    case GgufValueType::i16: { std::int16_t x = 0; Status s = c.read(x, "value", key); if (!s) return s; v.i = x; v.u = static_cast<std::uint64_t>(x); v.f = x; return s; }
    case GgufValueType::u32: { std::uint32_t x = 0; Status s = c.read(x, "value", key); if (!s) return s; v.u = x; v.i = x; v.f = x; return s; }
    case GgufValueType::i32: { std::int32_t x = 0; Status s = c.read(x, "value", key); if (!s) return s; v.i = x; v.u = static_cast<std::uint64_t>(x); v.f = x; return s; }
    case GgufValueType::f32: { float x = 0; Status s = c.read(x, "value", key); if (!s) return s; v.f = x; return s; }
    case GgufValueType::boolean: { std::uint8_t x = 0; Status s = c.read(x, "value", key); if (!s) return s; v.b = x != 0; v.u = x; v.i = x; return s; }
    case GgufValueType::u64: { std::uint64_t x = 0; Status s = c.read(x, "value", key); if (!s) return s; v.u = x; v.i = static_cast<std::int64_t>(x); v.f = static_cast<double>(x); return s; }
    case GgufValueType::i64: { std::int64_t x = 0; Status s = c.read(x, "value", key); if (!s) return s; v.i = x; v.u = static_cast<std::uint64_t>(x); v.f = static_cast<double>(x); return s; }
    case GgufValueType::f64: { double x = 0; Status s = c.read(x, "value", key); if (!s) return s; v.f = x; return s; }
    case GgufValueType::string: return c.read_string(v.s, "string value", key);
    case GgufValueType::array: break;
  }
  return Status::fail(Code::internal, "read_scalar called with an array type", c.pos(), key);
}

}  // namespace

Status GgufReader::parse(const os::FileMap& map, bool require_data) {
  map_ = &map;
  kv_.clear();
  tensors_.clear();
  file_bytes_ = map.size();
  Cursor c(map.data(), map.size());

  char magic[4];
  Status s = c.need(4, "the magic number");
  if (!s) return s;
  std::memcpy(magic, map.data(), 4);
  if (std::memcmp(magic, "GGUF", 4) != 0) return Status::fail(Code::corrupt, "not a GGUF file (magic number differs)", 0);
  c.skip(4, "magic");
  s = c.read(version_, "the version");
  if (!s) return s;
  if ((version_ & 0xFFFFu) == 0) return Status::fail(Code::corrupt, "version field looks byte-swapped (a file written on a big-endian machine?)", 4);
  if (version_ == 1) return Status::fail(Code::unsupported, "GGUF version 1 is not supported", 4);
  if (version_ > 3) return Status::fail(Code::unsupported, "GGUF version " + std::to_string(version_) + " is newer than this reader (up to 3)", 4);

  std::int64_t n_tensors = 0, n_kv = 0;
  s = c.read(n_tensors, "the tensor count");
  if (!s) return s;
  if (n_tensors < 0 || static_cast<std::uint64_t>(n_tensors) > kMaxCount) return Status::fail(Code::corrupt, "tensor count " + std::to_string(n_tensors) + " is out of range", 8);
  s = c.read(n_kv, "the key-value count");
  if (!s) return s;
  if (n_kv < 0 || static_cast<std::uint64_t>(n_kv) > kMaxCount) return Status::fail(Code::corrupt, "key-value count " + std::to_string(n_kv) + " is out of range", 16);
  // Each entry needs at least a few bytes; refuse counts the remaining bytes cannot hold before allocating anything.
  if (static_cast<std::uint64_t>(n_kv) * 12 > c.remaining() || static_cast<std::uint64_t>(n_tensors) * 28 > c.remaining())
    return Status::fail(Code::corrupt, "the file is too small for the counts in its header", 8);

  // key-value pairs
  std::unordered_set<std::string> seen_keys;
  kv_.reserve(static_cast<std::size_t>(n_kv));
  for (std::int64_t i = 0; i < n_kv; ++i) {
    GgufKv kv;
    const std::uint64_t at = c.pos();
    kv.file_begin = at;
    s = c.read_string(kv.key, "a key");
    if (!s) return s;
    if (kv.key.empty()) return Status::fail(Code::corrupt, "empty key at key-value pair " + std::to_string(i), at);
    if (!seen_keys.insert(kv.key).second) return Status::fail(Code::corrupt, "duplicate key", at, kv.key);
    std::uint32_t t = 0;
    s = c.read(t, "a value type", kv.key);
    if (!s) return s;
    if (!valid_type(t)) return Status::fail(Code::corrupt, "unknown value type " + std::to_string(t), c.pos() - 4, kv.key);
    if (t == static_cast<std::uint32_t>(GgufValueType::array)) {
      kv.value.type = GgufValueType::array;
      std::uint32_t et = 0;
      s = c.read(et, "an array element type", kv.key);
      if (!s) return s;
      if (!valid_type(et) || et == static_cast<std::uint32_t>(GgufValueType::array))
        return Status::fail(Code::corrupt, "unknown or nested array element type " + std::to_string(et), c.pos() - 4, kv.key);
      kv.value.array_type = static_cast<GgufValueType>(et);
      std::uint64_t n = 0;
      s = c.read(n, "an array length", kv.key);
      if (!s) return s;
      if (n > kMaxArrayElements) return Status::fail(Code::corrupt, "array length " + std::to_string(n) + " exceeds the limit", c.pos() - 8, kv.key);
      kv.value.array_count = n;
      kv.value.array_offset = c.pos();
      if (kv.value.array_type == GgufValueType::string) {
        for (std::uint64_t k = 0; k < n; ++k) {
          std::uint64_t len = 0;
          s = c.read(len, "a string array element", kv.key);
          if (!s) return s;
          if (len > kMaxStringBytes) return Status::fail(Code::corrupt, "string length in array exceeds the limit", c.pos() - 8, kv.key);
          s = c.skip(len, "a string array element", kv.key);
          if (!s) return s;
        }
      } else {
        const std::uint64_t es = scalar_size(kv.value.array_type);
        if (es == 0) return Status::fail(Code::internal, "array element type has no size", c.pos(), kv.key);  // unreachable: string and array are handled above
        if (n > std::numeric_limits<std::uint64_t>::max() / es) return Status::fail(Code::corrupt, "array size overflows", c.pos(), kv.key);
        s = c.skip(n * es, "array elements", kv.key);
        if (!s) return s;
      }
    } else {
      s = read_scalar(c, static_cast<GgufValueType>(t), kv.value, kv.key);
      if (!s) return s;
    }
    kv.file_end = c.pos();
    kv_.push_back(std::move(kv));
  }

  // alignment
  alignment_ = 32;
  if (const GgufValue* a = find("general.alignment")) {
    if (a->type != GgufValueType::u32) return Status::fail(Code::corrupt, "general.alignment must be a u32", 0, "general.alignment");
    alignment_ = static_cast<std::uint32_t>(a->u);
    if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0)
      return Status::fail(Code::corrupt, "alignment " + std::to_string(alignment_) + " is not a power of two", 0, "general.alignment");
  }

  // tensor table
  std::unordered_set<std::string> seen_names;
  tensors_.reserve(static_cast<std::size_t>(n_tensors));
  for (std::int64_t i = 0; i < n_tensors; ++i) {
    GgufTensor t;
    const std::uint64_t at = c.pos();
    s = c.read_string(t.name, "a tensor name");
    if (!s) return s;
    if (t.name.empty()) return Status::fail(Code::corrupt, "empty tensor name at tensor " + std::to_string(i), at);
    if (t.name.size() > kMaxNameLength) return Status::fail(Code::corrupt, "tensor name longer than " + std::to_string(kMaxNameLength) + " bytes", at, t.name.substr(0, 32) + "...");
    if (!seen_names.insert(t.name).second) return Status::fail(Code::corrupt, "duplicate tensor name", at, t.name);
    s = c.read(t.n_dims, "the dimension count", t.name);
    if (!s) return s;
    if (t.n_dims == 0 || t.n_dims > kMaxDims) return Status::fail(Code::corrupt, "dimension count " + std::to_string(t.n_dims) + " (allowed 1 to 4)", c.pos() - 4, t.name);
    for (std::uint32_t d = 0; d < t.n_dims; ++d) {
      std::int64_t ne = 0;
      s = c.read(ne, "a dimension", t.name);
      if (!s) return s;
      if (ne < 0) return Status::fail(Code::corrupt, "negative dimension", c.pos() - 8, t.name);
      t.ne[d] = static_cast<std::uint64_t>(ne);
    }
    // the element count must not overflow 64 bits (check pairwise before multiplying)
    std::uint64_t elems = 1;
    for (std::uint32_t d = 0; d < 4; ++d) {
      if (t.ne[d] != 0 && elems > std::numeric_limits<std::uint64_t>::max() / t.ne[d])
        return Status::fail(Code::corrupt, "element count overflows", at, t.name);
      elems *= t.ne[d];
    }
    s = c.read(t.type, "the tensor type", t.name);
    if (!s) return s;
    const quant::TypeInfo ti = quant::type_info(t.type);
    if (!ti.known) return Status::fail(Code::corrupt, "unknown tensor type " + std::to_string(t.type), c.pos() - 4, t.name);
    if (t.ne[0] % ti.block_elements != 0)
      return Status::fail(Code::corrupt, std::string("first dimension ") + std::to_string(t.ne[0]) + " is not a multiple of the " + ti.name + " block (" +
                                             std::to_string(ti.block_elements) + ")", at, t.name);
    const std::uint64_t blocks = elems / ti.block_elements;
    if (blocks != 0 && ti.block_bytes > std::numeric_limits<std::uint64_t>::max() / blocks) return Status::fail(Code::corrupt, "byte size overflows", at, t.name);
    t.bytes = blocks * ti.block_bytes;
    s = c.read(t.data_offset, "the data offset", t.name);
    if (!s) return s;
    tensors_.push_back(std::move(t));
  }

  // data section: starts at the next multiple of the alignment; offsets must be dense and in order, as the reference
  // reader requires (each tensor's offset equals the padded sum of the ones before it)
  const std::uint64_t header_end = c.pos();
  data_start_ = (header_end + alignment_ - 1) / alignment_ * alignment_;
  if (require_data && n_tensors > 0 && data_start_ > file_bytes_) return Status::fail(Code::corrupt, "file ends before the data section", header_end);
  std::uint64_t running = 0;
  for (GgufTensor& t : tensors_) {
    if (t.data_offset != running)
      return Status::fail(Code::corrupt, "tensor data offset " + std::to_string(t.data_offset) + " differs from the expected " + std::to_string(running), header_end, t.name);
    if (t.bytes > std::numeric_limits<std::uint64_t>::max() - (alignment_ - 1)) return Status::fail(Code::corrupt, "tensor size overflows after padding", header_end, t.name);
    const std::uint64_t padded = (t.bytes + alignment_ - 1) / alignment_ * alignment_;
    if (running > std::numeric_limits<std::uint64_t>::max() - padded) return Status::fail(Code::corrupt, "data section size overflows", header_end, t.name);
    t.file_offset = data_start_ + t.data_offset;
    if (require_data && (t.file_offset > file_bytes_ || t.bytes > file_bytes_ - t.file_offset))
      return Status::fail(Code::corrupt, "tensor data runs past the end of the file (needs " + std::to_string(t.bytes) + " bytes at byte " +
                                             std::to_string(t.file_offset) + ", file has " + std::to_string(file_bytes_) + ")", t.file_offset, t.name);
    running += padded;
  }
  data_bytes_ = running;
  return Status::good();
}

const GgufValue* GgufReader::find(std::string_view key) const {
  for (const GgufKv& kv : kv_)
    if (kv.key == key) return &kv.value;
  return nullptr;
}

const GgufTensor* GgufReader::find_tensor(std::string_view name) const {
  for (const GgufTensor& t : tensors_)
    if (t.name == name) return &t;
  return nullptr;
}

std::string GgufReader::get_string(std::string_view key, bool* found) const {
  const GgufValue* v = find(key);
  const bool ok = v && v->type == GgufValueType::string;
  if (found) *found = ok;
  return ok ? v->s : std::string();
}

std::uint64_t GgufReader::get_uint(std::string_view key, bool* found) const {
  const GgufValue* v = find(key);
  const bool ok = v && v->type != GgufValueType::string && v->type != GgufValueType::array && v->type != GgufValueType::f32 && v->type != GgufValueType::f64;
  if (found) *found = ok;
  return ok ? v->u : 0;
}

double GgufReader::get_float(std::string_view key, bool* found) const {
  const GgufValue* v = find(key);
  const bool ok = v && (v->type == GgufValueType::f32 || v->type == GgufValueType::f64);
  if (found) *found = ok;
  return ok ? v->f : 0.0;
}

std::string GgufReader::array_string(const GgufValue& v, std::uint64_t i, bool* found) const {
  if (found) *found = false;
  if (!map_ || v.type != GgufValueType::array || v.array_type != GgufValueType::string || i >= v.array_count) return {};
  Cursor c(map_->data(), map_->size());
  if (!c.skip(v.array_offset, "array")) return {};
  std::string s;
  for (std::uint64_t k = 0; k <= i; ++k) {
    std::uint64_t len = 0;
    if (!c.read(len, "string")) return {};
    if (k == i) {
      if (!c.need(len, "string")) return {};
      s.assign(reinterpret_cast<const char*>(map_->data() + c.pos()), static_cast<std::size_t>(len));
      if (found) *found = true;
      return s;
    }
    if (!c.skip(len, "string")) return {};
  }
  return {};
}

Status GgufReader::array_strings(const GgufValue& v, std::vector<std::string>& out) const {
  out.clear();
  if (!map_ || v.type != GgufValueType::array || v.array_type != GgufValueType::string)
    return Status::fail(Code::invalid_argument, "not a string array");
  Cursor c(map_->data(), map_->size());
  Status s = c.skip(v.array_offset, "array");
  if (!s) return s;
  out.resize(static_cast<std::size_t>(v.array_count));
  for (std::uint64_t k = 0; k < v.array_count; ++k) {
    s = c.read_string(out[static_cast<std::size_t>(k)], "string array element");
    if (!s) return s;
  }
  return Status::good();
}

Status GgufReader::array_ints(const GgufValue& v, std::vector<std::int64_t>& out) const {
  out.clear();
  if (!map_ || v.type != GgufValueType::array) return Status::fail(Code::invalid_argument, "not an array");
  const GgufValueType t = v.array_type;
  const bool is_int = t == GgufValueType::u8 || t == GgufValueType::i8 || t == GgufValueType::u16 || t == GgufValueType::i16 ||
                      t == GgufValueType::u32 || t == GgufValueType::i32 || t == GgufValueType::u64 || t == GgufValueType::i64 ||
                      t == GgufValueType::boolean;
  if (!is_int) return Status::fail(Code::invalid_argument, "not an integer array");
  Cursor c(map_->data(), map_->size());
  Status s = c.skip(v.array_offset, "array");
  if (!s) return s;
  out.resize(static_cast<std::size_t>(v.array_count));
  for (std::uint64_t k = 0; k < v.array_count; ++k) {
    std::int64_t x = 0;
    switch (t) {
      case GgufValueType::u8:
      case GgufValueType::boolean: { std::uint8_t e = 0; s = c.read(e, "array element"); x = e; break; }
      case GgufValueType::i8: { std::int8_t e = 0; s = c.read(e, "array element"); x = e; break; }
      case GgufValueType::u16: { std::uint16_t e = 0; s = c.read(e, "array element"); x = e; break; }
      case GgufValueType::i16: { std::int16_t e = 0; s = c.read(e, "array element"); x = e; break; }
      case GgufValueType::u32: { std::uint32_t e = 0; s = c.read(e, "array element"); x = e; break; }
      case GgufValueType::i32: { std::int32_t e = 0; s = c.read(e, "array element"); x = e; break; }
      case GgufValueType::u64: { std::uint64_t e = 0; s = c.read(e, "array element"); x = static_cast<std::int64_t>(e); break; }
      default: { std::int64_t e = 0; s = c.read(e, "array element"); x = e; break; }
    }
    if (!s) return s;
    out[static_cast<std::size_t>(k)] = x;
  }
  return Status::good();
}

}  // namespace pelias::io
